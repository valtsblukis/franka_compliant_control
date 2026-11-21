"""Dashboard setting endpoints with a recording backend, not a robot/plant."""
import threading

import pytest

from franka_compliant_control.client import CompliantClient
from test_policy_http import service


def test_recovery_toggle_updates_live_policy_without_recovering(service):
    http, backend, _ = service
    assert http.get('/automatic_recovery').json()['enabled'] is True
    result = http.post('/automatic_recovery', json={'enabled': False})
    assert result.status_code == 200 and not result.json()['enabled']
    assert result.json()['affects_future_attempts_only']
    assert not http.get('/health').json()['automatic_recovery_enabled']
    assert not http.get('/about?format=json').json()['automatic_recovery_enabled']
    assert not http.get('/state').json()['automatic_recovery_enabled']
    backend.phase, backend.reason = 'fault', 'measured_overspeed'
    assert http.post('/target_ee_pose', json={'position': [.45,0,.4]}).status_code == 409
    assert backend.recover_calls == 0
    assert http.post('/automatic_recovery', json={'enabled': True}).status_code == 200
    assert backend.recover_calls == 0  # Enabling does not itself perform recovery.
    assert http.post('/target_ee_pose', json={'position': [.45,0,.4]}).status_code == 202
    assert backend.recover_calls == 1


def test_toggle_while_active_does_not_stop_or_replace_target(service):
    http, backend, _ = service
    http.post('/target_ee_pose', json={'position': [.45,0,.4]})
    seq = backend.seq
    assert http.post('/automatic_recovery', json={'enabled': False}).status_code == 200
    assert backend.running and backend.seq == seq and backend.submit_calls == 1
    assert backend.recover_calls == 0


def test_toggle_during_lifecycle_operation_reports_busy_without_waiting(service):
    http, backend, _ = service
    engine = http.app.state.controller
    acquired, release = threading.Event(), threading.Event()
    def hold_lock():
        with engine.lock:
            acquired.set()
            release.wait(3)
    worker = threading.Thread(target=hold_lock);worker.start();assert acquired.wait(1)
    try:
        result = http.post('/automatic_recovery', json={'enabled': False})
        assert result.status_code == 409
        assert 'not changed' in result.json()['detail']
        assert engine.automatic_recovery is True
    finally:
        release.set();worker.join()
    assert backend.recover_calls == 0


@pytest.mark.parametrize('enabled', ['false', 0, 1, None])
def test_toggle_requires_boolean(service, enabled):
    http, backend, _ = service
    assert http.post('/automatic_recovery', json={'enabled': enabled}).status_code == 422
    assert http.get('/automatic_recovery').json()['enabled']
    assert backend.recover_calls == backend.start_calls == 0


def test_timeout_reset_requires_inactive_and_preserves_fault(service):
    http, backend, _ = service
    http.post('/command_timeout', json={'command_timeout_s': 100})
    http.post('/target_ee_pose', json={'position': [.45,0,.4]})
    assert http.post('/command_timeout/reset').status_code == 409
    assert backend.config.command_timeout == 100
    backend.running, backend.phase, backend.reason = False, 'fault', 'measured_overspeed'
    result = http.post('/command_timeout/reset')
    assert result.status_code == 200 and result.json()['command_timeout_s'] == 2
    assert backend.phase == 'fault' and backend.reason == 'measured_overspeed'
    assert backend.recover_calls == 0
    state = http.get('/state').json()
    assert state['command_timeout_s'] == 2 and state['last_timeout_request']['outcome'] == 'applied'


def test_read_only_cannot_enable_automatic_recovery(service):
    http, backend, _ = service
    backend.read_only = True
    assert http.post('/automatic_recovery', json={'enabled': False}).status_code == 200
    assert http.post('/automatic_recovery', json={'enabled': True}).status_code == 409
    assert not http.get('/automatic_recovery').json()['enabled']


def test_buttons_present_and_client_helpers_only_change_requested_setting(service):
    http, backend, _ = service
    page = http.get('/').text
    assert 'id="auto-recovery"' in page and 'id="reset-timeout"' in page
    assert "$('auto-recovery').onclick" in page
    assert "$('reset-timeout').onclick" in page
    client = CompliantClient();client.http.close();client.http = http
    assert client.set_automatic_recovery(False)['enabled'] is False
    assert client.get_automatic_recovery()['enabled'] is False
    assert client.reset_command_timeout()['command_timeout_s'] == 2
    assert backend.start_calls == backend.submit_calls == backend.recover_calls == 0
