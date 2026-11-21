"""Two-second defaults and HTTP leases; recording backend, no plant/robot."""
from pathlib import Path

import pytest

from franka_compliant_control import _native
from franka_compliant_control.client import CompliantClient
from franka_compliant_control.models import ControllerConfig
from test_policy_http import service


def test_native_schema_and_current_profiles_use_two_seconds():
    assert _native.Config().command_timeout == ControllerConfig().command_timeout == 2
    for name in ('auto', 'joint', 'osc', 'osc.progressive', 'osc.tuning'):
        config = ControllerConfig.model_validate_json(Path(f'configs/controller.fr3.{name}.json').read_text())
        assert config.command_timeout == config.native().command_timeout == 2
    for value in (2.001, 10, 1e12):
        assert ControllerConfig(command_timeout=value).native().command_timeout == value
    for value in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            ControllerConfig(command_timeout=value)


def test_default_setpoint_survives_short_policy_gap_and_expires_at_two_seconds(service):
    http, backend, clock = service
    response = http.post('/target_ee_pose', json={'position': [.45,0,.4]})
    assert response.status_code == 202
    target = response.json()
    assert 1.99 < target['deadline_s']-target['accepted_monotonic_s'] <= 2
    clock.offset += .5
    assert http.get('/health').json()['target_valid']
    assert backend.running
    clock.offset += 1.6
    assert http.get('/health').json()['reason'] == 'command_timeout'
    assert not backend.running


def test_timeout_api_and_shorter_explicit_lease(service):
    http, _, clock = service
    info = http.get('/command_timeout').json()
    assert info['command_timeout_s'] == info['default_s'] == 2
    assert info['maximum_s'] is None
    assert http.post('/command_timeout', json={'command_timeout_s': 2}).status_code == 200
    assert http.post('/command_timeout', json={'command_timeout_s': 2.001}).status_code == 200
    assert http.post('/target_ee_pose', json={'position': [.45,0,.4], 'valid_for_s': .1}).status_code == 202
    clock.offset += .2
    assert not http.get('/health').json()['target_valid']


@pytest.mark.parametrize('seconds', [2.001, 10, 1e12])
def test_longer_timeouts_are_accepted_and_the_actual_request_is_visible(service, seconds):
    http, backend, _ = service
    response = http.post('/command_timeout', json={'command_timeout_s': seconds})
    assert response.status_code == 200 and response.json()['command_timeout_s'] == seconds
    request = http.get('/state').json()['last_timeout_request']
    assert request['command_timeout_s'] == seconds and request['outcome'] == 'applied'
    assert backend.arm_calls == backend.submit_calls == 0


def test_rejected_numeric_timeout_is_logged_without_other_payload_fields(service):
    http, _, _ = service
    response = http.post('/command_timeout', json={'command_timeout_s': -10, 'unrelated': 'secret-value'})
    assert response.status_code == 422
    status = http.get('/status').json()
    assert 'command_timeout_s=-10' in status['last_rejection']['error']
    assert 'secret-value' not in str(status)
    assert status['state']['last_timeout_request']['outcome'] == 'rejected'


def test_client_accepts_two_second_lifetime_without_resending(monkeypatch):
    client = CompliantClient();calls = []
    def request(method, path, **kwargs):
        calls.append((path, kwargs))
        return {'command_timeout_s': 2}
    monkeypatch.setattr(client, '_request', request)
    client.set_command_timeout(2)
    client.send_pose([.45,0,.4], valid_for_s=2)
    assert calls[-1][1]['json']['valid_for_s'] == 2
    client.set_command_timeout(10)
    client.send_pose([.45,0,.4], valid_for_s=10)
    assert calls[-1][1]['json']['valid_for_s'] == 10
    with pytest.raises(ValueError):
        client.send_pose([.45,0,.4], valid_for_s=float('inf'))
    client.close()
