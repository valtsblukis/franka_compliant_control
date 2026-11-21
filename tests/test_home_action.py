"""Home-action and timeout-display contracts using the recording backend only."""
import math

import pytest

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.models import HomeRequest
from test_policy_http import service


def test_go_home_has_own_deadline_and_completion_parameters(service):
    http, backend, clock = service
    http.post('/command_timeout', json={'command_timeout_s': 1e12})
    result = http.post('/go_home')
    assert result.status_code == 200
    result = result.json()
    assert result['disarm_on_completion'] and result['action'] == 'go_home'
    assert result['home_options']['timeout_s'] == 30
    assert 29.9 < result['deadline_s']-clock.now() <= 30
    command = backend.last_command
    assert command.finish_at_target and command.action_timeout == 30
    assert command.completion_joint_tolerance == .02
    assert command.completion_position_tolerance == .01
    assert command.completion_rotation_tolerance == pytest.approx(math.radians(2))
    assert command.completion_settle_time == .2
    assert http.get('/command_timeout').json()['command_timeout_s'] == 1e12


def test_home_action_can_outlive_short_stream_timeout_without_extending_on_repeat(service):
    http, backend, clock = service
    http.post('/command_timeout', json={'command_timeout_s': .1})
    first = http.post('/go_home').json()
    clock.offset += .5
    assert backend.is_running()
    repeated = http.post('/go_home').json()
    assert repeated['already_in_progress']
    assert repeated['command_seq'] == first['command_seq']
    assert repeated['deadline_s'] == first['deadline_s']
    assert backend.start_calls == backend.submit_calls == 1
    assert http.post('/go_home', json={'timeout_s': 60}).status_code == 409
    assert backend.submit_calls == 1


def test_ordinary_setpoint_supersedes_home_action(service):
    http, backend, _ = service
    http.post('/go_home')
    result = http.post('/target_ee_pose', json={'position': [.5,0,.4]})
    assert result.status_code == 202
    assert result.json()['action'] == 'setpoint'
    assert not backend.last_command.finish_at_target
    assert http.get('/health').json()['homing_status'] == 'superseded'
    assert backend.start_calls == 1  # No restart for cancellation.


def test_home_timeout_and_explicit_stop_do_not_claim_home_was_reached(service):
    http, backend, clock = service
    http.post('/go_home', json={'timeout_s': 1})
    clock.offset += 1.1
    health = http.get('/health').json()
    assert not health['running'] and health['reason'] == 'home_timeout'
    assert health['homing_status'] == 'timed_out' and not health['home_reached']


@pytest.mark.parametrize('options', [{'timeout_s': 0}, {'timeout_s': .1, 'settle_time_s': .2},
    {'joint_tolerance_rad': -1}, {'position_tolerance_m': 0}, {'rotation_tolerance_rad': -1},
    {'settle_time_s': 0}, {'extra': True}])
def test_invalid_home_options_are_rejected_before_start(service, options):
    http, backend, _ = service
    assert http.post('/go_home', json=options).status_code == 422
    assert backend.start_calls == 0


def test_current_timeout_display_is_not_the_last_requested_value(service):
    http, backend, _ = service
    http.post('/command_timeout', json={'command_timeout_s': 10})
    http.post('/target_ee_pose', json={'position': [.45,0,.4]})
    assert http.post('/command_timeout', json={'command_timeout_s': 20}).status_code == 409
    view = http.get('/status').json()
    assert view['state']['command_timeout_s'] == 10
    assert view['state']['last_timeout_request']['command_timeout_s'] == 20
    assert view['state']['last_timeout_request']['outcome'] == 'rejected'
    html = http.get('/').text
    assert 'Configured streaming timeout:' in html
    assert 'Last timeout request:' in html
    assert 'homing_status' in html


def test_client_go_home_forwards_completion_options(service):
    http, backend, _ = service
    client = CompliantClient();client.http.close();client.http = http
    result = client.go_home(timeout_s=15, joint_tolerance_rad=.03)
    assert result['home_options']['timeout_s'] == 15
    assert backend.last_command.completion_joint_tolerance == .03
    client.stop()
