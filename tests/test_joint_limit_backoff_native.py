"""Real native bridge/backend tests; only offline samples and the toy sim, never hardware."""
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from franka_compliant_control import _native, _twin_native as twin
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import create_app


def config(upper7=.03, **changes):
    data = ControllerConfig.model_validate_json(Path('configs/controller.fr3.auto.json').read_text()).model_dump()
    if upper7 is not None:
        data['joint_upper'][6] = upper7
    data.update(changes)
    return ControllerConfig.model_validate(data)


def wait_state(client, predicate, timeout=3.):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = client.get('/state').json()
        if predicate(state):
            return state
        time.sleep(.005)
    raise AssertionError(state)


def offline_sample():
    s = twin.Sample()
    s.q = [0., -.4, 0., -1.9, 0., 1.5, 0.]
    s.position = [.45, 0., .4]
    s.jacobian = np.c_[np.diag([.2]*3 + [1.]*3), np.zeros(6)]
    s.mass = 3*np.eye(7)
    s.dynamics_valid = s.wrench_tcp_valid = s.relief_setup_valid = True
    return s


def offline_command(timeout=20.):
    r = _native.JointLimitBackoffOptions()
    r.confirm_clear_path = True
    cmd = _native.Command()
    cmd.mode = _native.Mode.Joint
    cmd.epoch = cmd.sequence = 1
    cmd.backoff_joint_limits = True
    cmd.backoff_options = r
    cmd.action_timeout = timeout
    return cmd


def test_offline_plans_native_targets_without_client_joint_values_and_keeps_deadline():
    s = offline_sample();cmd = offline_command(.21)
    ctrl = twin.OfflineController(config(-.02).native(), s, 1, 0., cmd, .21)
    out = ctrl.step(s, _native.Command(), .3, .001, 100.)
    assert out.backoff_mode and out.phase == 'stopping'
    assert out.reason == 'joint_limit_backoff_timeout' and out.backoff_deadline == .21
    assert out.backoff_directions.tolist() == [0.]*6 + [-1.]
    assert out.backoff_target_joints.tolist() == pytest.approx(list(s.q[:6]) + [-.12])
    assert out.friction_compensation_torque.tolist() == [0.]*7


@pytest.mark.parametrize('kind', ['start_moved', 'became_loaded', 'previous_torque', 'physical_bound'])
def test_offline_first_callback_rechecks_fresh_state_before_motion(kind):
    s = offline_sample();cmd = offline_command()
    ctrl = twin.OfflineController(config().native(), s, 1, 0., cmd, 20.)
    expected = ''
    if kind == 'start_moved':
        q = np.array(s.q);q[0] += .0011;s.q = q
        expected = 'joint_limit_backoff_start_changed'
    if kind == 'became_loaded':
        s.external_wrench = [15., 0., 0., 0., 0., 0.]
        s.external_wrench_tcp_base = [15., 0., 0., 0., 0., 0.]
        expected = 'contact_force_limit'
    if kind == 'previous_torque':
        s.previous_torque = [0.]*6 + [1.1]
        expected = 'invalid_robot_state'
    if kind == 'physical_bound':
        q = np.array(s.q);q[6] = 3.051;s.q = q
        # Both specified range and the action's outward envelope are violated;
        # the disqualifying outward fault must dominate the eligible limit one.
        expected = 'joint_limit_backoff_outward_motion'
    out = ctrl.step(s, cmd, .001, .001, 20.)
    assert out.phase == 'stopping' and not out.backoff_reached
    assert out.reason == expected


def test_native_noop_never_arms_or_changes_recovery_policy():
    with TestClient(create_app(config=config(None), backend_kind='sim', automatic_recovery=True)) as http:
        initial = http.get('/state').json()
        response = http.post('/backoff_joint_limits', json={'confirm_clear_path': True})
        assert response.status_code == 409 and 'no backoff needed' in response.text
        state = http.get('/state').json()
        assert state['joint_limit_backoff_available'] and not state['running']
        assert state['phase'] == 'disarmed' and state['epoch'] == initial['epoch']
        assert state['applied_seq'] == initial['applied_seq']
        assert http.get('/health').json()['automatic_recovery_enabled']


@pytest.mark.parametrize('upper7', [.03, -.02])
def test_native_backoff_near_or_outside_app_bound_completes_without_changing_limits(upper7):
    c = config(upper7)
    with TestClient(create_app(config=c, backend_kind='sim', automatic_recovery=True)) as http:
        initial = http.get('/state').json()
        response = http.post('/backoff_joint_limits', json={'confirm_clear_path': True})
        assert response.status_code == 202, response.text
        ack = response.json()
        assert ack['accepted_not_reached'] and ack['disarm_on_completion']
        state = wait_state(http, lambda s: s['joint_limit_backoff_active'])
        assert state['joint_limit_backoff_selected'] == [7]
        assert state['joint_limit_backoff_target_positions'][:6] == pytest.approx(initial['positions'][:6])
        assert state['joint_limit_backoff_target_positions'][6] == pytest.approx(min(-.1, upper7-.1))
        assert state['joint_limit_backoff_torque_limits'][6] == 1.
        assert state['reference_policy'] == 'native_joint_limit_backoff'
        assert http.post('/target_ee_pose', json={'position': initial['position']}).status_code == 409
        assert http.post('/go_home').status_code == 409
        assert http.post('/relieve_contact', json={'workspace_confirmed_clear': True}).status_code == 409
        assert http.post('/backoff_joint_limits', json={'confirm_clear_path': True}).status_code == 409
        assert http.get('/target_joint_state').json()['positions'] is None
        state = wait_state(http, lambda s: not s['running'], timeout=12.)
        assert state['phase'] == 'disarmed', state
        assert state['reason'] == 'joint_limit_backoff_reached'
        assert state['joint_limit_backoff_reached'] and state['joint_limit_backoff_status'] == 'completed'
        assert state['positions'][6] <= upper7-.08
        assert state['positions'][:6] == pytest.approx(initial['positions'][:6], abs=.01)
        assert state['friction_compensation_torques'] == [0.]*7
        assert http.app.state.controller.config.joint_upper[6] == upper7
        assert http.get('/health').json()['automatic_recovery_enabled']
        # Its fresh state now needs no additional retreat.
        response = http.post('/backoff_joint_limits', json={'confirm_clear_path': True})
        assert response.status_code == 409 and 'no backoff needed' in response.text


def test_native_timeout_stays_faulted_and_normal_target_cannot_autorecover_it():
    with TestClient(create_app(config=config(), backend_kind='sim', automatic_recovery=True)) as http:
        response = http.post('/backoff_joint_limits', json={'confirm_clear_path': True, 'timeout_s': .21})
        assert response.status_code == 202, response.text
        state = wait_state(http, lambda s: not s['running'])
        assert state['phase'] == 'fault' and state['reason'] == 'joint_limit_backoff_timeout'
        assert not state['joint_limit_backoff_reached'] and state['joint_limit_backoff_status'] == 'aborted'
        response = http.post('/target_ee_pose', json={'position': state['position']})
        assert response.status_code == 409 and 'explicit operator recovery' in response.text
        assert http.get('/state').json()['phase'] == 'fault'


def test_native_stop_cancels_and_allows_an_explicit_new_request_only():
    with TestClient(create_app(config=config(), backend_kind='sim', automatic_recovery=False)) as http:
        assert http.post('/backoff_joint_limits', json={'confirm_clear_path': True}).status_code == 202
        wait_state(http, lambda s: s['joint_limit_backoff_status'] == 'backing_off')
        result = http.post('/stop', params={'wait_s': 1.}).json()
        assert result['stopped']
        assert result['state']['joint_limit_backoff_status'] == 'cancelled'
        assert not result['state']['joint_limit_backoff_reached']
        response = http.post('/backoff_joint_limits', json={'confirm_clear_path': True})
        assert response.status_code == 202, response.text


def test_native_replaces_only_eligible_local_fault_without_sdk_recovery():
    # Create a local application-limit fault using the existing toy plant only.
    # The app bound is narrowed, not the FR3 specified physical bound.
    c = config(stop_velocity=.005, friction_compensation_torque=0., command_timeout=2.)
    with TestClient(create_app(config=c, backend_kind='sim', automatic_recovery=True)) as http:
        initial = http.get('/state').json()
        joints = initial['positions'].copy();joints[6] = .029
        response = http.post('/target_joint_state', json={'positions': joints, 'velocities': [0.]*6 + [.2]})
        assert response.status_code == 200, response.text
        state = wait_state(http, lambda s: not s['running'], timeout=3.)
        assert state['phase'] == 'fault' and state['reason'] == 'joint_limit', state
        assert state['positions'][6] < .08, state
        response = http.post('/backoff_joint_limits', json={'confirm_clear_path': True})
        assert response.status_code == 202, response.text
        state = wait_state(http, lambda s: s['joint_limit_backoff_active'])
        assert state['joint_limit_backoff_previous_reason'] == 'joint_limit'
        assert http.get('/health').json()['automatic_recovery_enabled']
        # Explicit cancellation, not an automatic retry or SDK recovery.
        assert http.post('/stop', params={'wait_s': 1.}).json()['stopped']
