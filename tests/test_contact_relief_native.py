"""Relief endpoint -> real native backend integration, using ONLY its toy sim backend.

Requires rebuilt manual-relief bindings; do not run against the older native module.
"""
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import create_app


def config():
    return ControllerConfig.model_validate_json(Path('configs/controller.fr3.auto.json').read_text())


def payload(state, **changes):
    p = state['position']
    return dict(mode='automatic', bias_wrench_tcp_base=[0.] * 6, bias_joint_positions=state['positions'],
                bias_source='synthetic known-free zero-wrench toy plant, not hardware calibration',
                bias_confirmed_free=True, workspace_confirmed_clear=True,
                workspace_lower=[x-.06 for x in p], workspace_upper=[x+.06 for x in p],
                **changes)


def wait_state(client, predicate, timeout=2.):
    end = time.monotonic()+timeout
    while time.monotonic()<end:
        state = client.get('/state').json()
        if predicate(state):
            return state
        time.sleep(.005)
    raise AssertionError(state)


def test_offline_initial_action_keeps_startup_deadline_and_fault_identity():
    from franka_compliant_control import _native, _twin_native as twin
    sample = twin.Sample()
    sample.q = [0., -.4, 0., -1.9, 0., 1.5, 0.]
    sample.position = [.45, 0., .4]
    sample.jacobian = np.c_[np.diag([.2]*3+[1.]*3), np.zeros(6)]
    sample.dynamics_valid = sample.wrench_tcp_valid = sample.relief_setup_valid = True
    options = _native.ContactReliefOptions()
    options.manual_stop = False
    options.bias_joint_positions = sample.q
    options.workspace_lower = [.39, -.06, .34]
    options.workspace_upper = [.51, .06, .46]
    options.bias_confirmed_free = options.workspace_confirmed_clear = True
    command = _native.Command()
    command.epoch = command.sequence = 1
    command.relieve_contact = True
    command.relief_options = options
    command.action_timeout = .21
    ctrl = twin.OfflineController(config().native(), sample, 1, 0., command, .21)
    # A later input/deadline cannot extend the immutable startup lease.
    output = ctrl.step(sample, _native.Command(), .3, .001, 100.)
    assert output.relief_mode and output.phase == 'stopping'
    assert output.reason == 'contact_relief_timeout' and output.relief_deadline == .21
    # A timely successful release may use the independent stop-settle interval.
    near_deadline = twin.OfflineController(config().native(), sample, 1, 0., command, .21)
    near_deadline.step(sample, command, 0., .001, .21)
    output = near_deadline.step(sample, command, .201, .001, .21)
    assert output.phase == 'stopping' and output.relief_released
    output = near_deadline.step(sample, command, .302, .001, .21)
    assert output.phase == 'disarmed' and output.relief_released
    assert output.reason == 'contact_relief_released'
    command.action_timeout = 10
    ctrl = twin.OfflineController(config().native(), sample, 1, 0., command, 10.)
    sample.external_wrench_tcp_base = [40., 0, 0, 0, 0, 0]
    output = ctrl.step(sample, _native.Command(), .001, .001, 10.)
    assert output.relief_mode and output.phase == 'stopping'
    assert output.stop_trigger == 'contact_force_limit' and not output.relief_released


def test_native_free_residual_holds_then_disarms_without_pose_jump():
    with TestClient(create_app(config=config(), backend_kind='sim', automatic_recovery=False)) as http:
        initial = http.get('/state').json()
        assert initial['external_wrench_tcp_valid'] and initial['contact_relief_available']
        result = http.post('/relieve_contact', json=payload(initial, upward_velocity=.005))
        assert result.status_code == 202, result.text
        state = wait_state(http, lambda s: not s['running'])
        assert state['phase'] == 'disarmed' and state['reason'] == 'contact_relief_released'
        assert state['contact_relief_status'] == 'completed' and state['contact_relief_released']
        assert state['contact_relief_mode'] == 'automatic' and state['contact_relief_bias_used']
        assert state['position'] == pytest.approx(initial['position'], abs=1e-10)
        assert state['reference_position'] == pytest.approx(initial['position'], abs=1e-10)
        assert state['friction_compensation_torques'] == [0.]*7
        assert not http.get('/target_ee_pose').json()['valid']


@pytest.mark.parametrize('explicit_box', [False, True])
def test_native_manual_no_load_does_not_auto_complete_and_user_stop_disarms(explicit_box):
    with TestClient(create_app(config=config(), backend_kind='sim', automatic_recovery=False)) as http:
        initial = http.get('/state').json()
        assert initial['contact_relief_mode'] == 'none' and not initial['contact_relief_bias_used']
        data = {'workspace_confirmed_clear': True}
        if explicit_box:
            data.update(workspace_lower=[x-.06 for x in initial['position']],
                        workspace_upper=[x+.06 for x in initial['position']])
        result = http.post('/relieve_contact', json=data)
        assert result.status_code == 202, result.text
        ack = result.json()
        assert ack['relief_options']['mode'] == 'manual' and not ack['automatic_release_detection']
        assert ack['bias_source'] is None and not ack['relief_options']['bias_confirmed_free']
        for name in ('bias_wrench_tcp_base', 'bias_joint_positions', 'bias_source'):
            assert ack['relief_options'][name] is None
        state = wait_state(http, lambda s: s['contact_relief_active'] and s['phase'] == 'active')
        # Longer than the automatic 200 ms release hold, with no external load.
        end = time.monotonic() + .35
        while time.monotonic() < end:
            state = http.get('/state').json()
            assert state['running'] and state['contact_relief_active']
            assert state['contact_relief_status'] == 'following_wrench'
            assert state['contact_relief_mode'] == 'manual'
            assert not state['contact_relief_released'] and not state['contact_relief_bias_used']
            time.sleep(.01)
        assert state['position'] == pytest.approx(initial['position'], abs=1e-10)
        assert state['friction_compensation_torques'] == [0.]*7
        stopped = http.post('/stop', params={'wait_s': 1}).json()
        assert stopped['stopped'], stopped
        state = stopped['state']
        assert not state['running'] and not state['contact_relief_active']
        assert state['phase'] == 'disarmed' and state['contact_relief_status'] == 'stopped'
        assert state['contact_relief_mode'] == 'manual'
        assert not state['contact_relief_released'] and not state['contact_relief_bias_used']
        assert not http.get('/target_ee_pose').json()['valid']


def test_native_action_owns_session_and_stop_cancels_it():
    with TestClient(create_app(config=config(), backend_kind='sim', automatic_recovery=False)) as http:
        initial = http.get('/state').json()
        # Constant robot-on-environment wrench; toy dynamics subtract this load.
        backend = http.app.state.controller.backend
        backend.inject_wrench([5.,0,0,0,0,0])
        result = http.post('/relieve_contact', json=payload(initial))
        assert result.status_code == 202, result.text
        state = wait_state(http, lambda s: s['contact_relief_active'] and s['phase']=='active')
        assert state['friction_compensation_torques'] == [0.]*7
        assert http.post('/target_ee_pose', json={'position':initial['position']}).status_code == 409
        assert http.post('/go_home').status_code == 409
        assert http.post('/relieve_contact', json=payload(initial)).status_code == 409
        backend.inject_wrench([0.]*6)
        stopped = http.post('/stop', params={'wait_s':1}).json()
        assert stopped['stopped'], stopped
        assert stopped['state']['contact_relief_status'] == 'cancelled'
        assert not stopped['state']['contact_relief_released']


def test_bias_posture_mismatch_rejects_before_control():
    with TestClient(create_app(config=config(), backend_kind='sim', automatic_recovery=False)) as http:
        initial = http.get('/state').json()
        data = payload(initial)
        data['bias_joint_positions'] = list(initial['positions'])
        data['bias_joint_positions'][0] += .201
        response = http.post('/relieve_contact', json=data)
        assert response.status_code == 409, response.text
        assert 'bias_posture_mismatch' in response.text
        state = http.get('/state').json()
        assert not state['running'] and state['phase']=='disarmed'
        assert state['applied_seq'] == 0


def test_bad_loaded_reading_faults_and_cannot_auto_recover():
    with TestClient(create_app(config=config(), backend_kind='sim', automatic_recovery=False)) as http:
        initial = http.get('/state').json()
        backend = http.app.state.controller.backend
        response = http.post('/relieve_contact', json=payload(initial))
        assert response.status_code == 202, response.text
        backend.inject_wrench([41.,0,0,0,0,0])
        state = wait_state(http, lambda s: s['phase'] in ('stopping','fault'))
        backend.inject_wrench([0.]*6)
        state = wait_state(http, lambda s: not s['running'])
        assert state['phase']=='fault' and state['stop_trigger']=='contact_force_limit'
        assert state['contact_relief_status']=='aborted' and not state['contact_relief_released']
        assert http.post('/automatic_recovery', json={'enabled':True}).status_code == 200
        response = http.post('/target_ee_pose', json={'position':state['position']})
        assert response.status_code == 409 and 'explicit operator recovery' in response.text
        assert http.get('/state').json()['phase']=='fault'
