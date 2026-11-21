"""Independent model checks and production-controller MuJoCo integration, no robot."""
from pathlib import Path
import hashlib

import numpy as np
import pytest

pytest.importorskip('mujoco')
native=pytest.importorskip('franka_compliant_control._twin_native')
from franka_compliant_control import _native
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.mujoco_twin import model_crosscheck, offline_model, preflight_path, simulation_trace, summarize, make_command
from franka_compliant_control.twin_experiment import ManualPosePolicy
from franka_compliant_control.twin_model import MujocoPlant, PlantParameters
from twin_helpers import home_fixture


@pytest.fixture(scope='module')
def home():return home_fixture()


def test_offline_kinematics_without_hardware_native_backend(home,monkeypatch):
    from franka_compliant_control.simulation.sdk import offline_kinematics
    from franka_compliant_control.twin_model import quaternion_matrix
    q=np.asarray(home['home']['joints'])
    sample=offline_model(home).sample(q,np.zeros(7),np.zeros(7),np.zeros(6))
    monkeypatch.delattr(_native,'OfflineKinematics',raising=False)
    pose,jac=offline_kinematics(home).evaluate(q)
    assert np.allclose(pose[:3,3],sample.position,atol=1e-12)
    assert np.allclose(pose[:3,:3],quaternion_matrix(sample.quaternion_xyzw),atol=1e-12)
    assert np.allclose(jac,sample.jacobian,atol=1e-12)
    assert preflight_path(home,ManualPosePolicy.from_home(home))['away_from_singularities']


def test_bridge_compiles_current_production_controller():
    assert hashlib.sha256(Path('native/controller.cpp').read_bytes()).hexdigest()==native.controller_source_sha256


def test_independent_mujoco_sdk_fk_jacobian_mass_gravity_coriolis(home):
    result=model_crosscheck(home,count=40)
    assert result['passed'],result
    assert result['max_errors']['jacobian_max_abs']<1e-10
    assert result['max_errors']['mass_max_abs']<1e-6
    assert result['max_errors']['gravity_Nm']<1e-10


def test_gravity_compensated_hold_has_no_double_coriolis_or_gravity(home):
    plant=MujocoPlant(home,PlantParameters(viscous=(0.,)*7,coulomb=(0.,)*7))
    model=offline_model(home)
    initial=plant.data.qpos.copy()
    for _ in range(1000):plant.step(model.gravity(plant.data.qpos))
    assert np.max(abs(plant.data.qpos-initial))<1e-9
    assert np.max(abs(plant.data.qvel))<1e-9


def test_sdk_filter_matches_documented_first_order_filter_and_rate_limit():
    a=2*np.pi*100*.001/(1+2*np.pi*100*.001)
    assert native.sdk_torque_filter(np.ones(7),np.zeros(7))==pytest.approx(np.ones(7)*a)
    assert np.max(native.sdk_torque_filter(np.ones(7)*100,np.zeros(7)))==pytest.approx(.999999)


def test_production_spring_torque_and_watchdog_in_offline_bridge(home):
    config=ControllerConfig.model_validate(home['controller_config']).native()
    config.posture_damping=config.posture_stiffness=0
    sdk=offline_model(home)
    sample=sdk.sample(home['home']['joints'],np.zeros(7),np.zeros(7),np.zeros(6))
    controller=native.OfflineController(config,sample,1,0)
    target={'position':sample.position+[.001,0,0],'quaternion_xyzw':sample.quaternion_xyzw}
    command=make_command(target,1)
    output=controller.step(sample,command,0,.001,.1)
    assert output.phase=='active'
    assert output.task_wrench==pytest.approx([.4,0,0,0,0,0],abs=1e-10)
    assert output.torque==pytest.approx(sample.jacobian.T@np.array([.4,0,0,0,0,0]),abs=1e-10)
    output=controller.step(sample,command,.11,.001,.1)
    assert output.phase=='stopping' and output.reason=='command_timeout'


def test_fixed_home_full_30cm_cycle_and_native_limits(home):
    policy=ManualPosePolicy.from_home(home)
    assert preflight_path(home,policy)['away_from_singularities']
    rows=list(simulation_trace(home,policy,PlantParameters()))
    result=summarize(rows,home,policy)
    assert result['completed'] and result['all_legs_settled']
    assert result['away_from_singularities'] and not result['hardware_stop_simulated']
    assert result['minimum_joint_position_margin_rad']>.3
    assert result['peak_commanded_force_N']<=home['controller_config']['commanded_force_limit']+1e-8
    assert result['peak_commanded_moment_Nm']<=home['controller_config']['commanded_moment_limit']+1e-8
    assert result['final_home_position_error_m']<.005
    assert rows[0]['policy']['position']==home['home']['position']
    assert rows[-1]['policy']['position']==home['home']['position']
    # Exactly 20 native ticks per held pose; no implicit native interpolation.
    assert rows[2020]['reference_position']==rows[2039]['reference_position']
    assert rows[2040]['reference_position']!=rows[2039]['reference_position']


def test_recorded_actions_use_native_tick_times_and_original_expiry(home):
    from itertools import islice
    policy=ManualPosePolicy.from_home(home)
    start=policy.sample(0)
    events=[{**start,'time_s':-.01,'deadline_s':.2},
            {**start,'position':(np.array(start['position'])+[.001,0,0]).tolist(),
             'time_s':.0259,'deadline_s':.04}]
    rows=list(islice(simulation_trace(home,policy,PlantParameters(),command_events=events),50))
    assert rows[25]['applied_seq']==1
    assert rows[26]['applied_seq']==2
    assert rows[26]['reference_position']==events[1]['position']
    assert rows[-1]['time_s']==pytest.approx(.04)
    assert rows[-1]['reason']=='command_timeout'
    assert rows[0]['external_wrench_valid'] is False


def test_nonfinite_and_invalid_sdk_inputs_rejected(home):
    model=offline_model(home)
    with pytest.raises(ValueError):model.sample([float('nan')]*7,[0]*7,[0]*7,[0]*6)
    with pytest.raises(ValueError):native.sdk_torque_filter([float('inf')]*7,[0]*7)
