"""Wrench point/sign conventions and approximate observer algebra; no robot I/O."""
import numpy as np
import pytest
pytest.importorskip('scipy')
from franka_compliant_control.wrench_observer import (
    canonical_franka_wrench,projected_wrench,tcp_to_spatial,spatial_to_tcp,WrenchObserver,score_wrenches)


def artifact(tc=0):
    return {'schema':'fcc-wrench-observer-v1','q_reference':[0.]*7,'resistance_gain':[.4]*7,
            'joint_offset':[0.]*7,'joint_position_slope':[0.]*7,'time_constant_s':tc,
            'nominal_contact_gain':1.,'rotation_scale':.2,'minimum_singular_value':.05}


def test_moment_reference_point_shift_and_round_trip():
    p=[.4,0,.5];w=np.array([0,2,0,0,0,0.])
    raw=tcp_to_spatial(w,p)
    assert raw==pytest.approx([0,2,0,-1,0,.8])
    assert spatial_to_tcp(raw,p)==pytest.approx(w)
    s={'external_wrench_stiffness':w.tolist(),'quaternion_xyzw':[0,0,0,1]}
    assert canonical_franka_wrench(s,stiffness_at_tcp=True)==pytest.approx(w)
    with pytest.raises(ValueError,match='coincidence'):canonical_franka_wrench(s)


def test_stiffness_axes_rotate_both_force_and_moment_without_spurious_translation():
    s={'external_wrench_stiffness':[1,0,0,0,1,0],
       'quaternion_xyzw':[0,0,np.sin(np.pi/4),np.cos(np.pi/4)]}
    assert canonical_franka_wrench(s,stiffness_at_tcp=True)==pytest.approx([0,1,0,-1,0,0],abs=1e-12)


def test_projection_preserves_robot_on_environment_sign_and_rejects_singular_state():
    j=np.c_[np.eye(6),np.zeros(6)];w=np.arange(1.,7.)
    result,sigma=projected_wrench(j,j.T@w)
    assert result==pytest.approx(w) and sigma==pytest.approx(.2)
    with pytest.raises(ValueError,match='conditioned'):projected_wrench(np.zeros((6,7)),np.zeros(7))


def test_free_space_attenuation_does_not_silently_attenuate_true_contact():
    o=WrenchObserver(artifact());j=np.c_[np.eye(6),np.zeros(6)]
    free=np.array([2,0,0,0,0,0,0.]);contact=np.array([3,0,0,0,0,0,0.])
    out=o.step(np.zeros(7),free,j,np.array([.4,0,.5]),.02,contact_joint_torque=contact)
    assert out['external_wrench_tcp_base'][0]==pytest.approx(.4*2+3)
    assert out['external_wrench'][4]==pytest.approx(.5*(.4*2+3))
    assert not out['true_force_calibrated']


def test_effective_lag_is_causal_and_dt_invariant():
    j=np.c_[np.eye(6),np.zeros(6)];z=np.zeros(7)
    one=WrenchObserver(artifact(.1));many=WrenchObserver(artifact(.1))
    for o in [one,many]:o.step(z,z,j,np.zeros(3),.02)
    a=one.step(z,np.ones(7),j,np.zeros(3),.02)['external_joint_torques']
    for _ in range(20):b=many.step(z,np.ones(7),j,np.zeros(3),.001)['external_joint_torques']
    assert a==pytest.approx(b)
    assert 0<a[0]<.4


def test_relative_tolerance_has_absolute_floor_and_sign_diagnostics():
    observed=np.array([[0,0,0,0,0,0],[2,0,0,0,1,0.]])
    predicted=np.array([[.49,0,0,.09,0,0],[2.5,0,0,0,1.2,0.]])
    score=score_wrenches(predicted,observed)
    assert score['force']['fraction_within_tolerance']==1
    assert score['moment']['fraction_within_tolerance']==1
    assert score['force_axis_sign_agreement'][0]==1
    assert score['force_direction']['fraction_within_30_degrees']==1


@pytest.mark.parametrize('changes',[{'resistance_gain':[-1]*7},{'joint_offset':[float('nan')]*7},
                                    {'time_constant_s':-1},{'nominal_contact_gain':0}])
def test_invalid_observer_parameters(changes):
    with pytest.raises(ValueError):WrenchObserver(artifact()|changes)


def test_observer_shadow_calculation_does_not_change_native_torque_or_plant():
    pytest.importorskip('mujoco');pytest.importorskip('franka_compliant_control._twin_native')
    from itertools import islice
    from twin_helpers import home_fixture
    from franka_compliant_control.mujoco_twin import simulation_trace
    from franka_compliant_control.twin_model import PlantParameters
    from franka_compliant_control.twin_experiment import ManualPosePolicy
    home=home_fixture();policy=ManualPosePolicy.from_home(home,offset_m=.03)
    base=list(islice(simulation_trace(home,policy,PlantParameters()),1700))
    observed=list(islice(simulation_trace(home,policy,PlantParameters(),observer_config=artifact(.1)),1700))
    assert np.array([r['positions'] for r in observed])==pytest.approx(np.array([r['positions'] for r in base]),abs=1e-12)
    assert np.array([r['commanded_joint_torques'] for r in observed])==pytest.approx(np.array([r['commanded_joint_torques'] for r in base]),abs=1e-12)
    assert all(r['external_wrench_valid'] for r in observed)
    for row in observed[::100]:
        assert spatial_to_tcp(row['external_wrench'],row['position'])==pytest.approx(row['external_wrench_tcp_base'])
