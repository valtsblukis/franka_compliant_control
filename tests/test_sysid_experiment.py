"""Calibration policy/guards and consent/lifecycle via recording clients only."""
from copy import deepcopy
import json

import numpy as np
import pytest
pytest.importorskip('mujoco')
from franka_compliant_control.sysid_experiment import build_plan,parse_plan,acquire,Envelope
from franka_compliant_control.twin_experiment import ManualPosePolicy,digest,write_json_new
from test_twin_real import Clock,Robot
from twin_helpers import home_fixture


@pytest.fixture
def trial(tmp_path):
    h=home_fixture();clock=Clock()
    base=Robot(h,clock,ManualPosePolicy.from_home(h))
    state=base.state();state['external_wrench']=[0]*6
    plan=build_plan(h,state,'training');policy,envelope=parse_plan(h,plan)
    class Recorded(Robot):
        def health(self):return {'running':self.running,'phase':'active' if self.running else 'disarmed','automatic_recovery_enabled':self.auto}
        def state(self):return {**super().state(),'external_wrench':[0.]*6}
        def set_target_ee_pose(self,p,q,*,valid_for_s):
            assert valid_for_s<=1
            return super().set_target_ee_pose(p,q)
    robot=Recorded(h,clock,policy)
    sim=tmp_path/'sim';sim.mkdir()
    write_json_new(sim/'manifest.json',{'plan_sha256':digest(plan)})
    write_json_new(sim/'summary.json',{'ready_for_review':True})
    return h,plan,policy,envelope,clock,robot,sim


def test_forward_up_policy_and_narrow_base_guard(trial):
    h,plan,p,e,*_=trial
    assert p.duration_s==39
    assert p.home_position==tuple(h['home']['position'])
    assert e.joint_lower[0]==pytest.approx(h['home']['joints'][0]-.05)
    assert e.joint_upper[0]==pytest.approx(h['home']['joints'][0]+.05)
    points=np.array([p.sample(t)['position'] for t in np.arange(0,p.duration_s+.001,.02)])
    assert points[:,0].min()>=h['home']['position'][0]-1e-12
    assert points[:,2].min()>=h['home']['position'][2]-1e-12
    assert np.ptp(points[:,1])<1e-12
    assert p.sample(0)['position']==h['home']['position']
    for field in ['twist','acceleration']:assert field not in p.sample(10)
    bad=np.array(h['home']['joints']);bad[0]+=.051
    with pytest.raises(RuntimeError,match='base range'):
        e.check(bad,np.zeros(7),h['home']['position'],h['home']['quaternion_xyzw'],np.zeros(6))


def test_validation_has_distinct_frozen_waypoints(trial):
    h,plan,p,e,clock,robot,_=trial
    val=build_plan(h,robot.state(),'validation');vp,_=parse_plan(h,val)
    assert val['role']=='validation' and val['policy_sha256']!=plan['policy_sha256']
    assert vp.home_position==p.home_position
    assert any(abs(w[1][0]-p.waypoints[0][1][0])>0 and abs(w[1][2]-p.waypoints[0][1][2])>0 for w in vp.waypoints[1:])


def test_customer_recipe_is_explicit_and_cannot_put_goals_outside_envelope(trial):
    from franka_compliant_control.sysid_experiment import front_recipe
    h,plan,p,e,clock,robot,sim=trial
    recipe=front_recipe();recipe['center_offset_m']=[.012,0,.025]
    other=build_plan(h,robot.state(),'validation',recipe=recipe)
    op,_=parse_plan(h,other)
    assert op.home_position==p.home_position
    assert op.waypoints[0][1]!=p.waypoints[0][1]
    recipe['center_offset_m']=[1,0,0]
    with pytest.raises(ValueError,match='outside'):
        build_plan(h,robot.state(),'training',recipe=recipe)


def test_acquisition_requires_consent_and_stops_after_schedule(trial,tmp_path):
    h,plan,p,e,clock,robot,sim=trial
    calls=[]
    def factory(*a,**k):calls.append('connect');return robot
    with pytest.raises(ValueError,match='Explicit'):
        acquire(h,plan,sim,tmp_path/'no',client_factory=factory)
    assert not calls
    result=acquire(h,plan,sim,tmp_path/'run',execute=True,clear_path=True,
                   client_factory=factory,clock=clock.now,sleep=clock.sleep)
    assert result['completed'] and result['stop_confirmed']
    assert robot.calls[-1]=='stop' and robot.closed
    assert robot.http.headers['Connection']=='close'


def test_base_excursion_aborts_without_recovery_or_restart(trial,tmp_path):
    h,plan,p,e,clock,robot,sim=trial
    original=robot.state
    def changed():
        state=original()
        if clock.t>.1:
            state['positions']=list(state['positions']);state['positions'][0]=e.joint_upper[0]+.01
        return state
    robot.state=changed
    with pytest.raises(RuntimeError,match='base range'):
        acquire(h,plan,sim,tmp_path/'run',execute=True,clear_path=True,
                client_factory=lambda *a,**k:robot,clock=clock.now,sleep=clock.sleep)
    assert robot.calls[-1]=='stop' and robot.closed
    assert (tmp_path/'run/failure.json').exists()
