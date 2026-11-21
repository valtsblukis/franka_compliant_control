"""Future hardware runner tested ONLY with an in-memory clock/client."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')
from franka_compliant_control.twin_experiment import ManualPosePolicy, digest, write_json_new
from franka_compliant_control.twin_real import run_real, validated_plan
from twin_helpers import home_fixture


class Clock:
    t=0.
    def now(self):return self.t
    def sleep(self,seconds):self.t+=seconds


class Robot:
    def __init__(self,home,clock,policy):
        self.home,self.clock,self.policy=home,clock,policy
        self.running=False;self.calls=[];self.closed=False;self.auto=False;self.fault_after=None
        self.http=SimpleNamespace(headers={})
    def health(self):return {'automatic_recovery_enabled':self.auto}
    def get_config(self):return deepcopy(self.home['controller_config'])
    def get_urdf(self):return {'urdf':self.home['urdf']}
    def refresh_state(self):self.calls.append('refresh')
    def state(self):
        fault=self.fault_after is not None and self.clock.t>self.fault_after
        position=self.policy.sample(self.clock.t)['position'] if self.running else self.home['home']['position']
        return {'backend':'robot','running':self.running and not fault,
                'phase':'fault' if fault else 'active' if self.running else 'disarmed',
                'robot_mode':'Move' if self.running else 'Idle','robot_has_errors':False,
                'sample_valid':True,'kinematics_valid':True,'sample_age_s':0.,
                'position':position,'quaternion_xyzw':self.home['home']['quaternion_xyzw'],
                'positions':self.home['home']['joints'],'velocities':[0.]*7,'twist':[0.]*6,
                'minimum_singular_value':.1,'osc_regularized':False,
                'server_monotonic_s':1000+self.clock.t,'robot_time_s':self.clock.t+1}
    def set_target_ee_pose(self,p,q):
        self.calls.append(('target',list(p),list(q)));self.running=True
        return {'accepted_monotonic_s':1000+self.clock.t}
    def stop(self,wait_s):self.calls.append('stop');self.running=False;return {'stopped':True}
    def close(self):self.closed=True


@pytest.fixture
def setup(tmp_path):
    home=home_fixture()
    policy=ManualPosePolicy.from_home(home,move_s=.2,dwell_s=.2,initial_hold_s=.2)
    sim=tmp_path/'sim';sim.mkdir()
    write_json_new(sim/'manifest.json',{'backend':'mujoco','home_sha256':home['sha256'],
        'controller_config':home['controller_config'],'policy':policy.manifest(),
        'policy_sha256':digest(policy.manifest()),'model_crosscheck':{'passed':True},
        'controller_source_sha256':'recording-test'})
    write_json_new(sim/'summary.json',{'ready_for_operator_review':True})
    clock=Clock();robot=Robot(home,clock,policy)
    def factory(*args,**kwargs):robot.calls.append('connect');return robot
    return home,sim,policy,clock,robot,factory


def test_explicit_execution_and_clear_path_required_before_connecting(setup,tmp_path):
    home,sim,_,clock,robot,factory=setup
    for flags in ({},{'execute':True},{'clear_path':True}):
        with pytest.raises(ValueError,match='execute'):
            run_real(home,sim,tmp_path/'real',client_factory=factory,**flags)
    assert not robot.calls and not (tmp_path/'real').exists()


def test_real_runner_never_reanchors_reconfigures_or_recovers(setup,tmp_path):
    home,sim,policy,clock,robot,factory=setup
    result=run_real(home,sim,tmp_path/'real',execute=True,clear_path=True,
                    client_factory=factory,clock=clock.now,sleep=clock.sleep)
    assert result['completed'] and result['stop_confirmed']
    assert robot.http.headers['Connection']=='close'
    targets=[c[1] for c in robot.calls if isinstance(c,tuple)]
    assert targets[0]==home['home']['position']
    assert np.max(np.array(targets)[:,1])==pytest.approx(policy.targets['left'][1])
    assert np.min(np.array(targets)[:,1])==pytest.approx(policy.targets['right'][1])
    assert targets[-1]==pytest.approx(home['home']['position'])
    assert robot.calls[-1]=='stop' and robot.calls.count('stop')==1 and robot.closed


def test_fault_aborts_without_replay_and_always_stops(setup,tmp_path):
    home,sim,_,clock,robot,factory=setup;robot.fault_after=.15
    with pytest.raises(RuntimeError,match='faulted'):
        run_real(home,sim,tmp_path/'real',execute=True,clear_path=True,
                 client_factory=factory,clock=clock.now,sleep=clock.sleep)
    assert robot.calls[-1]=='stop' and robot.closed
    assert (tmp_path/'real/failure.json').exists()
    assert not (tmp_path/'real/completion.json').exists()


def test_automatic_recovery_refused_not_silently_disabled(setup,tmp_path):
    home,sim,_,clock,robot,factory=setup;robot.auto=True
    with pytest.raises(RuntimeError,match='automatic recovery'):
        run_real(home,sim,tmp_path/'real',execute=True,clear_path=True,
                 client_factory=factory,clock=clock.now,sleep=clock.sleep)
    assert robot.calls==['connect'] and robot.closed


def test_changed_home_cannot_be_reused_with_previous_simulation(setup):
    home,sim,*_=setup
    home['sha256']['home']='different'
    with pytest.raises(ValueError,match='matching simulation'):validated_plan(home,sim)
