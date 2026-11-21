from copy import deepcopy
import gc
from types import SimpleNamespace
import numpy as np
from franka_compliant_control.twin_model import quaternion_matrix

class Clock:
    t=0.
    def now(self):return self.t
    def sleep(self,dt):self.t+=dt


class Robot:
    def __init__(self,home,clock):
        self.home=home;self.clock=clock;self.running=False;self.http=SimpleNamespace(headers={})
        self.pose=list(home['home']['position']);self.quat=list(home['home']['quaternion_xyzw'])
        self.seq=0;self.calls=[];self.closed=False;self.force=np.zeros(6);self.raise_after=None
    def health(self):return {'running':self.running,'phase':'active' if self.running else 'disarmed','automatic_recovery_enabled':False}
    def get_config(self):return deepcopy(self.home['controller_config'])
    def get_urdf(self):return {'urdf':self.home['urdf']}
    def refresh_state(self):self.calls.append('refresh')
    def state(self):
        rotation=quaternion_matrix(self.quat);w=self.force
        if self.raise_after is not None and self.clock.t>self.raise_after:w=np.array([9,0,0,0,0,0])
        return {'backend':'robot','phase':'active' if self.running else 'disarmed','running':self.running,
            'robot_mode':'Move' if self.running else 'Idle','robot_has_errors':False,'sample_valid':True,
            'kinematics_valid':True,'sample_age_s':0.,'position':self.pose.copy(),'quaternion_xyzw':self.quat.copy(),
            'positions':self.home['home']['joints'],'velocities':[0.]*7,'twist':[0.]*6,
            'external_wrench_stiffness':np.r_[rotation.T@w[:3],rotation.T@w[3:]].tolist(),
            'minimum_singular_value':.1,'robot_time_s':1+self.clock.t,'server_monotonic_s':1000+self.clock.t,
            'mode':'cartesian','epoch':1,'applied_seq':self.seq,'automatic_recovery_enabled':False}
    def set_target_ee_pose(self,p,q,*,valid_for_s):
        assert valid_for_s<=1
        assert not gc.isenabled()
        self.calls.append(('target',list(p)));self.pose=list(p);self.quat=list(q);self.running=True;self.seq+=1
        return {'epoch':1,'command_seq':self.seq,'accepted_monotonic_s':1000+self.clock.t}
    def stop(self,wait_s):self.calls.append('stop');self.running=False;return {'stopped':True}
    def close(self):self.closed=True
