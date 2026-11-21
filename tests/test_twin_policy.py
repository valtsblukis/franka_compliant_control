"""Policy/capture contracts. No robot connection or control."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from franka_compliant_control import _native
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.twin_experiment import ManualPosePolicy, capture_home, digest, load_home
from franka_compliant_control.twin_model import PlantParameters, combined_load


def test_fixed_home_targets_and_repeated_cycles_never_reanchor():
    policy=ManualPosePolicy((.4,.01,.5),(1.,0.,0.,0.),cycles=2)
    assert policy.targets['left']==pytest.approx([.4,.31,.5])
    assert policy.targets['right']==pytest.approx([.4,-.29,.5])
    assert policy.sample(0)['position']==list(policy.home_position)
    for time,goal in [(4,'left'),(9,'home'),(14,'right'),(19,'home'),(24,'left'),(29,'home'),(34,'right'),(39,'home')]:
        value=policy.sample(time)
        assert value['position']==pytest.approx(policy.targets[goal])
        assert value['quaternion_xyzw']==[1,0,0,0]
        assert 'twist' not in value and 'acceleration' not in value
    changed=policy.sample(4);changed['position'][0]=999
    assert policy.sample(4)['position']==pytest.approx([.4,.31,.5])


def test_policy_is_exactly_50hz_held_and_filtered_on_policy_side():
    p=ManualPosePolicy((.4,0.,.5),(1.,0.,0.,0.))
    assert p.sample(2.001)==p.sample(2.019)
    assert p.sample(2.02)!=p.sample(2.019)
    assert p.sample(1)['position']==list(p.home_position)
    assert p.sample(4)['position']==p.targets['left']
    assert p.sample(100)['position']==list(p.home_position)


@pytest.mark.parametrize('changes',[{'offset_m':float('nan')},{'move_s':0},{'dwell_s':-.1},
    {'hz':100},{'cycles':0},{'cycles':True},{'move_s':1.001}])
def test_invalid_policy(changes):
    with pytest.raises(ValueError):ManualPosePolicy((.4,0,.5),(1,0,0,0),**changes)


@pytest.mark.parametrize('changes',[{'viscous':[-1]*7},{'coulomb':[0]*6},{'armature':[float('nan')]*7},
    {'physics_dt':.0007},{'actuator_delay_steps':1.5},{'tool_mass_scale':0}])
def test_invalid_plant(changes):
    with pytest.raises(ValueError):PlantParameters(**changes)


def test_combine_tool_and_load_with_parallel_axis_theorem():
    h={'expected_end_effector':{'mass':1.,'com':[-.1,0,0],'inertia':[.01,0,0,0,.01,0,0,0,.01]},
       'load_mass':1.,'load_com':[.1,0,0],'load_inertia':[.02,0,0,0,.02,0,0,0,.02]}
    load=combined_load(h)
    assert load['mass']==2 and load['com']==[0,0,0]
    assert np.array(load['inertia']).reshape(3,3,order='F')==pytest.approx(np.diag([.03,.05,.05]))


class Response:
    def __init__(self,value):self.value=value
    def raise_for_status(self):pass
    def json(self):return deepcopy(self.value)


class CaptureHTTP:
    def __init__(self):
        self.calls=[];self.refreshed=False
        self.state={'backend':'robot','running':False,'phase':'disarmed','robot_mode':'Idle','robot_has_errors':False,
            'sample_valid':True,'kinematics_valid':True,'sample_age_s':0.,'robot_time_s':1.,
            'position':[.4,0,.5],'quaternion_xyzw':[0,0,0,1],'positions':[0,-.4,0,-1.9,0,1.5,0],'velocities':[0]*7}
        self.config=ControllerConfig.model_validate_json(Path('configs/controller.fr3.auto.json').read_text()).model_dump()
    def get(self,url,**kwargs):
        self.calls.append(('GET',url))
        if url.endswith('/controller_config'):return Response(self.config)
        if url.endswith('/urdf'):return Response({'urdf':'fake-model'})
        return Response({**self.state,'robot_time_s':self.state['robot_time_s']+int(self.refreshed)})
    def post(self,url,**kwargs):
        self.calls.append(('POST',url));assert url.endswith('/state/refresh')
        self.refreshed=True;return Response({})


def test_capture_only_refreshes_and_never_overwrites_or_sends_targets(tmp_path,monkeypatch):
    class Model:
        def __init__(self,*args):pass
        def evaluate(self,q):
            p=np.eye(4);p[:3,3]=[.4,0,.5]
            return p,np.zeros((6,7))
    monkeypatch.setattr(_native,'OfflineKinematics',Model)
    http=CaptureHTTP();path=tmp_path/'home.json'
    result=capture_home(path,'configs/hardware.wrist-load.estimated.json',http=http)
    assert result['home']['position']==[.4,0,.5]
    assert load_home(path)['home']==result['home']
    assert [url for method,url in http.calls if method=='POST']==['http://127.0.0.1:54321/state/refresh']
    count=len(http.calls)
    with pytest.raises(FileExistsError):capture_home(path,'configs/hardware.wrist-load.estimated.json',http=http)
    assert len(http.calls)==count
    damaged=json.loads(path.read_text());damaged['home']['position'][0]=.7
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError,match='fingerprint'):load_home(path)


@pytest.mark.parametrize('changes',[{'running':True},{'phase':'fault'},{'sample_age_s':1.},
    {'velocities':[.1]*7},{'robot_has_errors':True}])
def test_capture_rejects_unsafe_or_stale_state_before_saving(tmp_path,changes):
    http=CaptureHTTP();http.state.update(changes)
    with pytest.raises(RuntimeError):capture_home(tmp_path/'home.json','configs/hardware.wrist-load.estimated.json',http=http)
    assert not (tmp_path/'home.json').exists()
    assert all(method=='GET' or url.endswith('/state/refresh') for method,url in http.calls)


def test_measured_start_alignment_preserves_fixed_home_and_old_manifest():
    from franka_compliant_control.twin_model import rotation_distance
    home=(.4,0.,.5); quaternion=(0.,0.,0.,1.)
    angle=np.deg2rad(.8)
    start=(0.,0.,float(np.sin(angle/2)),float(np.cos(angle/2)))
    base=ManualPosePolicy(home,quaternion,offset_m=.03)
    aligned=ManualPosePolicy(home,quaternion,offset_m=.03,start_position=(.4001,0.,.5),
        start_quaternion_xyzw=start,start_joints=(0,-.4,0,-1.9,0,1.5,0),alignment_s=3.)
    assert 'alignment_s' not in base.manifest()
    assert aligned.targets==base.targets and aligned.duration_s==base.duration_s+3
    assert aligned.sample(0)['position']==pytest.approx([.4001,0,.5])
    assert aligned.sample(0)['quaternion_xyzw']==pytest.approx(start)
    assert aligned.sample(3)['position']==list(home)
    assert aligned.sample(3)['quaternion_xyzw']==list(quaternion)
    assert aligned.sample(7)['position']==base.targets['left']
    errors=[rotation_distance(aligned.sample(t)['quaternion_xyzw'],quaternion) for t in np.linspace(0,3,151)]
    assert all(a>=b-1e-12 for a,b in zip(errors,errors[1:]))
    assert max(abs(np.diff(errors)))*50<.02
    assert aligned.manifest()['home_position']==home


def test_alignment_rejects_large_or_incomplete_retargeting():
    with pytest.raises(ValueError,match='complete'):
        ManualPosePolicy((.4,0,.5),(0,0,0,1),start_position=(.4,0,.5),alignment_s=3)
    with pytest.raises(ValueError,match='2 mm'):
        ManualPosePolicy((.4,0,.5),(0,0,0,1),start_position=(.41,0,.5),
            start_quaternion_xyzw=(0,0,0,1),start_joints=(0,)*7,alignment_s=3)
