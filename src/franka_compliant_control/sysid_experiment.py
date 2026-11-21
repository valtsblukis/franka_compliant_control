"""FR3 adapter for bounded identification trials. No automatic recovery or re-homing.

Identification succeeds by executing a safe, well-recorded schedule, not by
pretending impedance tracking error is zero. Position error remains a reported
outcome; it is not silently relabeled as successful task accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

import numpy as np

from .client import CompliantClient
from .models import ControllerConfig
from .mujoco_twin import model_crosscheck, native_inputs_digest, offline_model, preflight_path, simulation_trace
from .trajectory import segment_sample
from .twin_experiment import digest, load_home, write_json_new
from .twin_model import PlantParameters, rotation_distance


@dataclass(frozen=True)
class WaypointPolicy:
    start_position: tuple
    start_quaternion_xyzw: tuple
    start_joints: tuple
    home_position: tuple  # Original saved anchor, NEVER recaptured.
    home_quaternion_xyzw: tuple
    waypoints: tuple  # Each entry: name, xyz, move_s, dwell_s.
    initial_hold_s: float = 1.
    hz: int = 50

    def __post_init__(self):
        for name,size in [('start_position',3),('start_quaternion_xyzw',4),('start_joints',7),
                          ('home_position',3),('home_quaternion_xyzw',4)]:
            v=tuple(getattr(self,name))
            if len(v)!=size or not np.isfinite(v).all():raise ValueError('Invalid policy vectors')
            object.__setattr__(self,name,v)
        if (abs(np.linalg.norm(self.start_quaternion_xyzw)-1)>1e-6
                or abs(np.linalg.norm(self.home_quaternion_xyzw)-1)>1e-6):raise ValueError('Invalid quaternion')
        if self.hz!=50 or not np.isfinite(self.initial_hold_s) or self.initial_hold_s<=0:
            raise ValueError('Invalid policy timing')
        points=[]
        for name,p,move,dwell in self.waypoints:
            if (not name.isalnum() or len(p)!=3 or not np.isfinite(p).all()
                    or not np.isfinite([move,dwell]).all() or move<1 or dwell<.5
                    or any(abs(t*50-round(t*50))>1e-8 for t in [move,dwell])):
                raise ValueError('Invalid waypoint/timing')
            points.append((name,tuple(p),float(move),float(dwell)))
        if not points or self.duration_s>120:raise ValueError('Need a finite <=120 s trial')
        object.__setattr__(self,'waypoints',tuple(points))

    @property
    def duration_s(self):return self.initial_hold_s+sum(m+d for _,_,m,d in self.waypoints)

    @property
    def targets(self):return {f'{name}{i}':list(p) for i,(name,p,_,_) in enumerate(self.waypoints)}

    def sample(self,t):
        if not np.isfinite(t) or t<0:raise ValueError('Invalid policy clock')
        tick=int(np.floor(t*50+1e-8));elapsed=tick/50
        q0=np.array(self.start_quaternion_xyzw);q1=np.array(self.home_quaternion_xyzw)
        if q0@q1<0:q1=-q1
        position=self.start_position;quaternion=q0;goal='start';goal_position=position;leg=-1;phase='initial_hold'
        if elapsed>=self.initial_hold_s:
            elapsed-=self.initial_hold_s
            previous=self.start_position
            for i,(name,end,move,dwell) in enumerate(self.waypoints):
                if elapsed<move+dwell-1e-9:
                    position,_,_=segment_sample(previous,end,elapsed,move)
                    u=min(1.,elapsed/move);fraction=u**3*(10-15*u+6*u*u)
                    if i==0:
                        angle=np.arccos(np.clip(q0@q1,-1,1))
                        quaternion=((1-fraction)*q0+fraction*q1 if angle<1e-8 else
                                    (np.sin((1-fraction)*angle)*q0+np.sin(fraction*angle)*q1)/np.sin(angle))
                        quaternion/=np.linalg.norm(quaternion)
                    else:quaternion=q1
                    goal,goal_position,leg=name,end,i
                    phase=('approach_center' if i==0 else 'move') if elapsed<move-1e-9 else 'dwell'
                    break
                elapsed-=move+dwell;previous=end
            else:
                goal,position,_,_=self.waypoints[-1];goal_position=position;quaternion=q1
                leg=len(self.waypoints);phase='complete'
        return {'policy_tick':tick,'policy_time_s':tick/50,'policy_phase':phase,'leg':leg,
                'goal':goal,'goal_position':list(goal_position),'position':list(position),
                'quaternion_xyzw':quaternion.tolist(),'moving':phase in ('move','approach_center')}

    def manifest(self):return {'schema':'fcc-waypoint-policy-v1',**asdict(self),'duration_s':self.duration_s,
                               'targets':self.targets,'reference':'50 Hz pose-only quintic/shortest-quaternion pacing'}


@dataclass(frozen=True)
class Envelope:
    tcp_lower: tuple
    tcp_upper: tuple
    joint_lower: tuple
    joint_upper: tuple
    orientation_xyzw: tuple
    translation_speed: float=.08
    rotation_speed: float=.15
    joint_speed: float=.25
    orientation_error_rad: float=np.deg2rad(2)
    minimum_singular_value: float=.05
    estimated_force_stop: float=8.
    estimated_moment_stop: float=2.

    def __post_init__(self):
        for name,size in [('tcp_lower',3),('tcp_upper',3),('joint_lower',7),('joint_upper',7),('orientation_xyzw',4)]:
            v=tuple(getattr(self,name))
            if len(v)!=size or not np.isfinite(v).all():raise ValueError('Invalid envelope')
            object.__setattr__(self,name,v)
        if np.any(np.array(self.tcp_lower)>=self.tcp_upper) or np.any(np.array(self.joint_lower)>=self.joint_upper):
            raise ValueError('Empty envelope')
        for v in [self.translation_speed,self.rotation_speed,self.joint_speed,self.orientation_error_rad,
                  self.minimum_singular_value,self.estimated_force_stop,self.estimated_moment_stop]:
            if not np.isfinite(v) or v<=0:raise ValueError('Invalid guard threshold')

    def check(self,q,dq,p,quat,twist,sigma=None,wrench=None):
        q,dq,p,twist=map(np.asarray,(q,dq,p,twist))
        if not all(np.isfinite(v).all() for v in (q,dq,p,twist)):raise RuntimeError('Nonfinite experiment state')
        if np.any(q<self.joint_lower) or np.any(q>self.joint_upper):
            raise RuntimeError('Identification joint envelope exceeded (including narrow base range)')
        if np.any(p<self.tcp_lower) or np.any(p>self.tcp_upper):raise RuntimeError('Identification workspace box exceeded')
        if (np.max(np.abs(dq))>self.joint_speed or np.linalg.norm(twist[:3])>self.translation_speed
                or np.linalg.norm(twist[3:])>self.rotation_speed):raise RuntimeError('Identification speed guard')
        if rotation_distance(quat,self.orientation_xyzw)>self.orientation_error_rad:raise RuntimeError('Identification orientation guard')
        if sigma is not None and (not np.isfinite(sigma) or sigma<self.minimum_singular_value):
            raise RuntimeError('Identification conditioning guard')
        if wrench is not None and (not np.isfinite(wrench).all()
                or np.linalg.norm(wrench[:3])>self.estimated_force_stop
                or np.linalg.norm(wrench[3:])>self.estimated_moment_stop):raise RuntimeError('Unexpected estimated contact load')


def front_recipe():
    training=[]
    for move in (3.,2.):
        training.extend([['forward',[.030,0,0],move,1.5],['center',[0,0,0],move,1.5],
                         ['up',[0,0,.020],move,1.5],['center',[0,0,0],move,1.5]])
    return {'center_offset_m':[.015,0,.020], 'tcp_lower_offset_m':[-.010,-.025,-.005],
            'tcp_upper_offset_m':[.065,.025,.065], 'joint_limit_margin_rad':.15,
            'base_joint_deviation_rad':.05,'approach_s':4.,'approach_dwell_s':2.,
            'training':training,
            'validation':[['diagonalA',[.023,0,.013],2.8,2.],['center',[0,0,0],2.8,2.],
                          ['diagonalB',[.010,0,.028],3.6,2.],['center',[0,0,0],3.6,2.]]}


def build_plan(home,state,role,*,recipe=None):
    if role not in ('training','validation'):raise ValueError('Unknown dataset role')
    c=ControllerConfig.model_validate(home['controller_config'])
    recipe=front_recipe() if recipe is None else recipe
    if set(recipe)!=set(front_recipe()):raise ValueError('Unknown or missing recipe fields')
    margin=recipe['joint_limit_margin_rad'];deviation=recipe['base_joint_deviation_rad']
    if not np.isfinite([margin,deviation]).all() or margin<0 or deviation<=0:raise ValueError('Invalid joint guard recipe')
    anchor=np.array(home['home']['position']);center=anchor+recipe['center_offset_m']
    lower=np.array(c.joint_lower)+margin;upper=np.array(c.joint_upper)-margin
    # These are operator-selected bounds, not an inferred pillar collision map.
    lower[0]=max(lower[0],home['home']['joints'][0]-deviation)
    upper[0]=min(upper[0],home['home']['joints'][0]+deviation)
    envelope=Envelope(tuple(anchor+recipe['tcp_lower_offset_m']),tuple(anchor+recipe['tcp_upper_offset_m']),
                      tuple(lower),tuple(upper),tuple(home['home']['quaternion_xyzw']))
    envelope.check(state['positions'],state['velocities'],state['position'],state['quaternion_xyzw'],state['twist'])
    points=[('center',center,recipe['approach_s'],recipe['approach_dwell_s'])]
    points.extend((name,center+offset,move,dwell) for name,offset,move,dwell in recipe[role])
    for _,point,_,_ in points:
        if np.any(point<envelope.tcp_lower) or np.any(point>envelope.tcp_upper):
            raise ValueError('Recipe waypoint lies outside its workspace box')
    policy=WaypointPolicy(tuple(state['position']),tuple(state['quaternion_xyzw']),tuple(state['positions']),
                           tuple(anchor),tuple(home['home']['quaternion_xyzw']),tuple(points))
    return {'schema':'fcc-sysid-plan-v1','role':role,'created_utc':datetime.now(timezone.utc).isoformat(),
            'home_sha256':home['sha256'],'controller_config':home['controller_config'],
            'start_observation':state,'policy':policy.manifest(),'policy_sha256':digest(policy.manifest()),
            'envelope':asdict(envelope),'environment_note':'Operator permits forward/above-table motion. Unsurveyed lateral camera pillars; narrow base range enforced. Not collision-free certification.'}


def parse_plan(home,plan):
    if (plan['schema']!='fcc-sysid-plan-v1' or plan['home_sha256']!=home['sha256']
            or plan['controller_config']!=home['controller_config']):raise ValueError('Plan/configuration mismatch')
    policy=WaypointPolicy(**{k:v for k,v in plan['policy'].items() if k in WaypointPolicy.__dataclass_fields__})
    if digest(policy.manifest())!=plan['policy_sha256']:raise ValueError('Policy fingerprint mismatch')
    return policy,Envelope(**plan['envelope'])


def capture_plan(home,output,role,url='http://127.0.0.1:54321',*,recipe=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    robot=CompliantClient(url,timeout_s=4);robot.http.headers['Connection']='close'
    try:
        health=robot.health()
        if health['running'] or health['phase']!='disarmed' or health['automatic_recovery_enabled']:
            raise RuntimeError('Need inactive disarmed controller, automatic recovery OFF')
        if ControllerConfig.model_validate(robot.get_config()).model_dump()!=home['controller_config']:
            raise RuntimeError('Controller configuration changed')
        robot.refresh_state();state=robot.state()
        if (state['robot_mode']!='Idle' or state['robot_has_errors'] or not state['sample_valid']
                or state['sample_age_s']>.05 or max(abs(v) for v in state['velocities'])>=.02):
            raise RuntimeError('Need fresh stationary Idle state')
        sample=offline_model(home).sample(state['positions'],np.zeros(7),np.zeros(7),np.zeros(6))
        if (np.linalg.norm(sample.position-state['position'])>1e-4
                or rotation_distance(sample.quaternion_xyzw,state['quaternion_xyzw'])>np.deg2rad(.05)):
            raise RuntimeError('Configured TCP model and measured pose disagree')
        plan=build_plan(home,state,role,recipe=recipe);plan['capture_source_url']=url
        write_json_new(output,plan);return plan
    finally:robot.close()  # No setpoint attempted: close sends no stop.


def simulate_plan(home,plan,output,parameters=PlantParameters()):
    from . import _twin_native
    policy,envelope=parse_plan(home,plan)
    if native_inputs_digest()!=_twin_native.twin_inputs_sha256:raise RuntimeError('Stale offline bridge')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    check=model_crosscheck(home)
    if not check['passed']:raise RuntimeError('Model conversion failed')
    def screen(q,p,quat,sigma):envelope.check(q,np.zeros(7),p,quat,np.zeros(6),sigma)
    path=preflight_path(home,policy,sample_guard=screen)
    write_json_new(output/'manifest.json',{'schema':'fcc-twin-run-v1','backend':'mujoco',
        'home_sha256':home['sha256'],'controller_config':home['controller_config'],
        'policy':policy.manifest(),'policy_sha256':plan['policy_sha256'],'plan_sha256':digest(plan),
        'plant':parameters.manifest(),'role':plan['role'],'envelope':asdict(envelope),'stream_command_lifetime_s':.2,
        'controller_source_sha256':_twin_native.controller_source_sha256,'model_crosscheck':check,'path_preflight':path})
    rows=[];failure=None
    with (output/'trace.jsonl').open('x') as f:
        for r in simulation_trace(home,policy,parameters,command_lifetime_s=.2):
            f.write(json.dumps(r,allow_nan=False)+'\n');rows.append(r)
            try:
                if r['phase']!='active':raise RuntimeError(r['reason'])
                envelope.check(r['positions'],r['velocities'],r['position'],r['quaternion_xyzw'],r['twist'],r['minimum_singular_value'])
            except RuntimeError as e:failure=str(e);break
    qv=np.array([r['velocities'] for r in rows]);v=np.array([r['twist'] for r in rows]);t=np.array([r['time_s'] for r in rows])
    ddq=np.diff(qv,axis=0)/np.diff(t)[:,None] if len(t)>1 else np.zeros((1,7))
    a=np.diff(v,axis=0)/np.diff(t)[:,None] if len(t)>1 else np.zeros((1,6))
    jerk=np.diff(ddq,axis=0)/np.diff(t)[1:,None] if len(t)>2 else np.zeros((1,7))
    c=ControllerConfig.model_validate(home['controller_config'])
    dynamic_ok=(np.max(abs(ddq))<=c.joint_acceleration and np.max(abs(jerk))<=c.joint_jerk
                and np.max(np.linalg.norm(a[:,:3],axis=1))<=c.translation_acceleration
                and np.max(np.linalg.norm(a[:,3:],axis=1))<=c.rotation_acceleration)
    summary={'completed':bool(failure is None and t[-1]>=policy.duration_s),'envelope_passed':failure is None,
             'acceleration_jerk_screen_passed':bool(dynamic_ok),'failure':failure,'samples':len(rows),
             'minimum_base_joint_rad':min(r['positions'][0] for r in rows),'maximum_base_joint_rad':max(r['positions'][0] for r in rows),
             'minimum_conditioning':min(r['minimum_singular_value'] for r in rows),
             'peak_tcp_speed_m_s':float(np.max(np.linalg.norm(v[:,:3],axis=1))),
             'peak_joint_acceleration_rad_s2':np.max(abs(ddq),axis=0).tolist(),
             'final_goal_error_m':float(np.linalg.norm(np.array(rows[-1]['position'])-rows[-1]['policy']['goal_position'])),
             'note':'Tracking error is data, not permission to exceed the safety envelope. Proprietary stop dynamics not simulated.'}
    summary['ready_for_review']=bool(summary['completed'] and dynamic_ok)
    write_json_new(output/'summary.json',summary);return summary


def acquire(home,plan,simulation,output,*,execute=False,clear_path=False,url='http://127.0.0.1:54321',
            client_factory=CompliantClient,clock=time.monotonic,sleep=time.sleep):
    if not execute or not clear_path:raise ValueError('Explicit execution and cleared-path acknowledgement required')
    policy,envelope=parse_plan(home,plan)
    simulation=Path(simulation)
    manifest=json.loads((simulation/'manifest.json').read_text());screen=json.loads((simulation/'summary.json').read_text())
    if manifest['plan_sha256']!=digest(plan) or not screen['ready_for_review']:raise ValueError('Need matching reviewed simulation')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json_new(output/'manifest.json',{'schema':'fcc-twin-run-v1','backend':'robot','role':plan['role'],
        'home_sha256':home['sha256'],'controller_config':home['controller_config'],'plan_sha256':digest(plan),
        'policy':policy.manifest(),'policy_sha256':plan['policy_sha256'],'envelope':asdict(envelope),
        'http_connection_policy':'close','torque_convention':'Franka tau_J_d excludes gravity',
        'initial_command_lifetime_s':1.,'stream_command_lifetime_s':.2,
        'tracking_accuracy_is_separate_from_schedule_completion':True,'source_url':url,
        'screened_simulation_manifest_sha256':digest(manifest)})
    robot=client_factory(url,timeout_s=4);robot.http.headers['Connection']='close'
    attempted=False;rows=[];stop=None
    try:
        health=robot.health()
        if health['running'] or health['phase']!='disarmed' or health['automatic_recovery_enabled']:
            raise RuntimeError('Need disarmed inactive controller and automatic recovery OFF')
        if (ControllerConfig.model_validate(robot.get_config()).model_dump()!=home['controller_config']
                or digest(robot.get_urdf()['urdf'])!=home['sha256']['urdf']):raise RuntimeError('Model/configuration mismatch')
        robot.refresh_state();initial=robot.state()
        if (initial['robot_mode']!='Idle' or initial['robot_has_errors'] or not initial['sample_valid']
                or initial['sample_age_s']>.05 or max(abs(v) for v in initial['velocities'])>=.02):
            raise RuntimeError('Need fresh stationary Idle state')
        if (np.linalg.norm(np.array(initial['position'])-policy.start_position)>.0002
                or rotation_distance(initial['quaternion_xyzw'],policy.start_quaternion_xyzw)>np.deg2rad(.05)
                or np.max(abs(np.array(initial['positions'])-policy.start_joints))>.001):
            raise RuntimeError('Start moved: replan/resimulate, never jump to a stale reference')
        envelope.check(initial['positions'],initial['velocities'],initial['position'],initial['quaternion_xyzw'],initial['twist'])
        write_json_new(output/'initial_state.json',initial)
        with (output/'trace.jsonl').open('x') as trace,(output/'actions.jsonl').open('x') as actions:
            first=policy.sample(0);attempted=True
            response=robot.set_target_ee_pose(first['position'],first['quaternion_xyzw'],valid_for_s=min(1.,home['controller_config']['command_timeout']))
            expected_epoch=response.get('epoch');last_own_sequence=response.get('command_seq')
            robot.timeout_s=.2;deadline=clock()+.5
            while True:
                state=robot.state()
                if state['running'] and state['phase']=='active':break
                if not state['running'] or state['phase']!='armed' or clock()>=deadline:raise RuntimeError('Startup failed')
                sleep(.005)
            origin=clock();server_origin=state['server_monotonic_s']
            write_json_new(output/'startup.json',{'state':state,'first_response':response,'policy_origin_server_monotonic_s':server_origin})
            actions.write(json.dumps({'policy':first,'response':response,'sent_time_s':0})+'\n')
            tick=0
            while True:
                due=origin+tick/50;sleep(max(0,due-clock()));elapsed=clock()-origin
                if clock()-due>.05:raise RuntimeError('Publisher schedule missed by >50 ms; no replay')
                state=robot.state()
                if (not state['running'] or state['phase']!='active' or not state['sample_valid']
                        or state['sample_age_s']>.02 or state.get('robot_has_errors')):raise RuntimeError('Fault/stale observation')
                if (state.get('mode','cartesian')!='cartesian'
                        or (expected_epoch is not None and state.get('epoch')!=expected_epoch)
                        or (last_own_sequence is not None and state.get('applied_seq',0)>last_own_sequence)):
                    raise RuntimeError('Another command producer changed mode/epoch/sequence')
                if state.get('automatic_recovery_enabled',False):raise RuntimeError('Automatic recovery was enabled during the trial')
                envelope.check(state['positions'],state['velocities'],state['position'],state['quaternion_xyzw'],
                               state['twist'],state['minimum_singular_value'],state['external_wrench'])
                measured=max(0.,state['server_monotonic_s']-state['sample_age_s']-server_origin)
                row={**state,'time_s':measured,'host_elapsed_s':elapsed,'policy':policy.sample(measured)}
                rows.append(row);trace.write(json.dumps(row,allow_nan=False)+'\n')
                if clock()-due>.05:raise RuntimeError('HTTP/processing deadline missed; no stale action sent')
                if elapsed>=policy.duration_s:break
                if tick:
                    target=policy.sample(elapsed)
                    response=robot.set_target_ee_pose(target['position'],target['quaternion_xyzw'],valid_for_s=.2)
                    last_own_sequence=response.get('command_seq')
                    actions.write(json.dumps({'policy':target,'response':response,'sent_time_s':elapsed},allow_nan=False)+'\n')
                tick+=1
    except BaseException as e:
        write_json_new(output/'failure.json',{'type':type(e).__name__,'error':str(e),'samples':len(rows)})
        raise
    finally:
        try:
            if attempted:
                robot.timeout_s=3.5;stop=robot.stop(wait_s=2.5);write_json_new(output/'stop.json',stop)
        finally:robot.close()
    if not stop or not stop['stopped']:
        write_json_new(output/'failure.json',{'error':'Stationary disarmed stop NOT confirmed'})
        raise RuntimeError('Stationary disarmed stop NOT confirmed')
    result={'completed':True,'stop_confirmed':True,'samples':len(rows),
            'final_goal_error_m':float(np.linalg.norm(np.array(rows[-1]['position'])-rows[-1]['policy']['goal_position'])),
            'minimum_base_joint_rad':min(r['positions'][0] for r in rows),'maximum_base_joint_rad':max(r['positions'][0] for r in rows),
            'note':'Completed calibration schedule, not a claim of exact task tracking or a calibrated model'}
    write_json_new(output/'completion.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    for name in ['plan','sim','run']:
        p=sub.add_parser(name);p.add_argument('--home',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
        if name=='plan':
            p.add_argument('--role',choices=['training','validation'],required=True)
            p.add_argument('--recipe',type=Path,help='Operator-reviewed relative waypoints and workspace/base-joint bounds')
        else:p.add_argument('--plan',required=True,type=Path)
        if name in ('plan','run'):p.add_argument('--url',default='http://127.0.0.1:54321')
        if name=='sim':p.add_argument('--plant',type=Path)
        if name=='run':
            p.add_argument('--simulation',required=True,type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--confirm-clear-path',action='store_true')
    args=parser.parse_args(argv);home=load_home(args.home)
    if args.operation=='plan':
        recipe=json.loads(args.recipe.read_text()) if args.recipe else None
        result=capture_plan(home,args.output,args.role,url=args.url,recipe=recipe)
    else:
        plan=json.loads(args.plan.read_text())
        if args.operation=='sim':
            params=PlantParameters(**json.loads(args.plant.read_text())) if args.plant else PlantParameters()
            result=simulate_plan(home,plan,args.output,params)
        else:
            if not args.execute:result={'dry_run':True,'robot_connection':False,'plan':plan}
            else:
                def interrupt(signum,_frame):raise KeyboardInterrupt(f'signal {signum}')
                for signum in (signal.SIGTERM,signal.SIGHUP):signal.signal(signum,interrupt)
                result=acquire(home,plan,args.simulation,args.output,execute=True,clear_path=args.confirm_clear_path,url=args.url)
    print(json.dumps(result,indent=2))
    return 1 if args.operation=='sim' and not result['ready_for_review'] else 0


if __name__=='__main__':raise SystemExit(main())
