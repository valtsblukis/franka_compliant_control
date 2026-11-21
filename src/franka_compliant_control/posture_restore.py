"""Explicit slow joint-posture restoration; never recaptures the saved TCP home.

Uses unchanged native joint impedance. A smooth 20 s joint-reference transition
includes the already identified local friction/bias equilibrium offset, followed
by 5 s dwell. This housekeeping move is separate from the learned pose policy.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time

import mujoco
import numpy as np
from . import _native,_twin_native
from .client import CompliantClient
from .models import ControllerConfig
from .mujoco_twin import offline_model,native_inputs_digest
from .sysid_experiment import Envelope
from .twin_experiment import digest,load_home,write_json_new
from .twin_model import MujocoPlant,PlantParameters,quaternion_matrix,rotation_distance
from .wrench_model import WrenchObserver,canonical_franka_wrench


def reference(plan,t):
    u=float(np.clip((t-1)/20,0,1));blend=u**3*(10-15*u+6*u*u)
    return np.array(plan['start']['positions'])+blend*(np.array(plan['equilibrium_joints'])-plan['start']['positions'])


def capture(home,plant,path):
    if Path(path).exists():raise FileExistsError(path)
    r=CompliantClient(timeout_s=4);r.http.headers['Connection']='close'
    try:
        health=r.health()
        if health['running'] or health['phase']!='disarmed' or health['automatic_recovery_enabled']:raise RuntimeError('Need disarmed controller, recovery off')
        if ControllerConfig.model_validate(r.get_config()).model_dump()!=home['controller_config']:raise RuntimeError('Controller changed')
        r.refresh_state();s=r.state()
        if s['robot_mode']!='Idle' or s['robot_has_errors'] or not s['sample_valid'] or s['sample_age_s']>.05 or max(abs(v) for v in s['velocities'])>=.02:raise RuntimeError('Need fresh stationary Idle state')
        target=np.array(home['home']['joints']);start=np.array(s['positions']);c=home['controller_config']
        if max(abs(target-start))>.35:raise ValueError('Posture restoration exceeds reviewed small joint displacement')
        compensation=(np.sign(target-start)*np.array(plant.coulomb)+plant.joint_bias)/np.array(c['joint_stiffness'])
        if max(abs(compensation))>.025:raise ValueError('Unexpected equilibrium compensation')
        p=np.array(home['home']['position']);lo=np.array(c['joint_lower'])+.15;hi=np.array(c['joint_upper'])-.15;b=np.deg2rad(18)
        lo[0]=max(lo[0],-b,target[0]-b);hi[0]=min(hi[0],b,target[0]+b)
        if np.any(target+compensation<lo) or np.any(target+compensation>hi):raise ValueError('Reference outside joint envelope')
        envelope=Envelope(tuple(p-.015),tuple(p+.015),tuple(lo),tuple(hi),tuple(home['home']['quaternion_xyzw']),
                          translation_speed=.03,rotation_speed=.08,joint_speed=.06)
        envelope.check(start,s['velocities'],s['position'],s['quaternion_xyzw'],s['twist'])
        plan={'schema':'fcc-posture-restore-v1','home_sha256':home['sha256'],'controller_config':c,
              'start':s,'target_joints':target.tolist(),'equilibrium_joints':(target+compensation).tolist(),
              'duration_s':26.,'envelope':asdict(envelope),'plant':asdict(plant),
              'source_sha256':digest(Path(__file__).read_text()),'native_settings_changed':False,
              'note':'Original joint posture and original TCP home retained; identified local resistance offset in joint reference only'}
        write_json_new(path,plan);return plan
    finally:r.close()


def screen(home,plan,observer_config,output):
    if native_inputs_digest()!=_twin_native.twin_inputs_sha256:raise RuntimeError('Stale native twin')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env=Envelope(**plan['envelope']);params=PlantParameters(**plan['plant']);plant=MujocoPlant(home,params)
    plant.data.qpos[:]=plan['start']['positions'];mujoco.mj_forward(plant.model,plant.data)
    shadow=mujoco.MjData(plant.model);sdk=offline_model(home);observer=WrenchObserver(observer_config)
    previous=np.zeros(7);sample=sdk.sample(plant.data.qpos,plant.data.qvel,previous,np.zeros(6))
    cfg=ControllerConfig.model_validate(home['controller_config']);controller=_twin_native.OfflineController(cfg.native(),sample,1,0.)
    cmd=_native.Command();cmd.epoch=1;cmd.sequence=1;cmd.mode=_native.Mode.Joint;cmd.joints=plan['start']['positions']
    peak_acc=0.;peak_jerk=0.;last_acc=None;settled_since=None;reason=None;rows=[]
    last_v=None;last_cart_acc=None;peak_cart_acc=np.zeros(2);peak_cart_jerk=np.zeros(2)
    write_json_new(output/'manifest.json',{'plan_sha256':digest(plan),'observer':observer_config,'hardware_authorized':False})
    with (output/'trace.jsonl').open('x') as log:
        for tick in range(26001):
            now=tick*.001;q=plant.data.qpos.copy();dq=plant.data.qvel.copy();sample=sdk.sample(q,dq,previous,np.zeros(6))
            v=sample.jacobian@dq;g=sdk.gravity(q)
            if last_v is not None:
                a=(v-last_v)/.001;peak_cart_acc=np.maximum(peak_cart_acc,[np.linalg.norm(a[:3]),np.linalg.norm(a[3:])])
                if last_cart_acc is not None:
                    j=(a-last_cart_acc)/.001;peak_cart_jerk=np.maximum(peak_cart_jerk,[np.linalg.norm(j[:3]),np.linalg.norm(j[3:])])
                last_cart_acc=a
            last_v=v.copy()
            shadow.qpos[:]=q;shadow.qvel[:]=dq;shadow.qfrc_applied[:]=previous+g-params.joint_bias;mujoco.mj_forward(plant.model,shadow)
            o=observer.step(q,previous-sample.mass@shadow.qacc-sample.coriolis,sample.jacobian,sample.position,.001)
            sample.external_wrench=o['external_wrench'];sample.external_torque=o['external_joint_torques']
            r=quaternion_matrix(sample.quaternion_xyzw);w=o['external_wrench_tcp_base'];sample.external_wrench_stiffness=np.r_[r.T@w[:3],r.T@w[3:]]
            try:env.check(q,dq,sample.position,sample.quaternion_xyzw,v,o['minimum_singular_value'],w)
            except RuntimeError as error:reason=str(error);break
            if tick%20==0:cmd.joints=reference(plan,now);cmd.sequence+=1
            out=controller.step(sample,cmd,now,.001,now+.2)
            if out.phase!='active':reason=out.reason;break
            torque=_twin_native.sdk_torque_filter(out.torque,previous)
            good=(max(abs(q-plan['target_joints']))<.01 and np.linalg.norm(sample.position-home['home']['position'])<.01
                  and rotation_distance(sample.quaternion_xyzw,home['home']['quaternion_xyzw'])<.02 and max(abs(dq))<.02)
            settled_since=(now if settled_since is None else settled_since) if good else None
            if tick%20==0:
                row={'time_s':now,'positions':q.tolist(),'velocities':dq.tolist(),'position':sample.position.tolist(),
                     'quaternion_xyzw':sample.quaternion_xyzw.tolist(),'twist':v.tolist(),'wrench':w.tolist(),'reference_joints':cmd.joints.tolist()}
                rows.append(row);log.write(json.dumps(row)+'\n')
            if tick==26000:break
            plant.step(torque+g);previous=torque
            acc=(plant.data.qvel-dq)/.001;peak_acc=max(peak_acc,float(max(abs(acc))))
            if last_acc is not None:peak_jerk=max(peak_jerk,float(max(abs(acc-last_acc))/.001))
            last_acc=acc
    cartesian_pass=bool(np.all(peak_cart_acc<=[cfg.translation_acceleration,cfg.rotation_acceleration])
                        and np.all(peak_cart_jerk<=[cfg.translation_jerk,cfg.rotation_jerk]))
    result={'passed':bool(reason is None and settled_since is not None and now-settled_since>=.5 and peak_acc<=cfg.joint_acceleration and peak_jerk<=cfg.joint_jerk and cartesian_pass),
        'peak_cartesian_acceleration':peak_cart_acc.tolist(),'peak_cartesian_jerk':peak_cart_jerk.tolist(),
        'reason':reason,'last_time_s':now,'peak_joint_acceleration':peak_acc,'peak_joint_jerk':peak_jerk,
        'final_joint_error_rad':(plant.data.qpos-plan['target_joints']).tolist(),
        'final_tcp_error_m':float(np.linalg.norm(sample.position-home['home']['position'])),
        'final_orientation_error_rad':rotation_distance(sample.quaternion_xyzw,home['home']['quaternion_xyzw']),
        'peak_force_N':max(float(np.linalg.norm(x['wrench'][:3])) for x in rows),
        'peak_moment_Nm':max(float(np.linalg.norm(x['wrench'][3:])) for x in rows)}
    write_json_new(output/'summary.json',result);return result


def run(home,plan,screen_path,output,*,execute=False):
    if not execute:raise ValueError('Explicit execution acknowledgement required')
    sm=json.loads((Path(screen_path)/'manifest.json').read_text());summary=json.loads((Path(screen_path)/'summary.json').read_text())
    if sm['plan_sha256']!=digest(plan) or not summary['passed'] or plan['home_sha256']!=home['sha256'] or plan['source_sha256']!=digest(Path(__file__).read_text()):raise ValueError('Need matching successful unchanged screen')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json_new(output/'manifest.json',plan)
    robot=CompliantClient(timeout_s=4);robot.http.headers['Connection']='close';attempted=False;stop=None;last=None
    sdk=offline_model(home);envelope=Envelope(**plan['envelope']);settled_since=None
    try:
        health=robot.health()
        if health['running'] or health['phase']!='disarmed' or health['automatic_recovery_enabled']:raise RuntimeError('Need inactive controller, recovery off')
        if ControllerConfig.model_validate(robot.get_config()).model_dump()!=home['controller_config'] or digest(robot.get_urdf()['urdf'])!=home['sha256']['urdf']:raise RuntimeError('Controller/model changed')
        robot.refresh_state();initial=robot.state()
        if initial['robot_mode']!='Idle' or initial['robot_has_errors'] or initial['sample_age_s']>.05 or not initial['sample_valid'] or max(abs(v) for v in initial['velocities'])>=.02:raise RuntimeError('Not stationary/fresh/Idle')
        if max(abs(np.array(initial['positions'])-plan['start']['positions']))>.001:raise RuntimeError('Start changed; rescreen')
        attempted=True;first=robot.set_target_joint_state(initial['positions'],valid_for_s=1.)
        robot.timeout_s=.2;deadline=time.monotonic()+.5
        while True:
            state=robot.state()
            if state['running'] and state['phase']=='active':break
            if not state['running'] or state['phase']!='armed' or time.monotonic()>deadline:raise RuntimeError('Startup failed')
            time.sleep(.005)
        origin=time.monotonic();epoch=first['epoch'];own=first['command_seq'];tick=0
        write_json_new(output/'startup.json',{'state':state,'response':first})
        with (output/'trace.jsonl').open('x') as log,(output/'actions.jsonl').open('x') as actions:
            while True:
                due=origin+tick*.02;time.sleep(max(0.,due-time.monotonic()));now=time.monotonic();elapsed=now-origin
                if now-due>.05:raise RuntimeError('Missed publication deadline')
                state=robot.state();last=state
                if not state['running'] or state['phase']!='active' or state['mode']!='joint' or not state['sample_valid'] or not state['kinematics_valid'] or state['robot_has_errors'] or not 0<=state['sample_age_s']<=.02:raise RuntimeError('Fault/stale state')
                if state['epoch']!=epoch or state['applied_seq']>own or state['automatic_recovery_enabled']:raise RuntimeError('Producer/recovery changed')
                w=canonical_franka_wrench(state,stiffness_at_tcp=True)
                sample=sdk.sample(state['positions'],state['velocities'],np.zeros(7),np.zeros(6));j=sample.jacobian.copy();j[3:]*=home['controller_config']['jacobian_rotation_scale']
                envelope.check(state['positions'],state['velocities'],state['position'],state['quaternion_xyzw'],state['twist'],np.linalg.svd(j,compute_uv=False)[-1],w)
                log.write(json.dumps({**state,'time_s':elapsed,'external_wrench_tcp_base':w.tolist()})+'\n')
                good=(max(abs(np.array(state['positions'])-plan['target_joints']))<.01
                    and np.linalg.norm(np.array(state['position'])-home['home']['position'])<.01
                    and rotation_distance(state['quaternion_xyzw'],home['home']['quaternion_xyzw'])<.02 and max(abs(v) for v in state['velocities'])<.02)
                settled_since=(elapsed if settled_since is None else settled_since) if good else None
                if elapsed>=plan['duration_s']:
                    if settled_since is None or elapsed-settled_since<.5:raise RuntimeError('Posture did not settle; no retry')
                    break
                if time.monotonic()-due>.05:raise RuntimeError('Transport/processing deadline missed')
                q=reference(plan,elapsed);response=robot.set_target_joint_state(q,valid_for_s=.2);own=response['command_seq']
                actions.write(json.dumps({'time_s':elapsed,'joints':q.tolist(),'response':response})+'\n');tick+=1
    except BaseException as error:
        write_json_new(output/'failure.json',{'error':str(error),'last_state':last});raise
    finally:
        try:
            if attempted:robot.timeout_s=3.5;stop=robot.stop(wait_s=2.5);write_json_new(output/'stop.json',stop)
        finally:robot.close()
    if not stop or not stop['stopped']:raise RuntimeError('Stop not confirmed')
    result={'completed':True,'stop_confirmed':True,'final_joint_error_rad':(np.array(last['positions'])-plan['target_joints']).tolist(),
            'final_tcp_error_m':float(np.linalg.norm(np.array(last['position'])-home['home']['position']))}
    write_json_new(output/'completion.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['plan','screen','run']);p.add_argument('--home',type=Path,required=True)
    p.add_argument('--plant',type=Path);p.add_argument('--observer',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--screen',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv);h=load_home(a.home)
    if a.operation=='plan':result=capture(h,PlantParameters(**json.loads(a.plant.read_text())),a.output)
    elif a.operation=='screen':result=screen(h,json.loads(a.plan.read_text()),json.loads(a.observer.read_text()),a.output)
    else:
        def interrupt(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupt)
        result=run(h,json.loads(a.plan.read_text()),a.screen,a.output,execute=a.execute)
    print(json.dumps(result,indent=2));return 1 if result.get('passed') is False else 0


if __name__=='__main__':raise SystemExit(main())
