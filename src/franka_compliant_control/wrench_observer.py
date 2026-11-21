"""Approximate Franka wrench telemetry, NOT an independently calibrated force sensor.

Canonical policy wrench: robot-on-environment, base axes, about configured TCP.
The recorded FR3's raw O_F_ext_hat_K moments instead obey the spatial transform
M_O = R_OK M_K + p_OK x F_O. Preserve raw SDK values and normalize explicitly.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .mujoco_twin import offline_model
from .system_identification import prepare_trial,trial_role
from .twin_experiment import digest,load_home,write_json_new
from .twin_model import quaternion_matrix


# Runtime-only implementation is below this fitting/evaluation CLI layer.
# Preserve the original public imports for saved experiment recipes.
from .wrench_model import (tcp_to_spatial, spatial_to_tcp, canonical_franka_wrench,
                           projected_wrench, WrenchObserver)


def _lowpass(values,time_constant,dt=.02):
    output=values.copy()
    if time_constant:
        a=np.exp(-dt/time_constant)
        for i in range(1,len(output)):output[i]=a*output[i-1]+(1-a)*values[i]
    return output


def observation_trial(directory,model):
    trial=prepare_trial(directory,model)
    rows=[json.loads(line) for line in (Path(directory)/'trace.jsonl').read_text().splitlines()]
    # prepare_trial deduplicates clocks; do likewise for measurement interpolation.
    unique=[];last=-np.inf
    for row in rows:
        if row['robot_time_s']>last:unique.append(row);last=row['robot_time_s']
    rows=unique
    times=np.array([r['robot_time_s'] for r in rows]);times=times-times[0]+rows[0]['time_s']
    def interp(values):
        a=np.asarray(values,dtype=float)
        return np.column_stack([np.interp(trial['time'],times,a[:,j]) for j in range(a.shape[1])])
    trial['external_torque']=interp([r['external_joint_torques'] for r in rows])
    trial['canonical_wrench']=interp([canonical_franka_wrench(r,stiffness_at_tcp=True) for r in rows])
    trial['raw_wrench']=interp([r['external_wrench'] for r in rows])
    trial['position']=interp([r['position'] for r in rows])
    errors=[];projected_errors=[]
    for row in rows[::4]:
        canonical=canonical_franka_wrench(row,stiffness_at_tcp=True)
        errors.append(np.asarray(row['external_wrench'])-tcp_to_spatial(canonical,row['position']))
        sample=model.sample(row['positions'],row['velocities'],np.zeros(7),np.zeros(6))
        projected,_=projected_wrench(sample.jacobian,row['external_joint_torques'])
        projected_errors.append(projected-canonical)
    trial['frame_audit']={
        'raw_spatial_transform_max_abs':np.max(np.abs(errors),axis=0).tolist(),
        'joint_torque_projection_max_abs':np.max(np.abs(projected_errors),axis=0).tolist(),
        'raw_moment_point_observed':'base_origin_spatial',
        'canonical_point':'configured_tcp','axes':'base','sign':'robot_on_environment'}
    if np.max(np.abs(errors))>1e-4 or np.max(np.abs(projected_errors))>1e-4:
        raise ValueError('Recorded frame/projection convention does not match this observer adapter')
    return trial


def fit_observer(home,training,output):
    """Train a compact free-space surrogate; do not fit unknown contact gain."""
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    for path in training:
        if trial_role(path)=='validation':
            raise ValueError('Validation trials must not enter observer fitting')
    model=offline_model(home);trials=[observation_trial(path,model) for path in training]
    qref=np.mean(np.concatenate([t['q'] for t in trials]),axis=0)
    candidates=[]
    # One common effective lag, not an independently fitted dead time per joint.
    for tc in [0.,.02,.04,.08,.12,.16,.24,.32,.48]:
        coefficients=[];losses=[];bounds=[]
        for joint in range(7):
            xx=[];yy=[]
            for t in trials:
                x=np.c_[t['residual'][:,joint],np.ones(len(t['q'])),t['q'][:,joint]-qref[joint]]
                x=_lowpass(x,tc);mask=t['valid'];xx.append(x[mask]);yy.append(t['external_torque'][mask,joint])
            x=np.concatenate(xx);y=np.concatenate(yy)
            ridge=np.diag([.02,.02,.005])*np.sqrt(len(y))
            p=lsq_linear(np.vstack([x,ridge]),np.r_[y,ridge@np.array([.4,0,0])],
                         bounds=([0,-2,-2],[1.5,2,2])).x
            coefficients.append(p.tolist());losses.append(float(np.mean((x@p-y)**2)))
            bounds.append(bool(np.any(p<np.array([0,-2,-2])+1e-5) or np.any(p>np.array([1.5,2,2])-1e-5)))
        candidates.append({'time_constant_s':tc,'coefficients':coefficients,'per_joint_mse':losses,
                           'mean_mse':float(np.mean(losses)),'at_bound':bounds})
    best=min(candidates,key=lambda x:x['mean_mse'])
    artifact={'schema':'fcc-wrench-observer-v1','q_reference':qref.tolist(),
        'resistance_gain':[p[0] for p in best['coefficients']],
        'joint_offset':[p[1] for p in best['coefficients']],
        'joint_position_slope':[p[2] for p in best['coefficients']],
        'time_constant_s':best['time_constant_s'],'sample_dt_fit_s':.02,
        'rotation_scale':home['controller_config']['jacobian_rotation_scale'],'minimum_singular_value':.05,
        'nominal_contact_gain':1.,'contact_gain_status':'assumed; no known-load/contact calibration',
        'scope':'Local low-speed approximation of Franka reported wrench; not true-force calibration',
        'time_constant_meaning':'effective free-space telemetry lag, not an identified proprietary filter or actuator delay',
        'canonical_convention':'base axes, about configured TCP, robot-on-environment',
        'raw_sdk_convention_observed':'base-origin spatial moment = TCP moment + p cross force',
        'home_model_sha256':{k:home['sha256'][k] for k in ['urdf','hardware_settings']},
        'joint_min_rad':np.min(np.concatenate([t['q'] for t in trials]),axis=0).tolist(),
        'joint_max_rad':np.max(np.concatenate([t['q'] for t in trials]),axis=0).tolist(),
        'training_trace_sha256':[t['quality']['trace_sha256'] for t in trials],
        'at_parameter_bound':best['at_bound'],
        'approximate_tolerance':{'relative':.3,'force_absolute_floor_N':.5,'moment_absolute_floor_Nm':.1}}
    output.mkdir(parents=True,exist_ok=False)
    write_json_new(output/'model.json',artifact)
    write_json_new(output/'fit.json',{'candidates':candidates,'selected_time_constant_s':best['time_constant_s'],
        'frame_audits':{str(p):t['frame_audit'] for p,t in zip(training,trials)},
        'normalization_does_not_change_existing_robot_telemetry_or_safety_limits':True})
    write_json_new(output/'frozen.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'model_sha256':digest(artifact),'weights_or_external_FT_used':False,
        'true_contact_force_accuracy_verified':False})
    return artifact


def score_wrenches(predicted,observed):
    p=np.asarray(predicted);y=np.asarray(observed)
    if p.shape!=y.shape or p.ndim!=2 or p.shape[1]!=6 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid wrench comparison')
    result={}
    for name,sl,floor in [('force',slice(0,3),.5),('moment',slice(3,6),.1)]:
        error=np.linalg.norm(p[:,sl]-y[:,sl],axis=1);magnitude=np.linalg.norm(y[:,sl],axis=1)
        tolerance=np.maximum(floor,.3*magnitude)
        result[name]={'vector_rmse':float(np.sqrt(np.mean(error**2))),'p95_vector_error':float(np.quantile(error,.95)),
                      'maximum_vector_error':float(error.max()),'fraction_within_tolerance':float(np.mean(error<=tolerance)),
                      'absolute_floor':floor,'relative_tolerance':.3}
    directions=np.linalg.norm(y[:,:3],axis=1)>1.
    nonzero=np.linalg.norm(p[:,:3],axis=1)>1e-8
    selected=directions & nonzero
    cosine=np.sum(p[selected,:3]*y[selected,:3],axis=1)/(np.linalg.norm(p[selected,:3],axis=1)*np.linalg.norm(y[selected,:3],axis=1))
    result['force_direction']={'samples':int(selected.sum()),
        'fraction_within_30_degrees':float(np.mean(cosine>=np.cos(np.deg2rad(30)))) if selected.any() else None,
        'note':'Only measured force magnitude >1 N; near-zero directions are not meaningful'}
    result['force_axis_sign_agreement']=[float(np.mean(np.sign(p[np.abs(y[:,j])>.5,j])==np.sign(y[np.abs(y[:,j])>.5,j])))
        if np.any(np.abs(y[:,j])>.5) else None for j in range(3)]
    return result


def evaluate_logged(home,artifact,directory,output):
    """Conditional observation check on recorded motion, NOT a contact/rollout test."""
    model=offline_model(home);t=observation_trial(directory,model);observer=WrenchObserver(artifact)
    predicted=[];truth=t['canonical_wrench']
    for q,v,res,p in zip(t['q'],t['dq'],t['residual'],t['position']):
        sample=model.sample(q,v,np.zeros(7),np.zeros(6))
        predicted.append(observer.step(q,res,sample.jacobian,p,.02)['external_wrench_tcp_base'])
    predicted=np.array(predicted)
    report={'schema':'fcc-wrench-observation-evaluation-v1','observed':str(directory),
        'evaluation':'conditional on recorded motion/torque residual; not an independent plant/contact rollout',
        'trace_was_used_for_fit':t['quality']['trace_sha256'] in artifact['training_trace_sha256'],
        'previously_viewed_development_data':True,'model_sha256':digest(artifact),
        'true_contact_force_accuracy_verified':False,'metrics':score_wrenches(predicted,truth),
        'frame_audit':t['frame_audit']}
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json_new(output/'evaluation.json',report)
    np.savez_compressed(output/'predictions.npz',time=t['time'],predicted=predicted,observed=truth)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
    for j,ax in enumerate(axes.T.flat):
        ax.plot(t['time'],truth[:,j],label='Franka (TCP/base normalized)',color='#2878c8')
        ax.plot(t['time'],predicted[:,j],label='Approximate observer',color='#25a26b')
        ax.set_ylabel(['Fx [N]','Fy [N]','Fz [N]','Mx [Nm]','My [Nm]','Mz [Nm]'][j]);ax.grid(True,alpha=.25)
    axes[0,0].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Recorded experiment time [s]')
    fig.suptitle('Franka telemetry surrogate — no weights/F/T reference; true contact force accuracy unverified')
    fig.tight_layout();fig.savefig(output/'wrench.png',dpi=140);plt.close(fig)
    return report


def evaluate_rollout(prediction,observed,output):
    prediction,observed,output=map(Path,(prediction,observed,output))
    sim=[json.loads(x) for x in (prediction/'trace.jsonl').read_text().splitlines()]
    real=[json.loads(x) for x in (observed/'trace.jsonl').read_text().splitlines()]
    ts=np.array([r['time_s'] for r in sim]);real=[r for r in real if ts[0]<=r['time_s']<=ts[-1]]
    if len(real)<2 or not all(r.get('external_wrench_valid') for r in sim):raise ValueError('Missing modeled observations/overlap')
    tr=np.array([r['time_s'] for r in real]);values=np.array([r['external_wrench_tcp_base'] for r in sim])
    pred=np.column_stack([np.interp(tr,ts,values[:,j]) for j in range(6)])
    truth=np.array([canonical_franka_wrench(r,stiffness_at_tcp=True) for r in real])
    metadata=json.loads((prediction/'manifest.json').read_text())
    if metadata.get('accepted_actions_sha256')!=digest((observed/'actions.jsonl').read_text()):
        raise ValueError('Rollout was not generated from this command stream')
    report={'schema':'fcc-wrench-rollout-evaluation-v1','prediction':str(prediction),'observed':str(observed),
        'evaluation':'Identified plant + observer, driven by recorded accepted actions; no recorded motion fed to the observer',
        'previously_viewed_development_data':True,'new_independent_contact_validation':False,
        'model_sha256':metadata['wrench_observer_sha256'],'true_contact_force_accuracy_verified':False,
        'metrics':score_wrenches(pred,truth),'samples':len(real)}
    output.mkdir(parents=True,exist_ok=False);write_json_new(output/'evaluation.json',report)
    np.savez_compressed(output/'predictions.npz',time=tr,predicted=pred,observed=truth)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
    for j,ax in enumerate(axes.T.flat):
        ax.plot(tr,truth[:,j],label='Franka (TCP/base normalized)',color='#2878c8')
        ax.plot(tr,pred[:,j],label='Plant + observer replay',color='#e29332')
        ax.set_ylabel(['Fx [N]','Fy [N]','Fz [N]','Mx [Nm]','My [Nm]','Mz [Nm]'][j]);ax.grid(True,alpha=.25)
    axes[0,0].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Recorded experiment time [s]')
    fig.suptitle('Approximate wrench prediction — recorded actions only; no true-force calibration')
    fig.tight_layout();fig.savefig(output/'wrench.png',dpi=140);plt.close(fig)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    fit=sub.add_parser('fit');fit.add_argument('--training',type=Path,nargs='+',required=True)
    test=sub.add_parser('evaluate');test.add_argument('--model',type=Path,required=True);test.add_argument('--observed',type=Path,required=True)
    roll=sub.add_parser('evaluate-rollout');roll.add_argument('--prediction',type=Path,required=True);roll.add_argument('--observed',type=Path,required=True);roll.add_argument('--output',type=Path,required=True)
    for p in (fit,test):p.add_argument('--context',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.operation=='evaluate-rollout':result=evaluate_rollout(args.prediction,args.observed,args.output)
    else:
        home=load_home(args.context)
        if args.operation=='fit':result=fit_observer(home,args.training,args.output)
        else:result=evaluate_logged(home,json.loads(args.model.read_text()),args.observed,args.output)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
