"""Low-frequency effective resistance identification; robot-independent regression.

The regression accepts arrays of any joint count. The saved-log adapter below is
Franka-specific: tau_J_d excludes robot-side gravity compensation. Never apply
that convention to another robot without adapting the inverse-dynamics residual.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

from .twin_experiment import digest, write_json_new
from .twin_model import PlantParameters


def trial_role(directory, manifest=None):
    """A validation sidecar can classify a trial without rewriting raw metadata."""
    directory=Path(directory)
    manifest=manifest or json.loads((directory/'manifest.json').read_text())
    sidecar=directory/'dataset-role.json'
    declared=json.loads(sidecar.read_text()).get('role') if sidecar.exists() else None
    if declared is not None and declared not in ('training','validation','development'):
        raise ValueError('Unknown dataset role')
    original=manifest.get('role')
    if original is not None and declared is not None and original!=declared:
        raise ValueError('Conflicting dataset classifications')
    return declared or original


def prepare_trial(directory, model, *, sample_period=.02, smoothing_s=.22):
    """Reconstruct low-frequency derivatives from positions, per trial, not across joins."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    role=trial_role(directory,manifest)
    if role is not None:manifest={**manifest,'role':role}
    stop=json.loads((directory/'stop.json').read_text())
    if not stop.get('stopped'):raise ValueError('Trial lacks a confirmed stationary stop')
    if manifest.get('role') in ('training','validation'):
        completion_path=directory/'completion.json'
        if not completion_path.is_file() or not json.loads(completion_path.read_text()).get('completed'):
            raise ValueError('Incomplete calibration trial; do not silently fit an aborted prefix')
    failure_path=directory/'failure.json'
    if failure_path.exists() and json.loads(failure_path.read_text()).get('error')!='Final home did not settle within the scheduled dwell':
        raise ValueError('Aborted/faulted trial is not an approved identification dataset')
    raw=[json.loads(line) for line in (directory/'trace.jsonl').read_text().splitlines() if line.strip()]
    if len(raw)<30:raise ValueError('At least 30 recorded samples are required')
    # Use the measurement clock, not HTTP receipt time. Retain its initial
    # experiment-clock offset for matching segment labels and references.
    unique=[];last=-np.inf
    for row in raw:
        stamp=row.get('robot_time_s')
        if stamp is None or not np.isfinite(stamp) or stamp<last:
            raise ValueError('Missing/nonmonotone robot measurement clock')
        if stamp>last:unique.append(row)
        last=stamp
    rows=unique
    if any(r.get('phase')!='active' or not r.get('sample_valid',True)
           or r.get('sample_age_s',0)>.02 for r in rows):
        raise ValueError('Invalid/faulted/stale samples; split or exclude this trial explicitly')
    t=np.array([r['robot_time_s'] for r in rows]);t=t-t[0]+rows[0]['time_s']
    if np.max(np.diff(t))>.08:raise ValueError('Measurement gap exceeds 80 ms')
    grid=np.arange(t[0],t[-1]+1e-10,sample_period)
    def interpolate(key):
        a=np.asarray([r[key] for r in rows],dtype=float)
        if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Invalid joint/torque observations')
        return np.column_stack([np.interp(grid,t,a[:,j]) for j in range(a.shape[1])])
    qraw=interpolate('positions');reported_dq=interpolate('velocities')
    torque=interpolate('last_desired_joint_torques')
    if torque.shape!=qraw.shape or reported_dq.shape!=qraw.shape:raise ValueError('Joint vector shapes differ')
    window=max(7,int(round(smoothing_s/sample_period))|1)
    if window>=len(grid):raise ValueError('Insufficient data for requested smoothing')
    q=savgol_filter(qraw,window,3,axis=0)
    dq=savgol_filter(qraw,window,3,deriv=1,delta=sample_period,axis=0)
    ddq=savgol_filter(qraw,window,3,deriv=2,delta=sample_period,axis=0)
    tau=savgol_filter(torque,window,3,axis=0)
    residual=np.empty_like(q)
    for i,(qi,vi,ai,ti) in enumerate(zip(q,dq,ddq,tau)):
        sample=model.sample(qi,vi,np.zeros(q.shape[1]),np.zeros(6))
        residual[i]=ti-sample.mass@ai-sample.coriolis
    indices=np.maximum(0,np.searchsorted(t,grid,side='right')-1)
    if manifest.get('schema') in ('fcc-force-aware-real-v1','fcc-learned-reference-real-v1'):
        # A goal-seeking controller has no prescribed move/dwell trajectory.
        # Use only the final scheduled second AND measured stationary samples
        # for the noise estimate; ringing earlier in the interval is motion.
        ends=[e['time_s'] for e in manifest['events'][1:]]+[manifest['duration_s']]
        dwell=np.array([any(end-1.<=t<end for end in ends)
                        and np.max(np.abs(rows[i]['velocities']))<.04
                        and np.linalg.norm(rows[i]['twist'][:3])<.015
                        for t,i in zip(grid,indices)])
        moving=~dwell
    else:
        moving=np.array([rows[i]['policy'].get('policy_phase')=='move' and rows[i]['policy'].get('leg',-1)>=0
                         for i in indices])
        dwell=np.array([rows[i]['policy'].get('policy_phase')=='dwell' for i in indices])
    valid=np.ones(len(grid),dtype=bool);valid[:window]=False;valid[-window:]=False
    # Noise floor is an observation property, estimated during declared dwells.
    if dwell.sum()<10:raise ValueError('Identification requires declared stationary dwells')
    # Gate the position-derived velocities using their own dwell variation.
    # Raw SDK dq noise is recorded separately; it is not the differentiated
    # signal used in this regression. A fixed floor prevents tiny-drift fits.
    noise=np.std(dq[dwell],axis=0)
    return {'directory':str(directory),'manifest':manifest,'time':grid,'q':q,'dq':dq,'ddq':ddq,
            'residual':residual,'torque':tau,'moving':moving,'dwell':dwell,'valid':valid,
            'velocity_noise_std':noise,'quality':{'raw_samples':len(raw),'unique_samples':len(rows),
            'median_period_s':float(np.median(np.diff(t))),'maximum_gap_s':float(np.max(np.diff(t))),
            'maximum_sample_age_s':max(r.get('sample_age_s',0) for r in rows),
            'sample_period_s':sample_period,'smoothing_s':window*sample_period,
            'reported_velocity_dwell_std':np.std(reported_dq[dwell],axis=0).tolist(),
            'derived_velocity_dwell_std':noise.tolist(),
            'source_task_failure':json.loads((directory/'failure.json').read_text()) if (directory/'failure.json').exists() else None,
            'derivative_method':'Savitzky-Golay position derivatives, cubic; same smoothing on delivered torque',
            'torque_convention':'Franka tau_J_d, gravity excluded; residual = tau - M*qdd - c',
            'trace_sha256':digest((directory/'trace.jsonl').read_text())}}


def _robust_fit(x,y,prior,lower,upper):
    weights=np.ones(len(y));params=np.array(prior,dtype=float)
    regularization=.01*len(y)
    for _ in range(8):
        a=np.vstack((np.sqrt(weights)[:,None]*x,np.sqrt(regularization)*np.eye(x.shape[1])))
        b=np.r_[np.sqrt(weights)*y,np.sqrt(regularization)*np.asarray(prior)]
        params=lsq_linear(a,b,bounds=(lower,upper)).x
        error=y-x@params
        scale=max(.03,1.4826*np.median(np.abs(error-np.median(error))))
        weights=np.minimum(1.,1.5*scale/np.maximum(np.abs(error),1e-12))
    return params


def fit_resistance(trials, *, prior_coulomb, prior_viscous, fit_bias=True,
                   minimum_speed=.002, maximum_acceleration=.30,
                   minimum_samples_each_direction=20, bootstrap=40, seed=7321,
                   coulomb_upper=3., bias_bound=2.):
    """Fit Fc>=0 and optional local bias; freeze viscous/inertia/delay.

    No independent-sample confidence claim: bootstrap uses contiguous 0.4 s
    blocks within each trial. Intervals are conditional on this model/smoothing.
    A joint without enough bidirectional excitation retains its prior.
    """
    if not trials:raise ValueError('Training trials are required')
    if any(t.get('manifest',{}).get('role')=='validation' for t in trials):
        raise ValueError('Held-out validation trials cannot be used for fitting')
    ndof=len(prior_coulomb)
    if len(prior_viscous)!=ndof or any(t['q'].shape[1]!=ndof for t in trials):
        raise ValueError('Inconsistent joint counts')
    if any(not np.isfinite(v) or v<0 for v in (*prior_coulomb,*prior_viscous)):
        raise ValueError('Invalid resistance priors')
    upper_fc=np.broadcast_to(np.asarray(coulomb_upper,dtype=float),(ndof,))
    bound_bias=np.broadcast_to(np.asarray(bias_bound,dtype=float),(ndof,))
    if (not np.isfinite(upper_fc).all() or not np.isfinite(bound_bias).all()
            or np.any(upper_fc<=0) or np.any(bound_bias<=0) or np.any(np.asarray(prior_coulomb)>upper_fc)):
        raise ValueError('Invalid fitting bounds')
    rng=np.random.default_rng(seed)
    fitted=list(prior_coulomb);bias=[0.]*ndof;reports=[]
    for joint in range(ndof):
        ys=[];xs=[];blocks=[];thresholds=[]
        for trial in trials:
            threshold=max(minimum_speed,4*float(trial['velocity_noise_std'][joint]))
            thresholds.append(threshold)
            mask=(trial['valid']&trial['moving']&(np.abs(trial['dq'][:,joint])>threshold)
                  &(np.abs(trial['ddq'][:,joint])<maximum_acceleration))
            sign=np.sign(trial['dq'][:,joint])
            x=np.column_stack((sign,np.ones(len(sign)))) if fit_bias else sign[:,None]
            y=trial['residual'][:,joint]-prior_viscous[joint]*trial['dq'][:,joint]
            xs.append(x[mask]);ys.append(y[mask])
            length=max(1,round(.4/trial['quality']['sample_period_s']))
            for start in range(0,len(mask),length):
                selection=np.flatnonzero(mask[start:start+length])+start
                if len(selection):blocks.append((x[selection],y[selection]))
        x=np.concatenate(xs);y=np.concatenate(ys)
        counts=[int(np.sum(x[:,0]>0)),int(np.sum(x[:,0]<0))]
        report={'joint':joint+1,'positive_samples':counts[0],'negative_samples':counts[1],
                'velocity_thresholds_rad_s':thresholds,'fitted':False,
                'viscous_status':'fixed prior; not enough bandwidth/excitation assumed for separate identification'}
        if min(counts)<minimum_samples_each_direction:
            report['reason']='insufficient bidirectional excitation; retain prior'
            reports.append(report);continue
        prior=[prior_coulomb[joint],0.] if fit_bias else [prior_coulomb[joint]]
        lower=[0.,-bound_bias[joint]] if fit_bias else [0.]
        upper=[upper_fc[joint],bound_bias[joint]] if fit_bias else [upper_fc[joint]]
        params=_robust_fit(x,y,prior,lower,upper)
        fitted[joint]=float(params[0]);bias[joint]=float(params[1]) if fit_bias else 0.
        predictions=x@params
        draws=[]
        for _ in range(bootstrap):
            selected=[blocks[i] for i in rng.integers(0,len(blocks),size=len(blocks))]
            bx=np.concatenate([b[0] for b in selected]);by=np.concatenate([b[1] for b in selected])
            if np.unique(bx[:,0]).size<2:continue
            draws.append(_robust_fit(bx,by,prior,lower,upper))
        report.update(fitted=True,coulomb_Nm=fitted[joint],local_bias_Nm=bias[joint],
            moving_residual_rmse_Nm=float(np.sqrt(np.mean((y-predictions)**2))),
            regressor_condition=float(np.linalg.cond(x)),
            at_bound=bool(np.any(params<np.array(lower)+1e-4) or np.any(params>np.array(upper)-1e-4)),
            conditional_block_bootstrap_95=np.quantile(draws,[.025,.975],axis=0).tolist() if draws else None)
        reports.append(report)
    return {'schema':'fcc-resistance-fit-v1','model':'Fc*sign(dq) + fixed_B*dq + local_joint_bias',
            'coulomb':fitted,'joint_bias':bias,'viscous':list(prior_viscous),'joints':reports,
            'settings':{'minimum_speed':minimum_speed,'maximum_acceleration':maximum_acceleration,
                        'minimum_samples_each_direction':minimum_samples_each_direction,'fit_bias':fit_bias,
                        'bootstrap_blocks_s':.4,'bootstrap_replicates':bootstrap,'seed':seed,
                        'coulomb_upper_Nm':upper_fc.tolist(),'bias_bound_Nm':bound_bias.tolist()},
            'training_trials':[{'directory':t['directory'],'quality':t['quality']} for t in trials],
            'data_scope':{'joint_min_rad':np.min(np.concatenate([t['q'] for t in trials]),axis=0).tolist(),
                          'joint_max_rad':np.max(np.concatenate([t['q'] for t in trials]),axis=0).tolist(),
                          'maximum_observed_speed_rad_s':np.max(np.abs(np.concatenate([t['dq'] for t in trials])),axis=0).tolist()},
            'not_identified':['rigid-body inertias','payload inertia','actuator delay','reflected motor inertia',
                              'separate static/Stribeck friction','global gravity/cable model','contact mechanics'],
            'scope':'local low-speed effective resistance under current robot compensation, not intrinsic motor friction'}


def fit_franka_logs(home, directories, output, *, fit_bias=True, friction_impedance=.9999):
    from .mujoco_twin import offline_model
    for directory in directories:
        if json.loads((Path(directory)/'manifest.json').read_text()).get('role')=='validation':
            raise ValueError('Held-out validation trials cannot be used for fitting')
    model=offline_model(home)
    trials=[prepare_trial(directory,model) for directory in directories]
    for trial in trials:
        m=trial['manifest']
        if any(m['home_sha256'][key]!=home['sha256'][key] for key in ('urdf','hardware_settings')):
            raise ValueError('Robot/TCP/load model changed across identification data')
        # Inverse dynamics uses delivered torque, not the controller gains or
        # watchdog. Closed-loop validation must replay each trial's own config.
    nominal=PlantParameters()
    fit=fit_resistance(trials,prior_coulomb=nominal.coulomb,prior_viscous=nominal.viscous,fit_bias=fit_bias)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    fit['numerical_friction']={'impedance':friction_impedance,'time_constant_s':.01,
                              'meaning':'fixed numerical convention tested for below-threshold stiction and timestep convergence, not a learned hardware parameter'}
    write_json_new(output/'fit.json',fit)
    values=asdict(nominal)|{'coulomb':fit['coulomb'],'joint_bias':fit['joint_bias'],
                           'friction_impedance':friction_impedance,'friction_timeconst':.01}
    parameters=PlantParameters(**values)
    write_json_new(output/'plant.json',asdict(parameters))
    return fit,parameters


def freeze_candidate(directory,minimum_relative_improvement=.5):
    from datetime import datetime,timezone
    if not np.isfinite(minimum_relative_improvement) or not 0<minimum_relative_improvement<1:
        raise ValueError('Improvement criterion must be between zero and one')
    directory=Path(directory)
    plant=json.loads((directory/'plant.json').read_text());PlantParameters(**plant)
    fit=json.loads((directory/'fit.json').read_text())
    if not any(j['fitted'] for j in fit['joints']):raise ValueError('No parameters were identified')
    if any(j.get('at_bound') for j in fit['joints']):raise ValueError('Review bound-saturated estimates before freezing')
    result={'frozen_utc':datetime.now(timezone.utc).isoformat(),'plant_sha256':digest(plant),
            'minimum_relative_improvement':minimum_relative_improvement,
            'primary_validation_criterion':f'At least {100*minimum_relative_improvement:g}% lower position RMS error on held-out data, without fitted time/pose shifts',
            'selection':'Bounded effective-resistance fit; operator must review training diagnostics before validation',
            'validation_not_yet_collected':True,
            'scope':'Local low-speed free-space; not full-robot/contact identification'}
    write_json_new(directory/'frozen.json',result);return result


def main(argv=None):
    import argparse
    from .twin_experiment import load_home
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',type=Path)
    parser.add_argument('--training',type=Path,nargs='+')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--freeze-existing',type=Path,help='Freeze an already reviewed fit, before collecting validation')
    parser.add_argument('--minimum-relative-improvement',type=float,default=.5)
    parser.add_argument('--no-bias',action='store_true')
    parser.add_argument('--soft-friction-ablation',action='store_true',
                        help='Diagnostic only: preserve MuJoCo default soft friction instead of tested near-static convention')
    args=parser.parse_args(argv)
    if args.freeze_existing:
        if args.context or args.training or args.output:parser.error('Do not mix fitting and freeze-existing arguments')
        print(json.dumps(freeze_candidate(args.freeze_existing,args.minimum_relative_improvement),indent=2));return 0
    if not args.context or not args.training or not args.output:
        parser.error('Fitting requires --context, --training and --output')
    fit,parameters=fit_franka_logs(load_home(args.context),args.training,args.output,
                                  fit_bias=not args.no_bias,friction_impedance=None if args.soft_friction_ablation else .9999)
    print(json.dumps({'fitted_joints':[j['joint'] for j in fit['joints'] if j['fitted']],
                      'plant':asdict(parameters),'output':str(args.output)},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
