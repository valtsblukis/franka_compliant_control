"""Evaluate a frozen model on a separate run; report prediction, not task success."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .twin_compare import compare,load_rows
from .twin_experiment import digest,write_json_new
from .twin_model import PlantParameters


def evaluate(candidate,observed,nominal,fitted,output):
    candidate,observed,nominal,fitted,output=map(Path,(candidate,observed,nominal,fitted,output))
    frozen=json.loads((candidate/'frozen.json').read_text())
    plant=json.loads((candidate/'plant.json').read_text())
    if digest(plant)!=frozen['plant_sha256']:raise ValueError('Candidate changed after freezing')
    manifest=json.loads((observed/'manifest.json').read_text())
    if manifest.get('role')!='validation':raise ValueError('Evaluation requires a held-out validation role')
    fit=json.loads((candidate/'fit.json').read_text())
    observation_hash=digest((observed/'trace.jsonl').read_text())
    if any(t['quality']['trace_sha256']==observation_hash for t in fit['training_trials']):
        raise ValueError('Validation data was used for fitting')
    first=json.loads((observed/'actions.jsonl').read_text().splitlines()[0])
    started=datetime.fromtimestamp(first['response']['accepted_timestamp_s'],timezone.utc)
    if datetime.fromisoformat(frozen['frozen_utc'])>=started:
        raise ValueError('Freeze must precede data collection (synchronize client/server wall clocks)')
    expected_actions=digest((observed/'actions.jsonl').read_text())
    for directory in (nominal,fitted):
        prediction=json.loads((directory/'manifest.json').read_text())
        if prediction.get('accepted_actions_sha256')!=expected_actions:raise ValueError('Predictions did not replay these accepted commands')
    prediction=json.loads((fitted/'manifest.json').read_text())
    predicted_plant={k:v for k,v in prediction['plant'].items() if k in PlantParameters.__dataclass_fields__}
    if predicted_plant!=plant:raise ValueError('Prediction used a different plant than the frozen candidate')
    output.mkdir(parents=True,exist_ok=False)
    before=compare(nominal,observed,output/'nominal')
    after=compare(fitted,observed,output/'fitted')
    required=float(frozen.get('minimum_relative_improvement',.5))
    if not np.isfinite(required) or not 0<required<1:raise ValueError('Invalid predeclared improvement threshold')
    if before['position_rmse_m']<=1e-12:raise ValueError('Baseline already at numerical floor; use an explicit absolute-error evaluation')
    result={'schema':'fcc-sysid-evaluation-v1','frozen_model_sha256':frozen['plant_sha256'],
            'validation_trace_sha256':observation_hash,'candidate_unchanged':True,'validation_not_used_for_fit':True,
            'full_coverage':before['full_run_comparison'] and after['full_run_comparison'],
            'nominal_position_rmse_m':before['position_rmse_m'],'fitted_position_rmse_m':after['position_rmse_m'],
            'relative_improvement':1-after['position_rmse_m']/before['position_rmse_m'],
            'nominal_max_position_error_m':before['position_max_error_m'],
            'fitted_max_position_error_m':after['position_max_error_m'],
            'nominal_orientation_rmse_rad':before['orientation_rmse_rad'],
            'fitted_orientation_rmse_rad':after['orientation_rmse_rad'],
            'criterion':frozen['primary_validation_criterion'],'minimum_relative_improvement':required,
            'position_source':'SDK/URDF encoder-derived TCP pose, not independent external metrology',
            'global_robot_calibrated':False}
    result['predeclared_criterion_passed']=bool(result['full_coverage'] and result['relative_improvement']>=required)
    write_json_new(output/'evaluation.json',result)
    real=load_rows(observed/'trace.jsonl');a=load_rows(nominal/'trace.jsonl');b=load_rows(fitted/'trace.jsonl')
    t=np.array([r['time_s'] for r in real]);positions=np.array([r['position'] for r in real])
    def interpolate(rows,key):
        tt=np.array([r['time_s'] for r in rows]);values=np.array([r[key] for r in rows])
        return np.column_stack([np.interp(t,tt,values[:,j]) for j in range(values.shape[1])])
    pa,pb=interpolate(a,'position'),interpolate(b,'position')
    # Secondary breakdown prevents a long quiet dwell from hiding moving errors.
    segments={}
    for label,mask in [('motion',np.array([r['policy']['policy_phase']=='move' for r in real])),
                       ('dwell',np.array([r['policy']['policy_phase']=='dwell' for r in real]))]:
        if mask.any():segments[label]={'samples':int(mask.sum()),
            'nominal_rmse_m':float(np.sqrt(np.mean(np.sum((pa[mask]-positions[mask])**2,axis=1)))),
            'fitted_rmse_m':float(np.sqrt(np.mean(np.sum((pb[mask]-positions[mask])**2,axis=1))))}
    write_json_new(output/'segment_metrics.json',segments)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,11),sharex=True)
    for j,label in enumerate('XYZ'):
        ax=axes[j,0];ax.plot(t,1000*positions[:,j],color='#2878c8',label='Real robot')
        ax.plot(t,1000*pa[:,j],'--',color='#999999',label='Original model')
        ax.plot(t,1000*pb[:,j],color='#25a26b',label='Frozen identified model')
        ax.set_ylabel(f'TCP {label} [mm, base frame]');ax.legend(fontsize=8)
    axes[0,1].plot(t,1000*np.linalg.norm(pa-positions,axis=1),'--',color='#999999',label='Original mismatch')
    axes[0,1].plot(t,1000*np.linalg.norm(pb-positions,axis=1),color='#25a26b',label='Identified mismatch')
    axes[0,1].set_ylabel('3D position prediction error [mm]');axes[0,1].legend(fontsize=8)
    for label,color,values in [('Real robot','#2878c8',np.array([r['twist'] for r in real])),
                               ('Original','#999999',interpolate(a,'twist')),('Identified','#25a26b',interpolate(b,'twist'))]:
        axes[1,1].plot(t,np.linalg.norm(values[:,:3],axis=1),label=label,color=color)
    axes[1,1].set_ylabel('TCP speed [m/s]');axes[1,1].legend(fontsize=8)
    for label,color,values in [('Real requested','#2878c8',np.array([r['commanded_task_wrench'] for r in real])),
                               ('Original requested','#999999',interpolate(a,'commanded_task_wrench')),
                               ('Identified requested','#25a26b',interpolate(b,'commanded_task_wrench'))]:
        axes[2,1].plot(t,np.linalg.norm(values[:,:3],axis=1),label=label,color=color)
    axes[2,1].set_ylabel('Requested force [N], not contact force');axes[2,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(True,alpha=.25)
    for ax in axes[-1]:ax.set_xlabel('Validation experiment time [s]')
    fig.suptitle(f'Held-out validation: {before["position_rmse_m"]*1000:.2f} → {after["position_rmse_m"]*1000:.2f} mm RMS; model frozen before collection')
    fig.tight_layout();fig.savefig(output/'validation.png',dpi=140);plt.close(fig)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['candidate','observed','nominal','fitted','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    result=evaluate(args.candidate,args.observed,args.nominal,args.fitted,args.output)
    print(json.dumps(result,indent=2));return 0 if result['predeclared_criterion_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
