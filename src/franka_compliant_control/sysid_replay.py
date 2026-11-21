"""Closed-loop prediction from recorded accepted commands; never fits time/pose shifts."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .mujoco_twin import simulation_trace,native_inputs_digest
from .sysid_experiment import WaypointPolicy
from .recorded_policy import RecordedGoalSchedule,recorded_policy_spec,accepted_events
from .twin_compare import compare
from .twin_experiment import ManualPosePolicy,load_home,write_json_new,digest
from .twin_model import PlantParameters


def replay(home,observed,parameters,output,*,observer_config=None):
    from . import _twin_native
    if native_inputs_digest()!=_twin_native.twin_inputs_sha256:raise RuntimeError('Stale controller bridge')
    observed,output=Path(observed),Path(output)
    manifest=json.loads((observed/'manifest.json').read_text())
    if any(home['sha256'][k]!=manifest['home_sha256'][k] for k in ('urdf','hardware_settings')):
        raise ValueError('Cannot replay a different robot/TCP/load model')
    if observer_config is not None and any(observer_config['home_model_sha256'][k]!=home['sha256'][k]
                                           for k in ('urdf','hardware_settings')):
        raise ValueError('Observer was fitted for a different robot/TCP/load model')
    context=deepcopy(home);context['controller_config']=manifest['controller_config'];context['sha256']=manifest['home_sha256']
    recorded=manifest.get('schema') in ('fcc-force-aware-real-v1','fcc-learned-reference-real-v1')
    if recorded:
        policy=RecordedGoalSchedule(recorded_policy_spec(manifest))
        policy_sha=digest(policy.manifest())
    else:
        cls=WaypointPolicy if manifest['policy']['schema']=='fcc-waypoint-policy-v1' else ManualPosePolicy
        policy=cls(**{k:v for k,v in manifest['policy'].items() if k in cls.__dataclass_fields__})
        policy_sha=manifest['policy_sha256']
        if digest(policy.manifest())!=policy_sha:raise ValueError('Policy fingerprint mismatch')
    startup=json.loads((observed/'startup.json').read_text())
    origin=startup['policy_origin_server_monotonic_s']
    actions=[json.loads(line) for line in (observed/'actions.jsonl').read_text().splitlines()]
    events=accepted_events(actions,origin,action_key='action' if recorded else 'policy')
    output.mkdir(parents=True,exist_ok=False)
    write_json_new(output/'manifest.json',{'schema':'fcc-twin-run-v1','backend':'mujoco',
        'home_sha256':manifest['home_sha256'],'controller_config':manifest['controller_config'],
        'policy':policy.manifest(),'policy_sha256':policy_sha,
        'plant':parameters.manifest(),'observed':str(observed),
        'wrench_observer':observer_config,'wrench_observer_sha256':digest(observer_config) if observer_config else None,
        'replay':'Actual server-accepted action times and deadlines, rounded up to next 1 ms callback; no time/pose fit',
        'initialization':'Observed first-active joints and desired torque; zero true initial velocity',
        'controller_source_sha256':_twin_native.controller_source_sha256,
        'accepted_actions_sha256':digest((observed/'actions.jsonl').read_text())})
    count=0;last=None
    with (output/'trace.jsonl').open('x') as f:
        for row in simulation_trace(context,policy,parameters,command_events=events,initial_state=startup['state'],observer_config=observer_config):
            f.write(json.dumps(row,allow_nan=False)+'\n');count+=1;last=row
    result={'completed':bool(last and last['time_s']>=policy.duration_s and last['phase']=='active'),
            'samples':count,'last_reason':last['reason'] if last else 'no samples'}
    write_json_new(output/'summary.json',result)
    comparison=compare(output,observed,output/'comparison')
    result['comparison']=comparison
    write_json_new(output/'prediction.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context',type=Path,required=True);p.add_argument('--observed',type=Path,required=True)
    p.add_argument('--plant',type=Path);p.add_argument('--observer',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    params=PlantParameters(**json.loads(a.plant.read_text())) if a.plant else PlantParameters()
    observer=json.loads(a.observer.read_text()) if a.observer else None
    print(json.dumps(replay(load_home(a.context),a.observed,params,a.output,observer_config=observer),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
