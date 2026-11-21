"""Frozen-model evaluation contracts using synthetic files only."""
from datetime import datetime,timezone
import json

import pytest
pytest.importorskip('matplotlib')
from dataclasses import asdict
from franka_compliant_control.twin_model import PlantParameters
from franka_compliant_control.twin_experiment import digest,write_json_new
from franka_compliant_control.sysid_evaluate import evaluate


@pytest.fixture
def campaign(tmp_path):
    candidate=tmp_path/'candidate';candidate.mkdir()
    plant=asdict(PlantParameters());write_json_new(candidate/'plant.json',plant)
    write_json_new(candidate/'fit.json',{'training_trials':[{'quality':{'trace_sha256':'training-only'}}]})
    freeze={'plant_sha256':digest(plant),'frozen_utc':datetime.fromtimestamp(100,timezone.utc).isoformat(),
            'primary_validation_criterion':'50% improvement','minimum_relative_improvement':.5}
    write_json_new(candidate/'frozen.json',freeze)
    observed=tmp_path/'observed';observed.mkdir()
    actions=json.dumps({'response':{'accepted_timestamp_s':200}})+'\n'
    (observed/'actions.jsonl').write_text(actions)
    meta={'backend':'robot','role':'validation','home_sha256':{'home':'fixed'},'policy_sha256':'same',
          'controller_config':{'mock':True},'policy':{'duration_s':1.}}
    paths=[]
    for name,offset in [('observed',0.),('nominal',.01),('fitted',.001)]:
        path=tmp_path/name;path.mkdir(exist_ok=True);paths.append(path)
        m={**meta,'backend':'robot' if name=='observed' else 'mujoco','plant':plant,
           'accepted_actions_sha256':digest(actions)}
        write_json_new(path/'manifest.json',m)
        write_json_new(path/'summary.json',{'completed':True})
        with (path/'trace.jsonl').open('w') as f:
            for t in (0.,.5,1.):
                f.write(json.dumps({'time_s':t,'robot_time_s':t,'position':[t+offset,0,0],
                    'quaternion_xyzw':[0,0,0,1],'positions':[t]*7,'velocities':[1]*7,'twist':[1,0,0,0,0,0],
                    'last_desired_joint_torques':[0]*7,'commanded_task_wrench':[0]*6,
                    'policy':{'policy_phase':'move'},'phase':'active','reason':'none'})+'\n')
    return candidate,*paths


def test_holdout_metrics_and_freeze_checks(campaign,tmp_path):
    c,obs,nom,fit=campaign
    result=evaluate(c,obs,nom,fit,tmp_path/'evaluation')
    assert result['predeclared_criterion_passed'] and result['full_coverage']
    assert result['relative_improvement']==pytest.approx(.9)
    assert not result['global_robot_calibrated']


def test_post_collection_freeze_is_rejected(campaign,tmp_path):
    c,obs,nom,fit=campaign
    path=c/'frozen.json';data=json.loads(path.read_text());data['frozen_utc']=datetime.fromtimestamp(300,timezone.utc).isoformat()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='precede'):evaluate(c,obs,nom,fit,tmp_path/'evaluation')


def test_validation_training_leak_is_rejected(campaign,tmp_path):
    c,obs,nom,fit=campaign
    (c/'fit.json').write_text(json.dumps({'training_trials':[{'quality':{'trace_sha256':digest((obs/'trace.jsonl').read_text())}}]}))
    with pytest.raises(ValueError,match='used for fitting'):evaluate(c,obs,nom,fit,tmp_path/'evaluation')


def test_changed_candidate_is_rejected(campaign,tmp_path):
    c,obs,nom,fit=campaign
    p=json.loads((c/'plant.json').read_text());p['coulomb'][0]=.9;(c/'plant.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='changed'):evaluate(c,obs,nom,fit,tmp_path/'evaluation')
