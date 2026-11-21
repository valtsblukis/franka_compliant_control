"""Comparison accounting: failed prefixes must not look like successful sys-id."""
import json
from pathlib import Path

import pytest
pytest.importorskip('matplotlib')

from franka_compliant_control.twin_compare import compare
from franka_compliant_control.twin_experiment import write_json_new


def make_run(path,*,complete=True,offset=0.,times=(0.,.5,1.)):
    path.mkdir()
    write_json_new(path/'manifest.json',{'backend':'mujoco','home_sha256':{'home':'fixed'},
        'policy_sha256':'policy','controller_config':{'test':'same'},'policy':{'duration_s':1.}})
    write_json_new(path/'summary.json',{'completed':complete})
    with (path/'trace.jsonl').open('w') as f:
        for t in times:
            f.write(json.dumps({'time_s':t,'robot_time_s':t,'position':[t+offset,0,0],
                'quaternion_xyzw':[0,0,0,1],'positions':[t]*7,'velocities':[1]*7,
                'twist':[1,0,0,0,0,0],'last_desired_joint_torques':[0]*7,
                'phase':'active' if complete else 'stopping','reason':'none' if complete else 'measured_overspeed'})+'\n')


def test_full_and_partial_comparisons_are_explicit(tmp_path):
    a,b,c=tmp_path/'a',tmp_path/'b',tmp_path/'c'
    make_run(a);make_run(b,offset=.001);make_run(c,complete=False,times=(0.,.1))
    result=compare(a,b,tmp_path/'full')
    assert result['full_run_comparison']
    assert result['position_rmse_m']==pytest.approx(.001)
    result=compare(a,c,tmp_path/'partial')
    assert result['position_rmse_m']==pytest.approx(0)
    assert not result['full_run_comparison'] and result['coverage_fraction']==pytest.approx(.1)
    assert result['warning'] and not result['observed_completed']


def test_reject_different_experiments(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';make_run(a);make_run(b)
    path=b/'manifest.json';manifest=json.loads(path.read_text());manifest['policy_sha256']='changed'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='different'):compare(a,b,tmp_path/'output')
