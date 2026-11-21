"""Offline checks for posture housekeeping, never real motion."""
import numpy as np
import pytest
pytest.importorskip('mujoco')
from franka_compliant_control.posture_restore import reference,run


def test_joint_reference_is_smooth_and_bounded():
    start=np.array([-.22,-.4,.1,-1.9,.02,1.5,.01]);end=start+[.14,0,-.1,.02,-.02,.01,-.01]
    plan={'start':{'positions':start.tolist()},'equilibrium_joints':end.tolist()}
    assert reference(plan,0)==pytest.approx(start)
    assert reference(plan,1)==pytest.approx(start)
    assert reference(plan,21)==pytest.approx(end)
    assert reference(plan,26)==pytest.approx(end)
    t=np.arange(0,26,.02);q=np.array([reference(plan,x) for x in t])
    v=np.diff(q,axis=0)/.02;a=np.diff(v,axis=0)/.02
    assert np.max(np.abs(v))<.02 and np.max(np.abs(a))<.003
    assert np.max(abs(v[49]))<1e-6


def test_no_execution_without_acknowledgement(tmp_path):
    with pytest.raises(ValueError,match='acknowledgement'):run({}, {}, tmp_path/'screen',tmp_path/'real')
    assert not (tmp_path/'real').exists()
