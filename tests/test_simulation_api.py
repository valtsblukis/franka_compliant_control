"""Dependency/library contracts, independent of the RL checkout or run archive."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import pytest


def test_core_has_no_learning_imports():
    root=Path('src/franka_compliant_control')
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and not node.level:names=[node.module or '']
            assert not any(n.split('.')[0] in {'franka_motion_rl','gymnasium','torch','stable_baselines3'} for n in names),path


def test_physics_and_presets_without_native_or_learning_imports():
    pytest.importorskip('mujoco')
    code='''import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('franka_motion_rl','gymnasium','torch','stable_baselines3','scipy') or name.endswith(('._native','._twin_native')):raise AssertionError(name)
sys.meta_path.insert(0,Block())
from franka_compliant_control.simulation.cli import smoke
assert smoke(seconds=.02)['passed']
'''
    subprocess.run([sys.executable,'-c',code],check=True)


def test_reference_inputs_are_packaged_non_deployable_and_isolated():
    from franka_compliant_control.simulation.presets import reference_context, identified_plant
    h=reference_context()
    assert h['simulation_only'] and h['hardware_settings']['read_only']
    assert h['hardware_settings']['robot_ip']=='offline.invalid'
    assert h['sha256']['home']==h['provenance']['source_home_sha256']
    h['home']['position'][0]=999
    assert reference_context()['home']['position'][0]!=999
    assert identified_plant().coulomb[5]==pytest.approx(.42549454657793156)


def test_installed_native_manifest_fallback(monkeypatch,tmp_path):
    pytest.importorskip('franka_compliant_control._twin_native')
    from franka_compliant_control.simulation import sdk
    from franka_compliant_control import _twin_native
    monkeypatch.setattr(sdk,'ROOT',tmp_path)
    assert sdk.native_inputs_digest()==_twin_native.twin_inputs_sha256
