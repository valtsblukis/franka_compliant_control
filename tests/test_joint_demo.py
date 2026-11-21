"""URDF kinematics and fake-clock HTTP producer tests; no robot or simulated plant."""
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from franka_compliant_control import _native
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control import joint_demo


@pytest.fixture
def config():
    return ControllerConfig.model_validate_json(Path("configs/controller.fr3.joint.json").read_text())


@pytest.fixture
def model():
    if not getattr(_native, "offline_kinematics_available", False):
        pytest.skip("offline SDK model requires hardware build >=0.18")
    hardware = json.loads(Path("configs/hardware.wrist-load.estimated.json").read_text())
    return _native.OfflineKinematics(Path("tests/fixtures/fr3-twin.urdf").read_text(),
                                     hardware["flange_to_tcp"])


def test_joint_start_script_forwards_configuration_and_arguments(tmp_path):
    root = tmp_path/"space in project"
    root.mkdir()
    shutil.copy2("start_joint_pos.sh", root/"start_joint_pos.sh")
    (root/"start.sh").write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    (root/"start.sh").chmod(0o755)
    result = subprocess.run(["bash", str(root/"start_joint_pos.sh"), "--port", "54325"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ["--port", "54325"]


def test_saved_model_and_jacobian_match_kinematics(model):
    q = np.array([0,-.4,0,-1.9,0,1.5,0])
    pose,jacobian = model.evaluate(q)
    assert pose[:3,3] == pytest.approx([.418133591554,0,.48463870918], abs=1e-9)
    for joint in range(7):
        plus,minus = q.copy(),q.copy();plus[joint] += 1e-6;minus[joint] -= 1e-6
        p,_ = model.evaluate(plus);m,_ = model.evaluate(minus)
        numerical = np.r_[(p[:3,3]-m[:3,3])/2e-6,
                           joint_demo.rotation_vector(p[:3,:3] @ m[:3,:3].T)/2e-6]
        assert numerical == pytest.approx(jacobian[:,joint], abs=1e-6)


def test_lower_endpoint_is_30cm_down_with_retained_orientation(model, config):
    home = np.array([0,-.4,0,-1.9,0,1.5,0])
    plan = joint_demo.plan_lower_target(model, home, config)
    lower,_ = model.evaluate(plan["lower_joints"])
    initial,_ = model.evaluate(home)
    assert lower[:3,3]-initial[:3,3] == pytest.approx([0,0,-.3], abs=2e-5)
    assert np.linalg.norm(joint_demo.rotation_vector(lower[:3,:3] @ initial[:3,:3].T)) < 1e-4
    assert not plan["collision_checked"]
    assert plan["planned_max_lateral_excursion_m"] > .02
    assert plan["move_time_s"] == 2 and plan["switch_every_s"] == 3
    assert plan["home_joints"] == home.tolist()


@pytest.mark.parametrize("kwargs", [{"drop_m": 0}, {"drop_m": -1}, {"drop_m": math.nan},
                                     {"move_time_s": 0}, {"move_time_s": 4, "switch_every_s": 3}])
def test_invalid_planning_requests(model, config, kwargs):
    with pytest.raises(ValueError):
        joint_demo.plan_lower_target(model, [0,-.4,0,-1.9,0,1.5,0], config, **kwargs)


def test_unreachable_or_too_fast_plan_is_rejected(model, config):
    with pytest.raises(ValueError, match="IK"):
        joint_demo.plan_lower_target(model, [0,-.4,0,-1.9,0,1.5,0], config, drop_m=1)
    with pytest.raises(ValueError, match="motion bound"):
        joint_demo.plan_lower_target(model, [0,-.4,0,-1.9,0,1.5,0], config, move_time_s=.1)


class FakeClient:
    def __init__(self, home):
        self.home=home;self.sent=[];self.stops=0;self.follow=False;self.fail=False
        self.current={"running":False,"phase":"disarmed","read_only":False,
                      "sample_valid":True,"sample_age_s":0,"positions":list(home),
                      "velocities":[0]*7,"reason":"none","error":""}
    def refresh_state(self):
        return {"state":dict(self.current)}
    def state(self):
        return dict(self.current)
    def set_target_joint_state(self, positions, velocities):
        if self.fail:
            raise RuntimeError("connection lost")
        started=not self.current["running"]
        self.sent.append((np.array(positions), np.array(velocities)))
        self.current.update(running=True,phase="active")
        if self.follow:
            self.current["positions"] = list(positions)
        return {"started_control":started,"seq":len(self.sent)}
    def stop(self, **_):
        self.stops+=1;self.current.update(running=False,phase="disarmed")
        return {"stopped":True,"state":dict(self.current)}


@pytest.fixture
def clock(monkeypatch):
    t=SimpleNamespace(now=0.,cancelled=False)
    class Event:
        def is_set(self):return t.cancelled
        def wait(self, seconds):t.now+=max(.02,seconds)
    monkeypatch.setattr(joint_demo,"time",SimpleNamespace(monotonic=lambda:t.now))
    return t,Event()


def plan():
    home=[0,-.4,0,-1.9,0,1.5,0]
    return {"home_joints":home,"lower_joints":[0,-.3,0,-2.3,0,2.0,0],"switch_every_s":3.,"move_time_s":2.}


def test_demo_switches_every_three_seconds_even_without_following(config, clock):
    t,event=clock;p=plan();client=FakeClient(p["home_joints"]);events=[]
    result=joint_demo.execute_joint_demo(client,p,config,cycles=2,stop_event=event,record=events.append)
    targets=[e for e in events if e["event"]=="target"]
    assert [e["name"] for e in targets]==["home","lower","home","lower"]
    assert [e["scheduled_elapsed_s"] for e in targets]==[0,3,6,9]
    assert result["elapsed_s"]==pytest.approx(12,abs=.021)
    assert client.stops==1
    assert client.sent[0][0]==pytest.approx(p["home_joints"])
    assert client.sent[-1][0]==pytest.approx(p["lower_joints"])


def test_demo_cancel_and_write_failure_stop_without_recovery(config, clock):
    t,event=clock;p=plan();client=FakeClient(p["home_joints"])
    def record(e):
        if e["event"]=="sample":t.cancelled=True
    result=joint_demo.execute_joint_demo(client,p,config,stop_event=event,record=record)
    assert result["reason"]=="interrupted" and client.stops==1
    t.cancelled=False;client=FakeClient(p["home_joints"]);client.fail=True
    with pytest.raises(RuntimeError,match="connection lost"):
        joint_demo.execute_joint_demo(client,p,config,stop_event=event)
    assert client.stops==1


def test_demo_refuses_changed_home_before_first_setpoint(config, clock):
    _,event=clock;p=plan();client=FakeClient(p["home_joints"])
    client.current["positions"][0]=.02
    with pytest.raises(RuntimeError,match="captured home"):
        joint_demo.execute_joint_demo(client,p,config,stop_event=event)
    assert not client.sent and client.stops==0


def test_cli_default_is_plan_only_and_sends_no_commands(model, config, tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location("demo_joint_pos","scripts/demo_joint_pos.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    home=[0,-.4,0,-1.9,0,1.5,0];pose,_=model.evaluate(home)
    class PlanningClient:
        sent=0
        def __init__(self,*_,**__):pass
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def health(self):return {"running":False,"phase":"disarmed","read_only":False}
        def get_config(self):return config.model_dump()
        def refresh_state(self):
            return {"state":{"sample_valid":True,"sample_age_s":0,"phase":"disarmed","running":False,
                             "positions":home,"position":pose[:3,3].tolist(),"quaternion_xyzw":[1,0,0,0]}}
        def set_target_joint_state(self,*_,**__):
            raise AssertionError("plan-only must not send any command")
    monkeypatch.setattr(module,"CompliantClient",PlanningClient)
    log=tmp_path/"plan.jsonl"
    assert module.main(["--output", str(log), "--urdf", "tests/fixtures/fr3-twin.urdf",
                        "--hardware-config", "configs/hardware.wrist-load.estimated.json"]) == 0
    events=[json.loads(l) for l in log.read_text().splitlines()]
    assert [e['event'] for e in events]==["before","plan"]
