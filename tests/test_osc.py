"""OSC schemas, transport, experiment state machine and simulation-only CLI."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
import time

import pytest

from franka_compliant_control import _native
from franka_compliant_control.local_client import LocalControllerClient
from franka_compliant_control.models import ControllerConfig, JointTarget
from franka_compliant_control import tracking


def profile():
    return ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.json").read_text())


def test_full_envelope_is_distinct_from_experiment_speed():
    config = profile()
    assert config.compliance_mode == "operational_space"
    assert config.effort_limit_policy == "franka" and config.collision_policy == "robot_configured"
    assert config.translation_velocity == config.measured_translation_velocity_limit == 3
    assert config.rotation_velocity == config.measured_rotation_velocity_limit == 2.5
    assert config.torque_limits == [87]*4+[12]*3 and config.torque_rate_limit == 1000
    assert config.joint_limit_margin == 0 and config.joint_lower[0] == -2.9007
    paced = ControllerConfig(**(config.model_dump() | {"translation_velocity": .1}))
    assert paced.measured_translation_velocity_limit == 3
    assert ControllerConfig().collision_policy == "configured"  # Old behavior unchanged.


def test_tuning_adds_physical_feedback_without_changing_operating_limits():
    old = ControllerConfig.model_validate_json(Path("tests/fixtures/controller/controller.fr3.osc-20260915-v1.json").read_text())
    new = ControllerConfig.model_validate_json(Path("tests/fixtures/controller/controller.fr3.osc-20260915-v2.json").read_text())
    fields = {"osc_translation_stiffness", "osc_translation_damping",
              "osc_rotation_stiffness", "osc_rotation_damping"}
    assert old.model_dump(exclude=fields) == new.model_dump(exclude=fields)
    assert all(getattr(old, name) == [0]*3 for name in fields)
    assert new.osc_translation_stiffness == [400]*3
    assert new.osc_translation_damping == [40]*3
    assert new.osc_rotation_stiffness == [120]*3
    assert new.osc_rotation_damping == [8]*3


@pytest.mark.parametrize("changes", [
    {"osc_translation_kp": [0, 25, 25]}, {"osc_rotation_kd": [10, -1, 10]},
    {"osc_regularization": float("nan")}, {"torque_rate_limit": 1001},
    {"torque_limits": [88]*4+[12]*3}, {"torque_rate_limit": 100},
    {"compliance_mode": "contact_yielding"}, {"bounded_cartesian_spring": True},
    {"motion_limit_policy": "configured"}, {"collision_policy": "off"},
    {"translation_velocity": 3.01}, {"translation_acceleration": 9.01},
    {"translation_jerk": 4501},
    {"osc_rotation_stiffness": [-1, 0, 0]},
    {"osc_translation_damping": [0, float("nan"), 0]},
])
def test_invalid_osc_config_rejected(changes):
    with pytest.raises(ValueError):
        ControllerConfig(**(profile().model_dump() | changes))


@pytest.mark.parametrize("policy", ["robot", "robot_configured"])
def test_local_transport_mode_and_robot_policy_telemetry(policy):
    config = ControllerConfig(**(profile().model_dump() | {"collision_policy": policy}))
    with LocalControllerClient(config, backend_kind="sim") as client:
        initial = client.arm()
        client.send_pose(initial["state"]["position"], initial["state"]["quaternion_xyzw"],
                         twist=[0]*6, posture=initial["state"]["positions"])
        with pytest.raises(ValueError, match="Cartesian targets only"):
            client.controller.submit(JointTarget(seq=2,
                deadline_s=_native.monotonic_time()+.1, positions=initial["state"]["positions"]))
        state = client.state()
        assert state["dynamics_valid"]
        assert state["contact_load_ratio"] == (None if policy == "robot" else 0)
        assert state["peak_contact_load_ratio"] == (None if policy == "robot" else 0)
        assert not state["software_contact_trips_enabled"]
        assert state["robot_collision_settings"]["status"] == ("not_applied" if policy == "robot" else "simulated")
        assert len(state["reference_acceleration"]) == 6
        assert client.stop(wait_s=2.5)["stopped"]


class FakeClient:
    def __init__(self):
        self.config = profile().model_dump()
        self.sent = []
        self.stops = self.arms = 0
        self.current = dict(backend="robot", read_only=False, running=False, phase="disarmed",
            position=[.4, .1, .5], quaternion_xyzw=[0,0,0,1], reason="none", error="",
            sample_valid=True, sample_age_s=0, velocities=[0]*7, reference_twist=[0]*6,
            reference_position=[.4, .1, .5], applied_seq=0)
        self.follow = True
        self.arm_error = self.send_error = None

    def state(self):
        return dict(self.current)

    def get_config(self):
        return self.config

    def arm(self):
        self.arms += 1
        if self.arm_error:
            raise RuntimeError(self.arm_error)
        self.current.update(position=[.5, .2, .6], phase="armed", running=True)
        return {"state": self.state()}

    def send_pose(self, position, orientation):
        if self.send_error:
            raise RuntimeError(self.send_error)
        self.sent.append((position, orientation))
        if self.follow:
            self.current.update(position=position, reference_position=position)
        self.current.update(phase="active", applied_seq=len(self.sent))
        return {"seq": len(self.sent)}

    def stop(self, **_):
        self.stops += 1
        return {"stopped": True, "state": self.state()}


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0)
    class Event:
        def is_set(self):
            return False
        def wait(self, seconds):
            clock.now += max(.02, seconds)
    monkeypatch.setattr(tracking, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return Event()


def test_post_arm_home_and_both_fixed_targets(clock):
    client = FakeClient()
    events = []
    result = tracking.track_two_targets(client, cycles=2, stop_event=clock, record=events.append)
    assert result["stopped"] and result["completed_legs"] == 4
    assert result["home_position"] == (.5, .2, .6)
    assert result["target_positions"] == [(.5, 0, .6), (.5, .4, .6)]
    assert [e["position"] for e in events if e["event"] == "target"] == result["target_positions"]*2
    assert all(q == (0,0,0,1) for _, q in client.sent)
    assert client.stops == client.arms == 1


@pytest.mark.parametrize("failure", ["stale", "blocked", "send", "arm", "record"])
def test_failures_stop_without_retry(clock, failure):
    client = FakeClient()
    if failure == "stale":
        client.current["sample_age_s"] = .1
    if failure == "blocked":
        client.follow = False
    if failure == "send":
        client.send_error = "send failed"
    if failure == "arm":
        client.arm_error = "arm acknowledgement lost"
    def record(event):
        if failure == "record" and event["event"] == "sample":
            raise RuntimeError("disk full")
    with pytest.raises(RuntimeError):
        tracking.track_two_targets(client, stop_event=clock, leg_timeout_s=1, record=record)
    assert client.stops == client.arms == 1


@pytest.mark.parametrize("follow", [False, True])
def test_timed_switching_does_not_wait_for_or_advance_early_on_reaching(clock, follow):
    client = FakeClient()
    client.follow = follow
    events = []
    result = tracking.track_two_targets(client, cycles=1, switch_every_s=5,
                                       stop_event=clock, record=events.append)
    targets = [e for e in events if e["event"] == "target"]
    assert [e["scheduled_elapsed_s"] for e in targets] == [0, 5]
    assert all(abs(e["elapsed_s"]-e["scheduled_elapsed_s"]) <= .021 for e in targets)
    assert result["completed_legs"] == 2 and result["elapsed_s"] == pytest.approx(10, abs=.021)
    assert result["reached_legs"] == (2 if follow else 0)
    intervals = [e for e in events if e["event"] == "target_interval_complete"]
    assert len(intervals) == 2 and all(e["reached_during_interval"] == follow for e in intervals)
    assert client.stops == client.arms == 1


def test_cancel_before_arm_is_nonintrusive():
    client = FakeClient()
    event = threading.Event();event.set()
    assert not tracking.track_two_targets(client, stop_event=event)["armed"]
    assert client.stops == client.arms == 0


@pytest.mark.parametrize("offset", [-.1, 0, float("inf"), float("nan")])
def test_invalid_offsets(offset):
    with pytest.raises(ValueError):
        tracking.targets_about_home([0,0,0], offset)


def test_hardware_requires_explicit_execute():
    result = subprocess.run([sys.executable, "scripts/osc_track.py", "--backend", "robot"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and "Hardware requires --execute" in result.stderr


@pytest.mark.parametrize("hold", [False, True])
def test_actual_foreground_cli_in_simulation(tmp_path, hold):
    log = tmp_path/"osc.jsonl"
    options = ["--hold", "--duration", ".25"] if hold else ["--offset", ".02", "--cycles", "1", "--switch-every", "0"]
    result = subprocess.run([sys.executable, "scripts/osc_track.py", "--output", str(log), *options],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, (result.stdout, result.stderr)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert events[0]["backend"] == "sim"
    assert events[0]["controller_config"]["measured_translation_velocity_limit"] == 3
    assert next(e for e in events if e["event"] == "stop")["result"]["stopped"]
    if not hold:
        reached = [e for e in events if e["event"] == "reached"]
        assert len(reached) == 2 and reached[0]["position"][1] == -.02 and reached[1]["position"][1] == .02
    assert "no service left running" in result.stdout


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_osc_signals_close_without_child_process(tmp_path, sig):
    log = tmp_path/"osc.jsonl"
    process = subprocess.Popen([sys.executable, "scripts/osc_track.py", "--cycles", "0", "--output", str(log)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        end = time.monotonic()+8
        while time.monotonic() < end:
            assert process.poll() is None, process.communicate()
            if log.exists() and '"event": "sample"' in log.read_text():
                break
            time.sleep(.02)
        else:
            pytest.fail("OSC did not start")
        assert not Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().strip()
        process.send_signal(sig)
        stdout, stderr = process.communicate(timeout=6)
        assert process.returncode == 0, (stdout, stderr)
        assert "Stopped and disarmed." in stdout
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)
