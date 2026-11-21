"""Progressive equilibrium control and live retargeting; simulation/offline only."""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.local_client import LocalControllerClient
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import create_app
from franka_compliant_control.trajectory import demo_pacing, segment_duration, segment_sample


def profile():
    # Preserve coverage of the original soft-core variant after the operator
    # selected the no-soft-zone saturated variant as the active profile.
    return ControllerConfig.model_validate_json(Path("tests/fixtures/controller/controller.fr3.osc.progressive-20260915-v1.json").read_text())


def applied(client, seq):
    end = time.monotonic()+.13
    while time.monotonic() < end:
        state = client.state()
        if state["applied_seq"] == seq:
            assert state["phase"] == "active"
            return state
        time.sleep(.001)
    pytest.fail(f"command {seq} was not observed before its expiry")


def test_current_profile_has_no_soft_zone_and_uses_half_franka_demo_pacing():
    old = profile()
    new = ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.progressive.json").read_text())
    assert new.osc_feedback_mode == "saturated"
    assert new.osc_translation_stiffness == [400]*3
    assert new.osc_rotation_stiffness == pytest.approx([8/math.radians(5)]*3)
    assert demo_pacing(new.osc_feedback_mode) == (1.5, 4.5, 2250)
    assert new.commanded_moment_limit == 8.0  # 2026-09-21: plateau at 5 degrees.
    assert old.commanded_moment_limit == 6.0
    for name in ("commanded_force_limit", "torque_limits", "torque_rate_limit",
                 "osc_translation_damping", "osc_rotation_damping", "collision_policy",
                 "contact_force_stop", "contact_moment_stop", "external_torque_limits"):
        assert getattr(new, name) == getattr(old, name)
    with pytest.raises(ValueError, match="positive physical stiffness"):
        ControllerConfig(**(new.model_dump() | {"osc_rotation_stiffness": [0]*3}))


def test_profile_is_opt_in_and_retains_robot_envelope():
    old = ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.json").read_text())
    new = profile()
    assert old.osc_feedback_mode == "linear" and old.effort_limit_policy == "franka"
    assert new.osc_feedback_mode == "progressive" and new.effort_limit_policy == "task_bounded"
    assert new.posture_stiffness == 0  # No hidden static joint-posture spring.
    assert new.progressive_position_soft_radius == .01
    assert new.progressive_position_saturation_radius == .1
    assert new.progressive_rotation_soft_radius == pytest.approx(math.radians(3))
    assert new.progressive_rotation_saturation_radius == pytest.approx(math.radians(20))
    for key in ("torque_limits", "torque_rate_limit", "joint_lower", "joint_upper", "joint_limit_margin",
                "measured_joint_velocity_limits", "measured_translation_velocity_limit",
                "measured_rotation_velocity_limit", "collision_policy", "contact_force_stop",
                "contact_moment_stop", "external_torque_limits"):
        assert getattr(new, key) == getattr(old, key)


@pytest.mark.parametrize("changes", [
    {"osc_feedback_mode": "unknown"}, {"osc_feedback_mode": "linear"},
    {"effort_limit_policy": "franka"}, {"compliance_mode": "continuous_impedance"},
    {"progressive_position_soft_radius": 0}, {"progressive_position_soft_radius": .1},
    {"progressive_position_saturation_radius": .009},
    {"progressive_rotation_saturation_radius": math.pi+.01},
    {"progressive_rotation_soft_radius": math.radians(20)},
    {"progressive_translation_soft_stiffness": -1},
    {"progressive_translation_soft_stiffness": 200},  # Must remain softer than shoulder average.
    {"progressive_rotation_soft_stiffness": 100},
    {"progressive_position_saturation_radius": float("inf")},
    {"osc_translation_damping": [0, 40, 40]}, {"osc_rotation_damping": [8, 0, 8]},
    {"commanded_force_limit": 0}, {"commanded_moment_limit": float("nan")},
    {"torque_rate_limit": 100},
])
def test_invalid_progressive_settings(changes):
    with pytest.raises(ValueError):
        ControllerConfig(**(profile().model_dump() | changes))


@pytest.mark.parametrize("transport", ["local", "http"])
def test_live_target_changes_redirect_the_spring_through_both_transports(transport):
    with ExitStack() as stack:
        if transport == "local":
            robot = stack.enter_context(LocalControllerClient(profile(), backend_kind="sim"))
            inject = robot.controller.backend.inject_wrench
        else:
            http = stack.enter_context(TestClient(create_app(config=profile(), enable_sim_injection=True)))
            robot = CompliantClient("http://testserver")
            robot.http.close()
            robot.http = http
            def inject(wrench):
                assert http.post("/sim/wrench", json={"wrench": wrench}).status_code == 200
        if transport == "local":
            initial = robot.arm()["state"]
        else:
            initial = robot.state()
        p, q = initial["position"], initial["quaternion_xyzw"]
        try:
            inject([20,0,0,0,0,0])  # Near force balance while commanding into a blocked direction.
            first = robot.send_pose([p[0]+1000,p[1],p[2]], q)
            state = applied(robot, first["seq"])
            assert state["reference_policy"] == "direct_equilibrium"
            assert state["spring_wrench"][:3] == pytest.approx([20,0,0], abs=.01)
            assert state["progressive_force_saturated"] and not state["yielding"]
            second = robot.send_pose([p[0]-1000,p[1],p[2]], q)
            state = applied(robot, second["seq"])
            assert state["reference_position"] == [p[0]-1000,p[1],p[2]]
            assert state["spring_wrench"][0] == pytest.approx(-20, abs=.001)
            assert state["commanded_task_wrench"][0] < 0
            inject([0]*6)
            current = robot.state()["position"]
            small = [current[0],current[1]+.005,current[2]]
            third = robot.send_pose(small, q)
            state = applied(robot, third["seq"])
            assert state["reference_position"] == small
            assert math.hypot(*state["spring_wrench"][:3]) < .6
            assert not state["progressive_force_saturated"]
            theta = .5
            fourth = robot.send_pose(small, [0,0,math.sin(theta/2),math.cos(theta/2)])
            state = applied(robot, fourth["seq"])
            assert state["progressive_moment_saturated"]
            assert math.hypot(*state["commanded_task_wrench"][:3]) <= 20.000001
            assert math.hypot(*state["commanded_task_wrench"][3:]) <= 6.000001
        finally:
            inject([0]*6)
            assert robot.stop(wait_s=2.5)["stopped"]


def test_progressive_ff_is_transmitted_and_capped():
    with LocalControllerClient(profile(), backend_kind="sim") as robot:
        initial = robot.arm()["state"]
        result = robot.send_pose(initial["position"], initial["quaternion_xyzw"],
            twist=[3,0,0,0,0,0], acceleration=[9,0,0,0,0,17])
        state = applied(robot, result["seq"])
        assert state["reference_twist"] == [3,0,0,0,0,0]
        assert state["reference_acceleration"] == [9,0,0,0,0,17]
        assert math.hypot(*state["commanded_task_wrench"][:3]) <= 20.000001
        assert math.hypot(*state["commanded_task_wrench"][3:]) <= 6.000001
        assert robot.stop(wait_s=2.5)["stopped"]


@pytest.mark.parametrize("kw", [
    {"twist": [3.01,0,0,0,0,0]}, {"twist": [2,2,2,0,0,0]},
    {"acceleration": [9.01,0,0,0,0,0]}, {"acceleration": [0,0,0,0,0,17.01]},
    {"acceleration": [float("nan")]*6}, {"acceleration": [0]*5},
])
def test_invalid_feedforward_is_rejected(kw):
    with LocalControllerClient(profile(), backend_kind="sim") as robot:
        initial = robot.arm()["state"]
        with pytest.raises(ValueError):
            robot.send_pose(initial["position"], initial["quaternion_xyzw"], **kw)
        assert robot.stop(wait_s=2.5)["stopped"]


def test_linear_profile_does_not_silently_ignore_nonzero_acceleration():
    config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.json").read_text())
    with LocalControllerClient(config, backend_kind="sim") as robot:
        initial = robot.arm()["state"]
        with pytest.raises(ValueError, match="only by bounded-equilibrium"):
            robot.send_pose(initial["position"], initial["quaternion_xyzw"], acceleration=[.1,0,0,0,0,0])
        assert robot.stop(wait_s=2.5)["stopped"]


def test_quintic_pacing_respects_resultant_derivative_budgets():
    start, end = [0,0,0], [.1,.2,-.3]
    v, a, j = .1,.3,2
    duration = segment_duration(start, end, v, a, j)
    times = np.linspace(0, duration, 2001)
    values = [segment_sample(start, end, t, duration) for t in times]
    velocity = np.array([x[1][:3] for x in values])
    acceleration = np.array([x[2][:3] for x in values])
    assert np.linalg.norm(velocity, axis=1).max() <= v+1e-12
    assert np.linalg.norm(acceleration, axis=1).max() <= a+1e-12
    assert np.linalg.norm(np.diff(acceleration, axis=0)/(times[1]-times[0]), axis=1).max() <= j+1e-8
    assert values[0] == (start, [0]*6, [0]*6)
    assert values[-1][0] == pytest.approx(end)
    assert values[-1][1:] == ([0]*6,[0]*6)


@pytest.mark.parametrize("hold", [False, True])
def test_progressive_cli_runs_only_in_simulation_by_default(tmp_path, hold):
    log = tmp_path/"progressive.jsonl"
    options = ["--hold", "--duration", ".25"] if hold else ["--offset", ".02", "--cycles", "1", "--switch-every", "0"]
    result = subprocess.run([sys.executable, "scripts/osc_track.py", "--profile", "progressive",
                             "--output", str(log), *options], capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, (result.stdout, result.stderr)
    events = [json.loads(l) for l in log.read_text().splitlines()]
    assert events[0]["backend"] == "sim"
    assert events[0]["controller_config"]["osc_feedback_mode"] == "saturated"
    assert events[0]["controller_config"]["translation_velocity"] == 1.5
    assert events[0]["controller_config"]["translation_acceleration"] == 4.5
    assert events[0]["controller_config"]["translation_jerk"] == 2250
    assert next(e for e in events if e["event"] == "stop")["result"]["stopped"]
    if not hold:
        before = next(e for e in events if e["event"] == "before")
        assert before["position_tolerance_m"] == .005
        assert before["rotation_tolerance_rad"] == pytest.approx(math.radians(2))
        targets = [e for e in events if e["event"] == "target"]
        assert len(targets) == 2 and all(t["paced_duration_s"] > 0 for t in targets)
        assert len([e for e in events if e["event"] == "reached"]) == 2


def test_progressive_cli_interrupt_stops_and_closes(tmp_path):
    log = tmp_path/"progressive.jsonl"
    proc = subprocess.Popen([sys.executable, "scripts/osc_track.py", "--profile", "progressive",
                             "--hold", "--duration", "0", "--output", str(log)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            assert proc.poll() is None, proc.communicate()
            if log.exists() and '"event": "sample"' in log.read_text():
                break
            time.sleep(.02)
        else:
            pytest.fail("progressive hold did not start")
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=6)
        assert proc.returncode == 0, (stdout, stderr)
        assert "Stopped and disarmed." in stdout
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=3)
