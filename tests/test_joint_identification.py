"""Joint probes are small references, not native-gain changes or robot connections."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
from franka_compliant_control.joint_identification import segments, reference, run
from franka_compliant_control.simulation.world import SimulationWorld
from franka_compliant_control.simulation.native import NativeControllerSimulation
from franka_compliant_control.twin_model import PlantParameters
from twin_helpers import home_fixture


def test_waveform_is_single_joint_continuous_bounded_and_returns():
    schedule = segments(np.deg2rad(1.4))
    q = np.array([0.1, 0.2, 0.3, -1.5, 0.1, 1.0, 0.2])
    plan = {"joint_index": 4, "segments": schedule, "start": {"positions": q.tolist()}}
    values = np.array(
        [
            reference(plan, t)[0]
            for t in np.arange(0, schedule[-1]["end_s"] + 0.02, 0.02)
        ]
    )
    assert np.max(abs(values[:, 4] - q[4])) <= np.deg2rad(1.4) + 1e-12
    assert np.allclose(values[:, [0, 1, 2, 3, 5, 6]], q[[0, 1, 2, 3, 5, 6]])
    assert np.allclose(values[-1], q)
    assert np.max(abs(np.diff(values[:, 4]))) / 0.02 < 0.05
    for s in schedule[1:]:
        assert np.allclose(
            reference(plan, s["start_s"] - 1e-7)[0],
            reference(plan, s["start_s"] + 1e-7)[0],
        )


def test_shared_native_core_joint_commands_keep_lease_semantics():
    h = home_fixture()
    world = SimulationWorld(h, PlantParameters())
    sim = NativeControllerSimulation(world)
    q = np.array(sim.measurement().positions)
    sim.submit_joint(q, valid_for_s=0.01)
    assert sim.step_control().phase == "active"
    for _ in range(20):
        step = sim.step_control()
        if step.phase != "active":
            break
    assert step.phase != "active"
    with pytest.raises(RuntimeError):
        sim.submit_joint(q)
    sim.reset()
    with pytest.raises(ValueError):
        sim.submit_joint([0] * 6)
    with pytest.raises(ValueError):
        sim.submit_joint([100] * 7)
    with pytest.raises(ValueError):
        sim.submit_joint(q, delay_steps=-1)


def test_recording_client_hold_stops_and_preserves_gc(tmp_path):
    import gc
    from robot_fakes import Robot, Clock
    from franka_compliant_control.joint_identification import capture
    from franka_compliant_control.twin_experiment import digest, write_json_new

    h = home_fixture()
    clock = Clock()
    robot = Robot(h, clock)
    plan = capture(
        h,
        mode="cartesian",
        role="development",
        hold_s=5.0,
        client_factory=lambda **kw: robot,
    )
    robot.closed = False
    screen = tmp_path / "screen"
    write_json_new(screen / "summary.json", {"passed": True})
    write_json_new(screen / "manifest.json", {"plan_sha256": digest(plan)})
    before = gc.isenabled()
    result = run(
        h,
        plan,
        screen,
        tmp_path / "real",
        execute=True,
        clear_path=True,
        client_factory=lambda **kw: robot,
        clock=clock.now,
        sleep=clock.sleep,
    )
    assert result["completed"] and result["stop_confirmed"]
    assert not robot.running and robot.closed and gc.isenabled() == before
    assert result["samples"] >= 900


def test_cartesian_reference_and_recorded_override_share_submit_path():
    from franka_compliant_control.joint_identification import submit
    from franka_compliant_control.sysid_experiment import WaypointPolicy, Envelope

    h = home_fixture()
    p = np.array(h["home"]["position"])
    q = h["home"]["quaternion_xyzw"]
    j = h["home"]["joints"]
    policy = WaypointPolicy(
        tuple(p),
        tuple(q),
        tuple(j),
        tuple(p),
        tuple(q),
        (("Probe", tuple(p + [0.005, 0, 0.005]), 2.0, 1.0),),
    )
    plan = {
        "start": {"positions": j, "position": p.tolist(), "quaternion_xyzw": q},
        "mode": "cartesian",
        "joint_index": None,
        "pose_policy": policy.manifest(),
        "envelope": asdict(
            Envelope(
                tuple(p - 0.05),
                tuple(p + 0.05),
                tuple(h["controller_config"]["joint_lower"]),
                tuple(h["controller_config"]["joint_upper"]),
                tuple(q),
            )
        ),
    }
    ref, phase = reference(plan, 2.0)
    assert phase["position"] != p.tolist() and np.allclose(ref, j)

    class Recording:
        def submit_pose(self, position, quaternion_xyzw, **kwargs):
            self.position = position

    sim = Recording()
    submit(sim, plan, ref, phase=phase)
    assert sim.position == phase["position"]
    actual = {"position": (p + [0, 0.002, 0]).tolist(), "quaternion_xyzw": q}
    submit(sim, plan, ref, phase=actual)
    assert sim.position == actual["position"]
    with pytest.raises(RuntimeError):
        submit(
            sim,
            plan,
            ref,
            phase={"position": (p + [1, 0, 0]).tolist(), "quaternion_xyzw": q},
        )


def test_no_robot_client_without_acknowledgements(tmp_path):
    def forbidden(*a, **k):
        raise AssertionError("Should not connect")

    for kwargs in [{}, {"execute": True}, {"clear_path": True}]:
        with pytest.raises(ValueError):
            run({}, {}, tmp_path, tmp_path / "out", client_factory=forbidden, **kwargs)
