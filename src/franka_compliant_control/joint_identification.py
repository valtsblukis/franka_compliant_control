"""Bounded joint/quiet-hold acquisition and native-controller replay.

No native rebuild, gains change, recovery, automatic retries or online fitting.
100/200 Hz HTTP polling is best-effort sampling, not 1 kHz telemetry. Reference
publication remains 50 Hz. Other joints retain references, not mechanical locks.
Optional recorded WaypointPolicy poses use the same acquisition/replay path.
"""

from copy import deepcopy
from functools import lru_cache
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import json
import signal
import time

import numpy as np

from .client import CompliantClient
from .models import ControllerConfig
from .simulation.world import SimulationWorld
from .simulation.native import NativeControllerSimulation
from .simulation.sensors import FrankaWrenchSurrogate
from .simulation.metrics import MotionMetrics
from .sysid_experiment import Envelope
from .twin_experiment import load_home, digest, write_json_new
from .twin_model import PlantParameters, rotation_distance
from .wrench_model import canonical_franka_wrench


def segments(amplitude, *, validation=False):
    """Smooth bidirectional probes at two speeds, with explicit dwells."""
    sign = -1 if validation else 1
    slow, fast = (3.0, 1.5) if validation else (4.0, 2.0)
    result = []
    now = 0.0
    value = 0.0
    for duration, target in [
        (5.0, 0.0),
        (slow, sign * amplitude),
        (2.0, sign * amplitude),
        (slow, 0.0),
        (2.0, 0.0),
        (slow, -sign * amplitude),
        (2.0, -sign * amplitude),
        (slow, 0.0),
        (3.0, 0.0),
        (fast, sign * amplitude),
        (2.0, sign * amplitude),
        (fast, 0.0),
        (2.0, 0.0),
        (fast, -sign * amplitude),
        (2.0, -sign * amplitude),
        (fast, 0.0),
        (5.0, 0.0),
    ]:
        result.append(
            {
                "start_s": now,
                "end_s": now + duration,
                "from_rad": value,
                "to_rad": target,
                "policy_phase": "dwell" if target == value else "move",
                "leg": len(result),
            }
        )
        now += duration
        value = target
    return result


@lru_cache(maxsize=8)
def _waypoint_policy(encoded):
    from .sysid_experiment import WaypointPolicy

    values = json.loads(encoded)
    return WaypointPolicy(
        **{k: v for k, v in values.items() if k in WaypointPolicy.__dataclass_fields__}
    )


def reference(plan, t):
    q = np.asarray(plan["start"]["positions"]).copy()
    if "pose_policy" in plan:
        return q, _waypoint_policy(
            json.dumps(plan["pose_policy"], sort_keys=True)
        ).sample(t)
    phase = {"policy_phase": "dwell", "leg": 0}
    if plan["joint_index"] is not None:
        seg = next(
            (s for s in plan["segments"] if t < s["end_s"]), plan["segments"][-1]
        )
        u = float(np.clip((t - seg["start_s"]) / (seg["end_s"] - seg["start_s"]), 0, 1))
        blend = u**3 * (10 - 15 * u + 6 * u * u)
        q[plan["joint_index"]] += seg["from_rad"] + blend * (
            seg["to_rad"] - seg["from_rad"]
        )
        phase = deepcopy(seg)
    return q, phase


def capture(
    home,
    *,
    joint=None,
    mode="joint",
    role="training",
    amplitude_deg=0.5,
    hold_s=30.0,
    poll_hz=200,
    record_native=False,
    client_factory=CompliantClient,
):
    if (
        joint not in (None, 1, 2, 3, 4, 5, 6, 7)
        or mode not in ("joint", "cartesian")
        or role not in ("training", "validation", "development")
    ):
        raise ValueError("Invalid probe selection")
    if (
        not np.isfinite(amplitude_deg)
        or not 0 < amplitude_deg <= 1.5
        or not np.isfinite(hold_s)
        or not 5 <= hold_s <= 60
    ):
        raise ValueError("Unreviewed amplitude/duration")
    if poll_hz not in (50, 100, 200):
        raise ValueError("Use reviewed 50, 100 or 200 Hz polling")
    if joint is not None and mode != "joint":
        raise ValueError("Individual probes use joint mode")
    client = client_factory(timeout_s=4)
    client.http.headers["Connection"] = "close"
    try:
        h = client.health()
        if h["running"] or h["phase"] != "disarmed" or h["automatic_recovery_enabled"]:
            raise RuntimeError("Need disarmed robot with recovery OFF")
        if (
            ControllerConfig.model_validate(client.get_config()).model_dump()
            != home["controller_config"]
            or digest(client.get_urdf()["urdf"]) != home["sha256"]["urdf"]
        ):
            raise RuntimeError("Model/controller changed")
        client.refresh_state()
        s = client.state()
        if (
            s["robot_mode"] != "Idle"
            or s["robot_has_errors"]
            or not s["sample_valid"]
            or not s["kinematics_valid"]
            or s["sample_age_s"] > 0.02
            or max(abs(x) for x in s["velocities"]) > 0.02
        ):
            raise RuntimeError("Need fresh stationary Idle state")
    finally:
        client.close()
    c = home["controller_config"]
    p = np.array(home["home"]["position"])
    lo = np.array(c["joint_lower"]) + 0.15
    hi = np.array(c["joint_upper"]) - 0.15
    b = np.deg2rad(18)
    lo[0] = max(lo[0], -b, home["home"]["joints"][0] - b)
    hi[0] = min(hi[0], b, home["home"]["joints"][0] + b)
    envelope = Envelope(
        tuple(p + [-0.025, -0.10, -0.012]),
        tuple(p + [0.04, 0.10, 0.04]),
        tuple(lo),
        tuple(hi),
        tuple(home["home"]["quaternion_xyzw"]),
        translation_speed=0.03,
        rotation_speed=0.08,
        joint_speed=0.06,
    )
    envelope.check(
        s["positions"], s["velocities"], s["position"], s["quaternion_xyzw"], s["twist"]
    )
    seg = (
        segments(np.deg2rad(amplitude_deg), validation=role == "validation")
        if joint
        else []
    )
    plan = {
        "schema": "fcc-joint-identification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "home_sha256": home["sha256"],
        "controller_config": c,
        "start": s,
        "mode": mode,
        "joint_index": joint - 1 if joint else None,
        "segments": seg,
        "duration_s": seg[-1]["end_s"] if seg else hold_s,
        "envelope": asdict(envelope),
        "reference_rate_hz": 50,
        "requested_poll_rate_hz": poll_hz,
        "native_recording_capacity": (
            int(np.ceil(((seg[-1]["end_s"] if seg else hold_s) + 5) * 1000))
            if record_native
            else None
        ),
        "native_rate_hz": 1000,
        "automatic_recovery": False,
        "source_sha256": digest(Path(__file__).read_text()),
        "source_native_adapter_sha256": digest(
            Path(__file__).with_name("simulation").joinpath("native.py").read_text()
        ),
        "purpose": "Effective resistance / observation diagnostics; no independent contact force calibration",
        "home_recaptured": False,
        "controller_gains_changed": False,
    }
    for t in np.arange(0, plan["duration_s"] + 0.001, 0.02):
        q, _ = reference(plan, t)
        if np.any(q < lo) or np.any(q > hi):
            raise ValueError("Joint reference outside envelope")
    return plan


def build_sim(home, parameters, observer, plan, initial=None):
    world = SimulationWorld(home, parameters)
    sim = NativeControllerSimulation(world, sensor=FrankaWrenchSurrogate(observer))
    state = initial or plan["start"]
    sim.reset(
        q=state["positions"],
        previous_torque=state.get("last_desired_joint_torques") if initial else None,
    )
    return sim


def snapshot(sim, phase):
    m = sim.measurement()
    return {
        "time_s": sim.now,
        "robot_time_s": sim.now,
        "positions": list(m.positions),
        "velocities": list(m.velocities),
        "position": list(m.position),
        "quaternion_xyzw": list(m.quaternion_xyzw),
        "twist": list(m.twist),
        "external_wrench_tcp_base": list(m.wrench_tcp_base),
        "last_desired_joint_torques": sim.previous.tolist(),
        "phase": sim.phase,
        "sample_valid": True,
        "sample_age_s": 0.0,
        "policy": phase,
    }


def pose_command(plan, phase=None):
    value = phase if phase is not None and "position" in phase else plan["start"]
    p = np.asarray(value["position"])
    q = np.asarray(value["quaternion_xyzw"])
    if np.any(p < plan["envelope"]["tcp_lower"]) or np.any(
        p > plan["envelope"]["tcp_upper"]
    ):
        raise RuntimeError("Pose reference outside planned workspace")
    if (
        rotation_distance(q, plan["envelope"]["orientation_xyzw"])
        > plan["envelope"]["orientation_error_rad"]
    ):
        raise RuntimeError("Pose reference outside orientation envelope")
    return {"position": p.tolist(), "quaternion_xyzw": q.tolist()}


def submit(sim, plan, q, valid_for_s=0.2, phase=None):
    if plan["mode"] == "joint":
        sim.submit_joint(q, valid_for_s=valid_for_s)
    else:
        pose = pose_command(plan, phase)
        sim.submit_pose(
            pose["position"], pose["quaternion_xyzw"], valid_for_s=valid_for_s
        )


def simulate(
    home, parameters, observer, plan, output, *, events=None, initial=None, end_s=None
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sim = build_sim(home, parameters, observer, plan, initial)
    env = Envelope(**plan["envelope"])
    metrics = MotionMetrics(sim.measurement())
    reason = None
    index = 0
    write_json_new(
        output / "manifest.json",
        {
            "schema": "fcc-joint-probe-screen-v1",
            "plan_sha256": digest(plan),
            "plan": plan,
            "simulation": sim.manifest(),
            "plant": parameters.manifest(),
            "observer": observer,
            "accepted_command_replay": events is not None,
            "hardware_authorized": False,
        },
    )
    duration = plan["duration_s"] if end_s is None else end_s
    with (output / "trace.jsonl").open("x") as log:
        for tick in range(int(np.ceil(duration * 1000)) + 1):
            q, phase = reference(plan, sim.now)
            try:
                m = sim.measurement()
                env.check(
                    m.positions,
                    m.velocities,
                    m.position,
                    m.quaternion_xyzw,
                    m.twist,
                    m.minimum_singular_value,
                    m.wrench_tcp_base,
                )
                metrics.update(m, False)
                if tick % 5 == 0:
                    log.write(json.dumps(snapshot(sim, phase), allow_nan=False) + "\n")
                if sim.now >= duration - 1e-9:
                    break
                if events is None:
                    if tick % 20 == 0:
                        submit(sim, plan, q, phase=phase)
                else:
                    while (
                        index < len(events)
                        and events[index]["time_s"] <= sim.now + 1e-10
                    ):
                        e = events[index]
                        remaining = e["deadline_s"] - sim.now
                        if remaining <= 0:
                            raise RuntimeError("Expired replay event")
                        submit(sim, plan, e["joints"], remaining, phase=e.get("pose"))
                        index += 1
                step = sim.step_control()
                if step.phase != "active":
                    raise RuntimeError(step.reason)
            except (ValueError, RuntimeError) as error:
                reason = str(error)
                break
    motion = metrics.report(home["controller_config"])
    passed = reason is None and max(motion["all_motion_ratios"]) <= 1
    result = {
        "passed": bool(passed),
        "reason": reason,
        "last_time_s": sim.now,
        "motion_metrics": motion,
        "final_positions": list(sim.measurement().positions),
        "all_scheduled_samples_only": True,
        "reference_tracking_accuracy_not_a_success_requirement": True,
        "hardware_ready": False,
    }
    write_json_new(output / "summary.json", result)
    return result


def run(
    home,
    plan,
    screen_path,
    output,
    *,
    execute=False,
    clear_path=False,
    client_factory=CompliantClient,
    clock=time.monotonic,
    sleep=time.sleep,
):
    if not execute or not clear_path:
        raise ValueError("Explicit execution and clear-path acknowledgements required")
    screen_path = Path(screen_path)
    if not json.loads((screen_path / "summary.json").read_text())[
        "passed"
    ] or json.loads((screen_path / "manifest.json").read_text())[
        "plan_sha256"
    ] != digest(
        plan
    ):
        raise ValueError("Matching successful screen required")
    if (
        plan["source_sha256"] != digest(Path(__file__).read_text())
        or plan["source_native_adapter_sha256"]
        != digest(
            Path(__file__).with_name("simulation").joinpath("native.py").read_text()
        )
        or plan["home_sha256"] != home["sha256"]
        or plan["controller_config"] != home["controller_config"]
    ):
        raise ValueError("Changed experiment/home/controller")
    if "pose_policy" in plan:
        if (
            plan["mode"] != "cartesian"
            or plan["joint_index"] is not None
            or plan.get("pose_policy_source_sha256")
            != digest(Path(__file__).with_name("sysid_experiment.py").read_text())
        ):
            raise ValueError("Changed or inconsistent Cartesian excitation policy")
    if plan["requested_poll_rate_hz"] not in (50, 100, 200):
        raise ValueError("Unsupported polling rate")
    poll_period = 1.0 / plan["requested_poll_rate_hz"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "manifest.json", plan)
    from .simulation.sdk import offline_model

    sdk = offline_model(home)
    env = Envelope(**plan["envelope"])
    client = client_factory(timeout_s=4)
    client.http.headers["Connection"] = "close"
    attempted = False
    stopped = None
    last = None
    enabled = gc.isenabled()
    from .clock_bounds import ClockOffsetBounds

    clock_bounds = ClockOffsetBounds()
    coherent_clocks = False
    count = 0
    try:
        if enabled:
            gc.collect()
            gc.disable()
        h = client.health()
        if h["running"] or h["phase"] != "disarmed" or h["automatic_recovery_enabled"]:
            raise RuntimeError("Need inactive robot, recovery OFF")
        if (
            ControllerConfig.model_validate(client.get_config()).model_dump()
            != home["controller_config"]
            or digest(client.get_urdf()["urdf"]) != home["sha256"]["urdf"]
        ):
            raise RuntimeError("Changed controller/model")
        client.refresh_state()
        s = client.state()
        start = plan["start"]
        if (
            s["robot_mode"] != "Idle"
            or s["robot_has_errors"]
            or not s["sample_valid"]
            or not s["kinematics_valid"]
            or s["sample_age_s"] > 0.02
            or max(abs(x) for x in s["velocities"]) > 0.02
        ):
            raise RuntimeError("Invalid initial state")
        if (
            max(abs(np.array(s["positions"]) - start["positions"])) > 0.001
            or np.linalg.norm(np.array(s["position"]) - start["position"]) > 0.0002
            or rotation_distance(s["quaternion_xyzw"], start["quaternion_xyzw"])
            > np.deg2rad(0.05)
        ):
            raise RuntimeError("Fresh start changed; replan/rescreen")
        env.check(
            s["positions"],
            s["velocities"],
            s["position"],
            s["quaternion_xyzw"],
            s["twist"],
        )

        def send(q, lease, phase=None):
            if plan["mode"] == "joint":
                return client.set_target_joint_state(q, valid_for_s=lease)
            pose = pose_command(plan, phase)
            return client.set_target_ee_pose(
                pose["position"], pose["quaternion_xyzw"], valid_for_s=lease
            )

        if plan.get("native_recording_capacity"):
            if not client.telemetry_status().get('native_experiment_speed_guard_available'):
                raise RuntimeError('Native-rate experiment speed guard upgrade required before another recorded motion')
            capture_status = client.prepare_telemetry(plan["native_recording_capacity"],speed_limits=[env.joint_speed,env.translation_speed,env.rotation_speed])
            write_json_new(output / "native-prepared.json", capture_status)
        for _ in range(5):
            begin = clock()
            clock_state = client.state()
            end = clock()
            if (
                "sample_timestamp_monotonic_s" not in clock_state
                or "state_timestamp_monotonic_s" not in clock_state
            ):
                break
            clock_bounds.observe(begin, end, clock_state["state_timestamp_monotonic_s"])
            coherent_clocks = True
        write_json_new(
            output / "observation-clock.json",
            {
                "coherent_native_timestamps": coherent_clocks,
                **clock_bounds.manifest(),
                "age_limit_s": 0.010,
            },
        )
        attempted = True
        first = send(start["positions"], 1.0)
        client.timeout_s = 0.2
        deadline = clock() + 0.5
        while True:
            s = client.state()
            if s["running"] and s["phase"] == "active":
                break
            if not s["running"] or s["phase"] != "armed" or clock() > deadline:
                raise RuntimeError("Startup failed")
            sleep(0.005)
        origin = clock()
        server_origin = s["server_monotonic_s"]
        epoch = first["epoch"]
        own = first["command_seq"]
        next_poll = origin
        command_tick = 1
        write_json_new(
            output / "startup.json",
            {
                "state": s,
                "response": first,
                "policy_origin_server_monotonic_s": server_origin,
            },
        )
        with (
            (output / "trace.jsonl").open("x") as log,
            (output / "actions.jsonl").open("x") as actions,
        ):
            actions.write(
                json.dumps(
                    {"time_s": 0.0, "joints": start["positions"], "response": first}
                )
                + "\n"
            )
            while True:
                sleep(max(0, next_poll - clock()))
                request = clock()
                elapsed = request - origin
                s = client.state()
                received = clock()
                if coherent_clocks:
                    clock_bounds.observe(
                        request, received, s["state_timestamp_monotonic_s"]
                    )
                last = s
                if (
                    not s["running"]
                    or s["phase"] != "active"
                    or s["mode"] != plan["mode"]
                    or s["epoch"] != epoch
                    or s["applied_seq"] > own
                    or s["automatic_recovery_enabled"]
                    or s["robot_has_errors"]
                    or not s["sample_valid"]
                    or not s["kinematics_valid"]
                    or not 0 <= s["sample_age_s"] <= 0.005
                ):
                    raise RuntimeError("Invalid/stale/lifecycle observation")
                w = canonical_franka_wrench(s, stiffness_at_tcp=True)
                sample = sdk.sample(
                    s["positions"], s["velocities"], np.zeros(7), np.zeros(6)
                )
                j = sample.jacobian.copy()
                j[3:] *= home["controller_config"]["jacobian_rotation_scale"]
                sigma = np.linalg.svd(j, compute_uv=False)[-1]
                env.check(
                    s["positions"],
                    s["velocities"],
                    s["position"],
                    s["quaternion_xyzw"],
                    s["twist"],
                    sigma,
                    w,
                )
                measured = s["server_monotonic_s"] - s["sample_age_s"] - server_origin
                q, phase = reference(plan, elapsed)
                log.write(
                    json.dumps(
                        {
                            **s,
                            "time_s": measured,
                            "host_elapsed_s": elapsed,
                            "external_wrench_tcp_base": w.tolist(),
                            "policy": phase,
                            "request_latency_s": clock() - request,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                count += 1
                if elapsed >= plan["duration_s"]:
                    break
                due = origin + command_tick * 0.02
                if clock() >= due:
                    if clock() - due > 0.05:
                        raise RuntimeError("Missed publication deadline; no catch-up")
                    age = (
                        clock_bounds.age_upper(
                            s["sample_timestamp_monotonic_s"], clock()
                        )
                        if coherent_clocks
                        else s["sample_age_s"] + clock() - request
                    )
                    if age > 0.01:
                        raise RuntimeError("Publication observation age exceeded 10 ms")
                    q, phase = reference(plan, clock() - origin)
                    response = send(q, 0.2, phase)
                    own = response["command_seq"]
                    actions.write(
                        json.dumps(
                            {
                                "time_s": clock() - origin,
                                "joints": q.tolist(),
                                "pose": (
                                    pose_command(plan, phase)
                                    if plan["mode"] == "cartesian"
                                    else None
                                ),
                                "response": response,
                                "publication_age_bound_s": age,
                                "legacy_roundtrip_age_bound_s": s["sample_age_s"]
                                + clock()
                                - request,
                                "coherent_native_age_bound": coherent_clocks,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    command_tick = int(np.floor((clock() - origin) / 0.02)) + 1
                next_poll = max(next_poll + poll_period, clock())
    except BaseException as error:
        write_json_new(
            output / "failure.json",
            {"error": str(error), "samples": count, "last_state": last},
        )
        raise
    finally:
        try:
            if attempted:
                client.timeout_s = 3.5
                stopped = client.stop(wait_s=2.5)
                write_json_new(output / "stop.json", stopped)
                if plan.get("native_recording_capacity") and stopped.get("stopped"):
                    try:
                        from .native_telemetry import quality

                        client.download_telemetry(output / "native.npz")
                        native_quality = quality(output / "native.npz")
                        write_json_new(output / "native-quality.json", native_quality)
                        if native_quality["capture_id"] != capture_status["capture_id"]:
                            raise RuntimeError("Native capture ID mismatch")
                        client.clear_telemetry(capture_status["capture_id"])
                    except Exception as error:
                        write_json_new(
                            output / "native-export-error.json",
                            {
                                "error": str(error),
                                "capture_retained_on_server_unless_cleared": True,
                            },
                        )
        finally:
            try:
                client.close()
            finally:
                if enabled:
                    gc.enable()
    if not stopped or not stopped["stopped"]:
        raise RuntimeError("Stop not confirmed")
    result = {
        "completed": True,
        "stop_confirmed": True,
        "samples": count,
        "mode": plan["mode"],
        "joint_index": plan["joint_index"],
        "tracking_success_not_claimed": True,
        "final_state": last,
    }
    write_json_new(output / "completion.json", result)
    return result


def replay(home, parameters, observer, observed, output):
    observed = Path(observed)
    plan = json.loads((observed / "manifest.json").read_text())
    startup = json.loads((observed / "startup.json").read_text())
    origin = startup["policy_origin_server_monotonic_s"]
    actions = [json.loads(l) for l in (observed / "actions.jsonl").open()]
    events = []
    last = -np.inf
    for a in actions:
        e = {
            "time_s": a["response"]["accepted_monotonic_s"] - origin,
            "deadline_s": a["response"]["deadline_s"] - origin,
            "joints": a["joints"],
            "pose": a.get("pose"),
        }
        if (
            not np.isfinite([e["time_s"], e["deadline_s"]]).all()
            or e["time_s"] <= last
            or e["deadline_s"] <= e["time_s"]
        ):
            raise ValueError("Invalid accepted event sequence")
        events.append(e)
        last = e["time_s"]
    if events[0]["time_s"] > 0:
        raise ValueError("Missing startup acceptance")
    rows = [json.loads(l) for l in (observed / "trace.jsonl").open()]
    result = simulate(
        home,
        parameters,
        observer,
        plan,
        output,
        events=events,
        initial=startup["state"],
        end_s=min(plan["duration_s"], rows[-1]["time_s"]),
    )
    completion = observed / "completion.json"
    real_completed = (
        completion.exists()
        and json.loads(completion.read_text()).get("completed", False)
        and not (observed / "failure.json").exists()
    )
    result.update(
        observed_completed=bool(real_completed),
        coverage_fraction=float(
            min(1.0, max(0.0, rows[-1]["time_s"]) / plan["duration_s"])
        ),
        full_run_comparison=bool(real_completed and result["reason"] is None),
        warning=(
            None
            if real_completed
            else "PARTIAL/FAILED observation: a small replay error is not full-run validation"
        ),
    )
    write_json_new(Path(output) / "replay.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["plan", "screen", "run", "replay"])
    p.add_argument("--home", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for n in ["plan", "plant", "observer", "screen", "observed"]:
        p.add_argument("--" + n, type=Path)
    p.add_argument("--joint", type=int)
    p.add_argument("--mode", choices=["joint", "cartesian"], default="joint")
    p.add_argument(
        "--role", choices=["training", "validation", "development"], default="training"
    )
    p.add_argument("--amplitude-deg", type=float, default=0.5)
    p.add_argument("--hold-s", type=float, default=30.0)
    p.add_argument("--poll-hz", type=int, choices=[50, 100, 200], default=200)
    p.add_argument("--record-native", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm-clear-path", action="store_true")
    a = p.parse_args(argv)
    h = load_home(a.home)
    if a.operation == "plan":
        r = capture(
            h,
            joint=a.joint,
            mode=a.mode,
            role=a.role,
            amplitude_deg=a.amplitude_deg,
            hold_s=a.hold_s,
            poll_hz=a.poll_hz,
            record_native=a.record_native,
        )
        write_json_new(a.output, r)
    elif a.operation == "run":

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupt)
        r = run(
            h,
            json.loads(a.plan.read_text()),
            a.screen,
            a.output,
            execute=a.execute,
            clear_path=a.confirm_clear_path,
        )
    else:
        params = PlantParameters(**json.loads(a.plant.read_text()))
        obs = json.loads(a.observer.read_text())
        r = (
            simulate(h, params, obs, json.loads(a.plan.read_text()), a.output)
            if a.operation == "screen"
            else replay(h, params, obs, a.observed, a.output)
        )
    print(
        json.dumps(
            {k: v for k, v in r.items() if k not in ("final_state", "start")}, indent=2
        )
    )
    return 1 if r.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
