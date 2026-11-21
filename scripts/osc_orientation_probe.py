"""Short local orientation step/return test. Simulation by default; no recovery.

Probe-specific abort bounds are not changes to the controller operating envelope.
The initial pose is captured after arm; never returns to a previously saved pose.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

from franka_compliant_control.local_client import LocalControllerClient
from franka_compliant_control.models import ControllerConfig, HardwareSettings
from franka_compliant_control.diagnostics import stop_failure_message, collision_application_message, final_error_message


def quaternion_multiply(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]


def angular_distance(a, b):
    dot = abs(sum(x*y for x, y in zip(a, b)))
    return 2*math.acos(min(1.0, dot))


def probe(client, *, axis="x", degrees=2.0, target_time=4.0, return_time=4.0,
          stop_event=None, record=lambda event: None, expected_backend="sim"):
    if axis not in "xyz" or len(axis) != 1:
        raise ValueError("axis must be x, y or z")
    if not math.isfinite(degrees) or not 0 < abs(degrees) <= 5:
        raise ValueError("this small-motion probe requires 0 < abs(degrees) <= 5")
    if any(not math.isfinite(t) or t <= 0 for t in (target_time, return_time)):
        raise ValueError("durations must be finite and positive")
    event = stop_event if stop_event is not None else threading.Event()
    before = client.state()
    if (before["backend"] != expected_backend or before["read_only"] or
            before["running"] or before["phase"] != "disarmed"):
        raise RuntimeError("Need an exclusively owned disarmed writable controller")
    config = client.get_config()
    if config["compliance_mode"] not in ("operational_space", "auto"):
        raise ValueError("probe requires operational_space")
    record({"event": "before", "state": before, "controller_config": config})
    if event.is_set():
        return {"armed": False, "reason": "cancelled_before_arm"}
    stage_results = []
    try:
        initial = client.arm()["state"]
        print(collision_application_message(initial), flush=True)
        position = tuple(initial["position"])
        orientation = tuple(initial["quaternion_xyzw"])
        increment = [0.0, 0.0, 0.0, math.cos(math.radians(degrees)/2)]
        increment["xyz".index(axis)] = math.sin(math.radians(degrees)/2)
        target = quaternion_multiply(increment, orientation)
        record({"event": "armed", "state": initial, "target_position": position,
                "initial_quaternion_xyzw": orientation, "step_quaternion_xyzw": target,
                "axis": axis, "degrees": degrees, "target_time_s": target_time,
                "return_time_s": return_time})
        start = next_tick = time.monotonic()
        for name, duration, quaternion in [("initial", 2.0, orientation),
                                           ("step", target_time, target),
                                           ("return", return_time, orientation)]:
            if event.is_set():
                break
            stage_start = time.monotonic()
            record({"event": "stage", "stage": name, "quaternion_xyzw": quaternion})
            print(f"Stage {name}: {duration:g} s", flush=True)
            last = None
            while not event.is_set() and time.monotonic()-stage_start < duration:
                client.send_pose(position, quaternion)
                state = client.state()
                now = time.monotonic()
                if state["phase"] not in ("armed", "active") or not state["running"]:
                    record({"event": "controller_stopped", "state": state})
                    raise RuntimeError(f"Controller stopped: {state['reason']}: {state['error']}")
                if (not state["sample_valid"] or not math.isfinite(state["sample_age_s"]) or
                        state["sample_age_s"] > .05):
                    raise RuntimeError("Invalid or stale (>50 ms) probe telemetry")
                distance = math.dist(position, state["position"])
                angle = angular_distance(orientation, state["quaternion_xyzw"])
                last = {"event": "sample", "elapsed_s": now-start, "stage": name,
                        "stage_elapsed_s": now-stage_start, "position_error_m": distance,
                        "rotation_error_rad": angular_distance(quaternion, state["quaternion_xyzw"]),
                        "rotation_from_initial_rad": angle, "state": state}
                record(last)
                if (distance > .025 or angle > math.radians(abs(degrees)+3) or
                        math.hypot(*state["twist"][:3]) > .1 or
                        math.hypot(*state["twist"][3:]) > .35):
                    raise RuntimeError("Probe-specific excursion/speed abort; controller envelope unchanged")
                next_tick = max(next_tick+.02, time.monotonic())
                event.wait(max(0, next_tick-time.monotonic()))
            if last:
                result = {"stage": name, "position_error_m": last["position_error_m"],
                          "rotation_error_deg": math.degrees(last["rotation_error_rad"]),
                          "rotation_from_initial_deg": math.degrees(last["rotation_from_initial_rad"])}
                stage_results.append(result)
                print(result, flush=True)
    finally:
        result = client.stop(wait_s=min(5.0, config["stop_timeout"]+.5))
        record({"event": "stop", "result": result})
        if not result["stopped"]:
            raise RuntimeError(stop_failure_message(result["state"]))
    summary = {"armed": True, "stopped": True, "stages": stage_results,
               "reason": "interrupted" if event.is_set() else "completed"}
    record({"event": "summary", **summary})
    return summary


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("sim", "robot"), default="sim")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--config", type=Path, default=root/"configs/controller.fr3.osc.json")
    p.add_argument("--axis", choices=("x", "y", "z"), default="x")
    p.add_argument("--degrees", type=float, default=2)
    p.add_argument("--target-time", type=float, default=4)
    p.add_argument("--return-time", type=float, default=4)
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.backend == "robot" and not a.execute:
        p.error("Hardware requires --execute and an operator ready at the physical stop")
    c = ControllerConfig.model_validate_json(a.config.read_text())
    # Linear OSC paces this probe through native Ruckig. Progressive OSC uses
    # direct equilibrium steps; these bounds only validate optional feedforward.
    # Measured-motion limits are unchanged in either case.
    c = ControllerConfig(**(c.model_dump() | {"translation_velocity": .05,
        "translation_acceleration": .2, "translation_jerk": 2,
        "rotation_velocity": .12, "rotation_acceleration": .4, "rotation_jerk": 4}))
    h = HardwareSettings.model_validate_json((root/"local-hardware.json").read_text()) if a.backend == "robot" else None
    output = a.output or root/"logs"/f"osc-probe-{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
    event = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: event.set())
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as f:
            def record(e):
                f.write(json.dumps({"time_monotonic_s": time.monotonic(), **e}, allow_nan=False)+"\n")
                f.flush()
            print(f"Log: {output}\nBackend: {a.backend}; axis: base-{a.axis}; step: {a.degrees:g} degrees", flush=True)
            if c.osc_feedback_mode != "linear":
                print("Bounded-equilibrium probe: DIRECT step, not a Ruckig-paced rotation.", flush=True)
            record({"event": "launch", "backend": a.backend, "controller_config": c.model_dump(),
                    "hardware_config": h.model_dump() if h else None})
            client = None
            try:
                if event.is_set():
                    return 0
                with LocalControllerClient(c, backend_kind=a.backend, hardware=h) as client:
                    result = probe(client, axis=a.axis, degrees=a.degrees,
                        target_time=a.target_time, return_time=a.return_time,
                        stop_event=event, record=record, expected_backend=a.backend)
                print("Stopped and disarmed; connection closed." if result.get("stopped") else "Cancelled before arm.", flush=True)
                return 0
            except Exception as error:
                if client is not None and getattr(client, "final_state", None) is not None:
                    record({"event": "controller_closed", "state": client.final_state})
                message = final_error_message(error, getattr(client, "final_state", None))
                record({"event": "error", "message": message, "initial_exception": str(error)})
                print(f"PROBE ENDED: {message}; no recovery attempted", file=sys.stderr, flush=True)
                return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
