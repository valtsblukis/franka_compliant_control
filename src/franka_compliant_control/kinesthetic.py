"""Kinesthetic-teaching recorder: hold a compliant pose target that follows the operator.

The target is the measured TCP pose at startup. When an operator pushes the
end effector more than a translation OR rotation threshold (default 5 cm /
5 deg) away from the current target, the target is reset to the measured pose.
The controller's gains, limits and collision settings are never changed, and
there is no recovery or retry: any controller stop ends the recording.

Output directory (new, never overwritten):
  manifest.json  controller config, URDF digest, service info, thresholds, initial state
  actions.jsonl  every sent target and its server response (accepted time/deadline)
  trace.jsonl    one full /state sample per tick, plus the target in force at that tick
  summary.json   end reason, stop confirmation and the covered workspace (TCP/joint bounds)
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from .client import CompliantClient
from .diagnostics import stop_failure_message
from .network import DEFAULT_URL
from .twin_experiment import digest, write_json_new
from .twin_model import rotation_distance

SCHEMA = "fcc-kinesthetic-v1"


def check_start_state(state, expected_backend):
    """Fresh, inactive, writable, error-free, stationary observation."""
    problems = []
    if state["backend"] != expected_backend:
        problems.append(f"backend={state['backend']} (expected {expected_backend})")
    if state["read_only"] or state["running"] or state["phase"] != "disarmed":
        problems.append(f"read_only={state['read_only']} running={state['running']} phase={state['phase']}")
    if state.get("motion_inhibited"):
        problems.append("software motion inhibit is asserted")
    if expected_backend == "robot" and (state.get("robot_mode") != "Idle" or state.get("robot_has_errors")):
        problems.append(f"robot_mode={state.get('robot_mode')} errors={state.get('robot_has_errors')}")
    if not state["sample_valid"] or not state.get("kinematics_valid", True) or not 0 <= state["sample_age_s"] <= .1:
        problems.append(f"sample not fresh/valid (age {state['sample_age_s']:.3f} s)")
    if np.max(np.abs(state["velocities"])) >= .02:
        problems.append("robot is not stationary")
    if problems:
        raise RuntimeError("Not ready to start: " + "; ".join(problems) + ". No stop/recovery was attempted.")


def _bounds(values):
    values = np.asarray(values, dtype=float)
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}


def collect(client, output_dir, *, translation_threshold_m=.05, rotation_threshold_rad=math.radians(5),
            duration_s=0., hz=50., lease_s=.2, expected_backend="robot", stop_event=None,
            report=lambda event: None, clock=time.monotonic):
    """Record one kinesthetic session into a new directory; return the summary.

    The first target starts torque control. Stop is always requested and
    confirmed before returning, including after errors and interruptions.
    """
    if not (math.isfinite(translation_threshold_m) and translation_threshold_m > 0
            and math.isfinite(rotation_threshold_rad) and rotation_threshold_rad > 0):
        raise ValueError("thresholds must be finite and positive")
    if not (math.isfinite(duration_s) and duration_s >= 0 and 0 < hz <= 100 and 0 < lease_s <= 1):
        raise ValueError("invalid duration, rate or lease")
    stop_event = stop_event if stop_event is not None else threading.Event()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    health = client.health()
    if health["automatic_recovery_enabled"]:
        raise RuntimeError("Disable automatic recovery on the service first; this recorder never changes settings")
    config = client.get_config()
    try:
        urdf_sha256 = digest(client.get_urdf()["urdf"])
    except Exception:
        if expected_backend == "robot":
            raise
        urdf_sha256 = None  # The toy test backend has no URDF.
    initial = client.refresh_state()["state"]
    check_start_state(initial, expected_backend)
    output_dir.mkdir(parents=True)
    write_json_new(output_dir/"manifest.json", {
        "schema": SCHEMA, "created": datetime.now().astimezone().isoformat(),
        "translation_threshold_m": translation_threshold_m, "rotation_threshold_rad": rotation_threshold_rad,
        "duration_s": duration_s, "hz": hz, "lease_s": lease_s, "expected_backend": expected_backend,
        "controller_config": config, "controller_config_sha256": digest(config),
        "urdf_sha256": urdf_sha256, "health": health, "initial_state": initial,
        "measurement": "HTTP latest-sample polling at the tick rate; not 1 kHz telemetry",
        "target_rule": "target := measured TCP pose whenever it is > threshold (translation OR rotation) from the target"})

    anchor_p, anchor_q = list(initial["position"]), list(initial["quaternion_xyzw"])
    positions, joints = [], []
    samples = retargets = 0
    path_length = max_force = max_moment = 0.
    min_singular = math.inf
    reason, error, stop_result, started = "duration", None, None, None
    attempted = False
    with (output_dir/"trace.jsonl").open("x") as trace, (output_dir/"actions.jsonl").open("x") as actions:
        def send(retarget, lease):
            response = client.send_pose(anchor_p, anchor_q, valid_for_s=lease)
            actions.write(json.dumps({"host_elapsed_s": clock()-started, "retarget": retarget,
                "action": {"position": anchor_p, "quaternion_xyzw": anchor_q}, "response": response},
                allow_nan=False) + "\n")
            return response
        try:
            if stop_event.is_set():
                reason = "cancelled_before_start"
            else:
                started = clock()
                attempted = True  # A lost response may still have started control.
                send(False, None)  # Service default lease covers controller startup.
                report({"event": "started", "position": anchor_p, "quaternion_xyzw": anchor_q})
                next_tick = started
                last_position = None
                while not stop_event.is_set():
                    now = clock()
                    if duration_s and now-started >= duration_s:
                        break
                    state = client.state()
                    if not state["running"] or state["phase"] not in ("armed", "active"):
                        reason = "controller_stopped"
                        raise RuntimeError(f"Controller stopped: {state['phase']}/{state['reason']}: {state['error']}")
                    if (not state["sample_valid"] or not state.get("kinematics_valid", True)
                            or not 0 <= state["sample_age_s"] <= .05):
                        reason = "stale_telemetry"
                        raise RuntimeError("Telemetry invalid or older than 50 ms")
                    p = np.asarray(state["position"], dtype=float)
                    distance = float(np.linalg.norm(p-anchor_p))
                    angle = rotation_distance(state["quaternion_xyzw"], anchor_q)
                    if not (math.isfinite(distance) and math.isfinite(angle)):
                        reason = "nonfinite_pose"
                        raise RuntimeError("Nonfinite measured pose")
                    retarget = distance > translation_threshold_m or angle > rotation_threshold_rad
                    trace.write(json.dumps({"host_elapsed_s": now-started, "target_position": anchor_p,
                        "target_quaternion_xyzw": anchor_q, "target_distance_m": distance,
                        "target_angle_rad": angle, "retarget": retarget, "state": state}, allow_nan=False) + "\n")
                    samples += 1
                    positions.append(p.tolist())
                    joints.append(state["positions"])
                    if last_position is not None:
                        path_length += float(np.linalg.norm(p-last_position))
                    last_position = p
                    wrench = state.get("external_wrench") or [0.]*6
                    max_force = max(max_force, float(np.linalg.norm(wrench[:3])))
                    max_moment = max(max_moment, float(np.linalg.norm(wrench[3:])))
                    min_singular = min(min_singular, state.get("minimum_singular_value", math.inf))
                    if retarget:
                        anchor_p, anchor_q = p.tolist(), list(state["quaternion_xyzw"])
                        retargets += 1
                    send(retarget, lease_s)
                    report({"event": "sample", "elapsed_s": now-started, "retargets": retargets,
                            "distance_m": distance, "angle_rad": angle, "retarget": retarget, "state": state})
                    # Skip missed slots; never burst stale commands to catch up.
                    next_tick = max(next_tick+1/hz, clock())
                    stop_event.wait(max(0., next_tick-clock()))
                if stop_event.is_set():
                    reason = "interrupted"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if reason == "duration":
                reason = "error"
        finally:
            if attempted:
                try:
                    stop_result = client.stop(wait_s=min(5., config.get("stop_timeout", 1.)+.5))
                except Exception as exc:
                    error = (error + "; " if error else "") + f"stop request failed: {exc}"
    final = (stop_result or {}).get("state") or {}
    if final.get("phase") == "fault":
        # Telemetry often goes invalid a tick before `running` drops; name the actual fault.
        reason = f"fault:{final.get('reason')}"
    stopped = bool(stop_result and stop_result.get("stopped")) or not attempted
    if stop_result and not stop_result.get("stopped"):
        error = (error + "; " if error else "") + stop_failure_message(stop_result["state"])
    summary = {"schema": SCHEMA, "reason": reason, "error": error, "stop_confirmed": stopped,
               "stop_result": stop_result, "elapsed_s": clock()-started if started is not None else 0.,
               "samples": samples, "retargets": retargets, "tcp_path_length_m": path_length,
               "max_external_force_n": max_force, "max_external_moment_nm": max_moment,
               "min_singular_value": min_singular if samples else None,
               "workspace": {"tcp_position_m": _bounds(positions), "joint_positions_rad": _bounds(joints)}
                            if samples else None}
    write_json_new(output_dir/"summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="New directory (default logs/kinesthetic-<timestamp>)")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--translation-threshold", type=float, default=.05, help="Metres (default .05)")
    parser.add_argument("--rotation-threshold-deg", type=float, default=5., help="Degrees (default 5)")
    parser.add_argument("--duration", type=float, default=0., help="Seconds; 0 (default) records until Ctrl+C")
    parser.add_argument("--hz", type=float, default=50.)
    parser.add_argument("--expected-backend", choices=("robot", "sim"), default="robot")
    parser.add_argument("--execute", action="store_true",
                        help="Start torque control. Without it, only a read-only readiness check is performed")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output = args.output or root/"logs"/f"kinesthetic-{datetime.now():%Y%m%d-%H%M%S}"
    client = CompliantClient(args.url, timeout_s=2.)
    # Persistent connections showed ~44 ms delayed-ACK latency on this stack (see twin_real).
    client.http.headers["Connection"] = "close"
    if not args.execute:
        with client:
            state = client.refresh_state()["state"]
            check_start_state(state, args.expected_backend)
            print(json.dumps({"ready": True, "automatic_recovery_enabled": client.health()["automatic_recovery_enabled"],
                              "position": state["position"], "quaternion_xyzw": state["quaternion_xyzw"],
                              "joints": state["positions"], "would_write": str(output)}, indent=2))
            print("Dry run only. Re-run with --execute to start compliant control.")
        return 0
    stop_event = threading.Event()
    previous = {n: signal.signal(n, lambda *_: stop_event.set()) for n in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    last_status = [0.]

    def report(event):
        if event["event"] == "started":
            print("Compliant control started. Guide the robot by hand. To finish, let go, wait for it to settle, then Ctrl+C.", flush=True)
        elif event["retarget"] or time.monotonic()-last_status[0] >= 1:
            last_status[0] = time.monotonic()
            s = event["state"]
            force = math.hypot(*s.get("external_wrench", [0.]*6)[:3])
            print(f"{event['elapsed_s']:7.1f}s  targets moved {event['retargets']:4d}  "
                  f"offset {event['distance_m']*1000:5.1f} mm / {math.degrees(event['angle_rad']):4.1f} deg  "
                  f"ext force {force:4.1f} N  tcp {np.round(s['position'], 3).tolist()}"
                  + ("  NEW TARGET" if event["retarget"] else ""), flush=True)

    try:
        print(f"Recording to {output}", flush=True)
        with client:
            summary = collect(client, output, translation_threshold_m=args.translation_threshold,
                              rotation_threshold_rad=math.radians(args.rotation_threshold_deg),
                              duration_s=args.duration, hz=args.hz, expected_backend=args.expected_backend,
                              stop_event=stop_event, report=report)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    print(json.dumps({k: summary[k] for k in ("reason", "error", "stop_confirmed", "samples", "retargets",
                                             "elapsed_s", "tcp_path_length_m", "workspace")}, indent=2))
    if not summary["stop_confirmed"]:
        print("STOP NOT CONFIRMED: use the physical stop if the robot is moving unexpectedly.", file=sys.stderr)
    return 0 if summary["stop_confirmed"] and summary["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
