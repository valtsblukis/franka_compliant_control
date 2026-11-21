"""Plan/stream a joint demo: captured current home and a TCP goal 30 cm lower.

Requires start_joint_pos.sh to be running. Default is PLAN ONLY (HTTP reads and
an explicit inactive state refresh); --execute sends setpoints. No auto recovery.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from franka_compliant_control import _native
from franka_compliant_control.client import CompliantClient
from franka_compliant_control.network import DEFAULT_URL
from franka_compliant_control.joint_demo import plan_lower_target, execute_joint_demo, quaternion_rotation, rotation_vector
from franka_compliant_control.models import ControllerConfig, HardwareSettings


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--drop", type=float, default=.30, help="Lower endpoint displacement along base -Z, metres")
    parser.add_argument("--switch-every", type=float, default=3, help="Seconds per selected target, irrespective of reaching")
    parser.add_argument("--move-time", type=float, default=2, help="Seconds of smooth joint interpolation per transition")
    parser.add_argument("--cycles", type=int, default=0, help="Home/lower interval pairs; 0 until Ctrl+C")
    parser.add_argument("--execute", action="store_true", help="Send planned joint setpoints; confirm swept workspace and physical stop readiness")
    parser.add_argument("--urdf", type=Path, default=root/"logs/robot-wrist-tool-import-20260909.json.urdf")
    parser.add_argument("--hardware-config", type=Path, default=root/"local-hardware.json", help="TCP transform used by the server")
    parser.add_argument("--output", type=Path, help="New JSONL filename; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.cycles < 0 or any(not math.isfinite(x) or x <= 0 for x in (args.drop,args.switch_every,args.move_time)):
        parser.error("drop/timing must be finite and positive; cycles must be nonnegative")
    if args.move_time > args.switch_every:
        parser.error("--move-time must fit within --switch-every")
    if not getattr(_native, "offline_kinematics_available", False):
        parser.error("This demo requires the local hardware-enabled libfranka >=0.18 offline model binding")
    hardware = HardwareSettings.model_validate_json(args.hardware_config.read_text())
    urdf = args.urdf.read_text()
    model = _native.OfflineKinematics(urdf, hardware.flange_to_tcp)
    stop_event = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop_event.set())
                for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    output = args.output or root/"logs"/f"joint-pos-demo-{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as log:
            def record(e):
                log.write(json.dumps({"time_monotonic_s": time.monotonic(), **e}, allow_nan=False)+"\n")
                log.flush()
            def report(e):
                s=e["state"]
                error=np.max(np.abs(np.asarray(e["target_joints"])-s["positions"]))
                print(f"{e['elapsed_s']:6.1f}s target={e['target_name']:5s} max joint error={math.degrees(error):5.2f} deg "
                      f"feedback-cap={s.get('joint_feedback_limited')} torque/slew={s.get('joint_torque_limited')}", flush=True)
            try:
                print(f"Log: {output}", flush=True)
                with CompliantClient(args.url, timeout_s=10) as client:
                    health = client.health()
                    if health["running"] or health["phase"] != "disarmed" or health["read_only"]:
                        raise RuntimeError("Need an inactive writable joint controller; stop the other producer and resolve faults explicitly")
                    config = ControllerConfig(**client.get_config())
                    if config.compliance_mode not in ("joint_impedance", "auto") or config.robot_model != hardware.robot_model:
                        raise RuntimeError("Run ./start_joint_pos.sh with the matching hardware configuration first")
                    state = client.refresh_state()["state"]
                    if (not state["sample_valid"] or state["sample_age_s"] > .05 or
                            state["phase"] != "disarmed" or state["running"] or
                            (state.get("backend") == "robot" and
                             (state.get("robot_mode") != "Idle" or state.get("robot_has_errors") is not False))):
                        raise RuntimeError("Need a fresh inactive valid robot observation")
                    pose, _ = model.evaluate(state["positions"])
                    position_difference = np.linalg.norm(pose[:3,3]-state["position"])
                    orientation_difference = np.linalg.norm(rotation_vector(pose[:3,:3] @ quaternion_rotation(state["quaternion_xyzw"]).T))
                    if position_difference > .001 or orientation_difference > math.radians(.2):
                        raise RuntimeError(f"Offline URDF/TCP disagrees with fresh robot pose ({position_difference*1000:.2f} mm / "
                                           f"{math.degrees(orientation_difference):.3f} deg); no setpoints sent")
                    record({"event": "before", "state": state, "controller_config": config.model_dump(),
                            "hardware_config": hardware.model_dump(), "urdf": str(args.urdf),
                            "urdf_sha256": hashlib.sha256(urdf.encode()).hexdigest()})
                    if stop_event.is_set():
                        return 0
                    plan = plan_lower_target(model, state["positions"], config, drop_m=args.drop,
                                             move_time_s=args.move_time, switch_every_s=args.switch_every)
                    record({"event": "plan", **plan})
                    print(f"Captured home TCP: {plan['home_position']}\nLower target TCP: {plan['lower_position']}", flush=True)
                    print(f"Home joints: {plan['home_joints']}\nLower joints: {plan['lower_joints']}", flush=True)
                    print(f"Plan: switch every {args.switch_every:g} s; each transition {args.move_time:g} s. "
                          f"Sampled peak TCP speed {plan['planned_peak_tcp_speed_m_s']:.3f} m/s; "
                          f"lowest TCP Z {plan['planned_min_tcp_height_m']:.3f} m.", flush=True)
                    print(f"Joint interpolation has up to {1000*plan['planned_max_lateral_excursion_m']:.1f} mm lateral / "
                          f"{plan['planned_max_orientation_excursion_deg']:.2f} deg orientation deviation between endpoints.", flush=True)
                    print("NO obstacle, table, self-collision or cable-clearance checking. Keep the entire swept path clear. "
                          "Do not intercept the moving arm with a hand; use a suitable compliant fixture for contact tests.", flush=True)
                    if not args.execute:
                        print("PLAN ONLY: no control started and no setpoints sent. Add --execute after reviewing clearance.", flush=True)
                        return 0
                    if stop_event.is_set():
                        return 0
                    if client.get_config() != config.model_dump():
                        raise RuntimeError("Controller configuration changed while planning; rerun the plan")
                    print(f"Starting at captured home, then lower at t={args.switch_every:g} s. Ctrl+C stops. No automatic recovery.", flush=True)
                    summary = execute_joint_demo(client, plan, config, cycles=args.cycles,
                                                 stop_event=stop_event, record=record, report=report)
                    print("Stopped and disarmed." if summary.get("stopped") else "Cancelled before control.", flush=True)
                    return 0
            except Exception as error:
                record({"event": "error", "message": str(error)})
                print(f"JOINT DEMO ENDED: {error}", file=sys.stderr, flush=True)
                return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
