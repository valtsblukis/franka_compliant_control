"""Foreground FR3 OSC experiment: captured home -> -20 cm Y -> +20 cm Y.

Simulation is the default. Hardware requires --backend robot --execute and an
operator-cleared path. Owns/closes one connection; no server, recovery or retry.
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

from franka_compliant_control.hold import hold_compliantly
from franka_compliant_control.diagnostics import collision_application_message, final_error_message
from franka_compliant_control.local_client import LocalControllerClient
from franka_compliant_control.models import ControllerConfig, HardwareSettings
from franka_compliant_control.tracking import track_two_targets
from franka_compliant_control.trajectory import demo_pacing


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sim", "robot"), default="sim")
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly authorize hardware torque control after clearing the path")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--profile", choices=("osc", "progressive"), default="osc")
    selection.add_argument("--config", type=Path)
    parser.add_argument("--hardware-config", type=Path, default=root/"local-hardware.json")
    parser.add_argument("--offset", type=float, default=.3, help="Base-Y distance each side of captured home, metres")
    parser.add_argument("--cycles", type=int, default=2, help="Number of -Y,+Y pairs; 0 until Ctrl+C")
    parser.add_argument("--speed", type=float, help="Reference speed: bounded profiles 1.5 m/s, linear OSC .5 m/s")
    parser.add_argument("--acceleration", type=float, help="Reference acceleration: bounded profiles 4.5 m/s^2, linear OSC .5")
    parser.add_argument("--jerk", type=float, help="Reference jerk: bounded profiles 2250 m/s^3, linear OSC 2")
    parser.add_argument("--switch-every", type=float, default=5, help="Switch target every N seconds, even if blocked; 0 waits for settling")
    parser.add_argument("--leg-timeout", type=float, default=30, help="Unfinished-leg timeout when --switch-every 0")
    parser.add_argument("--position-tolerance", type=float, help="Demo settling tolerance, metres; progressive default is 1.25 x soft radius")
    parser.add_argument("--rotation-tolerance-deg", type=float, help="Demo settling tolerance, degrees; progressive default is 1.25 x soft radius")
    parser.add_argument("--hold", action="store_true", help="Tune OSC at the captured pose without target changes")
    parser.add_argument("--duration", type=float, default=120, help="--hold duration; 0 until Ctrl+C")
    parser.add_argument("--output", type=Path, help="New JSONL file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.backend == "robot" and not args.execute:
        parser.error("Hardware requires --execute: confirm both endpoints AND the entire swept path are clear, with the physical stop ready")
    if args.cycles < 0 or any(not math.isfinite(v) or v <= 0 for v in
                             (args.offset, args.speed, args.acceleration, args.jerk, args.leg_timeout) if v is not None):
        parser.error("offset, trajectory parameters and timeout must be finite/positive; cycles must be nonnegative")
    if not math.isfinite(args.switch_every) or args.switch_every < 0 or 0 < args.switch_every < .02:
        parser.error("--switch-every must be 0 (settled mode) or at least .02 seconds")
    if args.leg_timeout <= .2 or not math.isfinite(args.duration) or args.duration < 0:
        parser.error("leg timeout must exceed .2 s; duration must be finite and nonnegative")
    if any(v is not None and (not math.isfinite(v) or v <= 0)
           for v in (args.position_tolerance, args.rotation_tolerance_deg)):
        parser.error("settling tolerances must be finite and positive")
    args.config = args.config or root/{"osc": "configs/controller.fr3.osc.json",
                                      "progressive": "configs/controller.fr3.osc.progressive.json"}[args.profile]
    config = ControllerConfig.model_validate_json(args.config.read_text())
    if config.compliance_mode not in ("operational_space", "auto"):
        parser.error("--config must select operational_space")
    defaults = demo_pacing(config.osc_feedback_mode)
    args.speed, args.acceleration, args.jerk = [default if value is None else value
        for value, default in zip((args.speed, args.acceleration, args.jerk), defaults)]
    # This changes only the requested experiment trajectory. Measured-motion
    # bounds and torque/collision policies remain those of the selected profile.
    try:
        config = ControllerConfig.model_validate(config.model_dump() | {
            "translation_velocity": args.speed, "translation_acceleration": args.acceleration,
            "translation_jerk": args.jerk})
    except ValueError as error:
        parser.error(str(error))
    hardware = (HardwareSettings.model_validate_json(args.hardware_config.read_text())
                if args.backend == "robot" else None)
    output = args.output or root/"logs"/f"osc-track-{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
    stop_event = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop_event.set())
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as log:
            def record(event):
                log.write(json.dumps({"time_monotonic_s": time.monotonic(), **event}, allow_nan=False)+"\n")
                log.flush()

            def report(event):
                if event["event"] == "armed":
                    print(f"Captured home: {event.get('home_position', event.get('target_position'))}", flush=True)
                    if "target_positions" in event:
                        print(f"Base-frame targets (-Y, +Y): {event['target_positions']}; orientation held.", flush=True)
                    print(collision_application_message(event.get("state", {})), flush=True)
                    print("OSC active. Ctrl+C stops; no automatic recovery or rearm.", flush=True)
                else:
                    s = event["state"]
                    saturation = (f" spring-cap F/M={s['progressive_force_saturated']}/{s['progressive_moment_saturated']}"
                                  if s.get("osc_feedback_mode", "linear") != "linear" else "")
                    print(f"{event['elapsed_s']:6.1f}s leg={event.get('leg', 'hold')} "
                          f"error={1000*event['position_error_m']:.1f} mm / "
                          f"{math.degrees(event['rotation_error_rad']):.2f} deg "
                          f"torque/slew-limited={s['joint_torque_limited']} "
                          f"regularized={s['osc_regularized']}{saturation}", flush=True)

            print(f"Config: {args.config}\nLog: {output}", flush=True)
            print(f"Trajectory v/a/j: {args.speed:g}/{args.acceleration:g}/{args.jerk:g} (SI); "
                  f"measured translation ceiling: {config.measured_translation_velocity_limit:g} m/s.", flush=True)
            print(f"Effort policy: {config.effort_limit_policy}; collision policy: {config.collision_policy}.", flush=True)
            if not args.hold:
                print(f"Target switching: every {args.switch_every:g} s, regardless of reaching." if args.switch_every else
                      "Target switching: wait for measured/reference settling.", flush=True)
            if config.collision_policy == "robot":
                print("Robot collision settings are preserved, NOT disabled; their thresholds are not readable here.", flush=True)
            else:
                print(f"On arm, REQUEST robot Cartesian upper thresholds {config.contact_force_stop:g} N / "
                      f"{config.contact_moment_stop:g} Nm per axis; joint upper {config.external_torque_limits} Nm.", flush=True)
                if config.collision_policy == "robot_configured":
                    print("Robot collision protection only; no duplicate software contact trips. No thresholds applied yet.", flush=True)
            if config.osc_feedback_mode != "linear":
                if config.osc_feedback_mode == "saturated":
                    print(f"NO SOFT ZONE: stiffness {config.osc_translation_stiffness} N/m / "
                          f"{config.osc_rotation_stiffness} Nm/rad, then force saturation.", flush=True)
                else:
                    print(f"Progressive spring: soft to {100*config.progressive_position_soft_radius:g} cm / "
                          f"{math.degrees(config.progressive_rotation_soft_radius):g} deg; plateau at "
                          f"{100*config.progressive_position_saturation_radius:g} cm / "
                          f"{math.degrees(config.progressive_rotation_saturation_radius):g} deg.", flush=True)
                print(f"Task effort caps: {config.commanded_force_limit:g} N / {config.commanded_moment_limit:g} Nm. "
                      "Direct, replaceable equilibrium; this demo paces target motion separately.", flush=True)
                print("Requested trajectory speed can be limited by the effort budget and move length.", flush=True)
            else:
                print(f"Additional Cartesian K/D: translation {config.osc_translation_stiffness} / "
                      f"{config.osc_translation_damping}; rotation {config.osc_rotation_stiffness} / "
                      f"{config.osc_rotation_damping} (physical SI units).", flush=True)
            print("Task-specific physical commissioning is required. No obstacle or full-path reachability check is provided.", flush=True)
            if args.backend == "sim":
                print("Simulation only: no robot connection. Test plant is not FR3 dynamics.", flush=True)
            record({"event": "launch", "backend": args.backend, "controller_config": config.model_dump(),
                    "hardware_config": hardware.model_dump() if hardware else None, "hold": args.hold,
                    "switch_every_s": args.switch_every if not args.hold else None})
            client = None
            try:
                if stop_event.is_set():
                    return 0
                with LocalControllerClient(config, backend_kind=args.backend, hardware=hardware) as client:
                    if args.hold:
                        summary = hold_compliantly(client, duration_s=args.duration,
                            max_displacement_m=None, max_rotation_rad=None, stop_event=stop_event,
                            record=record, report=report, expected_backend=args.backend)
                    else:
                        summary = track_two_targets(client, offset_m=args.offset, cycles=args.cycles,
                            leg_timeout_s=args.leg_timeout, switch_every_s=args.switch_every or None,
                            stop_event=stop_event, record=record,
                            position_tolerance_m=args.position_tolerance,
                            rotation_tolerance_rad=(math.radians(args.rotation_tolerance_deg)
                                                    if args.rotation_tolerance_deg is not None else None),
                            report=report, expected_backend=args.backend)
                print("Stopped and disarmed." if summary.get("stopped") else "Cancelled before arming.", flush=True)
                print("Controller closed; no service left running. No automatic return to home.", flush=True)
                return 0
            except Exception as error:
                if client is not None and getattr(client, "final_state", None) is not None:
                    record({"event": "controller_closed", "state": client.final_state})
                message = final_error_message(error, getattr(client, "final_state", None))
                print(f"OSC ENDED: {message}\nController closed; no recovery attempted.", file=sys.stderr, flush=True)
                record({"event": "error", "message": message, "initial_exception": str(error)})
                return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
