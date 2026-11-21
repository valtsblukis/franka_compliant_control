"""Connect directly and hold the robot's current pose in this foreground process.

Run when ready to start torque control. Ctrl+C/SIGTERM requests a stop, then
closes the robot connection. No HTTP server, background service or auto-recovery.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time

from franka_compliant_control.hold import hold_compliantly
from franka_compliant_control.diagnostics import final_error_message
from franka_compliant_control.local_client import LocalControllerClient
from franka_compliant_control.models import ControllerConfig, HardwareSettings


def main(argv=None, *, continuous_impedance=False):
    description = ("Hold the current pose using continuous bounded impedance in this foreground process. "
                   "The spring remains attached to the target during contact. Ctrl+C stops."
                   if continuous_impedance else __doc__)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", choices=("robot", "sim"), default="robot",
                        help="Direct robot connection (default), or local simulation")
    if continuous_impedance:
        parser.add_argument("--profile", choices=("contact", "impedance"), default="contact",
                            help="Bounded spring with wider contact tolerance (default), or previous impedance settings")
    else:
        parser.add_argument("--profile", choices=("commissioning", "perturbation"), default="perturbation",
                            help="Task tracking with compliant yielding (default), or original commissioning settings")
    parser.add_argument("--duration", type=float, default=120,
                        help="Seconds to hold (default 120); 0 means until Ctrl+C")
    parser.add_argument("--max-displacement", type=float,
                        help=("Optional excursion stop in metres (default: no added excursion bound)" if continuous_impedance else
                              "Excursion stop in metres (default .15; commissioning .05)"))
    parser.add_argument("--max-rotation-deg", type=float,
                        help=("Optional orientation excursion stop in degrees (default: no added excursion bound)" if continuous_impedance else
                              "Orientation excursion stop in degrees (default 30; commissioning 15)"))
    parser.add_argument("--output", type=Path, help="New JSONL filename; defaults to a timestamped log")
    args = parser.parse_args(argv)
    if args.max_displacement is None and not continuous_impedance:
        args.max_displacement = .05 if args.profile == "commissioning" else .15
    if args.max_rotation_deg is None and not continuous_impedance:
        args.max_rotation_deg = 15 if args.profile == "commissioning" else 30
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and nonnegative")
    if any(v is not None and (not math.isfinite(v) or v <= 0)
           for v in (args.max_displacement, args.max_rotation_deg)):
        parser.error("excursion limits must be finite and positive")
    root = Path(__file__).resolve().parents[1]
    config_path = root / {"commissioning": "local-controller.json",
                          "perturbation": "configs/controller.fr3.perturbation.json",
                          "contact": "configs/controller.fr3.contact.json",
                          "impedance": "configs/controller.fr3.impedance.json"}[args.profile]
    hardware_path = root / "local-hardware.json"
    config = ControllerConfig.model_validate_json(config_path.read_text())
    expected_mode = "continuous_impedance" if continuous_impedance else "contact_yielding"
    if config.compliance_mode != expected_mode:
        parser.error(f"{config_path} must select compliance_mode={expected_mode} for this launcher")
    hardware = (HardwareSettings.model_validate_json(hardware_path.read_text())
                if args.backend == "robot" else None)
    log_prefix = "impedance-hold" if continuous_impedance else "compliant-hold"
    output = args.output or root / "logs" / f"{log_prefix}-{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
    stop_event = threading.Event()
    previous_handlers = {}
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[number] = signal.signal(number, lambda *_: stop_event.set())

    def status(event):
        if event["event"] == "armed":
            print(f"Target position (m): {list(event['target_position'])}", flush=True)
            print("Holding position and orientation. Ctrl+C stops; release pressure to let it return.", flush=True)
            if continuous_impedance:
                print("Continuous impedance: the restoring spring stays attached to this target during contact.", flush=True)
        else:
            state = event["state"]
            force = math.sqrt(sum(v*v for v in state["external_wrench"][:3]))
            moment = math.sqrt(sum(v*v for v in state["external_wrench"][3:]))
            limits = [label for field, label in (("reference_limited", "reference-error"),
                      ("reference_speed_limited", "reference-speed"),
                      ("task_force_limited", "force-cap"), ("task_moment_limited", "moment-cap"),
                      ("joint_torque_limited", "torque/slew")) if state.get(field)]
            if state.get("drive_scale", 1) < 1:
                limits.append("contact")
            behavior = "IMPEDANCE" if continuous_impedance else ("YIELDING" if state["yielding"] else "TRACKING")
            print(f"{event['elapsed_s']:6.1f}s  {state['phase'].upper():6s}  "
                  f"{behavior:9s}  "
                  f"error {event['position_error_m']*1000:5.1f} mm / "
                  f"{math.degrees(event['rotation_error_rad']):4.1f} deg  "
                  f"estimated load {force:4.1f} N / {moment:.2f} Nm  "
                  f"contact-limit {state.get('contact_load_ratio', 0):.0%} "
                  f"(peak {state.get('peak_contact_load_ratio', 0):.0%})  "
                  f"limits={','.join(limits) or 'none'}", flush=True)

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as log:
            def record(event):
                log.write(json.dumps({"time_monotonic_s": time.monotonic(), **event}, allow_nan=False) + "\n")
                log.flush()
            print(f"Log: {output}", flush=True)
            print(f"Foreground process PID {os.getpid()} owns the controller and closes it on exit.", flush=True)
            print(f"Controller settings: {config_path}", flush=True)
            print(f"Control law: {config.compliance_mode}", flush=True)
            if config.bounded_cartesian_spring:
                print("Cartesian spring capped before damping; fixed target retained during contact.", flush=True)
            if args.profile == "contact":
                print("Contact profile uses provisional application thresholds; physical contact validation remains pending.", flush=True)
            print(f"Profile: {args.profile}; translation stiffness {config.translation_stiffness[0]:g} N/m; "
                  f"task effort caps {config.commanded_force_limit:g} N / {config.commanded_moment_limit:g} Nm.", flush=True)
            joint_trips = [v or config.measured_joint_velocity_limit or 1.5*config.joint_velocity
                           for v in config.measured_joint_velocity_limits]
            print(f"Estimated contact stops ({'per TCP axis' if continuous_impedance else 'resultant norm'}): "
                  f"{config.contact_force_stop:g} N / {config.contact_moment_stop:g} Nm; "
                  f"{'target-change' if continuous_impedance else 'return'} reference speed {config.translation_velocity:g} m/s.", flush=True)
            print(f"External joint-torque stops: {', '.join(f'{v:g}' for v in config.external_torque_limits)} Nm.", flush=True)
            if config.motion_limit_policy == "franka":
                print("Motion bounds: FR3 joint range with position- and direction-dependent speed limits; "
                      "listed joint speeds are upper ceilings, reduced near joint limits.", flush=True)
            print(f"Measured speed trips: {config.measured_translation_velocity_limit or 1.5*config.translation_velocity:g} m/s / "
                  f"{config.measured_rotation_velocity_limit or 1.5*config.rotation_velocity:g} rad/s angular; "
                  f"joints 1-7: {', '.join(f'{v:g}' for v in joint_trips)} rad/s.", flush=True)
            if hardware:
                print(f"Hardware settings: {hardware_path}", flush=True)
                print(f"Connecting directly to robot {hardware.robot_ip} (FCI).", flush=True)
            else:
                print("Using local simulation; no robot connection.", flush=True)
            client = None
            try:
                record({"event": "launch", "pid": os.getpid(), "backend": args.backend, "profile": args.profile,
                        "controller_config": config.model_dump(),
                        "hardware_config": hardware.model_dump() if hardware else None})
                if stop_event.is_set():
                    print("Cancelled before connecting.", flush=True)
                    return 0
                with LocalControllerClient(config, backend_kind=args.backend, hardware=hardware) as client:
                    displacement = "none" if args.max_displacement is None else f"{args.max_displacement*1000:g} mm"
                    rotation = "none" if args.max_rotation_deg is None else f"{args.max_rotation_deg:g} deg"
                    print(f"Starting torque control; added excursion bounds: {displacement} / {rotation}.", flush=True)
                    summary = hold_compliantly(client, duration_s=args.duration,
                        max_displacement_m=args.max_displacement,
                        max_rotation_rad=(math.radians(args.max_rotation_deg)
                                          if args.max_rotation_deg is not None else None),
                        stop_event=stop_event, record=record, report=status,
                        expected_backend=args.backend)
            except Exception as error:
                if client is not None and getattr(client, "final_state", None) is not None:
                    record({"event": "controller_closed", "state": client.final_state})
                message = final_error_message(error, getattr(client, "final_state", None))
                print(f"COMPLIANT CONTROL ENDED: {message}", file=sys.stderr, flush=True)
                print("Compliance is inactive. Controller closed; this command leaves no service running. No recovery attempted.", file=sys.stderr, flush=True)
                record({"event": "error", "message": message, "initial_exception": str(error)})
                return 1
            print("Stopped and disarmed." if summary.get("stopped") else "Cancelled before arming.", flush=True)
            print("Compliance is inactive. Controller closed; this command leaves no service running.", flush=True)
            return 0
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
