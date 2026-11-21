"""Foreground HTTP service; first setpoint starts control, with guarded fault recovery."""
import argparse
import os
from pathlib import Path

from franka_compliant_control.network import DEFAULT_PORT, run_http_server
from franka_compliant_control.models import ControllerConfig, HardwareSettings
from franka_compliant_control.service import create_app

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"local-controller.json",
                        help="Controller JSON; start.sh selects configs/controller.fr3.auto.json for both setpoint types")
    parser.add_argument("--hardware-config", type=Path, default=ROOT/"local-hardware.json")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 explicitly for a trusted-LAN policy client")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--status-interval", type=float, default=2.0, help="Terminal status interval in seconds; 0 disables")
    parser.add_argument("--no-auto-recover", action="store_true", help="Disable guarded recovery on new setpoints")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    hardware = HardwareSettings.model_validate_json(args.hardware_config.read_text())
    config = ControllerConfig.model_validate_json(args.config.read_text())
    app = create_app(config=config, backend_kind="robot", hardware=hardware,
                     status_interval_s=args.status_interval, automatic_recovery=not args.no_auto_recover)
    pid_path = ROOT/"logs/robot-service.pid"
    try:
        print(f"Foreground HTTP service PID {os.getpid()}; settings: {args.config.resolve()}", flush=True)
        print(f"Requesting {args.host}:{args.port}; starts DISARMED. First valid setpoint starts control; no /arm request needed.", flush=True)
        if config.compliance_mode == "auto":
            print("Automatic mode selection: /target_joint_state -> joint impedance; /target_ee_pose -> Cartesian OSC.", flush=True)
        print("A new setpoint can restart after stop/timeout. Automatic recovery: "
              + ("disabled." if args.no_auto_recover else "one guarded attempt per fault, on a fresh valid setpoint; never replay the old target."), flush=True)
        print(f"Read-only dashboard: http://127.0.0.1:{args.port}/ (or this host's LAN address).", flush=True)
        print("Authentication DISABLED, matching franky_service. Use a trusted/isolated robot network.", flush=True)
        print("One robot controller at a time. Do not run a standalone hold alongside this service.", flush=True)
        run_http_server(app, host=args.host, port=args.port, pid_path=pid_path)
    except RuntimeError as error:
        parser.exit(1, f"Controller startup failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
