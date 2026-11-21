"""Stream Cartesian JSONL setpoints to the existing HTTP service; never starts one.

Each line is a dict accepted by CompliantClient.set_target_ee_pose: position
(3 values), optional quaternion_xyzw (4 values), and optional derivatives.
Positions are absolute base-frame TCP coordinates. Review the actual setpoints;
no home pose, interpolation, or retries are supplied by this example.
"""
import argparse
from itertools import chain
import json
import math
from pathlib import Path
import time

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.network import DEFAULT_URL
from franka_compliant_control.diagnostics import stop_failure_message


def stream_setpoints(robot, setpoints, *, hz=50.0):
    """First setpoint starts control. The producer owns freshness and pacing."""
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be finite and positive")
    points = iter(setpoints)
    first = next(points, None)  # No loop starts until a real setpoint is available.
    if first is None:
        return {"sent": 0, "armed": False}
    if not isinstance(first, dict) or "position" not in first:
        raise ValueError("First setpoint must be a Cartesian object with 'position'")
    health = robot.health()
    if health["running"] or health["phase"] != "disarmed" or health["read_only"]:
        raise RuntimeError("Need an exclusively owned, disarmed writable service; no recovery attempted")
    if health.get("cartesian_reference_policy", health.get("reference_policy")) != "direct_equilibrium":
        raise RuntimeError("Select the bounded OSC profile for direct policy setpoints")
    sent = 0
    try:
        next_tick = time.monotonic()
        for point in chain((first,), points):
            if not isinstance(point, dict) or "position" not in point:
                raise ValueError("Each setpoint must be an object with Cartesian 'position'; joint vectors are not pose vectors")
            robot.set_target_ee_pose(**point)
            sent += 1
            next_tick = max(next_tick+1/hz, time.monotonic())
            time.sleep(max(0, next_tick-time.monotonic()))
        return {"sent": sent, "armed": True}
    finally:
        result = robot.stop(wait_s=2.5)
        if not result["stopped"]:
            raise RuntimeError(stop_failure_message(result["state"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--setpoints", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=50)
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize this file's first setpoint to start control")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required: inspect the absolute setpoints and prepare the robot before sending")
    with args.setpoints.open() as source:
        points = (json.loads(line) for line in source if line.strip())
        with CompliantClient(args.url) as robot:
            result = stream_setpoints(robot, points, hz=args.hz)
    print(f"Sent {result['sent']} setpoints; stop confirmed; HTTP client closed."
          if result["armed"] else "No setpoints: no arm or stop sent; HTTP client closed.")


if __name__ == "__main__":
    main()
