"""Request a two-second hardware hold at the measured pose, then stop and record results."""
import argparse
import json
from pathlib import Path
import time

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.commissioning import hold_current_pose

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hold-current-pose", action="store_true",
                    help="Start torque control; requires a clear workspace and operator at the physical stop")
parser.add_argument("--output", type=Path, required=True, help="New JSONL log file")
args = parser.parse_args()
if not args.hold_current_pose:
    parser.error("--hold-current-pose is required to start this powered test")
with args.output.open("x") as log:
    def record(event):
        log.write(json.dumps({"time_monotonic_s": time.monotonic(), **event}, allow_nan=False) + "\n")
        log.flush()
    with CompliantClient(timeout_s=5) as client:
        result = hold_current_pose(client, record)
    record({"event": "summary", **result})
    print(json.dumps(result, indent=2))
