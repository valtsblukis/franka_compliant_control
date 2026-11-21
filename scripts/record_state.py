"""Record HTTP telemetry without sending any robot commands."""
import argparse
import json
import pathlib
import time
from franka_compliant_control.client import CompliantClient
from franka_compliant_control.network import DEFAULT_URL

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default=DEFAULT_URL)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--duration", type=float, default=30)
parser.add_argument("--hz", type=float, default=50)
args = parser.parse_args()
if args.duration <= 0 or not 0 < args.hz <= 200:
    parser.error("duration must be positive; hz must be in (0, 200]")
with CompliantClient(args.url) as client:
    with args.output.open("x") as destination:
        start = time.monotonic()
        index = 0
        while time.monotonic() - start < args.duration:
            state = client.state()
            state["recorded_unix_s"] = time.time()
            destination.write(json.dumps(state, allow_nan=False) + "\n")
            index += 1
            time.sleep(max(0, start + index / args.hz - time.monotonic()))
print(f"Recorded {index} samples to {args.output}")
