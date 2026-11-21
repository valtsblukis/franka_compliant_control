"""Exercise the HTTP client against the mock backend only."""
import argparse
import math
import time
from franka_compliant_control.client import CompliantClient
from franka_compliant_control.network import DEFAULT_URL

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default=DEFAULT_URL)
args = parser.parse_args()
with CompliantClient(args.url) as client:
    initial = client.state()
    if initial["backend"] != "sim":
        raise SystemExit("This example is only for the mock backend.")
    start = time.monotonic()
    for i in range(150):
        position = list(initial["position"])
        position[0] += .01 * (1-math.cos(math.pi*min(i/100, 1)))
        client.send_pose(position, initial["quaternion_xyzw"])
        time.sleep(max(0, start+(i+1)*.02-time.monotonic()))
    final = client.state()
    print({"initial_position": initial["position"], "final_position": final["position"],
           "reference_position": final["reference_position"], "phase": final["phase"],
           "controller_compute_s": final["controller_compute_s"]})
