"""Inspect HTTP status without arming, moving or recovering the robot."""
import argparse
import json
import time

import requests

from franka_compliant_control.status import terminal_status
from franka_compliant_control.network import DEFAULT_URL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--watch", action="store_true", help="Poll cached status once per second")
    parser.add_argument("--refresh", action="store_true", help="Request ONE inactive robot read before displaying; never auto-recover")
    parser.add_argument("--json", action="store_true", help="Print full diagnostic JSON instead of a summary")
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")
    with requests.Session() as http:
        if args.refresh:
            response = http.post(base+"/state/refresh", timeout=5)
            response.raise_for_status()
        try:
            while True:
                response = http.get(base+"/status", timeout=5)
                response.raise_for_status()
                view = response.json()
                if args.json:
                    print(json.dumps(view, indent=2), flush=True)
                else:
                    print(terminal_status(view), flush=True)
                    if not args.watch:
                        for notice in view["notices"]:
                            print("- "+notice)
                        rejected = view.get("last_rejection")
                        if rejected:
                            print(f"Last rejected request: {rejected['operation']}: {rejected['error']}")
                if not args.watch:
                    return 0
                time.sleep(1)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
