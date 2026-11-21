"""Read saved contact logs and compare a candidate profile without a robot connection."""
import argparse
import json
from pathlib import Path

from franka_compliant_control.contact_analysis import analyze_log
from franka_compliant_control.models import ControllerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/controller.fr3.contact.json"))
    parser.add_argument("--output", type=Path, help="New JSON file; existing files are never overwritten")
    args = parser.parse_args()
    config = ControllerConfig.model_validate_json(args.config.read_text()).model_dump()
    report = {"candidate_config": config, "logs": [analyze_log(p, config) for p in args.logs],
              "physical_validation": False}
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as output:
            output.write(rendered)
        print(f"Offline observation comparison: {args.output}")
        for item in report["logs"]:
            print(f"{Path(item['log']).name}: peak {item['candidate_peak_threshold_fraction']:.1%} "
                  f"of candidate contact threshold; {item['candidate_threshold_crossing_observations']} crossings")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
