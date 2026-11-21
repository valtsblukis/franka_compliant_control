"""Compare configured return-reference times offline; never connects to a robot.

This measures the trajectory generator with ideal following, not physical robot
settling, tracking accuracy, latency under load, or safety.
"""
import argparse
import json
from pathlib import Path
import subprocess

from franka_compliant_control.models import ControllerConfig


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [f"{kind}_{limit}" for kind in ("translation", "rotation", "joint")
          for limit in ("velocity", "acceleration", "jerk")]
FIELDS += [f"measured_{kind}_velocity_limit" for kind in ("translation", "rotation", "joint")]


def benchmark(path):
    config = ControllerConfig.model_validate_json(path.read_text())
    if config.compliance_mode != "contact_yielding":
        raise ValueError("This yielding-release benchmark only applies to contact_yielding; continuous impedance retains its reference during contact")
    executable = ROOT / "build-robot/reference_benchmark"
    if not executable.exists():
        executable = ROOT / "build/reference_benchmark"
    if not executable.exists():
        raise RuntimeError("Build the native project first; reference_benchmark is missing")
    result = subprocess.run([str(executable), *[str(getattr(config, name)) for name in FIELDS],
                             *map(str, config.measured_joint_velocity_limits)],
                            capture_output=True, text=True, timeout=10, check=True)
    return {"config": str(path), **json.loads(result.stdout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, help="Optional new JSON report; refuses overwrite")
    args = parser.parse_args()
    paths = args.configs or [ROOT / "configs/archive/controller.fr3.perturbation-20260909-v2.json",
                            ROOT / "configs/controller.fr3.perturbation.json"]
    report = {"scope": "Reference generation with ideal following; not measured physical settling",
              "results": [benchmark(path) for path in paths]}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as out:
            out.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
