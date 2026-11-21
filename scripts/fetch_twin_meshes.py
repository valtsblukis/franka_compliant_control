#!/usr/bin/env python3
"""Optional pinned official FR3 visual assets; no robot I/O or model changes."""
import argparse
import hashlib
import json
from pathlib import Path
import requests

REVISION = "7aeeddc449edf8d62b594f9e36a81da53e7796f9"
BASE = f"https://raw.githubusercontent.com/frankarobotics/franka_description/{REVISION}/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"vendor/fr3-twin-meshes")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for relative in [f"meshes/robots/fr3/collision/link{i}.stl" for i in range(8)]+["LICENSE"]:
        result = requests.get(BASE+relative, timeout=30)
        result.raise_for_status()
        name = Path(relative).name
        target = args.output/name
        if target.exists() and target.read_bytes() != result.content:
            raise RuntimeError(f"Refusing to overwrite different asset: {target}")
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(result.content)
        records.append({"file":name,"url":BASE+relative,
                        "sha256":hashlib.sha256(result.content).hexdigest(),"bytes":len(result.content)})
    (args.output/"manifest.json").write_text(json.dumps({"repository":"frankarobotics/franka_description",
        "revision":REVISION,"purpose":"visual-only; not validated collision geometry","files":records},indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
