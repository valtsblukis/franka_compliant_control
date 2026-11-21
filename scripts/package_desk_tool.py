"""Validate and package a local tool configuration; no network or robot calls."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=root / "configs/desk/robotiq-2f85/endeffector-config.json")
parser.add_argument("--name", default="robotiq-2f85-nominal")
parser.add_argument("--output-dir", type=Path, default=root / "artifacts/desk")
args = parser.parse_args()
source = args.source.resolve()
if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.name):
    raise ValueError("Invalid output name")
config = json.loads(source.read_text())
required = {"id", "name", "color", "mass", "centerOfMass", "transformation", "inertia"}
if set(config) != required or not config["id"] or config["id"].startswith("ee-"):
    raise ValueError("Invalid custom-device schema")
if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in config["id"]):
    raise ValueError("Invalid device identifier")
mass = config["mass"]
com = np.asarray(config["centerOfMass"], dtype=float)
transform = np.asarray(config["transformation"], dtype=float).reshape(4, 4, order="F")
inertia = np.asarray(config["inertia"], dtype=float).reshape(3, 3, order="F")
if not 0 < mass <= 3 or com.shape != (3,) or not com.dtype.kind == "f":
    raise ValueError("Invalid mass or COM")
if not all(np.isfinite(v).all() for v in (com, transform, inertia)):
    raise ValueError("Nonfinite mechanical values")
r = transform[:3, :3]
if not (np.allclose(transform[3], [0, 0, 0, 1]) and
        np.allclose(r.T @ r, np.eye(3)) and np.isclose(np.linalg.det(r), 1)):
    raise ValueError("Invalid rigid transform")
eigenvalues = np.linalg.eigvalsh(inertia)
if not (np.allclose(inertia, inertia.T) and eigenvalues[0] > 0 and
        eigenvalues[-1] <= sum(eigenvalues[:2])):
    raise ValueError("Invalid COM inertia")
payload = (json.dumps(config, indent=2, allow_nan=False) + "\n").encode()
target = args.output_dir.resolve()
target.mkdir(parents=True, exist_ok=True)
archive = target / f"{args.name}.tar.gz"
with archive.open("wb") as output:
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            info = tarfile.TarInfo("endeffector-config.json")
            info.size, info.mode, info.mtime = len(payload), 0o644, 0
            tar.addfile(info, io.BytesIO(payload))
with tarfile.open(archive, "r:gz") as tar:
    members = tar.getmembers()
    if len(members) != 1 or not members[0].isfile() or members[0].name != "endeffector-config.json":
        raise ValueError("Unexpected archive contents")
    if members[0].size >= 20_000_000 or json.load(tar.extractfile(members[0])) != config:
        raise ValueError("Archive verification failed")
legacy = {key: config[key] for key in ("mass", "centerOfMass", "transformation", "inertia")}
raw = target / ("robotiq-2f85-parameters.json" if args.name == "robotiq-2f85-nominal"
                else f"{args.name}-parameters.json")
raw.write_text(json.dumps(legacy, indent=2, allow_nan=False) + "\n")
(target / "README.md").write_text((source.parent / "README.md").read_text().replace(
    "../../../docs/wrist-load.md", os.path.relpath(root / "docs/controller.md", target)))
for path in (archive, raw):
    print(f"{path.relative_to(root)}  sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
print("Archive structure, rigid transform and physical inertia checks passed; no robot changes.")
