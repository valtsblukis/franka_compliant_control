"""Validate SDK kinematics offline against a recorded read_robot observation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
observation = Path(sys.argv[1])
state = json.loads(observation.read_text())["state"]
prefix = root / "vendor/libfranka-rt-prefix"
output = root / "build-robot/model_test"
output.parent.mkdir(exist_ok=True)
subprocess.run([
    "c++", "-std=c++17", "-O2", str(root / "tests/libfranka_model_test.cpp"),
    "-I", str(prefix / "include"), "-I", str(root / ".venv/lib/python3.12/site-packages/cmeel.prefix/include/eigen3"),
    "-L", str(prefix / "lib"), f"-Wl,-rpath,{prefix}/lib",
    f"-Wl,-rpath-link,{root}/vendor/sdk-root/usr/lib/x86_64-linux-gnu",
    f"-Wl,-rpath-link,{root}/vendor/sdk-deps/cmeel.prefix/lib",
    "-lfranka", "-o", str(output),
], check=True)
with tempfile.NamedTemporaryFile(mode="w") as fixture:
    fixture.write(" ".join(map(str, state["q"] + state["F_T_EE"] + state["O_T_EE"])))
    fixture.flush()
    subprocess.run([str(output), str(observation) + ".urdf", fixture.name], check=True)
