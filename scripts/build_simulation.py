#!/usr/bin/env python3
"""Build offline modules in a fresh checkout, never a loaded robot tree.

_native is built without robot support. _twin_native links the offline SDK model.
No service or robot is started. Existing native modules are a hard refusal.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import sysconfig


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--franka-prefix", type=Path, required=True)
    p.add_argument("--dependency-prefix", type=Path, action="append", default=[])
    p.add_argument(
        "--ruckig-source", type=Path, help="Optional pre-fetched Ruckig 0.15.3 source"
    )
    a = p.parse_args()
    destination = root / "src/franka_compliant_control"
    if list(destination.glob("_native*.so")) or list(
        destination.glob("_twin_native*.so")
    ):
        p.error(
            "Existing native library: use a fresh checkout, never replace a loaded service binary"
        )
    prefix = a.franka_prefix.resolve()
    if not (prefix / "lib/cmake/Franka/FrankaConfig.cmake").is_file():
        p.error("Expected an installed libfranka >=0.18 SDK prefix (not bundled)")
    binary = Path(sys.executable).parent
    site = Path(sysconfig.get_paths()["purelib"])
    paths = [
        site / "pybind11/share/cmake/pybind11",
        site / "cmeel.prefix",
        prefix,
        *(x.resolve() for x in a.dependency_prefix),
    ]
    common = [
        "-G",
        "Ninja",
        f'-DCMAKE_MAKE_PROGRAM={binary/"ninja"}',
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DCMAKE_PREFIX_PATH=" + ";".join(str(x) for x in paths),
    ]
    cached = a.ruckig_source.resolve() if a.ruckig_source else None
    for build_name, flags, target in [
        (
            "build-offline-types",
            ["-DFCC_WITH_FRANKA=OFF", "-DFCC_BUILD_TWIN=OFF"],
            "_native",
        ),
        ("build-twin", ["-DFCC_WITH_FRANKA=ON", "-DFCC_BUILD_TWIN=ON"], "_twin_native"),
    ]:
        build = root / build_name
        extra = [f"-DFETCHCONTENT_SOURCE_DIR_RUCKIG={cached}"] if cached else []
        subprocess.run(
            [
                str(binary / "cmake"),
                "-S",
                str(root),
                "-B",
                str(build),
                *common,
                *flags,
                *extra,
            ],
            check=True,
        )
        subprocess.run(
            [
                str(binary / "cmake"),
                "--build",
                str(build),
                "--target",
                target,
                "--parallel",
                "2",
            ],
            check=True,
        )
        if cached is None:
            cached = build / "_deps/ruckig-src"
    print(
        "Offline modules built. _native has NO physical robot backend. No service started."
    )


if __name__ == "__main__":
    main()
