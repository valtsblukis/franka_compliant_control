"""Build only within this project, using the invoking Python environment."""
import argparse
import pathlib
import subprocess
import sys
import sysconfig

root = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
sdk = parser.add_mutually_exclusive_group()
sdk.add_argument("--franka-prefix", help="Prefix containing a matching libfranka installation")
sdk.add_argument("--franka-wheel-source", help="Matching libfranka source tree (including common submodule); link the runtime from this venv's franky-control wheel")
parser.add_argument("--build-type", default="RelWithDebInfo")
parser.add_argument("--dependency-prefix", action="append", default=[], help="Additional isolated SDK dependency prefix (repeatable)")
args = parser.parse_args()
site = pathlib.Path(sysconfig.get_paths()["purelib"])
bin_dir = pathlib.Path(sys.executable).parent
hardware = bool(args.franka_prefix or args.franka_wheel_source)
build = root / ("build-robot" if hardware else "build")
prefixes = [str(site / "pybind11/share/cmake/pybind11"), str(site / "cmeel.prefix")]
prefixes.extend(str(pathlib.Path(p).resolve()) for p in args.dependency_prefix)
if args.franka_prefix:
    prefixes.append(str(pathlib.Path(args.franka_prefix).resolve()))
extra = []
if args.franka_prefix:
    # A previous wheel build must not silently override the requested prefix.
    extra = ["-UFCC_FRANKA_SOURCE", "-UFCC_FRANKA_LIBRARY", "-UFranka_DIR"]
if args.franka_wheel_source:
    libraries = list((site / "franky_control.libs").glob("libfranka-*.so.*"))
    if len(libraries) != 1:
        parser.error("Install a matching franky-control wheel into this venv first")
    extra = [f"-DFCC_FRANKA_SOURCE={pathlib.Path(args.franka_wheel_source).resolve()}",
             f"-DFCC_FRANKA_LIBRARY={libraries[0]}"]
    # Reuse the already downloaded, hash-checked trajectory source.
    cached_ruckig = root / "build/_deps/ruckig-src"
    if cached_ruckig.is_dir():
        extra.append(f"-DFETCHCONTENT_SOURCE_DIR_RUCKIG={cached_ruckig}")
subprocess.run([
    str(bin_dir / "cmake"), "-S", str(root), "-B", str(build), "-G", "Ninja",
    f"-DCMAKE_MAKE_PROGRAM={bin_dir / 'ninja'}",
    f"-DPython_EXECUTABLE={sys.executable}",
    f"-DCMAKE_PREFIX_PATH={';'.join(prefixes)}",
    f"-DCMAKE_BUILD_TYPE={args.build_type}",
    f"-DFCC_WITH_FRANKA={'ON' if hardware else 'OFF'}", *extra,
], check=True)
subprocess.run([str(bin_dir / "cmake"), "--build", str(build), "--parallel", "2"], check=True)
subprocess.run([str(bin_dir / "ctest"), "--test-dir", str(build), "--output-on-failure"], check=True)
