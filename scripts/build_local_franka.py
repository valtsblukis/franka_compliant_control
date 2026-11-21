"""Build the locally patched 0.18.0 SDK into vendor/libfranka-rt-prefix.

Dependency preparation and patch provenance: docs/build.md.
No system installation, robot connection or automatic dependency downloads.
"""
from pathlib import Path
import subprocess
import sys
import sysconfig

root = Path(__file__).resolve().parents[1]
cmake = str(root / ".venv/bin/cmake")
prefix = root / "vendor/libfranka-rt-prefix"
deps = root / "vendor/sdk-deps/cmeel.prefix"
system_deps = root / "vendor/sdk-root/usr"
build = root / "build-libfranka-rt"
python_site = Path(sysconfig.get_paths()["purelib"])
subprocess.run([
    cmake, "-S", str(root / "vendor/libfranka-0.18.0"), "-B", str(build), "-G", "Ninja",
    f"-DCMAKE_MAKE_PROGRAM={root}/.venv/bin/ninja", "-DBUILD_TESTS=OFF", "-DBUILD_EXAMPLES=OFF",
    "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}",
    f"-DPYTHON_EXECUTABLE={sys.executable}", f"-DPython_EXECUTABLE={sys.executable}",
    f"-DCMAKE_PREFIX_PATH={deps};{system_deps};{root}/vendor/sdk-extra/cmeel.prefix;{python_site}/cmeel.prefix",
    f"-DCMAKE_INSTALL_RPATH={prefix}/lib;{deps}/lib;{system_deps}/lib/x86_64-linux-gnu",
    "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--disable-new-dtags", "-DCMAKE_POLICY_VERSION_MINIMUM=3.10",
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
], check=True)
subprocess.run([cmake, "--build", str(build), "--parallel", "2"], check=True)
subprocess.run(["c++", "-std=c++17", "-pthread", "-I", str(root / "vendor/libfranka-0.18.0/include"),
                str(root / "tests/libfranka_realtime_test.cpp"), "-o", str(build / "realtime_test")], check=True)
subprocess.run([str(build / "realtime_test")], check=True)
subprocess.run([cmake, "--install", str(build)], check=True)
