#!/usr/bin/env python3
"""Build ONLY the offline bridge; never overwrite the active hardware extension."""
import argparse
from pathlib import Path
import subprocess
import sys
import sysconfig


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--franka-prefix",type=Path,default=root/"vendor/libfranka-rt-prefix")
    parser.add_argument("--dependency-prefix",type=Path,action="append",default=[])
    args = parser.parse_args()
    if not (args.franka_prefix/"lib/cmake/Franka/FrankaConfig.cmake").is_file():
        parser.error("Provide an isolated libfranka >=0.18 prefix with --franka-prefix")
    binary = Path(sys.executable).parent
    site = Path(sysconfig.get_paths()["purelib"])
    prefixes = [site/"pybind11/share/cmake/pybind11",site/"cmeel.prefix",
                root/"vendor/sdk-root/usr",args.franka_prefix,*args.dependency_prefix]
    command = [str(binary/"cmake"),"-S",str(root),"-B",str(root/"build-twin"),"-G","Ninja",
               f"-DCMAKE_MAKE_PROGRAM={binary/'ninja'}",f"-DPython_EXECUTABLE={sys.executable}",
               "-DCMAKE_BUILD_TYPE=RelWithDebInfo","-DFCC_WITH_FRANKA=ON","-DFCC_BUILD_TWIN=ON",
               "-DCMAKE_PREFIX_PATH="+";".join(str(p.resolve()) for p in prefixes)]
    cached = root/"build/_deps/ruckig-src"
    if cached.is_dir(): command.append(f"-DFETCHCONTENT_SOURCE_DIR_RUCKIG={cached}")
    subprocess.run(command,check=True)
    subprocess.run([str(binary/"cmake"),"--build",str(root/"build-twin"),"--target","_twin_native","--parallel","2"],check=True)


if __name__ == "__main__":
    main()
