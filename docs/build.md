# Native builds

Use a fresh checkout/environment. `pip install -e '.[simulation,dev]'` installs
Python/build dependencies; it does **not** compile the native modules or install
libfranka. Pure MuJoCo physics needs neither native module nor SDK.

```sh
# Fresh offline workstation; no physical robot backend:
python scripts/build_simulation.py --franka-prefix /path/to/libfranka-prefix \
  --dependency-prefix /path/to/sdk-dependencies
```

The dependency-prefix option is repeatable. CMake fetches hash-pinned Ruckig 0.15.3;
`--ruckig-source` accepts separately audited sources on disconnected hosts. Tested
SDK: libfranka 0.18.0 with the flange-frame/realtime-detection patches in `patches/`.
SDK/transitive libraries are separately installed and must be visible to the
linker. The SDK build helper has origin-host vendor defaults, not bundled binaries.

`build_simulation.py` refuses existing native libraries and builds `_native` with
robot support OFF, then `_twin_native` with the URDF-only SDK model. The latter
compiles the actual production controller, not a second Python control law.
For an existing installation, `build_twin.py` rebuilds **only** the offline bridge;
physical builds use `build_native.py` with an explicit SDK prefix. Do not use these
as unattended deployment commands on a robot host.

The simulator checks its native source fingerprint. Editable installs use the
checkout's headers/sources; locally built wheels use CMake's packaged fingerprint.
Missing/stale build metadata fails closed. Rebuild matching sources, never edit
an old model/policy hash to bypass a mismatch.

```sh
python -m pytest -q
# After building native test targets in the chosen build directory:
ctest --test-dir build-offline-types --output-on-failure
```

The SDK-free build has 11 native suites; SDK-backed builds add model/diagnostic
checks. Tests use offline fixtures and recording clients, not robot connections.

No general project redistribution license is granted by this repository. Existing
third-party notices are preserved; SDKs, meshes and dependencies retain their own
licenses. Agree owner/collaborator terms before public distribution.
