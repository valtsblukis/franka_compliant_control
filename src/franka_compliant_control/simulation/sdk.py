"""Robot-specific SDK model/controller provenance, without experiment or device I/O."""

import hashlib
from pathlib import Path
from ..twin_model import combined_load

ROOT = Path(__file__).resolve().parents[3]
TWIN_INPUTS = (
    "controller.cpp",
    "controller.hpp",
    "fr3_limits.hpp",
    "operational_space.hpp",
    "progressive_spring.hpp",
    "joint_impedance.hpp",
    "joint_limit_backoff.hpp",
    "model_dynamics.hpp",
    "twin_bindings.cpp",
)


def native_inputs_digest():
    if not (ROOT/'native/controller.cpp').is_file():
        # A locally built wheel has no sibling C++ checkout. CMake records the
        # same source fingerprint in package data; absence still fails closed.
        import json
        from importlib.resources import files
        try:
            return json.loads(files('franka_compliant_control').joinpath('_native_inputs.json').read_text())['twin_inputs_sha256']
        except (OSError, KeyError, ValueError) as error:
            raise RuntimeError('Build the offline bridge with its source manifest') from error
    fingerprints = "".join(
        f'{name}:{hashlib.sha256((ROOT/"native"/name).read_bytes()).hexdigest()}\n'
        for name in TWIN_INPUTS
    )
    return hashlib.sha256(fingerprints.encode()).hexdigest()


class _ModelKinematics:
    """FK/J facade when the types-only _native intentionally has no robot SDK.

    Uses the same URDF-only SDK model as controller simulation, not a second IK law.
    """

    def __init__(self, home):
        self.model = offline_model(home)

    def evaluate(self, q):
        import numpy as np
        from ..twin_model import quaternion_matrix

        sample = self.model.sample(q, np.zeros(7), np.zeros(7), np.zeros(6))
        pose = np.eye(4)
        pose[:3, :3] = quaternion_matrix(sample.quaternion_xyzw)
        pose[:3, 3] = sample.position
        return pose, sample.jacobian


def offline_kinematics(home):
    from .. import _native

    if hasattr(_native, "OfflineKinematics"):
        return _native.OfflineKinematics(
            home["urdf"], home["hardware_settings"]["flange_to_tcp"]
        )
    return _ModelKinematics(home)


def offline_model(home):
    from .. import _twin_native

    load = combined_load(home["hardware_settings"])
    return _twin_native.OfflineModel(
        home["urdf"],
        home["hardware_settings"]["flange_to_tcp"],
        load["mass"],
        load["com"],
        load["inertia"],
    )
