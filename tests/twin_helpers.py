"""Shared offline fixtures; never construct a robot Backend."""

import json
from pathlib import Path

import numpy as np

from franka_compliant_control.models import ControllerConfig, HardwareSettings
from franka_compliant_control.twin_experiment import digest


def home_fixture():
    from franka_compliant_control import _native
    from franka_compliant_control.twin_model import matrix_quaternion
    from franka_compliant_control.simulation.presets import reference_context

    hardware = HardwareSettings.model_validate_json(
        Path("configs/hardware.wrist-load.estimated.json").read_text()
    ).model_dump()
    urdf = Path("tests/fixtures/fr3-twin.urdf").read_text()
    q = [0, -0.4, 0, -1.9, 0, 1.5, 0]
    if hasattr(_native, "OfflineKinematics"):
        pose, _ = _native.OfflineKinematics(urdf, hardware["flange_to_tcp"]).evaluate(q)
        position, quaternion = pose[:3, 3], matrix_quaternion(pose[:3, :3])
    else:
        # Collaborator builds intentionally disable the physical _native backend.
        # The separately built twin still provides the offline SDK model.
        from franka_compliant_control.simulation.sdk import offline_model

        sample = offline_model({"urdf": urdf, "hardware_settings": hardware}).sample(
            q, np.zeros(7), np.zeros(7), np.zeros(6)
        )
        position, quaternion = sample.position, sample.quaternion_xyzw
    home = {
        "schema": "fcc-twin-home-v1",
        "captured_utc": "offline-fixture",
        "home": {
            "position": position.tolist(),
            "quaternion_xyzw": quaternion.tolist(),
            "joints": q,
        },
        # Keep the offline baseline stable when the live policy profile is tuned.
        # Enabled policy defaults are checked separately in test_auto_mode.py;
        # this fixture's pure-spring/model tests assume the recorded off profile.
        "controller_config": ControllerConfig.model_validate(
            reference_context()["controller_config"]
        ).model_dump(),
        "hardware_settings": hardware,
        "urdf": urdf,
    }
    home["sha256"] = {
        key: digest(home[key])
        for key in ("home", "controller_config", "hardware_settings", "urdf")
    }
    return home
