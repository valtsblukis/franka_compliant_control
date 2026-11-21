"""New Python with an installed pre-compensation native module; no robot I/O."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from franka_compliant_control import _native
from franka_compliant_control.models import ControllerConfig


FRICTION_FIELDS = {"friction_compensation_torque", "friction_compensation_epsilon",
                   "friction_compensation_filter_time_s"}


def emulate_legacy_config(monkeypatch, missing=FRICTION_FIELDS):
    """Hide new fields even when tests run against the rebuilt native module."""
    native_config = _native.Config

    class LegacyConfig:
        __slots__ = ("inner",)

        def __init__(self, model):
            object.__setattr__(self, "inner", native_config(model))

        def __getattr__(self, name):
            if name in missing:
                raise AttributeError(name)
            return getattr(self.inner, name)

        def __setattr__(self, name, value):
            if name in missing:
                raise AttributeError(name)
            setattr(self.inner, name, value)

    monkeypatch.setattr(_native, "Config", LegacyConfig)


def test_legacy_native_accepts_off_without_losing_other_settings(monkeypatch):
    emulate_legacy_config(monkeypatch)
    for path in Path("configs").glob("controller*.json"):
        # Compatibility is deliberately off-only; the policy startup profile
        # now explicitly enables compensation and needs the matching new binary.
        values = json.loads(path.read_text()) | {"friction_compensation_torque": 0.0}
        config = ControllerConfig.model_validate(values)
        native = config.native()
        assert config.friction_compensation_torque == 0
        assert not hasattr(native, "friction_compensation_torque")
        for name, expected in config.model_dump().items():
            if name not in FRICTION_FIELDS:
                actual = getattr(native, name)
                if isinstance(expected, list):
                    np.testing.assert_array_equal(actual, expected)
                else:
                    assert actual == expected


@pytest.mark.parametrize("f,epsilon", [(.3, .05), (.6, .03), (1., .02)])
def test_legacy_native_rejects_enabling_compensation(monkeypatch, f, epsilon):
    emulate_legacy_config(monkeypatch)
    with pytest.raises(ValueError, match="requires a rebuilt native module"):
        ControllerConfig(friction_compensation_torque=f, friction_compensation_epsilon=epsilon)


@pytest.mark.parametrize("f,filter_time,accepted", [(0.,.035,True),(.6,0.,True),(.6,.035,False)])
def test_native_without_filter_never_silently_ignores_enabled_filter(monkeypatch, f, filter_time, accepted):
    emulate_legacy_config(monkeypatch, {"friction_compensation_filter_time_s"})
    if accepted:
        config = ControllerConfig(friction_compensation_torque=f,
                                  friction_compensation_filter_time_s=filter_time)
        assert config.native().friction_compensation_torque == f
    else:
        with pytest.raises(ValueError, match="Filtered friction compensation requires a rebuilt native module"):
            ControllerConfig(friction_compensation_torque=f,
                             friction_compensation_filter_time_s=filter_time)


def test_compatibility_does_not_skip_unrelated_missing_native_fields(monkeypatch):
    emulate_legacy_config(monkeypatch, FRICTION_FIELDS | {"osc_regularization"})
    with pytest.raises(AttributeError, match="osc_regularization"):
        ControllerConfig()


@pytest.mark.parametrize("missing", [FRICTION_FIELDS, {"friction_compensation_filter_time_s"}])
def test_enabled_policy_profile_rejects_stale_native(monkeypatch, missing):
    emulate_legacy_config(monkeypatch, missing)
    with pytest.raises(ValueError, match="requires a rebuilt native module"):
        ControllerConfig.model_validate_json(Path("configs/controller.fr3.auto.json").read_text())


def test_robot_entrypoint_accepts_explicit_off_profile_with_legacy_native(monkeypatch, tmp_path):
    emulate_legacy_config(monkeypatch)
    spec = importlib.util.spec_from_file_location("run_robot_compatibility", "scripts/run_robot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hardware = json.loads(Path("configs/hardware.template.json").read_text())
    hardware["read_only"] = True
    hardware_path = tmp_path / "hardware.json"
    hardware_path.write_text(json.dumps(hardware))
    profile = json.loads(Path("configs/controller.fr3.auto.json").read_text())
    profile["friction_compensation_torque"] = 0.0
    profile_path = tmp_path / "controller-off.json"
    profile_path.write_text(json.dumps(profile))
    app = object()
    calls = []

    def fake_create_app(**kwargs):
        assert kwargs["config"].friction_compensation_torque == 0
        assert kwargs["config"].compliance_mode == "auto"
        assert kwargs["hardware"].read_only
        assert not kwargs["automatic_recovery"]
        calls.append("construct_app")
        return app

    def fake_run_server(application, **kwargs):
        assert application is app
        calls.append("run_server")

    # Intercept before any Backend/Robot construction, server or network access.
    monkeypatch.setattr(module, "create_app", fake_create_app)
    monkeypatch.setattr(module, "run_http_server", fake_run_server)
    assert module.main(["--config", str(profile_path),
                        "--hardware-config", str(hardware_path), "--no-auto-recover"]) == 0
    assert calls == ["construct_app", "run_server"]
