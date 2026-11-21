"""Config/API/offline bridge checks; never instantiate a physical backend."""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from franka_compliant_control import _native
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import create_app


SETTINGS = [(0.0, 0.05), (0.3, 0.05), (0.6, 0.03), (1.0, 0.02)]


@pytest.mark.parametrize("filter_time", [0.0, .035])
@pytest.mark.parametrize("f,epsilon", SETTINGS)
def test_config_roundtrip_and_inactive_http_update(f, epsilon, filter_time):
    config = ControllerConfig(
        friction_compensation_torque=f, friction_compensation_epsilon=epsilon,
        friction_compensation_filter_time_s=filter_time,
    )
    native = config.native()
    assert native.friction_compensation_torque == f
    assert native.friction_compensation_epsilon == epsilon
    assert native.friction_compensation_filter_time_s == filter_time
    with TestClient(create_app()) as client:
        response = client.put("/controller_config", json=config.model_dump())
        assert response.status_code == 200
        assert response.json() == config.model_dump()
        state = client.get("/state").json()
        assert not state["running"] and state["backend"] == "sim"
        assert state["friction_compensation_torques"] == [0.0] * 7
        assert state["planned_joint_velocities"] == [0.0] * 7


@pytest.mark.parametrize("field,value", [
    ("friction_compensation_torque", -0.1),
    ("friction_compensation_torque", 8.01),  # Exceeds default wrist torque cap.
    ("friction_compensation_torque", float("nan")),
    ("friction_compensation_torque", float("inf")),
    ("friction_compensation_epsilon", 0.0),
    ("friction_compensation_epsilon", -0.01),
    ("friction_compensation_epsilon", float("nan")),
    ("friction_compensation_epsilon", float("inf")),
    ("friction_compensation_filter_time_s", -0.01),
    ("friction_compensation_filter_time_s", float("nan")),
    ("friction_compensation_filter_time_s", float("inf")),
])
def test_invalid_settings_rejected_by_python_and_native(field, value):
    with pytest.raises(ValueError):
        ControllerConfig(**{field: value})
    native = _native.Config()
    setattr(native, field, value)
    with pytest.raises(ValueError):
        native.validate()


def test_only_policy_startup_profile_enables_compensation():
    assert ControllerConfig().friction_compensation_torque == 0.0
    for path in Path("configs").glob("controller*.json"):
        config = ControllerConfig.model_validate_json(path.read_text())
        if path.name == "controller.fr3.auto.json":
            assert config.friction_compensation_torque == 1.0
            assert config.friction_compensation_epsilon == .02
            assert config.friction_compensation_filter_time_s == .035
        else:
            assert config.friction_compensation_torque == 0.0, path


@pytest.mark.parametrize("filter_time", [0.0, .035])
@pytest.mark.parametrize("f,epsilon", SETTINGS)
def test_offline_bridge_uses_production_compensation(f, epsilon, filter_time):
    twin = pytest.importorskip("franka_compliant_control._twin_native")
    config = ControllerConfig.model_validate(
        json.loads(Path("configs/controller.fr3.auto.json").read_text()) | {
            "friction_compensation_torque": f,
            "friction_compensation_epsilon": epsilon,
            "friction_compensation_filter_time_s": filter_time,
            "minimum_singular_value": .017,
            "osc_regularization": .04,
            "jacobian_rotation_scale": .3,
        }
    )
    sample = twin.Sample()
    sample.q = [0, -.4, 0, -1.9, 0, 1.5, 0]
    sample.dynamics_valid = True
    # Coupled, non-axis-aligned Jacobian tests the full scaled inverse mapping.
    jacobian = np.random.default_rng(7).normal(0, .2, (6, 7))
    sample.jacobian = jacobian
    command = _native.Command()
    command.epoch = command.sequence = 1
    command.position = [.001, -.002, .003]
    command.has_orientation = True
    angle = .004
    command.quaternion_xyzw = [0, np.sin(angle/2), 0, np.cos(angle/2)]
    controller = twin.OfflineController(config.native(), sample, 1, 0.0)
    out = controller.step(sample, command, 0.0, .001, 1.0)
    assert out.phase == "active"
    scale = np.array([1.0]*3 + [config.jacobian_rotation_scale]*3)
    js = scale[:, None]*jacobian
    error = np.array([.001, -.002, .003, 0, angle, 0])
    planned = js.T @ np.linalg.solve(
        js @ js.T + config.minimum_singular_value**2*np.eye(6), scale*error/.1
    )
    np.testing.assert_allclose(out.planned_joint_velocity, planned if f else np.zeros(7), atol=1e-12)
    alpha = -np.expm1(-.001/filter_time) if filter_time else 1.0
    np.testing.assert_allclose(out.friction_compensation_torque, alpha*f*np.tanh(planned/epsilon), atol=1e-12)
    stopped = controller.step(sample, command, .001, .001, 1.0, True)
    assert stopped.phase == "stopping"
    np.testing.assert_array_equal(stopped.friction_compensation_torque, np.zeros(7))
