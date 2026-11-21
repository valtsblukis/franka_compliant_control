"""Model boundaries must hold before connecting, on updates and in native control."""
import pytest
from fastapi.testclient import TestClient

from franka_compliant_control import _native
from franka_compliant_control.models import ControllerConfig, HardwareSettings, FR3_LOWER, FR3_UPPER
from franka_compliant_control.service import create_app


@pytest.mark.parametrize("model", ["panda", "fr3"])
def test_profile_defaults_match_native(model):
    config = ControllerConfig(robot_model=model)
    native = _native.Config(model)
    native.validate()
    for name, expected in config.model_dump().items():
        actual = getattr(native, name)
        assert (list(actual) if isinstance(expected, list) else actual) == expected


def test_fr3_bounds_reject_panda_and_allow_narrowing():
    fr3 = ControllerConfig(robot_model="fr3")
    assert fr3.joint_lower == FR3_LOWER
    assert fr3.joint_upper == FR3_UPPER
    panda = ControllerConfig()
    with pytest.raises(ValueError, match="model profile"):
        ControllerConfig(robot_model="fr3", joint_lower=panda.joint_lower, joint_upper=panda.joint_upper)
    narrower = fr3.model_dump()
    narrower["joint_lower"][5] = 1.0
    assert ControllerConfig(**narrower).joint_lower[5] == 1.0
    native = fr3.native()
    native.joint_lower = panda.joint_lower
    with pytest.raises(ValueError, match="model profile"):
        native.validate()


@pytest.mark.parametrize("change", [
    {"joint_velocity": 0.7}, {"joint_acceleration": 11}, {"joint_jerk": 5001},
    {"torque_limits": [88]*7}, {"torque_rate_limit": 1001},
    {"measured_joint_velocity_limit": 1.1},
    {"measured_joint_velocity_limit": .1},  # Below the default reference speed.
    {"measured_translation_velocity_limit": -.1},
    {"measured_rotation_velocity_limit": float("inf")},
])
def test_fr3_rejects_excessive_limits(change):
    with pytest.raises(ValueError):
        ControllerConfig(robot_model="fr3", **change)


def test_explicit_measured_speed_limits_roundtrip_without_faster_reference():
    from pathlib import Path
    config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.perturbation.json").read_text())
    native = config.native()
    assert native.translation_velocity == .2
    assert native.measured_translation_velocity_limit == .5
    assert native.measured_rotation_velocity_limit == 1.5
    assert native.measured_joint_velocity_limit == .8
    assert list(native.measured_joint_velocity_limits) == [1.5,.9,1.2,1.1,2,1.3,2]
    for field, value in [("measured_joint_velocity_limits", [1.5,1.1,1.2,1.1,2,1.3,2]),
                         ("measured_joint_velocity_limits", [1.5,.9,1.2,1.1,2,1.3,.1]),
                         ("measured_joint_velocity_limits", [-1,0,0,0,0,0,0]),
                         ("measured_joint_velocity_limits", [float("nan"),0,0,0,0,0,0]),
                         ("measured_translation_velocity_limit", -.1),
                         ("measured_rotation_velocity_limit", float("nan"))]:
        invalid = config.native()
        setattr(invalid, field, value)
        with pytest.raises(ValueError):
            invalid.validate()


@pytest.mark.parametrize("limits", [
    [0]*6, [0]*8, [-1,0,0,0,0,0,0], [float("inf"),0,0,0,0,0,0],
    [0,1.01,0,0,0,0,0], [0,0,0,0,0,1.51,0], [0,0,0,0,0,0,.1],
])
def test_per_joint_speed_overrides_reject_invalid_or_out_of_model_values(limits):
    with pytest.raises(ValueError):
        ControllerConfig(robot_model="fr3", measured_joint_velocity_limits=limits)


def test_model_mismatch_rejected_before_robot_construction():
    hardware = HardwareSettings(robot_ip="must-never-connect", robot_model="fr3",
        flange_to_tcp=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1], load_mass=0,
        load_com=[0]*3, load_inertia=[0]*9, calibration_confirmed=True)
    with pytest.raises(ValueError, match="robot_model must match"):
        create_app(config=ControllerConfig(), backend_kind="robot", hardware=hardware, api_token="a"*24)
    with pytest.raises(ValueError, match="robot_model must match"):
        _native.Backend(_native.Config(), "robot", hardware.native())


def test_fr3_update_cannot_switch_model_or_restore_panda_defaults():
    with TestClient(create_app(config=ControllerConfig(robot_model="fr3"))) as client:
        assert client.get("/health").json()["robot_model"] == "fr3"
        assert client.put("/controller_config", json={"commanded_force_limit": 5}).status_code == 422
        assert client.put("/controller_config", json=ControllerConfig().model_dump()).status_code == 422
        response = client.put("/controller_config", json={"robot_model": "fr3", "commanded_force_limit": 5})
        assert response.status_code == 200
        assert response.json()["joint_lower"] == FR3_LOWER
        pose = client.get("/state").json()["position"]
        initial = client.post("/target_ee_pose", json={"position": pose, "seq": 0})
        assert initial.status_code == 202
        assert client.get("/state").json()["robot_model"] == "fr3"
        invalid = {"seq": 1, "deadline_s": _native.monotonic_time() + .1,
                   "positions": [0,-.4,0,-1.9,0,.2,0]}
        assert client.post("/target_joint_state", json=invalid).status_code == 422
        invalid["position"] = pose
        invalid["posture"] = invalid.pop("positions")
        assert client.post("/target_ee_pose", json=invalid).status_code == 422
        assert client.post("/stop", params={"wait_s": 1}).json()["stopped"]


def test_fr3_native_update_cannot_change_model():
    backend = _native.Backend(_native.Config("fr3"), "sim", _native.HardwareConfig())
    try:
        with pytest.raises(ValueError, match="cannot change"):
            backend.configure(_native.Config("panda"))
        config = _native.Config("fr3")
        config.joint_lower = [0.1, *list(config.joint_lower)[1:]]
        backend.configure(config)
        with pytest.raises(RuntimeError, match="initial state.*joint 1 is.*strictly between"):
            backend.arm()
        assert not backend.state()["running"]
    finally:
        backend.close()


@pytest.mark.parametrize("change", [
    {"motion_limit_policy": "unchecked"}, {"robot_model": "panda"},
    {"joint_lower": [-2.901, -1.8361, -2.9007, -3.0770, -2.8763, .4398, -3.0508]},
    {"measured_joint_velocity_limits": [2.63, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]},
    {"measured_translation_velocity_limit": 3.01},
    {"measured_rotation_velocity_limit": 2.51}, {"joint_limit_margin": -.01},
])
def test_full_fr3_envelope_still_rejects_invalid_limits(change):
    from pathlib import Path
    import json
    values = json.loads(Path("configs/controller.fr3.impedance.json").read_text())
    with pytest.raises(ValueError):
        ControllerConfig(**(values | change))
