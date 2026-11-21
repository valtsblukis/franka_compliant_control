import pytest
from fastapi.testclient import TestClient

from franka_compliant_control import _native
from franka_compliant_control.models import HardwareSettings
from franka_compliant_control.service import create_app


def settings(**changes):
    return HardwareSettings(robot_ip="unused", robot_model="panda",
        flange_to_tcp=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
        load_mass=0, load_com=[0]*3, load_inertia=[0]*9, **changes)


def test_unconfirmed_settings_require_read_only():
    with pytest.raises(ValueError, match="read_only"):
        settings()
    assert settings(read_only=True).native().read_only
    assert not settings(calibration_confirmed=True).native().read_only


def test_read_only_native_blocks_arm_and_recovery():
    backend = _native.Backend(_native.Config(), "sim", settings(read_only=True).native())
    try:
        for action in (backend.arm, backend.recover):
            with pytest.raises(RuntimeError, match="read-only"):
                action()
        assert not backend.state()["running"]
        assert backend.state()["epoch"] == 0
    finally:
        backend.close()


def test_read_only_http_cannot_be_armed_by_config_or_recovery():
    with TestClient(create_app(hardware=settings(read_only=True))) as client:
        assert client.get("/health").json()["read_only"]
        assert client.get("/state").json()["wrench_point"] == "robot_stiffness_frame"
        assert client.put("/controller_config", json={}).status_code == 200
        assert client.post("/arm").status_code == 404
        for action, payload in (("target_ee_pose", {"position": [.45,0,.4]}), ("recover", None)):
            response = client.post("/"+action, json=payload)
            assert response.status_code == 409
            assert "read-only" in response.json()["detail"]
        assert client.get("/state").json()["phase"] == "disarmed"
