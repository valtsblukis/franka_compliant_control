"""Recovery/Franky compatibility through a recording backend, not robot dynamics."""
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from pydantic import BaseModel

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import create_app
from test_policy_http import service


class FrankyJointResponse(BaseModel):
    positions: list[float]
    velocities: list[float]
    seq: int | None
    accepted_timestamp_s: float
    age_s: float
    stop_latched: bool


class FrankyStopResponse(BaseModel):
    stopped: bool
    timestamp_s: float


def select(http, mode):
    if mode == "joint":
        config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.joint.json").read_text())
        assert http.put("/controller_config", json=config.model_dump()).status_code == 200
        return "/target_joint_state", {"positions": [0,-.4,0,-1.9,0,1.5,0]}, 200
    return "/target_ee_pose", {"position": [.45,0,.4]}, 202


@pytest.mark.parametrize("mode", ["joint", "osc"])
def test_new_command_recovers_once_and_applies_only_new_target(service, mode):
    http, backend, _ = service
    path, goal, status = select(http, mode)
    backend.phase, backend.reason, backend.running = "fault", "measured_overspeed", False
    assert http.get("/health").json()["automatic_recovery_enabled"]
    # Merely observing the fault must not call recovery.
    assert backend.recover_calls == 0
    response = http.post(path, json={**goal, "seq": 0})
    assert response.status_code == status, response.text
    assert response.json()["seq"] == 0 and response.json()["command_seq"] == 1
    assert backend.recover_calls == 1 and backend.submit_calls == 1
    assert backend.last_recovery["previous_reason"] == "measured_overspeed"
    assert http.post(path, json={**goal, "seq": 1}).status_code == status
    assert backend.recover_calls == 1


@pytest.mark.parametrize("mode", ["joint", "osc"])
def test_failed_recovery_does_not_loop_or_send_a_target(service, mode):
    http, backend, _ = service
    path, goal, _ = select(http, mode)
    backend.phase, backend.reason = "fault", "robot_joint_reflex"
    backend.recovery_error = "robot refused recovery"
    assert http.post(path, json=goal).status_code == 409
    assert backend.recover_calls == 1 and backend.submit_calls == 0
    assert http.post(path, json=goal).status_code == 409
    assert backend.recover_calls == 1 and backend.submit_calls == 0
    assert backend.phase == "fault"


def test_expired_or_invalid_target_cannot_trigger_recovery(service):
    http, backend, clock = service
    backend.phase, backend.reason = "fault", "measured_overspeed"
    assert http.post("/target_ee_pose", json={"position": [0,1]}).status_code == 422
    assert http.post("/target_joint_state", json={"positions": [0]*7}).status_code == 422
    assert http.post("/target_ee_pose", json={"position": [.45,0,.4], "deadline_s": clock.now()-1}).status_code == 409
    assert backend.recover_calls == backend.start_calls == 0


def test_recovery_does_not_extend_expiry_or_replay_old_setpoint(service):
    http, backend, _ = service
    backend.phase, backend.reason = "fault", "measured_overspeed"
    backend.recovery_duration = 2.1
    response = http.post("/target_ee_pose", json={"position": [.5,0,.4]})
    assert response.status_code == 422
    assert backend.recover_calls == 1 and backend.submit_calls == 0 and not backend.running
    assert backend.last_recovery["status"] == "succeeded"
    assert not http.get("/target_ee_pose").json()["has_target"]
    assert http.post("/target_ee_pose", json={"position": [.45,0,.4]}).status_code == 202
    assert backend.recover_calls == 1 and backend.submit_calls == 1


def test_automatic_recovery_can_be_disabled(service):
    config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.joint.json").read_text())
    with TestClient(create_app(config=config, automatic_recovery=False)) as http:
        backend = http.app.state.controller.backend
        backend.phase, backend.reason = "fault", "measured_overspeed"
        assert not http.get("/health").json()["automatic_recovery_enabled"]
        assert http.post("/target_joint_state", json={"positions": [0,-.4,0,-1.9,0,1.5,0]}).status_code == 409
        assert backend.recover_calls == 0


@pytest.mark.parametrize("mode", ["joint", "osc"])
def test_go_home_routes_through_selected_controller_and_can_recover(service, mode):
    http, backend, _ = service
    select(http, mode)
    backend.phase, backend.reason = "fault", "measured_overspeed"
    response = http.post("/go_home")
    assert response.status_code == 200, response.text
    goal = response.json()
    assert goal["home_joint_positions"] == [0,-.4,0,-1.9,0,1.5,0]
    assert goal["mode"] == ("joint" if mode == "joint" else "cartesian")
    assert goal["accepted_not_reached"] and goal["started_control"]
    if mode == "joint":
        assert goal["positions"] == goal["home_joint_positions"]
        FrankyJointResponse.model_validate(goal)
    else:
        assert goal["position"] == [.45,0,.4] and goal["posture"] == goal["home_joint_positions"]
    assert backend.recover_calls == 1 and backend.submit_calls == 1


def test_franky_joint_response_keys_and_sequence_echo(service):
    http, backend, _ = service
    path, goal, _ = select(http, "joint")
    empty = FrankyJointResponse.model_validate(http.get(path).json())
    assert empty.positions == [0]*7 and empty.seq is None and empty.stop_latched
    accepted = http.post(path, json={**goal, "seq": 20})
    assert accepted.status_code == 200
    parsed = FrankyJointResponse.model_validate(accepted.json())
    assert parsed.seq == 20 and not parsed.stop_latched and parsed.age_s == 0
    latest = http.get(path).json()
    assert FrankyJointResponse.model_validate(latest).seq == 20
    assert latest["command_seq"] == 1 and latest["applied"]
    health = http.get("/health").json()
    assert {"robot_connected", "stop_latched", "has_target", "last_control_error", "max_joint_speed_rad_s"} <= health.keys()
    assert health["last_control_error"] is None
    FrankyStopResponse.model_validate(http.post("/stop").json())


def test_client_go_home_uses_one_post_and_requests_stop_on_close(monkeypatch):
    robot = CompliantClient();calls = []
    def request(method, path, **kwargs):
        calls.append((method,path));return {"accepted": True}
    monkeypatch.setattr(robot, "_request", request)
    assert robot.go_home()["accepted"]
    robot.close()
    assert calls == [("POST","/go_home"),("POST","/stop")]
