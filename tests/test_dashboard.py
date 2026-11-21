"""Pure snapshot/HTTP tests. No robot connection or simulated plant."""
from pathlib import Path
import math
import logging

from fastapi.testclient import TestClient
from franka_compliant_control.service import create_app

import pytest

from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.status import diagnostic_status, terminal_status
from test_policy_http import service, payload  # Recording backend fixture only.


def config():
    return ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.progressive.json").read_text())


def state(**changes):
    return {"backend": "robot", "phase": "disarmed", "running": False, "read_only": False,
            "reason": "none", "robot_mode": "Idle", "robot_has_errors": False,
            "sample_age_s": .001, "sample_valid": True, "observation_source": "robot_read",
            "positions": [0,-.4,0,-3.0191168785,0,1.5,0], "velocities": [0]*7,
            "external_wrench": [0]*6, **changes}


def test_joint4_near_lower_limit_is_visible_without_claiming_a_violation():
    result = diagnostic_status(state(), config())
    j = result["joints"][3]
    assert j["position_status"] == "near"
    assert j["nearest_distance_deg"] == pytest.approx(3.31645, abs=.001)
    assert j["nearest_boundary"] == "lower"
    assert j["lower_velocity_rad_s"] < 0 < j["upper_velocity_rad_s"]
    assert abs(j["lower_velocity_rad_s"]) < j["upper_velocity_rad_s"]
    assert "J4 lower" in terminal_status(result)
    assert "not permission to arm" in result["readiness_note"]


def test_stale_fault_and_negative_margin_remain_distinct():
    q = state()["positions"];q[3] = -3.08
    result = diagnostic_status(state(positions=q, phase="fault", sample_age_s=147), config())
    assert not result["sample_fresh"]
    assert result["joints"][3]["position_status"] == "outside"
    assert result["joints"][3]["nearest_distance_deg"] < 0
    assert any("fault is latched" in x for x in result["notices"])
    assert "STALE" in terminal_status(result)


def test_nonfinite_and_missing_joint_values_are_not_readiness():
    result = diagnostic_status(state(positions=[None, math.nan], sample_age_s=None, sample_valid=False), config())
    assert not result["sample_fresh"]
    assert all(j["position_status"] == "unknown" for j in result["joints"])
    assert result["nearest_joint"] is None


def test_dashboard_polling_is_observation_only(service):
    http, backend, _ = service
    for _ in range(3):
        response = http.get("/status")
        assert response.status_code == 200
        assert len(response.json()["joints"]) == 7
    page = http.get("/")
    assert page.status_code == 200 and "Cache-Control" in page.headers
    html = page.text
    assert "Joint positions and directional speed limits" in html
    assert "setTimeout(poll,250)" in html
    assert "request('/state/refresh','POST')" in html
    for forbidden in ("request('/arm'", "request('/target_", "innerHTML"):
        assert forbidden not in html
    assert "$('recover').onclick=async()=>" in html
    assert "window.confirm(" in html
    assert "request('/recover','POST',30000)" in html
    assert backend.arm_calls == backend.recover_calls == backend.submit_calls == 0


def test_rejected_start_remains_visible_after_successful_refresh(service, monkeypatch, caplog):
    http, backend, _ = service
    def reject(*_, **kwargs):
        raise RuntimeError("initial state violates robot model joint margins: joint 4")
    monkeypatch.setattr(backend, "start", reject)
    assert http.post("/target_ee_pose", json={"position": [.5,0,.4]}).status_code == 409
    assert "joint 4" in http.get("/status").json()["last_rejection"]["error"]
    reads = []
    def refresh():
        if backend.running:
            raise RuntimeError("state refresh requires an inactive controller")
        reads.append(True)
    monkeypatch.setattr(backend, "refresh_state", refresh, raising=False)
    assert http.post("/state/refresh").status_code == 200
    result = http.get("/status").json()
    assert result["last_operation"]["operation"] == "refresh_state"
    assert "joint 4" in result["last_rejection"]["error"]
    assert "joint 4" in caplog.text
    assert reads == [True] and backend.recover_calls == backend.submit_calls == 0
    backend.running = True
    assert http.post("/state/refresh").status_code == 409
    assert len(reads) == 1


def test_invalid_setpoint_rejection_is_visible_but_not_raw_input(service):
    http, backend, _ = service
    rejected = http.post("/target_ee_pose", json=payload(http, position=[1,2]))
    assert rejected.status_code == 422
    status = http.get("/status").json()
    assert status["last_rejection"]["operation"] == "/target_ee_pose"
    assert "session_id" not in str(status)
    assert backend.submit_calls == 0


def test_recovery_api_remains_available_when_torque_controller_is_faulted(service):
    http, backend, _ = service
    backend.phase, backend.reason, backend.running = "fault", "measured_overspeed", False
    assert http.get("/status").json()["state"]["phase"] == "fault"
    response = http.post("/recover")
    assert response.status_code == 200
    assert response.json()["phase"] == "disarmed" and not response.json()["running"]
    assert backend.recover_calls == 1 and backend.start_calls == backend.submit_calls == 0
    assert http.get("/status").json()["last_operation"]["operation"] == "recover"


def test_recovery_rejected_while_loop_is_running(service):
    http, backend, _ = service
    backend.phase, backend.running = "active", True
    response = http.post("/recover")
    assert response.status_code == 409
    assert backend.recover_calls == 0


def test_terminal_reporter_only_reads_snapshots(service, caplog):
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        with TestClient(create_app(config=config(), status_interval_s=.01)) as http:
            for _ in range(3):
                assert http.get("/status").status_code == 200
            backend = http.app.state.controller.backend
            assert backend.arm_calls == backend.submit_calls == backend.recover_calls == 0
    assert "DISARMED running=False" in caplog.text


@pytest.mark.parametrize("interval", [-1, math.nan, math.inf])
def test_invalid_status_interval_rejected(interval):
    with pytest.raises(ValueError):
        create_app(config=config(), status_interval_s=interval)


def test_explicit_refresh_failure_keeps_existing_fault(service, monkeypatch):
    http, backend, _ = service
    backend.phase, backend.reason = "fault", "robot_joint_reflex"
    def fail():
        raise RuntimeError("read timeout")
    monkeypatch.setattr(backend, "refresh_state", fail, raising=False)
    assert http.post("/state/refresh").status_code == 409
    result = http.get("/status").json()
    assert result["state"]["phase"] == "fault"
    assert result["state"]["reason"] == "robot_joint_reflex"
    assert result["last_rejection"]["error"] == "read timeout"
    assert backend.recover_calls == 0
