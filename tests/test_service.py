import math
import time

import pytest
from fastapi.testclient import TestClient

from franka_compliant_control import _native
from franka_compliant_control.models import ControllerConfig, HardwareSettings
from franka_compliant_control.service import create_app


@pytest.fixture
def client():
    config = ControllerConfig(command_timeout=.15, stop_settle_time=.02, stop_timeout=.8)
    with TestClient(create_app(config=config, enable_sim_injection=True, automatic_recovery=False)) as c:
        yield c


def wait_state(client, predicate, timeout=1.5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = client.get("/state").json()
        if predicate(state):
            return state
        time.sleep(.005)
    pytest.fail(f"state did not satisfy predicate: {state}")


def arm(client):
    # Test setup starts with a real setpoint, never a separate arm endpoint.
    state = client.get("/state").json()
    r = client.post("/target_ee_pose", json={"position": state["position"], "seq": 0})
    assert r.status_code == 202, r.text
    return r.json()


def target(client, session, seq=1, **kw):
    state = client.get("/state").json()
    return {"seq": seq,
            "deadline_s": state["server_monotonic_s"] + .10,
            "position": state["position"], **kw}


def test_starts_disarmed_and_rejects_unowned_commands(client):
    assert client.get("/health").json()["phase"] == "disarmed"
    payload = {"session_id": "unknown", "seq": 1, "deadline_s": 1, "position": [0, 0, 0]}
    assert client.post("/target_ee_pose", json=payload).status_code == 422
    assert client.post("/arm").status_code == 404


def test_cartesian_joint_mode_changes_and_explicit_stop(client):
    session = arm(client)
    assert client.post("/arm").status_code == 404
    payload = target(client, session, position=[.46, 0, .4])
    assert client.post("/target_ee_pose", json=payload).status_code == 202
    wait_state(client, lambda s: s["applied_seq"] == 2)
    state = client.get("/state").json()
    joint = {k: v for k, v in target(client, session, seq=2).items() if k != "position"}
    joint["positions"] = state["positions"]
    assert client.post("/target_joint_state", json=joint).status_code == 200
    wait_state(client, lambda s: s["mode"] == "joint")
    stopped = client.post("/stop", params={"wait_s": 1}).json()
    assert stopped["stopped"]
    new = client.post("/target_ee_pose", json=target(client, session, seq=3))
    assert new.status_code == 202 and new.json()["started_control"]
    assert new.json()["epoch"] != session["epoch"]
    assert client.post("/target_ee_pose", json=target(client, session, seq=4)).status_code == 202


def test_duplicate_and_expired_commands_do_not_refresh_watchdog(client):
    session = arm(client)
    payload = target(client, session)
    assert client.post("/target_ee_pose", json=payload).status_code == 202
    assert client.post("/target_ee_pose", json=payload).status_code == 409
    expired = target(client, session, seq=2, deadline_s=.001)
    assert client.post("/target_ee_pose", json=expired).status_code == 409
    state = wait_state(client, lambda s: not s["running"])
    assert state["reason"] == "command_timeout"
    assert state["applied_seq"] == 2
    assert client.post("/target_ee_pose", json=target(client, session, seq=3)).status_code == 202


@pytest.mark.parametrize("change", [
    {"quaternion_xyzw": [0, 0, 0, 0]}, {"position": [0, 0]},
    {"frame": "camera"}, {"seq": True}, {"unexpected": 1},
    {"twist": [100, 0, 0, 0, 0, 0]},
])
def test_invalid_targets(client, change):
    session = arm(client)
    response = client.post("/target_ee_pose", json=target(client, session, **change))
    assert response.status_code == 422, response.text


def test_nonfinite_input(client):
    session = arm(client)
    payload = target(client, session)
    import json
    payload["position"] = [float("nan"), 0, .4]
    response = client.post("/target_ee_pose", content=json.dumps(payload), headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_native_watchdog_runs_while_python_holds_gil():
    cfg = ControllerConfig(command_timeout=.04, stop_settle_time=.01).native()
    b = _native.Backend(cfg, "sim", _native.HardwareConfig())
    try:
        b.arm()
        end = time.perf_counter() + .15
        while time.perf_counter() < end:
            pass
        state = b.state()
        assert state["reason"] == "command_timeout"
        assert not state["running"]
    finally:
        b.close()


def test_contact_trip_latches_fault_and_requires_recovery(client):
    session = arm(client)
    client.post("/target_ee_pose", json=target(client, session))
    client.post("/sim/wrench", json={"wrench": [16, 0, 0, 0, 0, 0]})
    state = wait_state(client, lambda s: s["reason"] == "contact_force_limit")
    assert state["phase"] in {"stopping", "fault"}
    client.post("/sim/wrench", json={"wrench": [0]*6})
    wait_state(client, lambda s: not s["running"])
    assert client.post("/arm").status_code == 404
    assert client.post("/target_ee_pose", json=target(client, session, seq=2)).status_code == 409
    assert client.post("/recover").json()["phase"] == "disarmed"
    arm(client)


def test_configuration_only_changes_when_inactive(client):
    cfg = client.get("/controller_config").json()
    cfg["commanded_force_limit"] = 5
    assert client.put("/controller_config", json=cfg).status_code == 200
    arm(client)
    assert client.put("/controller_config", json=cfg).status_code == 409
    client.post("/stop", params={"wait_s": 1})
    cfg["contact_force_stop"] = cfg["contact_force_soft"]
    assert client.put("/controller_config", json=cfg).status_code == 422


def test_authentication_and_sim_routes():
    with TestClient(create_app(api_token="test-token")) as c:
        assert c.get("/state").status_code == 401
        c.headers["Authorization"] = "Bearer test-token"
        assert c.get("/state").status_code == 200
        assert c.post("/sim/wrench", json={"wrench": [0]*6}).status_code == 404


def test_rejected_recovery_does_not_revoke_active_owner(client):
    session = arm(client)
    assert client.post("/recover").status_code == 409
    assert client.post("/target_ee_pose", json=target(client, session)).status_code == 202


def test_native_and_python_defaults_match():
    config = ControllerConfig()
    native = _native.Config()
    for name, expected in config.model_dump().items():
        actual = getattr(native, name)
        if isinstance(expected, list):
            assert list(actual) == expected
        else:
            assert actual == expected


def test_hardware_start_requires_explicit_configuration():
    with pytest.raises(ValueError):
        create_app(backend_kind="robot")
    with pytest.raises(ValueError):
        HardwareSettings(robot_ip="unreachable", robot_model="panda", flange_to_tcp=[0]*16,
                         load_mass=0, load_com=[0]*3, load_inertia=[0]*9, calibration_confirmed=True)
