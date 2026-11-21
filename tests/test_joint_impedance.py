"""Configuration and recording-backend API checks, without robot/plant dynamics."""
import math
from pathlib import Path

import numpy as np
import pytest

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.models import ControllerConfig
from test_policy_http import service


def config():
    return ControllerConfig.model_validate_json(Path("configs/controller.fr3.joint.json").read_text())


def test_joint_profile_is_explicit_and_preserves_manufacturer_envelope():
    c = config()
    cartesian = ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.progressive.json").read_text())
    assert c.compliance_mode == "joint_impedance" and c.effort_limit_policy == "joint_bounded"
    assert cartesian.compliance_mode == "operational_space"
    for key in ("joint_lower", "joint_upper", "joint_limit_margin", "torque_limits", "torque_rate_limit",
                "measured_joint_velocity_limits", "measured_translation_velocity_limit",
                "measured_rotation_velocity_limit", "collision_policy", "external_torque_limits",
                "contact_force_stop", "contact_moment_stop"):
        assert getattr(c, key) == getattr(cartesian, key)
    assert list(c.native().joint_feedback_torque_limits) == [8]*4+[3]*3


@pytest.mark.parametrize("changes", [
    {"effort_limit_policy": "franka"}, {"compliance_mode": "operational_space"},
    {"osc_feedback_mode": "saturated"}, {"robot_model": "panda"},
    {"joint_stiffness": [0]*7}, {"joint_damping": [-1]*7},
    {"joint_feedback_torque_limits": [0]*7}, {"joint_feedback_torque_limits": [13]*7},
    {"joint_feedback_torque_limits": [math.nan]*7}, {"joint_feedback_torque_limits": [1]*6},
])
def test_invalid_joint_configuration(changes):
    with pytest.raises(ValueError):
        ControllerConfig(**(config().model_dump() | changes))


def select_joint(http):
    response = http.put("/controller_config", json=config().model_dump())
    assert response.status_code == 200, response.text


def test_joint_setpoint_starts_direct_control_and_can_be_replaced(service):
    http, backend, _ = service
    select_joint(http)
    health = http.get("/health").json()
    assert health["supported_setpoint_modes"] == ["joint"]
    assert health["reference_policy"] == "direct_joint_equilibrium"
    assert not health["explicit_arm_required"]
    before = http.get("/joint_state").json()["positions"]
    goal = before.copy();goal[0] += .02
    first = http.post("/target_joint_state", json={"positions": goal})
    assert first.status_code == 200 and first.json()["started_control"]
    state = http.get("/state").json()
    assert state["mode"] == "joint"
    assert state["reference_joints"] == goal
    assert state["commanded_task_wrench"] is None and not state["task_wrench_applicable"]
    assert state["reference_position"] is None
    assert not state["software_contact_trips_enabled"]
    assert http.get("/status").json()["joints"][0]["tracking_error_rad"] == pytest.approx(.02)
    goal[0] = before[0]-.02
    second = http.post("/target_joint_state", json={"positions": goal, "velocities": [.05]+[0]*6})
    assert second.status_code == 200 and not second.json()["started_control"]
    assert backend.start_calls == 1
    assert backend.last_command.joint_velocity.tolist() == [.05]+[0]*6
    assert http.get("/target_joint_state").json()["positions"] == goal
    assert http.post("/stop").json()["stopped"]


def test_cartesian_commands_are_rejected_before_joint_start(service):
    http, backend, _ = service
    select_joint(http)
    response = http.post("/target_ee_pose", json={"position": [.5,0,.4]})
    assert response.status_code == 422 and "joint targets only" in response.text
    assert backend.start_calls == 0 and backend.recover_calls == 0


@pytest.mark.parametrize("values", [[0]*6, [math.nan]*7, [[0]]*7])
def test_client_rejects_bad_joint_vectors_locally(values):
    robot = CompliantClient("http://must-not-connect.invalid")
    try:
        with pytest.raises(ValueError):
            robot.set_target_joint_state(values)
        assert not robot._sent_setpoints
    finally:
        robot.close()


def test_policy_client_joint_stream_needs_no_arm(service):
    http, backend, _ = service
    select_joint(http)
    robot = CompliantClient("http://testserver")
    robot.http.close();robot.http = http
    goal = np.array([0,-.4,0,-1.9,0,1.5,0], dtype=np.float32)
    result = robot.set_target_joint_state(goal, seq=0)
    assert result["started_control"] and backend.start_calls == 1
    assert robot.get_current_joint_state()["positions"] != []
    robot.set_target_joint_state(goal, velocities=np.zeros(7), seq=1)
    assert backend.submit_calls == 2 and backend.start_calls == 1
    assert robot.stop()["stopped"]
