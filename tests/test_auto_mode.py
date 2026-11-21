"""Unified setpoint routing with a recording backend; no robot/plant execution."""
from pathlib import Path
import json

import pytest

from franka_compliant_control import _native
from franka_compliant_control.models import ControllerConfig
from test_policy_http import service


def config():
    return ControllerConfig.model_validate_json(Path("configs/controller.fr3.auto.json").read_text())


def test_policy_profile_tuning_and_common_envelope():
    mixed = config()
    cart = ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.progressive.json").read_text())
    joint = ControllerConfig.model_validate_json(Path("configs/controller.fr3.joint.json").read_text())
    assert mixed.compliance_mode == "auto" and mixed.effort_limit_policy == "mode_bounded"
    assert mixed.osc_translation_stiffness == [600.0] * 3
    assert mixed.osc_translation_damping == [48.99] * 3
    assert mixed.commanded_moment_limit == 6.0
    assert mixed.friction_compensation_torque == 1.0
    assert mixed.friction_compensation_epsilon == .02
    assert mixed.friction_compensation_filter_time_s == .035
    for key in ("osc_rotation_stiffness", "osc_rotation_damping", "commanded_force_limit",
                "posture_stiffness", "posture_damping", "osc_regularization", "osc_feedback_mode"):
        assert getattr(mixed, key) == getattr(cart, key)
    for key in ("joint_stiffness", "joint_damping", "joint_feedback_torque_limits"):
        assert getattr(mixed, key) == getattr(joint, key)
    for key in ("joint_lower", "joint_upper", "joint_limit_margin", "torque_limits", "torque_rate_limit",
                "measured_joint_velocity_limits", "measured_translation_velocity_limit", "measured_rotation_velocity_limit",
                "external_torque_limits", "contact_force_stop", "contact_moment_stop", "collision_policy"):
        assert getattr(mixed, key) == getattr(cart, key) == getattr(joint, key)


@pytest.mark.parametrize("changes", [{"effort_limit_policy": "task_bounded"},
    {"effort_limit_policy": "joint_bounded"}, {"compliance_mode": "joint_impedance"},
    {"osc_feedback_mode": "linear"}, {"osc_translation_stiffness": [0]*3},
    {"joint_feedback_torque_limits": [0]*7}, {"motion_limit_policy": "configured"}])
def test_bad_combined_profiles_are_rejected(changes):
    with pytest.raises(ValueError):
        ControllerConfig(**(config().model_dump() | changes))


@pytest.mark.parametrize("first_mode", ["joint", "cartesian"])
def test_interleaved_mode_requests_keep_one_worker_epoch_and_no_recovery(service, first_mode):
    http, backend, _ = service
    assert http.put("/controller_config", json=config().model_dump()).status_code == 200
    health = http.get("/health").json()
    assert health["supported_setpoint_modes"] == ["cartesian", "joint"]
    assert health["mode_switching_enabled"]
    assert health["cartesian_reference_policy"] == "direct_equilibrium"
    assert health["joint_reference_policy"] == "direct_joint_equilibrium"
    for seq in range(6):
        mode = first_mode if seq%2 == 0 else ("joint" if first_mode == "cartesian" else "cartesian")
        path = "/target_joint_state" if mode == "joint" else "/target_ee_pose"
        target = {"positions": [0,-.4,0,-1.9,0,1.5,0]} if mode == "joint" else {"position": [.45,.01,.4]}
        response = http.post(path, json={**target, "seq": seq})
        assert response.status_code == (200 if mode == "joint" else 202), response.text
        assert response.json()["started_control"] == (seq == 0)
        state = http.get("/state").json()
        assert state["epoch"] == 1 and state["applied_seq"] == seq+1
        assert state["mode"] == mode and state["compliance_mode"] == "auto"
        assert state["selected_control_law"] == ("joint_impedance" if mode == "joint" else "operational_space")
        assert state["reference_policy"] == ("direct_joint_equilibrium" if mode == "joint" else "direct_equilibrium")
        assert state["task_wrench_applicable"] == (mode == "cartesian")
        assert state["selected_effort_limit_policy"] == ("joint_bounded" if mode == "joint" else "task_bounded")
    assert backend.start_calls == backend.arm_calls == 1
    assert backend.recover_calls == 0 and backend.submit_calls == 6
    assert backend.config.compliance_mode == "auto"


def test_invalid_or_stale_opposite_mode_never_switches_controller(service):
    http, backend, _ = service
    http.put("/controller_config", json=config().model_dump())
    assert http.post("/target_joint_state", json={"positions": [0,-.4,0,-1.9,0,1.5,0], "seq": 2}).status_code == 200
    assert http.post("/target_ee_pose", json={"position": [.45,0,.4], "seq": 1}).status_code == 409
    assert http.post("/target_ee_pose", json={"position": [0,1], "seq": 3}).status_code == 422
    assert http.post("/target_ee_pose", json={"position": [.45,0,.4], "seq": 3, "deadline_s": 1}).status_code == 409
    assert http.get("/state").json()["mode"] == "joint"
    assert backend.start_calls == 1 and backend.submit_calls == 1
    assert http.put("/controller_config", json=config().model_dump()).status_code == 409


def test_joint_demo_plan_accepts_the_combined_profile_without_control():
    if not getattr(_native, "offline_kinematics_available", False):
        pytest.skip("offline SDK model unavailable")
    from franka_compliant_control.joint_demo import plan_lower_target
    tcp = json.loads(Path("configs/hardware.wrist-load.estimated.json").read_text())["flange_to_tcp"]
    model = _native.OfflineKinematics(Path("tests/fixtures/fr3-twin.urdf").read_text(), tcp)
    plan = plan_lower_target(model, [0,-.4,0,-1.9,0,1.5,0], config())
    assert plan["lower_position"][2] == pytest.approx(plan["home_position"][2]-.3, abs=2e-5)


def test_home_follows_most_recent_command_mode(service):
    http, backend, _ = service
    http.put("/controller_config", json=config().model_dump())
    assert http.post("/go_home").json()["mode"] == "cartesian"  # Before any mode selection.
    http.post("/target_joint_state", json={"positions": [0,-.4,0,-1.9,0,1.5,0]})
    assert http.post("/go_home").json()["mode"] == "joint"
    http.post("/stop")
    assert http.post("/go_home").json()["mode"] == "joint"
    http.post("/target_ee_pose", json={"position": [.45,0,.4]})
    assert http.post("/go_home").json()["mode"] == "cartesian"
    assert backend.recover_calls == 0
