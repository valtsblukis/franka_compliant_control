import copy
import json
from pathlib import Path

import pytest

from franka_compliant_control import _native
from franka_compliant_control.diagnostics import collision_application_message, stop_failure_message, final_error_message
from franka_compliant_control.local_client import LocalControllerClient
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import ControllerService


def profile():
    return ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.json").read_text())


def test_explicit_robot_thresholds_match_existing_contact_profile_and_keep_tuning():
    osc = profile()
    old = ControllerConfig.model_validate_json(Path("tests/fixtures/controller/controller.fr3.osc-20260915-v2.json").read_text())
    contact = ControllerConfig.model_validate_json(Path("configs/controller.fr3.contact.json").read_text())
    changed = {"collision_policy", "external_torque_limits"}
    assert old.model_dump(exclude=changed) == osc.model_dump(exclude=changed)
    assert osc.collision_policy == "robot_configured"
    for key in ("contact_force_soft", "contact_force_stop", "contact_moment_soft", "contact_moment_stop",
                "external_torque_limits"):
        assert getattr(osc, key) == getattr(contact, key)


def test_collision_metadata_is_not_inferred_from_current_config_or_claimed_applied_in_sim():
    config = profile()
    with LocalControllerClient(config, backend_kind="sim") as client:
        before = client.state()["robot_collision_settings"]
        assert before["status"] == "not_applied" and before["thresholds"] is None
        first = client.arm()["state"]
        info = first["robot_collision_settings"]
        assert info["status"] == "simulated" and not info["is_robot_readback"]
        assert info["acknowledged_monotonic_s"] is None
        assert not first["software_contact_trips_enabled"]
        thresholds = info["thresholds"]
        assert thresholds["upper_force_thresholds_nominal"] == [40]*3+[20]*3
        assert thresholds["lower_force_thresholds_nominal"] == [15]*3+[3]*3
        assert thresholds["upper_torque_thresholds_nominal"] == [16]*4+[8]*3
        for name in ("lower_force", "upper_force", "lower_torque", "upper_torque"):
            assert thresholds[f"{name}_thresholds_nominal"] == thresholds[f"{name}_thresholds_acceleration"]
        assert client.stop(wait_s=2.5)["stopped"]
        updated = ControllerConfig(**(config.model_dump() | {"contact_force_stop": 45}))
        client.controller.configure(updated)
        # The last application record must not be rewritten by a config PUT.
        assert client.state()["robot_collision_settings"] == info
        second = client.arm()["state"]["robot_collision_settings"]
        assert second["application_attempt"] == info["application_attempt"]+1
        assert second["thresholds"]["upper_force_thresholds_nominal"][:3] == [45]*3
        assert client.stop(wait_s=2.5)["stopped"]


def fault_state(fresh=True):
    return {"phase": "fault", "running": False, "reason": "robot_cartesian_reflex",
            "stop_trigger": "robot_cartesian_reflex", "robot_mode": "Reflex" if fresh else "unconfirmed",
            "last_observed_robot_mode": "Reflex" if fresh else "Move", "read_only": False,
            "error": 'libfranka: motion aborted by reflex! ["cartesian_reflex"]',
            "fault_diagnostics": {"sdk_record_count": 2, "sdk_records": [],
                "post_fault_observation": {"attempted": True, "completed": True, "fresh": fresh,
                    "valid": fresh, "sample_age_s": .01, "error": "" if fresh else "network read failed",
                    "state": {"robot_mode": "Reflex", "current_errors_text": '["cartesian_reflex"]'} if fresh else None}}}


def test_reflex_message_distinguishes_fresh_observation_from_failed_read():
    text = stop_failure_message(fault_state())
    assert text.startswith("Robot Cartesian collision reflex")
    assert "post-abort observation: Reflex" in text and "new robot timestamp" in text
    assert "SDK exception records=2" in text
    assert "Stationary disarmed stop not confirmed" not in text
    failed = stop_failure_message(fault_state(False))
    assert "post-abort robot mode unconfirmed" in failed and "cached last mode=Move" in failed
    assert "network read failed" in failed and "No recovery attempted" in failed


def test_pending_read_is_not_reported_as_current_move():
    state = fault_state(False)
    state["fault_diagnostics"]["post_fault_observation"]["completed"] = False
    text = stop_failure_message(state)
    assert "diagnostic read pending" in text and "cached last mode=Move" in text


def test_terminal_message_uses_final_observation_after_close():
    text = final_error_message(RuntimeError("diagnostic read pending"), fault_state())
    assert "post-abort observation: Reflex" in text
    assert "diagnostic read pending" not in text
    assert final_error_message(RuntimeError("disk full"), {"phase": "disarmed"}) == "disk full"


def test_inherited_policy_does_not_claim_to_have_just_applied_old_settings():
    text = collision_application_message({"collision_policy": "robot",
                                          "robot_collision_settings": {"status": "applied"}})
    assert "no collision setter was issued for this arm" in text


def test_legacy_reflex_error_does_not_become_a_network_diagnosis():
    state = fault_state(False)
    del state["fault_diagnostics"]
    state.update(reason="backend_error", robot_mode="Move")
    assert stop_failure_message(state).startswith("Robot Cartesian collision reflex")
    assert "no post-abort observation was recorded" in stop_failure_message(state)


@pytest.mark.parametrize("status", ["not_applied", "applying", "failed", "simulated"])
def test_collision_message_never_claims_unacknowledged_robot_settings(status):
    text = collision_application_message({"robot_collision_settings": {"status": status}})
    assert "acknowledged by SDK" not in text


def test_nonfinite_exception_values_do_not_hide_the_fault():
    state = fault_state()
    state["fault_diagnostics"]["sdk_records"] = [{"state": {"external_wrench": [float("nan")]*6}}]
    class Backend:
        def state(self):
            return copy.deepcopy(state)
    result = ControllerService(Backend(), profile()).state()
    assert result["fault_diagnostics"]["sdk_records"][0]["state"]["external_wrench"] == [None]*6
    json.dumps(result, allow_nan=False)
    assert result["reason"] == "robot_cartesian_reflex"


def test_local_close_preserves_late_post_fault_observation(monkeypatch):
    class Backend:
        def __init__(self, _config, kind, _hardware):
            assert kind == "sim"
            self.closed = False
        def close(self):
            self.closed = True
        def state(self):
            return fault_state(fresh=self.closed)
    monkeypatch.setattr(_native, "Backend", Backend)
    client = LocalControllerClient(profile(), backend_kind="sim")
    assert not client.state()["fault_diagnostics"]["post_fault_observation"]["fresh"]
    client.close()
    assert client.final_state["fault_diagnostics"]["post_fault_observation"]["fresh"]
    assert client.controller is None and client.final_state_error is None
    client.close()  # Still idempotent, preserving the final evidence.
    assert client.final_state is not None
