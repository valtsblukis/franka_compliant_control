from pathlib import Path
import math
import time

import pytest

from franka_compliant_control import _native
from franka_compliant_control.hold import hold_compliantly
from franka_compliant_control.local_client import LocalControllerClient
from franka_compliant_control.models import ControllerConfig


def profile():
    return ControllerConfig.model_validate_json(Path("configs/controller.fr3.impedance.json").read_text())


def test_original_comparison_profile_is_archived_and_effort_settings_preserved():
    old = ControllerConfig.model_validate_json(Path("configs/controller.fr3.perturbation.json").read_text())
    archived = ControllerConfig.model_validate_json(Path("tests/fixtures/controller/controller.fr3.impedance-20260915-v1.json").read_text())
    new = profile()
    assert old.compliance_mode == "contact_yielding"
    assert new.compliance_mode == "continuous_impedance"
    assert old.model_dump(exclude={"compliance_mode"}) == archived.model_dump(exclude={"compliance_mode"})
    changed = {"motion_limit_policy", "joint_lower", "joint_upper", "joint_limit_margin",
               "measured_translation_velocity_limit", "measured_rotation_velocity_limit",
               "measured_joint_velocity_limit", "measured_joint_velocity_limits"}
    assert archived.model_dump(exclude=changed) == new.model_dump(exclude=changed)
    assert new.motion_limit_policy == "franka" and new.joint_limit_margin == 0


@pytest.mark.parametrize("invalid", ["", "impedance", "yield", None])
def test_invalid_control_law_rejected(invalid):
    with pytest.raises(ValueError):
        ControllerConfig(compliance_mode=invalid)
    if invalid is not None:
        native = _native.Config()
        native.compliance_mode = invalid
        with pytest.raises(ValueError, match="compliance_mode"):
            native.validate()


def test_control_law_can_only_change_while_inactive():
    config = profile()
    with LocalControllerClient(config, backend_kind="sim") as client:
        assert client.state()["compliance_mode"] == "continuous_impedance"
        client.arm()
        alternate = ControllerConfig(**(config.model_dump() | {"compliance_mode": "contact_yielding"}))
        with pytest.raises(RuntimeError, match="disarm"):
            client.controller.configure(alternate)
        assert client.state()["compliance_mode"] == "continuous_impedance"
        assert client.stop(wait_s=2.5)["stopped"]
        client.controller.configure(alternate)
        assert client.state()["compliance_mode"] == "contact_yielding"


@pytest.mark.parametrize("name", ["impedance", "contact"])
def test_contact_keeps_original_equilibrium_and_release_returns_in_simulation(name):
    config = ControllerConfig.model_validate_json(Path(f"configs/controller.fr3.{name}.json").read_text())
    samples = []
    pushed = released = False
    with LocalControllerClient(config, backend_kind="sim") as client:
        def record(event):
            nonlocal pushed, released
            if event["event"] != "sample":
                return
            samples.append(event)
            if event["elapsed_s"] >= .2 and not pushed:
                client.controller.backend.inject_wrench([16,0,0,0,0,0])
                pushed = True
            elif event["elapsed_s"] >= .65 and not released:
                client.controller.backend.inject_wrench([0]*6)
                released = True
        result = hold_compliantly(client, duration_s=4, max_displacement_m=.15,
                                 max_rotation_rad=math.radians(30), record=record, expected_backend="sim")
    assert result["stopped"] and pushed and released
    active = [e for e in samples if e["state"]["phase"] == "active"]
    assert all(e["state"]["compliance_mode"] == "continuous_impedance" for e in active)
    assert all(not e["state"]["yielding"] and e["state"]["drive_scale"] == 1 for e in active)
    assert all(not e["state"]["reference_limited"] for e in active)
    assert all(math.dist(e["state"]["reference_position"], result["target_position"]) < 1e-10 for e in active)
    loaded = [e["state"] for e in active if .35 < e["elapsed_s"] < .65]
    assert loaded and all(s["external_wrench"][0] > config.contact_force_soft for s in loaded)
    assert all(s["commanded_task_wrench"][0]*(s["position"][0]-result["target_position"][0]) < 0 for s in loaded)
    assert all(math.hypot(*e["state"]["commanded_task_wrench"][:3]) <= 8.000001 for e in active)
    peak = max(e["position_error_m"] for e in active)
    assert peak > .005
    assert active[-1]["position_error_m"] < .75*peak


def test_contact_profile_retains_effort_and_motion_settings():
    previous = profile()
    contact = ControllerConfig.model_validate_json(Path("configs/controller.fr3.contact.json").read_text())
    changed = {"bounded_cartesian_spring", "external_torque_limits", "contact_force_stop", "contact_moment_stop"}
    assert previous.model_dump(exclude=changed) == contact.model_dump(exclude=changed)
    assert contact.bounded_cartesian_spring
    assert contact.external_torque_limits == [16]*4 + [8]*3
    assert contact.contact_force_stop == 40 and contact.contact_moment_stop == 20
    assert contact.native().bounded_cartesian_spring
    with pytest.raises(ValueError, match="requires continuous_impedance"):
        ControllerConfig(**(contact.model_dump() | {"compliance_mode": "contact_yielding"}))
    native = contact.native()
    native.compliance_mode = "contact_yielding"
    with pytest.raises(ValueError, match="requires continuous_impedance"):
        native.validate()


def test_contact_profile_tolerates_push_above_previous_joint_trip():
    config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.contact.json").read_text())
    injected = released = False
    loaded = []
    with LocalControllerClient(config, backend_kind="sim") as client:
        def record(event):
            nonlocal injected, released
            if event["event"] != "sample":
                return
            state = event["state"]
            if state["external_joint_torques"][3] > 8:
                loaded.append(state)
            if event["elapsed_s"] > .1 and not injected:
                client.controller.backend.inject_wrench([0,0,0,10,0,0])
                injected = True
            elif event["elapsed_s"] > .23 and not released:
                client.controller.backend.inject_wrench([0]*6)
                released = True
        result = hold_compliantly(client, duration_s=.7, max_displacement_m=None,
                                 max_rotation_rad=None, record=record, expected_backend="sim")
    assert result["stopped"] and injected and released and loaded
    assert all(s["phase"] == "active" and not s["yielding"] for s in loaded)
    assert all(s["contact_load_ratio"] == pytest.approx(10/16) for s in loaded)
    assert all(s["peak_contact_load_ratio"] >= 10/16 for s in loaded)


def test_first_native_stop_sample_survives_unloading():
    config = profile()
    with LocalControllerClient(config, backend_kind="sim") as client:
        client.arm()
        client.controller.backend.inject_wrench([0,0,0,9,0,0])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = client.state()
            if state["stop_sample"] is not None:
                break
            time.sleep(.002)
        else:
            pytest.fail("native contact guard did not record its triggering sample")
        trigger = state["stop_sample"]
        assert state["stop_trigger"] == "external_joint_torque_limit"
        assert trigger["external_wrench"][3] == 9
        assert trigger["external_joint_torques"][3] == 9
        client.controller.backend.inject_wrench([0]*6)
        result = client.stop(wait_s=2.5)
        assert result["state"]["phase"] == "fault"
        assert result["state"]["stop_sample"] == trigger
        assert result["state"]["external_wrench"] == [0]*6
