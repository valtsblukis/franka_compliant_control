from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
import math
import threading

from fastapi.testclient import TestClient
import pytest

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.hold import hold_compliantly
from franka_compliant_control.local_client import LocalControllerClient
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import create_app


@pytest.mark.parametrize("transport", ["http", "local"])
def test_push_yields_and_release_returns_toward_unchanged_target(transport):
    # Test-plant damping is chosen for a short, deterministic settling test;
    # this does not change the real robot's configuration.
    config = ControllerConfig(robot_model="fr3", translation_damping=[150]*3)
    events, sent = [], []
    with ExitStack() as stack:
        if transport == "http":
            http = stack.enter_context(TestClient(create_app(config=config, enable_sim_injection=True)))
            client = CompliantClient("http://testserver", timeout_s=5)
            client.http.close()
            client.http = http
            def inject(wrench):
                http.post("/sim/wrench", json={"wrench": wrench}).raise_for_status()
        else:
            client = stack.enter_context(LocalControllerClient(config, backend_kind="sim"))
            inject = client.controller.backend.inject_wrench
        original_send = client.send_pose
        pushed = released = False

        def send(position, quaternion):
            sent.append((tuple(position), tuple(quaternion)))
            return original_send(position, quaternion)

        def record(event):
            nonlocal pushed, released
            events.append(event)
            if event["event"] == "sample":
                if event["elapsed_s"] >= .2 and not pushed:
                    inject([9,0,0,0,0,0])
                    pushed = True
                elif event["elapsed_s"] >= .45 and not released:
                    inject([0]*6)
                    released = True

        client.send_pose = send
        summary = hold_compliantly(client, duration_s=3, record=record, expected_backend="sim")
        assert summary["stopped"] and summary["yielding_samples"] > 0
        assert pushed and released and len(set(sent)) == 1
        assert sent[0][0] == tuple(summary["target_position"])
        samples = [event for event in events if event["event"] == "sample"]
        peak = max(event["position_error_m"] for event in samples)
        assert peak > .0005  # The test actually displaced the end effector.
        assert samples[-1]["position_error_m"] < peak*.5
        assert not samples[-1]["state"]["yielding"]
        assert events[-2]["result"]["state"]["phase"] == "disarmed"


def test_perturbation_profile_resists_centimetre_displacement_and_returns():
    config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.perturbation.json").read_text())
    samples = []
    with LocalControllerClient(config, backend_kind="sim") as client:
        pushed = released = False
        def record(event):
            nonlocal pushed, released
            if event["event"] != "sample":
                return
            samples.append(event)
            if event["elapsed_s"] >= .15 and not pushed:
                client.controller.backend.inject_wrench([6,0,0,0,0,0])
                pushed = True
            elif event["elapsed_s"] >= 1.5 and not released:
                client.controller.backend.inject_wrench([0]*6)
                released = True
        result = hold_compliantly(client, duration_s=6, max_displacement_m=.15,
                                 record=record, expected_backend="sim")
    assert result["stopped"] and pushed and released
    peak = max(e["position_error_m"] for e in samples)
    assert peak > .01  # An actual centimetre-scale disturbance in the test plant.
    assert samples[-1]["position_error_m"] < peak*.75
    target_x = result["target_position"][0]
    disturbed = [e for e in samples if .8 < e["elapsed_s"] < 1.5]
    assert disturbed and all(e["state"]["commanded_task_wrench"][0] *
                            (e["state"]["position"][0]-target_x) < 0 for e in disturbed)
    assert max(math.dist([0]*3,e["state"]["commanded_task_wrench"][:3]) for e in samples) <= 8.00001


def test_revised_profile_recovers_orientation_more_strongly_than_v1():
    recovery_errors = []
    for path in ("tests/fixtures/controller/controller.fr3.perturbation-20260909-v1.json",
                 "configs/controller.fr3.perturbation.json"):
        config = ControllerConfig.model_validate_json(Path(path).read_text())
        samples = []
        pushed = released = False
        with LocalControllerClient(config, backend_kind="sim") as client:
            def record(event):
                nonlocal pushed, released
                if event["event"] != "sample":
                    return
                samples.append(event)
                if event["elapsed_s"] >= .1 and not pushed:
                    client.controller.backend.inject_wrench([0,0,0,2,0,0])
                    pushed = True
                elif event["elapsed_s"] >= 1 and not released:
                    client.controller.backend.inject_wrench([0]*6)
                    released = True
            result = hold_compliantly(client, duration_s=4, max_rotation_rad=math.radians(30),
                                     record=record, expected_backend="sim")
        assert result["stopped"] and pushed and released
        assert max(e["rotation_error_rad"] for e in samples) > math.radians(5)
        recovery_errors.append(sum(.5*(a["rotation_error_rad"]+b["rotation_error_rad"])*
                                   (b["elapsed_s"]-a["elapsed_s"])
                                   for a,b in zip(samples,samples[1:]) if a["elapsed_s"] >= 1.05))
    assert recovery_errors[1] < recovery_errors[0]*.9


def test_current_profile_yields_above_former_moment_stop_and_releases():
    config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.perturbation.json").read_text())
    samples = []
    pushed = released = False
    with LocalControllerClient(config, backend_kind="sim") as client:
        def record(event):
            nonlocal pushed, released
            if event["event"] != "sample":
                return
            samples.append(event)
            if event["elapsed_s"] >= .1 and not pushed:
                client.controller.backend.inject_wrench([0,0,0,6.5,0,0])
                pushed = True
            elif event["elapsed_s"] >= .3 and not released:
                client.controller.backend.inject_wrench([0]*6)
                released = True
        result = hold_compliantly(client, duration_s=.7, record=record, expected_backend="sim")
    assert result["stopped"] and pushed and released
    loaded = [e["state"] for e in samples if e["state"]["external_wrench"][3] > 6]
    assert loaded and all(s["yielding"] and s["phase"] == "active" for s in loaded)
    assert all(s["stop_trigger"] == "none" for s in loaded)
    assert not samples[-1]["state"]["yielding"]


class FakeClient:
    def __init__(self):
        self.armed = False
        self.stop_calls = 0
        self.arm_calls = 0
        self.sent = []
        self.change = {}
        self.stop_confirmed = True

    def state(self):
        state = {"backend": "robot", "phase": "active" if self.armed else "disarmed",
                 "running": self.armed, "read_only": False, "reason": "none", "error": "",
                 "sample_valid": True, "sample_age_s": 0, "applied_seq": len(self.sent),
                 "position": [.5,0,.4] if self.armed else [.4,0,.3],
                 "quaternion_xyzw": [0,0,0,1], "yielding": False}
        if self.sent:
            state.update(deepcopy(self.change))
        return state

    def get_config(self):
        return {"stop_timeout": 2}

    def arm(self):
        self.arm_calls += 1
        self.armed = True
        return {"state": self.state()}

    def send_pose(self, position, quaternion):
        self.sent.append((tuple(position), tuple(quaternion)))

    def stop(self, wait_s):
        self.stop_calls += 1
        self.armed = False
        return {"stopped": self.stop_confirmed,
                "state": {"phase": "disarmed" if self.stop_confirmed else "fault",
                          "reason": "user_stop" if self.stop_confirmed else "stop_timeout", "error": ""}}


def test_cancellation_during_yield_stops_without_retargeting():
    client = FakeClient()
    client.change = {"position": [.52,0,.4], "yielding": True}
    cancelled = threading.Event()
    def record(event):
        if event["event"] == "sample":
            cancelled.set()
    result = hold_compliantly(client, duration_s=0, stop_event=cancelled, record=record)
    assert result["reason"] == "interrupted" and result["stopped"]
    assert client.sent == [((.5,0,.4), (0,0,0,1))]
    assert client.stop_calls == 1


def test_optional_excursion_bounds_keep_target_and_still_stop_on_cancel():
    client = FakeClient()
    client.change = {"position": [.8,0,.4],
                     "quaternion_xyzw": [0,0,math.sin(.5),math.cos(.5)]}
    cancelled = threading.Event()
    def record(event):
        if event["event"] == "sample":
            cancelled.set()
    result = hold_compliantly(client, duration_s=0, max_displacement_m=None,
                             max_rotation_rad=None, stop_event=cancelled, record=record)
    assert result["stopped"] and result["max_position_error_m"] == pytest.approx(.3)
    assert result["max_rotation_error_rad"] == pytest.approx(1.)
    assert client.sent == [((.5,0,.4), (0,0,0,1))]


@pytest.mark.parametrize("bounds", [(None, .5), (.1, None)])
def test_each_optional_excursion_bound_can_still_stop_independently(bounds):
    client = FakeClient()
    client.change = {"position": [.8,0,.4],
                     "quaternion_xyzw": [0,0,math.sin(.5),math.cos(.5)]}
    with pytest.raises(RuntimeError, match="Excursion limit"):
        hold_compliantly(client, max_displacement_m=bounds[0], max_rotation_rad=bounds[1])
    assert client.stop_calls == client.arm_calls == 1


def test_immediate_cancellation_can_stop_before_first_target_is_observed():
    client = FakeClient()
    client.change = {"applied_seq": 0}
    cancelled = threading.Event()
    def record(event):
        if event["event"] == "sample":
            cancelled.set()
    result = hold_compliantly(client, duration_s=0, stop_event=cancelled, record=record)
    assert result["reason"] == "interrupted" and result["stopped"]
    assert result["applied_seq"] == 0 and client.stop_calls == 1


@pytest.mark.parametrize("change,match", [
    ({"sample_age_s": .2}, "older than 50 ms"),
    ({"position": [.56,0,.4], "yielding": True}, "Excursion limit"),
    ({"quaternion_xyzw": [0,0,math.sin(.2),math.cos(.2)]}, "Excursion limit"),
    ({"phase": "fault", "running": False, "reason": "contact_force_limit"}, "contact_force_limit"),
])
def test_guard_failure_stops_once_without_retry_or_recovery(change, match):
    client = FakeClient()
    client.change = change
    with pytest.raises(RuntimeError, match=match):
        hold_compliantly(client)
    assert client.arm_calls == client.stop_calls == 1
    assert len(client.sent) == 1


def test_logging_failure_still_requests_stop():
    client = FakeClient()
    def record(event):
        if event["event"] == "sample":
            raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        hold_compliantly(client, record=record)
    assert client.stop_calls == 1


def test_unconfirmed_stop_is_not_reported_as_success():
    client = FakeClient()
    client.stop_confirmed = False
    cancelled = threading.Event()
    def record(event):
        if event["event"] == "sample":
            cancelled.set()
    original_stop = client.stop
    def stop(wait_s):
        result = original_stop(wait_s)
        result["state"].update(stop_trigger="contact_moment_limit", robot_mode="Reflex")
        return result
    client.stop = stop
    with pytest.raises(RuntimeError, match="stop_timeout.*initial trigger=contact_moment_limit.*robot mode=Reflex"):
        hold_compliantly(client, stop_event=cancelled, record=record)


def test_existing_active_session_is_left_alone():
    client = FakeClient()
    client.armed = True
    with pytest.raises(RuntimeError, match="disarmed"):
        hold_compliantly(client)
    assert client.arm_calls == client.stop_calls == 0


def test_fault_preflight_includes_latched_reason_and_error():
    client = FakeClient()
    original = client.state
    client.state = lambda: original() | {"phase": "fault", "reason": "stop_timeout",
                                         "error": "robot-side stop did not produce confirmed stationary idle state"}
    with pytest.raises(RuntimeError, match="phase=fault, reason=stop_timeout, error='robot-side stop"):
        hold_compliantly(client)
    assert client.arm_calls == client.stop_calls == 0


def test_lost_arm_response_still_requests_stop():
    client = FakeClient()
    def arm():
        client.armed = True
        raise TimeoutError("lost response")
    client.arm = arm
    with pytest.raises(TimeoutError, match="lost response"):
        hold_compliantly(client)
    assert client.stop_calls == 1
