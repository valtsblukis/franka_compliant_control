"""Contact-relief wire/client contracts: recording backend only, no robot or plant.

Uses the rebuilt native Command/ContactReliefOptions value types. Do not run until
those bindings are available; these tests do not validate native relief dynamics.
"""
import asyncio
from copy import deepcopy
import json
import math
import threading

import httpx
import numpy as np
import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from franka_compliant_control import _native
from franka_compliant_control.client import CompliantClient
from franka_compliant_control.models import ContactReliefRequest, ControllerConfig
from franka_compliant_control.service import create_app
from test_policy_http import RecordingBackend, TOKEN


class FixedClock:
    offset = 100.0

    def now(self):
        return self.offset


def relief_payload(**changes):
    return {
        "mode": "automatic",
        "bias_wrench_tcp_base": [1.0, -2.0, .5, .1, -.2, .3],
        "bias_joint_positions": [0., -.4, 0., -1.9, 0., 1.5, 0.],
        "bias_source": "test-only independently verified free-space record 17",
        "bias_confirmed_free": True,
        "workspace_lower": [.4, -.1, .35],
        "workspace_upper": [.5, .1, .45],
        "workspace_confirmed_clear": True,
        **changes,
    }


@pytest.fixture(params=["manual", "automatic"])
def guarded_relief_payload(request):
    return {"workspace_confirmed_clear": True} if request.param == "manual" else relief_payload()


class ReliefRecordingBackend(RecordingBackend):
    """Only records transport/lifecycle, never constructs a native Backend."""

    def __init__(self, *args):
        super().__init__(*args)
        self.start_arguments = []
        self.relief_state = {
            "external_wrench_tcp_base": [2., -1., .5, .1, -.2, .3],
            "external_wrench_tcp_valid": True,
            "contact_relief_setup_valid": True,
            "contact_relief_active": False,
            "contact_relief_mode": "none",
            "contact_relief_bias_used": False,
            "contact_relief_status": "none",
            "contact_relief_released": False,
            "contact_relief_filtered_wrench_tcp_base": [0.] * 6,
            "contact_relief_translation": 0.,
            "contact_relief_rotation": 0.,
            "contact_relief_deadline_s": 0.,
            "contact_relief_reference_travel": 0.,
            "contact_relief_reference_rotation": 0.,
        }

    def state(self):
        if (self.relief_state["contact_relief_active"] and self.deadline is not None
                and self.clock.now() >= self.deadline):
            self.running, self.phase, self.reason = False, "disarmed", "contact_relief_timeout"
            self.relief_state.update(contact_relief_active=False, contact_relief_status="timed_out")
        return {**super().state(), **deepcopy(self.relief_state)}

    def start(self, command, deadline, automatic_recovery=False, motion_generation=None):
        self.start_arguments.append((deadline, automatic_recovery, motion_generation))
        if not command.relieve_contact:
            return super().start(command, deadline, automatic_recovery, motion_generation)
        self.motion_permit(motion_generation)
        if self.running or self.phase != "disarmed" or self.read_only:
            raise RuntimeError("Relief requires writable inactive Disarmed")
        if automatic_recovery:
            raise RuntimeError("Relief must not recover")
        if not self.clock.now() < deadline <= self.clock.now() + command.action_timeout:
            raise ValueError("Invalid relief deadline")
        self.start_calls += 1
        self.epoch += 1
        command.epoch = self.epoch
        self.seq, self.deadline, self.last_command = command.sequence, deadline, command
        self.running, self.phase, self.mode = True, "active", "cartesian"
        manual = command.relief_options.manual_stop
        self.relief_state.update(contact_relief_active=True,
                                 contact_relief_mode="manual" if manual else "automatic",
                                 contact_relief_bias_used=not manual,
                                 contact_relief_status="following_wrench" if manual else "moving",
                                 contact_relief_released=False, contact_relief_deadline_s=deadline)
        return self.epoch

    def stop(self):
        super().stop()
        if self.relief_state["contact_relief_active"]:
            self.relief_state.update(contact_relief_active=False, contact_relief_status="stopped")


@pytest.fixture
def relief_service(monkeypatch):
    # Real value types enforce the agreed native field names after the rebuild.
    assert callable(getattr(_native, "ContactReliefOptions", None)), "Rebuild native before running relief API tests"
    assert hasattr(_native.ContactReliefOptions, "manual_stop"), "Rebuild native manual relief bindings first"
    clock, backends = FixedClock(), []

    def factory(config, kind, hardware):
        assert kind == "sim"
        backend = ReliefRecordingBackend(config, kind, hardware, clock)
        backends.append(backend)
        return backend

    monkeypatch.setattr(_native, "Backend", factory)
    monkeypatch.setattr(_native, "monotonic_time", clock.now)
    with TestClient(create_app(config=ControllerConfig(command_timeout=.1),
                               automatic_recovery=False)) as http:
        yield http, backends[0], clock
    assert all(b.closed for b in backends)


def test_relief_starts_one_native_command_without_pose_or_recovery(relief_service, caplog):
    http, backend, clock = relief_service
    caplog.set_level("INFO", logger="uvicorn.error")
    payload = relief_payload()
    response = http.post("/relieve_contact", json=payload)
    assert response.status_code == 202, response.text
    ack = response.json()
    assert ack["accepted"] and ack["started_control"] and ack["accepted_not_released"]
    assert ack["disarm_on_completion"] and ack["action"] == "relieve_contact"
    assert ack["mode"] == "cartesian" and ack["epoch"] == backend.epoch
    assert ack["command_seq"] == 1 and ack["deadline_s"] == clock.now() + 10
    assert ack["automatic_release_detection"]
    assert ack["bias_source"] == payload["bias_source"]
    assert payload["bias_source"] in caplog.text
    assert "position" not in ack and "quaternion_xyzw" not in ack
    assert backend.start_calls == 1 and backend.start_arguments == [(110., False, 0)]
    assert backend.arm_calls == backend.submit_calls == backend.recover_calls == 0
    command, default = backend.last_command, _native.Command()
    assert command.relieve_contact and command.mode == _native.Mode.Cartesian
    assert command.action_timeout == 10 and command.sequence == ack["command_seq"]
    assert not command.relief_options.manual_stop
    expected = ContactReliefRequest(**payload).model_dump()
    assert ack["relief_options"] == expected
    for name, value in expected.items():
        if name not in {"mode", "timeout_s", "bias_source"} and value is not None:
            np.testing.assert_equal(getattr(command.relief_options, name), value)
    # Transport metadata/action flags only: no Python-captured pose or posture.
    for name in ("position", "quaternion_xyzw", "twist", "acceleration", "posture",
                 "joints", "joint_velocity", "has_orientation", "has_posture",
                 "finish_at_target", "completion_joint_tolerance", "completion_position_tolerance",
                 "completion_rotation_tolerance", "completion_settle_time"):
        np.testing.assert_equal(getattr(command, name), getattr(default, name))
    latest = http.get("/target_ee_pose").json()
    assert latest["action"] == "relieve_contact" and latest["applied"]
    assert latest["position"] is None and latest["quaternion_xyzw"] is None
    assert latest["relief_options"] == expected and latest["automatic_release_detection"]
    assert http.get("/state").json()["contact_relief_mode"] == "automatic"
    assert http.get("/health").json()["contact_relief_bias_used"]
    clock.offset += .5
    assert http.get("/health").json()["contact_relief_active"]  # Not the .1 s stream lease.
    assert http.get("/command_timeout").json()["command_timeout_s"] == .1


@pytest.mark.parametrize("box", [None, {"workspace_lower": [.4, -.1, .35], "workspace_upper": [.5, .1, .45]}])
def test_manual_default_maps_without_bias_and_forwards_native_mode_and_stop(relief_service, box):
    http, backend, clock = relief_service
    initial = http.get("/state").json()
    assert initial["contact_relief_mode"] == "none" and not initial["contact_relief_bias_used"]
    payload = {"workspace_confirmed_clear": True, **(box or {})}
    response = http.post("/relieve_contact", json=payload)
    assert response.status_code == 202, response.text
    ack = response.json()
    assert ack["relief_options"] == ContactReliefRequest(**payload).model_dump()
    assert ack["relief_options"]["mode"] == "manual" and ack["bias_source"] is None
    assert not ack["automatic_release_detection"] and ack["accepted_not_released"]
    assert ack["mode"] == "cartesian" and ack["deadline_s"] == clock.now() + 10
    command, defaults = backend.last_command, _native.ContactReliefOptions()
    assert command.relieve_contact and command.mode == _native.Mode.Cartesian
    assert command.relief_options.manual_stop and not command.relief_options.bias_confirmed_free
    assert command.action_timeout == 10
    assert command.relief_options.translation_speed == .005
    assert command.relief_options.rotation_speed == pytest.approx(math.radians(1))
    for name in ("bias_wrench_tcp_base", "bias_joint_positions"):
        assert ack["relief_options"][name] is None
        np.testing.assert_equal(getattr(command.relief_options, name), getattr(defaults, name))
    for name in ("workspace_lower", "workspace_upper"):
        np.testing.assert_equal(getattr(command.relief_options, name), box[name] if box else [0.] * 3)
    assert backend.start_arguments == [(110., False, 0)]
    assert backend.arm_calls == backend.submit_calls == backend.recover_calls == 0
    latest = http.get("/target_ee_pose").json()
    assert latest["relief_options"] == ack["relief_options"] and not latest["automatic_release_detection"]
    clock.offset += .5
    for path in ("/state", "/health"):
        state = http.get(path).json()
        assert state["contact_relief_active"] and state["contact_relief_mode"] == "manual"
        assert state["contact_relief_status"] == "following_wrench"
        assert not state["contact_relief_bias_used"] and not state["contact_relief_released"]
    stopped = http.post("/stop").json()
    assert stopped["stopped"] and stopped["state"]["contact_relief_status"] == "stopped"
    assert stopped["state"]["contact_relief_mode"] == "manual"
    assert not stopped["state"]["contact_relief_bias_used"] and not stopped["state"]["contact_relief_released"]


BOUNDS = {
    "timeout_s": (.2, 10.), "force_release": (0., 10.), "moment_release": (0., 3.),
    "max_translation": (0., .05), "max_rotation": (0., math.radians(10)),
    "translation_speed": (0., .02), "rotation_speed": (0., .1),
    "translation_admittance": (0., .01), "rotation_admittance": (0., .1),
    "upward_velocity": (0., .005),
}


@pytest.mark.parametrize("name", BOUNDS)
def test_schema_enforces_numeric_bounds_and_finiteness(name):
    lower, upper = BOUNDS[name]
    ContactReliefRequest(**relief_payload(**{name: upper}))
    if name == "upward_velocity":
        ContactReliefRequest(**relief_payload(**{name: 0}))
    bad_values = [-1., upper * 1.001, float("nan"), float("inf"), -float("inf")]
    if name != "upward_velocity":
        bad_values.append(lower)
    for value in bad_values:
        with pytest.raises(ValidationError):
            ContactReliefRequest(**relief_payload(**{name: value}))


@pytest.mark.parametrize("name,size", [("bias_wrench_tcp_base", 6), ("bias_joint_positions", 7),
                                       ("workspace_lower", 3), ("workspace_upper", 3)])
def test_schema_requires_finite_exact_size_vectors(name, size):
    for vector in ([0.] * (size - 1), [0.] * (size + 1), [float("nan")] * size,
                   [float("inf")] * size, None):
        with pytest.raises(ValidationError):
            ContactReliefRequest(**relief_payload(**{name: vector}))


@pytest.mark.parametrize("missing", list(relief_payload()))
def test_required_bias_provenance_and_clearance_never_autofilled(relief_service, missing):
    http, backend, _ = relief_service
    payload = relief_payload()
    del payload[missing]
    assert http.post("/relieve_contact", json=payload).status_code == 422
    assert backend.start_arguments == [] and backend.recover_calls == 0


@pytest.mark.parametrize("changes", [
    {"bias_source": ""}, {"bias_source": " \t\n"}, {"bias_source": "x" * 257}, {"bias_source": 17},
    {"bias_confirmed_free": False}, {"bias_confirmed_free": 1}, {"bias_confirmed_free": "true"},
    {"workspace_confirmed_clear": False}, {"workspace_confirmed_clear": 1},
    {"workspace_confirmed_clear": "true"}, {"workspace_lower": [.5, -.1, .35]},
    {"workspace_upper": [.3, .1, .45]}, {"timeout_s": .2}, {"timeout_s": 10.01},
    {"force_release": 0}, {"max_translation": .051}, {"rotation_speed": .101},
    {"upward_velocity": .006}, {"frame": "tool"}, {"wrench_point": "flange"},
    {"position": [.45, 0., .4]}, {"automatic_recovery": False}, {"seq": 1},
    {"deadline_s": 999}, {"valid_for_s": 1}, {"auto_bias": True},
])
def test_invalid_request_is_rejected_before_start(relief_service, changes):
    http, backend, _ = relief_service
    assert http.post("/relieve_contact", json=relief_payload(**changes)).status_code == 422
    assert backend.start_arguments == [] and backend.recover_calls == 0


def test_missing_body_is_not_an_implicit_default_action(relief_service):
    http, backend, _ = relief_service
    assert http.post("/relieve_contact").status_code == 422
    assert backend.start_arguments == []
    assert len(ContactReliefRequest(**relief_payload(bias_source="x" * 256)).bias_source) == 256


@pytest.mark.parametrize("condition", ["read_only", "running", "armed", "active", "stopping", "fault", "recovery_on"])
def test_relief_requires_writable_inactive_disarmed_and_recovery_off(relief_service, condition, guarded_relief_payload):
    http, backend, _ = relief_service
    engine = http.app.state.controller
    if condition in {"read_only", "running"}:
        setattr(backend, condition, True)
    elif condition == "recovery_on":
        engine.automatic_recovery = True
    else:
        backend.phase = condition
    response = http.post("/relieve_contact", json=guarded_relief_payload)
    assert response.status_code == 409, response.text
    assert engine.automatic_recovery is (condition == "recovery_on")
    assert backend.start_arguments == [] and backend.recover_calls == 0
    assert engine.accepted_seq == 0 and engine.latest_target is None


@pytest.mark.parametrize("running", [True, False])
def test_active_flag_blocks_targets_home_second_relief_and_enabling_recovery(
        relief_service, monkeypatch, running, guarded_relief_payload):
    http, backend, _ = relief_service
    first = http.post("/relieve_contact", json=guarded_relief_payload).json()
    # Guard specifically on native active state, not just running/phase/latest_target.
    backend.running, backend.phase = running, "active" if running else "disarmed"

    def unexpected_home():
        pytest.fail("home_command must not be requested during relief")

    monkeypatch.setattr(backend, "home_command", unexpected_home)
    for path, payload in [
        ("/target_ee_pose", {"position": [.45, 0., .4]}),
        ("/target_joint_state", {"positions": [0.] * 7}),
        ("/go_home", {}), ("/relieve_contact", guarded_relief_payload),
        ("/automatic_recovery", {"enabled": True}),
    ]:
        assert http.post(path, json=payload).status_code == 409
    assert http.post("/automatic_recovery", json={"enabled": False}).status_code == 200
    assert backend.start_calls == 1 and backend.submit_calls == backend.recover_calls == 0
    assert http.app.state.controller.latest_target["deadline_s"] == first["deadline_s"]
    assert not http.app.state.controller.automatic_recovery
    assert http.post("/stop").status_code == 200
    assert http.post("/automatic_recovery", json={"enabled": True}).status_code == 200


@pytest.mark.parametrize("missing", ["options_type", "manual_stop", "external_wrench_tcp_base", "external_wrench_tcp_valid", "contact_relief_setup_valid",
                                    "contact_relief_active", "contact_relief_mode", "contact_relief_bias_used",
                                    "contact_relief_status", "contact_relief_released",
                                    "contact_relief_filtered_wrench_tcp_base", "contact_relief_translation",
                                    "contact_relief_rotation", "contact_relief_deadline_s",
                                    "contact_relief_reference_travel", "contact_relief_reference_rotation"])
def test_old_or_partial_native_is_unavailable_without_breaking_normal_startup(relief_service, monkeypatch, missing):
    http, backend, _ = relief_service
    if missing == "options_type":
        monkeypatch.delattr(_native, "ContactReliefOptions")
    elif missing == "manual_stop":
        monkeypatch.setattr(_native, "ContactReliefOptions", object)  # Callable, but no manual binding.
    else:
        original = backend.state

        def older_state():
            state = original()
            del state[missing]
            return state

        monkeypatch.setattr(backend, "state", older_state)
    for _ in range(2):  # Filling defaults must never manufacture capability on a later poll.
        for path in ("/state", "/health"):
            assert http.get(path).json()["contact_relief_available"] is False
    assert http.post("/relieve_contact", json=relief_payload()).status_code == 503
    assert http.post("/relieve_contact", json={"workspace_confirmed_clear": True}).status_code == 503
    assert backend.start_arguments == []
    assert http.post("/target_ee_pose", json={"position": [.45, 0., .4]}).status_code == 202
    assert backend.start_calls == 1 and backend.recover_calls == 0


def test_partial_native_command_binding_fails_closed(relief_service, monkeypatch):
    http, backend, _ = relief_service
    monkeypatch.setattr(_native, "Command", object)
    assert http.post("/relieve_contact", json=relief_payload()).status_code == 503
    assert backend.start_arguments == []


def test_old_native_state_fallbacks_do_not_invent_wrench_or_completion(relief_service, monkeypatch):
    http, backend, _ = relief_service
    raw = {key: value for key, value in backend.state().items() if key not in backend.relief_state}
    monkeypatch.setattr(backend, "state", lambda: raw)  # Even a reused raw snapshot stays untouched.
    monkeypatch.setattr(_native, "ContactReliefOptions", None)
    for path in ("/state", "/health", "/state"):
        state = http.get(path).json()
        assert not state["contact_relief_available"] and not state["contact_relief_active"]
        assert state["contact_relief_status"] == "unavailable" and not state["contact_relief_released"]
        assert state["contact_relief_mode"] == "none" and not state["contact_relief_bias_used"]
        assert state["external_wrench_tcp_base"] is None and not state["external_wrench_tcp_valid"]
        assert state["contact_relief_filtered_wrench_tcp_base"] is None
        assert state["contact_relief_translation"] is None and state["contact_relief_deadline_s"] is None
    assert "contact_relief_active" not in raw
    assert http.post("/relieve_contact", json=relief_payload()).status_code == 503


def test_state_health_forward_native_status_without_estimating_success(relief_service):
    http, backend, clock = relief_service
    ack = http.post("/relieve_contact", json=relief_payload(timeout_s=1)).json()
    backend.relief_state.update(contact_relief_filtered_wrench_tcp_base=[1., -2., 3., -.1, .2, -.3],
                               contact_relief_translation=.002, contact_relief_rotation=.01,
                               contact_relief_reference_travel=.003, contact_relief_reference_rotation=.02)
    for path in ("/state", "/health"):
        state = http.get(path).json()
        assert state["contact_relief_available"] and state["contact_relief_active"]
        assert not state["contact_relief_released"]
        for key, value in backend.relief_state.items():
            assert state[key] == value
    clock.offset += 1.1
    health = http.get("/health").json()
    assert not health["contact_relief_active"] and not health["contact_relief_released"]
    assert health["contact_relief_status"] == "timed_out"
    assert health["latest_action"] == "relieve_contact"
    assert http.app.state.controller.latest_target["accepted_not_released"]
    # Only native status changes may report release, never HTTP acceptance/elapsed time.
    backend.relief_state.update(contact_relief_status="released", contact_relief_released=True)
    assert http.get("/state").json()["contact_relief_released"]
    assert ack["accepted_not_released"]


def test_nonfinite_native_relief_diagnostics_are_json_safe(relief_service):
    http, backend, _ = relief_service
    backend.relief_state.update(external_wrench_tcp_base=None, external_wrench_tcp_valid=False,
                               contact_relief_filtered_wrench_tcp_base=[float("nan")] * 6,
                               contact_relief_translation=float("inf"))
    for path in ("/state", "/health"):
        value = http.get(path).json()
        assert value["external_wrench_tcp_base"] is None and not value["external_wrench_tcp_valid"]
        assert value["contact_relief_filtered_wrench_tcp_base"] == [None] * 6
        assert value["contact_relief_translation"] is None


@pytest.mark.parametrize("delay,status", [(.4, 202), (1.1, 409)])
def test_timeout_is_measured_from_http_ingress_not_body_or_handler(
        relief_service, delay, status, guarded_relief_payload):
    http, backend, clock = relief_service
    received_at = clock.now()

    async def exercise():
        async def body():
            clock.offset += delay  # Middleware already ran; body parsing has not completed.
            yield json.dumps(guarded_relief_payload | {"timeout_s": 1}).encode()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=http.app), base_url="http://test") as client:
            return await client.post("/relieve_contact", content=body(), headers={"Content-Type": "application/json"})

    response = asyncio.run(exercise())
    assert response.status_code == status, response.text
    if status == 202:
        assert response.json()["deadline_s"] == received_at + 1
        assert backend.start_arguments == [(received_at + 1, False, 0)]
    else:
        assert backend.start_arguments == []


@pytest.mark.parametrize("arrives_inhibited", [False, True])
def test_ingress_generation_blocks_requests_across_inhibit_release(
        relief_service, arrives_inhibited, guarded_relief_payload):
    http, backend, _ = relief_service
    if arrives_inhibited:
        backend.inhibit_motion()

    async def exercise():
        entered, resume = asyncio.Event(), asyncio.Event()

        async def body():
            entered.set()
            await resume.wait()
            yield json.dumps(guarded_relief_payload).encode()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=http.app), base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/relieve_contact", content=body(),
                                                      headers={"Content-Type": "application/json"}))
            await asyncio.wait_for(entered.wait(), 2)
            if not arrives_inhibited:
                backend.inhibit_motion()
            status = backend.motion_inhibit_status()
            backend.release_motion_inhibit(status["generation"])
            resume.set()
            response = await asyncio.wait_for(pending, 3)
            assert response.status_code == 409, response.text

    asyncio.run(exercise())
    assert not backend.running and backend.start_arguments == []


@pytest.mark.parametrize("change", ["inhibit_cycle", "expiry"])
def test_rechecks_gate_and_expiry_after_waiting_for_service_lock(
        relief_service, monkeypatch, change, guarded_relief_payload):
    http, backend, clock = relief_service
    engine = http.app.state.controller
    entered, errors = threading.Event(), []
    original = engine._motion_generation

    def observed(expected=None):
        value = original(expected)
        if threading.current_thread().name == "queued-relief":
            entered.set()
        return value

    def request():
        try:
            engine.relieve_contact(ContactReliefRequest(**(guarded_relief_payload | {"timeout_s": 1})))
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(engine, "_motion_generation", observed)
    with engine.lock:
        thread = threading.Thread(target=request, name="queued-relief")
        thread.start()
        assert entered.wait(2)
        if change == "expiry":
            clock.offset += 1.1
        else:
            status = backend.inhibit_motion()
            backend.release_motion_inhibit(status["generation"])
    thread.join(3)
    assert not thread.is_alive() and len(errors) == 1
    assert isinstance(errors[0], HTTPException) and errors[0].status_code == 409
    assert backend.start_arguments == []


def test_inhibited_relief_and_native_generation_race_never_record_acceptance(relief_service, monkeypatch):
    http, backend, _ = relief_service
    asserted = backend.inhibit_motion()
    assert http.post("/relieve_contact", json=relief_payload()).status_code == 409
    assert backend.start_arguments == []
    backend.release_motion_inhibit(asserted["generation"])
    original = backend.start

    def raced(command, deadline, automatic_recovery=False, motion_generation=None):
        asserted = backend.inhibit_motion()
        backend.release_motion_inhibit(asserted["generation"])
        return original(command, deadline, automatic_recovery, motion_generation)

    monkeypatch.setattr(backend, "start", raced)
    assert http.post("/relieve_contact", json=relief_payload()).status_code == 409
    assert backend.start_arguments == [(110., False, 2)]
    assert backend.start_calls == 0 and not backend.running
    assert http.app.state.controller.latest_target is None
    assert http.app.state.controller.accepted_seq == 0


def test_native_start_rejection_preserves_previous_acceptance(relief_service, monkeypatch):
    http, backend, _ = relief_service
    first = http.post("/relieve_contact", json=relief_payload()).json()
    assert http.post("/stop").status_code == 200
    engine = http.app.state.controller
    previous = deepcopy(engine.latest_target)

    def rejected(*args, **kwargs):
        raise RuntimeError("native preflight rejected stale bias posture")

    monkeypatch.setattr(backend, "start", rejected)
    assert http.post("/relieve_contact", json=relief_payload()).status_code == 409
    assert engine.latest_target == previous and engine.accepted_seq == first["command_seq"]
    assert engine.epoch == first["epoch"] and backend.recover_calls == 0


def test_next_automatic_relief_needs_fresh_bias_and_advances_sequence(relief_service):
    http, backend, _ = relief_service
    first = http.post("/relieve_contact", json=relief_payload()).json()
    http.post("/stop")
    assert http.post("/relieve_contact", json={"timeout_s": 1}).status_code == 422
    second = http.post("/relieve_contact", json=relief_payload(bias_source="new free-space record")).json()
    assert second["command_seq"] == first["command_seq"] + 1
    assert second["epoch"] == first["epoch"] + 1
    assert backend.start_calls == 2


def test_authentication_and_catalog_include_relief(relief_service):
    with TestClient(create_app(api_token=TOKEN, automatic_recovery=False)) as http:
        assert http.post("/relieve_contact", json=relief_payload()).status_code == 401
        assert http.app.state.controller.backend.start_arguments == []
        http.headers["Authorization"] = "Bearer " + TOKEN
        catalog = http.get("/about", params={"format": "json"}).json()
        entry = next(e for e in catalog["endpoints"] if e["path"] == "/relieve_contact")
        assert entry["method"] == "POST" and "202" in entry["response_status_codes"]
        assert entry["request_schema"]["required"] == ["workspace_confirmed_clear"]
        assert entry["request_schema"]["properties"]["mode"]["default"] == "manual"
        assert entry["request_schema"]["properties"]["timeout_s"]["default"] == 10
        # OpenAPI may omit null defaults; nullable + not-required is the wire contract.
        bias_schema = entry["request_schema"]["properties"]["bias_source"]
        assert bias_schema.get("default") is None
        assert {"type": "null"} in bias_schema["anyOf"]
        assert http.post("/relieve_contact", json=relief_payload()).status_code == 202


def test_client_validates_vectors_and_sends_once_without_recovery_or_bias_capture(monkeypatch):
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"accepted": True, "accepted_not_released": True}

    monkeypatch.setattr(CompliantClient, "_request", request)
    client = CompliantClient()
    try:
        payload = relief_payload(bias_wrench_tcp_base=np.asarray([1., -2., .5, .1, -.2, .3]))
        assert client.relieve_contact(**payload)["accepted_not_released"]
        assert client._sent_setpoints
        assert [(method, path) for method, path, _ in calls] == [("POST", "/relieve_contact")]
        assert calls[0][2]["json"] == ContactReliefRequest(**relief_payload()).model_dump()
    finally:
        client.close()  # Existing client close contract requests stop after sending motion.
    assert [path for _, path, _ in calls] == ["/relieve_contact", "/stop"]


@pytest.mark.parametrize("changes", [{"bias_wrench_tcp_base": [True] * 6}, {"bias_joint_positions": [0.] * 6},
                                      {"bias_confirmed_free": False}, {"upward_velocity": .006},
                                      {"workspace_lower": [float("nan")] * 3}])
def test_client_invalid_options_do_not_send_any_request(monkeypatch, changes):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid relief must fail before any HTTP call")

    monkeypatch.setattr(CompliantClient, "_request", unexpected)
    with CompliantClient() as client:
        with pytest.raises(ValueError):
            client.relieve_contact(**relief_payload(**changes))
        assert not client._sent_setpoints


def test_client_does_not_retry_lost_relief_acceptance(monkeypatch):
    calls = []

    def request(self, method, path, **kwargs):
        calls.append(path)
        if path == "/relieve_contact":
            raise requests.Timeout("acceptance response lost")
        return {}

    monkeypatch.setattr(CompliantClient, "_request", request)
    with CompliantClient() as client:
        with pytest.raises(requests.Timeout):
            client.relieve_contact(**relief_payload())
        assert calls == ["/relieve_contact"] and client._sent_setpoints
    assert calls == ["/relieve_contact", "/stop"]
