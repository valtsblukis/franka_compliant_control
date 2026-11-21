"""Backoff schema/client/HTTP contracts; recording backend only, no robot or plant.

API tests use rebuilt native value types, not a native Backend. Do not run them
until the matching bindings have been rebuilt. Native eligibility, planning,
limits and dynamics are deliberately NOT reimplemented or validated here.
"""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
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
from franka_compliant_control.models import ControllerConfig, JointLimitBackoffRequest
from franka_compliant_control.service import JOINT_LIMIT_BACKOFF_STATE_DEFAULTS, create_app
from test_policy_http import RecordingBackend, TOKEN


DEFAULT_OPTIONS = dict(trigger_margin=.05, backoff_distance=.1, speed=.03,
                       torque_scale=1., timeout_s=20., confirm_clear_path=True)
OPTION_FIELDS = tuple(name for name in DEFAULT_OPTIONS if name != "timeout_s")
COMMAND_FIELDS = ("backoff_joint_limits", "backoff_options", "action_timeout", "mode")


def test_schema_minimal_input_and_defaults():
    assert JointLimitBackoffRequest(confirm_clear_path=True).model_dump() == DEFAULT_OPTIONS
    assert JointLimitBackoffRequest.model_json_schema()["required"] == ["confirm_clear_path"]
    with pytest.raises(ValidationError):
        JointLimitBackoffRequest()


@pytest.mark.parametrize("confirmation", [False, None, 0, 1, "true", "True", [], {}])
def test_schema_requires_strict_true_confirmation(confirmation):
    with pytest.raises(ValidationError):
        JointLimitBackoffRequest(confirm_clear_path=confirmation)


@pytest.mark.parametrize("name,lower,upper", [
    ("trigger_margin", 0., .15), ("backoff_distance", .02, .2),
    ("speed", 0., .05), ("torque_scale", 0., 1.), ("timeout_s", .2, 30.),
])
def test_schema_numeric_bounds_and_finiteness(name, lower, upper):
    # Keep the separate distance/margin relation valid while probing each bound.
    base = DEFAULT_OPTIONS | {"trigger_margin": .01, "backoff_distance": .2}
    JointLimitBackoffRequest(**(base | {name: upper}))
    if name == "backoff_distance":
        JointLimitBackoffRequest(**(base | {name: lower}))
    bad = [-1., lower - .0001, upper + .0001, float("nan"), float("inf"), -float("inf"), None]
    if name != "backoff_distance":
        bad.append(lower)
    for value in bad:
        with pytest.raises(ValidationError):
            JointLimitBackoffRequest(**(base | {name: value}))


@pytest.mark.parametrize("margin,distance", [(.05, .05), (.1, .05), (.15, .15)])
def test_schema_distance_must_strictly_exceed_margin(margin, distance):
    with pytest.raises(ValidationError):
        JointLimitBackoffRequest(confirm_clear_path=True, trigger_margin=margin, backoff_distance=distance)


@pytest.mark.parametrize("distance", [.02, .05, .1, .15])
def test_schema_entire_target_tolerance_band_must_clear_trigger_margin(distance):
    clearance = distance - min(.02, .2 * distance)
    # Distance > margin alone is insufficient: equality at the tolerance edge
    # and margins inside the band must both be rejected.
    for margin in (clearance, clearance + .0001):
        assert distance > margin
        with pytest.raises(ValidationError, match="must exceed trigger_margin"):
            JointLimitBackoffRequest(confirm_clear_path=True, trigger_margin=margin, backoff_distance=distance)
    accepted = JointLimitBackoffRequest(confirm_clear_path=True, trigger_margin=clearance - .0001,
                                       backoff_distance=distance)
    assert accepted.backoff_distance == distance


INVALID_OPTIONS = [
    {}, {"confirm_clear_path": False}, {"confirm_clear_path": "true"},
    {"confirm_clear_path": True, "trigger_margin": 0},
    {"confirm_clear_path": True, "backoff_distance": .05},
    {"confirm_clear_path": True, "trigger_margin": .08, "backoff_distance": .1},
    {"confirm_clear_path": True, "trigger_margin": .09, "backoff_distance": .1},
    {"confirm_clear_path": True, "speed": .051},
    {"confirm_clear_path": True, "torque_scale": 1.01},
    {"confirm_clear_path": True, "timeout_s": .2},
    *({"confirm_clear_path": True, name: value} for name, value in [
        ("joints", [7]), ("selected", [7]), ("positions", [0.] * 7),
        ("targets", [0.] * 7), ("velocities", [0.] * 7), ("torques", [0.] * 7),
        ("position", [0.] * 3), ("automatic_recovery", False),
        ("seq", 1), ("deadline_s", 999), ("valid_for_s", 1), ("mode", "joint"),
    ]),
]


@pytest.mark.parametrize("options", INVALID_OPTIONS)
def test_schema_rejects_invalid_and_client_selected_motion(options):
    with pytest.raises(ValidationError):
        JointLimitBackoffRequest(**options)


class FixedClock:
    offset = 100.

    def now(self):
        return self.offset


class BackoffRecordingBackend(RecordingBackend):
    """Records one start and publishes canned native diagnostics, never plans motion."""

    def __init__(self, *args):
        super().__init__(*args)
        self.start_arguments = []
        self.start_error = None
        self.backoff_state = deepcopy(JOINT_LIMIT_BACKOFF_STATE_DEFAULTS)
        self.backoff_state["joint_limit_backoff_status"] = "none"
        self.contact_relief_active = False
        # Deliberately unrelated to the ordinary recording backend's cached q.
        # Python must echo this supplied plan, not derive one from that cached q.
        self.native_plan = {
            "joint_limit_backoff_selected": [2, 7],
            "joint_limit_backoff_start_positions": [0., -1.5, 0., -1.9, 0., 1.5, 2.65],
            "joint_limit_backoff_target_positions": [0., -1.4, 0., -1.9, 0., 1.5, 2.55],
            "joint_limit_backoff_torque_limits": [8., 2., 8., 8., 3., 3., 1.],
        }

    def state(self):
        return {**super().state(), **deepcopy(self.backoff_state),
                "contact_relief_active": self.contact_relief_active}

    def start(self, command, deadline, automatic_recovery=False, motion_generation=None):
        self.start_arguments.append((deadline, automatic_recovery, motion_generation))
        if not command.backoff_joint_limits:
            return super().start(command, deadline, automatic_recovery, motion_generation)
        self.motion_permit(motion_generation)
        if self.start_error:
            raise RuntimeError(self.start_error)
        assert not automatic_recovery
        assert not self.running and not self.read_only and self.phase in ("disarmed", "fault")
        assert self.clock.now() < deadline <= self.clock.now() + command.action_timeout
        self.start_calls += 1
        self.epoch += 1
        command.epoch = self.epoch
        self.seq, self.deadline, self.last_command = command.sequence, deadline, command
        previous_reason = self.reason
        self.running, self.phase, self.mode = True, "active", "joint"
        self.backoff_state.update(self.native_plan)
        self.backoff_state.update(joint_limit_backoff_active=True, joint_limit_backoff_status="moving",
                                  joint_limit_backoff_reached=False, joint_limit_backoff_deadline_s=deadline,
                                  joint_limit_backoff_previous_reason=previous_reason)
        return self.epoch

    def stop(self):
        super().stop()
        if self.backoff_state["joint_limit_backoff_active"]:
            self.backoff_state.update(joint_limit_backoff_active=False, joint_limit_backoff_status="stopped")


@pytest.fixture
def backoff_service(monkeypatch):
    options_type = getattr(_native, "JointLimitBackoffOptions", None)
    assert callable(options_type), "Rebuild native before running backoff API tests"
    assert all(hasattr(options_type, name) for name in OPTION_FIELDS)
    assert all(hasattr(_native.Command, name) for name in COMMAND_FIELDS)
    clock, backends = FixedClock(), []

    def factory(config, kind, hardware):
        backend = BackoffRecordingBackend(config, kind, hardware, clock)
        backends.append(backend)
        return backend

    monkeypatch.setattr(_native, "Backend", factory)
    monkeypatch.setattr(_native, "monotonic_time", clock.now)
    # Reuse the existing valid auto profile, including its paired mode-bounded
    # effort/feedback policies and gains; only shorten the streaming lease.
    profile = Path(__file__).resolve().parents[1] / "configs/controller.fr3.auto.json"
    config = ControllerConfig.model_validate(json.loads(profile.read_text()) | {"command_timeout": .1})
    with TestClient(create_app(config=config, automatic_recovery=True)) as http:
        yield http, backends[0], clock
    assert all(backend.closed for backend in backends)


@pytest.mark.parametrize("automatic_recovery", [False, True])
@pytest.mark.parametrize("options", [{"confirm_clear_path": True}, {
    "confirm_clear_path": True, "trigger_margin": .08, "backoff_distance": .16,
    "speed": .02, "torque_scale": .5, "timeout_s": 25,
}])
def test_one_native_joint_command_no_targets_recovery_or_policy_change(backoff_service, automatic_recovery, options, caplog):
    http, backend, clock = backoff_service
    engine = http.app.state.controller
    engine.automatic_recovery = automatic_recovery
    config_before = engine.config.model_dump()
    caplog.set_level("INFO", logger="uvicorn.error")
    response = http.post("/backoff_joint_limits", json=options)
    assert response.status_code == 202, response.text
    ack = response.json()
    expected = JointLimitBackoffRequest(**options).model_dump()
    assert ack["accepted"] and ack["started_control"] and ack["accepted_not_reached"]
    assert ack["disarm_on_completion"] and ack["action"] == "backoff_joint_limits" and ack["mode"] == "joint"
    assert ack["backoff_options"] == expected
    assert ack["epoch"] == backend.epoch and ack["command_seq"] == 1
    assert ack["deadline_s"] == clock.now() + expected["timeout_s"]
    assert backend.start_arguments == [(ack["deadline_s"], False, 0)]
    assert backend.start_calls == 1
    assert backend.arm_calls == backend.submit_calls == backend.recover_calls == 0
    assert engine.automatic_recovery is automatic_recovery
    assert engine.config.model_dump() == config_before
    assert "backoff accepted" in caplog.text and "backoff_joint_limits completed" not in caplog.text
    command, default = backend.last_command, _native.Command()
    assert command.mode == _native.Mode.Joint and command.backoff_joint_limits
    assert command.action_timeout == expected["timeout_s"]
    for name in OPTION_FIELDS:
        assert getattr(command.backoff_options, name) == expected[name]
    for name in ("position", "quaternion_xyzw", "twist", "acceleration", "posture", "joints",
                 "joint_velocity", "has_orientation", "has_posture", "relieve_contact", "finish_at_target",
                 "completion_joint_tolerance", "completion_position_tolerance", "completion_rotation_tolerance",
                 "completion_settle_time"):
        np.testing.assert_equal(getattr(command, name), getattr(default, name))
    assert "positions" not in ack and "joints" not in ack
    for key, value in backend.native_plan.items():
        assert ack[key] == value
    latest = http.get("/target_joint_state").json()
    assert latest["action"] == "backoff_joint_limits" and latest["applied"]
    assert latest["positions"] is None and latest["velocities"] is None
    assert latest["backoff_options"] == expected
    clock.offset += .5  # Longer than stream timeout, much less than action timeout.
    assert http.get("/health").json()["joint_limit_backoff_active"]
    assert http.get("/command_timeout").json()["command_timeout_s"] == .1


@pytest.mark.parametrize("options", INVALID_OPTIONS)
def test_invalid_http_request_never_starts(backoff_service, options):
    http, backend, _ = backoff_service
    assert http.post("/backoff_joint_limits", json=options).status_code == 422
    assert backend.start_arguments == [] and backend.recover_calls == 0


def test_missing_body_never_implicitly_starts(backoff_service):
    http, backend, _ = backoff_service
    assert http.post("/backoff_joint_limits").status_code == 422
    assert backend.start_arguments == []


@pytest.mark.parametrize("condition", ["read_only", "running", "armed", "active", "stopping",
                                        "contact_relief_active", "homing"])
def test_lifecycle_and_other_actions_block_backoff(backoff_service, condition):
    http, backend, _ = backoff_service
    if condition in {"read_only", "running", "contact_relief_active"}:
        setattr(backend, condition, True)
    elif condition == "homing":
        backend.homing, backend.running = True, True
    else:
        backend.phase = condition
    assert http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).status_code == 409
    assert backend.start_arguments == [] and backend.recover_calls == 0


def test_fault_eligibility_is_native_not_python_reason_matching(backoff_service):
    http, backend, _ = backoff_service
    backend.phase, backend.reason = "fault", "opaque_native_eligible_local_reason"
    response = http.post("/backoff_joint_limits", json={"confirm_clear_path": True})
    assert response.status_code == 202, response.text
    assert response.json()["joint_limit_backoff_previous_reason"] == "opaque_native_eligible_local_reason"
    assert backend.start_arguments == [(120., False, 0)]
    assert backend.recover_calls == 0 and http.app.state.controller.automatic_recovery


@pytest.mark.parametrize("reason", ["No joints selected", "SDK fault diagnostics present", "Not stationary Idle",
                                    "Stale native sample", "Outside FR3 specified range", "Ineligible local fault"])
def test_native_rejection_is_409_without_recovery_or_acceptance(backoff_service, reason):
    http, backend, _ = backoff_service
    backend.phase, backend.reason, backend.start_error = "fault", "opaque_fault", reason
    response = http.post("/backoff_joint_limits", json={"confirm_clear_path": True})
    assert response.status_code == 409 and response.json()["detail"] == reason
    engine = http.app.state.controller
    assert backend.start_calls == backend.recover_calls == 0 and not backend.running
    assert backend.phase == "fault" and backend.reason == "opaque_fault"
    assert engine.latest_target is None and engine.accepted_seq == engine.epoch == 0
    assert engine.automatic_recovery


@pytest.mark.parametrize("running", [False, True])
def test_active_backoff_blocks_targets_actions_and_recovery_enable_but_allows_off(backoff_service, monkeypatch, running):
    http, backend, _ = backoff_service
    ack = http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).json()
    backend.running, backend.phase = running, "active" if running else "disarmed"

    def unexpected_home():
        pytest.fail("Must not capture a home command during backoff")

    monkeypatch.setattr(backend, "home_command", unexpected_home)
    for path, payload in [
        ("/target_ee_pose", {"position": [.45, 0., .4]}),
        ("/target_joint_state", {"positions": [0.] * 7}), ("/go_home", {}),
        ("/relieve_contact", {"workspace_confirmed_clear": True}),
        ("/backoff_joint_limits", {"confirm_clear_path": True}),
        ("/automatic_recovery", {"enabled": True}),
    ]:
        assert http.post(path, json=payload).status_code == 409
    assert http.app.state.controller.automatic_recovery  # Existing ON policy is not silently disabled.
    assert http.post("/automatic_recovery", json={"enabled": False}).status_code == 200
    assert backend.start_calls == 1 and backend.submit_calls == backend.recover_calls == 0
    assert http.app.state.controller.latest_target["deadline_s"] == ack["deadline_s"]
    assert http.post("/stop").status_code == 200
    assert http.post("/automatic_recovery", json={"enabled": True}).status_code == 200


@pytest.mark.parametrize("missing", ["options_type", *OPTION_FIELDS, *COMMAND_FIELDS,
                                     *JOINT_LIMIT_BACKOFF_STATE_DEFAULTS])
def test_partial_native_support_is_unavailable_and_never_manufactured(backoff_service, monkeypatch, missing):
    http, backend, _ = backoff_service
    if missing == "options_type":
        monkeypatch.delattr(_native, "JointLimitBackoffOptions")
    elif missing in OPTION_FIELDS:
        partial = type("PartialOptions", (), {key: None for key in OPTION_FIELDS if key != missing})
        monkeypatch.setattr(_native, "JointLimitBackoffOptions", partial)
    elif missing in COMMAND_FIELDS:
        partial = type("PartialCommand", (), {key: None for key in COMMAND_FIELDS if key != missing})
        monkeypatch.setattr(_native, "Command", partial)
    else:
        original = backend.state

        def older_state():
            state = original()
            del state[missing]
            return state

        monkeypatch.setattr(backend, "state", older_state)
    for path in ("/state", "/health", "/state"):
        assert not http.get(path).json()["joint_limit_backoff_available"]
    assert http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).status_code == 503
    assert backend.start_arguments == []
    if missing not in COMMAND_FIELDS:
        assert http.post("/target_joint_state", json={"positions": [0.] * 7}).status_code == 200


def test_old_state_safe_defaults_do_not_modify_raw_snapshot(backoff_service, monkeypatch):
    http, backend, _ = backoff_service
    raw = {key: value for key, value in backend.state().items() if key not in JOINT_LIMIT_BACKOFF_STATE_DEFAULTS}
    monkeypatch.setattr(backend, "state", lambda: raw)
    for path in ("/state", "/health", "/state"):
        state = http.get(path).json()
        assert not state["joint_limit_backoff_available"]
        for key, default in JOINT_LIMIT_BACKOFF_STATE_DEFAULTS.items():
            assert state[key] == default
    assert not (raw.keys() & JOINT_LIMIT_BACKOFF_STATE_DEFAULTS.keys())
    state = http.app.state.controller.state()
    state["joint_limit_backoff_selected"].append(7)
    assert http.app.state.controller.state()["joint_limit_backoff_selected"] == []


def test_native_state_health_and_plan_are_forwarded_not_estimated(backoff_service):
    http, backend, _ = backoff_service
    ack = http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).json()
    for path in ("/state", "/health"):
        state = http.get(path).json()
        assert state["joint_limit_backoff_available"]
        assert state["reference_policy"] == "native_joint_limit_backoff"
        for key, value in backend.backoff_state.items():
            assert state[key] == value
    for status, reached in [("timed_out", False), ("stalled", False), ("reached", True)]:
        backend.running, backend.phase = False, "disarmed" if reached else "fault"
        backend.backoff_state.update(joint_limit_backoff_active=False, joint_limit_backoff_status=status,
                                     joint_limit_backoff_reached=reached)
        for path in ("/state", "/health"):
            state = http.get(path).json()
            assert state["joint_limit_backoff_status"] == status
            assert state["joint_limit_backoff_reached"] is reached
    assert ack["accepted_not_reached"] and not ack["joint_limit_backoff_reached"]
    backend.backoff_state["joint_limit_backoff_target_positions"] = [float("nan")] * 7
    backend.backoff_state["joint_limit_backoff_deadline_s"] = float("inf")
    for path in ("/state", "/health"):
        state = http.get(path).json()
        assert state["joint_limit_backoff_target_positions"] == [None] * 7
        assert state["joint_limit_backoff_deadline_s"] is None


@pytest.mark.parametrize("delay,status", [(.4, 202), (1.1, 409)])
def test_deadline_starts_at_http_ingress_before_body_parsing(backoff_service, delay, status):
    http, backend, clock = backoff_service
    received_at = clock.now()

    async def exercise():
        async def body():
            clock.offset += delay
            yield json.dumps({"confirm_clear_path": True, "timeout_s": 1}).encode()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=http.app), base_url="http://test") as client:
            return await client.post("/backoff_joint_limits", content=body(), headers={"Content-Type": "application/json"})

    response = asyncio.run(exercise())
    assert response.status_code == status, response.text
    assert backend.start_arguments == ([(received_at + 1, False, 0)] if status == 202 else [])
    if status == 202:
        assert response.json()["deadline_s"] == received_at + 1


@pytest.mark.parametrize("arrives_inhibited", [False, True])
def test_ingress_generation_rejects_requests_spanning_inhibit_release(backoff_service, arrives_inhibited):
    http, backend, _ = backoff_service
    if arrives_inhibited:
        backend.inhibit_motion()

    async def exercise():
        async def body():
            if not arrives_inhibited:
                backend.inhibit_motion()
            status = backend.motion_inhibit_status()
            backend.release_motion_inhibit(status["generation"])
            yield json.dumps({"confirm_clear_path": True}).encode()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=http.app), base_url="http://test") as client:
            return await client.post("/backoff_joint_limits", content=body(), headers={"Content-Type": "application/json"})

    assert asyncio.run(exercise()).status_code == 409
    assert backend.start_arguments == []


@pytest.mark.parametrize("change", ["inhibit_cycle", "expiry"])
def test_gate_and_deadline_rechecked_after_service_lock(backoff_service, monkeypatch, change):
    http, backend, clock = backoff_service
    engine = http.app.state.controller
    entered, errors = threading.Event(), []
    original = engine._motion_generation

    def observed(expected=None):
        value = original(expected)
        if threading.current_thread().name == "queued-backoff":
            entered.set()
        return value

    def request():
        try:
            engine.backoff_joint_limits(JointLimitBackoffRequest(confirm_clear_path=True, timeout_s=1))
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(engine, "_motion_generation", observed)
    with engine.lock:
        thread = threading.Thread(target=request, name="queued-backoff")
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


def test_native_generation_race_does_not_record_acceptance(backoff_service, monkeypatch):
    http, backend, _ = backoff_service
    original = backend.start

    def raced(command, deadline, automatic_recovery=False, motion_generation=None):
        status = backend.inhibit_motion()
        backend.release_motion_inhibit(status["generation"])
        return original(command, deadline, automatic_recovery, motion_generation)

    monkeypatch.setattr(backend, "start", raced)
    assert http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).status_code == 409
    assert backend.start_arguments == [(120., False, 0)]
    assert backend.start_calls == 0 and not backend.running
    assert http.app.state.controller.latest_target is None
    assert http.app.state.controller.accepted_seq == 0


def test_rejection_preserves_previous_acceptance_and_next_request_advances(backoff_service):
    http, backend, _ = backoff_service
    first = http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).json()
    http.post("/stop")
    engine = http.app.state.controller
    previous = deepcopy(engine.latest_target)
    backend.start_error = "No joints selected"
    assert http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).status_code == 409
    assert engine.latest_target == previous and engine.accepted_seq == first["command_seq"]
    assert engine.epoch == first["epoch"] and backend.recover_calls == 0
    backend.start_error = None
    second = http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).json()
    assert second["command_seq"] == first["command_seq"] + 1
    assert second["epoch"] == first["epoch"] + 1


def test_sequence_exhaustion_rejects_before_start(backoff_service):
    http, backend, _ = backoff_service
    http.app.state.controller.accepted_seq = 2**63 - 1
    assert http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).status_code == 409
    assert backend.start_arguments == []


def test_authentication_and_catalog(backoff_service):
    with TestClient(create_app(api_token=TOKEN)) as http:
        assert http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).status_code == 401
        assert http.app.state.controller.backend.start_arguments == []
        http.headers["Authorization"] = "Bearer " + TOKEN
        catalog = http.get("/about", params={"format": "json"}).json()
        entry = next(item for item in catalog["endpoints"] if item["path"] == "/backoff_joint_limits")
        assert entry["method"] == "POST" and "202" in entry["response_status_codes"]
        assert entry["request_schema"]["required"] == ["confirm_clear_path"]
        assert set(entry["request_schema"]["properties"]) == set(DEFAULT_OPTIONS)
        assert entry["request_schema"]["properties"]["timeout_s"]["default"] == 20
        assert http.post("/backoff_joint_limits", json={"confirm_clear_path": True}).status_code == 202


def test_client_sends_once_validates_and_stops_on_close(monkeypatch):
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"accepted": True, "accepted_not_reached": True}

    monkeypatch.setattr(CompliantClient, "_request", request)
    with CompliantClient() as client:
        assert client.backoff_joint_limits(confirm_clear_path=True)["accepted_not_reached"]
        assert calls == [("POST", "/backoff_joint_limits", {"json": DEFAULT_OPTIONS})]
        assert client._sent_setpoints
    assert [path for _, path, _ in calls] == ["/backoff_joint_limits", "/stop"]


@pytest.mark.parametrize("options", INVALID_OPTIONS)
def test_invalid_client_input_never_sends(monkeypatch, options):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid backoff must not send any HTTP call")

    monkeypatch.setattr(CompliantClient, "_request", unexpected)
    with CompliantClient() as client:
        with pytest.raises(ValueError):
            client.backoff_joint_limits(**options)
        assert not client._sent_setpoints


def test_client_lost_ack_does_not_retry_or_recover_and_retains_stop_protection(monkeypatch):
    calls = []

    def request(self, method, path, **kwargs):
        calls.append(path)
        if path == "/backoff_joint_limits":
            raise requests.Timeout("acceptance response lost")
        return {}

    monkeypatch.setattr(CompliantClient, "_request", request)
    with CompliantClient() as client:
        with pytest.raises(requests.Timeout):
            client.backoff_joint_limits(confirm_clear_path=True)
        assert calls == ["/backoff_joint_limits"] and client._sent_setpoints
    assert calls == ["/backoff_joint_limits", "/stop"]
