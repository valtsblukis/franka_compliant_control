"""Software-inhibit contracts. Recording backend + native toy backend only."""

import asyncio
import json
import threading
import time

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from franka_compliant_control import _native
from franka_compliant_control.client import CompliantClient
from franka_compliant_control.models import CartesianTarget, ControllerConfig
from franka_compliant_control.service import create_app, ControllerService
from test_policy_http import service, RecordingBackend, Clock, TOKEN


def assert_latch(http):
    response = http.post("/motion_inhibit")
    assert response.status_code == 200, response.text
    value = response.json()
    assert (
        value["motion_inhibited"] and value["release_token"] and value["stop_requested"]
    )
    return value


def release(http, value):
    return http.post(
        "/motion_inhibit/release", json={"release_token": value["release_token"]}
    )


@pytest.mark.parametrize("mode", ["cartesian", "joint"])
def test_latch_blocks_motion_home_recovery_and_survives_stop_config(service, mode):
    http, backend, _ = service
    endpoint = "/target_ee_pose" if mode == "cartesian" else "/target_joint_state"
    payload = (
        {"position": [0.45, 0, 0.4]}
        if mode == "cartesian"
        else {"positions": [0, -0.4, 0, -1.9, 0, 1.5, 0]}
    )
    if mode == "joint":
        config = http.get("/controller_config").json()
        config["compliance_mode"] = "auto"
        config["effort_limit_policy"] = "mode_bounded"
        assert http.put("/controller_config", json=config).status_code == 200
    assert http.post(endpoint, json=payload).status_code in (200, 202)
    latch = assert_latch(http)
    before = (
        backend.start_calls,
        backend.arm_calls,
        backend.recover_calls,
        backend.submit_calls,
    )
    for _ in range(3):
        assert http.post(endpoint, json=payload).status_code == 409
        assert http.post("/go_home").status_code == 409
        assert http.post("/recover").status_code == 409
    assert http.post("/automatic_recovery", json={"enabled": True}).status_code == 200
    assert http.post("/stop").status_code == 200
    assert (
        http.put(
            "/controller_config", json=http.get("/controller_config").json()
        ).status_code
        == 200
    )
    assert before == (
        backend.start_calls,
        backend.arm_calls,
        backend.recover_calls,
        backend.submit_calls,
    )
    for path in ["/state", "/health"]:
        state = http.get(path).json()
        assert state["motion_inhibited"] and state["motion_inhibit_available"]
    target = http.get(endpoint).json()
    assert not target["valid"] and target["stop_latched"]
    result = release(http, latch)
    assert result.status_code == 200 and not result.json()["motion_resumed"]
    assert not result.json()["motion_inhibited"] and not backend.running
    assert not http.get(endpoint).json()["has_target"]
    assert before == (
        backend.start_calls,
        backend.arm_calls,
        backend.recover_calls,
        backend.submit_calls,
    )
    assert http.post(endpoint, json=payload).status_code in (200, 202)
    assert backend.start_calls == before[0] + 1


def test_reassertion_and_restart_invalidate_old_release_tokens(service):
    http, backend, _ = service
    first = assert_latch(http)
    second = assert_latch(http)
    assert second["generation"] > first["generation"]
    assert release(http, first).status_code == 409
    assert backend.inhibited
    # A newly constructed service wrapper has a different boot token, even if
    # the underlying generation happens to coincide after a process restart.
    other = ControllerService(backend, ControllerConfig())
    with pytest.raises(HTTPException):
        other.release_motion_inhibit(second["release_token"])
    assert release(http, second).status_code == 200
    assert release(http, second).status_code == 409
    for body in [
        {},
        {"release_token": None},
        {"release_token": ""},
        {"release_token": 42},
        {"release_token": "x", "force": True},
    ]:
        assert http.post("/motion_inhibit/release", json=body).status_code == 422


def test_release_refused_while_control_running_or_lifecycle_busy(service):
    http, backend, _ = service
    value = assert_latch(http)
    backend.running = True
    backend.phase = "stopping"
    assert release(http, value).status_code == 409 and backend.inhibited
    backend.running = False
    backend.phase = "disarmed"
    engine = http.app.state.controller
    entered = threading.Event()
    done = threading.Event()

    def hold_lock():
        with engine.lock:
            entered.set()
            done.wait(3)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert entered.wait(2)
    try:
        assert release(http, value).status_code == 409
    finally:
        done.set()
        thread.join(3)
    assert backend.inhibited and release(http, value).status_code == 200


def test_assertion_bypasses_session_lock_and_invalidates_queued_submit(
    service, monkeypatch
):
    http, backend, _ = service
    engine = http.app.state.controller
    entered = threading.Event()
    errors = []
    permit = engine._motion_generation

    def observed(expected=None):
        g = permit(expected)
        if threading.current_thread().name == "queued-publisher":
            entered.set()
        return g

    monkeypatch.setattr(engine, "_motion_generation", observed)

    def publisher():
        try:
            engine.submit(CartesianTarget(position=[0.45, 0, 0.4]))
        except Exception as error:
            errors.append(error)

    with engine.lock:
        thread = threading.Thread(target=publisher, name="queued-publisher")
        thread.start()
        assert entered.wait(2)
        # Separate HTTP worker can assert without needing the lock held here.
        values = []
        asserted = threading.Event()

        def stopper():
            try:
                values.append(assert_latch(http))
            finally:
                asserted.set()

        stop_thread = threading.Thread(target=stopper)
        stop_thread.start()
        assert asserted.wait(2), "Assertion blocked on the session lock"
        engine.release_motion_inhibit(values[0]["release_token"])
    stop_thread.join(3)
    thread.join(3)
    assert not thread.is_alive() and len(errors) == 1
    assert isinstance(errors[0], HTTPException) and errors[0].status_code == 409
    assert backend.start_calls == 0


@pytest.mark.parametrize("arrives_while_inhibited", [False, True])
def test_http_request_queued_before_release_cannot_start_afterward(
    service, arrives_while_inhibited
):
    http, backend, _ = service

    async def exercise():
        entered = asyncio.Event()
        continue_body = asyncio.Event()

        async def body():
            entered.set()
            await continue_body.wait()
            yield json.dumps({"position": [0.45, 0, 0.4]}).encode()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=http.app), base_url="http://test"
        ) as c:
            initial = assert_latch(http) if arrives_while_inhibited else None
            pending = asyncio.create_task(
                c.post(
                    "/target_ee_pose",
                    content=body(),
                    headers={"Content-Type": "application/json"},
                )
            )
            await asyncio.wait_for(entered.wait(), 2)
            value = initial or (await c.post("/motion_inhibit")).json()
            r = await c.post(
                "/motion_inhibit/release",
                json={"release_token": value["release_token"]},
            )
            assert r.status_code == 200
            continue_body.set()
            response = await asyncio.wait_for(pending, 3)
            assert response.status_code == 409, response.text

    asyncio.run(exercise())
    assert not backend.running and backend.start_calls == 0


def test_assertion_interrupts_further_startup_after_inflight_recovery(
    service, monkeypatch
):
    http, backend, _ = service
    backend.phase = "fault"
    backend.reason = "measured_overspeed"
    entered = threading.Event()
    finish = threading.Event()
    responses = []
    original = backend.recover

    def slow_recover(motion_generation=None):
        original(motion_generation=motion_generation)
        entered.set()
        assert finish.wait(3)

    monkeypatch.setattr(backend, "recover", slow_recover)
    thread = threading.Thread(
        target=lambda: responses.append(
            http.post("/target_ee_pose", json={"position": [0.45, 0, 0.4]})
        )
    )
    thread.start()
    assert entered.wait(2)
    try:
        value = assert_latch(http)
        assert not value["stopped"] and value["lifecycle_busy"]
        assert release(http, value).status_code == 409
    finally:
        finish.set()
        thread.join(3)
    assert not thread.is_alive() and responses[0].status_code == 409
    assert (
        backend.recover_calls == 1 and backend.start_calls == 0 and not backend.running
    )
    assert backend.inhibited
    assert release(http, value).status_code == 200


def test_release_does_not_clear_fault_or_readonly_configuration(service):
    http, backend, _ = service
    backend.phase = "fault"
    backend.reason = "measured_overspeed"
    backend.read_only = True
    value = assert_latch(http)
    assert http.post("/recover").status_code == 409
    assert release(http, value).status_code == 200
    assert backend.phase == "fault" and backend.read_only and backend.recover_calls == 0


def test_assert_and_release_do_not_queue_behind_exhausted_http_worker_pool(
    service, monkeypatch
):
    import anyio.to_thread

    with TestClient(create_app(api_token=TOKEN)) as http:
        backend = http.app.state.controller.backend
        entered, finish = threading.Event(), threading.Event()
        original = backend.start

        def slow_start(*args, **kwargs):
            entered.set()
            assert finish.wait(4)
            return original(*args, **kwargs)

        monkeypatch.setattr(backend, "start", slow_start)

        async def exercise():
            limiter = anyio.to_thread.current_default_thread_limiter()
            previous = limiter.total_tokens
            limiter.total_tokens = 1
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=http.app),
                    base_url="http://test",
                    headers={"Authorization": "Bearer " + TOKEN},
                ) as c:
                    pending = asyncio.create_task(
                        c.post("/target_ee_pose", json={"position": [0.45, 0, 0.4]})
                    )
                    try:
                        end = time.monotonic() + 2
                        while not entered.is_set() and time.monotonic() < end:
                            await asyncio.sleep(0.005)
                        assert entered.is_set()
                        response = await asyncio.wait_for(c.post("/motion_inhibit"), 1)
                        value = response.json()
                        assert response.status_code == 200 and value["motion_inhibited"]
                        assert value["lifecycle_busy"] and not value["stopped"]
                        status = await asyncio.wait_for(c.get("/motion_inhibit"), 1)
                        assert status.json()["motion_inhibited"]
                        released = await asyncio.wait_for(
                            c.post(
                                "/motion_inhibit/release",
                                json={"release_token": value["release_token"]},
                            ),
                            1,
                        )
                        assert released.status_code == 409
                    finally:
                        finish.set()
                        response = await asyncio.wait_for(pending, 2)
                    assert response.status_code == 409 and backend.start_calls == 0
            finally:
                limiter.total_tokens = previous

        asyncio.run(exercise())


def test_inhibit_does_not_confirm_stale_robot_stop(service, monkeypatch):
    http, backend, _ = service
    original = backend.state

    def stale():
        return {
            **original(),
            "backend": "robot",
            "robot_mode": "Idle",
            "sample_age_s": 5.0,
        }

    monkeypatch.setattr(backend, "state", stale)
    result = assert_latch(http)
    assert not result["stopped"] and result["motion_inhibited"]


def test_all_inhibit_routes_obey_authentication(service):
    with TestClient(create_app(api_token=TOKEN)) as http:
        backend = http.app.state.controller.backend
        for path in ["/motion_inhibit", "/motion_inhibit/release"]:
            assert http.post(path, json={"release_token": "x"}).status_code == 401
        assert http.get("/motion_inhibit").status_code == 401
        assert not backend.inhibited
        http.headers["Authorization"] = "Bearer " + TOKEN
        value = assert_latch(http)
        assert (
            http.get("/motion_inhibit").json()["release_token"]
            == value["release_token"]
        )
        assert release(http, value).status_code == 200


def test_old_backend_is_not_silently_treated_as_interlock_capable():
    class Old:
        started = False

        def start(self, *args):
            self.started = True

    old = Old()
    engine = ControllerService(old, ControllerConfig())
    assert not engine.motion_inhibit_status()["available"]
    with pytest.raises(HTTPException) as err:
        engine.submit(CartesianTarget(position=[0.45, 0, 0.4]))
    assert err.value.status_code == 503 and not old.started


def test_client_release_is_explicit_no_retry_or_motion(monkeypatch):
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"motion_inhibited": True, "release_token": "boot:1"}

    monkeypatch.setattr(CompliantClient, "_request", request)
    client = CompliantClient()
    try:
        value = client.inhibit_motion(wait_s=0)
        client.get_motion_inhibit()
        client.release_motion_inhibit(value["release_token"])
        with pytest.raises(ValueError):
            client.release_motion_inhibit(None)
    finally:
        client.close()
    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/motion_inhibit"),
        ("GET", "/motion_inhibit"),
        ("POST", "/motion_inhibit/release"),
    ]
    assert calls[-1][2]["json"] == {"release_token": "boot:1"}


@pytest.fixture
def native_backend():
    config = ControllerConfig(
        command_timeout=2.0, stop_settle_time=0.01, stop_timeout=1.0
    )
    backend = _native.Backend(config.native(), "sim", _native.HardwareConfig())
    try:
        yield backend, config
    finally:
        backend.close()


def wait_inactive(backend):
    deadline = time.monotonic() + 3
    while backend.is_running() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert not backend.is_running()


def command(backend):
    c = _native.Command()
    c.sequence = 1
    c.position = backend.state()["position"]
    return c


def test_native_blocks_every_motion_entry_and_ordinary_stop_cannot_release(
    native_backend,
):
    b, cfg = native_backend
    c = command(b)
    before = b.motion_inhibit_status()["generation"]
    value = b.inhibit_motion()
    for action in [
        lambda: b.arm(),
        lambda: b.start(c, _native.monotonic_time() + 1, True),
        lambda: b.submit(c, _native.monotonic_time() + 1),
        lambda: b.recover(),
    ]:
        with pytest.raises(RuntimeError, match="inhibit"):
            action()
    b.stop()
    b.configure(cfg.native())
    assert b.motion_inhibit_status()["motion_inhibited"]
    second = b.inhibit_motion()
    with pytest.raises(RuntimeError):
        b.release_motion_inhibit(value["generation"])
    b.release_motion_inhibit(second["generation"])
    assert not b.is_running() and not b.state()["motion_inhibited"]
    with pytest.raises(RuntimeError, match="changed"):
        b.start(c, _native.monotonic_time() + 1, False, motion_generation=before)
    assert not b.is_running()
    b.start(c, _native.monotonic_time() + 1)
    current = b.inhibit_motion()
    wait_inactive(b)
    assert b.state()["reason"] == "user_stop"
    b.release_motion_inhibit(current["generation"])
    time.sleep(0.03)
    assert not b.is_running()
    with pytest.raises(RuntimeError):
        b.submit(c, _native.monotonic_time() + 1)


def test_native_latch_stops_active_loop_and_explicit_local_hold_needs_rearm():
    from franka_compliant_control.local_client import LocalControllerClient

    with LocalControllerClient(
        ControllerConfig(stop_settle_time=0.01), backend_kind="sim"
    ) as client:
        client.arm()
        result = client.inhibit_motion(wait_s=2)
        assert result["stopped"] and result["motion_inhibited"]
        with pytest.raises(HTTPException):
            client.arm()
        client.release_motion_inhibit(result["release_token"])
        with pytest.raises(RuntimeError, match="Initialize"):
            client.send_pose([0.45, 0, 0.4], [0, 0, 0, 1])
        assert not client.state()["running"]
        client.arm()
        assert client.state()["running"]
