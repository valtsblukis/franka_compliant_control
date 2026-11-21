"""HTTP/client contract tests with a recording backend, not simulated robot dynamics."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import shutil
import subprocess
import threading

from fastapi.testclient import TestClient
import numpy as np
import pytest
import requests

from franka_compliant_control import _native
from franka_compliant_control.client import CompliantClient
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import create_app

TOKEN = "unit-test-token-not-a-real-secret"


class Clock:
    offset = 0.0
    def now(self):
        return time.monotonic()+self.offset


class RecordingBackend:
    """Only records calls and lifecycle/expiry. No worker, plant, SDK or robot."""
    def __init__(self, config, kind, hardware, clock):
        assert kind == "sim"  # Test sentinel only; no real simulation object is made.
        self.config, self.clock = config, clock
        self.running = False
        self.phase, self.reason = "disarmed", "none"
        self.epoch = self.seq = self.arm_calls = self.recover_calls = self.submit_calls = 0
        self.deadline = None
        self.read_only = hardware.read_only
        self.reference_position = [.45, 0., .4]
        self.reference_quaternion = [0., 0., 0., 1.]
        self.reference_joints = [0.,-.4,0.,-1.9,0.,1.5,0.]
        self.reference_joint_velocities = [0.]*7
        self.mode = "joint" if config.compliance_mode == "joint_impedance" else "cartesian"
        self.closed = False
        self.start_calls = 0
        self.auto_attempted = False
        self.recovery_error = None
        self.recovery_duration = 0.
        self.last_recovery = None
        self.urdf_cache = ""
        self.homing = False
        self.homing_status = "none"
        self.inhibited = False
        self.inhibit_generation = 0
        self.inhibit_lock = threading.Lock()

    def motion_inhibit_status(self):
        with self.inhibit_lock:
            return {'motion_inhibited': self.inhibited, 'generation': self.inhibit_generation}

    def inhibit_motion(self):
        with self.inhibit_lock:
            self.inhibit_generation += 1
            self.inhibited = True
            status = {'motion_inhibited': True, 'generation': self.inhibit_generation}
        self.stop()
        return status

    def release_motion_inhibit(self, generation):
        with self.inhibit_lock:
            if self.running or not self.inhibited or self.inhibit_generation != generation:
                raise RuntimeError('Motion inhibit release rejected')
            self.inhibit_generation += 1
            self.inhibited = False
            return {'motion_inhibited': False, 'generation': self.inhibit_generation}

    def motion_permit(self, generation=None):
        s = self.motion_inhibit_status()
        if s['motion_inhibited'] or (generation is not None and generation != s['generation']):
            raise RuntimeError('Motion inhibited or generation changed')
        return s['generation']

    def state(self):
        if self.running and self.deadline is not None and self.clock.now() >= self.deadline:
            self.running, self.phase = False, "disarmed"
            self.reason = "home_timeout" if self.homing else "command_timeout"
            if self.homing:
                self.homing_status = "timed_out"
        return {"backend": "sim", "read_only": self.read_only, "running": self.running,
                "phase": self.phase, "reason": self.reason, "error": "", "epoch": self.epoch,
                "applied_seq": self.seq, "position": [.45, 0., .4], "quaternion_xyzw": [0.,0.,0.,1.],
                "positions": [0.,-.4,0.,-1.9,0.,1.5,0.], "velocities": [0.]*7,
                "twist": [0.]*6, "kinematics_valid": True,
                "reference_position": self.reference_position.copy(),
                "reference_quaternion_xyzw": self.reference_quaternion.copy(),
                "reference_joints": self.reference_joints.copy(),
                "reference_joint_velocities": self.reference_joint_velocities.copy(), "mode": self.mode,
                "sample_age_s": .001, "sample_valid": True, "observation_source": "recording_test_backend",
                "robot_mode": "unavailable", "robot_has_errors": False,
                "contact_load_ratio": 0., "peak_contact_load_ratio": 0., "last_recovery": self.last_recovery,
                "homing_status": self.homing_status, "homing_active": self.homing and self.running}

    def arm(self, motion_generation=None):
        self.motion_permit(motion_generation)
        if self.running or self.phase == "fault" or self.read_only:
            raise RuntimeError("explicit inactive, recovered, writable backend required")
        self.arm_calls += 1
        self.epoch += 1
        self.seq = 0
        self.running, self.phase, self.reason = True, "armed", "none"
        self.deadline = self.clock.now()+self.config.command_timeout
        return self.epoch

    def validate_command_mode(self, command):
        if self.config.compliance_mode == "operational_space" and command.mode != _native.Mode.Cartesian:
            raise ValueError("operational_space accepts Cartesian targets only")
        if self.config.compliance_mode == "joint_impedance" and command.mode != _native.Mode.Joint:
            raise ValueError("joint_impedance accepts joint targets only")

    def model_urdf(self):
        return self.urdf_cache

    def home_command(self):
        command = _native.Command()
        command.joints = command.posture = [0,-.4,0,-1.9,0,1.5,0]
        if self.config.compliance_mode == "auto":
            command.mode = _native.Mode.Cartesian if self.mode == "cartesian" else _native.Mode.Joint
        else:
            command.mode = _native.Mode.Cartesian if self.config.compliance_mode == "operational_space" else _native.Mode.Joint
        command.position = [.45,0,.4]
        command.quaternion_xyzw = [0,0,0,1]
        return command

    def start(self, command, deadline, automatic_recovery=False, motion_generation=None):
        generation = self.motion_permit(motion_generation)
        self.validate_command_mode(command)
        if self.clock.now() >= deadline:
            raise ValueError("initial setpoint expired")
        if automatic_recovery and self.phase == "fault" and not self.read_only:
            if self.reason not in ("measured_overspeed", "robot_joint_reflex", "robot_cartesian_reflex", "joint_limit"):
                raise RuntimeError("fault requires explicit recovery")
            if self.auto_attempted:
                raise RuntimeError("automatic recovery already attempted")
            self.auto_attempted = True
            self.recover(motion_generation=generation)
            self.clock.offset += self.recovery_duration
            if self.clock.now() >= deadline:
                raise ValueError("initial setpoint expired during recovery")
        epoch = self.arm(motion_generation=generation)
        self.auto_attempted = False
        self.start_calls += 1
        command.epoch = epoch
        self.submit(command, deadline, motion_generation=generation)
        return epoch

    def submit(self, command, deadline, motion_generation=None):
        self.motion_permit(motion_generation)
        self.state()
        if not self.running:
            raise RuntimeError("controller is not accepting targets")
        self.validate_command_mode(command)
        if command.epoch != self.epoch or command.sequence <= self.seq:
            raise ValueError("stale session or sequence")
        lifetime = command.action_timeout if command.finish_at_target else self.config.command_timeout
        if not self.clock.now() < deadline <= self.clock.now()+lifetime:
            raise ValueError("invalid target lifetime")
        self.submit_calls += 1
        self.seq, self.deadline, self.phase = command.sequence, deadline, "active"
        self.homing_status = "moving" if command.finish_at_target else ("superseded" if self.homing else "none")
        self.homing = command.finish_at_target
        self.last_command = command
        self.mode = "cartesian" if command.mode == _native.Mode.Cartesian else "joint"
        if command.mode == _native.Mode.Cartesian:
            self.reference_position = command.position.tolist()
            if command.has_orientation:
                self.reference_quaternion = command.quaternion_xyzw
        else:
            self.reference_joints = command.joints.tolist()
            self.reference_joint_velocities = command.joint_velocity.tolist()

    def is_running(self):
        return self.state()["running"]

    def stop(self):
        self.running = False
        if self.phase != "fault":
            self.phase, self.reason = "disarmed", "user_stop"

    def recover(self, motion_generation=None):
        self.motion_permit(motion_generation)
        if self.running or self.read_only:
            raise RuntimeError("cannot recover")
        self.recover_calls += 1
        self.last_recovery = {"status": "failed" if self.recovery_error else "succeeded",
                              "previous_reason": self.reason}
        if self.recovery_error:
            raise RuntimeError(self.recovery_error)
        self.phase, self.reason = "disarmed", "none"

    def configure(self, config):
        if self.running:
            raise RuntimeError("disarm before changing configuration")
        self.config = config

    def close(self):
        self.stop()
        self.closed = True


@pytest.fixture
def service(monkeypatch):
    clock, backends = Clock(), []
    def factory(config, kind, hardware):
        backend = RecordingBackend(config, kind, hardware, clock)
        backends.append(backend)
        return backend
    monkeypatch.setattr(_native, "Backend", factory)
    monkeypatch.setattr(_native, "monotonic_time", clock.now)
    config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.progressive.json").read_text())
    monkeypatch.setenv("FCC_API_TOKEN", "ignored-ambient-token")
    with TestClient(create_app(config=config)) as http:
        yield http, backends[0], clock
    assert backends[0].closed


def payload(http, session=None, seq=1, **values):
    return {"seq": seq,
            "deadline_s": http.get("/state").json()["server_monotonic_s"]+.1,
            "position": [.5, .1, .4], **values}


def test_arm_route_and_session_wire_schema_are_removed(service):
    http, backend, _ = service
    assert http.post("/arm").status_code == 404
    assert "/arm" not in http.get("/openapi.json").json()["paths"]
    rejected = http.post("/target_ee_pose", json={"position": [.5,0,.4], "session_id": "obsolete"})
    assert rejected.status_code == 422
    assert backend.start_calls == backend.arm_calls == 0


def test_standalone_hold_initialization_has_no_http_session_and_does_not_restart(service):
    from franka_compliant_control.local_client import LocalControllerClient
    config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.progressive.json").read_text())
    with LocalControllerClient(config, backend_kind="sim") as local:
        initial = local.arm()["state"]
        assert not hasattr(local, "session_id")
        local.send_pose(initial["position"], initial["quaternion_xyzw"])
        backend = local.controller.backend
        assert backend.start_calls == 0 and backend.arm_calls == 1
        backend.running, backend.phase = False, "disarmed"
        with pytest.raises(RuntimeError, match="not accepting"):
            local.send_pose(initial["position"], initial["quaternion_xyzw"])
        assert backend.start_calls == 0


def test_start_shell_uses_project_venv_and_passes_overrides(tmp_path):
    root = tmp_path/"project with spaces"
    (root/".venv/bin").mkdir(parents=True)
    shutil.copy2("start.sh", root/"start.sh")
    fake_python = root/".venv/bin/python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\n')
    fake_python.chmod(0o755)
    result = subprocess.run(["bash", str(root/"start.sh"), "--port", "54325", "--host", "127.0.0.1"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(root), str(root/"scripts/run_robot.py"),
        "--config", str(root/"configs/controller.fr3.auto.json"), "--host", "0.0.0.0",
        "--port", "54325", "--host", "127.0.0.1"]


def test_first_plain_setpoint_starts_control_without_arm_or_session(service):
    http, backend, _ = service
    response = http.post("/target_ee_pose", json={"position": [.5,0,.4]})
    assert response.status_code == 202, response.text
    result = response.json()
    assert result["started_control"] and result["seq"] is None and result["command_seq"] == 1
    assert backend.start_calls == 1 and backend.submit_calls == 1
    assert http.get("/health").json()["explicit_arm_required"] is False
    assert http.get("/target_ee_pose").json()["valid"]
    response = http.post("/target_ee_pose", json={"position": [.5,.1,.4]})
    assert response.status_code == 202 and not response.json()["started_control"]
    assert backend.start_calls == 1 and backend.submit_calls == 2


def test_plain_commands_support_zero_based_optional_sequence(service):
    http, backend, _ = service
    first = {"position": [.5,0,.4], "seq": 0}
    response = http.post("/target_ee_pose", json=first)
    assert response.status_code == 202 and response.json()["client_seq"] == 0
    assert http.post("/target_ee_pose", json=first).status_code == 409
    assert http.post("/target_ee_pose", json={**first, "seq": 1}).status_code == 202
    assert backend.start_calls == 1


def test_plain_invalid_first_target_does_not_start_or_recover(service):
    http, backend, clock = service
    for data in ({"position": [1,2]}, {"position": [1,2,3], "quaternion_xyzw": [0]*4},
                 {"position": [1,2,3], "deadline_s": clock.now()-1},
                 {"position": [1,2,3], "valid_for_s": 2.1}):
        assert http.post("/target_ee_pose", json=data).status_code in (409,422)
    assert http.post("/target_joint_state", json={"positions": [0]*7}).status_code == 422
    assert backend.start_calls == backend.arm_calls == backend.recover_calls == 0
    backend.phase = "fault"
    assert http.post("/target_ee_pose", json={"position": [.5,0,.4]}).status_code == 409
    assert backend.start_calls == backend.recover_calls == 0


def test_new_setpoint_restarts_after_normal_stop_or_expiry_only(service):
    http, backend, clock = service
    point = {"position": [.5,0,.4]}
    assert http.post("/target_ee_pose", json=point).status_code == 202
    clock.offset += 2.1
    assert not http.get("/state").json()["running"]
    assert http.post("/target_ee_pose", json=point).json()["started_control"]
    assert http.post("/stop").json()["stopped"]
    assert http.post("/target_ee_pose", json=point).json()["started_control"]
    assert backend.start_calls == 3 and backend.recover_calls == 0


def test_plain_client_never_sends_an_arm_request(service):
    http, backend, _ = service
    robot = CompliantClient("http://testserver")
    robot.http.close();robot.http=http
    result = robot.set_target_ee_pose(np.array([.5,0,.4], dtype=np.float32), seq=0)
    assert result["started_control"] and backend.start_calls == 1
    assert robot.set_target_ee_pose([.5,.1,.4])["accepted"]
    assert robot.stop()["stopped"]
    assert robot.set_target_ee_pose([.5,0,.4], seq=0)["started_control"]
    assert backend.start_calls == 2
    robot.stop()


def test_lost_initial_setpoint_response_is_not_retried_and_close_requests_stop(monkeypatch):
    robot = CompliantClient()
    calls = []
    def request(method, path, **kwargs):
        calls.append(path)
        if path == "/target_ee_pose":
            raise requests.Timeout("response lost after possible startup")
        return {"stopped": True}
    monkeypatch.setattr(robot, "_request", request)
    with pytest.raises(requests.Timeout):
        robot.set_target_ee_pose([.5,0,.4])
    robot.close()
    assert calls == ["/target_ee_pose", "/stop"]


def test_reads_start_disarmed_and_never_move_or_refresh_targets(service):
    http, backend, _ = service
    assert "Authorization" not in http.headers
    health = http.get("/health").json()
    assert health["stop_latched"] and not health["has_target"]
    assert not health["authentication_enabled"]
    schema = http.get("/openapi.json").json()
    assert not schema.get("components", {}).get("securitySchemes")
    assert health["supported_setpoint_modes"] == ["cartesian"]
    assert health["reference_policy"] == "direct_equilibrium"
    for path in ("/target_ee_pose", "/target_joint_state"):
        target = http.get(path).json()
        assert not target["has_target"] and not target["valid"] and target["seq"] is None
    pose = http.get("/ee_pose").json()
    assert pose["position"] == [.45, 0, .4] and pose["sample_age_s"] == .001
    assert pose["phase"] == "disarmed" and pose["sample_valid"]
    assert len(http.get("/joint_state").json()["positions"]) == 7
    assert backend.arm_calls == backend.submit_calls == backend.recover_calls == 0


def test_latest_target_is_accepted_not_measured_and_has_no_session_secret(service):
    http, backend, clock = service
    session = None
    first = payload(http, session, seq=1, twist=[.01,0,0,0,0,0], acceleration=[.1,0,0,0,0,0])
    response = http.post("/target_ee_pose", json=first)
    assert response.status_code == 202
    assert response.json()["position"] == first["position"]
    assert "session_id" not in response.json()
    goal = http.get("/target_ee_pose").json()
    assert goal["valid"] and goal["applied"] and not goal["stop_latched"]
    assert goal["deadline_s"] == first["deadline_s"]
    assert goal["acceleration"] == first["acceleration"]
    assert goal["accepted_timestamp_s"] > 1_000_000_000
    assert goal["age_s"] >= 0 and "session_id" not in goal
    assert http.get("/ee_pose").json()["position"] != goal["position"]
    assert backend.last_command.acceleration.tolist() == first["acceleration"]
    count = backend.submit_calls
    clock.offset += .3
    expired = http.get("/target_ee_pose").json()
    assert expired["expired"] and not expired["valid"]
    assert expired["deadline_s"] == first["deadline_s"] and backend.submit_calls == count
    assert not http.get("/health").json()["target_valid"]


def test_stop_invalidates_target_and_new_setpoint_replaces_readback(service):
    http, backend, _ = service
    session = None
    command = payload(http, session)
    assert http.post("/target_ee_pose", json=command).status_code == 202
    result = http.post("/stop").json()
    assert result["stopped"] and result["timestamp_s"] > 0
    previous = http.get("/target_ee_pose").json()
    assert previous["has_target"] and not previous["valid"] and previous["stop_latched"]
    new = http.post("/target_ee_pose", json=payload(http, seq=2)).json()
    assert new["started_control"]
    assert new["epoch"] != previous["epoch"]
    assert http.get("/target_ee_pose").json()["valid"]
    assert backend.recover_calls == 0


def test_readback_is_a_copy_not_a_mutable_command_cache(service):
    http, _, _ = service
    session = None
    command = payload(http, session)
    assert http.post("/target_ee_pose", json=command).status_code == 202
    engine = http.app.state.controller
    view = engine.target("cartesian")
    view["position"][0] = 99
    assert engine.target("cartesian")["position"] == command["position"]


def test_stop_does_not_claim_stale_or_reflex_hardware_state_is_confirmed(service, monkeypatch):
    http, backend, _ = service
    original = backend.state
    override = {"backend": "robot", "robot_mode": "Reflex", "robot_has_errors": True}
    monkeypatch.setattr(backend, "state", lambda: {**original(), **override})
    assert not http.post("/stop").json()["stopped"]
    override.update(robot_mode="Idle", robot_has_errors=False, sample_age_s=1.0)
    assert not http.post("/stop").json()["stopped"]
    override["sample_age_s"] = .001
    assert http.post("/stop").json()["stopped"]
    override["velocities"] = [float("nan")]*7
    result = http.post("/stop")
    assert result.status_code == 200 and not result.json()["stopped"]


def test_rejected_update_does_not_replace_readback_or_rearm(service):
    http, backend, _ = service
    session = None
    first = payload(http, session)
    assert http.post("/target_ee_pose", json=first).status_code == 202
    assert http.post("/target_ee_pose", json=first).status_code == 409
    assert http.post("/target_ee_pose", json=payload(http, session, seq=2, quaternion_xyzw=[0]*4)).status_code == 422
    joint = {k:v for k,v in payload(http, session, seq=2).items() if k != "position"}
    joint["positions"] = [0,-.4,0,-1.9,0,1.5,0]
    response = http.post("/target_joint_state", json=joint)
    assert response.status_code == 422 and "Cartesian targets only" in response.text
    assert http.get("/target_ee_pose").json()["seq"] == 1
    assert not http.get("/target_joint_state").json()["has_target"]
    assert backend.submit_calls == 1 and backend.arm_calls == 1


def test_joint_readback_when_joint_control_is_selected(service):
    http, _, _ = service
    config = ControllerConfig(robot_model="fr3").model_dump()
    assert http.put("/controller_config", json=config).status_code == 200
    assert http.get("/health").json()["supported_setpoint_modes"] == ["cartesian", "joint"]
    session = None
    data = {k:v for k,v in payload(http, session).items() if k != "position"}
    data.update(positions=[.1,-.4,0,-1.9,0,1.5,0], velocities=[0]*7)
    assert http.post("/target_joint_state", json=data).status_code == 200
    assert http.get("/target_joint_state").json()["positions"] == data["positions"]
    assert not http.get("/target_ee_pose").json()["has_target"]


def test_timeout_is_finite_and_updates_only_inactive(service):
    http, backend, _ = service
    assert http.get("/command_timeout").json()["command_timeout_s"] == 2.0
    assert http.post("/command_timeout", json={"command_timeout_s": .4}).status_code == 200
    assert http.post("/target_ee_pose", json=payload(http)).status_code == 202
    assert http.post("/command_timeout", json={"command_timeout_s": .8}).status_code == 409
    assert backend.config.command_timeout == .4
    assert http.post("/command_timeout", json={"command_timeout_s": 1e12}).status_code == 409  # Valid value, active loop.
    assert http.post("/command_timeout", json={"command_timeout_s": 0}).status_code == 422
    home = http.post("/go_home")
    assert home.status_code == 200 and home.json()["home_kind"] == "nominal_franky_home"


@pytest.mark.parametrize("path", ["/health", "/state", "/ee_pose", "/joint_state",
                                   "/target_ee_pose", "/target_joint_state", "/command_timeout"])
def test_read_endpoints_need_no_authentication(service, path):
    http, backend, _ = service
    assert "Authorization" not in http.headers
    assert http.get(path).status_code == 200
    assert backend.arm_calls == backend.submit_calls == 0


def test_authentication_requires_explicit_factory_opt_in(service):
    # Backward-compatible custom deployments can still deliberately enable it;
    # neither normal launcher nor FCC_API_TOKEN does so automatically.
    config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.osc.progressive.json").read_text())
    with TestClient(create_app(config=config, api_token=TOKEN)) as protected:
        assert protected.get("/health").status_code == 401
        protected.headers["Authorization"] = f"Bearer {TOKEN}"
        assert protected.get("/health").json()["authentication_enabled"]


def test_numpy_policy_client_aliases_sequence_and_clock(service):
    http, backend, clock = service
    robot = CompliantClient("http://testserver")
    assert "Authorization" not in robot.http.headers
    robot.http.close()
    robot.http = http
    assert backend.arm_calls == 0
    assert not hasattr(robot, "arm")
    result = robot.set_target_ee_pose(np.array([.5,.1,.4], dtype=np.float32),
        np.array([0,0,0,1], dtype=np.float32), seq=np.int64(4))
    assert result["seq"] == 4 and robot.get_target_ee_pose()["command_seq"] == 1
    assert robot.get_current_ee_pose()["position"] == [.45,0,.4]
    assert len(robot.get_current_joint_state()["positions"]) == 7
    assert robot.set_target_ee_pose([.5,.2,.4])["client_seq"] == 5
    with pytest.raises(ValueError, match="seq"):
        robot.set_target_ee_pose([.5,.2,.4], seq=0)
    with pytest.raises(ValueError, match="numeric vector"):
        robot.set_target_ee_pose(np.zeros((3,1)))
    with pytest.raises(ValueError, match="finite"):
        robot.set_target_ee_pose([float("nan"),0,0])
    count = backend.submit_calls
    clock.offset += .02
    observed = robot.state()
    assert observed["server_monotonic_s"] == pytest.approx(clock.now(), abs=.02)
    assert backend.submit_calls == count  # Reads are not keepalives.
    assert robot.stop()["stopped"]
    assert robot.set_command_timeout(.3)["command_timeout_s"] == .3
    assert not hasattr(robot, "session_id") and robot.health()["stop_latched"]


def test_jsonl_generator_adapter_uses_one_session_and_stops(service):
    http, backend, _ = service
    robot = CompliantClient("http://testserver")
    robot.http.close()
    robot.http = http
    spec = importlib.util.spec_from_file_location("http_setpoint_example", "examples/http_setpoints.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = [{"position": [.5,0,.4]}, {"position": [.5,.1,.4]}]
    result = module.stream_setpoints(robot, iter(points), hz=100)
    assert result["sent"] == 2 and backend.arm_calls == 1
    assert backend.submit_calls == 2 and not backend.running
    assert backend.recover_calls == 0
    assert module.stream_setpoints(robot, []) == {"sent": 0, "armed": False}
    assert backend.arm_calls == 1


def test_client_does_not_retry_a_failed_write(monkeypatch):
    robot = CompliantClient()
    calls = []
    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise requests.Timeout("mock timeout")
    monkeypatch.setattr(robot, "_request", fail)
    with pytest.raises(requests.Timeout):
        robot.set_target_ee_pose([.5,0,.4])
    assert len(calls) == 1 and robot.sequence == 1
    robot.http.close()


def test_main_entrypoint_ignores_ambient_token_without_connecting(tmp_path, monkeypatch):
    from franka_compliant_control import service as module
    config = tmp_path/"controller.json"
    hardware = tmp_path/"hardware.json"
    config.write_text(Path("configs/controller.fr3.osc.progressive.json").read_text())
    hardware.write_text(Path("configs/hardware.wrist-load.estimated.json").read_text())
    monkeypatch.setenv("FCC_API_TOKEN", "ignored")
    monkeypatch.setattr(sys, "argv", ["franka-compliant-control", "--backend", "robot",
                                    "--config", str(config), "--hardware-config", str(hardware)])
    calls = []
    monkeypatch.setattr(module, "run_http_server", lambda app, **kw: calls.append((app, kw)))
    module.main()
    assert not calls[0][0].openapi().get("components", {}).get("securitySchemes")


def test_launcher_host_port_and_disarmed_app_without_starting_server(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("policy_robot_launcher", "scripts/run_robot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setenv("FCC_API_TOKEN", "ignored-ambient-token")
    hardware = json.loads(Path("configs/hardware.wrist-load.estimated.json").read_text())
    (tmp_path/"local-hardware.json").write_text(json.dumps(hardware))
    (tmp_path/"local-controller.json").write_text(Path("configs/controller.fr3.osc.progressive.json").read_text())
    # A directory at the former token path proves startup no longer reads it.
    (tmp_path/".fcc-api-token").mkdir()
    calls = []
    monkeypatch.setattr(module, "run_http_server", lambda app, **kwargs: calls.append((app, kwargs)))
    assert module.main(["--host", "0.0.0.0", "--port", "54325"]) == 0
    assert calls[0][1] == {"host": "0.0.0.0", "port": 54325, "pid_path": tmp_path/"logs/robot-service.pid"}
    assert not (tmp_path/"logs/robot-service.pid").exists()
    assert not calls[0][0].openapi().get("components", {}).get("securitySchemes")
    # Lifespan was never entered: no Backend (real or simulated) was constructed.
