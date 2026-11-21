"""HTTP command transport. Robot control and command expiry live in C++."""
from __future__ import annotations

import argparse
import asyncio
import logging
import hmac
import math
import os
import pathlib
import threading
import time
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, ConfigDict
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .models import AutomaticRecoveryUpdate, CartesianTarget, CommandTimeoutUpdate, ContactReliefRequest, ControllerConfig, HardwareSettings, HomeRequest, JointLimitBackoffRequest, JointTarget, WrenchInjection
from .status import diagnostic_status, terminal_status
from .network import DEFAULT_PORT, run_http_server
from .about import endpoint_catalog, render_about
from .constants import DEFAULT_COMMAND_TIMEOUT_S

LOG = logging.getLogger("uvicorn.error")

# Missing telemetry means unsupported, not a zero wrench or a successful action.
# Check capability against the raw backend state BEFORE supplying these defaults.
CONTACT_RELIEF_STATE_DEFAULTS = {
    "external_wrench_tcp_base": None,
    "external_wrench_tcp_valid": False,
    "contact_relief_setup_valid": False,
    "contact_relief_active": False,
    "contact_relief_mode": "none",
    "contact_relief_bias_used": False,
    "contact_relief_status": "unavailable",
    "contact_relief_released": False,
    "contact_relief_filtered_wrench_tcp_base": None,
    "contact_relief_translation": None,
    "contact_relief_rotation": None,
    "contact_relief_deadline_s": None,
    "contact_relief_reference_travel": None,
    "contact_relief_reference_rotation": None,
}


JOINT_LIMIT_BACKOFF_STATE_DEFAULTS = {
    "joint_limit_backoff_active": False,
    "joint_limit_backoff_status": "unavailable",
    "joint_limit_backoff_reached": False,
    "joint_limit_backoff_selected": [],
    "joint_limit_backoff_start_positions": None,
    "joint_limit_backoff_target_positions": None,
    "joint_limit_backoff_deadline_s": None,
    "joint_limit_backoff_previous_reason": None,
    "joint_limit_backoff_torque_limits": None,
}


class MotionInhibitRelease(BaseModel):
    model_config = ConfigDict(extra='forbid')
    release_token: str = Field(strict=True, min_length=1, max_length=128)


class TelemetryPrepare(BaseModel):
    model_config=ConfigDict(extra='forbid')
    capacity: int=Field(default=90000,ge=100,le=120000,strict=True)
    joint_speed: float=Field(default=0.,ge=0,le=3,allow_inf_nan=False)
    linear_speed: float=Field(default=0.,ge=0,le=1,allow_inf_nan=False)
    angular_speed: float=Field(default=0.,ge=0,le=3,allow_inf_nan=False)


class TelemetryClear(BaseModel):
    model_config=ConfigDict(extra='forbid')
    capture_id: int=Field(ge=1,strict=True)


class ControllerService:
    def __init__(self, backend, config: ControllerConfig, *, automatic_recovery=False):
        self.backend = backend
        self.config = config
        self.automatic_recovery = automatic_recovery
        self.lock = threading.RLock()
        self.epoch = 0
        self.accepted_seq = 0
        self.simple_sequence: int | None = None
        self.latest_target: dict | None = None
        self.last_operation: dict | None = None
        self.last_rejection: dict | None = None
        self.last_timeout_request: dict | None = None
        self._inhibit_instance = uuid.uuid4().hex

    def motion_inhibit_status(self):
        if not hasattr(self.backend, 'motion_inhibit_status'):
            return {'available': False, 'motion_inhibited': None, 'generation': None,
                    'release_token': None, 'scope': 'backend_lifetime'}
        status = self.backend.motion_inhibit_status()
        return {**status, 'available': True, 'scope': 'backend_lifetime',
                'release_token': (f'{self._inhibit_instance}:{status["generation"]}'
                                  if status['motion_inhibited'] else None)}

    def _motion_generation(self, expected=None):
        status = self.motion_inhibit_status()
        if not status['available']:
            raise HTTPException(503, 'Native motion-inhibit support unavailable; update the inactive backend')
        if status['motion_inhibited']:
            raise HTTPException(409, 'Motion inhibited; explicit release required')
        if expected is not None and expected != status['generation']:
            raise HTTPException(409, 'Motion inhibit changed; send a fresh request')
        return status['generation']

    def state(self):
        from . import _native
        state = dict(self.backend.state())
        relief_options_type = getattr(_native, "ContactReliefOptions", None)
        state["contact_relief_available"] = (
            callable(relief_options_type) and hasattr(relief_options_type, "manual_stop") and
            all(key in state for key in CONTACT_RELIEF_STATE_DEFAULTS))
        backoff_options_type = getattr(_native, "JointLimitBackoffOptions", None)
        command_type = getattr(_native, "Command", None)
        state["joint_limit_backoff_available"] = (
            callable(backoff_options_type) and callable(command_type) and
            all(hasattr(backoff_options_type, name) for name in
                JointLimitBackoffRequest.model_fields if name != "timeout_s") and
            all(hasattr(command_type, name) for name in
                ("backoff_joint_limits", "backoff_options", "action_timeout", "mode")) and
            all(key in state for key in JOINT_LIMIT_BACKOFF_STATE_DEFAULTS))
        for key, default in (CONTACT_RELIEF_STATE_DEFAULTS | JOINT_LIMIT_BACKOFF_STATE_DEFAULTS).items():
            state.setdefault(key, deepcopy(default))
        inhibit = self.motion_inhibit_status()
        state.update(motion_inhibit_available=inhibit['available'],
                     motion_inhibited=inhibit['motion_inhibited'],
                     motion_inhibit_generation=inhibit['generation'])
        state["server_monotonic_s"] = state['state_timestamp_monotonic_s'] if 'state_timestamp_monotonic_s' in state else _native.monotonic_time()
        state["accepted_seq"] = self.accepted_seq
        state["automatic_recovery_enabled"] = self.automatic_recovery
        state["last_timeout_request"] = deepcopy(self.last_timeout_request)
        state["command_timeout_s"] = self.config.command_timeout
        state["frame"] = "base"
        state["wrench_point"] = "robot_stiffness_frame" if state["read_only"] else "configured_tcp"
        state["hardware_available"] = _native.hardware_available
        state["libfranka_version"] = _native.libfranka_version
        state["robot_model"] = self.config.robot_model
        state["compliance_mode"] = self.config.compliance_mode
        automatic_mode = self.config.compliance_mode == "auto"
        backoff_active = state["joint_limit_backoff_active"]
        joint_control = (backoff_active or self.config.compliance_mode == "joint_impedance" or
                         (automatic_mode and state.get("mode") == "joint"))
        state["mode_switching_enabled"] = automatic_mode
        state["selected_control_law"] = ("joint_impedance" if joint_control else
                                         "operational_space" if automatic_mode else self.config.compliance_mode)
        state["effort_limit_policy"] = self.config.effort_limit_policy
        state["selected_effort_limit_policy"] = (("joint_bounded" if joint_control else "task_bounded")
                                                if automatic_mode else self.config.effort_limit_policy)
        state["osc_feedback_mode"] = self.config.osc_feedback_mode
        state["reference_policy"] = ("native_joint_limit_backoff" if backoff_active
                                     else "direct_joint_equilibrium" if joint_control
                                     else "direct_equilibrium" if self.config.osc_feedback_mode != "linear"
                                     else "ruckig")
        state["task_wrench_applicable"] = not joint_control
        if joint_control:
            # No Cartesian controller reference or claimed Cartesian effort bound.
            for key in ("reference_position", "reference_quaternion_xyzw", "reference_twist",
                        "reference_acceleration", "commanded_task_wrench", "spring_wrench"):
                state[key] = None
        state["collision_policy"] = self.config.collision_policy
        state["software_contact_trips_enabled"] = self.config.collision_policy == "configured"
        if self.config.collision_policy == "robot":
            # No readback API for the inherited robot thresholds; do not invent
            # a fraction of an unknown collision envelope.
            state["contact_load_ratio"] = state["peak_contact_load_ratio"] = None
        # Preserve fault visibility even if the robot supplied a nonfinite sample.
        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, list):
                return [finite_json(v) for v in value]
            if isinstance(value, dict):
                return {k: finite_json(v) for k, v in value.items()}
            return value
        return finite_json(state)

    def _initialize_local_hold(self):
        """Internal standalone-hold initialization, never exposed as an HTTP route."""
        generation = self._motion_generation()
        with self.lock:
            self._motion_generation(generation)
            self.epoch = self.backend.arm(motion_generation=generation)
            self.accepted_seq = 0
            self.simple_sequence = None
            self.latest_target = None
            return {"command_timeout_s": self.config.command_timeout, "state": self.state()}

    def submit(self, target: CartesianTarget | JointTarget, *, start_if_inactive=True,
               home: HomeRequest | None = None, motion_generation=None):
        from . import _native
        generation = self._motion_generation(motion_generation)
        received_at = _native.monotonic_time()
        with self.lock:
            self._motion_generation(generation)
            state = self.backend.state()
            if state.get("contact_relief_active", False):
                raise HTTPException(409, "Contact relief is active; normal targets are not accepted")
            if state.get("joint_limit_backoff_active", False):
                raise HTTPException(409, "Joint-limit backoff is active; normal targets are not accepted")
            if target.seq is not None and self.simple_sequence is not None and target.seq <= self.simple_sequence:
                raise HTTPException(409, "Sequence must strictly increase within the setpoint stream")
            now = _native.monotonic_time()
            lifetime = home.timeout_s if home is not None else (self.config.command_timeout if target.valid_for_s is None else target.valid_for_s)
            if home is None and lifetime > self.config.command_timeout:
                raise HTTPException(422, "valid_for_s exceeds command_timeout")
            deadline = target.deadline_s if target.deadline_s is not None else received_at+lifetime
            if deadline <= now:
                raise HTTPException(409, "Target expired in transit")
            if deadline > now + (home.timeout_s if home is not None else self.config.command_timeout):
                raise HTTPException(422, "Deadline exceeds command/action timeout in server monotonic time")
            command = _native.Command()
            if home is not None:
                command.finish_at_target = True
                command.action_timeout = home.timeout_s
                command.completion_joint_tolerance = home.joint_tolerance_rad
                command.completion_position_tolerance = home.position_tolerance_m
                command.completion_rotation_tolerance = home.rotation_tolerance_rad
                command.completion_settle_time = home.settle_time_s
            command.epoch = self.epoch
            command.sequence = self.accepted_seq+1
            if command.sequence >= 2**63:
                raise HTTPException(409, "Sequence range exhausted; restart the control session")
            if isinstance(target, CartesianTarget):
                command.mode = _native.Mode.Cartesian
                command.position = target.position
                command.twist = target.twist
                command.acceleration = target.acceleration
                if target.quaternion_xyzw is not None:
                    command.quaternion_xyzw = target.quaternion_xyzw
                    command.has_orientation = True
                if target.posture is not None:
                    command.posture = target.posture
                    command.has_posture = True
            else:
                command.mode = _native.Mode.Joint
                command.joints, command.joint_velocity = target.positions, target.velocities
            # A simple setpoint starts a clean inactive controller atomically:
            # native validation/preflight precede all control, and the target is
            # installed before the torque worker begins. Optional automatic
            # recovery is guarded natively and never replays an old target.
            running = self.backend.is_running() if hasattr(self.backend, "is_running") else self.backend.state()["running"]
            started = start_if_inactive and not running
            if started:
                self.epoch = self.backend.start(command, deadline, self.automatic_recovery, motion_generation=generation)
                self.record_operation("start_on_setpoint")
            else:
                self.backend.submit(command, deadline, motion_generation=generation)
            self.accepted_seq = command.sequence
            if target.seq is not None:
                self.simple_sequence = target.seq
            accepted_at = _native.monotonic_time()
            # Transport readback only: never a queue or a source of reissued commands.
            self.latest_target = {**target.model_dump(exclude={"valid_for_s"}),
                "seq": target.seq, "client_seq": target.seq, "command_seq": self.accepted_seq, "deadline_s": deadline,
                "mode": "cartesian" if isinstance(target, CartesianTarget) else "joint",
                "action": "go_home" if home is not None else "setpoint",
                "home_options": home.model_dump() if home is not None else None,
                "epoch": self.epoch, "accepted_monotonic_s": accepted_at,
                "accepted_timestamp_s": time.time()}
            inhibit = self.motion_inhibit_status()
            return {**deepcopy(self.latest_target), "accepted": True, "started_control": started,
                    'motion_inhibited': inhibit['motion_inhibited'],
                    'motion_inhibit_generation': inhibit['generation'],
                    "age_s": 0.0, "stop_latched": inhibit['generation'] != generation,
                    "server_monotonic_s": _native.monotonic_time()}

    def target(self, mode):
        """Latest accepted command, explicitly distinct from measured/reference state."""
        with self.lock:
            state = self.state()
            now = state["server_monotonic_s"]
            accepting = (state['motion_inhibited'] is False and state["running"]
                         and state["phase"] in ("armed", "active"))
            latest = self.latest_target
            selected = latest if latest is not None and latest["mode"] == mode else None
            fields = ({"position": None, "quaternion_xyzw": None, "twist": None,
                       "acceleration": None, "posture": None, "frame": "base"} if mode == "cartesian"
                      else {"positions": [0.0]*7, "velocities": [0.0]*7})
            if selected is not None and selected.get("action") == "backoff_joint_limits":
                # The native plan is separate telemetry, not a caller-supplied setpoint.
                fields = {"positions": None, "velocities": None}
            response = {**fields, "mode": mode, "has_target": selected is not None,
                "latest_mode": latest["mode"] if latest else None,
                "seq": None, "command_seq": None, "client_seq": None, "epoch": self.epoch, "deadline_s": None,
                "accepted_monotonic_s": None, "accepted_timestamp_s": 0.0,
                "age_s": 1_000_000.0, "expired": None, "valid": False, "applied": False,
                "applied_seq": state["applied_seq"], "phase": state["phase"],
                "stop_latched": not accepting, "server_monotonic_s": now}
            if selected is not None:
                response.update(deepcopy(selected))
                response["age_s"] = max(0, now-selected["accepted_monotonic_s"])
                response["expired"] = now >= selected["deadline_s"]
                response["valid"] = accepting and not response["expired"] and selected["epoch"] == self.epoch
                # Historical acknowledgement, not a reach/completion flag.
                response["applied"] = (state["epoch"] == selected["epoch"] and
                                       state["applied_seq"] == selected["command_seq"])
            return response

    def health(self):
        with self.lock:
            state = self.state()
            accepting = (state['motion_inhibited'] is False and state["running"]
                         and state["phase"] in ("armed", "active"))
            latest = self.latest_target
            result = {k: state[k] for k in ("backend", "robot_model", "read_only", "phase", "running",
                                            "reason", "error", "hardware_available")}
            result.update(motion_inhibited=state['motion_inhibited'],
                motion_inhibit_available=state['motion_inhibit_available'],
                motion_inhibit_generation=state['motion_inhibit_generation'],
                stop_latched=not accepting, has_target=latest is not None,
                target_valid=bool(latest and accepting and latest["epoch"] == self.epoch and
                                  latest["deadline_s"] > state["server_monotonic_s"]),
                accepted_seq=self.accepted_seq, applied_seq=state["applied_seq"],
                command_timeout_s=self.config.command_timeout, explicit_arm_required=False,
                start_on_setpoint=True, automatic_recovery_enabled=self.automatic_recovery,
                robot_connected=state["backend"] == "robot",
                robot_connection_status="handle_available_not_a_liveness_probe" if state["backend"] == "robot" else "simulation",
                last_control_error=state["error"] or (state["reason"] if state["phase"] == "fault" else None),
                max_joint_speed_rad_s=min(v or self.config.measured_joint_velocity_limit or 1.5*self.config.joint_velocity
                                          for v in self.config.measured_joint_velocity_limits),
                max_joint_speeds_rad_s=[v or self.config.measured_joint_velocity_limit or 1.5*self.config.joint_velocity
                                       for v in self.config.measured_joint_velocity_limits],
                supported_setpoint_modes=(["joint"] if self.config.compliance_mode == "joint_impedance"
                                           else ["cartesian"] if self.config.compliance_mode == "operational_space"
                                           else ["cartesian", "joint"]),
                reference_policy=state["reference_policy"], selected_control_law=state["selected_control_law"],
                homing_status=state.get("homing_status", "none"), home_reached=state.get("home_reached", False),
                mode_switching_enabled=state["mode_switching_enabled"],
                cartesian_reference_policy=(None if self.config.compliance_mode == "joint_impedance" else
                    "direct_equilibrium" if self.config.osc_feedback_mode != "linear" else "ruckig"),
                joint_reference_policy=("native_joint_limit_backoff" if state["joint_limit_backoff_active"] else
                    None if self.config.compliance_mode == "operational_space" else
                    "direct_joint_equilibrium" if self.config.compliance_mode in ("joint_impedance", "auto") else "ruckig"),
                server_monotonic_s=state["server_monotonic_s"])
            result.update({key: state[key] for key in CONTACT_RELIEF_STATE_DEFAULTS})
            result["contact_relief_available"] = state["contact_relief_available"]
            result.update({key: state[key] for key in JOINT_LIMIT_BACKOFF_STATE_DEFAULTS})
            result["joint_limit_backoff_available"] = state["joint_limit_backoff_available"]
            result["latest_action"] = latest.get("action") if latest else None
            return result

    def backoff_joint_limits(self, options: JointLimitBackoffRequest, motion_generation=None, *, received_at=None):
        from . import _native
        received_at = _native.monotonic_time() if received_at is None else received_at
        deadline = received_at + options.timeout_s
        generation = self._motion_generation(motion_generation)
        with self.lock:
            self._motion_generation(generation)
            state = self.state()
            if state["joint_limit_backoff_active"] or state["contact_relief_active"] or state.get("homing_active", False):
                raise HTTPException(409, "A finite action is already active; joint-limit backoff is not accepted")
            if not state["joint_limit_backoff_available"]:
                raise HTTPException(503, "Native joint-limit backoff support unavailable; update the inactive backend")
            if state["read_only"] or state["running"] or state["phase"] not in ("disarmed", "fault"):
                raise HTTPException(409, "Joint-limit backoff requires writable inactive Disarmed or a native-eligible local fault")
            if deadline <= _native.monotonic_time():
                raise HTTPException(409, "Joint-limit backoff request expired in transit")
            native_options = _native.JointLimitBackoffOptions()
            for name, value in options.model_dump(exclude={"timeout_s"}).items():
                setattr(native_options, name, value)
            # Native alone selects joints, captures measured holds and plans inward
            # targets from fresh state. Do not fill any target/feedforward fields.
            command = _native.Command()
            command.mode = _native.Mode.Joint
            command.backoff_joint_limits = True
            command.backoff_options = native_options
            command.action_timeout = options.timeout_s
            command.epoch = self.epoch
            command.sequence = self.accepted_seq + 1
            if command.sequence >= 2**63:
                raise HTTPException(409, "Sequence range exhausted; restart the control session")
            # Native decides local fault eligibility. This explicit action never
            # calls recovery or changes the global automatic-recovery policy.
            self.epoch = self.backend.start(command, deadline, False, motion_generation=generation)
            self.accepted_seq = command.sequence
            self.simple_sequence = None
            accepted_at = _native.monotonic_time()
            self.latest_target = {
                "action": "backoff_joint_limits", "mode": "joint",
                "backoff_options": options.model_dump(),
                "seq": None, "client_seq": None, "command_seq": self.accepted_seq,
                "epoch": self.epoch, "deadline_s": deadline,
                "accepted_monotonic_s": accepted_at, "accepted_timestamp_s": time.time(),
                "accepted_not_reached": True, "disarm_on_completion": True,
            }
            # Echo native diagnostics only; never infer selection or completion.
            native_state = self.state()
            self.latest_target.update({key: native_state[key] for key in JOINT_LIMIT_BACKOFF_STATE_DEFAULTS})
            LOG.info("Joint-limit backoff accepted: epoch=%s command_seq=%s", self.epoch, self.accepted_seq)
            inhibit = self.motion_inhibit_status()
            return {**deepcopy(self.latest_target), "accepted": True, "started_control": True,
                    "motion_inhibited": inhibit["motion_inhibited"],
                    "motion_inhibit_generation": inhibit["generation"],
                    "age_s": 0.0, "stop_latched": inhibit["generation"] != generation,
                    "server_monotonic_s": _native.monotonic_time()}

    def relieve_contact(self, options: ContactReliefRequest, motion_generation=None, *, received_at=None):
        from . import _native
        received_at = _native.monotonic_time() if received_at is None else received_at
        deadline = received_at + options.timeout_s
        generation = self._motion_generation(motion_generation)
        with self.lock:
            self._motion_generation(generation)
            state = self.state()
            if state["joint_limit_backoff_active"]:
                raise HTTPException(409, "Joint-limit backoff is active; contact relief is not accepted")
            if not state["contact_relief_available"]:
                raise HTTPException(503, "Native contact-relief support unavailable; update the inactive backend")
            if state["contact_relief_active"]:
                raise HTTPException(409, "Contact relief is already active")
            if state["read_only"] or state["running"] or state["phase"] != "disarmed":
                raise HTTPException(409, "Contact relief requires a writable, inactive Disarmed controller")
            if self.automatic_recovery:
                raise HTTPException(409, "Contact relief requires automatic_recovery OFF; disable it explicitly first")
            if deadline <= _native.monotonic_time():
                raise HTTPException(409, "Contact-relief request expired in transit")
            command = _native.Command()
            if not hasattr(command, "relieve_contact") or not hasattr(command, "relief_options"):
                raise HTTPException(503, "Native contact-relief command support unavailable; update the inactive backend")
            native_options = _native.ContactReliefOptions()
            if not hasattr(native_options, "manual_stop"):
                raise HTTPException(503, "Native manual contact-relief support unavailable; update the inactive backend")
            native_options.manual_stop = options.mode == "manual"
            # Absent manual bias stays UNUSED, not a synthetic zero calibration.
            # Absent box keeps the native manual-only zero/zero sentinel.
            for name, value in options.model_dump(
                    exclude={"mode", "timeout_s", "bias_source"}, exclude_none=True).items():
                setattr(native_options, name, value)
            # No Python pose capture/setpoint/servo: native captures the measured
            # pose at its first accepted callback. Leave other command fields alone.
            command.mode = _native.Mode.Cartesian
            command.relieve_contact = True
            command.relief_options = native_options
            command.action_timeout = options.timeout_s
            command.epoch = self.epoch
            command.sequence = self.accepted_seq + 1
            if command.sequence >= 2**63:
                raise HTTPException(409, "Sequence range exhausted; restart the control session")
            self.epoch = self.backend.start(command, deadline, False, motion_generation=generation)
            self.accepted_seq = command.sequence
            self.simple_sequence = None
            accepted_at = _native.monotonic_time()
            self.latest_target = {
                "action": "relieve_contact", "mode": "cartesian",
                "relief_options": options.model_dump(), "bias_source": options.bias_source,
                "automatic_release_detection": options.mode == "automatic",
                "seq": None, "client_seq": None, "command_seq": self.accepted_seq,
                "epoch": self.epoch, "deadline_s": deadline,
                "accepted_monotonic_s": accepted_at, "accepted_timestamp_s": time.time(),
                "accepted_not_released": True, "disarm_on_completion": True,
            }
            LOG.info("Contact relief accepted: epoch=%s command_seq=%s mode=%s bias_source=%r",
                     self.epoch, self.accepted_seq, options.mode, options.bias_source)
            inhibit = self.motion_inhibit_status()
            return {**deepcopy(self.latest_target), "accepted": True, "started_control": True,
                    "motion_inhibited": inhibit["motion_inhibited"],
                    "motion_inhibit_generation": inhibit["generation"],
                    "age_s": 0.0, "stop_latched": inhibit["generation"] != generation,
                    "server_monotonic_s": _native.monotonic_time()}

    def go_home(self, options: HomeRequest | None = None, motion_generation=None):
        from . import _native
        generation = self._motion_generation(motion_generation)
        options = options or HomeRequest()
        with self.lock:
            self._motion_generation(generation)
            state = self.state()
            if state["contact_relief_active"]:
                raise HTTPException(409, "Contact relief is active; homing is not accepted")
            if state["joint_limit_backoff_active"]:
                raise HTTPException(409, "Joint-limit backoff is active; homing is not accepted")
            latest = self.latest_target
            if (latest and latest.get("action") == "go_home" and state["running"] and
                    state["phase"] in ("armed", "active", "stopping")):
                if latest["home_options"] != options.model_dump():
                    raise HTTPException(409, "A home action is already running with different options; stop or supersede it first")
                return {**deepcopy(latest), "accepted": True, "already_in_progress": True,
                        "started_control": False, "homing_status": state.get("homing_status", "moving"),
                        "age_s": max(0, _native.monotonic_time()-latest["accepted_monotonic_s"]),
                        "stop_latched": state["phase"] == "stopping"}
            command = self.backend.home_command()
            if command.mode == _native.Mode.Joint:
                target = JointTarget(positions=command.joints.tolist())
            else:
                target = CartesianTarget(position=command.position.tolist(),
                    quaternion_xyzw=list(command.quaternion_xyzw), posture=command.posture.tolist())
            response = self.submit(target, home=options, motion_generation=generation)
            details = {"home_joint_positions": command.joints.tolist(), "home_kind": "nominal_franky_home",
                       "accepted_not_reached": True, "disarm_on_completion": True}
            self.latest_target.update(details)
            return {**response, **details, "already_in_progress": False}

    def status(self):
        # Dashboard polling never queues behind arm/recovery/configuration.
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Controller lifecycle operation in progress; retry status shortly")
        try:
            result = diagnostic_status(self.state(), self.config, deepcopy(self.last_operation))
            result["last_rejection"] = deepcopy(self.last_rejection)
            return result
        finally:
            self.lock.release()

    def refresh_state(self):
        # Native lifecycle lock rejects reads during control. This method issues
        # no setters, stop, recovery or new command, and preserves fault history.
        with self.lock:
            self.backend.refresh_state()
            return self.status()

    def record_operation(self, operation, error=None):
        entry = {"operation": operation, "error": error, "time_unix_s": time.time()}
        self.last_operation = entry
        if error:
            previous = self.last_rejection
            self.last_rejection = entry
            # Avoid flooding the terminal if a policy repeats the same invalid
            # request at publication rate; every rejection remains inspectable.
            if not previous or previous["operation"] != operation or previous["error"] != error:
                LOG.warning("Controller %s rejected: %s", operation, error)
        elif operation not in ("submit", "relieve_contact", "backoff_joint_limits"):
            # Finite actions log acceptance separately; only native state reports completion.
            LOG.info("Controller %s completed", operation)

    def recovery_policy(self):
        return {"enabled": self.automatic_recovery, "policy": "on_new_valid_setpoint_once",
                "affects_future_attempts_only": True}

    def set_automatic_recovery(self, enabled):
        # Do not let an OFF request wait silently behind a running recovery and
        # suggest that it cancelled it. Caller must see the busy response.
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Controller operation in progress; automatic recovery setting was not changed")
        try:
            state = self.backend.state()
            if enabled and state.get("contact_relief_active", False):
                raise HTTPException(409, "Automatic recovery cannot be enabled while contact relief is active")
            if enabled and state.get("joint_limit_backoff_active", False):
                raise HTTPException(409, "Automatic recovery cannot be enabled while joint-limit backoff is active")
            if enabled and state["read_only"]:
                raise RuntimeError("read-only connection: automatic recovery cannot be enabled")
            self.automatic_recovery = enabled
            LOG.info("Automatic recovery %s for future fresh setpoints", "enabled" if enabled else "disabled")
            return self.recovery_policy()
        finally:
            self.lock.release()

    def reset_command_timeout(self):
        return self.set_command_timeout(DEFAULT_COMMAND_TIMEOUT_S)

    def command_timeout(self):
        with self.lock:
            return {"command_timeout_s": self.config.command_timeout, "maximum_s": None,
                    "default_s": DEFAULT_COMMAND_TIMEOUT_S, "updates_require_inactive": True}

    def set_command_timeout(self, seconds):
        with self.lock:
            request = {"command_timeout_s": seconds, "time_unix_s": time.time(), "outcome": "requested"}
            self.last_timeout_request = request
            LOG.info("Command timeout requested: command_timeout_s=%r", seconds)
            try:
                self.configure(ControllerConfig.model_validate(self.config.model_dump() | {"command_timeout": seconds}))
            except Exception as error:
                self.last_timeout_request = {**request, "outcome": "rejected", "error": str(error)}
                raise
            self.last_timeout_request = {**request, "outcome": "applied"}
            if seconds > DEFAULT_COMMAND_TIMEOUT_S:
                LOG.warning("Command timeout set to %g s: the last target may remain active this long if the producer stalls", seconds)
            return self.command_timeout()

    def stop(self, wait_s=0.0):
        # Interrupt first, independently of an HTTP handler holding the session lock.
        self.backend.stop()
        with self.lock:
            self.simple_sequence = None
        return self._stop_result(wait_s)

    def _lifecycle_busy(self):
        acquired = self.lock.acquire(blocking=False)
        if acquired:
            self.lock.release()
        native_busy = getattr(self.backend, 'lifecycle_busy', lambda: False)()
        return not acquired or native_busy

    def _stop_result(self, wait_s, observed_after=None):
        end = time.monotonic() + wait_s
        is_running = getattr(self.backend, "is_running", lambda: self.backend.state()["running"])
        while (is_running() or self._lifecycle_busy()) and time.monotonic() < end:
            time.sleep(0.005)
        state = self.state()
        velocities = state.get("velocities", ())
        stationary = (state.get("sample_valid", False) and len(velocities) == 7 and
                      all(v is not None and math.isfinite(v) for v in velocities) and
                      max(abs(v) for v in velocities) < self.config.stop_velocity)
        robot_idle = (state.get("backend") != "robot" or
                      (state.get("robot_mode") == "Idle" and state.get("robot_has_errors") is False and
                       state.get("sample_age_s") is not None and 0 <= state["sample_age_s"] <= .05))
        if state.get('backend') == 'robot' and observed_after is not None:
            stamp = state.get('sample_timestamp_monotonic_s')
            robot_idle = robot_idle and stamp is not None and stamp >= observed_after
        busy = self._lifecycle_busy()
        stopped = not busy and not state["running"] and state["phase"] == "disarmed" and stationary and robot_idle
        return {"stop_requested": True, "stopped": stopped, "lifecycle_busy": busy,
                "timestamp_s": time.time(), "state": state}

    def _assert_motion_inhibit(self):
        # Never acquire the session/lifecycle lock before asserting. A producer
        # or SDK startup/recovery may hold it; the native latch is lock-free.
        if not hasattr(self.backend, 'inhibit_motion'):
            self.backend.stop()
            raise HTTPException(503, 'Native inhibit unavailable; ordinary stop requested only')
        from . import _native
        asserted_at = _native.monotonic_time()
        asserted = self.backend.inhibit_motion()
        return asserted_at, {**asserted, 'available': True, 'scope': 'backend_lifetime',
                             'release_token': f'{self._inhibit_instance}:{asserted["generation"]}'}

    def inhibit_motion(self, wait_s=0.0):
        if not math.isfinite(wait_s) or not 0 <= wait_s <= 5:
            raise ValueError('wait_s must be finite and in [0,5]')
        asserted_at, asserted = self._assert_motion_inhibit()
        return {**self._stop_result(wait_s, observed_after=asserted_at), **asserted}

    def release_motion_inhibit(self, token):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, 'Controller operation in progress; inhibit not released')
        try:
            status = self.motion_inhibit_status()
            if not status['available']:
                raise HTTPException(503, 'Native motion-inhibit support unavailable')
            if not status['motion_inhibited'] or token != status['release_token']:
                raise HTTPException(409, 'Stale release token or motion inhibit already released')
            self.backend.release_motion_inhibit(status['generation'])
            self.simple_sequence = None
            self.accepted_seq = 0
            self.latest_target = None
            return {**self.motion_inhibit_status(), 'released': True, 'motion_resumed': False}
        finally:
            self.lock.release()

    def recover(self, motion_generation=None):
        generation = self._motion_generation(motion_generation)
        with self.lock:
            self._motion_generation(generation)
            self.backend.recover(motion_generation=generation)
            self.simple_sequence = None
            self.latest_target = None
            return self.state()

    def configure(self, config):
        with self.lock:
            self.backend.configure(config.native())
            self.config = config
            self.simple_sequence = None
            self.latest_target = None
            return self.config


def _check_realtime_kernel():
    indicator = pathlib.Path("/sys/kernel/realtime")
    try:
        realtime = indicator.read_text().strip() == "1"
    except FileNotFoundError:
        # Linux emits this exact UTS_VERSION token for CONFIG_PREEMPT_RT.
        # A release name containing '-rt' or PREEMPT_DYNAMIC is insufficient.
        realtime = "PREEMPT_RT" in os.uname().version.split()
    if not realtime:
        raise RuntimeError("Hardware mode requires PREEMPT_RT (kernel RT indicator or PREEMPT_RT version flag).")
    from . import _native
    if not _native.has_realtime_kernel():
        raise RuntimeError("The linked libfranka does not recognize this RT kernel; rebuild with the local RT detection patch.")


def create_app(
    *, config: ControllerConfig | None = None, backend_kind: str = "sim",
    hardware: HardwareSettings | None = None, api_token: str | None = None,
    enable_sim_injection: bool = False, status_interval_s: float = 0.0,
    automatic_recovery: bool = True,
) -> FastAPI:
    if not math.isfinite(status_interval_s) or status_interval_s < 0:
        raise ValueError("status_interval_s must be finite and nonnegative")
    if backend_kind == "robot" and config is None:
        raise ValueError("Hardware mode requires an explicit controller configuration")
    config = config or ControllerConfig()
    if backend_kind not in {"sim", "robot"}:
        raise ValueError("backend_kind must be sim or robot")
    if backend_kind == "robot" and hardware is None:
        raise ValueError("Hardware mode requires hardware settings")
    if api_token is not None and (not api_token or (backend_kind == "robot" and len(api_token) < 24)):
        raise ValueError("An explicitly enabled API token must be nonempty (at least 24 characters for hardware)")
    if backend_kind == "robot" and hardware.robot_model != config.robot_model:
        raise ValueError("Hardware and controller robot_model must match")
    if enable_sim_injection and backend_kind != "sim":
        raise ValueError("Wrench injection is only supported in simulation")

    @asynccontextmanager
    async def lifespan(app):
        from . import _native
        if backend_kind == "robot":
            _check_realtime_kernel()
        backend = _native.Backend(config.native(), backend_kind,
                                  hardware.native() if hardware else _native.HardwareConfig())
        app.state.controller = ControllerService(backend, config, automatic_recovery=automatic_recovery)
        async def report_status():
            while True:
                try:
                    view = app.state.controller.status()
                    LOG.info("%s", terminal_status(view))
                except Exception as exc:
                    LOG.info("Controller status unavailable: %s", exc)
                await asyncio.sleep(status_interval_s)
        reporter = asyncio.create_task(report_status()) if status_interval_s else None
        try:
            yield
        finally:
            if reporter is not None:
                reporter.cancel()
                try:
                    await reporter
                except asyncio.CancelledError:
                    pass
            # pybind releases the GIL; the native watchdog already covers Python failure.
            backend.close()

    # Default launchers are unauthenticated, matching franky_service. Retain
    # explicit factory opt-in for existing embedders, but do not advertise an
    # OpenAPI security requirement when authentication is disabled.
    dependencies = []
    if api_token is not None:
        bearer = HTTPBearer(auto_error=False)

        async def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
            if credentials is None or not hmac.compare_digest(credentials.credentials, api_token):
                raise HTTPException(401, "Bearer token required", headers={"WWW-Authenticate": "Bearer"})

        dependencies.append(Depends(authenticate))

    app = FastAPI(title="Franka Compliant Control", version="0.1.0", lifespan=lifespan,
                  dependencies=dependencies)

    @app.middleware('http')
    async def capture_motion_generation(request: Request, call_next):
        # Capture before body parsing/dependencies/thread-pool queueing, so a
        # request admitted before/during a latch cycle cannot start after release.
        if request.method == 'POST' and request.url.path in {
            '/target_ee_pose', '/target_joint_state', '/go_home', '/relieve_contact', '/backoff_joint_limits', '/recover'
        }:
            if request.url.path in {'/relieve_contact', '/backoff_joint_limits'}:
                from . import _native
                request.state.received_monotonic_s = _native.monotonic_time()
            request.state.motion_generation = app.state.controller.motion_inhibit_status()['generation']
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc):
        # Never print raw inputs: they can include credentials or NaN/Inf.
        summary = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        if _request.url.path == "/command_timeout":
            body = exc.body if isinstance(exc.body, dict) else {}
            value = body.get("command_timeout_s")
            # Only this non-secret numeric setting is logged, never arbitrary
            # strings, extra fields, credentials, or a complete request body.
            rendered = repr(value)[:80] if isinstance(value, (int, float)) else f"<{type(value).__name__}>"
            summary = f"requested command_timeout_s={rendered}; {summary}"
            app.state.controller.last_timeout_request = {"requested_value": rendered,
                "time_unix_s": time.time(), "outcome": "rejected", "error": summary}
        app.state.controller.record_operation(_request.url.path, summary)
        # The rejected input may contain NaN/Inf and cannot be echoed in JSON.
        return JSONResponse(status_code=422, content={"detail": [
            {k: e[k] for k in ("loc", "msg", "type")} for e in exc.errors()
        ]})

    def engine():
        return app.state.controller

    def invoke(fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if fn.__name__ not in ("submit", "status"):
                engine().record_operation(fn.__name__)
            return result
        except HTTPException as exc:
            engine().record_operation(fn.__name__, str(exc.detail))
            raise
        except (ValueError, RuntimeError) as exc:
            if fn.__name__ != "status":
                engine().record_operation(fn.__name__, str(exc))
            raise HTTPException(422 if isinstance(exc, ValueError) else 409, str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(pathlib.Path(__file__).with_name("dashboard.html").read_text(),
                            headers={"Cache-Control": "no-store"})

    @app.get("/about", response_class=HTMLResponse)
    def about(format: Literal["html", "json"] = "html"):
        catalog = endpoint_catalog(app, engine().config, backend_kind=backend_kind,
                                   automatic_recovery=engine().automatic_recovery,
                                   authentication_enabled=api_token is not None)
        if format == "json":
            return JSONResponse(catalog, headers={"Cache-Control": "no-store"})
        return HTMLResponse(render_about(catalog), headers={"Cache-Control": "no-store"})

    @app.get("/urdf")
    def urdf():
        model = engine().backend.model_urdf()
        if not model:
            raise HTTPException(404, "Robot URDF is unavailable for this backend/SDK; no saved-model fallback is substituted")
        return {"urdf": model}

    @app.get("/status")
    def status():
        return {**invoke(engine().status), "authentication_enabled": api_token is not None}

    @app.post("/state/refresh")
    def refresh_state():
        return invoke(engine().refresh_state)

    @app.get("/health")
    def health():
        return {**engine().health(), "authentication_enabled": api_token is not None}

    @app.get("/state")
    def state():
        return engine().state()

    @app.post('/telemetry/prepare')
    def telemetry_prepare(payload: TelemetryPrepare):
        with engine().lock:
            invoke(engine().backend.prepare_telemetry,payload.capacity,[payload.joint_speed,payload.linear_speed,payload.angular_speed])
            return engine().backend.telemetry_status()

    @app.get('/telemetry/status')
    def telemetry_status():
        return engine().backend.telemetry_status()

    @app.get('/telemetry/data')
    def telemetry_data():
        # Native export refuses while running. Serialization happens only here,
        # outside RT and under the producer/lifecycle lock.
        import io
        import numpy as np
        from .native_telemetry import metadata
        with engine().lock:
            arrays=invoke(engine().backend.telemetry_data)
            info=metadata(engine().backend.telemetry_status())
            stream=io.BytesIO()
            import json
            np.savez_compressed(stream,**arrays,metadata_json=np.array(json.dumps(info)))
            return Response(stream.getvalue(),media_type='application/octet-stream',
                headers={'Content-Disposition':f'attachment; filename="native-{info["capture_id"]}.npz"','Cache-Control':'no-store'})

    @app.post('/telemetry/clear')
    def telemetry_clear(payload: TelemetryClear):
        with engine().lock:
            invoke(engine().backend.clear_telemetry,payload.capture_id)
            return engine().backend.telemetry_status()

    @app.get("/joint_state")
    def joint_state():
        s = engine().state()
        return {k: s[k] for k in ("positions", "velocities", "sample_age_s", "sample_valid",
                                 "phase", "running", "reason", "epoch", "applied_seq",
                                 "observation_source", "server_monotonic_s")}

    @app.get("/ee_pose")
    def ee_pose():
        s = engine().state()
        return {k: s[k] for k in ("position", "quaternion_xyzw", "twist", "kinematics_valid",
                                 "sample_age_s", "sample_valid", "phase", "running", "reason",
                                 "epoch", "applied_seq", "observation_source", "frame", "server_monotonic_s")}

    @app.get("/target_ee_pose")
    def get_target_ee():
        return engine().target("cartesian")

    @app.get("/target_joint_state")
    def get_target_joint():
        return engine().target("joint")

    @app.get("/command_timeout")
    def command_timeout():
        return engine().command_timeout()

    @app.post("/command_timeout")
    def set_command_timeout(payload: CommandTimeoutUpdate):
        return invoke(engine().set_command_timeout, payload.command_timeout_s)

    @app.post("/command_timeout/reset")
    def reset_command_timeout():
        return invoke(engine().reset_command_timeout)

    @app.get("/automatic_recovery")
    def automatic_recovery_policy():
        return engine().recovery_policy()

    @app.post("/automatic_recovery")
    def set_automatic_recovery(payload: AutomaticRecoveryUpdate):
        return invoke(engine().set_automatic_recovery, payload.enabled)

    @app.get("/controller_config", response_model=ControllerConfig)
    def get_config():
        return engine().config

    @app.put("/controller_config", response_model=ControllerConfig)
    def set_config(payload: ControllerConfig):
        return invoke(engine().configure, payload)

    @app.post("/target_ee_pose", status_code=202)
    def target_ee(payload: CartesianTarget, request: Request):
        return invoke(engine().submit, payload, motion_generation=request.state.motion_generation)

    @app.post("/target_joint_state", status_code=200)
    def target_joint(payload: JointTarget, request: Request):
        return invoke(engine().submit, payload, motion_generation=request.state.motion_generation)

    @app.post("/stop")
    def stop(wait_s: Annotated[float, Query(ge=0, le=5)] = 0.0):
        return invoke(engine().stop, wait_s)

    @app.get('/motion_inhibit')
    async def motion_inhibit_status():
        return engine().motion_inhibit_status()

    @app.post('/motion_inhibit')
    async def inhibit_motion(wait_s: Annotated[float, Query(ge=0, le=5)] = 0.0):
        # Assertion/authentication must not queue behind the sync worker pool:
        # those workers may all be waiting on a slow publisher/startup operation.
        asserted_at, asserted = invoke(engine()._assert_motion_inhibit)
        end = time.monotonic() + wait_s
        while (engine().backend.is_running() or engine()._lifecycle_busy()) and time.monotonic() < end:
            await asyncio.sleep(.005)
        return {**engine()._stop_result(0, observed_after=asserted_at), **asserted}

    @app.post('/motion_inhibit/release')
    async def release_motion_inhibit(payload: MotionInhibitRelease):
        return invoke(engine().release_motion_inhibit, payload.release_token)

    @app.post("/recover")
    def recover(request: Request):
        return invoke(engine().recover, request.state.motion_generation)

    @app.post("/go_home")
    def go_home(request: Request, payload: HomeRequest | None = None):
        return invoke(engine().go_home, payload, request.state.motion_generation)

    @app.post("/relieve_contact", status_code=202)
    def relieve_contact(payload: ContactReliefRequest, request: Request):
        return invoke(engine().relieve_contact, payload,
                      motion_generation=request.state.motion_generation,
                      received_at=request.state.received_monotonic_s)

    @app.post("/backoff_joint_limits", status_code=202)
    def backoff_joint_limits(payload: JointLimitBackoffRequest, request: Request):
        return invoke(engine().backoff_joint_limits, payload,
                      motion_generation=request.state.motion_generation,
                      received_at=request.state.received_monotonic_s)

    if enable_sim_injection:
        @app.post("/sim/wrench")
        def inject(payload: WrenchInjection):
            invoke(engine().backend.inject_wrench, payload.wrench)
            return {"injected": payload.wrench}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sim", "robot"], default="sim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--config", type=pathlib.Path, help="Controller configuration JSON")
    parser.add_argument("--hardware-config", type=pathlib.Path, help="Hardware JSON; read_only permits unconfirmed settings")
    parser.add_argument("--enable-sim-injection", action="store_true")
    parser.add_argument("--status-interval", type=float, default=2.0, help="Terminal snapshot interval, seconds; 0 disables")
    parser.add_argument("--no-auto-recover", action="store_true", help="Require manual recovery instead of one guarded attempt on a fresh setpoint")
    args = parser.parse_args()
    config = ControllerConfig.model_validate_json(args.config.read_text()) if args.config else ControllerConfig()
    hardware = HardwareSettings.model_validate_json(args.hardware_config.read_text()) if args.hardware_config else None
    if args.backend == "robot" and args.config is None:
        parser.error("hardware mode requires an explicit --config file")
    app = create_app(config=config, backend_kind=args.backend, hardware=hardware,
                     enable_sim_injection=args.enable_sim_injection, status_interval_s=args.status_interval,
                     automatic_recovery=not args.no_auto_recover)
    try:
        run_http_server(app, host=args.host, port=args.port)
    except RuntimeError as error:
        parser.exit(1, f"Controller startup failed: {error}\n")


if __name__ == "__main__":
    main()
