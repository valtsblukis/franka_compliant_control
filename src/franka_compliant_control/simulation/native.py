"""Layer 1: production native controller + SDK model/filter + reusable physics.

No Gym, task, reward, goal sampler, obstacle knowledge or robot transport.
Pose references are base-frame, quaternion xyzw, with zero twist/acceleration.
Optional joint references exercise the same native joint-impedance mode.
"""

from __future__ import annotations
from collections import deque
import heapq
import mujoco
import numpy as np
from .. import _native, _twin_native
from ..models import ControllerConfig
from .sdk import offline_model, native_inputs_digest
from ..twin_model import quaternion_matrix
from ..wrench_model import tcp_to_spatial
from .contacts import ContactReport
from .types import RobotMeasurement, ControlStep


class NativeControllerSimulation:
    control_dt = 0.001

    def __init__(self, world, *, sensor=None, kinematic_sensor=None):
        if native_inputs_digest() != _twin_native.twin_inputs_sha256:
            raise RuntimeError(
                "Offline native bridge is stale; hardware backend is not rebuilt here"
            )
        self.world = world
        self.home = world.home
        self.sensor = sensor
        self.kinematic_sensor = kinematic_sensor
        self.config = ControllerConfig.model_validate(self.home["controller_config"])
        self.sdk = offline_model(self.home)
        self.shadow = mujoco.MjData(world.model)
        self.reset()

    @property
    def now(self):
        return self.world.time_s

    def reset(self, *, q=None, dq=None, previous_torque=None, seed=0, sensor_seed=0):
        self.world.reset(q=q, dq=dq, seed=seed)
        mujoco.mj_resetData(self.world.model, self.shadow)
        self.previous = (
            np.zeros(7)
            if previous_torque is None
            else np.asarray(previous_torque, dtype=float).copy()
        )
        if self.previous.shape != (7,) or not np.isfinite(self.previous).all():
            raise ValueError("Invalid initial torque-filter state")
        self.applied = self.previous.copy()
        self.delay = deque(
            [np.zeros(7) for _ in range(self.world.parameters.actuator_delay_steps)]
        )
        self.phase = "active"
        self.reason = "none"
        self.pending = []
        self.sequence = 1
        self.last_output = None
        if self.sensor is not None:
            self.sensor.reset(sensor_seed)
        if self.kinematic_sensor is None:
            initial = self.sdk.sample(
                self.world.q, self.world.dq, self.previous, np.zeros(6)
            )
            self.controller = _twin_native.OfflineController(
                self.config.native(), initial, 1, 0.0
            )
            self.observe(self.control_dt)
        else:
            self.kinematic_sensor.reset(sensor_seed + 293)
            self.observe(self.control_dt)
            self.controller = _twin_native.OfflineController(
                self.config.native(), self.sample, 1, 0.0
            )
        self.command = self._command(
            self.sample.position, self.sample.quaternion_xyzw, 1
        )
        self.deadline = min(0.2, self.config.command_timeout)
        self.last_step_contacts = ()
        return self.measurement()

    @staticmethod
    def _command(position, quaternion, sequence):
        p = np.asarray(position, dtype=float)
        q = np.asarray(quaternion, dtype=float)
        if (
            p.shape != (3,)
            or q.shape != (4,)
            or not np.isfinite(np.r_[p, q]).all()
            or abs(np.linalg.norm(q) - 1) > 1e-5
        ):
            raise ValueError("Expected finite base position and unit xyzw quaternion")
        command = _native.Command()
        command.epoch = 1
        command.sequence = sequence
        command.mode = _native.Mode.Cartesian
        command.position = p
        command.has_orientation = True
        command.quaternion_xyzw = q
        return command

    def submit_pose(self, position, quaternion_xyzw, *, valid_for_s=0.2, delay_steps=0):
        command = self._command(position, quaternion_xyzw, self.sequence + 1)
        return self._submit(command, valid_for_s, delay_steps)

    def submit_joint(self, positions, *, valid_for_s=0.2, delay_steps=0):
        q = np.asarray(positions, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected seven finite joint references")
        if np.any(q < self.config.joint_lower) or np.any(q > self.config.joint_upper):
            raise ValueError("Joint reference outside configured limits")
        command = _native.Command()
        command.epoch = 1
        command.sequence = self.sequence + 1
        command.mode = _native.Mode.Joint
        command.joints = q
        return self._submit(command, valid_for_s, delay_steps)

    def _submit(self, command, valid_for_s, delay_steps):
        if self.phase != "active":
            raise RuntimeError("Stopped simulation requires reset")
        if (
            not np.isfinite(valid_for_s)
            or not 0 < valid_for_s <= self.config.command_timeout
        ):
            raise ValueError("Invalid command lease")
        if type(delay_steps) is not int or not 0 <= delay_steps <= 100:
            raise ValueError("Command delay must be 0..100 control ticks")
        self.sequence += 1
        heapq.heappush(
            self.pending,
            (
                self.now + delay_steps * self.control_dt,
                self.sequence,
                command,
                self.now + valid_for_s,
            ),
        )
        return self.sequence

    def observe(self, dt):
        w = self.world
        q = w.q
        dq = w.dq
        if self.kinematic_sensor is not None:
            q, dq = self.kinematic_sensor.observe(q, dq, dt)
        self.sample = self.sdk.sample(q, dq, self.previous, np.zeros(6))
        self.twist = self.sample.jacobian @ dq
        self.gravity = self.sdk.gravity(q)
        scaled = self.sample.jacobian.copy()
        scaled[3:] *= self.config.jacobian_rotation_scale
        self.sigma = float(np.linalg.svd(scaled, compute_uv=False)[-1])
        self.reading = None
        if self.sensor is not None:
            w.copy_kinematic_state(self.shadow)
            scene_robot_torque = w.fill_forces(self.shadow, self.applied + self.gravity)
            mujoco.mj_forward(w.model, self.shadow)
            contacts = w.oracle.read(self.shadow)
            external = (
                np.asarray(contacts.robot_on_environment_torque) - scene_robot_torque
            )
            total = (
                self.previous
                - self.sample.mass @ self.shadow.qacc[w.robot_dofs]
                - self.sample.coriolis
            )
            self.reading = self.sensor.observe(
                self.sample, total - external, external, dt
            )
            self.sample.external_wrench = tcp_to_spatial(
                self.reading.tcp_base, self.sample.position
            )
            self.sample.external_wrench_tcp_base = self.reading.tcp_base
            self.sample.wrench_tcp_valid = True
            self.sample.external_torque = self.reading.joint_torques
            r = quaternion_matrix(self.sample.quaternion_xyzw)
            self.sample.external_wrench_stiffness = np.r_[
                r.T @ self.reading.tcp_base[:3], r.T @ self.reading.tcp_base[3:]
            ]
        return self.measurement()

    def measurement(self):
        reading = self.reading
        return RobotMeasurement(
            self.now,
            tuple(self.sample.q),
            tuple(self.sample.dq),
            tuple(self.sample.position),
            tuple(self.sample.quaternion_xyzw),
            tuple(self.twist),
            tuple(reading.tcp_base) if reading else (0.0,) * 6,
            reading.valid if reading else False,
            reading.source if reading else "unmodeled_not_force_truth",
            reading.extrapolated if reading else False,
            self.sigma,
        )

    def step_control(self, *, stop=False, physics_guard=None):
        """One native tick; optional consumer guard runs after EVERY physics step.

        physics_guard(world, report) returns a stop reason or None. A stop ends
        this rollout; proprietary physical Franka braking is not simulated.
        """
        if self.phase != "active":
            raise RuntimeError("Stopped simulation requires reset")
        while self.pending and self.pending[0][0] <= self.now + 1e-12:
            _, seq, command, deadline = heapq.heappop(self.pending)
            if seq <= self.command.sequence or deadline <= self.now:
                self.world.emit(
                    "command_rejected", sequence=seq, reason="reordered_or_expired"
                )
                continue
            self.command = command
            self.deadline = deadline
        out = self.controller.step(
            self.sample, self.command, self.now, self.control_dt, self.deadline, stop
        )
        self.last_output = out
        if out.phase != "active":
            self.phase = out.phase
            self.reason = out.reason
            return ControlStep(self.measurement(), self.phase, self.reason, ())
        filtered = _twin_native.sdk_torque_filter(out.torque, self.previous)
        self.delay.append(filtered.copy())
        applied = self.delay.popleft()
        reports = []
        try:
            for _ in range(round(self.control_dt / self.world.physics_dt)):
                report = self.world.step_physics(applied + self.gravity)
                reports.append((self.world.physics_dt, report))
                reason = (
                    physics_guard(self.world, report)
                    if physics_guard is not None
                    else None
                )
                if reason:
                    self.phase = "stopped"
                    self.reason = str(reason)
                    break
            self.previous = filtered
            self.applied = applied
            self.observe(sum(dt for dt, _ in reports))
        except (RuntimeError, ValueError) as error:
            self.phase = "stopped"
            self.reason = str(error)
            raise
        self.last_step_contacts = tuple(reports)
        return ControlStep(
            self.measurement(), self.phase, self.reason, self.last_step_contacts
        )

    def manifest(self):
        return {
            "schema": "fcc-native-simulation-v1",
            "world": self.world.manifest(),
            "controller_config": self.config.model_dump(),
            "native_inputs_sha256": _twin_native.twin_inputs_sha256,
            "controller_source_sha256": _twin_native.controller_source_sha256,
            "sensor": self.sensor.manifest() if self.sensor else None,
            "kinematic_sensor": (
                self.kinematic_sensor.manifest() if self.kinematic_sensor else None
            ),
            "native_period_s": self.control_dt,
            "action": "absolute pose (zero twist/acceleration) or joint position (zero velocity feedforward)",
            "command_lifetime": "from submission; delivery delay never extends lease",
            "hardware_IO": False,
            "physical_stop_dynamics_modeled": False,
        }
