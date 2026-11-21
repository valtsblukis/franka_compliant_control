"""Wire schemas. All distances are metres, angles radians, times seconds."""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import DEFAULT_COMMAND_TIMEOUT_S

Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Vec4 = Annotated[list[float], Field(min_length=4, max_length=4)]
Vec6 = Annotated[list[float], Field(min_length=6, max_length=6)]
Vec7 = Annotated[list[float], Field(min_length=7, max_length=7)]
Positive = Annotated[float, Field(gt=0)]
RobotModel = Literal["panda", "fr3"]
FR3_LOWER = [-2.3476, -1.5454, -2.4937, -2.7714, -2.5100, 0.7773, -2.7045]
FR3_UPPER = [2.3476, 1.5454, 2.4937, -0.4226, 2.5100, 4.2841, 2.7045]
FR3_PHYSICAL_LOWER = [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, .4398, -3.0508]
FR3_PHYSICAL_UPPER = [2.9007, 1.8361, 2.9007, -.1169, 2.8763, 4.6216, 3.0508]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class ControllerConfig(Schema):
    """Initial tuning values, NOT a validated safe operating envelope."""

    robot_model: RobotModel = "panda"
    compliance_mode: Literal["contact_yielding", "continuous_impedance", "operational_space", "joint_impedance", "auto"] = "contact_yielding"
    motion_limit_policy: Literal["configured", "franka"] = "configured"
    bounded_cartesian_spring: bool = False
    osc_translation_kp: Vec3 = [25.0] * 3
    osc_translation_kd: Vec3 = [10.0] * 3
    osc_rotation_kp: Vec3 = [25.0] * 3
    osc_rotation_kd: Vec3 = [10.0] * 3
    osc_translation_stiffness: Vec3 = [0.0] * 3
    osc_translation_damping: Vec3 = [0.0] * 3
    osc_rotation_stiffness: Vec3 = [0.0] * 3
    osc_rotation_damping: Vec3 = [0.0] * 3
    osc_regularization: Positive = 0.02
    effort_limit_policy: Literal["configured", "franka", "task_bounded", "joint_bounded", "mode_bounded"] = "configured"
    osc_feedback_mode: Literal["linear", "progressive", "saturated"] = "linear"
    progressive_position_soft_radius: Positive = .01
    progressive_position_saturation_radius: Positive = .10
    progressive_rotation_soft_radius: Positive = math.radians(3)
    progressive_rotation_saturation_radius: Positive = math.radians(20)
    progressive_translation_soft_stiffness: Annotated[float, Field(ge=0)] = 50.0
    progressive_rotation_soft_stiffness: Annotated[float, Field(ge=0)] = 3.0
    collision_policy: Literal["configured", "robot", "robot_configured"] = "configured"
    translation_stiffness: Vec3 = [400.0] * 3
    rotation_stiffness: Vec3 = [15.0] * 3
    translation_damping: Vec3 = [40.0] * 3
    rotation_damping: Vec3 = [5.0] * 3
    joint_stiffness: Vec7 = [100.0] * 7
    joint_damping: Vec7 = [20.0] * 7
    # f [Nm] and epsilon [rad/s]; planned velocity is target error / 100 ms.
    friction_compensation_torque: Annotated[float, Field(ge=0)] = 0.0
    friction_compensation_epsilon: Positive = 0.05
    # First-order time constant. Zero preserves the unfiltered law.
    friction_compensation_filter_time_s: Annotated[float, Field(ge=0)] = 0.0
    joint_feedback_torque_limits: Vec7 = [8.0, 8.0, 8.0, 8.0, 3.0, 3.0, 3.0]
    stop_damping: Vec7 = [12.0] * 7
    joint_lower: Vec7 = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    joint_upper: Vec7 = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
    torque_limits: Vec7 = [20.0, 20.0, 20.0, 20.0, 8.0, 8.0, 8.0]
    external_torque_limits: Vec7 = [8.0] * 7
    commanded_force_limit: Positive = 10.0
    commanded_moment_limit: Positive = 1.0
    contact_force_soft: Positive = 8.0
    contact_force_stop: Positive = 15.0
    contact_moment_soft: Positive = 0.8
    contact_moment_stop: Positive = 2.0
    translation_velocity: Positive = 0.10
    translation_acceleration: Positive = 0.5
    translation_jerk: Positive = 5.0
    rotation_velocity: Positive = 0.5
    rotation_acceleration: Positive = 1.0
    rotation_jerk: Positive = 10.0
    joint_velocity: Positive = 0.2
    joint_acceleration: Positive = 0.5
    joint_jerk: Positive = 5.0
    # Zero uses the original 1.5 x reference speed trip; positive values are
    # independently configured measured-motion limits, still enforced natively.
    measured_translation_velocity_limit: Annotated[float, Field(ge=0)] = 0.0
    measured_rotation_velocity_limit: Annotated[float, Field(ge=0)] = 0.0
    measured_joint_velocity_limit: Annotated[float, Field(ge=0)] = 0.0
    # Positive entries override the shared setting; zero entries inherit it.
    measured_joint_velocity_limits: Vec7 = [0.0] * 7
    position_error_limit: Positive = 0.03
    rotation_error_limit: Positive = 0.15
    joint_error_limit: Positive = 0.15
    posture_stiffness: Annotated[float, Field(ge=0)] = 5.0
    posture_damping: Annotated[float, Field(ge=0)] = 3.0
    nullspace_torque_limit: Annotated[float, Field(ge=0)] = 2.0
    torque_rate_limit: Positive = 100.0
    command_timeout: Positive = DEFAULT_COMMAND_TIMEOUT_S
    stop_velocity: Positive = 0.02
    stop_settle_time: Positive = 0.1
    stop_timeout: Positive = 2.0
    joint_limit_margin: Annotated[float, Field(ge=0)] = 0.05
    minimum_singular_value: Positive = 0.02
    jacobian_rotation_scale: Positive = 0.2
    maximum_control_period: Annotated[float, Field(gt=0, le=0.02)] = 0.01

    @model_validator(mode="before")
    @classmethod
    def model_defaults(cls, data):
        if isinstance(data, dict) and data.get("robot_model") == "fr3":
            data = dict(data)
            full = data.get("motion_limit_policy") == "franka"
            data.setdefault("joint_lower", (FR3_PHYSICAL_LOWER if full else FR3_LOWER).copy())
            data.setdefault("joint_upper", (FR3_PHYSICAL_UPPER if full else FR3_UPPER).copy())
        return data

    @model_validator(mode="after")
    def consistent(self):
        for name in (
            "translation_stiffness", "rotation_stiffness", "translation_damping",
            "rotation_damping", "joint_stiffness", "joint_damping", "stop_damping",
            "torque_limits", "external_torque_limits",
        ):
            if any(v <= 0 for v in getattr(self, name)):
                raise ValueError(f"{name} must contain positive values")
        if self.contact_force_soft >= self.contact_force_stop:
            raise ValueError("contact_force_soft must be below contact_force_stop")
        if self.contact_moment_soft >= self.contact_moment_stop:
            raise ValueError("contact_moment_soft must be below contact_moment_stop")
        if self.stop_timeout <= self.stop_settle_time:
            raise ValueError("stop_timeout must exceed stop_settle_time")
        if any(hi - lo <= 2 * self.joint_limit_margin for lo, hi in zip(self.joint_lower, self.joint_upper)):
            raise ValueError("joint bounds must accommodate the joint margin")
        # One authority for model envelope validation in Python and direct native
        # calls, including position-dependent FR3 velocity constraints.
        self.native()
        return self

    def native(self):
        from . import _native
        result = _native.Config(self.robot_model)
        values = self.model_dump()
        # A staged controller update must not break the installed default-off
        # service. Older native builds lack these two fields; never silently
        # accept a request to enable compensation on such a build.
        missing_friction_fields = [name for name in (
            "friction_compensation_torque", "friction_compensation_epsilon"
        ) if not hasattr(result, name)]
        if missing_friction_fields:
            if self.friction_compensation_torque != 0:
                raise ValueError("Friction compensation requires a rebuilt native module; "
                                 "keep friction_compensation_torque=0 until deployment")
            for name in missing_friction_fields:
                del values[name]
        if not hasattr(result, "friction_compensation_filter_time_s"):
            if self.friction_compensation_torque != 0 and self.friction_compensation_filter_time_s != 0:
                raise ValueError("Filtered friction compensation requires a rebuilt native module; "
                                 "keep compensation off or filter_time_s=0 until deployment")
            del values["friction_compensation_filter_time_s"]
        for name, value in values.items():
            setattr(result, name, value)
        result.validate()
        return result


class EndEffectorSettings(Schema):
    mass: Annotated[float, Field(ge=0, le=3)]
    com: Vec3
    inertia: Annotated[list[float], Field(min_length=9, max_length=9)]

    @model_validator(mode="after")
    def physical_inertia(self):
        import numpy as np
        inertia = np.asarray(self.inertia).reshape(3, 3, order="F")
        eigenvalues = np.linalg.eigvalsh(inertia)
        if (not np.allclose(inertia, inertia.T, atol=1e-9) or eigenvalues[0] < -1e-9
                or eigenvalues[2] > eigenvalues[0] + eigenvalues[1] + 1e-9):
            raise ValueError("end-effector inertia must be physically consistent")
        return self


class HardwareSettings(Schema):
    robot_ip: Annotated[str, Field(min_length=1)]
    robot_model: RobotModel
    flange_to_tcp: Annotated[list[float], Field(min_length=16, max_length=16)]
    load_mass: Annotated[float, Field(ge=0, le=3)]
    load_com: Vec3
    load_inertia: Annotated[list[float], Field(min_length=9, max_length=9)]
    read_only: bool = False
    calibration_confirmed: bool = False
    configuration_basis: Literal["measured", "manufacturer_nominal", "estimated"] = "measured"
    configuration_source: Annotated[str, Field(min_length=1)] | None = None
    expected_end_effector: EndEffectorSettings | None = None

    @model_validator(mode="after")
    def rigid_transform_and_inertia(self):
        provisional = self.configuration_basis in ("manufacturer_nominal", "estimated")
        if provisional and (not self.configuration_source or not self.configuration_source.strip()
                            or self.expected_end_effector is None):
            raise ValueError("Provisional settings require a source and expected_end_effector for robot readback")
        if provisional and self.calibration_confirmed:
            raise ValueError("Provisional settings must not be labeled as measured calibration")
        if not self.read_only and not (self.calibration_confirmed or provisional):
            raise ValueError("Unconfirmed tool settings require read_only=true")
        import numpy as np
        t = np.asarray(self.flange_to_tcp).reshape(4, 4, order="F")
        if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7):
            raise ValueError("flange_to_tcp must be a column-major rigid transform")
        r = t[:3, :3]
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or abs(np.linalg.det(r) - 1) > 1e-6:
            raise ValueError("flange_to_tcp rotation must be orthonormal with determinant +1")
        inertia = np.asarray(self.load_inertia).reshape(3, 3, order="F")
        if not np.allclose(inertia, inertia.T, atol=1e-9):
            raise ValueError("load inertia must be symmetric")
        eigenvalues = np.linalg.eigvalsh(inertia)
        if eigenvalues[0] < -1e-9 or eigenvalues[2] > eigenvalues[0] + eigenvalues[1] + 1e-9:
            raise ValueError("load inertia must be physically consistent")
        if self.load_mass == 0 and (np.linalg.norm(inertia) > 1e-9 or np.linalg.norm(self.load_com) > 1e-9):
            raise ValueError("zero load mass requires zero centre of mass and inertia")
        return self

    def native(self):
        from . import _native
        result = _native.HardwareConfig()
        for name in ("robot_ip", "robot_model", "flange_to_tcp", "load_mass", "load_com", "load_inertia", "read_only"):
            setattr(result, name, getattr(self, name))
        if self.expected_end_effector is not None:
            result.check_end_effector = True
            result.expected_ee_mass = self.expected_end_effector.mass
            result.expected_ee_com = self.expected_end_effector.com
            result.expected_ee_inertia = self.expected_end_effector.inertia
        return result


class CommandTimeoutUpdate(Schema):
    command_timeout_s: Positive


class AutomaticRecoveryUpdate(Schema):
    enabled: Annotated[bool, Field(strict=True)]


class HomeRequest(Schema):
    timeout_s: Positive = 30.0
    joint_tolerance_rad: Positive = 0.02
    position_tolerance_m: Positive = 0.01
    rotation_tolerance_rad: Positive = math.radians(2)
    settle_time_s: Positive = 0.2

    @model_validator(mode="after")
    def completion_window(self):
        if self.timeout_s <= self.settle_time_s:
            raise ValueError("home timeout_s must exceed settle_time_s")
        return self


class JointLimitBackoffRequest(Schema):
    """One native inward-only action; the caller supplies no joints or targets."""

    trigger_margin: Annotated[float, Field(gt=0, le=.15)] = .05
    backoff_distance: Annotated[float, Field(ge=.02, le=.2, description=(
        "Radians; backoff_distance - min(0.02, 0.2 * backoff_distance) must exceed trigger_margin"))] = .1
    speed: Annotated[float, Field(gt=0, le=.05)] = .03
    torque_scale: Annotated[float, Field(gt=0, le=1)] = 1.0
    timeout_s: Annotated[float, Field(gt=.2, le=30)] = 20.0
    confirm_clear_path: Annotated[bool, Field(strict=True, description=(
        "Must be true: operator has reviewed and cleared the full arm/tool/cable retreat and braking path"))]

    @model_validator(mode="after")
    def inward_backoff(self):
        if not self.confirm_clear_path:
            raise ValueError("Joint-limit backoff requires confirm_clear_path=true")
        tolerance = min(.02, .2 * self.backoff_distance)
        if self.backoff_distance - tolerance <= self.trigger_margin:
            raise ValueError("backoff_distance - min(.02, .2 * backoff_distance) must exceed trigger_margin")
        return self


class ContactReliefRequest(Schema):
    """Bounded native action; manual follows raw wrench, automatic requires verified bias."""

    mode: Literal["manual", "automatic"] = Field(default="manual", description=(
        "Manual follows raw canonical SDK wrench until operator stop or a bound/guard; "
        "automatic uses verified bias and release detection"))
    bias_wrench_tcp_base: Vec6 | None = Field(default=None, description=(
        "Automatic only: SDK ROBOT-ON-ENVIRONMENT [Fx,Fy,Fz,Mx,My,Mz], base axes, about configured TCP; N/Nm"))
    bias_joint_positions: Vec7 | None = Field(default=None, description=(
        "Automatic only: joint angles at independently verified free-space bias measurement, rad"))
    bias_source: Annotated[str, Field(strict=True, min_length=1, max_length=256)] | None = None
    bias_confirmed_free: Annotated[bool, Field(strict=True)] = False
    workspace_lower: Vec3 | None = Field(default=None, description=(
        "Optional clear TCP box lower XYZ in base axes, m; both corners required in automatic mode"))
    workspace_upper: Vec3 | None = Field(default=None, description=(
        "Optional clear TCP box upper XYZ in base axes, m; provide both corners or neither"))
    workspace_confirmed_clear: Annotated[bool, Field(strict=True, description=(
        "Must be true: operator-supervised cleared local retreat/braking area, including arm/tool/cables"))]
    timeout_s: Annotated[float, Field(gt=.2, le=10)] = 10.0
    force_release: Annotated[float, Field(gt=0, le=10)] = 3.0
    moment_release: Annotated[float, Field(gt=0, le=3)] = 1.0
    max_translation: Annotated[float, Field(gt=0, le=.05)] = .05
    max_rotation: Annotated[float, Field(gt=0, le=math.radians(10))] = math.radians(10)
    translation_speed: Annotated[float, Field(gt=0, le=.02)] = .005
    rotation_speed: Annotated[float, Field(gt=0, le=.1)] = math.radians(1)
    translation_admittance: Annotated[float, Field(gt=0, le=.01)] = .002
    rotation_admittance: Annotated[float, Field(gt=0, le=.1)] = .05
    upward_velocity: Annotated[float, Field(ge=0, le=.005)] = 0.0

    @model_validator(mode="after")
    def caller_confirmations(self):
        if not self.workspace_confirmed_clear:
            raise ValueError("Contact relief requires an operator-supervised cleared local retreat/braking area")
        if (self.workspace_lower is None) != (self.workspace_upper is None):
            raise ValueError("Provide both workspace_lower and workspace_upper, or neither")
        if self.workspace_lower is not None and any(
                lo >= hi for lo, hi in zip(self.workspace_lower, self.workspace_upper)):
            raise ValueError("workspace_lower must be strictly below workspace_upper on every axis")
        if self.mode == "manual":
            if (self.bias_wrench_tcp_base is not None or self.bias_joint_positions is not None
                    or self.bias_source is not None or self.bias_confirmed_free):
                raise ValueError("Manual relief uses raw wrench; do not supply bias, provenance or a free-space bias confirmation")
        else:
            if self.workspace_lower is None:
                raise ValueError("Automatic relief requires an explicit clear TCP workspace box")
            if (self.bias_wrench_tcp_base is None or self.bias_joint_positions is None
                    or self.bias_source is None or not self.bias_source.strip() or not self.bias_confirmed_free):
                raise ValueError("Automatic relief requires verified free-space bias, joint positions, provenance and confirmation")
        return self


class Target(Schema):
    seq: Annotated[int, Field(strict=True, ge=0, lt=2**63)] | None = None
    deadline_s: Positive | None = Field(default=None, description="Optional server-monotonic expiry; otherwise stamped on receipt")
    valid_for_s: Positive | None = None

    @model_validator(mode="after")
    def transport_metadata(self):
        if self.deadline_s is not None and self.valid_for_s is not None:
            raise ValueError("provide deadline_s or valid_for_s, not both")
        return self


class CartesianTarget(Target):
    position: Vec3
    quaternion_xyzw: Vec4 | None = None
    twist: Vec6 = [0.0] * 6
    acceleration: Vec6 = [0.0] * 6
    posture: Vec7 | None = None
    frame: Literal["base"] = "base"

    @model_validator(mode="after")
    def quaternion(self):
        if self.quaternion_xyzw is not None:
            norm = math.sqrt(sum(x*x for x in self.quaternion_xyzw))
            if abs(norm-1) > 1e-3:
                raise ValueError("quaternion_xyzw must have unit norm (tolerance 0.001)")
            # Avoid assignment validation recursively invoking this validator.
            object.__setattr__(self, "quaternion_xyzw", [x/norm for x in self.quaternion_xyzw])
        return self


class JointTarget(Target):
    positions: Vec7
    velocities: Vec7 = [0.0] * 7


class WrenchInjection(Schema):
    wrench: Vec6
