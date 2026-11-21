"""Backend-neutral values passed between simulation, sensors, tasks and policies."""

from dataclasses import dataclass
import numpy as np
from .contacts import ContactReport


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self):
        p = np.asarray(self.position, dtype=float)
        q = np.asarray(self.quaternion_xyzw, dtype=float)
        if (
            p.shape != (3,)
            or q.shape != (4,)
            or not np.isfinite(np.r_[p, q]).all()
            or abs(np.linalg.norm(q) - 1) > 1e-5
        ):
            raise ValueError(
                "Expected finite base-frame position and unit xyzw quaternion"
            )
        object.__setattr__(self, "position", tuple(map(float, p)))
        object.__setattr__(self, "quaternion_xyzw", tuple(map(float, q)))


@dataclass(frozen=True)
class RobotMeasurement:
    time_s: float
    positions: tuple
    velocities: tuple
    position: tuple
    quaternion_xyzw: tuple
    twist: tuple
    wrench_tcp_base: tuple
    wrench_valid: bool
    wrench_source: str
    sensor_extrapolated: bool
    minimum_singular_value: float


@dataclass(frozen=True)
class ControlStep:
    measurement: RobotMeasurement
    phase: str
    reason: str
    physics_contacts: tuple[tuple[float, ContactReport], ...]
