"""Optional smooth external joint-torque disturbances; independent of obstacles."""

from dataclasses import dataclass, asdict
import numpy as np
from .components import SceneComponent


@dataclass(frozen=True)
class TorquePulse:
    start_s: float
    duration_s: float
    torque_Nm: tuple[float, ...]

    def __post_init__(self):
        if (
            not np.isfinite([self.start_s, self.duration_s]).all()
            or self.start_s < 0
            or self.duration_s <= 0
        ):
            raise ValueError("Invalid disturbance timing")
        if len(self.torque_Nm) != 7 or not np.isfinite(self.torque_Nm).all():
            raise ValueError("Expected seven finite external torques")


class JointTorqueDisturbances(SceneComponent):
    def __init__(self, pulses):
        self.pulses = tuple(pulses)

    def add_forces(self, world, data, generalized_force):
        for pulse in self.pulses:
            u = (data.time - pulse.start_s) / pulse.duration_s
            if 0 < u < 1:
                generalized_force[world.robot_dofs] += (
                    np.asarray(pulse.torque_Nm) * np.sin(np.pi * u) ** 2
                )

    def manifest(self):
        return {
            "component": type(self).__name__,
            "pulses": [asdict(p) for p in self.pulses],
            "convention": "external torque applied TO robot; smooth sin-squared envelope",
        }


def randomized_torque_disturbances(
    rng, *, count=1, maximum_torque_Nm=0.2, start_s=(1.0, 5.0), duration_s=(0.2, 0.8)
):
    if (
        type(count) is not int
        or count < 0
        or not np.isfinite(maximum_torque_Nm)
        or maximum_torque_Nm < 0
    ):
        raise ValueError("Invalid disturbance distribution")
    for bounds in (start_s, duration_s):
        if (
            len(bounds) != 2
            or not np.isfinite(bounds).all()
            or not 0 < bounds[0] <= bounds[1]
        ):
            raise ValueError("Invalid disturbance interval")
    pulses = []
    for _ in range(count):
        direction = rng.normal(size=7)
        direction /= max(np.linalg.norm(direction), 1e-12)
        pulses.append(
            TorquePulse(
                float(rng.uniform(*start_s)),
                float(rng.uniform(*duration_s)),
                tuple(direction * rng.uniform(0, maximum_torque_Nm)),
            )
        )
    return JointTorqueDisturbances(pulses)
