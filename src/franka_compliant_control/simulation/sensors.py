"""Policy-facing wrench estimates. Contact truth is a separate reward/debug channel."""

from dataclasses import dataclass, asdict
from copy import deepcopy
import numpy as np
from ..wrench_model import WrenchObserver, projected_wrench


@dataclass(frozen=True)
class WrenchReading:
    tcp_base: np.ndarray
    joint_torques: np.ndarray
    source: str
    valid: bool = True
    extrapolated: bool = False
    contact_calibrated: bool = False


class FrankaWrenchSurrogate:
    def __init__(self, artifact):
        self.artifact = deepcopy(artifact)
        self.reset()

    def reset(self, seed=0):
        self.observer = WrenchObserver(self.artifact)

    def observe(self, sample, free_residual, external_resisting_torque, dt):
        o = self.observer.step(
            sample.q,
            free_residual,
            sample.jacobian,
            sample.position,
            dt,
            contact_joint_torque=external_resisting_torque,
        )
        q = np.asarray(sample.q)
        outside = bool(
            np.any(q < np.array(self.artifact["joint_min_rad"]) - 0.02)
            or np.any(q > np.array(self.artifact["joint_max_rad"]) + 0.02)
        )
        return WrenchReading(
            o["external_wrench_tcp_base"],
            o["external_joint_torques"],
            "approximate_franka_surrogate",
            extrapolated=outside,
        )

    def manifest(self):
        return {
            "type": type(self).__name__,
            "artifact": self.artifact,
            "contact_calibrated": False,
        }


class AffineWrenchSensor:
    """Explicit per-episode measurement uncertainty around another sensor."""

    def __init__(self, base, gain, bias):
        self.base = base
        self.gain = np.asarray(gain, dtype=float)
        self.bias = np.asarray(bias, dtype=float)
        if (
            self.gain.shape != (6,)
            or self.bias.shape != (6,)
            or not np.isfinite([self.gain, self.bias]).all()
            or min(self.gain) < 0
        ):
            raise ValueError("Invalid sensor uncertainty")

    def reset(self, seed=0):
        self.base.reset(seed)

    def observe(self, sample, free_residual, external_resisting_torque, dt):
        reading = self.base.observe(
            sample, free_residual, external_resisting_torque, dt
        )
        wrench = reading.tcp_base * self.gain + self.bias
        return WrenchReading(
            wrench,
            reading.joint_torques + sample.jacobian.T @ (wrench - reading.tcp_base),
            reading.source + "+affine_uncertainty",
            reading.valid,
            reading.extrapolated,
            reading.contact_calibrated,
        )

    def manifest(self):
        return {
            "type": type(self).__name__,
            "base": self.base.manifest(),
            "gain": self.gain.tolist(),
            "bias": self.bias.tolist(),
        }


@dataclass(frozen=True)
class ContactSensorConfig:
    time_constant_s: float = 0.12
    external_gain: float = 1.0
    free_residual_gain: float = 0.0
    tcp_bias: tuple = (0.0,) * 6
    noise_std: tuple = (0.02, 0.02, 0.02, 0.003, 0.003, 0.003)
    rotation_scale: float = 0.2
    minimum_singular_value: float = 0.05

    def __post_init__(self):
        if (
            not np.isfinite(
                [
                    self.time_constant_s,
                    self.external_gain,
                    self.free_residual_gain,
                    self.rotation_scale,
                    self.minimum_singular_value,
                ]
            ).all()
            or self.time_constant_s < 0
            or self.external_gain <= 0
            or self.free_residual_gain < 0
            or self.rotation_scale <= 0
            or self.minimum_singular_value <= 0
        ):
            raise ValueError("Invalid sensor settings")
        if (
            len(self.tcp_bias) != 6
            or len(self.noise_std) != 6
            or not np.isfinite([self.tcp_bias, self.noise_std]).all()
            or min(self.noise_std) < 0
        ):
            raise ValueError("Invalid wrench bias/noise")


class FilteredContactSensor:
    """Explicit synthetic transfer model for general-workspace sim curricula.

    Avoids extrapolating a local fitted joint-position bias over the entire arm
    workspace. It is NOT newly calibrated Franka telemetry. External/contact
    torque is never attenuated using fitted free-space resistance gains.
    """

    def __init__(self, config=ContactSensorConfig()):
        self.config = config
        self.reset()

    def reset(self, seed=0):
        self.filtered = None
        self.rng = np.random.default_rng(seed)

    def observe(self, sample, free_residual, external_resisting_torque, dt):
        c = self.config
        raw = c.external_gain * np.asarray(
            external_resisting_torque
        ) + c.free_residual_gain * np.asarray(free_residual)
        alpha = np.exp(-dt / c.time_constant_s) if c.time_constant_s else 0.0
        self.filtered = (
            raw.copy()
            if self.filtered is None
            else alpha * self.filtered + (1 - alpha) * raw
        )
        w, _ = projected_wrench(
            sample.jacobian,
            self.filtered,
            rotation_scale=c.rotation_scale,
            minimum_singular_value=c.minimum_singular_value,
        )
        noisy = w + np.asarray(c.tcp_bias) + self.rng.normal(0, c.noise_std)
        tau = self.filtered + sample.jacobian.T @ (noisy - w)
        return WrenchReading(noisy, tau, "synthetic_filtered_contact_sensor")

    def manifest(self):
        return {
            "type": type(self).__name__,
            "config": asdict(self.config),
            "contact_calibrated": False,
            "franka_telemetry_calibrated": False,
        }
