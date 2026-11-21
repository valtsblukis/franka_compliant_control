"""Runtime wrench math/model only: NumPy, no fitting, plotting, SDK or transport."""

import numpy as np
from .twin_model import quaternion_matrix


def tcp_to_spatial(wrench, position):
    w = np.asarray(wrench, dtype=float).copy()
    p = np.asarray(position, dtype=float)
    if (
        w.shape != (6,)
        or p.shape != (3,)
        or not np.isfinite(w).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Expected finite wrench and reference point")
    w[3:] += np.cross(p, w[:3])
    return w


def spatial_to_tcp(wrench, position):
    w = np.asarray(wrench, dtype=float).copy()
    p = np.asarray(position, dtype=float)
    if (
        w.shape != (6,)
        or p.shape != (3,)
        or not np.isfinite(w).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Expected finite wrench and reference point")
    w[3:] -= np.cross(p, w[:3])
    return w


def canonical_franka_wrench(state, *, stiffness_at_tcp=False):
    """Requires known K==TCP; never infer coincidence from an inactive label."""
    if not stiffness_at_tcp:
        raise ValueError("Known stiffness-frame/TCP coincidence is required")
    k = np.asarray(state["external_wrench_stiffness"], dtype=float)
    if k.shape != (6,) or not np.isfinite(k).all():
        raise ValueError("Invalid stiffness-frame wrench")
    r = quaternion_matrix(state["quaternion_xyzw"])
    return np.r_[r @ k[:3], r @ k[3:]]


def projected_wrench(
    jacobian, torques, *, rotation_scale=0.2, minimum_singular_value=0.05
):
    j = np.asarray(jacobian, dtype=float)
    tau = np.asarray(torques, dtype=float)
    if (
        j.shape != (6, len(tau))
        or not np.isfinite(j).all()
        or not np.isfinite(tau).all()
    ):
        raise ValueError("Invalid Jacobian/torque vector")
    scaled = j.copy()
    scaled[3:] *= rotation_scale
    sigma = float(np.linalg.svd(scaled, compute_uv=False)[-1])
    if sigma < minimum_singular_value:
        raise ValueError("Wrench projection outside conditioned operating region")
    return np.linalg.lstsq(j.T, tau, rcond=1e-8)[0], sigma


class WrenchObserver:
    """Previously fitted local Franka surrogate; contact transfer is an assumption."""

    def __init__(self, artifact):
        if artifact["schema"] != "fcc-wrench-observer-v1":
            raise ValueError("Unknown observer schema")
        self.artifact = artifact
        self.qref = np.asarray(artifact["q_reference"])
        self.gain = np.asarray(artifact["resistance_gain"])
        self.offset = np.asarray(artifact["joint_offset"])
        self.slope = np.asarray(artifact["joint_position_slope"])
        if any(
            a.shape != (7,) or not np.isfinite(a).all()
            for a in [self.qref, self.gain, self.offset, self.slope]
        ):
            raise ValueError("Invalid observer coefficients")
        if (
            np.any(self.gain < 0)
            or np.any(self.gain > 1.5)
            or np.any(np.abs(self.offset) > 2 + 1e-8)
            or np.any(np.abs(self.slope) > 2 + 1e-8)
            or not np.isfinite(artifact["nominal_contact_gain"])
            or artifact["nominal_contact_gain"] <= 0
        ):
            raise ValueError("Observer coefficients outside declared bounds")
        self.tc = float(artifact["time_constant_s"])
        if not np.isfinite(self.tc) or self.tc < 0:
            raise ValueError("Invalid observer time constant")
        self.filtered = None

    def step(
        self, q, free_residual, jacobian, position, dt, *, contact_joint_torque=None
    ):
        q = np.asarray(q)
        r = np.asarray(free_residual)
        if (
            q.shape != (7,)
            or r.shape != (7,)
            or not np.isfinite(q).all()
            or not np.isfinite(r).all()
            or not 0 < dt <= 0.1
        ):
            raise ValueError("Invalid observer sample")
        external = (
            np.zeros(7)
            if contact_joint_torque is None
            else np.asarray(contact_joint_torque)
        )
        if external.shape != (7,) or not np.isfinite(external).all():
            raise ValueError("Invalid modeled contact torque")
        # Free-space attenuation must never be applied to true contact/disturbance.
        raw = (
            self.gain * r
            + self.offset
            + self.slope * (q - self.qref)
            + self.artifact["nominal_contact_gain"] * external
        )
        a = np.exp(-dt / self.tc) if self.tc else 0.0
        self.filtered = (
            raw.copy() if self.filtered is None else a * self.filtered + (1 - a) * raw
        )
        wrench, sigma = projected_wrench(
            jacobian,
            self.filtered,
            rotation_scale=self.artifact["rotation_scale"],
            minimum_singular_value=self.artifact["minimum_singular_value"],
        )
        return {
            "external_joint_torques": self.filtered.copy(),
            "external_wrench_tcp_base": wrench,
            "external_wrench": tcp_to_spatial(wrench, position),
            "minimum_singular_value": sigma,
            "source": "approximate_franka_estimator_surrogate",
            "true_force_calibrated": False,
        }
