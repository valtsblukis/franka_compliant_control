"""Optional fitted near-static kinematic observation variability, not plant truth.

A normalized VAR(1) generates correlated q/dq observation residuals at the native
sample period. It does not claim to model physical breakaway/elasticity. Fitting
and validation remain outside this runtime-only module.
"""

from copy import deepcopy
import numpy as np


class KinematicNoiseModel:
    def __init__(self, config):
        self.config = deepcopy(config)
        if config.get("schema") != "fcc-kinematic-noise-v1":
            raise ValueError("Unknown kinematic observation model")
        self.period = float(config["period_s"])
        self.scale = np.asarray(config["scale"])
        self.A = np.asarray(config["A"])
        self.Q = np.asarray(config["innovation_covariance"])
        self.P = np.asarray(config["stationary_covariance"])
        if (
            self.scale.shape != (14,)
            or self.A.shape != (14, 14)
            or self.Q.shape != (14, 14)
            or self.P.shape != (14, 14)
            or not np.isfinite(
                np.r_[
                    self.scale,
                    self.A.ravel(),
                    self.Q.ravel(),
                    self.P.ravel(),
                    self.period,
                ]
            ).all()
            or np.any(self.scale <= 0)
            or self.period <= 0
        ):
            raise ValueError("Invalid observation model arrays")
        if max(abs(np.linalg.eigvals(self.A))) >= 1:
            raise ValueError("Observation process must be stable")
        if not np.allclose(self.Q, self.Q.T) or not np.allclose(self.P, self.P.T):
            raise ValueError("Covariance must be symmetric")
        self.LQ = np.linalg.cholesky(self.Q)
        self.LP = np.linalg.cholesky(self.P)
        self.reset(0)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.z = self.LP @ self.rng.normal(size=14)

    def observe(self, q, dq, dt):
        if abs(dt - self.period) > 1e-10:
            raise ValueError("Observation model period changed")
        self.z = self.A @ self.z + self.LQ @ self.rng.normal(size=14)
        noise = self.z * self.scale
        return np.asarray(q) + noise[:7], np.asarray(dq) + noise[7:]

    def manifest(self):
        return {
            "model": type(self).__name__,
            "config": self.config,
            "scope": "effective local near-static observation variability; not a burst/contact model",
        }
