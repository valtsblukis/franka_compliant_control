"""Solver contact truth, separate from any policy-facing force estimate.

MuJoCo's contact force is from geom1 toward geom2 in the contact frame; frame
axes are rows. See MuJoCo computation/contact documentation. We transform all
six components and project at the actual contacting ROBOT BODY, not always TCP.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Contact:
    body1: str
    body2: str
    geom1: int
    geom2: int
    position: tuple[float, float, float]
    force_on_body2: tuple[float, float, float]
    moment_on_body2: tuple[float, float, float]
    normal_force_N: float
    penetration_m: float
    robot1: bool
    robot2: bool
    dynamic1: bool
    dynamic2: bool

    @property
    def self_contact(self):
        return self.robot1 and self.robot2

    @property
    def other_body(self):
        return self.body2 if self.robot1 else self.body1 if self.robot2 else None

    @property
    def force_norm_N(self):
        return float(np.linalg.norm(self.force_on_body2))


@dataclass(frozen=True)
class ContactReport:
    contacts: tuple[Contact, ...] = ()
    # Resisting generalized torque: negative of contact torque applied TO arm.
    robot_on_environment_torque: tuple[float, ...] = (0.0,) * 7

    @property
    def loads_by_body(self):
        """Sum magnitudes, NOT norm of sum: opposing/squeezing loads cannot cancel."""
        result = {}
        for c in self.contacts:
            if c.self_contact:
                keys = ["__robot_self__"]
            elif c.robot1 or c.robot2:
                keys = [c.other_body]
            else:
                # Secondary object/object and object/environment loads count too.
                keys = ([c.body1] if c.dynamic1 else []) + (
                    [c.body2] if c.dynamic2 else []
                )
            for key in keys:
                result[key] = result.get(key, 0.0) + c.force_norm_N
        return result

    @property
    def peak_body_force_N(self):
        return max(self.loads_by_body.values(), default=0.0)


class ContactOracle:
    def __init__(self, model, robot_bodies, robot_dofs):
        self.model = model
        self.robot_bodies = frozenset(robot_bodies)
        self.robot_dofs = np.asarray(robot_dofs)
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros_like(self._jp)
        self._wrench = np.zeros(6)

    def read(self, data) -> ContactReport:
        import mujoco

        m = self.model
        contacts = []
        resisting = np.zeros(m.nv)
        for index in range(data.ncon):
            c = data.contact[index]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0 or c.efc_address < 0:
                continue
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            mujoco.mj_contactForce(m, data, index, self._wrench)
            rotation = c.frame.reshape(3, 3).T
            force, moment = rotation @ self._wrench[:3], rotation @ self._wrench[3:]
            for body, sign, is_robot in [(b1, 1.0, r1), (b2, -1.0, r2)]:
                if is_robot:
                    mujoco.mj_jac(m, data, self._jp, self._jr, c.pos, body)
                    resisting += sign * (self._jp.T @ force + self._jr.T @ moment)
            contacts.append(
                Contact(
                    m.body(b1).name,
                    m.body(b2).name,
                    g1,
                    g2,
                    tuple(map(float, c.pos)),
                    tuple(map(float, force)),
                    tuple(map(float, moment)),
                    float(max(0.0, self._wrench[0])),
                    float(max(0.0, -c.dist)),
                    r1,
                    r2,
                    bool(m.body_mass[b1] > 0),
                    bool(m.body_mass[b2] > 0),
                )
            )
        return ContactReport(tuple(contacts), tuple(resisting[self.robot_dofs]))


@dataclass
class ContactAccumulator:
    """Episode/interval truth metrics; integrate once per actual physics step."""

    comfortable_force_N: float = 4.0
    peak_force_N: float = 0.0
    impulse_Ns: float = 0.0
    excess_force_squared_N2s: float = 0.0
    contact_time_s: float = 0.0
    maximum_penetration_m: float = 0.0
    self_contact: bool = False
    by_body_impulse_Ns: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not np.isfinite(self.comfortable_force_N) or self.comfortable_force_N < 0:
            raise ValueError("Invalid contact comfort threshold")

    def add(self, report: ContactReport, dt: float):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Contact integration needs a positive finite timestep")
        loads = report.loads_by_body
        self.peak_force_N = max(self.peak_force_N, report.peak_body_force_N)
        self.impulse_Ns += sum(loads.values()) * dt
        self.excess_force_squared_N2s += (
            sum(max(0.0, f - self.comfortable_force_N) ** 2 for f in loads.values())
            * dt
        )
        self.contact_time_s += dt if any(f > 1e-6 for f in loads.values()) else 0.0
        for name, load in loads.items():
            self.by_body_impulse_Ns[name] = (
                self.by_body_impulse_Ns.get(name, 0.0) + load * dt
            )
        self.maximum_penetration_m = max(
            self.maximum_penetration_m,
            max((c.penetration_m for c in report.contacts), default=0.0),
        )
        self.self_contact |= any(c.self_contact for c in report.contacts)
