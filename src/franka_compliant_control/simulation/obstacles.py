"""Optional finite-mass dynamic obstacles and randomized lifecycle.

These are compliant interaction proxies, NOT a calibrated human-arm model.
Appearance is collision-checked. Disappearance requests a force-limited
withdrawal and waits for unloaded, slow motion before disabling the body.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from .components import SceneComponent
from ..twin_model import _numbers


def bounded(vector, limit):
    v = np.asarray(vector, dtype=float)
    return v * min(1.0, limit / max(float(np.linalg.norm(v)), 1e-12))


@dataclass(frozen=True)
class MovableObstacleSpec:
    name: str
    mass_kg: float = 1.0
    radius_m: float = 0.045
    capsule_length_m: float = 0.20  # zero makes a sphere
    friction: float = 0.4
    contact_time_constant_s: float = 0.04
    support_stiffness_N_m: float = 25.0
    support_damping_Ns_m: float = 4.0
    support_force_limit_N: float = 8.0
    rotational_damping_Nms: float = 0.01
    gravity_supported: bool = True

    def __post_init__(self):
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("Obstacle name must be an identifier")
        if type(self.gravity_supported) is not bool:
            raise ValueError("Gravity support must be boolean")
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and (
                not np.isfinite(value) or value < 0
            ):
                raise ValueError("Invalid obstacle parameter: " + key)
        if (
            self.mass_kg <= 0
            or self.radius_m <= 0
            or self.contact_time_constant_s < 0.004
            or self.support_force_limit_N <= 0
        ):
            raise ValueError(
                "Obstacles require finite positive mass, size and compliant contact settings"
            )

    def inertia(self):
        r, length, mass = self.radius_m, self.capsule_length_m, self.mass_kg
        if length == 0:
            return np.full(3, 0.4 * mass * r * r)
        volume_cylinder, volume_caps = math.pi * r * r * length, 4 / 3 * math.pi * r**3
        cylinder = mass * volume_cylinder / (volume_cylinder + volume_caps)
        caps = mass - cylinder
        transverse = cylinder * (3 * r * r + length * length) / 12 + caps * (
            0.4 * r * r + length * length / 4 + 3 * length * r / 8
        )
        axial = 0.5 * cylinder * r * r + 0.4 * caps * r * r
        return np.array([transverse, transverse, axial])


@dataclass(frozen=True)
class ObstacleSchedule:
    enabled: bool = True
    initial_spawn_delay_s: tuple = (0.05, 0.25)
    spawn_interval_s: tuple = (0.4, 1.5)
    lifetime_s: tuple = (1.0, 4.0)
    persistent_probability: float = 0.15
    spawn_clearance_m: float = 0.008
    withdrawal_s: float = 0.8
    withdrawal_distance_m: float = 0.25
    quiet_s: float = 0.10
    retirement_speed_m_s: float = 0.15
    retirement_angular_speed_rad_s: float = 1.0
    unloaded_force_N: float = 0.05
    placement_attempts: int = 24

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("Schedule enable flag must be boolean")
        for r in (self.initial_spawn_delay_s, self.spawn_interval_s, self.lifetime_s):
            if len(r) != 2 or not np.isfinite(r).all() or not 0 < r[0] <= r[1]:
                raise ValueError("Invalid lifecycle interval")
        if not 0 <= self.persistent_probability <= 1:
            raise ValueError("Invalid persistence probability")
        if type(self.placement_attempts) is not int or self.placement_attempts < 1:
            raise ValueError("Invalid placement attempts")
        for v in [
            self.spawn_clearance_m,
            self.withdrawal_s,
            self.withdrawal_distance_m,
            self.quiet_s,
            self.retirement_speed_m_s,
            self.retirement_angular_speed_rad_s,
            self.unloaded_force_N,
        ]:
            if not np.isfinite(v) or v <= 0:
                raise ValueError("Lifecycle limits must be finite and positive")


class DynamicObstacles(SceneComponent):
    def __init__(self, specs, schedule=ObstacleSchedule()):
        self.specs = tuple(specs)
        self.schedule = schedule
        if not self.specs or len({s.name for s in self.specs}) != len(self.specs):
            raise ValueError("Unique obstacle specs are required")
        self.corridor = None

    def build(self, root, home):
        world = root.find("worldbody")
        if world is None:
            raise ValueError("MJCF worldbody missing")
        for i, s in enumerate(self.specs):
            b = ET.SubElement(
                world, "body", name="obstacle_" + s.name, pos=_numbers(self._parking(i))
            )
            ET.SubElement(b, "freejoint", name="obstacle_joint_" + s.name)
            ET.SubElement(
                b,
                "inertial",
                pos="0 0 0",
                mass=str(s.mass_kg),
                diaginertia=_numbers(s.inertia()),
            )
            attributes = {
                "name": "obstacle_geom_" + s.name,
                "type": "capsule" if s.capsule_length_m else "sphere",
                "size": (
                    _numbers([s.radius_m, s.capsule_length_m / 2])
                    if s.capsule_length_m
                    else str(s.radius_m)
                ),
                "mass": "0",
                "contype": "2",
                "conaffinity": "3",
                "condim": "3",
                "group": "4",
                "rgba": ".9 .55 .25 0",
                "friction": f"{s.friction} .005 .0001",
                "solref": f"{s.contact_time_constant_s} 1",
                "solimp": ".8 .95 .003 .5 2",
                "priority": "1",
            }
            ET.SubElement(b, "geom", **attributes)

    def bind(self, world):
        super().bind(world)
        m = world.model
        self.slots = []
        self.robot_geoms = [
            i
            for i in range(m.ngeom)
            if m.geom_bodyid[i] in world.robot_bodies and m.geom_contype[i] & 1
        ]
        if not self.robot_geoms:
            raise ValueError(
                "Dynamic obstacles need an explicit robot collision-geometry component"
            )
        for i, s in enumerate(self.specs):
            joint = m.joint("obstacle_joint_" + s.name).id
            self.slots.append(
                {
                    "spec": s,
                    "body": m.body("obstacle_" + s.name).id,
                    "geom": m.geom("obstacle_geom_" + s.name).id,
                    "qpos": int(m.jnt_qposadr[joint]),
                    "dof": int(m.jnt_dofadr[joint]),
                    "parking": self._parking(i),
                }
            )
        self.probe = mujoco.MjData(m)

    @staticmethod
    def _parking(index):
        return np.array([10.0 + index, 0.0, -10.0])

    def set_corridor(self, start, goal):
        points = np.asarray([start, goal], dtype=float)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError("Invalid motion corridor")
        self.corridor = points.copy()

    def reset(self, world, rng):
        self.rng = rng
        self.next_spawn = float(rng.uniform(*self.schedule.initial_spawn_delay_s))
        for slot in self.slots:
            slot.update(
                status="inactive",
                anchor=slot["parking"].copy(),
                death=None,
                quiet_since=None,
                maximum_active_displacement_m=0.0,
                maximum_active_speed_m_s=0.0,
            )
            self._park(world, slot)

    def _park(self, world, slot):
        a, d, g = slot["qpos"], slot["dof"], slot["geom"]
        world.data.qpos[a : a + 3] = slot["parking"]
        world.data.qpos[a + 3 : a + 7] = [1, 0, 0, 0]
        world.data.qvel[d : d + 6] = 0
        world.data.qacc_warmstart[d : d + 6] = 0
        world.model.geom_contype[g] = 0
        world.model.geom_conaffinity[g] = 0
        world.model.geom_rgba[g, 3] = 0
        world.model.body_contype[slot["body"]] = 0
        world.model.body_conaffinity[slot["body"]] = 0
        slot["status"] = "inactive"
        slot["anchor"] = slot["parking"].copy()

    def spawn(
        self,
        world,
        index,
        position,
        *,
        velocity=(0.0, 0.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        lifetime_s=None,
    ):
        slot = self.slots[index]
        p = np.asarray(position, dtype=float)
        v = np.asarray(velocity, dtype=float)
        q = np.asarray(quaternion_xyzw, dtype=float)
        if slot["status"] != "inactive":
            raise ValueError("Cannot teleport an active obstacle")
        if (
            p.shape != (3,)
            or v.shape != (3,)
            or q.shape != (4,)
            or not np.isfinite(np.r_[p, v, q]).all()
            or abs(np.linalg.norm(q) - 1) > 1e-5
        ):
            raise ValueError("Invalid spawn state")
        if lifetime_s is not None and (not np.isfinite(lifetime_s) or lifetime_s <= 0):
            raise ValueError("Invalid lifetime")
        a, d, g = slot["qpos"], slot["dof"], slot["geom"]
        probe = self.probe
        world.copy_kinematic_state(probe)
        probe.qpos[a : a + 3] = p
        probe.qpos[a + 3 : a + 7] = q[[3, 0, 1, 2]]
        probe.qvel[d : d + 6] = 0
        mujoco.mj_kinematics(world.model, probe)
        others = self.robot_geoms + [
            s["geom"] for s in self.slots if s["status"] != "inactive"
        ]
        # Include other collidable scene geometry, not display-only floor/meshes.
        others += [
            j
            for j in range(world.model.ngeom)
            if j != g
            and world.model.geom_bodyid[j] not in world.robot_bodies
            and world.model.geom_contype[j]
            and j not in others
        ]
        for other in others:
            distance = mujoco.mj_geomDistance(
                world.model,
                probe,
                g,
                other,
                self.schedule.spawn_clearance_m + 1.0,
                None,
            )
            if distance < self.schedule.spawn_clearance_m:
                return False
        world.data.qpos[a : a + 7] = probe.qpos[a : a + 7]
        world.data.qvel[d : d + 3] = v
        world.data.qvel[d + 3 : d + 6] = 0
        world.data.qacc_warmstart[d : d + 6] = 0
        world.model.geom_contype[g] = 2
        world.model.geom_conaffinity[g] = 1 | 2
        world.model.geom_rgba[g, 3] = 0.85
        world.model.body_contype[slot["body"]] = 2
        world.model.body_conaffinity[slot["body"]] = 1 | 2
        slot.update(
            status="active",
            anchor=p.copy(),
            spawn_position=p.copy(),
            death=None if lifetime_s is None else world.time_s + lifetime_s,
            quiet_since=None,
            maximum_active_displacement_m=0.0,
            maximum_active_speed_m_s=float(np.linalg.norm(v)),
        )
        world.emit(
            "obstacle_spawned",
            name=slot["spec"].name,
            position=p.tolist(),
            velocity=v.tolist(),
            mass_kg=slot["spec"].mass_kg,
            lifetime_s=lifetime_s,
            nonpenetrating=True,
        )
        mujoco.mj_forward(world.model, world.data)
        return True

    def request_despawn(self, world, index):
        slot = self.slots[index]
        if slot["status"] != "active":
            return False
        p = world.data.qpos[slot["qpos"] : slot["qpos"] + 3].copy()
        mujoco.mj_kinematics(world.model, world.data)
        origin = world.data.site_xpos[world.plant.tcp_id]
        contacts = [
            c
            for c in world.last_contacts.contacts
            if c.other_body == world.model.body(slot["body"]).name
        ]
        if contacts:
            origin = np.mean([c.position for c in contacts], axis=0)
        away = p - origin
        norm = np.linalg.norm(away)
        away = away / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
        target = self._withdrawal_target(world, slot, p, away)
        if target is None:
            # Do not force an environmental actuator into a blocked path. Keep
            # the finite body present, relax its anchor, and retry later.
            slot["anchor"] = p.copy()
            slot["death"] = world.time_s + 1.0
            world.emit(
                "obstacle_withdrawal_deferred",
                name=slot["spec"].name,
                reason="no clear swept path",
            )
            return False
        slot.update(
            status="withdrawing",
            withdraw_at=world.time_s,
            withdraw_from=p,
            withdraw_to=target,
            quiet_since=None,
        )
        world.emit(
            "obstacle_withdrawal_requested",
            name=slot["spec"].name,
            target=target.tolist(),
            swept_clearance_checked=True,
        )
        return True

    def _withdrawal_target(self, world, slot, position, away):
        """Check the current-orientation swept shape, not just the body centre.

        Existing touching pairs may separate, never deepen penetration. New
        pairs require a gap larger than half a translation sample plus margin.
        This is scene planning, NOT a guarantee against subsequent robot motion.
        """
        probe = self.probe
        world.copy_kinematic_state(probe)
        a, g = slot["qpos"], slot["geom"]
        m = world.model
        others = self.robot_geoms + [
            s["geom"] for s in self.slots if s is not slot and s["status"] != "inactive"
        ]
        others += [
            j
            for j in range(m.ngeom)
            if j != g
            and m.geom_bodyid[j] not in world.robot_bodies
            and m.geom_contype[j]
            and j not in others
        ]
        mujoco.mj_kinematics(m, probe)
        initial = {j: mujoco.mj_geomDistance(m, probe, g, j, 2.0, None) for j in others}
        basis = np.eye(3)
        directions = [away] + [x for axis in basis for x in (axis, -axis)]
        directions += [away + 2 * sign * axis for axis in basis for sign in (-1, 1)]
        distance = self.schedule.withdrawal_distance_m
        samples = 13
        step = distance / samples
        # Allow for free-body rotation/momentum; this is not a kinematic lock.
        spec = slot["spec"]
        velocity = world.data.qvel[slot["dof"] : slot["dof"] + 6]
        damping = max(
            spec.support_damping_Ns_m,
            2 * np.sqrt(spec.support_stiffness_N_m * spec.mass_kg),
        )
        rotation_damping = max(
            spec.rotational_damping_Nms,
            5 * max(spec.inertia()) / self.schedule.withdrawal_s,
        )
        rotation_bound = (
            np.linalg.norm(velocity[3:])
            * max(spec.inertia())
            / rotation_damping
            * (spec.radius_m + spec.capsule_length_m / 2)
        )
        translation_bound = (
            spec.mass_kg * np.linalg.norm(velocity[:3]) / max(damping, 1e-6)
        )
        clearance = step / 2 + 0.01 + min(0.08, rotation_bound + translation_bound)
        for direction in directions:
            direction = direction / max(np.linalg.norm(direction), 1e-9)
            # Do not reverse into the contacting surface just to find a clear endpoint.
            if direction @ away < -0.05:
                continue
            last = initial.copy()
            clear = True
            for i in range(1, samples + 1):
                probe.qpos[a : a + 3] = position + direction * distance * i / samples
                mujoco.mj_kinematics(m, probe)
                for j in others:
                    d = mujoco.mj_geomDistance(m, probe, g, j, 2.0, None)
                    if initial[j] < clearance:
                        if d < last[j] - 1e-5 or (i == samples and d < clearance):
                            clear = False
                            break
                    elif d < clearance:
                        clear = False
                        break
                    last[j] = d
                if not clear:
                    break
            if clear:
                return position + direction * distance
        return None

    def before_step(self, world, dt):
        for i, slot in enumerate(self.slots):
            if (
                slot["status"] == "active"
                and slot["death"] is not None
                and world.time_s >= slot["death"]
            ):
                self.request_despawn(world, i)
        if not self.schedule.enabled or world.time_s < self.next_spawn:
            return
        self.next_spawn = world.time_s + float(
            self.rng.uniform(*self.schedule.spawn_interval_s)
        )
        free = [i for i, s in enumerate(self.slots) if s["status"] == "inactive"]
        if not free or self.corridor is None:
            return
        i = int(self.rng.choice(free))
        start, goal = self.corridor
        # Reconsider current TCP so objects obstruct remaining motion rather than
        # repeatedly spawning behind the robot. Event timing stays independent.
        mujoco.mj_kinematics(world.model, world.data)
        start = world.data.site_xpos[world.plant.tcp_id].copy()
        if np.linalg.norm(goal - start) < 0.06:
            world.emit("obstacle_spawn_rejected", reason="remaining corridor too short")
            return
        life = (
            None
            if self.rng.random() < self.schedule.persistent_probability
            else float(self.rng.uniform(*self.schedule.lifetime_s))
        )
        for _ in range(self.schedule.placement_attempts):
            p = (
                start
                + self.rng.uniform(0.35, 0.85) * (goal - start)
                + self.rng.normal(0, 0.012, 3)
            )
            # Capsule azimuth varies; geometry/mass remain immutable this episode.
            angle = self.rng.uniform(-np.pi, np.pi)
            quat = [
                np.sin(np.pi / 4) * np.cos(angle),
                np.sin(np.pi / 4) * np.sin(angle),
                0.0,
                np.cos(np.pi / 4),
            ]
            if self.spawn(
                world,
                i,
                p,
                velocity=self.rng.uniform(-0.025, 0.025, 3),
                quaternion_xyzw=quat,
                lifetime_s=life,
            ):
                return
        world.emit(
            "obstacle_spawn_rejected",
            reason="no nonintersecting placement",
            name=self.specs[i].name,
        )

    def add_forces(self, world, data, generalized_force):
        for slot in self.slots:
            s = slot["spec"]
            a, d = slot["qpos"], slot["dof"]
            if s.gravity_supported or slot["status"] == "inactive":
                generalized_force[d : d + 3] -= s.mass_kg * world.model.opt.gravity
            if slot["status"] == "inactive":
                continue
            anchor = slot["anchor"]
            if slot["status"] == "withdrawing":
                u = float(
                    np.clip(
                        (data.time - slot["withdraw_at"]) / self.schedule.withdrawal_s,
                        0,
                        1,
                    )
                )
                blend = u**3 * (10 - 15 * u + 6 * u * u)
                anchor = slot["withdraw_from"] + blend * (
                    slot["withdraw_to"] - slot["withdraw_from"]
                )
            damping = s.support_damping_Ns_m
            rotation_damping = s.rotational_damping_Nms
            if slot["status"] == "withdrawing":
                damping = max(damping, 2 * np.sqrt(s.mass_kg * s.support_stiffness_N_m))
                rotation_damping = max(
                    rotation_damping, 5 * max(s.inertia()) / self.schedule.withdrawal_s
                )
            f = (
                s.support_stiffness_N_m * (anchor - data.qpos[a : a + 3])
                - damping * data.qvel[d : d + 3]
            )
            generalized_force[d : d + 3] += bounded(f, s.support_force_limit_N)
            generalized_force[d + 3 : d + 6] -= (
                rotation_damping * data.qvel[d + 3 : d + 6]
            )

    def after_step(self, world, contacts, dt):
        loads = contacts.loads_by_body
        for slot in self.slots:
            if slot["status"] == "active":
                slot["maximum_active_displacement_m"] = max(
                    slot["maximum_active_displacement_m"],
                    float(
                        np.linalg.norm(
                            world.data.qpos[slot["qpos"] : slot["qpos"] + 3]
                            - slot["spawn_position"]
                        )
                    ),
                )
                slot["maximum_active_speed_m_s"] = max(
                    slot["maximum_active_speed_m_s"],
                    float(
                        np.linalg.norm(world.data.qvel[slot["dof"] : slot["dof"] + 3])
                    ),
                )
            if slot["status"] != "withdrawing":
                continue
            name = world.model.body(slot["body"]).name
            unloaded = loads.get(
                name, 0.0
            ) <= self.schedule.unloaded_force_N and not any(
                name in (c.body1, c.body2) and c.penetration_m > 0
                for c in contacts.contacts
            )
            slow = (
                np.linalg.norm(world.data.qvel[slot["dof"] : slot["dof"] + 3])
                <= self.schedule.retirement_speed_m_s
                and np.linalg.norm(world.data.qvel[slot["dof"] + 3 : slot["dof"] + 6])
                <= self.schedule.retirement_angular_speed_rad_s
            )
            quiet = (
                unloaded
                and slow
                and world.time_s - slot["withdraw_at"] >= self.schedule.withdrawal_s
            )
            slot["quiet_since"] = (
                (world.time_s if slot["quiet_since"] is None else slot["quiet_since"])
                if quiet
                else None
            )
            if (
                slot["quiet_since"] is not None
                and world.time_s - slot["quiet_since"] >= self.schedule.quiet_s
            ):
                kinetic = self.kinetic_energy(world, slot)
                self._park(world, slot)
                world.emit(
                    "obstacle_deactivated",
                    name=slot["spec"].name,
                    removed_kinetic_energy_J=kinetic,
                    unloaded=True,
                )

    @staticmethod
    def kinetic_energy(world, slot):
        d = slot["dof"]
        v = world.data.qvel[d : d + 6]
        s = slot["spec"]
        return float(
            0.5 * s.mass_kg * (v[:3] @ v[:3]) + 0.5 * np.sum(s.inertia() * v[3:] ** 2)
        )

    def states(self, world):
        return [
            {
                "name": s["spec"].name,
                "status": s["status"],
                "position": world.data.qpos[s["qpos"] : s["qpos"] + 3].tolist(),
                "quaternion_xyzw": world.data.qpos[s["qpos"] + 3 : s["qpos"] + 7][
                    [1, 2, 3, 0]
                ].tolist(),
                "velocity": world.data.qvel[s["dof"] : s["dof"] + 6].tolist(),
                "kinetic_energy_J": self.kinetic_energy(world, s),
                "maximum_active_displacement_m": s["maximum_active_displacement_m"],
                "maximum_active_speed_m_s": s["maximum_active_speed_m_s"],
            }
            for s in self.slots
        ]

    def manifest(self):
        return {
            "component": type(self).__name__,
            "specs": [asdict(s) for s in self.specs],
            "schedule": asdict(self.schedule),
            "human_tissue_calibrated": False,
            "support": "finite-force translational spring/damper; optional gravity compensation",
            "disappearance": "bounded-force withdrawal; no deletion while loaded or translating quickly",
        }


@dataclass(frozen=True)
class ObstacleDistribution:
    sphere_probability: float = 0.5
    mass_kg: tuple = (0.35, 2.0)
    radius_m: tuple = (0.025, 0.055)
    capsule_length_m: tuple = (0.12, 0.28)
    friction: tuple = (0.2, 0.6)
    contact_time_constant_s: tuple = (0.03, 0.06)
    support_stiffness_N_m: tuple = (10.0, 35.0)
    support_damping_Ns_m: tuple = (3.0, 6.0)

    def __post_init__(self):
        if (
            not np.isfinite(self.sphere_probability)
            or not 0 <= self.sphere_probability <= 1
        ):
            raise ValueError("Invalid sphere probability")
        for name, bounds in asdict(self).items():
            if name == "sphere_probability":
                continue
            if (
                len(bounds) != 2
                or not np.isfinite(bounds).all()
                or not 0 <= bounds[0] <= bounds[1]
            ):
                raise ValueError("Invalid obstacle distribution: " + name)
        if (
            self.mass_kg[0] <= 0
            or self.radius_m[0] <= 0
            or self.contact_time_constant_s[0] < 0.004
        ):
            raise ValueError("Invalid mass/size/compliance distribution")


def randomized_obstacles(
    rng, count=2, *, schedule=ObstacleSchedule(), distribution=ObstacleDistribution()
):
    """Episode factory: random geometry/mass BEFORE compilation, not stale BVH edits."""
    if type(count) is not int or count < 1:
        raise ValueError("Positive obstacle count required")
    specs = []
    for i in range(count):
        values = {
            name: float(rng.uniform(*bounds))
            for name, bounds in asdict(distribution).items()
            if name != "sphere_probability"
        }
        if rng.random() < distribution.sphere_probability:
            values["capsule_length_m"] = 0.0
        specs.append(MovableObstacleSpec(f"movable_{i}", **values))
    return DynamicObstacles(specs, schedule)
