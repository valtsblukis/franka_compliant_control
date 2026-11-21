"""Layer 0: MuJoCo world + named robot state + optional scene components.

No SDK controller, policy, sensor model, reward, Gym, HTTP or robot device.
"""

from __future__ import annotations
from copy import deepcopy
import hashlib
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from ..twin_model import MujocoPlant, PlantParameters, build_mjcf
from .contacts import ContactOracle, ContactReport


class SimulationWorld:
    def __init__(
        self, home, parameters=PlantParameters(), *, components=(), mesh_dir=None
    ):
        self.home = deepcopy(home)
        self.parameters = parameters
        self.components = tuple(components)
        if len({id(c) for c in self.components}) != len(self.components):
            raise ValueError("Duplicate scene component instance")
        root = ET.fromstring(
            build_mjcf(home, parameters, mesh_dir=mesh_dir, targets={})
        )
        for component in self.components:
            component.build(root, self.home)
        # No-op world retains the exact old XML/model, including solver settings.
        xml = ET.tostring(root, encoding="unicode") if self.components else None
        self.plant = MujocoPlant(
            home, parameters, mesh_dir=mesh_dir, targets={}, mjcf=xml
        )
        self.model, self.data = self.plant.model, self.plant.data
        self.robot_qpos = self.plant.robot_qpos_indices
        self.robot_dofs = self.plant.robot_dof_indices
        root_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        if root_body < 0:
            raise ValueError("Robot root link0 not found")
        self.robot_bodies = frozenset(
            b for b in range(1, self.model.nbody) if self._is_descendant(b, root_body)
        )
        self.oracle = ContactOracle(self.model, self.robot_bodies, self.robot_dofs)
        self.events = []
        self.last_contacts = ContactReport()
        for component in self.components:
            component.bind(self)
        self.reset()

    def _is_descendant(self, body, parent):
        while body:
            if body == parent:
                return True
            body = int(self.model.body_parentid[body])
        return False

    @property
    def q(self):
        return self.plant.robot_positions

    @property
    def dq(self):
        return self.plant.robot_velocities

    @property
    def time_s(self):
        return float(self.data.time)

    @property
    def physics_dt(self):
        return float(self.model.opt.timestep)

    def emit(self, kind, **payload):
        self.events.append({"time_s": self.time_s, "kind": kind, **payload})

    def reset(self, *, q=None, dq=None, seed=0):
        q = np.asarray(self.home["home"]["joints"] if q is None else q, dtype=float)
        dq = np.zeros(7) if dq is None else np.asarray(dq, dtype=float)
        if q.shape != (7,) or dq.shape != (7,) or not np.isfinite(np.r_[q, dq]).all():
            raise ValueError("Expected finite seven-joint initial state")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot_qpos] = q
        self.data.qvel[self.robot_dofs] = dq
        self.events.clear()
        self.last_contacts = ContactReport()
        # Independent component RNG streams; scene randomness does not consume a
        # task's goal sampler or sensor RNG stream.
        children = np.random.SeedSequence(seed).spawn(len(self.components))
        for component, child in zip(self.components, children):
            component.reset(self, np.random.default_rng(child))
        mujoco.mj_forward(self.model, self.data)

    def scene_forces(self, data):
        force = np.zeros(self.model.nv)
        for component in self.components:
            component.add_forces(self, data, force)
        if not np.isfinite(force).all():
            raise RuntimeError("Scene produced nonfinite generalized forces")
        return force

    def fill_forces(self, data, robot_torque):
        """robot_torque includes any robot-side gravity compensation, exactly once."""
        torque = np.asarray(robot_torque, dtype=float)
        if torque.shape != (7,) or not np.isfinite(torque).all():
            raise ValueError("Expected seven finite robot torques")
        external = self.scene_forces(data)
        data.qfrc_applied[:] = external
        data.qfrc_applied[self.robot_dofs] += torque - self.parameters.joint_bias
        return external[self.robot_dofs].copy()

    def step_physics(self, robot_torque) -> ContactReport:
        """One physics substep, not one policy tick. Contacts never averaged away."""
        for component in self.components:
            component.before_step(self, self.physics_dt)
        self.fill_forces(self.data, robot_torque)
        mujoco.mj_step(self.model, self.data)
        self.plant.check_numerics()
        self.last_contacts = self.oracle.read(self.data)
        for component in self.components:
            component.after_step(self, self.last_contacts, self.physics_dt)
        return self.last_contacts

    def copy_kinematic_state(self, destination):
        # Observer scratch data is intentionally independent of the plant solver
        # warm start. This retains the validated free-space observer convention.
        destination.time = self.data.time
        destination.qpos[:] = self.data.qpos
        destination.qvel[:] = self.data.qvel
        destination.act[:] = self.data.act
        destination.mocap_pos[:] = self.data.mocap_pos
        destination.mocap_quat[:] = self.data.mocap_quat

    def manifest(self):
        return {
            "schema": "fcc-simulation-world-v1",
            "physics_dt": self.physics_dt,
            "home_sha256": self.home.get("sha256"),
            "compiled_mjcf_sha256": hashlib.sha256(self.plant.xml.encode()).hexdigest(),
            "plant": self.parameters.manifest(),
            "components": [c.manifest() for c in self.components],
            "robot_qpos_indices": self.robot_qpos.tolist(),
            "robot_dof_indices": self.robot_dofs.tolist(),
            "collision_geometry_calibrated": False,
            "hardware_authorized": False,
        }
