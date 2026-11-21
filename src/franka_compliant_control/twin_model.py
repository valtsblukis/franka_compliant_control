"""FR3 URDF-to-MuJoCo rigid-body twin, with explicit uncalibrated plant parameters.

Inertias and kinematic transforms come from the robot-exported URDF, not a Panda
substitute. The attached tool comes from the recorded flange-frame load model.
No network or robot I/O occurs in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def quaternion_matrix(xyzw):
    q = np.asarray(xyzw, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-5:
        raise ValueError("Expected normalized xyzw quaternion")
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def matrix_quaternion(matrix):
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(matrix, dtype=float).reshape(9))
    return q[[1,2,3,0]]


def rpy_matrix(rpy):
    r,p,y = rpy
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp,cp*sr,cp*cr]])


def rotation_distance(a, b):
    # Stable even for very small errors; atan2 rather than acos(trace).
    relative = quaternion_matrix(a) @ quaternion_matrix(b).T
    sine = np.linalg.norm([relative[2,1]-relative[1,2], relative[0,2]-relative[2,0],
                           relative[1,0]-relative[0,1]])/2
    cosine = np.clip((np.trace(relative)-1)/2, -1, 1)
    return float(np.arctan2(sine, cosine))


def combined_load(hardware):
    """Mass/COM/COM inertia, all expressed in flange axes; no double counting."""
    tool = hardware["expected_end_effector"]
    if tool is None:
        raise ValueError("Recorded end-effector load is required")
    masses = [tool["mass"], hardware["load_mass"]]
    coms = [np.array(tool["com"]), np.array(hardware["load_com"])]
    inertias = [np.array(tool["inertia"]).reshape(3,3,order="F"),
                np.array(hardware["load_inertia"]).reshape(3,3,order="F")]
    mass = sum(masses)
    com = sum((m*c for m,c in zip(masses,coms)), np.zeros(3))/mass if mass else np.zeros(3)
    inertia = np.zeros((3,3))
    for m,c,i in zip(masses,coms,inertias):
        d = c-com
        inertia += i + m*(d@d*np.eye(3)-np.outer(d,d))
    return {"mass": mass, "com": com.tolist(), "inertia": inertia.flatten(order="F").tolist()}


@dataclass(frozen=True)
class PlantParameters:
    # URDF's standard damping/friction values are a starting hypothesis, NOT
    # identified residual friction under Franka's internal compensation.
    viscous: tuple[float, ...] = (.003,)*7
    coulomb: tuple[float, ...] = (.2,)*7
    armature: tuple[float, ...] = (0.,)*7
    # Local effective resisting torque after robot-side compensation. Signed;
    # a fit may absorb load/cable/model bias here, not claim intrinsic friction.
    joint_bias: tuple[float, ...] = (0.,)*7
    # None preserves the original MuJoCo soft default. Near-rigid friction must
    # be selected explicitly and validated as a numerical model convention.
    friction_impedance: float | None = None
    friction_timeconst: float = .02
    tool_mass_scale: float = 1.0
    tool_inertia_scale: float = 1.0
    actuator_delay_steps: int = 0  # Additional whole native 1 ms cycles.
    physics_dt: float = .001

    def __post_init__(self):
        for values in (self.viscous, self.coulomb, self.armature):
            if len(values) != 7 or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("Plant friction/armature must be seven finite nonnegative values")
        if self.friction_impedance is not None and (not math.isfinite(self.friction_impedance)
                or not 0 < self.friction_impedance <= .9999):
            raise ValueError("Friction impedance must be in (0,.9999] or None")
        if not math.isfinite(self.friction_timeconst) or self.friction_timeconst < 2*self.physics_dt:
            raise ValueError("Friction time constant must be at least two physics steps")
        if len(self.joint_bias) != 7 or not all(math.isfinite(v) for v in self.joint_bias):
            raise ValueError("Plant joint bias must contain seven finite values")
        for value in (self.tool_mass_scale, self.tool_inertia_scale, self.physics_dt):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Plant scales and timestep must be finite and positive")
        if self.physics_dt > .001 or abs(.001/self.physics_dt-round(.001/self.physics_dt)) > 1e-8:
            raise ValueError("Physics step must divide the native 1 ms period")
        if type(self.actuator_delay_steps) is not int or not 0 <= self.actuator_delay_steps <= 20:
            raise ValueError("Actuator delay must be 0..20 native steps")

    def manifest(self):
        return {**asdict(self), "identified": False,
                "provenance": "URDF rigid bodies; estimated tool; configured plant hypotheses; see a separate fit report for estimated terms"}


def _numbers(values):
    return " ".join(format(float(v), ".17g") for v in values)


def _origin(element):
    if element is None:
        return np.zeros(3), np.eye(3)
    return (np.fromstring(element.get("xyz", "0 0 0"), sep=" "),
            rpy_matrix(np.fromstring(element.get("rpy", "0 0 0"), sep=" ")))


def build_mjcf(capture, parameters=PlantParameters(), *, mesh_dir=None, targets=None):
    """Generate a free-space plant. Meshes are visual, NOT cleared-path evidence.

    Environment/tool collision geometry has not been surveyed or validated.
    Joint limits remain active; no synthetic contact sensor is invented.
    """
    robot = ET.fromstring(capture["urdf"])
    links = {link.attrib["name"]: link for link in robot.findall("link")}
    joints = robot.findall("joint")
    moving = [j for j in joints if j.attrib["type"] != "fixed"]
    if [j.attrib["name"] for j in moving] != [f"joint{i}" for i in range(1,8)]:
        raise ValueError("Expected the robot-exported FR3 seven-joint URDF")
    roots = set(links)-{j.find("child").attrib["link"] for j in joints}
    if len(roots) != 1 or "link8" not in links:
        raise ValueError("Expected one base and link8 flange")
    children = {}
    for joint in joints:
        children.setdefault(joint.find("parent").attrib["link"], []).append(joint)
    root = ET.Element("mujoco", model="fcc_fr3_captured_load")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false", fusestatic="false")
    ET.SubElement(root, "option", timestep=str(parameters.physics_dt), gravity="0 0 -9.81",
                  integrator="implicitfast", solver="Newton", iterations="50", tolerance="1e-10")
    ET.SubElement(root, "size", njmax="200", nconmax="100")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="display_floor", type="plane", size="2 2 .01", pos="0 0 -.01",
                  rgba=".16 .18 .21 1", contype="0", conaffinity="0")
    targets = ({"home": capture["home"]["position"],
                "left": np.array(capture["home"]["position"])+[0,.3,0],
                "right": np.array(capture["home"]["position"])-[0,.3,0]} if targets is None else targets)
    palette = ("0 .9 .3 .8", ".2 .5 1 .8", "1 .6 .1 .8")
    for index, (label, position) in enumerate(targets.items()):
        ET.SubElement(world, "site", name="target_"+label, pos=_numbers(position), size=".012", rgba=palette[index%3])
    def body(parent, name, joint=None):
        pos, rot = _origin(joint.find("origin") if joint is not None else None)
        quat = matrix_quaternion(rot)[[3,0,1,2]]
        b = ET.SubElement(parent, "body", name=name, pos=_numbers(pos), quat=_numbers(quat))
        if joint is not None and joint.attrib["type"] != "fixed":
            index = int(joint.attrib["name"].removeprefix("joint"))-1
            limit = joint.find("limit")
            mj_joint = ET.SubElement(b, "joint", name=joint.attrib["name"], type="hinge",
                          axis=joint.find("axis").attrib["xyz"], limited="true",
                          range=f'{limit.attrib["lower"]} {limit.attrib["upper"]}',
                          damping=str(parameters.viscous[index]), frictionloss=str(parameters.coulomb[index]),
                          armature=str(parameters.armature[index]))
            if parameters.friction_impedance is not None:
                d = parameters.friction_impedance
                mj_joint.set("solimpfriction",f"{d} {d} .001 .5 2")
                mj_joint.set("solreffriction",f"{parameters.friction_timeconst} 1")
        inertial = links[name].find("inertial")
        if inertial is not None:
            pos_i, rot_i = _origin(inertial.find("origin"))
            a = {key:float(value) for key,value in inertial.find("inertia").attrib.items()}
            inertia = np.array([[a["ixx"],a["ixy"],a["ixz"]], [a["ixy"],a["iyy"],a["iyz"]],
                                [a["ixz"],a["iyz"],a["izz"]]])
            inertia = rot_i@inertia@rot_i.T
            ET.SubElement(b, "inertial", pos=_numbers(pos_i), mass=inertial.find("mass").attrib["value"],
                          fullinertia=_numbers(inertia[[0,1,2,0,0,1],[0,1,2,1,2,2]]))
        if mesh_dir is not None and name in {f"link{i}" for i in range(8)}:
            mesh = Path(mesh_dir)/f"{name}.stl"
            if not mesh.is_file():
                raise FileNotFoundError(mesh)
            ET.SubElement(asset, "mesh", name=name+"_mesh", file=str(mesh.resolve()))
            ET.SubElement(b, "geom", type="mesh", mesh=name+"_mesh", rgba=".8 .82 .85 1",
                          contype="0", conaffinity="0", group="1")
        if name == "link8":
            load = combined_load(capture["hardware_settings"])
            if load["mass"] > 0:
                tool = ET.SubElement(b, "body", name="tool_load")
                inertia = np.array(load["inertia"]).reshape(3,3,order="F")*parameters.tool_inertia_scale
                ET.SubElement(tool, "inertial", pos=_numbers(load["com"]),
                              mass=str(load["mass"]*parameters.tool_mass_scale),
                              fullinertia=_numbers(inertia[[0,1,2,0,0,1],[0,1,2,1,2,2]]))
                # Clearly approximate visual only. Never used for inertias/contact.
                ET.SubElement(tool, "geom", type="box", pos="0 0 .07", size=".045 .035 .07",
                              rgba=".18 .2 .23 1", contype="0", conaffinity="0")
            tcp = np.array(capture["hardware_settings"]["flange_to_tcp"]).reshape(4,4,order="F")
            ET.SubElement(b, "site", name="tcp", pos=_numbers(tcp[:3,3]),
                          quat=_numbers(matrix_quaternion(tcp[:3,:3])[[3,0,1,2]]),
                          size=".01", rgba="1 .1 .2 1")
        for child in children.get(name, []):
            body(b, child.find("child").attrib["link"], child)
    body(world, roots.pop())
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


class MujocoPlant:
    def __init__(self, capture, parameters=PlantParameters(), *, mesh_dir=None, targets=None, mjcf=None):
        import mujoco
        self.parameters = parameters
        self.xml = build_mjcf(capture, parameters, mesh_dir=mesh_dir, targets=targets) if mjcf is None else mjcf
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}') for i in range(1,8)]
        if min(joints)<0 or any(self.model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("Expected the named seven robot hinge joints")
        # Scene free joints need not be after the robot in generalized state.
        self.robot_qpos_indices = self.model.jnt_qposadr[joints].copy()
        self.robot_dof_indices = self.model.jnt_dofadr[joints].copy()
        self.data = mujoco.MjData(self.model)
        self.gravity_data = mujoco.MjData(self.model)
        self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.data.qpos[self.robot_qpos_indices] = capture["home"]["joints"]
        # Start stationary, not at a noisy measured velocity; record this choice.
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def model_quantities(self, q=None, dq=None):
        import mujoco
        if q is not None:
            self.data.qpos[self.robot_qpos_indices] = q
        if dq is not None:
            self.data.qvel[self.robot_dof_indices] = dq
        mujoco.mj_forward(self.model, self.data)
        jacp, jacr = np.zeros((3,self.model.nv)), np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp_id)
        mass = np.zeros((self.model.nv,self.model.nv))
        mujoco.mj_fullM(self.model, mass, self.data.qM)
        self.gravity_data.qpos[:] = self.data.qpos
        self.gravity_data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.gravity_data)
        gravity = self.gravity_data.qfrc_bias.copy()
        return {"position": self.data.site_xpos[self.tcp_id].copy(),
                "quaternion_xyzw": matrix_quaternion(self.data.site_xmat[self.tcp_id].reshape(3,3)),
                "jacobian": np.vstack((jacp,jacr))[:,self.robot_dof_indices],
                "mass": mass[np.ix_(self.robot_dof_indices,self.robot_dof_indices)],
                "gravity": gravity[self.robot_dof_indices],
                "coriolis": (self.data.qfrc_bias-gravity)[self.robot_dof_indices].copy()}

    @property
    def robot_positions(self):
        return self.data.qpos[self.robot_qpos_indices].copy()

    @property
    def robot_velocities(self):
        return self.data.qvel[self.robot_dof_indices].copy()

    def apply_robot_torque(self, applied_torque):
        torque=np.asarray(applied_torque,dtype=float)
        if torque.shape!=(7,) or not np.isfinite(torque).all():raise ValueError('Expected seven finite robot torques')
        self.data.qfrc_applied[self.robot_dof_indices] += torque-self.parameters.joint_bias

    def check_numerics(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo produced nonfinite state")
        if any(w.number for w in self.data.warning):
            raise RuntimeError("MuJoCo numerical/constraint warning; inspect logs")

    def step(self, applied_torque):
        """Legacy 1 ms torque stepping; scene-aware stepping is in simulation.World."""
        import mujoco
        self.data.qfrc_applied[:]=0
        self.apply_robot_torque(applied_torque)
        for _ in range(round(.001/self.parameters.physics_dt)):
            mujoco.mj_step(self.model, self.data)
        self.check_numerics()
