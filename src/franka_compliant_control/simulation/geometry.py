"""Optional explicit, approximate robot collision proxies. Never alter inertias."""

from dataclasses import asdict, dataclass
import xml.etree.ElementTree as ET
import numpy as np
from .components import SceneComponent
from ..twin_model import _numbers


@dataclass(frozen=True)
class CollisionGeometryConfig:
    arm_radius_m: float = 0.045
    tool_radius_m: float = 0.025
    tcp_radius_m: float = 0.030
    self_collisions: bool = False
    contact_time_constant_s: float = 0.04
    contact_damping_ratio: float = 1.0

    def __post_init__(self):
        if type(self.self_collisions) is not bool:
            raise ValueError("Self-collision switch must be boolean")
        for value in [
            self.arm_radius_m,
            self.tool_radius_m,
            self.tcp_radius_m,
            self.contact_time_constant_s,
            self.contact_damping_ratio,
        ]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "Collision proxy dimensions/contact settings must be positive"
                )
        if self.contact_time_constant_s < 0.004:
            raise ValueError("Contact time constant must be >=4 ms")


class RobotCollisionGeometry(SceneComponent):
    """Coarse capsules, NOT surveyed arm/tool/scene or human tissue geometry.

    Category bits: robot=1, obstacles=2, optional robot-self=4.
    Existing visual meshes/floor remain collision-disabled. Parent-neighborhood
    body pairs are excluded because proxies meet/overlap at their joints.
    """

    def __init__(self, config=CollisionGeometryConfig()):
        self.config = config

    def build(self, root, home):
        c = self.config
        if any(
            g.get("name", "").startswith("robot_contact_") for g in root.iter("geom")
        ):
            raise ValueError("Robot collision geometry already installed")
        bodies = {b.get("name"): b for b in root.iter("body")}
        attrs = {
            "density": "0",
            "mass": "0",
            "contype": str(1 | (4 if c.self_collisions else 0)),
            "conaffinity": str(2 | (4 if c.self_collisions else 0)),
            "condim": "3",
            "friction": ".5 .005 .0001",
            "solref": f"{c.contact_time_constant_s} {c.contact_damping_ratio}",
            "solimp": ".8 .95 .003 .5 2",
            "rgba": ".3 .6 .9 .25",
            "group": "3",
        }
        for i in range(8):
            body = bodies[f"link{i}"]
            child = body.find(f"body[@name='link{i+1}']")
            end = (
                np.fromstring(child.get("pos", "0 0 0"), sep=" ")
                if child is not None
                else np.zeros(3)
            )
            name = f"robot_contact_link{i}"
            if np.linalg.norm(end) > 0.02:
                ET.SubElement(
                    body,
                    "geom",
                    name=name,
                    type="capsule",
                    fromto=_numbers(np.r_[np.zeros(3), end]),
                    size=str(c.arm_radius_m),
                    **attrs,
                )
            else:
                ET.SubElement(
                    body,
                    "geom",
                    name=name,
                    type="sphere",
                    size=str(c.arm_radius_m),
                    **attrs,
                )
        tcp = np.array(home["hardware_settings"]["flange_to_tcp"]).reshape(
            4, 4, order="F"
        )[:3, 3]
        if np.linalg.norm(tcp) > 0.001:
            ET.SubElement(
                bodies["link8"],
                "geom",
                name="robot_contact_tool",
                type="capsule",
                fromto=_numbers(np.r_[np.zeros(3), tcp]),
                size=str(c.tool_radius_m),
                **attrs,
            )
        ET.SubElement(
            bodies["link8"],
            "geom",
            name="robot_contact_tcp",
            type="sphere",
            pos=_numbers(tcp),
            size=str(c.tcp_radius_m),
            **attrs,
        )
        if c.self_collisions:
            parent = {
                child.get("name"): b.get("name")
                for b in root.iter("body")
                for child in b.findall("body")
            }
            contact = root.find("contact")
            if contact is None:
                contact = ET.SubElement(root, "contact")
            names = [f"link{i}" for i in range(9)]
            for i, a in enumerate(names):
                for b in names[i + 1 :]:

                    def ancestors(n):
                        result = {n: 0}
                        while n in parent:
                            next_name = parent[n]
                            result[next_name] = result[n] + 1
                            n = next_name
                        return result

                    aa, bb = ancestors(a), ancestors(b)
                    distance = min(
                        (aa[n] + bb[n] for n in aa.keys() & bb.keys()), default=100
                    )
                    if distance <= 3:
                        ET.SubElement(contact, "exclude", body1=a, body2=b)

    def manifest(self):
        return {
            "component": type(self).__name__,
            "config": asdict(self.config),
            "geometry": "approximate massless capsules/spheres",
            "surveyed": False,
        }
