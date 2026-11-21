"""Immutable experiment inputs and a shared 50 Hz pose-only manual policy.

Only capture_home performs robot I/O, through an already-running HTTP service.
Its only POST is the existing inactive state read; it cannot send motion commands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import requests

from .models import ControllerConfig, HardwareSettings
from .trajectory import segment_sample


def digest(value):
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(text)


def load_home(path):
    home = json.loads(Path(path).read_text())
    if home.get("schema") != "fcc-twin-home-v1":
        raise ValueError("not a saved twin home capture")
    for key in ("controller_config", "hardware_settings", "urdf", "home"):
        if digest(home[key]) != home["sha256"][key]:
            raise ValueError(f"saved {key} fingerprint differs; do not silently reinitialize home")
    ControllerConfig.model_validate(home["controller_config"])
    HardwareSettings.model_validate(home["hardware_settings"])
    return home


def capture_home(path, hardware_path, url="http://127.0.0.1:54321", *, http=None):
    """Explicit fresh inactive read, never arm/stop/recover/configure; no overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Home is already fixed at {path}; choose a new explicit capture path")
    hardware = HardwareSettings.model_validate_json(Path(hardware_path).read_text())
    if hardware.robot_model != "fr3" or hardware.expected_end_effector is None:
        raise ValueError("FR3 and a recorded expected tool mass/COM/inertia are required")
    owns_http = http is None
    http = requests.Session() if owns_http else http
    base = url.rstrip("/")
    def get(endpoint):
        response = http.get(base + endpoint, timeout=5)
        response.raise_for_status()
        return response.json()
    try:
        before = get("/state")
        if before["backend"] != "robot" or before["running"] or before["phase"] != "disarmed":
            raise RuntimeError("Home capture requires an already-connected, inactive, disarmed robot service")
        response = http.post(base + "/state/refresh", timeout=5)
        response.raise_for_status()
        state = get("/state")
        if (state["running"] or state["phase"] != "disarmed" or state["robot_mode"] != "Idle"
                or state["robot_has_errors"] or not state["sample_valid"]
                or not state["kinematics_valid"] or state["sample_age_s"] > .1
                or state["robot_time_s"] <= before["robot_time_s"]):
            raise RuntimeError("No fresh valid error-free Idle observation; home not captured")
        if np.max(np.abs(state["velocities"])) >= .02:
            raise RuntimeError("Robot is not stationary; home not captured")
        config = ControllerConfig.model_validate(get("/controller_config"))
        if config.compliance_mode not in ("auto", "operational_space") or config.osc_feedback_mode != "saturated":
            raise ValueError("This experiment currently requires the saturated Cartesian OSC law")
        urdf = get("/urdf")["urdf"]
        from . import _native
        pose, _ = _native.OfflineKinematics(urdf, hardware.flange_to_tcp).evaluate(state["positions"])
        # A service can be inactive before its configured TCP was applied. Do not
        # relabel that measured pose as the configured TCP unless FK agrees.
        from .twin_model import quaternion_matrix
        position_error = float(np.linalg.norm(pose[:3, 3] - state["position"]))
        rotation_error = float(np.linalg.norm(pose[:3, :3] - quaternion_matrix(state["quaternion_xyzw"])))
        if position_error > 1e-4 or rotation_error > 1e-3:
            raise RuntimeError("Measured pose and configured-TCP FK disagree; no setters were applied")
        result = {
            "schema": "fcc-twin-home-v1",
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "source_url": base,
            "capture_operation": "GET state; POST inactive state/refresh; GET state/config/urdf; no motion",
            "home": {"position": state["position"], "quaternion_xyzw": state["quaternion_xyzw"],
                     "joints": state["positions"]},
            "robot_state": state,
            "controller_config": config.model_dump(),
            "hardware_settings": hardware.model_dump(),
            "urdf": urdf,
            "fk_position_error_m": position_error,
            "fk_rotation_matrix_error": rotation_error,
            "load_provenance": "local expected tool configuration; service refresh validates Desk tool; additional load from local file",
            "native_extension_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        }
        result["sha256"] = {key: digest(result[key]) for key in ("home", "urdf", "controller_config", "hardware_settings")}
        write_json_new(path, result)
        return result
    finally:
        if owns_http:
            http.close()


@dataclass(frozen=True)
class ManualPosePolicy:
    """Fixed home -> +base Y -> home -> -base Y -> home, repeating explicitly.

    Left means +base Y, right means -base Y, not camera/operator left/right.
    Quintic pacing is policy-side filtering. Only poses are transmitted.
    """
    home_position: tuple[float, float, float]
    home_quaternion_xyzw: tuple[float, float, float, float]
    offset_m: float = .30
    move_s: float = 3.0
    dwell_s: float = 2.0
    initial_hold_s: float = 1.0
    cycles: int = 1
    hz: int = 50
    # Optional immutable measured start for gentle alignment to the SAME home.
    # This is a policy-side prefix, not a redefinition of any experiment target.
    start_position: tuple[float, float, float] | None = None
    start_quaternion_xyzw: tuple[float, float, float, float] | None = None
    start_joints: tuple[float, ...] | None = None
    alignment_s: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "home_position", tuple(self.home_position))
        object.__setattr__(self, "home_quaternion_xyzw", tuple(self.home_quaternion_xyzw))
        if (len(self.home_position) != 3 or len(self.home_quaternion_xyzw) != 4
                or not all(math.isfinite(v) for v in (*self.home_position, *self.home_quaternion_xyzw))
                or abs(np.linalg.norm(self.home_quaternion_xyzw)-1) > 1e-6):
            raise ValueError("Invalid fixed home pose")
        starts = (self.start_position, self.start_quaternion_xyzw, self.start_joints)
        if any(value is not None for value in starts):
            if any(value is None for value in starts) or not math.isfinite(self.alignment_s) or self.alignment_s < 1:
                raise ValueError("Alignment requires a complete fixed measured start and at least one second")
            for name, size in (("start_position",3),("start_quaternion_xyzw",4),("start_joints",7)):
                value = tuple(getattr(self,name))
                if len(value) != size or not all(math.isfinite(v) for v in value):
                    raise ValueError("Invalid measured alignment start")
                object.__setattr__(self,name,value)
            if abs(np.linalg.norm(self.start_quaternion_xyzw)-1) > 1e-6:
                raise ValueError("Invalid alignment quaternion")
            if (np.linalg.norm(np.array(self.start_position)-self.home_position) > .002
                    or abs(np.dot(self.start_quaternion_xyzw,self.home_quaternion_xyzw)) < math.cos(math.radians(1)/2)):
                raise ValueError("Alignment is only for a start within 2 mm / 1 degree of fixed home")
        elif self.alignment_s != 0:
            raise ValueError("Alignment needs a measured start")
        for value in (self.offset_m, self.move_s, self.dwell_s, self.initial_hold_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Policy distances/times must be finite and positive")
        if self.hz != 50 or type(self.cycles) is not int or not 1 <= self.cycles <= 100:
            raise ValueError("This comparison uses 50 Hz and 1..100 finite cycles")
        # Stable integer publication schedule, including endpoint transitions.
        if any(abs(v*self.hz-round(v*self.hz)) > 1e-8
               for v in (self.move_s, self.dwell_s, self.initial_hold_s, self.alignment_s)):
            raise ValueError("Policy times must be multiples of the 50 Hz period")

    @classmethod
    def from_home(cls, capture, **kwargs):
        home = capture["home"]
        return cls(tuple(home["position"]), tuple(home["quaternion_xyzw"]), **kwargs)

    @property
    def duration_s(self):
        return self.alignment_s + self.initial_hold_s + self.cycles*4*(self.move_s+self.dwell_s)

    @property
    def targets(self):
        home = np.asarray(self.home_position)
        return {"home": home.tolist(), "left": (home+[0,self.offset_m,0]).tolist(),
                "right": (home-[0,self.offset_m,0]).tolist()}

    def sample(self, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Policy time must be finite and nonnegative")
        # Quantize intentionally: simulator and HTTP publisher use identical
        # zero-order-held 50 Hz poses, not a hidden 1 kHz reference trajectory.
        tick = int(math.floor(time_s*self.hz + 1e-8))
        time_s = tick/self.hz
        leg_s = self.move_s+self.dwell_s
        sequence = ("home", "left", "home", "right", "home")
        if time_s < self.alignment_s:
            position,_,_ = segment_sample(self.start_position,self.home_position,time_s,self.alignment_s)
            u = time_s/self.alignment_s
            fraction = u*u*u*(10+u*(-15+6*u))
            a,b = np.array(self.start_quaternion_xyzw),np.array(self.home_quaternion_xyzw)
            if a@b < 0: b = -b
            angle = math.acos(float(np.clip(a@b,-1,1)))
            q = ((1-fraction)*a+fraction*b if angle < 1e-8 else
                 (math.sin((1-fraction)*angle)*a+math.sin(fraction*angle)*b)/math.sin(angle))
            result = self._sample(tick,"align_home",-2,"home",position,True)
            result["quaternion_xyzw"] = (q/np.linalg.norm(q)).tolist()
            return result
        if time_s >= self.duration_s:
            return self._sample(tick, "complete", 4*self.cycles, "home", self.targets["home"], False)
        if time_s-self.alignment_s < self.initial_hold_s:
            return self._sample(tick, "initial_hold", -1, "home", self.targets["home"], False)
        elapsed = time_s-self.alignment_s-self.initial_hold_s
        leg = min(int((elapsed+1e-9)/leg_s), 4*self.cycles-1)
        within = elapsed-leg*leg_s
        start, goal = sequence[leg%4], sequence[leg%4+1]
        position, _, _ = segment_sample(self.targets[start], self.targets[goal], within, self.move_s)
        moving = within < self.move_s-1e-9
        return self._sample(tick, "move" if moving else "dwell", leg, goal, position, moving)

    def _sample(self, tick, phase, leg, goal, position, moving):
        return {"policy_tick": tick, "policy_time_s": tick/self.hz, "policy_phase": phase,
                "leg": leg, "goal": goal, "goal_position": self.targets[goal], "moving": moving,
                "position": position, "quaternion_xyzw": list(self.home_quaternion_xyzw)}

    def manifest(self):
        fields = asdict(self)
        if self.alignment_s == 0:
            # Preserve old manifests/fingerprints exactly for unaligned runs.
            for key in ("start_position","start_quaternion_xyzw","start_joints","alignment_s"):
                fields.pop(key)
        return {"schema": "fcc-manual-pose-policy-aligned-v1" if self.alignment_s else "fcc-manual-pose-policy-v1",
                **fields, "duration_s": self.duration_s,
                "targets": self.targets, "order": ["home", "left", "home", "right", "home"],
                "left_definition": "+base Y", "right_definition": "-base Y",
                "reference": "policy-side quintic pacing; pose only; 50 Hz zero-order hold",
                "twist_feedforward": False, "acceleration_feedforward": False}
