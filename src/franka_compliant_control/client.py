"""Thin HTTP setpoint client: first command starts control, with no arm API."""
from __future__ import annotations

import math
from numbers import Integral, Real
import threading
from typing import Sequence

import requests

from .network import DEFAULT_URL


def _vector(values, size, name):
    """Accept one-dimensional lists/tuples/NumPy vectors, emit JSON-native floats."""
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a {size}-element numeric vector") from exc
    if (len(items) != size or any(isinstance(v, bool) or not isinstance(v, Real) for v in items)):
        raise ValueError(f"{name} must be a {size}-element numeric vector")
    result = [float(v) for v in items]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite values")
    return result


class CompliantClient:
    def __init__(self, base_url=DEFAULT_URL, *, token=None, timeout_s=2.0):
        self.base_url = base_url.rstrip("/")
        self.http = requests.Session()
        if token:
            self.http.headers["Authorization"] = f"Bearer {token}"
        self.timeout_s = timeout_s
        self._sent_setpoints = False
        self.sequence = 0
        self.command_timeout_s = None
        self.lock = threading.RLock()

    def _request(self, method, path, **kwargs):
        response = self.http.request(method, self.base_url + path, timeout=self.timeout_s, **kwargs)
        response.raise_for_status()
        return response.json()

    def state(self):
        return self._request("GET", "/state")

    def refresh_state(self):
        """One inactive robot observation, without starting control or recovery."""
        return self._request("POST", "/state/refresh")

    def _send(self, path, payload, valid_for_s, seq=None):
        with self.lock:
            lifetime = valid_for_s
            if lifetime is not None and (not isinstance(lifetime, Real) or isinstance(lifetime, bool) or
                    not math.isfinite(lifetime) or lifetime <= 0):
                raise ValueError("valid_for_s must be finite, positive and no greater than command_timeout_s")
            if seq is None:
                seq = self.sequence+1
            minimum = self.sequence if self._sent_setpoints else -1
            if isinstance(seq, bool) or not isinstance(seq, Integral) or not minimum < seq < 2**63:
                raise ValueError("seq must be a nonnegative, strictly increasing integer below 2**63")
            self.sequence = int(seq)
            payload["seq"] = self.sequence
            if lifetime is not None:
                payload["valid_for_s"] = float(lifetime)
            self._sent_setpoints = True  # Even a lost response may have started control.
            # Do not retry expired or reordered motion requests.
            return self._request("POST", path, json=payload)

    def send_pose(self, position: Sequence[float], quaternion_xyzw=None, *, twist=None, acceleration=None,
                  posture=None, valid_for_s=None, seq=None):
        payload = {"position": _vector(position, 3, "position")}
        for name, value, size in (("quaternion_xyzw", quaternion_xyzw, 4), ("twist", twist, 6),
                                  ("acceleration", acceleration, 6), ("posture", posture, 7)):
            if value is not None:
                payload[name] = _vector(value, size, name)
        return self._send("/target_ee_pose", payload, valid_for_s, seq)

    def send_joint_state(self, positions: Sequence[float], velocities=None, *, valid_for_s=None, seq=None):
        payload = {"positions": _vector(positions, 7, "positions")}
        if velocities is not None:
            payload["velocities"] = _vector(velocities, 7, "velocities")
        return self._send("/target_joint_state", payload, valid_for_s, seq)

    # FrankyClient-style names: first setpoint starts control; no automatic recovery.
    set_target_ee_pose = send_pose
    set_target_joint_state = send_joint_state

    def get_target_ee_pose(self):
        return self._request("GET", "/target_ee_pose")

    def get_current_ee_pose(self):
        return self._request("GET", "/ee_pose")

    def get_target_joint_state(self):
        return self._request("GET", "/target_joint_state")

    def get_current_joint_state(self):
        return self._request("GET", "/joint_state")

    def health(self):
        return self._request("GET", "/health")

    def prepare_telemetry(self, capacity=90000, *, speed_limits=None):
        """Allocate one bounded capture while disarmed; optional native-rate experiment stops."""
        payload={'capacity':capacity}
        if speed_limits is not None:
            values=_vector(speed_limits,3,'joint/linear/angular speed limits')
            payload.update(zip(('joint_speed','linear_speed','angular_speed'),values))
        return self._request('POST','/telemetry/prepare',json=payload)

    def telemetry_status(self):
        return self._request('GET','/telemetry/status')

    def download_telemetry(self, path):
        from pathlib import Path
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        response=self.http.get(self.base_url+'/telemetry/data',timeout=max(30.,self.timeout_s))
        response.raise_for_status()
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(response.content)
        return self.telemetry_status()

    def clear_telemetry(self,capture_id):
        return self._request('POST','/telemetry/clear',json={'capture_id':capture_id})

    def get_urdf(self):
        return self._request("GET", "/urdf")

    def about(self):
        return self._request("GET", "/about", params={"format": "json"})

    def get_command_timeout(self):
        return self._request("GET", "/command_timeout")

    def set_command_timeout(self, command_timeout_s):
        if (isinstance(command_timeout_s, bool) or not isinstance(command_timeout_s, Real) or
                not math.isfinite(command_timeout_s) or command_timeout_s <= 0):
            raise ValueError("command_timeout_s must be finite and positive")
        with self.lock:
            result = self._request("POST", "/command_timeout", json={"command_timeout_s": float(command_timeout_s)})
            self._sent_setpoints = False
            self.sequence = 0
            self.command_timeout_s = result["command_timeout_s"]
            return result

    def reset_command_timeout(self):
        with self.lock:
            result = self._request("POST", "/command_timeout/reset")
            self._sent_setpoints = False
            self.sequence = 0
            self.command_timeout_s = result["command_timeout_s"]
            return result

    def get_automatic_recovery(self):
        return self._request("GET", "/automatic_recovery")

    def set_automatic_recovery(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        return self._request("POST", "/automatic_recovery", json={"enabled": enabled})

    def stop(self, wait_s=1.0):
        with self.lock:
            self._sent_setpoints = False
            self.sequence = 0
        return self._request("POST", "/stop", params={"wait_s": wait_s})

    def get_motion_inhibit(self):
        return self._request('GET', '/motion_inhibit')

    def inhibit_motion(self, wait_s=0.0):
        """Assert the global software latch. Never automatically release/retry it."""
        return self._request('POST', '/motion_inhibit', params={'wait_s': wait_s})

    def release_motion_inhibit(self, release_token):
        """Explicit release of the latest assertion; does not resume motion."""
        if not isinstance(release_token, str) or not release_token:
            raise ValueError('Expected the release_token from the latest assertion/status')
        return self._request('POST', '/motion_inhibit/release', json={'release_token': release_token})

    def go_home(self, **options):
        """Start finite homing; stops/disarms on completion. Returns acceptance, not completion."""
        with self.lock:
            self._sent_setpoints = True
            return self._request("POST", "/go_home", **({"json": options} if options else {}))

    def backoff_joint_limits(self, **options):
        """Request one native inward-only action; acceptance is NOT completion.

        Minimal input: confirm_clear_path=True, after operator path review.
        Native selects joints/targets and checks inactive local-fault eligibility.
        Never captures a pose, changes recovery policy, recovers, or retries.
        """
        from .models import JointLimitBackoffRequest
        payload = JointLimitBackoffRequest.model_validate(options).model_dump(mode="json")
        with self.lock:
            self._sent_setpoints = True  # A lost response may still have started backoff.
            return self._request("POST", "/backoff_joint_limits", json=payload)

    def relieve_contact(self, **options):
        """Request bounded native relief once; acceptance is NOT release/unjamming.

        Defaults to manual raw-wrench following: confirm an operator-supervised
        cleared local retreat/braking area, then stop explicitly. A TCP box is
        optional; automatic mode requires a box and verified bias/joints/provenance.
        Requires inactive Disarmed control with automatic recovery OFF. Never
        captures bias, changes recovery policy, recovers, or retries motion.
        """
        from .models import ContactReliefRequest
        for name, size in (("bias_wrench_tcp_base", 6), ("bias_joint_positions", 7),
                           ("workspace_lower", 3), ("workspace_upper", 3)):
            if options.get(name) is not None:
                options[name] = _vector(options[name], size, name)
        payload = ContactReliefRequest.model_validate(options).model_dump(mode="json")
        with self.lock:
            self._sent_setpoints = True  # A lost response may still have started relief.
            return self._request("POST", "/relieve_contact", json=payload)

    def recover(self):
        with self.lock:
            result = self._request("POST", "/recover")
            self._sent_setpoints = False
            self.sequence = 0
            return result

    def get_config(self):
        return self._request("GET", "/controller_config")

    def configure(self, **changes):
        with self.lock:
            config = self.get_config()
            config.update(changes)
            result = self._request("PUT", "/controller_config", json=config)
            self._sent_setpoints = False
            self.sequence = 0
            return result

    def close(self):
        try:
            if self._sent_setpoints:
                self.stop()
        finally:
            self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
