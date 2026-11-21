"""Versioned bounded native-callback recording schema and quality checks."""

import json
import numpy as np

SCHEMA = "fcc-native-telemetry-v1"
ID_COLUMNS = ["cycle", "epoch", "applied_seq", "mode", "phase", "reason", "flags"]
COLUMNS = [
    "robot_time_s",
    "callback_received_monotonic_s",
    "control_done_monotonic_s",
    "record_ready_monotonic_s",
    "period_s",
    "command_accepted_monotonic_s",
    "command_deadline_monotonic_s",
    "communication_success",
]
GROUPS = {}
for name, size in [
    ("q", 7),
    ("dq", 7),
    ("theta", 7),
    ("dtheta", 7),
    ("tau_measured", 7),
    ("tau_desired_previous", 7),
    ("controller_torque_before_sdk", 7),
    ("coriolis", 7),
    ("position", 3),
    ("quaternion_xyzw", 4),
    ("twist", 6),
    ("reference_position", 3),
    ("reference_quaternion_xyzw", 4),
    ("reference_joints", 7),
    ("reference_joint_velocity", 7),
    ("raw_base_wrench", 6),
    ("stiffness_wrench", 6),
    ("external_joint_torques", 7),
]:
    GROUPS[name] = slice(len(COLUMNS), len(COLUMNS) + size)
    COLUMNS.extend(f"{name}_{i}" for i in range(size))
assert len(COLUMNS) == 117


def metadata(status):
    return {
        **status,
        "columns": COLUMNS,
        "id_columns": ID_COLUMNS,
        "clock_semantics": "robot_time is SDK clock; host timestamps share native steady_clock; callback_received brackets receipt, not physical sensor acquisition",
        "torque_semantics": "tau_measured is raw SDK tau_J; tau_desired_previous is SDK tau_J_d (gravity excluded); newly computed controller torque is before the SDK 100 Hz filter/rate processing and affects later states",
        "motor_signals": "theta/dtheta and measured torque valid only when flags bit 256 is set",
        "capture_scope": "control callbacks only; proprietary post-MotionFinished braking not modelled by these rows",
        "overflow_policy": "keep earliest records; count drops; never overwrite/block RT; overflow invalidates identification",
        "flags": {
            "robot_errors": 1,
            "joint_torque_limited": 2,
            "task_force_limited": 4,
            "task_moment_limited": 8,
            "reference_limited": 16,
            "osc_regularized": 32,
            "reference_speed_limited": 64,
            "effort_limited": 128,
            "sdk_signals_valid": 256,
        },
    }


def load(path):
    with np.load(path, allow_pickle=False) as a:
        values = a["values"].copy()
        ids = a["ids"].copy()
        meta = json.loads(str(a["metadata_json"].item()))
    if (
        meta["schema"] != SCHEMA
        or meta["columns"] != COLUMNS
        or meta["id_columns"] != ID_COLUMNS
    ):
        raise ValueError("Unknown native recording schema")
    if (
        values.shape != (len(ids), 117)
        or ids.shape != (len(values), 7)
        or ids.dtype != np.uint64
        or not np.isfinite(values).all()
    ):
        raise ValueError("Malformed native recording")
    return values, ids, meta


def quality(path):
    v, ids, m = load(path)
    n = len(v)
    if n < 2:
        raise ValueError("Insufficient native samples")
    dt = np.diff(v[:, 0])
    hostdt = np.diff(v[:, 1])
    cycle = np.diff(ids[:, 0])
    sdk = np.all((ids[:, 6] & 256) != 0)
    result = {
        "schema": SCHEMA,
        "capture_id": m["capture_id"],
        "samples": n,
        "capacity": m["capacity"],
        "dropped": m["dropped"],
        "sdk_signals_valid": bool(sdk),
        "cycle_sequence_contiguous": bool(np.all(cycle == 1)),
        "robot_clock_strictly_increasing": bool(np.all(dt > 0)) if sdk else None,
        "robot_clock_gaps_over_1_5ms": int(np.sum(dt > 0.0015)) if sdk else None,
        "maximum_robot_period_s": float(max(dt)) if sdk else None,
        "median_robot_period_s": float(np.median(dt)) if sdk else None,
        "maximum_host_callback_gap_s": float(max(hostdt)),
        "host_timestamp_order_valid": bool(
            np.all(v[:, 1] <= v[:, 2]) and np.all(v[:, 2] <= v[:, 3])
        ),
        "callback_through_record_ready_us_quantiles": (
            1e6 * np.quantile(v[:, 3] - v[:, 1], [0.5, 0.95, 0.99, 1])
        ).tolist(),
        "record_pack_us_quantiles": (
            1e6 * np.quantile(v[:, 3] - v[:, 2], [0.5, 0.95, 0.99, 1])
        ).tolist(),
        "recording_is_complete": bool(
            m["dropped"] == 0
            and n == m["count"]
            and np.all(cycle == 1)
            and not m["active"]
        ),
    }
    result["suitable_for_contiguous_hardware_identification"] = bool(
        result["recording_is_complete"]
        and sdk
        and np.all(dt > 0)
        and not np.any(dt > 0.0015)
        and result["host_timestamp_order_valid"]
    )
    return result
