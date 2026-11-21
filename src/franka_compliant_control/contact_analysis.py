"""Compare recorded contact estimates with finite candidate thresholds, offline.

This is an observation comparison, not a dynamics replay or a safety validation.
It cannot reconstruct samples omitted by libfranka after a hardware reflex.
"""
import json
import math
from pathlib import Path


def _vector(values, length):
    if len(values) != length or not all(math.isfinite(v) for v in values):
        raise ValueError("Contact observations must contain finite vectors of the expected size")
    return list(values)


def _tcp_components(state):
    if state.get("external_wrench_stiffness") is not None:
        return _vector(state["external_wrench_stiffness"], 6)
    wrench = _vector(state["external_wrench"], 6)
    x, y, z, w = _vector(state["quaternion_xyzw"], 4)
    if abs(math.hypot(x, y, z, w) - 1) > 1e-3:
        raise ValueError("Contact observation orientation is not a unit quaternion")
    # R(O_T_EE)^T: K and EE coincide after the backend's setK(identity).
    rotation = ((1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)),
                (2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)),
                (2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)))
    return [sum(rotation[i][j]*wrench[start+j] for j in range(3))
            for start in (0, 3) for i in range(3)]


def _ratio(wrench, joint_torques, config):
    if config.get("_applied_thresholds") is not None:
        settings = config["_applied_thresholds"]
        upper = _vector(settings["upper_force_thresholds_nominal"], 6)
        joints = _vector(settings["upper_torque_thresholds_nominal"], 7)
        if min(upper+joints) <= 0:
            raise ValueError("Recorded collision thresholds must be positive")
        if (upper != settings["upper_force_thresholds_acceleration"] or
                joints != settings["upper_torque_thresholds_acceleration"]):
            raise ValueError("Different acceleration/nominal thresholds require phase-specific analysis")
        return max(*(abs(v)/limit for v, limit in zip(wrench, upper)),
                   *(abs(v)/limit for v, limit in zip(joint_torques, joints)))
    if config["compliance_mode"] != "contact_yielding":
        force, moment = max(map(abs, wrench[:3])), max(map(abs, wrench[3:]))
    else:
        force, moment = math.hypot(*wrench[:3]), math.hypot(*wrench[3:])
    return max(force/config["contact_force_stop"], moment/config["contact_moment_stop"],
               *(abs(t)/limit for t, limit in zip(joint_torques, config["external_torque_limits"])))


def analyze_log(path, candidate):
    """Return measured extrema and threshold fractions; never select new limits."""
    observed = []
    original = stop = None
    sdk_records_seen = set()
    post_observations_seen = set()
    def collect(state):
        observed.append(state)
        if state.get("stop_sample") is not None:
            observed.append(state["stop_sample"])
        fault = state.get("fault_diagnostics") or {}
        for index, record in enumerate(fault.get("sdk_records", [])):
            key = (fault.get("caught_at_monotonic_s"), index)
            if key not in sdk_records_seen:
                observed.append(record["state"])
                sdk_records_seen.add(key)
        post = fault.get("post_fault_observation") or {}
        key = post.get("sampled_at_monotonic_s")
        if post.get("fresh") and post.get("valid") and key not in post_observations_seen:
            observed.append(post["state"])
            post_observations_seen.add(key)
    with Path(path).open() as source:
        for line in source:
            event = json.loads(line)
            if event["event"] == "launch":
                original = event["controller_config"]
            elif event["event"] == "sample":
                collect(event["state"])
            elif event["event"] in ("stop", "controller_closed"):
                stop = event["result"]["state"] if event["event"] == "stop" else event["state"]
                collect(stop)
    if original is None or not observed:
        raise ValueError(f"{path}: expected a launch configuration and recorded observations")
    original = {"compliance_mode": "contact_yielding", **original}
    policy = original.get("collision_policy", "configured")
    recorded_known = policy == "configured"
    recorded_source = "recorded_controller_config" if recorded_known else "inherited_or_unconfirmed"
    if policy == "robot_configured" and stop:
        application = stop.get("robot_collision_settings") or {}
        if application.get("status") in ("applied", "simulated") and application.get("thresholds"):
            original["_applied_thresholds"] = application["thresholds"]
            recorded_known = True
            recorded_source = "sdk_acknowledged_request" if application["status"] == "applied" else "simulated_request"
    extrema = [0.0]*13
    old_ratios, new_ratios = [], []
    for state in observed:
        if state.get("sample_valid") is False:
            raise ValueError(f"{path}: invalid robot sample prevents a complete comparison")
        wrench = _tcp_components(state)
        torques = _vector(state["external_joint_torques"], 7)
        extrema = [max(a, abs(b)) for a, b in zip(extrema, wrench+torques)]
        if recorded_known:
            old_ratios.append(_ratio(wrench, torques, original))
        new_ratios.append(_ratio(wrench, torques, candidate))
    return {"log": str(path), "observations": len(observed),
            "peak_tcp_force_components_N": extrema[:3],
            "peak_tcp_moment_components_Nm": extrema[3:6],
            "peak_external_joint_torques_Nm": extrema[6:],
            "recorded_config_peak_threshold_fraction": max(old_ratios) if old_ratios else None,
            "recorded_threshold_source": recorded_source,
            "sdk_exception_record_observations": len(sdk_records_seen),
            "post_fault_observation_present": bool(post_observations_seen),
            "candidate_peak_threshold_fraction": max(new_ratios),
            "candidate_threshold_crossing_observations": sum(r >= 1 for r in new_ratios),
            "stop_trigger": stop.get("stop_trigger") if stop else None,
            "native_trigger_sample_present": bool(stop and stop.get("stop_sample") is not None),
            "limitation": "Recorded observations only; omitted robot-side collision peaks and changed dynamics are not reconstructed."}
