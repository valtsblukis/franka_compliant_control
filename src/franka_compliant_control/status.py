"""Read-only diagnostic view of a snapshot; never a safety/arming authority."""
import math


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def diagnostic_status(state, config, last_operation=None):
    """Display limits and observed blockers without claiming readiness to arm."""
    age = state.get("sample_age_s")
    fresh = finite(age) and 0 <= age <= .05 and bool(state.get("sample_valid"))
    joints = []
    q, dq = state.get("positions", []), state.get("velocities", [])
    for i, (lo, hi) in enumerate(zip(config.joint_lower, config.joint_upper)):
        lower, upper = lo+config.joint_limit_margin, hi-config.joint_limit_margin
        value = q[i] if i < len(q) and finite(q[i]) else None
        velocity = dq[i] if i < len(dq) and finite(dq[i]) else None
        dl, du = (value-lower, upper-value) if value is not None else (None, None)
        nearest = min(dl, du) if value is not None else None
        trip = config.measured_joint_velocity_limits[i] or config.measured_joint_velocity_limit or 1.5*config.joint_velocity
        low_speed, high_speed = -trip, trip
        if value is not None and config.motion_limit_policy == "franka":
            # Diagnostic mirror of native/fr3_limits.hpp, NOT a control guard.
            physical_lo = [-2.9007,-1.8361,-2.9007,-3.077,-2.8763,.4398,-3.0508][i]
            physical_hi = [2.9007,1.8361,2.9007,-.1169,2.8763,4.6216,3.0508][i]
            offset = [.6599,.2517,.2,.3533,.5757,.4878,.4628][i]
            decel = [6,2.585,3.5,4,17,5.5,17][i]
            ceiling = [2.62,2.62,2.62,2.62,5.26,4.18,5.26][i]
            low_speed = -min(trip, ceiling, max(0, math.sqrt(max(0,2*decel*(value-physical_lo)))-offset))
            high_speed = min(trip, ceiling, max(0, math.sqrt(max(0,2*decel*(physical_hi-value)))-offset))
        references = state.get("reference_joints") or []
        reference = (references[i] if state.get("mode") == "joint" and state.get("phase") == "active"
                     and i < len(references) and finite(references[i]) else None)
        joints.append({"joint": i+1, "position_rad": value, "velocity_rad_s": velocity,
            "reference_position_rad": reference,
            "tracking_error_rad": reference-value if reference is not None and value is not None else None,
            "lower_rad": lower, "upper_rad": upper, "lower_distance_rad": dl, "upper_distance_rad": du,
            "nearest_distance_deg": math.degrees(nearest) if nearest is not None else None,
            "nearest_boundary": ("lower" if dl <= du else "upper") if value is not None else None,
            "fraction": (value-lower)/(upper-lower) if value is not None else None,
            "position_status": ("unknown" if nearest is None else "outside" if nearest <= 0
                                else "near" if nearest < math.radians(5) else "inside"),
            "lower_velocity_rad_s": low_speed if value is not None else None,
            "upper_velocity_rad_s": high_speed if value is not None else None})
    notices = []
    if not fresh:
        notices.append("Snapshot is stale or invalid. Inactive GET polling does not read the robot. Request an inactive state read.")
    if state.get("read_only"):
        notices.append("Read-only connection: arming and recovery are disabled.")
    if state.get("phase") == "fault":
        notices.append("Controller fault is latched; reading state does not recover it. " +
                       ("A new valid setpoint can attempt guarded automatic recovery once. Stop the producer to prevent restart. "
                        if state.get("automatic_recovery_enabled") else "Automatic recovery is disabled. ") +
                       "Manual Recover fault / POST /recover is also available while inactive.")
    if state.get("backend") == "robot" and (state.get("robot_mode") != "Idle" or state.get("robot_has_errors") is not False):
        notices.append("Last observed robot state is not error-free Idle. Check Desk; no automatic recovery is performed.")
    for joint in joints:
        if joint["position_status"] in ("near", "outside"):
            notices.append(f"Joint {joint['joint']}: {joint['nearest_distance_deg']:.2f} deg to {joint['nearest_boundary']} boundary ({joint['position_status']}).")
    if any(finite(v) and abs(v) > config.stop_velocity for v in dq):
        notices.append("Observed joint velocity exceeds the stationary-before-arm threshold.")
    wrench = state.get("external_wrench")
    if wrench and len(wrench) == 6 and all(finite(v) for v in wrench):
        if math.hypot(*wrench[:3]) >= config.contact_force_soft or math.hypot(*wrench[3:]) >= config.contact_moment_soft:
            notices.append("Observed wrench exceeds an unloaded-startup soft threshold; arm preflight can reject this state.")
    nearest = min((j for j in joints if j["nearest_distance_deg"] is not None),
                  key=lambda j: j["nearest_distance_deg"], default=None)
    fault = state.get("fault_diagnostics") or {}
    post = fault.get("post_fault_observation") or {}
    snapshot = {k: v for k, v in state.items() if k != "fault_diagnostics"}
    return {"state": snapshot, "joints": joints, "sample_fresh": fresh,
            "near_limit_warning_deg": 5, "nearest_joint": nearest, "notices": notices,
            "readiness_note": "Diagnostic observations only, not permission to arm. Native preflight is authoritative.",
            "last_operation": last_operation, "fault_summary": {
                "exception_type": fault.get("exception_type"), "message": fault.get("message"),
                "sdk_record_count": fault.get("sdk_record_count", 0),
                "post_fault_observation": {k: v for k, v in post.items() if k != "state"},
                "post_fault_current_errors": (post.get("state") or {}).get("current_errors_text"),
                "post_fault_last_motion_errors": (post.get("state") or {}).get("last_motion_errors_text")}}


def terminal_status(view):
    state = view["state"]
    age = state.get("sample_age_s")
    age_text = f"{age:.3f}s" if finite(age) else "unknown"
    nearest = view.get("nearest_joint")
    joint_text = (f"J{nearest['joint']} {nearest['nearest_boundary']} margin={nearest['nearest_distance_deg']:.2f}deg"
                  if nearest else "joint margins unavailable")
    operation = view.get("last_rejection") or view.get("last_operation") or {}
    rejection = f" | historical {operation['operation']} rejection: {operation['error']}" if operation.get("error") else ""
    return (f"{state.get('phase', '?').upper()} running={state.get('running')} "
            f"robot={state.get('robot_mode', '?')} reason={state.get('reason', '?')} "
            f"sample={'FRESH' if view['sample_fresh'] else 'STALE'} age={age_text} | {joint_text} "
            f"| seq accepted/applied={state.get('accepted_seq')}/{state.get('applied_seq')}{rejection}")
