"""Human-readable fault and collision-configuration diagnostics; no robot I/O."""


def stop_failure_message(state):
    reason = state.get("reason", "unknown")
    error = state.get("error", "")
    names = {"robot_cartesian_reflex": "Robot Cartesian collision reflex",
             "robot_joint_reflex": "Robot joint collision reflex",
             "robot_reflex": "Robot reflex", "robot_control_error": "Robot control abort"}
    # Also explain old logs without interpreting a generic backend_error as a
    # network failure or pretending their cached mode is a post-fault read.
    if reason not in names:
        if '"cartesian_reflex"' in error:
            reason = "robot_cartesian_reflex"
        elif '"joint_reflex"' in error:
            reason = "robot_joint_reflex"
    if reason not in names:
        return (f"Stationary disarmed stop not confirmed: {state.get('phase')}/{reason}: {error}; "
                f"initial trigger={state.get('stop_trigger', 'unavailable')}; "
                f"robot mode={state.get('robot_mode', 'unavailable')}; no recovery attempted")

    fault = state.get("fault_diagnostics") or {}
    recovery = state.get("last_recovery")
    recovery_text = (f"Last recorded recovery: {recovery.get('status')}; {recovery.get('error', '')}."
                     if recovery else "No recovery attempted.")
    post = fault.get("post_fault_observation") or {}
    observed = post.get("state") or {}
    if post.get("fresh") and post.get("valid"):
        age = post.get("sample_age_s")
        age_text = f", age {age:.3f} s" if isinstance(age, (float, int)) else ""
        observation = (f"post-abort observation: {observed.get('robot_mode', 'unknown')} "
                       f"(new robot timestamp{age_text}); "
                       f"current errors={observed.get('current_errors_text', 'unknown')}")
    else:
        detail = ("diagnostic read pending" if post.get("attempted") and not post.get("completed")
                  else post.get("error") or "no post-abort observation was recorded")
        observation = (f"post-abort robot mode unconfirmed ({detail}); "
                       f"cached last mode={state.get('last_observed_robot_mode', state.get('robot_mode', 'unavailable'))}")
    return (f"{names[reason]}; fault remains latched. {observation}; "
            f"SDK exception records={fault.get('sdk_record_count', 0)}; "
            f"initial trigger={state.get('stop_trigger', reason)}. "
            f"Normal disarmed completion is not reported while the fault is latched; this is not evidence that the reflex stop was ignored. "
            f"{recovery_text} SDK detail: {error}")


def final_error_message(error, final_state):
    """Prefer completed post-fault diagnostics over an earlier wait-timeout message."""
    if final_state and final_state.get("phase") == "fault":
        fault = final_state.get("fault_diagnostics") or {}
        if (final_state.get("reason", "").startswith("robot_") or
                fault.get("exception_type") == "franka::ControlException"):
            return stop_failure_message(final_state)
    return str(error)


def collision_application_message(state):
    settings = state.get("robot_collision_settings") or {}
    if state.get("collision_policy") == "robot":
        return ("Collision policy inherits robot settings; no collision setter was issued for this arm. "
                f"Last request status from this connection: {settings.get('status', 'unknown')} (not a readback).")
    status = settings.get("status", "unknown")
    thresholds = settings.get("thresholds") or {}
    if status == "applied":
        return ("Robot collision settings acknowledged by SDK: TCP upper "
                f"{thresholds.get('upper_force_thresholds_nominal')} (N, Nm); joint upper "
                f"{thresholds.get('upper_torque_thresholds_nominal')} Nm; "
                "same acceleration/nominal settings. This is an acknowledged request, not a robot readback.")
    if status == "simulated":
        return "Collision settings recorded for simulation only; no robot settings applied."
    return f"Robot collision application status: {status}; no new successful application is established."
