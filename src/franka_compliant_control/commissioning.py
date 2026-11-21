"""Bounded commissioning hold. Calling hold_current_pose starts torque control."""
import math
import time

from .hold import initialize_hold


def hold_current_pose(client, record, *, expected_backend="robot"):
    before = client.state()
    record({"event": "before", "state": before})
    if (before["backend"] != expected_backend or before["phase"] != "disarmed"
            or before["running"] or before.get("read_only", False)):
        raise RuntimeError("Expected an available, disarmed service for the selected backend")
    samples = []
    initial = None
    try:
        initial = initialize_hold(client)
        # HTTP starts on its first captured-pose setpoint; local initialization
        # remains inside the owning foreground process.
        started = time.monotonic()
        record({"event": "armed", "state": initial})
        for i in range(100):
            if time.monotonic() - started >= 2.0:
                break
            client.send_pose(initial["position"], initial["quaternion_xyzw"])
            state = client.state()
            samples.append(state)
            record({"event": "sample", "state": state})
            if state["phase"] not in ("armed", "active") or not state["running"]:
                raise RuntimeError(f"Hold stopped: {state['reason']}: {state['error']}")
            if not state["sample_valid"] or state["sample_age_s"] > .05:
                raise RuntimeError("Hold telemetry is invalid or stale")
            drift = math.dist(state["position"], initial["position"])
            dot = abs(sum(a*b for a, b in zip(state["quaternion_xyzw"], initial["quaternion_xyzw"])))
            angle = 2*math.acos(min(1.0, dot))
            if drift > .005 or angle > .035:
                raise RuntimeError(f"Hold drift exceeded 5 mm / 2 degrees: {drift:.6f} m, {angle:.6f} rad")
            if state["yielding"]:
                raise RuntimeError("Controller yielded to estimated contact; unloaded hold not established")
            time.sleep(max(0, started + (i+1)*.02 - time.monotonic()))
    finally:
        # Also request stop if the startup request timed out after control began.
        result = client.stop(wait_s=3)
        record({"event": "stop", "result": result})
        if not result["stopped"]:
            raise RuntimeError(f"Stationary disarmed stop not confirmed: {result['state']}")
    if not samples or not any(s["applied_seq"] > 0 for s in samples):
        raise RuntimeError("No hold command was observed in the native loop")
    return {
        "samples": len(samples), "stopped": True,
        "max_position_drift_m": max(math.dist(s["position"], initial["position"]) for s in samples),
        "max_callback_s_observed": max(s["controller_compute_s"] for s in samples),
        "min_communication_success_observed": min(s["communication_success"] for s in samples if s["phase"] == "active"),
        "yielding_samples": sum(s["yielding"] for s in samples),
    }
