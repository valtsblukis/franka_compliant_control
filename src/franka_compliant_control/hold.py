"""Stream a fixed current-pose target while allowing compliant disturbances."""
import math
import threading
import time

from .diagnostics import stop_failure_message


def initialize_hold(client):
    """Local scripts initialize directly; HTTP holds start with a pose setpoint.

    There is no HTTP arm request. HTTP callers must have the intended TCP
    already configured when capturing a hold pose, as with any absolute goal.
    """
    if hasattr(client, "arm"):
        return client.arm()["state"]  # Standalone native connection, not an HTTP method.
    initial = client.refresh_state()["state"]
    if (initial["running"] or initial["phase"] != "disarmed" or initial["read_only"] or
            not initial["sample_valid"] or initial["sample_age_s"] > .05):
        raise RuntimeError("Need a fresh inactive writable observation to capture an HTTP hold")
    client.send_pose(initial["position"], initial["quaternion_xyzw"])
    return initial


def hold_compliantly(client, *, duration_s=120.0, max_displacement_m=.05,
                     max_rotation_rad=math.radians(15), stop_event=None,
                     record=lambda event: None, report=lambda event: None,
                     expected_backend="robot"):
    """Arm once, retain the post-arm pose, stream at 50 Hz, and confirm stop.

    duration_s=0 runs until cancellation. Yielding is expected, but faults,
    stale telemetry and excursions beyond any supplied initial-pose bounds abort.
    None disables an excursion check; robot/native fault checks still apply.
    No gains/limits are changed, and no recovery or motion retry is performed.
    """
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration must be finite and nonnegative (0 means until stopped)")
    if any(v is not None and (not math.isfinite(v) or v <= 0)
           for v in (max_displacement_m, max_rotation_rad)):
        raise ValueError("excursion limits must be finite and positive")
    stop_event = stop_event if stop_event is not None else threading.Event()
    before = client.state()
    if (before["backend"] != expected_backend or before["read_only"]
            or before["running"] or before["phase"] != "disarmed"):
        raise RuntimeError(
            "Need a disarmed, writable controller for the selected backend; "
            f"got backend={before['backend']}, read_only={before['read_only']}, "
            f"running={before['running']}, phase={before['phase']}, "
            f"reason={before['reason']}, error={before['error']!r}; "
            "no recovery or stop of an existing session was attempted")
    config = client.get_config()
    record({"event": "before", "state": before, "controller_config": config,
            "duration_s": duration_s, "max_displacement_m": max_displacement_m,
            "max_rotation_rad": max_rotation_rad})
    if stop_event.is_set():
        return {"armed": False, "reason": "cancelled_before_arm", "samples": 0}

    samples = yielding_samples = applied = 0
    max_distance = max_angle = 0.0
    previous_yielding = None
    started = None
    finish_reason = "duration"
    try:
        # Stop in finally even if the startup response is lost after control started.
        initial = initialize_hold(client)
        started = time.monotonic()
        position = tuple(initial["position"])
        quaternion = tuple(initial["quaternion_xyzw"])
        record({"event": "armed", "state": initial,
                "target_position": position, "target_quaternion_xyzw": quaternion})
        report({"event": "armed", "state": initial, "target_position": position,
                "target_quaternion_xyzw": quaternion})
        next_tick = next_status = started
        while not stop_event.is_set():
            now = time.monotonic()
            if duration_s and now - started >= duration_s:
                break
            # Never replace the fixed target with a displaced/yielding sample.
            client.send_pose(position, quaternion)
            state = client.state()
            if state["phase"] not in ("armed", "active") or not state["running"]:
                record({"event": "controller_stopped", "state": state})
                raise RuntimeError(f"Controller stopped: {state['phase']}/{state['reason']}: {state['error']}")
            if (not state["sample_valid"] or not math.isfinite(state["sample_age_s"])
                    or state["sample_age_s"] > .05):
                raise RuntimeError("Hold telemetry is invalid or older than 50 ms")
            distance = math.dist(position, state["position"])
            dot = abs(sum(a*b for a, b in zip(quaternion, state["quaternion_xyzw"])))
            angle = 2*math.acos(min(1.0, dot))
            if not math.isfinite(distance) or not math.isfinite(dot):
                raise RuntimeError("Nonfinite measured pose")
            samples += 1
            yielding = bool(state["yielding"])
            yielding_samples += yielding
            applied = max(applied, state["applied_seq"])
            max_distance, max_angle = max(max_distance, distance), max(max_angle, angle)
            event = {"event": "sample", "elapsed_s": time.monotonic() - started,
                     "position_error_m": distance, "rotation_error_rad": angle, "state": state}
            record(event)
            if ((max_displacement_m is not None and distance > max_displacement_m) or
                    (max_rotation_rad is not None and angle > max_rotation_rad)):
                raise RuntimeError(f"Excursion limit reached: {distance*1000:.1f} mm / {math.degrees(angle):.1f} deg from starting target")
            if yielding != previous_yielding or time.monotonic() >= next_status:
                report(event)
                previous_yielding = yielding
                next_status = time.monotonic() + 1
            # Skip missed update slots; do not burst stale commands to catch up.
            next_tick += .02
            if next_tick < time.monotonic():
                next_tick = time.monotonic()
            stop_event.wait(max(0, next_tick - time.monotonic()))
        if stop_event.is_set():
            finish_reason = "interrupted"
    finally:
        # Ask for stop before any logging/printing that might fail or block.
        try:
            result = client.stop(wait_s=min(5.0, config["stop_timeout"] + .5))
        except Exception as error:
            raise RuntimeError("Could not confirm the software stop; use the physical stop if the robot is moving unexpectedly") from error
        record({"event": "stop", "result": result})
        if not result["stopped"]:
            raise RuntimeError(stop_failure_message(result["state"]))
    # An immediate operator cancellation can precede the first native tick.
    # The confirmed stop is still successful; a completed-duration hold must
    # have evidence that its target was actually applied.
    if samples and applied == 0 and finish_reason != "interrupted":
        raise RuntimeError("No hold command was observed in the native loop")
    summary = {"armed": True, "stopped": True, "reason": finish_reason,
               "elapsed_s": time.monotonic() - started, "samples": samples,
               "yielding_samples": yielding_samples, "applied_seq": applied,
               "max_position_error_m": max_distance, "max_rotation_error_rad": max_angle,
               "target_position": position, "target_quaternion_xyzw": quaternion}
    record({"event": "summary", **summary})
    return summary
