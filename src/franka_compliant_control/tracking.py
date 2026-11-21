"""Foreground two-target Cartesian experiment; no robot connection on import."""
import math
import threading
import time

from .diagnostics import stop_failure_message
from .trajectory import segment_duration, segment_sample


def targets_about_home(position, offset_m=.2):
    if len(position) != 3 or not all(math.isfinite(v) for v in position):
        raise ValueError("home must be a finite base-frame position")
    if not math.isfinite(offset_m) or offset_m <= 0:
        raise ValueError("offset must be finite and positive")
    return [(position[0], position[1] - offset_m, position[2]),
            (position[0], position[1] + offset_m, position[2])]


def track_two_targets(client, *, offset_m=.2, cycles=2, leg_timeout_s=30.0,
                      settle_s=.2, position_tolerance_m=None,
                      rotation_tolerance_rad=None, stop_event=None,
                      record=lambda event: None, report=lambda event: None,
                      expected_backend="robot", switch_every_s=None):
    """Capture post-arm home, visit -Y,+Y for each cycle, then stop in place.

    Linear OSC uses native Ruckig. Bounded equilibrium modes use publisher-paced
    quintics with coherent derivatives. A switch interval advances on a fixed
    clock even while blocked; None retains wait-for-settling behavior. Reached
    events are tracked separately from completed target intervals. No recovery
    or catch-up replay. cycles=0 runs until stopped.
    """
    targets_about_home((0, 0, 0), offset_m)
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 0:
        raise ValueError("cycles must be a nonnegative integer")
    if any(not math.isfinite(v) or v <= 0 for v in
           (leg_timeout_s, settle_s, position_tolerance_m, rotation_tolerance_rad) if v is not None):
        raise ValueError("tracking timing and tolerances must be finite and positive")
    if switch_every_s is not None and (not math.isfinite(switch_every_s) or switch_every_s < .02):
        raise ValueError("target switching interval must be finite and at least 20 ms")
    if switch_every_s is None and leg_timeout_s <= settle_s:
        raise ValueError("leg timeout must exceed settling time")
    stop_event = stop_event if stop_event is not None else threading.Event()
    before = client.state()
    if (before["backend"] != expected_backend or before["read_only"] or
            before["running"] or before["phase"] != "disarmed"):
        raise RuntimeError("Need an exclusively owned, disarmed writable controller; no recovery attempted")
    config = client.get_config()
    if config["compliance_mode"] not in ("operational_space", "auto"):
        raise ValueError("two-target experiment requires operational_space")
    soft_core = config.get("osc_feedback_mode") == "progressive"
    bounded_equilibrium = config.get("osc_feedback_mode", "linear") != "linear"
    if position_tolerance_m is None:
        position_tolerance_m = 1.25*config["progressive_position_soft_radius"] if soft_core else .005
    if rotation_tolerance_rad is None:
        rotation_tolerance_rad = 1.25*config["progressive_rotation_soft_radius"] if soft_core else math.radians(2)
    if bounded_equilibrium:
        longest = segment_duration((0, -offset_m, 0), (0, offset_m, 0),
            config["translation_velocity"], config["translation_acceleration"], config["translation_jerk"])
        if switch_every_s is not None:
            if longest > switch_every_s:
                raise ValueError("switch interval must accommodate the paced trajectory; increase pacing or interval")
        elif longest+settle_s >= leg_timeout_s:
            raise ValueError("leg timeout must accommodate the paced trajectory and settling time")
    record({"event": "before", "state": before, "controller_config": config,
            "offset_m": offset_m, "cycles": cycles, "leg_timeout_s": leg_timeout_s,
            "settle_s": settle_s, "position_tolerance_m": position_tolerance_m,
            "rotation_tolerance_rad": rotation_tolerance_rad,
            "switch_every_s": switch_every_s,
            "completion_policy": "timed" if switch_every_s is not None else "settled"})
    if stop_event.is_set():
        return {"armed": False, "reason": "cancelled_before_arm", "completed_legs": 0}

    completed = samples = reached_count = 0
    try:
        initial = client.arm()["state"]
        started = time.monotonic()
        home, orientation = tuple(initial["position"]), tuple(initial["quaternion_xyzw"])
        targets = targets_about_home(home, offset_m)
        event = {"event": "armed", "state": initial, "home_position": home,
                 "target_positions": targets, "target_quaternion_xyzw": orientation}
        record(event)
        report(event)
        next_tick = next_report = started
        leg_started = started
        segment_origin = home
        motion_duration = None
        settled_since = None
        target_announced = leg_reached = False
        while not stop_event.is_set() and (cycles == 0 or completed < 2*cycles):
            now = time.monotonic()
            if switch_every_s is not None and now >= leg_started+switch_every_s:
                if now >= leg_started+2*switch_every_s:
                    raise RuntimeError("Missed a target switching interval; no catch-up replay")
                record({"event": "target_interval_complete", "leg": completed,
                        "elapsed_s": now-started, "reached_during_interval": leg_reached})
                segment_origin = targets[completed % 2]
                completed += 1
                if cycles and completed >= 2*cycles:
                    break
                leg_started = started+completed*switch_every_s
                settled_since = None
                target_announced = leg_reached = False
            target = targets[completed % 2]
            if bounded_equilibrium:
                if not target_announced:
                    motion_duration = segment_duration(segment_origin, target,
                        config["translation_velocity"], config["translation_acceleration"], config["translation_jerk"])
                position, twist, acceleration = segment_sample(segment_origin, target,
                    time.monotonic()-leg_started, motion_duration)
                accepted = client.send_pose(position, orientation, twist=twist, acceleration=acceleration)
            else:
                accepted = client.send_pose(target, orientation)
            if not target_announced:
                first_sequence = accepted.get("command_seq", accepted["seq"])
                record({"event": "target", "leg": completed, "position": target,
                        "paced_duration_s": motion_duration,
                        "scheduled_elapsed_s": leg_started-started,
                        "elapsed_s": time.monotonic()-started})
                target_announced = True
            state = client.state()
            if state["phase"] not in ("armed", "active") or not state["running"]:
                record({"event": "controller_stopped", "state": state})
                raise RuntimeError(f"Controller stopped: {state['phase']}/{state['reason']}: {state['error']}")
            if (not state["sample_valid"] or not math.isfinite(state["sample_age_s"]) or
                    state["sample_age_s"] > .05):
                raise RuntimeError("Tracking telemetry is invalid or older than 50 ms")
            distance = math.dist(target, state["position"])
            dot = abs(sum(a*b for a, b in zip(orientation, state["quaternion_xyzw"])))
            if not math.isfinite(distance) or not math.isfinite(dot):
                raise RuntimeError("Nonfinite measured pose")
            angle = 2*math.acos(min(1.0, dot))
            now = time.monotonic()
            samples += 1
            event = {"event": "sample", "elapsed_s": now-started, "leg": completed,
                     "target_position": target, "position_error_m": distance,
                     "rotation_error_rad": angle, "state": state}
            record(event)
            if now >= next_report:
                report(event)
                next_report = now+1
            # At least one command for this leg must have been applied. Comparing
            # to the newest submitted seq would race the independent 1 kHz loop.
            reference_settled = (math.dist(target, state["reference_position"]) < 1e-5 and
                                 math.hypot(*state["reference_twist"]) < 1e-4)
            reached = (state["phase"] == "active" and state["applied_seq"] >= first_sequence and
                       reference_settled and
                       distance < position_tolerance_m and angle < rotation_tolerance_rad and
                       max(abs(v) for v in state["velocities"]) < config["stop_velocity"])
            if reached and not leg_reached:
                if settled_since is None:
                    settled_since = now
                if now-settled_since >= settle_s:
                    record({"event": "reached", "leg": completed, "position": target,
                            "elapsed_s": now-started, "state": state})
                    reached_count += 1
                    leg_reached = True
                    if switch_every_s is None:
                        completed += 1
                        segment_origin = target
                        leg_started, settled_since = now, None
                        target_announced = leg_reached = False
            elif not reached:
                settled_since = None
            if switch_every_s is None and now-leg_started > leg_timeout_s:
                raise RuntimeError(f"Target {completed} did not settle within {leg_timeout_s:g} s; no retry")
            next_tick = max(next_tick+.02, time.monotonic())
            stop_event.wait(max(0, next_tick-time.monotonic()))
    finally:
        # Stop before logging, even on lost arm acknowledgement or output failure.
        try:
            result = client.stop(wait_s=min(5.0, config["stop_timeout"]+.5))
        except Exception as error:
            raise RuntimeError("Software stop unconfirmed; use the physical stop for unexpected motion") from error
        record({"event": "stop", "result": result})
        if not result["stopped"]:
            raise RuntimeError(stop_failure_message(result["state"]))
    summary = {"armed": True, "stopped": True, "completed_legs": completed,
               "reached_legs": reached_count, "switch_every_s": switch_every_s, "samples": samples,
               "reason": "interrupted" if stop_event.is_set() else "completed",
               "home_position": home, "target_positions": targets,
               "elapsed_s": time.monotonic()-started}
    record({"event": "summary", **summary})
    return summary
