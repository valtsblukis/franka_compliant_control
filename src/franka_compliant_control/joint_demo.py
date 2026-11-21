"""Offline kinematic planning and an HTTP joint-setpoint demo; no implicit recovery."""
import math
import time
import threading

import numpy as np

from .diagnostics import stop_failure_message


def rotation_vector(rotation):
    cosine = float(np.clip((np.trace(rotation)-1)/2, -1, 1))
    angle = math.acos(cosine)
    vee = np.array([rotation[2,1]-rotation[1,2], rotation[0,2]-rotation[2,0],
                    rotation[1,0]-rotation[0,1]])/2
    if angle < 1e-7:
        return vee
    if math.pi-angle < 1e-5:
        matrix = (rotation+np.eye(3))/2
        axis = matrix[:, int(np.argmax(np.diag(matrix)))]
        axis /= np.linalg.norm(axis)
        if axis @ vee < 0:
            axis = -axis
        return angle*axis
    return angle*vee/math.sin(angle)


def quaternion_rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .001:
        raise ValueError("Expected a finite unit quaternion [x,y,z,w]")
    x,y,z,w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _pose_error(target, actual):
    return np.r_[target[:3,3]-actual[:3,3], .2*rotation_vector(target[:3,:3] @ actual[:3,:3].T)]


def solve_pose(model, seed, target, lower, upper):
    """Small seeded damped-IK steps; fail rather than change the requested pose."""
    q = np.array(seed, dtype=float)
    for _ in range(250):
        pose, jacobian = model.evaluate(q)
        error = _pose_error(target, pose)
        if np.linalg.norm(error[:3]) < 1e-5 and np.linalg.norm(error[3:]) < 2e-5:
            return q
        j = np.array(jacobian, copy=True);j[3:] *= .2
        step = j.T @ np.linalg.solve(j @ j.T+1e-6*np.eye(6), error)
        step *= min(1., .08/max(1e-12, float(np.max(np.abs(step)))))
        for scale in (1., .5, .25, .125, .0625, .03125):
            candidate = q+scale*step
            if np.any(candidate <= lower) or np.any(candidate >= upper):
                continue
            next_pose, _ = model.evaluate(candidate)
            if np.linalg.norm(_pose_error(target, next_pose)) < np.linalg.norm(error):
                q = candidate
                break
        else:
            raise ValueError("Cannot find a local IK solution inside the joint bounds; no setpoints sent")
    raise ValueError("IK did not converge; no setpoints sent")


def joint_segment(start, end, elapsed, duration):
    u = min(1., max(0., elapsed/duration))
    delta = np.asarray(end)-start
    h = u**3*(10+u*(-15+6*u))
    dh = 30*u*u*(1-u)**2/duration
    ddh = 60*u*(1-u)*(1-2*u)/duration**2
    jerk = 60*(1-6*u+6*u*u)/duration**3 if 0 <= elapsed <= duration else 0.
    return np.asarray(start)+delta*h, delta*dh, delta*ddh, delta*jerk


def _joint_velocity_bounds(q, config):
    ceilings = np.array([x or config.measured_joint_velocity_limit or 1.5*config.joint_velocity
                         for x in config.measured_joint_velocity_limits])
    lower, upper = -ceilings, ceilings
    if config.motion_limit_policy == "franka":
        physical_lo = np.array([-2.9007,-1.8361,-2.9007,-3.077,-2.8763,.4398,-3.0508])
        physical_hi = np.array([2.9007,1.8361,2.9007,-.1169,2.8763,4.6216,3.0508])
        decel = np.array([6,2.585,3.5,4,17,5.5,17])
        offset = np.array([.6599,.2517,.2,.3533,.5757,.4878,.4628])
        lower = -np.minimum(ceilings, np.maximum(0, np.sqrt(np.maximum(0,2*decel*(q-physical_lo)))-offset))
        upper = np.minimum(ceilings, np.maximum(0, np.sqrt(np.maximum(0,2*decel*(physical_hi-q)))-offset))
    return lower, upper


def plan_lower_target(model, home, config, *, drop_m=.30, move_time_s=2., switch_every_s=3.):
    """Kinematic plan only. No robot I/O, collision map or guarantee of tracking."""
    if config.robot_model != "fr3" or config.compliance_mode not in ("joint_impedance", "auto"):
        raise ValueError("Demo requires the FR3 direct joint_impedance profile")
    if any(not math.isfinite(x) or x <= 0 for x in (drop_m, move_time_s, switch_every_s)):
        raise ValueError("Drop and timing must be finite and positive")
    if move_time_s > switch_every_s:
        raise ValueError("Move time must fit within each target interval")
    qhome = np.asarray(home, dtype=float)
    lo = np.asarray(config.joint_lower)+config.joint_limit_margin
    hi = np.asarray(config.joint_upper)-config.joint_limit_margin
    if qhome.shape != (7,) or not np.isfinite(qhome).all() or np.any(qhome <= lo) or np.any(qhome >= hi):
        raise ValueError("Starting joints must be finite and strictly inside configured bounds")
    home_pose, _ = model.evaluate(qhome)
    if not np.isfinite(home_pose).all():
        raise ValueError("Invalid initial model pose")
    qlower = qhome.copy()
    # Continuation encourages a nearby IK branch while keeping orientation fixed.
    for drop in np.linspace(0, drop_m, min(1000, max(2, math.ceil(drop_m/.02)+1)))[1:]:
        target = home_pose.copy();target[2,3] -= drop
        qlower = solve_pose(model, qlower, target, lo, hi)
    lower_pose, _ = model.evaluate(qlower)
    target = home_pose.copy();target[2,3] -= drop_m
    if np.linalg.norm(_pose_error(target, lower_pose)) > 1e-4:
        raise ValueError("Lower target verification failed")

    positions = [];orientation_excursions = [];max_speed = max_angular_speed = 0.
    max_joint_speed = np.zeros(7)
    for start, end in ((qhome, qlower), (qlower, qhome)):
        # Joint path bounds and derivative extrema are checked densely in both
        # directions. This is NOT collision checking or a continuous proof.
        for elapsed in np.linspace(0, move_time_s, 401):
            q,dq,ddq,jerk = joint_segment(start, end, elapsed, move_time_s)
            lower_speed, upper_speed = _joint_velocity_bounds(q, config)
            pose,jacobian = model.evaluate(q)
            twist = jacobian @ dq
            if (np.any(q <= lo) or np.any(q >= hi) or np.any(dq < lower_speed-1e-10) or
                    np.any(dq > upper_speed+1e-10) or np.max(np.abs(ddq)) > config.joint_acceleration+1e-9 or
                    np.max(np.abs(jerk)) > config.joint_jerk+1e-9 or
                    np.linalg.norm(twist[:3]) > (config.measured_translation_velocity_limit or 1.5*config.translation_velocity) or
                    np.linalg.norm(twist[3:]) > (config.measured_rotation_velocity_limit or 1.5*config.rotation_velocity)):
                raise ValueError("Planned joint path exceeds a configured motion bound; increase --move-time/--switch-every or reduce --drop")
            positions.append(pose[:3,3]);orientation_excursions.append(np.linalg.norm(rotation_vector(pose[:3,:3] @ home_pose[:3,:3].T)))
            max_speed = max(max_speed, float(np.linalg.norm(twist[:3])))
            max_angular_speed = max(max_angular_speed, float(np.linalg.norm(twist[3:])))
            max_joint_speed = np.maximum(max_joint_speed, np.abs(dq))
    positions = np.array(positions)
    return {"home_joints": qhome.tolist(), "lower_joints": qlower.tolist(),
            "home_position": home_pose[:3,3].tolist(), "lower_position": lower_pose[:3,3].tolist(),
            "home_rotation": home_pose[:3,:3].tolist(), "drop_m": drop_m,
            "move_time_s": move_time_s, "switch_every_s": switch_every_s,
            "planned_peak_joint_speed_rad_s": max_joint_speed.tolist(),
            "planned_peak_tcp_speed_m_s": max_speed, "planned_peak_angular_speed_rad_s": max_angular_speed,
            "planned_min_tcp_height_m": float(positions[:,2].min()),
            "planned_max_lateral_excursion_m": float(np.linalg.norm(positions[:,:2]-home_pose[:2,3], axis=1).max()),
            "planned_max_orientation_excursion_deg": math.degrees(max(orientation_excursions)),
            "collision_checked": False}


def execute_joint_demo(client, plan, config, *, cycles=0, stop_event=None,
                       record=lambda e: None, report=lambda e: None):
    """Stream the two joint goals at 50 Hz, switching every interval regardless of reach."""
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 0:
        raise ValueError("cycles must be a nonnegative integer")
    event = stop_event if stop_event is not None else threading.Event()
    qhome = np.asarray(plan["home_joints"], dtype=float)
    qlower = np.asarray(plan["lower_joints"], dtype=float)
    interval, duration = plan["switch_every_s"], plan["move_time_s"]
    if (qhome.shape != (7,) or qlower.shape != (7,) or not np.isfinite(qhome).all() or
            not np.isfinite(qlower).all() or not math.isfinite(interval) or not math.isfinite(duration) or
            not 0 < duration <= interval):
        raise ValueError("Invalid joint demo plan")
    if event.is_set():
        return {"sent": 0, "reason": "cancelled_before_start"}
    before = client.refresh_state()["state"]
    if (before["running"] or before["phase"] != "disarmed" or before["read_only"] or
            (before.get("backend") == "robot" and
             (before.get("robot_mode") != "Idle" or before.get("robot_has_errors") is not False)) or
            not before["sample_valid"] or before["sample_age_s"] > .05 or
            np.max(np.abs(np.asarray(before["positions"])-qhome)) > .005 or
            max(abs(v) for v in before["velocities"]) > config.stop_velocity):
        raise RuntimeError("Robot must remain stationary at captured home before starting; no setpoints sent")
    if event.is_set():
        return {"sent": 0, "reason": "cancelled_before_start"}
    home, low = qhome, qlower
    start = next_tick = next_report = time.monotonic()
    index = 0;announced = False;sent = 0;started_control = False
    try:
        while not event.is_set():
            now = time.monotonic()
            scheduled = int((now-start)/interval)
            if scheduled > index+1:
                raise RuntimeError("Missed a switching interval; no catch-up replay")
            if scheduled != index:
                index, announced = scheduled, False
            if cycles and index >= cycles*2:
                break
            # Home for t=[0,3), low for [3,6), home [6,9), ...
            target = home if index%2 == 0 else low
            origin = home if index == 0 else (low if index%2 == 0 else home)
            elapsed = now-start-index*interval
            q,dq,_,_ = joint_segment(origin, target, elapsed, duration)
            if not announced:
                record({"event": "target", "interval": index, "name": "home" if index%2 == 0 else "lower",
                        "scheduled_elapsed_s": index*interval, "target_joints": target.tolist()})
                announced = True
            # Do not intentionally resume a stopped/faulted trial. The HTTP
            # first-setpoint API can race a normal external stop; detect an
            # unexpected restart response and immediately request stop below.
            state = client.state()
            if started_control and (not state["running"] or state["phase"] not in ("armed", "active")):
                raise RuntimeError(f"Controller stopped during demo: {state['phase']}/{state['reason']}; no restart")
            if started_control and (not state["sample_valid"] or state["sample_age_s"] > .05):
                raise RuntimeError("Demo telemetry is invalid or stale")
            result = client.set_target_joint_state(q, velocities=dq)
            if started_control and result.get("started_control"):
                raise RuntimeError("Unexpected control restart; aborting demo")
            started_control = True;sent += 1
            state = client.state()
            if not state["running"] or state["phase"] not in ("armed", "active"):
                record({"event": "controller_stopped", "state": state})
                raise RuntimeError(f"Controller stopped: {state['phase']}/{state['reason']}")
            if not state["sample_valid"] or state["sample_age_s"] > .05:
                raise RuntimeError("Demo telemetry is invalid or stale")
            entry = {"event": "sample", "elapsed_s": time.monotonic()-start, "interval": index,
                     "target_name": "home" if index%2 == 0 else "lower", "target_joints": target.tolist(),
                     "published_joints": q.tolist(), "published_velocities": dq.tolist(), "state": state}
            record(entry)
            if now >= next_report:
                report(entry);next_report = now+1
            next_tick = max(next_tick+.02, time.monotonic())
            event.wait(max(0, next_tick-time.monotonic()))
    finally:
        result = client.stop(wait_s=min(5., config.stop_timeout+.5))
        record({"event": "stop", "result": result})
        if not result["stopped"]:
            raise RuntimeError(stop_failure_message(result["state"]))
    summary = {"sent": sent, "stopped": True, "reason": "interrupted" if event.is_set() else "completed",
               "elapsed_s": time.monotonic()-start, "completed_intervals": index}
    record({"event": "summary", **summary})
    return summary
