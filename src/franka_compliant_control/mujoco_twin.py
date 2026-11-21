"""Offline MuJoCo experiment using the unchanged native Controller implementation."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from . import _native
from .models import ControllerConfig
from .twin_experiment import ManualPosePolicy, capture_home, digest, load_home, write_json_new
from .twin_model import MujocoPlant, PlantParameters, combined_load, rotation_distance, matrix_quaternion

ROOT = Path(__file__).resolve().parents[2]
# Backward-compatible imports for frozen experiment recipes. Implementations now
# live below the experiment layer so reusable simulation has no CLI dependency.
from .simulation.sdk import TWIN_INPUTS, native_inputs_digest, offline_model, offline_kinematics


def model_crosscheck(home, *, count=32):
    """Independent MJ/SDK FK, J, M, c, g checks with matched ideal rigid bodies."""
    if type(count) is not int or count < 1:
        raise ValueError("Model crosscheck requires at least one posture")
    plant = MujocoPlant(home, PlantParameters(viscous=(0.,)*7, coulomb=(0.,)*7))
    sdk = offline_model(home)
    config = ControllerConfig.model_validate(home["controller_config"])
    rng = np.random.default_rng(7123)
    errors = {k:0. for k in ("position_m", "rotation_rad", "jacobian_max_abs",
                             "mass_max_abs", "coriolis_Nm", "gravity_Nm")}
    for index in range(count):
        q = np.array(home["home"]["joints"])
        if index:
            q += rng.uniform(-.4,.4,7)
            q = np.clip(q,np.array(config.joint_lower)+.1,np.array(config.joint_upper)-.1)
        dq = rng.uniform(-.5,.5,7)
        actual = plant.model_quantities(q,dq)
        sample = sdk.sample(q,dq,np.zeros(7),np.zeros(6))
        values = {
            "position_m": np.linalg.norm(actual["position"]-sample.position),
            "rotation_rad": rotation_distance(actual["quaternion_xyzw"],sample.quaternion_xyzw),
            "jacobian_max_abs": np.max(np.abs(actual["jacobian"]-sample.jacobian)),
            "mass_max_abs": np.max(np.abs(actual["mass"]-sample.mass)),
            "coriolis_Nm": np.max(np.abs(actual["coriolis"]-sample.coriolis)),
            "gravity_Nm": np.max(np.abs(actual["gravity"]-sdk.gravity(q))),
        }
        for key,value in values.items():
            errors[key] = max(errors[key],float(value))
    passed = all(value < 1e-6 for value in errors.values())
    return {"passed": passed, "postures": count, "max_errors": errors,
            "meaning": "implementation/model conversion consistency, NOT comparison with physical motion"}


def preflight_path(home, policy, *, sample_guard=None):
    """Continuous IK samples of the reference path; not obstacle collision checking."""
    config = ControllerConfig.model_validate(home["controller_config"])
    model = offline_kinematics(home)
    from .twin_model import quaternion_matrix
    q = np.array(policy.start_joints or home["home"]["joints"],dtype=float)
    minimum_sigma, minimum_margin, maximum_error = math.inf, math.inf, 0.
    lower, upper = np.array(config.joint_lower)+.02, np.array(config.joint_upper)-.02
    for t in np.linspace(0,policy.duration_s,201):
        target = policy.sample(float(t))
        rotation = quaternion_matrix(target["quaternion_xyzw"])
        for _ in range(150):
            pose,jac = model.evaluate(q)
            # Local base-frame orientation error, sufficient for fixed-orientation
            # small IK continuation steps; explicitly reject failed convergence.
            delta = rotation@pose[:3,:3].T
            angular = .5*np.array([delta[2,1]-delta[1,2],delta[0,2]-delta[2,0],delta[1,0]-delta[0,1]])
            error = np.r_[np.array(target["position"])-pose[:3,3],angular]
            if np.linalg.norm(error) < 1e-7:
                break
            increment = jac.T@np.linalg.solve(jac@jac.T+1e-6*np.eye(6),error)
            norm = np.max(np.abs(increment))
            q = np.clip(q+increment*min(1.,.05/max(norm,1e-12)),lower,upper)
        pose,jac = model.evaluate(q)
        position_error = np.linalg.norm(pose[:3,3]-target["position"])
        orientation_error = np.linalg.norm(pose[:3,:3]-rotation)
        if position_error > 1e-4 or orientation_error > 1e-3:
            raise RuntimeError(f"Reference IK failed at t={t:.3f}; no hardware execution permitted")
        maximum_error = max(maximum_error,float(position_error))
        scaled = jac.copy();scaled[3:]*=config.jacobian_rotation_scale
        sigma = float(np.linalg.svd(scaled,compute_uv=False)[-1])
        minimum_sigma = min(minimum_sigma,sigma)
        if sample_guard is not None:
            sample_guard(q,pose[:3,3],matrix_quaternion(pose[:3,:3]),sigma)
        minimum_margin = min(minimum_margin,float(np.min(np.minimum(q-config.joint_lower,config.joint_upper-q))))
    return {"reference_ik_passed": True, "sampled_configurations": 201,
            "minimum_scaled_jacobian_singular_value": minimum_sigma,
            "singularity_threshold": config.minimum_singular_value,
            "away_from_singularities": minimum_sigma >= config.minimum_singular_value,
            "minimum_joint_position_margin_rad": minimum_margin,
            "maximum_ik_position_error_m": maximum_error,
            "environment_collision_check": False,
            "note": "IK continuation is a feasibility screen; actual closed-loop joint path checked separately"}


def make_command(target, seq):
    cmd = _native.Command()
    cmd.epoch,cmd.sequence = 1,seq
    cmd.mode = _native.Mode.Cartesian
    cmd.position = target["position"]
    cmd.has_orientation = True
    cmd.quaternion_xyzw = target["quaternion_xyzw"]
    # Deliberately leave twist and acceleration zero: identical pose-only API.
    return cmd


def simulation_trace(home, policy, parameters, *, mesh_dir=None, command_lifetime_s=None,
                     command_events=None, initial_state=None, observer_config=None):
    """Yield synchronized native samples. No network, Python servo thread or sleep."""
    from . import _twin_native
    config = ControllerConfig.model_validate(home["controller_config"])
    sdk = offline_model(home)
    plant = MujocoPlant(home,parameters,mesh_dir=mesh_dir,targets=policy.targets)
    if policy.start_joints is not None:
        import mujoco
        plant.data.qpos[:] = policy.start_joints
        mujoco.mj_forward(plant.model,plant.data)
    previous = np.zeros(7)
    if initial_state is not None:
        import mujoco
        plant.data.qpos[:] = initial_state['positions']
        # Trials begin stationary; do not turn SDK velocity sensor noise into
        # true initial plant motion. Preserve the observed torque-filter state.
        plant.data.qvel[:] = 0
        previous = np.asarray(initial_state['last_desired_joint_torques'],dtype=float)
        mujoco.mj_forward(plant.model,plant.data)
    observer = observer_data = None
    if observer_config is not None:
        import mujoco
        from .wrench_model import WrenchObserver
        observer = WrenchObserver(observer_config)
        observer_data = mujoco.MjData(plant.model)  # Never perturb the plant solver's state.
    actuated_previous = previous.copy()
    sample = sdk.sample(plant.data.qpos,plant.data.qvel,previous,np.zeros(6))
    controller = _twin_native.OfflineController(config.native(),sample,1,0.)
    delayed = deque([np.zeros(7) for _ in range(parameters.actuator_delay_steps)])
    target = policy.sample(0)
    command = make_command(target,1)
    lifetime = config.command_timeout if command_lifetime_s is None else command_lifetime_s
    if not np.isfinite(lifetime) or not 0 < lifetime <= config.command_timeout:
        raise ValueError("Invalid simulated command lifetime")
    deadline = lifetime
    event_index = -1
    if command_events is not None:
        times = np.array([e['time_s'] for e in command_events])
        if not len(times) or not np.isfinite(times).all() or times[0]>0 or np.any(np.diff(times)<=0):
            raise ValueError('Replay needs increasing accepted-command times, including an initial command at/before zero')
    for tick in range(round(policy.duration_s*1000)+1):
        now = tick*.001
        if command_events is not None:
            target = policy.sample(now)
            while event_index+1 < len(command_events) and command_events[event_index+1]['time_s']<=now+1e-12:
                event_index += 1
                event = command_events[event_index]
                command = make_command(event,event_index+1)
                deadline = event['deadline_s']
        elif tick%20 == 0:
            target = policy.sample(now)
            command = make_command(target,tick//20+1)
            deadline = now+lifetime
        q,dq = plant.data.qpos.copy(),plant.data.qvel.copy()
        sample = sdk.sample(q,dq,previous,np.zeros(6))
        observation = None
        if observer is not None:
            observer_data.qpos[:] = q;observer_data.qvel[:] = dq
            observer_data.qfrc_applied[:] = actuated_previous+sdk.gravity(q)-parameters.joint_bias
            mujoco.mj_forward(plant.model,observer_data)
            residual = previous-sample.mass@observer_data.qacc-sample.coriolis
            observation = observer.step(q,residual,sample.jacobian,sample.position,.001)
            sample.external_wrench = observation['external_wrench']  # Raw deployed SDK convention.
            sample.external_torque = observation['external_joint_torques']
            from .twin_model import quaternion_matrix
            rotation = quaternion_matrix(sample.quaternion_xyzw)
            canonical = observation['external_wrench_tcp_base']
            sample.external_wrench_tcp_base = canonical
            sample.wrench_tcp_valid = True
            sample.external_wrench_stiffness = np.r_[rotation.T@canonical[:3],rotation.T@canonical[3:]]
        out = controller.step(sample,command,now,.001,deadline)
        filtered = _twin_native.sdk_torque_filter(out.torque,previous)
        delayed.append(filtered.copy())
        applied = delayed.popleft()
        gravity = sdk.gravity(q)
        velocity = sample.jacobian@dq
        yield {"time_s": now, "policy": target, "position": sample.position.tolist(),
               "quaternion_xyzw": sample.quaternion_xyzw.tolist(), "positions": q.tolist(),
               "velocities": dq.tolist(), "twist": velocity.tolist(),
               "reference_position": out.reference_position.tolist(),
               "external_wrench": observation['external_wrench'].tolist() if observation else [0.]*6,
               "external_wrench_tcp_base": observation['external_wrench_tcp_base'].tolist() if observation else None,
               "external_joint_torques": observation['external_joint_torques'].tolist() if observation else [0.]*7,
               "external_wrench_stiffness": sample.external_wrench_stiffness.tolist(),
               "wrench_source": observation['source'] if observation else "not_modeled_free_space_placeholder",
               "external_wrench_valid": observation is not None,
               "true_contact_force_calibrated": False,
               "commanded_task_wrench": out.task_wrench.tolist(),
               "commanded_joint_torques": out.torque.tolist(),
               "last_desired_joint_torques": previous.tolist(),
               "sdk_filtered_joint_torques": filtered.tolist(),
               "actuated_joint_torques_excluding_gravity": applied.tolist(),
               "gravity_compensation": gravity.tolist(),
               "applied_seq": out.sequence, "phase": out.phase, "reason": out.reason,
               "task_force_limited": out.task_force_limited,
               "task_moment_limited": out.task_moment_limited,
               "joint_torque_limited": out.joint_torque_limited,
               "osc_regularized": out.osc_regularized,
               "minimum_singular_value": out.singular_value,
               "sample_age_s": 0., "robot_time_s": now}
        if out.phase != "active":
            return
        if tick == round(policy.duration_s*1000):
            return
        # Franka internally gravity compensates. Do not add the entire bias
        # (which would double-count Coriolis already supplied by the controller).
        plant.step(applied+gravity)
        previous = filtered
        actuated_previous = applied


def summarize(rows, home, policy):
    if not rows:
        raise ValueError("No trajectory samples")
    config = ControllerConfig.model_validate(home["controller_config"])
    t = np.array([r["time_s"] for r in rows])
    p = np.array([r["position"] for r in rows])
    q = np.array([r["positions"] for r in rows])
    dq = np.array([r["velocities"] for r in rows])
    v = np.array([r["twist"] for r in rows])
    reference = np.array([r["policy"]["position"] for r in rows])
    errors = np.linalg.norm(p-reference,axis=1)
    angular = np.array([rotation_distance(r["quaternion_xyzw"],policy.home_quaternion_xyzw) for r in rows])
    dt = np.diff(t)
    acceleration = np.diff(v,axis=0)/dt[:,None] if len(dt) else np.zeros((1,6))
    joint_acceleration = np.diff(dq,axis=0)/dt[:,None] if len(dt) else np.zeros((1,7))
    jerk = np.diff(acceleration,axis=0)/dt[1:,None] if len(dt)>1 else np.zeros((1,6))
    joint_jerk = np.diff(joint_acceleration,axis=0)/dt[1:,None] if len(dt)>1 else np.zeros((1,7))
    legs = []
    for leg in range(policy.cycles*4):
        selected = [i for i,r in enumerate(rows) if r["policy"]["leg"] == leg]
        if not selected:
            continue
        last = selected[-1]
        goal = rows[last]["policy"]["goal_position"]
        # Terminal 0.2 s must be pose-settled and stationary, not one crossing.
        tail = [i for i in selected if t[i] >= t[last]-.2]
        settled = all(np.linalg.norm(p[i]-goal)<.005 and angular[i]<math.radians(2)
                      and np.max(np.abs(dq[i]))<config.stop_velocity for i in tail)
        legs.append({"leg":leg,"goal":rows[last]["policy"]["goal"],
                     "final_position_error_m":float(np.linalg.norm(p[last]-goal)),
                     "final_orientation_error_rad":float(angular[last]),
                     "settled_last_0_2s":bool(settled)})
    margins = np.minimum(q-np.array(config.joint_lower),np.array(config.joint_upper)-q)
    complete = abs(t[-1]-policy.duration_s)<.002 and all(r["phase"]=="active" for r in rows)
    sigma = min(r["minimum_singular_value"] for r in rows)
    result = {
        "completed":bool(complete), "duration_s":float(t[-1]), "samples":len(rows),
        "policy_rate_hz":50, "native_rate_hz":1000,
        "position_tracking_rmse_m":float(np.sqrt(np.mean(errors**2))),
        "maximum_position_tracking_error_m":float(max(errors)),
        "maximum_orientation_error_rad":float(max(angular)),
        "peak_tcp_speed_m_s":float(np.linalg.norm(v[:,:3],axis=1).max()),
        "peak_tcp_angular_speed_rad_s":float(np.linalg.norm(v[:,3:],axis=1).max()),
        "peak_tcp_acceleration_m_s2":float(np.linalg.norm(acceleration[:,:3],axis=1).max()),
        "peak_tcp_angular_acceleration_rad_s2":float(np.linalg.norm(acceleration[:,3:],axis=1).max()),
        "peak_tcp_jerk_m_s3":float(np.linalg.norm(jerk[:,:3],axis=1).max()),
        "peak_tcp_angular_jerk_rad_s3":float(np.linalg.norm(jerk[:,3:],axis=1).max()),
        "peak_joint_acceleration_rad_s2":np.max(np.abs(joint_acceleration),axis=0).tolist(),
        "peak_joint_jerk_rad_s3":np.max(np.abs(joint_jerk),axis=0).tolist(),
        "peak_joint_speed_rad_s":np.max(np.abs(dq),axis=0).tolist(),
        "peak_commanded_force_N":float(max(np.linalg.norm(r["commanded_task_wrench"][:3]) for r in rows)),
        "peak_commanded_moment_Nm":float(max(np.linalg.norm(r["commanded_task_wrench"][3:]) for r in rows)),
        "minimum_scaled_jacobian_singular_value":float(sigma),
        "away_from_singularities":bool(sigma>=config.minimum_singular_value),
        "minimum_joint_position_margin_rad":float(margins.min()),
        "saturation_fraction":{key:float(np.mean([r[key] for r in rows])) for key in
                               ("task_force_limited","task_moment_limited","joint_torque_limited","osc_regularized")},
        "final_home_position_error_m":float(np.linalg.norm(p[-1]-policy.home_position)),
        "final_max_joint_velocity_rad_s":float(np.max(np.abs(dq[-1]))),
        "legs":legs,
        "all_legs_settled":len(legs)==4*policy.cycles and all(leg["settled_last_0_2s"] for leg in legs),
        "hardware_stop_simulated":False,
        "measured_contact_comparison_available":False,
        "last_phase":rows[-1]["phase"], "last_reason":rows[-1]["reason"],
    }
    result["sampled_acceleration_jerk_within_config"] = bool(
        result["peak_tcp_acceleration_m_s2"] <= config.translation_acceleration
        and result["peak_tcp_angular_acceleration_rad_s2"] <= config.rotation_acceleration
        and result["peak_tcp_jerk_m_s3"] <= config.translation_jerk
        and result["peak_tcp_angular_jerk_rad_s3"] <= config.rotation_jerk
        and max(result["peak_joint_acceleration_rad_s2"]) <= config.joint_acceleration
        and max(result["peak_joint_jerk_rad_s3"]) <= config.joint_jerk)
    result["ready_for_operator_review"] = (result["completed"] and result["all_legs_settled"]
                                           and result["away_from_singularities"]
                                           and result["sampled_acceleration_jerk_within_config"])
    return result


def plot_trace(rows, home, policy, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["time_s"] for r in rows])
    p = np.array([r["position"] for r in rows])
    reference = np.array([r["policy"]["position"] for r in rows])
    v = np.array([r["twist"] for r in rows])
    wrench = np.array([r["commanded_task_wrench"] for r in rows])
    fig,axes = plt.subplots(5,1,figsize=(11,12),sharex=True)
    axes[0].plot(t,reference[:,1],"--",label="50 Hz pose reference")
    axes[0].plot(t,p[:,1],label="MuJoCo + native controller")
    axes[0].axhline(policy.home_position[1],color="gray",lw=.7,label="fixed captured home")
    axes[0].set_ylabel("Base Y [m]");axes[0].legend()
    axes[1].plot(t,1000*np.linalg.norm(p-reference,axis=1));axes[1].set_ylabel("Tracking error [mm]")
    axes[2].plot(t,np.linalg.norm(v[:,:3],axis=1));axes[2].set_ylabel("TCP speed [m/s]")
    axes[3].plot(t,np.linalg.norm(wrench[:,:3],axis=1),label="Requested force, NOT contact force")
    axes[3].set_ylabel("Requested force [N]");axes[3].legend()
    axes[4].plot(t,[r["minimum_singular_value"] for r in rows],label="Scaled J minimum singular value")
    axes[4].axhline(home["controller_config"]["minimum_singular_value"],color="red",ls="--",label="Experiment screen")
    axes[4].set_ylabel("Condition metric");axes[4].set_xlabel("Experiment time [s]");axes[4].legend()
    for axis in axes: axis.grid(True,alpha=.25)
    fig.suptitle("FR3 free-space twin — uncalibrated physical dynamics; no real motion yet")
    fig.tight_layout();fig.savefig(output,dpi=130);plt.close(fig)


def run_simulation(home, policy, output_dir, parameters=PlantParameters(), *, mesh_dir=None):
    from . import _twin_native
    import mujoco
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    config = ControllerConfig.model_validate(home["controller_config"])
    if native_inputs_digest() != _twin_native.twin_inputs_sha256:
        raise RuntimeError("Offline bridge is stale; rebuild _twin_native before comparing")
    crosscheck = model_crosscheck(home)
    if not crosscheck["passed"]:
        raise RuntimeError("MuJoCo/SDK rigid-body model conversion mismatch")
    path = preflight_path(home,policy)
    if not path["away_from_singularities"]:
        raise RuntimeError("Planned reference approaches the singularity threshold; no run")
    metadata = {
        "schema":"fcc-twin-run-v1", "backend":"mujoco", "home_sha256":home["sha256"],
        "home_capture":home["captured_utc"], "controller_config":config.model_dump(),
        "hardware_settings":home["hardware_settings"], "policy":policy.manifest(),
        "policy_sha256":digest(policy.manifest()), "plant":parameters.manifest(),
        "mujoco_version":mujoco.__version__, "sdk_version":_twin_native.libfranka_version,
        "controller_source_sha256":_twin_native.controller_source_sha256,
        "twin_inputs_sha256":_twin_native.twin_inputs_sha256,
        "native_extension_sha256":hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        "model_crosscheck":crosscheck, "path_preflight":path,
        "gravity":"SDK modeled gravity added once, representing robot-side compensation",
        "sdk_torque_processing":"actual libfranka 100 Hz lowpass then torque rate limiter",
        "initialization":("explicit measured start joints" if policy.start_joints is not None else "captured home joints")
                         + ", zero velocity and zero external commanded torque",
        "contact":"free-space only; no surveyed environment; wrench is truth zero, not estimated sensor output",
        "stop":"final home dwell; proprietary Franka MotionFinished braking is not modeled",
    }
    write_json_new(output_dir/"manifest.json",metadata)
    # Keep the generated MJCF and URDF alongside the experiment for inspection.
    (output_dir/"robot.xml").write_text(MujocoPlant(home,parameters,mesh_dir=mesh_dir,targets=policy.targets).xml)
    (output_dir/"robot.urdf").write_text(home["urdf"])
    rows = []
    began = time.monotonic()
    try:
        with (output_dir/"trace.jsonl").open("x") as stream:
            for row in simulation_trace(home,policy,parameters,mesh_dir=mesh_dir):
                stream.write(json.dumps(row,allow_nan=False)+"\n")
                rows.append(row)
        summary = summarize(rows,home,policy)
        summary["wall_time_s"] = time.monotonic()-began
        summary["model_crosscheck"] = crosscheck
        summary["path_preflight"] = path
        write_json_new(output_dir/"summary.json",summary)
        plot_trace(rows,home,policy,output_dir/"trajectory.png")
    except BaseException as error:
        write_json_new(output_dir/"failure.json",{"type":type(error).__name__,"error":str(error),"samples":len(rows)})
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation",required=True)
    capture = sub.add_parser("capture",help="fresh inactive HTTP read only; fix home once; never sends targets")
    capture.add_argument("--home",type=Path,required=True)
    capture.add_argument("--hardware-config",type=Path,default=ROOT/"local-hardware.json")
    capture.add_argument("--url",default="http://127.0.0.1:54321")
    simulate = sub.add_parser("sim",help="offline only; cannot connect to a robot")
    simulate.add_argument("--home",type=Path,required=True)
    simulate.add_argument("--output",type=Path,required=True)
    simulate.add_argument("--offset-m",type=float,default=.30,help="excursion each side of the SAME saved home")
    simulate.add_argument("--move-s",type=float,default=3.)
    simulate.add_argument("--dwell-s",type=float,default=2.)
    simulate.add_argument("--cycles",type=int,default=1)
    simulate.add_argument("--plant",type=Path,help="explicit JSON PlantParameters for sensitivity/sys-id")
    simulate.add_argument("--mesh-dir",type=Path,help="optional visual-only collision STL assets")
    args = parser.parse_args(argv)
    if args.operation == "capture":
        result = capture_home(args.home,args.hardware_config,args.url)
        print(json.dumps({"saved":str(args.home),"home":result["home"],"captured_utc":result["captured_utc"]},indent=2))
        return 0
    home = load_home(args.home)
    policy = ManualPosePolicy.from_home(home,offset_m=args.offset_m,move_s=args.move_s,dwell_s=args.dwell_s,cycles=args.cycles)
    parameters = PlantParameters(**json.loads(args.plant.read_text())) if args.plant else PlantParameters()
    result = run_simulation(home,policy,args.output,parameters,mesh_dir=args.mesh_dir)
    print(json.dumps(result,indent=2))
    return 0 if result["ready_for_operator_review"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
