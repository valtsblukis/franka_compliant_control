"""Guarded real counterpart of the saved MuJoCo manual policy.

Never invoked by the simulator. Requires explicit execution AND cleared-path
acknowledgement, a successful matching simulation and the unchanged fixed home.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .client import CompliantClient
from .models import ControllerConfig
from .twin_experiment import ManualPosePolicy, digest, load_home, write_json_new
from .twin_model import rotation_distance


def validated_plan(home, simulation_dir):
    simulation_dir = Path(simulation_dir)
    manifest = json.loads((simulation_dir/"manifest.json").read_text())
    summary = json.loads((simulation_dir/"summary.json").read_text())
    if (manifest["backend"] != "mujoco" or manifest["home_sha256"] != home["sha256"]
            or manifest["controller_config"] != home["controller_config"]
            or not summary["ready_for_operator_review"] or not manifest["model_crosscheck"]["passed"]):
        raise ValueError("A completed matching simulation and fixed home are required")
    fields = ManualPosePolicy.__dataclass_fields__
    policy = ManualPosePolicy(**{key:value for key,value in manifest["policy"].items() if key in fields})
    if (digest(policy.manifest()) != manifest["policy_sha256"]
            or not np.array_equal(policy.home_position,home["home"]["position"])
            or not np.array_equal(policy.home_quaternion_xyzw,home["home"]["quaternion_xyzw"])):
        raise ValueError("Policy/home fingerprint differs from the reviewed simulation")
    return policy,manifest


def check_start_state(state, home):
    if (state["backend"] != "robot" or state["running"] or state["phase"] != "disarmed"
            or state["robot_mode"] != "Idle" or state["robot_has_errors"]
            or not state["sample_valid"] or not state["kinematics_valid"]
            or not 0 <= state["sample_age_s"] <= .1):
        raise RuntimeError("A fresh, inactive, valid Idle robot observation is required")
    if np.max(np.abs(state["velocities"])) >= .02:
        raise RuntimeError("Robot must be stationary")
    if (np.linalg.norm(np.array(state["position"])-home["home"]["position"]) > .002
            or rotation_distance(state["quaternion_xyzw"],home["home"]["quaternion_xyzw"]) > np.deg2rad(1)
            or np.max(np.abs(np.array(state["positions"])-home["home"]["joints"])) > .02):
        raise RuntimeError("Robot moved away from the fixed captured home; no automatic return/reinitialization")


def run_real(home, simulation_dir, output_dir, *, execute=False, clear_path=False,
             url="http://127.0.0.1:54321", client_factory=CompliantClient,
             clock=time.monotonic, sleep=time.sleep):
    policy,simulation = validated_plan(home,simulation_dir)
    # Check both acknowledgements before constructing any network client.
    if not execute or not clear_path:
        raise ValueError("Real motion requires --execute AND --confirm-clear-path")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    config = ControllerConfig.model_validate(home["controller_config"])
    metadata = {"schema":"fcc-twin-run-v1", "backend":"robot", "home_sha256":home["sha256"],
                "controller_config":home["controller_config"], "policy":policy.manifest(),
                "policy_sha256":digest(policy.manifest()), "simulation_dir":str(simulation_dir),
                "simulation_controller_source_sha256":simulation["controller_source_sha256"],
                "measurement":"HTTP latest samples at approximately 50 Hz; not 1 kHz system identification",
                "settings_changes":False, "automatic_recovery_required":False,
                "http_connection_policy":"Connection: close; avoids observed persistent-connection delayed-ACK latency",
                "experiment_guards":{"max_home_distance_m":policy.offset_m+.02,
                    "max_orientation_error_deg":2.,"max_angular_speed_rad_s":.2,
                    "max_translation_speed_m_s":max(.1,3*1.875*policy.offset_m/policy.move_s)}}
    write_json_new(output_dir/"manifest.json",metadata)
    robot = client_factory(url,timeout_s=4.)
    # The deployed HTTP stack exhibited ~44 ms keepalive response latency.
    # Fresh local connections measured ~2 ms. This affects only experiment
    # transport, never the native FCI connection or controller configuration.
    robot.http.headers["Connection"] = "close"
    attempted = False
    stop_result = None
    rows = []
    try:
        health = robot.health()
        if health["automatic_recovery_enabled"]:
            raise RuntimeError("Disable automatic recovery explicitly before this experiment; runner never changes settings")
        current_config = ControllerConfig.model_validate(robot.get_config()).model_dump()
        if current_config != home["controller_config"]:
            raise RuntimeError("Running controller settings differ from captured/simulated settings")
        if digest(robot.get_urdf()["urdf"]) != home["sha256"]["urdf"]:
            raise RuntimeError("Robot URDF differs from the simulated model")
        robot.refresh_state()  # Read only, rejected by backend if another producer is active.
        initial = robot.state()
        check_start_state(initial,home)
        first_reference = policy.sample(0.)
        if (np.linalg.norm(np.array(initial["position"])-first_reference["position"]) > .0002
                or rotation_distance(initial["quaternion_xyzw"],first_reference["quaternion_xyzw"]) > np.deg2rad(.05)
                or (policy.start_joints is not None and
                    np.max(np.abs(np.array(initial["positions"])-policy.start_joints)) > .001)):
            raise RuntimeError("Measured start differs from the simulated first reference; resimulate alignment without changing home")
        write_json_new(output_dir/"initial_state.json",initial)
        # Neither this runner nor the policy uses /go_home. Every target comes
        # from the immutable saved origin, never from subsequent observations.
        with (output_dir/"trace.jsonl").open("x") as stream, (output_dir/"actions.jsonl").open("x") as actions:
            first = policy.sample(0.)
            attempted = True  # A lost response can still mean the first target started control.
            response = robot.set_target_ee_pose(first["position"],first["quaternion_xyzw"])
            robot.timeout_s = .2  # Streaming must not hide a long transport stall.
            startup_deadline = clock()+.5
            while True:
                startup = robot.state()
                if startup["running"] and startup["phase"] == "active":
                    break
                if not startup["running"] or startup["phase"] != "armed" or clock()>=startup_deadline:
                    raise RuntimeError("Native controller did not enter active state after startup")
                sleep(.005)
            # Arm/startup latency is logged separately. Start the paced policy
            # after the first active observation, while still holding saved home.
            origin = clock()
            server_origin = startup["server_monotonic_s"]
            write_json_new(output_dir/"startup.json",{"state":startup,"first_response":response,
                           "policy_origin_server_monotonic_s":server_origin})
            actions.write(json.dumps({"policy":first,"response":response,"sent_time_s":0.})+"\n")
            next_tick = 0
            while True:
                due = origin+next_tick/policy.hz
                sleep(max(0.,due-clock()))
                now = clock()
                if now-due > .05:
                    raise RuntimeError("Policy publisher missed its schedule by >50 ms; no catch-up replay")
                elapsed = now-origin
                state = robot.state()
                if (not state["running"] or state["phase"] not in ("armed","active")
                        or not state["sample_valid"] or not state["kinematics_valid"]
                        or not 0 <= state["sample_age_s"] <= .05):
                    raise RuntimeError("Inactive/faulted/stale observation; no restart or recovery")
                if (state["minimum_singular_value"] < config.minimum_singular_value
                        or state["osc_regularized"]):
                    raise RuntimeError("Singularity screen triggered")
                if (np.linalg.norm(np.array(state["position"])-home["home"]["position"]) > policy.offset_m+.02
                        or rotation_distance(state["quaternion_xyzw"],home["home"]["quaternion_xyzw"]) > np.deg2rad(2)):
                    raise RuntimeError("Experiment pose excursion exceeded the reviewed path neighborhood")
                twist = np.asarray(state["twist"])
                if (twist.shape != (6,) or not np.isfinite(twist).all()
                        or np.linalg.norm(twist[:3]) > max(.1,3*1.875*policy.offset_m/policy.move_s)
                        or np.linalg.norm(twist[3:]) > .2):
                    raise RuntimeError("Experiment measured-speed screen triggered")
                measured_time = state["server_monotonic_s"]-state["sample_age_s"]-server_origin
                row = {**state,"time_s":max(0.,measured_time),"host_elapsed_s":elapsed,
                       "policy":policy.sample(max(0.,measured_time))}
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+"\n")
                if elapsed >= policy.duration_s:
                    break
                target = policy.sample(elapsed)
                if next_tick:
                    result = robot.set_target_ee_pose(target["position"],target["quaternion_xyzw"])
                    actions.write(json.dumps({"policy":target,"response":result,"sent_time_s":elapsed},allow_nan=False)+"\n")
                next_tick += 1
            last = rows[-1]
            if (np.linalg.norm(np.array(last["position"])-home["home"]["position"]) > .005
                    or np.max(np.abs(last["velocities"])) >= config.stop_velocity):
                raise RuntimeError("Final home did not settle within the scheduled dwell")
    except BaseException as error:
        write_json_new(output_dir/"failure.json",{"type":type(error).__name__,"error":str(error),"samples":len(rows)})
        raise
    finally:
        try:
            if attempted:
                robot.timeout_s = 3.5  # Longer than the requested stop-confirmation wait.
                stop_result = robot.stop(wait_s=2.5)
                write_json_new(output_dir/"stop.json",stop_result)
        finally:
            # stop() clears the client's setpoint flag, so close does not reissue.
            robot.close()
    if not stop_result or not stop_result["stopped"]:
        write_json_new(output_dir/"failure.json",{"error":"stationary disarmed stop not confirmed","samples":len(rows)})
        raise RuntimeError("Stationary disarmed hardware stop was not confirmed; inspect stop.json")
    write_json_new(output_dir/"completion.json",{"completed":True,"samples":len(rows),"stop_confirmed":True})
    return {"completed":True,"samples":len(rows),"stop_confirmed":True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home",type=Path,required=True)
    parser.add_argument("--simulation",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--url",default="http://127.0.0.1:54321")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--confirm-clear-path",action="store_true")
    args = parser.parse_args(argv)
    home = load_home(args.home)
    policy,_ = validated_plan(home,args.simulation)
    if not args.execute:
        print(json.dumps({"dry_run":True,"robot_connection":False,"policy":policy.manifest(),
                          "required":"operator review of full swept arm/tool path, fixed-home check, automatic recovery disabled"},indent=2))
        return 0
    print(json.dumps(run_real(home,args.simulation,args.output,execute=args.execute,
                             clear_path=args.confirm_clear_path,url=args.url),indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
