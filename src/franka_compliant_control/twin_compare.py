"""Compare experiment-owned simulation/HTTP logs, without commanding either backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .twin_experiment import write_json_new,digest
from .recorded_policy import recorded_policy_spec
from .twin_model import rotation_distance


def load_rows(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows)<2 or not all(np.isfinite(r["time_s"]) for r in rows):
        raise ValueError("Need at least two finite timestamped observations")
    return rows


def compare(simulation_dir, observed_dir, output_dir):
    """No fitted time shift: retain transport/actuator latency in the residual."""
    simulation_dir,observed_dir,output_dir = map(Path,(simulation_dir,observed_dir,output_dir))
    sim = json.loads((simulation_dir/"manifest.json").read_text())
    obs = json.loads((observed_dir/"manifest.json").read_text())
    observed_policy_sha=obs.get('policy_sha256')
    if obs.get('schema') in ('fcc-force-aware-real-v1','fcc-learned-reference-real-v1'):
        observed_policy_sha=digest(recorded_policy_spec(obs))
        if sim.get('accepted_actions_sha256')!=digest((observed_dir/'actions.jsonl').read_text()):
            raise ValueError('Recorded-policy comparison requires the actual accepted-command stream')
    if (sim["home_sha256"]!=obs["home_sha256"] or sim["policy_sha256"]!=observed_policy_sha
            or sim["controller_config"]!=obs["controller_config"]):
        raise ValueError("Cannot compare different fixed origins, policies or controller settings")
    a,b = load_rows(simulation_dir/"trace.jsonl"),load_rows(observed_dir/"trace.jsonl")
    ta = np.array([r["time_s"] for r in a])
    if np.any(np.diff(ta)<=0):
        raise ValueError("Simulation times must strictly increase")
    # Repeated reads of the same cached observation are not independent samples.
    unique=[];last_robot_time=-np.inf
    for row in b:
        timestamp = row.get("robot_time_s",row["time_s"])
        if timestamp is None:
            raise ValueError("Observed log lacks measurement clock")
        if timestamp < last_robot_time:
            raise ValueError("Observed robot clock went backwards")
        if timestamp > last_robot_time and ta[0] <= row["time_s"] <= ta[-1]:
            unique.append(row)
        last_robot_time=timestamp
    if len(unique)<2:
        raise ValueError("Insufficient overlapping fresh observations")
    tb = np.array([r["time_s"] for r in unique])
    if np.any(np.diff(tb)<=0):
        raise ValueError("Observed experiment times must increase")
    def sampled(key):
        values = np.array([r[key] for r in a])
        return np.column_stack([np.interp(tb,ta,values[:,i]) for i in range(values.shape[1])])
    residuals={key:np.array([r[key] for r in unique])-sampled(key)
               for key in ("position","positions","velocities","twist","last_desired_joint_torques")}
    # Normalize interpolated quaternion components before angular comparison.
    # With dense 1 kHz small increments this is nlerp; align signs first.
    qa=np.array([r["quaternion_xyzw"] for r in a])
    for i in range(1,len(qa)):
        if qa[i]@qa[i-1]<0:qa[i]*=-1
    qs=np.column_stack([np.interp(tb,ta,qa[:,i]) for i in range(4)])
    qs/=np.linalg.norm(qs,axis=1)[:,None]
    orientation=np.array([rotation_distance(row["quaternion_xyzw"],q) for row,q in zip(unique,qs)])
    position=np.linalg.norm(residuals["position"],axis=1)
    def completed(folder):
        for filename in ("summary.json", "completion.json"):
            path=folder/filename
            if path.is_file():
                return bool(json.loads(path.read_text()).get("completed",False)) and not (folder/"failure.json").exists()
        return False
    coverage=float((tb[-1]-tb[0])/sim["policy"]["duration_s"])
    full=completed(simulation_dir) and completed(observed_dir) and coverage>=.98
    result={"schema":"fcc-twin-comparison-v1", "simulation":str(simulation_dir),
            "observed":str(observed_dir),"observed_backend":obs["backend"],
            "overlap_samples":len(unique),"overlap_s":float(tb[-1]-tb[0]),
            "reference_completed":completed(simulation_dir),"observed_completed":completed(observed_dir),
            "coverage_fraction":coverage,"full_run_comparison":full,
            "observed_last_phase":b[-1].get("phase"),"observed_last_reason":b[-1].get("reason"),
            "warning":None if full else "PARTIAL/FAILED run: low overlap RMSE does not imply a successful match",
            "position_rmse_m":float(np.sqrt(np.mean(position**2))),
            "position_max_error_m":float(position.max()),
            "orientation_rmse_rad":float(np.sqrt(np.mean(orientation**2))),
            "per_joint_position_rmse_rad":np.sqrt(np.mean(residuals["positions"]**2,axis=0)).tolist(),
            "per_joint_velocity_rmse_rad_s":np.sqrt(np.mean(residuals["velocities"]**2,axis=0)).tolist(),
            "per_joint_desired_torque_rmse_Nm":np.sqrt(np.mean(residuals["last_desired_joint_torques"]**2,axis=0)).tolist(),
            "alignment":"experiment timestamps; no optimized time shift or spatial alignment",
            "force_comparison":"not scored: free-space truth zero is not a Franka observer model",
            "calibrated":False}
    output_dir.mkdir(parents=True,exist_ok=False)
    write_json_new(output_dir/"comparison.json",result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    fig.suptitle("Full-run comparison" if full else "PARTIAL/FAILED run — overlap metrics only")
    axes[0].plot(ta,[r["position"][1] for r in a],label="simulation")
    axes[0].plot(tb,[r["position"][1] for r in unique],"--",label=obs["backend"]+" observation")
    axes[0].set_ylabel("Base Y [m]");axes[0].legend()
    axes[1].plot(tb,1000*position);axes[1].set_ylabel("Position residual [mm]")
    axes[2].plot(tb,residuals["velocities"]);axes[2].set_ylabel("Joint speed residual [rad/s]")
    axes[2].set_xlabel("Experiment time [s]")
    for axis in axes:axis.grid(True,alpha=.25)
    fig.tight_layout();fig.savefig(output_dir/"comparison.png",dpi=130);plt.close(fig)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation",required=True,type=Path)
    parser.add_argument("--observed",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(compare(args.simulation,args.observed,args.output),indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
