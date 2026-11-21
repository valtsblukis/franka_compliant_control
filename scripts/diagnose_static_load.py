"""Fit an equivalent static flange load from saved probes; never connects to a robot.

This diagnostic cannot distinguish payload from cable forces, friction or model
bias, and cannot identify inertia. Its output is deliberately not a Desk or
HardwareSettings configuration. Use observations from distinct stationary poses.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def gravity_regressor(observation):
    state = observation["state"]
    if state["robot_mode"] != "Idle" or any(state["current_errors"]):
        raise ValueError("Observation must be Idle without robot errors")
    if np.max(np.abs(state["dq"])) >= 0.02:
        raise ValueError("Observation is moving")
    pose = np.array(state["O_T_EE"]).reshape(4, 4, order="F")
    tool = np.array(state["F_T_EE"]).reshape(4, 4, order="F")
    jacobian = np.array(observation["zero_jacobian"]).reshape(6, 7, order="F")
    if not np.allclose(pose.ravel(order="F"), observation["model_pose"], atol=5e-5, rtol=0):
        raise ValueError("Recorded model pose disagrees with robot state")
    if np.linalg.matrix_rank(jacobian) < 6:
        raise ValueError("Invalid or singular recorded Jacobian")
    flange_rotation = (pose @ np.linalg.inv(tool))[:3, :3]
    # Shift the translational Jacobian from TCP back to flange, in base axes.
    jv_flange = jacobian[:3] + skew(flange_rotation @ tool[:3, 3]) @ jacobian[3:]
    gravity = np.asarray(state["O_ddP_O"])
    h = np.column_stack((-jv_flange.T @ gravity,
                         jacobian[3:].T @ skew(gravity) @ flange_rotation))
    nominal = np.r_[state["m_total"], state["m_total"] * np.array(state["F_x_Ctotal"])]
    observed = np.asarray(state["tau_ext_hat_filtered"]) + h @ nominal
    if not np.isfinite(h).all() or not np.isfinite(observed).all():
        raise ValueError("Nonfinite torque/regressor")
    return h, observed, flange_rotation.T @ gravity / np.linalg.norm(gravity)


def fit(observations):
    blocks = [gravity_regressor(observation) for observation in observations]
    h, y = np.vstack([v[0] for v in blocks]), np.concatenate([v[1] for v in blocks])
    theta, _, rank, singular = np.linalg.lstsq(h, y, rcond=None)
    if rank != 4 or theta[0] <= 0:
        raise ValueError("Mass/first moment not identifiable as a positive mass")
    residual = h @ theta - y
    directions = np.array([v[2] for v in blocks])
    separation = np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1))).max()
    # Columns have different units; scale first moments by a 0.1 m reference
    # length for a dimensionally comparable conditioning diagnostic.
    scaled_singular = np.linalg.svd(h @ np.diag([1, .1, .1, .1]), compute_uv=False)
    return {
        "diagnostic_only": True,
        "configuration_ready": False,
        "assumptions": "All static torque residual attributed to flange gravity load; no contact. Cable forces, link cable weight, friction and model/sensor bias are not separated. Inertia is unidentifiable from static observations.",
        "fitted_mass_kg": float(theta[0]),
        "fitted_flange_com_m": (theta[1:] / theta[0]).tolist(),
        "rank": int(rank),
        "singular_values_unscaled": singular.tolist(),
        "condition_number_unscaled": float(singular[0] / singular[-1]),
        "condition_number_at_0_1m_reference_length": float(scaled_singular[0] / scaled_singular[-1]),
        "max_gravity_direction_separation_deg": float(separation),
        "rms_joint_torque_residual_Nm": float(np.sqrt(np.mean(residual ** 2))),
        "max_joint_torque_residual_Nm": float(np.max(np.abs(residual))),
        "residual_by_observation_Nm": residual.reshape(-1, 7).tolist(),
    }


def load_stationary_window(path):
    """Average saved torque samples only after checking the entire idle window."""
    observation = json.loads(path.read_text())
    state = observation["state"]
    gravity_regressor(observation)  # Validate initial state, model and Jacobian.
    rows = [json.loads(line) for line in Path(str(path) + ".samples.jsonl").read_text().splitlines()]
    if len(rows) < 1000:
        raise ValueError(f"{path}: at least 1,000 samples required")
    if any(row["robot_mode"] != "Idle" or row["current_errors"] for row in rows):
        raise ValueError(f"{path}: window contains motion mode or robot errors")
    fixed_keys = ("m_total", "F_x_Ctotal", "I_total", "F_T_EE", "EE_T_K", "O_ddP_O")
    if any(row[key] != state[key] for row in rows for key in fixed_keys):
        raise ValueError(f"{path}: configuration or gravity changed within window")
    q, dq, wrench, tau, times = [np.asarray([row[key] for row in rows], dtype=float)
                               for key in ("q", "dq", "O_F_ext_hat_K", "tau_ext_hat_filtered", "time")]
    if not all(np.isfinite(v).all() for v in (q, dq, wrench, tau, times)):
        raise ValueError(f"{path}: nonfinite window data")
    if np.any(np.diff(times) <= 0) or times[-1] - times[0] < 999:
        raise ValueError(f"{path}: need at least one second of increasing sample times")
    max_speed = float(np.abs(dq).max())
    max_drift = float(np.abs(q - np.asarray(state["q"])).max())
    if max_speed >= .02 or max_drift >= .001:
        raise ValueError(f"{path}: window is not stationary relative to recorded geometry")
    summary = {
        "observation": str(path), "samples": len(rows),
        "duration_ms": float(times[-1] - times[0]),
        "stationary_idle_no_errors": True, "configuration_constant": True,
        "max_abs_joint_velocity_rad_s": max_speed,
        "max_joint_position_span_rad": float(np.ptp(q, axis=0).max()),
        "max_joint_drift_from_initial_rad": max_drift,
        "mean_wrench_N_Nm": wrench.mean(axis=0).tolist(),
        "std_wrench_N_Nm": wrench.std(axis=0).tolist(),
        "mean_joint_torque_residual_Nm": tau.mean(axis=0).tolist(),
        "std_joint_torque_residual_Nm": tau.std(axis=0).tolist(),
        "contact_free": "Relies on operator hands-off/no-contact procedure; cannot be inferred from stationary data",
    }
    # Keep the independently checked initial geometry. Its joint deviation from
    # all samples is below 0.001 rad. Only average measured residuals, not transforms.
    state["tau_ext_hat_filtered"] = summary["mean_joint_torque_residual_Nm"]
    state["O_F_ext_hat_K"] = summary["mean_wrench_N_Nm"]
    return observation, summary


def leave_one_out(observations, paths):
    checks = []
    for heldout, observation in enumerate(observations):
        try:
            trained = fit([d for i, d in enumerate(observations) if i != heldout])
        except ValueError as error:
            checks.append({"held_out_observation": str(paths[heldout]), "training_error": str(error)})
            continue
        h, y, _ = gravity_regressor(observation)
        mass = trained["fitted_mass_kg"]
        theta = np.r_[mass, mass * np.asarray(trained["fitted_flange_com_m"])]
        residual = h @ theta - y
        checks.append({
            "held_out_observation": str(paths[heldout]),
            "training_mass_kg": mass, "training_com_m": trained["fitted_flange_com_m"],
            "training_gravity_span_deg": trained["max_gravity_direction_separation_deg"],
            "training_scaled_condition_number": trained["condition_number_at_0_1m_reference_length"],
            "held_out_joint_torque_RMS_Nm": float(np.sqrt(np.mean(residual ** 2))),
            "held_out_max_abs_joint_torque_Nm": float(np.abs(residual).max()),
        })
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--average-windows", action="store_true",
                        help="Validate and average each probe's .samples.jsonl sidecar")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Diagnostic output already exists")
    summaries = []
    if args.average_windows:
        loaded = [load_stationary_window(path) for path in args.observations]
        observations, summaries = [v[0] for v in loaded], [v[1] for v in loaded]
    else:
        observations = [json.loads(path.read_text()) for path in args.observations]
    report = fit(observations)
    report["observations"] = [str(path) for path in args.observations]
    if summaries:
        report["window_summaries"] = summaries
        report["processing"] = "Mean torque residuals from validated stationary windows; initial geometry retained"
    if len(observations) >= 3:
        report["leave_one_pose_out"] = leave_one_out(observations, args.observations)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Equivalent static load: {report['fitted_mass_kg']:.3f} kg; diagnostic only")
    print(f"Observed gravity-direction span: {report['max_gravity_direction_separation_deg']:.2f} deg")
    print(f"Torque residual RMS: {report['rms_joint_torque_residual_Nm']:.3f} Nm")
    print(f"Saved {args.output}; no configuration or robot changes")


if __name__ == "__main__":
    main()
