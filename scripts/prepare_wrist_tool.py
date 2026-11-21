"""Prepare the provisional wrist assembly from the completed static survey.

No robot access. Output is a commissioning candidate, with a separately
documented inertia approximation; static data cannot identify inertia.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from franka_compliant_control.models import EndEffectorSettings, HardwareSettings

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--camera-model", choices=("zed2i", "zedmini"), default="zed2i")
args = parser.parse_args()
survey_path = root / "logs/load-survey-after-pose-05.json"
survey = json.loads(survey_path.read_text())
baseline_path = root / "configs/desk/robotiq-2f85/endeffector-config.json"
baseline = json.loads(baseline_path.read_text())
if len(survey["window_summaries"]) != 5 or not all(
        row["stationary_idle_no_errors"] and row["configuration_constant"]
        for row in survey["window_summaries"]):
    raise ValueError("Expected the completed five-pose stationary survey")

# Round the equivalent fit to avoid implying sub-gram/sub-millimetre accuracy.
mass = round(survey["fitted_mass_kg"], 2)
com = np.round(survey["fitted_flange_com_m"], 3)
base_mass = baseline["mass"]
base_com = np.asarray(baseline["centerOfMass"])
base_inertia = np.asarray(baseline["inertia"]).reshape(3, 3, order="F")
extra_mass = mass - base_mass
if not 0 < extra_mass < 1 or np.linalg.norm(com) > .2:
    raise ValueError("Survey result outside this candidate's assumed assembly range")
extra_com = (mass * com - base_mass * base_com) / extra_mass

# Camera dimensions are an explicit example assumption, not identification
# of the installed camera. Independently verify the actual assembly.
# Spread ALL equivalent excess mass over a camera-sized box, then average its
# COM inertia over orientations: I_iso = trace(I_box)/3 = m*sum(d^2)/18.
# This is a surrogate for camera/bracket/supported cabling, NOT measured
# component inertia, a certified bound, or a model of cables on other arm links.
dimensions = np.array([.1753, .0303, .0431] if args.camera_model == "zed2i"
                      else [.1245, .0305, .0265])
extra_inertia = np.eye(3) * extra_mass * np.dot(dimensions, dimensions) / 18

def parallel_axis(m, offset):
    return m * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))

inertia = (base_inertia + parallel_axis(base_mass, base_com - com)
           + extra_inertia + parallel_axis(extra_mass, extra_com - com))
inertia = np.round(inertia, 9)
settings = EndEffectorSettings(mass=mass, com=com.tolist(),
                              inertia=inertia.ravel(order="F").tolist())
config = {
    "id": "robotiq-wrist-load-estimated", "name": "Robotiq + wrist load (estimate)",
    "color": "#b37400", "mass": mass, "centerOfMass": settings.com,
    "transformation": baseline["transformation"], "inertia": settings.inertia,
}
target = root / "configs/desk/robotiq-wrist-estimated"
target.mkdir(parents=True, exist_ok=True)

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

write_json(target / "endeffector-config.json", config)
derivation = {
    "status": "provisional commissioning candidate; not applied or calibrated",
    "survey": str(survey_path.relative_to(root)),
    "survey_sha256": hashlib.sha256(survey_path.read_bytes()).hexdigest(),
    "fitted_mass_kg": survey["fitted_mass_kg"], "fitted_com_m": survey["fitted_flange_com_m"],
    "selected_mass_kg": mass, "selected_com_m": com.tolist(),
    "baseline": str(baseline_path.relative_to(root)),
    "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    "baseline_mass_kg": base_mass, "baseline_com_m": base_com.tolist(),
    "equivalent_extra_mass_kg": extra_mass, "equivalent_extra_com_m": extra_com.tolist(),
    "extra_box_dimensions_m": dimensions.tolist(),
    "camera_assumption": args.camera_model,
    "camera_model_status": "Selected camera dimensions are an example assumption, not identification of the installed camera. Verify the actual assembly before use.",
    "camera_evidence": "Public camera dimensions only; no installation-specific calibration is distributed",
    "dimensions_source": ("https://support.stereolabs.com/hc/en-us/article_attachments/27901419901463"
                          if args.camera_model == "zed2i" else
                          "https://support.stereolabs.com/hc/article_attachments/27901442262551"),
    "inertia_method": "Franka nominal gripper I_COM plus parallel-axis terms and orientation-averaged uniform-box inertia for all equivalent excess mass. Does not identify or model actual distributed cable inertia.",
    "equivalent_extra_inertia_at_com_kg_m2": extra_inertia.ravel(order="F").tolist(),
    "selected_total_inertia_at_com_kg_m2": settings.inertia,
    "inertia_eigenvalues_kg_m2": np.linalg.eigvalsh(inertia).tolist(),
    "configuration_ready_for_powered_control": False,
    "next_step": "Review candidate assumptions, import in Desk, verify readback and resting residuals before any bounded powered test",
}
write_json(target / "derivation.json", derivation)
hardware = json.loads((root / "local-hardware.json").read_text())
hardware.update(read_only=True, calibration_confirmed=False, configuration_basis="estimated",
                configuration_source="Five-pose static survey logs/load-survey-after-pose-05.json; explicit surrogate inertia in configs/desk/robotiq-wrist-estimated/derivation.json. Provisional and not yet validated on robot.",
                expected_end_effector=settings.model_dump())
candidate = HardwareSettings.model_validate(hardware)
assert candidate.native().read_only
write_json(root / "configs/hardware.wrist-load.estimated.json", candidate.model_dump())
print(f"Prepared {mass:.2f} kg, COM {com.tolist()} m; candidate remains read-only")
print(f"COM inertia eigenvalues: {np.linalg.eigvalsh(inertia).tolist()} kg m^2")
print("No robot settings, runtime configuration or controller process changed")
