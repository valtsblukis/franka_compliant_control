# Simulator / model contract

`SimulationWorld` owns MuJoCo state and scene components; it accepts robot torque
**including intended gravity compensation**. It imports no SDK, Gym or optimizer.
`NativeControllerSimulation` adds the production C++ law, SDK model/filter, command
leases and optional sensing. Tasks/rewards/Gym adapters belong to `franka_motion_rl`.

## Model

- Robot-exported FR3 URDF: link transforms, masses, COMs and full inertias.
- Recorded estimated 1.26 kg wrist assembly; TCP .174 m along flange Z. Load
  combination uses the parallel-axis theorem. Not a generic Panda substitution.
- MuJoCo: `implicitfast`, Newton, default 1 ms physics; substeps must divide 1 ms.
- Native loop: 1 kHz; poses normally published at 50 Hz with zero twist/acceleration
  feedforward. No hidden native interpolation or extra force controller.
- Torque order: native output → SDK 100 Hz low-pass → SDK rate limiter → optional
  extra-cycle delay → add SDK gravity once → plant. Do not add Coriolis twice.
- Plant resistance: MuJoCo Coulomb/viscous terms plus **subtracted** `joint_bias`.
  `tau_J_d` excludes robot-side gravity here; it is not raw measured motor effort.

Canonical inputs are under `simulation/data/`; `simulation.presets` returns fresh
copies. `reference_context()` preserves the saved home/TCP/dynamics but removes
live addresses and is marked simulation-only. `identified_plant()` selects the
local fit; bare `PlantParameters()` still means the earlier uncalibrated hypothesis.

The selected effective Fc/bias values are in the JSON, not duplicated here.
Viscosity remains .003 Nms/rad per joint; added armature/extra delay remain zero
**assumptions**. Near-static friction uses impedance .9999 and .01 s time constant.
Primary held-out SDK-TCP replay RMS improved 8.02 → 1.17 mm. Later rejected fits
are archived, not silently selected. This is local low-speed evidence, not global
calibration; frame-consistent SDK FK is not independent physical metrology.

## Sensing and scenes

- No wrench sensor means an invalid/unmodeled observation, not calibrated zero.
- `FrankaWrenchSurrogate`: local free-space SDK-estimate model, effective lag .24 s;
  true-contact gain is assumed, not calibrated. On this installation the raw moment
  includes `p_TCP × F`; use explicit TCP/base normalization with the known K transform.
- `FilteredContactSensor`: synthetic gain/lag/noise assumptions, separate from truth.
- `KinematicNoiseModel`: optional 1 kHz q/dq variability before SDK/native feedback;
  physical state is unchanged. Only near-static dq spread/angular-speed p99 were
  validated; it does not model wrist release. Not enabled by default.
- Scene components own one world and implement build/bind/reset/step/force hooks.
  Force hooks must be pure except for adding forces (sensing uses scratch data).
- Robot joints are indexed by name, never assumed to occupy `qpos[:7]` when scene
  free bodies exist. Visual meshes/floor are not collision-clearance evidence.
- Movable objects have finite mass/inertia, nonintersecting spawn and bounded-force
  unloaded withdrawal. Solver force magnitudes/impulse/history are accumulated at
  every physics substep; removal cannot erase cost. Proxies/materials are approximate.

A stopped rollout needs reset. Simulation termination is **not** a model of Franka
braking or a safety-rated stop. New postures, tool/cable changes, power cycles,
contact tasks and faster motion require separate validation. Do not fit validation
recordings or present a low-error failed prefix as a complete successful trial.
