# Franka compliant control

Native Cartesian/joint compliant controller, HTTP client/service, and a MuJoCo FR3
model running the same production controller. **No RL dependency.**

Motion tasks, policies and optimizers live in the separate `franka_motion_rl`
repository/package. The dependency is one-way: **RL → controller/simulator**.

## Install and build

Python ≥3.11; tested on Linux/Python 3.12. MuJoCo is optional.

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[simulation,dev]'

# Fresh simulator checkout: separately installed libfranka ≥0.18 required.
# Builds _native WITHOUT robot support, plus the offline _twin_native bridge.
.venv/bin/python scripts/build_simulation.py --franka-prefix /path/to/sdk-prefix

.venv/bin/franka-sim --controller --seconds 1
.venv/bin/python -m pytest -q
```

For **physics only**, skip the native build and run `franka-sim` without
`--controller`. For the toy API backend only, use `scripts/build_native.py` without
SDK arguments. Neither is a hardware deployment. See [build notes](docs/build.md).
**Never replace a native library loaded by a running robot service.**

### Conda alternative

A conda environment can be used alongside `.venv`:

```sh
conda create -n compliant_control python=3.12
conda activate compliant_control
cd /path/to/franka_compliant_control
python -m pip install -e '.[simulation,sysid,dev]'
python -m pytest -q  # after building matching native modules; see build notes
```

Python 3.12.13 was used for validation. Both environments can use the same
existing native libraries and SDK when the Python ABI matches; activation does
not rebuild them or restart the service. Initialize conda for your shell if
`conda activate` is unavailable.

`./start.sh` deliberately continues to use `.venv`. To use the activated conda
Python for the robot service instead, **stop the existing service and all command
producers first**, then run:

```sh
python scripts/run_robot.py --config configs/controller.fr3.auto.json --host 0.0.0.0
```

Do not run two robot services concurrently. The conda environment does not
install the separate RL package or Torch.

## Policy startup defaults

`./start.sh` selects `configs/controller.fr3.auto.json`: translation stiffness
`[600,600,600]`, damping `[48.99,48.99,48.99]`, commanded moment limit `6 Nm`, and
friction compensation `f=1.0 Nm`, `epsilon=0.02 rad/s`, with a `0.035 s` filter
(time constant). The service starts disarmed; the first valid setpoint can start
control with these settings. Generic configuration objects and other profiles
remain compensation-off. The policy simulator must use the same controller/filter
settings; frozen reference captures are not modified. See [controller notes](docs/controller.md).

## Manual contact relief (staged API)

```python
ack = client.relieve_contact(workspace_confirmed_clear=True)
```

Default manual mode follows raw SDK wrench with no bias subtraction or automatic
release. The acknowledgment means an **operator-supervised cleared local
retreat/braking area**; supervise and stop explicitly with `client.stop()`.
Raw bias may cause drift or motion in the wrong direction; **slow is not low force**.
Default caps are 0.005 m/s, 1 degree/s and 10 s, with at most 5 cm / 10 degrees of
travel and native hard guards. A TCP box is optional in manual mode. Automatic
release is opt-in with verified free-space bias and an explicit box. Both require
inactive Disarmed control and **automatic recovery OFF**; neither guarantees
unjamming. This is not a deployment/live-support claim. See the full
[API and safety boundaries](docs/controller.md#bounded-contact-relief-api).

## Simulator API

```python
from franka_compliant_control.simulation.presets import reference_context, identified_plant
from franka_compliant_control.simulation.world import SimulationWorld
from franka_compliant_control.simulation.native import NativeControllerSimulation

sim = NativeControllerSimulation(SimulationWorld(reference_context(), identified_plant()))
p, q = sim.sample.position.copy(), sim.sample.quaternion_xyzw.copy()
for tick in range(1000):
    if tick % 20 == 0:
        sim.submit_pose(p, q, valid_for_s=.2)
    result = sim.step_control()  # 1 ms; returns measurement, phase and contact reports
```

Poses are base-frame metres / **xyzw** quaternions. Reference shaping belongs to
the caller; the native controller sees held pose references at 1 kHz. The usual
policy rate is 50 Hz. [Model and layer contract](docs/simulation.md).

## Ownership

| Area | Location |
|---|---|
| Production torque controller / hardware backend | `native/controller.*`, `native/backend.*` |
| Service and client | `service.py`, `client.py` in the Python package |
| Model conversion / physical parameters | `twin_model.py` |
| Physics, native loop, sensing, contacts and scenes | `simulation/` |
| Small versioned reference model inputs | `simulation/data/`, loaded through `simulation.presets` |
| Identification / accepted-command replay | `system_identification.py`, `joint_identification.py`, `sysid_*`, `twin_*` |
| Native-rate acquisition | `native/telemetry_recorder.hpp`, `native_telemetry.py` |

`backend=sim` in the HTTP service is a **toy API test backend**, not the MuJoCo
model. Controller, pure physics and native-controller simulation can be used
without Gym or Torch. APIs and configuration validation are defined in code;
see [controller notes](docs/controller.md) for operational boundaries.

## Scope

The selected plant fits local effective resistance for a particular FR3/tool;
it is not a universal motor/contact model. SDK-based pose comparisons are not
independent TCP metrology. Wrist breakaway/internal-drive dynamics, physical stop
trajectories and human-contact safety remain unqualified. Example model inputs
and test coverage do not establish clearance for a hardware workspace.

Historical commissioning notes, raw runs and deployment records are not included
in this public snapshot. Only small model inputs and regression fixtures belong
in Git; new run output goes to an external or ignored directory.
