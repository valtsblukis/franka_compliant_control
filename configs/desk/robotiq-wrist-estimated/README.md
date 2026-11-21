# Robotiq and wrist assembly — illustrative estimated Desk configuration

**This is an example commissioning candidate, not a measured full inertia model
or approval to use these values on another assembly.** Raw commissioning records
and installation-specific observations are not distributed. Independently verify
the actual tool, camera, brackets, cabling and payload before hardware use.

| Property | Example value |
| --- | --- |
| Total empty assembly mass | 1.26 kg |
| Total COM in flange axes | [-0.019, -0.001, 0.035] m |
| Defined TCP | [0, 0, 0.174] m, axes parallel to flange |
| Additional `setLoad` mass | zero if the complete equivalent assembly is in Desk |
| Inertia | Provisional COM tensor in `endeffector-config.json` |
| Desk display name | `Robotiq + wrist load (estimate)` |

## Explicit inertia approximation

Static measurements can identify mass and first moments under a rigid-load
assumption; they do **not** identify inertia. This example combines:

1. The complete 0.900 kg Robotiq dynamics tuple from
   [Franka's example](https://github.com/frankarobotics/external_gripper_example/blob/master/panda_with_robotiq_gripper_example/config/endeffector-config.json).
2. An equivalent additional 0.360 kg at flange COM [-0.0665,-0.0035,-0.0200] m,
   giving the total mass and COM above.
3. A provisional COM inertia for the additional mass: an orientation average of
   a uniform box using ZED 2i external dimensions, 175.3 × 30.3 × 43.1 mm. Its
   isotropic inertia is `m*sum(dimensions²)/18`. The parallel-axis theorem combines
   the two bodies about their total COM.

The camera dimensions are from the
[Stereolabs ZED 2i datasheet](https://support.stereolabs.com/hc/en-us/article_attachments/27901419901463).
Spreading all excess mass over a camera-sized box is an engineering approximation,
not a measured assembly inertia, conservative bound, or model of cables attached
to other links. Camera dimensions do not identify the installed camera. Select
`--camera-model zedmini` only when that model and approximation are appropriate.
Never call this surrogate inertia measured or silently substitute another model.

`derivation.json` retains the numerical derivation and uncertainty statements.
Local survey records are not included. `scripts/prepare_wrist_tool.py` needs an
independently reviewed five-pose stationary survey in its expected input format
and a local hardware configuration; it performs no robot access. It overwrites
the candidate files, so preserve any installation-specific configuration first.

## Packaging and import

To package the example configuration without regenerating its values:

```bash
python scripts/package_desk_tool.py \
  --source configs/desk/robotiq-wrist-estimated/endeffector-config.json \
  --name robotiq-wrist-load-estimated \
  --output-dir artifacts/desk/wrist-load-estimated
```

After independent review, use Desk's end-effector import for the generated
`robotiq-wrist-load-estimated.tar.gz` and select the intended device. The adjacent
parameters JSON provides mechanical values for a legacy importer. These are
alternative formats for one tool, not two loads. The gripper-only example is
retained separately.

Before powered control, read back `m_ee`, `F_x_Cee`, `I_ee`, `F_T_NE`, `NE_T_EE`,
`F_T_EE`, `EE_T_K` and additional load, and check stationary residual force/moment
without contact. `configs/hardware.wrist-load.estimated.json` contains an IP
placeholder, remains read-only, and requires exact tool readback with
`configuration_basis=estimated` and `calibration_confirmed=false`. It is not a
ready-to-run hardware configuration. Packaging or importing is not authorization
to arm, recover, or move the robot.
