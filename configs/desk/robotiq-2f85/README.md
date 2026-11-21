# Robotiq 2F-85 nominal Desk import

This gripper-only example uses nominal settings for a single standard-mounted
Robotiq 2F-85. It excludes cameras, mounts, additional cabling and payloads.
Independently verify the actual assembly before hardware use. These are initial
commissioning values, not a calibrated model or a deployment record.

## Sources and selection

The complete dynamics tuple (mass, flange-relative COM and COM inertia) comes
from [Franka Robotics' Robotiq Desk example](https://github.com/frankarobotics/external_gripper_example/blob/master/panda_with_robotiq_gripper_example/config/endeffector-config.json),
with [deployment instructions](https://github.com/frankarobotics/external_gripper_example/blob/master/README.md)
that describe importing those mechanical settings into Desk. That example
was tested on Panda with libfranka 0.5.0; it is a manufacturer-provided nominal
tool baseline, not evidence of FR3 5.10 import acceptance or physical validation.
The tool properties are relative to the flange and independent of arm kinematics.

| Property | Selected value |
| --- | --- |
| Mass | 0.900 kg |
| COM relative to flange | [0, 0, 0.057] m |
| COM inertia diagonal | [0.002768, 0.003149, 0.000564] kg m² |
| Off-diagonal inertia | zero |
| Defined TCP relative to flange | [0, 0, 0.174] m, axes parallel to flange |

The dynamics tuple is preserved together. The later Robotiq manual's alternate
0.921 kg / 60 mm tuple and its inertia table remain recorded separately in
`configs/robotiq-2f85.nominal.json`; they are **not** combined with this inertia.
The manufacturer's example supplies a complete nominal dynamics tuple; it does
not establish the exact properties of a particular installation.

The one mechanical change from Franka's example is the nominal TCP: the example
uses the flange origin (identity transform). Here we define the tool point at
174 mm along flange Z using [Robotiq's manual, section 5.2.3](https://blog.robotiq.com/hubfs/support-files/2F-85_2F-140_General_PDF_20210623.pdf).
Changing the chosen TCP does not change the flange-relative mass properties.
The gripper opening/contact location can change the effective grasp point.

Assumptions: standard mount matching the example's flange axes, standard
fingertips, no camera/sensor/spacer/tool changer, no carried object. This package
only supplies mechanical defaults. Serial actuation is configured separately.
It supplies no geometry, Watchman safety envelope, or gripper driver.

## Packaging and import

From the project root:

```bash
.venv/bin/python scripts/package_desk_tool.py
```

This creates:

- `artifacts/desk/robotiq-2f85-nominal.tar.gz`: custom device archive. Upload it
  using the add/import device action on Desk's end-effector settings page and
  select `Robotiq 2F-85 (nominal)`.
- `artifacts/desk/robotiq-2f85-parameters.json`: the same mechanical values
  without device metadata, for an existing tool's mechanical-settings importer
  if that UI accepts JSON instead of a device archive.

The archive follows [Franka's documented commissioning format](https://download.franka.de/ride-cli-doc-v1.1.1/end_effector_commissioning.html):
gzip-compressed tar with `endeffector-config.json` at its root. Optional mesh,
URDF and icon files are omitted. Import compatibility and mechanical values
must be verified for the actual robot and software version.

After import, confirm the selected tool shows the values above. Then read back
`m_ee`, `F_x_Cee`, `I_ee`, `F_T_NE`, `NE_T_EE`, `F_T_EE`, `m_load`, and
`EE_T_K` through FCI before enabling powered control. Existing runtime frame
offsets can survive a device change. Additional carried load should be zero for
the empty tool; do not add the gripper mass again through `setLoad`.

Stop the existing service and all producers before commissioning. Importing a
file is not a request to arm or recover the robot.
