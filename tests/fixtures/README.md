# Offline FR3 twin fixture

`fr3-twin.urdf` is the rigid-body portion of the robot-exported FR3 URDF captured
through the existing service on 2026-09-23. Visual/collision mesh references and
massless accelerometer branches were removed for a small, self-contained test
fixture. Arm joint transforms, limits and inertias are unchanged. It contains no
runtime tool/load configuration and no saved home command.

The generated source identifies `franka_description/robots/fr3/fr3.urdf.xacro`.
Franka's description package is Apache-2.0 licensed; the license is retained in
`FRANKA_LICENSE`. Upstream: https://github.com/frankarobotics/franka_description

Runtime experiments use their own captured, fingerprinted robot URDF and home,
not this fixture. Passing model-conversion tests is not physical identification.
