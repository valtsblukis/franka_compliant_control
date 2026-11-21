"""Composable offline simulation stack.

Import the level you need; this package does not eagerly import SDK/Gym modules:
  world.SimulationWorld              physics and scene components
  native.NativeControllerSimulation  production controller and optional sensor
  obstacles.DynamicObstacles         optional movable obstacle lifecycle
  geometry.RobotCollisionGeometry    optional contact/self-collision proxies
  disturbances.JointTorqueDisturbances  optional external torque pulses
  contacts.ContactAccumulator        solver-truth metrics, independent of reward

No module in this package opens a robot device or sends HTTP commands.
"""
