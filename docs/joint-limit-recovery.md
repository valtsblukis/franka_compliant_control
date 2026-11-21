# FR3 joint limits and recovery

This guide distinguishes controller limits, SDK velocity envelopes and supported
recovery interfaces. It is not an assessment of a particular robot, a deployment
record, or hardware qualification. Obtain fresh state and follow the normal
[hardware preflight](controller.md) before considering any action.

## Position limits and directional velocity bounds

A configured effective joint limit, a robot-exported specified position limit,
and a directional velocity envelope are different constraints. A measured-speed
fault near a position boundary does not by itself identify a firmware
joint-position reflex. Preserve the original fault diagnostics; cleared or stale
records cannot establish current robot state.

The deprecated hardcoded libfranka helper uses this J7 outward-velocity formula:

```
min(5.26, max(0, sqrt(max(0, 34*(3.0196-q7))) - 0.35)) - 0.001
```

Its inner clamped term becomes zero at approximately **3.015997 rad**; the final
result subtracts a tolerance. This is a directional velocity-envelope boundary,
not a universal joint-position limit. Inward velocity has a separate bound.

libfranka 0.18 deprecates that helper and uses parameters from the robot's URDF
through `Robot::getUpperJointVelocityLimits` / `getLowerJointVelocityLimits`.
Its 0.18 changelog notes increased FCI joint-position limits with System >=5.9.
Use the version-matched SDK and exported model rather than substituting a legacy
formula for the active constraint. The URDF does not establish mechanical stops
or all configured safety functions.

Sources:
- `libfranka/include/franka/rate_limiting.h`
- `libfranka/src/joint_velocity_limits.cpp`
- libfranka 0.18 changelog
- `native/fr3_limits.hpp`
- [Official robot specifications](https://frankarobotics.github.io/docs/robot_specifications.html)

## Bounded inward-reference backoff

`POST /backoff_joint_limits` requests one finite native joint-space action.
Native code selects qualifying joints from fresh state; the caller does not
provide joints, targets or a saved pose:

```python
# Only after reviewing the entire arm/tool/cable retreat and braking path.
ack = client.backoff_joint_limits(confirm_clear_path=True)
```

See the [full API contract](controller.md#joint-limit-backoff-api) for options,
fault eligibility, torque/speed bounds, scoped position/velocity exceptions,
deadlines, diagnostics and exclusivity rules. Check `joint_limit_backoff_available`
in `/state` or `/health`; missing native capabilities reject the request with 503.
HTTP 202 means **accepted, not reached**. No selected joints returns 409 before
control. Supervise the action and use confirmed stop; do not retry automatically.

Strictly inward describes the shaped **references**, not guaranteed inward
physical motion or torque. Braking can oppose inward velocity. The bounds are
monitored guards, not physical barriers or a guarantee of low contact force.
Global configuration and SDK protection remain authoritative. Specified-range
violations or SDK/firmware refusal cannot be bypassed by this endpoint.

Startup requires inactive writable Disarmed control or a native-eligible local
software fault, with fresh stationary error-free Idle and no SDK fault diagnostics.
The action does not clear a fault, invoke SDK recovery, or change SDK settings.
The global automatic-recovery policy need not be OFF, but this action always
bypasses automatic recovery without changing that policy. Matching software and
successful offline tests do not qualify a hardware installation.

libfranka 0.18 torque-only control requests `kExternalController` with motion mode
`kNone`; firmware still decides whether Move is allowed. Source inspection alone
cannot establish whether a particular loaded or limited start will succeed.

## Supported Desk / SDK recovery options

The documented alternatives are not limited to hand-guiding:

- System 5.9/5.10 Desk provides individual joint jogging in Test & Jog, including
  moving out of soft joint limits. The attached physical **X4 external enabling
  device** must be held while operating the UI control.
- For the specific "A joint limit has been violated" safety error, the operating
  manual describes a safety-operator **Move Joint Out Of Limit** dialog, again
  used with the external enabling device.
- Being at the E-stop does not establish that X4 is available or held. Do not
  bypass enabling devices or safety functions.
- System 5.7.2 release notes describe a fix for an FCI joint-position-limit error
  that was not recoverable with `robot.automaticErrorRecovery()`. This does not
  establish that recovery physically repositions a joint or applies to every fault.
- Desk configuration/administration and clearing safety violations are not proof
  of an unattended joint-limit repositioning API. Joint-limit violations and lost
  joint-position/reference calibration require different recovery procedures.

If FCI is refused and the necessary operator-assisted workflow is unavailable,
qualified physical assistance or vendor guidance may be required.

References:
- [Operating Manual 5.9, sections 5.3.2 and 9.1.4 (PDF pp.22–23, 58–59)](https://www.franka.de/hubfs/Operating%20Manual%20Franka%20Research%203_5.9.0_5.9.1_R02216_1.2.3_EN.pdf)
- [System update history, 5.7.2 and 5.8.0 entries](https://download.franka.de/Franka_Robotics_Update_History.pdf)
- libfranka 0.18 `src/control_loop.cpp`, torque-only constructor

## Read-only observation

`scripts/read_robot.cpp` records SDK state, current and last-motion errors, and
SDK velocity bounds at measured and desired joint positions. It has no control,
recovery, setter or gripper call. Use it only with exclusive connection ownership
after stopping the existing service; never open a competing Robot connection.
Do not automatically restart the service, change limits, or enable a new buffer
as part of diagnosis.
