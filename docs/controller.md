# Controller / operations

The service publishes a latest-command mailbox to a native 1 kHz loop. Pose and
joint-position commands use the production controller and existing SDK torque
filter/rate limiting. Native code does not call Python in the servo callback;
mailbox/state access is nonblocking. Configuration and limits are defined by
`models.py`, `native/controller.hpp` and the reviewed `configs/` profile.

`CompliantClient` sends base-frame metres / xyzw quaternions and finite command
leases. **The first valid setpoint may start control**; absence of an explicit arm
request does not make a client read-only. Use one producer, fresh observations,
sequence/epoch checks and confirmed stop. No experiment should recover/retry or
relax a guard automatically. A software stop is not an emergency stop.

Before hardware use: fresh stationary error-free Idle; matching robot/URDF,
TCP/load/configuration; reviewed full arm/tool/cable path; exclusive producer;
physical stop available; exact trajectory screen; no competing training workers.
Keep saved home/joint references immutable. Matching TCP does not imply restored
redundant posture. A simulation-only reference context is not a deployment file.

Establish workspace clearance and operating limits for the actual installation.
Example profiles, identification speeds and simulation force thresholds are not
manufacturer limits or authorization for hardware motion or human contact.

## Latched software motion inhibit

| Endpoint | Effect |
|---|---|
| `POST /motion_inhibit?wait_s=0` | Assert the global latch and request stop; optional wait up to 5 s |
| `GET /motion_inhibit` | Read latch state, generation and current `release_token` |
| `POST /motion_inhibit/release` | Body `{"release_token":"…"}`; release only when control/lifecycle work is inactive |

`CompliantClient.inhibit_motion()`, `get_motion_inhibit()` and
`release_motion_inhibit(token)` mirror these routes. Assertion reaches a native
atomic gate without waiting for a publisher/startup lock. It blocks targets,
standalone arming, homing, contact relief, joint-limit backoff and recovery. `/stop` and configuration changes do not
release it. Reassertion invalidates older release tokens; release clears queued
references and **never resumes motion**. A fresh request is required afterward;
requests already admitted before/during the latch cycle are rejected, not replayed.
`/state` and `/health` expose `motion_inhibited` separately from legacy `stop_latched`.

This is a **process-local software interlock, not a safety-rated emergency stop**.
It does not survive backend/service restart (old-instance release tokens are invalid).
Network delivery and robot deceleration are not instantaneous; an SDK recovery call
already in progress cannot be cancelled. Inspect `stopped`, not just `stop_requested`;
confirmation is conservative during lifecycle work or without a fresh post-request
robot sample. Release does not clear faults or authorize another cell's motion.
Pause producers before release if you don't want their next fresh request to start.

Routes use the service's existing authentication policy; an unauthenticated server
also permits unauthenticated release. Protect the network / enable authentication
as appropriate. New Python with an old native backend fails motion requests closed;
activation requires a reviewed inactive native update, not overwriting a loaded module.

## Bounded contact-relief API

`POST /relieve_contact` requests one finite **native** Cartesian action, not a
Python servo or pose setpoint. The default is **manual force-following**:

```python
# Only after clearing the local retreat/braking area, with operator supervision.
ack = client.relieve_contact(workspace_confirmed_clear=True)
# Supervise motion and call client.stop() when desired; acceptance is not release.
```

The sole required acknowledgment, `workspace_confirmed_clear=true`, means an
**operator-supervised cleared local retreat/braking area**, including the arm,
tool, payload and cables. Use the ordinary preflight above, with the physical stop
available. **Never invoke this on a real jam without prior validation and an
operator at the robot. Slow is not low force.** Raw estimator bias can cause drift
or motion in the wrong direction. This is not an automatic unjamming procedure,
a collision planner, or a guarantee of clearance or low contact force.

Requires a writable, inactive **Disarmed** controller, motion inhibit released,
and **automatic recovery explicitly OFF**. Neither endpoint nor client disables
recovery for you, recovers a fault, captures/zeros a bias, or retries motion.
Check `contact_relief_available` in `/state` or `/health`: missing native
`manual_stop` binding or required state fields rejects relief with **503**, not a
fallback to ordinary motion. Use a matching native build; software validation
is **not hardware qualification**. For relief, start this service with
`./start.sh --no-auto-recover` or explicitly disable automatic recovery before
requesting the action.

### Manual mode and bounds

`mode="manual"` follows the **raw canonical SDK wrench without bias subtraction**.
It has no automatic low-load release detection: even a zero wrench does not
complete the action. Stop explicitly with `client.stop()` / `POST /stop` (or assert
motion inhibit); timeout, bounds and hard guards can also stop it. Active status is
`following_wrench`; a user stop reports `stopped`, never
`contact_relief_released=true`. Bias vectors, joint record and provenance default
to `None`; `bias_confirmed_free` defaults to `False`. Manual requests reject
non-null bias/provenance or a true free-space flag. Native default zero bias fields
are **unused, not a zero calibration**.

`workspace_lower` / `workspace_upper` are optional base-frame XYZ corners. Provide
both, strictly ordered on every axis, or neither. Omitting them in manual mode
means **no absolute TCP box**, not unlimited travel: local measured/reference-path
limits remain at most **5 cm / 10 degrees**, and joint displacement is limited to
**0.2 rad per joint from action start**. Joint/robot limits still apply. An explicit
box adds a TCP bound; neither it nor the local bounds guarantees full-arm clearance.
Limits are abort triggers, not physical barriers; allow for braking/overshoot.

The timeout defaults to **10 s**, must be **>0.2 and <=10 s**, and runs from HTTP
request receipt, including parsing/queueing. It is independent of the streaming
watchdog and cannot be renewed by repeating the request. Default speed caps are
**0.005 m/s / 1 degree/s** (`rotation_speed` is radians/s); schema maxima remain
0.02 m/s / 0.1 rad/s. All numeric gains/caps remain finite and bounded; `/docs` and
`/about?format=json` list them. `upward_velocity` defaults to zero and permits at
most 0.005 m/s in base +Z. Its component opposing the estimated environment push
is removed before the speed cap; this does not correct a biased/wrong estimate.

### Wrench convention and unchanged native guards

Wrenches are `[Fx,Fy,Fz,Mx,My,Mz]` in N/Nm, **SDK ROBOT-ON-ENVIRONMENT sign**, in
base axes, with moments **about the configured TCP**. `external_wrench_tcp_base`
is nullable; check `external_wrench_tcp_valid`. The native equilibrium follows
**minus** that SDK wrench (bias-subtracted only in automatic mode), with the fixed
50 ms time-constant filter: `v = clip(-A * filtered_wrench + upward_drift)`; the
same sign reversal applies to angular velocity.
`contact_relief_filtered_wrench_tcp_base` retains SDK sign.

Invalid frame/point data is rejected natively. Hardware derives the wrench from
the full reported stiffness-frame transform and verifies that the reported EE
matches the configured TCP, even before loaded startup. Relief additionally
requires `contact_relief_setup_valid`: EE/TCP, identity EE-to-stiffness transform,
declared expected end-effector model, and additional load mass/COM/inertia must
already match configured settings. It makes **no SDK frame, load or
collision-setting changes under contact**. An unconfigured/mismatched setup is
rejected, not repaired or calibrated using a contact sample.

Native captures the measured pose at the first accepted callback and disables
friction compensation and posture spring. Strict singularity, raw hard wrench,
external-joint-torque, measured-motion, reference-lead/path, workspace,
wrench-growth and timeout guards remain authoritative. Only a validated initial
relief command gets the startup soft-gate exception; there is no automatic recovery.
Hard norm stops use **un-debiased, unfiltered** canonical readings and the stricter
of configured thresholds or **40 N / 20 Nm**, plus existing external-joint-torque
limits. Conditioning floor: `max(minimum_singular_value, 0.05)`. Measured TCP speed
trips at 1.5 times requested speed caps and joint speed at 0.15 rad/s, additionally
to ordinary native limits. A target lead over 5 mm / 2 degrees aborts. A filtered
load rise above the best observed load by 2 N or 0.3 Nm for 50 ms aborts. No
threshold is automatically increased.

Action identity, options and deadline are fixed before the first callback;
delayed startup cannot fall back to an ordinary stream watchdog. Braking is not
instantaneous: normal SDK stop handling and post-stop stationary-Idle confirmation
remain required and may extend beyond the active timeout. The stationary
joint-speed ceiling is the stricter of configured stop velocity or 0.02 rad/s.
Fresh stop observations recheck frame/setup validity, valid OSC dynamics
(including a symmetric positive-definite mass matrix), conditioning, hard load
and workspace/travel limits. Controller/toy and hardware stop paths use the same
final-result predicate.

### Automatic release is opt-in

`mode="automatic"` retains bias-subtracted release detection. It requires an
explicit clear TCP box, all six `bias_wrench_tcp_base` values, seven
`bias_joint_positions`, `bias_confirmed_free=true`, and nonblank `bias_source`
provenance (maximum 256 characters, logged with acceptance). These must come from
an independently verified free-space record, never the current contact sample:

```python
ack = client.relieve_contact(
    mode="automatic",
    bias_wrench_tcp_base=free_bias_tcp_base,  # verified record: SDK sign, about TCP
    bias_joint_positions=free_bias_joints,
    bias_source=verified_record_id,
    bias_confirmed_free=True,
    workspace_lower=clear_tcp_lower, workspace_upper=clear_tcp_upper,
    workspace_confirmed_clear=True,
)
```

The names above are placeholders for verified records/reviewed bounds, not
hardware instructions. Do not negate the bias or substitute a flange/stiffness
moment. Tool/payload/frame/estimator conditions must match; posture proximity is
not proof of compatibility or freshness. Joint differences from the bias posture
must remain at most **0.2 rad per joint**, initially and throughout automatic relief.

Default release thresholds are 3 N / 1 Nm. Release requires low estimated residuals
for 200 ms **and stationary motion**, then disarm. The low-load hold freezes the
last bounded equilibrium without upward drift or a jump to measured pose; lead
and other guards still apply. A handoff losing its low-residual condition aborts
rather than reporting completion. **Low estimated residual is not proof the
object is unjammed or safe to move.** Automatic release detection is distinct from
automatic recovery, which must still be OFF.

### Acceptance and native state

HTTP **202 means accepted, not released**. Response/latest target records include
`action="relieve_contact"`, `relief_options.mode`, `bias_source` (null in manual),
`automatic_release_detection` (false in manual), epoch/command sequence, deadline,
`accepted_not_released=true` and `disarm_on_completion=true`. Backend command
`mode` stays `cartesian`; no requested pose is supplied. `/state` and `/health`
expose `contact_relief_mode` (`none` / `manual` / `automatic`),
`contact_relief_bias_used`, `contact_relief_active`, `contact_relief_status`,
`contact_relief_released`, filtered wrench, translation/rotation, deadline and
reference travel/rotation. These are native observations, not Python completion
guesses. While active, normal targets, homing, joint-limit backoff, a second relief
request and enabling automatic recovery are rejected. Stop/inhibit remain available; do not
automatically retry after rejection or abort.

For offline relief lifecycle tests, the bridge provides
`OfflineController(config, sample, epoch, now, initial_relief_command, deadline)`
to preserve the immutable startup action/deadline. The four-argument constructor
remains the ordinary streaming-controller interface.

## Joint-limit backoff API

`POST /backoff_joint_limits` requests one finite **native joint-space action**.
It is a general API, **not a one-off current-state recovery**, and is not hardware
qualified. Use a matching native build and check `joint_limit_backoff_available`
in `/state` or `/health`; missing native options, command bindings or raw state
fields return **503**, with safe unavailable defaults on reads.

```python
# Only after reviewing and clearing the full arm/tool/cable retreat/braking path.
ack = client.backoff_joint_limits(confirm_clear_path=True)
# Acceptance is not completion. Supervise; stop/inhibit remain available.
```

The sole required input is the **strict boolean** `confirm_clear_path=True`.
No joint list, current pose, targets, velocities or feedforward are accepted.
Confirmation is an operator assertion, not collision planning or clearance proof.
Use the ordinary preflight and physical stop availability described above.

| Option | Default | Allowed finite values |
|---|---:|---|
| `trigger_margin` | 0.05 rad | >0 and <=0.15 rad |
| `backoff_distance` | 0.1 rad | >=0.02 and <=0.2 rad, with the tolerance-clearance check below |
| `speed` | 0.03 rad/s | >0 and <=0.05 rad/s |
| `torque_scale` | 1.0 | >0 and <=1 |
| `timeout_s` | 20 s | >0.2 and <=30 s |

Target tolerance is `min(.02, .2 * backoff_distance)` rad. The schema requires
**`backoff_distance - min(.02, .2 * backoff_distance) > trigger_margin`**, so the
whole target-tolerance band clears the trigger margin. Merely choosing distance
larger than the margin is insufficient.

The deadline starts at **HTTP receipt**, including body parsing and queueing,
independently of the streaming watchdog. Ingress motion-generation checks reject
requests spanning an inhibit/release cycle. Native also refuses an expired
pre-control deadline **before entering SDK control**. Repeating the request does
not renew it. The client validates and sends once; it does not retry a lost acknowledgment
and retains its ordinary stop-on-close protection after sending motion.

### Native selection, limits and fault eligibility

Only native code selects joints from a **fresh** state near or beyond configured
effective limits. For an upper-side selection its target is
`min(start - backoff_distance, upper - margin - backoff_distance)`; for a lower-side
selection it is
`max(start + backoff_distance, lower + margin + backoff_distance)`, where `margin`
is the configured joint-limit margin. Selected-joint **references are strictly
inward**, not a guarantee of strictly inward physical motion or torque; braking
can apply torque opposing inward velocity. Unselected targets are captured
measured joint positions, not a saved home or a Python pose read.
**No joints selected: 409 before arming, without control.**

A selected joint may start at most **0.05 rad beyond a configured effective limit**,
but only while **strictly inside the FR3 specified position range**. This narrow
native exception does not widen global limits: the opposite position bound,
unselected-joint bounds, torque and collision guards remain in force. The scoped
selected-joint outward velocity exception is described below. Specified-range
violations and SDK/firmware refusals **cannot be bypassed**. No SDK setters,
firmware bypass, frame/load/collision changes or SDK recovery are used.

Starts require writable inactive **Disarmed**, or an eligible **LOCAL software
fault** from the native allowlist **JointLimit, Overspeed, BackoffTimeout,
BackoffStalled only**, with fresh, error-free, stationary **Idle** and no SDK fault
diagnostics. Eligibility is entirely native;
Python never clears a fault or calls recovery to make a request eligible. Unlike
contact relief, **automatic recovery need not be OFF**: the existing global policy
may remain true, but this action always passes `automatic_recovery=False` to native
startup and does not change that policy. Native blocks automatic recovery after a
backoff fault; no automatic retry or target replay follows an abort.

Native generates a slow, **strictly inward shaped reference**, with acceleration
`min(config, .05)` rad/s² and jerk `min(config, .5)` rad/s³, and suppresses friction
compensation. Selected-joint active torque caps are **2 Nm on J1–4 / 1 Nm on J5–7**,
times `torque_scale`, further bounded by configuration. **Other joints retain
normal hold-feedback budgets unchanged**; these selected caps are not global
actuator or contact-force limits. Scoped action guards include:

- measured-speed abort threshold **0.08 rad/s** (margin below 0.1 rad/s), with
  other applicable stricter limits retained except for the scoped replacement below;
- for **selected joints only**, a measured outward encoder/velocity allowance of
  up to **0.01 rad/s replaces the ordinary outward directional bound**, which may
  be zero near the specified position limit. It is **not** the minimum of 0.01
  and that ordinary outward bound. The selected joint's **inward directional
  bound** and **all unselected-joint directional bounds remain in force**;
- maximum **0.001 rad worse-than-start position**, **0.005 rad backtracking**,
  **0.01 rad held-joint drift** and **0.25 rad travel**. The progress high-water
  mark used to detect backtracking continues to update through stop observations;
- no-progress abort after **2 s** if reference progress is **>=0.02 rad** but
  **current actual measured progress** is **<0.002 rad**, not historical peak progress;
- start-mismatch, timeout and existing hard-guard aborts.

These are **monitored action guards, not physical barriers or guarantees of
strictly inward physical motion or torque**. Stop/braking is not instantaneous.
Global safety configuration and SDK protection remain unchanged; no SDK setters
are used.

### Acceptance, exclusivity and diagnostics

HTTP **202 means accepted, not reached**. Response/latest-command metadata has
`action="backoff_joint_limits"`, `mode="joint"`, echoed `backoff_options`, epoch,
command sequence, receipt-based deadline, `accepted_not_reached=true` and
`disarm_on_completion=true`. Available native plan/status fields are echoed as a
snapshot; `/state` and `/health` are authoritative for subsequent progress:

- `joint_limit_backoff_active`, `joint_limit_backoff_status`,
  `joint_limit_backoff_reached`;
- `joint_limit_backoff_selected` (**1-based joint indices**);
- `joint_limit_backoff_start_positions`, `joint_limit_backoff_target_positions`;
- `joint_limit_backoff_deadline_s`, `joint_limit_backoff_previous_reason`,
  `joint_limit_backoff_torque_limits`.

Missing telemetry is unavailable, not success: active/reached default false,
selected defaults to an empty list, and unavailable plans/deadlines/torque limits
are null. `/target_joint_state` does not fabricate caller positions/velocities for
this action; use the separately named native plan. Python neither estimates
completion nor guesses affected joints. During active backoff, normal targets,
homing, contact relief, a second backoff and enabling automatic recovery are
rejected; turning recovery **OFF** remains allowed. Backoff likewise cannot
supersede a running action. `mode_switching_enabled` reports a **configuration
capability only**: even if true, **no mode switching is permitted during this
exclusive action**. Stop/inhibit remain available.

A secondary disqualifying fault escalates the native summary `reason`, while
`stop_trigger` retains the first trigger for diagnostics. A retained first local
trigger therefore does not establish eligibility for another backoff. Python
forwards native diagnostics and never substitutes its own eligibility decision.

See [joint-limit recovery guidance](joint-limit-recovery.md) for the distinction
between configured limits, the specified position range and SDK refusal.
Nothing here authorizes recovery without the normal hardware preflight.

## Optional friction compensation

The production controller (hardware **and** native simulation) accepts the
settings below. Generic `Config()` / `ControllerConfig()` defaults remain off.
The **policy startup profile** `configs/controller.fr3.auto.json`, selected by
`start.sh`, explicitly enables **f=1.0, epsilon=0.02, filter time=0.035 s**.
It also uses translation stiffness **[600,600,600]**, translation damping
**[48.99,48.99,48.99]**, and commanded moment limit **6 Nm**. Other profiles and
frozen simulation/reference captures are unchanged. Starting the service remains
disarmed, but the first valid setpoint may start this enabled law. These requested
policy settings are not a declaration that the whole workspace/speed range is
qualified; use the same filter contract in the policy simulator.

| Setting | `friction_compensation_torque` (f, Nm/joint) | `friction_compensation_epsilon` (rad/s) |
|---|---:|---:|
| Off (generic default) | 0.0 | 0.05 |
| Example low setting | 0.3 | 0.05 |
| Example intermediate setting | 0.6 | 0.03 |
| Policy startup setting | 1.0 | 0.02 |

The additive joint torque is `tau_fc[i] = f * tanh(dq_plan[i] / epsilon)`.
The horizon is fixed at **0.100 s**; neither measured joint velocity nor a finite
difference of successive commands determines the direction:

- Joint target: `dq_plan = (q_target - q_measured) / 0.100`.
- Cartesian target: form base-frame translation and shortest-arc rotation error
  `e` against the measured TCP. With `S = diag(1,1,1,s,s,s)`,
  `s = jacobian_rotation_scale`, `Js = S*J`, and
  `lambda = minimum_singular_value`, use
  `dq_plan = Js.T * solve(Js*Js.T + lambda^2*I, S*e / 0.100)`.
  This is regularized geometric inverse kinematics, with no posture term.

These are **accepted target** errors, not the Ruckig-shaped reference errors.
`dq_plan` is only a direction/smoothing signal, not a joint-velocity command or a
promise to reach the target in 100 ms. This law produces breakaway torque even
when stationary. It also increases small-error restoring stiffness; it is not a
measured friction estimate and does not guarantee a 1–2 mm tracking error.

Compensation is disabled while armed without a target, stopping, faulted, or
contact-yielding. Legacy contact-yielding mode also applies its `drive_scale`.
At `f=0` the original torque law runs unchanged, without an extra inverse solve.
`f` must be finite, nonnegative and no larger than any configured actuator torque
limit; epsilon must be finite and strictly positive.

### Compensation-only low-pass filter

`friction_compensation_filter_time_s` selects a first-order time constant in
seconds. **0 (default) preserves the unfiltered law.** For a positive time
constant `T`, each native callback computes
`alpha = 1 - exp(-dt/T)` and `filtered += alpha * (raw_tau_fc - filtered)`.
Only this compensation term is filtered; targets, impedance feedback, damping,
and final actuator/slew limits are unchanged. The same code runs in simulation.

For example, **T=0.035 s** reaches 95% of a held torque change in **105 ms**, and
99% in **161 ms**. These are torque-filter response times, not TCP settling
promises. The steady-state formula is unchanged, but this is measured-error
restoring feedback: the added lag can reduce damping and cause oscillation.
A longer time constant is not automatically safer. Validate tracking, reversals,
and held-target stability at the intended speed, posture and control mode; a
Cartesian test does not qualify joint-mode use.

The filter starts at zero and resets on a new session, mode switch, yielding,
stop or fault. Ordinary accepted same-mode target updates do **not** restart it.
Legacy contact taper is applied after filtering so a falling `drive_scale` acts
immediately. Stop/fault/yield suppression has no filter tail; final actuator slew
and robot braking still apply. `friction_compensation_torques` reports the
filtered, tapered request before final actuator limits. The filter cannot
accumulate beyond the +/-f input bound; it is not an integrator.

```python
# Inactive controller only; example tuning, not hardware qualification.
client.configure(friction_compensation_torque=1.0,
                 friction_compensation_epsilon=0.02,
                 friction_compensation_filter_time_s=0.035)
```

The extra term is added **after task/joint feedback budgets**, but **before the
unchanged final actuator torque and torque-slew limits** and SDK filtering.
Thus `commanded_force_limit`, `commanded_moment_limit` and
`joint_feedback_torque_limits` do not bound this extra feedforward term. A
nonzero setting changes the total effort/compliance contract; don't interpret
feedback caps as total applied torque or contact-force bounds. Existing contact,
velocity, joint-position and watchdog stops remain authoritative.

Configure only while inactive, using the existing configuration route/client
(e.g. `client.configure(friction_compensation_torque=0.3,
friction_compensation_epsilon=0.05)`). Only the policy startup profile enables it
by default; use `friction_compensation_torque=0` to disable it. State fields
`planned_joint_velocities` and `friction_compensation_torques` expose the current
signal/request **before** final actuator/slew limits, not applied compensation.
The offline Output exposes the same vectors in singular form. The packed native
telemetry schema is unchanged; retain configuration alongside raw traces.

With an older installed `_native`, Python accepts the default-off configuration
without setting the unavailable friction fields, so ordinary service startup is
preserved. A nonzero compensation request fails with an explicit rebuild-required
error rather than being silently ignored. This compatibility does not enable the
feature on the old binary or permit mixing old/new native modules. Similarly,
a module with friction support but no filter support accepts the unfiltered
setting (`filter_time_s=0`), but rejects enabled filtered compensation with a
clear rebuild-required error. A filter setting while `f=0` is inert.

Rebuild matching `_native` and `_twin_native` modules together for the changed
Config/Output layout. `FCC_NATIVE_OUTPUT_DIRECTORY` stages both modules and their
source manifest without touching installed binaries. Never mix old/new modules
or overwrite a library loaded by a service. Enabling this on hardware requires
an inactive reviewed deployment and the normal preflight above. Run the tracking
test one setting at a time, starting off; stop immediately on buzzing/oscillation,
and do not automatically escalate, recover, or retry. An operator at the robot
must monitor the hold and have the physical stop available.

## Interpreting acceleration/jerk screens

The profile's `joint_acceleration=10` and `joint_jerk=5000` come from the FR3 SDK
motion-command limits (`kMaxJointAcceleration`, `kMaxJointJerk`, with an SDK safety
margin). They constrain generated joint references where that generator is used.
The bounded-equilibrium controller instead submits **torques**; its SDK path
filters torque and rate-limits torque commands, while native measured-speed,
joint-position, contact-policy and watchdog checks remain separate.

Some offline experiment helpers also compare finite differences of **simulated
measured velocity** against those reference ceilings. That is an additional
conservative experiment screen, not a calibrated prediction of a torque-mode
hardware fault. Its interpretation depends on the model, sample period and signal
being differentiated. An exceedance is not by itself proof of a physical limit
violation; a pass is not hardware safety qualification. Keep commanded-reference
limits, torque/slew limits, measured-motion guards and model diagnostics distinct.

## Native telemetry

Prepare a finite capture while inactive via `/telemetry/prepare`. The preallocated
recorder retains earliest callbacks, counts overflow and does no allocation,
blocking, file or network work in append. Export `/telemetry/data` only after stop,
then clear the matching capture ID explicitly. Up to 120,000 records × 992 bytes.
Schema/flags are in `native_telemetry.py`; status includes native speed-guard trips.

State at cycle n, newly computed pre-SDK torque n and robot-reported prior desired
torque are different stages. Raw measured torque is another channel. Preserve
those distinctions. Coherent sample/server timestamps support causal clock bounds;
old noncoherent HTTP snapshots do not provide complete 1 kHz history.

Check schedule completion, confirmed stop, native envelope/trip and gaps/drops.
A gapless recording of a failed prefix is still a failed experiment. Archive raw
records and declare fitting/validation roles before collection.

## Deployment

The HTTP service's toy `sim` backend and the MuJoCo/offline SDK bridge do not
require a robot. Physical support needs an explicit SDK-backed `_native` build.
Stage and test changes, stop the inactive service, back up configuration/binaries,
then perform a reviewed restart with recovery off. **Never overwrite a loaded
native library.** No build or installation command should implicitly deploy,
start motion, change gains or recover a fault.
