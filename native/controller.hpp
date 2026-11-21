#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <string>
#include <ruckig/ruckig.hpp>

namespace fcc {
using V3 = Eigen::Vector3d;
using V6 = Eigen::Matrix<double, 6, 1>;
using V7 = Eigen::Matrix<double, 7, 1>;
using M6 = Eigen::Matrix<double, 6, 6>;
using M7 = Eigen::Matrix<double, 7, 7>;
using J67 = Eigen::Matrix<double, 6, 7>;

enum class Mode { Cartesian, Joint };
enum class Phase { Disarmed, Armed, Active, Stopping, Fault };
enum class Reason {
  None, UserStop, Watchdog, ForceLimit, MomentLimit, JointContact,
  JointLimit, Singularity, Overspeed, InvalidState, Timing, StopTimeout,
  ReferenceError, Communication, RobotCartesianReflex, RobotJointReflex,
  RobotReflex, RobotControlError, HomeReached, HomeTimeout,
  ReliefReleased, ReliefTimeout, ReliefTravel, ReliefRotation, ReliefWorkspace,
  ReliefLoadGrowing, ReliefReferenceError, ReliefBiasMismatch, ReliefInvalidWrench,
  ReliefSetupMismatch, ReliefReleaseLost, ReliefJointTravel,
  BackoffReached, BackoffTimeout, BackoffStartChanged, BackoffOutward,
  BackoffHoldDrift, BackoffTravel, BackoffStalled, BackoffGoalLost, BackoffSetupMismatch
};
const char* phase_name(Phase);
const char* reason_name(Reason);
inline bool backoff_local_fault_allowed(Reason reason) {
  return reason==Reason::JointLimit || reason==Reason::Overspeed ||
         reason==Reason::BackoffTimeout || reason==Reason::BackoffStalled;
}

struct Config {
  explicit Config(const std::string& model = "panda");
  std::string robot_model;
  std::string compliance_mode = "contact_yielding";
  std::string motion_limit_policy = "configured"; // or FR3 position-dependent "franka"
  bool bounded_cartesian_spring = false;
  // OSC normalizes nominal acceleration dynamics; gains are s^-2 and s^-1.
  V3 osc_translation_kp = V3::Constant(25.0), osc_translation_kd = V3::Constant(10.0);
  V3 osc_rotation_kp = V3::Constant(25.0), osc_rotation_kd = V3::Constant(10.0);
  // Additional physical Cartesian feedback, independent of model inertia.
  // Zero preserves the original inertia-normalized law. Units: N/m, Ns/m,
  // Nm/rad and Nm*s/rad; not force/velocity limits or integral action.
  V3 osc_translation_stiffness = V3::Zero(), osc_translation_damping = V3::Zero();
  V3 osc_rotation_stiffness = V3::Zero(), osc_rotation_damping = V3::Zero();
  double osc_regularization = 0.02; // sqrt of scaled inverse task inertia
  std::string effort_limit_policy = "configured"; // franka: actuator only; task_bounded: progressive OSC
  std::string osc_feedback_mode = "linear"; // progressive/saturated: direct-equilibrium bounded springs
  double progressive_position_soft_radius = 0.01;
  double progressive_position_saturation_radius = 0.10;
  double progressive_rotation_soft_radius = 0.05235987755982989; // 3 deg
  double progressive_rotation_saturation_radius = 0.3490658503988659; // 20 deg
  double progressive_translation_soft_stiffness = 50.0; // N/m
  double progressive_rotation_soft_stiffness = 3.0; // Nm/rad
  // configured: SDK settings + native trips; robot: inherit; robot_configured: SDK only.
  std::string collision_policy = "configured";
  V3 translation_stiffness = V3::Constant(400.0);
  V3 rotation_stiffness = V3::Constant(15.0);
  V3 translation_damping = V3::Constant(40.0);
  V3 rotation_damping = V3::Constant(5.0);
  V7 joint_stiffness = V7::Constant(100.0);
  V7 joint_damping = V7::Constant(20.0);
  // Per-joint feedforward f*tanh(dq_plan/epsilon), with target error / 100 ms.
  // Off by default. Added outside feedback budgets but inside actuator/slew caps.
  double friction_compensation_torque = 0.0; // f, Nm (same on all seven joints)
  double friction_compensation_epsilon = 0.05; // rad/s, strictly positive
  // First-order time constant; 0 bypasses filtering (legacy law).
  double friction_compensation_filter_time_s = 0.0;
  // Used only by direct joint_impedance: bounds spring and combined feedback,
  // excluding Coriolis. Final actuator torque/slew limits remain separate.
  V7 joint_feedback_torque_limits = (V7()<<8,8,8,8,3,3,3).finished();
  V7 stop_damping = V7::Constant(12.0);
  V7 joint_lower = (V7() << -2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973).finished();
  V7 joint_upper = (V7() << 2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973).finished();
  V7 torque_limits = (V7() << 20,20,20,20,8,8,8).finished();
  V7 external_torque_limits = V7::Constant(8.0);
  double commanded_force_limit = 10.0;
  double commanded_moment_limit = 1.0;
  double contact_force_soft = 8.0;
  double contact_force_stop = 15.0;
  double contact_moment_soft = 0.8;
  double contact_moment_stop = 2.0;
  double translation_velocity = 0.10;
  double translation_acceleration = 0.5;
  double translation_jerk = 5.0;
  double rotation_velocity = 0.5;
  double rotation_acceleration = 1.0;
  double rotation_jerk = 10.0;
  double joint_velocity = 0.2;
  double joint_acceleration = 0.5;
  double joint_jerk = 5.0;
  // Zero preserves the legacy trip at 1.5 times reference speed. Explicit
  // limits let hand-driven motion be faster than the commanded return.
  double measured_translation_velocity_limit = 0.0;
  double measured_rotation_velocity_limit = 0.0;
  double measured_joint_velocity_limit = 0.0;
  // A positive entry overrides the shared limit for that joint. Zero falls
  // back to the shared/legacy setting, preserving existing configurations.
  V7 measured_joint_velocity_limits = V7::Zero();
  double position_error_limit = 0.03;
  double rotation_error_limit = 0.15;
  double joint_error_limit = 0.15;
  double posture_stiffness = 5.0;
  double posture_damping = 3.0;
  double nullspace_torque_limit = 2.0;
  double torque_rate_limit = 100.0; // Nm/s, using elapsed control period
  double command_timeout = 2.0;
  double stop_velocity = 0.02;
  double stop_settle_time = 0.1;
  double stop_timeout = 2.0;
  double joint_limit_margin = 0.05;
  double minimum_singular_value = 0.02;
  double jacobian_rotation_scale = 0.2; // metres: makes conditioning metric explicit
  double maximum_control_period = 0.01;
  double translation_speed_limit() const {
    return measured_translation_velocity_limit>0?measured_translation_velocity_limit:1.5*translation_velocity;
  }
  double rotation_speed_limit() const {
    return measured_rotation_velocity_limit>0?measured_rotation_velocity_limit:1.5*rotation_velocity;
  }
  double joint_speed_limit() const {
    return measured_joint_velocity_limit>0?measured_joint_velocity_limit:1.5*joint_velocity;
  }
  V7 joint_speed_limits() const {
    V7 limits;
    for(int i=0;i<7;i++) limits[i]=measured_joint_velocity_limits[i]>0?
        measured_joint_velocity_limits[i]:joint_speed_limit();
    return limits;
  }
  bool uses_osc_model() const {return compliance_mode=="operational_space" || compliance_mode=="auto";}
  void validate() const;
};

struct Sample {
  V7 q = V7::Zero(), dq = V7::Zero(), coriolis = V7::Zero();
  V7 previous_torque = V7::Zero(), external_torque = V7::Zero();
  V3 position = V3::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  J67 jacobian = J67::Zero();
  M7 mass = M7::Identity();
  V6 jacobian_dot_velocity = V6::Zero();
  bool dynamics_valid = false; // required by OSC; never assume identity on hardware
  V6 external_wrench = V6::Zero(); // Raw SDK base/spatial convention; preserve diagnostics.
  V6 external_wrench_stiffness = V6::Zero(), cartesian_collision = V6::Zero();
  // SDK robot-on-environment sign, base axes, about configured TCP. Directional
  // use requires explicit frame provenance, not the legacy raw-wrench label.
  V6 external_wrench_tcp_base = V6::Zero();
  bool wrench_tcp_valid = false;
  bool relief_setup_valid = false; // Configured EE/K/tool/load already match; no loaded setters.
  V7 joint_collision = V7::Zero();
  double robot_time = -1; // robot clock, not host monotonic time; unavailable in sim
  bool kinematics_valid = true; // false for raw post-fault observations without model evaluation
  double communication_success = 1.0;
  int robot_mode = -1; // libfranka mode, or unavailable for synthetic samples.
  bool robot_has_errors = false;
  bool finite() const;
};
// Identical policy/math for ordinary control and post-SDK stop observations.
Reason ordinary_hard_contact_reason(const Config&,const Sample&);

struct ContactReliefOptions {
  bool manual_stop=false; // Manual raw-wrench following: no bias or automatic release claim.
  V6 bias_wrench_tcp_base = V6::Zero();
  V7 bias_joint_positions = V7::Zero();
  V3 workspace_lower = V3::Zero(), workspace_upper = V3::Zero();
  bool bias_confirmed_free=false, workspace_confirmed_clear=false;
  double force_release=3., moment_release=1.;
  double max_translation=.05, max_rotation=.17453292519943295; // 10 deg
  double translation_speed=.02, rotation_speed=.1;
  double translation_admittance=.002, rotation_admittance=.05;
  double upward_velocity=0.;
  void validate(const Config&) const;
};
// Inactive preflight, also used by the first native callback. Never calibrates
// bias from contact and never weakens the ordinary arm path's unloaded gate.
void validate_contact_relief_start(const Config&, const Sample&, const ContactReliefOptions&);
Reason contact_relief_observation_reason(const Config&, const Sample&, const ContactReliefOptions&,
                                        const V3& anchor, const Eigen::Quaterniond& orientation,
                                        bool require_released=false, const V7* initial_joints=nullptr);

struct JointLimitBackoffOptions {
  bool confirm_clear_path=false;
  double trigger_margin=.05, backoff_distance=.1, speed=.03, torque_scale=1.;
  void validate(const Config&) const;
};
struct JointLimitBackoffPlan {
  V7 start=V7::Zero(), goal=V7::Zero(), direction=V7::Zero(); // +/-1 inward, 0 held
  bool prepared=false;
};

struct Command {
  std::uint64_t epoch = 0, sequence = 0;
  Mode mode = Mode::Cartesian;
  V3 position = V3::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  V6 twist = V6::Zero(), acceleration = V6::Zero(); // acceleration feedforward: progressive OSC only
  V7 joints = V7::Zero(), joint_velocity = V7::Zero(), posture = V7::Zero();
  bool has_orientation = false, has_posture = false;
  bool finish_at_target = false; // Reserved for the finite /go_home action.
  bool relieve_contact = false;
  ContactReliefOptions relief_options;
  bool backoff_joint_limits=false;
  JointLimitBackoffOptions backoff_options;
  JointLimitBackoffPlan backoff_plan; // Native-only fresh-state plan, never accepted from HTTP.
  double action_timeout = 30.0, completion_joint_tolerance = 0.02;
  double completion_position_tolerance = 0.01, completion_rotation_tolerance = 0.03490658503988659;
  double completion_settle_time = 0.2;
  double expires_at = 0.0; // host steady_clock; stamped only on accepted commands
  double accepted_at = 0.0;
};

struct Output {
  V7 torque = V7::Zero();
  V6 task_wrench = V6::Zero(), spring_wrench = V6::Zero();
  bool progressive_force_saturated=false, progressive_moment_saturated=false;
  V3 reference_position = V3::Zero();
  Eigen::Quaterniond reference_orientation = Eigen::Quaterniond::Identity();
  V7 reference_joints = V7::Zero(), reference_joint_velocity = V7::Zero();
  V7 joint_spring_torque = V7::Zero(), joint_feedback_torque = V7::Zero();
  // Requested compensation before final torque/slew limits, not applied torque.
  V7 planned_joint_velocity = V7::Zero(), friction_compensation_torque = V7::Zero();
  bool joint_spring_limited=false, joint_feedback_limited=false;
  V6 reference_twist = V6::Zero(), reference_acceleration = V6::Zero();
  Phase phase = Phase::Disarmed;
  Mode mode = Mode::Cartesian;
  Reason reason = Reason::None;
  Reason stop_trigger = Reason::None; // Preserve the first trigger across stop failures.
  bool finished = false, yielding = false, effort_limited = false;
  bool homing = false, home_reached = false, home_superseded = false;
  bool relief_mode=false, relief_manual=false, relief_settling=false, relief_released=false;
  V6 relief_filtered_wrench = V6::Zero(); // Canonical SDK sign; raw in manual, debiased in automatic.
  double relief_translation=0, relief_rotation=0, relief_reference_travel=0, relief_reference_rotation=0;
  double relief_deadline=0;
  V3 relief_anchor_position = V3::Zero();
  V7 relief_anchor_joints = V7::Zero();
  Eigen::Quaterniond relief_anchor_orientation = Eigen::Quaterniond::Identity();
  bool backoff_mode=false, backoff_reached=false, backoff_settling=false;
  JointLimitBackoffPlan backoff_plan;
  V7 backoff_torque_limits=V7::Zero(), backoff_best_progress=V7::Zero();
  double backoff_deadline=0;
  Reason backoff_previous_reason=Reason::None;
  std::uint64_t home_sequence = 0;
  double home_deadline = 0;
  bool reference_limited = false, task_force_limited = false, task_moment_limited = false;
  bool joint_torque_limited = false;
  bool reference_speed_limited = false;
  double drive_scale = 1.0;
  double contact_load_ratio = 0.0, peak_contact_load_ratio = 0.0;
  double singular_value = 0.0, command_age = 0.0;
  bool osc_regularized = false;
  std::uint64_t sequence = 0, epoch = 0;
};

class Controller {
 public:
  Controller(const Config&, const Sample&, std::uint64_t epoch, double now,
             const Command* initial_action=nullptr);
  Output step(const Sample&, const Command&, bool stop, double now, double dt);
 private:
  Config c_;
  const bool continuous_impedance_, operational_space_, bounded_equilibrium_, joint_impedance_, automatic_mode_;
  Output o_;
  Command target_;
  V7 posture_;
  V7 filtered_friction_torque_ = V7::Zero();
  V3 relief_anchor_position_ = V3::Zero();
  Eigen::Quaterniond relief_anchor_orientation_ = Eigen::Quaterniond::Identity();
  double relief_started_at_=0, relief_settled_at_=-1, relief_growing_at_=-1;
  double relief_best_force_=0, relief_best_moment_=0;
  double backoff_started_at_=0, backoff_settled_at_=-1;
  double armed_at_, stopping_at_ = 0, settled_at_ = -1, home_settled_at_ = -1;
  bool have_target_ = false, fault_ = false, action_pending_ = false;
  ruckig::Ruckig<3> translation_{0.001}, rotation_{0.001};
  ruckig::Ruckig<7> joints_{0.001};
  ruckig::InputParameter<3> ti_, ri_;
  ruckig::InputParameter<7> ji_;
  // update() retains its trajectory and elapsed time in the output object.
  // These must survive callbacks just like the input and Ruckig instances.
  ruckig::OutputParameter<3> translation_output_, rotation_output_;
  ruckig::OutputParameter<7> joint_output_;
  bool direct_joint_mode() const {return o_.backoff_mode || joint_impedance_ || (automatic_mode_ && o_.mode==Mode::Joint);}
  void reset_reference(const Sample&);
  void stop(Reason, double now, bool fault);
  bool advance_reference(const Sample&, double dt);
  void begin_contact_relief(const Sample&, double now);
  void advance_contact_relief(const Sample&, double now, double dt);
  void begin_joint_limit_backoff(const Sample&, double now);
  void advance_joint_limit_backoff(const Sample&, double now, double dt);
};
} // namespace fcc
