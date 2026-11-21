#include "controller.hpp"
#include "fr3_limits.hpp"
#include "operational_space.hpp"
#include "progressive_spring.hpp"
#include "joint_impedance.hpp"
#include "joint_limit_backoff.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fcc {
const char* phase_name(Phase p) {
  switch(p) {
    case Phase::Disarmed: return "disarmed";
    case Phase::Armed: return "armed";
    case Phase::Active: return "active";
    case Phase::Stopping: return "stopping";
    case Phase::Fault: return "fault";
  }
  return "fault";
}
const char* reason_name(Reason r) {
  switch(r) {
    case Reason::None: return "none";
    case Reason::UserStop: return "user_stop";
    case Reason::Watchdog: return "command_timeout";
    case Reason::ForceLimit: return "contact_force_limit";
    case Reason::MomentLimit: return "contact_moment_limit";
    case Reason::JointContact: return "external_joint_torque_limit";
    case Reason::JointLimit: return "joint_limit";
    case Reason::Singularity: return "singularity";
    case Reason::Overspeed: return "measured_overspeed";
    case Reason::InvalidState: return "invalid_robot_state";
    case Reason::Timing: return "control_deadline";
    case Reason::StopTimeout: return "stop_timeout";
    case Reason::ReferenceError: return "reference_generation_error";
    case Reason::Communication: return "backend_error";
    case Reason::RobotCartesianReflex: return "robot_cartesian_reflex";
    case Reason::RobotJointReflex: return "robot_joint_reflex";
    case Reason::RobotReflex: return "robot_reflex";
    case Reason::RobotControlError: return "robot_control_error";
    case Reason::HomeReached: return "home_reached";
    case Reason::HomeTimeout: return "home_timeout";
    case Reason::ReliefReleased: return "contact_relief_released";
    case Reason::ReliefTimeout: return "contact_relief_timeout";
    case Reason::ReliefTravel: return "contact_relief_travel_limit";
    case Reason::ReliefRotation: return "contact_relief_rotation_limit";
    case Reason::ReliefWorkspace: return "contact_relief_workspace_limit";
    case Reason::ReliefLoadGrowing: return "contact_relief_load_growing";
    case Reason::ReliefReferenceError: return "contact_relief_reference_error";
    case Reason::ReliefBiasMismatch: return "contact_relief_bias_posture_mismatch";
    case Reason::ReliefInvalidWrench: return "contact_relief_invalid_wrench_frame";
    case Reason::ReliefSetupMismatch: return "contact_relief_setup_mismatch";
    case Reason::ReliefReleaseLost: return "contact_relief_release_lost_during_stop";
    case Reason::ReliefJointTravel: return "contact_relief_joint_travel_limit";
    case Reason::BackoffReached: return "joint_limit_backoff_reached";
    case Reason::BackoffTimeout: return "joint_limit_backoff_timeout";
    case Reason::BackoffStartChanged: return "joint_limit_backoff_start_changed";
    case Reason::BackoffOutward: return "joint_limit_backoff_outward_motion";
    case Reason::BackoffHoldDrift: return "joint_limit_backoff_held_joint_drift";
    case Reason::BackoffTravel: return "joint_limit_backoff_travel_limit";
    case Reason::BackoffStalled: return "joint_limit_backoff_no_progress";
    case Reason::BackoffGoalLost: return "joint_limit_backoff_goal_lost_during_stop";
    case Reason::BackoffSetupMismatch: return "joint_limit_backoff_setup_mismatch";
  }
  return "unknown";
}
template<class T> static T capped(T x, double limit) {
  const double n = x.norm();
  if(n > limit) x *= limit/n;
  return x;
}
static V3 rotation_error(Eigen::Quaterniond desired, const Eigen::Quaterniond& actual) {
  Eigen::Quaterniond delta = desired * actual.conjugate();
  if(delta.w() < 0) delta.coeffs() *= -1;
  Eigen::AngleAxisd aa(delta.normalized());
  return aa.axis() * aa.angle();
}
static Eigen::Quaterniond rotation_increment(const V3& v) {
  if(v.norm() < 1e-12) return Eigen::Quaterniond::Identity();
  return Eigen::Quaterniond(Eigen::AngleAxisd(v.norm(), v.normalized()));
}
Config::Config(const std::string& model) : robot_model(model) {
  if(model=="fr3") {
    // Franka's suggested rectangular position/velocity envelope, deliberately
    // narrower than the physical limits. See docs/model-profiles.md.
    joint_lower << -2.3476,-1.5454,-2.4937,-2.7714,-2.5100,.7773,-2.7045;
    joint_upper << 2.3476,1.5454,2.4937,-.4226,2.5100,4.2841,2.7045;
  } else if(model!="panda") {
    throw std::invalid_argument("robot_model must be panda or fr3");
  }
}
void Config::validate() const {
  if(compliance_mode!="contact_yielding" && compliance_mode!="continuous_impedance" &&
     compliance_mode!="operational_space" && compliance_mode!="joint_impedance" && compliance_mode!="auto")
    throw std::invalid_argument("compliance_mode must be contact_yielding, continuous_impedance, operational_space, joint_impedance or auto");
  if(effort_limit_policy!="configured" && effort_limit_policy!="franka" &&
     effort_limit_policy!="task_bounded" && effort_limit_policy!="joint_bounded" && effort_limit_policy!="mode_bounded")
    throw std::invalid_argument("effort_limit_policy must be configured, franka, task_bounded, joint_bounded or mode_bounded");
  const bool automatic_mode=compliance_mode=="auto";
  if(automatic_mode!=(effort_limit_policy=="mode_bounded"))
    throw std::invalid_argument("auto mode and mode_bounded effort policy must be selected together");
  if((compliance_mode=="joint_impedance")!=(effort_limit_policy=="joint_bounded"))
    throw std::invalid_argument("joint_impedance and joint_bounded effort policy must be selected together");
  if(osc_feedback_mode!="linear" && osc_feedback_mode!="progressive" && osc_feedback_mode!="saturated")
    throw std::invalid_argument("osc_feedback_mode must be linear, progressive or saturated");
  const bool progressive=osc_feedback_mode=="progressive";
  const bool bounded_equilibrium=osc_feedback_mode!="linear";
  if(bounded_equilibrium!=(effort_limit_policy=="task_bounded" || automatic_mode))
    throw std::invalid_argument("progressive/saturated feedback and task_bounded effort policy must be selected together");
  for(double r : {progressive_position_soft_radius,progressive_position_saturation_radius,
                 progressive_rotation_soft_radius,progressive_rotation_saturation_radius})
    if(!std::isfinite(r) || r<=0) throw std::invalid_argument("progressive radii must be finite and positive");
  for(double k : {progressive_translation_soft_stiffness,progressive_rotation_soft_stiffness})
    if(!std::isfinite(k) || k<0) throw std::invalid_argument("progressive soft stiffness must be finite and nonnegative");
  if(progressive_position_saturation_radius<=progressive_position_soft_radius ||
     progressive_rotation_saturation_radius<=progressive_rotation_soft_radius ||
     progressive_rotation_saturation_radius>std::acos(-1.0))
    throw std::invalid_argument("progressive saturation radius must exceed soft radius; rotation cannot exceed pi");
  if(progressive &&
     (progressive_translation_soft_stiffness*progressive_position_saturation_radius>=commanded_force_limit ||
      progressive_rotation_soft_stiffness*progressive_rotation_saturation_radius>=commanded_moment_limit))
    throw std::invalid_argument("progressive effort caps must exceed soft_stiffness * saturation_radius");
  if(collision_policy!="configured" && collision_policy!="robot" && collision_policy!="robot_configured")
    throw std::invalid_argument("collision_policy must be configured, robot or robot_configured");
  if((effort_limit_policy!="configured" || collision_policy!="configured") &&
     ((compliance_mode!="operational_space" && compliance_mode!="joint_impedance" && !automatic_mode) ||
      robot_model!="fr3" || motion_limit_policy!="franka"))
    throw std::invalid_argument("full-envelope effort / robot collision policies require FR3 operational_space or joint_impedance with franka motion limits");
  if(bounded_cartesian_spring && compliance_mode!="continuous_impedance")
    throw std::invalid_argument("bounded_cartesian_spring requires continuous_impedance");
  if(motion_limit_policy!="configured" && motion_limit_policy!="franka")
    throw std::invalid_argument("motion_limit_policy must be configured or franka");
  const bool franka_limits=motion_limit_policy=="franka";
  if(franka_limits && robot_model!="fr3")
    throw std::invalid_argument("franka motion_limit_policy currently requires fr3");
  for(double x : {commanded_force_limit,commanded_moment_limit,contact_force_soft,
      contact_force_stop,contact_moment_soft,contact_moment_stop,translation_velocity,
      translation_acceleration,translation_jerk,rotation_velocity,rotation_acceleration,
      rotation_jerk,joint_velocity,joint_acceleration,joint_jerk,position_error_limit,
      rotation_error_limit,joint_error_limit,torque_rate_limit,command_timeout,
      stop_velocity,stop_settle_time,stop_timeout,
      minimum_singular_value,jacobian_rotation_scale,maximum_control_period,osc_regularization}) {
    if(!std::isfinite(x) || x <= 0) throw std::invalid_argument("limits must be finite and positive");
  }
  for(const auto& gain : {osc_translation_stiffness,osc_translation_damping,
                         osc_rotation_stiffness,osc_rotation_damping})
    if(!gain.allFinite() || (gain.array()<0).any())
      throw std::invalid_argument("OSC physical feedback gains must be finite and nonnegative");
  if(bounded_equilibrium && ((osc_translation_damping.array()<=0).any() ||
                             (osc_rotation_damping.array()<=0).any()))
    throw std::invalid_argument("bounded equilibrium feedback requires positive physical damping on every axis");
  if(osc_feedback_mode=="saturated" && ((osc_translation_stiffness.array()<=0).any() ||
                                       (osc_rotation_stiffness.array()<=0).any()))
    throw std::invalid_argument("saturated feedback requires positive physical stiffness on every axis");
  if(!std::isfinite(friction_compensation_torque) || friction_compensation_torque<0 ||
     friction_compensation_torque>torque_limits.minCoeff())
    throw std::invalid_argument("friction_compensation_torque must be finite, nonnegative and within every actuator torque limit");
  if(!std::isfinite(friction_compensation_epsilon) || friction_compensation_epsilon<=0)
    throw std::invalid_argument("friction_compensation_epsilon must be finite and positive");
  if(!std::isfinite(friction_compensation_filter_time_s) || friction_compensation_filter_time_s<0)
    throw std::invalid_argument("friction_compensation_filter_time_s must be finite and nonnegative");
  for(double x : {posture_stiffness,posture_damping,nullspace_torque_limit})
    if(!std::isfinite(x) || x < 0) throw std::invalid_argument("posture gains must be nonnegative");
  if(!std::isfinite(joint_limit_margin) || joint_limit_margin<0 ||
     (!franka_limits && joint_limit_margin==0))
    throw std::invalid_argument("joint limit margin must be positive (or zero with franka motion limits)");
  for(double x : {measured_translation_velocity_limit,measured_rotation_velocity_limit,
                 measured_joint_velocity_limit})
    if(!std::isfinite(x) || x < 0)
      throw std::invalid_argument("measured velocity limits must be finite and nonnegative (zero selects legacy default)");
  if(!measured_joint_velocity_limits.allFinite() || (measured_joint_velocity_limits.array()<0).any())
    throw std::invalid_argument("per-joint measured velocity limits must be finite and nonnegative");
  const V7 joint_trips=joint_speed_limits();
  if(translation_speed_limit()<translation_velocity || rotation_speed_limit()<rotation_velocity ||
     (joint_trips.array()<joint_velocity).any())
    throw std::invalid_argument("measured velocity limits must accommodate reference velocity");
  const auto positive = [](const auto& v) {return v.allFinite() && (v.array() > 0).all();};
  if(!positive(translation_stiffness) || !positive(rotation_stiffness) ||
     !positive(translation_damping) || !positive(rotation_damping) ||
     !positive(joint_stiffness) || !positive(joint_damping) || !positive(stop_damping) ||
     !positive(torque_limits) || !positive(external_torque_limits) || !positive(joint_feedback_torque_limits) ||
     !positive(osc_translation_kp) || !positive(osc_translation_kd) ||
     !positive(osc_rotation_kp) || !positive(osc_rotation_kd))
    throw std::invalid_argument("gain and torque vectors must be finite and positive");
  if(!joint_lower.allFinite() || !joint_upper.allFinite() ||
     ((joint_upper-joint_lower).array() <= 2*joint_limit_margin).any())
    throw std::invalid_argument("invalid joint bounds");
  const Config profile(robot_model);
  const V7 allowed_lower=franka_limits?fr3_limits::lower():profile.joint_lower;
  const V7 allowed_upper=franka_limits?fr3_limits::upper():profile.joint_upper;
  if((joint_lower.array()<allowed_lower.array()).any() ||
     (joint_upper.array()>allowed_upper.array()).any())
    throw std::invalid_argument("joint bounds exceed the selected robot model profile");
  if(robot_model=="fr3") {
    // Check the effective measured-overspeed trip, including legacy defaults.
    // Check the analytic
    // position-dependent limits at both ends of the entire permitted interval.
    const V7 physical_lower=fr3_limits::lower(), physical_upper=fr3_limits::upper();
    const V7 offset=fr3_limits::offset(), deceleration=fr3_limits::deceleration();
    const V7 rectangular_velocity=(V7()<<2,1,1.5,1.25,3,1.5,3).finished();
    for(int i=0;i<7;i++) {
      const double distance=std::min(joint_lower[i]+joint_limit_margin-physical_lower[i],
                                    physical_upper[i]-joint_upper[i]+joint_limit_margin);
      // With the full envelope, enforce the directional bound every callback;
      // the old profile keeps its conservative, whole-interval validation.
      const double velocity=franka_limits?fr3_limits::speed()[i]:std::min(rectangular_velocity[i],
          std::max(0.0,std::sqrt(2*deceleration[i]*distance)-offset[i]));
      if(joint_trips[i]>velocity)
        throw std::invalid_argument("FR3 measured velocity trip exceeds position-dependent limits");
    }
    if(joint_acceleration>10 || joint_jerk>5000)
      throw std::invalid_argument("FR3 joint acceleration or jerk exceeds model limits");
    if(translation_speed_limit()>3 || rotation_speed_limit()>2.5 ||
       translation_acceleration>9 || rotation_acceleration>17 ||
       translation_jerk>4500 || rotation_jerk>8500)
      throw std::invalid_argument("FR3 Cartesian motion exceeds model limits");
  }
  const V7 physical_torque=(V7()<<87,87,87,87,12,12,12).finished();
  if((joint_feedback_torque_limits.array()>physical_torque.array()).any())
    throw std::invalid_argument("joint feedback torque budget exceeds model actuator limits");
  if(effort_limit_policy!="configured" &&
     ((torque_limits-physical_torque).norm()!=0 || torque_rate_limit!=1000))
    throw std::invalid_argument("full-envelope effort policy requires full model torque limits and 1000 Nm/s slew");
  if((torque_limits.array()>physical_torque.array()).any() || torque_rate_limit>1000)
    throw std::invalid_argument("commanded torque or torque rate exceeds robot limits");
  if(contact_force_soft >= contact_force_stop || contact_moment_soft >= contact_moment_stop)
    throw std::invalid_argument("soft contact thresholds must be below stop thresholds");
  if(stop_timeout < stop_settle_time || maximum_control_period > .02)
    throw std::invalid_argument("invalid watchdog/stop timing");
}
static bool relief_options_valid(const Config& c,const ContactReliefOptions& r) {
  const bool local_envelope=r.manual_stop && r.workspace_lower.isZero(0) && r.workspace_upper.isZero(0);
  if(!c.uses_osc_model() || c.osc_feedback_mode=="linear" || !r.workspace_confirmed_clear ||
     !r.bias_wrench_tcp_base.allFinite() || !r.bias_joint_positions.allFinite() ||
     !r.workspace_lower.allFinite() || !r.workspace_upper.allFinite() ||
     (!local_envelope && (r.workspace_lower.array()>=r.workspace_upper.array()).any())) return false;
  if(r.manual_stop) {
    // Zero here means UNUSED, not a claimed free-space calibration.
    if(r.bias_confirmed_free || !r.bias_wrench_tcp_base.isZero(0) || !r.bias_joint_positions.isZero(0)) return false;
  } else if(!r.bias_confirmed_free || r.bias_wrench_tcp_base.head<3>().norm()>10 ||
     r.bias_wrench_tcp_base.tail<3>().norm()>3 ||
     (r.bias_joint_positions.array()<=c.joint_lower.array()+c.joint_limit_margin).any() ||
     (r.bias_joint_positions.array()>=c.joint_upper.array()-c.joint_limit_margin).any()) return false;
  const double values[]{r.force_release,r.moment_release,r.max_translation,r.max_rotation,
      r.translation_speed,r.rotation_speed,r.translation_admittance,r.rotation_admittance};
  const double maxima[]{10,3,.05,.17453292519943295,.02,.1,.01,.1};
  for(int i=0;i<8;i++) if(!std::isfinite(values[i]) || values[i]<=0 || values[i]>maxima[i]) return false;
  return std::isfinite(r.upward_velocity) && r.upward_velocity>=0 && r.upward_velocity<=.005 &&
      r.force_release<std::min(40.,c.contact_force_stop) && r.moment_release<std::min(20.,c.contact_moment_stop);
}
void ContactReliefOptions::validate(const Config& c) const {
  if(!relief_options_valid(c,*this))
    throw std::invalid_argument("contact relief requires bounded Cartesian OSC/auto, confirmed retreat clearance, finite bounded options, and verified bias/box only in automatic mode");
}
static bool relief_inside(const V3& p,const ContactReliefOptions& r) {
  if(r.manual_stop && r.workspace_lower.isZero(0) && r.workspace_upper.isZero(0)) return true; // Local excursion/path caps still apply.
  return (p.array()>=r.workspace_lower.array()).all() && (p.array()<=r.workspace_upper.array()).all();
}
static Reason relief_state_reason(const Config& c,const Sample& s,const ContactReliefOptions& r,bool stationary) {
  if(!relief_options_valid(c,r)) return Reason::ReferenceError;
  if(!s.finite() || !s.kinematics_valid || !s.dynamics_valid) return Reason::InvalidState;
  if(!s.wrench_tcp_valid) return Reason::ReliefInvalidWrench;
  if(!s.relief_setup_valid) return Reason::ReliefSetupMismatch;
  V6 model_wrench;V7 model_null;bool regularized;
  if(!operational_space_effort(s,c,V6::Zero(),V7::Zero(),model_wrench,model_null,regularized))
    return Reason::InvalidState;
  if(!relief_inside(s.position,r)) return Reason::ReliefWorkspace;
  if(!r.manual_stop && (s.q-r.bias_joint_positions).cwiseAbs().maxCoeff()>.2) return Reason::ReliefBiasMismatch;
  if(stationary && s.dq.cwiseAbs().maxCoeff()>=std::min(.02,c.stop_velocity)) return Reason::Overspeed;
  const V6 velocity=s.jacobian*s.dq;
  if(s.dq.cwiseAbs().maxCoeff()>.15 || velocity.head<3>().norm()>1.5*r.translation_speed ||
     velocity.tail<3>().norm()>1.5*r.rotation_speed ||
     (s.dq.cwiseAbs().array()>c.joint_speed_limits().array()).any() ||
     velocity.head<3>().norm()>c.translation_speed_limit() || velocity.tail<3>().norm()>c.rotation_speed_limit())
    return Reason::Overspeed;
  if(c.motion_limit_policy=="franka" &&
     ((s.dq.array()>fr3_limits::directional_speed(s.q,true).array()).any() ||
      (s.dq.array() < -fr3_limits::directional_speed(s.q,false).array()).any())) return Reason::Overspeed;
  if((s.q.array()<=c.joint_lower.array()+c.joint_limit_margin).any() ||
     (s.q.array()>=c.joint_upper.array()-c.joint_limit_margin).any()) return Reason::JointLimit;
  // Never debias hard-stop evidence. These norm checks are additionally bounded
  // by 40 N / 20 Nm even if a different ordinary-control profile is more permissive.
  if(s.external_wrench_tcp_base.head<3>().norm()>=std::min(40.,c.contact_force_stop)) return Reason::ForceLimit;
  if(s.external_wrench_tcp_base.tail<3>().norm()>=std::min(20.,c.contact_moment_stop)) return Reason::MomentLimit;
  if((s.external_torque.cwiseAbs().array()>=c.external_torque_limits.array()).any()) return Reason::JointContact;
  J67 j=s.jacobian;j.bottomRows<3>()*=c.jacobian_rotation_scale;
  const M6 gram=j*j.transpose();if(!gram.allFinite()) return Reason::InvalidState;
  Eigen::SelfAdjointEigenSolver<M6> eigen(gram,Eigen::EigenvaluesOnly);
  const double sigma=std::max(.05,c.minimum_singular_value);
  if(eigen.info()!=Eigen::Success || eigen.eigenvalues()[0]<sigma*sigma) return Reason::Singularity;
  return Reason::None;
}
void validate_contact_relief_start(const Config& c,const Sample& s,const ContactReliefOptions& r) {
  r.validate(c);
  const auto reason=relief_state_reason(c,s,r,true);
  if(reason!=Reason::None)
    throw std::runtime_error(std::string("contact relief preflight: ")+reason_name(reason));
}
Reason contact_relief_observation_reason(const Config& c,const Sample& s,const ContactReliefOptions& r,
                                        const V3& anchor,const Eigen::Quaterniond& orientation,bool released,const V7* initial_joints) {
  const auto reason=relief_state_reason(c,s,r,false);
  if(reason!=Reason::None) return reason;
  if(r.manual_stop && (!initial_joints || !initial_joints->allFinite())) return Reason::InvalidState;
  if(r.manual_stop && initial_joints && (s.q-*initial_joints).cwiseAbs().maxCoeff()>.2) return Reason::ReliefJointTravel;
  if((s.position-anchor).norm()>r.max_translation) return Reason::ReliefTravel;
  if(rotation_error(s.orientation,orientation).norm()>r.max_rotation) return Reason::ReliefRotation;
  const V6 residual=s.external_wrench_tcp_base-r.bias_wrench_tcp_base;
  if(released && (r.manual_stop || residual.head<3>().norm()>=r.force_release || residual.tail<3>().norm()>=r.moment_release))
    return Reason::ReliefReleaseLost;
  return Reason::None;
}
bool Sample::finite() const {
  return q.allFinite() && dq.allFinite() && coriolis.allFinite() && previous_torque.allFinite()
    && external_torque.allFinite() && position.allFinite() && orientation.coeffs().allFinite()
    && std::abs(orientation.norm()-1) < 1e-3 && jacobian.allFinite()
    && external_wrench.allFinite() && external_wrench_stiffness.allFinite()
    && cartesian_collision.allFinite() && joint_collision.allFinite()
    && std::isfinite(robot_time) && std::isfinite(communication_success)
    && (!wrench_tcp_valid || external_wrench_tcp_base.allFinite())
    && (!dynamics_valid || (mass.allFinite() && jacobian_dot_velocity.allFinite()));
}
Reason ordinary_hard_contact_reason(const Config& c,const Sample& s) {
  if(c.collision_policy!="configured") return Reason::None;
  const bool continuous=c.compliance_mode!="contact_yielding";
  const V3 force=s.orientation.conjugate()*V3(s.external_wrench.head<3>());
  const V3 moment=s.orientation.conjugate()*V3(s.external_wrench.tail<3>());
  if((continuous?force.cwiseAbs().maxCoeff():s.external_wrench.head<3>().norm())>=c.contact_force_stop)
    return Reason::ForceLimit;
  if((continuous?moment.cwiseAbs().maxCoeff():s.external_wrench.tail<3>().norm())>=c.contact_moment_stop)
    return Reason::MomentLimit;
  if((s.external_torque.cwiseAbs().array()>=c.external_torque_limits.array()).any()) return Reason::JointContact;
  return Reason::None;
}
Controller::Controller(const Config& c, const Sample& s, std::uint64_t epoch, double now,
                       const Command* initial_action)
  : c_(c), continuous_impedance_(c.compliance_mode!="contact_yielding"),
    operational_space_(c.uses_osc_model()),
    bounded_equilibrium_(c.osc_feedback_mode!="linear"),
    joint_impedance_(c.compliance_mode=="joint_impedance"), automatic_mode_(c.compliance_mode=="auto"),
    posture_(s.q), armed_at_(now) {
  c_.validate();
  if(!s.finite() || !s.kinematics_valid) throw std::invalid_argument("cannot arm with invalid state");
  if(operational_space_) {
    V6 wrench;V7 null_torque;bool regularized;
    if(!operational_space_effort(s,c_,V6::Zero(),V7::Zero(),wrench,null_torque,regularized))
      throw std::invalid_argument("cannot arm OSC without valid positive-definite model dynamics");
  }
  o_.epoch = epoch;
  if(initial_action) {
    const bool relief=initial_action->relieve_contact,backoff=initial_action->backoff_joint_limits;
    if(relief==backoff || initial_action->epoch!=epoch || initial_action->sequence==0 ||
       initial_action->mode!=(backoff?Mode::Joint:Mode::Cartesian) || initial_action->finish_at_target ||
       initial_action->has_orientation || initial_action->has_posture ||
       !initial_action->twist.isZero(0) || !initial_action->acceleration.isZero(0) ||
       !initial_action->joint_velocity.isZero(0) ||
       !std::isfinite(initial_action->expires_at) || !std::isfinite(initial_action->action_timeout) ||
       initial_action->action_timeout<=.2 || initial_action->action_timeout>(backoff?30.:10.))
      throw std::invalid_argument("invalid immutable bounded-action session command");
    if(relief) initial_action->relief_options.validate(c_);
    else if(!backoff_plan_valid(c_,initial_action->backoff_options,initial_action->backoff_plan))
      throw std::invalid_argument("invalid immutable joint-limit backoff plan");
    target_=*initial_action;action_pending_=true;
    if(relief) {
      o_.relief_mode=true;o_.relief_manual=target_.relief_options.manual_stop;
      o_.relief_deadline=target_.expires_at;
      o_.relief_anchor_position=s.position;o_.relief_anchor_orientation=s.orientation;o_.relief_anchor_joints=s.q;
    } else {
      o_.backoff_mode=true;o_.mode=Mode::Joint;o_.backoff_deadline=target_.expires_at;
      o_.backoff_plan=target_.backoff_plan;
      o_.backoff_torque_limits=backoff_torque_limits(c_,target_.backoff_options,o_.backoff_plan);
    }
  }
  if(joint_impedance_) o_.mode=Mode::Joint;
  o_.phase = Phase::Armed;
  o_.torque = s.previous_torque;
  reset_reference(s);
  for(int i=0;i<3;i++) {
    ti_.max_velocity[i] = c.translation_velocity/std::sqrt(3.0);
    ti_.max_acceleration[i] = c.translation_acceleration/std::sqrt(3.0);
    ti_.max_jerk[i] = c.translation_jerk/std::sqrt(3.0);
    ri_.max_velocity[i] = c.rotation_velocity/std::sqrt(3.0);
    ri_.max_acceleration[i] = c.rotation_acceleration/std::sqrt(3.0);
    ri_.max_jerk[i] = c.rotation_jerk/std::sqrt(3.0);
  }
  ji_.max_velocity.fill(c.joint_velocity);
  ji_.max_acceleration.fill(c.joint_acceleration);
  ji_.max_jerk.fill(c.joint_jerk);
}
void Controller::reset_reference(const Sample& s) {
  // New sessions/modes and yielding must not carry an old feedforward bias.
  filtered_friction_torque_.setZero();
  translation_.reset();rotation_.reset();joints_.reset();
  o_.reference_position = s.position;
  o_.reference_orientation = s.orientation;
  o_.reference_joints = s.q;
  o_.reference_joint_velocity.setZero();
  o_.reference_twist.setZero();
  o_.reference_acceleration.setZero();
  for(int i=0;i<3;i++) ti_.current_position[i]=s.position[i];
  ti_.current_velocity.fill(0); ti_.current_acceleration.fill(0);
  ri_.current_position.fill(0); ri_.current_velocity.fill(0); ri_.current_acceleration.fill(0);
  for(int i=0;i<7;i++) ji_.current_position[i]=s.q[i];
  ji_.current_velocity.fill(0); ji_.current_acceleration.fill(0);
}
void Controller::stop(Reason r, double now, bool fault) {
  if(o_.phase != Phase::Stopping) {
    o_.phase=Phase::Stopping; o_.reason=r; stopping_at_=now; settled_at_=-1;
    if(o_.stop_trigger==Reason::None) o_.stop_trigger=r;
  } else if(fault && (!fault_ ||
      (backoff_local_fault_allowed(o_.reason) && !backoff_local_fault_allowed(r)))) o_.reason=r;
  // Keep the first stop_trigger, but never hide a disqualifying later fault
  // behind a summary that an explicit joint-limit backoff is allowed to replace.
  fault_ = fault_ || fault;
  if(fault && o_.relief_mode) o_.relief_released=false;
  if(fault && o_.backoff_mode) o_.backoff_reached=false;
}
bool Controller::advance_reference(const Sample& s, double dt) {
  if(direct_joint_mode()) {
    o_.reference_joints=target_.joints;
    o_.reference_joint_velocity=target_.joint_velocity;
    return true;
  }
  if(bounded_equilibrium_) {
    // Direct equilibrium: no virtual trajectory can accumulate behind a wall.
    // Every newly accepted target redirects the spring on this callback. The
    // caller may pace targets and supply coherent twist/acceleration explicitly.
    o_.reference_position=target_.position;
    o_.reference_orientation=target_.orientation;
    o_.reference_twist=target_.twist;
    o_.reference_acceleration=target_.acceleration;
    return true;
  }
  translation_.delta_time=dt; rotation_.delta_time=dt; joints_.delta_time=dt;
  if(o_.mode==Mode::Cartesian) {
    for(int i=0;i<3;i++) {
      ti_.target_position[i]=target_.position[i];
      ti_.target_velocity[i]=target_.twist[i];
      ti_.target_acceleration[i]=0;
    }
    if(operational_space_) {
      // A one-axis move can use the full resultant budget (not v/sqrt(3)).
      // General multi-axis updates retain the conservative Cartesian box.
      int moving_axes=0;
      for(int i=0;i<3;i++)
        if(std::abs(ti_.target_position[i]-ti_.current_position[i])>1e-12 ||
           std::abs(ti_.target_velocity[i])+std::abs(ti_.current_velocity[i])+
           std::abs(ti_.current_acceleration[i])>1e-12) ++moving_axes;
      const double divisor=moving_axes<=1?1.0:std::sqrt(3.0);
      for(int i=0;i<3;i++) {
        ti_.max_velocity[i]=c_.translation_velocity/divisor;
        ti_.max_acceleration[i]=c_.translation_acceleration/divisor;
        ti_.max_jerk[i]=c_.translation_jerk/divisor;
      }
    }
    auto& tr=translation_output_;
    auto& rr=rotation_output_;
    if(translation_.update(ti_,tr)<0) return false;
    tr.pass_to_input(ti_);
    o_.reference_position=Eigen::Map<V3>(tr.new_position.data());
    o_.reference_twist.head<3>()=Eigen::Map<V3>(tr.new_velocity.data());
    // Reanchor each short rotational trajectory at the current reference. Its
    // coordinates and angular derivatives are expressed in the robot base.
    const V3 delta=rotation_error(target_.orientation,o_.reference_orientation);
    ri_.current_position.fill(0);
    for(int i=0;i<3;i++) {
      ri_.target_position[i]=delta[i];
      ri_.target_velocity[i]=target_.twist[i+3];
      ri_.target_acceleration[i]=0;
    }
    if(rotation_.update(ri_,rr)<0) return false;
    rr.pass_to_input(ri_);
    o_.reference_orientation=(rotation_increment(Eigen::Map<V3>(rr.new_position.data()))
                              *o_.reference_orientation).normalized();
    o_.reference_twist.tail<3>()=Eigen::Map<V3>(rr.new_velocity.data());
    o_.reference_acceleration.head<3>()=Eigen::Map<V3>(tr.new_acceleration.data());
    o_.reference_acceleration.tail<3>()=Eigen::Map<V3>(rr.new_acceleration.data());
    if(!continuous_impedance_ && ((o_.reference_position-s.position).norm()>c_.position_error_limit ||
       rotation_error(o_.reference_orientation,s.orientation).norm()>c_.rotation_error_limit)) {
      // A blocked robot must not accumulate reference displacement/velocity.
      o_.reference_position=s.position+capped(V3(o_.reference_position-s.position),c_.position_error_limit);
      const V3 bounded_rotation=capped(rotation_error(o_.reference_orientation,s.orientation),c_.rotation_error_limit);
      o_.reference_orientation=(rotation_increment(bounded_rotation)*s.orientation).normalized();
      for(int i=0;i<3;i++) ti_.current_position[i]=o_.reference_position[i];
      ti_.current_velocity.fill(0);ti_.current_acceleration.fill(0);
      ri_.current_velocity.fill(0);ri_.current_acceleration.fill(0);
      o_.reference_twist.setZero();
      o_.reference_acceleration.setZero();
      o_.reference_limited=true;
      o_.effort_limited=true;
    }
    for(int i=0;i<3;i++) o_.reference_speed_limited=o_.reference_speed_limited ||
        std::abs(o_.reference_twist[i])>=.99*ti_.max_velocity[i] ||
        std::abs(o_.reference_twist[i+3])>=.99*ri_.max_velocity[i];
  } else {
    for(int i=0;i<7;i++) {
      ji_.target_position[i]=target_.joints[i];
      ji_.target_velocity[i]=target_.joint_velocity[i];
      ji_.target_acceleration[i]=0;
    }
    auto& jr=joint_output_;
    if(joints_.update(ji_,jr)<0) return false;
    jr.pass_to_input(ji_);
    o_.reference_joints=Eigen::Map<V7>(jr.new_position.data());
    if(!continuous_impedance_ && (o_.reference_joints-s.q).cwiseAbs().maxCoeff()>c_.joint_error_limit) {
      for(int i=0;i<7;i++) {
        o_.reference_joints[i]=s.q[i]+std::clamp(o_.reference_joints[i]-s.q[i],-c_.joint_error_limit,c_.joint_error_limit);
        ji_.current_position[i]=o_.reference_joints[i];
      }
      ji_.current_velocity.fill(0);ji_.current_acceleration.fill(0);
      o_.reference_limited=true;
      o_.effort_limited=true;
    }
    o_.reference_joint_velocity=Eigen::Map<V7>(ji_.current_velocity.data());
    o_.reference_speed_limited=o_.reference_joint_velocity.cwiseAbs().maxCoeff()>=.99*c_.joint_velocity;
  }
  return true;
}
void Controller::begin_contact_relief(const Sample& s,double now) {
  o_.relief_mode=true;o_.relief_manual=target_.relief_options.manual_stop;
  o_.relief_settling=o_.relief_released=false;
  o_.relief_deadline=target_.expires_at;
  const auto reason=relief_state_reason(c_,s,target_.relief_options,true);
  if(reason!=Reason::None) {stop(reason,now,true);return;}
  relief_anchor_position_=s.position;relief_anchor_orientation_=s.orientation;
  o_.relief_anchor_position=s.position;o_.relief_anchor_orientation=s.orientation;o_.relief_anchor_joints=s.q;
  relief_started_at_=now;relief_settled_at_=relief_growing_at_=-1;
  o_.relief_translation=o_.relief_rotation=o_.relief_reference_travel=o_.relief_reference_rotation=0;
  o_.relief_filtered_wrench=s.external_wrench_tcp_base;
  if(!o_.relief_manual) o_.relief_filtered_wrench-=target_.relief_options.bias_wrench_tcp_base;
  relief_best_force_=o_.relief_filtered_wrench.head<3>().norm();
  relief_best_moment_=o_.relief_filtered_wrench.tail<3>().norm();
  // The initial equilibrium is measured on this callback, never a transport pose.
  reset_reference(s);target_.position=s.position;target_.orientation=s.orientation;
  target_.twist.setZero();target_.acceleration.setZero();
}
void Controller::advance_contact_relief(const Sample& s,double now,double dt) {
  const auto& r=target_.relief_options;
  if(!s.wrench_tcp_valid) {stop(Reason::ReliefInvalidWrench,now,true);return;}
  if(!s.relief_setup_valid) {stop(Reason::ReliefSetupMismatch,now,true);return;}
  if(!r.manual_stop && (s.q-r.bias_joint_positions).cwiseAbs().maxCoeff()>.2) {stop(Reason::ReliefBiasMismatch,now,true);return;}
  if(r.manual_stop && (s.q-o_.relief_anchor_joints).cwiseAbs().maxCoeff()>.2) {stop(Reason::ReliefJointTravel,now,true);return;}
  o_.relief_translation=(s.position-relief_anchor_position_).norm();
  o_.relief_rotation=rotation_error(s.orientation,relief_anchor_orientation_).norm();
  if(!relief_inside(s.position,r)) {stop(Reason::ReliefWorkspace,now,true);return;}
  if(o_.relief_translation>r.max_translation) {stop(Reason::ReliefTravel,now,true);return;}
  if(o_.relief_rotation>r.max_rotation) {stop(Reason::ReliefRotation,now,true);return;}
  if((o_.reference_position-s.position).norm()>.005 ||
     rotation_error(o_.reference_orientation,s.orientation).norm()>.03490658503988659) {
    stop(Reason::ReliefReferenceError,now,true);return;
  }
  V6 residual=s.external_wrench_tcp_base;
  if(!r.manual_stop) residual-=r.bias_wrench_tcp_base;
  const double alpha=-std::expm1(-dt/.050);
  o_.relief_filtered_wrench+=alpha*(residual-o_.relief_filtered_wrench);
  const double force=o_.relief_filtered_wrench.head<3>().norm(),moment=o_.relief_filtered_wrench.tail<3>().norm();
  relief_best_force_=std::min(relief_best_force_,force);relief_best_moment_=std::min(relief_best_moment_,moment);
  if(force>relief_best_force_+2. || moment>relief_best_moment_+.3) {
    if(relief_growing_at_<0) relief_growing_at_=now;
    if(now-relief_growing_at_>=.050) {stop(Reason::ReliefLoadGrowing,now,true);return;}
  } else relief_growing_at_=-1;
  if(!r.manual_stop && force<r.force_release && moment<r.moment_release) {
    // Freeze the last bounded equilibrium: no speed-cap bypass or uncounted
    // reference jump when the residual crosses a release threshold.
    if(!o_.relief_settling) {o_.relief_settling=true;relief_settled_at_=-1;}
    o_.reference_twist.setZero();o_.reference_acceleration.setZero();
    if(s.dq.cwiseAbs().maxCoeff()<std::min(.02,c_.stop_velocity)) {
      if(relief_settled_at_<0) relief_settled_at_=now;
      if(now-relief_settled_at_>=.200) {
        // Stopping skips the ordinary OSC solve, so validate the model before
        // certifying release on this callback as well as in final observations.
        const auto invalid=contact_relief_observation_reason(c_,s,r,relief_anchor_position_,relief_anchor_orientation_,false,&o_.relief_anchor_joints);
        if(invalid!=Reason::None) {stop(invalid,now,true);return;}
        o_.relief_released=true;stop(Reason::ReliefReleased,now,false);
      }
    } else relief_settled_at_=-1;
    return;
  }
  o_.relief_settling=false;relief_settled_at_=-1;
  if(now==relief_started_at_) return; // First callback has a zero-error, zero-twist equilibrium.
  // SDK sign is robot-on-environment: NEGATE to follow the environment's push.
  const V3 force_vector=o_.relief_filtered_wrench.head<3>();
  V3 upward(0,0,r.upward_velocity);
  // Optional preference may be tangential/away, never oppose the environment's push.
  const double against=upward.dot(force_vector);
  if(against>0 && force_vector.squaredNorm()>1e-12)
    upward-=against/force_vector.squaredNorm()*force_vector;
  const V3 linear=capped(V3(-r.translation_admittance*force_vector+upward),r.translation_speed);
  const V3 angular=capped(V3(-r.rotation_admittance*o_.relief_filtered_wrench.tail<3>()),r.rotation_speed);
  const V3 next=o_.reference_position+dt*linear;
  const auto orientation=(rotation_increment(V3(dt*angular))*o_.reference_orientation).normalized();
  const double travel=o_.relief_reference_travel+dt*linear.norm();
  const double rotation=o_.relief_reference_rotation+dt*angular.norm();
  if(!relief_inside(next,r)) {stop(Reason::ReliefWorkspace,now,true);return;}
  if(travel>r.max_translation || (next-relief_anchor_position_).norm()>r.max_translation) {
    stop(Reason::ReliefTravel,now,true);return;
  }
  if(rotation>r.max_rotation || rotation_error(orientation,relief_anchor_orientation_).norm()>r.max_rotation) {
    stop(Reason::ReliefRotation,now,true);return;
  }
  // Do not wind an equilibrium up behind a blocked robot, even within the box.
  if((next-s.position).norm()>.005 || rotation_error(orientation,s.orientation).norm()>.03490658503988659) {
    stop(Reason::ReliefReferenceError,now,true);return;
  }
  o_.reference_position=next;o_.reference_orientation=orientation;
  o_.reference_twist.head<3>()=linear;o_.reference_twist.tail<3>()=angular;
  o_.reference_acceleration.setZero();
  o_.relief_reference_travel=travel;o_.relief_reference_rotation=rotation;
}
void Controller::begin_joint_limit_backoff(const Sample& s,double now) {
  o_.backoff_mode=true;o_.backoff_reached=o_.backoff_settling=false;
  o_.backoff_deadline=target_.expires_at;o_.backoff_plan=target_.backoff_plan;
  const auto& r=target_.backoff_options;
  o_.backoff_best_progress.setZero();
  const auto invalid=backoff_start_observation_reason(c_,s,r,o_.backoff_plan,o_.backoff_best_progress);
  if(invalid!=Reason::None) {stop(invalid,now,true);return;}
  // Only the preselected joints move. Unselected targets are the first actual
  // callback's measured angles, not a remote or nominal-home reference.
  o_.backoff_plan.start=s.q;
  for(int i=0;i<7;i++) if(o_.backoff_plan.direction[i]==0) o_.backoff_plan.goal[i]=s.q[i];
  target_.backoff_plan=o_.backoff_plan;
  o_.backoff_torque_limits=backoff_torque_limits(c_,r,o_.backoff_plan);
  reset_reference(s);
  ji_.max_velocity.fill(r.speed);
  ji_.max_acceleration.fill(std::min(.05,c_.joint_acceleration));
  ji_.max_jerk.fill(std::min(.5,c_.joint_jerk));
  for(int i=0;i<7;i++) ji_.target_position[i]=o_.backoff_plan.goal[i];
  ji_.target_velocity.fill(0);ji_.target_acceleration.fill(0);
  backoff_started_at_=now;backoff_settled_at_=-1;
}
void Controller::advance_joint_limit_backoff(const Sample& s,double now,double dt) {
  const auto& r=target_.backoff_options;const auto& p=o_.backoff_plan;
  const auto invalid=backoff_observation_reason(c_,s,r,p,o_.backoff_best_progress,false,true);
  if(invalid!=Reason::None) {stop(invalid,now,true);return;}
  if(now==backoff_started_at_) return;
  joints_.delta_time=dt;
  if(joints_.update(ji_,joint_output_)<0) {stop(Reason::ReferenceError,now,true);return;}
  joint_output_.pass_to_input(ji_);
  const V7 next=Eigen::Map<V7>(joint_output_.new_position.data());
  const V7 velocity=Eigen::Map<V7>(joint_output_.new_velocity.data());
  for(int i=0;i<7;i++) {
    const double d=p.direction[i];
    if((d==0 && std::abs(next[i]-p.start[i])>1e-10) ||
       (d!=0 && (d*(next[i]-o_.reference_joints[i]) < -1e-10 ||
        d*(p.goal[i]-next[i]) < -1e-10 || std::abs(velocity[i])>r.speed+1e-9))) {
      stop(Reason::ReferenceError,now,true);return;
    }
    if(d!=0 && now-backoff_started_at_>=2 && d*(next[i]-p.start[i])>=.02 && d*(s.q[i]-p.start[i])<.002) {
      stop(Reason::BackoffStalled,now,true);return;
    }
  }
  o_.reference_joints=next;o_.reference_joint_velocity=velocity;
  for(int i=0;i<7;i++) if(p.direction[i]!=0 && p.direction[i]*(s.q[i]-p.goal[i])>=0)
    o_.reference_joint_velocity[i]=0; // No velocity drive farther inward after passing the goal.
  const bool reference_done=(next-p.goal).cwiseAbs().maxCoeff()<1e-6 && velocity.cwiseAbs().maxCoeff()<1e-4;
  o_.backoff_settling=reference_done && backoff_goal_reached(c_,s,r,p) &&
      s.dq.cwiseAbs().maxCoeff()<std::min(.01,c_.stop_velocity);
  if(o_.backoff_settling) {
    if(backoff_settled_at_<0) backoff_settled_at_=now;
    if(now-backoff_settled_at_>=.2) {o_.backoff_reached=true;stop(Reason::BackoffReached,now,false);}
  } else backoff_settled_at_=-1;
}
Output Controller::step(const Sample& s, const Command& incoming, bool stop_requested, double now, double dt) {
  if(o_.finished) return o_;
  // Loaded startup owns its original options/deadline before the first callback,
  // even if mailbox delivery is delayed or the request has already expired.
  const Command& cmd=action_pending_?target_:incoming;
  const bool active_budget=o_.phase==Phase::Armed || o_.phase==Phase::Active;
  o_.planned_joint_velocity.setZero();o_.friction_compensation_torque.setZero();
  if(!s.finite() || !s.kinematics_valid || (operational_space_ && !s.dynamics_valid) ||
     !std::isfinite(now) || !std::isfinite(dt) || dt<0) {
    filtered_friction_torque_.setZero();
    o_.phase=Phase::Fault; o_.reason=Reason::InvalidState; o_.finished=true;
    o_.relief_released=false;o_.backoff_reached=false;
    if(o_.stop_trigger==Reason::None) o_.stop_trigger=Reason::InvalidState;
    return o_; // terminate FCI control; never calculate with a corrupt sample
  }
  if(dt==0) dt=.001; // documented first FCI callback
  const bool accepting=cmd.epoch==o_.epoch && cmd.sequence>o_.sequence && cmd.expires_at>now;
  const bool relief_session=o_.relief_mode || (accepting && cmd.relieve_contact);
  const bool backoff_session=o_.backoff_mode || (accepting && cmd.backoff_joint_limits);
  if(accepting && (((o_.relief_mode || o_.backoff_mode) && !action_pending_) ||
      ((cmd.relieve_contact || cmd.backoff_joint_limits) && have_target_)))
    stop(Reason::ReferenceError,now,true); // No supersession or entry from a running stream.
  if(accepting && cmd.backoff_joint_limits &&
     (cmd.relieve_contact || cmd.finish_at_target || cmd.mode!=Mode::Joint || cmd.has_orientation || cmd.has_posture ||
      !cmd.twist.isZero(0) || !cmd.acceleration.isZero(0) || !cmd.joint_velocity.isZero(0) ||
      !backoff_plan_valid(c_,cmd.backoff_options,cmd.backoff_plan) || !std::isfinite(cmd.action_timeout) ||
      cmd.action_timeout<=.2 || cmd.action_timeout>30 || !std::isfinite(cmd.expires_at) ||
      cmd.expires_at>now+cmd.action_timeout)) stop(Reason::ReferenceError,now,true);
  if(accepting && cmd.relieve_contact &&
     (cmd.mode!=Mode::Cartesian || cmd.backoff_joint_limits || cmd.finish_at_target || cmd.has_posture || cmd.has_orientation ||
      !cmd.twist.isZero(0) || !cmd.acceleration.isZero(0) || !cmd.joint_velocity.isZero(0) ||
      !relief_options_valid(c_,cmd.relief_options) || !std::isfinite(cmd.action_timeout) ||
      cmd.action_timeout<=.2 || cmd.action_timeout>10 || !std::isfinite(cmd.expires_at) ||
      cmd.expires_at>now+cmd.action_timeout)) stop(Reason::ReferenceError,now,true);
  o_.effort_limited=false;
  o_.reference_limited=o_.task_force_limited=o_.task_moment_limited=o_.joint_torque_limited=false;
  o_.reference_speed_limited=false;
  o_.osc_regularized=false;
  o_.spring_wrench.setZero();
  o_.joint_spring_torque.setZero();o_.joint_feedback_torque.setZero();
  o_.joint_spring_limited=o_.joint_feedback_limited=false;
  o_.progressive_force_saturated=o_.progressive_moment_saturated=false;
  if(dt>c_.maximum_control_period) stop(Reason::Timing,now,true);
  const double f=s.external_wrench.head<3>().norm(), m=s.external_wrench.tail<3>().norm();
  // libfranka collision thresholds apply per stiffness-frame axis. setK(identity)
  // puts that frame at the TCP; O_F_ext_hat_K is expressed in base coordinates.
  // Rotate before comparing so diagonal loads do not trip a stricter norm check.
  const V3 contact_force=s.orientation.conjugate()*V3(s.external_wrench.head<3>());
  const V3 contact_moment=s.orientation.conjugate()*V3(s.external_wrench.tail<3>());
  const bool contact_guards=c_.collision_policy=="configured" && !relief_session && !backoff_session;
  if(relief_session) {
    if(!s.wrench_tcp_valid) stop(Reason::ReliefInvalidWrench,now,true);
    else if(!s.relief_setup_valid) stop(Reason::ReliefSetupMismatch,now,true);
    else {
      if(s.external_wrench_tcp_base.head<3>().norm()>=std::min(40.,c_.contact_force_stop)) stop(Reason::ForceLimit,now,true);
      if(s.external_wrench_tcp_base.tail<3>().norm()>=std::min(20.,c_.contact_moment_stop)) stop(Reason::MomentLimit,now,true);
    }
    if((s.external_torque.cwiseAbs().array()>=c_.external_torque_limits.array()).any()) stop(Reason::JointContact,now,true);
  }
  o_.contact_load_ratio=c_.collision_policy!="robot"?std::max({
      (continuous_impedance_?contact_force.cwiseAbs().maxCoeff():f)/c_.contact_force_stop,
      (continuous_impedance_?contact_moment.cwiseAbs().maxCoeff():m)/c_.contact_moment_stop,
      (s.external_torque.cwiseAbs().array()/c_.external_torque_limits.array()).maxCoeff()}):0.0;
  o_.peak_contact_load_ratio=std::max(o_.peak_contact_load_ratio,o_.contact_load_ratio);
  if(contact_guards) {
    const auto invalid=ordinary_hard_contact_reason(c_,s);
    if(invalid!=Reason::None) stop(invalid,now,true);
  }
  if(backoff_session) {
    const auto& action=o_.backoff_mode?target_:cmd;
    const auto& plan=o_.backoff_mode?o_.backoff_plan:cmd.backoff_plan;
    const auto invalid=have_target_?
        backoff_observation_reason(c_,s,action.backoff_options,plan,o_.backoff_best_progress,false,active_budget):
        backoff_start_observation_reason(c_,s,action.backoff_options,plan,o_.backoff_best_progress);
    if(invalid!=Reason::None) stop(invalid,now,true);
    else update_backoff_progress(s,plan,o_.backoff_best_progress); // Includes monitored stopping.
  } else if((s.q.array()<=c_.joint_lower.array()+c_.joint_limit_margin).any() ||
            (s.q.array()>=c_.joint_upper.array()-c_.joint_limit_margin).any())
    stop(Reason::JointLimit,now,true);
  J67 scaled=s.jacobian;
  scaled.bottomRows<3>()*=c_.jacobian_rotation_scale;
  Eigen::SelfAdjointEigenSolver<M6> eigen(scaled*scaled.transpose(),Eigen::EigenvaluesOnly);
  o_.singular_value=std::sqrt(std::max(0.0,eigen.eigenvalues()[0]));
  const Mode requested_mode=accepting?cmd.mode:o_.mode;
  if(!automatic_mode_ && ((operational_space_ && requested_mode!=Mode::Cartesian) ||
     (joint_impedance_ && requested_mode!=Mode::Joint))) stop(Reason::ReferenceError,now,true);
  const bool requested_joint_impedance=joint_impedance_ || (automatic_mode_ && requested_mode==Mode::Joint);
  const bool regularized_cartesian=continuous_impedance_ && requested_mode==Mode::Cartesian;
  if(!requested_joint_impedance && (eigen.info()!=Eigen::Success ||
     (!regularized_cartesian && o_.singular_value<c_.minimum_singular_value)))
    stop(Reason::Singularity,now,true);
  const V6 velocity=s.jacobian*s.dq;
  if(relief_session) {
    const auto& r=o_.relief_mode?target_.relief_options:cmd.relief_options;
    if(o_.relief_mode && r.manual_stop && (s.q-o_.relief_anchor_joints).cwiseAbs().maxCoeff()>.2)
      stop(Reason::ReliefJointTravel,now,true);
    if(eigen.info()!=Eigen::Success || o_.singular_value<std::max(.05,c_.minimum_singular_value))
      stop(Reason::Singularity,now,true); // SDK wrench may be zero near singularities.
    if(s.dq.cwiseAbs().maxCoeff()>.15 || velocity.head<3>().norm()>1.5*r.translation_speed ||
       velocity.tail<3>().norm()>1.5*r.rotation_speed) stop(Reason::Overspeed,now,true);
  }
  if((s.dq.cwiseAbs().array()>c_.joint_speed_limits().array()).any() ||
     velocity.head<3>().norm()>c_.translation_speed_limit() ||
     velocity.tail<3>().norm()>c_.rotation_speed_limit())
    stop(Reason::Overspeed,now,true);
  if(!backoff_session && c_.motion_limit_policy=="franka" &&
     ((s.dq.array()>fr3_limits::directional_speed(s.q,true).array()).any() ||
      (s.dq.array() < -fr3_limits::directional_speed(s.q,false).array()).any()))
    stop(Reason::Overspeed,now,true);
  if(stop_requested) stop(Reason::UserStop,now,false);
  // Old sessions, duplicate commands and expired samples cannot refresh the watchdog.
  if(o_.phase!=Phase::Stopping && cmd.epoch==o_.epoch && cmd.sequence>o_.sequence && cmd.expires_at>now) {
    if(!have_target_ || cmd.mode!=o_.mode) {
      reset_reference(s);
      if(automatic_mode_) posture_=s.q; // No stale secondary objective from the previous mode.
    }
    if(o_.homing && !cmd.finish_at_target) o_.home_superseded=true;
    o_.homing=cmd.finish_at_target;
    o_.home_reached=false;home_settled_at_=-1;
    if(cmd.finish_at_target) {
      o_.home_sequence=cmd.sequence;o_.home_deadline=cmd.expires_at;o_.home_superseded=false;
    }
    target_=cmd;
    if(!cmd.has_orientation) target_.orientation=o_.reference_orientation;
    if(cmd.has_posture) posture_=cmd.posture;
    o_.mode=cmd.mode; o_.sequence=cmd.sequence; have_target_=true; o_.phase=Phase::Active;
    if(cmd.relieve_contact) {action_pending_=false;begin_contact_relief(s,now);}
    if(cmd.backoff_joint_limits) {action_pending_=false;begin_joint_limit_backoff(s,now);}
  }
  const double deadline=(have_target_ || action_pending_) ? target_.expires_at : armed_at_+c_.command_timeout;
  o_.command_age=std::max(0.0,now-(have_target_ ? target_.accepted_at : armed_at_));
  if(now>=deadline) {
    if(o_.relief_mode || o_.backoff_mode) {
      // Finite action expiry does not reclassify an already-ended action during braking.
      if(o_.phase!=Phase::Stopping) stop(o_.backoff_mode?Reason::BackoffTimeout:Reason::ReliefTimeout,now,true);
    } else stop(o_.homing?Reason::HomeTimeout:Reason::Watchdog,now,false);
  }
  // The law is fixed for this armed session. Continuous impedance does not
  // use estimated contact to taper its spring or change its reference.
  double drive_scale=1.0;
  if(continuous_impedance_) {
    o_.yielding=false;
  } else {
    const double load=std::max(f/c_.contact_force_soft,m/c_.contact_moment_soft);
    o_.yielding=load>=1.0 || (o_.yielding && load>0.8);
    drive_scale=std::clamp((1.0-load)/0.3,0.0,1.0);
  }
  o_.drive_scale=drive_scale;
  V7 desired;
  o_.task_wrench.setZero();
  if(o_.phase==Phase::Stopping || o_.phase==Phase::Armed) {
    reset_reference(s);
    desired=s.coriolis-c_.stop_damping.cwiseProduct(s.dq);
  } else {
    if(o_.backoff_mode) advance_joint_limit_backoff(s,now,dt);
    else if(o_.relief_mode) advance_contact_relief(s,now,dt);
    else if(o_.yielding) reset_reference(s);
    else if(!advance_reference(s,dt)) stop(Reason::ReferenceError,now,true);
    if(o_.homing && o_.phase==Phase::Active) {
      const bool joint=o_.mode==Mode::Joint;
      const bool at_goal=joint?
          (s.q-target_.joints).cwiseAbs().maxCoeff()<=target_.completion_joint_tolerance:
          ((s.position-target_.position).norm()<=target_.completion_position_tolerance &&
           rotation_error(target_.orientation,s.orientation).norm()<=target_.completion_rotation_tolerance);
      const bool reference_done=joint?
          ((o_.reference_joints-target_.joints).cwiseAbs().maxCoeff()<1e-5 && o_.reference_joint_velocity.norm()<1e-3):
          ((o_.reference_position-target_.position).norm()<1e-5 &&
           rotation_error(target_.orientation,o_.reference_orientation).norm()<1e-4 && o_.reference_twist.norm()<1e-3);
      if(at_goal && reference_done && !o_.yielding && s.dq.cwiseAbs().maxCoeff()<c_.stop_velocity) {
        if(home_settled_at_<0) home_settled_at_=now;
        if(now-home_settled_at_>=target_.completion_settle_time) {
          o_.home_reached=true;
          stop(Reason::HomeReached,now,false);
        }
      } else home_settled_at_=-1;
    }
    if(o_.phase==Phase::Stopping) {
      desired=s.coriolis-c_.stop_damping.cwiseProduct(s.dq);
    } else if(direct_joint_mode()) {
      joint_impedance_effort(c_,s,o_,o_.backoff_mode?&o_.backoff_plan.direction:nullptr,
                             o_.backoff_mode?&o_.backoff_torque_limits:nullptr);
      desired=s.coriolis+o_.joint_feedback_torque;
      if(o_.backoff_mode) for(int i=0;i<7;i++) if(o_.backoff_plan.direction[i]!=0) {
        const double raw=desired[i],cap=o_.backoff_torque_limits[i];
        desired[i]=std::clamp(raw,-cap,cap);
        if(desired[i]!=raw) o_.effort_limited=o_.joint_torque_limited=true;
      }
    } else {
      const J67 projection_jacobian=regularized_cartesian?scaled:s.jacobian;
      Eigen::LDLT<M6> solve;
      if(!operational_space_) {
        M6 gram=projection_jacobian*projection_jacobian.transpose();
        if(regularized_cartesian)
          gram.diagonal().array()+=c_.minimum_singular_value*c_.minimum_singular_value;
        solve.compute(gram);
      }
      V7 null_torque;
      if(o_.mode==Mode::Cartesian) {
        V6 e;
        e.head<3>()=o_.reference_position-s.position;
        e.tail<3>()=rotation_error(o_.reference_orientation,s.orientation);
        const V7 post=(o_.relief_mode?0.:c_.posture_stiffness)*(posture_-s.q)-c_.posture_damping*s.dq;
        if(operational_space_) {
          V6 acceleration=o_.reference_acceleration;
          if(!bounded_equilibrium_) {
            acceleration.head<3>()+=c_.osc_translation_kp.cwiseProduct(e.head<3>())+
                c_.osc_translation_kd.cwiseProduct(V3(o_.reference_twist.head<3>()-velocity.head<3>()));
            acceleration.tail<3>()+=c_.osc_rotation_kp.cwiseProduct(e.tail<3>())+
                c_.osc_rotation_kd.cwiseProduct(V3(o_.reference_twist.tail<3>()-velocity.tail<3>()));
          }
          if(!operational_space_effort(s,c_,acceleration,post,o_.task_wrench,null_torque,o_.osc_regularized)) {
            stop(Reason::InvalidState,now,true);
            o_.task_wrench.setZero();null_torque.setZero();
          } else if(bounded_equilibrium_) {
            if(c_.osc_feedback_mode=="saturated") {
              o_.spring_wrench.head<3>()=saturated_spring(V3(e.head<3>()),c_.osc_translation_stiffness,
                  c_.commanded_force_limit,o_.progressive_force_saturated);
              o_.spring_wrench.tail<3>()=saturated_spring(V3(e.tail<3>()),c_.osc_rotation_stiffness,
                  c_.commanded_moment_limit,o_.progressive_moment_saturated);
            } else {
              o_.spring_wrench.head<3>()=progressive_spring(V3(e.head<3>()),
                  c_.progressive_position_soft_radius,c_.progressive_position_saturation_radius,
                  c_.progressive_translation_soft_stiffness,c_.commanded_force_limit,o_.progressive_force_saturated);
              o_.spring_wrench.tail<3>()=progressive_spring(V3(e.tail<3>()),
                  c_.progressive_rotation_soft_radius,c_.progressive_rotation_saturation_radius,
                  c_.progressive_rotation_soft_stiffness,c_.commanded_moment_limit,o_.progressive_moment_saturated);
            }
            // Replace ALL linear restoring feedback; no Lambda*Kp*e or old
            // physical stiffness can bypass the curve. Preserve damping after
            // spring saturation, then cap the combined task wrench below.
            o_.task_wrench+=o_.spring_wrench;
            o_.task_wrench.head<3>()+=c_.osc_translation_damping.cwiseProduct(
                V3(o_.reference_twist.head<3>()-velocity.head<3>()));
            o_.task_wrench.tail<3>()+=c_.osc_rotation_damping.cwiseProduct(
                V3(o_.reference_twist.tail<3>()-velocity.tail<3>()));
            o_.task_force_limited=o_.progressive_force_saturated;
            o_.task_moment_limited=o_.progressive_moment_saturated;
          } else {
            // The ideal inverse-dynamics law alone can request negligible
            // restoring torque in low-inertia directions. Add a physical
            // stiffness/damper without cancelling measured external forces.
            o_.task_wrench.head<3>()+=c_.osc_translation_stiffness.cwiseProduct(e.head<3>())+
                c_.osc_translation_damping.cwiseProduct(V3(o_.reference_twist.head<3>()-velocity.head<3>()));
            o_.task_wrench.tail<3>()+=c_.osc_rotation_stiffness.cwiseProduct(e.tail<3>())+
                c_.osc_rotation_damping.cwiseProduct(V3(o_.reference_twist.tail<3>()-velocity.tail<3>()));
          }
        } else {
          V3 spring_force=c_.translation_stiffness.cwiseProduct(e.head<3>());
          V3 spring_moment=c_.rotation_stiffness.cwiseProduct(e.tail<3>());
          if(c_.bounded_cartesian_spring) {
            // Preserve damping at large errors: clipping K*e - D*v only after
            // summation can mask D*v behind a saturated restoring spring.
            o_.task_force_limited=spring_force.norm()>c_.commanded_force_limit;
            o_.task_moment_limited=spring_moment.norm()>c_.commanded_moment_limit;
            spring_force=capped(spring_force,c_.commanded_force_limit);
            spring_moment=capped(spring_moment,c_.commanded_moment_limit);
          }
          o_.task_wrench.head<3>()=drive_scale*(spring_force+
              c_.translation_damping.cwiseProduct(o_.reference_twist.head<3>()))-
              c_.translation_damping.cwiseProduct(velocity.head<3>());
          o_.task_wrench.tail<3>()=drive_scale*(spring_moment+
              c_.rotation_damping.cwiseProduct(o_.reference_twist.tail<3>()))-
              c_.rotation_damping.cwiseProduct(velocity.tail<3>());
          null_torque=post-projection_jacobian.transpose()*solve.solve(projection_jacobian*post);
        }
      } else {
        const V7 raw=drive_scale*(c_.joint_stiffness.cwiseProduct(o_.reference_joints-s.q)+
            c_.joint_damping.cwiseProduct(Eigen::Map<V7>(ji_.current_velocity.data())))-
            c_.joint_damping.cwiseProduct(s.dq);
        // Static task/nullspace decomposition, not a measured contact-force bound.
        o_.task_wrench=solve.solve(s.jacobian*raw);
        null_torque=raw-s.jacobian.transpose()*o_.task_wrench;
      }
      if(c_.effort_limit_policy=="configured" || bounded_equilibrium_) {
        const V6 before=o_.task_wrench;
        o_.task_force_limited=o_.task_force_limited || before.head<3>().norm()>c_.commanded_force_limit;
        o_.task_moment_limited=o_.task_moment_limited || before.tail<3>().norm()>c_.commanded_moment_limit;
        o_.task_wrench.head<3>()=capped(V3(before.head<3>()),c_.commanded_force_limit);
        o_.task_wrench.tail<3>()=capped(V3(before.tail<3>()),c_.commanded_moment_limit);
        o_.effort_limited=o_.effort_limited || o_.task_force_limited || o_.task_moment_limited ||
            (before-o_.task_wrench).norm()>1e-9 || drive_scale<1;
        if(c_.effort_limit_policy=="configured")
          null_torque=drive_scale*capped(null_torque,c_.nullspace_torque_limit);
      }
      if(o_.phase==Phase::Stopping) desired=s.coriolis-c_.stop_damping.cwiseProduct(s.dq);
      else desired=s.coriolis+s.jacobian.transpose()*o_.task_wrench+null_torque;
    }
  }
  if(o_.phase==Phase::Active && !o_.yielding && !o_.relief_mode && !o_.backoff_mode && c_.friction_compensation_torque>0) {
    // Use the accepted target error, not measured dq or target finite differences:
    // a stationary joint still needs breakaway torque to reach a held target.
    constexpr double horizon=0.100;
    bool valid=true;
    if(o_.mode==Mode::Joint) {
      o_.planned_joint_velocity=(target_.joints-s.q)/horizon;
    } else {
      V6 error;
      error.head<3>()=target_.position-s.position;
      error.tail<3>()=c_.jacobian_rotation_scale*rotation_error(target_.orientation,s.orientation);
      // Geometric damped least squares, using the existing scaled Jacobian and
      // geometric singularity regularizer (not the OSC inverse-inertia scale).
      M6 gram=scaled*scaled.transpose();
      gram.diagonal().array()+=c_.minimum_singular_value*c_.minimum_singular_value;
      Eigen::LDLT<M6> solve(gram);
      valid=solve.info()==Eigen::Success && (solve.vectorD().array()>0).all();
      if(valid) o_.planned_joint_velocity=scaled.transpose()*solve.solve(error/horizon);
    }
    if(!valid || !o_.planned_joint_velocity.allFinite()) {
      o_.planned_joint_velocity.setZero();filtered_friction_torque_.setZero();
      stop(Reason::InvalidState,now,true);
      desired=s.coriolis-c_.stop_damping.cwiseProduct(s.dq);
    } else {
      V7 raw;
      for(int i=0;i<7;i++) raw[i]=c_.friction_compensation_torque*
          std::tanh(o_.planned_joint_velocity[i]/c_.friction_compensation_epsilon);
      if(c_.friction_compensation_filter_time_s==0) filtered_friction_torque_=raw;
      else {
        // Exact first-order update for held input, using the actual callback dt.
        // Only compensation is filtered, never impedance feedback or damping.
        const double alpha=-std::expm1(-dt/c_.friction_compensation_filter_time_s);
        filtered_friction_torque_+=alpha*(raw-filtered_friction_torque_);
      }
      // Apply contact taper AFTER filtering so a falling drive_scale is immediate.
      o_.friction_compensation_torque=drive_scale*filtered_friction_torque_;
      desired+=o_.friction_compensation_torque;
    }
  } else filtered_friction_torque_.setZero(); // No filter tail through stop/yield/off.
  if(!desired.allFinite()) {
    filtered_friction_torque_.setZero();
    o_.planned_joint_velocity.setZero();o_.friction_compensation_torque.setZero();
    o_.phase=Phase::Fault; o_.reason=Reason::InvalidState; o_.finished=true;
    o_.relief_released=false;o_.backoff_reached=false;
    if(o_.stop_trigger==Reason::None) o_.stop_trigger=Reason::InvalidState;
    return o_;
  }
  // FCI's last applied command, rather than an assumed previous output, is the
  // slew baseline. Filtering or lost packets can otherwise hide discontinuities.
  for(int i=0;i<7;i++) {
    const double bounded=std::clamp(desired[i],-c_.torque_limits[i],c_.torque_limits[i]);
    o_.torque[i]=s.previous_torque[i]+std::clamp(bounded-s.previous_torque[i],
          -c_.torque_rate_limit*dt,c_.torque_rate_limit*dt);
    o_.effort_limited=o_.effort_limited || std::abs(o_.torque[i]-desired[i])>1e-9;
    o_.joint_torque_limited=o_.joint_torque_limited || std::abs(o_.torque[i]-desired[i])>1e-9;
  }
  if(o_.phase==Phase::Stopping) {
    if(o_.relief_mode) {
      o_.relief_translation=(s.position-o_.relief_anchor_position).norm();
      o_.relief_rotation=rotation_error(s.orientation,o_.relief_anchor_orientation).norm();
      const auto invalid=contact_relief_observation_reason(c_,s,target_.relief_options,
          o_.relief_anchor_position,o_.relief_anchor_orientation,false,&o_.relief_anchor_joints);
      if(invalid!=Reason::None) stop(invalid,now,true);
    }
    if(o_.backoff_mode) {
      const auto invalid=backoff_observation_reason(c_,s,target_.backoff_options,o_.backoff_plan,o_.backoff_best_progress);
      if(invalid!=Reason::None) stop(invalid,now,true);
    }
    const double stationary=o_.backoff_mode?std::min(.01,c_.stop_velocity):
        (o_.relief_mode?std::min(.02,c_.stop_velocity):c_.stop_velocity);
    if(s.dq.cwiseAbs().maxCoeff()<stationary && (o_.torque-desired).norm()<0.1) {
      if(settled_at_<0) settled_at_=now;
      if(now-settled_at_>=c_.stop_settle_time) {
        if(o_.relief_mode && o_.relief_released) {
          const auto invalid=contact_relief_observation_reason(c_,s,target_.relief_options,
              o_.relief_anchor_position,o_.relief_anchor_orientation,true,&o_.relief_anchor_joints);
          if(invalid!=Reason::None) stop(invalid,now,true);
        }
        if(o_.backoff_mode && o_.backoff_reached) {
          const auto invalid=backoff_observation_reason(c_,s,target_.backoff_options,o_.backoff_plan,o_.backoff_best_progress,true);
          if(invalid!=Reason::None) stop(invalid,now,true);
        }
        o_.finished=true; o_.phase=fault_?Phase::Fault:Phase::Disarmed;
      }
    } else settled_at_=-1;
    if(now-stopping_at_>c_.stop_timeout) {
      o_.finished=true; o_.phase=Phase::Fault; o_.reason=Reason::StopTimeout;o_.relief_released=false;o_.backoff_reached=false;
    }
  }
  return o_;
}
} // namespace fcc
