#pragma once
#include "controller.hpp"

namespace fcc {
inline bool automatically_recoverable(Reason reason) {
  switch(reason) {
    case Reason::None: // A robot-side reflex may be present after service startup.
    case Reason::ForceLimit: case Reason::MomentLimit: case Reason::JointContact:
    case Reason::JointLimit: case Reason::Singularity: case Reason::Overspeed:
    case Reason::RobotCartesianReflex: case Reason::RobotJointReflex: case Reason::RobotReflex:
      return true;
    default:return false;
  }
}
inline const char* automatic_recovery_blocker(const Config& c, const Sample& s,
                                             Reason reason, bool allowed_mode,
                                             bool collision_errors_only) {
  if(!automatically_recoverable(reason)) return "fault requires explicit recovery (control/timing/model/communication failure)";
  if(!s.finite()) return "robot state is invalid";
  if(!allowed_mode) return "robot is not Idle or Reflex; physical/user stop and guiding require operator action";
  if(!collision_errors_only) return "robot has non-collision errors requiring operator action";
  if(s.dq.cwiseAbs().maxCoeff()>=c.stop_velocity) return "robot is not stationary";
  if((s.q.array()<=c.joint_lower.array()+c.joint_limit_margin).any() ||
     (s.q.array()>=c.joint_upper.array()-c.joint_limit_margin).any())
    return "robot is outside configured joint margins";
  if(s.external_wrench.head<3>().norm()>=c.contact_force_soft ||
     s.external_wrench.tail<3>().norm()>=c.contact_moment_soft ||
     (s.external_torque.cwiseAbs().array()>=.7*c.external_torque_limits.array()).any())
    return "estimated load is not below unloaded/contact-reporting thresholds";
  return nullptr;
}
inline V7 nominal_home_joints() { return (V7()<<0,-.4,0,-1.9,0,1.5,0).finished(); }
} // namespace fcc
