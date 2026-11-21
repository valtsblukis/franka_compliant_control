#pragma once
#include "controller.hpp"
#include <algorithm>

namespace fcc {
// Direct diagonal joint impedance. Bound spring BEFORE adding velocity-error
// damping so damping remains effective at large displacements. No target reset,
// Cartesian projection, inertia shaping or contact-state switching.
inline void joint_impedance_effort(const Config& c, const Sample& s, Output& out,
                                   const V7* inward_direction=nullptr,const V7* action_limits=nullptr) {
  const V7 spring=c.joint_stiffness.cwiseProduct(out.reference_joints-s.q);
  for(int i=0;i<7;i++) {
    const double limit=action_limits?std::min(c.joint_feedback_torque_limits[i],(*action_limits)[i]):c.joint_feedback_torque_limits[i];
    const bool suppress_outward=inward_direction && (*inward_direction)[i]!=0 && (*inward_direction)[i]*spring[i]<0;
    out.joint_spring_torque[i]=suppress_outward?0.:std::clamp(spring[i],-limit,limit);
    out.joint_spring_limited=out.joint_spring_limited || suppress_outward || std::abs(spring[i])>limit;
    const double feedback=out.joint_spring_torque[i]+
        c.joint_damping[i]*(out.reference_joint_velocity[i]-s.dq[i]);
    out.joint_feedback_torque[i]=std::clamp(feedback,-limit,limit);
    out.joint_feedback_limited=out.joint_feedback_limited || std::abs(feedback)>limit;
  }
  out.effort_limited=out.effort_limited || out.joint_spring_limited || out.joint_feedback_limited;
}
} // namespace fcc
