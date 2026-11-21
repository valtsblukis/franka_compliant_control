#pragma once
#include "controller.hpp"
#include <franka/model.h>
#include <algorithm>

namespace fcc {
// Local kinematic directional derivative at measured q,dq; does not difference
// noisy samples in time. Perturbation norm <= 1e-5 rad, in the same TCP/base
// convention as zeroJacobian. No robot I/O or Python callback is involved.
inline void sample_dynamics(franka::Model& model, const franka::RobotState& rs, Sample& s) {
  const auto mass=model.mass(rs);
  s.mass=Eigen::Map<const M7>(mass.data());
  s.jacobian_dot_velocity.setZero();
  if(s.dq.squaredNorm()>0) {
    const double h=1e-5/std::max(1.0,s.dq.norm());
    auto plus=rs.q, minus=rs.q;
    for(int i=0;i<7;i++) {plus[i]+=h*s.dq[i];minus[i]-=h*s.dq[i];}
    const auto jp=model.zeroJacobian(franka::Frame::kEndEffector,plus,rs.F_T_EE,rs.EE_T_K);
    const auto jm=model.zeroJacobian(franka::Frame::kEndEffector,minus,rs.F_T_EE,rs.EE_T_K);
    s.jacobian_dot_velocity=((Eigen::Map<const J67>(jp.data())-Eigen::Map<const J67>(jm.data()))/(2*h))*s.dq;
  }
  s.dynamics_valid=true;
}
} // namespace fcc
