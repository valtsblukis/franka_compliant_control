#pragma once
#include "controller.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>

namespace fcc {
// Fixed-size solves only. S nondimensionalizes translation/rotation for the
// singularity regularizer. Away from that regularizer this is exact OSC:
// tau = c + J^T Lambda (a_cmd - Jdot*dq) + (I-J^T Jbar^T) tau_posture.
// No measured-force cancellation: nominal tracking is inertia normalized, but
// contact acceleration still depends on the actual operational-space inertia.
inline bool operational_space_effort(const Sample& s, const Config& c,
                                     const V6& acceleration, const V7& posture,
                                     V6& wrench, V7& null_torque, bool& regularized) {
  if(!s.dynamics_valid || !s.mass.allFinite() || !s.jacobian_dot_velocity.allFinite() ||
     !s.mass.isApprox(s.mass.transpose(),1e-10)) return false;
  Eigen::LDLT<M7> mass(s.mass);
  if(mass.info()!=Eigen::Success || (mass.vectorD().array()<=0).any()) return false;
  J67 js=s.jacobian;
  js.bottomRows<3>()*=c.jacobian_rotation_scale;
  const Eigen::Matrix<double,7,6> minv_jt=mass.solve(js.transpose());
  M6 inverse_inertia=js*minv_jt;
  Eigen::SelfAdjointEigenSolver<M6> eigen(inverse_inertia,Eigen::EigenvaluesOnly);
  if(eigen.info()!=Eigen::Success) return false;
  // Exact inverse outside the deficient region, continuous damping inside it.
  const double damping=std::max(0.0,c.osc_regularization*c.osc_regularization-eigen.eigenvalues()[0]);
  regularized=damping>0;
  inverse_inertia.diagonal().array()+=damping;
  Eigen::LDLT<M6> task(inverse_inertia);
  if(task.info()!=Eigen::Success || (task.vectorD().array()<=0).any()) return false;
  V6 rhs=acceleration-s.jacobian_dot_velocity;
  rhs.tail<3>()*=c.jacobian_rotation_scale;
  wrench=task.solve(rhs);
  wrench.tail<3>()*=c.jacobian_rotation_scale;
  null_torque=posture-js.transpose()*task.solve(js*mass.solve(posture));
  return wrench.allFinite() && null_torque.allFinite();
}
} // namespace fcc
