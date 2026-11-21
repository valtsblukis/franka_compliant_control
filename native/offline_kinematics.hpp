#pragma once
#include "controller.hpp"
#include <franka/model.h>
#include <stdexcept>
#include <utility>

namespace fcc {
// URDF-only SDK model: no Robot construction, network connection or control.
// Called with the GIL held by the planning script, never in a torque callback.
class OfflineKinematics {
 public:
  OfflineKinematics(const std::string& urdf, const std::array<double,16>& flange_to_tcp)
      : model_(urdf), tcp_(flange_to_tcp) {
    const Eigen::Map<const Eigen::Matrix4d> t(tcp_.data());
    const Eigen::Matrix3d r=t.topLeftCorner<3,3>();
    if(!t.allFinite() || !t.row(3).isApprox(Eigen::RowVector4d(0,0,0,1),1e-9) ||
       !(r.transpose()*r).isApprox(Eigen::Matrix3d::Identity(),1e-7) ||
       std::abs(r.determinant()-1)>1e-7)
      throw std::invalid_argument("flange_to_tcp must be a finite column-major rigid transform");
  }
  std::pair<Eigen::Matrix4d,J67> evaluate(const V7& q) {
    if(!q.allFinite()) throw std::invalid_argument("kinematic joint positions must be finite");
    std::array<double,7> joints;Eigen::Map<V7>(joints.data())=q;
    const std::array<double,16> identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    const auto pose=model_.pose(franka::Frame::kEndEffector,joints,tcp_,identity);
    const auto jacobian=model_.zeroJacobian(franka::Frame::kEndEffector,joints,tcp_,identity);
    return {Eigen::Map<const Eigen::Matrix4d>(pose.data()),Eigen::Map<const J67>(jacobian.data())};
  }
 private:
  franka::Model model_;
  std::array<double,16> tcp_;
};
} // namespace fcc
