#pragma once
#include "backend.hpp"
#include <franka/exception.h>
#include <iomanip>
#include <locale>
#include <sstream>

namespace fcc {
// Same capacity as libfranka's documented default; no change to the RT logger's
// allocation/copy workload. The SDK may not provide the exact trigger sample.
constexpr std::size_t kExceptionLogCapacity=50;

// Fixed-size, no model calls and no strings. Also used to seed normal samples.
inline Sample observation_sample(const franka::RobotState& rs) {
  Sample s;
  s.q=Eigen::Map<const V7>(rs.q.data());s.dq=Eigen::Map<const V7>(rs.dq.data());
  s.previous_torque=Eigen::Map<const V7>(rs.tau_J_d.data());
  s.external_torque=Eigen::Map<const V7>(rs.tau_ext_hat_filtered.data());
  const Eigen::Map<const Eigen::Matrix4d> pose(rs.O_T_EE.data());
  s.position=pose.topRightCorner<3,1>();
  s.orientation=Eigen::Quaterniond(pose.topLeftCorner<3,3>()).normalized();
  s.external_wrench=Eigen::Map<const V6>(rs.O_F_ext_hat_K.data());
  s.external_wrench_stiffness=Eigen::Map<const V6>(rs.K_F_ext_hat_K.data());
  s.cartesian_collision=Eigen::Map<const V6>(rs.cartesian_collision.data());
  s.joint_collision=Eigen::Map<const V7>(rs.joint_collision.data());
  s.communication_success=rs.control_command_success_rate;
  s.robot_mode=static_cast<int>(rs.robot_mode);
  s.robot_has_errors=static_cast<bool>(rs.current_errors);
  s.robot_time=rs.time.toSec();s.kinematics_valid=false;
  return s;
}
// SDK robot-on-environment wrench about the reported EE, in base axes. The
// full K->EE moment shift is essential if EE_T_K is not identity. Callers must
// separately verify that reported EE is the configured TCP before labeling it TCP.
inline bool sdk_wrench_at_ee_base(const franka::RobotState& rs,V6& wrench) {
  const Eigen::Map<const Eigen::Matrix4d> o_ee(rs.O_T_EE.data()),ee_k(rs.EE_T_K.data());
  const Eigen::Map<const V6> k(rs.K_F_ext_hat_K.data());
  if(!o_ee.allFinite() || !ee_k.allFinite() || !k.allFinite()) return false;
  const Eigen::Matrix3d r=o_ee.topLeftCorner<3,3>(),rk=ee_k.topLeftCorner<3,3>();
  if(!o_ee.row(3).isApprox(Eigen::RowVector4d(0,0,0,1),1e-6) ||
     !ee_k.row(3).isApprox(Eigen::RowVector4d(0,0,0,1),1e-6) ||
     !(r.transpose()*r).isApprox(Eigen::Matrix3d::Identity(),1e-6) ||
     !(rk.transpose()*rk).isApprox(Eigen::Matrix3d::Identity(),1e-6) ||
     std::abs(r.determinant()-1)>1e-6 || std::abs(rk.determinant()-1)>1e-6) return false;
  wrench.head<3>()=r*rk*V3(k.head<3>());
  wrench.tail<3>()=r*rk*V3(k.tail<3>())+(r*V3(ee_k.topRightCorner<3,1>())).cross(V3(wrench.head<3>()));
  return wrench.allFinite();
}
// Loaded relief may not change its bias/estimator basis. Check the existing
// setup, including payload (not just Desk end-effector), with fixed-size math.
inline bool relief_hardware_setup_matches(const HardwareConfig& h,const franka::RobotState& rs) {
  if(!h.check_end_effector) return false;
  const auto match=[](double a,double b,double tolerance) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tolerance;
  };
  if(!match(rs.m_ee,h.expected_ee_mass,1e-5) || !match(rs.m_load,h.load_mass,1e-5)) return false;
  for(int i=0;i<3;i++)
    if(!match(rs.F_x_Cee[i],h.expected_ee_com[i],1e-6) || !match(rs.F_x_Cload[i],h.load_com[i],1e-6)) return false;
  for(int i=0;i<9;i++)
    if(!match(rs.I_ee[i],h.expected_ee_inertia[i],1e-8) || !match(rs.I_load[i],h.load_inertia[i],1e-8)) return false;
  for(int i=0;i<16;i++) {
    const double identity=i%5==0?1.:0.;
    if(!match(rs.F_T_EE[i],h.flange_to_tcp[i],1e-6) || !match(rs.EE_T_K[i],identity,1e-6)) return false;
  }
  return true;
}
// Everything below runs outside the torque callback. The raw SDK serialization
// retains fields not repeated in the compact observation, at full precision.
inline RobotObservation capture_observation(const franka::RobotState& rs) {
  RobotObservation out;
  out.sample=observation_sample(rs);
  out.measured_torque=Eigen::Map<const V7>(rs.tau_J.data());
  out.base_to_ee=rs.O_T_EE;out.flange_to_ee=rs.F_T_EE;out.ee_to_stiffness=rs.EE_T_K;
  out.current_errors=static_cast<std::string>(rs.current_errors);
  out.last_motion_errors=static_cast<std::string>(rs.last_motion_errors);
  out.cartesian_reflex=rs.current_errors.cartesian_reflex;
  out.joint_reflex=rs.current_errors.joint_reflex;
  std::ostringstream raw;raw.imbue(std::locale::classic());raw<<std::setprecision(17)<<rs;
  out.robot_state_json=raw.str();
  return out;
}
inline Reason classify_control_exception(const FaultDiagnostics& fault) {
  // Named SDK errors remain useful when the SDK supplied no records. Do not
  // use stale last_motion_errors from a previous, explicitly recovered session.
  if(fault.message.find("\"cartesian_reflex\"")!=std::string::npos) return Reason::RobotCartesianReflex;
  if(fault.message.find("\"joint_reflex\"")!=std::string::npos) return Reason::RobotJointReflex;
  for(auto it=fault.records.rbegin();it!=fault.records.rend();++it) {
    if(it->state.cartesian_reflex) return Reason::RobotCartesianReflex;
    if(it->state.joint_reflex) return Reason::RobotJointReflex;
  }
  if(fault.message.find("motion aborted by reflex")!=std::string::npos)
    return Reason::RobotReflex;
  for(const auto& record:fault.records)
    if(record.state.sample.robot_mode==static_cast<int>(franka::RobotMode::kReflex))
      return Reason::RobotReflex;
  return Reason::RobotControlError;
}
inline std::shared_ptr<FaultDiagnostics> capture_control_exception(const franka::ControlException& error,
                                                                  double now) {
  auto out=std::make_shared<FaultDiagnostics>();
  out->exception_type="franka::ControlException";out->message=error.what();out->caught_at=now;
  out->records.reserve(error.log.size());
  for(const auto& record:error.log) {
    ExceptionRecord r;r.state=capture_observation(record.state);
    r.sent_torques=record.command.torques.tau_J;
    r.sent_joint_positions=record.command.joint_positions.q;
    r.sent_joint_velocities=record.command.joint_velocities.dq;
    r.sent_pose=record.command.cartesian_pose.O_T_EE;
    r.sent_twist=record.command.cartesian_velocities.O_dP_EE;
    out->records.push_back(std::move(r));
  }
  out->reason=classify_control_exception(*out);
  return out;
}
inline void latch_control_failure(Output& out, Reason reason) {
  out.phase=Phase::Fault;out.reason=reason;out.finished=true;
  out.planned_joint_velocity.setZero();out.friction_compensation_torque.setZero();
  if(out.relief_mode) out.relief_released=false;
  if(out.backoff_mode) out.backoff_reached=false;
  if(out.stop_trigger==Reason::None) out.stop_trigger=reason;
}
// A single existing-connection read; no stop/recovery/rearm/setter/reconnection.
// Its wall duration is governed by the SDK network timeout. Never replace the
// primary exception if this diagnostic read fails or returns an old timestamp.
template<class Read, class Clock>
void observe_after_fault(FaultDiagnostics& fault, double last_robot_time, Read read, Clock clock) {
  for(const auto& record:fault.records)
    last_robot_time=std::max(last_robot_time,record.state.sample.robot_time);
  fault.post_read_attempted=true;
  try {
    const auto rs=read();fault.post_read_at=clock();
    fault.post_observation=capture_observation(rs);
    fault.post_read_fresh=rs.time.toSec()>last_robot_time;
    fault.post_read_valid=fault.post_observation.sample.finite();
    if(!fault.post_read_fresh) fault.post_read_error="post-fault robot timestamp did not advance";
    else if(!fault.post_read_valid) fault.post_read_error="post-fault robot sample is nonfinite or invalid";
  } catch(const std::exception& error) {fault.post_read_error=error.what();}
  fault.post_read_completed=true;
}
} // namespace fcc
