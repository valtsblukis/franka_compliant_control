// Offline-only bridge. Uses the production Controller and SDK model, never Robot.
#include "controller.hpp"
#include "model_dynamics.hpp"
#include "joint_limit_backoff.hpp"
#include <franka/lowpass_filter.h>
#include <franka/rate_limiting.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Eigenvalues>

namespace py = pybind11;
using namespace fcc;

namespace {
class OfflineModel {
 public:
  OfflineModel(const std::string& urdf, const std::array<double,16>& tcp,
               double mass, const std::array<double,3>& com,
               const std::array<double,9>& inertia) : model_(urdf) {
    const Eigen::Map<const Eigen::Matrix4d> t(tcp.data());
    const Eigen::Matrix3d r=t.topLeftCorner<3,3>();
    const Eigen::Map<const Eigen::Matrix3d> i(inertia.data());
    if(!t.allFinite() || !t.row(3).isApprox(Eigen::RowVector4d(0,0,0,1),1e-9) ||
       !(r.transpose()*r).isApprox(Eigen::Matrix3d::Identity(),1e-7) ||
       std::abs(r.determinant()-1)>1e-7 || !std::isfinite(mass) || mass<0 ||
       !Eigen::Map<const V3>(com.data()).allFinite() || !i.allFinite() ||
       !i.isApprox(i.transpose(),1e-10))
      throw std::invalid_argument("invalid offline TCP/load model");
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(i);
    if(eigen.info()!=Eigen::Success || eigen.eigenvalues().minCoeff()<-1e-12 ||
       eigen.eigenvalues()[2]>eigen.eigenvalues()[0]+eigen.eigenvalues()[1]+1e-10)
      throw std::invalid_argument("nonphysical offline load inertia");
    state_.F_T_EE=tcp;
    state_.EE_T_K={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    state_.m_total=mass; state_.F_x_Ctotal=com; state_.I_total=inertia;
  }
  Sample sample(const V7& q, const V7& dq, const V7& previous, const V6& wrench) {
    if(!q.allFinite() || !dq.allFinite() || !previous.allFinite() || !wrench.allFinite())
      throw std::invalid_argument("nonfinite offline state");
    Eigen::Map<V7>(state_.q.data())=q; Eigen::Map<V7>(state_.dq.data())=dq;
    Sample s; s.q=q; s.dq=dq; s.previous_torque=previous; s.external_wrench=wrench;
    s.external_wrench_tcp_base=wrench;s.wrench_tcp_valid=true;s.relief_setup_valid=true; // Explicit synthetic model/TCP wrench.
    const auto pose=model_.pose(franka::Frame::kEndEffector,state_);
    const Eigen::Map<const Eigen::Matrix4d> t(pose.data());
    s.position=t.topRightCorner<3,1>();
    s.orientation=Eigen::Quaterniond(t.topLeftCorner<3,3>()).normalized();
    const auto jac=model_.zeroJacobian(franka::Frame::kEndEffector,state_);
    s.jacobian=Eigen::Map<const J67>(jac.data());
    const auto coriolis=model_.coriolis(state_); s.coriolis=Eigen::Map<const V7>(coriolis.data());
    sample_dynamics(model_,state_,s); // Same loaded M and Jdot*dq as hardware.
    s.external_wrench_stiffness.head<3>()=s.orientation.conjugate()*V3(wrench.head<3>());
    s.external_wrench_stiffness.tail<3>()=s.orientation.conjugate()*V3(wrench.tail<3>());
    // Synthetic TCP-only load. This is not Franka's joint-torque observer.
    s.external_torque=s.jacobian.transpose()*wrench;
    return s;
  }
  V7 gravity(const V7& q) {
    if(!q.allFinite()) throw std::invalid_argument("nonfinite joints");
    std::array<double,7> joints; Eigen::Map<V7>(joints.data())=q;
    const auto g=model_.gravity(joints,state_.m_total,state_.F_x_Ctotal,{0,0,-9.81});
    return Eigen::Map<const V7>(g.data());
  }
 private:
  franka::Model model_;
  franka::RobotState state_{};
};

// Commands are constructed with the existing _native.Command binding. The
// offline clock is explicit; no thread, wall clock or HTTP loop is involved.
class OfflineController {
 public:
  OfflineController(const Config& c, const Sample& s, std::uint64_t epoch, double now)
      : controller_(c,s,epoch,now), accepted_at_(now) {}
  OfflineController(const Config& c,const Sample& s,std::uint64_t epoch,double now,
                    const Command& initial,double deadline)
      : initial_(initial_relief(c,s,initial,epoch,now,deadline)),
        controller_(c,s,epoch,now,&initial_),accepted_at_(now),
        last_epoch_(epoch),last_sequence_(initial_.sequence) {}
  Output step(const Sample& s, const Command& input, double now, double dt,
              double deadline, bool stop) {
    if(!std::isfinite(deadline) || !std::isfinite(now))
      throw std::invalid_argument("nonfinite offline command clock");
    Command cmd=input;
    cmd.expires_at=deadline;
    if(cmd.epoch!=last_epoch_ || cmd.sequence!=last_sequence_) {
      accepted_at_=now;last_epoch_=cmd.epoch;last_sequence_=cmd.sequence;
    }
    cmd.accepted_at=accepted_at_;
    return controller_.step(s,cmd,stop,now,dt);
  }
 private:
  static Command initial_relief(const Config& c,const Sample& s,const Command& input,
                               std::uint64_t epoch,double now,double deadline) {
    if(input.relieve_contact==input.backoff_joint_limits || !std::isfinite(now) || !std::isfinite(deadline) ||
       !std::isfinite(input.action_timeout) || deadline>now+input.action_timeout)
      throw std::invalid_argument("invalid offline initial bounded-action command/deadline");
    Command cmd=input;
    if(input.relieve_contact) validate_contact_relief_start(c,s,input.relief_options);
    else cmd.backoff_plan=plan_joint_limit_backoff(c,s,input.backoff_options);
    cmd.epoch=epoch;cmd.expires_at=deadline;cmd.accepted_at=now;return cmd;
  }
  Command initial_;
  Controller controller_;
  double accepted_at_;
  std::uint64_t last_epoch_=0,last_sequence_=0;
};
}

PYBIND11_MODULE(_twin_native,m) {
  // Reuse registered Config, Command and Mode types; do NOT rebuild/replace
  // the hardware extension or introduce another robot connection.
  py::module_::import("franka_compliant_control._native");
  m.doc()="Offline production controller + libfranka model; no robot I/O";
  m.attr("controller_source_sha256")=FCC_CONTROLLER_SOURCE_SHA256;
  m.attr("twin_inputs_sha256")=FCC_TWIN_INPUTS_SHA256;
  m.attr("libfranka_version")=FCC_FRANKA_VERSION;
  py::class_<Sample>(m,"Sample")
    .def(py::init<>())
#define S(x) .def_readwrite(#x,&Sample::x)
    S(q) S(dq) S(coriolis) S(previous_torque) S(external_torque) S(position)
    S(jacobian) S(mass) S(jacobian_dot_velocity) S(dynamics_valid)
    S(external_wrench) S(external_wrench_stiffness) S(kinematics_valid)
    S(external_wrench_tcp_base) S(wrench_tcp_valid) S(relief_setup_valid)
#undef S
    .def_property("quaternion_xyzw",[](const Sample& s){return Eigen::Vector4d(s.orientation.coeffs());},
      [](Sample& s,const Eigen::Vector4d& q){s.orientation=Eigen::Quaterniond(q[3],q[0],q[1],q[2]);});
  py::class_<Output>(m,"Output")
#define O(x) .def_readonly(#x,&Output::x)
    O(torque) O(task_wrench) O(spring_wrench) O(reference_position) O(reference_twist)
    O(reference_acceleration) O(sequence) O(epoch) O(finished) O(yielding) O(effort_limited)
    O(task_force_limited) O(task_moment_limited) O(joint_torque_limited)
    O(progressive_force_saturated) O(progressive_moment_saturated) O(singular_value)
    O(osc_regularized) O(command_age) O(planned_joint_velocity) O(friction_compensation_torque)
    O(relief_mode) O(relief_manual) O(relief_settling) O(relief_released) O(relief_filtered_wrench)
    O(relief_translation) O(relief_rotation) O(relief_reference_travel) O(relief_reference_rotation) O(relief_deadline)
    O(backoff_mode) O(backoff_reached) O(backoff_settling) O(backoff_deadline) O(backoff_torque_limits) O(backoff_best_progress)
#undef O
    .def_property_readonly("backoff_start_joints",[](const Output& o){return o.backoff_plan.start;})
    .def_property_readonly("backoff_target_joints",[](const Output& o){return o.backoff_plan.goal;})
    .def_property_readonly("backoff_directions",[](const Output& o){return o.backoff_plan.direction;})
    .def_property_readonly("phase",[](const Output& o){return phase_name(o.phase);})
    .def_property_readonly("reason",[](const Output& o){return reason_name(o.reason);})
    .def_property_readonly("stop_trigger",[](const Output& o){return reason_name(o.stop_trigger);});
  py::class_<OfflineModel>(m,"OfflineModel")
    .def(py::init<const std::string&,const std::array<double,16>&,double,
                 const std::array<double,3>&,const std::array<double,9>&>())
    .def("sample",&OfflineModel::sample)
    .def("gravity",&OfflineModel::gravity);
  py::class_<OfflineController>(m,"OfflineController")
    .def(py::init<const Config&,const Sample&,std::uint64_t,double>())
    .def(py::init<const Config&,const Sample&,std::uint64_t,double,const Command&,double>(),
         py::arg("config"),py::arg("sample"),py::arg("epoch"),py::arg("now"),
         py::arg("initial_relief"),py::arg("deadline"))
    .def("step",&OfflineController::step,py::arg("sample"),py::arg("command"),
         py::arg("now"),py::arg("dt"),py::arg("deadline"),py::arg("stop")=false);
  m.def("sdk_torque_filter",[](const V7& desired,const V7& previous) {
    if(!desired.allFinite() || !previous.allFinite()) throw std::invalid_argument("nonfinite torque");
    std::array<double,7> current, last;
    Eigen::Map<V7>(last.data())=previous;
    for(int i=0;i<7;i++) current[i]=franka::lowpassFilter(franka::kDeltaT,desired[i],previous[i],100.0);
    const auto filtered=franka::limitRate(franka::kMaxTorqueRate,current,last);
    return V7(Eigen::Map<const V7>(filtered.data()));
  });
}
