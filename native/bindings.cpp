#include "backend.hpp"
#include "hardware_stop.hpp"
#include "joint_limit_backoff.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#ifdef FCC_WITH_FRANKA
#include <franka/control_tools.h>
#endif
#ifdef FCC_OFFLINE_KINEMATICS
#include "offline_kinematics.hpp"
#endif
namespace py=pybind11;
using namespace fcc;

template<class V> py::list list(const V& v) {
  py::list result;
  for(int i=0;i<v.size();i++) result.append(v[i]);
  return result;
}
static py::dict inhibit_dict(const MotionInhibitStatus& s) {
  py::dict d;d["motion_inhibited"]=s.active;d["generation"]=s.generation;return d;
}
static py::dict collision_dict(const CollisionSettings& s) {
  py::dict d;
  d["status"]=collision_application_name(s.status);
  d["application_attempt"]=s.attempt;
  d["acknowledged_monotonic_s"]=s.acknowledged_at>0?py::cast(s.acknowledged_at):py::none();
  d["is_robot_readback"]=false;
  d["thresholds"]=py::none();
  if(s.attempt) {
    py::dict t;
    t["lower_torque_thresholds_acceleration"]=s.lower_torque;
    t["upper_torque_thresholds_acceleration"]=s.upper_torque;
    t["lower_torque_thresholds_nominal"]=s.lower_torque;
    t["upper_torque_thresholds_nominal"]=s.upper_torque;
    t["lower_force_thresholds_acceleration"]=s.lower_wrench;
    t["upper_force_thresholds_acceleration"]=s.upper_wrench;
    t["lower_force_thresholds_nominal"]=s.lower_wrench;
    t["upper_force_thresholds_nominal"]=s.upper_wrench;
    d["thresholds"]=t;
  }
  return d;
}
static py::dict observation_dict(const RobotObservation& o) {
  const auto& s=o.sample;
  py::dict d;
  d["robot_time_s"]=s.robot_time;
  d["robot_mode"]=robot_mode_name(s.robot_mode);d["robot_has_errors"]=s.robot_has_errors;
  d["sample_valid"]=s.finite();
  d["positions"]=list(s.q);d["velocities"]=list(s.dq);
  d["position"]=list(s.position);d["quaternion_xyzw"]=list(s.orientation.coeffs());
  d["external_joint_torques"]=list(s.external_torque);
  d["external_wrench"]=list(s.external_wrench);
  d["external_wrench_stiffness"]=list(s.external_wrench_stiffness);
  d["measured_joint_torques"]=list(o.measured_torque);
  d["last_desired_joint_torques"]=list(s.previous_torque);
  d["cartesian_collision"]=list(s.cartesian_collision);d["joint_collision"]=list(s.joint_collision);
  d["O_T_EE"]=o.base_to_ee;d["F_T_EE"]=o.flange_to_ee;d["EE_T_K"]=o.ee_to_stiffness;
  d["current_errors_text"]=o.current_errors;d["last_motion_errors_text"]=o.last_motion_errors;
  d["communication_success"]=s.communication_success;
  d["robot_state_serialized"]=o.robot_state_json;
  return d;
}
static py::dict fault_dict(const FaultDiagnostics& f) {
  py::dict d;
  d["reason"]=reason_name(f.reason);d["exception_type"]=f.exception_type;
  d["message"]=f.message;d["caught_at_monotonic_s"]=f.caught_at;
  d["sdk_record_count"]=f.records.size();
  d["sdk_record_alignment"]="command at robot timestamp n, paired state at n+1; not host timestamps";
  d["exact_trigger_sample_guaranteed"]=false;
  py::list records;
  for(const auto& record:f.records) {
    py::dict row, command;
    row["state"]=observation_dict(record.state);
    command["torques"]=record.sent_torques;command["joint_positions"]=record.sent_joint_positions;
    command["joint_velocities"]=record.sent_joint_velocities;
    command["pose"]=record.sent_pose;command["twist"]=record.sent_twist;
    row["command"]=command;records.append(row);
  }
  d["sdk_records"]=records;
  py::dict post;
  post["attempted"]=f.post_read_attempted;post["completed"]=f.post_read_completed;
  post["fresh"]=f.post_read_fresh;
  post["valid"]=f.post_read_valid;post["error"]=f.post_read_error;
  post["sampled_at_monotonic_s"]=f.post_read_at>0?py::cast(f.post_read_at):py::none();
  post["sample_age_s"]=f.post_read_at>0?py::cast(std::max(0.0,steady_seconds()-f.post_read_at)):py::none();
  post["state"]=py::none();
  if(f.post_read_at>0) post["state"]=observation_dict(f.post_observation);
  d["post_fault_observation"]=post;
  return d;
}
PYBIND11_MODULE(_native,m) {
  m.doc()="Native impedance controller; no Python callback runs in the control thread.";
#ifdef FCC_WITH_FRANKA
  m.attr("hardware_available")=true;
  m.attr("libfranka_version")=FCC_FRANKA_VERSION;
  m.def("has_realtime_kernel",&franka::hasRealtimeKernel);
#else
  m.attr("hardware_available")=false;
  m.attr("libfranka_version")=py::none();
  m.def("has_realtime_kernel",[](){return false;});
#endif
#ifdef FCC_OFFLINE_KINEMATICS
  m.attr("offline_kinematics_available")=true;
  py::class_<OfflineKinematics>(m,"OfflineKinematics")
    .def(py::init<const std::string&,const std::array<double,16>&>())
    .def("evaluate",&OfflineKinematics::evaluate);
#else
  m.attr("offline_kinematics_available")=false;
#endif
  m.def("monotonic_time",&steady_seconds);
  py::enum_<Mode>(m,"Mode").value("Cartesian",Mode::Cartesian).value("Joint",Mode::Joint);
  auto config=py::class_<Config>(m,"Config")
    .def(py::init<const std::string&>(),py::arg("robot_model")="panda")
    .def("validate",&Config::validate);
#define FIELD(x) config.def_readwrite(#x,&Config::x);
  FIELD(robot_model) FIELD(compliance_mode) FIELD(motion_limit_policy)
  FIELD(bounded_cartesian_spring) FIELD(effort_limit_policy) FIELD(collision_policy)
  FIELD(osc_translation_kp) FIELD(osc_translation_kd) FIELD(osc_rotation_kp) FIELD(osc_rotation_kd)
  FIELD(osc_regularization) FIELD(osc_feedback_mode)
  FIELD(progressive_position_soft_radius) FIELD(progressive_position_saturation_radius)
  FIELD(progressive_rotation_soft_radius) FIELD(progressive_rotation_saturation_radius)
  FIELD(progressive_translation_soft_stiffness) FIELD(progressive_rotation_soft_stiffness)
  FIELD(osc_translation_stiffness) FIELD(osc_translation_damping)
  FIELD(osc_rotation_stiffness) FIELD(osc_rotation_damping)
  FIELD(translation_stiffness) FIELD(rotation_stiffness)
  FIELD(translation_damping) FIELD(rotation_damping) FIELD(joint_stiffness)
  FIELD(friction_compensation_torque) FIELD(friction_compensation_epsilon) FIELD(friction_compensation_filter_time_s)
  FIELD(joint_damping) FIELD(joint_feedback_torque_limits) FIELD(stop_damping) FIELD(joint_lower) FIELD(joint_upper)
  FIELD(torque_limits) FIELD(external_torque_limits) FIELD(commanded_force_limit)
  FIELD(commanded_moment_limit) FIELD(contact_force_soft) FIELD(contact_force_stop)
  FIELD(contact_moment_soft) FIELD(contact_moment_stop) FIELD(translation_velocity)
  FIELD(translation_acceleration) FIELD(translation_jerk) FIELD(rotation_velocity)
  FIELD(rotation_acceleration) FIELD(rotation_jerk) FIELD(joint_velocity)
  FIELD(measured_translation_velocity_limit) FIELD(measured_rotation_velocity_limit)
  FIELD(measured_joint_velocity_limit) FIELD(measured_joint_velocity_limits)
  FIELD(joint_acceleration) FIELD(joint_jerk) FIELD(position_error_limit)
  FIELD(rotation_error_limit) FIELD(joint_error_limit) FIELD(posture_stiffness)
  FIELD(posture_damping) FIELD(nullspace_torque_limit) FIELD(torque_rate_limit)
  FIELD(command_timeout) FIELD(stop_velocity) FIELD(stop_settle_time) FIELD(stop_timeout)
  FIELD(joint_limit_margin) FIELD(minimum_singular_value) FIELD(jacobian_rotation_scale)
  FIELD(maximum_control_period)
#undef FIELD
  py::class_<HardwareConfig>(m,"HardwareConfig")
    .def(py::init<>())
    .def_readwrite("robot_ip",&HardwareConfig::robot_ip)
    .def_readwrite("robot_model",&HardwareConfig::robot_model)
    .def_readwrite("read_only",&HardwareConfig::read_only)
    .def_readwrite("check_end_effector",&HardwareConfig::check_end_effector)
    .def_readwrite("expected_ee_mass",&HardwareConfig::expected_ee_mass)
    .def_readwrite("expected_ee_com",&HardwareConfig::expected_ee_com)
    .def_readwrite("expected_ee_inertia",&HardwareConfig::expected_ee_inertia)
    .def("validate_end_effector",&HardwareConfig::validate_end_effector)
    .def_readwrite("flange_to_tcp",&HardwareConfig::flange_to_tcp)
    .def_readwrite("load_mass",&HardwareConfig::load_mass)
    .def_readwrite("load_com",&HardwareConfig::load_com)
    .def_readwrite("load_inertia",&HardwareConfig::load_inertia);
  auto relief=py::class_<ContactReliefOptions>(m,"ContactReliefOptions")
    .def(py::init<>()).def("validate",&ContactReliefOptions::validate);
#define RELIEF_FIELD(x) relief.def_readwrite(#x,&ContactReliefOptions::x);
  RELIEF_FIELD(manual_stop) RELIEF_FIELD(bias_wrench_tcp_base) RELIEF_FIELD(bias_joint_positions)
  RELIEF_FIELD(workspace_lower) RELIEF_FIELD(workspace_upper)
  RELIEF_FIELD(bias_confirmed_free) RELIEF_FIELD(workspace_confirmed_clear)
  RELIEF_FIELD(force_release) RELIEF_FIELD(moment_release)
  RELIEF_FIELD(max_translation) RELIEF_FIELD(max_rotation)
  RELIEF_FIELD(translation_speed) RELIEF_FIELD(rotation_speed)
  RELIEF_FIELD(translation_admittance) RELIEF_FIELD(rotation_admittance) RELIEF_FIELD(upward_velocity)
#undef RELIEF_FIELD
  auto backoff=py::class_<JointLimitBackoffOptions>(m,"JointLimitBackoffOptions")
    .def(py::init<>()).def("validate",&JointLimitBackoffOptions::validate);
#define BACKOFF_FIELD(x) backoff.def_readwrite(#x,&JointLimitBackoffOptions::x);
  BACKOFF_FIELD(confirm_clear_path) BACKOFF_FIELD(trigger_margin) BACKOFF_FIELD(backoff_distance)
  BACKOFF_FIELD(speed) BACKOFF_FIELD(torque_scale)
#undef BACKOFF_FIELD
  py::class_<Command>(m,"Command")
    .def(py::init<>())
    .def_readwrite("epoch",&Command::epoch).def_readwrite("sequence",&Command::sequence)
    .def_readwrite("mode",&Command::mode).def_readwrite("position",&Command::position)
    .def_readwrite("twist",&Command::twist).def_readwrite("acceleration",&Command::acceleration)
    .def_readwrite("joints",&Command::joints)
    .def_readwrite("joint_velocity",&Command::joint_velocity)
    .def_readwrite("posture",&Command::posture)
    .def_readwrite("has_posture",&Command::has_posture)
    .def_readwrite("has_orientation",&Command::has_orientation)
    .def_readwrite("finish_at_target",&Command::finish_at_target)
    .def_readwrite("relieve_contact",&Command::relieve_contact)
    .def_readwrite("relief_options",&Command::relief_options)
    .def_readwrite("backoff_joint_limits",&Command::backoff_joint_limits)
    .def_readwrite("backoff_options",&Command::backoff_options)
    .def_readwrite("action_timeout",&Command::action_timeout)
    .def_readwrite("completion_joint_tolerance",&Command::completion_joint_tolerance)
    .def_readwrite("completion_position_tolerance",&Command::completion_position_tolerance)
    .def_readwrite("completion_rotation_tolerance",&Command::completion_rotation_tolerance)
    .def_readwrite("completion_settle_time",&Command::completion_settle_time)
    .def_property("quaternion_xyzw",[](const Command& c){return list(c.orientation.coeffs());},
      [](Command& c,const std::array<double,4>& q){c.orientation=Eigen::Quaterniond(q[3],q[0],q[1],q[2]);});
  py::class_<Backend>(m,"Backend")
    .def(py::init<const Config&,const std::string&,const HardwareConfig&>(),py::call_guard<py::gil_scoped_release>())
    .def("arm",&Backend::arm,py::arg("motion_generation")=py::none(),py::call_guard<py::gil_scoped_release>())
    .def("start",&Backend::start,py::arg("first"),py::arg("deadline"),py::arg("automatic_recovery")=false,
         py::arg("motion_generation")=py::none(),py::call_guard<py::gil_scoped_release>())
    .def("home_command",&Backend::home_command,py::call_guard<py::gil_scoped_release>())
    .def("model_urdf",&Backend::model_urdf,py::call_guard<py::gil_scoped_release>())
    .def("refresh_state",&Backend::refresh_state,py::call_guard<py::gil_scoped_release>())
    .def("submit",&Backend::submit,py::arg("command"),py::arg("deadline"),py::arg("motion_generation")=py::none(),py::call_guard<py::gil_scoped_release>())
    .def("stop",&Backend::request_stop)
    .def("inhibit_motion",[](Backend& b){return inhibit_dict(b.inhibit_motion());})
    .def("motion_inhibit_status",[](Backend& b){return inhibit_dict(b.motion_inhibit_status());})
    .def("release_motion_inhibit",[](Backend& b,std::uint64_t generation){
      MotionInhibitStatus s;{py::gil_scoped_release release;s=b.release_motion_inhibit(generation);}
      return inhibit_dict(s);
    },py::arg("generation"))
    .def("is_running",&Backend::running)
    .def("lifecycle_busy",&Backend::lifecycle_busy)
    .def("recover",&Backend::recover,py::arg("motion_generation")=py::none(),py::call_guard<py::gil_scoped_release>())
    .def("configure",&Backend::configure,py::call_guard<py::gil_scoped_release>())
    .def("close",&Backend::close,py::call_guard<py::gil_scoped_release>())
    .def("inject_wrench",&Backend::inject_wrench)
    .def("prepare_telemetry",[](Backend& b,std::size_t capacity,std::array<double,3> limits){py::gil_scoped_release release;b.prepare_telemetry(capacity,limits);},py::arg("capacity"),py::arg("speed_limits")=std::array<double,3>{0,0,0})
    .def("clear_telemetry",&Backend::clear_telemetry,py::call_guard<py::gil_scoped_release>())
    .def("telemetry_status",[](Backend& b){
      TelemetryStatus s;{py::gil_scoped_release release;s=b.telemetry_status();}
      py::dict d;d["capture_id"]=s.capture_id;d["count"]=s.count;d["dropped"]=s.dropped;
      d["capacity"]=s.capacity;d["prepared"]=s.prepared;d["active"]=s.active;
      d["record_bytes"]=sizeof(TelemetryRecord);d["schema"]="fcc-native-telemetry-v1";
      d["speed_guard_trip"]=s.speed_guard_trip;d["native_experiment_speed_guard_available"]=true;return d;
    })
    .def("telemetry_data",[](Backend& b){
      std::vector<TelemetryRecord> rows;{py::gil_scoped_release release;rows=b.telemetry_data();}
      py::array_t<double> values({static_cast<py::ssize_t>(rows.size()),static_cast<py::ssize_t>(kTelemetryValues)});
      py::array_t<std::uint64_t> ids({static_cast<py::ssize_t>(rows.size()),static_cast<py::ssize_t>(7)});
      for(std::size_t i=0;i<rows.size();++i) {
        std::copy(rows[i].values.begin(),rows[i].values.end(),values.mutable_data()+i*kTelemetryValues);
        std::copy(rows[i].ids.begin(),rows[i].ids.end(),ids.mutable_data()+i*7);
      }
      py::dict d;d["values"]=values;d["ids"]=ids;return d;
    })
    .def("state",[](Backend& b) {
      Snapshot s;
      {py::gil_scoped_release release;s=b.snapshot();}
      py::dict d;
      const auto inhibit=b.motion_inhibit_status();
      d["motion_inhibited"]=inhibit.active;d["motion_inhibit_generation"]=inhibit.generation;
      d["backend"]=b.is_simulation()?"sim":"robot";
      d["read_only"]=b.is_read_only();
      d["running"]=b.running(); d["phase"]=phase_name(s.output.phase);
      d["reason"]=reason_name(s.output.reason); d["error"]=b.error();
      d["stop_trigger"]=reason_name(s.output.stop_trigger);
      const auto& out=s.output;
      const bool relief_active=out.relief_mode && (out.phase==Phase::Armed || out.phase==Phase::Active || out.phase==Phase::Stopping);
      const char* relief_status="none";
      if(out.relief_mode) {
        if(out.phase==Phase::Fault) relief_status="aborted";
        else if(out.phase==Phase::Disarmed) relief_status=out.relief_released?"completed":
            (out.reason==Reason::UserStop?(out.relief_manual?"stopped":"cancelled"):"aborted");
        else if(out.phase==Phase::Stopping) relief_status=out.relief_released?"stopping_after_release":"stopping";
        else if(out.phase==Phase::Armed) relief_status="arming";
        else relief_status=out.relief_manual?"following_wrench":(out.relief_settling?"settling":"relieving");
      }
      d["contact_relief_active"]=relief_active;d["contact_relief_status"]=relief_status;
      d["contact_relief_mode"]=out.relief_mode?(out.relief_manual?"manual":"automatic"):"none";
      d["contact_relief_bias_used"]=out.relief_mode && !out.relief_manual;
      d["contact_relief_released"]=out.relief_released;
      d["contact_relief_filtered_wrench_tcp_base"]=list(out.relief_filtered_wrench);
      d["contact_relief_translation"]=out.relief_translation;d["contact_relief_rotation"]=out.relief_rotation;
      d["contact_relief_reference_travel"]=out.relief_reference_travel;
      d["contact_relief_reference_rotation"]=out.relief_reference_rotation;
      d["contact_relief_deadline_s"]=out.relief_deadline>0?py::cast(out.relief_deadline):py::none();
      d["friction_compensation_suppressed_for_relief"]=relief_active;
      const bool backoff_active=out.backoff_mode && (out.phase==Phase::Armed || out.phase==Phase::Active || out.phase==Phase::Stopping);
      const char* backoff_status="none";
      if(out.backoff_mode) {
        if(out.phase==Phase::Fault) backoff_status="aborted";
        else if(out.phase==Phase::Disarmed) backoff_status=out.backoff_reached?"completed":(out.reason==Reason::UserStop?"cancelled":"aborted");
        else if(out.phase==Phase::Stopping) backoff_status="stopping";
        else if(out.phase==Phase::Armed) backoff_status="arming";
        else backoff_status=out.backoff_settling?"settling":"backing_off";
      }
      py::list selected;
      if(out.backoff_mode) for(int i=0;i<7;i++) if(out.backoff_plan.direction[i]!=0) selected.append(i+1);
      d["joint_limit_backoff_active"]=backoff_active;d["joint_limit_backoff_status"]=backoff_status;
      d["joint_limit_backoff_reached"]=out.backoff_reached;d["joint_limit_backoff_selected"]=selected;
      d["joint_limit_backoff_start_positions"]=out.backoff_mode?py::object(list(out.backoff_plan.start)):py::none();
      d["joint_limit_backoff_target_positions"]=out.backoff_mode?py::object(list(out.backoff_plan.goal)):py::none();
      d["joint_limit_backoff_torque_limits"]=out.backoff_mode?py::object(list(out.backoff_torque_limits)):py::none();
      d["joint_limit_backoff_deadline_s"]=out.backoff_deadline>0?py::cast(out.backoff_deadline):py::none();
      d["joint_limit_backoff_previous_reason"]=reason_name(out.backoff_previous_reason);
      const char* home_status="none";
      if(out.home_superseded) home_status="superseded";
      else if(out.homing) {
        if(out.phase==Phase::Fault || (out.phase==Phase::Stopping && !normal_stop_reason(out.reason))) home_status="fault";
        else if(out.reason==Reason::HomeTimeout) home_status="timed_out";
        else if(out.home_reached) home_status=out.phase==Phase::Disarmed?"completed":"stopping_at_home";
        else if(out.phase==Phase::Stopping || out.phase==Phase::Disarmed) home_status="cancelled";
        else home_status="moving";
      }
      d["homing_status"]=home_status;d["home_reached"]=out.home_reached;
      d["homing_active"]=out.homing && (out.phase==Phase::Armed || out.phase==Phase::Active);
      d["homing_command_seq"]=out.home_sequence;
      d["homing_deadline_s"]=out.home_deadline>0?py::cast(out.home_deadline):py::none();
      d["stop_sample"]=py::none();
      if(s.stop_sample_valid) {
        py::dict trigger;
        trigger["time_monotonic_s"]=s.stop_sampled_at;
        trigger["external_wrench"]=list(s.stop_sample.external_wrench);
        trigger["external_joint_torques"]=list(s.stop_sample.external_torque);
        trigger["positions"]=list(s.stop_sample.q);
        trigger["velocities"]=list(s.stop_sample.dq);
        trigger["quaternion_xyzw"]=list(s.stop_sample.orientation.coeffs());
        trigger["robot_mode"]=robot_mode_name(s.stop_sample.robot_mode);
        trigger["robot_time_s"]=s.stop_sample.robot_time;
        trigger["external_wrench_stiffness"]=list(s.stop_sample.external_wrench_stiffness);
        trigger["cartesian_collision"]=list(s.stop_sample.cartesian_collision);
        trigger["joint_collision"]=list(s.stop_sample.joint_collision);
        d["stop_sample"]=trigger;
      }
      const bool cached_after_fault=s.observation_source==ObservationSource::CachedAfterFault;
      d["robot_mode"]=cached_after_fault?"unconfirmed":robot_mode_name(s.sample.robot_mode);
      d["robot_has_errors"]=cached_after_fault?py::none():py::cast(s.sample.robot_has_errors);
      d["last_observed_robot_mode"]=robot_mode_name(s.sample.robot_mode);
      d["last_observed_robot_has_errors"]=s.sample.robot_has_errors;
      d["observation_source"]=observation_source_name(s.observation_source);
      d["robot_time_s"]=s.sample.robot_time>=0?py::cast(s.sample.robot_time):py::none();
      d["robot_collision_settings"]=collision_dict(s.collision_settings);
      d["fault_diagnostics"]=py::none();
      if(s.fault_diagnostics) d["fault_diagnostics"]=fault_dict(*s.fault_diagnostics);
      d["last_recovery"]=py::none();
      if(s.recovery) {
        const auto& r=*s.recovery;py::dict recovery;
        recovery["status"]=r.status;recovery["error"]=r.error;recovery["automatic"]=r.automatic;
        recovery["robot_recovery_called"]=r.robot_recovery_called;recovery["attempt"]=r.attempt;
        recovery["started_at_monotonic_s"]=r.started_at;recovery["completed_at_monotonic_s"]=r.completed_at;
        recovery["previous_reason"]=reason_name(r.previous_reason);
        recovery["previous_stop_trigger"]=reason_name(r.previous_trigger);
        recovery["previous_robot_mode"]=robot_mode_name(r.previous_sample.robot_mode);
        recovery["previous_positions"]=list(r.previous_sample.q);
        recovery["previous_external_joint_torques"]=list(r.previous_sample.external_torque);
        recovery["previous_external_wrench"]=list(r.previous_sample.external_wrench);
        recovery["previous_error"]=r.previous_fault_diagnostics?r.previous_fault_diagnostics->message:"";
        recovery["previous_sdk_record_count"]=r.previous_fault_diagnostics?r.previous_fault_diagnostics->records.size():0;
        d["last_recovery"]=recovery;
      }
      d["epoch"]=s.output.epoch; d["applied_seq"]=s.output.sequence;
      d["mode"]=s.output.mode==Mode::Cartesian?"cartesian":"joint";
      d["position"]=list(s.sample.position);
      d["quaternion_xyzw"]=list(s.sample.orientation.coeffs());
      d["positions"]=list(s.sample.q);d["velocities"]=list(s.sample.dq);
      d["kinematics_valid"]=s.sample.kinematics_valid;
      d["twist"]=py::none();
      if(s.sample.kinematics_valid) d["twist"]=list(V6(s.sample.jacobian*s.sample.dq));
      d["external_wrench"]=list(s.sample.external_wrench);
      d["external_wrench_stiffness"]=list(s.sample.external_wrench_stiffness);
      d["external_wrench_tcp_valid"]=s.sample.wrench_tcp_valid;
      d["contact_relief_setup_valid"]=s.sample.relief_setup_valid;
      d["external_wrench_tcp_base"]=s.sample.wrench_tcp_valid?py::object(list(s.sample.external_wrench_tcp_base)):py::none();
      d["last_desired_joint_torques"]=list(s.sample.previous_torque);
      d["cartesian_collision"]=list(s.sample.cartesian_collision);
      d["joint_collision"]=list(s.sample.joint_collision);
      d["external_joint_torques"]=list(s.sample.external_torque);
      d["commanded_joint_torques"]=list(s.output.torque);
      d["commanded_task_wrench"]=list(s.output.task_wrench);
      d["spring_wrench"]=list(s.output.spring_wrench);
      d["progressive_force_saturated"]=s.output.progressive_force_saturated;
      d["progressive_moment_saturated"]=s.output.progressive_moment_saturated;
      d["reference_position"]=list(s.output.reference_position);
      d["reference_quaternion_xyzw"]=list(s.output.reference_orientation.coeffs());
      d["reference_joints"]=list(s.output.reference_joints);
      d["reference_joint_velocities"]=list(s.output.reference_joint_velocity);
      d["joint_spring_torques"]=list(s.output.joint_spring_torque);
      d["joint_feedback_torques"]=list(s.output.joint_feedback_torque);
      d["planned_joint_velocities"]=list(s.output.planned_joint_velocity);
      d["friction_compensation_torques"]=list(s.output.friction_compensation_torque);
      d["joint_spring_limited"]=s.output.joint_spring_limited;
      d["joint_feedback_limited"]=s.output.joint_feedback_limited;
      d["reference_twist"]=list(s.output.reference_twist);
      d["reference_acceleration"]=list(s.output.reference_acceleration);
      d["dynamics_valid"]=s.sample.dynamics_valid;
      d["osc_regularized"]=s.output.osc_regularized;
      d["yielding"]=s.output.yielding;d["effort_limited"]=s.output.effort_limited;
      d["drive_scale"]=s.output.drive_scale;
      d["contact_load_ratio"]=s.output.contact_load_ratio;
      d["peak_contact_load_ratio"]=s.output.peak_contact_load_ratio;
      d["reference_limited"]=s.output.reference_limited;
      d["reference_speed_limited"]=s.output.reference_speed_limited;
      d["task_force_limited"]=s.output.task_force_limited;
      d["task_moment_limited"]=s.output.task_moment_limited;
      d["joint_torque_limited"]=s.output.joint_torque_limited;
      d["minimum_singular_value"]=s.output.singular_value;
      const double state_at=steady_seconds();
      d["sample_timestamp_monotonic_s"]=s.sampled_at;
      d["state_timestamp_monotonic_s"]=state_at;
      d["sample_age_s"]=std::max(0.0,state_at-s.sampled_at);
      d["sample_valid"]=s.sample.finite();
      d["controller_compute_s"]=s.cycle_time;
      d["command_age_s"]=s.output.command_age;
      d["communication_success"]=s.sample.communication_success;
      return d;
    });
}
