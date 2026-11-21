#include "backend.hpp"
#include "hardware_stop.hpp"
#include "recovery_policy.hpp"
#include "joint_limit_backoff.hpp"
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#ifdef FCC_WITH_FRANKA
#include <franka/robot.h>
#include <franka/model.h>
#include <franka/exception.h>
#include "model_dynamics.hpp"
#include "robot_diagnostics.hpp"
#endif

namespace fcc {
const char* robot_mode_name(int mode) {
#ifdef FCC_WITH_FRANKA
  switch(mode) {
    case static_cast<int>(franka::RobotMode::kOther): return "Other";
    case static_cast<int>(franka::RobotMode::kIdle): return "Idle";
    case static_cast<int>(franka::RobotMode::kMove): return "Move";
    case static_cast<int>(franka::RobotMode::kGuiding): return "Guiding";
    case static_cast<int>(franka::RobotMode::kReflex): return "Reflex";
    case static_cast<int>(franka::RobotMode::kUserStopped): return "UserStopped";
    case static_cast<int>(franka::RobotMode::kAutomaticErrorRecovery): return "AutomaticErrorRecovery";
  }
#else
  (void)mode;
#endif
  return "unavailable";
}
void HardwareConfig::validate_end_effector(double mass, const std::array<double,3>& com,
                                          const std::array<double,9>& inertia) const {
  if(!check_end_effector) return;
  const auto matches=[](double actual,double expected,double tolerance) {
    return std::isfinite(actual) && std::isfinite(expected) && std::abs(actual-expected)<=tolerance;
  };
  bool valid=matches(mass,expected_ee_mass,1e-5);
  for(int i=0;i<3;i++) valid=valid && matches(com[i],expected_ee_com[i],1e-6);
  for(int i=0;i<9;i++) valid=valid && matches(inertia[i],expected_ee_inertia[i],1e-8);
  if(!valid) throw std::runtime_error("Desk end-effector mass/COM/inertia differ from configured tool");
}
double steady_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Backend::RobotImpl {
#ifdef FCC_WITH_FRANKA
  franka::Robot robot;
  franka::Model model;
  std::array<double,16> home_pose;
  HardwareConfig expected_hardware;
  std::string urdf;
  explicit RobotImpl(const HardwareConfig& h)
    : robot(h.robot_ip,franka::RealtimeConfig::kEnforce,kExceptionLogCapacity),model(robot.loadModel()),expected_hardware(h) {
    const auto& tcp=h.flange_to_tcp;
    std::array<double,7> q;Eigen::Map<V7>(q.data())=nominal_home_joints();
    const std::array<double,16> identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    home_pose=model.pose(franka::Frame::kEndEffector,q,tcp,identity);
#ifdef FCC_OFFLINE_KINEMATICS
    // Supported by the >=0.18 SDK. Read once at startup, never in control or GET handlers.
    urdf=robot.getRobotModel();
#endif
  }
  Sample sample(const franka::RobotState& rs, bool dynamics) {
    Sample s=observation_sample(rs);
    const Eigen::Map<const Eigen::Matrix4d> actual_tcp(rs.F_T_EE.data()),expected_tcp(expected_hardware.flange_to_tcp.data());
    s.wrench_tcp_valid=actual_tcp.allFinite() && expected_tcp.allFinite() &&
        actual_tcp.isApprox(expected_tcp,1e-6) && sdk_wrench_at_ee_base(rs,s.external_wrench_tcp_base);
    s.relief_setup_valid=relief_hardware_setup_matches(expected_hardware,rs);
    const auto c=model.coriolis(rs);
    s.coriolis=Eigen::Map<const V7>(c.data());
    const auto j=model.zeroJacobian(franka::Frame::kEndEffector,rs);
    s.jacobian=Eigen::Map<const J67>(j.data());
    s.kinematics_valid=true;
    if(dynamics) sample_dynamics(model,rs,s);
    return s;
  }
  static void require_idle(const Sample& s) {
    if(s.robot_mode!=static_cast<int>(franka::RobotMode::kIdle) || s.robot_has_errors)
      throw std::runtime_error(std::string("robot is in ")+robot_mode_name(s.robot_mode)+
          (s.robot_has_errors?" with active errors":"")+
          "; restore an error-free Idle state in Desk before arming; no recovery attempted");
  }
#endif
};
Backend::Backend(const Config& c,const std::string& kind,const HardwareConfig& h)
  :config_(c),hardware_(h),simulation_(kind=="sim") {
  c.validate();
  if(kind!="sim" && kind!="robot") throw std::invalid_argument("backend must be sim or robot");
  if(!simulation_) {
    if(h.robot_model!=c.robot_model)
      throw std::invalid_argument("hardware and controller robot_model must match");
#ifdef FCC_WITH_FRANKA
    if(h.robot_ip.empty()) throw std::invalid_argument("robot_ip required");
    robot_=std::make_unique<RobotImpl>(h);
#else
    throw std::runtime_error("This build has no libfranka backend. Rebuild with FCC_WITH_FRANKA=ON.");
#endif
  }
  snapshot_.sample.q << 0,-.4,0,-1.9,0,1.5,0;
  snapshot_.sample.position << .45,0,.4;
  for(int i=0;i<6;i++) snapshot_.sample.jacobian(i,i)=i<3?.2:1.0;
  snapshot_.sample.jacobian(0,6)=.05;
  snapshot_.sample.mass=3*M7::Identity();
  snapshot_.sample.dynamics_valid=true;
  snapshot_.sample.wrench_tcp_valid=true; // Synthetic test plant has an explicit TCP wrench.
  snapshot_.sample.relief_setup_valid=true;
  if(!simulation_) snapshot_.sample=read_idle();
  snapshot_.observation_source=simulation_?ObservationSource::Simulation:ObservationSource::RobotRead;
  snapshot_.sampled_at=steady_seconds();
  snapshot_.output.reference_position=snapshot_.sample.position;
  snapshot_.output.reference_orientation=snapshot_.sample.orientation;
  snapshot_.output.reference_joints=snapshot_.sample.q;
  if(c.compliance_mode=="joint_impedance") snapshot_.output.mode=Mode::Joint;
  home_command_.joints=home_command_.posture=nominal_home_joints();
  home_command_.position=snapshot_.sample.position;
  home_command_.orientation=snapshot_.sample.orientation;
#ifdef FCC_WITH_FRANKA
  if(!simulation_) {
    const Eigen::Map<const Eigen::Matrix4d> pose(robot_->home_pose.data());
    home_command_.position=pose.topRightCorner<3,1>();
    home_command_.orientation=Eigen::Quaterniond(pose.topLeftCorner<3,3>()).normalized();
  }
#endif
}
Command Backend::home_command() {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  auto home=home_command_;
  if(config_.compliance_mode=="auto") {
    home.mode=snapshot().output.mode;
    if(last_sequence_>0) {
      std::lock_guard<std::mutex> command_lock(command_mutex_);
      home.mode=command_.mode; // Most recently accepted mode, including after a stop.
    }
  } else home.mode=config_.compliance_mode=="operational_space"?Mode::Cartesian:Mode::Joint;
  home.has_orientation=home.has_posture=home.mode==Mode::Cartesian;
  validate_command(home);
  return home;
}
std::string Backend::model_urdf() const {
#ifdef FCC_WITH_FRANKA
  if(!simulation_) return robot_->urdf;
#endif
  return {};
}
Backend::~Backend() {close();}
Sample Backend::read_idle() {
#ifdef FCC_WITH_FRANKA
  if(!simulation_) {
    const auto rs=robot_->robot.readOnce();
    hardware_.validate_end_effector(rs.m_ee,rs.F_x_Cee,rs.I_ee);
    return robot_->sample(rs,config_.uses_osc_model());
  }
#endif
  return snapshot().sample;
}
Snapshot Backend::snapshot() {std::lock_guard<std::mutex> lock(snapshot_mutex_);return snapshot_;}
std::string Backend::error() {std::lock_guard<std::mutex> lock(error_mutex_);return error_;}
void Backend::refresh_state() {
  std::lock_guard<std::mutex> lifecycle(lifecycle_mutex_);
  if(running_) throw std::runtime_error("state refresh requires an inactive controller; use callback telemetry while running");
  const auto previous=snapshot();
  const auto observed=read_idle();
  if(!simulation_ && observed.robot_time<=previous.sample.robot_time)
    throw std::runtime_error("robot observation timestamp did not advance; cached state retained");
  std::lock_guard<std::mutex> lock(snapshot_mutex_);
  snapshot_.sample=observed;
  snapshot_.sampled_at=steady_seconds();
  snapshot_.observation_source=simulation_?ObservationSource::Simulation:ObservationSource::RobotRead;
  // Preserve latched faults, references, first-stop sample and SDK diagnostics.
}
void Backend::publish(const Snapshot& s) {
  // The RT callback must never wait for HTTP/state readers.
  std::unique_lock<std::mutex> lock(snapshot_mutex_,std::try_to_lock);
  if(lock.owns_lock()) snapshot_=s;
}
std::uint64_t Backend::arm(std::optional<std::uint64_t> expected) {
  const auto generation=motion_inhibit_.permit(expected);
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  return arm_locked(generation);
}
std::uint64_t Backend::start(Command first,double deadline,bool automatic_recovery,
                             std::optional<std::uint64_t> expected) {
  const auto generation=motion_inhibit_.permit(expected);
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  motion_inhibit_.check(generation);
  // Validate the complete target BEFORE robot reads/setters or torque control.
  validate_command(first);
  const double now=steady_seconds();
  if((first.relieve_contact || first.backoff_joint_limits) && automatic_recovery)
    throw std::invalid_argument("bounded relief/backoff actions never permit automatic recovery");
  const double lifetime=(first.finish_at_target || first.relieve_contact || first.backoff_joint_limits)?first.action_timeout:config_.command_timeout;
  if(first.sequence==0 || !std::isfinite(deadline) || deadline<=now || deadline>now+lifetime)
    throw std::invalid_argument("invalid initial target sequence or lifetime");
  if(running_) throw std::runtime_error("controller is already running");
  stop_requested_=false;
  if(automatic_recovery && !running_ && !hardware_.read_only) try_automatic_recovery_locked(generation);
  return arm_locked(generation,&first,deadline);
}
std::uint64_t Backend::arm_locked(std::uint64_t generation,const Command* first,double deadline) {
  motion_inhibit_.check(generation);
  if(telemetry_.used()) throw std::runtime_error("Export and clear previous native capture before another session");
  if(hardware_.read_only) throw std::runtime_error("read-only connection: arming is disabled");
  if(running_) throw std::runtime_error("controller is already running");
  if(!first) stop_requested_=false;
  if(stop_requested_) throw std::runtime_error("startup interrupted by a stop request");
  if(worker_.joinable()) worker_.join();
  motion_inhibit_.check(generation);
  const bool relief=first && first->relieve_contact;
  const bool backoff=first && first->backoff_joint_limits;
  const auto prior=snapshot();
  const bool local_backoff_fault=backoff && !prior.output.relief_mode && !prior.fault_diagnostics &&
      backoff_local_fault_allowed(prior.output.reason);
  if(prior.output.phase==Phase::Fault && !local_backoff_fault)
    throw std::runtime_error("explicit recovery required; fault is not eligible for joint-limit backoff");
  Command prepared;if(first) prepared=*first;
  Sample s=read_idle();
  {std::lock_guard<std::mutex> l(snapshot_mutex_);
    snapshot_.sample=s;snapshot_.sampled_at=steady_seconds();
    snapshot_.observation_source=simulation_?ObservationSource::Simulation:ObservationSource::RobotRead;}
#ifdef FCC_WITH_FRANKA
  if(!simulation_) RobotImpl::require_idle(s);
#endif
  if(!s.finite() || s.dq.cwiseAbs().maxCoeff()>config_.stop_velocity)
    throw std::runtime_error("robot must have valid state and be stationary before arming");
  if((s.previous_torque.cwiseAbs().array()>config_.torque_limits.array()).any())
    throw std::runtime_error("initial torque exceeds configured command limits");
  if(backoff) prepared.backoff_plan=plan_joint_limit_backoff(config_,s,prepared.backoff_options);
  else for(int i=0;i<7;i++) {
    const double lower=config_.joint_lower[i]+config_.joint_limit_margin;
    const double upper=config_.joint_upper[i]-config_.joint_limit_margin;
    if(s.q[i]<=lower || s.q[i]>=upper) {
      std::ostringstream message;
      message << "initial state violates robot model joint margins: joint " << i+1
              << " is " << s.q[i] << " rad; must be strictly between " << lower
              << " and " << upper << " rad";
      throw std::runtime_error(message.str());
    }
  }
  if(relief) validate_contact_relief_start(config_,s,first->relief_options);
#ifdef FCC_WITH_FRANKA
  if(!simulation_) {
    // An unloaded commissioning hold cannot be established while the initial
    // wrench already demands yielding. Reject before any robot setters run.
    if(!relief && (s.external_wrench.head<3>().norm()>=config_.contact_force_soft ||
                  s.external_wrench.tail<3>().norm()>=config_.contact_moment_soft))
      throw std::runtime_error("initial estimated wrench exceeds a soft contact threshold; investigate tool/contact load before arming");
    // Bounded actions require an already matching setup and change NO SDK
    // frame/load/collision settings. Backoff retains the ordinary unloaded gate.
    if(!relief && !backoff) {
    // All model/configuration calls are outside the real-time callback.
    const Eigen::Map<const Eigen::Matrix4d> f_tcp(hardware_.flange_to_tcp.data());
    const auto rs=robot_->robot.readOnce();
    hardware_.validate_end_effector(rs.m_ee,rs.F_x_Cee,rs.I_ee);
    const Eigen::Map<const Eigen::Matrix4d> f_ne(rs.F_T_NE.data());
    const Eigen::Matrix4d ne_tcp=f_ne.inverse()*f_tcp;
    std::array<double,16> transform;
    Eigen::Map<Eigen::Matrix4d>(transform.data())=ne_tcp;
    motion_inhibit_.check(generation);
    robot_->robot.setEE(transform);
    motion_inhibit_.check(generation);
    robot_->robot.setK({1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1});
    motion_inhibit_.check(generation);
    robot_->robot.setLoad(hardware_.load_mass,hardware_.load_com,hardware_.load_inertia);
    motion_inhibit_.check(generation);
    if(applies_robot_collision_settings(config_)) {
      auto request=collision_request(config_);request.attempt=++collision_attempt_;
      request.status=CollisionApplication::Applying;
      {std::lock_guard<std::mutex> l(snapshot_mutex_);snapshot_.collision_settings=request;}
      try {
        apply_collision_request(robot_->robot,request);
        request.acknowledged_at=steady_seconds();
      } catch(...) {
        std::lock_guard<std::mutex> l(snapshot_mutex_);snapshot_.collision_settings=request;
        throw;
      }
      // Persist the successful request even if a later arm preflight fails.
      {std::lock_guard<std::mutex> l(snapshot_mutex_);snapshot_.collision_settings=request;}
    }
    }
    s=read_idle();
    {std::lock_guard<std::mutex> l(snapshot_mutex_);
      snapshot_.sample=s;snapshot_.sampled_at=steady_seconds();
      snapshot_.observation_source=ObservationSource::RobotRead;}
    RobotImpl::require_idle(s);
  }
#endif
  motion_inhibit_.check(generation);
  if(stop_requested_) throw std::runtime_error("arming interrupted by a stop request");
  const double checked_at=steady_seconds();
  if(relief) validate_contact_relief_start(config_,s,first->relief_options);
  if(backoff) prepared.backoff_plan=plan_joint_limit_backoff(config_,s,prepared.backoff_options);
  Command preflight_command;
  if(relief || backoff) {
    preflight_command=prepared;preflight_command.epoch=epoch_+1;preflight_command.expires_at=deadline;
  }
  Controller preflight(config_,s,epoch_+1,checked_at,(relief || backoff)?&preflight_command:nullptr);
  const auto checked=preflight.step(s,preflight_command,false,checked_at,0);
  // Startup must be unloaded for either law, even though continuous impedance
  // deliberately never sets the runtime yielding flag.
  if(!simulation_ && !relief && (s.external_wrench.head<3>().norm()>=config_.contact_force_soft ||
                     s.external_wrench.tail<3>().norm()>=config_.contact_moment_soft))
    throw std::runtime_error("arm preflight failed: initial estimated wrench exceeds a soft contact threshold");
  if(checked.phase!=((relief || backoff)?Phase::Active:Phase::Armed))
    throw std::runtime_error(std::string("arm preflight failed: ")+reason_name(checked.reason));
  Command initial;
  if(first) {
    const double now=steady_seconds();
    if(deadline<=now) throw std::invalid_argument("initial setpoint expired during preflight; control was not started");
    initial=prepared;initial.epoch=epoch_+1;initial.expires_at=deadline;initial.accepted_at=now;
  }
  motion_inhibit_.check(generation);
  ++epoch_; last_sequence_=initial.sequence;
  {std::lock_guard<std::mutex> l(command_mutex_);command_=initial;}
  {std::lock_guard<std::mutex> l(error_mutex_);error_.clear();}
  Snapshot start;
  start.collision_settings=snapshot().collision_settings;
  start.recovery=snapshot().recovery;
  if(simulation_ && applies_robot_collision_settings(config_)) {
    start.collision_settings=collision_request(config_);
    start.collision_settings.status=CollisionApplication::Simulated;
    start.collision_settings.attempt=++collision_attempt_;
  }
  start.observation_source=simulation_?ObservationSource::Simulation:ObservationSource::RobotRead;
  start.sample=s; start.sampled_at=steady_seconds();
  start.output.phase=Phase::Armed; start.output.epoch=epoch_;
  start.output.relief_mode=relief;
  if(relief) start.output.relief_manual=initial.relief_options.manual_stop;
  if(relief) start.output.relief_deadline=initial.expires_at;
  if(backoff) {
    start.output.backoff_mode=true;start.output.backoff_deadline=initial.expires_at;
    start.output.backoff_plan=initial.backoff_plan;
    start.output.backoff_torque_limits=backoff_torque_limits(config_,initial.backoff_options,initial.backoff_plan);
    start.output.backoff_previous_reason=prior.output.phase==Phase::Fault?prior.output.reason:Reason::None;
  }
  if(backoff || config_.compliance_mode=="joint_impedance") start.output.mode=Mode::Joint;
  start.output.reference_position=s.position; start.output.reference_orientation=s.orientation;
  start.output.reference_joints=s.q;
  {std::lock_guard<std::mutex> l(snapshot_mutex_);snapshot_=start;}
  automatic_recovery_used_=false;
  telemetry_.begin();
  running_=true;
  worker_=std::thread(&Backend::run,this,s,config_,epoch_,generation,initial);
  return epoch_;
}
void Backend::submit(Command c,double deadline,std::optional<std::uint64_t> expected) {
  const auto generation=motion_inhibit_.permit(expected);
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  motion_inhibit_.check(generation);
  const auto phase=snapshot().output.phase;
  if(!running_ || stop_requested_ || (phase!=Phase::Armed && phase!=Phase::Active))
    throw std::runtime_error("controller is not accepting targets");
  if(command_.relieve_contact || command_.backoff_joint_limits || c.relieve_contact || c.backoff_joint_limits)
    throw std::runtime_error("bounded relief/backoff is an exclusive finite session; stop before another action");
  if(c.epoch!=epoch_ || c.sequence<=last_sequence_ || c.sequence==0)
    throw std::invalid_argument("stale session or sequence");
  const double now=steady_seconds();
  validate_command(c);
  const double lifetime=c.finish_at_target?c.action_timeout:config_.command_timeout;
  if(!std::isfinite(deadline) || deadline<=now || deadline>now+lifetime)
    throw std::invalid_argument("invalid target lifetime");
  c.expires_at=deadline;
  c.accepted_at=now;
  {std::lock_guard<std::mutex> l(command_mutex_);
    motion_inhibit_.check(generation);command_=c;}
  last_sequence_=c.sequence;
}
void Backend::validate_command(const Command& c) const {
  if(c.backoff_joint_limits) {
    c.backoff_options.validate(config_);
    if(c.relieve_contact || c.finish_at_target || c.mode!=Mode::Joint || c.has_orientation || c.has_posture ||
       !c.twist.isZero(0) || !c.acceleration.isZero(0) || !c.joint_velocity.isZero(0) ||
       !std::isfinite(c.action_timeout) || c.action_timeout<=.2 || c.action_timeout>30)
      throw std::invalid_argument("joint-limit backoff requires a fresh native-planned joint action and timeout in (.2,30] s");
  }
  if(c.relieve_contact) {
    c.relief_options.validate(config_);
    if(c.mode!=Mode::Cartesian || c.backoff_joint_limits || c.finish_at_target || c.has_orientation || c.has_posture ||
       !c.twist.isZero(0) || !c.acceleration.isZero(0) || !c.joint_velocity.isZero(0) ||
       !std::isfinite(c.action_timeout) || c.action_timeout<=.2 || c.action_timeout>10)
      throw std::invalid_argument("contact relief requires a fresh Cartesian action, no client reference/feedforward, and timeout in (.2,10] s");
  }
  if(c.finish_at_target) {
    for(double value:{c.action_timeout,c.completion_joint_tolerance,c.completion_position_tolerance,
                      c.completion_rotation_tolerance,c.completion_settle_time})
      if(!std::isfinite(value) || value<=0) throw std::invalid_argument("home completion settings must be finite and positive");
    if(c.action_timeout<=c.completion_settle_time || !c.twist.isZero(0) ||
       !c.acceleration.isZero(0) || !c.joint_velocity.isZero(0))
      throw std::invalid_argument("home action requires zero target velocity/acceleration and timeout greater than settle time");
  }
  if(config_.compliance_mode=="operational_space" && c.mode!=Mode::Cartesian)
    throw std::invalid_argument("operational_space accepts Cartesian targets only");
  if(config_.compliance_mode=="joint_impedance" && c.mode!=Mode::Joint)
    throw std::invalid_argument("joint_impedance accepts joint targets only");
  if(!c.position.allFinite() || !c.orientation.coeffs().allFinite() ||
     !c.twist.allFinite() || !c.acceleration.allFinite() || !c.joints.allFinite() ||
     !c.joint_velocity.allFinite() || !c.posture.allFinite())
    throw std::invalid_argument("target contains nonfinite values");
  if(c.has_orientation && std::abs(c.orientation.norm()-1)>1e-6)
    throw std::invalid_argument("quaternion must be normalized");
  const auto valid_joints=[&](const V7& q) {
    return (q.array()>config_.joint_lower.array()+config_.joint_limit_margin).all() &&
           (q.array()<config_.joint_upper.array()-config_.joint_limit_margin).all();
  };
  if((c.mode==Mode::Joint && !c.backoff_joint_limits && !valid_joints(c.joints)) || (c.has_posture && !valid_joints(c.posture)))
    throw std::invalid_argument("target violates joint margins");
  if(config_.osc_feedback_mode!="linear") {
    if(c.twist.head<3>().norm()>config_.translation_velocity*(1+1e-12) ||
       c.twist.tail<3>().norm()>config_.rotation_velocity*(1+1e-12) ||
       c.acceleration.head<3>().norm()>config_.translation_acceleration*(1+1e-12) ||
       c.acceleration.tail<3>().norm()>config_.rotation_acceleration*(1+1e-12))
      throw std::invalid_argument("feedforward velocity/acceleration exceeds reference limits");
  } else {
    if(!c.acceleration.isZero(0))
      throw std::invalid_argument("acceleration feedforward is supported only by bounded-equilibrium OSC");
    if(c.twist.head<3>().cwiseAbs().maxCoeff()>config_.translation_velocity/std::sqrt(3.0) ||
       c.twist.tail<3>().cwiseAbs().maxCoeff()>config_.rotation_velocity/std::sqrt(3.0))
      throw std::invalid_argument("feedforward velocity exceeds reference limits");
  }
  if(config_.compliance_mode=="joint_impedance" || (config_.compliance_mode=="auto" && c.mode==Mode::Joint)) {
    if((c.joint_velocity.cwiseAbs().array()>config_.joint_speed_limits().array()).any())
      throw std::invalid_argument("joint feedforward velocity exceeds per-joint model/configured ceilings");
    if(c.has_orientation || c.has_posture || !c.twist.isZero(0) || !c.acceleration.isZero(0))
      throw std::invalid_argument("joint_impedance does not accept Cartesian or posture feedforward");
  } else if(c.joint_velocity.cwiseAbs().maxCoeff()>config_.joint_velocity)
    throw std::invalid_argument("feedforward velocity exceeds reference limits");
}
void Backend::request_stop() {stop_requested_=true;}
MotionInhibitStatus Backend::release_motion_inhibit(std::uint64_t generation) {
  std::unique_lock<std::mutex> lock(lifecycle_mutex_,std::try_to_lock);
  if(!lock.owns_lock()) throw std::runtime_error("lifecycle operation in progress; inhibit not released");
  if(running_) throw std::runtime_error("wait for control to stop before releasing motion inhibit");
  // Clearing never starts control, performs SDK recovery, or replays a mailbox.
  {std::lock_guard<std::mutex> l(command_mutex_);command_=Command{};last_sequence_=0;}
  return motion_inhibit_.release(generation);
}
void Backend::configure(const Config& c) {
  c.validate();
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if(running_) throw std::runtime_error("disarm before changing configuration");
  if(c.robot_model!=config_.robot_model)
    throw std::invalid_argument("robot_model cannot change on an existing backend; restart with the intended profile");
  config_=c;
}
void Backend::recover(std::optional<std::uint64_t> expected) {
  const auto generation=motion_inhibit_.permit(expected);
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  recover_locked(false,generation);
  automatic_recovery_used_=false;
}
void Backend::try_automatic_recovery_locked(std::uint64_t generation) {
  motion_inhibit_.check(generation);
  const auto before=snapshot();
  if((before.output.relief_mode || before.output.backoff_mode) && before.output.phase==Phase::Fault)
    throw std::runtime_error("bounded relief/backoff abort requires explicit operator recovery; no automatic recovery");
  auto s=before.sample;
  bool allowed_mode=true, collision_errors_only=true, robot_needs_recovery=false;
#ifdef FCC_WITH_FRANKA
  if(!simulation_) {
    const auto rs=robot_->robot.readOnce();
    hardware_.validate_end_effector(rs.m_ee,rs.F_x_Cee,rs.I_ee);
    s=robot_->sample(rs,config_.uses_osc_model());
    allowed_mode=rs.robot_mode==franka::RobotMode::kIdle || rs.robot_mode==franka::RobotMode::kReflex;
    const auto errors=static_cast<std::string>(rs.current_errors);
    collision_errors_only=errors=="[]" || errors=="[\"joint_reflex\"]" ||
        errors=="[\"cartesian_reflex\"]" || errors=="[\"joint_reflex\", \"cartesian_reflex\"]";
    robot_needs_recovery=rs.robot_mode!=franka::RobotMode::kIdle || static_cast<bool>(rs.current_errors);
  }
#endif
  if(before.output.phase!=Phase::Fault && !robot_needs_recovery) return;
  if(automatic_recovery_used_)
    throw std::runtime_error("automatic recovery already attempted for this fault; explicit recovery required");
  const auto reason=before.output.phase!=Phase::Fault?Reason::None:
      (before.output.reason==Reason::None?Reason::InvalidState:before.output.reason);
  const char* blocked=automatic_recovery_blocker(config_,s,reason,allowed_mode,collision_errors_only);
  if(blocked) {
    auto info=std::make_shared<RecoveryInfo>();info->automatic=true;info->status="blocked";
    info->error=blocked;info->started_at=info->completed_at=steady_seconds();
    info->attempt=recovery_attempt_;info->previous_reason=before.output.reason;
    info->previous_trigger=before.output.stop_trigger;info->previous_sample=before.sample;
    info->previous_fault_diagnostics=before.fault_diagnostics;
    {std::lock_guard<std::mutex> l(snapshot_mutex_);snapshot_.recovery=std::move(info);}
    throw std::runtime_error(std::string("automatic recovery blocked: ")+blocked);
  }
  automatic_recovery_used_=true;
  recover_locked(true,generation);
}
void Backend::recover_locked(bool automatic,std::uint64_t generation) {
  motion_inhibit_.check(generation);
  if(hardware_.read_only) throw std::runtime_error("read-only connection: recovery is disabled");
  if(running_) throw std::runtime_error("wait for stopping to finish before recovery");
  if(worker_.joinable()) worker_.join();
  const auto before=snapshot();
  auto info=std::make_shared<RecoveryInfo>();info->automatic=automatic;info->status="in_progress";
  info->started_at=steady_seconds();info->attempt=++recovery_attempt_;
  info->previous_reason=before.output.reason;info->previous_trigger=before.output.stop_trigger;
  info->previous_sample=before.sample;info->previous_fault_diagnostics=before.fault_diagnostics;
  {std::lock_guard<std::mutex> l(snapshot_mutex_);snapshot_.recovery=std::make_shared<RecoveryInfo>(*info);}
  try {
    auto s=read_idle();
    if(!s.finite() || s.dq.cwiseAbs().maxCoeff()>=config_.stop_velocity)
      throw std::runtime_error("recovery requires a valid stationary robot observation");
#ifdef FCC_WITH_FRANKA
    if(automatic && !simulation_) {
      const auto rs=robot_->robot.readOnce();
      hardware_.validate_end_effector(rs.m_ee,rs.F_x_Cee,rs.I_ee);
      s=robot_->sample(rs,config_.uses_osc_model());
      const auto errors=static_cast<std::string>(rs.current_errors);
      const bool allowed_errors=errors=="[]" || errors=="[\"joint_reflex\"]" ||
          errors=="[\"cartesian_reflex\"]" || errors=="[\"joint_reflex\", \"cartesian_reflex\"]";
      const bool allowed_mode=rs.robot_mode==franka::RobotMode::kIdle || rs.robot_mode==franka::RobotMode::kReflex;
      const auto reason=before.output.phase==Phase::Fault?before.output.reason:Reason::None;
      if(const char* blocked=automatic_recovery_blocker(config_,s,reason,allowed_mode,allowed_errors))
        throw std::runtime_error(std::string("automatic recovery blocked: ")+blocked);
    }
    if(!simulation_ && (s.robot_mode!=static_cast<int>(franka::RobotMode::kIdle) || s.robot_has_errors)) {
      motion_inhibit_.check(generation);
      info->robot_recovery_called=true;
      robot_->robot.automaticErrorRecovery();
      // An SDK call already in progress cannot be cancelled by this software latch.
      motion_inhibit_.check(generation);
      s=read_idle();
      RobotImpl::require_idle(s);
    }
#endif
    if(!s.finite() || s.dq.cwiseAbs().maxCoeff()>=config_.stop_velocity)
      throw std::runtime_error("recovery did not produce a valid stationary robot observation");
    motion_inhibit_.check(generation);
    info->status="succeeded";info->completed_at=steady_seconds();
    {std::lock_guard<std::mutex> l(snapshot_mutex_);
      snapshot_.sample=s; snapshot_.sampled_at=steady_seconds();
      snapshot_.stop_sample_valid=false;snapshot_.fault_diagnostics.reset();
      snapshot_.observation_source=ObservationSource::Recovery;
      snapshot_.output=Output{};snapshot_.recovery=info;}
    {std::lock_guard<std::mutex> l(error_mutex_);error_.clear();}
  } catch(const std::exception& error) {
    info->status="failed";info->error=error.what();info->completed_at=steady_seconds();
    {std::lock_guard<std::mutex> l(snapshot_mutex_);snapshot_.recovery=info;}
    throw;
  }
}
void Backend::inject_wrench(const V6& f) {
  if(!simulation_ || !f.allFinite()) throw std::invalid_argument("finite wrench injection is simulation-only");
  std::lock_guard<std::mutex> lock(injection_mutex_); injected_wrench_=f;
}
TelemetryStatus Backend::prepare_telemetry(std::size_t capacity, std::array<double,3> speed_limits) {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if(running_ || snapshot().output.phase!=Phase::Disarmed) throw std::runtime_error("Prepare telemetry only while disarmed");
  if(worker_.joinable()) worker_.join();
  telemetry_.prepare(capacity,speed_limits);telemetry_exported_=false;return telemetry_.status();
}
TelemetryStatus Backend::telemetry_status() {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);return telemetry_.status();
}
std::vector<TelemetryRecord> Backend::telemetry_data() {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if(running_) throw std::runtime_error("Export telemetry only after control stops");
  if(worker_.joinable()) worker_.join();
  auto data=telemetry_.copy();telemetry_exported_=true;return data;
}
void Backend::clear_telemetry(std::uint64_t id) {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if(running_ || (!telemetry_exported_ && telemetry_.status().count)) throw std::runtime_error("Export stopped capture before clearing");
  if(worker_.joinable()) worker_.join();
  telemetry_.clear(id);telemetry_exported_=false;
}
void Backend::run(Sample s,Config c,std::uint64_t epoch,std::uint64_t motion_generation,Command initial) {
  Snapshot current; current.sample=s;current.sampled_at=steady_seconds();
  current.output=snapshot().output; // Preserve action/epoch if SDK startup fails before its first callback.
  const bool relief_session=initial.relieve_contact,backoff_session=initial.backoff_joint_limits;
  const double relief_deadline=initial.expires_at;
  const Reason backoff_previous=current.output.backoff_previous_reason;
  current.collision_settings=snapshot().collision_settings;
  current.recovery=snapshot().recovery;
  std::shared_ptr<FaultDiagnostics> fault;
  auto cancelled_before_control=[&]() {
    const auto reason=pre_control_stop_reason(initial,steady_seconds(),
        stop_requested_ || !motion_inhibit_.permitted(motion_generation));
    if(reason==Reason::None) return false;
    // No SDK control was entered. Do not stamp the old preflight sample as a
    // fresh post-stop robot observation; an inactive refresh can confirm it.
    current=snapshot();current.sampled_at=0;
    current.output.phase=normal_stop_reason(reason)?Phase::Disarmed:Phase::Fault;current.output.reason=reason;
    current.output.stop_trigger=reason;current.output.finished=true;
    current.output.relief_released=current.output.backoff_reached=false;
    {std::lock_guard<std::mutex> lock(snapshot_mutex_);snapshot_=current;}
    telemetry_.finish();running_=false;return true;
  };
  if(cancelled_before_control()) return;
  try {
    Controller controller(c,s,epoch,steady_seconds(),(relief_session || backoff_session)?&initial:nullptr);
    Command latest;
    std::uint64_t record_cycle=0;
    auto record=[&](const std::array<double,7>& measured,const std::array<double,7>& theta,
                    const std::array<double,7>& dtheta,double received,double dt,bool sdk_values) {
      if(!telemetry_.active()) return;
      TelemetryRecord row;
      const auto& sample=current.sample;const auto& out=current.output;
      const std::uint64_t flags=(sample.robot_has_errors?1:0)|(out.joint_torque_limited?2:0)|
        (out.task_force_limited?4:0)|(out.task_moment_limited?8:0)|(out.reference_limited?16:0)|
        (out.osc_regularized?32:0)|(out.reference_speed_limited?64:0)|(out.effort_limited?128:0)|(sdk_values?256:0)|(telemetry_.trip()?512:0);
      row.ids={record_cycle++,out.epoch,out.sequence,static_cast<std::uint64_t>(out.mode),
               static_cast<std::uint64_t>(out.phase),static_cast<std::uint64_t>(out.reason),flags};
      row.values[0]=sample.robot_time;row.values[1]=received;row.values[2]=steady_seconds();
      row.values[4]=dt;row.values[5]=latest.accepted_at;row.values[6]=latest.expires_at;
      row.values[7]=sample.communication_success;
      std::size_t k=8;
      const auto copy=[&](const auto& v,int n){for(int i=0;i<n;++i) row.values[k++]=v[i];};
      copy(sample.q,7);copy(sample.dq,7);copy(theta,7);copy(dtheta,7);copy(measured,7);
      copy(sample.previous_torque,7);copy(out.torque,7);copy(sample.coriolis,7);
      copy(sample.position,3);copy(sample.orientation.coeffs(),4);
      const V6 twist=sample.jacobian*sample.dq;copy(twist,6);
      copy(out.reference_position,3);copy(out.reference_orientation.coeffs(),4);
      copy(out.reference_joints,7);copy(out.reference_joint_velocity,7);
      copy(sample.external_wrench,6);copy(sample.external_wrench_stiffness,6);copy(sample.external_torque,7);
      row.values[3]=steady_seconds();telemetry_.append(row);
    };
    auto tick=[&](const Sample& sample,double dt,double received) {
      const double begin=steady_seconds();
      {std::unique_lock<std::mutex> lock(command_mutex_,std::try_to_lock);
        if(lock.owns_lock()) latest=command_;}
      current.sample=sample; current.sampled_at=received;
      current.observation_source=simulation_?ObservationSource::Simulation:ObservationSource::ControlCallback;
      const V6 measured_twist=sample.jacobian*sample.dq;
      const bool experiment_stop=telemetry_.speed_trip(sample.dq.cwiseAbs().maxCoeff(),measured_twist.head<3>().norm(),measured_twist.tail<3>().norm());
      current.output=controller.step(sample,latest,stop_requested_ || !motion_inhibit_.permitted(motion_generation) || experiment_stop,begin,dt);
      if(relief_session) {
        // Keep action identity through a missed first mailbox read or an invalid
        // first callback; this changes diagnostics, never controller decisions.
        current.output.relief_mode=true;current.output.relief_deadline=relief_deadline;
        current.output.relief_manual=initial.relief_options.manual_stop;
      }
      if(backoff_session) {
        current.output.backoff_mode=true;current.output.backoff_deadline=initial.expires_at;
        current.output.backoff_previous_reason=backoff_previous;
      }
      if(!current.stop_sample_valid && current.output.stop_trigger!=Reason::None) {
        current.stop_sample=sample;
        current.stop_sampled_at=begin;
        current.stop_sample_valid=true;
      }
      current.cycle_time=steady_seconds()-begin;
      publish(current);
      return current.output;
    };
    if(simulation_) {
      auto next=std::chrono::steady_clock::now();
      while(true) {
        {std::unique_lock<std::mutex> lock(injection_mutex_,std::try_to_lock);
          if(lock.owns_lock()) s.external_wrench=injected_wrench_;}
        s.external_wrench_tcp_base=s.external_wrench;s.wrench_tcp_valid=true;s.relief_setup_valid=true;
        s.external_wrench_stiffness.head<3>()=s.orientation.conjugate()*V3(s.external_wrench.head<3>());
        s.external_wrench_stiffness.tail<3>()=s.orientation.conjugate()*V3(s.external_wrench.tail<3>());
        s.external_torque=s.jacobian.transpose()*s.external_wrench;
        const double received=steady_seconds();
        const auto out=tick(s,.001,received);
        const std::array<double,7> unavailable{};record(unavailable,unavailable,unavailable,received,.001,false);
        if(out.finished) break;
        // Deliberately simple test plant, NOT a Panda dynamics/contact simulator.
        const V7 acceleration=(out.torque-s.jacobian.transpose()*s.external_wrench-2*s.dq)/3;
        s.dq+=.001*acceleration; s.q+=.001*s.dq;
        const V6 v=s.jacobian*s.dq;
        s.position+=.001*v.head<3>();
        const V3 angle=.001*v.tail<3>();
        if(angle.norm()>1e-12) s.orientation=(Eigen::Quaterniond(Eigen::AngleAxisd(angle.norm(),angle.normalized()))*s.orientation).normalized();
        s.previous_torque=out.torque;
        next+=std::chrono::milliseconds(1);
        std::this_thread::sleep_until(next);
      }
    } else {
#ifdef FCC_WITH_FRANKA
      if(cancelled_before_control()) return;
      robot_->robot.control([&](const franka::RobotState& rs,franka::Duration period)->franka::Torques {
        const double callback_begin=steady_seconds();
        const auto out=tick(robot_->sample(rs,c.uses_osc_model()),period.toSec(),callback_begin);
        record(rs.tau_J,rs.theta,rs.dtheta,callback_begin,period.toSec(),true);
        current.cycle_time=steady_seconds()-callback_begin;
        publish(current);
        std::array<double,7> values;
        Eigen::Map<V7>(values.data())=out.torque;
        franka::Torques torque(values);
        return handoff_hardware_stop(out)?franka::MotionFinished(torque):torque;
      },true,100.0);
      // The SDK waits for the motion-finished acknowledgement. The final torque
      // callback can still show velocity; use new robot samples to confirm stop.
      HardwareStopMonitor monitor(steady_seconds(),c,backoff_session?.01:(relief_session?.02:0.));
      if(relief_session || backoff_session) {current.output.phase=Phase::Stopping;current.output.finished=false;}
      while(true) {
        const auto rs=robot_->robot.readOnce();
        current.sample=robot_->sample(rs,c.uses_osc_model());
        current.sampled_at=steady_seconds();
        current.observation_source=ObservationSource::StopRead;
        const bool idle=rs.robot_mode==franka::RobotMode::kIdle && !rs.current_errors;
        const auto observed=monitor.observe(current.sampled_at,current.sample,idle);
        observe_stop_guards(c,current.sample,current.output,!relief_session && !backoff_session);
        if(relief_session) {
          auto& out=current.output;
          out.relief_translation=(current.sample.position-out.relief_anchor_position).norm();
          out.relief_rotation=current.sample.orientation.angularDistance(out.relief_anchor_orientation);
          const auto invalid=contact_relief_observation_reason(c,current.sample,initial.relief_options,
              out.relief_anchor_position,out.relief_anchor_orientation,
              observed==StopObservation::Confirmed && out.relief_released,&out.relief_anchor_joints);
          if(invalid!=Reason::None) {
            if(normal_stop_reason(out.reason)) out.reason=invalid;
            if(out.stop_trigger==Reason::None) out.stop_trigger=invalid;
            out.relief_released=false;
          }
        }
        if(backoff_session) {
          auto& out=current.output;
          const auto invalid=backoff_observation_reason(c,current.sample,initial.backoff_options,
              out.backoff_plan,out.backoff_best_progress,observed==StopObservation::Confirmed && out.backoff_reached);
          if(invalid!=Reason::None) {
            if(normal_stop_reason(out.reason) ||
               (backoff_local_fault_allowed(out.reason) && !backoff_local_fault_allowed(invalid))) out.reason=invalid;
            if(out.stop_trigger==Reason::None) out.stop_trigger=invalid;
            out.backoff_reached=false;
          } else update_backoff_progress(current.sample,out.backoff_plan,out.backoff_best_progress);
        }
        current.output.torque=current.sample.previous_torque;
        current.output.task_wrench.setZero();
        publish(current);
        if(observed==StopObservation::Confirmed) {
          current.output.phase=normal_stop_reason(current.output.reason)?Phase::Disarmed:Phase::Fault;
          current.output.finished=true;
          break;
        }
        if(observed==StopObservation::TimedOut) {
          // Build diagnostics outside the torque callback before replacing the
          // summary reason. A reflex may otherwise appear only as a timeout.
          const std::string detail=std::string("robot-side stop did not produce confirmed stationary idle state; initial trigger=")+
              reason_name(current.output.stop_trigger)+"; robot_mode="+robot_mode_name(current.sample.robot_mode)+
              "; current_errors="+static_cast<std::string>(rs.current_errors)+
              "; last_motion_errors="+static_cast<std::string>(rs.last_motion_errors);
          robot_->robot.stop();
          current.output.phase=Phase::Fault;
          current.output.reason=Reason::StopTimeout;
          current.output.finished=true;
          throw std::runtime_error(detail);
        }
      }
#endif
    }
  }
#ifdef FCC_WITH_FRANKA
  catch(const franka::ControlException& e) {
    fault=capture_control_exception(e,steady_seconds());
    latch_control_failure(current.output,fault->reason);
    std::lock_guard<std::mutex> lock(error_mutex_);error_=e.what();
  }
#endif
  catch(const std::exception& e) {
    current.output.planned_joint_velocity.setZero();current.output.friction_compensation_torque.setZero();
    if(current.output.relief_mode) current.output.relief_released=false;
    if(current.output.backoff_mode) current.output.backoff_reached=false;
    current.output.phase=Phase::Fault;
    if(current.output.reason==Reason::None || normal_stop_reason(current.output.reason))
      current.output.reason=Reason::Communication;
    if(current.output.stop_trigger==Reason::None) current.output.stop_trigger=current.output.reason;
    current.output.finished=true;
    fault=std::make_shared<FaultDiagnostics>();
    fault->reason=current.output.reason;fault->exception_type="backend_exception";
    fault->message=e.what();fault->caught_at=steady_seconds();
    std::lock_guard<std::mutex> lock(error_mutex_); error_=e.what();
  }
  if(fault) {
    // Preserve the exception records even if the subsequent diagnostic read
    // blocks. Publish an immutable copy; readers never race its construction.
    current.observation_source=ObservationSource::CachedAfterFault;
    auto pending=std::make_shared<FaultDiagnostics>(*fault);
    pending->post_read_attempted=!simulation_;
    current.fault_diagnostics=std::move(pending);
    publish(current);
#ifdef FCC_WITH_FRANKA
    if(!simulation_) {
      observe_after_fault(*fault,current.sample.robot_time,
          [&](){return robot_->robot.readOnce();},steady_seconds);
      if(fault->post_read_fresh && fault->post_read_valid) {
        current.sample=fault->post_observation.sample;
        current.sampled_at=fault->post_read_at;
        current.observation_source=ObservationSource::PostFaultRead;
        current.output.torque=current.sample.previous_torque;
        current.output.task_wrench.setZero();
      }
    }
#endif
    current.fault_diagnostics=std::move(fault);
  }
  {std::lock_guard<std::mutex> lock(snapshot_mutex_);snapshot_=current;}
  telemetry_.finish();
  running_=false;
}
void Backend::close() {
  request_stop();
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if(worker_.joinable()) {
    const double deadline=steady_seconds()+config_.stop_timeout+.5;
    while(running_ && steady_seconds()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
#ifdef FCC_WITH_FRANKA
    if(running_ && !simulation_) {try {robot_->robot.stop();} catch(...) {}}
#endif
    worker_.join();
  }
}
} // namespace fcc
