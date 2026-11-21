// Offline exception and collision-setter tests. Never constructs franka::Robot.
#include "robot_diagnostics.hpp"
#include <cassert>
#include <iostream>
using namespace fcc;

static franka::RobotState state(std::uint64_t ms, bool reflex=false) {
  franka::RobotState s{};
  s.O_T_EE={1,0,0,0,0,1,0,0,0,0,1,0,.4,.1,.5,1};
  s.F_T_EE={1,0,0,0,0,1,0,0,0,0,1,0,0,0,.174,1};
  s.EE_T_K={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  s.time=franka::Duration(ms);s.robot_mode=reflex?franka::RobotMode::kReflex:franka::RobotMode::kMove;
  if(reflex) {
    std::array<bool,41> flags{};flags[7]=true;s.current_errors=franka::Errors(flags);
    assert(s.current_errors.cartesian_reflex);
    s.last_motion_errors=s.current_errors;s.cartesian_collision[4]=1;
  }
  s.O_F_ext_hat_K={1,2,3,4,5,6};s.K_F_ext_hat_K={6,5,4,3,2,1};
  s.control_command_success_rate=1;
  return s;
}
struct FakeRobot {
  int calls=0;bool fail=false;
  void setCollisionBehavior(const std::array<double,7>& la,const std::array<double,7>& ua,
      const std::array<double,7>& ln,const std::array<double,7>& un,
      const std::array<double,6>& fa,const std::array<double,6>& ga,
      const std::array<double,6>& fn,const std::array<double,6>& gn) {
    ++calls;
    assert(la==ln && ua==un && fa==fn && ga==gn);
    assert((ua==std::array<double,7>{16,16,16,16,8,8,8}));
    assert(la[0]==.7*16 && la[6]==.7*8);
    assert((ga==std::array<double,6>{40,40,40,20,20,20}));
    assert((fa==std::array<double,6>{15,15,15,3,3,3}));
    if(fail) throw std::runtime_error("mock setter rejected request");
  }
};
static void collision_acknowledgement_is_explicit() {
  Config c("fr3");c.external_torque_limits<<16,16,16,16,8,8,8;
  c.contact_force_soft=15;c.contact_moment_soft=3;c.contact_force_stop=40;c.contact_moment_stop=20;
  for(const auto policy:{"configured","robot_configured","robot"}) {
    c.collision_policy=policy;FakeRobot robot;auto request=collision_request(c);
    if(applies_robot_collision_settings(c)) apply_collision_request(robot,request);
    assert(robot.calls==(c.collision_policy=="robot"?0:1));
    assert(request.status==(robot.calls?CollisionApplication::Applied:CollisionApplication::NotApplied));
  }
  auto request=collision_request(c);FakeRobot robot;robot.fail=true;
  try {apply_collision_request(robot,request);assert(false);} catch(const std::runtime_error&) {}
  assert(robot.calls==1 && request.status==CollisionApplication::Failed && request.acknowledged_at==0);
}
static void sdk_records_and_reason_are_preserved() {
  std::vector<franka::Record> log(2);log[0].state=state(1000);log[1].state=state(1001,true);
  log[0].command.torques.tau_J[0]=1.2345678901234567;
  log[1].command.torques.tau_J[6]=.25;
  franka::ControlException error("libfranka: motion aborted by reflex! [\"cartesian_reflex\"]",log);
  auto f=capture_control_exception(error,123.4);
  assert(f->reason==Reason::RobotCartesianReflex && f->caught_at==123.4);
  assert(f->records.size()==2 && f->records[0].sent_torques[0]==log[0].command.torques.tau_J[0]);
  assert(f->records[1].state.sample.robot_time==1.001);
  assert(f->records[1].state.sample.cartesian_collision[4]==1);
  assert(f->records[1].state.sample.external_wrench[0]==1);
  assert(f->records[1].state.sample.external_wrench_stiffness[0]==6);
  assert(f->records[1].state.current_errors.find("cartesian_reflex")!=std::string::npos);
  assert(f->records[1].state.robot_state_json.find("\"F_T_EE\"")!=std::string::npos);
  assert(f->records[1].state.robot_state_json.find("\"cartesian_collision\"")!=std::string::npos);
  Output out;out.phase=Phase::Active;out.stop_trigger=Reason::MomentLimit;
  out.relief_mode=true;out.relief_released=true;
  out.planned_joint_velocity.setConstant(.1);out.friction_compensation_torque.setConstant(.5);
  latch_control_failure(out,f->reason);
  assert(out.planned_joint_velocity.isZero(0) && out.friction_compensation_torque.isZero(0));
  assert(!out.relief_released);
  assert(out.reason==Reason::RobotCartesianReflex && out.stop_trigger==Reason::MomentLimit);
  assert(out.phase==Phase::Fault && out.finished);
  auto empty=capture_control_exception(franka::ControlException("abort [\"joint_reflex\"]"),1);
  assert(empty->records.empty() && empty->reason==Reason::RobotJointReflex);
  log[1].state.current_errors=franka::Errors();log[1].state.robot_mode=franka::RobotMode::kMove;
  auto stale=capture_control_exception(franka::ControlException("invalid controller command",log),1);
  assert(stale->reason==Reason::RobotControlError); // Old last_motion_errors must not classify this abort.
}
static void post_fault_read_is_read_only_and_preserves_primary_fault() {
  const franka::ControlException error("abort [\"cartesian_reflex\"]",{{state(1001,true),{}}});
  for(int scenario=0;scenario<4;scenario++) {
    auto f=capture_control_exception(error,123.4);int reads=0;
    observe_after_fault(*f,1.0,[&]()->franka::RobotState {
      ++reads;
      if(scenario==1) throw std::runtime_error("mock receive timeout");
      auto s=state(scenario==2?1001:1020,true);
      if(scenario==3) s.q[0]=NAN;
      return s;
    },[](){return 123.5;});
    assert(reads==1 && f->post_read_attempted && f->post_read_completed);
    assert(f->reason==Reason::RobotCartesianReflex && f->message==error.what());
    assert(f->records.size()==1);
    if(scenario==0) {
      assert(f->post_read_fresh && f->post_read_valid && f->post_read_error.empty());
      assert(!f->post_observation.sample.kinematics_valid); // Do not fabricate a twist after the abort.
      assert(f->post_observation.sample.robot_has_errors);
    } else if(scenario==1) {
      assert(!f->post_read_fresh && !f->post_read_valid && f->post_read_error=="mock receive timeout");
    } else if(scenario==2) {
      assert(!f->post_read_fresh); // Newer than cached callback is insufficient if SDK log is newer still.
    } else assert(f->post_read_fresh && !f->post_read_valid);
  }
}
static void canonical_wrench_uses_full_stiffness_transform() {
  auto rs=state(1);rs.K_F_ext_hat_K={0,2,0,0,0,0};
  Eigen::Matrix4d ee=Eigen::Matrix4d::Identity(),k=Eigen::Matrix4d::Identity();
  ee.topLeftCorner<3,3>()=Eigen::AngleAxisd(std::acos(-1.)/2,V3::UnitZ()).toRotationMatrix();
  ee.topRightCorner<3,1>()=V3(.4,.1,.5);
  k.topLeftCorner<3,3>()=Eigen::AngleAxisd(std::acos(-1.)/2,V3::UnitX()).toRotationMatrix();
  k.topRightCorner<3,1>()=V3(.1,0,0);
  Eigen::Map<Eigen::Matrix4d>(rs.O_T_EE.data())=ee;
  Eigen::Map<Eigen::Matrix4d>(rs.EE_T_K.data())=k;
  V6 wrench;
  assert(sdk_wrench_at_ee_base(rs,wrench));
  assert((wrench.head<3>()-V3(0,0,2)).norm()<1e-12);
  assert((wrench.tail<3>()-V3(.2,0,0)).norm()<1e-12);
  // Absolute EE translation must not leak a base-origin moment into TCP relief.
  ee.topRightCorner<3,1>()=V3(9,8,7);Eigen::Map<Eigen::Matrix4d>(rs.O_T_EE.data())=ee;
  V6 translated;assert(sdk_wrench_at_ee_base(rs,translated));
  assert((translated-wrench).norm()<1e-12);
  rs.EE_T_K[15]=0;assert(!sdk_wrench_at_ee_base(rs,wrench));
}
static void loaded_relief_requires_unchanged_tool_load_and_frames() {
  auto rs=state(1);HardwareConfig h;h.check_end_effector=true;
  h.flange_to_tcp=rs.F_T_EE;h.expected_ee_mass=.9;rs.m_ee=.9;
  h.expected_ee_com={0,0,.05};rs.F_x_Cee=h.expected_ee_com;
  h.expected_ee_inertia={.001,0,0,0,.001,0,0,0,.001};rs.I_ee=h.expected_ee_inertia;
  h.load_mass=.5;rs.m_load=.5;h.load_com={0,.01,.02};rs.F_x_Cload=h.load_com;
  h.load_inertia={.001,0,0,0,.001,0,0,0,.001};rs.I_load=h.load_inertia;
  assert(relief_hardware_setup_matches(h,rs));
  for(int cause=0;cause<7;cause++) {
    auto bad=rs;auto config=h;
    if(cause==0) bad.m_load=0; // Model basis would change if setLoad were called.
    if(cause==1) bad.F_x_Cload[1]+=.01;
    if(cause==2) bad.I_load[0]+=.001;
    if(cause==3) bad.m_ee+=.1;
    if(cause==4) bad.F_T_EE[14]+=.01;
    if(cause==5) bad.EE_T_K[12]+=.01;
    if(cause==6) config.check_end_effector=false;
    assert(!relief_hardware_setup_matches(config,bad));
  }
}
int main() {
  collision_acknowledgement_is_explicit();sdk_records_and_reason_are_preserved();
  post_fault_read_is_read_only_and_preserves_primary_fault();
  canonical_wrench_uses_full_stiffness_transform();
  loaded_relief_requires_unchanged_tool_load_and_frames();
  std::cout<<"Offline collision acknowledgement, SDK exception and post-fault read checks passed\n";
}
