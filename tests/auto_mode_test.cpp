// Fixed samples only: verifies native law selection, not simulated robot motion.
#include "controller.hpp"
#include "fr3_limits.hpp"
#include <cassert>
#include <iostream>
using namespace fcc;

static Config config() {
  Config c("fr3");c.compliance_mode="auto";c.effort_limit_policy="mode_bounded";
  c.osc_feedback_mode="saturated";c.motion_limit_policy="franka";c.collision_policy="robot_configured";
  c.joint_lower=fr3_limits::lower();c.joint_upper=fr3_limits::upper();c.joint_limit_margin=0;
  c.torque_limits<<87,87,87,87,12,12,12;c.torque_rate_limit=1000;
  c.measured_joint_velocity_limits=fr3_limits::speed();
  c.measured_translation_velocity_limit=3;c.measured_rotation_velocity_limit=2.5;
  c.osc_translation_stiffness.setConstant(400);c.osc_translation_damping.setConstant(40);
  c.osc_rotation_stiffness.setConstant(120);c.osc_rotation_damping.setConstant(8);
  c.commanded_force_limit=20;c.commanded_moment_limit=6;
  c.joint_stiffness<<100,100,100,100,50,50,30;c.joint_damping<<10,10,10,10,5,5,3;
  c.joint_feedback_torque_limits<<8,8,8,8,3,3,3;c.validate();return c;
}
static Sample sample() {
  Sample s;s.q<<0,-.4,0,-1.9,0,1.5,0;s.position<<.45,0,.4;
  s.mass=3*M7::Identity();s.dynamics_valid=true;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1;
  s.coriolis.setConstant(.02);return s;
}
static Command command(const Sample& s, Mode mode, std::uint64_t seq) {
  Command cmd;cmd.epoch=1;cmd.sequence=seq;cmd.expires_at=1;
  cmd.mode=mode;cmd.position=s.position;cmd.joints=s.q;
  return cmd;
}
static void matches_standalone_laws_and_switches_without_restart() {
  auto c=config();auto s=sample();Controller mixed(c,s,1,0);
  auto joint_config=c;joint_config.compliance_mode="joint_impedance";
  joint_config.effort_limit_policy="joint_bounded";joint_config.osc_feedback_mode="linear";
  auto cart_config=c;cart_config.compliance_mode="operational_space";cart_config.effort_limit_policy="task_bounded";
  for(int tick=1;tick<=20;tick++) {
    const auto mode=tick%2?Mode::Joint:Mode::Cartesian;
    auto cmd=command(s,mode,tick);
    cmd.joints[0]+=.1;cmd.joint_velocity[0]=.02;
    cmd.position[0]+=.03;
    Controller single(mode==Mode::Joint?joint_config:cart_config,s,1,tick*.001);
    const auto expected=single.step(s,cmd,false,tick*.001,.001);
    const auto out=mixed.step(s,cmd,false,tick*.001,.001);
    assert(out.phase==Phase::Active && out.mode==mode && out.epoch==1 && out.sequence==cmd.sequence);
    assert((out.torque-expected.torque).norm()<1e-10);
    assert((out.task_wrench-expected.task_wrench).norm()<1e-10);
    assert((out.joint_feedback_torque-expected.joint_feedback_torque).norm()<1e-10);
    assert((out.torque-s.previous_torque).cwiseAbs().maxCoeff()<=1+1e-12);
    assert(!out.yielding && !out.reference_limited);
    s.previous_torque=out.torque;
  }
}
static void omitted_orientation_and_posture_reseed_on_cartesian_reentry() {
  auto c=config();auto s=sample();Controller mixed(c,s,1,0);
  auto cart=command(s,Mode::Cartesian,1);cart.has_orientation=true;
  cart.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.5,V3::UnitZ()));
  cart.has_posture=true;cart.posture=s.q;cart.posture[6]=1;
  mixed.step(s,cart,false,0,.001);
  auto joint=command(s,Mode::Joint,2);joint.joints[6]=.4;
  mixed.step(s,joint,false,.001,.001);
  s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.2,V3::UnitY()));s.q[6]=.3;
  cart=command(s,Mode::Cartesian,3); // No explicit orientation/posture.
  const auto out=mixed.step(s,cart,false,.002,.001);
  assert(out.phase==Phase::Active && out.reference_orientation.angularDistance(s.orientation)<1e-12);
  assert(out.task_wrench.norm()<1e-10);
  assert((out.torque-s.coriolis).norm()<1e-10); // No old posture spring leaks back in.
}
static void stale_updates_do_not_switch_and_stops_remain_authoritative() {
  auto c=config();auto s=sample();Controller mixed(c,s,1,0);
  auto joint=command(s,Mode::Joint,5);auto out=mixed.step(s,joint,false,0,.001);
  assert(out.mode==Mode::Joint);
  auto cart=command(s,Mode::Cartesian,4);
  out=mixed.step(s,cart,false,.001,.001);assert(out.mode==Mode::Joint && out.sequence==5);
  cart.sequence=6;cart.expires_at=.001;
  out=mixed.step(s,cart,false,.002,.001);assert(out.mode==Mode::Joint && out.sequence==5);
  cart.expires_at=1;cart.epoch=2;
  out=mixed.step(s,cart,false,.003,.001);assert(out.mode==Mode::Joint && out.sequence==5);
  cart.epoch=1;
  out=mixed.step(s,cart,true,.004,.001);assert(out.phase==Phase::Stopping && out.mode==Mode::Joint);
  assert(out.stop_trigger==Reason::UserStop);
  Controller timeout(c,s,1,0);joint.expires_at=.02;timeout.step(s,joint,false,0,.001);
  cart.expires_at=.01;
  assert(timeout.step(s,cart,false,.03,.001).stop_trigger==Reason::Watchdog);
}
static void rank_deficient_joints_and_regularized_cartesian_both_work() {
  auto c=config();auto s=sample();s.jacobian.setZero();Controller mixed(c,s,1,0);
  auto joint=command(s,Mode::Joint,1);joint.joints[0]+=.1;
  assert(mixed.step(s,joint,false,0,.001).phase==Phase::Active);
  auto cart=command(s,Mode::Cartesian,2);cart.position[0]+=.1;
  const auto out=mixed.step(s,cart,false,.001,.001);
  assert(out.phase==Phase::Active && out.osc_regularized && out.torque.allFinite());
}
int main() {
  Eigen::internal::set_is_malloc_allowed(false);
  matches_standalone_laws_and_switches_without_restart();
  omitted_orientation_and_posture_reseed_on_cartesian_reentry();
  stale_updates_do_not_switch_and_stops_remain_authoritative();
  rank_deficient_joints_and_regularized_cartesian_both_work();
  std::cout<<"Unified joint/Cartesian switching checks passed; no robot or plant integration\n";
}
