// Fixed-sample controller checks only: no plant integration or robot connection.
#include "controller.hpp"
#include "fr3_limits.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace fcc;

static Config config() {
  Config c("fr3");c.compliance_mode="joint_impedance";c.effort_limit_policy="joint_bounded";
  c.motion_limit_policy="franka";c.collision_policy="robot_configured";
  c.joint_lower=fr3_limits::lower();c.joint_upper=fr3_limits::upper();c.joint_limit_margin=0;
  c.measured_joint_velocity_limits=fr3_limits::speed();
  c.measured_translation_velocity_limit=3;c.measured_rotation_velocity_limit=2.5;
  c.torque_limits<<87,87,87,87,12,12,12;c.torque_rate_limit=1000;
  c.joint_stiffness<<100,100,100,100,50,50,30;
  c.joint_damping<<10,10,10,10,5,5,3;
  c.joint_feedback_torque_limits<<8,8,8,8,3,3,3;
  c.validate();return c;
}
static Sample sample() {
  Sample s;s.q<<0,-.4,0,-1.9,0,1.5,0;
  // Rank-deficient Jacobian, no mass model: neither should inhibit joint PD.
  s.jacobian.setZero();s.dynamics_valid=false;s.coriolis.setConstant(.05);
  return s;
}
static Command target(const Sample& s) {
  Command c;c.epoch=1;c.sequence=1;c.mode=Mode::Joint;c.joints=s.q;c.expires_at=10;
  return c;
}
static void exact_joint_law_and_direct_replacement() {
  const auto c=config();auto s=sample();auto cmd=target(s);Controller controller(c,s,1,0);
  const auto preflight=controller.step(s,Command{},false,0,.001);
  assert(preflight.phase==Phase::Armed && preflight.mode==Mode::Joint);
  cmd.joints[0]+=.002;cmd.joint_velocity[0]=.01;s.dq[0]=.005;
  auto out=controller.step(s,cmd,false,.001,.001);
  assert(out.phase==Phase::Active && out.mode==Mode::Joint && !out.yielding);
  assert((out.reference_joints-cmd.joints).norm()<1e-12);
  assert(out.reference_joint_velocity[0]==.01);
  assert(std::abs(out.joint_spring_torque[0]-.2)<1e-12);
  assert(std::abs(out.joint_feedback_torque[0]-.25)<1e-12);
  assert(std::abs(out.torque[0]-.3)<1e-12); // Coriolis added once, no gravity term.
  assert(out.task_wrench.isZero() && !out.joint_feedback_limited);
  cmd.joints[0]=s.q[0]-.5;++cmd.sequence;
  s.previous_torque=out.torque;
  out=controller.step(s,cmd,false,.002,.001);
  assert(out.joint_spring_torque[0]==-8 && out.joint_spring_limited);
  assert(out.reference_joints[0]==cmd.joints[0]);
  assert((out.torque-s.previous_torque).cwiseAbs().maxCoeff()<=1+1e-12);
  cmd.joints=s.q;cmd.joint_velocity=s.dq;++cmd.sequence;
  out=controller.step(s,cmd,false,.003,.001);
  assert(out.joint_feedback_torque.norm()<1e-12); // No stored equilibrium/integral.
}
static void saturation_keeps_damping_and_separate_actuator_limits() {
  const auto c=config();
  for(int joint=0;joint<7;joint++) {
    for(double sign:{-1.,1.}) {
      auto s=sample();auto cmd=target(s);Controller control(c,s,1,0);
      cmd.joints[joint]+=sign*.3; // Exceeds spring cap and old joint-error leash.
      s.dq[joint]=sign*.1;
      auto out=control.step(s,cmd,false,.001,.001);
      const double expected=sign*(c.joint_feedback_torque_limits[joint]-.1*c.joint_damping[joint]);
      assert(out.phase==Phase::Active && out.joint_spring_limited && !out.reference_limited);
      assert(std::abs(out.joint_feedback_torque[joint]-expected)<1e-12);
      s.dq[joint]=-sign*.1;
      out=control.step(s,cmd,false,.002,.001);
      assert(out.joint_feedback_limited);
      assert(std::abs(out.joint_feedback_torque[joint]-sign*c.joint_feedback_torque_limits[joint])<1e-12);
      assert((out.torque.cwiseAbs().array()<=c.torque_limits.array()).all());
      assert(out.effort_limited);
    }
  }
}
static void contact_does_not_retarget_and_guards_still_stop() {
  const auto c=config();auto s=sample();auto cmd=target(s);cmd.joints[0]+=.2;
  Controller control(c,s,1,0);
  s.external_torque.setConstant(100);s.external_wrench.setConstant(100);
  auto out=control.step(s,cmd,false,.001,.001);
  assert(out.phase==Phase::Active && !out.yielding && out.drive_scale==1);
  assert((out.reference_joints-cmd.joints).norm()<1e-12);
  s.external_torque.setZero();s.external_wrench.setZero();
  out=control.step(s,cmd,false,.002,.001);
  assert(out.joint_spring_torque[0]==8); // Release retains original spring.
  for(Reason expected:{Reason::Watchdog,Reason::UserStop,Reason::JointLimit,Reason::Overspeed,
                       Reason::Timing,Reason::InvalidState,Reason::ReferenceError}) {
    auto observed=sample();auto command=target(observed);Controller guarded(c,observed,1,0);
    guarded.step(observed,command,false,0,.001);
    if(expected==Reason::JointLimit) observed.q[0]=c.joint_lower[0];
    if(expected==Reason::Overspeed) observed.dq[0]=2.63;
    if(expected==Reason::InvalidState) observed.q[0]=NAN;
    if(expected==Reason::ReferenceError) {command.mode=Mode::Cartesian;++command.sequence;}
    out=guarded.step(observed,command,expected==Reason::UserStop,
        expected==Reason::Watchdog?11:.001,expected==Reason::Timing?.011:.001);
    assert(out.phase==Phase::Fault || out.phase==Phase::Stopping);
    assert(out.stop_trigger==expected);
  }
  auto near=sample();near.q[0]=c.joint_upper[0]-.05;
  auto command=target(near);Controller guarded(c,near,1,0);
  guarded.step(near,command,false,0,.001);
  near.dq[0]=.12;
  assert(guarded.step(near,command,false,.001,.001).stop_trigger==Reason::Overspeed);
}
int main() {
  Eigen::internal::set_is_malloc_allowed(false);
  exact_joint_law_and_direct_replacement();saturation_keeps_damping_and_separate_actuator_limits();
  contact_does_not_retarget_and_guards_still_stop();
  std::cout<<"Direct joint impedance algebra, retargeting and guards passed; no plant or robot\n";
}
