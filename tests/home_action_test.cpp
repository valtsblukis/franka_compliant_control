// Fixed-sample action lifecycle tests. No robot or simulated motion integration.
#include "controller.hpp"
#include "hardware_stop.hpp"
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
  c.validate();return c;
}
static Sample sample() {
  Sample s;s.q<<0,-.4,0,-1.9,0,1.5,0;s.position<<.45,0,.4;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1;
  s.mass=3*M7::Identity();s.dynamics_valid=true;return s;
}
static Command home(const Sample& s,Mode mode) {
  Command cmd;cmd.mode=mode;cmd.epoch=1;cmd.sequence=1;cmd.expires_at=30;
  cmd.joints=s.q;cmd.position=s.position;cmd.orientation=s.orientation;cmd.has_orientation=mode==Mode::Cartesian;
  cmd.finish_at_target=true;return cmd;
}
int main() {
  Eigen::internal::set_is_malloc_allowed(false);
  const auto c=config();
  for(auto mode:{Mode::Joint,Mode::Cartesian}) {
    auto s=sample();auto cmd=home(s,mode);Controller ctrl(c,s,1,0);
    auto out=ctrl.step(s,cmd,false,0,.001);
    assert(out.phase==Phase::Active && out.homing && !out.home_reached);
    assert(ctrl.step(s,cmd,false,.19,.001).phase==Phase::Active);
    out=ctrl.step(s,cmd,false,.201,.001);
    assert(out.phase==Phase::Stopping && out.reason==Reason::HomeReached && out.home_reached);
    assert(handoff_hardware_stop(out) && normal_stop_reason(out.reason));
    s.previous_torque=out.torque;
    out=ctrl.step(s,cmd,false,.302,.001);
    assert(out.phase==Phase::Disarmed && out.finished && out.home_reached);

    Controller stream(c,s,1,0);cmd.finish_at_target=false;cmd.expires_at=2;
    stream.step(s,cmd,false,0,.001);
    assert(stream.step(s,cmd,false,1,.001).phase==Phase::Active); // Ordinary pose/joint hold unchanged.

    cmd=home(s,mode);Controller moving(c,s,1,0);moving.step(s,cmd,false,0,.001);
    s.dq[0]=.05;moving.step(s,cmd,false,.15,.001);s.dq[0]=0;
    moving.step(s,cmd,false,.2,.001);
    assert(moving.step(s,cmd,false,.35,.001).phase==Phase::Active);
    assert(moving.step(s,cmd,false,.401,.001).reason==Reason::HomeReached);

    s=sample();cmd=home(s,mode);Controller far(c,s,1,0);
    if(mode==Mode::Joint) s.q[0]+=.1;
    else s.position[0]+=.1;
    assert(far.step(s,cmd,false,1,.001).phase==Phase::Active);
    out=far.step(s,cmd,false,30.001,.001);
    assert(out.reason==Reason::HomeTimeout && !out.home_reached && normal_stop_reason(out.reason));

    s=sample();cmd=home(s,mode);Controller cancelled(c,s,1,0);
    cancelled.step(s,cmd,false,0,.001);
    auto ordinary=cmd;ordinary.sequence=2;ordinary.finish_at_target=false;
    cancelled.step(s,ordinary,false,.1,.001);
    out=cancelled.step(s,ordinary,false,.5,.001);
    assert(out.phase==Phase::Active && !out.homing && out.home_superseded && !out.home_reached);
    assert(cancelled.step(s,cmd,false,.6,.001).phase==Phase::Active); // Old home cannot return.

    Controller stopped(c,s,1,0);stopped.step(s,cmd,false,0,.001);
    out=stopped.step(s,cmd,true,.1,.001);
    assert(out.reason==Reason::UserStop && !out.home_reached);
    Controller faulted(c,s,1,0);faulted.step(s,cmd,false,0,.001);
    out=faulted.step(s,cmd,false,.201,.011);
    assert(out.reason==Reason::Timing && !out.home_reached && !normal_stop_reason(out.reason));
  }
  std::cout<<"Finite homing reach/dwell/stop/timeout/supersession checks passed; no robot or plant\n";
}
