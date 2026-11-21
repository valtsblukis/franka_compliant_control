// Fixed-sample production-law tests; no robot connection or plant assumptions.
#include "controller.hpp"
#include "fr3_limits.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
using namespace fcc;

static Config config(double f=0.6, double epsilon=0.03) {
  Config c("fr3");c.compliance_mode="auto";c.effort_limit_policy="mode_bounded";
  c.osc_feedback_mode="saturated";c.motion_limit_policy="franka";
  c.joint_lower=fr3_limits::lower();c.joint_upper=fr3_limits::upper();c.joint_limit_margin=0;
  c.torque_limits<<87,87,87,87,12,12,12;c.torque_rate_limit=1000;
  c.osc_translation_stiffness.setConstant(400);c.osc_translation_damping.setConstant(40);
  c.osc_rotation_stiffness.setConstant(90);c.osc_rotation_damping.setConstant(8);
  c.posture_stiffness=0;c.posture_damping=0;
  c.friction_compensation_torque=f;c.friction_compensation_epsilon=epsilon;
  c.validate();return c;
}
static Sample sample() {
  Sample s;s.q<<0,-.4,0,-1.9,0,1.5,0;s.position<<.45,0,.4;
  s.mass=3*M7::Identity();s.dynamics_valid=true;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1;
  return s;
}
static Command command(const Sample& s, Mode mode=Mode::Joint) {
  Command cmd;cmd.epoch=1;cmd.sequence=1;cmd.expires_at=1;
  cmd.mode=mode;cmd.position=s.position;cmd.joints=s.q;
  cmd.has_orientation=mode==Mode::Cartesian;cmd.orientation=s.orientation;return cmd;
}
static void close(const V7& a, const V7& b) {assert((a-b).norm()<1e-11);}
static Output step(const Config& c, const Sample& s, const Command& cmd) {
  Controller ctrl(c,s,1,0);return ctrl.step(s,cmd,false,0,.01);
}
static void formula_and_off_default() {
  assert(Config().friction_compensation_torque==0);
  for(const auto setting : {std::pair<double,double>{0,.05},{.3,.05},{.6,.03},{1,.02}}) {
    const auto c=config(setting.first,setting.second);auto s=sample();auto cmd=command(s);
    // Includes zero, both signs, linear transition and saturated regions.
    V7 delta;delta<<0,.001,-.002,.003,-.004,.05,-.05;cmd.joints+=delta;
    // Opposite measured/commanded velocities must not determine compensation.
    s.dq=-delta;cmd.joint_velocity=-2*delta;
    const auto out=step(c,s,cmd);assert(out.phase==Phase::Active);
    V7 expected=V7::Zero();
    for(int i=0;i<7;i++) expected[i]=setting.first*std::tanh(delta[i]/.1/setting.second);
    close(out.friction_compensation_torque,expected);
    if(setting.first>0) close(out.planned_joint_velocity,delta/.1);
    else close(out.planned_joint_velocity,V7::Zero());
    auto off=c;off.friction_compensation_torque=0;
    close(out.torque-step(off,s,cmd).torque,expected);
    assert(out.friction_compensation_torque.cwiseAbs().maxCoeff()<=setting.first);
    auto zero=command(s);close(step(c,s,zero).friction_compensation_torque,V7::Zero());
  }
}
static void cartesian_target_error_and_orientation() {
  auto c=config();auto s=sample();auto cmd=command(s,Mode::Cartesian);
  cmd.position+=V3(.001,-.002,.003);
  // Nonidentity actual orientation, base-frame rotational target delta.
  s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.7,V3::UnitY()));
  cmd.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(-.004,V3::UnitZ()))*s.orientation;
  const auto out=step(c,s,cmd);assert(out.phase==Phase::Active);
  V7 expected=V7::Zero();
  expected.head<3>()=(cmd.position-s.position)*.2/(.04+.0004)/.1;
  expected[5]=-.004*.04/(.04+.0004)/.1;
  close(out.planned_joint_velocity,expected);
  V7 torque;
  for(int i=0;i<7;i++) torque[i]=c.friction_compensation_torque*std::tanh(expected[i]/c.friction_compensation_epsilon);
  close(out.friction_compensation_torque,torque);
  auto off=c;off.friction_compensation_torque=0;
  close(out.torque-step(off,s,cmd).torque,torque);
  // Linear mode has a shaped reference but compensation uses the accepted target.
  c.compliance_mode="operational_space";c.effort_limit_policy="configured";c.osc_feedback_mode="linear";
  const auto shaped=step(c,s,cmd);
  assert((shaped.reference_position-cmd.position).norm()>.001);
  close(shaped.planned_joint_velocity,expected);
  // Equivalent quaternion sign must not change the direction.
  cmd.orientation.coeffs()*=-1;
  close(step(c,s,cmd).planned_joint_velocity,expected);
  // Rank-deficient Cartesian Jacobian is finite; an unreachable task gives zero.
  s.jacobian.setZero();const auto singular=step(c,s,cmd);
  assert(singular.phase==Phase::Active && singular.torque.allFinite());
  close(singular.friction_compensation_torque,V7::Zero());
}
static void stale_targets_and_mode_switch() {
  const auto c=config();auto s=sample();auto cmd=command(s);cmd.joints[0]+=.002;
  Controller ctrl(c,s,1,0);const auto first=ctrl.step(s,cmd,false,0,.01);
  auto rejected=cmd;rejected.joints[0]-=.1;
  close(ctrl.step(s,rejected,false,.01,.01).friction_compensation_torque,first.friction_compensation_torque);
  rejected.sequence=2;rejected.epoch=2;
  close(ctrl.step(s,rejected,false,.02,.01).friction_compensation_torque,first.friction_compensation_torque);
  rejected.epoch=1;rejected.expires_at=.02;
  close(ctrl.step(s,rejected,false,.03,.01).friction_compensation_torque,first.friction_compensation_torque);
  // Switching to a zero-error Cartesian target must not retain joint compensation.
  auto cart=command(s,Mode::Cartesian);cart.sequence=2;
  close(ctrl.step(s,cart,false,.04,.01).friction_compensation_torque,V7::Zero());
  cmd.sequence=3;cmd.joints[0]=s.q[0]-.002;
  close(ctrl.step(s,cmd,false,.05,.01).friction_compensation_torque,-first.friction_compensation_torque);
  // Omitted orientation reseeds from measurement on Cartesian reentry.
  s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.3,V3::UnitX()));
  cart.sequence=4;cart.has_orientation=false;
  close(ctrl.step(s,cart,false,.06,.01).friction_compensation_torque,V7::Zero());
}
static void rejected_commands_do_not_extend_lease_or_restart() {
  for(const Mode mode : {Mode::Cartesian,Mode::Joint}) {
    const auto c=config();auto s=sample();auto cmd=command(s,mode);
    cmd.position[0]+=.003;cmd.joints[0]+=.003;cmd.expires_at=.02;
    Controller ctrl(c,s,1,0);const auto initial=ctrl.step(s,cmd,false,0,.001);
    assert(initial.friction_compensation_torque[0]>0);
    // Duplicate sequence with a new target and expiry must not refresh anything.
    cmd.position[0]-=.01;cmd.joints[0]-=.01;cmd.expires_at=10;
    close(ctrl.step(s,cmd,false,.01,.001).friction_compensation_torque,initial.friction_compensation_torque);
    auto out=ctrl.step(s,cmd,false,.02,.001);
    assert(out.phase==Phase::Stopping && out.stop_trigger==Reason::Watchdog);
    close(out.friction_compensation_torque,V7::Zero());
    // Even fresh, unexpired commands cannot resurrect this stopped controller.
    for(int tick=21;tick<=150;tick++) {
      ++cmd.sequence;out=ctrl.step(s,cmd,false,tick*.001,.001);
      close(out.planned_joint_velocity,V7::Zero());close(out.friction_compensation_torque,V7::Zero());
      assert(out.phase!=Phase::Active);
    }
    assert(out.finished && out.phase==Phase::Disarmed);
  }
}
static void lifecycle_and_guards() {
  const auto c=config();auto s=sample();auto cmd=command(s);cmd.joints[0]+=.01;
  Controller armed(c,s,1,0);
  close(armed.step(s,Command{},false,0,.001).friction_compensation_torque,V7::Zero());
  for(int cause=0;cause<8;cause++) {
    Controller ctrl(c,s,1,0);assert(ctrl.step(s,cmd,false,0,.001).friction_compensation_torque[0]>0);
    auto bad=s;double now=.01,dt=.001;
    if(cause==1) now=1; // Watchdog.
    if(cause==2) bad.external_wrench[0]=c.contact_force_stop;
    if(cause==3) bad.dq[0]=c.joint_speed_limit()+.01;
    if(cause==4) bad.q[0]=c.joint_upper[0];
    if(cause==5) dt=.02;
    if(cause==6) bad.mass(0,0)=-1; // Dynamics failure inside Cartesian effort calculation.
    if(cause==7) bad.q[0]=std::numeric_limits<double>::quiet_NaN();
    auto input=cmd;
    if(cause==6) {input=command(s,Mode::Cartesian);input.sequence=2;input.position[0]+=.01;}
    const auto out=ctrl.step(bad,input,cause==0,now,dt);
    assert(out.phase==Phase::Stopping || out.phase==Phase::Fault);
    close(out.planned_joint_velocity,V7::Zero());close(out.friction_compensation_torque,V7::Zero());
  }
  // Legacy contact-yielding taper must also taper feedforward, and suppress it
  // completely when yielding (no stale target can push through the contact).
  Config legacy;legacy.friction_compensation_torque=.6;auto lsample=sample();
  auto target=command(lsample);target.joints[0]+=.01;
  lsample.external_wrench[0]=.85*legacy.contact_force_soft;
  auto out=step(legacy,lsample,target);assert(!out.yielding);
  assert(std::abs(out.friction_compensation_torque[0]-.5*.6*std::tanh(.1/.05))<1e-12);
  lsample.external_wrench[0]=legacy.contact_force_soft;
  out=step(legacy,lsample,target);assert(out.yielding);
  close(out.friction_compensation_torque,V7::Zero());
}
static void compensation_is_outside_feedback_budgets() {
  auto c=config();c.joint_feedback_torque_limits.setConstant(.1);
  auto s=sample();auto cmd=command(s);cmd.joints.array()+=.01;
  auto off=c;off.friction_compensation_torque=0;
  const auto on_joint=step(c,s,cmd), off_joint=step(off,s,cmd);
  assert(on_joint.joint_feedback_torque.cwiseAbs().maxCoeff()==.1);
  close(on_joint.joint_feedback_torque,off_joint.joint_feedback_torque);
  close(on_joint.torque-off_joint.torque,on_joint.friction_compensation_torque);
  c.commanded_force_limit=.1;off=c;off.friction_compensation_torque=0;
  cmd=command(s,Mode::Cartesian);cmd.position[0]+=.01;
  const auto on_cart=step(c,s,cmd), off_cart=step(off,s,cmd);
  assert(on_cart.task_force_limited && on_cart.task_wrench.head<3>().norm()==.1);
  assert((on_cart.task_wrench-off_cart.task_wrench).norm()==0);
  close(on_cart.torque-off_cart.torque,on_cart.friction_compensation_torque);
}
static void final_limits_and_validation() {
  auto c=config(1,.02);auto s=sample();auto cmd=command(s);cmd.joints[0]+=.1;
  s.coriolis[0]=c.torque_limits[0]-.2;
  s.previous_torque[0]=c.torque_limits[0]-.1;
  const auto out=step(c,s,cmd);
  assert(out.friction_compensation_torque[0]>.99);
  assert(out.torque[0]==c.torque_limits[0] && out.joint_torque_limited);
  s.coriolis.setZero();s.previous_torque.setZero();
  Controller ctrl(c,s,1,0);const auto slew=ctrl.step(s,cmd,false,0,.001);
  assert(slew.torque[0]==1 && slew.joint_torque_limited);
  for(double f : {-1.,13.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    auto bad=c;bad.friction_compensation_torque=f;bool threw=false;
    try {bad.validate();} catch(const std::invalid_argument&) {threw=true;}assert(threw);
  }
  for(double epsilon : {0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    auto bad=c;bad.friction_compensation_epsilon=epsilon;bool threw=false;
    try {bad.validate();} catch(const std::invalid_argument&) {threw=true;}assert(threw);
  }
}
static void filter_step_response_and_streaming_updates() {
  auto c=config(1,.02);c.friction_compensation_filter_time_s=.035;
  auto s=sample();auto cmd=command(s);cmd.joints[0]+=.001;
  auto unfiltered=c;unfiltered.friction_compensation_filter_time_s=0;
  const V7 raw=step(unfiltered,s,cmd).friction_compensation_torque;
  Controller ctrl(c,s,1,0);double elapsed=0;
  for(int tick=1;tick<=350;tick++) {
    // Ordinary new targets must NOT restart the filter every publication.
    if(tick%20==0) ++cmd.sequence;
    elapsed+=.001;
    const auto out=ctrl.step(s,cmd,false,elapsed,.001);
    assert(out.phase==Phase::Active);
    close(out.friction_compensation_torque,raw*(1-std::exp(-elapsed/.035)));
    assert(out.friction_compensation_torque.cwiseAbs().maxCoeff()<=1);
    if(tick==105) assert(out.friction_compensation_torque[0]/raw[0]>.95);
    if(tick==162) assert(out.friction_compensation_torque[0]/raw[0]>.99);
    // The physical impedance feedback itself must not be filtered.
    assert(std::abs(out.joint_feedback_torque[0]-.1)<1e-12);
    s.previous_torque=out.torque;
  }
  // Reversal is smooth and converges to the new law, not the old direction.
  cmd.joints[0]=s.q[0]-.001;++cmd.sequence;
  const double previous=raw[0]*(1-std::exp(-elapsed/.035));
  const auto reversed=ctrl.step(s,cmd,false,elapsed+.001,.001);
  assert(std::abs(reversed.friction_compensation_torque[0]-
      (-raw[0]+(previous+raw[0])*std::exp(-.001/.035)))<1e-12);
  // Variable callback periods integrate the same elapsed response.
  Controller variable(c,s,1,0);cmd.joints[0]=s.q[0]+.001;++cmd.sequence;
  elapsed=0;
  for(double dt : {.001,.002,.007}) {
    elapsed+=dt;const auto out=variable.step(s,cmd,false,elapsed,dt);
    close(out.friction_compensation_torque,raw*(1-std::exp(-elapsed/.035)));
  }
  Controller first(c,s,1,0);
  close(first.step(s,cmd,false,0,0).friction_compensation_torque,raw*(1-std::exp(-.001/.035)));
}
static void filter_lifecycle_and_contact_taper() {
  auto c=config(1,.02);c.friction_compensation_filter_time_s=.035;
  auto s=sample();auto cmd=command(s);cmd.joints[0]+=.001;
  for(int cause=0;cause<8;cause++) {
    Controller ctrl(c,s,1,0);
    const auto on=ctrl.step(s,cmd,false,0,.01);assert(on.friction_compensation_torque[0]>0);
    auto bad=s;double now=.01,dt=.001;
    if(cause==1) now=1;
    if(cause==2) bad.external_wrench[0]=c.contact_force_stop;
    if(cause==3) bad.dq[0]=c.joint_speed_limit()+.01;
    if(cause==4) bad.q[0]=c.joint_upper[0];
    if(cause==5) dt=.02;
    if(cause==6) bad.dynamics_valid=false;
    if(cause==7) bad.q[0]=std::numeric_limits<double>::quiet_NaN();
    const auto stopped=ctrl.step(bad,cmd,cause==0,now,dt);
    assert(stopped.phase==Phase::Stopping || stopped.phase==Phase::Fault);
    close(stopped.friction_compensation_torque,V7::Zero());
    close(ctrl.step(s,cmd,false,now+.001,.001).friction_compensation_torque,V7::Zero());
  }
  Controller mixed(c,s,1,0);mixed.step(s,cmd,false,0,.01);
  auto cart=command(s,Mode::Cartesian);cart.sequence=2;
  close(mixed.step(s,cart,false,.01,.001).friction_compensation_torque,V7::Zero());
  cmd.sequence=3;auto fresh=step(c,s,cmd);
  close(mixed.step(s,cmd,false,.02,.01).friction_compensation_torque,fresh.friction_compensation_torque);
  // A taper must act immediately on already-filtered torque, without a delayed
  // tail pushing against contact. Yield/release starts from a zero filter state.
  Config legacy;legacy.friction_compensation_torque=.6;legacy.friction_compensation_filter_time_s=.035;
  Controller contact(legacy,s,1,0);cmd=command(s);cmd.joints[0]+=.001;
  const double raw=.6*std::tanh(.001/.1/.05);
  contact.step(s,cmd,false,0,.01);
  s.external_wrench[0]=.85*legacy.contact_force_soft;
  auto out=contact.step(s,cmd,false,.01,.01);
  assert(!out.yielding);
  assert(std::abs(out.friction_compensation_torque[0]-.5*raw*(1-std::exp(-.02/.035)))<1e-12);
  s.external_wrench[0]=legacy.contact_force_soft;
  out=contact.step(s,cmd,false,.02,.001);assert(out.yielding);
  close(out.friction_compensation_torque,V7::Zero());
  s.external_wrench.setZero();out=contact.step(s,cmd,false,.03,.001);
  assert(!out.yielding);
  assert(std::abs(out.friction_compensation_torque[0]-raw*(1-std::exp(-.001/.035)))<1e-12);
  // Filtering disabled compensation must not alter the original controller law.
  auto off=c;off.friction_compensation_torque=0;auto no_filter=off;no_filter.friction_compensation_filter_time_s=0;
  close(step(off,s,cmd).torque,step(no_filter,s,cmd).torque);
  for(double value : {-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    auto bad=c;bad.friction_compensation_filter_time_s=value;bool threw=false;
    try {bad.validate();} catch(const std::invalid_argument&) {threw=true;}assert(threw);
  }
}
int main() {
  Eigen::internal::set_is_malloc_allowed(false);
  formula_and_off_default();cartesian_target_error_and_orientation();stale_targets_and_mode_switch();
  rejected_commands_do_not_extend_lease_or_restart();lifecycle_and_guards();
  compensation_is_outside_feedback_budgets();final_limits_and_validation();
  filter_step_response_and_streaming_updates();filter_lifecycle_and_contact_taper();
  std::cout<<"Friction compensation formula, lifecycle, regularization and final-limit checks passed\n";
}
