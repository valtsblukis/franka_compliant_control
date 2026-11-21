// Fixed-state and simple contact-plant checks; never connects to a robot.
#include "controller.hpp"
#include "fr3_limits.hpp"
#include "hardware_stop.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
using namespace fcc;

static Config config() {
  Config c("fr3");c.compliance_mode="auto";c.effort_limit_policy="mode_bounded";
  c.osc_feedback_mode="saturated";c.motion_limit_policy="franka";c.collision_policy="robot_configured";
  c.joint_lower=fr3_limits::lower();c.joint_upper=fr3_limits::upper();c.joint_limit_margin=0;
  c.torque_limits<<87,87,87,87,12,12,12;c.torque_rate_limit=1000;
  c.osc_translation_stiffness.setConstant(600);c.osc_translation_damping.setConstant(48.99);
  c.osc_rotation_stiffness.setConstant(90);c.osc_rotation_damping.setConstant(8);
  c.commanded_force_limit=20;c.commanded_moment_limit=6;
  c.contact_force_soft=15;c.contact_force_stop=40;c.contact_moment_soft=3;c.contact_moment_stop=20;
  c.external_torque_limits<<16,16,16,16,8,8,8;
  c.friction_compensation_torque=1;c.friction_compensation_epsilon=.02;c.friction_compensation_filter_time_s=.035;
  c.validate();return c;
}
static Sample sample() {
  Sample s;s.q<<0,-.4,0,-1.9,0,1.5,0;s.position<<.45,0,.4;
  s.mass=3*M7::Identity();s.dynamics_valid=true;s.wrench_tcp_valid=true;s.relief_setup_valid=true;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1;
  return s;
}
static ContactReliefOptions options(const Sample& s) {
  ContactReliefOptions r;r.bias_joint_positions=s.q;
  r.workspace_lower=s.position-V3::Constant(.06);r.workspace_upper=s.position+V3::Constant(.06);
  r.bias_confirmed_free=r.workspace_confirmed_clear=true;return r;
}
static Command command(const Sample& s) {
  Command c;c.epoch=1;c.sequence=1;c.expires_at=10;c.action_timeout=10;
  c.relieve_contact=true;c.relief_options=options(s);return c;
}
static void close(const V3& a,const V3& b) {assert((a-b).norm()<1e-11);}
static void sign_reanchor_bias_and_release() {
  auto c=config();auto s=sample();auto cmd=command(s);
  cmd.position<<999,999,999; // Never used as the internal relief equilibrium.
  cmd.relief_options.bias_wrench_tcp_base<<1,-2,.5,.1,.2,-.3;
  s.external_wrench_tcp_base=cmd.relief_options.bias_wrench_tcp_base;
  s.external_wrench_tcp_base[0]+=5;s.external_wrench_tcp_base[5]+=1.5;
  s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.4,V3::UnitY()));
  Controller ctrl(c,s,1,0);auto out=ctrl.step(s,cmd,false,0,.001);
  assert(out.phase==Phase::Active && out.relief_mode && !out.relief_released);
  close(out.reference_position,s.position);assert(out.reference_orientation.angularDistance(s.orientation)<1e-12);
  assert(out.reference_twist.isZero(0) && out.friction_compensation_torque.isZero(0));
  out=ctrl.step(s,cmd,false,.001,.001);
  // Positive SDK wrench means robot pushes environment: retreat is NEGATIVE.
  assert(out.reference_twist[0]<0 && out.reference_twist[5]<0);
  assert(std::abs(out.reference_twist[0]+.01)<1e-12);
  assert(out.reference_twist.head<3>().norm()<=.02 && out.reference_twist.tail<3>().norm()<=.1);
  assert(out.reference_position[0]<s.position[0]);
  assert(out.friction_compensation_torque.isZero(0));
  s.position=out.reference_position;s.orientation=out.reference_orientation;s.previous_torque=out.torque;
  s.external_wrench_tcp_base=cmd.relief_options.bias_wrench_tcp_base;
  // Filter starts with real load, not zero; release cannot be declared instantly.
  out=ctrl.step(s,cmd,false,.002,.001);assert(!out.relief_released);
  for(int tick=3;tick<600 && !out.finished;tick++) {
    out=ctrl.step(s,cmd,false,tick*.001,.001);s.previous_torque=out.torque;
  }
  assert(out.finished && out.phase==Phase::Disarmed && out.relief_released);
  assert(out.reason==Reason::ReliefReleased && normal_stop_reason(out.reason));
  assert(out.reference_twist.isZero(0) && out.friction_compensation_torque.isZero(0));
}
static void stationary_free_hold_has_no_upward_drift() {
  auto c=config();auto s=sample();auto cmd=command(s);cmd.relief_options.upward_velocity=.005;
  cmd.relief_options.bias_wrench_tcp_base<<1,2,-1,.2,.3,-.4;
  s.external_wrench_tcp_base=cmd.relief_options.bias_wrench_tcp_base;
  Controller ctrl(c,s,1,0);auto out=ctrl.step(s,cmd,false,0,.001);
  assert(out.relief_settling && !out.relief_released);close(out.reference_position,s.position);
  out=ctrl.step(s,cmd,false,.199,.001);assert(!out.relief_released);
  out=ctrl.step(s,cmd,false,.201,.001);
  assert(out.relief_released && out.phase==Phase::Stopping);close(out.reference_position,s.position);
}
static void hard_limits_use_unbiased_evidence() {
  auto c=config();auto s=sample();auto cmd=command(s);
  // A loaded start above the ordinary soft force threshold is explicitly allowed.
  s.external_wrench_tcp_base[0]=20;validate_contact_relief_start(c,s,cmd.relief_options);
  for(int cause=0;cause<5;cause++) {
    auto bad=s;Reason expected=Reason::None;
    if(cause==0) {bad.external_wrench_tcp_base[0]=40;expected=Reason::ForceLimit;}
    if(cause==1) {bad.external_wrench_tcp_base[3]=20;expected=Reason::MomentLimit;}
    if(cause==2) {bad.external_torque[6]=8;expected=Reason::JointContact;}
    if(cause==3) {bad.wrench_tcp_valid=false;expected=Reason::ReliefInvalidWrench;}
    if(cause==4) {bad.jacobian.setZero();expected=Reason::Singularity;}
    bool rejected=false;
    try {validate_contact_relief_start(c,bad,cmd.relief_options);} catch(const std::runtime_error&) {rejected=true;}
    assert(rejected);
    Controller ctrl(c,s,1,0);ctrl.step(s,cmd,false,0,.001);
    auto out=ctrl.step(bad,cmd,false,.001,.001);
    assert(out.phase==Phase::Stopping && out.stop_trigger==expected);
    assert(out.friction_compensation_torque.isZero(0));
  }
}
static void bounds_growth_timeout_and_exclusivity() {
  auto c=config();auto s=sample();s.external_wrench_tcp_base[0]=5;auto cmd=command(s);
  for(int cause=0;cause<8;cause++) {
    Controller ctrl(c,s,1,0);ctrl.step(s,cmd,false,0,.001);
    auto changed=s;auto input=cmd;double now=.001;bool stop=false;Reason expected=Reason::None;
    if(cause==0) {changed.position[1]+=.051;expected=Reason::ReliefTravel;}
    if(cause==1) {changed.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.18,V3::UnitY()));expected=Reason::ReliefRotation;}
    if(cause==2) {changed.position[0]+=.061;expected=Reason::ReliefWorkspace;}
    if(cause==3) {changed.q[6]+=.201;expected=Reason::ReliefBiasMismatch;}
    if(cause==4) {now=10;expected=Reason::ReliefTimeout;}
    if(cause==5) {stop=true;expected=Reason::UserStop;}
    if(cause==6) {input.sequence=2;input.relieve_contact=false;input.position=s.position;expected=Reason::ReferenceError;}
    if(cause==7) {changed.dq[0]=.151;expected=Reason::Overspeed;}
    const auto out=ctrl.step(changed,input,stop,now,.001);
    assert(out.phase==Phase::Stopping && out.stop_trigger==expected && !out.relief_released);
  }
  // Stale/wrong-epoch input cannot change direction, restart or refresh expiry.
  Controller stale(c,s,1,0);stale.step(s,cmd,false,0,.001);
  auto old=cmd;old.epoch=2;old.sequence=2;old.expires_at=100;old.relief_options.bias_wrench_tcp_base[0]=9;
  auto out=stale.step(s,old,false,.001,.001);assert(out.reference_twist[0]<0 && out.sequence==1);
  assert(stale.step(s,old,false,10,.001).reason==Reason::ReliefTimeout);
  // Sustained increase over best filtered force aborts, despite staying below hard stop.
  Controller growing(c,s,1,0);growing.step(s,cmd,false,0,.001);
  auto higher=s;higher.external_wrench_tcp_base[0]=9;
  for(int tick=1;tick<250;tick++) {
    out=growing.step(higher,cmd,false,tick*.001,.001);
    higher.position=out.reference_position;higher.orientation=out.reference_orientation;
    if(out.phase==Phase::Stopping) break;
  }
  assert(out.stop_trigger==Reason::ReliefLoadGrowing);
  // A fixed blocked sample must not accumulate an arbitrarily distant target.
  Controller blocked(c,s,1,0);
  for(int tick=0;tick<1000;tick++) {
    out=blocked.step(s,cmd,false,tick*.001,.001);
    if(out.phase==Phase::Stopping) break;
  }
  assert(out.stop_trigger==Reason::ReliefReferenceError);
  assert((out.reference_position-s.position).norm()<=.005+1e-12);
}
static void release_freezes_bounded_reference_and_hold_lead_guard_remains() {
  auto c=config();auto s=sample();s.external_wrench_tcp_base[0]=4;auto cmd=command(s);
  Controller ctrl(c,s,1,0);ctrl.step(s,cmd,false,0,.001);
  V3 previous=s.position;double previous_travel=0;
  s.position[1]+=.002;s.external_wrench_tcp_base.setZero();
  Output out;
  for(int tick=1;tick<100;tick++) {
    out=ctrl.step(s,cmd,false,tick*.001,.001);
    assert((out.reference_position-previous).norm()<=.02*.001+1e-12);
    if(out.relief_settling) {
      close(out.reference_position,previous);
      assert(out.relief_reference_travel==previous_travel);
      break;
    }
    previous=out.reference_position;previous_travel=out.relief_reference_travel;
  }
  assert(out.phase==Phase::Active && out.relief_settling);
  assert((out.reference_position-s.position).norm()>.001); // No teleport to measurement.
  // Still below release threshold: a held reference may not silently stretch
  // beyond the lead bound before release completes.
  s.position[1]+=.006;
  out=ctrl.step(s,cmd,false,.100,.001);
  assert(out.phase==Phase::Stopping && out.stop_trigger==Reason::ReliefReferenceError);
}
static void simple_wall_contact_unloads() {
  auto c=config();auto s=sample();auto cmd=command(s);
  const double initial_x=s.position[0],wall=initial_x-.005;
  const V7 q0=s.q;cmd.relief_options.bias_wrench_tcp_base[0]=1.;
  s.external_wrench_tcp_base[0]=6.; // 1 N free bias + 5 N physical resisting contact.
  Controller ctrl(c,s,1,0);Output out;
  for(int tick=0;tick<4000;tick++) {
    const double contact=1000*std::max(0.,s.position[0]-wall);
    s.external_wrench_tcp_base=cmd.relief_options.bias_wrench_tcp_base;
    s.external_wrench_tcp_base[0]+=contact;
    s.external_torque=s.jacobian.transpose()*(V6()<<contact,0,0,0,0,0).finished();
    out=ctrl.step(s,cmd,false,tick*.001,.001);
    assert(out.friction_compensation_torque.isZero(0));
    if(out.finished) break;
    s.dq+=.001*(out.torque-s.external_torque-2*s.dq)/3;
    s.q+=.001*s.dq;s.position[0]=initial_x+.2*(s.q[0]-q0[0]);
    s.previous_torque=out.torque;
  }
  assert(out.finished && out.phase==Phase::Disarmed && out.relief_released);
  assert(s.position[0]<initial_x && 1000*std::max(0.,s.position[0]-wall)<3);
}
static void immutable_startup_deadline_guards_and_identity() {
  auto c=config();c.command_timeout=30;auto s=sample();auto cmd=command(s);
  cmd.expires_at=.21;cmd.action_timeout=.21;
  Controller expired(c,s,1,0,&cmd);
  auto out=expired.step(s,Command{},false,.3,.001); // No mailbox delivery; original deadline already expired.
  assert(out.relief_mode && out.relief_deadline==.21 && out.phase==Phase::Stopping);
  assert(out.reason==Reason::ReliefTimeout && out.stop_trigger==Reason::ReliefTimeout);
  cmd=command(s);Controller hard(c,s,1,0,&cmd);
  s.external_wrench_tcp_base[0]=40;
  out=hard.step(s,Command{},false,.001,.001);
  assert(out.relief_mode && out.phase==Phase::Stopping && out.stop_trigger==Reason::ForceLimit);
  s.external_wrench_tcp_base.setZero();out=hard.step(s,Command{},false,.102,.001);
  assert(out.phase==Phase::Fault && out.relief_mode && !out.relief_released);
  Controller corrupt(c,s,1,0,&cmd);s.q[0]=std::numeric_limits<double>::quiet_NaN();
  out=corrupt.step(s,Command{},false,.001,.001);
  assert(out.phase==Phase::Fault && out.relief_mode && !out.relief_released);
}
static void successful_release_can_finish_stopping_after_action_deadline() {
  auto c=config();auto s=sample();auto cmd=command(s);
  cmd.expires_at=.21;cmd.action_timeout=.21;
  Controller ctrl(c,s,1,0,&cmd);
  assert(ctrl.step(s,Command{},false,0,.001).phase==Phase::Active);
  auto out=ctrl.step(s,cmd,false,.201,.001);
  assert(out.phase==Phase::Stopping && out.relief_released && out.reason==Reason::ReliefReleased);
  out=ctrl.step(s,cmd,false,.21,.001);
  assert(out.phase==Phase::Stopping && out.relief_released);
  out=ctrl.step(s,cmd,false,.302,.001);
  assert(out.finished && out.phase==Phase::Disarmed && out.relief_released);
  assert(out.reason==Reason::ReliefReleased && out.relief_deadline==.21);
}
static void upward_drift_cannot_reverse_relief_and_stationarity_is_capped() {
  auto c=config();auto s=sample();auto cmd=command(s);
  cmd.relief_options.translation_admittance=.001;cmd.relief_options.upward_velocity=.005;
  s.external_wrench_tcp_base[2]=4;
  Controller up(c,s,1,0);up.step(s,cmd,false,0,.001);
  auto out=up.step(s,cmd,false,.001,.001);
  assert(std::abs(out.reference_twist[2]+.004)<1e-12);
  assert(out.reference_twist.head<3>().dot(s.external_wrench_tcp_base.head<3>())<0);
  c.stop_velocity=.2;s=sample();cmd=command(s);
  Controller moving(c,s,1,0);moving.step(s,cmd,false,0,.001);
  s.dq[6]=.1;out=moving.step(s,cmd,false,.3,.001);
  assert(out.phase==Phase::Active && !out.relief_released);
  s.dq[6]=0;moving.step(s,cmd,false,.4,.001);
  assert(!moving.step(s,cmd,false,.599,.001).relief_released);
  assert(moving.step(s,cmd,false,.601,.001).relief_released);
  HardwareStopMonitor monitor(0,c,.02);s.dq[6]=.1;
  assert(monitor.observe(0,s,true)==StopObservation::Pending);
  assert(monitor.observe(.2,s,true)==StopObservation::Pending);
  s.dq[6]=0;assert(monitor.observe(.3,s,true)==StopObservation::Pending);
  assert(monitor.observe(.401,s,true)==StopObservation::Confirmed);
}
static void post_handoff_observations_cannot_falsely_certify_release() {
  auto c=config();auto s=sample();auto r=options(s);
  assert(contact_relief_observation_reason(c,s,r,s.position,s.orientation,true)==Reason::None);
  for(int cause=0;cause<7;cause++) {
    auto bad=s;Reason expected=Reason::None;
    if(cause==0) {bad.wrench_tcp_valid=false;expected=Reason::ReliefInvalidWrench;}
    if(cause==1) {bad.relief_setup_valid=false;expected=Reason::ReliefSetupMismatch;}
    if(cause==2) {bad.jacobian.setZero();expected=Reason::Singularity;}
    if(cause==3) {bad.external_wrench_tcp_base[0]=40;expected=Reason::ForceLimit;}
    if(cause==4) {bad.position[0]+=.061;expected=Reason::ReliefWorkspace;}
    if(cause==5) {bad.position[0]+=.051;expected=Reason::ReliefTravel;}
    if(cause==6) {bad.external_wrench_tcp_base[0]=3.1;expected=Reason::ReliefReleaseLost;}
    assert(contact_relief_observation_reason(c,bad,r,s.position,s.orientation,true)==expected);
  }
}
static void invalid_model_on_release_and_lost_release_during_stop() {
  auto c=config();auto s=sample();auto cmd=command(s);
  for(int kind=0;kind<3;kind++) {
    Controller ctrl(c,s,1,0,&cmd);ctrl.step(s,Command{},false,0,.001);
    auto bad=s;
    if(kind==0) bad.mass=-M7::Identity();
    if(kind==1) bad.mass(0,0)=0;
    if(kind==2) bad.mass(0,1)=.2;
    const auto out=ctrl.step(bad,cmd,false,.201,.001);
    assert(out.phase==Phase::Stopping && out.reason==Reason::InvalidState && !out.relief_released);
    assert(contact_relief_observation_reason(c,bad,cmd.relief_options,s.position,s.orientation,true)==Reason::InvalidState);
  }
  for(int kind=0;kind<3;kind++) {
    Controller ctrl(c,s,1,0,&cmd);ctrl.step(s,Command{},false,0,.001);
    auto out=ctrl.step(s,cmd,false,.201,.001);
    assert(out.phase==Phase::Stopping && out.relief_released);
    auto late=s;Reason expected;
    if(kind==0) {late.external_wrench_tcp_base[0]=3.1;expected=Reason::ReliefReleaseLost;}
    else if(kind==1) {late.position[0]+=.061;expected=Reason::ReliefWorkspace;}
    else {late.position[0]+=.051;expected=Reason::ReliefTravel;}
    out=ctrl.step(late,cmd,false,.302,.001);
    assert(out.finished && out.phase==Phase::Fault && out.reason==expected && !out.relief_released);
    assert(out.stop_trigger==Reason::ReliefReleased);
    assert(std::abs(out.relief_translation-(late.position-s.position).norm())<1e-12);
  }
}
static void manual_follow_needs_no_bias_and_never_claims_release() {
  auto c=config();auto s=sample();auto cmd=command(s);
  cmd.relief_options=ContactReliefOptions{};
  auto& r=cmd.relief_options;r.manual_stop=true;r.workspace_confirmed_clear=true;
  r.translation_speed=.005;r.rotation_speed=std::acos(-1.)/180.;
  r.validate(c);validate_contact_relief_start(c,s,r); // No bias or absolute box.
  Controller quiet(c,s,1,0,&cmd);auto out=quiet.step(s,Command{},false,0,.001);
  out=quiet.step(s,cmd,false,.3,.001);
  assert(out.phase==Phase::Active && out.relief_manual && !out.relief_released && !out.relief_settling);
  close(out.reference_position,s.position);
  quiet.step(s,cmd,true,.301,.001);out=quiet.step(s,cmd,false,.402,.001);
  assert(out.finished && out.phase==Phase::Disarmed && out.reason==Reason::UserStop && !out.relief_released);

  // Follow even a below-release-threshold raw signal; it is not relabeled bias.
  s.external_wrench_tcp_base[0]=1.;Controller loaded(c,s,1,0,&cmd);
  loaded.step(s,Command{},false,0,.001);out=loaded.step(s,cmd,false,.001,.001);
  assert(out.reference_twist[0]<0 && std::abs(out.reference_twist[0]+.002)<1e-12);
  assert(out.friction_compensation_torque.isZero(0));
  assert(out.relief_filtered_wrench[0]==1. && !out.relief_released);
  auto moved=s;moved.q[6]+=.201;
  out=loaded.step(moved,cmd,false,.002,.001);
  assert(out.stop_trigger==Reason::ReliefJointTravel);

  assert(contact_relief_observation_reason(c,s,r,s.position,s.orientation,false,&s.q)==Reason::None);
  assert(contact_relief_observation_reason(c,s,r,s.position,s.orientation,true,&s.q)==Reason::ReliefReleaseLost);
  assert(contact_relief_observation_reason(c,s,r,s.position,s.orientation,false)==Reason::InvalidState);
  // Supplied bias cannot be silently ignored in manual mode.
  auto bad=r;bad.bias_wrench_tcp_base[0]=1.;bool rejected=false;
  try {bad.validate(c);} catch(const std::invalid_argument&) {rejected=true;}assert(rejected);
  // An explicit box is still honored; local-only mode is not a joint-limit bypass.
  auto box=r;box.workspace_lower=s.position+V3::Constant(.01);box.workspace_upper=s.position+V3::Constant(.02);
  rejected=false;try {validate_contact_relief_start(c,s,box);} catch(const std::runtime_error&) {rejected=true;}assert(rejected);
  auto outside=s;outside.q[6]=c.joint_upper[6]+.001;
  rejected=false;try {validate_contact_relief_start(c,outside,r);} catch(const std::runtime_error&) {rejected=true;}assert(rejected);
}
static void invalid_options_reject() {
  auto c=config();auto s=sample();auto r=options(s);r.validate(c);
  for(int cause=0;cause<9;cause++) {
    auto bad=r;
    if(cause==0) bad.bias_confirmed_free=false;
    if(cause==1) bad.workspace_confirmed_clear=false;
    if(cause==2) bad.workspace_upper=bad.workspace_lower;
    if(cause==3) bad.translation_speed=.021;
    if(cause==4) bad.max_translation=.051;
    if(cause==5) bad.max_rotation=.18;
    if(cause==6) bad.bias_wrench_tcp_base[0]=std::numeric_limits<double>::quiet_NaN();
    if(cause==7) bad.upward_velocity=-.001;
    if(cause==8) bad.bias_joint_positions.setZero();
    bool rejected=false;try {bad.validate(c);} catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
  }
}
int main() {
  Eigen::internal::set_is_malloc_allowed(false);
  sign_reanchor_bias_and_release();stationary_free_hold_has_no_upward_drift();hard_limits_use_unbiased_evidence();
  bounds_growth_timeout_and_exclusivity();release_freezes_bounded_reference_and_hold_lead_guard_remains();
  simple_wall_contact_unloads();immutable_startup_deadline_guards_and_identity();
  successful_release_can_finish_stopping_after_action_deadline();
  upward_drift_cannot_reverse_relief_and_stationarity_is_capped();post_handoff_observations_cannot_falsely_certify_release();
  invalid_model_on_release_and_lost_release_during_stop();manual_follow_needs_no_bias_and_never_claims_release();
  invalid_options_reject();
  std::cout<<"Bounded native contact-relief sign, guards, lifecycle and wall-unloading checks passed\n";
}
