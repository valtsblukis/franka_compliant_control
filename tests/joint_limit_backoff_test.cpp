// All joints/sides, fresh native planning, fixed-state guards and shaped references.
// No robot connection or hardware recovery.
#include "controller.hpp"
#include "joint_limit_backoff.hpp"
#include "hardware_stop.hpp"
#include <cassert>
#include <iostream>
using namespace fcc;

static Config config() {
  Config c("fr3");c.compliance_mode="auto";c.effort_limit_policy="mode_bounded";
  c.osc_feedback_mode="saturated";c.motion_limit_policy="franka";c.collision_policy="robot_configured";
  c.joint_lower=fr3_limits::lower();c.joint_upper=fr3_limits::upper();c.joint_limit_margin=0;
  c.torque_limits<<87,87,87,87,12,12,12;c.torque_rate_limit=1000;
  c.osc_translation_stiffness.setConstant(600);c.osc_translation_damping.setConstant(48.99);
  c.osc_rotation_stiffness.setConstant(90);c.osc_rotation_damping.setConstant(8);
  c.contact_force_soft=15;c.contact_force_stop=40;c.contact_moment_soft=3;c.contact_moment_stop=20;
  c.friction_compensation_torque=1;c.friction_compensation_epsilon=.02;c.friction_compensation_filter_time_s=.035;
  c.validate();return c;
}
static Sample sample(const Config& c) {
  Sample s;s.q=.5*(c.joint_lower+c.joint_upper);s.position<<.45,0,.4;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1.;
  s.mass=3*M7::Identity();s.dynamics_valid=true;s.kinematics_valid=true;
  s.wrench_tcp_valid=s.relief_setup_valid=true;return s;
}
static JointLimitBackoffOptions options() {JointLimitBackoffOptions r;r.confirm_clear_path=true;return r;}
static Command command(const Config& c,const Sample& s) {
  Command cmd;cmd.mode=Mode::Joint;cmd.epoch=cmd.sequence=1;cmd.expires_at=20;cmd.action_timeout=20;
  cmd.backoff_joint_limits=true;cmd.backoff_options=options();
  cmd.backoff_plan=plan_joint_limit_backoff(c,s,cmd.backoff_options);return cmd;
}
static void all_joints_both_sides_and_multiple_selection() {
  const auto c=config();const auto r=options();
  for(int joint=0;joint<7;joint++) for(double direction:{-1.,1.}) {
    auto s=sample(c);s.q[joint]=direction>0?c.joint_lower[joint]+.01:c.joint_upper[joint]-.01;
    const auto p=plan_joint_limit_backoff(c,s,r);
    assert(p.prepared && p.direction[joint]==direction);
    assert(std::abs(p.goal[joint]-s.q[joint]-direction*.1)<1e-12);
    for(int i=0;i<7;i++) if(i!=joint) assert(p.direction[i]==0 && p.goal[i]==s.q[i]);
  }
  auto s=sample(c);s.q[0]=c.joint_lower[0]+.01;s.q[3]=c.joint_upper[3]-.02;s.q[6]=c.joint_upper[6]-.03;
  auto p=plan_joint_limit_backoff(c,s,r);assert(p.direction[0]==1 && p.direction[3]==-1 && p.direction[6]==-1);
  assert(p.direction[1]==0 && p.direction[2]==0 && p.direction[4]==0 && p.direction[5]==0);
}
static void scoped_configured_limit_exception_not_physical_bypass() {
  auto c=config();c.joint_upper[6]-=.2;c.validate();auto s=sample(c);s.q[6]=c.joint_upper[6]+.03;
  const auto cmd=command(c,s);assert(cmd.backoff_plan.goal[6]==c.joint_upper[6]-.1);
  Controller ordinary(c,s,1,0);
  assert(ordinary.step(s,Command{},false,0,.001).reason==Reason::JointLimit);
  Controller backoff(c,s,1,0,&cmd);const auto out=backoff.step(s,Command{},false,0,.001);
  assert(out.phase==Phase::Active && out.backoff_mode && out.mode==Mode::Joint);
  assert(out.reference_joints==s.q && out.friction_compensation_torque.isZero(0));
  auto too_far=s;too_far.q[6]=c.joint_upper[6]+.051;
  bool rejected=false;try {plan_joint_limit_backoff(c,too_far,options());} catch(const std::runtime_error&) {rejected=true;}assert(rejected);
  too_far=s;too_far.q[6]=fr3_limits::upper()[6]+.001;
  rejected=false;try {plan_joint_limit_backoff(c,too_far,options());} catch(const std::runtime_error&) {rejected=true;}assert(rejected);
  // At the specified bound, no exception is available either.
  too_far.q[6]=fr3_limits::upper()[6];
  rejected=false;try {plan_joint_limit_backoff(c,too_far,options());} catch(const std::runtime_error&) {rejected=true;}assert(rejected);
}
static void selected_outward_noise_tolerance_does_not_allow_worsening() {
  auto c=config();auto s=sample(c);s.q[6]=c.joint_upper[6]-.001;s.dq[6]=.001;
  assert(fr3_limits::directional_speed(s.q,true)[6]==0);
  auto cmd=command(c,s);Controller ctrl(c,s,1,0,&cmd);
  auto out=ctrl.step(s,Command{},false,0,.001);assert(out.phase==Phase::Active);
  s.q[6]+=.0005;s.dq[6]=.011;
  out=ctrl.step(s,cmd,false,.001,.001);assert(out.reason==Reason::BackoffOutward);
}
static void reference_moves_only_selected_inward_then_disarms() {
  auto c=config();auto s=sample(c);s.q[1]=c.joint_lower[1]+.02;s.q[6]=c.joint_upper[6]-.02;
  auto cmd=command(c,s);Controller ctrl(c,s,1,0,&cmd);
  const auto start=s.q;auto previous=start;Output out;
  for(int tick=0;tick<19000;tick++) {
    out=ctrl.step(s,cmd,false,tick*.001,.001);
    assert(out.friction_compensation_torque.isZero(0));
    if(out.phase==Phase::Active) {
      assert(out.reference_joint_velocity.cwiseAbs().maxCoeff()<=.03+1e-9);
      for(int i=0;i<7;i++) {
        const double d=cmd.backoff_plan.direction[i];
        if(d==0) assert(out.reference_joints[i]==start[i]);
        else {
          assert(d*(out.reference_joints[i]-previous[i])>=-1e-10);
          assert(std::abs(out.torque[i])<=out.backoff_torque_limits[i]+1e-12);
        }
      }
      previous=out.reference_joints;s.q=out.reference_joints;s.dq=out.reference_joint_velocity;
    } else s.dq.setZero();
    s.previous_torque=out.torque;
    if(out.finished) break;
  }
  assert(out.finished && out.phase==Phase::Disarmed && out.backoff_reached && out.reason==Reason::BackoffReached);
  assert(normal_stop_reason(out.reason));
}
static void guards_lifecycle_and_missing_motion() {
  auto c=config();auto s=sample(c);s.q[6]=c.joint_upper[6]-.02;auto cmd=command(c,s);
  for(int cause=0;cause<7;cause++) {
    Controller ctrl(c,s,1,0,&cmd);ctrl.step(s,Command{},false,0,.001);
    auto bad=s;auto input=cmd;double now=.001;bool cancel=false;Reason expected=Reason::None;
    if(cause==0) {bad.q[6]+=.0011;expected=Reason::BackoffOutward;}
    if(cause==1) {bad.q[0]+=.0101;expected=Reason::BackoffHoldDrift;}
    if(cause==2) {bad.dq[6]=-.081;expected=Reason::Overspeed;}
    if(cause==3) {bad.external_wrench_tcp_base[0]=40;expected=Reason::ForceLimit;}
    if(cause==4) {now=20;expected=Reason::BackoffTimeout;}
    if(cause==5) {cancel=true;expected=Reason::UserStop;}
    if(cause==6) {input.backoff_joint_limits=false;input.sequence=2;expected=Reason::ReferenceError;}
    const auto out=ctrl.step(bad,input,cancel,now,.001);
    assert(out.phase==Phase::Stopping && out.stop_trigger==expected && !out.backoff_reached);
  }
  Controller blocked(c,s,1,0,&cmd);Output out;
  for(int tick=0;tick<3000;tick++) {
    out=blocked.step(s,cmd,false,tick*.001,.001);s.previous_torque=out.torque;
    if(out.phase!=Phase::Active) break;
  }
  assert(out.reason==Reason::BackoffStalled);
  cmd.action_timeout=.21;cmd.expires_at=.21;c.command_timeout=30;
  Controller expired(c,s,1,0,&cmd);out=expired.step(s,Command{},false,.3,.001);
  assert(out.backoff_mode && out.reason==Reason::BackoffTimeout && out.backoff_deadline==.21);
}
static void reject_noop_invalid_options_and_unsafe_start() {
  const auto c=config();auto s=sample(c);auto r=options();
  bool rejected=false;try {plan_joint_limit_backoff(c,s,r);} catch(const std::runtime_error&) {rejected=true;}assert(rejected);
  s.q[6]=c.joint_upper[6]-.02;
  for(int kind=0;kind<4;kind++) {
    auto bad=s;
    if(kind==0) bad.relief_setup_valid=false;
    if(kind==1) bad.robot_has_errors=true;
    if(kind==2) bad.external_wrench[0]=c.contact_force_soft;
    if(kind==3) bad.previous_torque[6]=1.1;
    rejected=false;try {plan_joint_limit_backoff(c,bad,r);} catch(const std::runtime_error&) {rejected=true;}assert(rejected);
  }
  r.confirm_clear_path=false;rejected=false;
  try {r.validate(c);} catch(const std::invalid_argument&) {rejected=true;}assert(rejected);
  assert(backoff_local_fault_allowed(Reason::JointLimit));assert(backoff_local_fault_allowed(Reason::Overspeed));
  assert(!backoff_local_fault_allowed(Reason::RobotJointReflex));assert(!backoff_local_fault_allowed(Reason::InvalidState));
}
static void secondary_fault_cannot_remain_backoff_eligible() {
  auto c=config();c.stop_settle_time=1.;auto s=sample(c);s.q[6]=c.joint_upper[6]-.02;
  auto cmd=command(c,s);cmd.action_timeout=.21;cmd.expires_at=.21;
  Controller ctrl(c,s,1,0,&cmd);ctrl.step(s,cmd,false,0,.001);
  auto out=ctrl.step(s,cmd,false,.22,.001);assert(out.reason==Reason::BackoffTimeout);
  s.q[0]+=.0101;out=ctrl.step(s,cmd,false,.23,.001);
  assert(out.reason==Reason::BackoffHoldDrift && out.stop_trigger==Reason::BackoffTimeout);
  assert(!backoff_local_fault_allowed(out.reason));
  s=sample(c);s.q[6]=c.joint_upper[6]-.02;cmd=command(c,s);
  Controller other(c,s,1,0,&cmd);other.step(s,cmd,false,0,.001);
  s.dq[6]=-.081;out=other.step(s,cmd,false,.01,.001);assert(out.reason==Reason::Overspeed);
  s.dq.setZero();s.external_wrench_tcp_base[0]=40;out=other.step(s,cmd,false,.02,.001);
  assert(out.reason==Reason::ForceLimit && out.stop_trigger==Reason::Overspeed);
  assert(!backoff_local_fault_allowed(out.reason));
}
static void early_excursion_does_not_defeat_stall() {
  auto c=config();auto s=sample(c);s.q[6]=c.joint_upper[6]-.02;auto cmd=command(c,s);
  Controller ctrl(c,s,1,0,&cmd);Output out;
  for(int tick=0;tick<3000;tick++) {
    if(tick>0 && tick<=300) {s.q[6]-=.00001;s.dq[6]=-.01;}
    else if(tick>300 && tick<=600) {s.q[6]+=.00001;s.dq[6]=.01;}
    else s.dq.setZero();
    out=ctrl.step(s,cmd,false,tick*.001,.001);s.previous_torque=out.torque;
    if(out.phase!=Phase::Active) break;
  }
  assert(out.backoff_best_progress[6]>.0029);
  assert(out.reason==Reason::BackoffStalled);
}
static void high_water_guard_continues_during_stop() {
  auto c=config();c.stop_settle_time=1.;auto s=sample(c);s.q[6]=c.joint_upper[6]-.02;
  const auto cmd=command(c,s);Controller ctrl(c,s,1,0,&cmd);Output out;int tick=0;
  for(;tick<18000;tick++) {
    out=ctrl.step(s,cmd,false,tick*.001,.001);s.previous_torque=out.torque;
    if(out.phase==Phase::Stopping) break;
    s.q=out.reference_joints;s.dq=out.reference_joint_velocity;
  }
  assert(out.backoff_reached && out.reason==Reason::BackoffReached);
  const double prior_best=out.backoff_best_progress[6];
  for(int i=0;i<100;i++) {
    s.q[6]-=.00007;s.dq[6]=-.07;
    out=ctrl.step(s,cmd,false,(++tick)*.001,.001);s.previous_torque=out.torque;
    assert(out.phase==Phase::Stopping && out.backoff_reached);
  }
  assert(out.backoff_best_progress[6]>prior_best+.0069);
  for(int i=0;i<700;i++) {
    s.q[6]+=.000009;s.dq[6]=.009;
    out=ctrl.step(s,cmd,false,(++tick)*.001,.001);s.previous_torque=out.torque;
    if(out.reason==Reason::BackoffOutward) break;
  }
  assert(out.reason==Reason::BackoffOutward && !out.backoff_reached);
  assert(out.stop_trigger==Reason::BackoffReached);
}
static void mixed_guard_evidence_never_hides_disqualification() {
  auto c=config();auto initial=sample(c);initial.q[6]=c.joint_upper[6]-.02;
  auto cmd=command(c,initial);const auto& p=cmd.backoff_plan;const auto r=options();const V7 best=V7::Zero();
  for(double outward:{.011,.081}) {
    Controller ctrl(c,initial,1,0,&cmd);ctrl.step(initial,cmd,false,0,.001);
    auto s=initial;s.dq[6]=outward;
    auto out=ctrl.step(s,cmd,false,.001,.001);
    assert(out.reason==Reason::BackoffOutward && !backoff_local_fault_allowed(out.reason));
    out=ctrl.step(initial,cmd,false,.002,.001);
    assert(out.reason==Reason::BackoffOutward); // A later clean sample cannot erase it.
  }
  auto s=initial;s.dq[6]=-.081;
  assert(backoff_observation_reason(c,s,r,p,best)==Reason::Overspeed);
  s=initial;s.dq[0]=.081;s.q[5]+=.0101;
  assert(backoff_observation_reason(c,s,r,p,best)==Reason::BackoffHoldDrift);
  s=initial;s.q[6]=fr3_limits::upper()[6]+.001;s.external_wrench_tcp_base[0]=40;
  assert(backoff_observation_reason(c,s,r,p,best)==Reason::ForceLimit);
  s.external_wrench_tcp_base.setZero();s.external_torque[0]=c.external_torque_limits[0];
  assert(backoff_observation_reason(c,s,r,p,best)==Reason::JointContact);
  s=initial;s.dq[6]=-.081;s.previous_torque[6]=1.01;
  assert(backoff_observation_reason(c,s,r,p,best,false,true)==Reason::InvalidState);
  s=initial;s.dq[6]=-.081;s.external_wrench[0]=c.contact_force_soft;
  assert(backoff_start_observation_reason(c,s,r,p,best)==Reason::ForceLimit);
  s=initial;s.dq[6]=-.081;s.q[6]-=.002;
  assert(backoff_start_observation_reason(c,s,r,p,best)==Reason::BackoffStartChanged);
}
static void stop_read_guards_latch_ordinary_loads_and_invalid_samples() {
  auto c=config();c.collision_policy="configured";auto s=sample(c);Output out;
  out.phase=Phase::Stopping;out.reason=out.stop_trigger=Reason::JointLimit;
  s.external_wrench[0]=c.contact_force_stop;
  assert(observe_stop_guards(c,s,out,true)==Reason::ForceLimit);
  assert(out.reason==Reason::ForceLimit && out.stop_trigger==Reason::JointLimit);
  s.external_wrench.setZero();assert(observe_stop_guards(c,s,out,true)==Reason::None);
  assert(!backoff_local_fault_allowed(out.reason));
  out.reason=out.stop_trigger=Reason::BackoffReached;out.backoff_reached=true;s.robot_has_errors=true;
  assert(observe_stop_guards(c,s,out,false)==Reason::InvalidState);
  s.robot_has_errors=false;assert(observe_stop_guards(c,s,out,false)==Reason::None);
  assert(out.reason==Reason::InvalidState && !out.backoff_reached);
  // Preserve ordinary policy: inherited robot-only guards are not silently replaced.
  c.collision_policy="robot_configured";s.external_wrench[0]=40;
  assert(ordinary_hard_contact_reason(c,s)==Reason::None);
}
static void final_goal_loss_is_independent_of_backtrack_limit() {
  const auto c=config();auto s=sample(c);s.q[6]=c.joint_upper[6]-.02;
  const auto p=plan_joint_limit_backoff(c,s,options());V7 best=V7::Zero();
  s.q[6]=p.goal[6]+.019;update_backoff_progress(s,p,best);
  assert(backoff_observation_reason(c,s,options(),p,best,true)==Reason::None);
  s.q[6]+=.003;
  assert(backoff_observation_reason(c,s,options(),p,best,false)==Reason::None);
  assert(backoff_observation_reason(c,s,options(),p,best,true)==Reason::BackoffGoalLost);
}
static void expired_pre_control_gate_does_not_call_sdk_stub() {
  auto c=config();auto s=sample(c);s.q[6]=c.joint_upper[6]-.02;auto cmd=command(c,s);
  cmd.expires_at=.21;int sdk_control_calls=0;
  const auto enter=[&](double worker_time,bool cancelled) {
    const auto reason=pre_control_stop_reason(cmd,worker_time,cancelled);
    if(reason==Reason::None) ++sdk_control_calls;
    return reason;
  };
  assert(enter(.3,false)==Reason::BackoffTimeout && sdk_control_calls==0);
  assert(enter(.3,true)==Reason::UserStop && sdk_control_calls==0);
  assert(enter(.1,false)==Reason::None && sdk_control_calls==1);
  cmd.backoff_joint_limits=false;cmd.relieve_contact=true;
  assert(enter(.3,false)==Reason::ReliefTimeout && sdk_control_calls==1);
}
int main() {
  Eigen::internal::set_is_malloc_allowed(false);
  all_joints_both_sides_and_multiple_selection();scoped_configured_limit_exception_not_physical_bypass();
  selected_outward_noise_tolerance_does_not_allow_worsening();reference_moves_only_selected_inward_then_disarms();
  guards_lifecycle_and_missing_motion();reject_noop_invalid_options_and_unsafe_start();
  secondary_fault_cannot_remain_backoff_eligible();early_excursion_does_not_defeat_stall();
  high_water_guard_continues_during_stop();expired_pre_control_gate_does_not_call_sdk_stub();
  mixed_guard_evidence_never_hides_disqualification();stop_read_guards_latch_ordinary_loads_and_invalid_samples();
  final_goal_loss_is_independent_of_backtrack_limit();
  std::cout<<"General joint-limit backoff planning, directional references, scoped guards and stop lifecycle passed\n";
}
