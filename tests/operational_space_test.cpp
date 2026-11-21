#include "controller.hpp"
#include "operational_space.hpp"
#include "fr3_limits.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace fcc;

static Config config() {
  Config c("fr3");c.compliance_mode="operational_space";
  c.motion_limit_policy="franka";c.effort_limit_policy="franka";c.collision_policy="robot";
  c.joint_lower=fr3_limits::lower();c.joint_upper=fr3_limits::upper();c.joint_limit_margin=0;
  c.torque_limits << 87,87,87,87,12,12,12;c.torque_rate_limit=1000;
  c.measured_joint_velocity_limits=fr3_limits::speed();
  c.measured_translation_velocity_limit=3;c.measured_rotation_velocity_limit=2.5;
  c.translation_velocity=.1;c.translation_acceleration=.3;c.translation_jerk=2;
  c.posture_stiffness=2;c.posture_damping=2;c.validate();return c;
}
static Sample sample() {
  Sample s;s.q << 0,-.4,0,-1.9,0,1.5,0;s.position << .45,0,.4;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1;
  s.jacobian.col(6) << .04,-.01,.03,.1,-.08,.06;
  M7 factor=M7::Identity();
  for(int i=0;i<7;i++) {factor(i,i)=1+.1*i;if(i) factor(i,i-1)=.2;}
  s.mass=factor*factor.transpose();s.dynamics_valid=true;
  return s;
}
static Command command(const Sample& s) {
  Command cmd;cmd.epoch=1;cmd.sequence=1;cmd.expires_at=100;
  cmd.position=s.position;cmd.orientation=s.orientation;cmd.has_orientation=true;return cmd;
}
static void inverse_dynamics_and_dynamic_nullspace() {
  auto c=config();auto s=sample();
  V6 a;a << .3,-.2,.1,.2,-.15,.12;
  s.jacobian_dot_velocity << .03,-.07,.01,.04,.03,-.06;
  V7 post;post << 1,-2,3,-4,5,-6,7;
  for(double inertia : {.5,1.,3.}) {
    auto loaded=s;loaded.mass=inertia*s.mass;
    V6 wrench;V7 null;bool regularized;
    assert(operational_space_effort(loaded,c,a,post,wrench,null,regularized));
    assert(!regularized);
    const auto mass=loaded.mass.ldlt();
    assert((loaded.jacobian*mass.solve(null)).norm()<1e-10);
    const V6 actual=loaded.jacobian*mass.solve(loaded.jacobian.transpose()*wrench+null)+loaded.jacobian_dot_velocity;
    assert((actual-a).norm()<1e-10);
    V6 without_bias;V7 n;
    loaded.jacobian_dot_velocity.setZero();
    assert(operational_space_effort(loaded,c,a,post,without_bias,n,regularized));
    assert((without_bias-wrench).norm()>.01); // Jdot*dq compensation matters.
  }
}
static void invalid_dynamics_and_singularities() {
  auto c=config();auto s=sample();V6 wrench;V7 null;bool regularized;
  for(int invalid=0;invalid<5;invalid++) {
    auto bad=s;
    if(invalid==0) bad.dynamics_valid=false;
    if(invalid==1) bad.mass(0,0)=-1;
    if(invalid==2) bad.mass(0,1)+=1;
    if(invalid==3) bad.mass(0,0)=NAN;
    if(invalid==4) bad.jacobian_dot_velocity[0]=NAN;
    assert(!operational_space_effort(bad,c,V6::Zero(),V7::Zero(),wrench,null,regularized));
    bool rejected=false;
    try {Controller invalid_start(c,bad,1,0);} catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
    auto cmd=command(s);Controller control(c,s,1,0);control.step(s,cmd,false,0,.001);
    const auto out=control.step(bad,cmd,false,.001,.001);
    assert(out.phase==Phase::Fault || out.phase==Phase::Stopping);
    assert(out.reason==Reason::InvalidState);
  }
  for(double scale : {1.,.01,0.,.01,1.}) {
    auto deficient=s;deficient.jacobian=scale*s.jacobian;
    assert(operational_space_effort(deficient,c,V6::Ones(),V7::Ones(),wrench,null,regularized));
    assert(wrench.allFinite() && null.allFinite());
    if(scale==0) assert(regularized);
  }
}
static void policies_limits_and_reference_acceleration() {
  auto c=config();auto s=sample();auto cmd=command(s);Controller control(c,s,1,0);
  control.step(s,cmd,false,0,.001);
  s.position[1]+=.2;
  // No hidden task-wrench cap or duplicate software collision trip in this
  // opt-in profile. This does NOT emulate or disable a robot collision reflex.
  s.external_wrench.setConstant(1000);s.external_torque.setConstant(1000);
  auto out=control.step(s,cmd,false,.001,.001);
  assert(out.phase==Phase::Active && !out.yielding && !out.task_force_limited);
  assert(out.task_wrench.head<3>().norm()>8);
  assert(out.torque.cwiseAbs().maxCoeff()<=1.0+1e-12); // 1000 Nm/s, 1 ms
  for(int i=2;i<500;i++) {
    s.previous_torque=out.torque;out=control.step(s,cmd,false,i*.001,.001);
    assert((out.torque.cwiseAbs().array()<=c.torque_limits.array()+1e-12).all());
  }
  s.dq[0]=2.63;out=control.step(s,cmd,false,.5,.001);
  assert(out.stop_trigger==Reason::Overspeed);
  c.collision_policy="robot_configured";s=sample();cmd=command(s);Controller sdk_only(c,s,1,0);
  sdk_only.step(s,cmd,false,0,.001);
  s.external_wrench[0]=2*c.contact_force_stop;s.external_torque.setConstant(100);
  out=sdk_only.step(s,cmd,false,.001,.001);
  assert(out.phase==Phase::Active && out.contact_load_ratio>=2 && !out.yielding);
  c.collision_policy="configured";s=sample();cmd=command(s);Controller guarded(c,s,1,0);
  guarded.step(s,cmd,false,0,.001);s.external_wrench[0]=c.contact_force_stop;
  assert(guarded.step(s,cmd,false,.001,.001).stop_trigger==Reason::ForceLimit);
  c=config();s=sample();cmd=command(s);Controller moving(c,s,1,0);
  moving.step(s,cmd,false,0,.001);cmd.position[1]+=.2;++cmd.sequence;
  V6 previous=V6::Zero();double max_speed=0;
  for(int i=1;i<4000;i++) {
    out=moving.step(s,cmd,false,.001*i,.001);
    assert(out.phase==Phase::Active);
    assert(std::abs(out.reference_acceleration[1])<=c.translation_acceleration+1e-10);
    assert(std::abs(out.reference_acceleration[1]-previous[1])<=c.translation_jerk*.001+1e-9);
    max_speed=std::max(max_speed,std::abs(out.reference_twist[1]));previous=out.reference_acceleration;
    s.previous_torque=out.torque;
  }
  assert(max_speed>.999*c.translation_velocity); // full budget for pure Y
  assert((out.reference_position-cmd.position).norm()<1e-10);
  auto joint=cmd;joint.mode=Mode::Joint;++joint.sequence;
  assert(moving.step(s,joint,false,4,.001).stop_trigger==Reason::ReferenceError);
}
static void two_targets_with_mass_plant() {
  auto c=config();auto s=sample();s.mass=3*M7::Identity();s.jacobian.col(6).setZero();
  const auto home=s;auto cmd=command(s);Controller control(c,s,1,0);control.step(s,cmd,false,0,.001);
  double now=0,max_error=0;
  for(double side : {-1.,1.,-1.,1.}) {
    cmd.position=home.position+side*.2*V3::UnitY();++cmd.sequence;
    for(int i=0;i<6500;i++) {
      now+=.001;auto out=control.step(s,cmd,false,now,.001);
      assert(out.phase==Phase::Active && !out.yielding);
      const V7 ddq=s.mass.ldlt().solve(out.torque-s.coriolis-2*s.dq);
      s.dq+=.001*ddq;s.q+=.001*s.dq;
      s.position+=.001*(s.jacobian*s.dq).head<3>();s.previous_torque=out.torque;
      max_error=std::max(max_error,(out.reference_position-s.position).norm());
    }
    assert((s.position-cmd.position).norm()<.001 && s.dq.norm()<.01);
  }
  std::cout << "OSC constant-mass plant max reference error: " << max_error << " m\n";
}
static void physical_feedback_does_not_vanish_with_small_model_inertia() {
  auto c=config();auto augmented=c;
  augmented.osc_translation_stiffness.setConstant(200);
  augmented.osc_translation_damping.setConstant(30);
  augmented.osc_rotation_stiffness.setConstant(10);
  augmented.osc_rotation_damping.setConstant(2);augmented.validate();
  auto s=sample();s.mass=.003*M7::Identity();auto cmd=command(s);
  Controller original(c,s,1,0), physical(augmented,s,1,0);
  original.step(s,cmd,false,0,.001);physical.step(s,cmd,false,0,.001);
  s.position[0]+=.01;s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.1,V3::UnitX()));
  s.dq[0]=.05;s.dq[3]=-.02;
  const auto a=original.step(s,cmd,false,.001,.001), b=physical.step(s,cmd,false,.001,.001);
  V6 difference=V6::Zero();difference[0]=-2.3;difference[3]=-.96;
  assert((b.task_wrench-a.task_wrench-difference).norm()<1e-10);
  assert(b.phase==Phase::Active && !b.task_force_limited && !b.task_moment_limited);

  // Illustrative low-inertia wrist with dry friction and a mass mismatch.
  // Unlike the original viscous-only plant, this reproduces a stuck offset.
  for(bool enabled : {false,true}) {
    auto gains=enabled?augmented:c;
    s=sample();s.jacobian.col(6).setZero();s.mass=.003*M7::Identity();cmd=command(s);
    Controller control(gains,s,1,0);control.step(s,cmd,false,0,.001);
    const double origin=s.q[3];double angle=.1, speed=0;
    for(int i=1;i<=3000;i++) {
      s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(angle,V3::UnitX()));
      s.q[3]=origin+angle;s.dq[3]=speed;
      const auto out=control.step(s,cmd,false,i*.001,.001);
      assert(out.phase==Phase::Active);
      const double tau=out.torque[3];
      if(std::abs(speed)<1e-4 && std::abs(tau)<=.05) speed=0;
      else {
        const double friction=.05*(speed!=0?std::copysign(1.0,speed):std::copysign(1.0,tau));
        const double next=speed+.001*(tau-friction)/.02;
        speed=(speed*next<0 && std::abs(tau)<=.05)?0:next;
      }
      angle+=.001*speed;s.previous_torque=out.torque;
    }
    assert(enabled?std::abs(angle)<.01:std::abs(angle)>.09);
  }
}
int main() {
  Eigen::internal::set_is_malloc_allowed(false);
  physical_feedback_does_not_vanish_with_small_model_inertia();
  inverse_dynamics_and_dynamic_nullspace();invalid_dynamics_and_singularities();
  policies_limits_and_reference_acceleration();two_targets_with_mass_plant();
  std::cout << "OSC algebra, no-allocation, limits and tracking checks passed\n";
}
