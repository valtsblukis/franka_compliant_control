#include "controller.hpp"
#include "fr3_limits.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace fcc;

static Config config() {
  Config c("fr3");c.compliance_mode="continuous_impedance";
  c.contact_force_soft=15;c.contact_force_stop=25;
  c.contact_moment_soft=3;c.contact_moment_stop=10;
  c.commanded_force_limit=8;c.commanded_moment_limit=1.5;
  c.position_error_limit=.08;c.rotation_error_limit=.3;
  c.translation_velocity=.2;c.translation_acceleration=1;c.translation_jerk=10;
  c.measured_translation_velocity_limit=.5;c.measured_rotation_velocity_limit=1.5;
  c.measured_joint_velocity_limits << 1.5,.9,1.2,1.1,2,1.3,2;
  c.validate();return c;
}
static Sample sample() {
  Sample s;s.q << 0,-.4,0,-1.9,0,1.5,0;s.position << .45,0,.4;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1;
  return s;
}
static Command hold(const Sample& s) {
  Command cmd;cmd.epoch=1;cmd.sequence=1;cmd.expires_at=10;
  cmd.position=s.position;cmd.orientation=s.orientation;cmd.has_orientation=true;
  cmd.joints=s.q;return cmd;
}
static void unchanged_spring_across_contact_thresholds() {
  const Config c=config();
  // Small errors test the linear spring/damper; large ones exceed the old
  // reference leash and test bounded effort with an unchanged equilibrium.
  for(bool large : {false,true}) {
    auto s=sample();const auto cmd=hold(s);Controller control(c,s,1,0);
    control.step(s,cmd,false,0,.001);
    const double displacement=large?.12:.005, angle=large?.4:.02;
    s.position[0]+=displacement;s.q[0]+=displacement/.2;
    s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(angle,V3::UnitX()));s.q[3]+=angle;
    s.dq[0]=.05;s.dq[3]=-.03;
    const double expected_force=std::clamp(-400*displacement-40*.01,-8.0,8.0);
    const double expected_moment=std::clamp(-15*angle+5*.03,-1.5,1.5);
    int tick=1;
    for(double load : {0.,.69,.70,.80,.999,1.,1.01,1.2,.9,.79,0.}) {
      s.external_wrench[0]=load*c.contact_force_soft;
      s.external_wrench[3]=load*c.contact_moment_soft;
      s.external_torque=s.jacobian.transpose()*s.external_wrench;
      const auto out=control.step(s,cmd,false,tick++*.001,.001);
      assert(out.phase==Phase::Active && !out.yielding && out.drive_scale==1);
      assert(!out.reference_limited);
      assert((out.reference_position-cmd.position).norm()<1e-12);
      assert(out.reference_orientation.angularDistance(cmd.orientation)<1e-12);
      assert(out.reference_twist.norm()<1e-12);
      assert(std::abs(out.task_wrench[0]-expected_force)<1e-12);
      assert(std::abs(out.task_wrench[3]-expected_moment)<1e-12);
      assert((out.torque-s.previous_torque).cwiseAbs().maxCoeff()<=c.torque_rate_limit*.001+1e-12);
      s.previous_torque=out.torque;
    }
  }
}
static void existing_law_remains_distinct() {
  Config old=config();old.compliance_mode="contact_yielding";
  auto s=sample();const auto cmd=hold(s);
  Controller previous(old,s,1,0), continuous(config(),s,1,0);
  previous.step(s,cmd,false,0,.001);continuous.step(s,cmd,false,0,.001);
  s.position[0]+=.12;s.q[0]+=.6;s.external_wrench[3]=3.6;
  const auto a=previous.step(s,cmd,false,.001,.001);
  const auto b=continuous.step(s,cmd,false,.001,.001);
  assert(a.yielding && a.drive_scale==0 && a.task_wrench.norm()<1e-12);
  assert((a.reference_position-s.position).norm()<1e-12);
  assert(!b.yielding && b.drive_scale==1 && b.task_wrench[0]==-8);
  assert((b.reference_position-cmd.position).norm()<1e-12);
  s.external_wrench.setZero();
  const auto released=previous.step(s,cmd,false,.002,.001);
  const auto retained=continuous.step(s,cmd,false,.002,.001);
  assert((released.reference_position-s.position).norm()<1e-6);
  assert((retained.reference_position-cmd.position).norm()<1e-12);
}
static void target_changes_are_shaped_without_measured_pose_reseeding() {
  auto s=sample();auto cmd=hold(s);const auto c=config();
  Controller control(c,s,1,0);control.step(s,cmd,false,0,.001);
  cmd.sequence=2;cmd.position[0]+=.1;
  V3 previous=s.position;Output out;
  for(int i=1;i<2000;i++) {
    if(i%20==0) ++cmd.sequence;
    out=control.step(s,cmd,false,i*.001,.001);
    assert(out.phase==Phase::Active && !out.reference_limited);
    assert((out.reference_position-previous).norm()<=c.translation_velocity*.001+1e-12);
    assert(out.task_wrench.head<3>().norm()<=8+1e-12);
    previous=out.reference_position;s.previous_torque=out.torque;
  }
  assert((out.reference_position-cmd.position).norm()<1e-10);
  // The measured pose never followed; a reference leash would have stopped at 8 cm.
  assert((out.reference_position-s.position).norm()>.099);

  s=sample();cmd=hold(s);cmd.mode=Mode::Joint;
  Controller joint(c,s,1,0);joint.step(s,cmd,false,0,.001);
  s.q[0]+=.2;s.position[0]+=.04;s.external_wrench[3]=3.6;
  out=joint.step(s,cmd,false,.001,.001);
  assert(out.phase==Phase::Active && !out.yielding && !out.reference_limited);
  assert((out.reference_joints-cmd.joints).norm()<1e-12);
  assert(out.task_wrench[0]<0);
}
static void independent_stops_remain_active() {
  const auto c=config();
  for(Reason expected : {Reason::ForceLimit,Reason::MomentLimit,Reason::JointContact,
                        Reason::JointLimit,Reason::Overspeed,Reason::Singularity,
                        Reason::InvalidState,Reason::Timing,Reason::Watchdog,Reason::UserStop}) {
    auto s=sample();auto cmd=hold(s);cmd.expires_at=.1;
    // Joint-space decomposition still needs a nonsingular Jacobian. Cartesian
    // continuous impedance now uses a regularized posture projection instead.
    if(expected==Reason::Singularity) cmd.mode=Mode::Joint;
    Controller control(c,s,1,0);control.step(s,cmd,false,0,.001);
    if(expected==Reason::ForceLimit) s.external_wrench[0]=c.contact_force_stop;
    if(expected==Reason::MomentLimit) s.external_wrench[3]=c.contact_moment_stop;
    if(expected==Reason::JointContact) s.external_torque[0]=c.external_torque_limits[0];
    if(expected==Reason::JointLimit) s.q[0]=c.joint_upper[0];
    if(expected==Reason::Overspeed) s.dq[6]=c.joint_speed_limits()[6]+.01;
    if(expected==Reason::Singularity) s.jacobian.setZero();
    if(expected==Reason::InvalidState) s.q[0]=NAN;
    const auto out=control.step(s,cmd,expected==Reason::UserStop,
        expected==Reason::Watchdog?.101:.001,expected==Reason::Timing?.011:.001);
    assert(out.phase==Phase::Stopping || out.phase==Phase::Fault);
    assert(out.reason==expected && out.stop_trigger==expected);
  }
}
static void contact_limits_use_stiffness_axes_not_resultant_norm() {
  const auto c=config();
  for(bool rotated : {false,true}) {
    auto s=sample();
    if(rotated) s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.73,V3(1,2,3).normalized()));
    auto cmd=hold(s);Controller control(c,s,1,0);control.step(s,cmd,false,0,.001);
    // Resultants exceed the old 25 N / 10 Nm stops; every K-frame axis is below
    // the unchanged robot-side thresholds. External joint torques remain low.
    s.external_wrench.head<3>()=s.orientation*V3(20,20,0);
    s.external_wrench.tail<3>()=s.orientation*V3(8,8,0);
    auto out=control.step(s,cmd,false,.001,.001);
    assert(out.phase==Phase::Active && !out.yielding && out.drive_scale==1);
    s.external_wrench.tail<3>()=s.orientation*V3(10.01,0,0);
    out=control.step(s,cmd,false,.002,.001);
    assert(out.stop_trigger==Reason::MomentLimit);
  }
}
static void cartesian_impedance_remains_bounded_through_singularity() {
  auto s=sample();auto cmd=hold(s);const auto c=config();
  Controller control(c,s,1,0);control.step(s,cmd,false,0,.001);
  s.position[0]+=.2;s.q[6]+=.2;
  int tick=1;
  for(double scale : {1.,.1,.001,0.,.001,.1,1.}) {
    s.jacobian=scale*sample().jacobian;
    auto out=control.step(s,cmd,false,tick++*.001,.001);
    assert(out.phase==Phase::Active && out.torque.allFinite());
    assert((out.reference_position-cmd.position).norm()<1e-12);
    assert(out.task_wrench.head<3>().norm()<=8+1e-12);
    assert((out.torque-s.previous_torque).cwiseAbs().maxCoeff()<=.1+1e-12);
    s.previous_torque=out.torque;
  }
}
static Config full_envelope_config() {
  auto c=config();c.motion_limit_policy="franka";
  c.joint_lower=fr3_limits::lower();c.joint_upper=fr3_limits::upper();
  c.joint_limit_margin=0;c.measured_joint_velocity_limits=fr3_limits::speed();
  c.measured_translation_velocity_limit=3;c.measured_rotation_velocity_limit=2.5;
  c.validate();return c;
}
static void franka_motion_envelope_is_directional() {
  const auto c=full_envelope_config();
  for(bool upper : {false,true}) {
    for(bool outward : {false,true}) {
      auto s=sample();
      s.q[0]=upper?2.8507:-2.8507; // 0.05 rad from the manufacturer's bound.
      const auto cmd=hold(s);Controller control(c,s,1,0);
      control.step(s,cmd,false,0,.001);
      s.dq[0]=(upper==outward?1:-1)*.12;
      const auto out=control.step(s,cmd,false,.001,.001);
      // Joint 1 has ~0.114697 rad/s remaining outward speed here; motion
      // inward is still allowed, even outside the old rectangular profile.
      assert(outward?out.stop_trigger==Reason::Overspeed:out.phase==Phase::Active);
    }
  }
  auto s=sample();auto cmd=hold(s);Controller control(c,s,1,0);
  control.step(s,cmd,false,0,.001);
  s.position[0]+=.3;s.dq[0]=2.;
  auto out=control.step(s,cmd,false,.001,.001);
  assert(out.phase==Phase::Active && out.task_wrench.head<3>().norm()<=8+1e-12);
  s.dq[0]=2.63;
  out=control.step(s,cmd,false,.002,.001);
  assert(out.stop_trigger==Reason::Overspeed);
}
static Config contact_config() {
  auto c=full_envelope_config();c.bounded_cartesian_spring=true;
  c.external_torque_limits << 16,16,16,16,8,8,8;
  c.contact_force_stop=40;c.contact_moment_stop=20;
  c.validate();return c;
}
static void capped_spring_keeps_damping_effective() {
  const auto c=contact_config();
  for(double v : {-.3,-.1,0.,.1,.3}) {
    auto s=sample();const auto cmd=hold(s);Controller control(c,s,1,0);
    control.step(s,cmd,false,0,.001);
    s.position[0]+=.1;s.q[0]+=.5;s.dq[0]=v/.2;
    s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.4,V3::UnitX()));
    s.dq[3]=-.4;
    const auto out=control.step(s,cmd,false,.001,.001);
    assert(out.phase==Phase::Active && !out.yielding && out.drive_scale==1);
    assert((out.reference_position-cmd.position).norm()<1e-12);
    assert(std::abs(out.task_wrench[0]-std::clamp(-8-40*v,-8.0,8.0))<1e-12);
    assert(std::abs(out.task_wrench[3]-.5)<1e-12); // Brakes angular return.
    assert(out.task_force_limited && out.task_moment_limited && out.effort_limited);
    assert((out.torque-s.previous_torque).cwiseAbs().maxCoeff()<=.1+1e-12);
  }
  // Preserve the small-error linear law exactly.
  auto s=sample();auto cmd=hold(s);Controller control(c,s,1,0);
  control.step(s,cmd,false,0,.001);s.position[0]+=.005;s.dq[0]=-.05;
  const auto out=control.step(s,cmd,false,.001,.001);
  assert(std::abs(out.task_wrench[0]-(-2+.4))<1e-12);
  assert(!out.task_force_limited);
}
static void contact_envelope_allows_interaction_and_still_stops_overloads() {
  const auto c=contact_config();auto s=sample();auto cmd=hold(s);
  Controller control(c,s,1,0);control.step(s,cmd,false,0,.001);
  s.external_wrench << 30,0,0,0,15,0;s.external_torque[0]=12;
  auto out=control.step(s,cmd,false,.001,.001);
  assert(out.phase==Phase::Active && out.contact_load_ratio==.75);
  assert(out.peak_contact_load_ratio==.75 && !out.yielding);
  s.external_wrench.setZero();s.external_torque.setZero();
  out=control.step(s,cmd,false,.002,.001);
  assert(out.phase==Phase::Active && out.contact_load_ratio==0 && out.peak_contact_load_ratio==.75);
  for(int guard=0;guard<4;guard++) {
    s=sample();cmd=hold(s);Controller overload(c,s,1,0);
    overload.step(s,cmd,false,0,.001);
    if(guard==0) s.external_wrench[0]=40;
    if(guard==1) s.external_wrench[3]=20;
    if(guard==2) s.external_torque[0]=16;
    if(guard==3) s.external_torque[6]=8;
    out=overload.step(s,cmd,false,.001,.001);
    const Reason expected=guard==0?Reason::ForceLimit:(guard==1?Reason::MomentLimit:Reason::JointContact);
    assert(out.phase==Phase::Stopping && out.stop_trigger==expected);
    assert(out.contact_load_ratio==1 && out.peak_contact_load_ratio==1);
  }
}
static void sustained_wall_contact_and_commanded_withdrawal() {
  auto c=contact_config();auto s=sample();const auto origin=s;
  auto cmd=hold(s);cmd.expires_at=100;
  Controller control(c,s,1,0);control.step(s,cmd,false,0,.001);
  cmd.position[0]+=.1;
  double x=0,v=0,max_force=0;int loaded_ticks=0;Output out;
  // A one-axis mass and elastic wall test, not a Franka contact simulator.
  // The commanded target remains 10 cm behind the wall for 2.5 seconds.
  for(int i=1;i<=6000;i++) {
    if(i%20==1) ++cmd.sequence;
    if(i==2501) {cmd.position[0]=origin.position[0]-.04;++cmd.sequence;}
    const double wall=x>0?3000*x+8*std::max(0.0,v):0;
    s.position[0]=origin.position[0]+x;s.q[0]=origin.q[0]+x/.2;s.dq[0]=v/.2;
    s.external_wrench[0]=wall;s.external_torque=s.jacobian.transpose()*s.external_wrench;
    out=control.step(s,cmd,false,i*.001,.001);
    assert(out.phase==Phase::Active && !out.yielding && out.drive_scale==1);
    assert(std::abs(out.task_wrench[0])<=8+1e-12);
    assert((out.torque-s.previous_torque).cwiseAbs().maxCoeff()<=.1+1e-12);
    if(i>1800 && i<2500) {
      assert(std::abs(out.reference_position[0]-origin.position[0]-.1)<1e-10);
      if(wall>7 && wall<9) ++loaded_ticks;
    }
    max_force=std::max(max_force,wall);
    // tau=J^T F with J=.2; inertia expressed here in Cartesian coordinates.
    const double acceleration=(out.torque[0]/.2-wall-2*v)/1.5;
    v+=.001*acceleration;x+=.001*v;s.previous_torque=out.torque;
  }
  assert(loaded_ticks>600 && max_force<40);
  assert(std::abs(x+.04)<.001 && std::abs(v)<.001);
  assert(std::abs(out.reference_position[0]-cmd.position[0])<1e-10);
  std::cout << "Wall test peak contact " << max_force << " N; final error " << std::abs(x+.04) << " m\n";
}
int main() {
  Eigen::internal::set_is_malloc_allowed(false);
  unchanged_spring_across_contact_thresholds();
  existing_law_remains_distinct();
  target_changes_are_shaped_without_measured_pose_reseeding();
  independent_stops_remain_active();
  contact_limits_use_stiffness_axes_not_resultant_norm();
  cartesian_impedance_remains_bounded_through_singularity();
  franka_motion_envelope_is_directional();
  capped_spring_keeps_damping_effective();
  contact_envelope_allows_interaction_and_still_stops_overloads();
  sustained_wall_contact_and_commanded_withdrawal();
  std::cout << "Continuous impedance and existing yielding comparison checks passed\n";
}
