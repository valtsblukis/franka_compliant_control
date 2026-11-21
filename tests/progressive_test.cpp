#include "controller.hpp"
#include "progressive_spring.hpp"
#include "fr3_limits.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace fcc;

static Config config() {
  Config c("fr3");c.compliance_mode="operational_space";c.osc_feedback_mode="progressive";
  c.motion_limit_policy="franka";c.effort_limit_policy="task_bounded";c.collision_policy="robot_configured";
  c.joint_lower=fr3_limits::lower();c.joint_upper=fr3_limits::upper();c.joint_limit_margin=0;
  c.torque_limits<<87,87,87,87,12,12,12;c.torque_rate_limit=1000;
  c.measured_joint_velocity_limits=fr3_limits::speed();
  c.measured_translation_velocity_limit=3;c.measured_rotation_velocity_limit=2.5;
  c.translation_acceleration=9;c.rotation_acceleration=17;
  c.osc_translation_damping.setConstant(40);c.osc_rotation_damping.setConstant(8);
  c.commanded_force_limit=20;c.commanded_moment_limit=6;
  c.contact_force_soft=15;c.contact_force_stop=40;c.contact_moment_soft=3;c.contact_moment_stop=20;
  c.external_torque_limits<<16,16,16,16,8,8,8;c.validate();return c;
}
static Sample sample() {
  Sample s;s.q<<0,-.4,0,-1.9,0,1.5,0;s.position<<.45,0,.4;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1;
  s.mass=3*M7::Identity();s.dynamics_valid=true;return s;
}
static Command command(const Sample& s) {
  Command cmd;cmd.epoch=1;cmd.sequence=1;cmd.expires_at=100;
  cmd.position=s.position;cmd.orientation=s.orientation;cmd.has_orientation=true;return cmd;
}
static void smooth_monotone_spring_and_plateau() {
  const auto c=config();
  for(bool rotation:{false,true}) {
    const double soft=rotation?c.progressive_rotation_soft_radius:c.progressive_position_soft_radius;
    const double end=rotation?c.progressive_rotation_saturation_radius:c.progressive_position_saturation_radius;
    const double gain=rotation?c.progressive_rotation_soft_stiffness:c.progressive_translation_soft_stiffness;
    const double cap=rotation?c.commanded_moment_limit:c.commanded_force_limit;
    auto magnitude=[&](double r){return progressive_magnitude(r,soft,end,gain,cap);};
    double last=0;
    for(int i=0;i<=2000;i++) {
      const double value=magnitude(2*end*i/2000);
      assert(value>=last-1e-12 && value<=cap+1e-12);last=value;
    }
    assert(magnitude(soft/2)==gain*soft/2);
    constexpr double h=1e-7;
    assert(std::abs((magnitude(soft+h)-magnitude(soft-h))/(2*h)-gain)<.01);
    assert(std::abs((magnitude(end+h)-magnitude(end-h))/(2*h))<.01);
    assert((magnitude((soft+end)/2+h)-magnitude((soft+end)/2-h))/(2*h)>gain);
    for(double distance:{end,2*end,1000.,1e300}) {
      bool saturated=false;const V3 direction=V3(1,-2,3).normalized();
      const V3 force=progressive_spring(V3(distance*direction),soft,end,gain,cap,saturated);
      assert((force-cap*direction).norm()<1e-10);
      assert(saturated || std::abs(distance-end)<1e-12);
    }
  }
}
static void frozen_robot_can_be_retargeted_without_reference_windup() {
  auto c=config();auto s=sample();auto cmd=command(s);Controller control(c,s,1,0);
  control.step(s,cmd,false,0,.001);
  cmd.position=s.position+1000*V3::UnitX();++cmd.sequence;
  Output out;
  for(int i=1;i<=2000;i++) {
    // Refresh the same equilibrium, as a real client must do for the watchdog.
    if(i%20==0) ++cmd.sequence;
    out=control.step(s,cmd,false,i*.001,.001);
    assert(out.phase==Phase::Active && !out.yielding && !out.reference_limited);
    assert((out.reference_position-cmd.position).norm()<1e-12);
    assert(std::abs(out.task_wrench[0]-20)<1e-12 && out.progressive_force_saturated);
    s.previous_torque=out.torque;
  }
  // Withdraw to just 1 mm behind the frozen tool. A stale far-away Ruckig
  // reference must not keep pressing into the old target for seconds.
  cmd.position=s.position-.001*V3::UnitX();++cmd.sequence;
  out=control.step(s,cmd,false,2.001,.001);
  assert(out.phase==Phase::Active && out.sequence==cmd.sequence);
  assert(std::abs(out.spring_wrench[0]+.05)<1e-12 && out.task_wrench[0]<0);
  assert((out.torque-s.previous_torque).cwiseAbs().maxCoeff()<=1+1e-12);
  cmd.position=s.position+.2*V3::UnitY();++cmd.sequence;
  out=control.step(s,cmd,false,2.002,.001);
  assert(std::abs(out.task_wrench[0])<1e-12 && std::abs(out.task_wrench[1]-20)<1e-12);
  cmd.position=s.position;
  for(double angle:{.5,-.5,0.}) {
    cmd.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(angle,V3::UnitZ()));++cmd.sequence;
    out=control.step(s,cmd,false,2.003,.001);
    assert(out.phase==Phase::Active);
    assert(std::abs(out.task_wrench[5]-(angle>0?6:(angle<0?-6:0)))<1e-10);
  }
  assert(out.task_wrench.norm()<1e-10); // No stored integral/reference force remains.
}
static void damping_and_all_task_terms_share_the_budget() {
  auto c=config();
  // Deliberately huge old gains must not bypass the new nonlinear law.
  c.osc_translation_kp.setConstant(1e6);c.osc_rotation_kp.setConstant(1e6);
  c.osc_translation_stiffness.setConstant(1e6);c.osc_rotation_stiffness.setConstant(1e6);
  for(double velocity:{-.75,0.,.1,.75}) {
    auto s=sample();s.jacobian(0,0)=1;auto cmd=command(s);Controller control(c,s,1,0);
    control.step(s,cmd,false,0,.001);cmd.position[0]+=.5;++cmd.sequence;s.dq[0]=velocity;
    const auto out=control.step(s,cmd,false,.001,.001);
    assert(out.phase==Phase::Active && out.progressive_force_saturated);
    assert(std::abs(out.spring_wrench[0]-20)<1e-12);
    assert(std::abs(out.task_wrench[0]-std::clamp(20-40*velocity,-20.,20.))<1e-10);
    assert(out.task_wrench.head<3>().norm()<=20+1e-12);
  }
  auto s=sample();auto cmd=command(s);cmd.acceleration<<.3,.3,.3,2,2,2;
  Controller control(c,s,1,0);const auto out=control.step(s,cmd,false,0,.001);
  assert(out.task_wrench.head<3>().norm()<=20+1e-12);
  assert(out.task_wrench.tail<3>().norm()<=6+1e-12);
  assert(out.task_force_limited && out.task_moment_limited);
}
static void contact_is_not_a_switch_and_stops_still_work() {
  auto c=config();auto s=sample();auto cmd=command(s);cmd.position[0]+=.5;
  Controller control(c,s,1,0);control.step(s,cmd,false,0,.001);
  s.external_wrench.setConstant(100);s.external_torque.setConstant(100);
  auto out=control.step(s,cmd,false,.001,.001);
  assert(out.phase==Phase::Active && out.progressive_force_saturated && !out.yielding);
  auto expired=cmd;expired.expires_at=.001;expired.sequence++;
  out=control.step(s,expired,false,.002,.001);
  assert(out.sequence==cmd.sequence); // An invalid update cannot replace the target.
  out=control.step(s,cmd,false,101,.001);
  assert(out.stop_trigger==Reason::Watchdog);
  Controller stopped(c,s,1,0);
  assert(stopped.step(s,cmd,true,.001,.001).stop_trigger==Reason::UserStop);
  s=sample();Controller overspeed(c,s,1,0);overspeed.step(s,cmd,false,0,.001);
  s.dq[0]=2.63;
  assert(overspeed.step(s,cmd,false,.001,.001).stop_trigger==Reason::Overspeed);
}
static void wall_withdrawal_and_release_keep_the_target() {
  auto c=config();auto s=sample();const auto home=s;auto cmd=command(s);
  Controller control(c,s,1,0);control.step(s,cmd,false,0,.001);
  cmd.position[0]+=.2;++cmd.sequence;
  double x=0,velocity=0;bool withdrew=false;double max_wall=0,max_push_error=0;
  for(int i=1;i<=32000;i++) {
    if(i%20==0) ++cmd.sequence;
    if(i==5001) {cmd.position[0]=home.position[0]-.04;++cmd.sequence;}
    if(i==15001) {cmd.position[0]=home.position[0]+.2;++cmd.sequence;}
    // Wall exists only for the first two phases. Last phase is free release /
    // return toward a retained target; no reference reseeding is permitted.
    const double wall=(i<=15000 && x>0)?1000*x:0;
    const double push=(i>20000 && i<=22000)?35:0;
    s.position[0]=home.position[0]+x;s.q[0]=home.q[0]+x/.2;s.dq[0]=velocity/.2;
    s.external_wrench[0]=wall+push;s.external_torque=s.jacobian.transpose()*s.external_wrench;
    const auto out=control.step(s,cmd,false,i*.001,.001);
    assert(out.phase==Phase::Active && !out.yielding && !out.reference_limited);
    assert(out.task_wrench.head<3>().norm()<=20+1e-10);
    assert((out.reference_position-cmd.position).norm()<1e-10);
    if(i==5001) assert(out.spring_wrench[0]<0); // First new-goal callback withdraws.
    if(i==15000) withdrew=std::abs(x+.04)<.01;
    if(push>0) max_push_error=std::max(max_push_error,std::abs(x-.2));
    const double acceleration=(out.torque[0]/.2-wall-push-50*velocity)/75;
    velocity+=.001*acceleration;x+=.001*velocity;s.previous_torque=out.torque;
    max_wall=std::max(max_wall,wall);
  }
  assert(withdrew && max_push_error>.1 && std::abs(x-.2)<.005 && std::abs(velocity)<.005);
  std::cout<<"Progressive toy-wall peak "<<max_wall<<" N; final release error "<<std::abs(x-.2)<<" m\n";
}
int main() {
  Eigen::internal::set_is_malloc_allowed(false);
  smooth_monotone_spring_and_plateau();frozen_robot_can_be_retargeted_without_reference_windup();
  damping_and_all_task_terms_share_the_budget();contact_is_not_a_switch_and_stops_still_work();
  wall_withdrawal_and_release_keep_the_target();
  std::cout<<"Progressive curve, live retargeting, effort budgets and release checks passed\n";
}
