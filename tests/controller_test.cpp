#include "controller.hpp"
#include "hardware_stop.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace fcc;
static Sample sample() {
  Sample s;
  s.q << 0,-.4,0,-1.9,0,1.5,0;
  s.position << .45,0,.4;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1;
  return s;
}
static Command command(double expires=10) {
  Command c;
  c.epoch=1;c.sequence=1;c.position << .55,0,.4;c.expires_at=expires;
  return c;
}
static void reference_lifetime_regression() {
  // Ruckig stores the trajectory and elapsed time in OutputParameter. Compare
  // the controller's persistent stream to a correctly retained oracle output.
  Config config;
  auto s=sample();
  auto target=command();target.position=s.position+V3(.015,-.012,.010);
  target.has_orientation=true;target.orientation=s.orientation;
  Controller control(config,s,1,0);
  ruckig::Ruckig<3> oracle(.001);
  ruckig::InputParameter<3> input;
  ruckig::OutputParameter<3> output;
  for(int j=0;j<3;j++) {
    input.current_position[j]=s.position[j];input.target_position[j]=target.position[j];
    input.max_velocity[j]=config.translation_velocity/std::sqrt(3.0);
    input.max_acceleration[j]=config.translation_acceleration/std::sqrt(3.0);
    input.max_jerk[j]=config.translation_jerk/std::sqrt(3.0);
  }
  for(int i=0;i<1200;i++) {
    assert(oracle.update(input,output)>=0);output.pass_to_input(input);
    if(i%20==0) ++target.sequence;
    const auto actual=control.step(s,target,false,i*.001,.001);
    assert(actual.phase==Phase::Active);
    assert((actual.reference_position-Eigen::Map<V3>(output.new_position.data())).norm()<1e-10);
    assert((actual.reference_twist.head<3>()-Eigen::Map<V3>(output.new_velocity.data())).norm()<1e-10);
    s.previous_torque=actual.torque;
  }
  assert((Eigen::Map<V3>(output.new_position.data())-target.position).norm()<1e-10);

  s=sample();Controller joint_control(config,s,1,0);
  target=command();target.mode=Mode::Joint;target.joints=s.q;target.joints[0]+=.05;
  ruckig::Ruckig<7> joint_oracle(.001);
  ruckig::InputParameter<7> joint_input;
  ruckig::OutputParameter<7> joint_output;
  for(int j=0;j<7;j++) {
    joint_input.current_position[j]=s.q[j];joint_input.target_position[j]=target.joints[j];
    joint_input.max_velocity[j]=config.joint_velocity;
    joint_input.max_acceleration[j]=config.joint_acceleration;
    joint_input.max_jerk[j]=config.joint_jerk;
  }
  for(int i=0;i<1200;i++) {
    assert(joint_oracle.update(joint_input,joint_output)>=0);joint_output.pass_to_input(joint_input);
    const auto actual=joint_control.step(s,target,false,i*.001,.001);
    assert(actual.phase==Phase::Active);
    assert((actual.reference_joints-Eigen::Map<V7>(joint_output.new_position.data())).norm()<1e-10);
    s.previous_torque=actual.torque;
  }
  // A release must resume continuously from the displaced equilibrium rather
  // than snapping the reference to the original target between callbacks.
  s=sample();config.position_error_limit=.08;config.rotation_error_limit=.5;
  Controller released(config,s,1,0);
  target=command();target.position=s.position;
  target.has_orientation=true;target.orientation=s.orientation;
  for(int i=0;i<50;i++) released.step(s,target,false,i*.001,.001);
  s.position+=V3(.04,-.02,.01);
  s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.3,V3::UnitY()));
  s.external_wrench[3]=config.contact_moment_soft;
  auto previous=released.step(s,target,false,.050,.001);
  assert(previous.yielding);
  s.external_wrench.setZero();
  for(int i=51;i<1250;i++) {
    auto actual=released.step(s,target,false,i*.001,.001);
    assert(actual.phase==Phase::Active && !actual.yielding);
    assert((actual.reference_position-previous.reference_position).norm()<=config.translation_velocity*.001+1e-9);
    assert(actual.reference_orientation.angularDistance(previous.reference_orientation)<=config.rotation_velocity*.001+1e-9);
    previous=actual;s.previous_torque=actual.torque;
  }
  assert((previous.reference_position-target.position).norm()<1e-4);
}
int main() {
  Config cfg;
  cfg.validate();
  auto s=sample();
  // This assertion is enabled even in release builds. The controller uses only
  // fixed-size Eigen matrices; accidental dynamic allocation in step is a failure.
  Eigen::internal::set_is_malloc_allowed(false);
  reference_lifetime_regression();
  {
    Config faster("fr3");
    faster.measured_translation_velocity_limit=.5;
    faster.measured_rotation_velocity_limit=1.5;
    faster.measured_joint_velocity_limit=.8;
    faster.measured_joint_velocity_limits << 1.5,.9,1.2,1.1,2,1.3,2;
    faster.validate();
    // Every joint may pass the former shared trip, but must still stop at its
    // own boundary, in both directions. Cartesian speeds stay below their trips.
    for(int joint=0;joint<7;joint++) for(double sign : {-1.0,1.0}) {
      auto observed=sample();observed.jacobian.bottomRows<3>()*=.2;
      auto target=command();target.position=observed.position;
      Controller control(faster,observed,1,0);
      observed.dq[joint]=sign*(faster.joint_speed_limits()[joint]-.01);
      auto out=control.step(observed,target,false,0,.001);
      assert(out.phase==Phase::Active);
      observed.previous_torque=out.torque;
      observed.dq[joint]=sign*(faster.joint_speed_limits()[joint]+.01);
      out=control.step(observed,target,false,.001,.001);
      assert(out.phase==Phase::Stopping && out.reason==Reason::Overspeed);
    }
    // Cartesian guards remain independent of the faster joint overrides.
    for(int axis : {0,3}) {
      auto observed=sample();observed.jacobian(axis,axis)=1;
      const double limit=axis==0?faster.translation_speed_limit():faster.rotation_speed_limit();
      const int joint=axis==0?0:4;
      if(axis==3) {
        observed.jacobian(3,3)=0;observed.jacobian(3,4)=1;
        observed.jacobian(4,4)=0;observed.jacobian(4,3)=1;
      }
      Controller control(faster,observed,1,0);
      observed.dq[joint]=limit-.01;
      assert(control.step(observed,command(),false,0,.001).phase==Phase::Active);
      observed.dq[joint]=limit+.01;
      const auto out=control.step(observed,command(),false,.001,.001);
      assert(out.phase==Phase::Stopping && out.reason==Reason::Overspeed);
    }
    Config inherited("fr3");inherited.joint_velocity=.3;
    inherited.measured_joint_velocity_limit=.8;
    inherited.measured_joint_velocity_limits[5]=1.3;
    inherited.validate();
    assert(inherited.joint_speed_limits()[0]==.8 && inherited.joint_speed_limits()[5]==1.3);
    inherited.measured_joint_velocity_limit=0;
    inherited.validate();
    assert(std::abs(inherited.joint_speed_limits()[0]-.45)<1e-12);
    assert(inherited.joint_speed_limits()[5]==1.3);
  }
  {
    Config diagnostic;
    diagnostic.translation_stiffness.setConstant(10000);
    diagnostic.rotation_stiffness.setConstant(100);
    diagnostic.commanded_force_limit=.5;diagnostic.commanded_moment_limit=.1;
    diagnostic.position_error_limit=.01;
    auto observed=sample();Controller control(diagnostic,observed,1,0);
    auto target=command();target.has_orientation=true;
    target.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.05,V3::UnitX()));
    Output out;
    for(int i=0;i<1000;i++) {
      out=control.step(observed,target,false,i*.001,.001);
      observed.previous_torque=out.torque;
    }
    assert(out.task_force_limited && out.task_moment_limited && out.reference_limited);
    assert(out.drive_scale==1);
    observed.external_wrench[0]=.9*diagnostic.contact_force_soft;
    out=control.step(observed,target,false,1,.001);
    assert(std::abs(out.drive_scale-1.0/3)<1e-9);
  }
  {
    Controller control(cfg,s,1,0);
    Output o;
    for(int i=0;i<500;i++) {
      o=control.step(s,command(),false,i*.001,.001);
      assert(o.task_wrench.head<3>().norm()<=cfg.commanded_force_limit+1e-9);
      assert(o.task_wrench.tail<3>().norm()<=cfg.commanded_moment_limit+1e-9);
      assert((o.torque-s.previous_torque).cwiseAbs().maxCoeff()<=cfg.torque_rate_limit*.001+1e-9);
      s.previous_torque=o.torque;
    }
    assert(o.phase==Phase::Active);
    s.external_wrench[0]=cfg.contact_force_soft;
    o=control.step(s,command(),false,.5,.001);
    assert(o.yielding);
    assert((o.reference_position-s.position).norm()<1e-9);
    s.external_wrench[0]=cfg.contact_force_stop;
    o=control.step(s,command(),false,.501,.001);
    assert(o.phase==Phase::Stopping && o.reason==Reason::ForceLimit);
  }
  {
    s=sample();Controller control(cfg,s,1,0);
    auto c=command(.1);
    control.step(s,c,false,0,.001);
    c.sequence=2; c.epoch=2;c.expires_at=10;
    auto o=control.step(s,c,false,.11,.001);
    assert(o.phase==Phase::Stopping && o.reason==Reason::Watchdog);
  }
  {
    s=sample();Controller control(cfg,s,1,0);
    s.jacobian.setZero();
    auto o=control.step(s,command(),false,0,.001);
    assert(o.reason==Reason::Singularity);
  }
  {
    s=sample();Controller control(cfg,s,1,0);
    s.q[0]=NAN;
    auto o=control.step(s,command(),false,0,.001);
    assert(o.finished && o.phase==Phase::Fault);
  }
  {
    s=sample();Controller control(cfg,s,1,0);
    auto c=command();c.mode=Mode::Joint;c.joints=s.q;c.joints[0]+=.1;
    auto o=control.step(s,c,false,0,.001);
    assert(o.mode==Mode::Joint && o.phase==Phase::Active);
    o=control.step(s,c,true,.001,.001);
    assert(o.phase==Phase::Stopping);
    for(int i=2;i<250;i++) {o=control.step(s,c,false,i*.001,.001);s.previous_torque=o.torque;}
    assert(o.finished && o.phase==Phase::Disarmed);
  }
  {
    s=sample();Controller control(cfg,s,1,0);
    auto c=command();c.has_orientation=true;
    c.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(.10,V3::UnitZ()));
    V6 previous_velocity=V6::Zero(), previous_acceleration=V6::Zero();
    for(int i=0;i<400;i++) {
      const auto o=control.step(s,c,false,i*.001,.001);
      assert(o.phase==Phase::Active);
      const V6 a=(o.reference_twist-previous_velocity)/.001;
      assert(o.reference_twist.head<3>().norm()<=cfg.translation_velocity+1e-8);
      assert(o.reference_twist.tail<3>().norm()<=cfg.rotation_velocity+1e-8);
      assert(a.head<3>().norm()<=cfg.translation_acceleration+1e-6);
      assert(a.tail<3>().norm()<=cfg.rotation_acceleration+1e-6);
      assert(((a-previous_acceleration)/.001).head<3>().norm()<=cfg.translation_jerk+1e-4);
      assert(((a-previous_acceleration)/.001).tail<3>().norm()<=cfg.rotation_jerk+1e-3);
      previous_velocity=o.reference_twist;previous_acceleration=a;
      // Ideal following avoids intentionally overridden limits in contact/leash handling.
      s.position=o.reference_position;s.orientation=o.reference_orientation;
      s.previous_torque=o.torque;
    }
  }
  for(const auto reason : {Reason::MomentLimit,Reason::JointContact,Reason::JointLimit,Reason::Overspeed,Reason::Timing}) {
    s=sample();Controller control(cfg,s,1,0);
    double dt=.001;
    if(reason==Reason::MomentLimit) s.external_wrench[3]=cfg.contact_moment_stop;
    if(reason==Reason::JointContact) s.external_torque[6]=cfg.external_torque_limits[6];
    if(reason==Reason::JointLimit) s.q[0]=cfg.joint_lower[0];
    if(reason==Reason::Overspeed) s.dq[0]=1;
    if(reason==Reason::Timing) dt=.011;
    const auto o=control.step(s,command(),false,0,dt);
    assert(o.phase==Phase::Stopping && o.reason==reason);
  }
  Eigen::internal::set_is_malloc_allowed(true);
  {
    // A hand-driven displacement may exceed reference speed while the return
    // reference remains slow. Every measured-speed guard must still trip.
    Config p("fr3");
    p.translation_velocity=.02;p.rotation_velocity=.1;p.joint_velocity=.1;
    p.measured_translation_velocity_limit=.2;
    p.measured_rotation_velocity_limit=.6;p.measured_joint_velocity_limit=.6;
    p.translation_stiffness.setConstant(100);p.translation_damping.setConstant(20);
    p.rotation_stiffness.setConstant(5);p.rotation_damping.setConstant(2.5);
    p.contact_moment_soft=3;p.contact_moment_stop=10;
    p.commanded_force_limit=8;p.commanded_moment_limit=.8;p.position_error_limit=.08;
    s=sample();auto target=command();target.position=s.position;
    target.has_orientation=true;target.orientation=s.orientation;
    Controller control(p,s,1,0);
    control.step(s,target,false,0,.001);
    s.position[0]+=.1;s.dq[0]=.5; // 10 cm displacement, 10 cm/s hand-driven speed
    s.external_wrench[3]=2.1; // Used to stop at only 2 Nm.
    auto o=control.step(s,target,false,.001,.001);
    assert(o.phase==Phase::Active && !o.yielding);
    assert(o.task_wrench[0]<0 && o.task_wrench.head<3>().norm()<=8+1e-9);
    assert(o.reference_twist.head<3>().norm()<=.02+1e-9);
    s.external_wrench[3]=3.1;
    o=control.step(s,target,false,.002,.001);
    assert(o.phase==Phase::Active && o.yielding);
    assert(o.task_wrench[0]<0); // Damping still opposes the disturbance.
    s.external_wrench[3]=6.5; // Above the former 6 Nm stop, still yielding.
    s.external_torque=s.jacobian.transpose()*s.external_wrench;
    o=control.step(s,target,false,.003,.001);
    assert(o.phase==Phase::Active && o.yielding && o.stop_trigger==Reason::None);
    assert(o.task_wrench[0]<0);
    s.external_torque.setZero();
    s.external_wrench.setZero();s.dq.setZero();
    for(int i=4;i<1500;i++) {
      o=control.step(s,target,false,i*.001,.001);
      assert(o.phase==Phase::Active);
      assert(o.reference_twist.head<3>().norm()<=.02+1e-9);
    }
    assert(!o.yielding && o.reference_position[0]<s.position[0]-.005);
    assert(o.task_wrench[0]<0); // Release restores the original-target pull.
    for(int axis=0;axis<3;axis++) {
      s=sample();Config limits=p;
      if(axis==0) limits.measured_translation_velocity_limit=.09;
      if(axis==1) limits.measured_rotation_velocity_limit=.4;
      Controller guarded(limits,s,1,0);
      if(axis==0) s.dq[0]=.5; // 0.1 m/s, below the joint guard.
      if(axis==1) s.dq[3]=.5; // 0.5 rad/s, below the joint guard.
      if(axis==2) s.dq[6]=.7; // Nullspace motion, zero Cartesian speed here.
      const auto stopped=guarded.step(s,target,false,0,.001);
      assert(stopped.phase==Phase::Stopping && stopped.reason==Reason::Overspeed);
    }
    for(const auto reason : {Reason::MomentLimit,Reason::ForceLimit,Reason::JointContact}) {
      s=sample();Controller guarded(p,s,1,0);
      if(reason==Reason::MomentLimit) s.external_wrench[3]=p.contact_moment_stop;
      if(reason==Reason::ForceLimit) s.external_wrench[0]=p.contact_force_stop;
      if(reason==Reason::JointContact) s.external_torque[0]=p.external_torque_limits[0];
      const auto stopped=guarded.step(s,target,false,0,.001);
      assert(stopped.reason==reason && stopped.stop_trigger==reason);
    }
  }
  {
    // Regression from first hardware hold: a persistent uncompensated load can
    // sustain joint velocity during a damping-only stop. Handoff must occur on
    // the first stop callback, not after a two-second timeout.
    s=sample();
    s.dq[3]=-.04;
    s.external_wrench[2]=3.8;
    Controller control(cfg,s,1,0);
    const auto active=control.step(s,command(),false,.001,.001);
    assert(!handoff_hardware_stop(active));
    const auto stopping=control.step(s,command(),true,.002,.001);
    assert(stopping.phase==Phase::Stopping && !stopping.finished);
    assert(handoff_hardware_stop(stopping));
    HardwareStopMonitor monitor(.002,cfg);
    assert(monitor.observe(.003,s,true)==StopObservation::Pending);
    s.dq.setZero();
    assert(monitor.observe(.010,s,false)==StopObservation::Pending);
    assert(monitor.observe(.020,s,true)==StopObservation::Pending);
    s.dq[3]=-.04; // one slow sample must not confirm a still-moving robot
    assert(monitor.observe(.080,s,true)==StopObservation::Pending);
    s.dq.setZero();
    assert(monitor.observe(.100,s,true)==StopObservation::Pending);
    assert(monitor.observe(.201,s,true)==StopObservation::Confirmed);
    assert(normal_stop_reason(Reason::UserStop));
    assert(normal_stop_reason(Reason::Watchdog));
    assert(!normal_stop_reason(Reason::MomentLimit));
    HardwareStopMonitor timeout(0,cfg);
    s.dq[3]=-.04;
    assert(timeout.observe(cfg.stop_timeout,s,true)==StopObservation::TimedOut);
    // Failure to settle must not erase the contact event that initiated stop.
    s=sample();s.dq[3]=.04;
    Controller timed(cfg,s,1,0);
    s.external_wrench[3]=cfg.contact_moment_stop;
    auto out=timed.step(s,command(),false,0,.001);
    s.external_wrench.setZero();
    for(int i=1;i<2100 && !out.finished;i++) {
      s.previous_torque=out.torque;
      out=timed.step(s,command(),false,i*.001,.001);
    }
    assert(out.finished && out.reason==Reason::StopTimeout);
    assert(out.stop_trigger==Reason::MomentLimit);
    s=sample();Controller fault(cfg,s,1,0);s.q[0]=NAN;
    assert(handoff_hardware_stop(fault.step(s,Command{},false,.001,.001)));
  }
  {
    Config fr3("fr3");
    fr3.validate();
    s=sample();
    // FR3 permits joint 6 above Panda's upper limit in this profile.
    s.q[5]=4.0;
    Controller control(fr3,s,1,0);
    assert(control.step(s,Command{},false,0,0).phase==Phase::Armed);
    for(int i=0;i<7;i++) {
      for(double edge : {fr3.joint_lower[i]+fr3.joint_limit_margin,
                         fr3.joint_upper[i]-fr3.joint_limit_margin}) {
        s=sample();
        Controller boundary(fr3,s,1,0);
        s.q[i]=edge;
        const auto o=boundary.step(s,Command{},false,0,0);
        assert(o.phase==Phase::Stopping && o.reason==Reason::JointLimit);
      }
    }
  }
  std::cout << "Native controller checks passed\n";
}
