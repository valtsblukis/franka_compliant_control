// Pure recovery admission checks. No robot/plant or recovery call.
#include "recovery_policy.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace fcc;
int main() {
  Config c("fr3");Sample s;s.q=nominal_home_joints();
  for(Reason reason:{Reason::None,Reason::ForceLimit,Reason::MomentLimit,Reason::JointContact,
                    Reason::Overspeed,Reason::JointLimit,Reason::RobotCartesianReflex,Reason::RobotJointReflex})
    assert(automatic_recovery_blocker(c,s,reason,true,true)==nullptr);
  for(Reason reason:{Reason::Timing,Reason::InvalidState,Reason::Communication,Reason::StopTimeout,Reason::ReferenceError})
    assert(automatic_recovery_blocker(c,s,reason,true,true)!=nullptr);
  assert(automatic_recovery_blocker(c,s,Reason::Overspeed,false,true)); // User stop/guiding, not Idle/Reflex.
  assert(automatic_recovery_blocker(c,s,Reason::RobotReflex,true,false)); // Non-collision error.
  auto bad=s;bad.dq[2]=c.stop_velocity;
  assert(std::strstr(automatic_recovery_blocker(c,bad,Reason::Overspeed,true,true),"stationary"));
  bad=s;bad.q[3]=c.joint_lower[3]+c.joint_limit_margin;
  assert(std::strstr(automatic_recovery_blocker(c,bad,Reason::JointLimit,true,true),"joint margins"));
  bad=s;bad.q[3]=NAN;assert(automatic_recovery_blocker(c,bad,Reason::JointLimit,true,true));
  for(int kind=0;kind<3;kind++) {
    bad=s;
    if(kind==0) bad.external_wrench[0]=c.contact_force_soft;
    if(kind==1) bad.external_wrench[3]=c.contact_moment_soft;
    if(kind==2) bad.external_torque[6]=.7*c.external_torque_limits[6];
    assert(std::strstr(automatic_recovery_blocker(c,bad,Reason::RobotJointReflex,true,true),"load"));
  }
  assert((nominal_home_joints()-(V7()<<0,-.4,0,-1.9,0,1.5,0).finished()).norm()==0);
  std::cout<<"Recovery admission and nominal-home checks passed; no robot or simulated plant\n";
}
