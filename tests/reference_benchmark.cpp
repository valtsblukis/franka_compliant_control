// Offline reference-generator benchmark. No libfranka or robot connection.
#include "controller.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace fcc;

struct ReferenceResult { double seconds; bool speed_limited; };

static ReferenceResult settle_reference(const Config& config,bool rotation) {
  Sample s;
  s.q << 0,-.4,0,-1.9,0,1.5,0;
  s.position << .45,0,.4;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1;
  Command target;
  target.epoch=1;target.sequence=1;target.expires_at=30;
  target.position=s.position;target.orientation=s.orientation;target.has_orientation=true;
  Controller controller(config,s,1,0);
  controller.step(s,target,false,0,.001);
  // Impose a displacement while yielding, then release and retain the target.
  if(rotation) {
    s.orientation=Eigen::Quaterniond(Eigen::AngleAxisd(20*M_PI/180,V3::UnitX()));
    s.q[3]+=20*M_PI/180;
  } else {
    s.position[0]-=.1;s.q[0]-=.5;
  }
  s.external_wrench[0]=config.contact_force_soft;
  const auto yielded=controller.step(s,target,false,.001,.001);
  if(!yielded.yielding) throw std::runtime_error("benchmark did not enter yielding");
  s.external_wrench.setZero();
  bool saw_reference_speed_limit=false;
  for(int i=2;i<25000;i++) {
    if(i%20==0) ++target.sequence;
    const auto out=controller.step(s,target,false,i*.001,.001);
    if(out.phase!=Phase::Active) throw std::runtime_error(reason_name(out.reason));
    saw_reference_speed_limit=saw_reference_speed_limit || out.reference_speed_limited;
    const double position_error=(out.reference_position-target.position).norm();
    const double angle_error=out.reference_orientation.angularDistance(target.orientation);
    if(position_error<1e-6 && angle_error<1e-5 && out.reference_twist.norm()<1e-4) {
      return {(i-1)*.001,saw_reference_speed_limit};
    }
    // Ideal reference following isolates command shaping from robot dynamics.
    s.position=out.reference_position;s.orientation=out.reference_orientation;
    for(int j=0;j<6;j++) {
      s.dq[j]=out.reference_twist[j]/(j<3?.2:1);
      s.q[j]+=.001*s.dq[j];
    }
    s.previous_torque=out.torque;
  }
  throw std::runtime_error("reference did not settle in 25 s");
}

int main(int argc,char** argv) {
  try {
    if(argc!=13 && argc!=20) throw std::runtime_error("expected translation/rotation/joint v,a,j, three measured-speed limits and optionally seven joint overrides");
    Config c("fr3");
    c.translation_velocity=std::stod(argv[1]);c.translation_acceleration=std::stod(argv[2]);c.translation_jerk=std::stod(argv[3]);
    c.rotation_velocity=std::stod(argv[4]);c.rotation_acceleration=std::stod(argv[5]);c.rotation_jerk=std::stod(argv[6]);
    c.joint_velocity=std::stod(argv[7]);c.joint_acceleration=std::stod(argv[8]);c.joint_jerk=std::stod(argv[9]);
    c.measured_translation_velocity_limit=std::stod(argv[10]);
    c.measured_rotation_velocity_limit=std::stod(argv[11]);c.measured_joint_velocity_limit=std::stod(argv[12]);
    if(argc==20) for(int i=0;i<7;i++) c.measured_joint_velocity_limits[i]=std::stod(argv[13+i]);
    c.validate();
    const auto translation=settle_reference(c,false), rotation=settle_reference(c,true);
    std::cout << std::setprecision(8) << std::boolalpha
              << "{\"translation_10cm_reference_s\":" << translation.seconds
              << ",\"rotation_20deg_reference_s\":" << rotation.seconds
              << ",\"translation_reached_speed_limit\":" << translation.speed_limited
              << ",\"rotation_reached_speed_limit\":" << rotation.speed_limited << "}\n";
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';return 1;
  }
}
