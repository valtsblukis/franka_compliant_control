// Runs entirely offline against a saved robot URDF; never constructs Robot.
#include "model_dynamics.hpp"
#include "operational_space.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
using namespace fcc;

int main(int argc, char** argv) {
  if(argc!=2) return 2;
  std::ifstream file(argv[1]);if(!file) return 2;
  franka::Model model(std::string((std::istreambuf_iterator<char>(file)),{}));
  franka::RobotState rs{};
  rs.q={0,-.4,0,-1.9,0,1.5,0};rs.dq={.2,-.1,.15,-.12,.17,-.08,.1};
  rs.F_T_EE={1,0,0,0,0,1,0,0,0,0,1,0,0,0,.174,1};
  rs.EE_T_K={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  rs.m_total=1.26;rs.F_x_Ctotal={-.019,-.001,.035};
  rs.I_total={.004965866,-.00005985,-.0013167,-.00005985,.006480866,-.0000693,-.0013167,-.0000693,.002374416};
  Sample s;s.q=Eigen::Map<const V7>(rs.q.data());s.dq=Eigen::Map<const V7>(rs.dq.data());
  Config config("fr3");
  for(int posture=0;posture<3;posture++) {
    rs.q[0]+=.1;rs.q[2]-=.05;rs.q[5]+=.03;
    s.q=Eigen::Map<const V7>(rs.q.data());
    sample_dynamics(model,rs,s);
    assert(s.mass.isApprox(s.mass.transpose(),1e-10));
    assert((s.mass.ldlt().vectorD().array()>0).all());
    const auto jac=model.zeroJacobian(franka::Frame::kEndEffector,rs);
    s.jacobian=Eigen::Map<const J67>(jac.data());
    // Independently differentiate poses twice along q(t)=q+t*dq.
    constexpr double h=1e-4;
    auto plus=rs.q,minus=rs.q;
    for(int i=0;i<7;i++) {plus[i]+=h*rs.dq[i];minus[i]-=h*rs.dq[i];}
    const auto p=model.pose(franka::Frame::kEndEffector,plus,rs.F_T_EE,rs.EE_T_K);
    const auto m=model.pose(franka::Frame::kEndEffector,minus,rs.F_T_EE,rs.EE_T_K);
    const auto z=model.pose(franka::Frame::kEndEffector,rs);
    const Eigen::Matrix4d second=(Eigen::Map<const Eigen::Matrix4d>(p.data())-
        2*Eigen::Map<const Eigen::Matrix4d>(z.data())+Eigen::Map<const Eigen::Matrix4d>(m.data()))/(h*h);
    assert((second.topRightCorner<3,1>()-s.jacobian_dot_velocity.head<3>()).norm()<1e-6);
    const Eigen::Matrix3d spin=second.topLeftCorner<3,3>()*
        Eigen::Map<const Eigen::Matrix4d>(z.data()).topLeftCorner<3,3>().transpose();
    V3 alpha;alpha << (spin(2,1)-spin(1,2))/2,(spin(0,2)-spin(2,0))/2,(spin(1,0)-spin(0,1))/2;
    assert((alpha-s.jacobian_dot_velocity.tail<3>()).norm()<1e-6);
    V6 a=V6::Constant(.1),wrench;V7 null;bool regularized;
    assert(operational_space_effort(s,config,a,V7::Ones(),wrench,null,regularized));
    assert(!regularized);
    const V6 actual=s.jacobian*s.mass.ldlt().solve(s.jacobian.transpose()*wrench+null)+s.jacobian_dot_velocity;
    assert((actual-a).norm()<1e-9);
  }
  std::vector<double> times;times.reserve(2000);
  for(int i=0;i<2000;i++) {
    const auto begin=std::chrono::steady_clock::now();
    const auto jac=model.zeroJacobian(franka::Frame::kEndEffector,rs);
    const auto coriolis=model.coriolis(rs);
    s.jacobian=Eigen::Map<const J67>(jac.data());s.coriolis=Eigen::Map<const V7>(coriolis.data());
    sample_dynamics(model,rs,s);
    V6 wrench;V7 null;bool regularized;
    assert(operational_space_effort(s,config,V6::Ones(),V7::Ones(),wrench,null,regularized));
    times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count());
  }
  std::sort(times.begin(),times.end());
  std::cout << "Offline model + OSC solve (not FCI callback) p50/p99/max us: "
            << times[1000] << '/' << times[1980] << '/' << times.back() << '\n';
  std::cout << "Saved-URDF mass, Jdot*dq and OSC checks passed; no robot connection\n";
}
