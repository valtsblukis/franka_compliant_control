// Offline regression for the robot-supplied URDF, including fixed sensor branches.
#include <franka/model.h>
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: model_test ROBOT.urdf OBSERVATION.txt");
    std::ifstream urdf_file(argv[1]), state_file(argv[2]);
    if (!urdf_file || !state_file) throw std::runtime_error("cannot open fixture");
    const std::string urdf((std::istreambuf_iterator<char>(urdf_file)), {});
    franka::Model model(urdf);
    // SDK 0.18.0 skips an inertia update when gravity(m_total=0) is called;
    // use a separate pristine model for the bare-arm reference, not a model
    // whose cache has already contained a payload.
    franka::Model bare_model(urdf);
    std::array<double, 7> q{};
    std::array<double, 16> tool{}, expected{};
    const std::array<double, 16> identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    for (auto& v : q) state_file >> v;
    for (auto& v : tool) state_file >> v;
    for (auto& v : expected) state_file >> v;
    if (!state_file) throw std::runtime_error("invalid observation fixture");
    const auto initial = model.pose(franka::Frame::kEndEffector, q, tool, identity);
    double pose_error = 0;
    for (size_t i=0; i<16; ++i) pose_error=std::max(pose_error,std::abs(initial[i]-expected[i]));
    std::cout << "model/state pose max error: " << pose_error << std::endl;
    if (!std::isfinite(pose_error) || pose_error > 5e-5)
      throw std::runtime_error("model pose disagrees with robot-reported pose");
    double derivative_error=0, gravity_error=0;
    for (int sample=0; sample<3; ++sample) {
      if (sample) {q[0]+=.07; q[2]-=.05; q[5]+=.03; tool[14]+=.035;}
      const auto raw=model.zeroJacobian(franka::Frame::kEndEffector,q,tool,identity);
      const Eigen::Matrix<double,6,7> jac=Eigen::Map<const Eigen::Matrix<double,6,7>>(raw.data());
      auto scaled=jac;
      scaled.bottomRows<3>()*=.2;
      const double minimum=scaled.jacobiSvd().singularValues().minCoeff();
      if (!jac.allFinite() || minimum < .02) throw std::runtime_error("Jacobian is invalid or singular");
      const auto pose=model.pose(franka::Frame::kEndEffector,q,tool,identity);
      const Eigen::Matrix4d t=Eigen::Map<const Eigen::Matrix4d>(pose.data());
      const Eigen::Matrix4d tool_inverse=Eigen::Map<const Eigen::Matrix4d>(tool.data()).inverse();
      // Independently check payload gravity against the derivative of its
      // potential energy, including a COM offset in all three flange axes.
      constexpr double mass=1.23;
      const std::array<double,3> com{-.021,.017,.049}, gravity{0,0,-9.81};
      const Eigen::Vector4d com_h(com[0],com[1],com[2],1);
      const auto loaded=model.gravity(q,mass,com,gravity);
      const auto unloaded=bare_model.gravity(q,0,{0,0,0},gravity);
      for (size_t joint=0; joint<7; ++joint) {
        auto plus=q, minus=q;
        constexpr double h=1e-6;
        plus[joint]+=h; minus[joint]-=h;
        const auto p=model.pose(franka::Frame::kEndEffector,plus,tool,identity);
        const auto m=model.pose(franka::Frame::kEndEffector,minus,tool,identity);
        const Eigen::Matrix4d delta=(Eigen::Map<const Eigen::Matrix4d>(p.data())-
            Eigen::Map<const Eigen::Matrix4d>(m.data()))/(2*h);
        const double payload_gravity=mass*9.81*(delta*tool_inverse*com_h).z();
        gravity_error=std::max(gravity_error,std::abs(loaded[joint]-unloaded[joint]-payload_gravity));
        Eigen::Matrix<double,6,1> numeric;
        numeric.head<3>()=delta.topRightCorner<3,1>();
        const Eigen::Matrix3d spin=delta.topLeftCorner<3,3>()*t.topLeftCorner<3,3>().transpose();
        numeric.tail<3>() << spin(2,1),spin(0,2),spin(1,0);
        derivative_error=std::max(derivative_error,(numeric-jac.col(joint)).cwiseAbs().maxCoeff());
      }
      std::cout << "scaled minimum singular value: " << minimum << '\n';
    }
    std::cout << "Jacobian finite-difference max error: " << derivative_error << '\n';
    if (!std::isfinite(derivative_error) || derivative_error > 1e-6)
      throw std::runtime_error("Jacobian disagrees with pose derivatives");
    std::cout << "Payload gravity/potential derivative max error: " << gravity_error << '\n';
    if (!std::isfinite(gravity_error) || gravity_error > 1e-6)
      throw std::runtime_error("SDK payload gravity disagrees with potential energy derivatives");
    std::cout << "Offline robot model checks passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
