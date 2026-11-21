// Read-only commissioning probe: no setters, recovery, control or gripper calls.
#include <franka/robot.h>
#include <franka/model.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <chrono>
template<class T> void array_json(std::ostream& out, const T& values) {
  out << '[';
  for(size_t i=0;i<values.size();++i) {if(i) out << ',';out << values[i];}
  out << ']';
}
int main(int argc, char** argv) {
  try {
    if(argc!=3 && argc!=4) throw std::runtime_error("usage: read_robot ROBOT_IP OUTPUT.json [SAMPLE_DURATION_MS]");
    const int duration_ms=argc==4?std::stoi(argv[3]):0;
    if(duration_ms<0 || duration_ms>5000) throw std::runtime_error("sample duration must be 0..5000 ms");
    if(std::filesystem::exists(argv[2])) throw std::runtime_error("output already exists");
    const std::string urdf_path=std::string(argv[2])+".urdf";
    if(std::filesystem::exists(urdf_path)) throw std::runtime_error("URDF output already exists");
    const std::string samples_path=std::string(argv[2])+".samples.jsonl";
    if(duration_ms && std::filesystem::exists(samples_path)) throw std::runtime_error("samples output already exists");
    franka::Robot robot(argv[1],franka::RealtimeConfig::kEnforce);
    const auto urdf=robot.getRobotModel();
    auto model=robot.loadModel();
    const auto state=robot.readOnce();
    const auto coriolis=model.coriolis(state);
    const auto jacobian=model.zeroJacobian(franka::Frame::kEndEffector,state);
    std::ofstream output(argv[2]);
    if(!output) throw std::runtime_error("cannot open output file");
    output << std::setprecision(17) << "{\"server_protocol_version\":" << robot.serverVersion()
           << ",\"state\":" << state << ",\"coriolis\":";
    array_json(output,coriolis);
    output << ",\"zero_jacobian\":";array_json(output,jacobian);
    output << ",\"model_pose\":";array_json(output,model.pose(franka::Frame::kEndEffector,state));
    // These getters calculate bounds from the robot-exported URDF; no control,
    // recovery, setters or deprecated hardcoded velocity-envelope helpers.
    output << ",\"lower_joint_velocity_limits_at_q\":";array_json(output,robot.getLowerJointVelocityLimits(state.q));
    output << ",\"upper_joint_velocity_limits_at_q\":";array_json(output,robot.getUpperJointVelocityLimits(state.q));
    output << ",\"lower_joint_velocity_limits_at_q_d\":";array_json(output,robot.getLowerJointVelocityLimits(state.q_d));
    output << ",\"upper_joint_velocity_limits_at_q_d\":";array_json(output,robot.getUpperJointVelocityLimits(state.q_d));
    output << "}\n";
    if(!output) throw std::runtime_error("failed writing robot observation");
    std::ofstream urdf_output(urdf_path);
    urdf_output << urdf;
    if(!urdf_output) throw std::runtime_error("failed writing robot model");
    if(duration_ms) {
      std::ofstream samples_output(samples_path);
      if(!samples_output) throw std::runtime_error("cannot open samples output");
      samples_output << std::setprecision(17);
      const auto end=std::chrono::steady_clock::now()+std::chrono::milliseconds(duration_ms);
      while(std::chrono::steady_clock::now()<end) samples_output << robot.readOnce() << '\n';
      if(!samples_output) throw std::runtime_error("failed writing samples");
      std::cout << "Read-only resting-state samples saved to " << samples_path << '\n';
    }
    std::cout << "Read-only robot state and model saved to " << argv[2] << '\n';
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';return 1;
  }
}
