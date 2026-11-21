#pragma once
#include "controller.hpp"
#include "collision_settings.hpp"
#include "telemetry_recorder.hpp"
#include "motion_inhibit.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fcc {
double steady_seconds();
const char* robot_mode_name(int mode);
struct HardwareConfig {
  std::string robot_ip;
  std::string robot_model;
  bool read_only=false;
  bool check_end_effector=false;
  double expected_ee_mass=0;
  std::array<double,3> expected_ee_com{0,0,0};
  std::array<double,9> expected_ee_inertia{0,0,0,0,0,0,0,0,0};
  void validate_end_effector(double mass, const std::array<double,3>& com,
                            const std::array<double,9>& inertia) const;
  std::array<double,16> flange_to_tcp{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  double load_mass=0;
  std::array<double,3> load_com{0,0,0};
  std::array<double,9> load_inertia{0,0,0,0,0,0,0,0,0};
};
// Large/string diagnostics are constructed only AFTER leaving the control
// callback. RT snapshots hold a null shared pointer during normal control.
struct RobotObservation {
  Sample sample;
  V7 measured_torque=V7::Zero();
  std::array<double,16> base_to_ee{}, flange_to_ee{}, ee_to_stiffness{};
  std::string current_errors, last_motion_errors, robot_state_json;
  bool cartesian_reflex=false, joint_reflex=false;
};
struct ExceptionRecord {
  RobotObservation state;
  std::array<double,7> sent_torques{}, sent_joint_positions{}, sent_joint_velocities{};
  std::array<double,16> sent_pose{};
  std::array<double,6> sent_twist{};
};
struct FaultDiagnostics {
  Reason reason=Reason::Communication;
  std::string exception_type, message;
  double caught_at=0;
  std::vector<ExceptionRecord> records;
  bool post_read_attempted=false, post_read_completed=false, post_read_fresh=false, post_read_valid=false;
  double post_read_at=0;
  RobotObservation post_observation;
  std::string post_read_error;
};
struct RecoveryInfo {
  std::string status, error;
  bool automatic=false, robot_recovery_called=false;
  std::uint64_t attempt=0;
  double started_at=0, completed_at=0;
  Reason previous_reason=Reason::None, previous_trigger=Reason::None;
  Sample previous_sample;
  std::shared_ptr<const FaultDiagnostics> previous_fault_diagnostics;
};
enum class ObservationSource { RobotRead, Simulation, ControlCallback, StopRead,
                               CachedAfterFault, PostFaultRead, Recovery };
inline const char* observation_source_name(ObservationSource source) {
  switch(source) {
    case ObservationSource::RobotRead:return "robot_read";
    case ObservationSource::Simulation:return "simulation";
    case ObservationSource::ControlCallback:return "control_callback";
    case ObservationSource::StopRead:return "stop_observation";
    case ObservationSource::CachedAfterFault:return "cached_before_fault";
    case ObservationSource::PostFaultRead:return "post_fault_read";
    case ObservationSource::Recovery:return "explicit_recovery_read";
  }
  return "unknown";
}
struct Snapshot {
  Sample sample;
  ObservationSource observation_source=ObservationSource::Simulation;
  CollisionSettings collision_settings;
  std::shared_ptr<const FaultDiagnostics> fault_diagnostics;
  std::shared_ptr<const RecoveryInfo> recovery;
  Sample stop_sample;
  bool stop_sample_valid=false;
  double stop_sampled_at=0;
  Output output;
  double sampled_at=0, cycle_time=0;
};
class Backend {
 public:
  Backend(const Config&, const std::string& kind, const HardwareConfig&);
  ~Backend();
  Backend(const Backend&)=delete;
  std::uint64_t arm(std::optional<std::uint64_t> motion_generation=std::nullopt);
  std::uint64_t start(Command first, double deadline, bool automatic_recovery=false,
                      std::optional<std::uint64_t> motion_generation=std::nullopt);
  Command home_command(); // Cached configured-TCP FK, never robot I/O or a command.
  std::string model_urdf() const; // Startup cache only; empty when unavailable.
  void submit(Command, double validity_seconds, std::optional<std::uint64_t> motion_generation=std::nullopt);
  void request_stop();
  MotionInhibitStatus inhibit_motion() noexcept {return motion_inhibit_.assert_inhibit();}
  MotionInhibitStatus motion_inhibit_status() const noexcept {return motion_inhibit_.status();}
  MotionInhibitStatus release_motion_inhibit(std::uint64_t generation);
  void recover(std::optional<std::uint64_t> motion_generation=std::nullopt);
  void configure(const Config&);
  Snapshot snapshot();
  void refresh_state(); // Inactive read only: no setters, recovery or arming.
  std::string error();
  bool running() const {return running_.load();}
  bool lifecycle_busy() {
    std::unique_lock<std::mutex> lock(lifecycle_mutex_,std::try_to_lock);
    return !lock.owns_lock();
  }
  bool is_simulation() const {return simulation_;}
  bool is_read_only() const {return hardware_.read_only;}
  void inject_wrench(const V6&);
  TelemetryStatus prepare_telemetry(std::size_t capacity, std::array<double,3> speed_limits);
  TelemetryStatus telemetry_status();
  std::vector<TelemetryRecord> telemetry_data();
  void clear_telemetry(std::uint64_t capture_id);
  void close();
 private:
  struct RobotImpl;
  std::unique_ptr<RobotImpl> robot_;
  Config config_;
  HardwareConfig hardware_;
  bool simulation_;
  std::atomic<bool> running_{false}, stop_requested_{false};
  MotionInhibit motion_inhibit_;
  std::uint64_t epoch_=0, last_sequence_=0, collision_attempt_=0, recovery_attempt_=0;
  bool automatic_recovery_used_=false;
  std::mutex lifecycle_mutex_, command_mutex_, snapshot_mutex_, error_mutex_, injection_mutex_;
  std::thread worker_;
  Command command_, home_command_;
  Snapshot snapshot_;
  std::string error_;
  V6 injected_wrench_=V6::Zero();
  TelemetryRecorder telemetry_;
  bool telemetry_exported_=false;
  void run(Sample, Config, std::uint64_t, std::uint64_t motion_generation, Command initial);
  void publish(const Snapshot&);
  Sample read_idle();
  std::uint64_t arm_locked(std::uint64_t motion_generation, const Command* first=nullptr, double deadline=0);
  void validate_command(const Command&) const;
  void recover_locked(bool automatic, std::uint64_t motion_generation);
  void try_automatic_recovery_locked(std::uint64_t motion_generation);
};
} // namespace fcc
