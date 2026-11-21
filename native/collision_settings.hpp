#pragma once
#include "controller.hpp"
#include <array>

namespace fcc {
enum class CollisionApplication { NotApplied, Applying, Applied, Failed, Simulated };
struct CollisionSettings {
  CollisionApplication status=CollisionApplication::NotApplied;
  std::array<double,7> lower_torque{}, upper_torque{};
  std::array<double,6> lower_wrench{}, upper_wrench{};
  double acknowledged_at=0;
  std::uint64_t attempt=0;
};
inline bool applies_robot_collision_settings(const Config& c) {
  return c.collision_policy!="robot";
}
inline CollisionSettings collision_request(const Config& c) {
  CollisionSettings result;
  for(int i=0;i<7;i++) {
    result.upper_torque[i]=c.external_torque_limits[i];
    result.lower_torque[i]=.7*result.upper_torque[i];
  }
  for(int i=0;i<6;i++) {
    result.lower_wrench[i]=i<3?c.contact_force_soft:c.contact_moment_soft;
    result.upper_wrench[i]=i<3?c.contact_force_stop:c.contact_moment_stop;
  }
  return result;
}
// Both acceleration and nominal thresholds are explicitly identical. The
// status means SDK acknowledgement, NOT a readback of robot configuration.
// Called only while inactive, never in the torque callback.
template<class Robot> void apply_collision_request(Robot& robot, CollisionSettings& request) {
  request.status=CollisionApplication::Applying;
  request.acknowledged_at=0;
  try {
    robot.setCollisionBehavior(request.lower_torque,request.upper_torque,
        request.lower_torque,request.upper_torque,request.lower_wrench,request.upper_wrench,
        request.lower_wrench,request.upper_wrench);
    request.status=CollisionApplication::Applied;
  } catch(...) {
    request.status=CollisionApplication::Failed;
    throw;
  }
}
inline const char* collision_application_name(CollisionApplication status) {
  switch(status) {
    case CollisionApplication::NotApplied:return "not_applied";
    case CollisionApplication::Applying:return "applying";
    case CollisionApplication::Applied:return "applied";
    case CollisionApplication::Failed:return "failed";
    case CollisionApplication::Simulated:return "simulated";
  }
  return "unknown";
}
} // namespace fcc
