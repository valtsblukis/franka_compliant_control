#pragma once
#include "controller.hpp"
#include <algorithm>
#include <cmath>

namespace fcc {
// Hardware delegates deceleration to Franka as soon as stopping is requested.
// Waiting for the test plant's damping-only stop permits sag under a steady load.
inline bool handoff_hardware_stop(const Output& out) {
  return out.phase==Phase::Stopping || out.finished;
}

// Used at worker entry and immediately before SDK control. A callback expiry
// check is still necessary for a deadline crossed inside the SDK start request.
inline Reason pre_control_stop_reason(const Command& initial,double now,bool cancelled) {
  if(cancelled) return Reason::UserStop;
  if(initial.sequence==0) return Reason::None; // Explicit hold-only arm, no leased command.
  if(!std::isfinite(now) || !std::isfinite(initial.expires_at)) return Reason::ReferenceError;
  if(now<initial.expires_at) return Reason::None;
  if(initial.backoff_joint_limits) return Reason::BackoffTimeout;
  if(initial.relieve_contact) return Reason::ReliefTimeout;
  return initial.finish_at_target?Reason::HomeTimeout:Reason::Watchdog;
}
enum class StopObservation { Pending, Confirmed, TimedOut };
class HardwareStopMonitor {
 public:
  HardwareStopMonitor(double now,const Config& c,double velocity_ceiling=0)
    :deadline_(now+c.stop_timeout),settle_time_(c.stop_settle_time),
     velocity_(velocity_ceiling>0?std::min(c.stop_velocity,velocity_ceiling):c.stop_velocity) {}
  StopObservation observe(double now,const Sample& sample,bool idle_without_errors) {
    const bool stationary=idle_without_errors && sample.finite() &&
        sample.dq.cwiseAbs().maxCoeff()<velocity_;
    if(stationary) {
      if(stationary_since_<0) stationary_since_=now;
      if(now-stationary_since_>=settle_time_) return StopObservation::Confirmed;
    } else stationary_since_=-1;
    return now>=deadline_?StopObservation::TimedOut:StopObservation::Pending;
  }
 private:
  double deadline_,settle_time_,velocity_,stationary_since_=-1;
};
inline bool normal_stop_reason(Reason reason) {
  return reason==Reason::UserStop || reason==Reason::Watchdog ||
         reason==Reason::HomeReached || reason==Reason::HomeTimeout || reason==Reason::ReliefReleased ||
         reason==Reason::BackoffReached;
}
inline Reason observe_stop_guards(const Config& c,const Sample& s,Output& out,bool ordinary_contact) {
  Reason invalid=Reason::None;
  if(!s.finite() || s.robot_has_errors) invalid=Reason::InvalidState;
  else if(ordinary_contact) invalid=ordinary_hard_contact_reason(c,s);
  if(invalid!=Reason::None) {
    if(normal_stop_reason(out.reason) || backoff_local_fault_allowed(out.reason)) out.reason=invalid;
    if(out.stop_trigger==Reason::None) out.stop_trigger=invalid;
    out.relief_released=out.backoff_reached=false;
  }
  return invalid;
}
} // namespace fcc
