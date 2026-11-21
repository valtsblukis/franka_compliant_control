#pragma once
#include "controller.hpp"
#include "fr3_limits.hpp"
#include "operational_space.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fcc {
inline double backoff_tolerance(const JointLimitBackoffOptions& r) {
  return std::min(.02,.2*r.backoff_distance);
}
inline bool backoff_options_valid(const Config& c,const JointLimitBackoffOptions& r) {
  return c.robot_model=="fr3" && c.motion_limit_policy=="franka" &&
      (c.compliance_mode=="auto" || c.compliance_mode=="joint_impedance") && r.confirm_clear_path &&
      std::isfinite(r.trigger_margin) && r.trigger_margin>0 && r.trigger_margin<=.15 &&
      std::isfinite(r.backoff_distance) && r.backoff_distance>=.02 && r.backoff_distance<=.2 &&
      r.backoff_distance-backoff_tolerance(r)>r.trigger_margin &&
      std::isfinite(r.speed) && r.speed>0 && r.speed<=std::min(.05,c.joint_velocity) &&
      std::isfinite(r.torque_scale) && r.torque_scale>0 && r.torque_scale<=1;
}
inline void JointLimitBackoffOptions::validate(const Config& c) const {
  if(!backoff_options_valid(c,*this))
    throw std::invalid_argument("joint-limit backoff requires FR3 auto/joint impedance, confirmed clear path, and finite bounded options with clearance beyond the trigger band");
}
inline V7 backoff_torque_limits(const Config& c,const JointLimitBackoffOptions& r,
                              const JointLimitBackoffPlan& p) {
  V7 limits=c.joint_feedback_torque_limits;
  for(int i=0;i<7;i++) if(p.direction[i]!=0)
    limits[i]=std::min({limits[i],c.torque_limits[i],r.torque_scale*(i<4?2.:1.)});
  return limits;
}
inline Reason backoff_basic_state_reason(const Config& c,const Sample& s) {
  if(!s.finite() || !s.kinematics_valid || s.robot_has_errors) return Reason::InvalidState;
  if(!s.wrench_tcp_valid || !s.relief_setup_valid) return Reason::BackoffSetupMismatch;
  if(c.uses_osc_model()) {
    V6 w;V7 n;bool regularized;
    if(!operational_space_effort(s,c,V6::Zero(),V7::Zero(),w,n,regularized)) return Reason::InvalidState;
  }
  // Disqualifying load evidence must dominate a simultaneous local limit fault.
  if(s.external_wrench_tcp_base.head<3>().norm()>=std::min(40.,c.contact_force_stop)) return Reason::ForceLimit;
  if(s.external_wrench_tcp_base.tail<3>().norm()>=std::min(20.,c.contact_moment_stop)) return Reason::MomentLimit;
  if((s.external_torque.cwiseAbs().array()>=c.external_torque_limits.array()).any()) return Reason::JointContact;
  // Never allow a software exception outside the FR3 specified position range.
  if((s.q.array()<=fr3_limits::lower().array()).any() ||
     (s.q.array()>=fr3_limits::upper().array()).any()) return Reason::JointLimit;
  return Reason::None;
}
inline bool backoff_plan_valid(const Config& c,const JointLimitBackoffOptions& r,
                              const JointLimitBackoffPlan& p) {
  if(!backoff_options_valid(c,r) || !p.prepared || !p.start.allFinite() ||
     !p.goal.allFinite() || !p.direction.allFinite()) return false;
  bool selected=false;
  for(int i=0;i<7;i++) {
    const double lo=c.joint_lower[i]+c.joint_limit_margin,hi=c.joint_upper[i]-c.joint_limit_margin;
    const double d=p.direction[i];
    if(p.start[i]<=fr3_limits::lower()[i] || p.start[i]>=fr3_limits::upper()[i]) return false;
    if(p.start[i]<lo-.05 || p.start[i]>hi+.05) return false;
    if(d==0) {
      if(p.goal[i]!=p.start[i] || p.start[i]<=lo || p.start[i]>=hi) return false;
    } else {
      if(d!=1 && d!=-1) return false;
      selected=true;
      // The fixed selected set may be reseeded to the first callback sample,
      // within the 1 mrad startup tolerance, without adding another joint.
      if((d>0 && p.start[i]-lo>r.trigger_margin+.001) || (d<0 && hi-p.start[i]>r.trigger_margin+.001)) return false;
      if(d*(p.goal[i]-p.start[i])<=0 || std::abs(p.goal[i]-p.start[i])>.25 ||
         p.goal[i]<=lo+r.trigger_margin || p.goal[i]>=hi-r.trigger_margin) return false;
    }
  }
  return selected;
}
inline JointLimitBackoffPlan plan_joint_limit_backoff(const Config& c,const Sample& s,
                                                     const JointLimitBackoffOptions& r) {
  r.validate(c);
  const auto invalid=backoff_basic_state_reason(c,s);
  if(invalid!=Reason::None)
    throw std::runtime_error(std::string("joint-limit backoff preflight: ")+reason_name(invalid));
  if(s.dq.cwiseAbs().maxCoeff()>=std::min(.01,c.stop_velocity))
    throw std::runtime_error("joint-limit backoff requires a stationary start");
  // This action does not get contact relief's loaded-start exception.
  if(s.external_wrench.head<3>().norm()>=c.contact_force_soft ||
     s.external_wrench.tail<3>().norm()>=c.contact_moment_soft)
    throw std::runtime_error("joint-limit backoff requires the normal unloaded startup checks");
  JointLimitBackoffPlan p;p.start=p.goal=s.q;p.prepared=true;
  bool selected=false;
  for(int i=0;i<7;i++) {
    const double lo=c.joint_lower[i]+c.joint_limit_margin,hi=c.joint_upper[i]-c.joint_limit_margin;
    if(s.q[i]<lo-.05 || s.q[i]>hi+.05)
      throw std::runtime_error("joint-limit backoff starting configured-limit violation exceeds 0.05 rad");
    const bool low=s.q[i]-lo<=r.trigger_margin,high=hi-s.q[i]<=r.trigger_margin;
    if(low && high) throw std::runtime_error("joint-limit backoff interval is too narrow/ambiguous");
    if(low) {p.direction[i]=1;p.goal[i]=std::max(s.q[i]+r.backoff_distance,lo+r.backoff_distance);selected=true;}
    if(high) {p.direction[i]=-1;p.goal[i]=std::min(s.q[i]-r.backoff_distance,hi-r.backoff_distance);selected=true;}
  }
  if(!selected) throw std::runtime_error("no joints are near or beyond configured limits; no backoff needed");
  if(!backoff_plan_valid(c,r,p)) throw std::runtime_error("joint-limit backoff target cannot fit inside the configured interval");
  const V7 limits=backoff_torque_limits(c,r,p);
  for(int i=0;i<7;i++) if(p.direction[i]!=0 && std::abs(s.previous_torque[i])>limits[i])
    throw std::runtime_error("initial command torque exceeds selected-joint backoff budget");
  return p;
}
inline bool backoff_goal_reached(const Config& c,const Sample& s,const JointLimitBackoffOptions& r,
                                const JointLimitBackoffPlan& p) {
  const double tolerance=backoff_tolerance(r);
  for(int i=0;i<7;i++) if(p.direction[i]!=0) {
    const double lo=c.joint_lower[i]+c.joint_limit_margin,hi=c.joint_upper[i]-c.joint_limit_margin;
    if(s.q[i]<=lo || s.q[i]>=hi || p.direction[i]*(s.q[i]-p.goal[i]) < -tolerance) return false;
    const double clearance=p.direction[i]>0?s.q[i]-lo:hi-s.q[i];
    if(clearance<r.backoff_distance-tolerance || clearance<=r.trigger_margin) return false;
  }
  return true;
}
inline void update_backoff_progress(const Sample& s,const JointLimitBackoffPlan& p,V7& best_progress) {
  for(int i=0;i<7;i++) if(p.direction[i]!=0)
    best_progress[i]=std::max(best_progress[i],p.direction[i]*(s.q[i]-p.start[i]));
}
inline Reason backoff_observation_reason(const Config& c,const Sample& s,const JointLimitBackoffOptions& r,
                                        const JointLimitBackoffPlan& p,const V7& best_progress,
                                        bool require_goal=false,bool active_budget=false) {
  if(!backoff_plan_valid(c,r,p) || !best_progress.allFinite()) return Reason::ReferenceError;
  Reason result=backoff_basic_state_reason(c,s);
  if(result!=Reason::None && !backoff_local_fault_allowed(result)) return result;
  // Do not short-circuit eligible position/speed faults: another predicate or
  // another joint in the same sample may forbid local fault replacement.
  const auto eligible=[&](Reason reason) {if(result==Reason::None) result=reason;};
  const V6 velocity=s.jacobian*s.dq;
  if(s.dq.cwiseAbs().maxCoeff()>=.08 ||
     (s.dq.cwiseAbs().array()>c.joint_speed_limits().array()).any() ||
     velocity.head<3>().norm()>std::min(.15,c.translation_speed_limit()) ||
     velocity.tail<3>().norm()>std::min(.3,c.rotation_speed_limit())) eligible(Reason::Overspeed);
  const V7 positive=fr3_limits::directional_speed(s.q,true),negative=fr3_limits::directional_speed(s.q,false);
  const V7 limits=backoff_torque_limits(c,r,p);
  for(int i=0;i<7;i++) {
    const double lo=c.joint_lower[i]+c.joint_limit_margin,hi=c.joint_upper[i]-c.joint_limit_margin;
    const double d=p.direction[i];
    if(d==0) {
      if(s.q[i]<=lo || s.q[i]>=hi) eligible(Reason::JointLimit);
      if(std::abs(s.q[i]-p.start[i])>.01) return Reason::BackoffHoldDrift;
      if(s.dq[i]>positive[i] || s.dq[i]<-negative[i]) eligible(Reason::Overspeed);
    } else {
      if(active_budget && std::abs(s.previous_torque[i])>limits[i]+1e-6) return Reason::InvalidState;
      const double progress=d*(s.q[i]-p.start[i]);
      if(progress<-.001 || progress<best_progress[i]-.005 || d*s.dq[i]<-.01) return Reason::BackoffOutward;
      if(std::abs(s.q[i]-p.start[i])>.25) return Reason::BackoffTravel;
      if((d>0 && (s.q[i]<lo-.05 || s.q[i]>=hi)) ||
         (d<0 && (s.q[i]>hi+.05 || s.q[i]<=lo))) eligible(Reason::JointLimit);
      // Preserve the inward directional bound; only a tiny outward/noise
      // allowance is scoped to selected recovery joints near their boundary.
      if((d>0 && s.dq[i]>positive[i]) || (d<0 && s.dq[i]<-negative[i])) eligible(Reason::Overspeed);
    }
  }
  if(require_goal && !backoff_goal_reached(c,s,r,p)) return Reason::BackoffGoalLost;
  return result;
}
inline Reason backoff_start_observation_reason(const Config& c,const Sample& s,const JointLimitBackoffOptions& r,
                                              const JointLimitBackoffPlan& p,const V7& best_progress) {
  const auto result=backoff_observation_reason(c,s,r,p,best_progress,false,true);
  if(result!=Reason::None && !backoff_local_fault_allowed(result)) return result;
  if((s.q-p.start).cwiseAbs().maxCoeff()>.001) return Reason::BackoffStartChanged;
  if(s.external_wrench.head<3>().norm()>=c.contact_force_soft) return Reason::ForceLimit;
  if(s.external_wrench.tail<3>().norm()>=c.contact_moment_soft) return Reason::MomentLimit;
  if(s.dq.cwiseAbs().maxCoeff()>=std::min(.01,c.stop_velocity)) return Reason::Overspeed;
  return result;
}
} // namespace fcc
