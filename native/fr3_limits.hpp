#pragma once
#include "controller.hpp"
#include <algorithm>
#include <cmath>

namespace fcc::fr3_limits {
// Franka FCI specification: robot_specifications.html, FR3 position-based
// velocity limits. These are motion limits, not external-contact thresholds.
inline V7 lower() { return (V7()<<-2.9007,-1.8361,-2.9007,-3.0770,-2.8763,.4398,-3.0508).finished(); }
inline V7 upper() { return (V7()<<2.9007,1.8361,2.9007,-.1169,2.8763,4.6216,3.0508).finished(); }
inline V7 speed() { return (V7()<<2.62,2.62,2.62,2.62,5.26,4.18,5.26).finished(); }
inline V7 offset() { return (V7()<<.6599,.2517,.2000,.3533,.5757,.4878,.4628).finished(); }
inline V7 deceleration() { return (V7()<<6,2.585,3.5,4,17,5.5,17).finished(); }
inline V7 directional_speed(const V7& q, bool positive) {
  const V7 distance=positive?V7(upper()-q):V7(q-lower());
  const V7 nominal=speed(), off=offset(), dec=deceleration();
  V7 result;
  for(int i=0;i<7;i++)
    result[i]=std::min(nominal[i],std::max(0.0,std::sqrt(std::max(0.0,2*dec[i]*distance[i]))-off[i]));
  return result;
}
}
