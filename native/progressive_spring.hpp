#pragma once
#include "controller.hpp"
#include <algorithm>

namespace fcc {
// Magnitude law: linear soft core, monotone cubic-Hermite shoulder, constant
// plateau. Config validation enforces k_soft*r_saturation < effort_limit,
// making the shoulder monotone and stiffer than the core in its middle.
// Value and first derivative match at both boundaries. This is a spring law,
// not a contact detector, state switch, reference reset or integral controller.
inline double progressive_magnitude(double radius, double soft_radius,
                                    double saturation_radius, double soft_stiffness,
                                    double effort_limit) {
  if(radius<=soft_radius) return soft_stiffness*radius;
  if(radius>=saturation_radius) return effort_limit;
  const double width=saturation_radius-soft_radius;
  const double u=(radius-soft_radius)/width, u2=u*u, u3=u2*u;
  const double at_soft=soft_stiffness*soft_radius;
  return std::clamp((2*u3-3*u2+1)*at_soft+(u3-2*u2+u)*width*soft_stiffness+
                    (-2*u3+3*u2)*effort_limit,0.0,effort_limit);
}
inline V3 progressive_spring(const V3& error, double soft_radius,
                             double saturation_radius, double soft_stiffness,
                             double effort_limit, bool& saturated) {
  const double scale=error.cwiseAbs().maxCoeff();
  saturated=false;
  if(scale==0) return V3::Zero();
  // Normalize without squaring potentially large finite target coordinates.
  const V3 scaled=error/scale;
  const double length=scaled.norm();
  saturated=scale>=saturation_radius || scale*length>=saturation_radius;
  const double radius=saturated?saturation_radius:scale*length;
  return (scaled/length)*progressive_magnitude(radius,soft_radius,saturation_radius,
                                               soft_stiffness,effort_limit);
}
// No soft zone: linear stiffness immediately, with a norm plateau. Scaling
// avoids overflow in the norm for large finite virtual equilibrium errors.
inline V3 saturated_spring(const V3& error, const V3& stiffness,
                           double effort_limit, bool& saturated) {
  const V3 raw=stiffness.cwiseProduct(error);
  saturated=false;
  if(!raw.allFinite()) return raw; // Controller's invalid-effort guard handles this.
  const double scale=raw.cwiseAbs().maxCoeff();
  if(scale==0) return V3::Zero();
  const V3 direction=raw/scale;
  const double length=direction.norm();
  saturated=scale>=effort_limit/length;
  return saturated?V3((direction/length)*effort_limit):raw;
}
} // namespace fcc
