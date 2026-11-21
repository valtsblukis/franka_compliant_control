// Pure spring algebra: no simulated dynamics or robot connection.
#include "progressive_spring.hpp"
#include <cassert>
#include <iostream>
using namespace fcc;
int main() {
  Eigen::internal::set_is_malloc_allowed(false);
  bool saturated=false;
  const V3 k=V3::Constant(400);
  const auto near=saturated_spring(V3(.001,0,0),k,20,saturated);
  assert(std::abs(near[0]-.4)<1e-12 && !saturated); // No soft/dead zone.
  const auto boundary=saturated_spring(V3(.05,0,0),k,20,saturated);
  assert(std::abs(boundary[0]-20)<1e-12 && saturated);
  for(double error:{.1,1.,1000.,1e300}) {
    const auto forward=saturated_spring(V3(error,0,0),k,20,saturated);
    assert(saturated && std::abs(forward[0]-20)<1e-12);
    const auto reverse=saturated_spring(V3(-error,0,0),k,20,saturated);
    assert(saturated && std::abs(reverse[0]+20)<1e-12);
  }
  const auto angular=saturated_spring(V3(0,.01,0),V3::Constant(120),6,saturated);
  assert(std::abs(angular[1]-1.2)<1e-12 && !saturated);
  const auto moment=saturated_spring(V3(0,1,0),V3::Constant(120),6,saturated);
  assert(std::abs(moment.norm()-6)<1e-12 && saturated);
  const auto diagonal=saturated_spring(V3(1,2,3),k,20,saturated);
  assert(std::abs(diagonal.norm()-20)<1e-12 && saturated);
  assert(saturated_spring(V3::Zero(),k,20,saturated).isZero() && !saturated);
  std::cout<<"No-soft-zone spring algebra passed (no plant or robot)\n";
}
