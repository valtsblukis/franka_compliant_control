#include "motion_inhibit.hpp"
#include <cassert>
#include <thread>
#include <atomic>

using fcc::MotionInhibit;
template<class F> void rejects(F action) {
  bool failed=false;
  try {action();} catch(const std::runtime_error&) {failed=true;}
  assert(failed);
}
int main() {
  MotionInhibit gate;
  const auto initial=gate.permit();
  assert(initial==0 && !gate.status().active && gate.permitted(initial));
  auto first=gate.assert_inhibit();
  assert(first.active && !gate.permitted(initial));
  rejects([&](){gate.permit();});
  auto cleared=gate.release(first.generation);
  assert(!cleared.active && cleared.generation>first.generation);
  rejects([&](){gate.permit(initial);}); // Queued-before-latch requests stay invalid.
  rejects([&](){gate.release(first.generation);});
  const auto fresh=gate.permit(cleared.generation);
  const auto old=gate.assert_inhibit();
  const auto latest=gate.assert_inhibit();
  rejects([&](){gate.release(old.generation);});
  assert(gate.status().active && !gate.permitted(fresh));
  gate.release(latest.generation);

  // A concurrent reassertion can never be cleared by an older release.
  for(int i=0;i<500;++i) {
    const auto stop=gate.assert_inhibit();
    std::atomic<bool> ready{false};
    std::thread release([&](){while(!ready.load()) {};
      try {gate.release(stop.generation);} catch(const std::runtime_error&) {}});
    std::thread assertion([&](){while(!ready.load()) {};gate.assert_inhibit();});
    ready=true;release.join();assertion.join();
    assert(gate.status().active);
    gate.release(gate.status().generation);
  }
}
