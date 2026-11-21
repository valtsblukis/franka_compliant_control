#pragma once
#include <atomic>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

namespace fcc {
struct MotionInhibitStatus {
  bool active;
  std::uint64_t generation;
};

// One atomic word: coherent assertion + generation, including across release.
// Assertion never waits for lifecycle/SDK work. RT only calls permitted().
class MotionInhibit {
 public:
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                "Motion inhibit requires lock-free 64-bit atomics");
  MotionInhibitStatus status() const noexcept {
    const auto s=state_.load();
    return {bool(s&1),s>>1};
  }
  MotionInhibitStatus assert_inhibit() noexcept {
    auto old=state_.load();
    for(;;) {
      const auto generation=old>>1;
      // Exhaustion stays latched permanently, never wraps to an old permit.
      const auto next=generation>=maximum?((maximum<<1)|1):(((generation+1)<<1)|1);
      if(state_.compare_exchange_weak(old,next)) return {true,next>>1};
    }
  }
  std::uint64_t permit(std::optional<std::uint64_t> expected=std::nullopt) const {
    const auto s=status();
    if(s.active) throw std::runtime_error("motion inhibited; explicit release required");
    if(expected && *expected!=s.generation)
      throw std::runtime_error("motion inhibit changed; fresh request required");
    return s.generation;
  }
  bool permitted(std::uint64_t generation) const noexcept {
    return generation<=maximum && state_.load()==(generation<<1);
  }
  void check(std::uint64_t generation) const {
    if(!permitted(generation))
      throw std::runtime_error("motion inhibited or generation changed; fresh request required");
  }
  MotionInhibitStatus release(std::uint64_t generation) {
    if(generation>=maximum) throw std::runtime_error("motion inhibit generation exhausted");
    auto expected=(generation<<1)|1;
    const auto cleared=(generation+1)<<1;
    if(!state_.compare_exchange_strong(expected,cleared))
      throw std::runtime_error("motion inhibit changed or already released");
    return {false,generation+1};
  }
 private:
  static constexpr auto maximum=std::numeric_limits<std::uint64_t>::max()>>1;
  std::atomic<std::uint64_t> state_{0};
};
} // namespace fcc
