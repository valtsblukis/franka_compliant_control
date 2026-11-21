#pragma once
// Single RT writer, bounded preallocated storage. Export/reset only after join.
// No Python, allocation, locks, files, network or overwrite in append().
#include <array>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fcc {
constexpr std::size_t kTelemetryValues=117;
struct TelemetryRecord {
  std::array<std::uint64_t,7> ids{}; // cycle, epoch, applied sequence, mode, phase, reason, flags
  std::array<double,kTelemetryValues> values{};
};
struct TelemetryStatus {
  std::uint64_t capture_id=0, count=0, dropped=0, capacity=0;
  bool prepared=false, active=false;
  int speed_guard_trip=0;
};
class TelemetryRecorder {
 public:
  void prepare(std::size_t capacity, std::array<double,3> speed_limits={0,0,0}) {
    if(prepared_ || active_.load()) throw std::runtime_error("Export/clear previous telemetry first");
    if(capacity<100 || capacity>120000) throw std::invalid_argument("Telemetry capacity must be 100..120000 callbacks");
    records_.resize(capacity); // allocate AND touch pages before starting control
    for(auto& r:records_) r=TelemetryRecord{};
    for(double x:speed_limits) if(!std::isfinite(x) || x<0) throw std::invalid_argument("Invalid native experiment speed limit");
    limits_=speed_limits;trip_=0;
    count_=0;dropped_=0;++id_;prepared_=true;used_=false;
  }
  void begin() {
    if(prepared_ && !used_) {used_=true;active_.store(true,std::memory_order_release);}
  }
  bool active() const noexcept {return active_.load(std::memory_order_relaxed);}
  bool used() const noexcept {return used_;}
  bool speed_trip(double joint,double linear,double angular) noexcept {
    if(!active()) return false;
    if(trip_.load(std::memory_order_relaxed)) return true;
    const std::array<double,3> values{joint,linear,angular};
    for(int i=0;i<3;++i) if(limits_[i]>0 && values[i]>limits_[i]) {trip_.store(i+1,std::memory_order_relaxed);return true;}
    return false;
  }
  int trip() const noexcept {return trip_.load(std::memory_order_relaxed);}
  void append(const TelemetryRecord& row) noexcept {
    if(!active()) return;
    const auto n=count_.load(std::memory_order_relaxed);
    if(n>=records_.size()) {dropped_.fetch_add(1,std::memory_order_relaxed);return;}
    records_[n]=row;
    count_.store(n+1,std::memory_order_release);
  }
  void finish() noexcept {active_.store(false,std::memory_order_release);}
  TelemetryStatus status() const {
    return {id_,count_.load(std::memory_order_acquire),dropped_.load(),records_.size(),prepared_,active(),trip()};
  }
  std::vector<TelemetryRecord> copy() const {
    if(active()) throw std::runtime_error("Telemetry export requires inactive control");
    return {records_.begin(),records_.begin()+count_.load()};
  }
  void clear(std::uint64_t id) {
    if(active() || id!=id_) throw std::runtime_error("Inactive matching telemetry capture required");
    records_.clear();records_.shrink_to_fit();prepared_=false;used_=false;count_=0;dropped_=0;
  }
 private:
  std::vector<TelemetryRecord> records_;
  std::atomic<std::uint64_t> count_{0},dropped_{0};
  std::atomic<bool> active_{false};
  std::uint64_t id_=0;
  bool prepared_=false,used_=false;
  std::array<double,3> limits_{};
  std::atomic<int> trip_{0};
};
} // namespace fcc
