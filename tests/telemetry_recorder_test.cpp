#include "telemetry_recorder.hpp"
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>
static bool forbid_allocations=false;
void* operator new(std::size_t n) {if(forbid_allocations) std::abort();if(auto p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
int main() {
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
  static_assert(std::atomic<bool>::is_always_lock_free);
  fcc::TelemetryRecorder r;
  try {r.prepare(1);assert(false);} catch(const std::invalid_argument&) {}
  r.prepare(100);r.begin();fcc::TelemetryRecord row;
  forbid_allocations=true;
  for(int i=0;i<120;++i) {row.ids[0]=i;row.values[0]=i*.001;r.append(row);}
  r.finish();forbid_allocations=false;
  assert(r.status().count==100 && r.status().dropped==20);
  auto rows=r.copy();assert(rows.front().ids[0]==0 && rows.back().ids[0]==99);
  try {r.prepare(100);assert(false);} catch(const std::runtime_error&) {}
  try {r.clear(r.status().capture_id+1);assert(false);} catch(const std::runtime_error&) {}
  r.clear(r.status().capture_id);r.prepare(120000);r.begin();
  const auto start=std::chrono::steady_clock::now();forbid_allocations=true;
  for(int i=0;i<120000;++i){row.ids[0]=i;r.append(row);}
  r.finish();forbid_allocations=false;
  const double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/120000;
  assert(r.status().count==120000 && r.status().dropped==0);
  std::cout<<"Record bytes="<<sizeof(row)<<"; mean append ns="<<ns<<" (not a worst-case RT guarantee)\n";
  r.clear(r.status().capture_id);r.prepare(100,{.01,.02,.03});r.begin();
  assert(!r.speed_trip(.001,.001,.001));assert(r.speed_trip(.02,0,0));
  assert(r.trip()==1 && r.speed_trip(0,0,0));r.finish();
}
