// Compiled only by scripts/build_local_franka.py, against the patched SDK.
#include "../vendor/libfranka-0.18.0/src/control_tools.cpp"
#include <cassert>
#include <iostream>
int main() {
  assert(franka::hasRealtimeVersionFlag("#1 SMP PREEMPT_RT Tue Sep 9"));
  for(const char* version : {"", "#1 SMP PREEMPT_DYNAMIC", "#1 SMP PREEMPT",
                             "#1 SMP NOT_PREEMPT_RT", "#1 SMP PREEMPT_RT_EXTRA"}) {
    assert(!franka::hasRealtimeVersionFlag(version));
  }
  std::cout << "libfranka RT detection checks passed; running kernel recognized: "
            << franka::hasRealtimeKernel() << '\n';
}
