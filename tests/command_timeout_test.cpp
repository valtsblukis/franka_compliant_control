// Fixed-sample watchdog checks; no robot or simulated plant integration.
#include "controller.hpp"
#include <cassert>
#include <iostream>
#include <limits>
using namespace fcc;
int main() {
  Config c;assert(c.command_timeout==2.0);c.validate();
  Sample s;s.q<<0,-.4,0,-1.9,0,1.5,0;
  for(int i=0;i<6;i++) s.jacobian(i,i)=i<3?.2:1.;
  Controller waiting(c,s,1,100.);
  assert(waiting.step(s,Command{},false,100.21,.001).phase==Phase::Armed);
  assert(waiting.step(s,Command{},false,101.99,.001).phase==Phase::Armed);
  assert(waiting.step(s,Command{},false,102.001,.001).stop_trigger==Reason::Watchdog);
  Command command;command.epoch=1;command.sequence=1;command.position=s.position;
  command.accepted_at=100.;command.expires_at=102.;
  Controller active(c,s,1,100.);
  assert(active.step(s,command,false,100.,.001).phase==Phase::Active);
  assert(active.step(s,command,false,101.99,.001).phase==Phase::Active);
  assert(active.step(s,command,false,102.001,.001).stop_trigger==Reason::Watchdog);
  Controller short_lived(c,s,1,100.);command.expires_at=100.1;
  short_lived.step(s,command,false,100.,.001);
  assert(short_lived.step(s,command,false,100.101,.001).stop_trigger==Reason::Watchdog);
  for(double timeout:{2.001,10.,1e12}) {c.command_timeout=timeout;c.validate();}
  c.command_timeout=10.;Controller longer(c,s,1,100.);command.expires_at=110.;
  assert(longer.step(s,command,false,109.99,.001).phase==Phase::Active);
  assert(longer.step(s,command,false,110.001,.001).stop_trigger==Reason::Watchdog);
  for(double invalid:{0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected=false;c.command_timeout=invalid;
    try {c.validate();} catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
  }
  std::cout<<"Two-second default and explicit shorter expiry checks passed; no robot or plant\n";
}
