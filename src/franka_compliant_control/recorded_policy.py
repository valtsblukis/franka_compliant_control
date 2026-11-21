"""Immutable goal annotations for accepted-command replay (not a new controller)."""
from copy import deepcopy

import numpy as np

from .twin_experiment import digest


def recorded_policy_spec(manifest):
    return {'schema':'fcc-recorded-goals-v1',
            **{k:deepcopy(manifest[k]) for k in ('events','duration_s','policy_config','policy_code_sha256')}}


class RecordedGoalSchedule:
    """Only annotates replay; actual references come from accepted actions."""
    def __init__(self, spec):
        self.spec=deepcopy(spec)
        self.events=self.spec['events'];self.duration_s=float(self.spec['duration_s'])
        times=np.array([e['time_s'] for e in self.events])
        if not len(times) or times[0]!=0 or np.any(np.diff(times)<=0) or times[-1]>self.duration_s:
            raise ValueError('Invalid recorded goal schedule')
        self.start_joints=None
        self.targets={e['name']:e['position'] for e in self.events}

    def manifest(self):
        return deepcopy(self.spec)

    def sample(self, now):
        index=max(i for i,e in enumerate(self.events) if e['time_s']<=max(0,now))
        e=self.events[index]
        return {**deepcopy(e),'goal':e['name'],'goal_position':e['position'],
                'leg':index,'policy_phase':'recorded_goal_annotation_only'}


def accepted_events(actions, origin, *, action_key):
    events=[]
    last=-np.inf
    for a in actions:
        r=a['response'];t=float(r['accepted_monotonic_s'])-origin
        deadline=float(r['deadline_s'])-origin
        if not np.isfinite([t,deadline]).all() or t<=last or deadline<=t:
            raise ValueError('Invalid recorded acceptance/deadline sequence')
        target=a[action_key]
        events.append({'time_s':t,'deadline_s':deadline,
                       'position':target['position'],'quaternion_xyzw':target['quaternion_xyzw']})
        last=t
    if not events or events[0]['time_s']>0:
        raise ValueError('Replay must include the startup command at/before origin')
    return events
