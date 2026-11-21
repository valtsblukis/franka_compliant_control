"""Offline/publisher-side straight-line minimum-jerk pacing, not a motion planner."""
import math


def demo_pacing(feedback_mode):
    """Requested demo limits, separate from measured-motion trips.

    Bounded-equilibrium profiles use half FR3's published translation v/a/j.
    The linear OSC demo keeps the operator's existing 0.5/0.5 pacing.
    """
    if feedback_mode == "linear":
        return .5, .5, 2.0
    if feedback_mode in ("progressive", "saturated"):
        return 1.5, 4.5, 2250.0
    raise ValueError("unknown OSC feedback mode")


def segment_duration(start, end, velocity, acceleration, jerk):
    if len(start) != 3 or len(end) != 3 or not all(math.isfinite(v) for v in (*start, *end)):
        raise ValueError("segment endpoints must be finite 3-vectors")
    if any(not math.isfinite(v) or v <= 0 for v in (velocity, acceleration, jerk)):
        raise ValueError("segment v/a/j must be finite and positive")
    distance = math.dist(start, end)
    # Exact derivative maxima of h(u)=10u^3-15u^4+6u^5, u in [0,1].
    duration = max(1.875*distance/velocity,
                   math.sqrt((10/math.sqrt(3))*distance/acceleration),
                   math.cbrt(60*distance/jerk), .001)
    if not math.isfinite(duration):
        raise ValueError("segment duration is not finite")
    return duration*(1+1e-9)  # Avoid boundary rounding above a reference limit.


def segment_sample(start, end, elapsed, duration):
    if len(start) != 3 or len(end) != 3 or not all(math.isfinite(v) for v in (*start, *end)):
        raise ValueError("segment endpoints must be finite 3-vectors")
    if not math.isfinite(elapsed) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("segment timing must be finite with positive duration")
    u = min(1., max(0., elapsed/duration))
    h = u*u*u*(10+u*(-15+6*u))
    dh = 30*u*u*(1-u)*(1-u)/duration
    ddh = 60*u*(1-u)*(1-2*u)/(duration*duration)
    delta = [b-a for a, b in zip(start, end)]
    return ([a+d*h for a, d in zip(start, delta)],
            [d*dh for d in delta]+[0.0]*3,
            [d*ddh for d in delta]+[0.0]*3)
