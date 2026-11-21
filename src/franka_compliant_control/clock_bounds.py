"""Causal client/server steady-clock offset bounds, never a fitted trajectory shift.

Requires coherent native sample and state timestamps. A delayed response AFTER
sampling increases age; delay BEFORE sampling does not automatically stale it.
Bounded relative drift is explicit; inconsistent/nonmonotone clocks fail closed.
"""

import math


class ClockOffsetBounds:
    def __init__(self, drift_ppm=100.0, slack_s=2e-6):
        if (
            not math.isfinite(drift_ppm)
            or not 0 <= drift_ppm <= 1000
            or not math.isfinite(slack_s)
            or slack_s < 0
        ):
            raise ValueError("Invalid clock-bound assumptions")
        self.drift = drift_ppm * 1e-6
        self.slack = slack_s
        self.low = None
        self.high = None
        self.at = None

    def observe(self, request_started, response_received, server_timestamp):
        if (
            not all(
                math.isfinite(x)
                for x in (request_started, response_received, server_timestamp)
            )
            or response_received < request_started
        ):
            raise RuntimeError("Invalid request/response clocks")
        low = server_timestamp - response_received - self.slack
        high = server_timestamp - request_started + self.slack
        if self.at is not None:
            if response_received < self.at:
                raise RuntimeError("Client clock went backwards")
            margin = self.drift * (response_received - self.at)
            low = max(low, self.low - margin)
            high = min(high, self.high + margin)
            if low > high:
                raise RuntimeError("Inconsistent server/client clock bounds")
        self.low, self.high, self.at = low, high, response_received

    def age_upper(self, sample_timestamp, now):
        if (
            self.at is None
            or not math.isfinite(sample_timestamp)
            or not math.isfinite(now)
            or now < self.at
        ):
            raise RuntimeError("Uninitialized/nonmonotone observation clock")
        margin = self.drift * (now - self.at)
        age = now + self.high + margin - sample_timestamp
        if age < -self.slack:
            raise RuntimeError("Sample timestamp is in the future")
        return max(0.0, age)

    def manifest(self):
        return {
            "offset_lower_s": self.low,
            "offset_upper_s": self.high,
            "reference_client_monotonic_s": self.at,
            "relative_drift_ppm": self.drift * 1e6,
            "slack_s": self.slack,
            "method": "causal RTT interval intersections; coherent native timestamps required",
        }
