import pytest
from franka_compliant_control.clock_bounds import ClockOffsetBounds


def test_pre_snapshot_delay_is_not_confused_with_old_measurement():
    b = ClockOffsetBounds()
    b.observe(100.0, 100.004, 600.003)
    b.observe(100.020, 100.034, 600.033)
    age = b.age_upper(600.032, 100.034)
    assert 0.002 <= age < 0.006


def test_post_snapshot_delay_still_exceeds_unchanged_limit():
    b = ClockOffsetBounds()
    b.observe(100.0, 100.004, 600.003)
    b.observe(100.04, 100.06, 600.041)
    assert b.age_upper(600.040, 100.06) > 0.010


def test_clock_jump_and_future_sample_fail_closed():
    b = ClockOffsetBounds()
    b.observe(100.0, 100.004, 600.003)
    with pytest.raises(RuntimeError):
        b.observe(100.020, 100.024, 601.023)
    with pytest.raises(RuntimeError):
        b.age_upper(700.0, 100.005)


def test_causal_bound_over_many_variable_delays():
    import numpy as np

    rng = np.random.default_rng(45)
    b = ClockOffsetBounds()
    for i in range(1000):
        req = 100 + i * 0.1
        forward, back = rng.uniform(0, 0.02, 2)
        generated = req + forward
        received = generated + back
        sample = generated - 0.001
        b.observe(req, received, generated + 500)
        assert b.age_upper(sample + 500, received) >= received - sample - 1e-9
