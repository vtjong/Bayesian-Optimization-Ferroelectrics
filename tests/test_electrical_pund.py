"""The Round-1 slide-7 partition rule on a synthetic N-ramp current shaped like Sample1 R9C9."""

import numpy as np

from electrical.pund import partition_ramp


def synthetic_n_ramp(dt=5e-9, noise=5e-6, shoulder=True, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 1e-6, dt)
    current = -1.0e-4 * np.ones_like(t)  # dielectric ramp current (C dV/dt)
    current += -3.3e-4 * np.exp(-0.5 * ((t - 0.28e-6) / 0.12e-6) ** 2)  # main switching peak
    if shoulder:
        current += -2.0e-4 * np.exp(-0.5 * ((t - 0.82e-6) / 0.14e-6) ** 2)  # late shoulder
    return t + 9.0e-6, current + rng.normal(0, noise, len(t))


def test_the_dip_between_peak_and_shoulder_is_found():
    t, current = synthetic_n_ramp()
    q1, q2, t_split, rule, _ = partition_ramp(current, t, -1, 5e-9)
    assert rule == "dip"
    assert 9.55e-6 < t_split < 9.72e-6, t_split
    assert q1 < 0 and q2 < 0 and 0.55 < q1 / (q1 + q2) < 0.8


def test_without_a_shoulder_the_split_falls_back_to_a_fixed_fraction():
    t, current = synthetic_n_ramp(shoulder=False, noise=1e-6)
    q1, q2, t_split, rule, _ = partition_ramp(current, t, -1, 5e-9)
    assert rule == "fallback"
    assert abs(t_split - (9.0e-6 + 0.65e-6)) < 0.02e-6
