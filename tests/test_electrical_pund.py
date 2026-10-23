"""Pulse finding and the three O / T split rules on synthetic currents shaped like the real ones."""

import numpy as np
import pytest

from electrical.pund import PulseLabel, SplitRule, segment_pulses, split_ramp

AREA_CM2 = 9.0e-6
DT_S = 5e-9


def synthetic_n_ramp(dt=DT_S, noise=5e-6, shoulder=True, seed=0):
    """A 1 us N-ramp current: dielectric step, main switching peak and an optional late shoulder."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 1e-6 + dt / 2, dt)
    current = -1.0e-4 * np.ones_like(t)
    current += -3.3e-4 * np.exp(-0.5 * ((t - 0.28e-6) / 0.12e-6) ** 2)
    if shoulder:
        current += -2.0e-4 * np.exp(-0.5 * ((t - 0.82e-6) / 0.14e-6) ** 2)
    return t + 9.0e-6, current + rng.normal(0, noise, len(t))


def trapezoid_train(signs, dt=DT_S, v_peak=3.0, rise=1e-6, plateau=1e-6, gap=1e-6, lead=0.5e-6):
    """V(t) of trapezoid pulses with the given signs (rise / plateau / fall), separated by gaps."""
    pieces = [np.zeros(int(lead / dt))]
    for sign in signs:
        ramp = np.linspace(0, v_peak, int(rise / dt), endpoint=False)
        pieces += [
            sign * ramp,
            sign * np.full(int(plateau / dt), v_peak),
            sign * ramp[::-1],
            np.zeros(int(gap / dt)),
        ]
    v = np.concatenate(pieces)
    return np.arange(len(v)) * dt, v


class TestPulseFinding:
    @pytest.mark.parametrize(
        ("signs", "labels"),
        [
            ((1, 1, -1, -1), ["P", "U", "N", "D"]),
            ((-1, -1, 1, 1), ["N", "D", "P", "U"]),
            ((-1, 1, 1, -1, -1), ["preset", "P", "U", "N", "D"]),
        ],
    )
    def test_the_expected_patterns_are_labelled(self, signs, labels):
        _, v = trapezoid_train(signs)
        assert [p.label.value for p in segment_pulses(v)] == labels

    def test_any_other_pattern_is_left_unlabelled(self):
        _, v = trapezoid_train((1, -1, 1))
        assert {p.label for p in segment_pulses(v)} == {PulseLabel.UNKNOWN}

    def test_ramp_and_plateau_are_located_on_the_waveform(self):
        # The pulse starts where |V| crosses the threshold (0.05 us up a 3 V / 1 us ramp) and the
        # ramp ends where |V| reaches 95 % of the peak, so the detected rise is 0.9 us and the
        # plateau runs between the two 95 % points; the full nominal ramp is recovered from it.
        t, v = trapezoid_train((1, 1, -1, -1))
        p = segment_pulses(v)[0]
        assert t[p.i_ramp_end] - t[p.i_start] == pytest.approx(0.90e-6, abs=2 * DT_S)
        assert t[p.i_plateau_end] - t[p.i_ramp_end] == pytest.approx(1.10e-6, abs=3 * DT_S)
        assert p.ramp.stop - p.ramp.start == pytest.approx(0.90e-6 / 0.95 / DT_S + 1, abs=2)


class TestSplitRules:
    @pytest.mark.parametrize("rule", list(SplitRule))
    def test_every_rule_puts_the_split_in_the_dip_between_peak_and_shoulder(self, rule):
        t, current = synthetic_n_ramp()
        split = split_ramp(current, t, -1, DT_S, rule, AREA_CM2)
        assert split.rule is rule and split.found
        assert 9.55e-6 < split.t_split_s < 9.72e-6, split.t_split_s
        assert split.q_main_c < 0 and split.q_shoulder_c < 0
        assert 0.55 < split.fraction_main < 0.8
        assert split.total_uc_cm2 == pytest.approx(abs(np.trapezoid(current, t)) * 1e6 / AREA_CM2)

    def test_the_rules_agree_with_each_other_within_one_smoothing_frame(self):
        t, current = synthetic_n_ramp()
        times = [split_ramp(current, t, -1, DT_S, rule, AREA_CM2).t_split_s for rule in SplitRule]
        assert max(times) - min(times) < 41 * DT_S

    @pytest.mark.parametrize("rule", list(SplitRule))
    def test_without_a_shoulder_the_split_falls_back_to_a_fixed_fraction(self, rule):
        t, current = synthetic_n_ramp(shoulder=False, noise=1e-6)
        split = split_ramp(current, t, -1, DT_S, rule, AREA_CM2)
        if split.found:  # the derivative rules can still find a zero on a smooth decay
            assert 9.5e-6 < split.t_split_s < 9.9e-6
        else:
            assert split.t_split_s == pytest.approx(9.0e-6 + 0.65e-6, abs=0.02e-6)

    def test_a_too_short_ramp_is_the_fallback(self):
        t, current = synthetic_n_ramp()
        split = split_ramp(current[:5], t[:5], -1, DT_S, SplitRule.DIP, AREA_CM2)
        assert not split.found
