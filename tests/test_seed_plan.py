"""Integration tests for the seed design.

These encode the properties the design is supposed to guarantee, and are the reason the previous
seed was replaced: it satisfied none of the last three.
"""

import numpy as np
import pytest
from conftest import LADDER_HI_MS, LADDER_LO_MS

from run_flash_plan import (
    BAND_HI_C,
    BAND_LO_C,
    LADDER_TIMES,
    LADDER_TMAX_LEVELS,
    N_REPLICATES,
    T_SEARCH_HI,
    T_SEARCH_LO,
    make_plan,
)

LADDER_ISO_TOL_C = 2.0  # how tightly the ladder rungs must share one peak temperature
SATURATED_X = 0.98  # above this every hypothesis agrees the film is fully crystalline
DEAD_ZONE_X = 0.02  # below this every hypothesis agrees the film is amorphous
INFORMATIVE_SPREAD = 0.05  # ensemble disagreement that still makes a saturated shot worth firing
MIN_INFORMATIVE_FRACTION = 0.5  # the replaced design managed 3 of 12 (25%)
MIN_DETECTION_POWER = 0.90  # required probability of detecting a tilt that is really there


class TestDesignInvariants:
    def test_expected_block_composition(self, seed_plan):
        blocks = seed_plan["block"]
        assert blocks.count("A") == len(LADDER_TIMES) * len(LADDER_TMAX_LEVELS)
        assert blocks.count("B") == 5
        assert blocks.count("E") == 1
        assert 1 <= blocks.count("D") <= N_REPLICATES
        assert len(blocks) == blocks.count("A") + 5 + 1 + blocks.count("D")

    def test_conditions_are_settable_on_the_tool(self, seed_plan):
        """Whole volts and 0.1 ms steps -- anything else cannot be dialled in."""
        assert np.all(seed_plan["V"] == np.round(seed_plan["V"]))
        np.testing.assert_allclose(seed_plan["t"], np.round(seed_plan["t"] * 10) / 10)

    def test_no_condition_in_the_unsupported_time_gap(self, seed_plan):
        """The table has no node below 2.6 ms, so Tmax quoted there is a spline artifact."""
        assert np.all(seed_plan["t"] >= T_SEARCH_LO - 1e-9)
        assert np.all(seed_plan["t"] <= T_SEARCH_HI + 1e-9)

    def test_all_conditions_within_the_measured_box(self, seed_plan):
        assert np.all(seed_plan["V"] >= 506) and np.all(seed_plan["V"] <= 716)


class TestLadderIsTheTiltTest:
    def test_rungs_share_one_peak_temperature(self, seed_plan):
        """Otherwise a trend across the ladder confounds dwell with peak temperature."""
        tm = np.array(
            [t for t, b in zip(seed_plan["tmax"], seed_plan["block"]) if b in ("A", "D")]
        )
        for level in LADDER_TMAX_LEVELS:
            on = tm[np.abs(tm - level) < 15.0]
            assert on.size >= len(LADDER_TIMES), f"level {level:.0f} C under-populated"
            assert on.max() - on.min() < LADDER_ISO_TOL_C

    def test_rungs_span_the_full_supported_time_range(self, seed_plan):
        times = sorted({t for t, b in zip(seed_plan["t"], seed_plan["block"]) if b == "A"})
        assert times[0] == pytest.approx(LADDER_LO_MS)
        assert times[-1] == pytest.approx(LADDER_HI_MS)

    def test_ladder_separates_the_hypotheses(self, seed_plan, ensemble):
        """Flat under iso-Tmax, large swing under any width-tracking cooling law."""
        sel = [i for i, b in enumerate(seed_plan["block"]) if b == "A"]
        v = seed_plan["V"][sel]
        t = seed_plan["t"][sel]
        tm = seed_plan["tmax"][sel]
        # Read ALONG one iso-Tmax level; mixing levels would confound tilt with peak temperature.
        on = np.abs(tm - max(LADDER_TMAX_LEVELS)) < 15.0
        v, t = v[on], t[on]
        lo, hi = np.argmin(t), np.argmax(t)
        swings = {k: m.fraction(v, t)[hi] - m.fraction(v, t)[lo] for k, m in ensemble.items()}
        assert abs(swings["isoT"]) < 0.05, "the null hypothesis must stay flat along a level"
        assert swings["diffusion"] > swings["ramp"] > 0.0, "tilt ordering must survive the design"

    def test_replicates_land_on_ladder_rungs(self, seed_plan):
        """A replicate is only a reproducibility control if it repeats an existing condition."""
        rungs = {
            (int(v), round(float(t), 1))
            for v, t, b in zip(seed_plan["V"], seed_plan["t"], seed_plan["block"])
            if b == "A"
        }
        reps = [
            (int(v), round(float(t), 1))
            for v, t, b in zip(seed_plan["V"], seed_plan["t"], seed_plan["block"])
            if b == "D"
        ]
        assert reps and all(r in rungs for r in reps)


class TestBudgetIsNotWasted:
    def test_core_points_lie_in_the_informative_band(self, seed_plan):
        tm = [t for t, b in zip(seed_plan["tmax"], seed_plan["block"]) if b != "E"]
        assert min(tm) >= BAND_LO_C - 1.0
        assert max(tm) <= BAND_HI_C + 1.0

    def test_most_shots_carry_boundary_information(self, seed_plan, ensemble):
        """A shot is informative if it is mid-transition under some hypothesis, or the hypotheses
        disagree about it. The replaced design managed only 3 of 12 by this measure."""
        mean_x = np.mean([m.fraction(seed_plan["V"], seed_plan["t"]) for m in ensemble.values()], 0)
        spread = seed_plan["disagree"]
        informative = ((mean_x > DEAD_ZONE_X) & (mean_x < SATURATED_X)) | (
            spread > INFORMATIVE_SPREAD
        )
        assert informative.mean() > MIN_INFORMATIVE_FRACTION

    def test_exactly_one_deliberate_dead_zone_shot(self, seed_plan, ensemble):
        """The floor anchor is meant to be uninformative about the boundary -- it pins the readout
        floor. Any OTHER shot landing that cold is waste, so this bounds the accidental ones."""
        mean_x = np.mean([m.fraction(seed_plan["V"], seed_plan["t"]) for m in ensemble.values()], 0)
        blocks = np.array(seed_plan["block"])
        assert np.all(mean_x[blocks == "E"] < DEAD_ZONE_X), "the floor anchor should be cold"
        accidental = int(np.sum((mean_x < DEAD_ZONE_X) & (blocks != "E")))
        assert accidental <= 2, f"{accidental} core shots landed in the dead zone"

    def test_core_block_alone_carries_information(self, seed_plan, ensemble):
        """Scored over block B ONLY.

        The all-shots version passes on the strength of the six ladder shots, which are
        informative by construction, so it cannot detect a wasteful core.
        """
        idx = [i for i, b in enumerate(seed_plan["block"]) if b == "B"]
        v, t = seed_plan["V"][idx], seed_plan["t"][idx]
        mean_x = np.mean([m.fraction(v, t) for m in ensemble.values()], 0)
        spread = seed_plan["disagree"][idx]
        informative = ((mean_x > DEAD_ZONE_X) & (mean_x < SATURATED_X)) | (
            spread > INFORMATIVE_SPREAD
        )
        assert informative.mean() >= 0.5, (
            f"only {informative.sum()}/{len(idx)} core shots carry boundary information"
        )

    def test_design_is_stratified_not_clustered(self, seed_plan):
        """No two distinct conditions may collide; replicates are the only repeats."""
        pairs = list(zip(seed_plan["V"].astype(int), np.round(seed_plan["t"], 1)))
        non_replicate = [p for p, b in zip(pairs, seed_plan["block"]) if b != "D"]
        assert len(set(non_replicate)) == len(non_replicate)

    def test_reproducible_for_a_fixed_seed(self):
        a, b = make_plan(n_core=5, seed=7), make_plan(n_core=5, seed=7)
        np.testing.assert_array_equal(a["V"], b["V"])
        np.testing.assert_array_equal(a["t"], b["t"])


class TestSeedPower:
    """The acceptance test: the ladder must be able to answer the question it was bought for."""


    def test_ladder_includes_the_replicates(self, seed_power):
        v, _ = seed_power["conditions"]
        assert len(v) >= len(LADDER_TIMES) * len(LADDER_TMAX_LEVELS)

    def test_detects_a_tilt_when_one_exists(self, seed_power):
        keys, conf = seed_power["keys"], seed_power["confusion"]
        tilted = [i for i, k in enumerate(keys) if k != "isoT"]
        detected = np.mean([sum(conf[i, j] for j in tilted) / seed_power["trials"] for i in tilted])
        assert detected > MIN_DETECTION_POWER

    def test_confirms_a_flat_boundary_when_there_is_no_tilt(self, seed_power):
        keys, conf = seed_power["keys"], seed_power["confusion"]
        i = keys.index("isoT")
        assert conf[i, i] / seed_power["trials"] > MIN_DETECTION_POWER

    def test_never_confuses_zero_tilt_with_large_tilt(self, seed_power):
        """Mistaking isoT for diffusion (or vice versa) would invert the campaign's conclusion."""
        keys, conf = seed_power["keys"], seed_power["confusion"]
        i, j = keys.index("isoT"), keys.index("diffusion")
        # A Monte-Carlo count of exactly zero is not a testable claim -- with n trials the rule of
        # three only supports p < 3/n. Assert a rate, not an exact zero.
        assert conf[i, j] / seed_power["trials"] < 0.005
        assert conf[j, i] / seed_power["trials"] < 0.005
