"""Integration tests for the seed design.

These encode the properties the design is supposed to guarantee, and are the reason the previous
seed was replaced: it satisfied none of the last three.
"""

import numpy as np
import pytest

from campaign.plan import (
    N_DRAWN,
    N_REPLICATES,
    SEED_SIZE,
    T_SEARCH_HI,
    T_SEARCH_LO,
    make_plan,
)
from physics.constants import T_TRANSITION_REF_C

LADDER_ISO_TOL_C = 2.0  # how tightly the ladder rungs must share one peak temperature
SATURATED_X = 0.98  # above this every hypothesis agrees the film is fully crystalline
DEAD_ZONE_X = 0.02  # below this every hypothesis agrees the film is amorphous
INFORMATIVE_SPREAD = 0.05  # ensemble disagreement that still makes a saturated shot worth firing
MIN_INFORMATIVE_FRACTION = 0.5  # the replaced design managed 3 of 12 (25%)
MIN_DETECTION_POWER = 0.90  # required probability of detecting a tilt that is really there


class TestDesignInvariants:
    def test_expected_block_composition(self, seed_plan):
        blocks = seed_plan["block"]
        assert blocks.count("L") == N_DRAWN, "wrong number of drawn conditions"
        assert blocks.count("R") == N_REPLICATES, "wrong number of replicates"
        assert len(blocks) == SEED_SIZE
        assert set(blocks) <= {"L", "R"}, "an unknown block appeared"

    def test_conditions_are_settable_on_the_tool(self, seed_plan):
        v, t = seed_plan["V"], seed_plan["t"]
        assert np.all(v == np.round(v)), "voltages must be whole volts"
        assert np.allclose(t, np.round(t * 10) / 10), "times must land on 0.1 ms"

    def test_no_condition_in_the_unsupported_time_gap(self, seed_plan):
        """The table has no node below 2.6 ms; anything quoted there is spline artifact."""
        assert seed_plan["t"].min() >= T_SEARCH_LO - 1e-9
        assert seed_plan["t"].max() <= T_SEARCH_HI + 1e-9

    def test_all_conditions_within_the_measured_box(self, seed_plan):
        assert np.all(seed_plan["V"] >= 506) and np.all(seed_plan["V"] <= 716)

    def test_drawn_conditions_are_distinct(self, seed_plan):
        """Snapping to whole volts can collapse two draws; replicates are the only repeats."""
        pairs = list(zip(seed_plan["V"].astype(int), np.round(seed_plan["t"], 1)))
        drawn = [p for p, b in zip(pairs, seed_plan["block"]) if b == "L"]
        assert len(set(drawn)) == len(drawn)

    def test_replicates_match_a_drawn_condition(self, seed_plan):
        """Vacuous while N_REPLICATES is 0, and kept so it bites the moment any are added."""
        pairs = list(zip(seed_plan["V"].astype(int), np.round(seed_plan["t"], 1)))
        drawn = {p for p, b in zip(pairs, seed_plan["block"]) if b == "L"}
        reps = [p for p, b in zip(pairs, seed_plan["block"]) if b == "R"]
        assert len(reps) == N_REPLICATES
        assert all(p in drawn for p in reps)

    def test_every_condition_is_distinct_when_nothing_is_replicated(self, seed_plan):
        """With no replicates, a repeated condition can only be a snapping collision."""
        if N_REPLICATES:
            pytest.skip("replicates are deliberate repeats")
        pairs = list(zip(seed_plan["V"].astype(int), np.round(seed_plan["t"], 1)))
        assert len(set(pairs)) == len(pairs)

    def test_reproducible_for_a_fixed_seed(self):
        a, b = make_plan(n_core=N_DRAWN, seed=7), make_plan(n_core=N_DRAWN, seed=7)
        np.testing.assert_array_equal(a["V"], b["V"])
        np.testing.assert_array_equal(a["t"], b["t"])


class TestTheDesignDoesNotBetOnTheBracket:
    """The property the whole design rests on: coverage that does not presume a location."""

    def test_covers_a_wide_span_of_peak_temperature(self, seed_plan):
        tm = seed_plan["tmax"]
        assert tm.max() - tm.min() > 120.0, "the draw has collapsed into a narrow band"

    def test_brackets_the_measured_transition_on_both_sides(self, seed_plan):
        """A boundary is located by straddling it, not by landing on it."""
        tm = seed_plan["tmax"]
        assert (tm < T_TRANSITION_REF_C).sum() >= 3, "nothing below the bracket"
        assert (tm > T_TRANSITION_REF_C).sum() >= 3, "nothing above the bracket"

    def test_still_brackets_a_transition_well_away_from_the_bracket(self, seed_plan):
        """The point of not betting: it must straddle a transition the bracket did not predict."""
        tm = seed_plan["tmax"]
        for hypothetical in (400.0, 430.0, 470.0, 500.0):
            assert (tm < hypothetical).any() and (tm > hypothetical).any(), (
                f"no bracketing if the transition is really at {hypothetical:.0f} C"
            )

    def test_replicates_are_spread_not_stacked(self, seed_plan):
        """Replicates at the same temperature ask the same question twice."""
        tm = np.array([t for t, b in zip(seed_plan["tmax"], seed_plan["block"]) if b == "R"])
        if len(tm) < 2:
            pytest.skip("fewer than two replicates")
        assert abs(tm[0] - tm[1]) > 20.0, "replicates landed in the same regime"
