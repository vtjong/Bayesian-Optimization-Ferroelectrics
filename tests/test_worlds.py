"""The randomized worlds must respect the one thing that is measured, and nothing else.

If a world is allowed to disagree with the 30 measured peak temperatures, every design that
navigates in thermal coordinates is penalised for an error that does not exist, and the comparison
silently favours designs that ignore the table. If a world is NOT allowed to disagree between the
nodes, the comparison silently favours designs that interpolate confidently. Both failure modes are
easy to introduce and invisible in the summary numbers, so they are asserted here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from physics.thermal_model import FLASH, FLASH_T, FLASH_V  # noqa: E402
from validation.worlds import INTERP_SIGMAS, sample_world  # noqa: E402

N_WORLDS = 40
# The GP's posterior sd at a measured node, times the licence a world is given, plus room for the
# interpolator's own grid resolution.
NODE_TOLERANCE_C = 0.5
BETWEEN_ROWS = ((632.0, 3.8), (653.0, 3.8), (674.0, 6.4))


@pytest.fixture(scope="module")
def worlds():
    rng = np.random.default_rng(0)
    return [sample_world(rng) for _ in range(N_WORLDS)]


class TestTheMeasuredTableIsRespected:
    def test_worlds_agree_with_the_table_at_every_measured_node(self, worlds):
        """The 30 points are a direct measurement; no world may contradict them."""
        vv, tt = np.meshgrid(FLASH_V, FLASH_T)
        base = FLASH.tmax(vv, tt)
        worst = max(float(np.abs(w.true_tmax(vv, tt) - base).max()) for w in worlds)
        assert worst < NODE_TOLERANCE_C, (
            f"a world deviated {worst:.2f} C from a MEASURED node; the interpolator for the GP "
            "posterior has probably lost the node coordinates from its grid"
        )

    def test_worlds_may_disagree_between_rows(self, worlds):
        """Interpolation is genuinely uncertain; the test is worthless if that is not used."""
        v = np.array([p[0] for p in BETWEEN_ROWS])
        t = np.array([p[1] for p in BETWEEN_ROWS])
        base = FLASH.tmax(v, t)
        spread = [float(np.abs(w.true_tmax(v, t) - base).max()) for w in worlds]
        assert max(spread) > 5.0, "no world deviated between rows; the perturbation is inert"
        assert np.median(spread) > 1.0, "between-row deviation is too rare to test anything"

    def test_deviation_stays_within_the_licence_it_was_given(self, worlds):
        """The field is L1-normalized so the deviation cannot exceed INTERP_SIGMAS * sd."""
        v = np.array([p[0] for p in BETWEEN_ROWS])
        t = np.array([p[1] for p in BETWEEN_ROWS])
        base = FLASH.tmax(v, t)
        # sd between the 2.6 and 5.1 ms rows is ~16 C; allow the full licence plus a margin.
        cap = INTERP_SIGMAS * 20.0
        worst = max(float(np.abs(w.true_tmax(v, t) - base).max()) for w in worlds)
        assert worst <= cap, f"deviation {worst:.1f} C exceeds the {cap:.0f} C licence"


class TestTheRestIsFree:
    def test_the_response_is_a_fraction(self, worlds):
        v = np.linspace(506, 716, 25)
        t = np.geomspace(2.6, 10.1, 25)
        for w in worlds:
            x = w.truth(v, t)
            assert np.all((x >= 0.0) & (x <= 1.0)) and np.isfinite(x).all()

    def test_worlds_span_flat_and_strongly_tilted_boundaries(self, worlds):
        """A suite with no flat worlds cannot measure a false-positive rate, and vice versa."""
        tilts = np.array([w.tilt_k for w in worlds])
        assert (tilts <= 5.0).any(), "no effectively flat world"
        assert (tilts >= 25.0).any(), "no strongly tilted world"

    def test_the_readout_is_not_assumed_calibrated(self, worlds):
        """An unknown floor and span are the point; a design must not rely on y being a fraction."""
        v = np.full(12, 640.0)
        t = np.full(12, 5.1)
        rng = np.random.default_rng(1)
        offsets = []
        for w in worlds:
            x = w.truth(v, t)
            offsets.append(float(np.mean(w.observe(np.zeros_like(x), rng))))
        assert np.ptp(offsets) > 0.05, "every world reports the same floor; the readout is fixed"

    def test_observation_is_stochastic(self, worlds):
        v = np.full(8, 650.0)
        t = np.full(8, 5.1)
        rng = np.random.default_rng(2)
        x = worlds[0].truth(v, t)
        assert not np.allclose(worlds[0].observe(x, rng), worlds[0].observe(x, rng))
