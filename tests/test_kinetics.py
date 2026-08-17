"""Tests for the boundary-model ensemble: anchoring, the closed-form tilt, and numerical stability.

The scientific claim under test is that the four hypotheses differ ONLY in their cooling law. If
they drift apart at the calibration anchor, any comparison between them silently confounds the
cooling law with the onset temperature.
"""

import numpy as np
import pytest
from conftest import LADDER_HI_MS, LADDER_LO_MS

from discovery.constants import (
    AVRAMI_N,
    CELSIUS_TO_KELVIN,
    EA_EV,
    KB_EV,
    T_ONSET_C,
    T_REF_MS,
    T_ROOM_C,
)
from discovery.kinetics import (
    KineticBoundary,
    disagreement,
    logistic_sharpness,
    theta_kelvin,
)
from discovery.synthetic import SHAPES, thermal_model

ANCHOR_TOL_X = 5e-3  # |X - 1/2| at the calibration anchor
TILT_TOL_C = 2.0  # agreement with the closed-form tilt law
QUADRATURE_TOL_C = 0.5  # boundary shift permitted under a much finer integration grid


@pytest.fixture(scope="module")
def anchor_voltage() -> float:
    """Voltage reaching the onset temperature at the reference flash time."""
    return thermal_model(SHAPES["isoT"]).voltage_for_tmax(T_ONSET_C, T_REF_MS)


class TestAnchoring:
    def test_all_models_cross_half_at_the_anchor(self, ensemble, anchor_voltage):
        for name, model in ensemble.items():
            x = float(
                np.atleast_1d(model.fraction(np.array([anchor_voltage]), np.array([T_REF_MS])))[0]
            )
            assert abs(x - 0.5) < ANCHOR_TOL_X, f"{name} drifted off the anchor: X = {x:.4f}"

    def test_fractions_are_bounded(self, ensemble):
        v = np.linspace(506.0, 716.0, 40)
        t = np.full_like(v, 5.1)
        for name, model in ensemble.items():
            x = model.fraction(v, t)
            assert np.all(x >= 0.0) and np.all(x <= 1.0), name

    def test_fraction_grid_matches_scattered_evaluation(self, ensemble):
        v = np.array([560.0, 620.0, 680.0])
        t = np.array([2.6, 7.6])
        for name, model in ensemble.items():
            grid = model.fraction_grid(v, t)
            assert grid.shape == (t.size, v.size), name
            for i, t_val in enumerate(t):
                expected = model.fraction(v, np.full_like(v, t_val))
                np.testing.assert_allclose(grid[i], expected, atol=1e-9, err_msg=name)

    def test_fraction_increases_with_voltage(self, ensemble):
        """Crystallinity is monotone in thermal severity at fixed pulse width."""
        v = np.linspace(506.0, 716.0, 60)
        for name, model in ensemble.items():
            x = model.fraction(v, np.full_like(v, 5.1))
            assert np.all(np.diff(x) >= -1e-9), name


class TestTilt:
    def test_iso_tmax_has_exactly_zero_tilt(self, ensemble):
        """Measure the tilt from the MODEL, not from a hardcoded return value.

        ``tilt_c`` used to return a literal 0.0, so this assertion could not fail; mutating the
        model to carry a real 30 C tilt left it green. Bisecting X = 1/2 out of ``fraction`` at
        both pulse widths tests the claim rather than the constant.
        """
        model = ensemble["isoT"]

        def boundary(t):
            lo, hi = 100.0, 1200.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                v = model.thermal.voltage_for_tmax(mid, t)
                inside = np.isfinite(v) and float(model.fraction(v, t)[0]) >= 0.5
                lo, hi = (lo, mid) if inside else (mid, hi)
            return 0.5 * (lo + hi)

        assert abs(boundary(LADDER_LO_MS) - boundary(LADDER_HI_MS)) < 1e-6

    def test_diffusion_matches_the_closed_form(self, ensemble):
        """t_eff proportional to t implies tilt = theta * ln(t2 / t1) with no free parameters."""
        predicted = theta_kelvin(T_ONSET_C) * np.log(LADDER_HI_MS / LADDER_LO_MS)
        measured = ensemble["diffusion"].tilt_c(LADDER_LO_MS, LADDER_HI_MS)
        assert abs(measured - predicted) < TILT_TOL_C

    def test_ensemble_is_ordered_by_tilt(self, ensemble):
        tilts = {k: m.tilt_c(LADDER_LO_MS, LADDER_HI_MS) for k, m in ensemble.items()}
        assert tilts["isoT"] < tilts["ramp"] < tilts["diffusion"]
        assert abs(tilts["diffusion"] - tilts["rect"]) < TILT_TOL_C

    def test_tilt_lowers_the_boundary_for_longer_pulses(self, ensemble):
        """More dwell buys crystallization at a lower peak temperature."""
        for name in ("ramp", "diffusion", "rect"):
            model = ensemble[name]
            assert model.boundary_tmax(LADDER_HI_MS) < model.boundary_tmax(LADDER_LO_MS), name


class TestSharpness:
    def test_derived_from_activation_parameters(self):
        """The transition width is a consequence of (Ea, n), not an independent constant."""
        sharp = logistic_sharpness(T_ONSET_C, EA_EV, AVRAMI_N)
        assert sharp == pytest.approx(2 * np.log(2.0) * AVRAMI_N / theta_kelvin(T_ONSET_C))
        assert 0.05 < sharp < 0.30

    def test_higher_barrier_sharpens_the_transition(self):
        assert logistic_sharpness(T_ONSET_C, 2.0, AVRAMI_N) > logistic_sharpness(
            T_ONSET_C, 1.0, AVRAMI_N
        )


class TestDisagreement:
    def test_zero_where_every_hypothesis_agrees(self, ensemble):
        """Deep in either phase the models coincide, so a shot there settles nothing."""
        cold = disagreement(ensemble, np.array([506.0]), np.array([5.1]))
        assert cold[0] < 1e-3

    def test_large_on_the_iso_tmax_ladder(self, ensemble):
        """The short-pulse rung is where the hypotheses separate most."""
        v = thermal_model(SHAPES["isoT"]).voltage_for_tmax(385.0, LADDER_LO_MS)
        spread = disagreement(ensemble, np.array([v]), np.array([LADDER_LO_MS]))
        assert spread[0] > 0.3


class TestNumericalStability:
    def test_boundary_is_stable_under_finer_quadrature(self):
        """The NUMERICAL constants are tuning knobs; results must not depend on them."""
        model = KineticBoundary(thermal_model(SHAPES["diffusion"]), ea_ev=EA_EV, n=AVRAMI_N)
        coarse = model.boundary_tmax(LADDER_LO_MS)

        shape = SHAPES["diffusion"]
        tau = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, 3.0 * LADDER_LO_MS, 400_000),
                    np.geomspace(3.0 * LADDER_LO_MS, shape.duration_ms, 40_000),
                ]
            )
        )
        target = (-np.log(0.5)) ** (1.0 / AVRAMI_N) / model.nu

        def budget(tmax_c: float) -> float:
            t_k = T_ROOM_C + (tmax_c - T_ROOM_C) * shape(tau, LADDER_LO_MS) + CELSIUS_TO_KELVIN
            return float(np.trapezoid(np.exp(-EA_EV / (KB_EV * t_k)), tau))

        lo, hi = 100.0, 1200.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if budget(mid) < target:
                lo = mid
            else:
                hi = mid
        assert abs(0.5 * (lo + hi) - coarse) < QUADRATURE_TOL_C
