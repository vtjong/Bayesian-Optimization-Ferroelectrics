"""Tests for the measured peak-temperature table and the candidate cooling laws."""

import numpy as np
import pytest

from discovery.constants import FLASH_TABLE_CSV, T_REF_MS
from discovery.synthetic import (
    FLASH,
    FLASH_T,
    FLASH_TMAX,
    FLASH_V,
    SHAPES,
    load_flash_table,
    tmax,
)

NODE_TOL_C = 1e-6  # spline must interpolate a measured node to well below readout precision
PEAK_TOL = 1e-3  # |max(g) - 1| for a shape normalized to a unit peak
PLATEAU_CUTOFF = 0.10  # separates "decays toward room" from "holds a permanent plateau"


class TestMeasuredTable:
    """The table is loaded from CSV, so these guard the parse and the interpolant."""

    def test_axes_ascending_and_shape_matches(self):
        assert np.all(np.diff(FLASH_V) > 0)
        assert np.all(np.diff(FLASH_T) > 0)
        assert FLASH_TMAX.shape == (FLASH_T.size, FLASH_V.size)

    def test_csv_is_the_only_source_of_truth(self):
        """Re-reading the file reproduces the module-level arrays exactly."""
        voltages, times, table = load_flash_table(FLASH_TABLE_CSV)
        np.testing.assert_array_equal(voltages, FLASH_V)
        np.testing.assert_array_equal(times, FLASH_T)
        np.testing.assert_array_equal(table, FLASH_TMAX)

    def test_parsed_values_match_the_file_literally(self):
        """Anchor the parse to known literals.

        Comparing the parser's output to arrays the parser itself produced only catches
        non-determinism: mutating the parser to return ``table * 1.15`` leaves every other table
        test green. These literals are read off data/flash_temp_table.csv by eye.
        """
        assert FLASH_TMAX.shape == (5, 6)
        assert FLASH_V[0] == 506.0 and FLASH_V[-1] == 716.0
        assert FLASH_T[0] == 0.1 and FLASH_T[-1] == 10.1
        assert FLASH_TMAX[0, 0] == 81.6
        assert FLASH_TMAX[1, -1] == 563.335
        assert FLASH_TMAX[-1, -1] == 421.551

    def test_parser_round_trips_a_synthetic_table(self, tmp_path):
        """A file with known contents parses back to exactly those numbers."""
        f = tmp_path / "tiny.csv"
        f.write_text(",V=100,V=200\nt=1.0,10.5,20.5\nt=2.0,30.5,40.5\n")
        voltages, times, table = load_flash_table(f)
        np.testing.assert_array_equal(voltages, [100.0, 200.0])
        np.testing.assert_array_equal(times, [1.0, 2.0])
        np.testing.assert_array_equal(table, [[10.5, 20.5], [30.5, 40.5]])

    def test_missing_table_raises_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_flash_table(tmp_path / "absent.csv")

    def test_spline_interpolates_every_node(self):
        for j, t in enumerate(FLASH_T):
            for i, v in enumerate(FLASH_V):
                assert abs(float(tmax(v, t)) - FLASH_TMAX[j, i]) < NODE_TOL_C

    def test_tmax_is_reentrant_in_time(self):
        """Peak temperature rises then falls with pulse width, so a Tmax level set folds."""
        hottest = int(np.argmax(FLASH_TMAX.max(axis=1)))
        assert 0 < hottest < len(FLASH_T) - 1

    def test_voltage_inversion_round_trips(self):
        for t in (2.6, 5.1, 7.6, 10.1):
            target = 0.5 * sum(FLASH.tmax_range(t))
            v = FLASH.voltage_for_tmax(target, t)
            assert np.isfinite(v)
            assert abs(float(FLASH.tmax(v, t)) - target) < 1e-6

    def test_voltage_inversion_reports_unreachable(self):
        assert not np.isfinite(FLASH.voltage_for_tmax(2000.0, 5.1))


class TestPulseShapes:
    """Every cooling law must be peak-normalized; only the legacy ones keep a plateau."""

    @pytest.mark.parametrize("name", sorted(SHAPES))
    def test_normalized_to_unit_peak(self, name):
        shape = SHAPES[name]
        tau = np.linspace(0.0, shape.duration_ms, 200_000)
        assert abs(shape(tau, T_REF_MS).max() - 1.0) < PEAK_TOL

    @pytest.mark.parametrize("name", sorted(SHAPES))
    def test_bounded_and_causal(self, name):
        shape = SHAPES[name]
        tau = np.linspace(-5.0, shape.duration_ms, 20_000)
        g = shape(tau, T_REF_MS)
        assert np.all(g >= 0.0) and np.all(g <= 1.0 + PEAK_TOL)
        assert np.all(g[tau < 0] == 0.0), "no heating before the pulse"

    def test_diffusion_returns_toward_room_temperature(self):
        shape = SHAPES["diffusion"]
        assert float(shape(np.array([shape.duration_ms]), T_REF_MS)[0]) < PLATEAU_CUTOFF

    def test_legacy_shape_keeps_its_unphysical_plateau(self):
        """Documents the known-wrong boundary condition retained for reproducibility."""
        shape = SHAPES["isoT"]
        assert float(shape(np.array([shape.duration_ms]), T_REF_MS)[0]) > PLATEAU_CUTOFF

    def test_only_the_legacy_shape_ignores_pulse_width(self):
        """The property that decides whether the boundary can carry a tilt at all."""
        for name, shape in SHAPES.items():
            tau = np.linspace(0.0, shape.duration_ms, 8000)
            differs = not np.allclose(shape(tau, 2.6), shape(tau, 10.1))
            assert differs == (name != "isoT"), name

    def test_diffusion_is_self_similar_in_tau_over_t(self):
        """Conduction has no intrinsic timescale: g depends on tau only through tau / t_pulse."""
        shape = SHAPES["diffusion"]
        u = np.linspace(0.0, 20.0, 5000)
        np.testing.assert_allclose(shape(u * 2.6, 2.6), shape(u * 10.1, 10.1), atol=1e-12)


class TestTraceAssembly:
    """The thermal model must actually route the commanded pulse width into the trace."""

    def test_trace_peaks_at_the_table_value(self):
        _, temp = FLASH.trace(674.0, 5.1)
        assert abs(temp.max() - float(tmax(674.0, 5.1))) < 1.0

    def test_trace_depends_on_pulse_width(self):
        _, short = FLASH.trace(674.0, 2.6)
        _, long = FLASH.trace(674.0, 10.1)
        assert not np.allclose(short, long)
