"""Ingestion must refuse ambiguity rather than resolve it.

Each failure below is one that already happened in this project's archival data, so these are
regression tests against real defects, not hypotheticals.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from discovery.results import (  # noqa: E402
    ACTUAL_T_COLUMN,
    ACTUAL_V_COLUMN,
    CONTROL_INDEX,
    CROSSCHECK_COLUMN,
    PLAN_KEY,
    READOUT_COLUMN,
    READOUT_KIND_COLUMN,
    STATUS_COLUMN,
    blank_template,
    load,
    unfired,
)

A_REAL_ZERO = 0.0  # an amorphous specimen reads zero; this is data, not absence


@pytest.fixture
def plan(seed_plan):
    """The committed plan in the CSV's own column names."""
    return pd.DataFrame(
        {
            PLAN_KEY: np.arange(1, len(seed_plan["V"]) + 1),
            "block": seed_plan["block"],
            "voltage_V": seed_plan["V"],
            "time_ms": seed_plan["t"],
            "pred_Tmax_C": np.round(seed_plan["tmax"], 1),
            READOUT_KIND_COLUMN: "eps_r",
            "note": seed_plan["note"],
        }
    )


def _filled(plan, **overrides):
    """A fully measured sheet, with per-column overrides applied afterwards.

    Includes the as-deposited control row, which the template always carries and which is exempt
    from the as-fired requirement because it is never flashed.
    """
    # object dtype throughout: the sheet is a CSV, and tests must be free to write "" or "n/a"
    # into any cell exactly as an operator would.
    t = blank_template(plan).astype(object)
    t[STATUS_COLUMN] = "measured"
    t["specimen_id"] = [f"S{i:02d}" for i in range(len(t))]
    t[READOUT_COLUMN] = np.linspace(0.05, 0.95, len(t))
    for k, v in overrides.items():
        t[k] = v
    return t


def _write(tmp_path, frame):
    p = tmp_path / "results.csv"
    frame.to_csv(p, index=False)
    return p


class TestBlanksAreNeverNumbers:
    def test_unfilled_sheet_is_rejected_not_read_as_not_run(self, plan, tmp_path):
        """The defect that matters most: an empty sheet must not look like a completed campaign."""
        with pytest.raises(ValueError, match="status"):
            load(_write(tmp_path, blank_template(plan)), plan)

    def test_measured_row_without_a_value_is_an_error(self, plan, tmp_path):
        f = _filled(plan)
        f.loc[3, READOUT_COLUMN] = ""
        with pytest.raises(ValueError, match="no readout value"):
            load(_write(tmp_path, f), plan)

    def test_value_on_a_non_measured_row_is_an_error(self, plan, tmp_path):
        """A number whose status says it is not a measurement is worse than a missing one."""
        f = _filled(plan)
        f.loc[2, STATUS_COLUMN] = "not_run"
        with pytest.raises(ValueError, match="not marked"):
            load(_write(tmp_path, f), plan)

    def test_a_genuine_zero_survives_as_zero(self, plan, tmp_path):
        """The archival failure was the converse: a blank silently became 0.0."""
        f = _filled(plan)
        f.loc[f[PLAN_KEY] == 1, READOUT_COLUMN] = A_REAL_ZERO
        res = load(_write(tmp_path, f), plan)
        row = res.table[res.table[PLAN_KEY] == 1]
        assert row[READOUT_COLUMN].iloc[0] == A_REAL_ZERO
        assert res.n_measured == len(plan) + 1

    def test_non_numeric_reading_is_rejected(self, plan, tmp_path):
        f = _filled(plan)
        f.loc[1, READOUT_COLUMN] = "n/a"
        with pytest.raises(ValueError, match="non-numeric"):
            load(_write(tmp_path, f), plan)


class TestCorrespondenceWithThePlan:
    def test_missing_condition_is_rejected(self, plan, tmp_path):
        with pytest.raises(ValueError, match="1:1"):
            load(_write(tmp_path, _filled(plan).iloc[:-1]), plan)

    def test_duplicate_index_is_rejected(self, plan, tmp_path):
        f = _filled(plan)
        f.loc[5, PLAN_KEY] = f.loc[4, PLAN_KEY]
        with pytest.raises(ValueError, match="1:1|duplicate"):
            load(_write(tmp_path, f), plan)

    def test_missing_column_is_rejected(self, plan, tmp_path):
        with pytest.raises(ValueError, match="missing required columns"):
            load(_write(tmp_path, _filled(plan).drop(columns=[CROSSCHECK_COLUMN])), plan)

    def test_readout_kind_may_not_differ_from_the_plan(self, plan, tmp_path):
        """The noise model and the boundary anchor are specific to one readout."""
        f = _filled(plan)
        f.loc[f[PLAN_KEY] == 1, READOUT_KIND_COLUMN] = "two_pr"
        with pytest.raises(ValueError, match="readout kind"):
            load(_write(tmp_path, f), plan)


class TestAsFiredConditions:
    def test_models_see_the_as_fired_condition_not_the_planned_one(self, plan, tmp_path):
        f = _filled(plan)
        f.loc[f[PLAN_KEY] == 1, ACTUAL_V_COLUMN] = float(plan.loc[0, "voltage_V"]) + 13.0
        res = load(_write(tmp_path, f), plan)
        d = res.table[res.table[PLAN_KEY] == 1]
        assert d[ACTUAL_V_COLUMN].iloc[0] == pytest.approx(float(plan.loc[0, "voltage_V"]) + 13.0)

    def test_drift_beyond_tolerance_is_reported(self, plan, tmp_path):
        f = _filled(plan)
        f.loc[f[PLAN_KEY] == 1, ACTUAL_V_COLUMN] = float(plan.loc[0, "voltage_V"]) + 13.0
        res = load(_write(tmp_path, f), plan)
        assert list(res.drift[PLAN_KEY]) == [1]

    def test_no_drift_reported_when_the_tool_delivered_the_plan(self, plan, tmp_path):
        assert load(_write(tmp_path, _filled(plan)), plan).drift.empty

    def test_as_fired_condition_required_on_measured_rows(self, plan, tmp_path):
        f = _filled(plan)
        f.loc[f[PLAN_KEY] != CONTROL_INDEX, ACTUAL_T_COLUMN] = ""
        with pytest.raises(ValueError, match="missing or non-numeric"):
            load(_write(tmp_path, f), plan)

    def test_the_as_deposited_control_needs_no_condition(self, plan, tmp_path):
        """It is never flashed, so demanding a voltage would make the sheet unfillable."""
        res = load(_write(tmp_path, _filled(plan)), plan)
        row = res.table[res.table[PLAN_KEY] == CONTROL_INDEX]
        assert len(row) == 1, "the control row must survive the join"
        assert row["block"].iloc[0] == "R"
        assert np.isfinite(row[READOUT_COLUMN].iloc[0]), "the control must carry a reading"

    def test_control_readout_kind_is_checked_too(self, plan, tmp_path):
        """It is read on the same instrument, so a mismatch there is the same error."""
        f = _filled(plan)
        f.loc[f[PLAN_KEY] == CONTROL_INDEX, READOUT_KIND_COLUMN] = "two_pr"
        with pytest.raises(ValueError, match="readout kind"):
            load(_write(tmp_path, f), plan)

    def test_control_is_not_reported_as_drift(self, plan, tmp_path):
        """Its blank condition must not read as a delivery error."""
        res = load(_write(tmp_path, _filled(plan)), plan)
        assert CONTROL_INDEX not in list(res.drift[PLAN_KEY])


class TestPartialBatches:
    def test_unfired_conditions_are_named_not_silently_dropped(self, plan, tmp_path):
        f = _filled(plan)
        f.loc[f[PLAN_KEY].isin([2, 5]), STATUS_COLUMN] = "failed"
        f.loc[f[PLAN_KEY].isin([2, 5]), READOUT_COLUMN] = ""
        res = load(_write(tmp_path, f), plan)
        assert res.n_measured == len(plan) + 1 - 2
        assert unfired(res) == [2, 5]

    def test_conditions_returns_only_measured_rows(self, plan, tmp_path):
        f = _filled(plan)
        f.loc[f[PLAN_KEY].isin([CONTROL_INDEX, 1]), STATUS_COLUMN] = "not_run"
        f.loc[f[PLAN_KEY].isin([CONTROL_INDEX, 1]), READOUT_COLUMN] = ""
        v, t, y = load(_write(tmp_path, f), plan).conditions()
        assert len(v) == len(t) == len(y) == len(plan) - 1  # control + one row not_run
        assert np.isfinite(y).all()
