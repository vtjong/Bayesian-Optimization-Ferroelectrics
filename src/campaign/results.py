"""Ingest measured outcomes for a fired batch, and refuse to guess when the record is ambiguous.

The campaign's whole value is a chain of inference from a readout back to a boundary, so the point
where numbers enter the repository is the point where the chain is most easily broken. Three
failures already present in this project's archival data motivated every rule below:

  * a blank cell silently read as ``0`` -- indistinguishable from a genuine amorphous reading;
  * a row shifted one column, putting a 1e6-cycle value into a 3-cycle field, so a real measurement
    became a wrong measurement rather than a missing one;
  * the condition a specimen actually received never recorded, only the condition it was assigned,
    so a delivery error is invisible and reads as specimen scatter.

So: STATUS IS EXPLICIT AND MANDATORY. A reading is present because the row says ``measured``, never
because a cell happens to be non-empty. A ``measured`` row without a finite value is an error, and
so is a value on a row marked ``not_run`` or ``failed``. Nothing is inferred from emptiness.

The AS-FIRED conditions are recorded separately from the planned ones. They are what the models are
conditioned on; the planned values are kept only so the two can be compared. A specimen that was
meant to get 632 V and got 645 V is a usable measurement at 645 V and a corrupt one at 632 V.

This module reads and validates. It does not fit anything.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Join key back to the plan, then what the operator records.
PLAN_KEY = "index"
# The as-deposited reference: a specimen that is NEVER FLASHED, carried as index 0 so it travels
# with the batch and is read by the same loader. It exists because the campaign currently has no
# measured amorphous value at all -- every P-V sample in the archive was annealed, and the "floor"
# quoted from them selects loops with 2Pr < 2, which means NOT FERROELECTRIC rather than NOT
# CRYSTALLIZED. Without this row, block E's floor anchor has nothing to anchor against and a
# reading of 38 cannot be told apart from an already-transformed non-ferroelectric film.
CONTROL_INDEX = 0
STATUS_COLUMN = "status"
STATUS_MEASURED = "measured"
STATUS_FAILED = "failed"  # fired, but the specimen or the readout was lost
STATUS_NOT_RUN = "not_run"  # never fired
VALID_STATUS = (STATUS_MEASURED, STATUS_FAILED, STATUS_NOT_RUN)

SPECIMEN_COLUMN = "specimen_id"
ACTUAL_V_COLUMN = "voltage_V_actual"
ACTUAL_T_COLUMN = "time_ms_actual"
READOUT_COLUMN = "readout_value"
READOUT_KIND_COLUMN = "readout"
CROSSCHECK_COLUMN = "two_pr_uC_cm2"  # independent PUND cross-check, optional
DATE_COLUMN = "date"
OPERATOR_COLUMN = "operator"
NOTES_COLUMN = "notes"

RESULT_COLUMNS = (
    PLAN_KEY,
    SPECIMEN_COLUMN,
    STATUS_COLUMN,
    ACTUAL_V_COLUMN,
    ACTUAL_T_COLUMN,
    READOUT_KIND_COLUMN,
    READOUT_COLUMN,
    CROSSCHECK_COLUMN,
    DATE_COLUMN,
    OPERATOR_COLUMN,
    NOTES_COLUMN,
)

# A shot whose as-fired condition differs from plan by more than this is reported. Not an error --
# the measurement is still usable at its as-fired condition -- but an iso-Tmax ladder rung that
# drifted is no longer on its level, which changes what the shot can be used for.
DRIFT_V_TOLERANCE = 0.5
DRIFT_T_TOLERANCE_MS = 0.05


@dataclass
class SeedResults:
    """Validated outcomes for a fired batch.

    :param table: one row per plan condition, planned and as-fired columns joined.
    :param measured: boolean mask of rows carrying a usable reading.
    :param drift: rows whose as-fired condition differs from plan beyond tolerance.
    """

    table: pd.DataFrame
    measured: np.ndarray
    drift: pd.DataFrame

    @property
    def n_measured(self) -> int:
        """How many conditions returned a usable reading."""
        return int(self.measured.sum())

    def conditions(self) -> tuple:
        """As-fired ``(V, t, y)`` for the measured rows only -- what a model may be fitted to."""
        d = self.table[self.measured]
        return (
            d[ACTUAL_V_COLUMN].to_numpy(float),
            d[ACTUAL_T_COLUMN].to_numpy(float),
            d[READOUT_COLUMN].to_numpy(float),
        )


def blank_template(plan: pd.DataFrame) -> pd.DataFrame:
    """A results sheet pre-filled with the plan and with every outcome field empty.

    The as-fired columns are pre-filled with the PLANNED values as a convenience, and the operator
    is expected to correct them where reality differed. Status is left empty deliberately: an
    unfilled sheet must fail validation rather than read as sixteen not-run specimens.

    :param plan: the fired plan, in the CSV's column names.
    """
    out = pd.DataFrame({PLAN_KEY: np.concatenate([[CONTROL_INDEX], plan[PLAN_KEY].to_numpy()])})
    out[SPECIMEN_COLUMN] = ""
    out[STATUS_COLUMN] = ""
    out[ACTUAL_V_COLUMN] = np.concatenate([[np.nan], plan["voltage_V"].to_numpy()])
    out[ACTUAL_T_COLUMN] = np.concatenate([[np.nan], plan["time_ms"].to_numpy()])
    out[READOUT_KIND_COLUMN] = np.concatenate(
        [[plan[READOUT_KIND_COLUMN].iloc[0]], plan[READOUT_KIND_COLUMN].to_numpy()]
    )
    out[READOUT_COLUMN] = ""
    out[CROSSCHECK_COLUMN] = ""
    out[DATE_COLUMN] = ""
    out[OPERATOR_COLUMN] = ""
    out[NOTES_COLUMN] = ""
    out.loc[out[PLAN_KEY] == CONTROL_INDEX, NOTES_COLUMN] = (
        "AS-DEPOSITED CONTROL -- do not flash. Leave voltage/time blank."
    )
    return out[list(RESULT_COLUMNS)]


def _blank(series: pd.Series) -> np.ndarray:
    """Mask of cells that are empty, whitespace, or NA -- never coerced to a number."""
    s = series.astype("object")
    return s.isna().to_numpy() | s.astype(str).str.strip().isin(["", "nan", "None"]).to_numpy()


def load(results_path: Path, plan: pd.DataFrame) -> SeedResults:
    """Read a filled results sheet, validate it against the plan, and join the two.

    Raises ``ValueError`` on any ambiguity rather than making a choice on the operator's behalf.

    :param results_path: filled results CSV.
    :param plan: the seed plan the results correspond to.
    """
    # keep_default_na=False so an empty cell stays an empty string and is never confused with a
    # legitimate zero; every numeric column is converted explicitly below.
    raw = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]

    missing = [c for c in RESULT_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"results sheet is missing required columns: {missing}")

    idx = pd.to_numeric(raw[PLAN_KEY], errors="coerce")
    if idx.isna().any():
        raise ValueError(f"non-numeric {PLAN_KEY} in rows {list(np.where(idx.isna())[0] + 2)}")
    raw[PLAN_KEY] = idx.astype(int)

    want, got = set(plan[PLAN_KEY]) | {CONTROL_INDEX}, set(raw[PLAN_KEY])
    if want != got:
        raise ValueError(
            f"results do not correspond 1:1 with the plan -- missing {sorted(want - got)}, "
            f"unexpected {sorted(got - want)}"
        )
    if raw[PLAN_KEY].duplicated().any():
        dup = sorted(raw[PLAN_KEY][raw[PLAN_KEY].duplicated()].unique())
        raise ValueError(f"duplicate plan indices in the results sheet: {dup}")

    status = raw[STATUS_COLUMN].astype(str).str.strip().str.lower()
    bad = sorted(set(status) - set(VALID_STATUS))
    if bad:
        raise ValueError(
            f"unrecognised {STATUS_COLUMN} values {bad}; every row must be one of {VALID_STATUS}"
        )

    is_measured = (status == STATUS_MEASURED).to_numpy()
    value_blank = _blank(raw[READOUT_COLUMN])

    if (is_measured & value_blank).any():
        rows = list(raw.loc[is_measured & value_blank, PLAN_KEY])
        raise ValueError(f"rows marked '{STATUS_MEASURED}' with no readout value: {rows}")
    if (~is_measured & ~value_blank).any():
        rows = list(raw.loc[~is_measured & ~value_blank, PLAN_KEY])
        raise ValueError(
            f"rows carrying a readout value but not marked '{STATUS_MEASURED}': {rows}. "
            "A value that is not a measurement must be removed, not left with a status that "
            "hides it."
        )

    value = pd.to_numeric(raw[READOUT_COLUMN].where(~value_blank), errors="coerce")
    if (is_measured & ~np.isfinite(value.to_numpy(float))).any():
        rows = list(raw.loc[is_measured & ~np.isfinite(value.to_numpy(float)), PLAN_KEY])
        raise ValueError(f"non-numeric readout values in rows {rows}")

    # The as-deposited control is exempt: it has no as-fired condition because it was never fired.
    needs_condition = is_measured & (raw[PLAN_KEY].to_numpy() != CONTROL_INDEX)
    for col in (ACTUAL_V_COLUMN, ACTUAL_T_COLUMN):
        as_num = pd.to_numeric(raw[col].where(~_blank(raw[col])), errors="coerce")
        bad = needs_condition & ~np.isfinite(as_num.to_numpy(float))
        if bad.any():
            rows = list(raw.loc[bad, PLAN_KEY])
            raise ValueError(f"'{col}' missing or non-numeric on measured rows {rows}")
        raw[col] = as_num

    raw[READOUT_COLUMN] = value
    raw[CROSSCHECK_COLUMN] = pd.to_numeric(
        raw[CROSSCHECK_COLUMN].where(~_blank(raw[CROSSCHECK_COLUMN])), errors="coerce"
    )

    # The control is read on the same instrument as the batch, so it is checked too.
    kind_plan = plan.set_index(PLAN_KEY)[READOUT_KIND_COLUMN]
    expected = {i: kind_plan[i] for i in kind_plan.index}
    expected[CONTROL_INDEX] = kind_plan.iloc[0]
    kind_got = raw.set_index(PLAN_KEY)[READOUT_KIND_COLUMN].str.strip()
    clash = [i for i in expected if kind_got[i] and kind_got[i] != expected[i]]
    if clash:
        raise ValueError(
            f"readout kind differs from the plan on rows {clash}; the noise model and the "
            "boundary anchor are specific to one readout and cannot be mixed"
        )

    table = plan.merge(
        raw, on=PLAN_KEY, how="right", suffixes=("", "_result"), validate="one_to_one"
    )
    table["block"] = table["block"].fillna("R")  # R = reference, never flashed
    table = table.sort_values(PLAN_KEY).reset_index(drop=True)
    measured = (
        table[STATUS_COLUMN].astype(str).str.strip().str.lower() == STATUS_MEASURED
    ).to_numpy()

    dv = (table[ACTUAL_V_COLUMN] - table["voltage_V"]).abs()
    dt = (table[ACTUAL_T_COLUMN] - table["time_ms"]).abs()
    drifted = measured & ((dv > DRIFT_V_TOLERANCE) | (dt > DRIFT_T_TOLERANCE_MS)).fillna(False)
    drifted = drifted.to_numpy()
    return SeedResults(table=table, measured=measured, drift=table[drifted])


def unfired(results: SeedResults) -> List[int]:
    """Plan indices with no usable reading, so a caller can report coverage honestly."""
    return sorted(results.table.loc[~results.measured, PLAN_KEY].astype(int))
