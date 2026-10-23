"""Build the tables a round produces: per capacitor, per sample, and the campaign across rounds.

The per-capacitor table has one row per capacitor and state. Its columns are a contract with the
correlation study: each feature record becomes a block of columns named ``<prefix><field>``,
and the order of the blocks in the ``flatten_*`` functions below is the order of the table.
``CAPACITOR_COLUMNS`` is that layout with every block empty.

The per-sample table reduces each sample and state to mean, sd, n, median and MAD per feature.
Features that only mean something on a switching capacitor (PUND switching-peak voltages, the
O / T split, wake-up) are averaged over the switching capacitors alone. The P-V coercive
voltages are averaged over every capacitor, as the group's summary does.

The campaign table holds every round's per-capacitor rows and is what the correlation study
reads. A rerun round replaces its own rows, and the file is written atomically.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from electrical.extract_endurance_features import EnduranceFeatures
from electrical.extract_pund_features import (
    SPLIT_KEYS,
    Coercivity,
    Leakage,
    PulseCharge,
    PundDielectric,
    PundFeatures,
    PundWaveform,
    RampSplit,
    SwitchedPolarization,
    SwitchingPeak,
)
from electrical.extract_pv_features import (
    CoerciveVoltage,
    NegativeHalfLoop,
    PvDielectric,
    PvFeatures,
    PvSweep,
    Remanence,
)
from electrical.find_measurement_files import CapacitorId
from electrical.flag_issues import OUTLIER_FLAGS_COLUMN, Issue, describe_issues, has_error
from electrical.read_measurement_runs import PRISTINE_STATE, InstrumentCharges
from utils.file_io import read_csv_with_columns, write_csv_atomic

# ------------------------------------------------------------------------- per capacitor

#: Columns that say which capacitor and state a row is.
IDENTITY_COLUMNS = (
    "round",
    "sample",
    "sample_key",
    "sample_num",
    "cap",
    "row",
    "col",
    "state",
    "side_um",
    "area_cm2",
)
#: Columns that close every row: when the runs were measured, and the issues found.
TRAILING_COLUMNS = ("pund_executed", "pv_executed", "n_issues", "has_error", "issues")
#: Ratios added to every row, defined on switching capacitors only.
DERIVED_COLUMNS = ("psw_over_pr", "psw_inst_over_pr")
#: Key that identifies one row across rounds.
CAPACITOR_KEY = ("round", "sample_key", "cap", "state")


@dataclass(frozen=True, slots=True, kw_only=True)
class CapacitorRecord:
    """Everything known about one capacitor in one state."""

    round_label: int
    sample_key: str
    sample_name: str
    sample_number: int | None
    capacitor: CapacitorId
    state: str
    side_um: int
    area_cm2: float
    pund_executed_at: datetime | None = None
    pv_executed_at: datetime | None = None
    pund: PundFeatures | None = None
    pv: PvFeatures | None = None
    endurance: EnduranceFeatures | None = None
    issues: tuple[Issue, ...] = ()


def _record_columns(prefix: str, record_type: type, record: object | None) -> dict[str, object]:
    """One record's fields as ``prefix + field`` columns; NaN throughout when it is absent."""
    names = [f.name for f in fields(record_type) if f.name != "issues"]
    if record is None:
        return {prefix + name: np.nan for name in names}
    values = (getattr(record, name) for name in names)
    return {prefix + name: v.value if isinstance(v, Enum) else v for name, v in zip(names, values)}


def flatten_pund_features(f: PundFeatures | None) -> dict[str, object]:
    """Lay out the PUND features as columns, in table order."""
    row = {}
    row |= _record_columns("", PundWaveform, f and f.waveform)
    row |= _record_columns("inst_", InstrumentCharges, f and f.instrument)
    row |= _record_columns("q_p_", PulseCharge, f and f.charge_p)
    row |= _record_columns("q_u_", PulseCharge, f and f.charge_u)
    row |= _record_columns("q_n_", PulseCharge, f and f.charge_n)
    row |= _record_columns("q_d_", PulseCharge, f and f.charge_d)
    row |= _record_columns("", SwitchedPolarization, f and f.polarization)
    for key in SPLIT_KEYS:
        row |= _record_columns(f"split_{key}_", RampSplit, f and f.splits.get(key))
    row |= _record_columns("peak_pos_", SwitchingPeak, f and f.peak_pos)
    row |= _record_columns("peak_neg_", SwitchingPeak, f and f.peak_neg)
    row |= _record_columns("", Coercivity, f and f.coercivity)
    row |= _record_columns("", Leakage, f and f.leakage)
    row |= _record_columns("", PundDielectric, f and f.dielectric)
    return row


def flatten_pv_features(f: PvFeatures | None) -> dict[str, object]:
    """Lay out the P-V features as columns, in table order."""
    row = {}
    row |= _record_columns("", PvSweep, f and f.sweep)
    row |= _record_columns("", Remanence, f and f.remanence)
    row |= _record_columns("", CoerciveVoltage, f and f.coercive)
    row |= _record_columns("", NegativeHalfLoop, f and f.half_loop)
    row |= _record_columns("", PvDielectric, f and f.dielectric)
    return row


def flatten_endurance_features(f: EnduranceFeatures | None) -> dict[str, object]:
    """Lay out the endurance features as ``endurance_*`` columns."""
    return _record_columns("endurance_", EnduranceFeatures, f)


CAPACITOR_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    *flatten_pund_features(None),
    *flatten_pv_features(None),
    *flatten_endurance_features(None),
    *TRAILING_COLUMNS,
)


def build_capacitor_row(record: CapacitorRecord) -> dict[str, object]:
    """Lay out one record as a row keyed by ``CAPACITOR_COLUMNS``."""
    return {
        "round": record.round_label,
        "sample": record.sample_name,
        "sample_key": record.sample_key,
        "sample_num": record.sample_number,
        "cap": record.capacitor.label,
        "row": record.capacitor.row,
        "col": record.capacitor.col,
        "state": record.state,
        "side_um": record.side_um,
        "area_cm2": record.area_cm2,
        **flatten_pund_features(record.pund),
        **flatten_pv_features(record.pv),
        **flatten_endurance_features(record.endurance),
        "pund_executed": record.pund_executed_at,
        "pv_executed": record.pv_executed_at,
        "n_issues": len(record.issues),
        "has_error": has_error(record.issues),
        "issues": describe_issues(record.issues) or None,
    }


def build_capacitor_table(records: Iterable[CapacitorRecord]) -> pd.DataFrame:
    """Build the per-capacitor table and add the Psw / Pr ratios."""
    table = pd.DataFrame.from_records(
        [build_capacitor_row(r) for r in records], columns=list(CAPACITOR_COLUMNS)
    )
    table["switching"] = table["switching"].fillna(False).astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        table["psw_over_pr"] = table["psw_uc_cm2"] / table["pr_uc_cm2"]
        table["psw_inst_over_pr"] = table["psw_inst_uc_cm2"] / table["pr_uc_cm2"]
    table.loc[~table["switching"], list(DERIVED_COLUMNS)] = np.nan
    return table


# ---------------------------------------------------------------------------- per sample

#: A sample switches when its mean Pr exceeds this, uC/cm^2.
SWITCHING_SAMPLE_PR_UC_CM2 = 2.0
#: A sample is inhomogeneous when its relative Pr scatter exceeds this many times the median.
INHOMOGENEITY_FACTOR = 3
#: Statistics reported for every headline feature, in column order.
STATISTICS = ("mean", "sd", "n", "median", "mad")

SPLIT_FEATURES = (
    *(f"split_{key}_fraction_main" for key in SPLIT_KEYS),
    "split_raw_n_dip_t_split_s",
)
ENDURANCE_FEATURES = (
    "endurance_psw_cycle0_uc_cm2",
    "endurance_psw_cycle_max_uc_cm2",
    "endurance_cycles_at_max",
    "endurance_psw_cycle_final_uc_cm2",
    "endurance_wake_ratio",
    "endurance_wake_ratio_max",
    "endurance_fatigue_ratio",
    "endurance_psw_gain_uc_cm2",
    "endurance_cycles_half_wakeup",
    "endurance_psw_asym_final_uc_cm2",
)
#: Features that get the full mean / sd / n / median / MAD treatment.
HEADLINE_FEATURES = (
    "psw_uc_cm2",
    "psw_pos_uc_cm2",
    "psw_neg_uc_cm2",
    "psw_inst_uc_cm2",
    "psw_asym_uc_cm2",
    "psw_plateau_pos_uc_cm2",
    "psw_plateau_neg_uc_cm2",
    "p_raw_n_ramp_uc_cm2",
    "split_raw_n_dip_total_uc_cm2",
    *SPLIT_FEATURES,
    "peak_pos_coercive_v",
    "peak_neg_coercive_v",
    "vc_mean_v",
    "imprint_v",
    "peak_pos_current_a",
    "peak_neg_current_a",
    "peak_pos_t_peak_s",
    "peak_neg_t_peak_s",
    "j_leak_a_cm2",
    "kappa_pund",
    "kappa_inst_u",
    *ENDURANCE_FEATURES,
    "pr_uc_cm2",
    "pr_neg_uc_cm2",
    "pr_gap_uc_cm2",
    "pr_gap_corrected_uc_cm2",
    "pmax_uc_cm2",
    "pmax_minus_pr_uc_cm2",
    "ec_pos_v",
    "ec_neg_v",
    "pinching_index",
    "loop_width_at_zero_uc_cm2",
    "loop_width_max_uc_cm2",
    "loop_area_uc_v_cm2",
    "kappa_summary",
    "kappa0",
    "dkappa_rel",
    "kappa_neg_branch",
    "kappa_rise1",
    *DERIVED_COLUMNS,
)
#: Features averaged over switching capacitors only.
SWITCHING_ONLY_FEATURES = (
    *SPLIT_FEATURES,
    "peak_pos_coercive_v",
    "peak_neg_coercive_v",
    "vc_mean_v",
    "imprint_v",
    "peak_pos_t_peak_s",
    "peak_neg_t_peak_s",
    "pinching_index",
    *DERIVED_COLUMNS,
    "endurance_wake_ratio",
    "endurance_wake_ratio_max",
    "endurance_fatigue_ratio",
    "endurance_cycles_half_wakeup",
    "endurance_cycles_at_max",
)
SAMPLE_COLUMNS: tuple[str, ...] = (
    "round",
    "sample",
    "sample_key",
    "sample_num",
    "state",
    "n_capacitors",
    "n_switching",
    "frac_switching",
    "n_errors",
    *(f"{feature}_{stat}" for feature in HEADLINE_FEATURES for stat in STATISTICS),
    "pr_grad_row",
    "pr_grad_col",
    "pr_rel_sd",
    "inhomogeneous",
)


def summarize_samples(capacitors: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-capacitor table to one row per (round, sample, state)."""
    features = [f for f in HEADLINE_FEATURES if f in capacitors.columns]
    rows = []
    for (rnd, sample, state), g in capacitors.groupby(["round", "sample", "state"], sort=False):
        switching = g["switching"].astype(bool)
        row = {
            "round": rnd,
            "sample": sample,
            "sample_key": g["sample_key"].iloc[0],
            "sample_num": g["sample_num"].iloc[0],
            "state": state,
            "n_capacitors": len(g),
            "n_switching": int(switching.sum()),
            "frac_switching": float(switching.mean()),
            "n_errors": int(g["has_error"].sum()),
        }
        for feature in features:
            x = g[feature][switching] if feature in SWITCHING_ONLY_FEATURES else g[feature]
            x = pd.to_numeric(x, errors="coerce").dropna()
            row[f"{feature}_mean"] = x.mean() if len(x) else np.nan
            row[f"{feature}_sd"] = x.std(ddof=1) if len(x) > 1 else np.nan
            row[f"{feature}_n"] = len(x)
            row[f"{feature}_median"] = x.median() if len(x) else np.nan
            row[f"{feature}_mad"] = (x - x.median()).abs().median() if len(x) else np.nan
        row["pr_grad_row"], row["pr_grad_col"] = _pr_gradient(g)
        rows.append(row)
    samples = pd.DataFrame(rows)

    # Inhomogeneous: relative Pr scatter far above the median, on pristine switching films.
    pristine_switching = (samples["pr_uc_cm2_mean"].abs() > SWITCHING_SAMPLE_PR_UC_CM2) & (
        samples["state"] == PRISTINE_STATE
    )
    relative = (samples["pr_uc_cm2_sd"] / samples["pr_uc_cm2_mean"].abs()).where(pristine_switching)
    threshold = INHOMOGENEITY_FACTOR * relative.median() if relative.notna().any() else np.inf
    samples["pr_rel_sd"] = relative
    samples["inhomogeneous"] = relative > threshold
    return samples[[c for c in SAMPLE_COLUMNS if c in samples.columns]]


def _pr_gradient(group: pd.DataFrame) -> tuple[float, float]:
    """Linear trend of Pr across the capacitor array, uC/cm^2 per row and per column."""
    if group["row"].nunique() < 2 and group["col"].nunique() < 2:
        return np.nan, np.nan
    design = np.column_stack([np.ones(len(group)), group["row"], group["col"]]).astype(float)
    y = group["pr_uc_cm2"].astype(float).to_numpy()
    ok = np.isfinite(y)
    if ok.sum() < 4 or np.linalg.matrix_rank(design[ok]) < 3:
        return np.nan, np.nan
    b = np.linalg.lstsq(design[ok], y[ok], rcond=None)[0]
    return float(b[1]), float(b[2])


# ------------------------------------------------------------------------------ campaign

#: Columns of the campaign table: the per-capacitor contract, the ratios and the outlier flags.
CAMPAIGN_COLUMNS: tuple[str, ...] = (*CAPACITOR_COLUMNS, *DERIVED_COLUMNS, OUTLIER_FLAGS_COLUMN)


def read_campaign_table(path: Path) -> pd.DataFrame:
    """Read the campaign table, checking it has exactly the columns this version writes.

    :raises ContractError: If columns are missing or unexpected; rebuild by rerunning every round.
    """
    return read_csv_with_columns(path, CAMPAIGN_COLUMNS)


def update_campaign_table(path: Path, capacitors: pd.DataFrame) -> pd.DataFrame:
    """Replace one round's rows in the campaign table, keep every other round's, write atomically.

    :param path: The campaign CSV; created if absent.
    :param capacitors: The round's flagged per-capacitor table.
    :return: The table as written.
    """
    incoming = capacitors[list(CAMPAIGN_COLUMNS)]
    merged = incoming
    if path.exists():
        existing = read_campaign_table(path)
        key = list(CAPACITOR_KEY)
        kept = existing[~existing.set_index(key).index.isin(incoming.set_index(key).index)]
        merged = pd.concat([kept, incoming], ignore_index=True)
    order = ["round", "sample_num", "sample_key", "row", "col", "state"]
    merged = merged.sort_values(order).reset_index(drop=True)
    write_csv_atomic(path, merged)
    return merged
