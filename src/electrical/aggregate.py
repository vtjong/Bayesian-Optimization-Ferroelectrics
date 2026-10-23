"""Per-capacitor table -> per-sample table with repeat scatter and homogeneity flags.

Statistics are taken per (round, sample, state). Features that only mean something on a
switching capacitor (coercive voltages, the O / T split, wake-up) are averaged over the
switching capacitors alone; everything else over every capacitor measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from electrical.pund import SPLIT_KEYS

#: A sample switches when its mean Pr exceeds this, uC/cm^2.
SWITCHING_SAMPLE_PR_UC_CM2 = 2.0
#: A sample is inhomogeneous when its relative Pr scatter exceeds this many times the median.
INHOMOGENEITY_FACTOR = 3
#: Statistics reported for every headline feature, in column order.
STATISTICS = ("mean", "sd", "n", "median", "mad")

SPLIT_FEATURES = tuple(f"split_{key}_fraction_main" for key in SPLIT_KEYS) + (
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
    "psw_over_pr",
    "psw_inst_over_pr",
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
    "ec_neg_v",
    "pinching_index",
    "psw_over_pr",
    "psw_inst_over_pr",
    "endurance_wake_ratio",
    "endurance_wake_ratio_max",
    "endurance_fatigue_ratio",
    "endurance_cycles_half_wakeup",
    "endurance_cycles_at_max",
)
#: Ratios added to the per-capacitor table, defined on switching capacitors only.
DERIVED_COLUMNS = ("psw_over_pr", "psw_inst_over_pr")
SAMPLE_IDENTITY_COLUMNS = ("round", "sample", "sample_key", "sample_num", "state")
SAMPLE_COUNT_COLUMNS = ("n_capacitors", "n_switching", "frac_switching", "n_errors")
SAMPLE_SHAPE_COLUMNS = ("pr_grad_row", "pr_grad_col", "pr_rel_sd", "inhomogeneous")
SAMPLE_COLUMNS: tuple[str, ...] = (
    *SAMPLE_IDENTITY_COLUMNS,
    *SAMPLE_COUNT_COLUMNS,
    *(f"{feature}_{stat}" for feature in HEADLINE_FEATURES for stat in STATISTICS),
    *SAMPLE_SHAPE_COLUMNS,
)


def add_derived(capacitors: pd.DataFrame) -> pd.DataFrame:
    """Add the Psw / Pr ratios, defined on switching capacitors only."""
    capacitors = capacitors.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        capacitors["psw_over_pr"] = capacitors["psw_uc_cm2"] / capacitors["pr_uc_cm2"]
        capacitors["psw_inst_over_pr"] = capacitors["psw_inst_uc_cm2"] / capacitors["pr_uc_cm2"]
    not_switching = ~capacitors["switching"].astype(bool)
    capacitors.loc[not_switching, ["psw_over_pr", "psw_inst_over_pr"]] = np.nan
    return capacitors


def median_absolute_deviation(values) -> float:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def pr_gradient(group: pd.DataFrame) -> tuple[float, float]:
    """Linear trend of Pr over the capacitor array, uC/cm^2 per row and per column."""
    if group["row"].nunique() < 2 and group["col"].nunique() < 2:
        return np.nan, np.nan
    design = np.column_stack([np.ones(len(group)), group["row"], group["col"]]).astype(float)
    y = group["pr_uc_cm2"].astype(float).to_numpy()
    ok = np.isfinite(y)
    if ok.sum() < 4 or np.linalg.matrix_rank(design[ok]) < 3:
        return np.nan, np.nan
    b = np.linalg.lstsq(design[ok], y[ok], rcond=None)[0]
    return float(b[1]), float(b[2])


def per_sample(capacitors: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-capacitor table to one row per (round, sample, state)."""
    capacitors = add_derived(capacitors)
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
            "n_errors": int(g["has_error"].sum()) if "has_error" in g else 0,
        }
        for feature in features:
            x = g[feature]
            if feature in SWITCHING_ONLY_FEATURES:
                x = x[switching]
            x = pd.to_numeric(x, errors="coerce").dropna()
            row[f"{feature}_mean"] = x.mean() if len(x) else np.nan
            row[f"{feature}_sd"] = x.std(ddof=1) if len(x) > 1 else np.nan
            row[f"{feature}_n"] = len(x)
            row[f"{feature}_median"] = x.median() if len(x) else np.nan
            row[f"{feature}_mad"] = median_absolute_deviation(x)
        row["pr_grad_row"], row["pr_grad_col"] = pr_gradient(g)
        rows.append(row)
    samples = pd.DataFrame(rows)
    # inhomogeneity: relative Pr scatter well above the campaign's median, pristine switching films
    pristine_switching = (samples["pr_uc_cm2_mean"].abs() > SWITCHING_SAMPLE_PR_UC_CM2) & (
        samples["state"] == "pris"
    )
    relative = (samples["pr_uc_cm2_sd"] / samples["pr_uc_cm2_mean"].abs()).where(pristine_switching)
    threshold = INHOMOGENEITY_FACTOR * relative.median() if relative.notna().any() else np.inf
    samples["pr_rel_sd"] = relative
    samples["inhomogeneous"] = relative > threshold
    return samples[[c for c in SAMPLE_COLUMNS if c in samples.columns]]
