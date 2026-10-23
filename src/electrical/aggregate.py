"""Per-capacitor table -> per-sample table with repeat scatter and homogeneity flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLS = ["round", "sample", "sample_num", "cap", "row", "col", "state", "size_um"]
# features that get the full mean/sd/n/median/MAD treatment
HEADLINE = [
    "Psw",
    "Psw_pos",
    "Psw_neg",
    "Psw_inst",
    "Psw_asym",
    "Psw_plateau_pos",
    "Psw_plateau_neg",
    "P_raw_N_ramp",
    "P_rawN_ramp_total",
    "QO_frac_rawN",
    "QO_frac_sub_neg",
    "QO_frac_sub_pos",
    "t_split_rawN_s",
    "Vc_pos",
    "Vc_neg",
    "Vc_mean",
    "imprint_PUND",
    "Isw_peak_pos_A",
    "Isw_peak_neg_A",
    "t_peak_pos_s",
    "t_peak_neg_s",
    "J_leak_A_cm2",
    "kappa_PUND",
    "kappa_inst_U",
    "Pr",
    "Pr_summary",
    "Pr_pos",
    "Pr_neg",
    "Pr_gap",
    "Pr_gap_corr",
    "Pmax",
    "Pmax_minus_Pr",
    "Ec_neg",
    "pinching_index",
    "loop_width_neg_at0",
    "loop_width_neg_max",
    "loop_area_neg",
    "kappa_summary",
    "kappa0",
    "dkappa_rel",
    "kappa_neg_branch",
    "kappa_rise1",
    "Psw_over_Pr",
    "Psw_inst_over_Pr",
    "Psw_cyc0",
    "Psw_cyc_max",
    "cycles_at_max",
    "Psw_cyc_final",
    "wake_ratio",
    "wake_ratio_max",
    "fatigue_ratio",
    "Psw_gain_abs",
    "cycles_half_wakeup",
    "Psw_asym_cyc_final",
]
# features that only make sense on switching capacitors
SWITCHING_ONLY = [
    "QO_frac_rawN",
    "QO_frac_sub_neg",
    "QO_frac_sub_pos",
    "t_split_rawN_s",
    "Vc_pos",
    "Vc_neg",
    "Vc_mean",
    "imprint_PUND",
    "t_peak_pos_s",
    "t_peak_neg_s",
    "Ec_neg",
    "pinching_index",
    "Psw_over_Pr",
    "Psw_inst_over_Pr",
    "wake_ratio",
    "wake_ratio_max",
    "fatigue_ratio",
    "cycles_half_wakeup",
    "cycles_at_max",
]


def add_derived(cap: pd.DataFrame) -> pd.DataFrame:
    cap = cap.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cap["Psw_over_Pr"] = cap["Psw"] / cap["Pr"]
        cap["Psw_inst_over_Pr"] = cap["Psw_inst"] / cap["Pr"]
    cap.loc[~cap["switching"].astype(bool), ["Psw_over_Pr", "Psw_inst_over_Pr"]] = np.nan
    return cap


def _mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def per_sample(cap: pd.DataFrame) -> pd.DataFrame:
    cap = add_derived(cap)
    feats = [c for c in HEADLINE if c in cap.columns]
    out = []
    for (rnd, sample, state), g in cap.groupby(["round", "sample", "state"], sort=False):
        row = {
            "round": rnd,
            "sample": sample,
            "state": state,
            "sample_num": g["sample_num"].iloc[0],
            "n_capacitors": len(g),
            "n_switching": int(g["switching"].sum()),
            "frac_switching": float(g["switching"].mean()),
            "n_problems": int(g["problem"].notna().sum()) if "problem" in g else 0,
        }
        for c in feats:
            x = g[c]
            if c in SWITCHING_ONLY:
                x = x[g["switching"].astype(bool)]
            x = pd.to_numeric(x, errors="coerce").dropna()
            row[f"{c}_mean"] = x.mean() if len(x) else np.nan
            row[f"{c}_sd"] = x.std(ddof=1) if len(x) > 1 else np.nan
            row[f"{c}_n"] = len(x)
            row[f"{c}_median"] = x.median() if len(x) else np.nan
            row[f"{c}_mad"] = _mad(x)
        # spatial gradient of Pr over the capacitor array (uC/cm^2 per row / per column)
        if g["row"].nunique() > 1 or g["col"].nunique() > 1:
            X = np.column_stack([np.ones(len(g)), g["row"], g["col"]]).astype(float)
            y = g["Pr"].astype(float).to_numpy()
            ok = np.isfinite(y)
            if ok.sum() >= 4 and np.linalg.matrix_rank(X[ok]) == 3:
                b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
                row["Pr_grad_row"], row["Pr_grad_col"] = float(b[1]), float(b[2])
        out.append(row)
    df = pd.DataFrame(out)
    # inhomogeneity flag: sd/|mean| of Pr above 3x the campaign median
    # (switching samples, pristine state)
    rel = (df["Pr_sd"] / df["Pr_mean"].abs()).where(
        (df["Pr_mean"].abs() > 2.0) & (df["state"] == "pris")
    )
    thr = 3 * rel.median() if rel.notna().any() else np.inf
    df["Pr_rel_sd"] = rel
    df["inhomogeneous"] = rel > thr
    return df
