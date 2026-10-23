"""P-V (really Q-V) features from one Keithley `doubleSweepSeg` export.

The sweep is a single 0 -> +V1 -> 0 -> V2 -> 0 triangle from the pristine state,
so the first quarter starts from an unpoled film and the closed part of the loop
is +V1 -> 0 -> V2 -> 0.  `Charge` is the cumulative integral of `Imeas`.

Three quantities reproduce the group's summary sheet (checked on Sample14):
  Pr_summary    = [Q(V=0 on the +V1->0 branch) - (Qmax+Qmin)/2] / A
  kappa_summary = slope of Q vs V over 0.8..2.25 V on the +V1->0 branch, x t/(eps0 A)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from electrical.device import EPS0, T_HZO_M, area_cm2

KAPPA_WINDOW_V = (0.8, 2.25)  # the group's dielectric-fit window
KAPPA0_WINDOW_V = 0.3  # |V| < 0.3 V for the zero-field slope


def branches(V: np.ndarray):
    """Index slices of the four legs from the turning points of V."""
    i_max, i_min = int(np.argmax(V)), int(np.argmin(V))
    if i_max < i_min:
        return {
            "rise1": slice(0, i_max + 1),
            "fall_pos": slice(i_max, None),
            "order": "pos_first",
            "i_max": i_max,
            "i_min": i_min,
        }
    return {"order": "neg_first", "i_max": i_max, "i_min": i_min}


def _cross(v: np.ndarray, q: np.ndarray, level: float = 0.0, which: str = "first"):
    """Interpolated q where v crosses `level` (v monotone-ish on the slice)."""
    s = np.sign(v - level)
    k = np.flatnonzero(np.diff(s) != 0)
    if len(k) == 0:
        return np.nan
    k = k[0] if which == "first" else k[-1]
    v0, v1, q0, q1 = v[k], v[k + 1], q[k], q[k + 1]
    if v1 == v0:
        return float(q0)
    return float(q0 + (level - v0) * (q1 - q0) / (v1 - v0))


def _cross_v(v: np.ndarray, q: np.ndarray, level: float):
    """Interpolated v where q crosses `level`."""
    s = np.sign(q - level)
    k = np.flatnonzero(np.diff(s) != 0)
    if len(k) == 0:
        return np.nan
    k = k[0]
    q0, q1, v0, v1 = q[k], q[k + 1], v[k], v[k + 1]
    if q1 == q0:
        return float(v0)
    return float(v0 + (level - q0) * (v1 - v0) / (q1 - q0))


def _slope(v, q, lo, hi):
    m = (v >= lo) & (v <= hi)
    if m.sum() < 5:
        return np.nan
    return float(np.polyfit(v[m], q[m], 1)[0])


def pv_features(df: pd.DataFrame, size_um: int | None = 30) -> tuple[dict, dict]:
    A = area_cm2(size_um)
    A_m2 = A * 1e-4
    V = df["Vforce"].astype(float).to_numpy()
    Q = df["Charge"].astype(float).to_numpy()
    current = df["Imeas"].astype(float).to_numpy()
    T = df["Time"].astype(float).to_numpy()
    uC = 1e6 / A
    kap = lambda C: float(C * T_HZO_M / (EPS0 * A_m2))  # noqa: E731
    f: dict = {
        "area_cm2": A,
        "dt_s": float(np.median(np.diff(T))),
        "n_points": len(T),
        "Vmax": float(V.max()),
        "Vmin": float(V.min()),
    }
    traces = {"V": V, "Q": Q, "I": current, "T": T}

    i_max, i_min = int(np.argmax(V)), int(np.argmin(V))
    if not i_max < i_min:
        f["problem"] = "sweep is not positive-first; branch logic assumes 0->+V->0->-V->0"
        return f, traces
    rise1 = slice(0, i_max + 1)  # 0 -> +V1 (from the pristine state)
    fall = slice(i_max, i_min + 1)  # +V1 -> 0 -> V2 (descending)
    rise2 = slice(i_min, len(V))  # V2 -> 0 (ascending)
    fall_pos = slice(i_max, i_max + int(np.argmin(np.abs(V[fall]))) + 1)  # +V1 -> 0 only

    Qc = 0.5 * (Q.max() + Q.min())
    f["Q_centre_C"] = Qc
    Q0_fall = _cross(V[fall], Q[fall], 0.0, "first")
    Q0_rise2 = _cross(V[rise2], Q[rise2], 0.0, "last") if np.any(V[rise2] > 0) else float(Q[-1])
    f["Pr_summary"] = (Q0_fall - Qc) * uC
    f["Pr_pos"] = f["Pr_summary"]
    f["Pr_neg"] = (Q0_rise2 - Qc) * uC
    f["Pr_gap"] = 0.5 * (Q0_fall - Q0_rise2) * uC  # offset-free in Q, not in current
    # current-offset (drift) correction: the instrument integrates a small DC offset
    # over the whole 50 us record; estimate it on the V=0 rest periods and re-integrate
    rest = np.abs(V) < 0.05
    rest[i_max - 50 : i_min + 50] = False  # keep only the flanking rests
    I_off = float(np.median(current[rest])) if rest.sum() > 20 else 0.0
    Qc_ = np.concatenate(
        [[0.0], np.cumsum(0.5 * (current[1:] + current[:-1] - 2 * I_off) * np.diff(T))]
    )
    f["I_offset_A"] = I_off
    Q0_fall_c = _cross(V[fall], Qc_[fall], 0.0, "first")
    Q0_rise2_c = (
        _cross(V[rise2], Qc_[rise2], 0.0, "last") if np.any(V[rise2] > 0) else float(Qc_[-1])
    )
    f["Pr_gap_corr"] = 0.5 * (Q0_fall_c - Q0_rise2_c) * uC
    # headline Pr = the group's definition (centred single crossing); the gap forms carry
    # a small positive bias from dielectric loss (+0.3 uC/cm^2 on the Sample14 dielectric)
    f["Pr"] = f["Pr_summary"]
    traces["Q_corr"] = Qc_
    f["Pmax"] = 0.5 * (Q.max() - Q.min()) * uC
    f["Pmax_minus_Pr"] = f["Pmax"] - f["Pr"]

    # coercive voltage on the descending branch (Q crosses the centre at V < 0)
    f["Ec_neg"] = _cross_v(V[fall], Q[fall], Qc)
    # pinching on the closed negative half-loop: branch separation at V=0 vs its maximum
    vd, qd = V[fall], Q[fall]  # descending, includes 0 -> V2
    va, qa = V[rise2], Q[rise2]  # ascending V2 -> 0
    grid = np.linspace(V.min() * 0.98, -0.02, 60)
    md = vd <= 0
    qd_i = np.interp(grid, vd[md][::-1], qd[md][::-1])  # vd decreasing -> reverse for interp
    qa_i = np.interp(grid, va, qa)
    dP = (qd_i - qa_i) * uC
    f["loop_width_neg_at0"] = float(dP[-1])
    f["loop_width_neg_max"] = float(dP.max())
    f["pinching_index"] = float(1.0 - dP[-1] / dP.max()) if dP.max() > 0 else np.nan
    f["loop_area_neg"] = float(np.trapezoid(dP, grid))  # uC/cm^2 * V

    # dielectric slopes
    C_sum = _slope(V[fall_pos], Q[fall_pos], *KAPPA_WINDOW_V)
    f["C_summary_F"] = C_sum
    f["kappa_summary"] = kap(C_sum)
    C0 = _slope(V[fall_pos], Q[fall_pos], -KAPPA0_WINDOW_V, KAPPA0_WINDOW_V)
    f["kappa0"] = kap(C0)
    f["dkappa_rel"] = (
        f["kappa_summary"] / f["kappa0"] - 1.0 if np.isfinite(C0) and C0 != 0 else np.nan
    )
    C_neg = _slope(V[rise2], Q[rise2], -KAPPA_WINDOW_V[1], -KAPPA_WINDOW_V[0])
    f["kappa_neg_branch"] = kap(C_neg)
    C_r1 = _slope(V[rise1], Q[rise1], *KAPPA_WINDOW_V)
    f["kappa_rise1"] = kap(C_r1)
    f["Q_end_C"] = float(Q[-1])
    return f, traces
