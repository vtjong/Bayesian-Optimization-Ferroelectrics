"""PUND features from one Keithley `pundTest` export.

Waveform seen on Sample14: four trapezoid pulses P, U (+Vp) and N, D (-Vp) with
rise / plateau / fall of trf / tp / trf and gaps of td; no preset pulse (the P-V
sweep run just before leaves the film negatively poled).  Everything below is
derived from V(t) and I(t); the instrument's own ramp-integrated scalars on the
first data row are kept as `*_inst` for regression against the group's summary.

Sign convention: charges are reported as magnitudes in uC/cm^2 unless the name
says otherwise; voltages keep their sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from electrical.device import EPS0, T_HZO_M, area_cm2

INSTRUMENT_SCALARS = ["P", "Pa", "U", "Ua", "N", "Na", "D", "Da", "Psw", "Qsw"]


@dataclass
class Pulse:
    label: str  # P, U, N, D (or X for unexpected extras)
    sign: int
    i0: int  # first sample of the pulse (|V| crosses threshold)
    i_ramp_end: int  # first sample at the plateau level
    i_plat_end: int  # last sample at the plateau level
    i1: int  # last sample of the pulse
    vp: float

    @property
    def rise(self):
        return slice(self.i0, self.i_ramp_end + 1)

    @property
    def plateau(self):
        return slice(self.i_ramp_end, self.i_plat_end + 1)

    @property
    def fall(self):
        return slice(self.i_plat_end, self.i1 + 1)

    @property
    def whole(self):
        return slice(self.i0, self.i1 + 1)


def segment_pulses(
    V: np.ndarray,
    thresh: float = 0.15,
    min_run: int = 100,
    med_k: int = 5,
    plateau_frac: float = 0.95,
) -> list[Pulse]:
    """Pulses as runs of |median-filtered V| > thresh lasting at least min_run samples."""
    Vm = median_filter(V, size=med_k, mode="nearest")
    on = np.abs(Vm) > thresh
    edges = np.flatnonzero(np.diff(on.astype(int))) + 1
    runs = [r for r in np.split(np.arange(len(V)), edges) if on[r[0]] and len(r) >= min_run]
    pulses = []
    for r in runs:
        i0, i1 = int(r[0]), int(r[-1])
        seg = Vm[i0 : i1 + 1]
        sign = 1 if seg.mean() > 0 else -1
        vp = float(np.max(np.abs(seg)))
        at_plateau = np.flatnonzero(np.abs(seg) >= plateau_frac * vp)
        i_ramp_end = i0 + int(at_plateau[0])
        i_plat_end = i0 + int(at_plateau[-1])
        pulses.append(Pulse("X", sign, i0, i_ramp_end, i_plat_end, i1, vp))
    # Label: expected pattern (+,+,-,-); with a 5th leading pulse treat it as a preset.
    signs = [p.sign for p in pulses]
    labels = None
    if signs == [1, 1, -1, -1]:
        labels = ["P", "U", "N", "D"]
    elif signs == [-1, -1, 1, 1]:
        labels = ["N", "D", "P", "U"]
    elif len(signs) == 5 and signs[1:] in ([1, 1, -1, -1], [-1, -1, 1, 1]):
        labels = ["preset"] + (["P", "U", "N", "D"] if signs[1] == 1 else ["N", "D", "P", "U"])
    if labels:
        for p, lab in zip(pulses, labels):
            p.label = lab
    return pulses


def _trapz(y, x):
    return float(np.trapezoid(y, x))


def _smooth(current: np.ndarray, dt: float, window_s: float = 100e-9) -> np.ndarray:
    w = max(5, int(round(window_s / dt)) | 1)
    w = min(w, len(current) - (1 - len(current) % 2))
    return savgol_filter(current, w, 2) if len(current) > w else current


def partition_ramp(
    current: np.ndarray, t: np.ndarray, sign: int, dt: float, fallback_frac: float = 0.65
):
    """Round-1 slide-7 rule on a rising-ramp current.

    Smooth; main peak = largest |current| in the first 70 % of the ramp; late shoulder =
    the largest |current| after the main peak that follows a dip; split = the dip (local
    minimum of |current| between the two).  Without a dip the split falls back to a fixed
    fraction of the ramp and the rule is reported as 'fallback'.
    Returns Q1, Q2 (signed, C), t_split (s), rule, peak index.
    """
    mag = sign * _smooth(current, dt)  # positive-going switching lobe
    n = len(mag)
    if n < 10:
        return np.nan, np.nan, np.nan, "too-short", -1
    k_peak = int(np.argmax(mag[: max(5, int(0.7 * n))]))
    after = mag[k_peak:]
    # local minima after the peak
    dips = np.flatnonzero((after[1:-1] < after[:-2]) & (after[1:-1] <= after[2:])) + 1
    rule = "fallback"
    k_split = min(n - 1, int(round(fallback_frac * n)))
    for d in dips:
        rest = after[d:]
        if len(rest) > 3 and rest.max() > after[d] * 1.02 and after[d] > 0.15 * mag[k_peak]:
            k_split, rule = k_peak + int(d), "dip"
            break
    Q1 = _trapz(current[: k_split + 1], t[: k_split + 1])
    Q2 = _trapz(current[k_split:], t[k_split:])
    return Q1, Q2, float(t[k_split]), rule, k_peak


def pund_features(df: pd.DataFrame, size_um: int | None = 30) -> tuple[dict, dict]:
    """Features (dict) and traces (dict of arrays for QC plots) from a Data sheet."""
    A = area_cm2(size_um)
    A_m2 = A * 1e-4
    V = df["V"].astype(float).to_numpy()
    current = df["I"].astype(float).to_numpy()
    t = df["t"].astype(float).to_numpy()
    dt = float(np.median(np.diff(t)))
    f: dict = {"area_cm2": A, "dt_s": dt, "n_points": len(t)}
    for k in INSTRUMENT_SCALARS:
        f[f"{k}_inst_C"] = float(df[k].iloc[0]) if k in df.columns else np.nan

    pulses = segment_pulses(V)
    f["n_pulses"] = len(pulses)
    f["pulse_pattern"] = "".join(p.label[0] for p in pulses)
    by = {p.label: p for p in pulses}
    traces = {"t": t, "V": V, "I": current, "pulses": pulses}
    if not {"P", "U", "N", "D"} <= by.keys():
        f["problem"] = f"unexpected pulse pattern {[p.sign for p in pulses]}"
        return f, traces

    first = min(p.i0 for p in pulses)
    base = float(current[: max(10, first - 20)].mean()) if first > 30 else 0.0
    f["I_baseline_A"] = base
    Ic = current - base
    vp = by["P"].vp
    f["Vp_V"] = vp
    ramp_s = t[by["P"].i_ramp_end] - t[by["P"].i0]
    f["t_rise_s"] = ramp_s
    f["t_plateau_s"] = t[by["P"].i_plat_end] - t[by["P"].i_ramp_end]

    # per-pulse integrals
    Q = {}
    for lab, p in by.items():
        for part in ("rise", "plateau", "fall", "whole"):
            sl = getattr(p, part)
            Q[(lab, part)] = _trapz(Ic[sl], t[sl])
            f[f"Q_{lab}_{part}_C"] = Q[(lab, part)]
    uC = 1e6 / A  # C -> uC/cm^2

    # switched polarization, whole-pulse (dielectric displacement cancels rise vs fall)
    f["Psw_pos"] = (Q[("P", "whole")] - Q[("U", "whole")]) * uC
    f["Psw_neg"] = -(Q[("N", "whole")] - Q[("D", "whole")]) * uC
    f["Psw"] = 0.5 * (f["Psw_pos"] + f["Psw_neg"])
    # ramp-only version (what the instrument reports) and the instrument value itself
    f["Psw_pos_ramp"] = (Q[("P", "rise")] - Q[("U", "rise")]) * uC
    f["Psw_neg_ramp"] = -(Q[("N", "rise")] - Q[("D", "rise")]) * uC
    f["Psw_inst"] = (f["P_inst_C"] - f["U_inst_C"]) * uC
    f["Psw_inst_neg"] = -(f["N_inst_C"] - f["D_inst_C"]) * uC
    # raw N ramp charge / 2A, the Round-1 'Ptotal' style quantity
    f["P_raw_N_ramp"] = -Q[("N", "rise")] * uC
    f["P_raw_P_ramp"] = Q[("P", "rise")] * uC
    f["Psw_asym"] = f["Psw_pos"] - f["Psw_neg"]

    # leakage: last 20 % of the U and D plateaus
    def tail_mean(p):
        sl = p.plateau
        n = sl.stop - sl.start
        stop = sl.stop - min(10, n // 10)  # stay clear of the fall onset
        return float(np.median(Ic[stop - max(5, n // 5) : stop]))

    f["J_leak_pos_A_cm2"] = tail_mean(by["U"]) / A
    f["J_leak_neg_A_cm2"] = tail_mean(by["D"]) / A
    f["J_leak_A_cm2"] = 0.5 * (abs(f["J_leak_pos_A_cm2"]) + abs(f["J_leak_neg_A_cm2"]))

    # dielectric capacitance from the non-switching ramps (mean current / slew rate)
    def ramp_C(p):
        sl = p.rise
        v = V[sl]
        i = Ic[sl]
        tt = t[sl]
        m = (np.abs(v) > 0.3 * vp) & (np.abs(v) < 0.85 * vp)
        if m.sum() < 5:
            return np.nan
        slew = abs(np.polyfit(tt[m], v[m], 1)[0])  # measured dV/dt on the ramp
        return float(np.abs(i[m].mean()) / slew)

    C_U, C_D = ramp_C(by["U"]), ramp_C(by["D"])
    f["C_U_F"], f["C_D_F"] = C_U, C_D
    f["kappa_PUND"] = float(np.nanmean([C_U, C_D]) * T_HZO_M / (EPS0 * A_m2))
    f["kappa_inst_U"] = float(f["U_inst_C"] / vp * T_HZO_M / (EPS0 * A_m2))

    # switching-current traces (U/D-subtracted), aligned at pulse start
    def sub(a, b):
        n = min(a.i1 - a.i0, b.i1 - b.i0) + 1
        return (
            Ic[a.i0 : a.i0 + n] - Ic[b.i0 : b.i0 + n],
            t[a.i0 : a.i0 + n] - t[a.i0],
            V[a.i0 : a.i0 + n],
        )

    Isw_p, tp_, Vp_ = sub(by["P"], by["U"])
    Isw_n, tn_, Vn_ = sub(by["N"], by["D"])
    traces.update(
        {
            "Isw_pos": Isw_p,
            "Isw_neg": Isw_n,
            "t_rel_pos": tp_,
            "t_rel_neg": tn_,
            "V_P": Vp_,
            "V_N": Vn_,
        }
    )

    # dynamic coercive voltage: V at the switching-current peak (ramp + plateau)
    def peak(Isw, tt, VV, sign, p):
        n = p.i_plat_end - p.i0 + 1
        s = sign * _smooth(Isw[:n], dt)
        k = int(np.argmax(s))
        return float(s[k]), float(tt[k]), float(VV[k])

    f["Isw_peak_pos_A"], f["t_peak_pos_s"], f["Vc_pos"] = peak(Isw_p, tp_, Vp_, 1, by["P"])
    f["Isw_peak_neg_A"], f["t_peak_neg_s"], f["Vc_neg"] = peak(Isw_n, tn_, Vn_, -1, by["N"])
    f["imprint_PUND"] = 0.5 * (f["Vc_pos"] + f["Vc_neg"])
    f["Vc_mean"] = 0.5 * (f["Vc_pos"] - f["Vc_neg"])

    # Round-1 partition on the N ramp: raw current (prototype) and U/D-subtracted
    pN = by["N"]
    rs = pN.rise
    Q1, Q2, ts, rule, kpk = partition_ramp(current[rs], t[rs], -1, dt)
    f["QO_rawN_C"], f["QT_rawN_C"], f["t_split_rawN_s"], f["split_rule_rawN"] = Q1, Q2, ts, rule
    f["QO_frac_rawN"] = Q1 / (Q1 + Q2) if (Q1 + Q2) != 0 else np.nan
    f["P_rawN_ramp_total"] = -(Q1 + Q2) * uC
    nr = pN.i_ramp_end - pN.i0 + 1
    Q1s, Q2s, tss, rule_s, _ = partition_ramp(Isw_n[:nr], tn_[:nr] + t[pN.i0], -1, dt)
    f["QO_sub_neg_C"], f["QT_sub_neg_C"], f["t_split_sub_neg_s"], f["split_rule_sub_neg"] = (
        Q1s,
        Q2s,
        tss,
        rule_s,
    )
    f["QO_frac_sub_neg"] = Q1s / (Q1s + Q2s) if (Q1s + Q2s) != 0 else np.nan
    pP = by["P"]
    nrp = pP.i_ramp_end - pP.i0 + 1
    Q1p, Q2p, tsp, rule_p, _ = partition_ramp(Isw_p[:nrp], tp_[:nrp] + t[pP.i0], 1, dt)
    f["QO_sub_pos_C"], f["QT_sub_pos_C"], f["t_split_sub_pos_s"], f["split_rule_sub_pos"] = (
        Q1p,
        Q2p,
        tsp,
        rule_p,
    )
    f["QO_frac_sub_pos"] = Q1p / (Q1p + Q2p) if (Q1p + Q2p) != 0 else np.nan
    # charge continuing on the plateau after the ramp (what ramp-only integration misses)
    f["Psw_plateau_pos"] = (Q[("P", "plateau")] - Q[("U", "plateau")]) * uC
    f["Psw_plateau_neg"] = -(Q[("N", "plateau")] - Q[("D", "plateau")]) * uC

    f["switching"] = bool(f["Psw"] > 2.0)
    return f, traces
