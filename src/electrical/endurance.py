"""Endurance (wake-up / fatigue) features from one Keithley `pundEndurance` export.

The Data sheet is a short table: one row per logged iteration (cycle count 0, 1, 8,
52, ... 1e6) with the instrument's ramp-integrated P, Pa, U, Ua, N, Na, D, Da, Psw,
Qsw in coulombs.  Psw here is the same ramp-only (P − U) the summary sheet uses, so
iteration 0 should agree with the pristine PUND file's `Psw_inst`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from electrical.device import area_cm2


def endurance_features(df: pd.DataFrame, size_um: int | None = 30) -> tuple[dict, pd.DataFrame]:
    A = area_cm2(size_um)
    uC = 1e6 / A
    it = pd.to_numeric(df["iteration"], errors="coerce").to_numpy(float)
    P, U, N, D = (
        pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in ("P", "U", "N", "D")
    )
    ok = np.isfinite(it) & np.isfinite(P) & np.isfinite(U)
    it, P, U, N, D = it[ok], P[ok], U[ok], N[ok], D[ok]
    curve = pd.DataFrame({"iteration": it, "Psw_pos": (P - U) * uC, "Psw_neg": -(N - D) * uC})
    curve["Psw"] = 0.5 * (curve["Psw_pos"] + curve["Psw_neg"])
    f: dict = {"n_iterations": int(len(it)), "max_cycles": float(it.max()) if len(it) else np.nan}
    if len(it) < 2:
        f["problem_end"] = "endurance table too short"
        return f, curve
    psw = curve["Psw"].to_numpy()
    k = int(np.argmax(psw))
    f["Psw_cyc0"] = float(psw[0])
    f["Psw_cyc_max"] = float(psw[k])
    f["cycles_at_max"] = float(it[k])
    f["Psw_cyc_final"] = float(psw[-1])
    f["wake_ratio"] = float(psw[-1] / psw[0]) if abs(psw[0]) > 0.5 else np.nan  # final / pristine
    f["wake_ratio_max"] = float(psw[k] / psw[0]) if abs(psw[0]) > 0.5 else np.nan
    f["fatigue_ratio"] = float(psw[-1] / psw[k]) if abs(psw[k]) > 0.5 else np.nan  # 1 = no fatigue
    f["Psw_gain_abs"] = float(psw[-1] - psw[0])
    # cycles to reach half of the total wake-up gain (log-interpolated)
    gain = psw - psw[0]
    if gain[-1] > 0.5:
        half = 0.5 * gain[-1]
        j = int(np.argmax(gain >= half))
        if j > 0 and it[j] > 0 and it[j - 1] >= 0:
            x0, x1 = np.log10(max(it[j - 1], 1)), np.log10(it[j])
            f["cycles_half_wakeup"] = float(10 ** np.interp(half, [gain[j - 1], gain[j]], [x0, x1]))
        else:
            f["cycles_half_wakeup"] = float(it[j])
    else:
        f["cycles_half_wakeup"] = np.nan
    f["Psw_asym_cyc_final"] = float(curve["Psw_pos"].iloc[-1] - curve["Psw_neg"].iloc[-1])
    return f, curve
