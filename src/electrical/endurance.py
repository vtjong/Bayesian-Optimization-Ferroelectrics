"""Wake-up and fatigue features from one ``pundEndurance`` log.

The log is one row per cycle count (0, 1, 8, 52, ... 1e6) with the instrument's ramp-only
P, U, N, D charges, so its Psw is the same definition as the group's sheet and the cycle-0
row should agree with the pristine PUND run's ``psw_inst``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from electrical.device import area_cm2
from electrical.issues import IssueCode, ValidationIssue, error
from electrical.workbook import EnduranceTable

#: Ratios are not formed when the reference Psw is below this, uC/cm^2.
WAKEUP_MIN_PSW_UC_CM2 = 0.5


@dataclass(frozen=True, slots=True, kw_only=True)
class EnduranceFeatures:
    """How the switched charge evolved with cycling.

    ``wake_ratio`` is the final Psw over the pristine one, ``fatigue_ratio`` the final over
    the maximum (1 means no fatigue), ``cycles_half_wakeup`` the log-interpolated cycle count
    at half the total gain.
    """

    n_iterations: int
    max_cycles: float
    psw_cycle0_uc_cm2: float = np.nan
    psw_cycle_max_uc_cm2: float = np.nan
    cycles_at_max: float = np.nan
    psw_cycle_final_uc_cm2: float = np.nan
    wake_ratio: float = np.nan
    wake_ratio_max: float = np.nan
    fatigue_ratio: float = np.nan
    psw_gain_uc_cm2: float = np.nan
    cycles_half_wakeup: float = np.nan
    psw_asym_final_uc_cm2: float = np.nan
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True, kw_only=True)
class EnduranceCurve:
    """Psw against cycle count, uC/cm^2."""

    cycles: np.ndarray
    psw_pos_uc_cm2: np.ndarray
    psw_neg_uc_cm2: np.ndarray
    psw_uc_cm2: np.ndarray

    def to_frame(self, round_label, sample: str, capacitor: str) -> pd.DataFrame:
        """Return the curve as rows of ``endurance_curves.csv``."""
        return pd.DataFrame(
            {
                "round": round_label,
                "sample": sample,
                "cap": capacitor,
                "cycles": self.cycles,
                "psw_pos_uc_cm2": self.psw_pos_uc_cm2,
                "psw_neg_uc_cm2": self.psw_neg_uc_cm2,
                "psw_uc_cm2": self.psw_uc_cm2,
            }
        )


def endurance_features(table: EnduranceTable) -> tuple[EnduranceFeatures, EnduranceCurve]:
    """Reduce one endurance log to its wake-up features and its Psw curve."""
    uc = 1e6 / area_cm2(table.source.file.side_um)
    ok = np.isfinite(table.cycles) & np.isfinite(table.p_c) & np.isfinite(table.u_c)
    cycles = table.cycles[ok]
    psw_pos = (table.p_c[ok] - table.u_c[ok]) * uc
    psw_neg = -(table.n_c[ok] - table.d_c[ok]) * uc
    psw = 0.5 * (psw_pos + psw_neg)
    curve = EnduranceCurve(
        cycles=cycles, psw_pos_uc_cm2=psw_pos, psw_neg_uc_cm2=psw_neg, psw_uc_cm2=psw
    )
    n = len(cycles)
    if n < 2:
        issue = error(
            IssueCode.ENDURANCE_TOO_SHORT,
            f"endurance table too short ({n} usable rows)",
            sample_key=table.source.file.sample_key,
            capacitor=table.source.file.capacitor.label,
            path=table.source.file.path,
        )
        return EnduranceFeatures(
            n_iterations=n, max_cycles=float(cycles.max()) if n else np.nan, issues=(issue,)
        ), curve

    k = int(np.argmax(psw))
    first, last, best = float(psw[0]), float(psw[-1]), float(psw[k])
    gain = psw - psw[0]
    cycles_half = np.nan
    if gain[-1] > WAKEUP_MIN_PSW_UC_CM2:
        half = 0.5 * gain[-1]
        j = int(np.argmax(gain >= half))
        if j > 0 and cycles[j] > 0 and cycles[j - 1] >= 0:
            x0, x1 = np.log10(max(cycles[j - 1], 1)), np.log10(cycles[j])
            cycles_half = float(10 ** np.interp(half, [gain[j - 1], gain[j]], [x0, x1]))
        else:
            cycles_half = float(cycles[j])
    features = EnduranceFeatures(
        n_iterations=n,
        max_cycles=float(cycles.max()),
        psw_cycle0_uc_cm2=first,
        psw_cycle_max_uc_cm2=best,
        cycles_at_max=float(cycles[k]),
        psw_cycle_final_uc_cm2=last,
        wake_ratio=last / first if abs(first) > WAKEUP_MIN_PSW_UC_CM2 else np.nan,
        wake_ratio_max=best / first if abs(first) > WAKEUP_MIN_PSW_UC_CM2 else np.nan,
        fatigue_ratio=last / best if abs(best) > WAKEUP_MIN_PSW_UC_CM2 else np.nan,
        psw_gain_uc_cm2=last - first,
        cycles_half_wakeup=cycles_half,
        psw_asym_final_uc_cm2=float(psw_pos[-1] - psw_neg[-1]),
    )
    return features, curve
