"""P-V (really Q-V) features from one ``doubleSweepSeg`` run.

The sweep is a single 0 -> +V1 -> 0 -> V2 -> 0 triangle from the pristine state, so the
first quarter starts from an unpoled film and the closed part of the loop is +V1 -> 0 -> V2
-> 0. ``Charge`` is the instrument's running integral of ``Imeas``.

Two features reproduce the group's summary sheet: ``pr`` is the charge at V = 0 on the
+V1 -> 0 branch measured from the loop centre (Qmax + Qmin) / 2, and ``kappa_summary`` is
the slope of Q against V over 0.8-2.25 V on that branch, scaled by t / (eps0 A). On an
imprinted loop the two V = 0 crossings differ, so ``pr_gap`` (half their separation) is
carried beside ``pr`` for comparison with the switched charge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from electrical.device import EPS0, T_HZO_M, area_cm2
from electrical.issues import IssueCode, ValidationIssue, error
from electrical.workbook import PvRun

#: The group's dielectric-fit window on the +V1 -> 0 branch, V.
KAPPA_WINDOW_V = (0.8, 2.25)
#: |V| below this is the zero-field slope.
KAPPA0_WINDOW_V = 0.3
#: |V| below this is a rest period, used to estimate the instrument's current offset.
REST_VOLTAGE_V = 0.05
#: Samples either side of the sweep kept out of the rest estimate.
REST_GUARD_SAMPLES = 50
REST_MIN_SAMPLES = 20
#: The negative half-loop is compared on this many voltages between V2 and zero.
LOOP_GRID_POINTS = 60


@dataclass(frozen=True, slots=True, kw_only=True)
class PvSweep:
    """What was applied and the loop's gross position."""

    v_max_v: float
    v_min_v: float
    pv_dt_s: float
    pv_n_points: int
    q_centre_c: float = np.nan
    q_end_c: float = np.nan
    i_offset_a: float = np.nan


@dataclass(frozen=True, slots=True, kw_only=True)
class Remanence:
    """Remanent and maximum polarization, uC/cm^2.

    ``pr`` is the group's definition (centred crossing on the +V1 -> 0 branch); ``pr_neg`` the
    same on the V2 -> 0 branch; ``pr_gap`` half the V = 0 separation of the two branches,
    raw and with the instrument's DC current offset removed.
    """

    pr_uc_cm2: float
    pr_neg_uc_cm2: float
    pr_gap_uc_cm2: float
    pr_gap_corrected_uc_cm2: float
    pmax_uc_cm2: float
    pmax_minus_pr_uc_cm2: float


@dataclass(frozen=True, slots=True, kw_only=True)
class NegativeHalfLoop:
    """The closed negative half-loop: coercive voltage, opening and pinching."""

    ec_neg_v: float
    loop_width_at_zero_uc_cm2: float
    loop_width_max_uc_cm2: float
    pinching_index: float
    loop_area_uc_v_cm2: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PvDielectric:
    """Permittivity from the Q-V slope: the group's window, zero field, and the other branches."""

    c_summary_f: float
    kappa_summary: float
    kappa0: float
    dkappa_rel: float
    kappa_neg_branch: float
    kappa_rise1: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PvFeatures:
    """Everything one P-V run yields; groups are ``None`` when the sweep is not positive-first."""

    sweep: PvSweep
    remanence: Remanence | None = None
    half_loop: NegativeHalfLoop | None = None
    dielectric: PvDielectric | None = None
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True, kw_only=True)
class PvTraces:
    """Arrays the QC report plots."""

    voltage_v: np.ndarray
    charge_c: np.ndarray
    current_a: np.ndarray
    time_s: np.ndarray
    area_cm2: float
    charge_corrected_c: np.ndarray | None = None


def charge_at_zero_crossing(v: np.ndarray, q: np.ndarray, which: str) -> float:
    """Return q interpolated where v crosses zero, at the first or last crossing on the slice."""
    s = np.sign(v)
    k = np.flatnonzero(np.diff(s) != 0)
    if len(k) == 0:
        return np.nan
    k = k[0] if which == "first" else k[-1]
    v0, v1, q0, q1 = v[k], v[k + 1], q[k], q[k + 1]
    if v1 == v0:
        return float(q0)
    return float(q0 + (0.0 - v0) * (q1 - q0) / (v1 - v0))


def voltage_at_charge(v: np.ndarray, q: np.ndarray, level: float) -> float:
    """Return v interpolated where q first crosses ``level`` on the slice."""
    s = np.sign(q - level)
    k = np.flatnonzero(np.diff(s) != 0)
    if len(k) == 0:
        return np.nan
    k = k[0]
    q0, q1, v0, v1 = q[k], q[k + 1], v[k], v[k + 1]
    if q1 == q0:
        return float(v0)
    return float(v0 + (level - q0) * (v1 - v0) / (q1 - q0))


def slope(v: np.ndarray, q: np.ndarray, lo: float, hi: float) -> float:
    """Return dQ/dV fitted where lo <= v <= hi, or NaN with fewer than five points."""
    m = (v >= lo) & (v <= hi)
    if m.sum() < 5:
        return np.nan
    return float(np.polyfit(v[m], q[m], 1)[0])


def pv_features(run: PvRun) -> tuple[PvFeatures, PvTraces]:
    """Reduce one P-V run to its features and the traces the QC report plots."""
    area = area_cm2(run.source.file.side_um)
    area_m2 = area * 1e-4
    uc = 1e6 / area
    kappa_of = lambda c: float(c * T_HZO_M / (EPS0 * area_m2))  # noqa: E731
    v, q, i, t = run.voltage_v, run.charge_c, run.current_a, run.time_s
    traces = PvTraces(voltage_v=v, charge_c=q, current_a=i, time_s=t, area_cm2=area)
    sweep = dict(
        v_max_v=float(v.max()), v_min_v=float(v.min()), pv_dt_s=run.dt_s, pv_n_points=len(t)
    )

    i_max, i_min = int(np.argmax(v)), int(np.argmin(v))
    if not i_max < i_min:
        issue = error(
            IssueCode.SWEEP_NOT_POSITIVE_FIRST,
            "sweep is not positive-first; branch logic assumes 0 -> +V -> 0 -> -V -> 0",
            sample_key=run.source.file.sample_key,
            capacitor=run.source.file.capacitor.label,
            path=run.source.file.path,
            location=run.source.sheet,
        )
        return PvFeatures(sweep=PvSweep(**sweep), issues=(issue,)), traces

    rise1 = slice(0, i_max + 1)  # 0 -> +V1 from the pristine state
    fall = slice(i_max, i_min + 1)  # +V1 -> 0 -> V2
    rise2 = slice(i_min, len(v))  # V2 -> 0
    fall_pos = slice(i_max, i_max + int(np.argmin(np.abs(v[fall]))) + 1)  # +V1 -> 0 only

    q_centre = 0.5 * (q.max() + q.min())
    q0_fall = charge_at_zero_crossing(v[fall], q[fall], "first")
    q0_rise2 = (
        charge_at_zero_crossing(v[rise2], q[rise2], "last")
        if np.any(v[rise2] > 0)
        else float(q[-1])
    )
    pr = (q0_fall - q_centre) * uc

    # The instrument integrates a small DC current offset over the whole record; estimate it
    # on the rest periods either side of the sweep and re-integrate without it.
    rest = np.abs(v) < REST_VOLTAGE_V
    rest[i_max - REST_GUARD_SAMPLES : i_min + REST_GUARD_SAMPLES] = False
    i_offset = float(np.median(i[rest])) if rest.sum() > REST_MIN_SAMPLES else 0.0
    q_corrected = np.concatenate(
        [[0.0], np.cumsum(0.5 * (i[1:] + i[:-1] - 2 * i_offset) * np.diff(t))]
    )
    q0_fall_c = charge_at_zero_crossing(v[fall], q_corrected[fall], "first")
    if np.any(v[rise2] > 0):
        q0_rise2_c = charge_at_zero_crossing(v[rise2], q_corrected[rise2], "last")
    else:
        q0_rise2_c = float(q_corrected[-1])
    pmax = 0.5 * (q.max() - q.min()) * uc
    remanence = Remanence(
        pr_uc_cm2=pr,
        pr_neg_uc_cm2=(q0_rise2 - q_centre) * uc,
        pr_gap_uc_cm2=0.5 * (q0_fall - q0_rise2) * uc,
        pr_gap_corrected_uc_cm2=0.5 * (q0_fall_c - q0_rise2_c) * uc,
        pmax_uc_cm2=pmax,
        pmax_minus_pr_uc_cm2=pmax - pr,
    )

    # The negative half-loop: descending branch against ascending branch on a common grid.
    vd, qd = v[fall], q[fall]
    va, qa = v[rise2], q[rise2]
    grid = np.linspace(v.min() * 0.98, -0.02, LOOP_GRID_POINTS)
    md = vd <= 0
    qd_i = np.interp(grid, vd[md][::-1], qd[md][::-1])  # vd decreasing -> reverse for interp
    qa_i = np.interp(grid, va, qa)
    opening = (qd_i - qa_i) * uc
    half_loop = NegativeHalfLoop(
        ec_neg_v=voltage_at_charge(v[fall], q[fall], q_centre),
        loop_width_at_zero_uc_cm2=float(opening[-1]),
        loop_width_max_uc_cm2=float(opening.max()),
        pinching_index=float(1.0 - opening[-1] / opening.max()) if opening.max() > 0 else np.nan,
        loop_area_uc_v_cm2=float(np.trapezoid(opening, grid)),
    )

    c_summary = slope(v[fall_pos], q[fall_pos], *KAPPA_WINDOW_V)
    c_zero = slope(v[fall_pos], q[fall_pos], -KAPPA0_WINDOW_V, KAPPA0_WINDOW_V)
    kappa_summary = kappa_of(c_summary)
    kappa0 = kappa_of(c_zero)
    dielectric = PvDielectric(
        c_summary_f=c_summary,
        kappa_summary=kappa_summary,
        kappa0=kappa0,
        dkappa_rel=kappa_summary / kappa0 - 1.0 if np.isfinite(c_zero) and c_zero != 0 else np.nan,
        kappa_neg_branch=kappa_of(
            slope(v[rise2], q[rise2], -KAPPA_WINDOW_V[1], -KAPPA_WINDOW_V[0])
        ),
        kappa_rise1=kappa_of(slope(v[rise1], q[rise1], *KAPPA_WINDOW_V)),
    )
    features = PvFeatures(
        sweep=PvSweep(**sweep, q_centre_c=q_centre, q_end_c=float(q[-1]), i_offset_a=i_offset),
        remanence=remanence,
        half_loop=half_loop,
        dielectric=dielectric,
    )
    traces = PvTraces(
        voltage_v=v,
        charge_c=q,
        current_a=i,
        time_s=t,
        area_cm2=area,
        charge_corrected_c=q_corrected,
    )
    return features, traces
