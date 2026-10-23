"""Extract P-V loop features: remanent polarization, coercive voltage, pinching, κ from the slope.

The sweep is a single 0 -> +V1 -> 0 -> V2 -> 0 triangle from the pristine state, so the first
quarter starts from an unpoled film and the closed part of the loop is +V1 -> 0 -> V2 -> 0.
``Charge`` is the instrument's running integral of ``Imeas``.

The group's definitions, confirmed against their summary sheet and the Q-V plots their script
draws for every capacitor: ``pr`` is the charge at V = 0 on the +V1 -> 0 branch measured from
the loop centre (Qmax + Qmin) / 2; ``ec_pos`` and ``ec_neg`` are the voltages where that
centred charge crosses zero on the first 0 -> +V1 rise and on the +V1 -> V2 fall; and
``kappa_summary`` is the slope of Q against V over 0.8-2.25 V on the +V1 -> 0 branch, scaled
by t / (eps0 A). On the pristine run the first rise starts from the as-deposited film, so
``ec_pos`` there is not the mirror of ``ec_neg``. On an imprinted loop the two V = 0 crossings
differ, so ``pr_gap`` (half their separation) is carried beside ``pr`` for comparison with
the switched charge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electrical.convert_units import area_cm2, kappa_from_capacitance, uc_cm2_per_coulomb
from electrical.flag_issues import Issue, IssueCode, error
from electrical.read_measurement_runs import PvRun

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
    same on the V2 -> 0 branch; ``pr_gap`` half the V = 0 separation of the two branches, raw
    and with the instrument's DC current offset removed.
    """

    pr_uc_cm2: float
    pr_neg_uc_cm2: float
    pr_gap_uc_cm2: float
    pr_gap_corrected_uc_cm2: float
    pmax_uc_cm2: float
    pmax_minus_pr_uc_cm2: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CoerciveVoltage:
    """Voltages where the centred charge crosses zero: on the first rise and on the fall."""

    ec_pos_v: float
    ec_neg_v: float


@dataclass(frozen=True, slots=True, kw_only=True)
class NegativeHalfLoop:
    """The closed negative half-loop: opening and pinching."""

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
    coercive: CoerciveVoltage | None = None
    half_loop: NegativeHalfLoop | None = None
    dielectric: PvDielectric | None = None
    issues: tuple[Issue, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class PvTraces:
    """Arrays the QC report plots."""

    voltage_v: np.ndarray
    charge_c: np.ndarray
    current_a: np.ndarray
    time_s: np.ndarray
    area_cm2: float
    charge_corrected_c: np.ndarray | None = None


def _value_at_crossing(
    signal: np.ndarray, values: np.ndarray, level: float = 0.0, which: str = "first"
) -> float:
    """Return ``values`` interpolated where ``signal`` crosses ``level`` (first or last time)."""
    k = np.flatnonzero(np.diff(np.sign(signal - level)) != 0)
    if len(k) == 0:
        return np.nan
    k = k[0] if which == "first" else k[-1]
    s0, s1, v0, v1 = signal[k], signal[k + 1], values[k], values[k + 1]
    if s1 == s0:
        return float(v0)
    return float(v0 + (level - s0) * (v1 - v0) / (s1 - s0))


def _fitted_slope(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Return dy/dx fitted where lo <= x <= hi, or NaN with fewer than five points."""
    m = (x >= lo) & (x <= hi)
    if m.sum() < 5:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


def extract_pv_features(run: PvRun) -> tuple[PvFeatures, PvTraces]:
    """Reduce one P-V run to its features and the traces the QC report plots."""
    area = area_cm2(run.source.file.side_um)
    scale = uc_cm2_per_coulomb(area)
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
            **run.source.where(),
        )
        return PvFeatures(sweep=PvSweep(**sweep), issues=(issue,)), traces

    rise1 = slice(0, i_max + 1)  # 0 -> +V1 from the pristine state
    fall = slice(i_max, i_min + 1)  # +V1 -> 0 -> V2
    rise2 = slice(i_min, len(v))  # V2 -> 0
    fall_pos = slice(i_max, i_max + int(np.argmin(np.abs(v[fall]))) + 1)  # +V1 -> 0 only

    def zero_field_charges(charge: np.ndarray) -> tuple[float, float]:
        """Charge at V = 0 on the descending branch and on the closing ascending branch."""
        on_fall = _value_at_crossing(v[fall], charge[fall])
        if np.any(v[rise2] > 0):
            return on_fall, _value_at_crossing(v[rise2], charge[rise2], which="last")
        return on_fall, float(charge[-1])

    q_centre = 0.5 * (q.max() + q.min())
    q0_fall, q0_rise2 = zero_field_charges(q)
    pr = (q0_fall - q_centre) * scale

    # The instrument integrates a small DC current offset over the whole record; estimate it
    # on the rest periods either side of the sweep and re-integrate without it.
    rest = np.abs(v) < REST_VOLTAGE_V
    rest[i_max - REST_GUARD_SAMPLES : i_min + REST_GUARD_SAMPLES] = False
    i_offset = float(np.median(i[rest])) if rest.sum() > REST_MIN_SAMPLES else 0.0
    q_corrected = np.concatenate(
        [[0.0], np.cumsum(0.5 * (i[1:] + i[:-1] - 2 * i_offset) * np.diff(t))]
    )
    q0_fall_c, q0_rise2_c = zero_field_charges(q_corrected)
    pmax = 0.5 * (q.max() - q.min()) * scale
    remanence = Remanence(
        pr_uc_cm2=pr,
        pr_neg_uc_cm2=(q0_rise2 - q_centre) * scale,
        pr_gap_uc_cm2=0.5 * (q0_fall - q0_rise2) * scale,
        pr_gap_corrected_uc_cm2=0.5 * (q0_fall_c - q0_rise2_c) * scale,
        pmax_uc_cm2=pmax,
        pmax_minus_pr_uc_cm2=pmax - pr,
    )

    # The negative half-loop: descending branch against ascending branch on a common grid.
    vd, qd = v[fall], q[fall]
    va, qa = v[rise2], q[rise2]
    grid = np.linspace(v.min() * 0.98, -0.02, LOOP_GRID_POINTS)
    descending = vd <= 0
    qd_i = np.interp(
        grid, vd[descending][::-1], qd[descending][::-1]
    )  # vd falls; interp needs rising
    qa_i = np.interp(grid, va, qa)
    opening = (qd_i - qa_i) * scale
    coercive = CoerciveVoltage(
        ec_pos_v=_value_at_crossing(q[rise1], v[rise1], q_centre),
        ec_neg_v=_value_at_crossing(q[fall], v[fall], q_centre),
    )
    half_loop = NegativeHalfLoop(
        loop_width_at_zero_uc_cm2=float(opening[-1]),
        loop_width_max_uc_cm2=float(opening.max()),
        pinching_index=float(1.0 - opening[-1] / opening.max()) if opening.max() > 0 else np.nan,
        loop_area_uc_v_cm2=float(np.trapezoid(opening, grid)),
    )

    c_summary = _fitted_slope(v[fall_pos], q[fall_pos], *KAPPA_WINDOW_V)
    c_zero = _fitted_slope(v[fall_pos], q[fall_pos], -KAPPA0_WINDOW_V, KAPPA0_WINDOW_V)
    c_neg = _fitted_slope(v[rise2], q[rise2], -KAPPA_WINDOW_V[1], -KAPPA_WINDOW_V[0])
    c_rise1 = _fitted_slope(v[rise1], q[rise1], *KAPPA_WINDOW_V)
    kappa_summary = kappa_from_capacitance(c_summary, area)
    kappa0 = kappa_from_capacitance(c_zero, area)
    dielectric = PvDielectric(
        c_summary_f=c_summary,
        kappa_summary=kappa_summary,
        kappa0=kappa0,
        dkappa_rel=kappa_summary / kappa0 - 1.0 if np.isfinite(c_zero) and c_zero != 0 else np.nan,
        kappa_neg_branch=kappa_from_capacitance(c_neg, area),
        kappa_rise1=kappa_from_capacitance(c_rise1, area),
    )
    features = PvFeatures(
        sweep=PvSweep(**sweep, q_centre_c=q_centre, q_end_c=float(q[-1]), i_offset_a=i_offset),
        remanence=remanence,
        coercive=coercive,
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
