"""Extract PUND features: switched polarization, O / T charge split, coercive voltage, leakage, κ.

The waveform is four trapezoid pulses P, U (+Vp) and N, D (-Vp), each rise / plateau / fall,
with no preset pulse: the P-V sweep run just before leaves the film negatively poled. Every
feature comes from V(t) and I(t); the instrument's own ramp-integrated charges are carried
beside them because the group's summary sheet is built from those.

Charges are magnitudes in uC/cm^2 unless the name says otherwise; voltages keep their sign.
On a switching film the N-ramp current has a main peak and a late shoulder, and the charge on
either side of the split between them is the O / T partition the correlation study asks for.
Three split rules are reported side by side: the dip rule behind the Round-1 summary slide and
the two rules in the group's MATLAB scripts (``PeakSplit_dIdt_*.m``). All three work on the
ramp as the instrument programmed it (9.0 to 10.0 us for the N pulse), the window the scripts
and the slide use; the scripts' time windows are expressed relative to the start of that ramp.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from electrical.convert_units import area_cm2, kappa_from_capacitance, uc_cm2_per_coulomb
from electrical.flag_issues import Issue, IssueCode, error, warning
from electrical.read_measurement_runs import InstrumentCharges, PundRun

#: A pulse is a run of |median-filtered V| above this, lasting at least the minimum samples.
PULSE_THRESHOLD_V = 0.15
PULSE_MIN_RUN_SAMPLES = 100
PULSE_MEDIAN_SAMPLES = 5
#: The plateau is where |V| is within this fraction of the pulse's peak.
PLATEAU_FRACTION = 0.95
#: Baseline current is the mean before the first pulse, leaving a gap before it.
BASELINE_GAP_SAMPLES = 20
BASELINE_MIN_SAMPLES = 10
BASELINE_NEEDS_SAMPLES = 30
#: Savitzky-Golay smoothing of the switching current for peak finding and the dip rule.
SMOOTH_WINDOW_S = 100e-9
SMOOTH_ORDER = 2
#: Dip rule: main peak in the first part of the ramp, shoulder after a dip that rises again.
DIP_PEAK_SEARCH_FRACTION = 0.7
DIP_SHOULDER_RISE = 1.02
DIP_FLOOR_FRACTION = 0.15
#: Where the split lands when a rule finds nothing, as a fraction of the ramp.
SPLIT_FALLBACK_FRACTION = 0.65
#: The group's crossover rule: smoothing, and the window after ramp start for the dI/dt zero.
CROSSOVER_SG_ORDER = 3
CROSSOVER_SG_FRAME = 41
CROSSOVER_WINDOW_S = (0.5e-6, 0.7e-6)
#: The group's linear rule: smoothing, and the windows for the dI/dt extrema the line joins.
LINEAR_SG_ORDER = 3
LINEAR_SG_FRAME = 21
LINEAR_FIRST_WINDOW_S = (0.3e-6, 0.5e-6)
LINEAR_SECOND_WINDOW_S = (0.7e-6, 0.9e-6)
#: Leakage is the median current over the tail of the U and D plateaus.
LEAK_TAIL_FRACTION = 0.2
LEAK_GUARD_SAMPLES = 10
#: Dielectric slope is fitted where the ramp is between these fractions of the peak voltage.
RAMP_FIT_WINDOW = (0.3, 0.85)
#: A capacitor switches when its whole-pulse Psw exceeds this.
SWITCHING_PSW_UC_CM2 = 2.0


class PulseLabel(Enum):
    P = "P"
    U = "U"
    N = "N"
    D = "D"
    PRESET = "preset"
    UNKNOWN = "X"


class SplitRule(Enum):
    DIP = "dip"
    CROSSOVER = "crossover"
    LINEAR = "linear"


#: The ramps the O / T split is reported on, with the rules applied to each.
SPLIT_LAYOUT = (
    ("raw_n", (SplitRule.DIP, SplitRule.CROSSOVER, SplitRule.LINEAR)),
    ("sub_n", (SplitRule.DIP, SplitRule.CROSSOVER, SplitRule.LINEAR)),
    ("sub_p", (SplitRule.DIP,)),
)
SPLIT_KEYS = tuple(f"{ramp}_{rule.value}" for ramp, rules in SPLIT_LAYOUT for rule in rules)


@dataclass(frozen=True, slots=True, kw_only=True)
class Pulse:
    """One pulse located in the record, by sample index."""

    label: PulseLabel
    sign: int
    i_start: int
    i_ramp_end: int
    i_plateau_end: int
    i_end: int
    peak_v: float

    @property
    def rise(self) -> slice:
        return slice(self.i_start, self.i_ramp_end + 1)

    @property
    def plateau(self) -> slice:
        return slice(self.i_ramp_end, self.i_plateau_end + 1)

    @property
    def fall(self) -> slice:
        return slice(self.i_plateau_end, self.i_end + 1)

    @property
    def whole(self) -> slice:
        return slice(self.i_start, self.i_end + 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class PundWaveform:
    """What was applied: pulse count and pattern, amplitude, timing, baseline current."""

    n_pulses: int
    pulse_pattern: str
    pulse_v: float = np.nan
    t_rise_s: float = np.nan
    t_plateau_s: float = np.nan
    baseline_a: float = np.nan
    pund_dt_s: float = np.nan
    pund_n_points: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class PulseCharge:
    """Charge under one pulse, by segment, coulombs (baseline-corrected)."""

    rise_c: float
    plateau_c: float
    fall_c: float
    whole_c: float


@dataclass(frozen=True, slots=True, kw_only=True)
class SwitchedPolarization:
    """Switched charge per area, by definition.

    ``psw`` integrates whole pulses (rise, plateau, fall), so the dielectric displacement
    cancels and what remains is switching plus leakage. ``psw_inst`` is the instrument's
    ramp-only P - U, the definition behind the group's sheet. ``psw_plateau_*`` is what
    ramp-only integration leaves out.
    """

    psw_uc_cm2: float
    psw_pos_uc_cm2: float
    psw_neg_uc_cm2: float
    psw_asym_uc_cm2: float
    psw_pos_ramp_uc_cm2: float
    psw_neg_ramp_uc_cm2: float
    psw_inst_uc_cm2: float
    psw_inst_neg_uc_cm2: float
    psw_plateau_pos_uc_cm2: float
    psw_plateau_neg_uc_cm2: float
    p_raw_n_ramp_uc_cm2: float
    p_raw_p_ramp_uc_cm2: float
    switching: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class RampSplit:
    """The O / T partition of one ramp current under one rule.

    :param rule: Which rule placed the split.
    :param found: Whether the rule found its feature; otherwise the split is the fallback.
    :param t_split_s: Absolute time of the split.
    :param q_main_c: Signed charge before the split (main peak), coulombs.
    :param q_shoulder_c: Signed charge after the split (late shoulder), coulombs.
    :param fraction_main: ``q_main / (q_main + q_shoulder)``.
    :param total_uc_cm2: Magnitude of the whole ramp charge per area.
    """

    rule: SplitRule
    found: bool
    t_split_s: float
    q_main_c: float
    q_shoulder_c: float
    fraction_main: float
    total_uc_cm2: float


@dataclass(frozen=True, slots=True, kw_only=True)
class SwitchingPeak:
    """The switching-current peak on one polarity: its size, time and the voltage there."""

    current_a: float
    t_peak_s: float
    coercive_v: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Coercivity:
    """Imprint and mean coercive voltage from the two switching peaks."""

    imprint_v: float
    vc_mean_v: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Leakage:
    """Current density over the tail of the non-switching plateaus."""

    j_leak_pos_a_cm2: float
    j_leak_neg_a_cm2: float
    j_leak_a_cm2: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PundDielectric:
    """Capacitance from the non-switching ramps and the permittivity it implies.

    On a switching film the U ramp still carries some switching current, so ``kappa_pund``
    reads high there; the Q-V slope from the P-V sweep is the permittivity of record.
    """

    c_u_f: float
    c_d_f: float
    kappa_pund: float
    kappa_inst_u: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PundFeatures:
    """Everything one PUND run yields; a group is ``None`` when the pulse pattern was not found."""

    waveform: PundWaveform
    instrument: InstrumentCharges
    charge_p: PulseCharge | None = None
    charge_u: PulseCharge | None = None
    charge_n: PulseCharge | None = None
    charge_d: PulseCharge | None = None
    polarization: SwitchedPolarization | None = None
    splits: Mapping[str, RampSplit] = field(default_factory=dict)
    peak_pos: SwitchingPeak | None = None
    peak_neg: SwitchingPeak | None = None
    coercivity: Coercivity | None = None
    leakage: Leakage | None = None
    dielectric: PundDielectric | None = None
    issues: tuple[Issue, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class PundTraces:
    """Arrays the QC report plots: the record and the U/D-subtracted switching currents."""

    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    pulses: tuple[Pulse, ...]
    isw_pos_a: np.ndarray | None = None
    isw_neg_a: np.ndarray | None = None
    t_rel_pos_s: np.ndarray | None = None
    t_rel_neg_s: np.ndarray | None = None
    v_pos_v: np.ndarray | None = None
    v_neg_v: np.ndarray | None = None


def switched_polarization(p, u, n, d, scale):
    """Return (P - U), -(N - D) and their mean, each times ``scale``.

    U and D repeat P and N on an already-switched film, so subtracting them removes the
    dielectric and leakage charge and leaves what switched. Works on scalars and arrays.
    """
    positive = (p - u) * scale
    negative = -(n - d) * scale
    return positive, negative, 0.5 * (positive + negative)


def segment_pulses(voltage_v: np.ndarray) -> tuple[Pulse, ...]:
    """Locate the pulses and label them P, U, N, D by their sign pattern.

    The expected pattern is (+, +, -, -) or its mirror; a fifth leading pulse is a preset.
    Any other pattern leaves every pulse labelled UNKNOWN.
    """
    smoothed = median_filter(voltage_v, size=PULSE_MEDIAN_SAMPLES, mode="nearest")
    on = np.abs(smoothed) > PULSE_THRESHOLD_V
    edges = np.flatnonzero(np.diff(on.astype(int))) + 1
    runs = [
        r
        for r in np.split(np.arange(len(voltage_v)), edges)
        if on[r[0]] and len(r) >= PULSE_MIN_RUN_SAMPLES
    ]
    located = []
    for r in runs:
        i_start, i_end = int(r[0]), int(r[-1])
        segment = smoothed[i_start : i_end + 1]
        peak_v = float(np.max(np.abs(segment)))
        at_plateau = np.flatnonzero(np.abs(segment) >= PLATEAU_FRACTION * peak_v)
        located.append(
            dict(
                sign=1 if segment.mean() > 0 else -1,
                i_start=i_start,
                i_ramp_end=i_start + int(at_plateau[0]),
                i_plateau_end=i_start + int(at_plateau[-1]),
                i_end=i_end,
                peak_v=peak_v,
            )
        )
    signs = tuple(p["sign"] for p in located)
    patterns = {(1, 1, -1, -1): "PUND", (-1, -1, 1, 1): "NDPU"}
    if signs in patterns:
        labels = [PulseLabel(c) for c in patterns[signs]]
    elif len(signs) == 5 and signs[1:] in patterns:
        labels = [PulseLabel.PRESET, *(PulseLabel(c) for c in patterns[signs[1:]])]
    else:
        labels = [PulseLabel.UNKNOWN] * len(located)
    return tuple(Pulse(label=label, **p) for label, p in zip(labels, located))


@dataclass(frozen=True, slots=True, kw_only=True)
class PulseTiming:
    """The PUND waveform as programmed: delay before and between pulses, ramp time, plateau."""

    delay_s: float
    ramp_s: float
    plateau_s: float


def read_pulse_timing(inputs: Mapping[str, str]) -> PulseTiming | None:
    """Read ``td``, ``trf`` and ``tp`` from a run's Settings inputs; ``None`` if any is missing."""
    try:
        return PulseTiming(
            delay_s=float(inputs["td"]), ramp_s=float(inputs["trf"]), plateau_s=float(inputs["tp"])
        )
    except (KeyError, ValueError):
        return None


def programmed_ramp(order: int, timing: PulseTiming, t: np.ndarray, dt: float) -> slice:
    """Return the samples of the ``order``-th pulse's rising ramp as the instrument programmed it.

    Pulse k starts at td + k (2 trf + tp + td) and ramps for trf, so the N ramp of a PUND run
    is 9.0 to 10.0 us. The measured voltage lags this by about 20 ns; the group's scripts and the
    Round-1 summary slide use the programmed window, and so does every split here.
    """
    start = timing.delay_s + order * (2 * timing.ramp_s + timing.plateau_s + timing.delay_s)
    inside = np.flatnonzero((t >= start - dt / 2) & (t < start + timing.ramp_s - dt / 2))
    return slice(int(inside[0]), int(inside[-1]) + 1)


def split_ramp(
    current: np.ndarray, t: np.ndarray, sign: int, dt: float, rule: SplitRule, area: float
) -> RampSplit:
    """Split one ramp current into the charge before and after the main-peak / shoulder boundary.

    All three rules work on the positive-going lobe ``sign * current`` and integrate the raw
    current on either side of the split, as the group's scripts do.

    * DIP: smooth; the main peak is the largest value in the first part of the ramp; the split
      is the first local minimum after it that rises again and stays above a floor.
    * CROSSOVER: heavier smoothing; dI/dt zero crossings that are local minima of the lobe,
      inside the window after ramp start; the one with the smallest current wins.
    * LINEAR: the extremes of dI/dt in two windows, joined by a line; its zero is the split.

    :return: The split, with ``found`` false and the fallback position when the rule fails.
    """
    n = len(current)
    k_split, found = min(n - 1, int(round(SPLIT_FALLBACK_FRACTION * n))), False
    if n >= 10:
        lobe = sign * current
        t_rel = t - t[0]
        if rule is SplitRule.DIP:
            k = _dip_index(_smooth(lobe, dt))
        elif rule is SplitRule.CROSSOVER:
            k = _crossover_index(lobe, t_rel)
        else:
            k = _linear_index(lobe, t_rel)
        if k is not None:
            k_split, found = k, True
    q_main = _trapz(current[: k_split + 1], t[: k_split + 1])
    q_shoulder = _trapz(current[k_split:], t[k_split:])
    total = q_main + q_shoulder
    return RampSplit(
        rule=rule,
        found=found,
        t_split_s=float(t[k_split]) if n else np.nan,
        q_main_c=q_main,
        q_shoulder_c=q_shoulder,
        fraction_main=q_main / total if total != 0 else np.nan,
        total_uc_cm2=abs(total) * uc_cm2_per_coulomb(area),
    )


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def _savgol(values: np.ndarray, frame: int, order: int) -> np.ndarray:
    """Savitzky-Golay smoothing, the frame clamped to the largest odd length that fits."""
    frame = min(frame, len(values) - (1 - len(values) % 2))
    return savgol_filter(values, frame, order) if frame > order else values


def _smooth(current: np.ndarray, dt: float) -> np.ndarray:
    """Smooth a current over ``SMOOTH_WINDOW_S`` for peak finding and the dip rule."""
    return _savgol(current, max(5, int(round(SMOOTH_WINDOW_S / dt)) | 1), SMOOTH_ORDER)


def _dip_index(lobe: np.ndarray) -> int | None:
    n = len(lobe)
    k_peak = int(np.argmax(lobe[: max(5, int(DIP_PEAK_SEARCH_FRACTION * n))]))
    after = lobe[k_peak:]
    dips = np.flatnonzero((after[1:-1] < after[:-2]) & (after[1:-1] <= after[2:])) + 1
    for d in dips:
        rest = after[d:]
        deep_enough = after[d] > DIP_FLOOR_FRACTION * lobe[k_peak]
        if len(rest) > 3 and rest.max() > after[d] * DIP_SHOULDER_RISE and deep_enough:
            return k_peak + int(d)
    return None


def _crossover_index(lobe: np.ndarray, t_rel: np.ndarray) -> int | None:
    smooth = _savgol(lobe, CROSSOVER_SG_FRAME, CROSSOVER_SG_ORDER)
    d = np.gradient(smooth, t_rel)
    k = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))  # lobe minima: dI/dt crosses zero upward
    with np.errstate(divide="ignore", invalid="ignore"):
        crossings = t_rel[k] - d[k] * (t_rel[k + 1] - t_rel[k]) / (d[k + 1] - d[k])
    lo, hi = CROSSOVER_WINDOW_S
    crossings = crossings[(crossings >= lo) & (crossings <= hi)]
    if len(crossings) == 0:
        return None
    chosen = crossings[int(np.argmin(np.interp(crossings, t_rel, smooth)))]
    return int(np.flatnonzero(t_rel >= chosen)[0])


def _linear_index(lobe: np.ndarray, t_rel: np.ndarray) -> int | None:
    smooth = _savgol(lobe, LINEAR_SG_FRAME, LINEAR_SG_ORDER)
    d = np.gradient(smooth, t_rel)
    first = (t_rel >= LINEAR_FIRST_WINDOW_S[0]) & (t_rel <= LINEAR_FIRST_WINDOW_S[1])
    second = (t_rel >= LINEAR_SECOND_WINDOW_S[0]) & (t_rel <= LINEAR_SECOND_WINDOW_S[1])
    if not first.any() or not second.any():
        return None
    k_a = np.flatnonzero(first)[int(np.argmin(d[first]))]
    k_b = np.flatnonzero(second)[int(np.argmax(d[second]))]
    if t_rel[k_b] == t_rel[k_a] or d[k_b] == d[k_a]:
        return None
    slope = (d[k_b] - d[k_a]) / (t_rel[k_b] - t_rel[k_a])
    t_zero = t_rel[k_a] - d[k_a] / slope
    inside = np.flatnonzero(t_rel >= t_zero)
    if len(inside) == 0 or t_zero < t_rel[0]:
        return None
    return int(inside[0])


def extract_pund_features(run: PundRun) -> tuple[PundFeatures, PundTraces]:
    """Reduce one PUND run to its features and the traces the QC report plots."""
    area = area_cm2(run.source.file.side_um)
    scale = uc_cm2_per_coulomb(area)
    t, voltage, current = run.time_s, run.voltage_v, run.current_a
    dt = run.dt_s
    where = run.source.where()

    pulses = segment_pulses(voltage)
    by = {p.label: p for p in pulses}
    pattern = "".join(p.label.value[0] for p in pulses)
    traces = PundTraces(time_s=t, voltage_v=voltage, current_a=current, pulses=pulses)
    if not {PulseLabel.P, PulseLabel.U, PulseLabel.N, PulseLabel.D} <= by.keys():
        waveform = PundWaveform(
            n_pulses=len(pulses), pulse_pattern=pattern, pund_dt_s=dt, pund_n_points=len(t)
        )
        issue = error(
            IssueCode.UNEXPECTED_PULSE_PATTERN,
            f"unexpected pulse pattern {[p.sign for p in pulses]}",
            **where,
        )
        return PundFeatures(waveform=waveform, instrument=run.instrument, issues=(issue,)), traces

    p, u, n, d = by[PulseLabel.P], by[PulseLabel.U], by[PulseLabel.N], by[PulseLabel.D]
    first = min(pulse.i_start for pulse in pulses)
    baseline = 0.0
    if first > BASELINE_NEEDS_SAMPLES:
        baseline = float(current[: max(BASELINE_MIN_SAMPLES, first - BASELINE_GAP_SAMPLES)].mean())
    corrected = current - baseline
    waveform = PundWaveform(
        n_pulses=len(pulses),
        pulse_pattern=pattern,
        pulse_v=p.peak_v,
        t_rise_s=float(t[p.i_ramp_end] - t[p.i_start]),
        t_plateau_s=float(t[p.i_plateau_end] - t[p.i_ramp_end]),
        baseline_a=baseline,
        pund_dt_s=dt,
        pund_n_points=len(t),
    )

    def charge(pulse: Pulse) -> PulseCharge:
        return PulseCharge(
            rise_c=_trapz(corrected[pulse.rise], t[pulse.rise]),
            plateau_c=_trapz(corrected[pulse.plateau], t[pulse.plateau]),
            fall_c=_trapz(corrected[pulse.fall], t[pulse.fall]),
            whole_c=_trapz(corrected[pulse.whole], t[pulse.whole]),
        )

    qp, qu, qn, qd = charge(p), charge(u), charge(n), charge(d)
    inst = run.instrument
    psw_pos, psw_neg, psw = switched_polarization(
        qp.whole_c, qu.whole_c, qn.whole_c, qd.whole_c, scale
    )
    ramp_pos, ramp_neg, _ = switched_polarization(qp.rise_c, qu.rise_c, qn.rise_c, qd.rise_c, scale)
    inst_pos, inst_neg, _ = switched_polarization(inst.p_c, inst.u_c, inst.n_c, inst.d_c, scale)
    plateau_pos, plateau_neg, _ = switched_polarization(
        qp.plateau_c, qu.plateau_c, qn.plateau_c, qd.plateau_c, scale
    )
    polarization = SwitchedPolarization(
        psw_uc_cm2=psw,
        psw_pos_uc_cm2=psw_pos,
        psw_neg_uc_cm2=psw_neg,
        psw_asym_uc_cm2=psw_pos - psw_neg,
        psw_pos_ramp_uc_cm2=ramp_pos,
        psw_neg_ramp_uc_cm2=ramp_neg,
        psw_inst_uc_cm2=inst_pos,
        psw_inst_neg_uc_cm2=inst_neg,
        psw_plateau_pos_uc_cm2=plateau_pos,
        psw_plateau_neg_uc_cm2=plateau_neg,
        p_raw_n_ramp_uc_cm2=-qn.rise_c * scale,
        p_raw_p_ramp_uc_cm2=qp.rise_c * scale,
        switching=bool(psw > SWITCHING_PSW_UC_CM2),
    )

    def tail_median(pulse: Pulse) -> float:
        sl = pulse.plateau
        count = sl.stop - sl.start
        stop = sl.stop - min(LEAK_GUARD_SAMPLES, count // 10)  # stay clear of the fall onset
        return float(np.median(corrected[stop - max(5, int(count * LEAK_TAIL_FRACTION)) : stop]))

    j_pos, j_neg = tail_median(u) / area, tail_median(d) / area
    leakage = Leakage(
        j_leak_pos_a_cm2=j_pos, j_leak_neg_a_cm2=j_neg, j_leak_a_cm2=0.5 * (abs(j_pos) + abs(j_neg))
    )

    def ramp_capacitance(pulse: Pulse) -> float:
        sl = pulse.rise
        v, i, tt = voltage[sl], corrected[sl], t[sl]
        lo, hi = RAMP_FIT_WINDOW
        m = (np.abs(v) > lo * p.peak_v) & (np.abs(v) < hi * p.peak_v)
        if m.sum() < 5:
            return np.nan
        slew = abs(np.polyfit(tt[m], v[m], 1)[0])  # measured dV/dt on the ramp
        return float(np.abs(i[m].mean()) / slew)

    c_u, c_d = ramp_capacitance(u), ramp_capacitance(d)
    dielectric = PundDielectric(
        c_u_f=c_u,
        c_d_f=c_d,
        kappa_pund=kappa_from_capacitance(np.nanmean([c_u, c_d]), area),
        kappa_inst_u=kappa_from_capacitance(inst.u_c / p.peak_v, area),
    )

    def subtracted(a: Pulse, b: Pulse):
        count = min(a.i_end - a.i_start, b.i_end - b.i_start) + 1
        sa, sb = slice(a.i_start, a.i_start + count), slice(b.i_start, b.i_start + count)
        return corrected[sa] - corrected[sb], t[sa] - t[a.i_start], voltage[sa]

    isw_pos, t_rel_pos, v_pos = subtracted(p, u)
    isw_neg, t_rel_neg, v_neg = subtracted(n, d)
    traces = PundTraces(
        time_s=t,
        voltage_v=voltage,
        current_a=current,
        pulses=pulses,
        isw_pos_a=isw_pos,
        isw_neg_a=isw_neg,
        t_rel_pos_s=t_rel_pos,
        t_rel_neg_s=t_rel_neg,
        v_pos_v=v_pos,
        v_neg_v=v_neg,
    )

    def peak(isw, tt, vv, sign: int, pulse: Pulse) -> SwitchingPeak:
        count = pulse.i_plateau_end - pulse.i_start + 1
        lobe = sign * _smooth(isw[:count], dt)
        k = int(np.argmax(lobe))
        return SwitchingPeak(
            current_a=float(lobe[k]), t_peak_s=float(tt[k]), coercive_v=float(vv[k])
        )

    peak_pos, peak_neg = (
        peak(isw_pos, t_rel_pos, v_pos, 1, p),
        peak(isw_neg, t_rel_neg, v_neg, -1, n),
    )
    coercivity = Coercivity(
        imprint_v=0.5 * (peak_pos.coercive_v + peak_neg.coercive_v),
        vc_mean_v=0.5 * (peak_pos.coercive_v - peak_neg.coercive_v),
    )

    timing = read_pulse_timing(run.source.settings.inputs)

    def ramp(pulse: Pulse) -> slice:
        if timing is None:
            return pulse.rise
        return programmed_ramp(pulses.index(pulse), timing, t, dt)

    def ramp_minus(a: Pulse, b: Pulse) -> tuple[np.ndarray, slice]:
        """The ramp of ``a`` with the same samples of ``b``'s ramp subtracted."""
        ra, rb = ramp(a), ramp(b)
        count = min(ra.stop - ra.start, rb.stop - rb.start)
        ra = slice(ra.start, ra.start + count)
        return corrected[ra] - corrected[rb.start : rb.start + count], ra

    n_ramp = ramp(n)
    sub_n, sub_n_ramp = ramp_minus(n, d)
    sub_p, sub_p_ramp = ramp_minus(p, u)
    ramps = {
        "raw_n": (current[n_ramp], t[n_ramp], -1),
        "sub_n": (sub_n, t[sub_n_ramp], -1),
        "sub_p": (sub_p, t[sub_p_ramp], 1),
    }
    splits = {
        f"{ramp}_{rule.value}": split_ramp(*ramps[ramp], dt, rule, area)
        for ramp, rules in SPLIT_LAYOUT
        for rule in rules
    }

    issues = []
    if timing is None:
        issues.append(
            warning(
                IssueCode.TIMING_NOT_IN_SETTINGS,
                "td, trf or tp missing from Settings; the split uses the detected ramp",
                **where,
            )
        )
    if polarization.switching and not splits["raw_n_dip"].found:
        issues.append(
            warning(
                IssueCode.SPLIT_NOT_FOUND, "no dip between peak and shoulder on the N ramp", **where
            )
        )
    features = PundFeatures(
        waveform=waveform,
        instrument=inst,
        charge_p=qp,
        charge_u=qu,
        charge_n=qn,
        charge_d=qd,
        polarization=polarization,
        splits=splits,
        peak_pos=peak_pos,
        peak_neg=peak_neg,
        coercivity=coercivity,
        leakage=leakage,
        dielectric=dielectric,
        issues=tuple(issues),
    )
    return features, traces
