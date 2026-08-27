"""Candidate seed designs, including ones that use ONLY the measured peak-temperature table.

The committed seed places blocks A and B using a transition bracket, an activation energy, an
Avrami exponent, a noise model and a five-member pulse-shape ensemble. Exactly one of those --
none of them -- is a campaign measurement; the peak-temperature table is. Everything else is
archival, from a different sample set, a different tool, or a readout whose calibration is now in
doubt.

That is a lot of prior for a first batch. The designs here let the question be settled by
measurement rather than by argument: build alternatives that lean on progressively less, score them
all against ground truths none of them came from (``run_seed_stress``), and keep whichever wins.

  ``naive_lhs``      Latin hypercube over the raw knobs (V, log t). Assumes nothing at all, and is
                     the design the campaign originally rejected.
  ``thermal_lhs``    Latin hypercube stratified over PEAK TEMPERATURE, inverted through the table.
                     Uses the table and nothing else.
  ``paired_sweep``   Matched-Tmax pairs at several temperature levels spanning the reachable range.
                     Also uses only the table -- and it answers the dwell question at EVERY level,
                     so it does not have to know in advance which level the transition sits at.

The last is the interesting one. The committed design buys dwell contrast at two tightly-placed
levels, which is efficient if the transition is where the bracket says and blind if it is not. A
sweep of pairs buys less precision per level and cannot be wrong about which level to watch.
"""

from typing import Sequence, Tuple

import numpy as np
from scipy.stats import qmc

from design_space import V_HI, V_LO, snap, snap_all
from physics.thermal_model import FLASH

N_SHOTS = 16  # every candidate is compared at the committed budget

# Crystallizing 10 nm HZO on a millisecond timescale is not plausible below this, on any kinetics.
# It is a physical floor, not an archival one -- which is the point of using it here.
SWEEP_LO_C = 350.0
LEVEL_MARGIN_C = 6.0  # keep levels off the reachable edge, where the inversion is ill-conditioned


def naive_lhs(n: int, t_lo: float, t_hi: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Latin hypercube over the raw knobs, stratified in log t.

    :param n: number of conditions.
    :param t_lo: shortest supported flash time (ms).
    :param t_hi: longest supported flash time (ms).
    :param seed: RNG seed.
    """
    u = qmc.LatinHypercube(d=2, seed=seed).random(n)
    v = V_LO + u[:, 0] * (V_HI - V_LO)
    lo, hi = np.log10(t_lo), np.log10(t_hi)
    t = 10.0 ** (lo + u[:, 1] * (hi - lo))
    out = [snap(a, b) for a, b in zip(v, t)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out], float)


def thermal_lhs(n: int, t_lo: float, t_hi: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Latin hypercube stratified over PEAK TEMPERATURE, inverted through the measured table.

    Draws a flash time, then a temperature quantile within the range reachable at that time, then
    solves the table for the voltage. Uses the table and nothing else -- no onset, no kinetics.

    :param n: number of conditions.
    :param t_lo: shortest supported flash time (ms).
    :param t_hi: longest supported flash time (ms).
    :param seed: RNG seed.
    """
    u = qmc.LatinHypercube(d=2, seed=seed).random(n)
    lo, hi = np.log10(t_lo), np.log10(t_hi)
    t = snap_all(np.zeros_like(u[:, 1]), 10.0 ** (lo + u[:, 1] * (hi - lo)))[1]
    v = np.empty(n)
    for i, ti in enumerate(t):
        r_lo, r_hi = FLASH.tmax_range(float(ti))
        lo_c = max(r_lo + LEVEL_MARGIN_C, SWEEP_LO_C)
        hi_c = r_hi - LEVEL_MARGIN_C
        if hi_c <= lo_c:  # this flash time cannot reach the window at all
            lo_c, hi_c = r_lo + LEVEL_MARGIN_C, r_hi - LEVEL_MARGIN_C
        v[i] = FLASH.voltage_for_tmax(lo_c + u[i, 0] * (hi_c - lo_c), float(ti))
    out = [snap(a, b) for a, b in zip(v, t)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out], float)


def paired_sweep(
    n_levels: int, times: Sequence[float], lo_c: float = SWEEP_LO_C
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matched-Tmax pairs at ``n_levels`` levels spanning the range reachable at every time.

    Every level is fired at every time, so the dwell contrast is measured wherever the transition
    turns out to be. The committed design concentrates that contrast at two levels, which is more
    precise if the bracket is right and blind if it is not.

    Returns ``(V, t, level)`` so a caller can report which level each shot belongs to.

    :param n_levels: number of peak-temperature levels.
    :param times: flash times, which should be measured table rows.
    :param lo_c: coldest level to place.
    """
    reach_lo = max(FLASH.tmax_range(float(t))[0] for t in times) + LEVEL_MARGIN_C
    reach_hi = min(FLASH.tmax_range(float(t))[1] for t in times) - LEVEL_MARGIN_C
    levels = np.linspace(max(reach_lo, lo_c), reach_hi, n_levels)
    v, t, lv = [], [], []
    for level in levels:
        for ti in times:
            vs, ts = snap(FLASH.voltage_for_tmax(float(level), float(ti)), float(ti))
            v.append(vs)
            t.append(ts)
            lv.append(level)
    return np.array(v), np.array(t, float), np.array(lv)


def with_replicates(
    v: np.ndarray, t: np.ndarray, idx: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Repeat the given conditions on separate specimens.

    :param v: voltages.
    :param t: flash times (ms).
    :param idx: positions to replicate.
    """
    idx = list(idx)
    return np.concatenate([v, v[idx]]), np.concatenate([t, t[idx]])


def catalogue(committed, ladder, explore, floor, t_lo: float, t_hi: float, seed: int) -> dict:
    """Every candidate seed, all at the same shot count, keyed by name.

    The committed design is INJECTED rather than imported so that this module stays a library and
    the comparison scripts stay thin -- otherwise the design catalogue and the design generator
    import each other and a CLI flag can silently change a published number.

    :param committed: ``(V, t)`` of the committed plan.
    :param ladder: ``(V, t)`` of the committed plan's dwell ladder and its replicates.
    :param explore: callable ``(n, avoid_v, avoid_t) -> (V, t)`` placing model-agnostic probes.
    :param floor: ``(V, t)`` of the amorphous floor anchor.
    :param t_lo: shortest supported flash time (ms).
    :param t_hi: longest supported flash time (ms).
    :param seed: realization seed for the randomized designs.
    """
    fv, ft = floor
    v_c, t_c = committed
    out = {"committed (A4 B4 C3 D4 E1)": (np.asarray(v_c), np.asarray(t_c))}
    out["naive LHS (V, log t)"] = naive_lhs(N_SHOTS, t_lo, t_hi, seed)
    out["thermal LHS (Tmax, log t)"] = thermal_lhs(N_SHOTS, t_lo, t_hi, seed)

    v, t, _ = paired_sweep(5, (2.6, 5.1))
    v, t = with_replicates(v, t, [1, 3, 5, 7, 9])
    out["paired sweep 5x2 +5r"] = (np.append(v, int(fv)), np.append(t, ft))

    # ladder + replicates kept; the model-informed core replaced by table-only coverage
    va, ta = (np.asarray(a) for a in ladder)
    vl, tl = thermal_lhs(4, t_lo, t_hi, seed)
    vc, tc = explore(3, np.concatenate([va, vl, [fv]]), np.concatenate([ta, tl, [ft]]))
    out["synthesis (A4 L4 C3 D4 E1)"] = (
        np.concatenate([va, vl, vc, [fv]]),
        np.concatenate([ta, tl, tc, [ft]]),
    )
    return out
