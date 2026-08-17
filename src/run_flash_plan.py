"""Generate the seed flash-condition plan for the crystallization-boundary campaign.

The design box is (flash voltage, flash time), but crystallization responds to PEAK TEMPERATURE,
which spans 82-563 C across that box while the transition is only tens of degrees wide. A Latin
hypercube laid out uniformly in (V, t) is therefore NOT uniform in the quantity that matters: most
of its points land where every hypothesis predicts the same thing. This generator stratifies the
measured peak temperature instead -- LHS over (Tmax, log t), inverted through the measured table to
(V, t) -- and spends part of the seed on a designed contrast rather than on coverage.

Four blocks, 16 conditions of an 80-shot budget:

  A  iso-Tmax dwell ladder   4  TWO peak-temperature levels crossed with the two measured table
                                rows that can reach them.
                                Reading ALONG a level measures the boundary tilt; reading ACROSS
                                the levels measures the transition sharpness. One level would
                                confound the two, and would slide into the floor or the ceiling if
                                the true onset differs from the prior.
  B  stratified core         7  (Tmax, log t) Latin hypercube over the range where the ensemble
                                actually transitions, maximin against block A.
  D  ladder replicates      +4  every block-A condition on a second specimen. Needed for
                                POWER -- one specimen per rung confirms a flat boundary only 86% of
                                the time, against the 90% the design is held to -- and so that a
                                discrepant replicate, which means a variable outside (V, t) is in
                                play, can be diagnosed at any rung rather than only a chosen one.
  E  amorphous floor         1  one cold condition to pin the readout's amorphous floor.

Every condition is scored against the whole boundary-model ensemble, and the CSV records the
spread rather than a single predicted label: the disagreement IS what the seed buys.

Usage:  python src/run_flash_plan.py [--n-core 7] [--seed 7]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

sys.path.append(str(Path(__file__).resolve().parent))

from discovery.constants import (
    NOISE_BOUNDARY,
    NOISE_FLOOR,
    T_ONSET_C,
    T_ONSET_SIGMA_C,
    T_REF_MS,
)
from discovery.kinetics import (
    DEFAULT_MODEL,
    build_ensemble,
    disagreement,
    logistic_sharpness,
    theta_kelvin,
)
from discovery.synthetic import FLASH, FLASH_T, T_HI, T_LO, V_HI, V_LO
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "flash_plan"

# Boundary-search points are restricted to flash times the measured table actually supports. The
# table has NO node between 0.1 and 2.6 ms even though Tmax climbs 189 -> 563 C across that gap and
# the surface maximum sits inside it, so any Tmax quoted there is an artifact of the spline.
T_SEARCH_LO, T_SEARCH_HI = float(FLASH_T[1]), float(FLASH_T[-1])  # 2.6 .. 10.1 ms

# Block A uses TWO peak-temperature levels, not one. A single-level ladder is rank-deficient: it
# identifies only the product of transition sharpness and tilt, and it slides bodily into the
# saturated floor or ceiling if the true onset differs from the prior by ~1 sigma, at which point
# it reports "no tilt" even when the tilt is large. Two levels straddling the onset prior keep the
# estimate well-conditioned and roughly flat in precision across T_ONSET_C +/- T_ONSET_SIGMA_C.
#
# Two constraints leave exactly two usable flash times, and they are both MEASURED TABLE ROWS.
#
# The long rows are out of reach. The reachable ceiling falls with pulse width -- 563 C at 2.6 ms,
# 515 at 5.1, 438 at 7.6, 422 at 10.1 -- and the levels must be reachable at EVERY ladder time, so
# a level near the onset rules out everything beyond ~7 ms.
#
# The gaps between rows cannot be used either. At a measured row the spline and the GP fitted to
# the same 30 nodes agree to ~0.1 C and the GP's own uncertainty is ~0.15 C. Midway between the 2.6
# and 5.1 ms rows they disagree by 10.9 C and that uncertainty is 15.8 C -- comparable to the 16.5 C
# separation between the two levels, so an off-node rung cannot be placed on the same iso-Tmax
# contour to a tolerance the tilt estimate can use. A 3.8 ms rung would improve worst-case SE(tilt)
# from 28.7 to 25.1 K on paper while spending that gain on a rung whose temperature is unknown to
# the width of the contrast being measured.
#
# The dwell contrast is therefore ln(5.1/2.6) = 0.67 e-folds. That is a limit of the tool at this
# onset rather than of the design: the boundary sits exactly where the reachable ceiling is
# collapsing. Widening it requires a new peak-temperature row, not a different design.
LADDER_TIMES = (2.6, 5.1)
LADDER_MARGIN_C = 6.0  # keep levels off the reachable edge, where inversion is ill-conditioned
LADDER_MIN_GAP_C = 6.0  # two levels closer than this are one level
LADDER_LEVEL_STEP_C = 2.0  # search resolution
# Tilt is a DIFFERENCE between two dwells, so it is meaningless without the range it spans. Every
# tilt this script reports is quoted over the ladder's own range, which is what block A measures.
LADDER_DWELL_RANGE = (min(LADDER_TIMES), max(LADDER_TIMES))

# Resolution of the scan that locates the transition band, over a band ~100 C wide. Near the
# long-dwell end the reachable ceiling clips the band to a sliver; below about 0.3 C of width the
# scan finds fewer than two nodes inside and reports it unreachable. That is the intended
# behaviour, not a defect -- a band narrower than that offers no range to stratify over, and the
# caller drops those times from its interpolation table.
BAND_SCAN_NODES = 600

# Outer bound on where a boundary-search condition may sit: the onset prior widened by the tilt a
# boundary could show across the ladder's dwell range and by the transition half-width. Derived
# from those three inputs rather than hardcoded, so that a change to any of them cannot leave a
# stale literal behind. Block B narrows further, to the range where the ensemble mean actually
# transitions (_transition_band).
# Quoted over the range block B actually spans, not the ladder's -- this bound constrains block B.
_BAND_TILT_C = theta_kelvin(T_ONSET_C) * np.log(T_SEARCH_HI / T_SEARCH_LO)
_BAND_HALFWIDTH_C = np.log(9.0) / logistic_sharpness(T_ONSET_C)  # 10-90% half-width
BAND_LO_C = T_ONSET_C - T_ONSET_SIGMA_C - _BAND_TILT_C - _BAND_HALFWIDTH_C
BAND_HI_C = T_ONSET_C + T_ONSET_SIGMA_C + _BAND_TILT_C + _BAND_HALFWIDTH_C


def _tilt_precision(levels, t0: float, sharp: float, tilt: float = 18.0) -> float:
    """Standard error on the boundary tilt from a ladder, with sharpness left FREE.

    The ladder measures a boundary of the form ``X = sigmoid(s * [Tmax - T0 + beta*ln(t/t_ref)])``.
    Reading along one level constrains only the PRODUCT of the sharpness ``s`` and the tilt
    ``beta``; a single level therefore cannot separate a sharp boundary that barely moves from a
    broad one that moves a lot. Two levels break that degeneracy, but only if they are placed well.
    This returns sqrt of the (beta, beta) entry of the inverse Fisher information, i.e. the
    uncertainty on the quantity the first batch exists to measure.

    :param levels: candidate peak-temperature levels (deg C).
    :param t0: assumed onset temperature (deg C).
    :param sharp: assumed transition sharpness (per deg C).
    :param tilt: assumed tilt, at which the information is evaluated (K per e-fold of dwell).
    """
    times = np.asarray(LADDER_TIMES, float)
    temp = np.repeat(np.asarray(levels, float), times.size)
    t = np.tile(times, len(levels))
    u = np.log(t / T_REF_MS)
    arg = temp - t0 + tilt * u
    mu = 1.0 / (1.0 + np.exp(-sharp * arg))
    grad_z = mu * (1.0 - mu)
    sd = NOISE_FLOOR + NOISE_BOUNDARY * mu * (1.0 - mu)
    jac = np.column_stack([grad_z * -sharp, grad_z * arg, grad_z * sharp * u])
    jac = jac / sd[:, None]
    fisher = jac.T @ jac
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        return np.inf
    return float(np.sqrt(cov[2, 2])) if cov[2, 2] > 0 else np.inf


def ladder_levels() -> tuple:
    """Two iso-Tmax levels chosen to measure the tilt as precisely as possible, MINIMAX over the
    onset prior.

    Placing the levels at ``T_ONSET_C +/- T_ONSET_SIGMA_C`` is intuitive and wrong: with a wide
    onset prior it puts one level in the amorphous floor and the other near saturation, where the
    response is flat under every hypothesis. Instead the pair is chosen to minimise the WORST-CASE
    uncertainty on the tilt as the true onset ranges over its prior, which keeps the ladder useful
    whether the onset sits at the centre or the edge of what we believe.

    Levels are additionally constrained to be reachable at EVERY ladder time -- the reachable
    ceiling falls with dwell, so the long-pulse arm sets the upper limit.
    """
    sharp = logistic_sharpness(T_ONSET_C)
    lo = max(FLASH.tmax_range(t)[0] for t in LADDER_TIMES) + LADDER_MARGIN_C
    hi = min(FLASH.tmax_range(t)[1] for t in LADDER_TIMES) - LADDER_MARGIN_C
    grid = np.arange(np.ceil(lo), np.floor(hi) + 1, LADDER_LEVEL_STEP_C)
    onsets = np.linspace(T_ONSET_C - T_ONSET_SIGMA_C, T_ONSET_C + T_ONSET_SIGMA_C, 9)
    best, best_score = None, np.inf
    for i, a in enumerate(grid):
        for b in grid[i + int(LADDER_MIN_GAP_C / LADDER_LEVEL_STEP_C) :]:
            score = max(_tilt_precision((a, b), t0, sharp) for t0 in onsets)
            if score < best_score:
                best, best_score = (float(a), float(b)), score
    if best is None:
        raise RuntimeError("no feasible ladder levels")
    return best


LADDER_TMAX_LEVELS = ladder_levels()
FLOOR_CONDITION = (V_LO, 5.1)  # coldest reachable column at a supported time


def _snap(v: float, t: float) -> tuple:
    """Snap a condition to what the tool can actually be set to: whole volts, 0.1 ms."""
    return int(round(v)), round(t * 10.0) / 10.0


def _norm(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map conditions to the unit box, with time on a log axis (the boundary is closer to
    isotropic in (V, log t) than in (V, t))."""
    x = (np.asarray(v, float) - V_LO) / (V_HI - V_LO)
    y = (np.log10(np.asarray(t, float)) - np.log10(T_LO)) / (np.log10(T_HI) - np.log10(T_LO))
    return np.column_stack([x, y])


def _min_separation(v: np.ndarray, t: np.ndarray) -> float:
    """Smallest pairwise distance in the normalized box (the maximin design criterion)."""
    p = _norm(v, t)
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def ladder_block() -> tuple:
    """Block A: two peak-temperature levels crossed with three pulse widths.

    Voltage is inverted from the measured Tmax table at each (level, time), so each level is an
    iso-Tmax contour to within the snapping resolution. Reading along a level measures the tilt;
    reading across the two levels measures the transition sharpness, which is what makes the two
    separable instead of confounded.
    """
    vs, ts, levels = [], [], []
    for level in LADDER_TMAX_LEVELS:
        for t in LADDER_TIMES:
            v = FLASH.voltage_for_tmax(level, t)
            if not np.isfinite(v):
                raise ValueError(f"{level} C unreachable at t = {t} ms")
            v, t = _snap(v, t)
            vs.append(v)
            ts.append(t)
            levels.append(level)
    return np.array(vs), np.array(ts), np.array(levels)


def _transition_band(models: dict, t: float, lo_x: float = 0.03, hi_x: float = 0.97) -> tuple:
    """Peak-temperature range over which the ENSEMBLE MEAN fraction runs from ``lo_x`` to ``hi_x``.

    Uses the mean across all hypotheses rather than any single one, so the stratification hedges
    the same way the rest of the design does.

    Returns ``(nan, nan)`` when the transition is NOT REACHABLE at this flash time. The reachable
    ceiling falls with pulse width, so beyond about 8 ms no voltage in the box gets the film near
    the onset. Reporting the full prior band there would hand the caller a range that looks like a
    transition band but lies entirely in the amorphous floor, and draws taken from it would look
    stratified while carrying no information.

    :param models: the boundary-model ensemble.
    :param t: flash time (ms).
    :param lo_x: lower fraction defining the informative range.
    :param hi_x: upper fraction defining the informative range.
    """
    grid = np.linspace(BAND_LO_C, BAND_HI_C, BAND_SCAN_NODES)
    v = FLASH.voltages_for_tmax(grid, np.full_like(grid, t))
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    x = np.mean([m.fraction(v[ok], np.full(ok.sum(), t)) for m in models.values()], axis=0)
    inside = grid[ok][(x >= lo_x) & (x <= hi_x)]
    if inside.size < 2:
        return float("nan"), float("nan")
    return float(inside.min()), float(inside.max())


def core_block(
    n: int, seed: int, avoid_v: np.ndarray, avoid_t: np.ndarray, models: dict, n_tries: int = 400
):
    """Block B: Latin hypercube over (Tmax, log t), inverted to (V, t).

    Draws a stratified flash time and a stratified peak-temperature quantile within the band that
    is actually reachable at that time, then solves the measured table for the voltage. Repeats
    over ``n_tries`` LHS seeds and keeps the design with the largest minimum separation from block
    A and from itself -- a maximin criterion applied in the coordinates the GP sees.

    The per-draw Latin hypercube is deliberately NOT discrepancy-optimized. Selection here is by
    maximin in normalized (V, log t) AFTER inverting through the temperature table, so optimizing
    centered discrepancy in the raw unit square first would spend most of the runtime refining
    candidates against a different criterion, only to discard them.

    :param n: number of core conditions.
    :param seed: base RNG seed.
    :param avoid_v: voltages of already-placed conditions.
    :param avoid_t: flash times of already-placed conditions.
    :param n_tries: how many LHS realizations to score.
    """
    best, best_sep = None, -np.inf
    log_lo, log_hi = np.log10(T_SEARCH_LO), np.log10(T_SEARCH_HI)
    # Precompute the transition band once on a coarse time grid and interpolate: recomputing it
    # per candidate design would dominate the search cost. Times where the band is unreachable are
    # dropped from the interpolation table rather than carried as nan, which np.interp would spread
    # over both neighbouring intervals and turn into a silent, unintended exclusion zone.
    band_t = np.geomspace(T_SEARCH_LO, T_SEARCH_HI, 24)
    _edges = np.array([_transition_band(models, x) for x in band_t])
    keep = np.isfinite(_edges[:, 0])
    if keep.sum() < 2:
        raise RuntimeError("the transition is unreachable across the supported flash times")
    band_t = band_t[keep]
    band_lo_v, band_hi_v = _edges[keep, 0], _edges[keep, 1]
    for k in range(n_tries):
        u = qmc.LatinHypercube(d=2, seed=seed + k).random(n)
        t = np.round(10.0 ** (log_lo + u[:, 0] * (log_hi - log_lo)) * 10.0) / 10.0
        reach_lo = np.array([FLASH.tmax_range(x)[0] for x in t])
        reach_hi = np.array([FLASH.tmax_range(x)[1] for x in t])
        # Stratify across the range where the ensemble actually transitions, not uniformly over
        # the full prior band: the band is ~4x the width of the transition, so uniform
        # stratification spends most shots where every hypothesis already agrees.
        edges = np.column_stack([np.interp(t, band_t, band_lo_v), np.interp(t, band_t, band_hi_v)])
        band_lo = np.maximum(np.maximum(reach_lo, BAND_LO_C), edges[:, 0])
        band_hi = np.minimum(np.minimum(reach_hi, BAND_HI_C), edges[:, 1])
        if np.any(band_hi <= band_lo):
            continue
        target = band_lo + u[:, 1] * (band_hi - band_lo)
        v = FLASH.voltages_for_tmax(target, t)
        if not np.all(np.isfinite(v)):
            continue
        v = np.round(v)
        sep = _min_separation(np.concatenate([avoid_v, v]), np.concatenate([avoid_t, t]))
        if sep > best_sep:
            best_sep, best = sep, (v.astype(int), t)
    if best is None:
        raise RuntimeError("no feasible core design found")
    return best


def make_plan(n_core: int, seed: int) -> dict:
    """Assemble the full seed plan and score every condition against the model ensemble.

    :param n_core: size of the stratified core block.
    :param seed: RNG seed for the core Latin hypercube.
    """
    models = build_ensemble()
    v_a, t_a, lvl_a = ladder_block()
    v_e, t_e = _snap(*FLOOR_CONDITION)
    v_e, t_e = np.array([v_e]), np.array([t_e])
    v_b, t_b = core_block(
        n_core, seed, np.concatenate([v_a, v_e]), np.concatenate([t_a, t_e]), models
    )

    # EVERY ladder condition is replicated on a second specimen, rather than a hand-picked subset.
    # Two independent reasons, and the design needs both:
    #
    #   power       -- with only two usable flash times the ladder is short, and a single specimen
    #                  per rung leaves the probability of correctly confirming a FLAT boundary at
    #                  0.841, under the 0.9 the design is held to. Doubling the rungs raises it to
    #                  0.941. (4000 trials; at 400 the third digit is pure Monte Carlo noise.)
    #                  Confirming the null is the harder of the two jobs: a tilted boundary is
    #                  recovered at 0.975 unreplicated and 0.989 replicated.
    #   attribution -- a discrepant replicate says a variable outside (V, t) is in play. Replicating
    #                  only some rungs makes that diagnosis conditional on which rung happened to be
    #                  picked, and unreplicated rungs would carry scatter that reads as tilt.
    #
    # A pair is still a sentinel, not a variance estimate; two specimens cannot separate specimen
    # scatter from a thermal-delivery error at that condition.
    rep_idx = list(range(len(v_a)))

    v = np.concatenate([v_a, v_b, v_e, v_a[rep_idx]])
    t = np.concatenate([t_a, t_b, t_e, t_a[rep_idx]])
    block = ["A"] * len(v_a) + ["B"] * len(v_b) + ["E"] * len(v_e) + ["D"] * len(rep_idx)
    note = (
[f"ladder {lv:.0f} C @ t={x} ms" for lv, x in zip(lvl_a, t_a)]
        + ["stratified core"] * len(v_b)
        + ["amorphous floor anchor"]
        + [f"replicate of A{i + 1} (separate specimen)" for i in rep_idx]
    )
    order = np.lexsort((v, t, np.array([("ABDE".index(b)) for b in block])))
    return {
        "V": v[order],
        "t": t[order],
        "block": [block[i] for i in order],
        "note": [note[i] for i in order],
        "tmax": FLASH.tmax(v[order], t[order]),
        "preds": {k: m.fraction(v[order], t[order]) for k, m in models.items()},
        "disagree": disagreement(models, v[order], t[order]),
        "models": models,
        "ladder": (v_a, t_a, lvl_a),
    }


def _check(plan: dict) -> None:
    """Assert the design properties the plan is supposed to guarantee."""
    v, t, block, tm = plan["V"], plan["t"], plan["block"], plan["tmax"]
    assert np.all(t >= T_SEARCH_LO - 1e-9), "a condition lands in the un-noded 0.1-2.6 ms gap"
    assert np.all(t <= T_SEARCH_HI + 1e-9), "a condition exceeds the measured time range"
    assert np.all(v == np.round(v)) and np.allclose(t, np.round(t * 10) / 10), "not snapped"
    lad_tm = np.array([tm[i] for i in range(len(tm)) if block[i] in ("A", "D")])
    # Assign each ladder shot to its NEAREST level. A fixed-width window would be wrong whenever
    # the levels sit closer together than the window -- which LADDER_MIN_GAP_C explicitly permits,
    # and which happens as soon as the transition is narrow enough to pull the pair together.
    owner = np.argmin(np.abs(lad_tm[:, None] - np.asarray(LADDER_TMAX_LEVELS)[None, :]), axis=1)
    for k, level in enumerate(LADDER_TMAX_LEVELS):
        on = lad_tm[owner == k]
        assert on.size >= len(LADDER_TIMES), f"ladder level {level:.0f} C under-populated"
        assert on.max() - on.min() < 2.0, f"ladder level {level:.0f} C not iso-Tmax"
    core = np.array([tm[i] for i in range(len(tm)) if block[i] != "E"])
    assert core.min() >= BAND_LO_C - 1.0, f"a non-anchor point is too cold: {core.min():.0f} C"
    assert core.max() <= BAND_HI_C + 1.0, f"a non-anchor point is too hot: {core.max():.0f} C"
    assert sum(b == "E" for b in block) == 1, "expected exactly one floor anchor"


def _write_csv(plan: dict, path: Path) -> None:
    """Write the plan, recording every hypothesis rather than a single predicted label."""
    keys = list(plan["preds"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["index", "block", "voltage_V", "time_ms", "pred_Tmax_C"]
            + [f"pred_X_{k}" for k in keys]
            + ["model_disagreement", "tmax_node_supported", "readout", "note"]
        )
        for i in range(len(plan["V"])):
            w.writerow(
                [
                    i + 1,
                    plan["block"][i],
                    int(plan["V"][i]),
                    plan["t"][i],
                    round(float(plan["tmax"][i]), 1),
                ]
                + [round(float(plan["preds"][k][i]), 3) for k in keys]
                + [
                    round(float(plan["disagree"][i]), 3),
                    bool(plan["t"][i] >= T_SEARCH_LO),
                    "eps_r",
                    plan["note"][i],
                ]
            )


def _figure(plan: dict, path: Path) -> None:
    """Three panels: where the points sit, what the hypotheses disagree about, and the ladder."""
    models = plan["models"]
    vg = np.linspace(V_LO, V_HI, 220)
    tg = np.linspace(T_SEARCH_LO, T_SEARCH_HI, 220)
    vv, tt = np.meshgrid(vg, tg)
    colors = {"A": "#e41a1c", "B": "#377eb8", "E": "#4daf4a", "D": "#ff7f00"}
    labels = {
        "A": "A  iso-T$_{max}$ ladder",
        "B": "B  stratified core",
        "E": "E  floor anchor",
        "D": "D  replicate",
    }

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    a = axes[0]
    cf = a.contourf(vv, tt, FLASH.tmax(vv, tt), levels=18, cmap="inferno")
    fig.colorbar(cf, ax=a).set_label("measured T$_{max}$ (°C)")
    styles = {"isoT": ":", "ramp": "--", "lamp": "-", "diffusion": "-.", "rect": "-."}
    for key, m in models.items():
        a.contour(
            vv,
            tt,
            m.fraction_grid(vg, tg),
            levels=[0.5],
            colors="cyan",
            linewidths=2.4 if key == DEFAULT_MODEL else 1.4,
            linestyles=styles[key],
        )
    for b in "ABED":
        s = [i for i, x in enumerate(plan["block"]) if x == b]
        if s:
            a.scatter(
                plan["V"][s],
                plan["t"][s],
                c=colors[b],
                s=110,
                marker="D" if b == "D" else "o",
                edgecolors="w",
                linewidths=1.5,
                zorder=5,
                label=labels[b],
            )
    a.set_xlabel("flash voltage V (V)")
    a.set_ylabel("flash time t (ms)")
    a.set_title(
        "Seed conditions on the measured T$_{max}$ field\n"
        "cyan = X=0.5 under each hypothesis (solid = diffusion, dotted = iso-T$_{max}$)",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(loc="upper left", fontsize=8, framealpha=0.9)

    a = axes[1]
    dg = np.stack([m.fraction_grid(vg, tg) for m in models.values()])
    cf = a.contourf(vv, tt, dg.max(0) - dg.min(0), levels=18, cmap="viridis")
    fig.colorbar(cf, ax=a).set_label("ensemble disagreement (max − min X)")
    for b in "ABED":
        s = [i for i, x in enumerate(plan["block"]) if x == b]
        if s:
            a.scatter(
                plan["V"][s],
                plan["t"][s],
                c=colors[b],
                s=110,
                marker="D" if b == "D" else "o",
                edgecolors="w",
                linewidths=1.5,
                zorder=5,
            )
    a.set_xlabel("flash voltage V (V)")
    a.set_ylabel("flash time t (ms)")
    a.set_title(
        "Where the hypotheses disagree\n(a shot where they agree settles nothing)",
        fontweight="bold",
        fontsize=10,
    )

    a = axes[2]
    v_a, t_a, lvl_a = plan["ladder"]
    for li, level in enumerate(LADDER_TMAX_LEVELS):
        on = np.isclose(lvl_a, level)
        order = np.argsort(t_a[on])
        for key, m in models.items():
            a.plot(
                t_a[on][order],
                m.fraction(v_a[on], t_a[on])[order],
                marker="o" if li == 0 else "s",
                lw=2.4 if key == DEFAULT_MODEL else 1.4,
                ls=styles[key],
                alpha=1.0 if li == 0 else 0.55,
                label=f"{key} (tilt {m.tilt_c(*LADDER_DWELL_RANGE):.0f} °C)" if li == 0 else None,
            )
    a.axhline(0.5, color="gray", lw=1, ls=":")
    a.set_xscale("log")
    a.set_xlabel("flash time t (ms)   [same T$_{max}$ at every point]")
    a.set_ylabel("predicted crystalline fraction X")
    a.set_ylim(-0.05, 1.05)
    a.set_title(
        "Block A: the tilt test, two T$_{max}$ levels\n"
        f"({LADDER_TMAX_LEVELS[0]:.0f} / {LADDER_TMAX_LEVELS[1]:.0f} °C) — "
        "flat lines = pure temperature threshold",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(fontsize=8.5)

    plt.tight_layout()
    save_figure(fig, str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-core", type=int, default=7, help="size of the stratified core block")
    ap.add_argument("--seed", type=int, default=7, help="base RNG seed for the core LHS")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    plan = make_plan(args.n_core, args.seed)
    _check(plan)
    keys = list(plan["preds"])

    n = len(plan["V"])
    print(f"Seed flash plan: {n} shots over V=[{V_LO:.0f},{V_HI:.0f}] V, t=[{T_LO},{T_HI}] ms")
    print(f"  boundary search restricted to t in [{T_SEARCH_LO}, {T_SEARCH_HI}] ms (table nodes)")
    print(f"  informative band Tmax in [{BAND_LO_C:.0f}, {BAND_HI_C:.0f}] C\n")
    head = "  #  blk    V   t(ms)   Tmax  " + "".join(f"{k:>11s}" for k in keys) + "   spread"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for i in range(n):
        row = f"{i + 1:3d}   {plan['block'][i]}   {int(plan['V'][i]):4d} {plan['t'][i]:6.1f} "
        row += f"{plan['tmax'][i]:6.1f}  "
        row += "".join(f"{plan['preds'][k][i]:11.3f}" for k in keys)
        row += f"   {plan['disagree'][i]:6.3f}"
        print(row)

    v_a, t_a, lvl_a = plan["ladder"]
    print("\n  Block A tilt test -- swing in X along each level, shortest to longest pulse:")
    hdr = "    " + f"{'model':10s}" + "".join(f"{lv:>12.0f} C" for lv in LADDER_TMAX_LEVELS)
    print(hdr + f"{'tilt':>10s}")
    for k, m in plan["models"].items():
        row = f"    {k:10s}"
        for level in LADDER_TMAX_LEVELS:
            on = np.isclose(lvl_a, level)
            x = m.fraction(v_a[on], t_a[on])
            row += f"{x[np.argmax(t_a[on])] - x[np.argmin(t_a[on])]:+14.3f}"
        print(row + f"{m.tilt_c(*LADDER_DWELL_RANGE):9.0f} C")

    _write_csv(plan, ROOT / "data" / "flash_plan_seed.csv")
    _figure(plan, OUT / "flash_plan.png")
    print(f"\nSaved -> data/flash_plan_seed.csv, {OUT / 'flash_plan.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
