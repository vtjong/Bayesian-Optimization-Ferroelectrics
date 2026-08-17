"""Generate the seed flash-condition plan for the crystallization-boundary campaign.

The design box is (flash voltage, flash time), but crystallization responds to PEAK TEMPERATURE,
which spans 82-563 C across that box while the transition is only tens of degrees wide. A Latin
hypercube laid out uniformly in (V, t) is therefore NOT uniform in the quantity that matters: most
of its points land where every hypothesis predicts the same thing. This generator stratifies the
measured peak temperature instead -- LHS over (Tmax, log t), inverted through the measured table to
(V, t) -- and spends part of the seed on a designed contrast rather than on coverage.

Four blocks (12 primary + 2 replicates of 80 total shots):

  A  iso-Tmax dwell ladder   4  same peak temperature, four pulse widths. Under a boundary that is
                                a pure Tmax level set all four read the same fraction; under
                                activated kinetics with a width-tracking cooling law they sweep
                                nearly 0 to 1. This is the one contrast that measures the boundary
                                tilt, and it costs nothing because the rung also brackets the
                                prior onset.
  B  stratified core        7  (Tmax, log t) Latin hypercube over the informative band, maximin
                                against block A. This is the GP's actual space-filling seed.
  E  amorphous floor        1  one cold condition to pin the readout's amorphous floor.
  D  replicates            +2  the two most boundary-adjacent A conditions, repeated on separate
                                specimens. If these scatter by more than measurement noise, a
                                variable outside (V, t) is in play and the 2-D map is not valid --
                                which we want to learn in the first batch, not the last.

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
)
from discovery.synthetic import FLASH, FLASH_T, T_HI, T_LO, V_HI, V_LO
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "flash_plan"

# Boundary-search points are restricted to flash times the measured table actually supports. The
# table has NO node between 0.1 and 2.6 ms even though Tmax climbs 189 -> 563 C across that gap and
# the surface maximum sits inside it, so any Tmax quoted there is an artifact of the spline.
T_SEARCH_LO, T_SEARCH_HI = float(FLASH_T[1]), float(FLASH_T[-1])  # 2.6 .. 10.1 ms

# Informative band. Onset prior 380 +/- 30 C; tilt 0-51 C across the usable time range puts the
# boundary anywhere in ~325-435 C; widening by the transition half-width gives roughly [292, 468].
# Outside [310, 490] every hypothesis in the ensemble agrees, so a shot there settles nothing.
BAND_LO_C, BAND_HI_C = 310.0, 490.0

# Block A uses TWO peak-temperature levels, not one. A single-level ladder is rank-deficient: it
# identifies only the product of transition sharpness and tilt, and it slides bodily into the
# saturated floor or ceiling if the true onset differs from the prior by ~1 sigma, at which point
# it reports "no tilt" even when the tilt is large. Two levels straddling the onset prior keep the
# estimate well-conditioned and roughly flat in precision across T_ONSET_C +/- T_ONSET_SIGMA_C.
LADDER_TIMES = (2.6, 5.1, 10.1)  # measured table rows -> no spline extrapolation
LADDER_MARGIN_C = 6.0  # keep levels off the reachable edge, where inversion is ill-conditioned
LADDER_MIN_GAP_C = 6.0  # two levels closer than this are one level
LADDER_LEVEL_STEP_C = 2.0  # search resolution


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
N_REPLICATES = 2


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

    :param models: the boundary-model ensemble.
    :param t: flash time (ms).
    :param lo_x: lower fraction defining the informative range.
    :param hi_x: upper fraction defining the informative range.
    """
    grid = np.linspace(BAND_LO_C, BAND_HI_C, 400)
    v = FLASH.voltages_for_tmax(grid, np.full_like(grid, t))
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return BAND_LO_C, BAND_HI_C
    x = np.mean([m.fraction(v[ok], np.full(ok.sum(), t)) for m in models.values()], axis=0)
    inside = grid[ok][(x >= lo_x) & (x <= hi_x)]
    if inside.size < 2:
        return BAND_LO_C, BAND_HI_C
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
    # per candidate design would dominate the search cost.
    band_t = np.geomspace(T_SEARCH_LO, T_SEARCH_HI, 24)
    _edges = np.array([_transition_band(models, x) for x in band_t])
    band_lo_v, band_hi_v = _edges[:, 0], _edges[:, 1]
    for k in range(n_tries):
        u = qmc.LatinHypercube(d=2, seed=seed + k).random(n)
        t = np.round(10.0 ** (log_lo + u[:, 0] * (log_hi - log_lo)) * 10.0) / 10.0
        reach_lo = np.array([FLASH.tmax_range(x)[0] for x in t])
        reach_hi = np.array([FLASH.tmax_range(x)[1] for x in t])
        # Stratify across the range where the ensemble actually transitions, not uniformly over the
        # full prior band: the band is ~180 C wide while the transition is ~47 C, so uniform
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

    # Replicates go where a hidden variable would do the most damage, which is two different
    # places: the rung sitting mid-transition under EVERY hypothesis (largest readout noise, so
    # the most sensitive test of reproducibility), and the rung where the hypotheses disagree most
    # (a scatter artifact there would masquerade as boundary tilt).
    stack = np.stack([m.fraction(v_a, t_a) for m in models.values()])
    rep_idx = list(dict.fromkeys([int(np.argmin(np.abs(stack.mean(0) - 0.5))),
                                  int(np.argmax(stack.max(0) - stack.min(0)))]))[:N_REPLICATES]

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
    for level in LADDER_TMAX_LEVELS:
        on = lad_tm[np.abs(lad_tm - level) < 15.0]
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
                label=f"{key} (tilt {m.tilt_c():.0f} °C)" if li == 0 else None,
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
    ap.add_argument("--n-core", type=int, default=5, help="size of the stratified core block")
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
        print(row + f"{m.tilt_c():9.0f} C")

    _write_csv(plan, ROOT / "data" / "flash_plan_seed.csv")
    _figure(plan, OUT / "flash_plan.png")
    print(f"\nSaved -> data/flash_plan_seed.csv, {OUT / 'flash_plan.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
