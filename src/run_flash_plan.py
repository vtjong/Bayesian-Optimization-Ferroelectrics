"""Generate the seed flash-condition plan for the crystallization-boundary campaign.

A Latin hypercube over the two knobs, stratified in log time, plus two replicates. That is a
deliberately plain design, and it is plain because a targeted one was tested against it and lost.

WHY NOT A TARGETED DESIGN. The obvious move is to spend most of the batch near where the
transition is believed to be, and to buy a matched-temperature dwell contrast there. That design
was built, and it is better -- IF the belief is right. Counting distinct conditions that land in
the transition, as a function of where the transition actually turns out to be:

    true onset      targeted      hypercube
       380 C               0              3
       440 C               5              3
       447 C               7              2      <- what the targeted design bets on
       480 C               0              2
       500 C               0              2

Seven against two where the bet lands, zero against two once it is 33 C out. The belief is a
bracket from six archival samples at one flash time, on a switching figure of merit, from a
different sample set, bracketed by a single voltage step. Averaged over the range that belief
honestly spans, both designs put the same 1.2 conditions in the transition -- but the hypercube
does it reliably and the targeted design does it as a coin flip.

The comparison was run over randomly drawn worlds in which the response family, the transition
location and width, the dwell dependence, a non-thermal voltage channel, the noise scale and the
readout's floor, span and saturation are all sampled independently, with only the 30 MEASURED peak
temperatures held fixed (``discovery.worlds``). Worst-case misclassified area over the supported
box, at the 90th percentile: hypercube 0.25, targeted 0.58. The targeted design also lost on the
dwell question it was built for -- 0.30 power against 0.36 -- so it was dominated on both.

WHY TWO REPLICATES. Averaged over five hypercube realizations, 0 and 2 replicates are
indistinguishable on coverage (p90 0.250 vs 0.247) while 3 and 4 degrade it (0.292, 0.440). Two
therefore cost nothing measurable and buy two things: a check that (V, t) describes the experiment
at all -- archival replicates on this tool disagree by up to 2.08x -- and the repeated measurement
that separates readout noise from surface structure, which is otherwise confounded at this budget.

WHAT THE BATCH CANNOT DO. No 16-shot design answers the dwell question under honest uncertainty.
The best of everything tested calls a large tilt 47% of the time at a 16% false-positive rate;
this one manages 36% at 22%. That is not a defect of the design, it is the budget. The dwell
question has to accumulate across the campaign, informed by where the boundary actually turns out
to be -- which is what this batch is for.

Every condition is scored against the whole boundary-model ensemble and the CSV records each
member's prediction, so no single model's label is presented as the answer. The scoring is
REPORTING only; nothing in the placement uses it.

Usage:  python src/run_flash_plan.py [--n-core 14] [--seed 7]
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

from discovery.constants import T_TRANSITION_REF_C, T_TRANSITION_SIGMA_C
from discovery.kinetics import (
    DEFAULT_MODEL,
    build_ensemble,
    disagreement,
)
from discovery.synthetic import FLASH, FLASH_T, T_HI, T_LO, V_HI, V_LO
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "flash_plan"

# Boundary-search points are restricted to flash times the measured table actually supports. The
# table has NO node between 0.1 and 2.6 ms even though Tmax climbs 189 -> 563 C across that gap and
# the surface maximum sits inside it, so any Tmax quoted there is an artifact of the spline.
T_SEARCH_LO, T_SEARCH_HI = float(FLASH_T[1]), float(FLASH_T[-1])  # 2.6 .. 10.1 ms

# The batch is restricted to flash times the measured table supports, and to nothing else. A
# targeted design that concentrated conditions near the believed transition -- two matched
# peak-temperature levels crossed with two flash times -- was built and tested against this one. It
# was dominated on both the boundary-mapping metric and the dwell question it existed for, because
# concentration only pays if the belief is precise and ours is a bracket from six archival samples.
# See docs for the comparison; the machinery for it has been removed rather than left dormant.

SEED_SIZE = 16  # of an 80-specimen campaign
N_REPLICATES = 2  # measured: 0 and 2 are indistinguishable on coverage, 3+ degrade it
N_DRAWN = SEED_SIZE - N_REPLICATES

# The hypercube is stratified over PEAK TEMPERATURE and inverted through the measured table, not
# drawn over the raw knobs. Which of those wins turns entirely on how much is assumed about where
# the transition sits, and that dependence is worth stating because it nearly went unnoticed:
#
#   transition sampled uniformly 360-620 C  ->  knobs 0.508, temperature 0.511   (a tie)
#   transition sampled 420-560 C            ->  knobs 0.455, temperature 0.301
#
# (worst-case misclassified area, 90th percentile, over randomized worlds). The second prior is the
# defensible one. Six campaign-tool samples at 5.0 ms put a non-crystallized film at 434.7 C and a
# crystallized one at 458.7 C, so a transition at 380 C is not merely unlikely, it is contradicted
# by the measurement. A uniform prior reaching down to 360 C throws that evidence away, and under
# it the raw knobs look better only because nothing else is known.
#
# Stratifying temperature is therefore NOT a bet on the bracket midpoint -- the draw spans the whole
# reachable range and the design is scored against transitions anywhere in it. It is a bet that the
# film responds to temperature rather than to voltage, which the measured table already asserts.
LHS_TRIES = 400  # realizations to score; the best by maximin separation is kept

# Coldest peak temperature worth a specimen. Crystallizing 10 nm HZO on a millisecond timescale is
# not plausible below this on any kinetics, and the campaign-tool samples put the transition far
# above it. A physical floor, not an archival one.
DRAW_LO_C = 350.0
DRAW_MARGIN_C = 6.0  # keep draws off the reachable edge, where the inversion is ill-conditioned
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






def draw_block(n: int, seed: int, n_tries: int = LHS_TRIES) -> tuple:
    """Latin hypercube over ``(Tmax, log t)``, inverted through the measured table.

    Repeats over ``n_tries`` realizations and keeps the one with the largest minimum separation in
    normalized coordinates. A hypercube guarantees each axis is stratified but says nothing about
    the pair, so a realization can still leave two conditions almost coincident; maximin removes
    that without importing any belief about where the boundary is.

    :param n: number of conditions.
    :param seed: base RNG seed.
    :param n_tries: how many realizations to score.
    """
    best, best_sep = None, -np.inf
    log_lo, log_hi = np.log10(T_SEARCH_LO), np.log10(T_SEARCH_HI)
    for k in range(n_tries):
        u = qmc.LatinHypercube(d=2, seed=seed + k).random(n)
        t = np.round(10.0 ** (log_lo + u[:, 1] * (log_hi - log_lo)), 1)
        v = np.empty(n)
        for i, ti in enumerate(t):
            r_lo, r_hi = FLASH.tmax_range(float(ti))
            lo_c = max(r_lo + DRAW_MARGIN_C, DRAW_LO_C)
            hi_c = r_hi - DRAW_MARGIN_C
            if hi_c <= lo_c:  # this flash time cannot reach the window at all
                lo_c, hi_c = r_lo + DRAW_MARGIN_C, r_hi - DRAW_MARGIN_C
            v[i] = FLASH.voltage_for_tmax(lo_c + u[i, 0] * (hi_c - lo_c), float(ti))
        snapped = [_snap(a, b) for a, b in zip(v, t)]
        v = np.array([p[0] for p in snapped])
        t = np.array([p[1] for p in snapped], float)
        if len(set(zip(v.tolist(), t.tolist()))) < n:
            continue  # snapping collapsed two conditions onto one another
        sep = _min_separation(v, t)
        if sep > best_sep:
            best_sep, best = sep, (v, t)
    if best is None:
        raise RuntimeError("no feasible hypercube found")
    return best


def replicate_indices(v: np.ndarray, t: np.ndarray, n: int) -> list:
    """Which conditions to repeat on a second specimen: spread across peak temperature.

    The tempting choice is the condition nearest mid-transition, where the readout is noisiest and
    a disagreement is most visible. That was tried and rejected for two reasons. It needs the
    ensemble to say where mid-transition IS, which is the bet this design exists to avoid; and when
    the draw happens to place nothing near the boundary it silently replicates a SATURATED point
    instead, where two specimens agree trivially and the check is worthless.

    Spreading the replicates over the temperature range asks the reproducibility question in more
    than one regime and cannot degenerate that way. It also assumes nothing.

    :param v: drawn voltages.
    :param t: drawn flash times (ms).
    :param n: how many to replicate.
    """
    tm = FLASH.tmax(v, t)
    order = np.argsort(tm)
    # evenly spaced in rank, avoiding the two extremes where a replicate is least informative
    picks = np.linspace(0, len(order) - 1, n + 2)[1:-1]
    return [int(order[int(round(p))]) for p in picks]


def make_plan(n_core: int, seed: int) -> dict:
    """Assemble the seed plan and score every condition against the model ensemble.

    :param n_core: number of drawn conditions, before replicates.
    :param seed: RNG seed for the hypercube.
    """
    models = build_ensemble()
    v_d, t_d = draw_block(n_core, seed)
    rep = replicate_indices(v_d, t_d, N_REPLICATES)

    v = np.concatenate([v_d, v_d[rep]])
    t = np.concatenate([t_d, t_d[rep]])
    block = ["L"] * len(v_d) + ["R"] * len(rep)
    note = ["hypercube draw"] * len(v_d) + [
        f"replicate of L{i + 1} (separate specimen)" for i in rep
    ]
    order = np.lexsort((v, t, np.array([("LR".index(b)) for b in block])))
    return {
        "V": v[order],
        "t": t[order],
        "block": [block[i] for i in order],
        "note": [note[i] for i in order],
        "tmax": FLASH.tmax(v[order], t[order]),
        "preds": {k: m.fraction(v[order], t[order]) for k, m in models.items()},
        "disagree": disagreement(models, v[order], t[order]),
        "models": models,
    }


def _check(plan: dict) -> None:
    """Assert the design properties the plan is supposed to guarantee."""
    v, t, block = plan["V"], plan["t"], plan["block"]
    assert len(v) == SEED_SIZE, f"expected {SEED_SIZE} shots, got {len(v)}"
    assert np.all(t >= T_SEARCH_LO - 1e-9), "a condition lands in the un-noded 0.1-2.6 ms gap"
    assert np.all(t <= T_SEARCH_HI + 1e-9), "a condition exceeds the measured time range"
    assert np.all(v >= V_LO) and np.all(v <= V_HI), "a condition leaves the voltage box"
    assert np.all(v == np.round(v)) and np.allclose(t, np.round(t * 10) / 10), "not snapped"
    assert block.count("R") == N_REPLICATES, "wrong number of replicates"

    pairs = list(zip(v.astype(int).tolist(), np.round(t, 1).tolist()))
    drawn = [p for p, b in zip(pairs, block) if b == "L"]
    assert len(set(drawn)) == len(drawn), "two drawn conditions collided after snapping"
    for p, b in zip(pairs, block):
        if b == "R":
            assert p in drawn, "a replicate does not match any drawn condition"


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
    """Three panels: where the conditions sit, where the hypotheses disagree, and coverage."""
    models = plan["models"]
    vg = np.linspace(V_LO, V_HI, 220)
    tg = np.linspace(T_SEARCH_LO, T_SEARCH_HI, 220)
    vv, tt = np.meshgrid(vg, tg)
    colors = {"L": "#377eb8", "R": "#ff7f00"}
    labels = {"L": "hypercube draw", "R": "replicate"}

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
    # How much of the reachable temperature range the batch actually covers -- the property that
    # decided this design, since a targeted batch covers a narrow band superbly and the rest not
    # at all.
    tm = plan["tmax"]
    a.hist(tm, bins=12, color="#7fa8d1", edgecolor="k", linewidth=0.5)
    lo = T_TRANSITION_REF_C - T_TRANSITION_SIGMA_C
    hi = T_TRANSITION_REF_C + T_TRANSITION_SIGMA_C
    for x in (lo, hi):
        a.axvline(x, color="#b23", lw=1.4, ls="--")
    a.axvline(T_TRANSITION_REF_C, color="#b23", lw=2, label="proxy transition bracket")
    a.set_xlabel("peak temperature T$_{max}$ (°C)")
    a.set_ylabel("conditions")
    a.set_title(
        "Coverage of the reachable range\n(the batch does not bet on the bracket)",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(fontsize=8.5)

    plt.tight_layout()
    save_figure(fig, str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-core", type=int, default=N_DRAWN, help="number of drawn conditions before replicates"
    )
    ap.add_argument("--seed", type=int, default=7, help="base RNG seed for the hypercube")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    plan = make_plan(args.n_core, args.seed)
    _check(plan)
    keys = list(plan["preds"])

    n = len(plan["V"])
    print(f"Seed flash plan: {n} shots over V=[{V_LO:.0f},{V_HI:.0f}] V, t=[{T_LO},{T_HI}] ms")
    print(f"  restricted to t in [{T_SEARCH_LO}, {T_SEARCH_HI}] ms; the table has no node below")
    tm = plan["tmax"]
    print(f"  covers Tmax {tm.min():.0f}-{tm.max():.0f} C, and bets on none of it\n")
    head = "  #  blk    V   t(ms)   Tmax  " + "".join(f"{k:>11s}" for k in keys) + "   spread"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for i in range(n):
        row = f"{i + 1:3d}   {plan['block'][i]}   {int(plan['V'][i]):4d} {plan['t'][i]:6.1f} "
        row += f"{plan['tmax'][i]:6.1f}  "
        row += "".join(f"{plan['preds'][k][i]:11.3f}" for k in keys)
        row += f"   {plan['disagree'][i]:6.3f}"
        print(row)

    tm = plan["tmax"]
    print("\n  What the ensemble expects, as a check on coverage rather than a prediction:")
    mean_x = np.mean([plan["preds"][k] for k in keys], axis=0)
    print(f"    conditions the ensemble puts in the transition (0.05 < X < 0.95): "
          f"{int(np.sum((mean_x > 0.05) & (mean_x < 0.95)))} of {n}")
    print(f"    conditions it puts at the floor  (X < 0.05): {int(np.sum(mean_x <= 0.05))}")
    print(f"    conditions it puts at saturation (X > 0.95): {int(np.sum(mean_x >= 0.95))}")
    print("    Those counts assume the ensemble is right about WHERE the transition is. It is the")
    print("    bet this design declines to make -- but if the bracket is right, they are the cost.")

    _write_csv(plan, ROOT / "data" / "flash_plan_seed.csv")
    _figure(plan, OUT / "flash_plan.png")
    print(f"\nSaved -> data/flash_plan_seed.csv, {OUT / 'flash_plan.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
