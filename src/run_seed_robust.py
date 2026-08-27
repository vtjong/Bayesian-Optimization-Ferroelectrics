"""Which seed survives when EVERY assumption that can be wrong is wrong at once?

``run_seed_stress`` varies the crystallization response but holds the measured peak-temperature
table fixed and shared between the truth and the design. That guarantees the single thing the
design most depends on, and it flatters designs that navigate in thermal coordinates. It also
assumes the calibrated noise model and a readout that reports crystalline fraction directly --
and the readout calibration is now known to be mislabeled.

This script removes all of those guarantees. Each trial draws a world from ``validation.worlds`` in
which the temperature scale, the local table accuracy, the response family, its location, width,
tilt and curvature, a non-thermal voltage channel, the noise scale, a heavy tail, per-specimen
scatter, and the readout's floor, span and saturation are ALL sampled independently.

    The DESIGN still navigates by the measured table, because that is all anyone has.
    The FILM responds to the world. They are not the same thing.

THE METRIC IS SCALE-FREE, and it has to be: with an unknown readout floor and span, the observed
number is an affine function of the crystalline fraction, so thresholding it at 1/2 would measure
the readout rather than the design. The fitted surface is instead normalized over the box to its
own range, split at the midpoint, and compared to the true X = 1/2 contour. That is what an
experimenter with an uncalibrated instrument would actually do.

Reported as the mean and the 90th percentile over worlds. The tail is the point: a seed is chosen
once, and the question is how badly it can leave us misled, not how it does on average.

Naive LHS is included deliberately. It assumes nothing at all, which is exactly why it is the
benchmark every model-informed design has to beat.

Usage:  python src/run_seed_robust.py [--worlds 150] [--seed 0]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from campaign.plan import (
    CORE_SIZE,
    FLOOR_CONDITION,
    T_SEARCH_HI,
    T_SEARCH_LO,
    explore_block,
    make_plan,
)
from validation.designs import catalogue
from validation.evaluate import has_boundary, misclassified_area, supported_grid
from validation.worlds import sample_world
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "seed_robust"
TAIL_PERCENTILE = 90


def build_designs(seed: int) -> dict:
    """The candidate catalogue, with the committed plan supplied from the generator.

    :param seed: realization seed for the randomized designs.
    """
    plan = make_plan(n_core=CORE_SIZE, seed=seed)
    is_ladder = np.isin(np.array(plan["block"]), ["A", "D"])
    return catalogue(
        committed=(plan["V"], plan["t"]),
        ladder=(plan["V"][is_ladder], plan["t"][is_ladder]),
        explore=explore_block,
        floor=FLOOR_CONDITION,
        t_lo=T_SEARCH_LO,
        t_hi=T_SEARCH_HI,
        seed=seed,
    )


def score(n_worlds: int, seed: int, realizations) -> dict:
    """Misclassification for every design across randomly sampled worlds.

    :param n_worlds: how many worlds to draw.
    :param seed: RNG seed.
    :param realizations: design-realization seeds to average over.
    """
    vv, tt = supported_grid(T_SEARCH_LO, T_SEARCH_HI)
    names = list(build_designs(realizations[0]))
    res = {n: [] for n in names}
    worst = {n: (None, -1.0) for n in names}
    rng = np.random.default_rng(seed)
    for _ in range(n_worlds):
        w = sample_world(rng)
        if not has_boundary(w.truth, vv, tt):
            continue
        for r in realizations:
            for name, (v, t) in build_designs(r).items():
                y = w.observe(w.truth(v, t), rng)
                m = misclassified_area(v, t, y, w.truth, vv, tt)
                res[name].append(m)
                if m > worst[name][1]:
                    worst[name] = (w.label, m)
    return {"res": res, "worst": worst}


def _report(out: dict) -> None:
    """Mean, tail and worst observed, plus which world hurt each design most."""
    res, worst = out["res"], out["worst"]
    names = list(res)
    n = min(len(v) for v in res.values())
    print(f"Misclassified fraction of the box, over {n} evaluated worlds per design.")
    print("The design navigates by the measured table; the film obeys the world.\n")
    tail = f"p{TAIL_PERCENTILE}"
    print(f"{'design':30s} {'mean':>8s} {'median':>8s} {tail:>8s} {'max':>8s}")
    ranked = sorted(names, key=lambda k: np.percentile(res[k], TAIL_PERCENTILE))
    for k in ranked:
        a = np.array(res[k])
        print(
            f"{k:30s} {a.mean():8.3f} {np.median(a):8.3f} "
            f"{np.percentile(a, TAIL_PERCENTILE):8.3f} {a.max():8.3f}"
        )
    print(f"\nRanked by p{TAIL_PERCENTILE} -- the tail is what a one-shot choice has to survive.\n")
    print("worst world for each design:")
    for k in ranked:
        print(f"  {k:30s} {worst[k][1]:.3f}   {worst[k][0]}")


def _figure(out: dict, path: Path) -> None:
    """Distribution of misclassification per design."""
    res = out["res"]
    names = sorted(res, key=lambda k: np.percentile(res[k], TAIL_PERCENTILE))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    data = [np.array(res[k]) for k in names]
    bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#7fa8d1")
        patch.set_alpha(0.75)
    for i, a in enumerate(data, start=1):
        ax.scatter(
            np.percentile(a, TAIL_PERCENTILE),
            i,
            marker="D",
            s=42,
            color="#b23",
            zorder=5,
            label=f"p{TAIL_PERCENTILE}" if i == 1 else None,
        )
    ax.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_xlabel("misclassified fraction of the supported box")
    ax.set_title(
        "Seed robustness when every assumption may be wrong at once\n"
        "(thermal scale and warp, response family, noise, readout floor/span/saturation)",
        fontweight="bold",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_figure(fig, str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worlds", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realizations", type=int, nargs="+", default=[7, 11])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== seed robustness under randomized assumptions ===")
    print(f"  {args.worlds} worlds x {len(args.realizations)} design realizations\n")
    out = score(args.worlds, args.seed, args.realizations)
    _report(out)
    _figure(out, OUT / "seed_robust.png")
    print(f"\nSaved -> {OUT / 'seed_robust.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
