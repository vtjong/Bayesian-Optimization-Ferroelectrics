"""How many of the 80 specimens should the seed take?

The question is not "which seed size maps the boundary best" -- a bigger seed always maps better,
because it is more data. It is how the WHOLE campaign performs, because every specimen spent on
the seed is one the adaptive loop does not get. A seed too small leaves the surrogate unable to
propose anything sensible and the loop wanders; a seed too large spends on blind coverage what
targeted follow-up would have spent better.

So this runs the closed loop end to end: seed, fit, propose a batch, measure, refit, repeat to the
budget. It reports the misclassified area the campaign FINISHES at, and how many specimens it
needed to reach a usable accuracy, as a function of seed size.

THE ACQUISITION IS TARGETED VARIANCE, NOT BOUNDARY ENTROPY. Entropy on the class probability is
maximal on the whole predicted contour for any posterior width -- verified identical to machine
precision across three orders of magnitude in the latent sd -- so it cannot rank a resolved
boundary point against an unexplored one, and it is the criterion that failed to converge on three
of four benchmarks in the founding level-set paper. Targeted variance weights the posterior
variance toward the threshold, so it collapses at a measured point and genuinely moves on.

Worlds come from ``validation.worlds``: the 30 measured peak temperatures are held fixed and
everything else -- response family, transition location and width, dwell dependence, a non-thermal
voltage channel, noise scale, and the readout's floor, span and saturation -- is sampled.

Usage:  python src/run_seed_size.py [--worlds 40] [--budget 80] [--batch 4]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from active_learning.surrogate import BoundarySurrogate
from campaign.plan import N_REPLICATES, T_SEARCH_HI, T_SEARCH_LO
from design_space import snap
from validation.designs import thermal_lhs, with_replicates
from validation.evaluate import has_boundary, supported_grid
from validation.worlds import sample_world

SEED_SIZES = (8, 12, 16, 20, 24, 32)
BATCH_SIZES = (2, 4, 6, 8)
USABLE_TOLERANCE = 0.05  # misclassified area a campaign could act on
CAND_V, CAND_T = 48, 40  # proposal grid


def _candidates() -> tuple:
    """Settable conditions the loop may propose."""
    vv, tt = supported_grid(T_SEARCH_LO, T_SEARCH_HI)
    v = np.linspace(vv.min(), vv.max(), CAND_V)
    t = np.geomspace(T_SEARCH_LO, T_SEARCH_HI, CAND_T)
    g = np.array([snap(a, b) for a in v for b in t])
    return g[:, 0], g[:, 1]


def targeted_variance(gp: BoundarySurrogate, v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Posterior variance weighted toward the boundary; zero where the field is already resolved.

    :param gp: fitted surrogate.
    :param v: candidate voltages.
    :param t: candidate flash times (ms).
    """
    mu, sd = gp.latent(v, t)
    total = sd**2 + 1e-9
    return sd**2 * np.exp(-0.5 * mu**2 / total) / np.sqrt(2.0 * np.pi * total)


def propose(gp: BoundarySurrogate, k: int, taken: set, cv: np.ndarray, ct: np.ndarray) -> tuple:
    """Greedy batch by targeted variance, refusing conditions already fired.

    Each pick is removed from the pool with a small exclusion around it, which is the cheap stand-in
    for fantasy conditioning; at k = 4 the difference is not what decides this study.

    :param gp: fitted surrogate.
    :param k: batch size.
    :param taken: conditions already fired.
    :param cv: candidate voltages.
    :param ct: candidate flash times (ms).
    """
    score = targeted_variance(gp, cv, ct)
    picked_v, picked_t = [], []
    for _ in range(k):
        order = np.argsort(-score)
        for j in order:
            key = (int(cv[j]), round(float(ct[j]), 1))
            if key in taken:
                continue
            picked_v.append(cv[j])
            picked_t.append(ct[j])
            taken.add(key)
            near = (np.abs(cv - cv[j]) < 12) & (np.abs(np.log(ct / ct[j])) < 0.15)
            score[near] *= 0.05
            break
    return np.array(picked_v), np.array(picked_t)


def run_campaign(world, n_seed: int, budget: int, batch: int, seed: int, vv, tt) -> dict:
    """One full campaign: seed, then adaptive batches to the budget.

    :param world: sampled world.
    :param n_seed: specimens spent on the seed.
    :param budget: total specimens.
    :param batch: specimens per adaptive round.
    :param seed: RNG seed for the seed design.
    :param vv: evaluation voltage grid.
    :param tt: evaluation time grid.
    """
    n_draw = max(n_seed - N_REPLICATES, 2)
    v, t = thermal_lhs(n_draw, T_SEARCH_LO, T_SEARCH_HI, seed)
    reps = list(np.argsort(np.argsort(np.linspace(0, n_draw - 1, N_REPLICATES + 2)[1:-1])))
    v, t = with_replicates(v, t, reps[:N_REPLICATES])
    rng = np.random.default_rng(seed)
    y = world.observe(world.truth(v, t), rng)

    cv, ct = _candidates()
    taken = {(int(a), round(float(b), 1)) for a, b in zip(v, t)}
    truth_cls = world.truth(vv.ravel(), tt.ravel()) > 0.5
    first_usable = None
    mis = 1.0
    while True:
        gp = BoundarySurrogate().fit(v, t, y)
        mis = float(np.mean(gp.crystalline_side(vv.ravel(), tt.ravel()) != truth_cls))
        if first_usable is None and mis < USABLE_TOLERANCE:
            first_usable = len(v)
        if len(v) + batch > budget:
            break
        nv, nt = propose(gp, batch, taken, cv, ct)
        if nv.size == 0:
            break
        ny = world.observe(world.truth(nv, nt), rng)
        v, t, y = np.concatenate([v, nv]), np.concatenate([t, nt]), np.concatenate([y, ny])
    return {"final": mis, "to_usable": first_usable, "spent": len(v)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worlds", type=int, default=40)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--batch", type=int, nargs="+", default=list(BATCH_SIZES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vv, tt = supported_grid(T_SEARCH_LO, T_SEARCH_HI)
    batches = list(args.batch)
    acc = {(n, k): {"final": [], "usable": []} for n in SEED_SIZES for k in batches}
    rng = np.random.default_rng(args.seed)
    kept = 0
    while kept < args.worlds:
        w = sample_world(rng)
        if not has_boundary(w.truth, vv, tt):
            continue
        kept += 1
        for n in SEED_SIZES:
            for k in batches:
                r = run_campaign(w, n, args.budget, k, seed=7 + kept, vv=vv, tt=tt)
                acc[(n, k)]["final"].append(r["final"])
                acc[(n, k)]["usable"].append(r["to_usable"])

    print(f"=== seed size x batch size, closed loop to a {args.budget}-specimen budget ===")
    print(f"  {kept} worlds\n")
    print("MEAN misclassified area where the campaign ENDS")
    print("%7s" % "n_seed" + "".join(f"{'k=' + str(k):>10s}" for k in batches))
    for n in SEED_SIZES:
        row = f"{n:7d}"
        for k in batches:
            row += f"{np.mean(acc[(n, k)]['final']):10.3f}"
        print(row)
    print("\np90 (the tail a one-shot campaign has to survive)")
    print("%7s" % "n_seed" + "".join(f"{'k=' + str(k):>10s}" for k in batches))
    for n in SEED_SIZES:
        row = f"{n:7d}"
        for k in batches:
            row += f"{np.percentile(acc[(n, k)]['final'], 90):10.3f}"
        print(row)
    print("\n%% of worlds ever reaching <5%% misclassified, and median specimens to get there")
    print("%7s" % "n_seed" + "".join(f"{'k=' + str(k):>12s}" for k in batches))
    for n in SEED_SIZES:
        row = f"{n:7d}"
        for k in batches:
            u = [x for x in acc[(n, k)]["usable"] if x is not None]
            tot = len(acc[(n, k)]["usable"])
            row += f"{len(u) / max(tot, 1):8.0%}/{int(np.median(u)) if u else 0:<3d}"
        print(row)
    print("\n  'final mis' is where the campaign ENDS after spending all 80 specimens.")
    print("  A seed too small leaves the loop proposing from a surrogate that knows nothing;")
    print("  a seed too large spends on blind coverage what targeted follow-up would spend better.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
