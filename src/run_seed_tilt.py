"""Can a seed recover the DWELL COEFFICIENT when every assumption may be wrong?

``run_seed_robust`` asks how well a design maps the boundary. That is not the campaign's question.
The campaign's question is whether pulse width matters at fixed peak temperature -- the coefficient
``beta`` in

    X = f( Tmax - T0 + beta * ln(t / t_ref) )

-- because a zero answer means the boundary is a temperature contour and the remaining 64 specimens
have an easy job, while a large one means it is a folded surface.

THE ESTIMATOR IS A PROFILE, NOT A LOCAL OPTIMIZATION, and that is not a detail. Fitting all five
parameters at once with a local optimizer gives an answer that depends on where the optimizer
starts: on these worlds, moving the ``beta`` start from 0 to {0, 20, 40} swings RMSE by a factor of
two to three and flips the sign of the bias. The likelihood is nearly flat in ``beta``, so a point
estimate from a local fit is an artefact of the search. Profiling on a fixed grid -- re-fitting the
other four parameters at each ``beta`` -- removes the start dependence entirely, and the shape of
the profile is itself the answer: if SSE at ``beta = 0`` is barely worse than at the optimum, the
design cannot tell a flat boundary from a tilted one, whatever number the optimizer returns.

Metrics, in increasing order of usefulness:

  RMSE, corr    how close the profile minimum lands to the truth
  SSE ratio     SSE(beta = 0) / SSE(best). 1.0 means a flat boundary explains the data just as
                well as the true tilted one
  power / FPR   at a fixed ratio threshold: how often the design correctly calls a large tilt, and
                how often it cries tilt when the truth is flat. This is the pair a campaign
                actually cares about

Usage:  python src/run_seed_tilt.py [--worlds 160] [--seed 5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.append(str(Path(__file__).resolve().parent))

from discovery.constants import T_REF_MS
from discovery.evaluate import has_boundary, supported_grid
from discovery.synthetic import FLASH
from discovery.worlds import sample_world
from run_flash_plan import T_SEARCH_HI, T_SEARCH_LO
from run_seed_robust import build_designs

BETA_GRID = np.linspace(-20.0, 80.0, 21)  # K per e-fold; the sampler draws truth in [0, 60]
T0_STARTS = (410.0, 480.0, 545.0)
LARGE_TILT_K = 25.0  # a tilt this big changes what the campaign has to do
FLAT_TILT_K = 5.0  # at or below this the boundary is a temperature contour for practical purposes
RATIO_THRESHOLD = 1.5  # call it tilted when a flat fit is this much worse


def profile(v: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple:
    """Profile the sum of squares over ``beta``, re-fitting the other parameters at each value.

    The inner fit is warm-started from the previous grid point, which is both faster and more
    stable than restarting cold; the first point is fitted from several onsets so the sweep does
    not begin in a bad basin.

    :param v: as-fired voltages.
    :param t: as-fired flash times (ms).
    :param y: observed readings.
    """
    tm = FLASH.tmax(v, t)
    u = np.log(t / T_REF_MS)
    lo, span = float(np.min(y)), float(np.ptp(y)) or 1.0

    def sse(p, b):
        fl, amp, t0, ls = p
        z = np.clip(np.exp(ls) * (tm - t0 + b * u), -500.0, 500.0)
        return float(np.sum((y - (fl + amp / (1.0 + np.exp(-z)))) ** 2))

    out = np.empty(BETA_GRID.size)
    warm = None
    for i, b in enumerate(BETA_GRID):
        starts = [warm] if warm is not None else []
        starts += [np.array([lo, span, t0, -2.0]) for t0 in T0_STARTS] if i == 0 else []
        best, bv = None, np.inf
        for s0 in starts:
            r = minimize(
                sse, s0, args=(b,), method="Nelder-Mead", options={"maxiter": 4000, "fatol": 1e-10}
            )
            if r.fun < bv:
                best, bv = r.x, r.fun
        out[i] = bv
        warm = best
    return float(BETA_GRID[int(np.argmin(out))]), out


def run(n_worlds: int, seed: int, realizations) -> dict:
    """Profile every design over randomly drawn worlds.

    :param n_worlds: how many informative worlds to accumulate.
    :param seed: RNG seed.
    :param realizations: design-realization seeds to average over.
    """
    vv, tt = supported_grid(T_SEARCH_LO, T_SEARCH_HI)
    names = list(build_designs(realizations[0]))
    acc = {n: {"est": [], "true": [], "ratio": []} for n in names}
    rng = np.random.default_rng(seed)
    kept = tried = 0
    while kept < n_worlds and tried < 20 * n_worlds:
        tried += 1
        w = sample_world(rng)
        if not has_boundary(w.truth, vv, tt):
            continue  # uninformative for every design alike
        kept += 1
        for r in realizations:
            for name, (v, t) in build_designs(r).items():
                y = w.observe(w.truth(v, t), rng)
                b, prof = profile(v, t, y)
                zero = prof[int(np.argmin(np.abs(BETA_GRID)))]
                acc[name]["est"].append(b)
                acc[name]["true"].append(w.tilt_k)
                acc[name]["ratio"].append(zero / max(prof.min(), 1e-12))
    return {"acc": acc, "kept": kept}


def _report(out: dict) -> None:
    """RMSE and correlation, then the decision-relevant power / false-positive pair."""
    acc, kept = out["acc"], out["kept"]
    print(f"{kept} informative worlds (boundary inside the box), profile estimator.\n")
    print(f"{'design':30s} {'RMSE':>8s} {'corr':>7s} {'SSE(0)/SSE*':>13s}")
    rows = []
    for n, d in acc.items():
        e = np.asarray(d["est"], float)
        tr = np.asarray(d["true"], float)
        rows.append((np.sqrt(np.mean((e - tr) ** 2)), n, np.corrcoef(e, tr)[0, 1], d))
    for rmse, n, c, d in sorted(rows):
        print(f"{n:30s} {rmse:8.1f} {c:7.2f} {np.median(d['ratio']):13.2f}")

    print(f"\ncalling a tilt at SSE(0)/SSE* > {RATIO_THRESHOLD}:")
    hdr = f"power (tilt>{int(LARGE_TILT_K)})"
    print(f"{'design':30s} {hdr:>18s} {'FPR (tilt<5)':>14s}")
    for _, n, _, d in sorted(rows):
        tr = np.asarray(d["true"], float)
        ratio = np.asarray(d["ratio"], float)
        big, flat = tr >= LARGE_TILT_K, tr <= FLAT_TILT_K
        power = float(np.mean(ratio[big] > RATIO_THRESHOLD)) if big.any() else np.nan
        fpr = float(np.mean(ratio[flat] > RATIO_THRESHOLD)) if flat.any() else np.nan
        print(f"{n:30s} {power:18.2f} {fpr:14.2f}   (n={big.sum()}, {flat.sum()})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worlds", type=int, default=160)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--realizations", type=int, nargs="+", default=[7, 11])
    args = ap.parse_args()
    print("=== tilt recovery under randomized assumptions ===")
    print(f"  {args.worlds} worlds x {len(args.realizations)} realizations\n")
    _report(run(args.worlds, args.seed, args.realizations))
    return 0


if __name__ == "__main__":
    sys.exit(main())
