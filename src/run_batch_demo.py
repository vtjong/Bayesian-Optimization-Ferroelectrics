"""Batch ("qEntropy") vs sequential boundary mapping -- for beamtime, where a round measures many.

Beamtime measures several (V, t) conditions per round, not one. This compares a greedy BATCH
entropy acquisition -- pick the top point, fantasize its value at the posterior mean (Kriging
believer), re-condition at fixed hyperparameters, repeat q times -- against pure SEQUENTIAL
(1 shot/round), at a fixed active-shot budget. The GP's own correlations spread the batch along
the boundary, so the q points are diverse without measuring in between (the entropy analogue of
BoTorch's qUCB/qEI sequential greedy).

Reports (A) the convergence of the boundary-map error vs cumulative shots up to an 80-measurement
budget, for sequential q=1 and small batches q=2-4; and (B) one greedy batch spread along the
boundary.

Usage:  python src/run_batch_demo.py [--seeds 12]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

import validation.picker as pk
from physics.thermal_model import T_HI, T_LO, V_HI, V_LO
from visualization.base import save_figure

OUT = Path(__file__).resolve().parent.parent / "predictions" / "batch_demo"
N_SEED = 10  # LHS seed points
TOTAL = 80  # total budget (delivered as two halves of 40)
QS = [1, 2, 3, 4]  # sequential (q=1) + small feasible batches
COL = {1: "#7a7a7a", 2: "#2f7bbf", 3: "#2e8b57", 4: "#c26a1f"}
CFG = pk.BoundaryConfig(noise_boundary=0.30)  # permittivity noise regime (peak sigma_n ~0.10)


def convergence(q, seeds):
    """Boundary-map error vs cumulative shots (to TOTAL) for batch size q."""
    nr = max(1, round((TOTAL - N_SEED) / q))
    H = np.array(
        [
            pk.run_active_batch(q=q, n_seed=N_SEED, n_rounds=nr, seed=s, cfg=CFG)["err"]
            for s in range(seeds)
        ]
    )
    samples = N_SEED + q * np.arange(H.shape[1])
    return samples, H.mean(0), H.std(0) / np.sqrt(seeds), nr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    conv = {}
    print(f"convergence to {TOTAL} shots ({N_SEED} LHS seed), {args.seeds} seeds")
    print("batch q | rounds | err @40 shots | err @80 shots")
    for q in QS:
        s, m, e, nr = convergence(q, args.seeds)
        conv[q] = (s, m, e)
        print(f"  q={q}    |  {nr:2d}   |    {np.interp(40, s, m):.3f}      |    {m[-1]:.3f}")

    # one greedy batch (q=4) from an LHS seed, to show the spread along the boundary
    rng = np.random.default_rng(0)
    Vseed, tseed = pk.lhs_seed(N_SEED, rng)
    ys = list(pk.measure(np.array(Vseed), np.array(tseed), rng))
    gp = pk.fit_gp(list(Vseed), list(tseed), ys)
    Vc, tc = pk.candidate_grid()
    Vb, tb = pk.propose_batch(gp, list(Vseed), list(tseed), ys, 4, pk.DEFAULT_ACQ, Vc, tc)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    # panel A: convergence to 80 shots for q = 1..4
    for q in QS:
        s, m, e = conv[q]
        lbl = "sequential (q=1)" if q == 1 else f"batch q={q}"
        a1.plot(s, m, "-", color=COL[q], lw=1.9, label=lbl)
        a1.fill_between(s, m - e, m + e, color=COL[q], alpha=0.12)
    for x, lab in [(40, "40"), (80, "budget = 80")]:
        a1.axvline(x, color="#444", ls=":", lw=1.1)
        a1.text(x - 1, 0.23, lab, rotation=90, va="top", ha="right", fontsize=8, color="#444")
    a1.set_xlabel("cumulative shots  (LHS seed + batch active rounds)")
    a1.set_ylabel("boundary-map error (misclassification area)")
    a1.set_title(
        "A. Convergence of the boundary map\n(batch q=2-4 tracks sequential)",
        fontweight="bold",
        fontsize=11,
    )
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)

    # panel B: the greedy batch spread along the boundary
    vs = np.linspace(V_LO, V_HI, 120)
    ts = np.linspace(T_LO, T_HI, 120)
    VV, TT = np.meshgrid(vs, ts)
    ftrue = pk.true_f(VV.ravel(), TT.ravel()).reshape(VV.shape)
    a2.contourf(VV, TT, pk.noise_sigma(ftrue), levels=15, cmap="Purples")
    a2.contour(VV, TT, ftrue, levels=[pk.DEFAULT.theta], colors="k", linewidths=2)
    a2.scatter(Vseed, tseed, c="white", edgecolors="k", s=35, label="LHS seed (10)")
    a2.scatter(Vb, tb, c="#c26a1f", edgecolors="k", s=80, zorder=5, label="greedy batch (q=4)")
    for i, (v, t) in enumerate(zip(Vb, tb), 1):
        a2.annotate(str(i), (v, t), fontsize=8, fontweight="bold", ha="center", va="center")
    a2.set_xlabel("voltage V")
    a2.set_ylabel("pulse time t (ms)")
    a2.set_title(
        "B. One greedy batch (q=4) spreads along the boundary\n(numbered in pick order)",
        fontweight="bold",
        fontsize=11,
    )
    a2.legend(fontsize=8, loc="upper right")
    save_figure(fig, str(OUT / "batch_demo.png"))
    print(f"\nSaved -> {OUT / 'batch_demo.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
