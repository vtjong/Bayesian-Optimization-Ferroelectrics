"""Batch ("qEntropy") vs sequential boundary mapping -- for beamtime, where a round measures many.

Beamtime measures several (V, t) conditions per round, not one. This compares a greedy BATCH
entropy acquisition -- pick the top point, fantasize its value at the posterior mean (Kriging
believer), re-condition at fixed hyperparameters, repeat q times -- against pure SEQUENTIAL
(1 shot/round), at a fixed active-shot budget. The GP's own correlations spread the batch along
the boundary, so the q points are diverse without measuring in between (the entropy analogue of
BoTorch's qUCB/qEI sequential greedy).

Reports (A) final boundary-map error vs batch size q: batching cuts the number of rounds from ~24
to a handful with no meaningful loss; and (B) one greedy batch spread along the boundary.

Usage:  python src/run_batch_demo.py [--seeds 20]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

import discovery.picker as pk
from discovery.synthetic import T_HI, T_LO, V_HI, V_LO
from visualization.base import save_figure

OUT = Path(__file__).resolve().parent.parent / "predictions" / "batch_demo"
ACTIVE = 24                       # fixed active-shot budget (on top of the LHS seed)
QS = [1, 2, 4, 6, 8, 12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pk._S1 = 0.30                  # permittivity noise regime (peak sigma_n ~ 0.10)

    means, sems, rounds = [], [], []
    print(f"batch q | rounds | final boundary-map err (mean +/- SEM, {args.seeds} seeds), "
          f"{ACTIVE} active shots")
    for q in QS:
        nr = max(1, round(ACTIVE / q))
        e = [pk.run_active_batch(q=q, n_seed=10, n_rounds=nr, seed=s)["err"][-1]
             for s in range(args.seeds)]
        means.append(np.mean(e)); sems.append(np.std(e) / np.sqrt(args.seeds)); rounds.append(nr)
        print(f"  q={q:2d}   |  {nr:2d}    | {means[-1]:.3f} +/- {sems[-1]:.3f}")
    means, sems = np.array(means), np.array(sems)

    # one greedy batch (q=8) from an LHS seed, to show the spread along the boundary
    rng = np.random.default_rng(0)
    Vseed, tseed = pk.lhs_seed(10, rng)
    ys = list(pk.measure(np.array(Vseed), np.array(tseed), rng))
    gp = pk.fit_gp(list(Vseed), list(tseed), ys)
    Vc, tc = pk.candidate_grid()
    Vb, tb = pk.propose_batch(gp, list(Vseed), list(tseed), ys, 8, "lse", Vc, tc)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    # panel A: batch size vs error, with the sequential (q=1) band for reference
    a1.axhspan(means[0] - sems[0], means[0] + sems[0], color="#7a7a7a", alpha=0.18,
               label="sequential (q=1) +/-1 SEM")
    a1.errorbar(QS, means, yerr=sems, fmt="o-", color="#c26a1f", capsize=3, label="batch qEntropy")
    for q, m, nr in zip(QS, means, rounds):
        a1.annotate(f"{nr} rounds", (q, m), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#555")
    a1.set_xlabel("batch size q  (shots measured together per round)")
    a1.set_ylabel("final boundary-map error")
    a1.set_title("A. Batch matches sequential in far fewer rounds\n(same ~24 active shots)",
                 fontweight="bold", fontsize=11)
    a1.legend(fontsize=9); a1.grid(alpha=0.3)

    # panel B: the greedy batch spread along the boundary
    vs = np.linspace(V_LO, V_HI, 120); ts = np.linspace(T_LO, T_HI, 120)
    VV, TT = np.meshgrid(vs, ts)
    ftrue = pk.true_f(VV.ravel(), TT.ravel()).reshape(VV.shape)
    a2.contourf(VV, TT, pk.noise_sigma(ftrue), levels=15, cmap="Purples")
    a2.contour(VV, TT, ftrue, levels=[pk.THETA], colors="k", linewidths=2)
    a2.scatter(Vseed, tseed, c="white", edgecolors="k", s=35, label="LHS seed (10)")
    a2.scatter(Vb, tb, c="#c26a1f", edgecolors="k", s=70, zorder=5, label="greedy batch (q=8)")
    for i, (v, t) in enumerate(zip(Vb, tb), 1):
        a2.annotate(str(i), (v, t), fontsize=8, fontweight="bold", ha="center", va="center")
    a2.set_xlabel("voltage V"); a2.set_ylabel("pulse time t (ms)")
    a2.set_title("B. One greedy batch spreads along the boundary\n(numbered in pick order)",
                 fontweight="bold", fontsize=11)
    a2.legend(fontsize=8, loc="upper right")
    save_figure(fig, str(OUT / "batch_demo.png"))
    print(f"\nSaved -> {OUT / 'batch_demo.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
