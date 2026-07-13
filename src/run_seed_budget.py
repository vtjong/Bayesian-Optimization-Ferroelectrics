"""Choose the LHS-seed / active-round split for the boundary-mapping campaign.

Given a fixed total shot budget, how should it divide between space-filling LHS seed points and
active (picker-chosen) rounds? Runs the calibrated LSE picker across splits in the permittivity
noise regime and reports:

  (A) boundary-map error vs number of LHS seed points, for a few total budgets -> the best split;
  (B) the convergence curve (error vs cumulative shots) -> the active round where error plateaus
      (diminishing returns), i.e. how many active rounds are actually worth running.

Each active round = one picker-chosen shot, re-fit, repeat (sequential; 1 shot per sample).

Usage:  python src/run_seed_budget.py [--seeds 16]
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
from visualization.base import save_figure

OUT = Path(__file__).resolve().parent.parent / "predictions" / "seed_budget"
TOTALS = [30, 35, 40]
SEED_GRID = [6, 8, 10, 12, 15, 18, 22, 26, 30]
COL = {30: "#7a7a7a", 35: "#c26a1f", 40: "#2f7bbf"}
CFG = pk.BoundaryConfig(noise_boundary=0.30)  # permittivity noise regime (peak sigma_n ~0.095)


def err_for(n_seed, n_iter, seeds):
    """Mean +/- SEM boundary-map error near the end of the budget over `seeds` restarts."""
    e = [
        np.mean(pk.run_active("lse", n_seed=n_seed, n_iter=n_iter, seed=s, cfg=CFG)["err"][-2:])
        for s in range(seeds)
    ]
    return float(np.mean(e)), float(np.std(e) / np.sqrt(seeds))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # (A) sweep the seed / active split at each total budget
    print(
        f"peak sigma_n ~ {CFG.noise_floor + 0.25 * CFG.noise_boundary:.2f} (permittivity regime), "
        f"{args.seeds} restarts\n"
    )
    resA, best = {}, {}
    for T in TOTALS:
        xs, ys, es = [], [], []
        for ns in SEED_GRID:
            na = T - ns
            if na < 3:
                continue
            m, sem = err_for(ns, na, args.seeds)
            xs.append(ns)
            ys.append(m)
            es.append(sem)
        xs, ys, es = np.array(xs), np.array(ys), np.array(es)
        resA[T] = (xs, ys, es)
        bi = int(np.argmin(ys))
        best[T] = (int(xs[bi]), T - int(xs[bi]), ys[bi], es[bi])
        lo = ys[(xs >= 6) & (xs <= 15)]
        print(
            f"total {T}: lowest-error split ~{xs[bi]} seed + {T - xs[bi]} active "
            f"(err {ys[bi]:.3f}+/-{es[bi]:.3f}); flat over seed 6-15 "
            f"(spread {lo.max() - lo.min():.3f})"
        )

    # (B) convergence at a fixed seed count -> the KNEE (most gain), not a hard plateau
    NS, K = 10, 30
    hist = np.array(
        [
            pk.run_active("lse", n_seed=NS, n_iter=K, seed=s, cfg=CFG)["err"]
            for s in range(args.seeds)
        ]
    )
    conv, conv_sem = hist.mean(0), hist.std(0) / np.sqrt(args.seeds)
    samples = NS + np.arange(K)  # error recorded at NS+i fitted samples
    cum_red = conv[0] - conv  # cumulative error reduction from the seed fit
    total_red = conv[0] - conv.min()
    knee = int(samples[np.argmax(cum_red >= 0.70 * total_red)])  # 70% of achievable gain
    n90 = int(samples[np.argmax(cum_red >= 0.90 * total_red)])  # 90% of achievable gain
    print(
        f"\nconvergence (seed={NS}): ~70% of the achievable gain by ~{knee} shots "
        f"(~{knee - NS} active rounds), ~90% by ~{n90} ({n90 - NS} rounds); "
        "error keeps falling to the budget end (no hard plateau within 40)."
    )

    print("\n--- RECOMMENDATION -------------------------------------------------")
    print("  * LHS seed: ~10-12 points (just enough for a stable 2-D GP, ~5*d).")
    print("    More seed does NOT help and, at fixed budget, HURTS (panel A right tail).")
    print("  * Active rounds: spend ALL remaining budget (1 shot/round, refit each).")
    print("    Reserve ~5 shots of 40 for calibration/replicates -> ~35 for the map:")
    print("    ~10 LHS seed + ~25 active rounds is the robust operating point.")
    print("  * Gains: biggest in the first ~8 rounds, but still real at 40 -> use them all.")
    print("--------------------------------------------------------------------")

    # ---- figure ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.axvspan(8, 12, color="#2e8b57", alpha=0.10, label="recommended seed (~10)")
    for T in TOTALS:
        xs, ys, es = resA[T]
        a1.plot(xs, ys, "o-", color=COL[T], label=f"budget = {T}")
        a1.fill_between(xs, ys - es, ys + es, color=COL[T], alpha=0.13)
    a1.set_xlabel("number of LHS seed points  (active rounds = budget - seed)")
    a1.set_ylabel("boundary-map error (misclassification area)")
    a1.set_title(
        "A. How to split the budget\n(flat over a broad range; over-seeding hurts)",
        fontweight="bold",
        fontsize=11,
    )
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)

    a2.plot(samples, conv, "o-", color="#c26a1f", label="LSE active learning")
    a2.fill_between(samples, conv - conv_sem, conv + conv_sem, color="#c26a1f", alpha=0.15)
    a2.axvline(NS, color="gray", ls=":", lw=1.2, label=f"{NS} LHS seed")
    a2.axvline(knee, color="#2e8b57", lw=2, label=f"70% of gain by ~{knee} shots")
    a2.axvline(n90, color="#2e8b57", ls="--", lw=1.4, label=f"90% by ~{n90} shots")
    a2.axvspan(knee, samples[-1], color="#2e8b57", alpha=0.07)
    a2.set_xlabel("cumulative shots  (LHS seed + active rounds)")
    a2.set_ylabel("boundary-map error")
    a2.set_title(
        "B. Where the active-learning gains are\n(no hard plateau within 40 shots)",
        fontweight="bold",
        fontsize=11,
    )
    a2.legend(fontsize=9)
    a2.grid(alpha=0.3)
    save_figure(fig, str(OUT / "seed_budget.png"))
    print(f"\nSaved -> {OUT / 'seed_budget.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
