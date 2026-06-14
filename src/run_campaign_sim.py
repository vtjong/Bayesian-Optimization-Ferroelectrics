"""Closed-loop campaign simulator — test the round plan BEFORE spending the XRD budget.

Two studies on a synthetic process-window ground truth in (V, t):
  (1) Strategy comparison — active level-set acquisition vs random vs space-filling, scored
      by window-F1 (how well the polar-phase window is mapped) vs number of experiments.
  (2) Schedule sweep — several (batch q, rounds R) at the same total budget → which reaches
      the target F1 in the fewest experiments. Prints a recommendation.

Synthetic — no real data needed. This is the round-planning de-risking the plan calls for.

Usage:  python src/run_campaign_sim.py [--reps N] [--target F1]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from inference.campaign import average_runs, experiments_to_target
from visualization.base import save_figure

OUT = Path(__file__).resolve().parent.parent / "predictions" / "campaign"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3, help="Seeds averaged per setting")
    ap.add_argument("--target", type=float, default=0.8, help="Target window F1")
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (1) strategy comparison at a fixed schedule
    print("Strategy comparison (q=4, R=8)...")
    for strat, color in [("active", "#1a7a1a"), ("random", "#999999"),
                         ("grid", "#b00000")]:
        n_exp, f1 = average_runs(strat, q=4, n_rounds=8, reps=args.reps)
        ax1.plot(n_exp, f1, "o-", color=color, label=strat)
    ax1.axhline(args.target, color="gray", ls="--", lw=1, label=f"target {args.target}")
    ax1.set_xlabel("# experiments")
    ax1.set_ylabel("window F1 (mapping accuracy)")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Active level-set vs baselines", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # (2) schedule sweep (active), ~same total budget
    print("Schedule sweep (active)...")
    schedules = [(2, 16), (4, 8), (8, 4), (16, 2)]
    rec = []
    for q, n_rounds in schedules:
        n_exp, f1 = average_runs("active", q=q, n_rounds=n_rounds, reps=args.reps)
        nstar = experiments_to_target(n_exp, f1, args.target)
        rec.append((f"q={q}, R={n_rounds}", nstar, float(f1[-1])))
        ax2.plot(n_exp, f1, "o-", label=f"q={q}, R={n_rounds}")
    ax2.axhline(args.target, color="gray", ls="--", lw=1)
    ax2.set_xlabel("# experiments")
    ax2.set_ylabel("window F1")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Batch/round schedule sweep", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(OUT / "campaign_sim.png"))

    print(f"\nSchedule → experiments to reach F1 ≥ {args.target}:")
    for name, nstar, final in rec:
        verdict = f"n={nstar}" if nstar else "not reached at this budget"
        print(f"  {name:12s}: {verdict:28s} (final F1={final:.2f})")
    reached = [(name, nstar) for name, nstar, _ in rec if nstar]
    if reached:
        best = min(reached, key=lambda x: x[1])
        print(f"\nRECOMMEND: '{best[0]}' reaches F1={args.target} in {best[1]} experiments "
              "(finer batches re-steer more often → fewer total experiments).")
    print(f"Saved -> {OUT / 'campaign_sim.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
