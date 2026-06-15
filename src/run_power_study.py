"""Run the design-stage Bayesian power study and report the go/no-go (REVISION 1 #1).

Prints P(correct model selection) vs sample size n and noise, saves a plot, and prints
the minimum n that reaches the target power per (model, noise). PROTOTYPE: 2 shape-distinct
kinetic models, 1-D design, grid-quadrature evidence; upgrade to the real model set +
nested sampling for the paper.

Usage:  python src/run_power_study.py [--reps N] [--target P]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))

from inference.power_study import min_n_for_power, run_power_study
from visualization.base import save_figure

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "predictions" / "power_study"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=200, help="Replicates per (n, sigma) cell")
    ap.add_argument("--target", type=float, default=0.8, help="Target power for go/no-go")
    ap.add_argument("--threshold", type=float, default=1.0, help="log10 BF decision threshold")
    ap.add_argument("--cal-err", type=float, default=0.03, help="Injected thermal-calibration error")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    print(f"Running power study (reps={args.reps}, log10BF>{args.threshold}, "
          f"cal_err={args.cal_err})...")
    rows = run_power_study(reps=args.reps, threshold_log10bf=args.threshold, cal_err=args.cal_err)

    print(f"\n{'true_model':10s} {'n':>4s} {'sigma':>7s} {'P(correct)':>11s} {'med log10BF':>12s}")
    for r in rows:
        print(f"{r['true_model']:10s} {r['n']:>4d} {r['sigma']:>7.3f} "
              f"{r['p_correct']:>11.2f} {r['median_log10bf']:>12.2f}")

    gono = min_n_for_power(rows, target=args.target)
    print(f"\nGO/NO-GO — minimum n to reach P(correct) >= {args.target}:")
    for k, v in gono.items():
        verdict = f"n={v}" if v is not None else "NOT REACHED at n<=40"
        print(f"  {k:24s} -> {verdict}")

    # plot: p_correct vs n, one line per (model, sigma)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    keys = sorted({(r["true_model"], r["sigma"]) for r in rows})
    for tm, sig in keys:
        sub = sorted((r for r in rows if r["true_model"] == tm and r["sigma"] == sig),
                     key=lambda r: r["n"])
        ax.plot([r["n"] for r in sub], [r["p_correct"] for r in sub],
                "o-", label=f"{tm}, σ={sig}")
    ax.axhline(args.target, color="green", ls="--", lw=1, label=f"target {args.target}")
    ax.set_xlabel("sample size n", fontsize=12)
    ax.set_ylabel("P(correctly select the true model)", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_title("Design-stage power study: mechanism discriminability vs n",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, str(out / "power_curve.png"))
    print(f"\nSaved plot -> {out / 'power_curve.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
