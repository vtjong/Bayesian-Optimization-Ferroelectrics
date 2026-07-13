"""Misspecification stress test: how robust is the chart comparison to a wrong thermal model?

Generates data from a perturbed thermal model (strength delta: shifted Tmax constants +
temperature-dependent cooling) while the analysis builds charts from the canonical model.
Reports P(identify family) and the (effective) Ea recovery error vs delta. delta=0 is the
matched baseline. Saves a figure + results.json.

Usage:  python src/run_misspec_study.py [--reps N] [--n N]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))

from discovery.misspec import run_misspecification
from visualization.base import save_figure

OUT = Path(__file__).resolve().parent.parent / "predictions" / "misspec_study"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--readout", default="xrd")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(
        f"Misspecification stress test (Scenario A, {args.readout}, n={args.n}, "
        f"reps={args.reps})\n  delta=0 is matched; larger delta = more wrong thermal model.\n"
    )
    rows = run_misspecification(readout=args.readout, n=args.n, reps=args.reps)

    print(f"{'delta':>6} {'P(family)':>11} {'med|Ea_eff err|':>16}")
    for r in rows:
        print(f"{r['delta']:>6.2f} {r['p_family']:>11.2f} {r['median_ea_err']:>14.2f}eV")

    d = [r["delta"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(d, [r["p_family"] for r in rows], "o-", color="#1f6fb2", label="P(identify family)")
    ax1.axhline(0.8, color="green", ls="--", lw=1)
    ax1.set_xlabel("thermal-model misspecification strength  δ", fontsize=12)
    ax1.set_ylabel("P(identify the controlling family)", color="#1f6fb2", fontsize=12)
    ax1.set_ylim(0, 1.02)
    ax1.tick_params(axis="y", labelcolor="#1f6fb2")
    ax2 = ax1.twinx()
    ax2.plot(
        d, [r["median_ea_err"] for r in rows], "s--", color="#d1772b", label="median |Ea_eff error|"
    )
    ax2.set_ylabel("median |Ea_eff error| (eV)", color="#d1772b", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#d1772b")
    ax1.set_title(
        "Misspecification stress test: recovery vs wrong thermal model\n"
        "(δ=0 matched; analysis always uses the canonical model)",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, str(OUT / "misspec.png"))

    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
