"""Functional-trace regression: recover an UNLISTED temperature-window controller.

Crystallization is planted to be controlled by the time the trace spends in a temperature
WINDOW [lo, hi] -- a controller NOT in the discrete chart menu (Tmax / TBac(Ea) / dwell).
A smoothness-regularized regression of logit(crystalline fraction) on each shot's
time-temperature occupancy histogram recovers the weighting w(T), which peaks in the window;
meanwhile the discrete chart comparison goes diffuse (no chart wins). This demonstrates that a
functional regression on the full single-pulse trace finds contributors the menu cannot.

Usage:  python src/run_functional_demo.py [--n 300] [--lo 500] [--hi 560]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from discovery.charts import build_charts
from discovery.compare import compare
from discovery.functional import (TEMP_CENTERS, fit_weighting, make_window_dataset,
                                  occupancy)
from discovery.synthetic import SCENARIOS, make_dataset
from visualization.base import save_figure

OUT = Path(__file__).resolve().parent.parent / "predictions" / "functional_demo"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lo", type=float, default=500.0)
    ap.add_argument("--hi", type=float, default=560.0)
    ap.add_argument("--readout", default="xrd")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    V, t, y = make_window_dataset(args.n, args.readout, rng, lo=args.lo, hi=args.hi)
    w, sd, r2 = fit_weighting(occupancy(V, t), y)
    occ = sd > 0.05 * sd.max()                       # bins actually visited
    peakT = TEMP_CENTERS[occ][np.argmax(w[occ])]

    # discrete chart comparison on the SAME data, and on scenario A as a reference
    m_win = compare(build_charts(V, t), y, args.readout)["margin_over_vt"]
    Va, ta, ya = make_dataset(args.n, SCENARIOS["A"], args.readout,
                              np.random.default_rng(1))
    m_a = compare(build_charts(Va, ta), ya, args.readout)["margin_over_vt"]

    print(f"Functional regression (fit R2={r2:.2f}): learned weighting peaks at "
          f"{peakT:.0f} C  (planted window {args.lo:.0f}-{args.hi:.0f} C)")
    print(f"Discrete chart comparison margin over (V,t): window data = {m_win:.0f} "
          f"(diffuse) vs single-coordinate reference (scenario A) = {m_a:.0f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.axvspan(args.lo, args.hi, color="#d9544d", alpha=0.18, label="planted window")
    a1.plot(TEMP_CENTERS[occ], w[occ], "o-", color="#1f6fb2", lw=2,
            label="learned weighting w(T)")
    a1.axhline(0, color="gray", lw=0.6)
    a1.set_xlabel("temperature T (C)"); a1.set_ylabel("learned weight (standardized)")
    a1.set_title("Functional regression recovers the UNLISTED\ntemperature-window controller",
                 fontweight="bold", fontsize=11)
    a1.legend(fontsize=9)

    a2.bar(["single-coordinate\n(scenario A)", "window\n(this demo)"], [m_a, m_win],
           color=["#2e8b57", "#d1772b"])
    a2.set_ylabel("discrete-chart margin over (V,t)")
    a2.set_title("Discrete chart menu: wins for a listed coordinate,\n"
                 "goes diffuse for the unlisted window", fontweight="bold", fontsize=11)
    a2.annotate("menu fails ->\nneed the trace", (1, m_win), (0.6, m_a * 0.5),
                fontsize=9, arrowprops=dict(arrowstyle="-|>"))
    save_figure(fig, str(OUT / "functional_demo.png"))
    print(f"\nSaved -> {OUT / 'functional_demo.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
