"""Generate the seed flash-condition plan for the boundary-mapping campaign.

Produces a space-filling Latin-hypercube design of N conditions over the measured (flash voltage,
flash time) box, annotates each with the predicted peak temperature (from the calibrated table)
and its predicted crystalline/amorphous state, and writes the plan to data/ + a figure. These are
the conditions the experimental team flashes first; the active-learning rounds follow.

Usage:  python src/run_flash_plan.py [--n 12] [--seed 7]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

sys.path.append(str(Path(__file__).resolve().parent))

from discovery.synthetic import (T_HI, T_LO, T_ONSET_C, V_HI, V_LO, tmax)
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "flash_plan"


def make_plan(n: int, seed: int):
    """Optimized Latin-hypercube seed over (V, t), snapped to achievable settings."""
    u = qmc.LatinHypercube(d=2, optimization="random-cd", seed=seed).random(n)
    V = np.round(V_LO + u[:, 0] * (V_HI - V_LO)).astype(int)          # nearest volt
    t = np.round((T_LO + u[:, 1] * (T_HI - T_LO)) * 10) / 10.0        # nearest 0.1 ms
    Tm = tmax(V.astype(float), t)
    order = np.lexsort((V, t))                                        # tidy: by time then voltage
    return V[order], t[order], Tm[order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    V, t, Tm = make_plan(args.n, args.seed)
    cryst = Tm > T_ONSET_C
    print(f"Seed flash plan: {args.n} conditions over V=[{V_LO:.0f},{V_HI:.0f}] V, "
          f"t=[{T_LO},{T_HI}] ms; onset {T_ONSET_C:.0f} C")
    print(f"predicted crystalline: {cryst.sum()} / {args.n}  (both classes bracket the boundary)\n")
    print("  #   V (V)   t (ms)   pred Tmax (C)   pred state")
    for i in range(args.n):
        print(f" {i+1:2d}   {V[i]:4d}   {t[i]:5.1f}    {Tm[i]:8.0f}       "
              f"{'crystalline' if cryst[i] else 'amorphous'}")

    # save the plan
    import csv
    with open(ROOT / "data" / "flash_plan_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "voltage_V", "time_ms", "pred_Tmax_C", "pred_state"])
        for i in range(args.n):
            w.writerow([i + 1, V[i], t[i], round(float(Tm[i]), 1),
                        "crystalline" if cryst[i] else "amorphous"])

    # figure: points on the calibrated Tmax field + onset boundary
    vv = np.linspace(V_LO, V_HI, 160); tg = np.linspace(T_LO, T_HI, 160)
    VV, TT = np.meshgrid(vv, tg)
    ZZ = tmax(VV.ravel(), TT.ravel()).reshape(VV.shape)
    fig, ax = plt.subplots(figsize=(8.2, 6))
    cf = ax.contourf(VV, TT, ZZ, levels=18, cmap="inferno")
    plt.colorbar(cf, ax=ax, label="predicted Tmax (C)")
    ax.contour(VV, TT, ZZ, [T_ONSET_C], colors="cyan", linewidths=2.5)
    for i in range(args.n):
        mk = "o" if cryst[i] else "s"
        ax.scatter(V[i], t[i], c="white" if cryst[i] else "none", marker=mk, s=120,
                   edgecolors="cyan" if cryst[i] else "white", linewidths=1.8, zorder=5)
        ax.annotate(str(i + 1), (V[i], t[i]), fontsize=8, fontweight="bold",
                    ha="center", va="center", color="k" if cryst[i] else "white")
    ax.set_xlabel("flash voltage V (V)"); ax.set_ylabel("flash time t (ms)")
    ax.set_title(f"{args.n} seed flash conditions on the calibrated Tmax field\n"
                 f"(cyan = {T_ONSET_C:.0f} C onset; filled = predicted crystalline)",
                 fontweight="bold", fontsize=11)
    save_figure(fig, str(OUT / "flash_plan.png"))
    print(f"\nSaved -> data/flash_plan_seed.csv, {OUT / 'flash_plan.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
