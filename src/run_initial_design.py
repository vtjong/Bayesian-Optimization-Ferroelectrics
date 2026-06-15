"""Generate the round-0 Latin-hypercube seed over the (voltage, time) box.

Shows that the chosen design box BRACKETS the crystallization boundary: the thermal model's
predicted Tmax spans from sub-crystallization (cool corner) to over-budget (hot corner), so a
space-filling LHS seed straddles the boundary. This is the boss's "restrict the domain by
physical insight (e.g. 100% confidence it won't crystallize) or tool limits" guidance.

The box is in NORMALIZED voltage (the thermal model is not yet calibrated to real volts);
map v_norm -> real V once the thermal model is calibrated.

Usage:  python src/run_initial_design.py [--n 12]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from optimization.sampling import latin_hypercube
from thermal import simulate_profile
from visualization.base import save_figure

OUT = Path(__file__).resolve().parent.parent / "predictions" / "initial_design"

# Physically-restricted design box (normalized voltage, pulse time in ms).
# Chosen so Tmax spans ~sub-crystallization -> over-budget (brackets the boundary).
V_LO, V_HI = 0.55, 1.00
T_LO, T_HI = 0.5, 5.0
THRESH_C = 500.0  # nominal crystallization threshold


def tmax_grid(nv=70, nt=70):
    vs = np.linspace(V_LO, V_HI, nv)
    ts = np.linspace(T_LO, T_HI, nt)
    tmax = np.array([[simulate_profile(v, t)[1].max() for v in vs] for t in ts])
    return vs, ts, tmax


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12, help="Number of LHS seed points")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_pts = latin_hypercube([(V_LO, V_HI), (T_LO, T_HI)], args.n, seed=args.seed)
    vs, ts, tmax = tmax_grid()

    fig, ax = plt.subplots(figsize=(8.5, 6))
    cf = ax.contourf(vs, ts, tmax, levels=20, cmap="inferno")
    fig.colorbar(cf, ax=ax).set_label("predicted Tmax (°C)")
    cs = ax.contour(vs, ts, tmax, levels=[THRESH_C], colors="cyan", linewidths=2.5)
    ax.clabel(cs, fmt=f"~{int(THRESH_C)}°C boundary", fontsize=9)
    ax.scatter(seed_pts[:, 0], seed_pts[:, 1], c="white", edgecolors="black",
               s=70, zorder=5, label=f"LHS seed (n={args.n})")
    ax.set_xlabel("normalized flash voltage")
    ax.set_ylabel("pulse time (ms)")
    ax.set_title("Round-0 LHS over the (V, t) box — the seed brackets the boundary",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(OUT / "initial_design.png"))

    below = np.mean([simulate_profile(v, t)[1].max() < THRESH_C for v, t in seed_pts])
    print(f"LHS seed: {args.n} points over V∈[{V_LO},{V_HI}], t∈[{T_LO},{T_HI}]ms")
    print(f"  {below*100:.0f}% below and {(1-below)*100:.0f}% above the ~{int(THRESH_C)}°C "
          "boundary → the seed straddles it (good).")
    print(f"Saved -> {OUT / 'initial_design.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
