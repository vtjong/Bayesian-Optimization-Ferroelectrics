"""Compare the lumped vs the 1-D through-thickness thermal model (the field-standard upgrade).

The 1-D model solves the heat equation in the film+substrate with an absorbed-flux boundary,
so it captures a realistic *diffusive* quench (sharp surface drop when the flash ends, then a
slow tail) — unlike the lumped model's single exponential. Peaks land in the HZO crystallization
window. See docs/thermal_budget_litreview.md §6 for the methodology this follows.

Usage:  python src/run_thermal_1d.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from thermal import simulate_profile, simulate_profile_1d
from visualization.base import save_figure

OUT = Path(__file__).resolve().parent.parent / "predictions" / "thermal_sim"


def main() -> int:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#b00000", "#e07b00", "#1f6fb2"]
    for (v, tp), c in zip([(0.6, 1.0), (0.8, 2.0), (1.0, 3.0)], colors):
        t1, T1 = simulate_profile_1d(v, tp)
        tl, Tl = simulate_profile(v, tp)
        ax.plot(t1, T1, color=c, lw=2.4, label=f"1-D  V={v}, t={tp:.0f}ms (Tmax={T1.max():.0f}°C)")
        ax.plot(tl, Tl, color=c, lw=1.6, ls="--", alpha=0.7)
    ax.axhline(500, color="gray", ls=":", lw=1)
    ax.text(0.1, 515, "crystallization threshold ~500°C", color="gray", fontsize=9)
    ax.plot([], [], color="k", lw=2.4, label="── 1-D through-thickness (upgrade)")
    ax.plot([], [], color="k", lw=1.6, ls="--", label="-- lumped (prototype)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("surface (film) temperature (°C)")
    ax.set_xlim(0, 8)
    ax.set_title("1-D model: realistic diffusive quench (sharp drop + slow tail)\n"
                 "vs lumped exponential", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(OUT / "lumped_vs_1d.png"))
    print(f"Saved -> {OUT / 'lumped_vs_1d.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
