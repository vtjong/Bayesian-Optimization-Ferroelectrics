"""Simulate flash-anneal temperature profiles and the thermal descriptors from (V, t).

Produces three figures:
  1. profiles      — T(t) for a few (V, t) recipes (heating + quench + Tmax)
  2. descriptor_maps — Tmax and Arrhenius budget K over the (V, t) box
  3. collinearity  — correlation matrix of the descriptors (they collapse onto (V,t))

This is the boss's thermal-feature layer: (V,t) -> T(t) -> {Tmax, K, rates, dwell}.

Usage:  python src/run_thermal_sim.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from thermal import extract_descriptors, simulate_profile
from visualization.base import save_figure

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "predictions" / "thermal_sim"
T_MIN, T_MAX = 0.5, 5.0  # pulse duration range (ms)


def fig_profiles():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for v, tp in [(0.6, 1.0), (0.8, 2.0), (1.0, 3.0)]:
        t, T = simulate_profile(v, tp)
        ax.plot(t, T, lw=2.3, label=f"V={v:.1f}, t={tp:.0f}ms  (Tmax={T.max():.0f}°C)")
    ax.axhline(500, color="gray", ls="--", lw=1)
    ax.text(0.1, 515, "crystallization threshold ~500°C", color="gray", fontsize=9)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Simulated flash-anneal profiles: heat → peak → quench",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def descriptor_grid(nv=26, nt=26, ea_eV=1.5):
    vs = np.linspace(0.25, 1.0, nv)
    ts = np.linspace(T_MIN, T_MAX, nt)
    keys = None
    rows = []
    for tp in ts:
        for v in vs:
            t, T = simulate_profile(v, tp)
            d = extract_descriptors(t, T, ea_eV=ea_eV)
            if keys is None:
                keys = list(d.keys())
            rows.append([v, tp] + [d[k] for k in keys])
    arr = np.array(rows)
    return vs, ts, keys, arr  # arr cols: V, t, <descriptors...>


def fig_maps(vs, ts, keys, arr):
    nv, nt = len(vs), len(ts)
    Tmax = arr[:, 2 + keys.index("Tmax")].reshape(nt, nv)
    K = arr[:, 2 + keys.index("arrhenius_K")].reshape(nt, nv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, name, cmap in [(axes[0], Tmax, "Tmax (°C)", "inferno"),
                                 (axes[1], np.log10(K + 1e-30), "log₁₀  Arrhenius K", "viridis")]:
        cf = ax.contourf(vs, ts, data, levels=20, cmap=cmap)
        fig.colorbar(cf, ax=ax).set_label(name)
        ax.set_xlabel("normalized voltage V")
        ax.set_ylabel("pulse time t (ms)")
        ax.set_title(name, fontweight="bold")
    fig.suptitle("Thermal descriptors over the (V, t) box", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def _pca_evr(D):
    """Cumulative explained-variance ratio of the standardized descriptors."""
    Ds = (D - D.mean(0)) / (D.std(0) + 1e-12)
    s = np.linalg.svd(Ds, full_matrices=False, compute_uv=False)
    evr = s ** 2 / np.sum(s ** 2)
    return evr, np.cumsum(evr)


def fig_collinearity(keys, arr):
    D = arr[:, 2:]
    corr = np.corrcoef(D, rowvar=False)
    evr, cum = _pca_evr(D)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(keys, fontsize=8)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(corr[i, j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax).set_label("Pearson correlation")
    ax.set_title("Descriptor correlation matrix", fontsize=11, fontweight="bold")

    n = np.arange(1, len(evr) + 1)
    ax2.bar(n, evr, color="#4da6ff", label="per-component")
    ax2.plot(n, cum, "o-", color="#b00000", label="cumulative")
    ax2.axhline(0.99, color="gray", ls="--", lw=1)
    ax2.set_xlabel("principal component")
    ax2.set_ylabel("explained variance ratio")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.set_title(f"First 2 PCs ≈ {cum[1] * 100:.0f}% of variance\n"
                  "intrinsically 2 DOF (all are functions of V,t); tail = nonlinear curvature",
                  fontsize=10, fontweight="bold")
    fig.suptitle("Single-pulse identifiability: descriptors are functions of just (V, t)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    save_figure(fig_profiles(), str(OUT / "profiles.png"))
    vs, ts, keys, arr = descriptor_grid()
    save_figure(fig_maps(vs, ts, keys, arr), str(OUT / "descriptor_maps.png"))
    save_figure(fig_collinearity(keys, arr), str(OUT / "collinearity.png"))

    # report the effective dimensionality of the descriptor cloud
    evr, cum = _pca_evr(arr[:, 2:])
    print(f"Descriptors ({len(keys)}): {keys}")
    print(f"PCA: first 2 components explain {cum[1] * 100:.0f}% of descriptor variance "
          f"(EVR: {np.round(evr, 3)})")
    print(f"→ all {len(keys)} descriptors are deterministic functions of just (V,t): "
          "intrinsically 2 DOF (the linear-PCA tail is nonlinear curvature, not new info).")
    print(f"Figures saved to {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
