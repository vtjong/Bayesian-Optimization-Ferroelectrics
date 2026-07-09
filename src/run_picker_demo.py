"""Acquisition comparison for noisy crystallization-boundary mapping (Layer 2).

Grounds the acquisition choice empirically: predictive entropy vs BALD vs the latent-variance
level-set entropy (LSE), across observation-noise levels, on the calibrated boundary whose noise
is worst AT the boundary (the permittivity "leaky-near-the-transition" regime). Reports final
boundary-map error (mean +/- SEM over seeds) and shows the default LSE picker clustering shots on
the boundary.

Finding: in the realistic (permittivity) noise regime the three are COMPARABLE -- overlapping
+/-1 SEM bands, no robust dominance. We adopt LSE as a principled default (boundary-focused,
noise-aware; best-or-tied at the noise extremes), with BALD/entropy as baselines.

Usage:  python src/run_picker_demo.py [--seeds 16]
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

OUT = Path(__file__).resolve().parent.parent / "predictions" / "picker_demo"
COL = {"entropy": "#7a7a7a", "bald": "#d1772b", "lse": "#1f6fb2"}
LBL = {"entropy": "predictive entropy", "bald": "BALD", "lse": "level-set entropy (latent)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    s1_grid = [0.2, 0.4, 0.8, 1.5, 2.5]
    peaks = [pk._S0 + 0.25 * s for s in s1_grid]
    curves = {a: [] for a in ("entropy", "bald", "lse")}
    sems = {a: [] for a in ("entropy", "bald", "lse")}
    print(f"peak sigma_n | entropy | bald | lse   (final boundary-map error, mean+/-SEM, "
          f"{args.seeds} seeds)")
    for s1 in s1_grid:
        pk._S1 = s1
        for a in ("entropy", "bald", "lse"):
            e = [np.mean(pk.run_active(a, n_seed=10, n_iter=22, seed=s)["err"][-4:])
                 for s in range(args.seeds)]
            curves[a].append(np.mean(e)); sems[a].append(np.std(e) / np.sqrt(args.seeds))
        print(f"   {pk._S0 + 0.25*s1:.2f}      | "
              + " | ".join(f"{curves[a][-1]:.3f}+/-{sems[a][-1]:.3f}"
                           for a in ("entropy", "bald", "lse")))

    # spatial shot placement for the (default) LSE picker at a representative moderate noise
    pk._S1 = 0.4
    h = pk.run_active("lse", n_seed=10, n_iter=25, seed=0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for a in ("entropy", "bald", "lse"):
        m, e = np.array(curves[a]), np.array(sems[a])
        a1.plot(peaks, m, "o-", color=COL[a], label=LBL[a])
        a1.fill_between(peaks, m - e, m + e, color=COL[a], alpha=0.15)
    a1.axvspan(0.08, 0.16, color="#2e8b57", alpha=0.12, label="permittivity regime")
    a1.set_xlabel("peak observation noise  sigma_n  (worst at the boundary)")
    a1.set_ylabel("boundary-map error (misclassification area)")
    a1.set_title("Entropy-family acquisitions are comparable\n(overlapping +/-1 SEM bands)",
                 fontweight="bold", fontsize=11)
    a1.legend(fontsize=8); a1.grid(alpha=0.3)

    vs = np.linspace(V_LO, V_HI, 120); ts = np.linspace(T_LO, T_HI, 120)
    VV, TT = np.meshgrid(vs, ts)
    ftrue = pk.true_f(VV.ravel(), TT.ravel()).reshape(VV.shape)
    band = pk.noise_sigma(ftrue)
    a2.contourf(VV, TT, band, levels=15, cmap="Purples")
    a2.contour(VV, TT, ftrue, levels=[pk.THETA], colors="k", linewidths=2)
    ns = 10
    a2.scatter(h["V"][:ns], h["t"][:ns], c="white", edgecolors="k", s=35, label="LHS seed")
    a2.scatter(h["V"][ns:], h["t"][ns:], c="#1f6fb2", edgecolors="k", s=35,
               label="LSE-chosen shots")
    a2.set_xlabel("voltage V"); a2.set_ylabel("pulse time t (ms)")
    a2.set_title("LSE picker clusters shots on the boundary\n(shaded = observation noise band)",
                 fontweight="bold", fontsize=11)
    a2.legend(fontsize=8, loc="upper right")
    save_figure(fig, str(OUT / "picker_acquisition_comparison.png"))
    print(f"\nSaved -> {OUT / 'picker_acquisition_comparison.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
