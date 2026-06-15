"""Decorrelation arm: a preheat axis breaks single-pulse collinearity → unlocks mechanism ID.

Under a single pulse, peak temperature (Tmax) and quench/cooling rate are *perfectly*
collinear, so "Tmax controls crystallization" vs "cooling rate controls it" are
INDISTINGUISHABLE. Adding a substrate-preheat axis lets you reach a given Tmax with less
flash power → Tmax and cooling rate vary independently → the true mechanism becomes
identifiable. This is the Phase-2 decorrelation DoE, demonstrated on synthetic data.

Usage:  python src/run_decorrelation.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.append(str(Path(__file__).resolve().parent))
from thermal import extract_descriptors, simulate_profile
from visualization.base import save_figure

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "predictions" / "decorrelation"


def design(n, decorrelate, seed):
    """Sample a design; return (Tmax, cooling_rate) descriptor arrays."""
    rng = np.random.default_rng(seed)
    V = rng.uniform(0.45, 1.0, n)
    tp = rng.uniform(0.5, 5.0, n)
    pre = rng.uniform(25.0, 500.0, n) if decorrelate else np.full(n, 25.0)
    Tmax, cool = np.empty(n), np.empty(n)
    for i in range(n):
        t, T = simulate_profile(V[i], tp[i], t_preheat=pre[i])
        d = extract_descriptors(t, T)
        Tmax[i], cool[i] = d["Tmax"], d["cooling_rate"]
    return Tmax, cool


def true_cryst(Tmax, rng, t_c=560.0, w=70.0, noise=0.04):
    """Ground truth: crystallinity is controlled by Tmax (a sigmoid onset)."""
    y = 1.0 / (1.0 + np.exp(-(Tmax - t_c) / w))
    return np.clip(y + rng.normal(0, noise, len(y)), 0, 1)


def loo_r2(x, y):
    """Leave-one-out R^2 of a 1-D GP regression y ~ f(x): how well x explains y."""
    xs = ((x - x.mean()) / (x.std() + 1e-9)).reshape(-1, 1)
    # fixed-noise (alpha) GP — no WhiteKernel hyperparameter to hit a bound (no warnings)
    kernel = ConstantKernel(1.0) * RBF(1.0)
    preds = np.empty(len(y))
    for tr, te in LeaveOneOut().split(xs):
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, normalize_y=True,
                                      n_restarts_optimizer=1).fit(xs[tr], y[tr])
        preds[te] = gp.predict(xs[te])
    ss_res, ss_tot = np.sum((y - preds) ** 2), np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def main() -> int:
    rng = np.random.default_rng(0)
    n = 45
    Tmax_s, cool_s = design(n, decorrelate=False, seed=1)   # single pulse
    Tmax_d, cool_d = design(n, decorrelate=True, seed=2)    # + preheat
    y_s, y_d = true_cryst(Tmax_s, rng), true_cryst(Tmax_d, rng)

    r2 = {
        "single": {"Tmax": loo_r2(Tmax_s, y_s), "cool": loo_r2(cool_s, y_s)},
        "decorr": {"Tmax": loo_r2(Tmax_d, y_d), "cool": loo_r2(cool_d, y_d)},
    }
    print("Which descriptor explains crystallinity? (LOO R^2; truth = Tmax)")
    print(f"  SINGLE PULSE : f(Tmax)={r2['single']['Tmax']:.2f}   "
          f"f(cooling_rate)={r2['single']['cool']:.2f}   -> ~equal, CANNOT distinguish")
    print(f"  + PREHEAT    : f(Tmax)={r2['decorr']['Tmax']:.2f}   "
          f"f(cooling_rate)={r2['decorr']['cool']:.2f}   -> only the TRUE one wins")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    axA.scatter(Tmax_s, cool_s, c="#1f6fb2", s=45, label="single pulse (preheat=25°C)")
    axA.scatter(Tmax_d, cool_d, c="#ff8c1a", s=45, label="+ preheat (decorrelated)")
    axA.set_xlabel("Tmax (°C)")
    axA.set_ylabel("cooling / quench rate (°C/ms)")
    axA.set_title("Single pulse: Tmax & cooling rate locked on a LINE\n"
                  "+ preheat: they spread into a CLOUD", fontweight="bold", fontsize=11)
    axA.legend()
    axA.grid(alpha=0.25)

    x = np.arange(2)
    axB.bar(x - 0.2, [r2["single"]["Tmax"], r2["single"]["cool"]], 0.4,
            label="single pulse", color="#1f6fb2")
    axB.bar(x + 0.2, [r2["decorr"]["Tmax"], r2["decorr"]["cool"]], 0.4,
            label="+ preheat", color="#ff8c1a")
    axB.set_xticks(x)
    axB.set_xticklabels(["f(Tmax)\n[TRUE driver]", "f(cooling rate)\n[decoy]"])
    axB.set_ylabel("LOO R²  (how well it explains crystallinity)")
    axB.axhline(0, color="k", lw=0.5)
    axB.set_title("Single pulse: both explain it equally → ambiguous\n"
                  "+ preheat: only the TRUE descriptor wins → IDENTIFIABLE",
                  fontweight="bold", fontsize=11)
    axB.legend()
    axB.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    save_figure(fig, str(OUT / "decorrelation.png"))
    print(f"\nSaved -> {OUT / 'decorrelation.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
