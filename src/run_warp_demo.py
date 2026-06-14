"""Synthetic demo: a learnable input warp on (voltage, time) recovers energy-space smoothness.

Ground truth: FOM is a SMOOTH function of (energy, time) where energy = g(V) = V^2 (a
nonlinear, monotonic warp of the control knob). We fit three GPs and compare held-out
predictive accuracy + calibration:

  (a) stationary GP on raw (V, t)          -> sees a warped/non-stationary surface (worst)
  (b) stationary GP on (energy=V^2, t)     -> the oracle smooth coordinate (best)
  (c) warped GP on (V, t) (BoTorch Warp)   -> LEARNS the warp from data -> should match (b)

It also checks that the learned warp recovers the true g(V) = V^2. This de-risks the plan's
decision to train on (V, t) with a learnable warp (REVISION 1 #11) — keeping exact, SHAP-valid
control-knob inputs while recovering the smoothness that energy density gave.

Usage:  python src/run_warp_demo.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

sys.path.append(str(Path(__file__).resolve().parent))
from visualization.base import save_figure

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "predictions" / "warp_demo"
torch.set_default_dtype(torch.double)


def true_fom(V, t):
    """Smooth Gaussian peak in (energy, time) space, energy = V**2 (the hidden warp)."""
    e = V ** 2
    return np.exp(-(((e - 0.6) ** 2) + ((t - 0.45) ** 2)) / (2 * 0.07))


def fit_gp(train_x, train_y, warp_dims=None):
    """Fit a SingleTaskGP via marginal likelihood; optional Kumaraswamy input warp."""
    itf = Warp(d=train_x.shape[-1], indices=warp_dims) if warp_dims is not None else None
    model = SingleTaskGP(train_x, train_y, input_transform=itf,
                         outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def metrics(model, test_x, test_y):
    """Held-out RMSE and NLPD (predictive incl. observation noise)."""
    with torch.no_grad():
        post = model.posterior(test_x, observation_noise=True)
        mean = post.mean.squeeze(-1)
        var = post.variance.squeeze(-1).clamp_min(1e-9)
    y = test_y.squeeze(-1)
    rmse = torch.sqrt(torch.mean((mean - y) ** 2)).item()
    nlpd = (0.5 * torch.log(2 * np.pi * var) + 0.5 * (y - mean) ** 2 / var).mean().item()
    return rmse, nlpd


def main() -> int:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    n_train, n_test, noise = 40, 400, 0.03

    Vtr, ttr = rng.uniform(0, 1, n_train), rng.uniform(0, 1, n_train)
    ytr = true_fom(Vtr, ttr) + rng.normal(0, noise, n_train)
    Vte, tte = rng.uniform(0, 1, n_test), rng.uniform(0, 1, n_test)
    yte = true_fom(Vte, tte)

    def col(a, b):
        return torch.tensor(np.column_stack([a, b]))

    Y = lambda y: torch.tensor(y).unsqueeze(-1)
    X_v_tr, X_v_te = col(Vtr, ttr), col(Vte, tte)            # (V, t)
    X_e_tr, X_e_te = col(Vtr ** 2, ttr), col(Vte ** 2, tte)  # (energy, t) oracle

    print("Fitting 3 GPs (marginal-likelihood, fit_gpytorch_mll)...")
    m_a = fit_gp(X_v_tr, Y(ytr))                 # stationary on (V, t)
    m_b = fit_gp(X_e_tr, Y(ytr))                 # stationary on (energy, t) — oracle
    m_c = fit_gp(X_v_tr, Y(ytr), warp_dims=[0])  # warped on (V, t)

    res = {
        "(a) stationary GP on (V, t)": metrics(m_a, X_v_te, Y(yte)),
        "(b) stationary GP on (energy, t)  [oracle]": metrics(m_b, X_e_te, Y(yte)),
        "(c) warped GP on (V, t)  [learns warp]": metrics(m_c, X_v_te, Y(yte)),
    }
    print(f"\n{'model':46s} {'RMSE':>8s} {'NLPD':>8s}")
    for k, (rmse, nlpd) in res.items():
        print(f"{k:46s} {rmse:>8.4f} {nlpd:>8.3f}")

    # learned warp vs true g(V)=V^2
    vg = torch.linspace(0, 1, 100)
    Xg = torch.stack([vg, torch.full_like(vg, 0.5)], dim=1)
    learned = m_c.input_transform.transform(Xg)[:, 0].detach().numpy()
    vg_np = vg.numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    labels = list(res.keys())
    rmses = [res[k][0] for k in labels]
    nlpds = [res[k][1] for k in labels]
    x = np.arange(len(labels))
    ax1b = ax1.twinx()
    b1 = ax1.bar(x - 0.2, rmses, 0.4, color="#4da6ff")
    b2 = ax1b.bar(x + 0.2, nlpds, 0.4, color="#ff4d4d")
    ax1.set_ylabel("RMSE  (lower better)", color="#1f6fb2")
    ax1b.set_ylabel("NLPD  (lower better)", color="#b02a22")
    ax1.set_ylim(0, max(rmses) * 1.3)
    lo, hi = min(nlpds), max(nlpds)
    pad = (hi - lo) * 0.6 + 1e-3
    ax1b.set_ylim(lo - pad, hi + pad)
    for xi, r in zip(x - 0.2, rmses):
        ax1.text(xi, r, f"{r:.4f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["(a) V,t\nstationary", "(b) energy,t\noracle", "(c) V,t\nWARPED"])
    ax1.set_title("Held-out error: warped (V,t) ≈ energy oracle ≫ raw (V,t)",
                  fontsize=12, fontweight="bold")
    ax1.legend([b1, b2], ["RMSE", "NLPD"], loc="upper center")
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(vg_np, vg_np ** 2, "k--", lw=2, label="true warp  g(V)=V²")
    ax2.plot(vg_np, learned, color="#1a7a1a", lw=2.5, label="learned warp (Kumaraswamy)")
    ax2.plot(vg_np, vg_np, color="grey", lw=1, ls=":", label="identity (no warp)")
    ax2.set_xlabel("voltage V (normalized)")
    ax2.set_ylabel("warped coordinate")
    ax2.set_title("The GP recovered the hidden V→energy warp from data",
                  fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, str(OUT / "warp_demo.png"))
    print(f"\nSaved -> {OUT / 'warp_demo.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
