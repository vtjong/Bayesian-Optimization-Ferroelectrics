"""LOO model selection: compare Matern nu and fixed-vs-learned noise.

Runs the leave-one-out harness (:mod:`evaluation.loo`) on the real HZO
experimental data and prints a small ranked table of LOO-RMSE / LOO-NLPD /
95%-coverage for each candidate configuration:

  - Matern nu in {0.5, 1.5, 2.5}
  - noise: fixed (FixedNoiseGaussianLikelihood) vs learned (GaussianLikelihood
    + weakly-informative prior)

All candidates are fit with the principled marginal-likelihood fitter
(``fit_gp_mll`` / ``build_and_fit_gp``) and weakly-informative ARD lengthscale
priors. A summary bar chart is saved to predictions/model_selection/.

Usage:  python src/run_model_selection.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

import gpytorch  # noqa: E402

from evaluation.loo import loo_cross_validate  # noqa: E402
from fit import build_and_fit_gp  # noqa: E402
from preprocessing.loaders import load_experimental_data  # noqa: E402
from preprocessing.transforms import (  # noqa: E402
    TorchMinMaxScaler,
    prepare_gp_training_tensors,
)
from visualization.base import save_figure  # noqa: E402

warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning)
warnings.filterwarnings("ignore", message=".*input.*not contained.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "predictions" / "model_selection"

# Noise center: midpoint of the measured experimental noise (~0.045-0.1).
NOISE_CENTER = 0.07
MIN_LENGTHSCALE = 0.03


def make_fit_fn(matern_nu: float, learn_noise: bool):
    """Build a (train_x, train_y) -> (likelihood, model) fitter for one config.

    :param matern_nu: Matern smoothness (0.5, 1.5, or 2.5)
    :param learn_noise: Use learned GaussianLikelihood when True, else fixed
    :return: A fitter callable for the LOO harness
    """

    def _fit(x_tr, y_tr):
        return build_and_fit_gp(
            train_x=x_tr,
            train_y=y_tr,
            kernel_type="matern",
            noise=NOISE_CENTER,
            matern_nu=matern_nu,
            min_lengthscale=MIN_LENGTHSCALE,
            learn_noise=learn_noise,
            lengthscale_prior="lognormal",
            verbose=False,
        )

    return _fit


def _plot(results, save_path):
    """Render a ranked RMSE/NLPD bar chart for the candidate configs."""
    ordered = sorted(results, key=lambda r: r.rmse)
    labels = [r.label for r in ordered]
    rmses = [r.rmse for r in ordered]
    nlpds = [r.nlpd for r in ordered]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    b1 = ax1.bar(x - 0.2, rmses, 0.4, color="#4da6ff")
    b2 = ax2.bar(x + 0.2, nlpds, 0.4, color="#ff4d4d")
    ax1.set_ylabel("LOO-RMSE  (lower better)", color="#1f6fb2")
    ax2.set_ylabel("LOO-NLPD  (lower better)", color="#b02a22")
    for xi, r in zip(x - 0.2, rmses):
        ax1.text(xi, r, f"{r:.4f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_title(
        "GP model selection by leave-one-out (ranked by LOO-RMSE)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend([b1, b2], ["LOO-RMSE", "LOO-NLPD"], loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, str(save_path))


def main() -> int:
    """Run the LOO comparison and print a ranked table."""
    print("\n" + "=" * 70)
    print("GP MODEL SELECTION  —  LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 70)

    print("\nLoading experimental data...")
    fe_data = load_experimental_data()
    scaler = TorchMinMaxScaler()
    train_x, train_y = prepare_gp_training_tensors(fe_data, scaler)
    n = train_x.shape[0]
    print(f"Loaded {n} observations; features = (Time, Energy density)")

    configs = []
    for nu in (0.5, 1.5, 2.5):
        for learn_noise in (False, True):
            noise_tag = "learned-noise" if learn_noise else "fixed-noise"
            label = f"Matern nu={nu}, {noise_tag}"
            configs.append((label, nu, learn_noise))

    print(f"\nRunning LOO ({n} refits) for {len(configs)} configurations...")
    results = []
    for label, nu, learn_noise in configs:
        print(f"\n-> {label}")
        res = loo_cross_validate(
            train_x=train_x,
            train_y=train_y,
            fit_fn=make_fit_fn(nu, learn_noise),
            label=label,
            verbose=False,
        )
        results.append(res)
        print(
            f"   LOO-RMSE={res.rmse:.4f}  LOO-NLPD={res.nlpd:.3f}  "
            f"cov95={res.coverage95:.2f}"
        )

    ranked = sorted(results, key=lambda r: r.rmse)
    print("\n" + "=" * 70)
    print("RANKED RESULTS (by LOO-RMSE; ties broken by LOO-NLPD)")
    print("=" * 70)
    header = f"{'rank':>4} {'config':34s} {'LOO-RMSE':>9} {'LOO-NLPD':>9} {'cov95':>6}"
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(sorted(ranked, key=lambda r: (r.rmse, r.nlpd)), 1):
        print(
            f"{rank:>4} {r.label:34s} {r.rmse:>9.4f} {r.nlpd:>9.3f} "
            f"{r.coverage95:>6.2f}"
        )

    best = min(results, key=lambda r: (r.rmse, r.nlpd))
    print(f"\nBest by LOO-RMSE: {best.label}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig_path = OUT / "model_selection.png"
    _plot(results, fig_path)
    print(f"Saved figure -> {fig_path}")

    print("\n" + "=" * 70)
    print("MODEL SELECTION COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
