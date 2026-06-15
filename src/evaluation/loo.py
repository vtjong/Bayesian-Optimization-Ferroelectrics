"""Leave-one-out cross-validation for small-n GP model selection.

At n~40 the cheapest defensible way to estimate out-of-sample predictive
quality is exact leave-one-out (LOO): refit the GP on n-1 points and score the
held-out point. This module computes:

- **LOO-RMSE**: point-prediction error on held-out targets.
- **LOO-NLPD**: mean negative log predictive density (lower = better calibrated
  AND accurate); uses the *predictive* variance including observation noise.
- **LOO coverage**: fraction of held-out points inside the 95% predictive
  interval (≈0.95 indicates well-calibrated uncertainty).

It is model-agnostic: the caller passes a ``build_and_fit`` callable that takes
(train_x, train_y) and returns a fitted ``(likelihood, model)`` pair, so the
same harness compares fixed-vs-learned noise, kernels, and Matern nu.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class LOOResult:
    """Aggregated leave-one-out metrics for one model configuration.

    :ivar label: Human-readable configuration name (for ranked tables)
    :ivar rmse: LOO root-mean-squared error on held-out targets
    :ivar nlpd: Mean LOO negative log predictive density (lower is better)
    :ivar coverage95: Fraction of held-out points inside the 95% interval
    :ivar means: Per-point held-out predictive means (n,)
    :ivar stds: Per-point held-out predictive stds incl. noise (n,)
    :ivar targets: The held-out targets, in fold order (n,)
    """

    label: str
    rmse: float
    nlpd: float
    coverage95: float
    means: np.ndarray
    stds: np.ndarray
    targets: np.ndarray


# A fitter maps (train_x, train_y) -> (likelihood, fitted_model).
FitFn = Callable[[torch.Tensor, torch.Tensor], Tuple[object, object]]


def _observation_noise_var(likelihood) -> float:
    """Read the scalar observation-noise variance from a Gaussian likelihood.

    Works for both ``GaussianLikelihood`` (learned, homoskedastic) and
    ``FixedNoiseGaussianLikelihood`` (fixed per-point noise, assumed constant).
    Returning a scalar lets us add predictive noise explicitly rather than
    relying on ``observation_noise=True`` at a query point, which is brittle
    for the fixed-noise likelihood on unseen inputs.

    :param likelihood: Fitted Gaussian likelihood
    :return: Observation-noise variance as a Python float
    """
    noise = likelihood.noise
    return float(noise.reshape(-1)[0].item())


def _predict_one(
    model, likelihood, test_x: torch.Tensor
) -> Tuple[float, float]:
    """Predict mean and predictive std (incl. observation noise) at one point.

    Uses the BoTorch latent ``posterior`` (so any attached input transform
    such as a learnable Warp is applied) and adds the observation-noise variance
    explicitly. The resulting std reflects the full predictive distribution used
    by NLPD and coverage.

    :param model: Fitted GP model (eval mode)
    :param likelihood: Fitted likelihood (source of the noise variance)
    :param test_x: Single query point of shape (1, n_features)
    :return: (mean, std) as Python floats
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        posterior = model.posterior(test_x, observation_noise=False)
        mean = posterior.mean.reshape(-1)[0].item()
        latent_var = posterior.variance.reshape(-1)[0].item()
    pred_var = max(latent_var + _observation_noise_var(likelihood), 1e-9)
    return mean, float(np.sqrt(pred_var))


def loo_cross_validate(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    fit_fn: FitFn,
    label: str = "model",
    verbose: bool = False,
) -> LOOResult:
    """Run exact leave-one-out cross-validation, refitting per left-out point.

    :param train_x: Full input set (n_samples, n_features), scaled to [0, 1]
    :param train_y: Full target vector (n_samples,)
    :param fit_fn: Callable (train_x, train_y) -> (likelihood, fitted_model)
    :param label: Configuration name carried into the result
    :param verbose: Print per-fold progress when True
    :return: :class:`LOOResult` with aggregated LOO metrics
    """
    n = train_x.shape[0]
    means = np.empty(n)
    stds = np.empty(n)
    targets = train_y.detach().cpu().numpy().astype(float)

    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool)
        mask[i] = False
        x_tr, y_tr = train_x[mask], train_y[mask]
        x_te = train_x[i:i + 1]

        likelihood, model = fit_fn(x_tr, y_tr)
        mean, std = _predict_one(model, likelihood, x_te)
        means[i], stds[i] = mean, std

        if verbose:
            print(f"  [{label}] fold {i + 1}/{n}: y={targets[i]:.4f} "
                  f"pred={mean:.4f}+/-{std:.4f}")

    errors = means - targets
    rmse = float(np.sqrt(np.mean(errors**2)))

    # Negative log predictive density under a Gaussian predictive density.
    var = stds**2
    nlpd = float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * errors**2 / var))

    # 95% interval coverage (1.96 sigma).
    inside = np.abs(errors) <= 1.96 * stds
    coverage95 = float(np.mean(inside))

    return LOOResult(
        label=label,
        rmse=rmse,
        nlpd=nlpd,
        coverage95=coverage95,
        means=means,
        stds=stds,
        targets=targets,
    )
