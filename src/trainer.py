"""GP training utilities with MLE optimization."""

from typing import List, Tuple

import gpytorch
import torch


def train_gp_model(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    learning_rate: float = 0.003,
    n_epochs: int = 3000,
    log_interval: int = 500,
    train_lengthscale: bool = True,
) -> Tuple[gpytorch.models.ExactGP, gpytorch.likelihoods.Likelihood, List[float]]:
    """Train GP model via maximum likelihood estimation.

    Optimizes kernel hyperparameters (lengthscales, outputscale) and
    constant mean using Adam optimizer on negative marginal log-likelihood.

    :param model: GP model to train
    :param likelihood: Gaussian likelihood
    :param train_x: Training inputs (n_samples, n_features)
    :param train_y: Training targets (n_samples,)
    :param learning_rate: Adam learning rate
    :param n_epochs: Number of training iterations
    :param log_interval: Log progress every N iterations
    :param train_lengthscale: Whether to train lengthscales
    :return: (trained_model, trained_likelihood, loss_history)
    """
    # Freeze lengthscales if specified
    if not train_lengthscale:
        model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)

    # Set to training mode
    model.train()
    likelihood.train()

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    loss_history = []

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        # Log progress
        if epoch == 1 or epoch % log_interval == 0:
            lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy()
            noise = model.likelihood.noise.detach().numpy()
            print(
                f"Epoch {epoch}/{n_epochs} - "
                f"Loss: {loss.item():.3f}  "
                f"Lengthscale: {lengthscale}  "
                f"Noise: {noise[0]:.4f}"
            )
            loss_history.append(loss.item())

        optimizer.step()

    # Print final outputscale
    outputscale = model.covar_module.outputscale.detach().numpy()
    print(f"\nFinal outputscale: {outputscale:.4f}")

    return model, likelihood, loss_history


def save_model_checkpoint(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    loss_history: List[float],
    save_path: str = "models/model_state.pth",
) -> None:
    """Save trained model checkpoint.

    :param model: Trained GP model
    :param likelihood: Trained likelihood
    :param loss_history: Training loss history
    :param save_path: Path to save checkpoint
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "likelihood_state_dict": likelihood.state_dict(),
        "loss": loss_history,
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model_checkpoint(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    load_path: str = "models/model_state.pth",
) -> Tuple[gpytorch.models.ExactGP, gpytorch.likelihoods.Likelihood, List[float]]:
    """Load trained model checkpoint.

    :param model: Model instance (for architecture)
    :param likelihood: Likelihood instance
    :param load_path: Path to checkpoint file
    :return: (loaded_model, loaded_likelihood, loss_history)
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    loss_history = checkpoint["loss"]

    print(f"Model loaded from {load_path}")
    return model, likelihood, loss_history
