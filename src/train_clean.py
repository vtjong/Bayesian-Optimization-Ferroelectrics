"""Clean training script for GP model with Bayesian Optimization.

This script demonstrates the clean workflow:
1. Load config
2. Load & preprocess data
3. Create GP model
4. Train model
5. Evaluate performance
6. Suggest next experiments

Use this as a template for cleaning up training.ipynb
"""

import sys
import warnings

import gpytorch
import numpy as np
import pandas as pd
import torch

sys.path.append(".")

from config_loader import config_to_args, load_config
from evaluator import evaluate_model
from models.factory import create_gp_model
from optimization.acquisition import (
    format_suggestions,
    suggest_next_experiments_analytic,
    suggest_next_experiments_mc,
)
from preprocessing.loaders import load_experimental_data
from preprocessing.transforms import TorchMinMaxScaler, prepare_gp_training_tensors
from trainer import save_model_checkpoint, train_gp_model

warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning)


def main():
    """Main training and optimization workflow."""

    # ========================================================================
    # 1. LOAD CONFIGURATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING CONFIGURATION")
    print("=" * 70)

    config = load_config("config/training_config.yaml")

    # Set random seeds
    torch.manual_seed(config.compute.seed)
    np.random.seed(config.compute.seed)

    print(f"Kernel: {config.model.kernel}")
    print(f"Matérn ν: {config.model.matern_nu}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Learning rate: {config.training.learning_rate}")

    # ========================================================================
    # 2. LOAD & PREPROCESS DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: LOADING & PREPROCESSING DATA")
    print("=" * 70)

    fe_data = load_experimental_data()
    print(f"Loaded {len(fe_data)} experimental observations")

    scaler = TorchMinMaxScaler()
    train_x, train_y = prepare_gp_training_tensors(fe_data, scaler)
    num_samples, num_params = train_x.shape

    print(f"Tensor shapes: X={train_x.shape}, y={train_y.shape}")
    print(f"Input range: [{train_x.min():.3f}, {train_x.max():.3f}]")
    print(f"Output range: [{train_y.min():.3f}, {train_y.max():.3f}]")

    # ========================================================================
    # 3. CREATE GP MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: CREATING GP MODEL")
    print("=" * 70)

    likelihood, model, lengthscales = create_gp_model(
        train_x=train_x,
        train_y=train_y,
        kernel_type=config.model.kernel,
        lengthscale=config.model.lengthscale_prior,
        noise=config.model.noise_prior,
        num_dims=config.model.input_dim,
        min_lengthscale=config.model.min_lengthscale,
        matern_nu=config.model.matern_nu,
    )

    # ========================================================================
    # 4. TRAIN MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING MODEL")
    print("=" * 70)

    if config.training.train_flag:
        model, likelihood, loss_history = train_gp_model(
            model=model,
            likelihood=likelihood,
            train_x=train_x,
            train_y=train_y,
            learning_rate=config.training.learning_rate,
            n_epochs=config.training.epochs,
            log_interval=config.training.log_interval,
            train_lengthscale=config.model.train_lengthscale,
        )

        # Save checkpoint
        save_model_checkpoint(model, likelihood, loss_history)
    else:
        print("Skipping training (train_flag=False)")
        loss_history = []

    # ========================================================================
    # 5. EVALUATE MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATING MODEL PERFORMANCE")
    print("=" * 70)

    y_pred_mean, y_pred_std, metrics = evaluate_model(
        model=model,
        likelihood=likelihood,
        test_x=train_x,
        test_y=train_y,
    )

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:12s}: {metric_value:.4f}")

    # ========================================================================
    # 6. SUGGEST NEXT EXPERIMENTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: SUGGESTING NEXT EXPERIMENTS")
    print("=" * 70)

    bounds = torch.tensor([[0.0] * num_params, [1.0] * num_params])
    feature_names = ["Pulse Time (ms)", "Energy Density (J/cm²)"]

    if config.acquisition.mc_or_analytic == "analytic":
        suggestions = suggest_next_experiments_analytic(
            model=model,
            likelihood=likelihood,
            train_y=train_y,
            bounds=bounds,
            beta=5.0,
        )
    else:
        suggestions = suggest_next_experiments_mc(
            model=model,
            likelihood=likelihood,
            train_y=train_y,
            bounds=bounds,
            q=config.acquisition.num_suggestions,
            beta=5.0,
            seed=config.compute.seed,
            acq_functions=config.acquisition.functions,
        )

    format_suggestions(suggestions, scaler, feature_names)

    # ========================================================================
    # 7. EXPORT SUGGESTIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: EXPORTING SUGGESTIONS")
    print("=" * 70)

    suggestion_rows = []
    for acq_name, (candidates, predictions) in suggestions.items():
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
            predictions = [predictions]

        candidates_original = scaler.inverse_transform(
            torch.from_numpy(candidates).float()
        ).numpy()

        for i, (cand, pred) in enumerate(zip(candidates_original, predictions)):
            suggestion_rows.append(
                {
                    "Acquisition_Function": acq_name,
                    "Candidate_ID": i + 1,
                    "Pulse_Time_ms": cand[0],
                    "Energy_Density_J_cm2": cand[1],
                    "Predicted_FOM": pred,
                }
            )

    suggestions_df = pd.DataFrame(suggestion_rows)
    suggestions_df.to_csv("predictions/next_experiments.csv", index=False)
    print("\nSuggestions saved to: predictions/next_experiments.csv")
    print(suggestions_df)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
