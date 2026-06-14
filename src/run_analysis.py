"""Standalone script for parameter correlation and feature importance analysis.

This script can be run independently to analyze existing trained GP models,
or can analyze the current experimental data without requiring a trained model
for correlation analysis.

Usage:
    python src/run_analysis.py
"""

import sys
import warnings

import gpytorch
import torch

sys.path.append(".")

from analysis import ParameterAnalyzer
from config_loader import load_config
from models.factory import create_gp_model
from preprocessing.loaders import load_experimental_data
from preprocessing.transforms import TorchMinMaxScaler, prepare_gp_training_tensors

warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning)


def main():
    """Run comprehensive parameter analysis."""
    print("\n" + "=" * 70)
    print("PARAMETER CORRELATION & FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Load config
    config = load_config("../config/training_config.yaml")

    # Load data
    print("\nLoading experimental data...")
    fe_data = load_experimental_data()
    print(f"Loaded {len(fe_data)} experimental observations")

    scaler = TorchMinMaxScaler()
    train_x, train_y = prepare_gp_training_tensors(fe_data, scaler)
    feature_names = ["Pulse Time (ms)", "Energy Density (J/cm²)"]

    # Initialize analyzer
    analyzer = ParameterAnalyzer(
        train_x=train_x,
        train_y=train_y,
        feature_names=feature_names,
    )

    # Always compute correlations (doesn't require trained model)
    print("\n" + "-" * 70)
    print("STEP 1: CORRELATION ANALYSIS (Data-only)")
    print("-" * 70)
    correlations = analyzer.compute_correlations(methods=["pearson", "spearman"])

    print("\nParameter-FOM Correlations:")
    for i, name in enumerate(feature_names):
        print(
            f"  {name:30s}: Pearson={correlations.pearson[i]:6.4f}, "
            f"Spearman={correlations.spearman[i]:6.4f}, "
            f"Partial={correlations.partial[i]:6.4f}"
        )

    print("\nParameter-Parameter Correlation Matrix:")
    param_param = correlations.param_param_matrix
    print(
        f"  Correlation between {feature_names[0]} and {feature_names[1]}: "
        f"{param_param[0, 1]:.4f}"
    )

    # Load and analyze trained model if available
    try:
        print("\n" + "-" * 70)
        print("STEP 2: FEATURE IMPORTANCE FROM GP MODEL")
        print("-" * 70)

        # Create model (will load checkpoint if available)
        likelihood, model, _ = create_gp_model(
            train_x=train_x,
            train_y=train_y,
            kernel_type=config.model.kernel,
            lengthscale=config.model.lengthscale_prior,
            noise=config.model.noise_prior,
            num_dims=config.model.input_dim,
            min_lengthscale=config.model.min_lengthscale,
            matern_nu=config.model.matern_nu,
        )

        # Try to load trained model
        try:
            checkpoint = torch.load("../models/model_state.pth")
            model.load_state_dict(checkpoint["model_state_dict"])
            likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
            print("Loaded trained model from checkpoint")
        except FileNotFoundError:
            print("No trained model found. Using initial model parameters.")
            print("  (Run training first for accurate feature importance)")

        model.eval()
        likelihood.eval()

        # ARD importance
        print("\nComputing ARD feature importance from lengthscales...")
        ard_result = analyzer.extract_ard_importance(model)
        lengthscales = ard_result.metadata.get("lengthscales")
        print("\nARD Lengthscales:")
        for i, name in enumerate(feature_names):
            print(
                f"  {name:30s}: {lengthscales[i]:8.4f} "
                f"(Importance: {ard_result.importance_scores[i]:6.4f})"
            )

        # Optional: Sobol indices (may be slow)
        print("\n" + "-" * 70)
        print("STEP 3: ADVANCED ANALYSES (Optional)")
        print("-" * 70)

        try:
            print("\nComputing Sobol sensitivity indices (global sensitivity)...")
            sobol_result = analyzer.compute_sobol_indices(model, likelihood, n_samples=5000)
            print("\nSobol First-Order Indices (S1):")
            for i, name in enumerate(feature_names):
                print(f"  {name:30s}: {sobol_result.importance_scores[i]:8.4f}")
            print("\nSobol Total-Order Indices (ST):")
            total_order = sobol_result.metadata.get("total_order")
            for i, name in enumerate(feature_names):
                print(f"  {name:30s}: {total_order[i]:8.4f}")
        except Exception as e:
            print(f"  Sobol analysis skipped: {e}")
            print("  (Install SALib: pip install SALib)")

    except Exception as e:
        print(f"\nModel-based analysis skipped: {e}")
        print("  (This is normal if no trained model is available)")

    # Generate summary report
    print("\n" + "-" * 70)
    print("STEP 4: GENERATING SUMMARY REPORT")
    print("-" * 70)

    summary_df = analyzer.generate_summary_report()
    summary_df.to_csv("../predictions/parameter_analysis_summary.csv", index=False)
    print("\nSummary report saved to: predictions/parameter_analysis_summary.csv")
    print("\n" + summary_df.to_string(index=False))

    # Generate visualizations
    print("\n" + "-" * 70)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("-" * 70)

    try:
        analyzer.plot_correlation_matrix(save_path="../predictions/correlation_matrix.png")
        print("  ✓ Correlation matrix: predictions/correlation_matrix.png")

        if analyzer.importance_results:
            analyzer.plot_feature_importance_comparison(
                save_path="../predictions/feature_importance_comparison.png"
            )
            print("  ✓ Feature importance: " "predictions/feature_importance_comparison.png")

        analyzer.plot_correlation_barplot(save_path="../predictions/correlation_barplot.png")
        print("  ✓ Correlation barplot: predictions/correlation_barplot.png")
    except Exception as e:
        print(f"  Visualization error: {e}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
