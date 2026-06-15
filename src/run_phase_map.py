"""Render a (time, energy) map with a level-set boundary from a trained GP -> PNG.

Demo on the existing experimental data: trains a GP on the electrical FOM and draws the
predicted-value heatmap with a threshold "boundary" contour plus the predictive
uncertainty map, with the observed experiments overlaid (axes in physical units). The
FOM stands in for a continuous crystallinity metric until XRD labels are available;
the same code then renders the real crystallization boundary.

Usage:
    python src/run_phase_map.py [--out DIR] [--epochs N] [--threshold V] [--num-points N]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from config_loader import load_config
from fit import build_and_fit_gp
from preprocessing.loaders import load_experimental_data
from preprocessing.transforms import TorchMinMaxScaler, prepare_gp_training_tensors
from visualization.grid_predictor import build_phase_map_result
from visualization.phase_map import PhaseMapPlotter

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "predictions" / "phase_map"
CONFIG_PATH = REPO_ROOT / "config" / "training_config.yaml"


def main() -> int:
    """Train a GP on the FOM data and render the phase-map + uncertainty PNGs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--epochs", type=int, default=800, help="GP training epochs")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Boundary level (default: median observed FOM)",
    )
    parser.add_argument("--num-points", type=int, default=60, help="Grid resolution/axis")
    args = parser.parse_args()

    config = load_config(str(CONFIG_PATH))

    print("Loading experimental data...")
    fe_data = load_experimental_data()
    scaler = TorchMinMaxScaler()
    train_x, train_y = prepare_gp_training_tensors(fe_data, scaler)
    print(f"  {len(train_y)} observations")

    print("Training GP on (time, voltage) + learnable warp, fit by marginal likelihood...")
    likelihood, model = build_and_fit_gp(
        train_x,
        train_y,
        kernel_type=config.model.kernel,
        matern_nu=config.model.matern_nu,
        min_lengthscale=config.model.min_lengthscale,
        learn_noise=True,
        warp_dims=list(range(train_x.shape[-1])),  # warp all dims → recovers smoothness
    )

    threshold = args.threshold if args.threshold is not None else float(np.median(train_y.numpy()))
    print(f"Boundary threshold (stand-in for crystallization): {threshold:.3f}")

    result = build_phase_map_result(
        model, likelihood, train_x, scaler,
        num_points=args.num_points,
        threshold=threshold,
        value_label="FOM 2Qsw/(U+|D|)",
        train_y=train_y,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter = PhaseMapPlotter()

    map_path = out_dir / "phase_map.png"
    plotter.plot_crystallinity_map(result, save_path=str(map_path))
    unc_path = out_dir / "phase_map_uncertainty.png"
    plotter.plot_uncertainty_map(result, save_path=str(unc_path))

    print(f"Wrote:\n  - {map_path}\n  - {unc_path}")
    print("(FOM is a stand-in target; swap in XRD crystallinity to map the real boundary.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
