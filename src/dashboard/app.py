"""Streamlit dashboard for the HZO Bayesian-optimization project.

Run locally:   streamlit run src/dashboard/app.py
Or via Docker:  scripts/run_dashboard.sh docker

Tabs: Data & Runs | Phase Map | Crystal Structures | Run Experiments. All compute is
delegated to the reusable packages; this file is presentation only.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering for st.pyplot

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# Make the `src/` packages importable regardless of launch directory.
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent
PREDICTIONS_DIR = REPO_ROOT / "predictions"
CONFIG_PATH = REPO_ROOT / "config" / "training_config.yaml"

from config_loader import load_config
from models.factory import create_gp_model
from preprocessing.loaders import load_experimental_data
from preprocessing.transforms import TorchMinMaxScaler, prepare_gp_training_tensors
from trainer import train_gp_model
from visualization.grid_predictor import build_phase_map_result
from visualization.phase_map import PhaseMapPlotter


# --------------------------------------------------------------------------- #
# Cached backend helpers (reuse the existing pipeline)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training GP on experimental data...")
def get_trained_model(epochs: int):
    """Load data and train a GP; cached so widgets don't retrigger training."""
    config = load_config(str(CONFIG_PATH))
    fe_data = load_experimental_data()
    scaler = TorchMinMaxScaler()
    train_x, train_y = prepare_gp_training_tensors(fe_data, scaler)
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
    model, likelihood, _ = train_gp_model(
        model,
        likelihood,
        train_x,
        train_y,
        learning_rate=config.training.learning_rate,
        n_epochs=epochs,
        log_interval=max(1, epochs // 2),
    )
    return model, likelihood, scaler, train_x, train_y


@st.cache_data(show_spinner=False)
def load_run_csvs():
    """List CSV outputs under predictions/ as (name, DataFrame) pairs."""
    if not PREDICTIONS_DIR.exists():
        return []
    return [(p.name, pd.read_csv(p)) for p in sorted(PREDICTIONS_DIR.glob("*.csv"))]


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_data_and_runs():
    st.header("Data & past runs")
    fe_data = load_experimental_data()
    st.subheader(f"Experimental data ({len(fe_data)} rows)")
    st.dataframe(fe_data, use_container_width=True)

    csvs = load_run_csvs()
    if csvs:
        st.subheader("Saved run outputs (predictions/*.csv)")
        name = st.selectbox("Pick a file", [n for n, _ in csvs])
        df = dict(csvs)[name]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No predictions/*.csv yet. Run an experiment in the last tab.")


def _build_result(epochs: int, num_points: int, threshold: float):
    model, likelihood, scaler, train_x, train_y = get_trained_model(epochs)
    result = build_phase_map_result(
        model,
        likelihood,
        train_x,
        scaler,
        num_points=num_points,
        threshold=threshold,
        value_label="FOM 2Qsw/(U+|D|)",
        train_y=train_y,
    )
    return result


def tab_phase_map():
    st.header("Phase map + GP surface")
    st.caption("FOM stands in for crystallinity until XRD labels are available.")
    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Training epochs", 100, 3000, 800, step=100)
    num_points = col2.slider("Grid resolution", 20, 100, 60, step=10)
    threshold = col3.number_input("Boundary threshold", value=3.0, step=0.1)

    result = _build_result(epochs, num_points, threshold)
    plotter = PhaseMapPlotter()

    st.subheader("Crystallinity / FOM map with boundary contour")
    st.pyplot(plotter.plot_crystallinity_map(result))
    st.subheader("Predictive uncertainty")
    st.pyplot(plotter.plot_uncertainty_map(result))

    st.subheader("3D GP surface")
    surface = go.Figure(
        go.Surface(z=result.mean, x=result.x_coords, y=result.y_coords, colorscale="Viridis")
    )
    if result.obs_value is not None:
        surface.add_trace(
            go.Scatter3d(
                x=result.obs_x,
                y=result.obs_y,
                z=result.obs_value,
                mode="markers",
                marker=dict(size=4, color="black"),
                name="experiments",
            )
        )
    surface.update_layout(
        height=600,
        scene=dict(
            xaxis_title=result.x_label,
            yaxis_title=result.y_label,
            zaxis_title=result.value_label,
        ),
    )
    st.plotly_chart(surface, use_container_width=True)


def tab_structures():
    st.header("HfO2 crystal structures")
    st.caption("Only the polar orthorhombic (Pca2_1) phase is ferroelectric.")
    try:
        from visualization.structures import (
            CachedStructureProvider,
            CrystalStructureVisualizer,
            MaterialsProjectProvider,
            available_phase_keys,
            get_phase,
        )

        viz = CrystalStructureVisualizer(
            provider=CachedStructureProvider(MaterialsProjectProvider())
        )
    except (ImportError, ValueError) as exc:
        st.warning(
            f"Crystal-structure rendering unavailable: {exc}\n\n"
            "Install `pip install -r requirements-viz.txt` and set MP_API_KEY."
        )
        return

    keys = available_phase_keys()
    cols = st.columns(2)
    for i, key in enumerate(keys):
        phase = get_phase(key)
        with cols[i % 2]:
            st.markdown(f"**{phase.name}** — {phase.space_group}")
            st.caption(phase.description)
            with st.spinner(f"Rendering {key}..."):
                st.pyplot(viz.render_phase(key))


def tab_run_experiments():
    st.header("Run experiments")
    st.write("Train the GP and run the acquisition to propose next experiments.")
    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Epochs", 100, 3000, 800, step=100, key="exp_epochs")
    q = col2.slider("Batch size (q)", 1, 8, 4)
    beta = col3.number_input("UCB beta", value=5.0, step=0.5)

    if not st.button("Suggest next experiments"):
        return

    from optimization.acquisition import suggest_next_experiments_mc

    model, likelihood, scaler, train_x, train_y = get_trained_model(epochs)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    with st.spinner("Optimizing acquisition..."):
        suggestions = suggest_next_experiments_mc(
            model,
            likelihood,
            train_y,
            bounds,
            q=q,
            beta=beta,
            acq_functions=["qUCB"],
        )

    candidates_scaled, preds = suggestions["qUCB"]
    physical = scaler.inverse_transform(torch.as_tensor(candidates_scaled)).numpy()
    table = pd.DataFrame(
        {
            "Pulse Time (ms)": physical[:, 0],
            "Energy density (J/cm^2)": physical[:, 1],
            "Predicted FOM": [float(p) for p in preds.reshape(-1)[: len(physical)]],
        }
    )
    st.subheader("Suggested next experiments (qUCB)")
    st.dataframe(table, use_container_width=True)

    # Overlay suggestions on the phase map.
    result = build_phase_map_result(
        model,
        likelihood,
        train_x,
        scaler,
        num_points=60,
        threshold=float(train_y.median()),
        value_label="FOM",
        train_y=train_y,
    )
    fig = PhaseMapPlotter().plot_crystallinity_map(result)
    fig.axes[0].scatter(
        table["Pulse Time (ms)"],
        table["Energy density (J/cm^2)"],
        marker="*",
        s=260,
        c="red",
        edgecolors="black",
        zorder=6,
        label="suggested",
    )
    fig.axes[0].legend(loc="best")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="HZO BO Dashboard", layout="wide")
    st.title("HZO Bayesian-Optimization Dashboard")
    st.sidebar.markdown(
        "Local tool for the HZO project.\n\n"
        "- **Data & Runs** — browse data + saved outputs\n"
        "- **Phase Map** — GP map + boundary + 3D surface\n"
        "- **Crystal Structures** — HfO2 polymorphs (needs MP_API_KEY)\n"
        "- **Run Experiments** — train + suggest next points"
    )
    tabs = st.tabs(["Data & Runs", "Phase Map", "Crystal Structures", "Run Experiments"])
    with tabs[0]:
        tab_data_and_runs()
    with tabs[1]:
        tab_phase_map()
    with tabs[2]:
        tab_structures()
    with tabs[3]:
        tab_run_experiments()


if __name__ == "__main__":
    main()
