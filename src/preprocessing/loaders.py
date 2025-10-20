"""Data loading and exploratory visualization for ferroelectric experiments.

This module handles:
- Loading experimental data from Excel files
- Data cleaning and validation
- Exploratory visualization with scatter matrices
"""

import pandas as pd
import plotly.express as px

# Visualization constant
DEFAULT_COLOR = "#72356c"


def read_dat(
    dir="/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/data/",
    src_file="Bolometer_readings_PulseForge.xlsx",
    sheet="Combined",
) -> pd.DataFrame:
    """Load and clean ferroelectric experimental data.

    ML Reasoning:
    -------------
    Small dataset quality is critical for GP performance. We aggressively
    filter invalid data because:
    - GPs are sensitive to outliers and noise
    - Missing values can destabilize covariance matrix inversions
    - Zero measurements indicate failed experiments (not informative)
    - Clean data enables more accurate lengthscale learning

    Data Properties:
    ----------------
    - Input 1: Pulse width (Time in ms) - controls thermal diffusion
    - Input 2: Energy density (J/cm²) - controls peak temperature
    - Output: Figure of merit (2 Qsw/(U+|D|) 1e6cycles) - quality

    :param dir: Directory containing experimental data
    :param src_file: Excel file with bolometer readings
    :param sheet: Worksheet name containing combined measurements
    :return: Cleaned data with 3 columns (2 inputs, 1 output)
    :rtype: pd.DataFrame
    """
    file = dir + src_file

    # Load only relevant columns to minimize memory usage
    fe_data = pd.read_excel(
        file,
        sheet_name=sheet,
        usecols=[
            "Energy density new cone (J/cm^2)",  # X₁: Energy input
            "Time (ms)",  # X₂: Pulse duration
            "2 Qsw/(U+|D|) 1e6cycles",  # y: Target metric
        ],
    )

    # Data cleaning: Critical for small datasets
    # Remove NaN values - cannot train with missing labels
    fe_data.dropna(subset=["2 Qsw/(U+|D|) 1e6cycles"], inplace=True)

    # Remove zero measurements - indicates experimental failure
    # Zero is physically invalid (division by polarization)
    fe_data = fe_data[fe_data["2 Qsw/(U+|D|) 1e6cycles"] != 0]

    return fe_data


def display_data(fe_data: pd.DataFrame) -> None:
    """Create scatter matrix to visualize parameter relationships.

    ML Reasoning:
    -------------
    Exploratory visualization is essential for:
    - Identifying input-output correlations (guides lengthscale priors)
    - Detecting non-linear relationships (justifies GP over linear models)
    - Spotting outliers or clusters (informs kernel choice)
    - Understanding parameter coupling (helps with ARD kernel selection)

    Visualization Strategy:
    -----------------------
    Scatter matrix shows all pairwise relationships:
    - Diagonal: Distribution of each variable
    - Off-diagonal: Bivariate relationships
    - Helps identify if Matern/RBF kernels are appropriate

    :param fe_data: DataFrame with experimental measurements
    :type fe_data: pd.DataFrame
    """
    # Create scatter matrix for all variable pairs
    fig = px.scatter_matrix(
        fe_data,
        dimensions=[
            "Energy density new cone (J/cm^2)",
            "Time (ms)",
            "2 Qsw/(U+|D|) 1e6cycles",
        ],
        color_discrete_sequence=[DEFAULT_COLOR] * 23,
    )

    # Apply clean styling
    fig.update_layout(template="ggplot2", width=800, height=800)
    fig.show()
