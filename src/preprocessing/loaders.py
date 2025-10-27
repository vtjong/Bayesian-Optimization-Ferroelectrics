"""Data loading and visualization for ferroelectric experiments."""

import pandas as pd
import plotly.express as px

# Visualization constant
DEFAULT_COLOR = "#72356c"


def load_experimental_data(
    dir="/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/data/",
    src_file="Bolometer_readings_PulseForge.xlsx",
    sheet="Combined",
) -> pd.DataFrame:
    """Load and clean ferroelectric experimental data from Excel.

    Removes NaN and zero values from target metric to ensure data quality
    for Gaussian Process training.

    :param dir: Directory containing experimental data
    :param src_file: Excel file with bolometer readings
    :param sheet: Worksheet name containing combined measurements
    :return: DataFrame with 3 columns: energy density, pulse time, and FOM
    :rtype: pd.DataFrame
    """
    file = dir + src_file

    fe_data = pd.read_excel(
        file,
        sheet_name=sheet,
        usecols=[
            "Energy density new cone (J/cm^2)",
            "Time (ms)",
            "2 Qsw/(U+|D|) 1e6cycles",
        ],
    )

    # Remove invalid measurements
    fe_data.dropna(subset=["2 Qsw/(U+|D|) 1e6cycles"], inplace=True)
    fe_data = fe_data[fe_data["2 Qsw/(U+|D|) 1e6cycles"] != 0]

    return fe_data


def plot_input_output_scatter_matrix(fe_data: pd.DataFrame) -> None:
    """Display scatter matrix of input parameters vs figure of merit.

    Shows pairwise relationships between energy density, pulse time, and
    the target figure of merit to identify correlations and non-linearities.

    :param fe_data: DataFrame with experimental measurements (inputs and FOM)
    :type fe_data: pd.DataFrame
    """
    fig = px.scatter_matrix(
        fe_data,
        dimensions=[
            "Energy density new cone (J/cm^2)",
            "Time (ms)",
            "2 Qsw/(U+|D|) 1e6cycles",
        ],
        color_discrete_sequence=[DEFAULT_COLOR] * 23,
    )

    fig.update_layout(template="ggplot2", width=800, height=800)
    fig.show()
