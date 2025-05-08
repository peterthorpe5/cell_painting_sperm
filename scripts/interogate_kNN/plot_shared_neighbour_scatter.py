#!/usr/bin/env python3
"""
Scatter plot of shared neighbours for two compounds.

Plots (1 - distance) values for neighbours shared between two compounds.
Includes:
    - KDE-based static plot (matplotlib)
    - Interactive scatter plot (plotly)

Usage:
    python plot_shared_neighbour_scatter.py --input nn.tsv --compound1 C1 --compound2 C2

"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import argparse
import plotly.express as px


def plot_kde_scatter(df: pd.DataFrame, cpd1: str, cpd2: str, output_prefix: str) -> None:
    """
    Plot scatter of 1-distance values for neighbours shared between two compounds.

    Includes:
        - Matplotlib static KDE plot
        - Plotly interactive scatter plot with neighbour_id tooltips

    Parameters:
        df (pd.DataFrame): Nearest neighbour dataframe.
        cpd1 (str): First compound ID.
        cpd2 (str): Second compound ID.
        output_prefix (str): Prefix for saved plot filenames.
    """
    df1 = df[df["cpd_id"] == cpd1].set_index("neighbour_id")
    df2 = df[df["cpd_id"] == cpd2].set_index("neighbour_id")
    common = df1.index.intersection(df2.index)

    if common.empty:
        print("No shared neighbours found.")
        return

    x = 1 - df1.loc[common, "distance"]
    y = 1 - df2.loc[common, "distance"]
    neighbours = common.tolist()

    # KDE for colour density (matplotlib version)
    xy = np.vstack([x.values, y.values])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x_sorted = x.values[idx]
    y_sorted = y.values[idx]
    z_sorted = z[idx]
    neighbours_sorted = np.array(neighbours)[idx]

    # --- Static KDE plot (matplotlib) ---
    plt.figure(figsize=(6, 6))
    plt.scatter(x_sorted, y_sorted, c=z_sorted, cmap="viridis", s=60, edgecolor="")
    cb = plt.colorbar()
    cb.set_label("Density")

    plt.xlabel(f"{cpd1} (1 - distance)")
    plt.ylabel(f"{cpd2} (1 - distance)")
    plt.title("Neighbour similarity comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_scatter_kde.pdf", dpi=1200)
    plt.close()
    print(f"Static KDE scatter saved to {output_prefix}_scatter_kde.pdf")

    # --- Interactive Plotly plot ---
    df_plot = pd.DataFrame({
        f"{cpd1} (1-distance)": x.values,
        f"{cpd2} (1-distance)": y.values,
        "neighbour_id": neighbours,
    })

    fig = px.scatter(
        df_plot,
        x=f"{cpd1} (1-distance)",
        y=f"{cpd2} (1-distance)",
        hover_name="neighbour_id",
        title="Interactive Neighbour Similarity Scatter",
        width=700,
        height=700,
    )
    fig.update_traces(marker=dict(size=10, colour="blue"))
    fig.write_html(f"{output_prefix}_interactive_scatter.html")
    print(f"Interactive scatter saved to {output_prefix}_interactive_scatter.html")


def main():
    parser = argparse.ArgumentParser(description="Plot KDE scatter and interactive scatter of shared neighbours.")
    parser.add_argument("--input", required=True, help="Path to nearest neighbour TSV file")
    parser.add_argument("--compound1", required=True, help="First compound ID")
    parser.add_argument("--compound2", required=True, help="Second compound ID")
    parser.add_argument("--output_prefix", default="scatter_output", help="Prefix for saved output file")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    plot_kde_scatter(df, args.compound1, args.compound2, args.output_prefix)


if __name__ == "__main__":
    main()
