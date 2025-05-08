#!/usr/bin/env python3
"""
Scatter plot of shared neighbours for two compounds.

Plots (1 - distance) values for neighbours shared between two compounds.
Includes:
    - KDE-based static plot (matplotlib)
    - Interactive scatter plot (plotly)

Usage:
    python plot_shared_neighbour_scatter.py --input nn.tsv --compound1 C1 --compound2 C2

    python plot.py --input nearest_neighbours.tsv --compound1 DDD02955130 --compound2 DDD02459457

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
    """
    # Deduplicate by keeping the smallest distance for each (cpd_id, neighbour_id)
    df = df.sort_values("distance").drop_duplicates(subset=["cpd_id", "neighbour_id"], keep="first")

    # Filter for each compound
    df1 = df[df["cpd_id"] == cpd1].set_index("neighbour_id")
    df2 = df[df["cpd_id"] == cpd2].set_index("neighbour_id")

    # Identify shared neighbours with no missing distance
    common = df1.index.intersection(df2.index)
    common = [n for n in common if pd.notnull(df1.at[n, "distance"]) and pd.notnull(df2.at[n, "distance"])]

    if not common:
        print("No valid shared neighbours found (after cleaning).")
        return

    # Sort and align
    common = sorted(common)
    x = 1 - df1.loc[common, "distance"]
    y = 1 - df2.loc[common, "distance"]

    # KDE for density
    xy = np.vstack([x.to_numpy(), y.to_numpy()])
    z = gaussian_kde(xy)(xy)

    # Sort by density
    idx = z.argsort()
    x_sorted = x.to_numpy()[idx]
    y_sorted = y.to_numpy()[idx]
    z_sorted = z[idx]
    neighbours_sorted = np.array(common)[idx]

    # --- Static KDE plot ---
    plt.figure(figsize=(6, 6))
    plt.scatter(x_sorted, y_sorted, c=z_sorted, cmap="viridis", s=60, edgecolors="none")
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
        f"{cpd1} (1-distance)": x,
        f"{cpd2} (1-distance)": y,
        "neighbour_id": common,
    })

    # Save clean shared neighbour data to TSV
    df_plot.to_csv(f"{output_prefix}_shared_neighbours.tsv", sep="\t", index=False)
    print(f"Shared neighbours saved to {output_prefix}_shared_neighbours.tsv")

    fig = px.scatter(
        df_plot,
        x=f"{cpd1} (1-distance)",
        y=f"{cpd2} (1-distance)",
        hover_name="neighbour_id",
        title="Interactive Neighbour Similarity Scatter",
        width=700,
        height=700,
    )
    fig.update_traces(marker=dict(size=10, color="blue"))
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
