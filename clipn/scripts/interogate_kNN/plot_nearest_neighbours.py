#!/usr/bin/env python3
"""
Visualise nearest neighbour distances between two compounds.

Generates:
1. A scatter plot comparing neighbour similarity between two compounds.
   - Includes optional KDE-based density colouring.
2. A bar chart showing top-N nearest neighbours for each compound.

Usage:
    python compare_neighbours.py --input nn.tsv --compound1 DDD00071692 --compound2 DDD00088797 --top_n 5 --plot_type kde


"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gaussian_kde
import numpy as np


def load_nn_file(file_path: str) -> pd.DataFrame:
    """
    Load the nearest neighbour TSV file.

    Parameters:
        file_path (str): Path to nearest neighbour file.

    Returns:
        pd.DataFrame: Loaded DataFrame with cpd_id, neighbour_id, and distance.
    """
    return pd.read_csv(file_path, sep="\t")


def plot_comparison_scatter(df: pd.DataFrame, cpd1: str, cpd2: str, output_prefix: str, plot_type: str = "kde") -> None:
    """
    Plot scatter of 1-distance values for neighbours shared between two compounds.

    Parameters:
        df (pd.DataFrame): Nearest neighbour dataframe.
        cpd1 (str): First compound ID.
        cpd2 (str): Second compound ID.
        output_prefix (str): Prefix for saved plot filename.
        plot_type (str): Type of scatter plot ('scatter' or 'kde').
    """
    df1 = df[df["cpd_id"] == cpd1].set_index("neighbour_id")
    df2 = df[df["cpd_id"] == cpd2].set_index("neighbour_id")
    common = df1.index.intersection(df2.index)

    if common.empty:
        print("No common neighbours found between the two compounds.")
        return

    x = 1 - df1.loc[common, "distance"]
    y = 1 - df2.loc[common, "distance"]

    plt.figure(figsize=(6, 6))

    if plot_type == "kde":
        # Perform KDE
        xy = np.vstack([x.values, y.values])
        z = gaussian_kde(xy)(xy)

        # Sort points by density
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = x.values[idx], y.values[idx], z[idx]

        plt.scatter(x_sorted, y_sorted, c=z_sorted, cmap="viridis", s=60, edgecolor="")
        cb = plt.colorbar()
        cb.set_label("Density")

    else:
        plt.scatter(x, y, s=60)
        for neighbour in common:
            plt.text(x[neighbour], y[neighbour], neighbour, fontsize=8)

    plt.xlabel(f"{cpd1} (1 - distance)")
    plt.ylabel(f"{cpd2} (1 - distance)")
    plt.title("Neighbour similarity comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_scatter_comparison.png", dpi=300)
    plt.close()
    print(f"Scatter plot saved to {output_prefix}_scatter_comparison.png")


def plot_top_n_bar(df: pd.DataFrame, cpd1: str, cpd2: str, top_n: int, output_prefix: str) -> None:
    """
    Plot top-N nearest neighbours for each compound as bar charts.

    Parameters:
        df (pd.DataFrame): Nearest neighbour dataframe.
        cpd1 (str): First compound ID.
        cpd2 (str): Second compound ID.
        top_n (int): Number of top matches to plot.
        output_prefix (str): Prefix for saved plot filename.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, cpd in enumerate([cpd1, cpd2]):
        subset = df[df["cpd_id"] == cpd].copy()
        subset["similarity"] = 1 - subset["distance"]
        top_hits = subset.nlargest(top_n, "similarity")

        ax.bar(
            [f"{cpd}\n{n}" for n in top_hits["neighbour_id"]],
            top_hits["similarity"],
            label=cpd,
            alpha=0.7
        )

    ax.set_ylabel("1 - distance (Similarity)")
    ax.set_title(f"Top {top_n} Nearest Neighbours")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_topN_bar.png", dpi=300)
    plt.close()
    print(f"Bar chart saved to {output_prefix}_topN_bar.png")


def main():
    parser = argparse.ArgumentParser(description="Compare nearest neighbours for two compounds.")
    parser.add_argument("--input", required=True, help="Path to nearest neighbour .tsv file")
    parser.add_argument("--compound1", required=True, help="First compound ID")
    parser.add_argument("--compound2", required=True, help="Second compound ID")
    parser.add_argument("--top_n", type=int, default=5, help="Top N neighbours to include in bar plot")
    parser.add_argument("--output_prefix", default="comparison", help="Prefix for output plot files")
    parser.add_argument("--plot_type", choices=["scatter", "kde"], default="kde", help="Scatter plot style: scatter or kde")
    args = parser.parse_args()

    df = load_nn_file(args.input)
    plot_comparison_scatter(df, args.compound1, args.compound2, args.output_prefix, plot_type=args.plot_type)
    plot_top_n_bar(df, args.compound1, args.compound2, args.top_n, args.output_prefix)


if __name__ == "__main__":
    main()
