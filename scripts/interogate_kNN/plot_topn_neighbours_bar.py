#!/usr/bin/env python3
"""
Bar chart of top-N nearest neighbours for two compounds.

Plots (1 - distance) for top N nearest neighbours per compound.

Usage:
    python plot_topn_neighbours_bar.py --input nn.tsv --compound1 C1 --compound2 C2 --top_n 10


"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse


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

    for cpd in [cpd1, cpd2]:
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
    plt.savefig(f"{output_prefix}_top{top_n}_bar.png", dpi=300)
    plt.close()
    print(f"Top-{top_n} bar chart saved to {output_prefix}_top{top_n}_bar.png")


def main():
    parser = argparse.ArgumentParser(description="Plot top-N neighbours for two compounds.")
    parser.add_argument("--input", required=True, help="Path to nearest neighbour TSV file")
    parser.add_argument("--compound1", required=True, help="First compound ID")
    parser.add_argument("--compound2", required=True, help="Second compound ID")
    parser.add_argument("--top_n", type=int, default=10, help="Top N neighbours to plot")
    parser.add_argument("--output_prefix", default="bar_output", help="Prefix for saved output file")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    plot_top_n_bar(df, args.compound1, args.compound2, args.top_n, args.output_prefix)


if __name__ == "__main__":
    main()
