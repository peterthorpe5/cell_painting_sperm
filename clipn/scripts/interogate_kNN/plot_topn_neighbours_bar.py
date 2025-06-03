#!/usr/bin/env python3
"""
Bar chart of top-N nearest neighbours for a single compound.

Plots (1 - distance) similarity values for top N neighbours.

Usage:
    python plot_topn_neighbours_bar.py --input nn.tsv --compound DDD02955130 --top_n 10
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_top_n_bar(df: pd.DataFrame, cpd: str, top_n: int, output_prefix: str) -> None:
    """
    Plot top-N nearest neighbours for a single compound as a bar chart.

    Parameters:
        df (pd.DataFrame): Nearest neighbour dataframe.
        cpd (str): Compound ID.
        top_n (int): Number of top matches to plot.
        output_prefix (str): Prefix for saved plot filename.
    """
    subset = df[df["cpd_id"] == cpd].copy()
    if subset.empty:
        print(f"No matches found for compound '{cpd}' in the input file.")
        return

    subset["similarity"] = 1 - subset["distance"]
    top_hits = subset.nlargest(top_n, "similarity")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        top_hits["neighbour_id"],
        top_hits["similarity"],
        color="steelblue"
    )

    ax.set_ylabel("1 - distance (Similarity)")
    ax.set_title(f"Top {top_n} Nearest Neighbours for {cpd}")
    ax.set_xticklabels(top_hits["neighbour_id"], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_{cpd}_top{top_n}_bar.pdf", dpi=1200)
    plt.close()
    print(f"Top-{top_n} bar chart saved to {output_prefix}_{cpd}_top{top_n}_bar.pdf")


def main():
    parser = argparse.ArgumentParser(description="Plot top-N neighbours for a single compound.")
    parser.add_argument("--input", required=True, help="Path to nearest neighbour TSV file")
    parser.add_argument("--compound", required=True, help="Compound ID")
    parser.add_argument("--top_n", type=int, default=10, help="Top N neighbours to plot")
    parser.add_argument("--output_prefix", default="bar_output", help="Prefix for saved output file")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    plot_top_n_bar(df, args.compound, args.top_n, args.output_prefix)


if __name__ == "__main__":
    main()
