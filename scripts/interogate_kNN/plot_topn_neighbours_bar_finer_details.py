#!/usr/bin/env python3
"""
Horizontal bar chart of top-N nearest neighbours for a single compound.

Plots raw distances (lower = more similar) for top N neighbours.

Usage:
    python plot_topn_neighbours_bar.py --input nn.tsv --compound DDD02955130 --top_n 10
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_top_n_bar(df: pd.DataFrame, cpd: str, top_n: int, output_prefix: str) -> None:
    """
    Plot top-N nearest neighbours for a single compound as a horizontal bar chart.

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

    top_hits = subset.nsmallest(top_n, "distance")  # lower distance = better
    top_hits = top_hits.sort_values("distance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top_hits["neighbour_id"], top_hits["distance"], color="steelblue")

    # Label each bar with the raw distance value
    for bar, value in zip(bars, top_hits["distance"]):
        ax.text(value + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va='center', ha='left', fontsize=8)

    ax.set_xlabel("Distance (lower is more similar)")
    ax.set_title(f"Top {top_n} Nearest Neighbours for {cpd}")
    ax.invert_yaxis()  # most similar at the top
    plt.tight_layout()
    output_file = f"{output_prefix}_{cpd}_top{top_n}_bar.pdf"
    plt.savefig(output_file, dpi=1200)
    plt.close()
    print(f"Top-{top_n} horizontal bar chart saved to {output_file}")


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
