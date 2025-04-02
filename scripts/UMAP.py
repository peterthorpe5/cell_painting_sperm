#!/usr/bin/env python3
# coding: utf-8

"""
Flexible UMAP Projection for CLIPn Latent Space

This script:
- Loads a CLIPn latent space output file with metadata (TSV format).
- Projects data to 2D using UMAP with configurable parameters.
- Optionally applies KMeans clustering (if requested).
- Merges in additional metadata (e.g. compound functions) if provided.
- Saves UMAP coordinates and both static and interactive plots.
- Supports colouring by multiple metadata fields.

Usage:
------
    python clipn_umap_plot.py \
        --input latent.tsv \
        --output_dir umap_output \
        --umap_n_neighbors 15 \
        --umap_min_dist 0.1 \
        --umap_metric euclidean \
        --colour_by Library cpd_type function \
        --add_labels \
        --compound_metadata compound_function.tsv

Output:
-------
    - clipn_umap_coordinates_[metric]_n[neigh]_d[dist].tsv
    - one UMAP plot PDF per colour_by column
    - one interactive HTML plot per colour_by column

"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
import plotly.express as px
from sklearn.cluster import KMeans


def run_umap_analysis(input_path, output_dir, args):
    """Run UMAP and optionally KMeans clustering on a latent feature DataFrame.

    Parameters
    ----------
    input_path : str
        Path to the input TSV file with metadata and latent features.

    output_dir : str
        Directory where output files will be saved.

    args : argparse.Namespace
        Command-line arguments for UMAP and clustering parameters.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path, sep='\t')

    if args.compound_metadata:
        compound_meta = pd.read_csv(args.compound_metadata, sep='\t')
        df = pd.merge(df, compound_meta, on="cpd_id", how="left")

    metadata_cols = ["cpd_id", "cpd_type", "Library", "Dataset", "Sample"]
    latent_features = df.drop(columns=metadata_cols, errors="ignore")

    umap_model = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        n_components=2,
        random_state=42
    )
    latent_umap = umap_model.fit_transform(latent_features)

    df["UMAP1"] = latent_umap[:, 0]
    df["UMAP2"] = latent_umap[:, 1]

    if args.num_clusters:
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=42)
        df["Cluster"] = kmeans.fit_predict(latent_umap)
    else:
        df["Cluster"] = "NA"

    coord_filename = f"clipn_umap_coordinates_{args.umap_metric}_n{args.umap_n_neighbors}_d{args.umap_min_dist}.tsv"
    coords_file = os.path.join(output_dir, coord_filename)
    df.to_csv(coords_file, sep='\t', index=False)

    # Loop over all colour_by options
    colour_fields = args.colour_by if args.colour_by else [None]

    for colour_col in colour_fields:
        label = colour_col if colour_col else "uncoloured"

        # Static plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            df["UMAP1"],
            df["UMAP2"],
            s=5,
            alpha=0.6,
            c=df[colour_col] if colour_col else "grey",
            cmap="tab10"
        )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"CLIPn UMAP coloured by {label}")
        if colour_col:
            plt.colorbar(scatter, label=label)
        plt.tight_layout()

        plot_filename = f"clipn_umap_plot_{label}_{args.umap_metric}_n{args.umap_n_neighbors}_d{args.umap_min_dist}.pdf"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # Interactive Plot
        hover_cols = ["cpd_id", "cpd_type", "Library"]
        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color=colour_col if colour_col else None,
            hover_data=hover_cols,
            title=f"CLIPn UMAP (Interactive, coloured by {label})",
            template="simple_white"
        )
        if args.add_labels:
            fig.update_traces(text=df["cpd_id"], textposition="top center")

        html_filename = f"clipn_umap_plot_{label}_{args.umap_metric}_n{args.umap_n_neighbors}_d{args.umap_min_dist}.html"
        fig.write_html(os.path.join(output_dir, html_filename))

        print(f"Saved UMAP plot: {plot_path}")
        print(f"Saved interactive UMAP: {html_filename}")

    print(f"UMAP coordinates saved to: {coords_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flexible UMAP Projection for CLIPn Latent Data")
    parser.add_argument("--input", required=True, help="Path to input TSV with latent + metadata")
    parser.add_argument("--output_dir", required=True, help="Directory to save UMAP plot and coordinates")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP: number of neighbours")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP: minimum distance")
    parser.add_argument("--umap_metric", type=str, default="euclidean", help="UMAP: distance metric")
    parser.add_argument("--num_clusters", type=int, default=None, help="Optional: number of KMeans clusters")
    parser.add_argument("--colour_by", nargs="*", default=None, help="List of metadata columns to colour UMAP by")
    parser.add_argument("--add_labels", action="store_true", help="Add `cpd_id` text labels to interactive UMAP")
    parser.add_argument("--compound_metadata", type=str, default=None, help="Optional file with compound annotations to merge on `cpd_id`")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_umap_analysis(args.input, args.output_dir, args)
