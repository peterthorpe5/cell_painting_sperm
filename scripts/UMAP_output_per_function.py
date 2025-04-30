#!/usr/bin/env python3
# coding: utf-8

"""
Flexible UMAP Projection for CLIPn Latent Space

This script:
- Loads a CLIPn latent space output file with metadata (TSV format).
- Projects data to 2D using UMAP with configurable parameters.
- Optionally applies KMeans clustering (if requested).
- Merges in additional metadata (e.g. compound functions) if provided.
- Supports colouring by multiple metadata fields.
- Adds all merged compound annotations to hover tooltips.
- Highlights compounds of interest using stars.
- Uses diamond markers for entries where 'Library' contains MCP.
- Saves a cluster summary table by `cpd_type` and compound annotations.
- Saves coordinates, static and interactive UMAP plots.

Usage:
------
    python clipn_umap_plot.py \
        --input latent.tsv \
        --output_dir umap_output \
        --umap_n_neighbors 15 \
        --umap_min_dist 0.1 \
        --umap_metric euclidean \
        --colour_by Library cpd_type function \
        --highlight_prefix MCP \
        --add_labels \
        --compound_metadata compound_function.tsv

Output:
-------
    - Subfolder per `colour_by` with:
        - clipn_umap_coordinates.tsv
        - clipn_umap_plot.pdf
        - clipn_umap_plot.html
        - cluster_summary_by_cpd_type.tsv (if clustered)
        - cluster_summary_by_<column>.tsv (if compound metadata provided)

"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import umap.umap_ as umap
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

def summarise_clusters(df, outdir, compound_columns):
    """
    Summarise cluster composition by cpd_type, marker symbol, and additional metadata fields.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing UMAP coordinates and cluster labels.
    outdir : str
        Directory to save cluster summary TSV files.
    compound_columns : list
        List of compound annotation columns to summarise per cluster.
    """
    if "Cluster" not in df.columns or df["Cluster"].nunique() <= 1:
        return

    if "cpd_type" in df.columns:
        cpd_summary = df.groupby("Cluster")["cpd_type"].value_counts().unstack(fill_value=0)
        cpd_summary.to_csv(os.path.join(outdir, "cluster_summary_by_cpd_type.tsv"), sep="\t")

    if "marker_symbol" in df.columns:
        marker_summary = df.groupby("Cluster")["marker_symbol"].value_counts().unstack(fill_value=0)
        marker_summary.to_csv(os.path.join(outdir, "cluster_summary_by_marker_symbol.tsv"), sep="\t")

    for col in compound_columns:
        if col in df.columns:
            func_summary = df.groupby("Cluster")[col].value_counts().unstack(fill_value=0)
            func_summary.to_csv(os.path.join(outdir, f"cluster_summary_by_{col}.tsv"), sep="\t")


def run_umap_analysis(input_path, output_dir, args):
    """
    Perform UMAP projection and optional clustering, save results and plots.

    Parameters
    ----------
    input_path : str
        Path to input latent space TSV file.
    output_dir : str
        Directory where results will be saved.
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path, sep='\t')
    compound_columns = []

    print(f"[DEBUG] Input data shape: {df.shape}")
    print(f"[DEBUG] Input columns: {list(df.columns)}")

    if args.compound_metadata:
        compound_meta = pd.read_csv(args.compound_metadata, sep='\t')
        print(f"[DEBUG] Compound metadata columns before rename: {list(compound_meta.columns)}")

        if "Library" in df.columns and "library" in compound_meta.columns:
            compound_meta = compound_meta.drop(columns=["library"])

        if "publish own other" in compound_meta.columns:
            compound_meta = compound_meta.rename(columns={"publish own other": "published_other"})

        df = pd.merge(df, compound_meta, on="cpd_id", how="left")
        compound_columns = [col for col in compound_meta.columns if col not in ["cpd_id", "library"]]

        print(f"[DEBUG] Compound columns merged: {compound_columns}")
        print(f"[DEBUG] Data shape after merge: {df.shape}")

    latent_features = df[[col for col in df.columns if col.isdigit()]].copy()
    print(f"[INFO] Using {latent_features.shape[1]} numeric columns for UMAP:")
    print(latent_features.columns.tolist())

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

    def is_highlighted(row):
        cpd_match = str(row["cpd_id"]).upper().startswith(args.highlight_prefix.upper()) if args.highlight_prefix else False
        lib_match = "MCP" in str(row.get("Library", "")).upper()
        return cpd_match or lib_match

    df["is_highlighted"] = df.apply(is_highlighted, axis=1)
    df["is_library_mcp"] = df["Library"].astype(str).str.upper().str.contains("MCP")

    df["marker_symbol"] = df.apply(
        lambda row: "star" if row["is_highlighted"]
        else ("diamond" if row["is_library_mcp"] else "circle"),
        axis=1
    )

    print(f"[DEBUG] MCP in Library (diamond shape): {df['is_library_mcp'].sum()}/{len(df)} entries")
    print(f"[DEBUG] Highlighted compounds (star or diamond): {(df['marker_symbol'] != 'circle').sum()}/{len(df)}")

    colour_fields = args.colour_by if args.colour_by else [None]

    for colour_col in colour_fields:
        label = colour_col if colour_col else "uncoloured"
        label_folder = os.path.join(output_dir, label)
        os.makedirs(label_folder, exist_ok=True)

        coord_filename = f"clipn_umap_coordinates_{args.umap_metric}_n{args.umap_n_neighbors}_d{args.umap_min_dist}.tsv"
        coords_file = os.path.join(label_folder, coord_filename)
        df.to_csv(coords_file, sep='\t', index=False)

        plt.figure(figsize=(12, 8))
        if colour_col and colour_col in df.columns:
            unique_vals = df[colour_col].unique()
            colour_map = {val: idx for idx, val in enumerate(unique_vals)}
            colours = df[colour_col].map(colour_map)
            cmap = colormaps['tab10']
        else:
            colours = "grey"
            cmap = None

        scatter = plt.scatter(
            df["UMAP1"], df["UMAP2"], s=5, alpha=0.6, c=colours, cmap=cmap
        )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"CLIPn UMAP coloured by {label}")
        if colour_col:
            plt.colorbar(scatter, label=label)
        plt.tight_layout()

        plot_filename = f"clipn_umap_plot_{label}_{args.umap_metric}_n{args.umap_n_neighbors}_d{args.umap_min_dist}.pdf"
        plot_path = os.path.join(label_folder, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()

        hover_keys = [
            "cpd_id", "cpd_type", "Library", "Dataset",
            colour_col, "name", "published_phenotypes",
            "published_target", "published_other"
        ]
        hover_cols = [col for col in hover_keys if col in df.columns]


        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color=colour_col if colour_col in df.columns else None,
            symbol="marker_symbol",
            hover_data=hover_cols,
            title=f"CLIPn UMAP (Interactive, coloured by {label})",
            template="plotly_white"
        )

        fig.update_traces(
            marker=dict(
                size=df["marker_symbol"].apply(lambda x: 14 if x in ["star", "diamond"] else 6),
                line=dict(
                    width=df["is_highlighted"].apply(lambda x: 2 if x else 0),
                    color="black"
                )
            )
        )

        if args.add_labels:
            fig.update_traces(text=df["cpd_id"], textposition="top center")

        html_filename = f"clipn_umap_plot_{label}_{args.umap_metric}_n{args.umap_n_neighbors}_d{args.umap_min_dist}.html"
        fig.write_html(os.path.join(label_folder, html_filename))

        summarise_clusters(df, label_folder, compound_columns)

        print(f"Saved UMAP plot: {plot_path}")
        print(f"Saved interactive UMAP: {html_filename}")
        print(f"Saved coordinates: {coords_file}")




def parse_args():
    """
    Parse command-line arguments for UMAP plotting script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments object.
    """
    parser = argparse.ArgumentParser(description="Flexible UMAP Projection for CLIPn Latent Data")
    parser.add_argument("--input", required=True, help="Path to input TSV with latent + metadata")
    parser.add_argument("--output_dir", required=True, help="Directory to save UMAP plot and coordinates")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP: number of neighbours")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP: minimum distance")
    parser.add_argument("--umap_metric", type=str, default="euclidean", help="UMAP: distance metric")
    parser.add_argument("--num_clusters", type=int, default=None, help="Optional: number of KMeans clusters")
    parser.add_argument("--colour_by", nargs="*", default=None, help="List of metadata columns to colour UMAP by")
    parser.add_argument("--add_labels", action="store_true", help="Add `cpd_id` text labels to interactive UMAP")
    parser.add_argument("--highlight_prefix", type=str, default="MCP", help="Highlight compounds with this prefix")
    parser.add_argument("--compound_metadata", type=str, default=None, help="Optional file with compound annotations to merge on `cpd_id`")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_umap_analysis(args.input, args.output_dir, args)
