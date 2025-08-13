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
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import io
import csv

def read_table_auto(file_path):
    """
    Read a delimited text table with automatic delimiter detection.

    Parameters
    ----------
    file_path : str
        Path to a CSV/TSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cleaned column names.

    Notes
    -----
    - Uses pandas' engine-based inference (sep=None) first.
    - Falls back from tab to comma if only one column is produced.
    - Strips BOMs and surrounding whitespace on column names.
    """
    df = pd.read_csv(filepath_or_buffer=file_path, sep=None, engine="python")
    if df.shape[1] == 1 and "," in df.columns[0]:
        # Header got swallowed as one column → force comma
        df = pd.read_csv(filepath_or_buffer=file_path, sep=",")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def ensure_cpd_id(df):
    """
    Ensure a 'cpd_id' column exists by harmonising common synonyms.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame which should contain compound identifiers.

    Returns
    -------
    pandas.DataFrame
        DataFrame guaranteed to have a 'cpd_id' column (string, stripped).

    Raises
    ------
    ValueError
        If no suitable identifier column is found.
    """
    rename_map = {
        "Compound": "cpd_id",
        "compound": "cpd_id",
        "compound_id": "cpd_id",
        "Compound ID": "cpd_id",
        "cpdID": "cpd_id",
        "name": "cpd_id",
    }
    df = df.rename(columns=rename_map)
    if "cpd_id" not in df.columns:
        raise ValueError("Could not find a 'cpd_id' column or a known synonym in compound metadata.")
    df["cpd_id"] = df["cpd_id"].astype(str).str.strip()
    return df



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
        compound_meta = read_table_auto(file_path=args.compound_metadata)
        print(f"[DEBUG] Compound metadata columns before harmonise: {list(compound_meta.columns)}")

        # Keep your existing special-case rename, then harmonise identifiers
        if "publish own other" in compound_meta.columns:
            compound_meta = compound_meta.rename(columns={"publish own other": "published_other"})

        compound_meta = ensure_cpd_id(df=compound_meta)

        # Avoid duplicate lowercase/uppercase 'Library' clashes
        if "Library" in df.columns and "library" in compound_meta.columns:
            compound_meta = compound_meta.drop(columns=["library"])

        df = pd.merge(left=df, right=compound_meta, on="cpd_id", how="left")
        compound_columns = [c for c in compound_meta.columns if c not in ["cpd_id", "library"]]
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
        if getattr(args, "clustering_method", "kmeans") == "hierarchical":
            clustering = AgglomerativeClustering(n_clusters=args.num_clusters, linkage="ward")
            df["Cluster"] = clustering.fit_predict(latent_umap)
        else:
            kmeans = KMeans(n_clusters=args.num_clusters, random_state=42)
            df["Cluster"] = kmeans.fit_predict(latent_umap)
    else:
        df["Cluster"] = np.nan


    if args.highlight_list:
        highlight_set = {c.upper() for c in args.highlight_list}
        name_cols = [c for c in ["cpd_id", "Compound", "compound_name"] if c in df.columns]
        def _row_is_highlighted(row):
            for col in name_cols:
                if str(row[col]).upper() in highlight_set:
                    return True
            return False
        df["is_highlighted"] = df.apply(func=_row_is_highlighted, axis=1)
    else:
        df["is_highlighted"] = df["cpd_id"].astype(str).str.upper().str.startswith(args.highlight_prefix.upper()) if args.highlight_prefix else False


    if "Library" not in df.columns and "library" in df.columns:
        df["Library"] = df["library"]
    df["is_library_mcp"] = df["Library"].astype(str).str.upper().str.contains("MCP")
    print(f"[DEBUG] MCP in Library (diamond shape): {df['is_library_mcp'].sum()}/{len(df)} entries")

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

        base_hover = ["cpd_id", "cpd_type", "Library"]
        hover_cols = [col for col in base_hover + compound_columns if col in df.columns]

        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color=colour_col if colour_col in df.columns else None,
            hover_data=hover_cols,
            title=f"CLIPn UMAP (Interactive, coloured by {label})",
            template="plotly_white"
        )

        fig.update_traces(
            marker=dict(
                size=df["is_highlighted"].apply(lambda x: 14 if x else 6),
                symbol=df.apply(
                    lambda row: "star" if row["is_highlighted"] else ("diamond" if row["is_library_mcp"] else "circle"),
                    axis=1
                ),
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
    parser.add_argument("--umap_metric", type=str, default="cosine", help="UMAP: distance metric")
    parser.add_argument("--num_clusters", type=int, default=None, help="Optional: number of KMeans clusters")
    parser.add_argument("--clustering_method", default="hierarchical", choices=["kmeans", "hierarchical"], help="Clustering method for UMAP ('kmeans' or 'hierarchical').")
    parser.add_argument("--colour_by", nargs="*", default=None, help="List of metadata columns to colour UMAP by")
    parser.add_argument("--add_labels", action="store_true", help="Add `cpd_id` text labels to interactive UMAP")
    parser.add_argument("--highlight_prefix", type=str, default="MCP", help="Highlight compounds with this prefix")
    parser.add_argument("--highlight_list",
                        nargs="+",
                        default=["MCP09", "MCP05",
                                                        'DDD02387619', 'DDD02454019',  
                                                        'DDD02591200', 'DDD02591362', 'DDD02941115', 
                                                        'DDD02941193', 'DDD02947912', 'DDD02947919', 'DDD02948915', 
                                                        'DDD02948916', 'DDD02948926', 'DDD02952619', 'DDD02952620', 
                                                        'DDD02955130'],
                        help="List of specific compound IDs to highlight regardless of prefix"
                    )

    parser.add_argument("--compound_metadata", type=str, default=None, help="Optional file with compound annotations to merge on `cpd_id`")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_umap_analysis(args.input, args.output_dir, args)
