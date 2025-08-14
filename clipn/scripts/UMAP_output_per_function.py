#!/usr/bin/env python3
# coding: utf-8

"""
Flexible UMAP Projection for CLIPn Latent Space

This script:
- Loads a CLIPn latent space output file with metadata (TSV format).
- Projects data to 2D using UMAP with configurable parameters.
- Optionally applies clustering (KMeans or hierarchical).
- Merges additional compound metadata (CSV/TSV auto-detected).
- Supports colouring by multiple metadata fields.
- Highlights compounds by prefix or explicit list.
- Uses diamond markers for entries where 'Library' contains 'MCP'.
- Saves a cluster summary table and coordinates, plus static (PDF) and interactive (HTML) plots.

All outputs are tab-separated (TSV).
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# Robust colormap import (supports older Matplotlib)
try:
    from matplotlib import colormaps as _mpl_cmaps

    def _get_cmap(name: str):
        return _mpl_cmaps[name]
except Exception:  # pragma: no cover
    import matplotlib.cm as cm

    def _get_cmap(name: str):
        return cm.get_cmap(name)

import plotly.express as px
import umap.umap_ as umap
from sklearn.cluster import AgglomerativeClustering, KMeans


def read_table_auto(file_path: str) -> pd.DataFrame:
    """
    Read a delimited text table with automatic delimiter detection.

    Parameters
    ----------
    file_path : str
        Path to a CSV/TSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cleaned column names (BOM/whitespace stripped) and
        correct delimiter even for comma-delimited CSVs.
    """
    df = pd.read_csv(filepath_or_buffer=file_path, sep=None, engine="python")
    if df.shape[1] == 1 and "," in df.columns[0]:
        # Header swallowed as a single column -> enforce comma
        df = pd.read_csv(filepath_or_buffer=file_path, sep=",")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def ensure_cpd_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a single string 'cpd_id' column exists.
    - Deduplicates columns.
    - If multiple cpd_id-like columns exist, coalesce to first non-null.
    - Cast to string and strip whitespace.
    """
    df = df.copy()

    # 1) Drop duplicate column names globally (keep first)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # 2) Find candidate columns that could be the ID
    candidates = [c for c in df.columns
                  if c.lower() in {"cpd_id", "compound_id", "compound", "cpd"}]

    if not candidates:
        raise ValueError("No 'cpd_id'-like column found in metadata (looked for cpd_id/compound_id/compound/cpd).")

    # 3) If more than one candidate, coalesce left→right (first non-null)
    if len(candidates) > 1:
        s = df[candidates].bfill(axis=1).iloc[:, 0]
    else:
        s = df[candidates[0]]
        # If duplicate name slipped through and we got a DataFrame, take first col
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]

    # 4) Finalise as clean string
    df["cpd_id"] = s.astype(str).str.strip()
    return df


def summarise_clusters(df: pd.DataFrame, outdir: str, compound_columns: List[str]) -> None:
    """
    Summarise cluster composition by cpd_type, marker symbol, and extra metadata.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing UMAP coordinates and cluster labels.
    outdir : str
        Directory to save cluster summary TSV files.
    compound_columns : list of str
        Additional compound annotation columns to summarise per cluster.
    """
    if "Cluster" not in df.columns or df["Cluster"].nunique(dropna=True) <= 1:
        return

    if "cpd_type" in df.columns:
        cpd_summary = df.groupby("Cluster", dropna=False)["cpd_type"].value_counts().unstack(fill_value=0)
        cpd_summary.to_csv(os.path.join(outdir, "cluster_summary_by_cpd_type.tsv"), sep="\t")

    # Only produce marker symbol summary if you later add a 'marker_symbol' column to df
    if "marker_symbol" in df.columns:
        marker_summary = df.groupby("Cluster", dropna=False)["marker_symbol"].value_counts().unstack(fill_value=0)
        marker_summary.to_csv(os.path.join(outdir, "cluster_summary_by_marker_symbol.tsv"), sep="\t")

    for col in compound_columns:
        if col in df.columns:
            func_summary = df.groupby("Cluster", dropna=False)[col].value_counts().unstack(fill_value=0)
            func_summary.to_csv(os.path.join(outdir, f"cluster_summary_by_{col}.tsv"), sep="\t")


def run_umap_analysis(input_path: str, output_dir: str, args: argparse.Namespace) -> None:
    """
    Perform UMAP projection and optional clustering, save results and plots.

    Parameters
    ----------
    input_path : str
        Path to input latent space TSV file (with latent dims named '0','1',...).
    output_dir : str
        Directory where results will be saved.
    args : argparse.Namespace
        Parsed command-line arguments controlling UMAP, colouring, and metadata merge.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Main latent+metadata table (TSV from run_clipn)
    df = pd.read_csv(filepath_or_buffer=input_path, sep="\t")
    
    # NORMALISE IDs EARLY
    if "cpd_id" in df.columns:
        df["cpd_id"] = df["cpd_id"].astype(str).str.upper().str.strip()
        
    compound_columns: List[str] = []

    print(f"[DEBUG] Input data shape: {df.shape}")
    print(f"[DEBUG] Input columns: {list(df.columns)}")

    # Merge compound metadata if provided (CSV/TSV auto-detected)
    if args.compound_metadata:
        compound_meta = read_table_auto(file_path=args.compound_metadata)
        print(f"[DEBUG] Compound metadata columns before harmonise: {list(compound_meta.columns)}")

        if "publish own other" in compound_meta.columns:
            compound_meta = compound_meta.rename(columns={"publish own other": "published_other"})

        compound_meta = ensure_cpd_id(df=compound_meta)

        compound_meta["cpd_id"] = compound_meta["cpd_id"].astype(str).str.upper().str.strip()
        compound_meta = compound_meta.drop_duplicates(subset=["cpd_id"])

        # Avoid duplicate lowercase/uppercase 'Library' clashes
        if "Library" in df.columns and "library" in compound_meta.columns:
            compound_meta = compound_meta.drop(columns=["library"])

        if "cpd_id" in df.columns:
            df = pd.merge(left=df, right=compound_meta, on="cpd_id", how="left")
            compound_columns = [c for c in compound_meta.columns if c not in ["cpd_id", "library"]]
            print(f"[DEBUG] Compound columns merged: {compound_columns}")
            print(f"[DEBUG] Data shape after merge: {df.shape}")
        else:
            print("[WARN] 'cpd_id' not present in input table; skipping compound metadata merge.")

    # Detect latent feature columns robustly
    latent_cols = [c for c in df.columns if str(c).isdigit()]
    df = df.drop_duplicates(subset=["cpd_id"] + latent_cols)
    latent_features = df[latent_cols].copy()
    n_lat = latent_features.shape[1]
    if n_lat < 2:
        raise ValueError(
            f"No sufficient latent columns detected for UMAP. "
            f"Found {n_lat}; expected >=2. Columns seen: {latent_cols}"
        )
    print(f"[INFO] Using {n_lat} numeric columns for UMAP:")
    print(latent_features.columns.tolist())

    # Compute UMAP
    umap_model = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        n_components=2,
        random_state=42,
    )
    latent_umap = umap_model.fit_transform(latent_features.to_numpy())
    df["UMAP1"] = latent_umap[:, 0]
    df["UMAP2"] = latent_umap[:, 1]

    # Optional clustering on UMAP space
    if args.num_clusters:
        if getattr(args, "clustering_method", "kmeans") == "hierarchical":
            clustering = AgglomerativeClustering(n_clusters=args.num_clusters, linkage="ward")
            df["Cluster"] = clustering.fit_predict(latent_umap)
        else:
            kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init="auto")
            df["Cluster"] = kmeans.fit_predict(latent_umap)
    else:
        df["Cluster"] = np.nan

    # Highlighting logic
    if args.highlight_list:
        highlight_set = {str(c).upper() for c in args.highlight_list}
        name_cols = [c for c in ["cpd_id", "Compound", "compound_name"] if c in df.columns]
        def _row_is_highlighted(row) -> bool:
            for col in name_cols:
                if str(row[col]).upper() in highlight_set:
                    return True
            return False
        df["is_highlighted"] = df.apply(func=_row_is_highlighted, axis=1) if name_cols else False
    else:
        df["is_highlighted"] = (
            df["cpd_id"].astype(str).str.upper().str.startswith(args.highlight_prefix.upper())
            if "cpd_id" in df.columns and args.highlight_prefix
            else False
        )

    # Library normalisation and MCP marker flag
    if "Library" not in df.columns and "library" in df.columns:
        df["Library"] = df["library"]
    df["is_library_mcp"] = df.get("Library", pd.Series("", index=df.index)).astype(str).str.upper().str.contains("MCP")
    print(f"[DEBUG] MCP in Library (diamond shape): {df['is_library_mcp'].sum()}/{len(df)} entries")

    colour_fields = args.colour_by if args.colour_by else [None]

    for colour_col in colour_fields:
        label = colour_col if colour_col else "uncoloured"
        label_folder = os.path.join(output_dir, label)
        os.makedirs(label_folder, exist_ok=True)

        # Save UMAP coordinates
        coord_filename = f"clipn_umap_coordinates_{args.umap_metric}_n{args.umap_n_neighbors}_d{args.umap_min_dist}.tsv"
        coords_file = os.path.join(label_folder, coord_filename)
        df.to_csv(coords_file, sep="\t", index=False)

        # Static Matplotlib plot
        plt.figure(figsize=(12, 8))
        if colour_col and colour_col in df.columns:
            unique_vals = pd.unique(df[colour_col])
            colour_map = {val: idx for idx, val in enumerate(unique_vals)}
            colours = df[colour_col].map(colour_map)
            cmap = _get_cmap("tab10")
        else:
            colours = "grey"
            cmap = None

        scatter = plt.scatter(df["UMAP1"], df["UMAP2"], s=5, alpha=0.6, c=colours, cmap=cmap)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"CLIPn UMAP coloured by {label}")
        if colour_col and colour_col in df.columns:
            plt.colorbar(scatter, label=label)  # continuous legend; adequate for quick-look
        plt.tight_layout()

        plot_filename = f"clipn_umap_plot_{label}_{args.umap_metric}_n{args.umap_n_neighbors}_d{args.umap_min_dist}.pdf"
        plot_path = os.path.join(label_folder, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # Interactive Plotly plot
        base_hover = ["cpd_id", "cpd_type", "Library"]
        hover_cols = [col for col in base_hover + compound_columns if col in df.columns]

        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color=colour_col if (colour_col and colour_col in df.columns) else None,
            hover_data=hover_cols,
            title=f"CLIPn UMAP (Interactive, coloured by {label})",
            template="plotly_white",
        )

        # Markers: star for highlighted, diamond if Library contains MCP, else circle
        # Plotly accepts sequence values for size/line.width/symbol
        sizes = df["is_highlighted"].apply(lambda x: 14 if x else 6)
        widths = df["is_highlighted"].apply(lambda x: 2 if x else 0)
        symbols = df.apply(
            lambda row: "star" if row["is_highlighted"]
            else ("diamond" if row["is_library_mcp"] else "circle"),
            axis=1,
        )

        fig.update_traces(
            marker=dict(
                size=sizes,
                symbol=symbols,
                line=dict(width=widths, color="black"),
            )
        )

        if args.add_labels and "cpd_id" in df.columns:
            fig.update_traces(text=df["cpd_id"], textposition="top center")

        html_filename = f"clipn_umap_plot_{label}_{args.umap_metric}_n{args.umap_n_neighbors}_d{args.umap_min_dist}.html"
        fig.write_html(os.path.join(label_folder, html_filename))

        summarise_clusters(df, label_folder, compound_columns)

        print(f"Saved UMAP plot: {plot_path}")
        print(f"Saved interactive UMAP: {os.path.join(label_folder, html_filename)}")
        print(f"Saved coordinates: {coords_file}")


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--num_clusters", type=int, default=None, help="Optional: number of clusters")
    parser.add_argument(
        "--clustering_method",
        default="hierarchical",
        choices=["kmeans", "hierarchical"],
        help="Clustering method for UMAP ('kmeans' or 'hierarchical').",
    )
    parser.add_argument("--colour_by", nargs="*", default=None, help="List of metadata columns to colour UMAP by")
    parser.add_argument("--add_labels", action="store_true", help="Add `cpd_id` text labels to interactive UMAP")
    parser.add_argument("--highlight_prefix", type=str, default="MCP", help="Highlight compounds with this prefix")
    parser.add_argument(
        "--highlight_list",
        nargs="+",
        default=[
            "MCP09",
            "MCP05",
            "DDD02387619",
            "DDD02454019",
            "DDD02591200",
            "DDD02591362",
            "DDD02941115",
            "DDD02941193",
            "DDD02947912",
            "DDD02947919",
            "DDD02948915",
            "DDD02948916",
            "DDD02948926",
            "DDD02952619",
            "DDD02952620",
            "DDD02955130",
        ],
        help="List of specific compound IDs to highlight regardless of prefix",
    )
    parser.add_argument(
        "--compound_metadata",
        type=str,
        default=None,
        help="Optional file with compound annotations to merge on `cpd_id`",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_umap_analysis(input_path=args.input, output_dir=args.output_dir, args=args)
