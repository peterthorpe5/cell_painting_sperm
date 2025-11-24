#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D visualisation of compound embeddings and MOA centroids.

Supports:
- PCA projection
- UMAP projection
- OR both (via --projection both)

Plots:
1. All compounds in embedding space
2. MOA centroids overlaid
3. Optional colouring by metadata column (e.g., cpd_type, Library)

Inputs are compatible with:
- CLIPn latent embeddings
- CellProfiler filtered feature embeddings

Outputs:
- <out_prefix>_pca.pdf
- <out_prefix>_umap.pdf
- Optional interactive HTML versions

All outputs are tab-separated.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analysis_pipeline.embedding_utils import (
    configure_logging,
    load_tsv_safely,
    detect_metadata_columns,
    numeric_feature_columns,
    l2_normalise,
    project_pca,
    project_umap,
)


# --------------------------------------------------------------------------- #
# Plotting utilities
# --------------------------------------------------------------------------- #

def _make_colour_map(categories: List[str]) -> dict:
    """
    Create a categorical colour map.

    Parameters
    ----------
    categories : list of str
        Distinct values of a metadata column.

    Returns
    -------
    dict
        Mapping from category to RGB colour.
    """
    import matplotlib.cm as cm
    import numpy as np

    cmap = cm.get_cmap("tab20", len(categories))
    return {cat: cmap(i) for i, cat in enumerate(categories)}


def _plot_2d(
    *,
    coords: np.ndarray,
    centroids: np.ndarray,
    centroid_labels: List[str],
    df: pd.DataFrame,
    id_col: str,
    feature_cols: List[str],
    colour_by: Optional[str],
    out_path: Path,
    title: str,
) -> None:
    """
    Produce a single static 2D scatter plot with centroids overlaid.

    Parameters
    ----------
    coords : np.ndarray
        2D coordinates for all compounds.
    centroids : np.ndarray
        2D coordinates for centroids.
    centroid_labels : list of str
        MOA labels per centroid.
    df : pd.DataFrame
        Original embedding table.
    id_col : str
        Identifier column.
    feature_cols : list of str
        Embedding columns.
    colour_by : str or None
        Column used to colour compounds (optional).
    out_path : Path
        Path to write the PDF.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if colour_by and colour_by in df.columns:
        categories = sorted(df[colour_by].astype(str).unique())
        cmap = _make_colour_map(categories)
        for cat in categories:
            mask = df[colour_by].astype(str) == cat
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=15,
                label=str(cat),
                colour=cmap[cat],
                alpha=0.7,
            )
        ax.legend(title=colour_by, fontsize=7)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.7, colour="grey")

    # Centroids
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=120,
        colour="red",
        marker="X",
        label="centroid",
        edgecolour="black",
    )

    for i, lab in enumerate(centroid_labels):
        ax.text(
            centroids[i, 0],
            centroids[i, 1],
            lab,
            fontsize=6,
            ha="left",
            va="bottom",
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    """Main entry point for 2D projection and plotting."""

    parser = argparse.ArgumentParser(
        description="2D visualisation of MOA centroids and compound embeddings."
    )

    parser.add_argument("--embeddings_tsv", type=str, required=True,
                        help="Embeddings file (TSV).")
    parser.add_argument("--anchors_tsv", type=str, required=True,
                        help="Pseudo-anchor assignments TSV.")
    parser.add_argument("--out_prefix", type=str, required=True,
                        help="Prefix for output PDF/HTML files.")

    parser.add_argument("--id_col", type=str, default="cpd_id")
    parser.add_argument("--moa_col", type=str, default="pseudo_moa")
    parser.add_argument("--metadata_cols", type=str, default="")

    parser.add_argument(
        "--projection",
        type=str,
        default="umap",
        choices=["umap", "pca", "both"],
        help="Projection method."
    )

    parser.add_argument(
        "--colour_by",
        type=str,
        default=None,
        help="Metadata column to colour compounds by (optional)."
    )

    parser.add_argument("--umap_n_neighbours", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()
    configure_logging()

    # ------------------------------------------------------------------ #
    # Load input tables
    # ------------------------------------------------------------------ #
    df = load_tsv_safely(path=args.embeddings_tsv)
    anchors = load_tsv_safely(path=args.anchors_tsv)

    # Metadata + feature detection
    user_metadata = [x for x in args.metadata_cols.split(",") if x] or None
    metadata_cols = detect_metadata_columns(df=df, user_metadata=user_metadata)

    if args.id_col not in df.columns:
        raise KeyError(f"Identifier column '{args.id_col}' not found.")

    feature_cols = numeric_feature_columns(df=df, metadata_cols=metadata_cols)
    if len(feature_cols) == 0:
        raise ValueError("No numeric embedding features detected.")

    # ------------------------------------------------------------------ #
    # Prepare embeddings
    # ------------------------------------------------------------------ #
    X = df[feature_cols].to_numpy().astype(float)
    X = l2_normalise(X=X)

    ids = df[args.id_col].astype(str).tolist()

    # ------------------------------------------------------------------ #
    # Build centroids from anchors
    # ------------------------------------------------------------------ #
    anchors = anchors.copy()
    anchors[args.id_col] = anchors[args.id_col].astype(str)
    anchors = anchors[anchors[args.id_col].isin(ids)]

    if anchors.shape[0] == 0:
        raise ValueError("No anchors match embedding table.")

    # Compute centroid vectors
    centroid_vecs = []
    centroid_labels = []

    for moa, sub in anchors.groupby(args.moa_col):
        idxs = [ids.index(cid) for cid in sub[args.id_col].tolist()]
        Xm = X[idxs, :]

        if Xm.shape[0] == 1:
            vec = Xm[0]
        else:
            vec = np.median(Xm, axis=0)

        centroid_vecs.append(vec)
        centroid_labels.append(str(moa))

    centroid_vecs = np.vstack(centroid_vecs)

    # ------------------------------------------------------------------ #
    # Projection: PCA
    # ------------------------------------------------------------------ #
    if args.projection in ("pca", "both"):
        coords_pca = project_pca(X=X, n_components=2)
        cent_pca = project_pca(X=centroid_vecs, n_components=2)

        out_pdf = Path(f"{args.out_prefix}_pca.pdf")
        _plot_2d(
            coords=coords_pca,
            centroids=cent_pca,
            centroid_labels=centroid_labels,
            df=df,
            id_col=args.id_col,
            feature_cols=feature_cols,
            colour_by=args.colour_by,
            out_path=out_pdf,
            title="PCA projection of embeddings and MOA centroids",
        )

    # ------------------------------------------------------------------ #
    # Projection: UMAP
    # ------------------------------------------------------------------ #
    if args.projection in ("umap", "both"):
        coords_umap = project_umap(
            X=X,
            n_components=2,
            n_neighbours=args.umap_n_neighbours,
            min_dist=args.umap_min_dist,
            random_state=args.random_seed,
        )

        cent_umap = project_umap(
            X=centroid_vecs,
            n_components=2,
            n_neighbours=args.umap_n_neighbours,
            min_dist=args.umap_min_dist,
            random_state=args.random_seed,
        )

        out_pdf = Path(f"{args.out_prefix}_umap.pdf")
        _plot_2d(
            coords=coords_umap,
            centroids=cent_umap,
            centroid_labels=centroid_labels,
            df=df,
            id_col=args.id_col,
            feature_cols=feature_cols,
            colour_by=args.colour_by,
            out_path=out_pdf,
            title="UMAP projection of embeddings and MOA centroids",
        )

    # ------------------------------------------------------------------ #
    # Optional HTML interactive (Plotly)
    # ------------------------------------------------------------------ #
    try:
        import plotly.express as px
        import plotly.io as pio

        if args.projection in ("pca", "both"):
            df_pca = pd.DataFrame({
                "x": coords_pca[:, 0],
                "y": coords_pca[:, 1],
                args.id_col: ids,
                args.colour_by: df[args.colour_by] if args.colour_by in df.columns else None,
            })
            fig_pca = px.scatter(df_pca, x="x", y="y", colour=args.colour_by)
            pio.write_html(fig_pca, file=f"{args.out_prefix}_pca.html")

        if args.projection in ("umap", "both"):
            df_umap = pd.DataFrame({
                "x": coords_umap[:, 0],
                "y": coords_umap[:, 1],
                args.id_col: ids,
                args.colour_by: df[args.colour_by] if args.colour_by in df.columns else None,
            })
            fig_umap = px.scatter(df_umap, x="x", y="y", colour=args.colour_by)
            pio.write_html(fig_umap, file=f"{args.out_prefix}_umap.html")

    except Exception:
        # Missing Plotly: skip HTML silently
        pass


if __name__ == "__main__":
    main()
