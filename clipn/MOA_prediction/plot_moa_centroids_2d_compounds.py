#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot PCA/UMAP maps for per-compound embeddings and pseudo-MOA anchors.

This script is designed for situations where the input embeddings table is
very large (e.g. per-cell rows) but you only want a plot at the compound level.

Workflow
--------
1) Load embeddings TSV (may be per-cell/per-image/per-well)
2) Detect numeric feature columns
3) Aggregate rows to one vector per compound (median/mean)
4) Load pseudo-anchors TSV (id_col, moa_col)
5) Merge anchors onto per-compound table
6) Plot PCA and/or UMAP for all compounds (one point per compound)
7) Optionally overlay pseudo-MOA centroids in the SAME projection space

All outputs are tab-separated elsewhere; plots are saved as PDF.
UK English spelling is used throughout.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_pipeline.embedding_utils import (
    configure_logging,
    load_tsv_safely,
    detect_metadata_columns,
    numeric_feature_columns,
    aggregate_replicates,
    l2_normalise,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot PCA/UMAP for per-compound embeddings and pseudo-MOA anchors."
    )

    parser.add_argument(
        "--embeddings_tsv",
        type=str,
        required=True,
        help="Input embeddings TSV (can be per-cell; will be aggregated to per-compound).",
    )
    parser.add_argument(
        "--anchors_tsv",
        type=str,
        required=True,
        help="Pseudo-anchors TSV (from make_pseudo_anchors.py).",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        required=True,
        help="Output prefix for plots (e.g. moa_output/moa_map).",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="cpd_id",
        help="Compound identifier column.",
    )
    parser.add_argument(
        "--moa_col",
        type=str,
        default="pseudo_moa",
        help="MOA/anchor label column in anchors file.",
    )
    parser.add_argument(
        "--aggregate_method",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Aggregation method to make per-compound vectors.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="both",
        choices=["pca", "umap", "both"],
        help="Which projection(s) to generate.",
    )

    parser.add_argument(
        "--colour_by",
        type=str,
        default="",
        help="Optional column to colour points by (must exist after merge). "
             "If not provided, uses moa_col where available.",
    )

    parser.add_argument(
        "--plot_centroids",
        action="store_true",
        help="Overlay pseudo-MOA centroids on the same projection.",
    )

    parser.add_argument(
        "--umap_n_neighbours",
        type=int,
        default=15,
        help="UMAP n_neighbours (will be clipped to n_points-1).",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed (used for reproducibility; UMAP may run single-threaded).",
    )
    parser.add_argument(
        "--label_points",
        action="store_true",
        help="Label each compound point with its identifier.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="",
        help="Column to use for labels (default: id_col).",
    )
    parser.add_argument(
        "--label_jitter",
        type=float,
        default=0.0,
        help="Optional jitter applied to labels (not points) to reduce overlap.",
    )


    return parser.parse_args()


def fit_pca_and_transform(
    *,
    X: np.ndarray,
    X_extra: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fit PCA on X and transform X (and optionally X_extra) into the same PCA space.

    Parameters
    ----------
    X : np.ndarray
        Main matrix to fit PCA on.
    X_extra : np.ndarray, optional
        Extra matrix to transform with the same PCA model.

    Returns
    -------
    coords : np.ndarray
        PCA coordinates for X (n_samples, 2).
    extra_coords : np.ndarray or None
        PCA coordinates for X_extra, or None if not provided.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    extra_coords = None
    if X_extra is not None and X_extra.size > 0:
        extra_coords = pca.transform(X_extra)

    return coords, extra_coords


def fit_umap_and_transform_stacked(
    *,
    X: np.ndarray,
    X_extra: Optional[np.ndarray],
    n_neighbours: int,
    min_dist: float,
    random_state: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fit UMAP on stacked data so that X and X_extra are in the same UMAP space.

    For small datasets UMAP can fail; this function raises a ValueError in that case.

    Parameters
    ----------
    X : np.ndarray
        Main data matrix.
    X_extra : np.ndarray, optional
        Extra rows to embed jointly.
    n_neighbours : int
        UMAP n_neighbours (will be clipped to n_total - 1).
    min_dist : float
        UMAP min_dist.
    random_state : int
        Random seed.

    Returns
    -------
    coords : np.ndarray
        UMAP coordinates for X.
    extra_coords : np.ndarray or None
        UMAP coordinates for X_extra, or None if not provided.
    """
    try:
        import umap
    except Exception as exc:
        raise ImportError("UMAP is required but could not be imported.") from exc

    if X_extra is not None and X_extra.size > 0:
        XY = np.vstack([X, X_extra])
        split = X.shape[0]
    else:
        XY = X
        split = X.shape[0]

    n_total = XY.shape[0]
    if n_total < 3:
        raise ValueError(f"UMAP requires at least 3 points; got {n_total}.")

    n_nb = min(int(n_neighbours), n_total - 1)
    if n_nb < 2:
        raise ValueError(f"UMAP requires n_neighbours >= 2; got {n_nb}.")

    model = umap.UMAP(
        n_components=2,
        n_neighbors=n_nb,
        min_dist=float(min_dist),
        random_state=int(random_state),
    )

    XY_coords = model.fit_transform(XY)

    coords = XY_coords[:split, :]
    extra_coords = None
    if split < n_total:
        extra_coords = XY_coords[split:, :]

    return coords, extra_coords


def compute_centroids(
    *,
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute one centroid per label (median of member vectors).

    Parameters
    ----------
    df : pd.DataFrame
        Per-compound table containing label_col.
    id_col : str
        Compound identifier (unused except for clarity).
    label_col : str
        Label to group by (e.g. pseudo_moa).
    feature_cols : list of str
        Feature columns.

    Returns
    -------
    pd.DataFrame
        Centroid table with columns [label_col] + feature_cols.
    """
    _ = id_col  # intentionally unused
    cent_rows = []
    for lab, sub in df.dropna(subset=[label_col]).groupby(label_col, sort=False):
        X = sub[feature_cols].to_numpy(dtype=float)
        vec = np.median(X, axis=0)
        cent_rows.append({label_col: str(lab), **{c: v for c, v in zip(feature_cols, vec)}})
    return pd.DataFrame(cent_rows)


def plot_scatter(
    *,
    coords: np.ndarray,
    labels: pd.Series,
    title: str,
    out_pdf: Path,
    point_ids: Optional[pd.Series] = None,
    label_points: bool = False,
    label_jitter: float = 0.0,
    centroid_coords: Optional[np.ndarray] = None,
    centroid_labels: Optional[pd.Series] = None,
) -> None:
    """
    Plot a simple 2D scatter and save to PDF.

    Parameters
    ----------
    coords : np.ndarray
        2D coordinates for points.
    labels : pd.Series
        Label per point (used for colour legend).
    title : str
        Plot title.
    out_pdf : Path
        Output PDF path.
    point_ids : pd.Series, optional
        Text labels for points (e.g. cpd_id).
    label_points : bool
        Whether to annotate each point with point_ids.
    label_jitter : float
        Jitter applied to label positions to reduce overlap.
    centroid_coords : np.ndarray, optional
        2D coordinates for centroids.
    centroid_labels : pd.Series, optional
        Label per centroid.
    """
    fig, ax = plt.subplots(figsize=(9.5, 7.0))

    labels = labels.astype(str).fillna("NA")
    uniq = sorted(labels.unique().tolist())

    for lab in uniq:
        mask = labels == lab
        ax.scatter(coords[mask.values, 0], coords[mask.values, 1], s=55, alpha=0.9, label=lab)

    # Label each point (compound)
    if label_points and point_ids is not None:
        ids = point_ids.astype(str).tolist()

        # Deterministic small offsets to separate labels when points overlap
        # (spiral-like offsets around the point)
        offsets = [
            (0.0, 0.0),
            (0.01, 0.01),
            (-0.01, 0.01),
            (0.01, -0.01),
            (-0.01, -0.01),
            (0.02, 0.0),
            (-0.02, 0.0),
            (0.0, 0.02),
            (0.0, -0.02),
        ]

        rng = np.random.RandomState(0)
        for i, cid in enumerate(ids):
            x, y = coords[i, 0], coords[i, 1]
            dx, dy = offsets[i % len(offsets)]
            if label_jitter > 0:
                dx += rng.normal(loc=0.0, scale=label_jitter)
                dy += rng.normal(loc=0.0, scale=label_jitter)

            ax.text(
                x + dx,
                y + dy,
                cid,
                fontsize=8,
                ha="left",
                va="bottom",
            )

    # Centroids
    if centroid_coords is not None and centroid_labels is not None and centroid_coords.size > 0:
        cent_labs = centroid_labels.astype(str).tolist()
        ax.scatter(
            centroid_coords[:, 0],
            centroid_coords[:, 1],
            s=190,
            marker="X",
            alpha=1.0,
            linewidths=0.6,
            edgecolors="black",
            label="Centroids",
        )
        for i, lab in enumerate(cent_labs):
            ax.text(
                centroid_coords[i, 0],
                centroid_coords[i, 1],
                str(lab),
                fontsize=9,
                ha="left",
                va="bottom",
            )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=8, frameon=True)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    configure_logging()

    out_prefix = Path(args.out_prefix)

    LOGGER.info("Loading embeddings: %s", args.embeddings_tsv)
    df = load_tsv_safely(path=args.embeddings_tsv)

    user_metadata = None
    metadata_cols = detect_metadata_columns(df=df, user_metadata=user_metadata)

    if args.id_col not in df.columns:
        raise KeyError(
            f"Identifier column '{args.id_col}' not found. "
            f"Available columns include: {list(df.columns)[:25]} ..."
        )

    feature_cols = numeric_feature_columns(df=df, metadata_cols=metadata_cols)
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns detected.")

    LOGGER.info("Rows=%s, cols=%s; features=%s", df.shape[0], df.shape[1], len(feature_cols))
    LOGGER.info("Aggregating to per-compound vectors using %s.", args.aggregate_method)

    emb = aggregate_replicates(
        df=df,
        id_col=args.id_col,
        feature_cols=feature_cols,
        method=args.aggregate_method,
    )

    emb[args.id_col] = emb[args.id_col].astype(str)
    LOGGER.info("Per-compound table has %s compounds.", emb.shape[0])

    LOGGER.info("Loading anchors: %s", args.anchors_tsv)
    anchors = load_tsv_safely(path=args.anchors_tsv)
    if args.id_col not in anchors.columns or args.moa_col not in anchors.columns:
        raise KeyError(
            f"Anchors file must contain '{args.id_col}' and '{args.moa_col}'. "
            f"Found columns: {list(anchors.columns)}"
        )

    anchors = anchors[[args.id_col, args.moa_col]].copy()
    anchors[args.id_col] = anchors[args.id_col].astype(str)

    plot_df = emb.merge(anchors, how="left", on=args.id_col)

    label_col = args.label_col.strip() if args.label_col.strip() else args.id_col
    if label_col not in plot_df.columns:
        raise KeyError(f"Label column '{label_col}' not found in plot table.")
    point_ids = plot_df[label_col].astype(str)


    if args.colour_by.strip():
        colour_col = args.colour_by.strip()
        if colour_col not in plot_df.columns:
            raise KeyError(
                f"--colour_by '{colour_col}' not found after merge. "
                f"Available columns: {list(plot_df.columns)[:25]} ..."
            )
    else:
        colour_col = args.moa_col

    X = plot_df[feature_cols].to_numpy(dtype=float)
    X = l2_normalise(X=X)

    centroid_coords_pca = None
    centroid_coords_umap = None
    centroid_labels = None
    X_cent = None

    if args.plot_centroids:
        cent_df = compute_centroids(
            df=plot_df,
            id_col=args.id_col,
            label_col=args.moa_col,
            feature_cols=feature_cols,
        )
        if cent_df.shape[0] >= 1:
            centroid_labels = cent_df[args.moa_col].astype(str)
            X_cent = cent_df[feature_cols].to_numpy(dtype=float)
            X_cent = l2_normalise(X=X_cent)
            LOGGER.info("Computed %s centroids for overlay.", cent_df.shape[0])
        else:
            LOGGER.info("No centroids computed (no labelled points).")

    # PCA
    if args.projection in {"pca", "both"}:
        coords_pca, cent_pca = fit_pca_and_transform(X=X, X_extra=X_cent)
        centroid_coords_pca = cent_pca

        out_pdf = Path(f"{out_prefix}_pca.pdf")
        plot_scatter(
            coords=coords_pca,
            labels=plot_df[colour_col],
            title=f"PCA (n={plot_df.shape[0]}) coloured by {colour_col}",
            out_pdf=out_pdf,
            point_ids=point_ids,
            label_points=args.label_points,
            label_jitter=args.label_jitter,
            centroid_coords=centroid_coords_pca,
            centroid_labels=centroid_labels,
        )

        LOGGER.info("Wrote: %s", out_pdf)

    # UMAP
    if args.projection in {"umap", "both"}:
        try:
            coords_umap, cent_umap = fit_umap_and_transform_stacked(
                X=X,
                X_extra=X_cent,
                n_neighbours=args.umap_n_neighbours,
                min_dist=args.umap_min_dist,
                random_state=args.random_seed,
            )
            centroid_coords_umap = cent_umap

            out_pdf = Path(f"{out_prefix}_umap.pdf")
            plot_scatter(
                coords=coords_umap,
                labels=plot_df[colour_col],
                title=f"UMAP (n={plot_df.shape[0]}) coloured by {colour_col}",
                out_pdf=out_pdf,
                point_ids=point_ids,
                label_points=args.label_points,
                label_jitter=args.label_jitter,
                centroid_coords=centroid_coords_umap,
                centroid_labels=centroid_labels,
            )
            LOGGER.info("Wrote: %s", out_pdf)
        except Exception as exc:
            LOGGER.warning("UMAP failed (%s). Skipping UMAP output.", str(exc))

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
