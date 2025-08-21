#!/usr/bin/env python3
# coding: utf-8
"""
Heatmap of CLIPn Latent Dimensions
----------------------------------

This standalone script loads a TSV of CLIPn latent embeddings with metadata,
optionally aggregates rows to one-per-`cpd_id` (median), applies optional
normalisation/z-scoring, and renders a heatmap (optionally hierarchically
clustered) to PDF/PNG. It also writes the processed matrix and the row/column
orders to TSV (tab-separated only; never commas).

Typical usage
-------------
python heatmap_latent_space.py \
    --latent_csv HGTx_LSS02_latent.tsv \
    --outdir plots/heatmap \
    --latent_prefix z_ \
    --aggregate auto \
    --zscore feature \
    --cluster_rows --cluster_cols \
    --distance cosine \
    --row_annotations Dataset cpd_type

Inputs
------
- A TSV file with latent columns (digit-named: "0","1",... or prefixed via
  --latent_prefix) plus metadata columns such as `cpd_id`, `Dataset`,
  `cpd_type`, etc.

Outputs (under --outdir)
------------------------
- heatmap.pdf / heatmap.png : the rendered heatmap
- heatmap_matrix.tsv        : matrix used in the heatmap (rows × latent dims)
- row_order.tsv             : order of `cpd_id` used in the heatmap
- col_order.tsv             : order of latent columns used in the heatmap
- heatmap.log               : log file

Notes
-----
- Aggregation: If `--aggregate auto` (default), the script will detect duplicate
  `cpd_id` rows and aggregate by median. You can force on/off via
  `--aggregate yes|no`.
- Z-scoring: `--zscore feature` standardises each latent dimension across
  rows; `--zscore row` standardises each row across dimensions; `none` leaves
  values unchanged (apart from NaN handling).
- Clustering: Requires SciPy. If unavailable or disabled, rows/cols are
  ordered as-is (or by variance for columns).
- Annotation bars: optional row-side categorical colour bars for columns like
  `Dataset` or `cpd_type`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Optional SciPy for hierarchical clustering
try:
    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =============================================================
# Logging
# =============================================================

def setup_logging(*, output_dir: str | Path) -> logging.Logger:
    """Configure console and file logging.

    Parameters
    ----------
    output_dir : str | Path
        Directory where the log file will be written.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "heatmap.log"

    logger = logging.getLogger("clipn_heatmap")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    fileh = logging.FileHandler(filename=str(log_path), mode="w", encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(stream)
    logger.addHandler(fileh)
    logger.info("Logging to %s", log_path)
    return logger


# =============================================================
# Helpers
# =============================================================

def validate_columns(*, df: pd.DataFrame, required: Iterable[str], logger: logging.Logger) -> None:
    """Validate the presence of required columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    required : Iterable[str]
        Column names that must be present.
    logger : logging.Logger
        Logger for reporting issues.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")


def select_latent_columns(*, df: pd.DataFrame, prefix: Optional[str], logger: logging.Logger) -> List[str]:
    """Select latent feature columns by prefix or digit-named convention.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing latent features and metadata.
    prefix : str | None
        If provided, select numeric dtype columns whose names start with this
        prefix. Otherwise, select digit-named numeric columns (e.g. "0","1",...).
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    list[str]
        Names of latent feature columns.

    Raises
    ------
    ValueError
        If no latent columns are found.
    """
    if prefix:
        cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix) and pd.api.types.is_numeric_dtype(df[c])]
    else:
        cols = [c for c in df.columns if isinstance(c, str) and c.isdigit() and pd.api.types.is_numeric_dtype(df[c])]

    if not cols:
        logger.error("No latent feature columns found (prefix=%s).", prefix)
        raise ValueError("No latent feature columns found. Check column names and --latent_prefix.")

    logger.info("Selected %d latent columns.", len(cols))
    return cols


def aggregate_by_id(*, df: pd.DataFrame, id_col: str, latent_cols: List[str], how: str, logger: logging.Logger) -> pd.DataFrame:
    """Aggregate rows to one per `id_col` using the specified statistic.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame with latent features and metadata.
    id_col : str
        Identifier column name (e.g. `cpd_id`).
    latent_cols : list[str]
        Latent feature column names.
    how : str
        Aggregation method: 'median' or 'mean'.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        Aggregated DataFrame with one row per `id_col`.
    """
    agg_func = {"median": np.median, "mean": np.mean}.get(how.lower())
    if agg_func is None:
        raise ValueError("Unsupported aggregation: use 'median' or 'mean'.")

    dup_count = int(df.duplicated(subset=[id_col]).sum())
    if dup_count == 0:
        logger.info("No duplicate %s values detected; skipping aggregation.", id_col)
        return df.copy()

    logger.info("Aggregating to one row per %s using %s (duplicates: %d).", id_col, how, dup_count)

    # Keep first occurrence of metadata for annotations; aggregate only latent cols
    meta_cols = [c for c in df.columns if c not in latent_cols]
    meta_first = df[meta_cols].drop_duplicates(subset=[id_col]).set_index(id_col)

    agg_latent = (
        df[[id_col] + latent_cols]
        .groupby(by=id_col, dropna=False)
        .agg(func=agg_func)
    )

    out = meta_first.join(other=agg_latent, how="right")
    out = out.reset_index()  # restore id_col as a column
    return out


def zscore_matrix(*, X: pd.DataFrame, mode: str, logger: logging.Logger) -> pd.DataFrame:
    """Z-score normalisation of the matrix by feature or row.

    Parameters
    ----------
    X : pandas.DataFrame
        Matrix of values (numeric) to standardise.
    mode : str
        One of: 'none', 'feature', 'row'.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        Standardised matrix (copy).
    """
    mode = mode.lower()
    Xp = X.copy()
    if mode == "none":
        return Xp

    if mode == "feature":
        mu = Xp.mean(axis=0)
        sd = Xp.std(axis=0, ddof=0).replace(to_replace=0, value=1.0)
        logger.info("Z-scoring per feature (columns).")
        Xp = (Xp - mu) / sd
        return Xp

    if mode == "row":
        mu = Xp.mean(axis=1)
        sd = Xp.std(axis=1, ddof=0).replace(to_replace=0, value=1.0)
        logger.info("Z-scoring per row.")
        Xp = (Xp.subtract(other=mu, axis=0)).divide(other=sd, axis=0)
        return Xp

    raise ValueError("--zscore must be one of: none, feature, row")


def robust_clip(*, X: pd.DataFrame, lower_q: float, upper_q: float, logger: logging.Logger) -> pd.DataFrame:
    """Clip extreme values to reduce the impact of outliers.

    Parameters
    ----------
    X : pandas.DataFrame
        Input numeric matrix.
    lower_q : float
        Lower quantile in [0,1).
    upper_q : float
        Upper quantile in (0,1].
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        Clipped matrix (copy).
    """
    lo = float(X.quantile(q=lower_q).min())
    hi = float(X.quantile(q=upper_q).max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        logger.info("Skipping robust clipping (ill-defined quantiles).")
        return X.copy()
    logger.info("Clipping values to [%0.3f, %0.3f] based on quantiles.", lo, hi)
    return X.clip(lower=lo, upper=hi, axis=None)


def hierarchical_order(
    *, X: pd.DataFrame, distance: str, linkage: str, axis: int, logger: logging.Logger
) -> List[int]:
    """Compute hierarchical clustering order of rows or columns.

    Parameters
    ----------
    X : pandas.DataFrame
        Numeric matrix.
    distance : str
        Distance metric for pairwise distances (e.g., 'euclidean', 'cosine').
    linkage : str
        Linkage method for clustering (e.g., 'average', 'complete', 'ward').
    axis : int
        0 for row clustering, 1 for column clustering.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    list[int]
        Order (indices) for rows or columns.
    """
    if not SCIPY_AVAILABLE:
        logger.warning("SciPy not available; skipping hierarchical clustering.")
        return list(range(X.shape[axis]))

    if axis == 0:
        data = X.values
    else:
        data = X.values.T

    # Ward requires Euclidean distance
    if linkage.lower() == "ward" and distance.lower() != "euclidean":
        logger.warning("Ward linkage requires Euclidean distance; switching to 'average'.")
        linkage = "average"

    pdist = ssd.pdist(data, metric=distance)
    if not np.isfinite(pdist).all():
        logger.warning("Non-finite distances encountered; replacing NaN/inf with large values.")
        pdist = np.nan_to_num(x=pdist, nan=np.nanmax(pdist[np.isfinite(pdist)]) * 10.0, posinf=None, neginf=None)

    Z = sch.linkage(pdist, method=linkage)
    order = sch.leaves_list(Z).tolist()
    return order


def write_tsv(*, df: pd.DataFrame, path: str | Path, logger: logging.Logger, index: bool = False) -> None:
    """Write a DataFrame to a TSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to write.
    path : str | Path
        Output file path.
    logger : logging.Logger
        Logger instance.
    index : bool
        Whether to include the index column.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_or_buf=path, sep="\t", index=index)
    logger.info("Wrote %d rows to %s", len(df), path)


# =============================================================
# Plotting
# =============================================================

def draw_heatmap(
    *,
    X: pd.DataFrame,
    row_labels: List[str],
    col_labels: List[str],
    row_ann: Optional[pd.DataFrame],
    out_pdf: Path,
    out_png: Path,
    title: str,
    logger: logging.Logger,
) -> None:
    """Render a heatmap with optional row-side categorical colour bars.

    Parameters
    ----------
    X : pandas.DataFrame
        Numeric matrix with shape (rows, cols).
    row_labels : list[str]
        Labels for rows (e.g., cpd_id).
    col_labels : list[str]
        Labels for columns (latent dims).
    row_ann : pandas.DataFrame | None
        Optional row annotations (categorical) with index aligned to X rows.
    out_pdf : pathlib.Path
        Output path for the PDF image.
    out_png : pathlib.Path
        Output path for the PNG image.
    title : str
        Title for the figure.
    logger : logging.Logger
        Logger instance.
    """
    n_rows, n_cols = X.shape

    # Dynamic figure size with sensible caps
    width = max(6.0, min(0.12 * n_cols + 2.0, 22.0))
    height = max(6.0, min(0.18 * n_rows + 2.0, 28.0))

    # GridSpec: optional narrow strip for row annotations
    if row_ann is not None and not row_ann.empty:
        ann_width = 0.30 * len(row_ann.columns)  # ~0.3 inch per annotation
    else:
        ann_width = 0.0

    fig = plt.figure(figsize=(width + ann_width, height))
    gs = fig.add_gridspec(nrows=1, ncols=2 if ann_width > 0 else 1, width_ratios=[ann_width, 1.0] if ann_width > 0 else [1.0], wspace=0.02)

    # Row annotation panel (left)
    if ann_width > 0:
        ax_ann = fig.add_subplot(gs[0, 0])
        # Build colour blocks for each annotation column
        ann_mat_list = []
        legends: List[Tuple[str, dict]] = []
        for col in row_ann.columns:
            cats = row_ann[col].astype(str).fillna("nan").tolist()
            uniq = sorted(set(cats))
            cmap = mpl.colormaps.get("tab20", mpl.colormaps["tab20"])  # categorical palette
            colour_map = {c: cmap(i % cmap.N) for i, c in enumerate(uniq)}
            legends.append((col, colour_map))
            ann_mat_list.append(np.array([colour_map[c] for c in cats]).reshape(-1, 1, 4))
        ann_mat = np.concatenate(ann_mat_list, axis=1)  # shape: rows × ann_cols × rgba
        # Display as image (transpose ann_cols to width)
        ax_ann.imshow(ann_mat, aspect="auto", interpolation="nearest")
        ax_ann.set_xticks([])
        ax_ann.set_yticks([])
    else:
        ax_ann = None

    # Heatmap panel (right or only)
    ax_heat = fig.add_subplot(gs[0, -1])
    im = ax_heat.imshow(X.values, aspect="auto", interpolation="nearest")
    ax_heat.set_title(title)
    ax_heat.set_xticks(np.arange(n_cols))
    ax_heat.set_xticklabels(col_labels, rotation=90, fontsize=7)

    # Row labels: only if not too many
    if n_rows <= 80:
        ax_heat.set_yticks(np.arange(n_rows))
        ax_heat.set_yticklabels(row_labels, fontsize=7)
    else:
        ax_heat.set_yticks([])

    # Colourbar
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("value", rotation=-90, va="bottom")

    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # Log legends for annotations (textual mapping only)
    if row_ann is not None and not row_ann.empty and ax_ann is not None:
        logger.info("Row annotation columns: %s", list(row_ann.columns))


# =============================================================
# Main
# =============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the heatmap script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Render a heatmap of CLIPn latent dimensions with optional aggregation and clustering.")
    p.add_argument("--latent_csv", required=True, help="Input TSV with latent features + metadata.")
    p.add_argument("--outdir", required=True, help="Output directory for images and TSVs.")

    # Columns and selection
    p.add_argument("--id_col", default="cpd_id", help="Identifier column name (default: cpd_id).")
    p.add_argument("--latent_prefix", default=None, help="Prefix for latent columns (default: use digit-named columns).")

    # Aggregation behaviour
    p.add_argument("--aggregate", choices=["auto", "yes", "no"], default="auto", help="Aggregate to one row per id (median) if duplicates present (default: auto).")
    p.add_argument("--agg_method", choices=["median", "mean"], default="median", help="Aggregation method when aggregating (default: median).")

    # Preprocessing
    p.add_argument("--zscore", choices=["none", "feature", "row"], default="feature", help="Z-scoring mode (default: feature).")
    p.add_argument("--clip_lower_q", type=float, default=0.02, help="Lower quantile for robust clipping (default: 0.02).")
    p.add_argument("--clip_upper_q", type=float, default=0.98, help="Upper quantile for robust clipping (default: 0.98).")

    # Clustering and ordering
    p.add_argument("--cluster_rows", action=argparse.BooleanOptionalAction, default=True, help="Enable hierarchical clustering of rows (default: enabled).")
    p.add_argument("--cluster_cols", action=argparse.BooleanOptionalAction, default=True, help="Enable hierarchical clustering of columns (default: enabled).")
    p.add_argument("--distance", default="cosine", help="Distance metric for clustering (default: cosine).")
    p.add_argument("--linkage", default="average", help="Linkage method for clustering (default: average).")

    # Row annotations
    p.add_argument("--row_annotations", nargs="*", default=None, help="Optional categorical columns to show as row-side annotation bars (e.g., Dataset cpd_type).")

    # Subsetting
    p.add_argument("--max_rows", type=int, default=None, help="Optional cap on number of rows to plot (selects the first N after ordering).")

    return p.parse_args()


def main() -> None:
    """Entry point: read latent TSV, prepare matrix, and render heatmap."""
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir=outdir)

    logger.info("Args: %s", vars(args))
    logger.info("Reading: %s", args.latent_csv)
    df = pd.read_csv(filepath_or_buffer=args.latent_csv, sep="\t")

    # Ensure id_col exists
    validate_columns(df=df, required=[args.id_col], logger=logger)

    # Select latent columns
    latent_cols = select_latent_columns(df=df, prefix=args.latent_prefix, logger=logger)

    # Fill NaNs in latent with per-column medians, then 0 as last resort
    Xfull = df[latent_cols].copy()
    Xfull = Xfull.apply(func=lambda s: s.fillna(value=s.median()), axis=0)
    Xfull = Xfull.fillna(value=0)
    df[latent_cols] = Xfull

    # Decide aggregation
    do_agg = False
    if args.aggregate == "yes":
        do_agg = True
    elif args.aggregate == "no":
        do_agg = False
    else:  # auto
        do_agg = bool(df.duplicated(subset=[args.id_col]).any())

    if do_agg:
        df = aggregate_by_id(df=df, id_col=args.id_col, latent_cols=latent_cols, how=args.agg_method, logger=logger)
        logger.info("Aggregated rows: %d unique %s", len(df), args.id_col)
    else:
        logger.info("Aggregation disabled or not required (rows: %d).", len(df))

    # Build matrix and labels
    row_labels = df[args.id_col].astype(str).tolist()
    X = df[latent_cols].copy()

    # Z-score and clipping
    X = zscore_matrix(X=X, mode=args.zscore, logger=logger)
    X = robust_clip(X=X, lower_q=args.clip_lower_q, upper_q=args.clip_upper_q, logger=logger)

    # Determine row/column orders
    if args.cluster_rows:
        row_order = hierarchical_order(X=X, distance=args.distance, linkage=args.linkage, axis=0, logger=logger)
    else:
        row_order = list(range(X.shape[0]))

    if args.cluster_cols:
        col_order = hierarchical_order(X=X, distance=args.distance, linkage=args.linkage, axis=1, logger=logger)
    else:
        # Order columns by decreasing variance (helps readability)
        col_order = (
            X.var(axis=0).sort_values(ascending=False).index.to_list()
        )
        # convert names to positional indices for consistent handling
        name_to_idx = {c: i for i, c in enumerate(X.columns)}
        col_order = [name_to_idx[c] for c in col_order]

    # Reorder
    Xo = X.iloc[row_order, col_order]
    row_labels_o = [row_labels[i] for i in row_order]
    col_labels_o = [X.columns[i] for i in col_order]

    # Optionally subset rows
    if args.max_rows is not None and args.max_rows > 0 and Xo.shape[0] > args.max_rows:
        logger.info("Subsetting to first %d rows after ordering (from %d).", args.max_rows, Xo.shape[0])
        Xo = Xo.iloc[: args.max_rows, :]
        row_labels_o = row_labels_o[: args.max_rows]

    # Prepare row annotations
    row_ann = None
    if args.row_annotations:
        cols = [c for c in args.row_annotations if c in df.columns]
        if cols:
            row_ann = df.iloc[row_order][cols].reset_index(drop=True)
        else:
            logger.warning("None of the requested row annotation columns found: %s", args.row_annotations)

    # Write matrix and orders
    write_tsv(df=pd.DataFrame({"cpd_id": row_labels_o}).reset_index(drop=True), path=outdir / "row_order.tsv", logger=logger, index=False)
    write_tsv(df=pd.DataFrame({"latent_dim": col_labels_o}).reset_index(drop=True), path=outdir / "col_order.tsv", logger=logger, index=False)

    mat_out = Xo.copy()
    mat_out.insert(loc=0, column="cpd_id", value=row_labels_o)
    write_tsv(df=mat_out, path=outdir / "heatmap_matrix.tsv", logger=logger, index=False)

    # Draw heatmap
    pdf_path = outdir / "heatmap.pdf"
    png_path = outdir / "heatmap.png"
    title = "CLIPn latent heatmap"
    if args.zscore != "none":
        title += f" (zscore={args.zscore})"
    if args.cluster_rows or args.cluster_cols:
        title += " + cluster"

    draw_heatmap(
        X=Xo,
        row_labels=row_labels_o,
        col_labels=col_labels_o,
        row_ann=row_ann,
        out_pdf=pdf_path,
        out_png=png_path,
        title=title,
        logger=logger,
    )

    logger.info("Saved heatmap to %s and %s", pdf_path, png_path)


if __name__ == "__main__":
    main()
