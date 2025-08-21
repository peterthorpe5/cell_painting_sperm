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
- heatmap.pdf : the rendered heatmap (default)
- heatmap.png : only if --also_png is provided
- heatmap_matrix.tsv        : matrix used in the heatmap (rows × latent dims)
- row_order.tsv             : order of `cpd_id` used in the heatmap
- col_order.tsv             : order of latent columns used in the heatmap
- cpd_id_clusters.tsv          : optional row cluster labels if --k_rows is set
- col_clusters.tsv          : optional column cluster labels if --k_cols is set
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
  `Dataset` or `cpd_type` (narrow by default; adjustable via `--ann_width_per_col`).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

def detect_separator(*, file_path: str | Path, logger: logging.Logger) -> str:
    """Auto-detect the field separator for a delimited text file.

    Parameters
    ----------
    file_path : str | Path
        Path to the input file.
    logger : logging.Logger
        Logger for messages.

    Returns
    -------
    str
        Detected delimiter character. Falls back to "	" if unsure.
    """
    try:
        with open(file=file_path, mode="r", encoding="utf-8", errors="ignore") as fh:
            sample = fh.read(1024 * 1024)
        dialect = csv.Sniffer().sniff(sample, delimiters="	,;| ")
        sep = dialect.delimiter
        return sep
    except Exception:
        logger.warning("Could not auto-detect delimiter; defaulting to tab (\t).")
        return "	"


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
        cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    else:
        cols = [c for c in df.columns if isinstance(c, str) and c.isdigit()]

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
) -> Tuple[List[int], Optional[np.ndarray]]:
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
    tuple[list[int], numpy.ndarray | None]
        (Order indices for rows/cols, linkage matrix or None if SciPy unavailable).
    """
    if not SCIPY_AVAILABLE:
        logger.warning("SciPy not available; skipping hierarchical clustering.")
        return list(range(X.shape[axis])), None

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
    return order, Z


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
    out_png: Optional[Path],
    title: str,
    logger: logging.Logger,
    cmap: str,
    vcenter: Optional[float],
    row_Z: Optional[np.ndarray],
    col_Z: Optional[np.ndarray],
    show_dendrograms: bool,
    dendro_width: float,
    dendro_height: float,
    ann_width_per_col: float,
    cbar_location: str,
) -> None:
    """Render a heatmap with optional dendrograms and row-side categorical colour bars.

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
    out_png : pathlib.Path | None
        Output path for the PNG image, if requested.
    title : str
        Title for the figure.
    logger : logging.Logger
        Logger instance.
    cmap : str
        Matplotlib colour map name (e.g., 'bwr').
    vcenter : float | None
        Centre value for diverging colour map; None disables centring.
    row_Z : numpy.ndarray | None
        Linkage matrix for rows (SciPy). Required to draw row dendrograms.
    col_Z : numpy.ndarray | None
        Linkage matrix for columns (SciPy). Required to draw column dendrograms.
    show_dendrograms : bool
        Whether to draw dendrograms at left (rows) and top (columns).
    dendro_width : float
        Width in inches allocated to the left dendrogram.
    dendro_height : float
        Height in inches allocated to the top dendrogram.
    """
    n_rows, n_cols = X.shape

    # Dynamic figure size with sensible caps for the heatmap panel
    heat_w = max(6.0, min(0.12 * n_cols + 2.0, 22.0))
    heat_h = max(6.0, min(0.18 * n_rows + 2.0, 28.0))

    # Annotation strip width (narrower by default; user-adjustable)
    if row_ann is not None and not row_ann.empty:
        ann_cols = len(row_ann.columns)
        ann_w = ann_width_per_col * ann_cols
    else:
        ann_w = 0.0

    use_dendro = show_dendrograms and SCIPY_AVAILABLE and (row_Z is not None or col_Z is not None)

    # Build figure + layout
    if use_dendro:
        fig_w = heat_w + ann_w + (dendro_width if row_Z is not None else 0.0)
        fig_h = heat_h + (dendro_height if col_Z is not None else 0.0)

        if ann_w > 0:
            gs = plt.figure(figsize=(fig_w, fig_h)).add_gridspec(
                nrows=2,
                ncols=3,
                width_ratios=[dendro_width if row_Z is not None else 0.01, ann_w, 1.0],
                height_ratios=[dendro_height if col_Z is not None else 0.01, 1.0],
                wspace=0.02,
                hspace=0.02,
            )
            fig = plt.gcf()
            ax_row_d = fig.add_subplot(gs[1, 0]) if row_Z is not None else None
            ax_ann = fig.add_subplot(gs[1, 1])
            ax_heat = fig.add_subplot(gs[1, 2])
            ax_col_d = fig.add_subplot(gs[0, 2]) if col_Z is not None else None
        else:
            gs = plt.figure(figsize=(fig_w, fig_h)).add_gridspec(
                nrows=2,
                ncols=2,
                width_ratios=[dendro_width if row_Z is not None else 0.01, 1.0],
                height_ratios=[dendro_height if col_Z is not None else 0.01, 1.0],
                wspace=0.02,
                hspace=0.02,
            )
            fig = plt.gcf()
            ax_row_d = fig.add_subplot(gs[1, 0]) if row_Z is not None else None
            ax_heat = fig.add_subplot(gs[1, 1])
            ax_ann = None
            ax_col_d = fig.add_subplot(gs[0, 1]) if col_Z is not None else None
    else:
        # No dendrograms: keep a simpler layout (annotations left, heatmap right)
        fig = plt.figure(figsize=(heat_w + ann_w, heat_h))
        gs = fig.add_gridspec(
            nrows=1,
            ncols=2 if ann_w > 0 else 1,
            width_ratios=[ann_w, 1.0] if ann_w > 0 else [1.0],
            wspace=0.02,
        )
        ax_ann = fig.add_subplot(gs[0, 0]) if ann_w > 0 else None
        ax_heat = fig.add_subplot(gs[0, -1])
        ax_row_d = None
        ax_col_d = None

    # Row annotations (if present)
    if ax_ann is not None:
        # Build colour blocks for each annotation column
        ann_mat_list = []
        for col in row_ann.columns:
            cats = row_ann[col].astype(str).fillna("nan").tolist()
            uniq = sorted(set(cats))
            cat_cmap = mpl.colormaps.get("tab20", mpl.colormaps["tab20"])  # categorical palette
            colour_map = {c: cat_cmap(i % cat_cmap.N) for i, c in enumerate(uniq)}
            ann_mat_list.append(np.array([colour_map[c] for c in cats]).reshape(-1, 1, 4))
        ann_mat = np.concatenate(ann_mat_list, axis=1) if ann_mat_list else np.zeros((n_rows, 1, 4))
        ax_ann.imshow(ann_mat, aspect="auto", interpolation="nearest")
        ax_ann.set_xticks([])
        ax_ann.set_yticks([])

    # Heatmap
    norm = TwoSlopeNorm(vcenter=vcenter) if vcenter is not None else None
    im = ax_heat.imshow(X.values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax_heat.set_title(title)
    ax_heat.set_xticks(np.arange(n_cols))
    ax_heat.set_xticklabels(col_labels, rotation=90, fontsize=7)

    # Row labels on the RIGHT (per your request)
    if n_rows <= 80:
        ax_heat.set_yticks(np.arange(n_rows))
        ax_heat.tick_params(left=False, right=True, labelleft=False, labelright=True)
        ax_heat.set_yticklabels(row_labels, fontsize=7)
    else:
        ax_heat.set_yticks([])

    # Colourbar positioning
    if cbar_location.lower() == "top":
        # Place a horizontal colourbar above the heatmap axis without overlapping labels
        pos = ax_heat.get_position()
        cax = fig.add_axes([pos.x0, min(0.98, pos.y1 + 0.01), pos.width, 0.02])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.set_xlabel("value")
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.tick_top()
    elif cbar_location.lower() == "bottom":
        cbar = fig.colorbar(im, ax=ax_heat, orientation='horizontal', fraction=0.04, pad=0.10)
        cbar.ax.set_xlabel("value")
    else:  # right (default fallback)
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.06)
        cbar.ax.set_ylabel("value", rotation=-90, va="bottom")

    # Dendrograms
    if use_dendro:
        if col_Z is not None and ax_col_d is not None:
            sch.dendrogram(col_Z, ax=ax_col_d, orientation="top", no_labels=True, color_threshold=None)
            ax_col_d.set_xticks([])
            ax_col_d.set_yticks([])
            for spine in ax_col_d.spines.values():
                spine.set_visible(False)
        if row_Z is not None and ax_row_d is not None:
            sch.dendrogram(row_Z, ax=ax_row_d, orientation="left", no_labels=True, color_threshold=None)
            ax_row_d.set_xticks([])
            ax_row_d.set_yticks([])
            for spine in ax_row_d.spines.values():
                spine.set_visible(False)

    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    if out_png is not None:
        fig.savefig(out_png, dpi=200)
    plt.close(fig)


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
    p.add_argument("--sep", choices=["auto", "tab", "comma", "semicolon", "pipe", "space"], default="auto", help="Field separator for --latent_csv (default: auto-detect).")
    p.add_argument("--also_png", action="store_true", help="Also save a PNG alongside the PDF.")
    p.add_argument("--cbar_location", choices=["top", "right", "bottom"], default="bottom", help="Location of colourbar/key (default: top).")

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

    # Dendrogram display
    p.add_argument("--show_dendrograms", action=argparse.BooleanOptionalAction, default=True, help="Draw dendrograms above (columns) and left (rows); disable with --no-show-dendrograms.")
    p.add_argument("--dendro_width", type=float, default=0.5, help="Width in inches for the left row dendrogram (default: 1.2).")
    p.add_argument("--dendro_height", type=float, default=0.5, help="Height in inches for the top column dendrogram (default: 1.2).")
    p.add_argument("--ann_width_per_col", type=float, default=0.12, help="Width in inches allocated per annotation column (default: 0.12).")

    # Colour map and centring
    p.add_argument("--cmap", default="bwr", help="Matplotlib colour map for the heatmap (default: bwr).")
    p.add_argument("--vcenter", type=float, default=0.0, help="Centre value for diverging colour map (default: 0.0).")
    p.add_argument("--no_vcenter", action="store_true", help="Disable centring the colour map at the given value.")

    # Cluster cuts
    p.add_argument("--k_rows", type=int, default=None, help="Cut the row dendrogram into K clusters and write row_clusters.tsv.")
    p.add_argument("--k_cols", type=int, default=None, help="Cut the column dendrogram into K clusters and write col_clusters.tsv.")

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
    # Auto-detect separator unless overridden
    if args.sep == "auto":
        sep_detected = detect_separator(file_path=args.latent_csv, logger=logger)
        if sep_detected == " ":
            df = pd.read_csv(filepath_or_buffer=args.latent_csv, sep=r"\s+", engine="python")
        else:
            df = pd.read_csv(filepath_or_buffer=args.latent_csv, sep=sep_detected)
        logger.info("Detected field separator: %r", sep_detected)
    else:
        sep_map = {"tab": "	", "comma": ",", "semicolon": ";", "pipe": "|", "space": r"\s+"}
        chosen = sep_map.get(args.sep, "	")
        if args.sep == "space":
            df = pd.read_csv(filepath_or_buffer=args.latent_csv, sep=chosen, engine="python")
        else:
            df = pd.read_csv(filepath_or_buffer=args.latent_csv, sep=chosen)
        logger.info("Using field separator: %s", args.sep)

    # Ensure id_col exists
    validate_columns(df=df, required=[args.id_col], logger=logger)

    # Select latent columns
    latent_cols = select_latent_columns(df=df, prefix=args.latent_prefix, logger=logger)

    # Fill NaNs in latent with per-column medians, then 0 as last resort
    Xfull = df[latent_cols].copy()
    Xfull = Xfull.apply(pd.to_numeric, errors='coerce')
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
        row_order, row_Z = hierarchical_order(X=X, distance=args.distance, linkage=args.linkage, axis=0, logger=logger)
    else:
        row_order, row_Z = list(range(X.shape[0])), None

    if args.cluster_cols:
        col_order, col_Z = hierarchical_order(X=X, distance=args.distance, linkage=args.linkage, axis=1, logger=logger)
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

    # Optional cluster cuts output
    if args.k_rows is not None and row_Z is not None and SCIPY_AVAILABLE:
        row_cluster_labels = sch.fcluster(row_Z, t=int(args.k_rows), criterion='maxclust')
        # Align to heatmap order
        row_clusters_o = [int(row_cluster_labels[i]) for i in row_order]
        write_tsv(
            df=pd.DataFrame({"cpd_id": row_labels_o, "Cluster": row_clusters_o}),
            path=outdir / "cpd_id_clusters.tsv",
            logger=logger,
            index=False,
        )
    if args.k_cols is not None and col_Z is not None and SCIPY_AVAILABLE:
        col_cluster_labels = sch.fcluster(col_Z, t=int(args.k_cols), criterion='maxclust')
        col_clusters_o = [int(col_cluster_labels[i]) for i in col_order]
        write_tsv(
            df=pd.DataFrame({"latent_dim": col_labels_o, "Cluster": col_clusters_o}),
            path=outdir / "col_clusters.tsv",
            logger=logger,
            index=False,
        )

    # Draw heatmap
    pdf_path = outdir / "heatmap.pdf"
    png_path = (outdir / "heatmap.png") if args.also_png else None
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
        cmap=args.cmap,
        vcenter=None if args.no_vcenter else args.vcenter,
        row_Z=row_Z,
        col_Z=col_Z,
        show_dendrograms=args.show_dendrograms,
        dendro_width=float(args.dendro_width),
        dendro_height=float(args.dendro_height),
        ann_width_per_col=float(args.ann_width_per_col),
        cbar_location=args.cbar_location,
    )

    if png_path is not None:
        logger.info("Saved heatmap to %s and %s", pdf_path, png_path)
    else:
        logger.info("Saved heatmap to %s", pdf_path)


if __name__ == "__main__":
    main()
