#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
k-NN Baseline for Cell Painting Features (TSV-only)
--------------------------------------------------

This script computes a transparent k-nearest neighbour (k-NN) baseline on
pre-CLIPn feature space. It loads multiple datasets, harmonises and scales
features, aggregates to a chosen granularity, and outputs nearest-neighbour
tables, an optional graph edge list, and optional UMAP coordinates. All
outputs are TSV (no comma-separated files).

Typical usage
-------------
python knn_baseline_cellpainting.py \
    --datasets_csv datasets.tsv \
    --out out/knn_baseline \
    --experiment EXP1 \
    --knn_level compound \
    --knn_metric cosine \
    --knn_k 10 \
    --scaling_mode all \
    --scaling_method robust \
    --graph \
    --umap

Requirements
------------
- Python 3.9+
- pandas, numpy, scikit-learn
- Optional: umap-learn (for UMAP); otherwise falls back to PCA(2D)

Notes
-----
- British English spelling is used (e.g., neighbour).
- No positional arguments are used.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA

# Try to import UMAP; fall back to PCA if unavailable
try:
    import umap  # type: ignore
    _HAVE_UMAP = True
except Exception:  # pragma: no cover
    umap = None
    _HAVE_UMAP = False


# -----------------
# Logging utilities
# -----------------

def setup_logging(out_dir: str | Path, experiment: str) -> logging.Logger:
    """
    Configure logging to both stderr and a file.

    Parameters
    ----------
    out_dir : str | Path
        Directory where the log file will be written.
    experiment : str
        Experiment name used in the log filename.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"{experiment}_knn_baseline.log"

    logger = logging.getLogger("knn_baseline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    sh = logging.StreamHandler(stream=sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    fh = logging.FileHandler(filename=log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info("Starting k-NN baseline. Experiment: %s", experiment)
    logger.info("Command-line arguments: %s", " ".join(sys.argv))
    return logger


# ----------------------
# Light-weight I/O utils
# ----------------------

def detect_csv_delimiter(csv_path: str) -> str:
    """
    Detect the delimiter of a text file, preferring tab if ambiguous.

    Parameters
    ----------
    csv_path : str
        Path to a small text file.

    Returns
    -------
    str
        Detected delimiter ('\\t' or ',').
    """
    opener = open
    if str(csv_path).endswith(".gz"):
        import gzip  # lazy import
        opener = gzip.open  # type: ignore

    with opener(csv_path, mode="rt", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)

    has_tab = "\t" in sample
    has_comma = "," in sample
    if has_tab and has_comma:
        return "\t"
    if has_tab:
        return "\t"
    if has_comma:
        return ","
    return "\t"


def _read_csv_fast(path: str, delimiter: str) -> pd.DataFrame:
    """
    Read CSV/TSV using pyarrow if available, otherwise fall back to Python engine.

    Parameters
    ----------
    path : str
        Path to the input table.
    delimiter : str
        Field delimiter.

    Returns
    -------
    pandas.DataFrame
        Loaded table.
    """
    try:
        return pd.read_csv(path, delimiter=delimiter, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, delimiter=delimiter, engine="python", compression="infer")


def safe_to_csv(df: pd.DataFrame, path: Path | str, sep: str = "\t", logger: Optional[logging.Logger] = None) -> None:
    """
    Save a DataFrame to disk as TSV, flattening MultiIndex columns if present.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to write.
    path : Path | str
        Output file path.
    sep : str, optional
        Field separator (default '\\t').
    logger : logging.Logger, optional
        Logger for status messages.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["__".join(map(str, t)) for t in out.columns.to_list()]
    else:
        out.columns = out.columns.map(str)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep=sep, index=False)
    if logger:
        logger.info("Wrote %d rows × %d cols -> %s", out.shape[0], out.shape[1], path)


# ------------------------------------
# Minimal harmonisation + preprocessing
# ------------------------------------

def ensure_library_column(df: pd.DataFrame, filepath_or_name: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Ensure a 'Library' column exists; if missing, set from file stem.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    filepath_or_name : str
        Path or name to derive a fallback library label.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        Table with a 'Library' column.
    """
    if "Library" not in df.columns:
        fallback = Path(filepath_or_name).stem
        df["Library"] = fallback
        logger.info("'Library' absent; set to '%s'.", fallback)
    return df


def load_single_dataset(name: str, path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load one dataset and lightly standardise metadata columns.

    Parameters
    ----------
    name : str
        Dataset name to assign.
    path : str
        Path to input TSV/CSV.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        Harmonised table with required metadata columns if present.
    """
    sep = detect_csv_delimiter(path)
    df = _read_csv_fast(path, sep)

    # Ensure metadata presence; attempt light renaming if common variants exist
    rename_map = {
        "Plate": "Plate_Metadata",
        "Well": "Well_Metadata",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = ensure_library_column(df, filepath_or_name=path, logger=logger)

    # Warn if mandatory metadata missing (we can still proceed for image-level)
    for col in ["cpd_id", "cpd_type", "Plate_Metadata", "Well_Metadata", "Library"]:
        if col not in df.columns:
            logger.warning("[%s] Metadata column '%s' missing.", name, col)

    # Attach a MultiIndex if none
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["Dataset", "Sample"]:
        df = df.reset_index(drop=True)
        df.index = pd.MultiIndex.from_product([[name], range(len(df))], names=["Dataset", "Sample"])

    return df


def harmonise_numeric_columns(dfs: Dict[str, pd.DataFrame], logger: logging.Logger) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Subset each dataset to the shared intersection of numeric features.

    Parameters
    ----------
    dfs : dict[str, pandas.DataFrame]
        Mapping of dataset name to table.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    tuple
        (updated mapping, list of common numeric feature names)
    """
    num_sets = [set(df.select_dtypes(include=[np.number]).columns) for df in dfs.values()]
    common = sorted(set.intersection(*num_sets)) if num_sets else []
    logger.info("Harmonised numeric feature intersection: %d columns.", len(common))

    meta_cols = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
    for name, df in dfs.items():
        X = df[common] if common else df.select_dtypes(include=[np.number])
        M = df[[c for c in meta_cols if c in df.columns]]
        dfs[name] = pd.concat([X, M], axis=1)
    return dfs, common


def clean_nonfinite_features(df: pd.DataFrame, feature_cols: list[str], logger: logging.Logger, label: str = "") -> pd.DataFrame:
    """
    Replace ±inf with NaN in selected numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    feature_cols : list[str]
        Numeric column names.
    logger : logging.Logger
        Logger instance.
    label : str
        Text label for log messages.

    Returns
    -------
    pandas.DataFrame
        Updated table.
    """
    out = df.copy()
    if feature_cols:
        out.loc[:, feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    n_missing = int(out[feature_cols].isna().sum().sum()) if feature_cols else 0
    if n_missing:
        logger.info("[%s] Missing values after inf→NaN: %d", label, n_missing)
    return out


def clean_and_impute_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    logger: logging.Logger,
    *,
    groupby_cols: list[str] | None = None,
    max_nan_col_frac: float = 0.30,
    max_nan_row_frac: float = 0.80,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop very sparse features/rows and impute remaining NaNs (median per group).

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    feature_cols : list[str]
        Numeric features to consider.
    logger : logging.Logger
        Logger instance.
    groupby_cols : list[str] or None
        Columns used to compute per-group medians for imputation.
    max_nan_col_frac : float
        Drop a feature if its NaN fraction exceeds this threshold.
    max_nan_row_frac : float
        Drop a row if its NaN fraction across kept features exceeds this threshold.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        (cleaned table, list of dropped feature names)
    """
    if not feature_cols:
        return df, []

    df = df.copy()
    # Drop sparse features
    col_nan = df[feature_cols].isna().mean(axis=0)
    drop_feats = col_nan[col_nan > max_nan_col_frac].index.tolist()
    keep_feats = [c for c in feature_cols if c not in drop_feats]
    if drop_feats:
        logger.warning("Dropping %d/%d features with > %.0f%% NaN.", len(drop_feats), len(feature_cols), max_nan_col_frac * 100)

    # Drop sparse rows
    if keep_feats:
        row_nan = df[keep_feats].isna().mean(axis=1)
        to_drop = row_nan > max_nan_row_frac
        nrows = int(to_drop.sum())
        if nrows:
            logger.warning("Dropping %d rows with > %.0f%% NaN across kept features.", nrows, max_nan_row_frac * 100)
        df = df.loc[~to_drop].copy()
    else:
        logger.error("All features would be dropped. Loosen thresholds or inspect inputs.")
        return df.iloc[0:0], feature_cols

    # Median imputation per group
    groups = groupby_cols or (["Dataset", "Plate_Metadata"] if "Plate_Metadata" in df.columns else ["Dataset"])
    def _impute(g: pd.DataFrame) -> pd.DataFrame:
        med = g[keep_feats].median(numeric_only=True)
        g.loc[:, keep_feats] = g[keep_feats].fillna(med)
        return g

    df = df.groupby(groups, dropna=False, sort=False).apply(_impute)
    # Remove group keys from index if introduced
    try:
        if isinstance(df.index, pd.MultiIndex) and df.index.names[: len(groups)] == tuple(groups):
            df.index = df.index.droplevel(list(range(len(groups))))
    except Exception:
        pass

    # Global backup
    remain = int(df[keep_feats].isna().sum().sum())
    if remain:
        logger.warning("Filling %d remaining NaNs with global medians.", remain)
        global_med = df[keep_feats].median(numeric_only=True)
        df.loc[:, keep_feats] = df[keep_feats].fillna(global_med)

    return df, drop_feats


def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    plate_col: str | None,
    mode: str,
    method: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Scale features globally or per-plate.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    feature_cols : list[str]
        Numeric features to scale.
    plate_col : str or None
        Plate column name (required for 'per_plate').
    mode : str
        One of {'all', 'per_plate', 'none'}.
    method : str
        One of {'robust', 'standard'}.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        Scaled table.
    """
    if not feature_cols:
        logger.warning("No numeric features found; skipping scaling.")
        return df

    if mode == "none":
        logger.info("Skipping scaling (mode='none').")
        return df

    scaler_cls = RobustScaler if method == "robust" else StandardScaler
    out = df.copy()

    if mode == "all":
        scaler = scaler_cls()
        out.loc[:, feature_cols] = scaler.fit_transform(out[feature_cols])
        logger.info("Scaled all features using %s scaler.", method)
        return out

    if mode == "per_plate":
        if plate_col is None or plate_col not in out.columns:
            raise ValueError("plate_col must exist for per_plate scaling.")
        for plate, idx in out.groupby(plate_col).groups.items():
            scaler = scaler_cls()
            idx = list(idx)
            out.loc[idx, feature_cols] = scaler.fit_transform(out.loc[idx, feature_cols])
        logger.info("Scaled features per-plate (%s).", method)
        return out

    logger.warning("Unknown scaling mode '%s'; no scaling applied.", mode)
    return df


# -------------------------
# k-NN + aggregation helpers
# -------------------------

def _mode_strict(series: pd.Series) -> Optional[str]:
    """
    Return the most frequent non-null value in a Series, or None.

    Parameters
    ----------
    series : pandas.Series
        Input values.

    Returns
    -------
    Optional[str]
        Modal value if present, otherwise None.
    """
    s = series.dropna()
    if s.empty:
        return None
    m = s.mode(dropna=True)
    return None if m.empty else str(m.iloc[0])


def aggregate_for_knn(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    level: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate to the desired granularity for k-NN.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned and scaled table with metadata.
    feature_cols : list[str]
        Numeric feature columns.
    level : str
        One of {'compound', 'well', 'image'}.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    (X, meta) : tuple[pandas.DataFrame, pandas.DataFrame]
        X: numeric matrix aligned to entities.
        meta: metadata per entity with an 'EntityID' column.
    """
    meta_cols = [c for c in ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"] if c in df.columns]

    if level == "compound":
        if "cpd_id" not in df.columns:
            raise ValueError("Compound-level aggregation requires 'cpd_id'.")
        g = df.groupby("cpd_id", sort=False, dropna=False)
        X = g[feature_cols].median(numeric_only=True)
        meta = pd.DataFrame({
            "EntityID": X.index.astype(str),
            "cpd_id": X.index.astype(str),
        })
        if "cpd_type" in df.columns:
            meta["cpd_type"] = g["cpd_type"].apply(_mode_strict).reset_index(drop=True)
        if "Library" in df.columns:
            meta["Library"] = g["Library"].apply(_mode_strict).reset_index(drop=True)
        meta = meta.set_index(X.index).reset_index(drop=True)

    elif level == "well":
        need = {"Plate_Metadata", "Well_Metadata"}
        if not need.issubset(df.columns):
            raise ValueError("Well-level aggregation requires Plate_Metadata and Well_Metadata.")
        g = df.groupby(["Plate_Metadata", "Well_Metadata"], sort=False, dropna=False)
        X = g[feature_cols].median(numeric_only=True)
        idx = X.reset_index()[["Plate_Metadata", "Well_Metadata"]]
        meta = pd.DataFrame({
            "EntityID": idx["Plate_Metadata"].astype(str) + "::" + idx["Well_Metadata"].astype(str),
            "Plate_Metadata": idx["Plate_Metadata"].astype(str),
            "Well_Metadata": idx["Well_Metadata"].astype(str),
        })
        if "cpd_id" in df.columns:
            meta["cpd_id"] = g["cpd_id"].apply(_mode_strict).reset_index(drop=True)
        if "cpd_type" in df.columns:
            meta["cpd_type"] = g["cpd_type"].apply(_mode_strict).reset_index(drop=True)
        if "Library" in df.columns:
            meta["Library"] = g["Library"].apply(_mode_strict).reset_index(drop=True)
        meta = meta.set_index(X.index).reset_index(drop=True)

    elif level == "image":
        if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["Dataset", "Sample"]:
            raise ValueError("Image-level expects MultiIndex ['Dataset','Sample'] as the index.")
        X = df[feature_cols].copy()
        meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=X.index)
        meta = meta.reset_index()
        meta["EntityID"] = meta["Dataset"].astype(str) + "::" + meta["Sample"].astype(str)
        meta = meta.set_index(X.index).reset_index(drop=True)

    else:
        raise ValueError("level must be one of {'compound','well','image'}.")

    # Normalise indices
    X = X.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    logger.info("Aggregation level '%s' -> %d entities.", level, len(X))
    return X, meta


def run_knn_analysis(
    *,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    k: int,
    metric: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Compute top-k nearest neighbours for each entity.

    Parameters
    ----------
    X : pandas.DataFrame
        Numeric feature matrix.
    meta : pandas.DataFrame
        Metadata aligned to X with 'EntityID'.
    k : int
        Number of neighbours per entity (excluding self).
    metric : str
        One of {'cosine','euclidean','correlation'}.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        Long-format neighbour table.
    """
    if metric in {"cosine", "euclidean"}:
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric=metric)
        nn.fit(X.values)
        dists, idxs = nn.kneighbors(X.values, return_distance=True)
    elif metric == "correlation":
        logger.info("Computing pairwise correlation distances (may be memory intensive).")
        D = pairwise_distances(X.values, metric="correlation")
        idxs = np.argsort(D, axis=1)[:, : min(k + 1, X.shape[0])]
        row_idx = np.arange(X.shape[0])[:, None]
        dists = D[row_idx, idxs]
        del D
    else:
        raise ValueError("metric must be 'cosine', 'euclidean' or 'correlation'.")

    rows: List[Dict[str, object]] = []
    entity_ids = meta["EntityID"].astype(str).tolist()
    meta_cols = [c for c in meta.columns if c != "EntityID"]

    for i in range(len(X)):
        neighbours = [(j, d) for j, d in zip(idxs[i], dists[i]) if j != i]
        neighbours = neighbours[:k]
        for rank, (j, d) in enumerate(neighbours, start=1):
            rec = {
                "QueryID": entity_ids[i],
                "NeighbourID": entity_ids[j],
                "rank": rank,
                "distance": float(d),
            }
            for c in meta_cols:
                rec[f"Query_{c}"] = meta.iloc[i][c]
                rec[f"Neighbour_{c}"] = meta.iloc[j][c]
            rows.append(rec)

    out = pd.DataFrame(rows)
    logger.info("Computed k-NN pairs: %d", out.shape[0])
    return out


def distance_to_similarity(metric: str, distance: float) -> float:
    """
    Convert a distance to a similarity score in [0,1] where possible.

    Parameters
    ----------
    metric : str
        Distance metric ('cosine','euclidean','correlation').
    distance : float
        Non-negative distance.

    Returns
    -------
    float
        Similarity score; higher means more similar.
    """
    if metric in {"cosine", "correlation"}:
        # distances already in [0,2] or [0,2]; map 0->1 (identical), 1->0 (orthogonal)
        return max(0.0, 1.0 - float(distance))
    # Euclidean: scale to (0,1] via a soft transform
    return 1.0 / (1.0 + float(distance))


def build_graph_edges(knn_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Build an undirected edge list from k-NN pairs with a similarity weight.

    Parameters
    ----------
    knn_df : pandas.DataFrame
        Output of run_knn_analysis().
    metric : str
        Metric used to compute distances (for similarity transform).

    Returns
    -------
    pandas.DataFrame
        Edge list with columns: ['SourceID','TargetID','weight'].
    """
    edges = knn_df[["QueryID", "NeighbourID", "distance"]].copy()
    edges["weight"] = edges["distance"].apply(lambda d: distance_to_similarity(metric, d))
    edges = edges.rename(columns={"QueryID": "SourceID", "NeighbourID": "TargetID"})
    # Optional: drop duplicates for undirected graph by canonical ordering
    canon = edges.apply(
        lambda r: tuple(sorted((r["SourceID"], r["TargetID"]))), axis=1
    )
    edges["__pair__"] = canon
    edges = edges.sort_values(by=["__pair__", "weight"], ascending=[True, False])
    edges = edges.drop_duplicates(subset="__pair__", keep="first").drop(columns="__pair__")
    return edges


def simple_knn_qc(knn_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute simple neighbour quality metrics when labels are available.

    Parameters
    ----------
    knn_df : pandas.DataFrame
        Long-format neighbour table.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        One-row summary with available metrics.
    """
    mets: Dict[str, float] = {}

    if {"Query_cpd_id", "Neighbour_cpd_id"}.issubset(knn_df.columns):
        same = (knn_df["Query_cpd_id"].astype(str) == knn_df["Neighbour_cpd_id"].astype(str))
        mets["same_cpd_id_rate"] = float(same.mean())

    if {"Query_cpd_type", "Neighbour_cpd_type"}.issubset(knn_df.columns):
        same = (knn_df["Query_cpd_type"].astype(str) == knn_df["Neighbour_cpd_type"].astype(str))
        mets["same_cpd_type_rate"] = float(same.mean())

    if {"Query_Library", "Neighbour_Library"}.issubset(knn_df.columns):
        same_lib = (knn_df["Query_Library"].astype(str) == knn_df["Neighbour_Library"].astype(str))
        mets["same_library_neighbour_rate"] = float(same_lib.mean())

    out = pd.DataFrame([mets]) if mets else pd.DataFrame([{}])
    logger.info("QC summary: %s", out.to_dict(orient="records")[0])
    return out


def save_knn_outputs(
    *,
    knn_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    out_dir: Path,
    experiment: str,
    save_full_matrix: bool,
    metric: str,
    logger: logging.Logger,
) -> None:
    """
    Save neighbour table, QC summary, and optional full pairwise distance matrix.

    Parameters
    ----------
    knn_df : pandas.DataFrame
        k-NN neighbour pairs.
    qc_df : pandas.DataFrame
        One-row QC summary.
    X : pandas.DataFrame
        Numeric matrix used for k-NN.
    meta : pandas.DataFrame
        Metadata aligned to X.
    out_dir : pathlib.Path
        Output directory.
    experiment : str
        Experiment name for filenames.
    save_full_matrix : bool
        Whether to save the full pairwise distance matrix (guarded to small n).
    metric : str
        Distance metric used.
    logger : logging.Logger
        Logger instance.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    nn_path = out_dir / f"{experiment}_nearest_neighbours.tsv"
    knn_df.to_csv(nn_path, sep="\t", index=False)
    logger.info("Wrote neighbours -> %s", nn_path)

    qc_path = out_dir / f"{experiment}_knn_qc_summary.tsv"
    qc_df.to_csv(qc_path, sep="\t", index=False)
    logger.info("Wrote QC summary -> %s", qc_path)

    if save_full_matrix:
        n = X.shape[0]
        if n > 5000:
            logger.warning("Skipping full pairwise matrix (n=%d too large).", n)
            return
        D = pairwise_distances(X.values, metric=metric if metric != "correlation" else "correlation")
        dm = pd.DataFrame(D, index=meta["EntityID"], columns=meta["EntityID"])
        dm_path = out_dir / f"{experiment}_pairwise_distance_matrix.tsv"
        dm.to_csv(dm_path, sep="\t")
        logger.info("Wrote pairwise matrix %d×%d -> %s", n, n, dm_path)


def compute_umap_coords(
    *,
    X: pd.DataFrame,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Compute 2D UMAP (or PCA fallback) coordinates for sanity-checking.

    Parameters
    ----------
    X : pandas.DataFrame
        Numeric matrix.
    n_neighbors : int
        UMAP n_neighbors parameter.
    min_dist : float
        UMAP min_dist parameter.
    metric : str
        UMAP distance metric.
    random_state : int
        Random seed.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['UMAP_1','UMAP_2'] (or ['PC1','PC2'] on fallback).
    """
    if _HAVE_UMAP:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric=metric,
            random_state=random_state,
        )
        emb = reducer.fit_transform(X.values)
        df = pd.DataFrame(emb, columns=["UMAP_1", "UMAP_2"])
        logger.info("Computed UMAP coordinates (n_neighbors=%d, min_dist=%.3f).", n_neighbors, min_dist)
        return df

    # Fallback to PCA
    pca = PCA(n_components=2, random_state=random_state)
    emb = pca.fit_transform(X.values)
    df = pd.DataFrame(emb, columns=["PC1", "PC2"])
    logger.warning("umap-learn not available; wrote PCA(2D) instead.")
    return df


# ----
# Main
# ----

def main(args: argparse.Namespace) -> None:
    """
    Execute the k-NN baseline workflow from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    logger = setup_logging(out_dir=args.out, experiment=args.experiment)

    # Load datasets
    sep = detect_csv_delimiter(args.datasets_csv)
    ds = pd.read_csv(args.datasets_csv, delimiter=sep)
    if not {"dataset", "path"}.issubset(ds.columns):
        raise ValueError("datasets_csv must contain columns: 'dataset', 'path'.")

    dataframes: Dict[str, pd.DataFrame] = {}
    for name, path in ds.set_index("dataset")["path"].to_dict().items():
        dataframes[name] = load_single_dataset(name=name, path=path, logger=logger)

    dataframes, common_cols = harmonise_numeric_columns(dataframes, logger)
    logger.info("Loaded %d datasets; common numeric features: %d", len(dataframes), len(common_cols))

    # Concatenate in stable order
    combined = pd.concat([dataframes[n] for n in dataframes.keys()], axis=0, sort=False, copy=False)
    if list(combined.index.names) != ["Dataset", "Sample"]:
        combined.index = combined.index.set_names(["Dataset", "Sample"])

    # Identify numeric features (exclude metadata)
    meta_cols = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
    feature_cols = [c for c in combined.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(combined[c])]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after harmonisation.")

    # Clean → impute → (optional) scale
    combined = clean_nonfinite_features(combined, feature_cols, logger, label="pre-scale")
    combined, dropped = clean_and_impute_features(
        combined, feature_cols, logger,
        groupby_cols=["Dataset", "Plate_Metadata"] if "Plate_Metadata" in combined.columns else ["Dataset"],
        max_nan_col_frac=args.trim_nan_feature_frac,
        max_nan_row_frac=args.trim_min_features_per_object,
    )
    if dropped:
        feature_cols = [c for c in feature_cols if c not in dropped]

    if args.scaling_mode != "none":
        combined = scale_features(
            df=combined,
            feature_cols=feature_cols,
            plate_col="Plate_Metadata",
            mode=args.scaling_mode,
            method=args.scaling_method,
            logger=logger,
        )
        combined = clean_nonfinite_features(combined, feature_cols, logger, label="post-scale")

    # Aggregate to chosen level
    X, meta = aggregate_for_knn(df=combined, feature_cols=feature_cols, level=args.knn_level, logger=logger)

    # k-NN
    knn_df = run_knn_analysis(X=X, meta=meta, k=args.knn_k, metric=args.knn_metric, logger=logger)
    qc_df = simple_knn_qc(knn_df=knn_df, logger=logger)

    # Outputs
    out_dir = Path(args.out)
    save_knn_outputs(
        knn_df=knn_df,
        qc_df=qc_df,
        X=X,
        meta=meta,
        out_dir=out_dir,
        experiment=args.experiment,
        save_full_matrix=args.knn_save_full_matrix,
        metric=args.knn_metric,
        logger=logger,
    )

    if args.graph:
        edges = build_graph_edges(knn_df, metric=args.knn_metric)
        edges_path = out_dir / f"{args.experiment}_knn_graph_edges.tsv"
        edges.to_csv(edges_path, sep="\t", index=False)
        logger.info("Wrote graph edges -> %s", edges_path)

        # Nodes table with available metadata
        nodes = meta.copy()
        nodes = nodes.rename(columns={"EntityID": "NodeID"})
        nodes_path = out_dir / f"{args.experiment}_knn_graph_nodes.tsv"
        nodes.to_csv(nodes_path, sep="\t", index=False)
        logger.info("Wrote graph nodes -> %s", nodes_path)

    if args.umap:
        coords = compute_umap_coords(
            X=X,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.seed,
            logger=logger,
        )
        coords = pd.concat([meta[["EntityID"]].reset_index(drop=True), coords], axis=1)
        umap_path = out_dir / f"{args.experiment}_umap_coords.tsv"
        coords.to_csv(umap_path, sep="\t", index=False)
        logger.info("Wrote UMAP/PCA coords -> %s", umap_path)

    logger.info("k-NN baseline completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-NN baseline on Cell Painting features (TSV-only outputs).")

    # Inputs / outputs
    parser.add_argument("--datasets_csv", required=True, help="TSV/CSV with columns: 'dataset', 'path'.")
    parser.add_argument("--out", required=True, help="Output directory for TSVs.")
    parser.add_argument("--experiment", required=True, help="Experiment name used for filenames.")

    # Cleaning and scaling
    parser.add_argument(
        "--trim_nan_feature_frac",
        type=float,
        default=0.30,
        help="Max NaN fraction permitted per feature before dropping (default: 0.30).",
    )
    parser.add_argument(
        "--trim_min_features_per_object",
        type=float,
        default=0.80,
        help="Max NaN fraction permitted per object before dropping (default: 0.80).",
    )
    parser.add_argument(
        "--scaling_mode",
        choices=["all", "per_plate", "none"],
        default="all",
        help="Feature scaling mode (default: all).",
    )
    parser.add_argument(
        "--scaling_method",
        choices=["robust", "standard"],
        default="robust",
        help="Scaler to use (default: robust).",
    )

    # k-NN controls
    parser.add_argument(
        "--knn_level",
        choices=["compound", "well", "image"],
        default="compound",
        help="Entity granularity for k-NN (default: compound).",
    )
    parser.add_argument(
        "--knn_metric",
        choices=["cosine", "euclidean", "correlation"],
        default="cosine",
        help="Distance metric (default: cosine).",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=10,
        help="Number of neighbours per entity (default: 10).",
    )
    parser.add_argument(
        "--knn_save_full_matrix",
        action="store_true",
        help="Also save the full pairwise distance matrix (guarded; n <= 5000).",
    )

    # Graph + UMAP
    parser.add_argument("--graph", action="store_true", help="Emit k-NN graph edge and node TSVs.")
    parser.add_argument("--umap", action="store_true", help="Emit UMAP (or PCA fallback) 2D coordinates TSV.")
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15).",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.10,
        help="UMAP min_dist (default: 0.10).",
    )
    parser.add_argument(
        "--umap_metric",
        type=str,
        default="euclidean",
        help="UMAP metric (default: euclidean).",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )

    main(parser.parse_args())
