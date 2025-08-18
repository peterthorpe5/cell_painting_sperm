#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SHAP-based Feature Attribution for Cell Painting Nearest-Neighbour Similarity
-----------------------------------------------------------------------------

Given a well-level feature table and an NN table, explain which features
distinguish a query compound (target=1) from its nearest neighbours (target=0)
using a simple classifier (LogReg for small N, Random Forest otherwise) and SHAP.

Inputs
------
--features            : TSV of well-level features OR a CSV/TSV “file-of-files”
                        containing a column 'path' with TSVs to load and merge.
--nn_file             : TSV with columns [cpd_id|query_id, neighbour_id, distance].
--query_id            : Comma-separated list OR a text file (one ID per line).
--output_dir          : Output directory.
--n_neighbors         : Number of nearest neighbours to use per query (default: 5).
--n_top_features      : Number of top features to report/plot (default: 10).
--log_file            : Log file name inside output_dir (default: shap_explain.log).

Outputs (per query)
-------------------
Tables (TSV):
- <query>_top_shap_features_driving_difference.tsv
- <query>_most_similar_features.tsv

Plots (PDF):
- <query>_top_shap_features_bar.pdf
- <query>_most_similar_features_bar.pdf
- <query>_shap_summary_bar.pdf
- <query>_shap_summary_beeswarm.pdf
- <query>_shap_summary_top_bar.pdf
- <query>_shap_summary_top_beeswarm.pdf
- <query>_shap_summary_similar_bar.pdf
- <query>_shap_summary_similar_beeswarm.pdf
- <query>_shap_bar_clustered_difference.pdf
- <query>_shap_bar_clustered_similarity.pdf
- <query>_shap_waterfall_query_median.pdf
- <query>_shap_heatmap.pdf
- <query>_dependence_plots/*.pdf

Notes
-----
- Only numeric columns are used as features; metadata are dropped automatically.
- Robust to different SHAP value shapes (list/2D/3D); always uses class 1.
- All output tables are TSV; no comma-separated files.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from typing import Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ------------------------- Logging -------------------------------------------

def setup_logger(log_file: str) -> logging.Logger:
    """
    Create a logger that writes DEBUG to file and INFO to stderr.

    Parameters
    ----------
    log_file : str
        Path to the log file to write.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("shap_explain")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ------------------------- I/O helpers ---------------------------------------

META_COLS_DEFAULT = {
    "cpd_id", "cpd_type", "Dataset", "Library", "Plate_Metadata", "Well_Metadata", "target"
}


def _read_table(path: str) -> pd.DataFrame:
    """
    Read CSV/TSV with automatic delimiter detection.

    Parameters
    ----------
    path : str
        Path to a delimited text file.

    Returns
    -------
    pandas.DataFrame
        Loaded table.
    """
    return pd.read_csv(path, sep=None, engine="python")


def load_feature_files(list_or_single: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load well-level features.

    If `list_or_single` is a CSV/TSV with 'path' column, read and harmonise all.
    Otherwise treat it as a single TSV of well-level data.

    Parameters
    ----------
    list_or_single : str
        Path to a single features TSV or a file-of-files with a 'path' column.
    logger : logging.Logger
        Logger for messages.

    Returns
    -------
    pandas.DataFrame
        Concatenated and harmonised well-level features.
    """
    ext = os.path.splitext(list_or_single)[-1].lower()
    try:
        df_list = pd.read_csv(list_or_single, sep=None, engine="python")
        if "path" in df_list.columns:
            logger.info("Detected file-of-files with %d inputs.", len(df_list))
            dfs = []
            for p in df_list["path"]:
                d = _read_table(str(p))
                dfs.append(d)
            # Harmonise by intersection of columns
            common = set(dfs[0].columns)
            for d in dfs[1:]:
                common &= set(d.columns)
            if not common:
                raise ValueError("No common columns across feature files.")
            dfs = [d[list(sorted(common))] for d in dfs]
            out = pd.concat(dfs, ignore_index=True)
            logger.info("Loaded & harmonised well-level features: %s", out.shape)
            return out
        # Fall through to single TSV if no 'path'
    except Exception as e:
        logger.info(
            "Not a file-of-files (or failed to parse as such): %s. Trying as single TSV…",
            e,
        )

    out = _read_table(list_or_single)
    logger.info("Loaded single well-level feature file: %s shape=%s", list_or_single, out.shape)
    return out


def parse_query_ids(arg: str) -> List[str]:
    """
    Parse query IDs from a comma-separated string or a newline-delimited file.

    Parameters
    ----------
    arg : str
        Comma-separated IDs or path to a text file with one ID per line.

    Returns
    -------
    list of str
        Parsed query IDs.
    """
    if os.path.isfile(arg):
        with open(arg) as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    return [x.strip() for x in arg.split(",") if x.strip()]


def load_neighbours(nn_file: str,
                    query_id: str,
                    n_neighbors: int,
                    logger: logging.Logger) -> List[str]:
    """
    Return top-N neighbour IDs for a given query from an NN table.

    Parameters
    ----------
    nn_file : str
        Path to nearest-neighbour TSV.
    query_id : str
        Query compound identifier.
    n_neighbors : int
        Number of neighbours to retrieve.
    logger : logging.Logger
        Logger for messages.

    Returns
    -------
    list of str
        Neighbour IDs (unique, order may be sorted by distance if present).
    """
    nn = pd.read_csv(nn_file, sep=None, engine="python")
    # Accept either 'cpd_id' or 'query_id'
    qcol = "cpd_id" if "cpd_id" in nn.columns else ("query_id" if "query_id" in nn.columns else None)
    if qcol is None:
        raise ValueError("NN file must have 'cpd_id' or 'query_id' column.")
    mask = nn[qcol].astype(str).str.upper() == str(query_id).upper()
    hits = nn.loc[mask].copy()
    if "distance" in hits.columns:
        hits = hits.sort_values("distance")
    if "neighbour_id" not in hits.columns:
        raise ValueError("NN file must contain a 'neighbour_id' column.")
    out = hits["neighbour_id"].astype(str).unique().tolist()[:n_neighbors]
    logger.info("Found %d neighbours for %s: %s", len(out), query_id, out)
    return out


# ------------------------- Plot helpers --------------------------------------

def plot_feature_importance_bar(features: Iterable[str],
                                importance: np.ndarray,
                                out_pdf: str,
                                title: str,
                                logger: logging.Logger) -> None:
    """
    Save a horizontal bar chart for (feature, importance).

    Parameters
    ----------
    features : Iterable[str]
        Feature names.
    importance : numpy.ndarray
        Importance values aligned to features.
    out_pdf : str
        Output PDF path.
    title : str
        Plot title.
    logger : logging.Logger
        Logger for messages.
    """
    try:
        features = list(features)
        idx = np.argsort(importance)  # ascending; invert for plotting top at bottom
        y = np.arange(len(features))
        plt.figure(figsize=(8, 0.5 * len(features) + 2))
        plt.barh(y, importance[idx], align="center")
        plt.yticks(y, [features[i] for i in idx])
        plt.xlabel("Mean |SHAP|")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_pdf)
        plt.close()
        logger.info("Wrote bar plot: %s", out_pdf)
    except Exception as e:
        logger.warning("Bar plot failed: %s", e)


def plot_summary_pair(X: pd.DataFrame,
                      shap_values: np.ndarray,
                      feature_names: List[str],
                      out_prefix: str,
                      logger: logging.Logger,
                      n_top: int = 10) -> None:
    """
    Write bar + beeswarm SHAP summary plots.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    shap_values : numpy.ndarray
        SHAP values aligned to X.
    feature_names : list of str
        Feature names.
    out_prefix : str
        Output prefix (without extension).
    logger : logging.Logger
        Logger for messages.
    n_top : int
        Number of features to display.
    """
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X.values, feature_names=feature_names,
                          show=False, max_display=n_top, plot_type="bar")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_bar.pdf")
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X.values, feature_names=feature_names,
                          show=False, max_display=n_top, plot_type="dot")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_beeswarm.pdf")
        plt.close()
        logger.info("Wrote SHAP summary plots: %s_bar/beeswarm.pdf", out_prefix)
    except Exception as e:
        logger.warning("SHAP summary plotting failed: %s", e)


def plot_waterfall_for_median_query(X: pd.DataFrame,
                                    shap_values: np.ndarray,
                                    feature_names: List[str],
                                    target: pd.Series,
                                    out_pdf: str,
                                    logger: logging.Logger,
                                    max_display: int = 10) -> None:
    """
    Waterfall for the median query well (closest to median query vector).

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    shap_values : numpy.ndarray
        SHAP values aligned to X.
    feature_names : list of str
        Feature names.
    target : pandas.Series
        Binary target vector (1=query, 0=neighbour).
    out_pdf : str
        Output PDF path.
    logger : logging.Logger
        Logger for messages.
    max_display : int
        Maximum features to display in the waterfall.
    """
    try:
        q = X[target == 1]
        if q.empty:
            logger.info("No query wells (target==1); skipping waterfall.")
            return
        if len(q) == 1:
            idx = X.index.get_loc(q.index[0])
        else:
            med = q.median(axis=0)
            d = np.linalg.norm(q.values - med.values, axis=1)
            idx = X.index.get_loc(q.index[int(np.argmin(d))])

        plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[idx],
                base_values=0.0,
                data=X.iloc[idx].values,
                feature_names=feature_names,
            ),
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(out_pdf)
        plt.close()
        logger.info("Wrote waterfall: %s", out_pdf)
    except Exception as e:
        logger.warning("Waterfall failed: %s", e)


def plot_heatmap(X: pd.DataFrame,
                 shap_values: np.ndarray,
                 feature_names: List[str],
                 out_pdf: str,
                 logger: logging.Logger,
                 max_display: int = 20,
                 font_size: int = 7) -> None:
    """
    SHAP heatmap (top N features).

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    shap_values : numpy.ndarray
        SHAP values.
    feature_names : list of str
        Feature names.
    out_pdf : str
        Output PDF path.
    logger : logging.Logger
        Logger for messages.
    max_display : int
        Maximum number of features to display.
    font_size : int
        Font size for y tick labels.
    """
    try:
        plt.figure(figsize=(max_display, min(X.shape[0], 40) / 2 + 4))
        shap.plots.heatmap(
            shap.Explanation(values=shap_values, data=X.values, feature_names=feature_names),
            max_display=max_display,
            show=False
        )
        ax = plt.gca()
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(font_size)
        plt.tight_layout()
        plt.savefig(out_pdf)
        plt.close()
        logger.info("Wrote heatmap: %s", out_pdf)
    except Exception as e:
        logger.warning("Heatmap failed: %s", e)


def plot_shap_bar_clustered(X: pd.DataFrame,
                            shap_values: np.ndarray,
                            feature_names: List[str],
                            output_file: str,
                            logger: logging.Logger,
                            max_display: int = 20) -> None:
    """
    Generate a SHAP bar plot using shap.Explanation (cluster-aware ordering).

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    shap_values : numpy.ndarray
        SHAP values aligned to X.
    feature_names : list of str
        Feature names.
    output_file : str
        Output PDF path.
    logger : logging.Logger
        Logger for messages.
    max_display : int
        Maximum number of features to display.
    """
    try:
        plt.figure(figsize=(10, 6))
        shap.plots.bar(
            shap.Explanation(
                values=shap_values,
                data=X.values,
                feature_names=feature_names
            ),
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logger.info("Wrote clustered SHAP bar plot: %s", output_file)
    except Exception as e:
        logger.warning("Clustered SHAP bar plot failed: %s", e)


def plot_shap_dependence_plots(X: pd.DataFrame,
                               shap_values: np.ndarray,
                               top_features: List[str],
                               output_dir: str,
                               query_id: str,
                               logger: logging.Logger,
                               n_dependence: int = 5) -> None:
    """
    Generate and save SHAP dependence plots for the top N features.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    shap_values : numpy.ndarray
        SHAP values aligned to X.
    top_features : list of str
        Top feature names to plot.
    output_dir : str
        Base output directory (a subfolder is created).
    query_id : str
        Query identifier used in filenames/folders.
    logger : logging.Logger
        Logger for messages.
    n_dependence : int
        Number of dependence plots to write.
    """
    compound_dir = os.path.join(output_dir, f"{query_id}_dependence_plots")
    os.makedirs(compound_dir, exist_ok=True)
    try:
        n_to_plot = min(n_dependence, len(top_features))
        for i, feat in enumerate(top_features[:n_to_plot]):
            shap.dependence_plot(feat, shap_values, X, show=False)
            plt.tight_layout()
            out_file = os.path.join(compound_dir, f"{i+1:02d}_{feat}_dependence.pdf")
            plt.savefig(out_file)
            plt.close()
            logger.info("Wrote SHAP dependence plot: %s", out_file)
    except Exception as e:
        logger.warning("Dependence plotting failed for %s: %s", query_id, e)


# ------------------------- Core ----------------------------------------------

def _prepare_X_y(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Select numeric feature columns (excluding known metadata) and target.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Input features with metadata and 'target' column.

    Returns
    -------
    tuple
        (X, y, feature_cols) where X is numeric feature matrix, y is target,
        and feature_cols are the selected feature names.
    """
    numeric_cols = [c for c in features_df.columns
                    if pd.api.types.is_numeric_dtype(features_df[c])]
    feature_cols = [c for c in numeric_cols if c not in META_COLS_DEFAULT]
    X = features_df[feature_cols]
    y = features_df["target"].astype(int)
    return X, y, feature_cols


def _normalise_shap_array(shap_raw: np.ndarray,
                          X: pd.DataFrame,
                          logger: logging.Logger) -> np.ndarray:
    """
    Convert SHAP output to a 2D array aligned with X.

    Parameters
    ----------
    shap_raw : numpy.ndarray or list
        Raw SHAP output from explainer.
    X : pandas.DataFrame
        Feature matrix.
    logger : logging.Logger
        Logger for messages.

    Returns
    -------
    numpy.ndarray
        SHAP values of shape (n_samples, n_features).

    Raises
    ------
    ValueError
        If array cannot be reshaped/selected to match X.
    """
    shap_arr = np.asarray(shap_raw)
    shap_arr = np.squeeze(shap_arr)
    if isinstance(shap_raw, list):
        # If list of arrays (e.g. binary classifier), prefer class 1
        try:
            shap_arr = np.asarray(shap_raw[1])
        except Exception:
            shap_arr = np.asarray(shap_raw[0])

    if shap_arr.ndim == 3 and shap_arr.shape[-1] == 2:
        logger.warning("3D SHAP with class axis detected; taking class 1.")
        shap_arr = shap_arr[:, :, 1]

    if shap_arr.ndim != 2 or shap_arr.shape[1] != X.shape[1]:
        raise ValueError(f"SHAP shape {shap_arr.shape} incompatible with X {X.shape}.")

    return shap_arr


def run_shap(features_df: pd.DataFrame,
             n_top_features: int,
             out_dir: str,
             query_id: str,
             logger: logging.Logger,
             small_sample_threshold: int = 30) -> None:
    """
    Fit simple classifier to distinguish query vs neighbours and compute SHAP.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Features (including 'target' column).
    n_top_features : int
        Number of top/lowest features to report/plot.
    out_dir : str
        Output directory for this query.
    query_id : str
        Query compound identifier.
    logger : logging.Logger
        Logger for messages.
    small_sample_threshold : int
        Use logistic regression if n_samples < this threshold.
    """
    # Select features (numeric only), drop known metadata
    X, y, feature_cols = _prepare_X_y(features_df)

    if not feature_cols:
        logger.error("No numeric feature columns available after excluding metadata.")
        return

    logger.info("X shape: %s; positives=%d, negatives=%d",
                X.shape, int((y == 1).sum()), int((y == 0).sum()))

    # Guardrails
    if X.shape[0] < 4 or y.nunique() < 2 or y.value_counts().min() < 2:
        logger.error("Not enough samples/classes to run SHAP reliably. Skipping.")
        return

    # Choose model and compute SHAP
    try:
        if X.shape[0] < small_sample_threshold:
            logger.info("Using LogisticRegression (n=%d < %d).", X.shape[0], small_sample_threshold)
            model = LogisticRegression(max_iter=2000, n_jobs=None, random_state=42)
            model.fit(X, y)
            explainer = shap.Explainer(model, X)
            shap_raw = explainer(X).values
        else:
            logger.info("Using RandomForest (n=%d).", X.shape[0])
            model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
            shap_raw = sv[1] if isinstance(sv, list) and len(sv) >= 2 else np.asarray(sv)
    except Exception as e:
        logger.error("Model/SHAP failed: %s", e)
        logger.debug(traceback.format_exc())
        return

    # Make SHAP 2D (n_samples, n_features)
    try:
        shap_arr = _normalise_shap_array(shap_raw, X, logger)
    except Exception as e:
        logger.error("Failed to normalise SHAP output: %s", e)
        logger.debug(traceback.format_exc())
        return

    # Rank features by mean |SHAP|
    mean_abs = np.abs(shap_arr).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n_top_features]
    low_idx = np.argsort(mean_abs)[:n_top_features]

    top_feats = [feature_cols[i] for i in top_idx]
    low_feats = [feature_cols[i] for i in low_idx]
    top_vals = mean_abs[top_idx]
    low_vals = mean_abs[low_idx]

    # Outputs
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({"feature": top_feats, "mean_abs_shap": top_vals}) \
        .to_csv(os.path.join(out_dir, f"{query_id}_top_shap_features_driving_difference.tsv"),
                sep="\t", index=False)
    pd.DataFrame({"feature": low_feats, "mean_abs_shap": low_vals}) \
        .to_csv(os.path.join(out_dir, f"{query_id}_most_similar_features.tsv"),
                sep="\t", index=False)

    # Simple bar plots (difference/similarity)
    plot_feature_importance_bar(top_feats, top_vals,
                                os.path.join(out_dir, f"{query_id}_top_shap_features_bar.pdf"),
                                f"{query_id}: Features driving difference", logger)
    plot_feature_importance_bar(low_feats, low_vals,
                                os.path.join(out_dir, f"{query_id}_most_similar_features_bar.pdf"),
                                f"{query_id}: Features driving similarity", logger)

    # SHAP summary plots for full feature set
    plot_summary_pair(X, shap_arr, feature_cols,
                      os.path.join(out_dir, f"{query_id}_shap_summary"), logger,
                      n_top=n_top_features)

    # SHAP summary plots for top/lowest subsets
    X_top = X.iloc[:, top_idx]
    X_low = X.iloc[:, low_idx]
    plot_summary_pair(X_top, shap_arr[:, top_idx], top_feats,
                      os.path.join(out_dir, f"{query_id}_shap_summary_top"), logger,
                      n_top=n_top_features)
    plot_summary_pair(X_low, shap_arr[:, low_idx], low_feats,
                      os.path.join(out_dir, f"{query_id}_shap_summary_similar"), logger,
                      n_top=n_top_features)

    # Clustered bar plots via shap.Explanation
    plot_shap_bar_clustered(
        X_top, shap_arr[:, top_idx], top_feats,
        os.path.join(out_dir, f"{query_id}_shap_bar_clustered_difference.pdf"),
        logger, max_display=n_top_features
    )
    plot_shap_bar_clustered(
        X_low, shap_arr[:, low_idx], low_feats,
        os.path.join(out_dir, f"{query_id}_shap_bar_clustered_similarity.pdf"),
        logger, max_display=n_top_features
    )

    # Waterfall: median query well
    plot_waterfall_for_median_query(
        X, shap_arr, feature_cols, features_df["target"],
        os.path.join(out_dir, f"{query_id}_shap_waterfall_query_median.pdf"),
        logger, max_display=n_top_features
    )

    # Heatmap on full set (top-N display controlled in function)
    plot_heatmap(
        X, shap_arr, feature_cols,
        os.path.join(out_dir, f"{query_id}_shap_heatmap.pdf"),
        logger, max_display=n_top_features, font_size=6
    )

    # Dependence plots for the top-N features
    plot_shap_dependence_plots(
        X, shap_arr, top_feats, out_dir, query_id, logger, n_dependence=min(5, n_top_features)
    )


def main() -> None:
    """
    Entry point: parse arguments, load inputs, and run per-query SHAP analyses.
    """
    parser = argparse.ArgumentParser(description="Explain NN similarity (SHAP) for Cell Painting.")
    parser.add_argument("--features", required=True,
                        help="TSV of well-level features or file-of-files with 'path'.")
    parser.add_argument("--nn_file", required=True, help="Nearest-neighbours TSV.")

    parser.add_argument("--query_id", required=False, default=None,
                        help="Comma list or file with one ID per line.")
    parser.add_argument("--query_ids", nargs="+",
                        help=("Alias for --query_id. Provide one or more IDs; "
                              "names with spaces must be quoted."))

    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--n_neighbors", type=int, default=5,
                        help="Number of neighbours per query.")
    parser.add_argument("--n_top_features", type=int, default=10,
                        help="Top-N features to plot/report.")
    parser.add_argument("--log_file", default="shap_explain.log",
                        help="Log filename (inside output_dir).")
    args = parser.parse_args()

    # Fold --query_ids into --query_id
    if args.query_ids:
        merged = ",".join(args.query_ids)
        args.query_id = merged if args.query_id in (None, "") else f"{args.query_id},{merged}"

    if not args.query_id:
        parser.error("Provide queries via --query_id (comma list or file) or --query_ids.")

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, args.log_file))
    logger.info(sys.version_info)
    logger.info("Arguments: %s", vars(args))

    features_all = load_feature_files(args.features, logger)
    # Ensure we have cpd_id (case-insensitive safeguard)
    if "cpd_id" not in features_all.columns:
        raise ValueError("Feature table(s) must include a 'cpd_id' column.")

    queries = parse_query_ids(args.query_id)
    logger.info("Processing %d queries: %s", len(queries), queries)

    for q in queries:
        logger.info("=== Query: %s ===", q)
        try:
            nn_ids = load_neighbours(args.nn_file, q, args.n_neighbors, logger)
            ids_upper = {s.upper() for s in ([q] + list(nn_ids))}
            subset = features_all[
                features_all["cpd_id"].astype(str).str.upper().isin(ids_upper)
            ].copy()

            if subset.empty:
                logger.warning("No wells found for %s and its NNs in feature table; skipping.", q)
                continue
            subset["target"] = (subset["cpd_id"].astype(str) == str(q)).astype(int)
            out_dir = os.path.join(args.output_dir, q)
            os.makedirs(out_dir, exist_ok=True)
            run_shap(subset, args.n_top_features, out_dir, q, logger)
        except Exception as e:
            logger.error("Failed for query %s: %s", q, e)
            logger.debug(traceback.format_exc())

    logger.info("All SHAP analyses complete.")


if __name__ == "__main__":
    main()
