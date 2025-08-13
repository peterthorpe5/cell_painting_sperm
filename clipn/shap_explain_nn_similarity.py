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
- <query>_top_shap_features_driving_difference.tsv/.pdf
- <query>_most_similar_features.tsv/.pdf
- <query>_shap_summary_bar.pdf and _beeswarm.pdf
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
    """Create a logger that writes DEBUG to file and INFO to stderr."""
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
    """Read CSV/TSV with automatic delimiter detection."""
    return pd.read_csv(path, sep=None, engine="python")


def load_feature_files(list_or_single: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load well-level features.

    If `list_or_single` is a CSV/TSV with 'path' column, read and harmonise all.
    Otherwise treat it as a single TSV of well-level data.
    """
    ext = os.path.splitext(list_or_single)[-1].lower()
    try:
        df_list = pd.read_csv(list_or_single, sep=None, engine="python")
        if "path" in df_list.columns:
            logger.info(f"Detected file-of-files with {len(df_list)} inputs.")
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
            logger.info(f"Loaded & harmonised well-level features: {out.shape}")
            return out
        # Fall through to single TSV if no 'path'
    except Exception as e:
        logger.info(f"Not a file-of-files (or failed to parse as such): {e}. "
                    f"Trying as single TSV…")

    out = _read_table(list_or_single)
    logger.info(f"Loaded single well-level feature file: {list_or_single} shape={out.shape}")
    return out


def parse_query_ids(arg: str) -> List[str]:
    """Comma-separated or file with one ID per line."""
    if os.path.isfile(arg):
        with open(arg) as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    return [x.strip() for x in arg.split(",") if x.strip()]


def load_neighbours(nn_file: str,
                    query_id: str,
                    n_neighbors: int,
                    logger: logging.Logger) -> List[str]:
    """Return top-N neighbour IDs for a given query from an NN table."""
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
    logger.info(f"Found {len(out)} neighbours for {query_id}: {out}")
    return out


# ------------------------- Plot helpers --------------------------------------

def plot_feature_importance_bar(features: Iterable[str],
                                importance: np.ndarray,
                                out_pdf: str,
                                title: str,
                                logger: logging.Logger) -> None:
    """Save a horizontal bar chart for (feature, importance)."""
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
        logger.info(f"Wrote bar plot: {out_pdf}")
    except Exception as e:
        logger.warning(f"Bar plot failed: {e}")


def plot_summary_pair(X: pd.DataFrame,
                      shap_values: np.ndarray,
                      feature_names: List[str],
                      out_prefix: str,
                      logger: logging.Logger,
                      n_top: int = 10) -> None:
    """Write bar + beeswarm SHAP summary plots."""
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
        logger.info(f"Wrote SHAP summary plots: {out_prefix}_bar/beeswarm.pdf")
    except Exception as e:
        logger.warning(f"SHAP summary plotting failed: {e}")


def plot_waterfall_for_median_query(X: pd.DataFrame,
                                    shap_values: np.ndarray,
                                    feature_names: List[str],
                                    target: pd.Series,
                                    out_pdf: str,
                                    logger: logging.Logger,
                                    max_display: int = 10) -> None:
    """Waterfall for the median query well (closest to median query vector)."""
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
        logger.info(f"Wrote waterfall: {out_pdf}")
    except Exception as e:
        logger.warning(f"Waterfall failed: {e}")


def plot_heatmap(X: pd.DataFrame,
                 shap_values: np.ndarray,
                 feature_names: List[str],
                 out_pdf: str,
                 logger: logging.Logger,
                 max_display: int = 20,
                 font_size: int = 7) -> None:
    """SHAP heatmap (top N features)."""
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
        logger.info(f"Wrote heatmap: {out_pdf}")
    except Exception as e:
        logger.warning(f"Heatmap failed: {e}")


# ------------------------- Core ----------------------------------------------

def run_shap(features_df: pd.DataFrame,
             n_top_features: int,
             out_dir: str,
             query_id: str,
             logger: logging.Logger,
             small_sample_threshold: int = 30) -> None:
    """
    Fit simple classifier to distinguish query vs neighbours and compute SHAP.
    """
    # Select features (numeric only), drop known metadata
    numeric_cols = [c for c in features_df.columns
                    if pd.api.types.is_numeric_dtype(features_df[c])]
    feature_cols = [c for c in numeric_cols if c not in META_COLS_DEFAULT]

    if not feature_cols:
        logger.error("No numeric feature columns available after excluding metadata.")
        return

    X = features_df[feature_cols]
    y = features_df["target"].astype(int)
    logger.info(f"X shape: {X.shape}; positives={int((y==1).sum())}, negatives={(y==0).sum()}")

    # Guardrails
    if X.shape[0] < 4 or y.nunique() < 2 or y.value_counts().min() < 2:
        logger.error("Not enough samples/classes to run SHAP reliably. Skipping.")
        return

    # Choose model
    try:
        if X.shape[0] < small_sample_threshold:
            logger.info(f"Using LogisticRegression (n={X.shape[0]} < {small_sample_threshold}).")
            model = LogisticRegression(max_iter=2000, n_jobs=None, random_state=42)
            model.fit(X, y)
            explainer = shap.Explainer(model, X)
            shap_raw = explainer(X).values
        else:
            logger.info(f"Using RandomForest (n={X.shape[0]}).")
            model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
            shap_raw = sv[1] if isinstance(sv, list) and len(sv) >= 2 else np.asarray(sv)
    except Exception as e:
        logger.error(f"Model/SHAP failed: {e}")
        logger.debug(traceback.format_exc())
        return

    # Make SHAP 2D (n_samples, n_features)
    shap_arr = np.asarray(shap_raw)
    if shap_arr.ndim == 3 and shap_arr.shape[-1] == 2:
        logger.warning("3D SHAP with class axis detected; taking class 1.")
        shap_arr = shap_arr[:, :, 1]
    shap_arr = np.squeeze(shap_arr)
    if shap_arr.ndim != 2 or shap_arr.shape[1] != X.shape[1]:
        logger.error(f"SHAP shape {shap_arr.shape} incompatible with X {X.shape}.")
        return
    
    base_value = None
    try:
        ev = explainer.expected_value
        base_value = ev[1] if isinstance(ev, (list, np.ndarray)) and len(np.ravel(ev)) >= 2 else ev
    except Exception:
        base_value = 0.0

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

    plot_feature_importance_bar(top_feats, top_vals,
                                os.path.join(out_dir, f"{query_id}_top_shap_features_bar.pdf"),
                                f"{query_id}: Features driving difference", logger)
    plot_feature_importance_bar(low_feats, low_vals,
                                os.path.join(out_dir, f"{query_id}_most_similar_features_bar.pdf"),
                                f"{query_id}: Features driving similarity", logger)

    plot_summary_pair(X.iloc[:, top_idx], shap_arr[:, top_idx], top_feats,
                      os.path.join(out_dir, f"{query_id}_shap_summary_top"), logger,
                      n_top_features=n_top_features)
    plot_summary_pair(X.iloc[:, low_idx], shap_arr[:, low_idx], low_feats,
                      os.path.join(out_dir, f"{query_id}_shap_summary_similar"), logger,
                      n_top_features=n_top_features)

    # Full summaries / heatmap / waterfall
    plot_summary_pair(X, shap_arr, feature_cols,
                      os.path.join(out_dir, f"{query_id}_shap_summary"), logger,
                      n_top_features=n_top_features)

    plot_waterfall_for_median_query(X, shap_arr, feature_cols, features_df["target"],
                                    os.path.join(out_dir, f"{query_id}_shap_waterfall_query_median.pdf"),
                                    logger, max_display=n_top_features)
    

    plot_heatmap(X, shap_arr, feature_cols,
                 os.path.join(out_dir, f"{query_id}_shap_heatmap.pdf"),
                 logger, max_display=n_top_features, font_size=6)

    # Dependence plots for the top N features
    dep_dir = os.path.join(out_dir, f"{query_id}_dependence_plots")
    os.makedirs(dep_dir, exist_ok=True)
    for i, feat in enumerate(top_feats, start=1):
        try:
            shap.dependence_plot(feat, shap_arr, X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(dep_dir, f"{i:02d}_{feat}_dependence.pdf"))
            plt.close()
        except Exception as e:
            logger.debug(f"Dependence plot failed for {feat}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain NN similarity (SHAP) for Cell Painting.")
    parser.add_argument("--features", required=True, help="TSV of well-level features or file-of-files with 'path'.")
    parser.add_argument("--nn_file", required=True, help="Nearest-neighbours TSV.")

    parser.add_argument("--query_id", required=False, default=None,
                    help="Comma list or file with one ID per line.")
    parser.add_argument("--query_ids", nargs="+",
                    help="Alias for --query_id. Provide one or more IDs; "
                         "names with spaces must be quoted.")   #  default="DDD02387619,DDD02948916,DDD02955130,DDD02958365",

    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbours per query.")
    parser.add_argument("--n_top_features", type=int, default=10, help="Top-N features to plot/report.")
    parser.add_argument("--log_file", default="shap_explain.log", help="Log filename (inside output_dir).")
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
    logger.info(f"Processing {len(queries)} queries: {queries}")

    for q in queries:
        logger.info(f"=== Query: {q} ===")
        try:
            nn_ids = load_neighbours(args.nn_file, q, args.n_neighbors, logger)
            ids_upper = {s.upper() for s in ([q] + list(nn_ids))}
            subset = features_all[
                features_all["cpd_id"].astype(str).str.upper().isin(ids_upper)].copy()

            if subset.empty:
                logger.warning(f"No wells found for {q} and its NNs in feature table; skipping.")
                continue
            subset["target"] = (subset["cpd_id"].astype(str) == str(q)).astype(int)
            out_dir = os.path.join(args.output_dir, q)
            os.makedirs(out_dir, exist_ok=True)
            run_shap(subset, args.n_top_features, out_dir, q, logger)
        except Exception as e:
            logger.error(f"Failed for query {q}: {e}")
            logger.debug(traceback.format_exc())

    logger.info("All SHAP analyses complete.")


if __name__ == "__main__":
    main()
