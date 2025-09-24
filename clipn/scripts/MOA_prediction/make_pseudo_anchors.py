#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create pseudo-anchors (unsupervised clusters) for prototype MOA scoring.
-------------------------------------------------------------------------

This script clusters aggregated compound embeddings and writes:
1) anchors TSV mapping each compound to a pseudo-MOA cluster;
2) a one-row run summary TSV with algorithm/quality metrics;
3) a per-cluster stats TSV including per-cluster cosine silhouette and
   nearest-centroid cosine (and nearest cluster id).

- Input can be well-level; we aggregate to compound using a robust method.
- Clustering:
    * 'kmeans' (default): supports fixed k or auto-k via cosine silhouette.
    * 'auto': HDBSCAN-first; if unavailable/degenerate, fall back to KMeans auto-k.
    * 'hdbscan': force HDBSCAN (error if not installed).

Outputs (TSV; never comma-separated)
------------------------------------
- anchors_pseudo.tsv          : columns [id_col, moa, cluster_id]
- anchors_pseudo_summary.tsv  : one row of run metadata and QC
- anchors_pseudo_clusters.tsv : per-cluster stats:
    [cluster_id, moa, n_members, silhouette_cosine,
     nearest_cluster_id, nearest_centroid_cosine]

In this script, silhouette is a quality score for clustering computed with cosine distance on the L2-normalised embeddings.
     
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------- maths helpers ------------------------------- #

def l2_normalise(*, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalisation.

    Parameters
    ----------
    X : np.ndarray
        Matrix (n_samples × n_features).
    eps : float, optional
        Numerical stabiliser added to norms, by default 1e-12.

    Returns
    -------
    np.ndarray
        Row-normalised matrix.
    """
    nrm = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    return X / nrm


def geometric_median(*, X: np.ndarray, max_iter: int = 256, tol: float = 1e-6) -> np.ndarray:
    """
    Compute the geometric median of points using Weiszfeld's algorithm.

    Parameters
    ----------
    X : np.ndarray
        Points of shape (n_points, n_features).
    max_iter : int, optional
        Maximum iterations, by default 256.
    tol : float, optional
        Convergence tolerance, by default 1e-6.

    Returns
    -------
    np.ndarray
        1D array of length n_features representing the geometric median.
    """
    if X.shape[0] == 1:
        return X[0].copy()
    y = X.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - y, axis=1)
        if np.any(d < 1e-12):
            return X[np.argmin(d)].copy()
        w = 1.0 / d
        y_new = np.average(X, axis=0, weights=w)
        if np.linalg.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y


def trimmed_mean(*, X: np.ndarray, trim_frac: float = 0.1) -> np.ndarray:
    """
    Compute a per-feature trimmed mean.

    Parameters
    ----------
    X : np.ndarray
        Matrix of shape (n_points, n_features).
    trim_frac : float, optional
        Fraction to trim at each tail, by default 0.1.

    Returns
    -------
    np.ndarray
        1D trimmed-mean vector.
    """
    if X.shape[0] == 1 or trim_frac <= 0:
        return X.mean(axis=0)
    lo = int(np.floor(trim_frac * X.shape[0]))
    hi = int(np.ceil((1 - trim_frac) * X.shape[0]))
    Xs = np.sort(X, axis=0)
    Xc = Xs[lo:hi, :]
    return Xc.mean(axis=0)


# -------------------------------- I/O helpers -------------------------------- #

def detect_id_column(*, df: pd.DataFrame, id_col: Optional[str]) -> str:
    """
    Detect or validate the identifier column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_col : Optional[str]
        Desired id column name, if provided.

    Returns
    -------
    str
        The resolved identifier column name.

    Raises
    ------
    ValueError
        If no suitable id column is found.
    """
    if id_col is not None:
        if id_col in df.columns:
            return id_col
        raise ValueError(f"Identifier column '{id_col}' not found.")
    for c in ["cpd_id", "compound_id", "Compound", "compound", "QueryID", "id"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect an identifier column (tried common candidates).")


# ------------------------ replicate aggregation (to compounds) ---------------- #

def aggregate_compounds(
    *,
    df: pd.DataFrame,
    id_col: str,
    method: str = "median",
    trimmed_frac: float = 0.1,
) -> pd.DataFrame:
    """
    Aggregate replicate rows per compound into a single embedding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least id_col and numeric embedding columns.
    id_col : str
        Identifier column for grouping (e.g., 'cpd_id').
    method : str, optional
        Aggregation method: 'median', 'mean', 'trimmed_mean', 'geometric_median',
        by default 'median'.
    trimmed_frac : float, optional
        Trimming fraction for 'trimmed_mean', by default 0.1.

    Returns
    -------
    pd.DataFrame
        One row per compound with aggregated numeric columns and the id_col.
    """
    num_cols = numeric_feature_columns(df)
    rows: List[Dict[str, float]] = []
    for cid, sub in df.groupby(by=id_col, sort=False):
        X = sub[num_cols].to_numpy()
        if method == "median":
            vec = np.median(X, axis=0)
        elif method == "mean":
            vec = np.mean(X, axis=0)
        elif method == "trimmed_mean":
            vec = trimmed_mean(X=X, trim_frac=trimmed_frac)
        elif method == "geometric_median":
            vec = geometric_median(X=X)
        else:
            raise ValueError(f"Unknown aggregate_method '{method}'")
        rows.append({"__id__": cid, **{c: float(v) for c, v in zip(num_cols, vec)}})
    out = pd.DataFrame(data=rows).rename(columns={"__id__": id_col})
    return out[[id_col] + num_cols]


# ------------------------------- clustering utils ---------------------------- #

def cosine_silhouette_scores(*, X: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute overall and per-sample cosine silhouette scores.

    Parameters
    ----------
    X : np.ndarray
        Normalised data (n_samples × n_features).
    labels : np.ndarray
        Cluster labels (length n_samples).

    Returns
    -------
    Tuple[float, np.ndarray]
        (overall_silhouette, per_sample_silhouette) where the first is NaN if
        there are <2 clusters with at least 2 members.
    """
    try:
        from sklearn.metrics import silhouette_score, silhouette_samples
    except Exception as exc:
        raise SystemExit("scikit-learn is required for silhouette computation.") from exc

    uniq = np.unique(labels)
    # Need at least two clusters with >=2 members
    valid_clusters = [u for u in uniq if (labels == u).sum() >= 2]
    if len(valid_clusters) < 2:
        return float("nan"), np.full(shape=(X.shape[0],), fill_value=np.nan, dtype=float)

    overall = float(silhouette_score(X=X, labels=labels, metric="cosine"))
    per_sample = silhouette_samples(X=X, labels=labels, metric="cosine")
    return overall, per_sample


def try_hdbscan(
    *,
    X: np.ndarray,
    min_cluster_size: Optional[int],
    min_samples: Optional[int],
) -> Optional[np.ndarray]:
    """
    Attempt HDBSCAN clustering. Returns labels or None if unavailable/fails.

    Parameters
    ----------
    X : np.ndarray
        Normalised data (n_samples × n_features).
    min_cluster_size : Optional[int]
        HDBSCAN min_cluster_size. If None, a heuristic is used.
    min_samples : Optional[int]
        HDBSCAN min_samples. If None, HDBSCAN's default is used.

    Returns
    -------
    Optional[np.ndarray]
        Labels array of length n_samples (with -1 for noise) or None on failure.
    """
    try:
        import hdbscan  # type: ignore
    except Exception:
        return None

    n = X.shape[0]
    mcs = int(max(5, min_cluster_size if min_cluster_size is not None else np.ceil(0.01 * n)))
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, prediction_data=False)
        labels = clusterer.fit_predict(X)
        return labels.astype(int)
    except Exception:
        return None


def handle_noise(
    *,
    X: np.ndarray,
    labels: np.ndarray,
    strategy: str = "nearest",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Handle HDBSCAN noise labels (-1) according to a chosen strategy.

    Parameters
    ----------
    X : np.ndarray
        Normalised data (n_samples × n_features).
    labels : np.ndarray
        Labels with possible -1 values for noise.
    strategy : str, optional
        One of {"own_cluster", "drop", "nearest"}, by default "nearest".

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        (labels_out, keep_mask)
        - labels_out: updated labels; for "drop" the returned labels have no
          sentinel codes; for others length equals X.shape[0].
        - keep_mask: for "drop", a boolean mask of kept rows; otherwise None.
    """
    assert strategy in {"own_cluster", "drop", "nearest"}
    if (-1 not in labels):
        return labels, None

    noise_mask = labels == -1
    if strategy == "drop":
        keep_mask = ~noise_mask
        return labels[keep_mask], keep_mask

    if strategy == "own_cluster":
        lbls = labels.copy()
        non_noise = lbls != -1
        start = (lbls[non_noise].max(initial=-1) + 1) if non_noise.any() else 0
        lbls[noise_mask] = np.arange(start, start + noise_mask.sum(), dtype=int)
        return lbls, None

    # "nearest": assign noise to nearest non-noise centroid by cosine
    lbls = labels.copy()
    non_noise_mask = lbls != -1
    if not non_noise_mask.any():
        # All noise: make each its own cluster
        start = 0
        lbls[noise_mask] = np.arange(start, start + noise_mask.sum(), dtype=int)
        return lbls, None

    uniq = np.unique(lbls[non_noise_mask])
    centroids = []
    for u in uniq:
        idx = np.where(lbls == u)[0]
        c = X[idx, :].mean(axis=0)
        nrm = np.linalg.norm(c)
        centroids.append(c / nrm if nrm > 0 else c)
    C = np.vstack(centroids)
    sims = X[noise_mask, :] @ C.T
    assign = np.argmax(sims, axis=1)
    lbls[noise_mask] = uniq[assign]
    return lbls, None

def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the columns that are actual embedding features.

    Priority:
    1) Columns named as plain integers: "0","1","2",...
    2) Else, columns starting with "feat_" followed by digits.
    3) Else, fallback to numeric dtype but EXCLUDE common metadata names.

    This prevents leaking columns like 'Sample' into the embedding.
    """
    cols = [str(c) for c in df.columns]

    # 1) strictly-integer column names
    int_like = [c for c in cols if c.isdigit()]
    if int_like:
        return sorted(int_like, key=lambda s: int(s))

    # 2) feat_### pattern
    import re
    feat_like = [c for c in cols if re.fullmatch(r"feat_\d+", c)]
    if feat_like:
        # keep order as they appear
        return feat_like

    # 3) fallback: numeric dtype minus known metadata
    meta_blocklist = {
        "sample", "dataset", "plate", "well", "row", "col",
        "time", "replicate", "dose", "concentration",
        "cpd_id", "compound_id", "id"
    }
    num_cols = df.select_dtypes(include=[np.number]).columns
    keep = [c for c in num_cols if c.lower() not in meta_blocklist]
    return keep



def overlay_given_labels(
    *,
    clusters_df: Optional[pd.DataFrame] = None,
    anchors: Optional[pd.DataFrame] = None,
    labels_tsv: str,
    id_col: str,
    labels_id_col: Optional[str] = None,
    labels_label_col: str = "label",
) -> pd.DataFrame:
    """
    Merge user-provided labels onto cluster assignments and derive a final MOA.

    Backwards compatible with previous usage:
    `_overlay_given_labels(anchors=..., labels_tsv=..., id_col=..., labels_id_col=..., labels_label_col=...)`
    or the newer `clusters_df=...`.

    Parameters
    ----------
    clusters_df : Optional[pd.DataFrame]
        DataFrame with per-compound cluster assignment. Must contain at least
        [id_col, "moa"]. If available, "cluster_id" will also be preserved.
    anchors : Optional[pd.DataFrame]
        Alias for `clusters_df` (for backwards compatibility).
    labels_tsv : str
        Path to a TSV with at least [id_col (or labels_id_col), labels_label_col].
        May contain multiple rows per compound; these will be collapsed.
    id_col : str
        Identifier column name used in `clusters_df` / embeddings (e.g., 'cpd_id').
    labels_id_col : Optional[str]
        Column name in labels_tsv that corresponds to the identifier. If None,
        defaults to `id_col`.
    labels_label_col : str
        Column in labels_tsv containing the human-provided label (default 'label').

    Returns
    -------
    pd.DataFrame
        Columns: [id_col, "moa", "cluster_id", "given_label", "is_labelled", "moa_final"].
    """
    # Accept both parameter names for the clustered table
    if clusters_df is None and anchors is None:
        raise ValueError("Provide either clusters_df or anchors.")
    df = clusters_df if clusters_df is not None else anchors
    df = df.copy()

    # Ensure id is a proper column (not only an index)
    if df.index.name == id_col:
        df = df.reset_index()
    if id_col not in df.columns:
        if df.index.name is None:
            df = df.reset_index().rename(columns={"index": id_col})
        else:
            raise KeyError(f"'{id_col}' is neither a column nor the index in the cluster table.")

    # Normalise id and guarantee 'cluster_id'
    df[id_col] = df[id_col].astype(str).str.strip()
    if "cluster_id" not in df.columns:
        df["cluster_id"] = pd.NA

    # Load labels
    lab = pd.read_csv(labels_tsv, sep="\t", dtype=str, keep_default_na=False, na_values=[""], 
                        low_memory=False)
    src_id = labels_id_col if labels_id_col is not None else id_col
    if src_id not in lab.columns:
        raise ValueError(
            f"Labels file '{labels_tsv}' must contain an id column "
            f"('{src_id}' or '{id_col}'). Found columns: {list(lab.columns)}"
        )
    if labels_label_col not in lab.columns:
        raise ValueError(
            f"Labels file '{labels_tsv}' must contain label column '{labels_label_col}'. "
            f"Found columns: {list(lab.columns)}"
        )

    if src_id != id_col:
        lab = lab.rename(columns={src_id: id_col})
    lab[id_col] = lab[id_col].astype(str).str.strip()
    lab[labels_label_col] = lab[labels_label_col].astype(str).str.strip()

    # Collapse multiple labels per compound (keep unique; stable order)
    lab_agg = (
        lab.groupby(id_col, sort=False)[labels_label_col]
           .apply(lambda s: "; ".join(pd.unique([x for x in s.values if x != ""])))
           .reset_index()
           .rename(columns={labels_label_col: "given_label"})
    )

    # Note any labels that do not appear in clustered set
    missing = set(lab_agg[id_col]) - set(df[id_col])
    if missing:
        print(f"NOTE: {len(missing)} labelled compounds not in clustered set; ignoring.", flush=True)

    # Join
    out = df.merge(lab_agg, on=id_col, how="left", validate="one_to_one")

    # Flags
    out["given_label"] = out["given_label"].replace("", np.nan)
    out["is_labelled"] = out["given_label"].notna()

    # Final MOA: keep cluster MOA and append label(s) if present and different.
    base = out["moa"].astype(str).str.strip().fillna("")
    labv = out["given_label"].astype(str).str.strip().fillna("")
    same = base.str.lower() == labv.str.lower()
    out["moa_final"] = np.where(
        (labv.eq("")) | same,
        base,
        base + " | " + labv,
    )

    return out[[id_col, "moa", "cluster_id", "given_label", "is_labelled", "moa_final"]]




def auto_kmeans(
    *,
    X: np.ndarray,
    random_seed: int,
    min_k: int,
    max_k: int,
    sample_size: int = 2000,
) -> Tuple[np.ndarray, int]:
    """
    Run KMeans with auto selection of k via cosine silhouette on a subsample.

    Parameters
    ----------
    X : np.ndarray
        Normalised data (n_samples × n_features).
    random_seed : int
        Random seed.
    min_k : int
        Minimum k to consider (≥2).
    max_k : int
        Maximum k to consider (≥ min_k + 1).
    sample_size : int, optional
        Subsample size for silhouette evaluation, by default 2000.

    Returns
    -------
    Tuple[np.ndarray, int]
        (labels, chosen_k)
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    if n < 3:
        return np.zeros(shape=(n,), dtype=int), 1

    # Candidate ks
    ks: List[int] = sorted(set([
        max(2, min_k),
        int(round(np.sqrt(n))),
        int(round(1.5 * np.sqrt(n))),
        int(round(2.0 * np.sqrt(n))),
        min(max_k, max(2, n - 1)),
    ]))
    ks = [k for k in ks if 2 <= k <= min(max_k, n - 1)]
    if not ks:
        ks = [min(2, n - 1)]

    rng = np.random.default_rng(random_seed)
    sample_idx = rng.choice(n, size=min(n, sample_size), replace=False)

    best_k: Optional[int] = None
    best_score = -np.inf
    best_labels: Optional[np.ndarray] = None

    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_seed, n_init="auto")
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X=X[sample_idx, :], labels=labels[sample_idx], metric="cosine")
        if (score > best_score) or (np.isclose(score, best_score) and (best_k is None or k < best_k)):
            best_score = float(score)
            best_k = int(k)
            best_labels = labels

    if best_labels is None:
        # Fallback: k=2
        from sklearn.cluster import KMeans as _KMeans
        km = _KMeans(n_clusters=2, random_state=random_seed, n_init="auto")
        best_labels = km.fit_predict(X)
        best_k = 2

    return best_labels, int(best_k)


# ----------------------------------- main ------------------------------------ #

def main() -> None:
    """
    Parse arguments and write pseudo-anchors TSVs (anchors, summary, clusters).
    """
    parser = argparse.ArgumentParser(description="Make pseudo-anchors by clustering embeddings (TSV in/out, UK English).")
    parser.add_argument("--embeddings_tsv", type=str, required=True, help="TSV with embeddings (well- or compound-level).")
    parser.add_argument("--out_anchors_tsv", type=str, required=True, help="Output TSV path for pseudo-anchors.")
    parser.add_argument("--out_summary_tsv", type=str, required=True, help="Output TSV path for run summary.")
    parser.add_argument("--out_clusters_tsv", type=str, required=True, help="Output TSV path for per-cluster stats.")
    parser.add_argument("--id_col", type=str, default="cpd_id", help="Identifier column name (default: cpd_id).")

    parser.add_argument("--labels_tsv", type=str, default=None,
                        help="Optional TSV mapping IDs to labels for overlay (e.g., cpd_id\\tlabel).")
    parser.add_argument("--labels_id_col", type=str, default="cpd_id",
                        help="ID column name in labels TSV (default: cpd_id).")
    parser.add_argument("--labels_label_col", type=str, default="label",
                        help="Label column name in labels TSV (default: label).")

    parser.add_argument("--aggregate_method", type=str, default="median",
                        choices=["median", "mean", "trimmed_mean", "geometric_median"],
                        help="Replicate aggregation method (default: median).")
    parser.add_argument("--trimmed_frac", type=float, default=0.1, help="Trim fraction for trimmed_mean (default: 0.1).")

    parser.add_argument("--clusterer", type=str, default="kmeans",
                        choices=["kmeans", "auto", "hdbscan"],
                        help="Clustering algorithm: 'kmeans' (default), 'auto' (HDBSCAN-first), or 'hdbscan'.")
    parser.add_argument("--n_clusters", type=int, default=-1, help="KMeans clusters (-1 for auto-k).")
    parser.add_argument("--auto_min_clusters", type=int, default=8, help="Minimum k considered in auto-k (default: 8).")
    parser.add_argument("--auto_max_clusters", type=int, default=64, help="Maximum k considered in auto-k (default: 64).")

    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=-1, help="HDBSCAN min_cluster_size (<=0 → heuristic).")
    parser.add_argument("--hdbscan_min_samples", type=int, default=-1, help="HDBSCAN min_samples (<=0 → default).")
    parser.add_argument("--hdbscan_noise", type=str, default="nearest",
                        choices=["own_cluster", "drop", "nearest"],
                        help="How to handle HDBSCAN noise (default: nearest).")

    parser.add_argument("--hdbscan_min_silhouette", type=float, default=0.02,
                        help="If HDBSCAN cosine silhouette < this, fall back to KMeans.")
    parser.add_argument("--max_singleton_frac", type=float, default=0.15,
                        help="If fraction of singleton clusters > this, fall back to KMeans.")
    parser.add_argument("--silhouette_sample_size", type=int, default=2000,
                        help="Subsample size for auto-k silhouette scoring (default: 2000).")

    parser.add_argument("--random_seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    # Load embeddings and aggregate to compounds
    df = pd.read_csv(args.embeddings_tsv, sep="\t", low_memory=False)
    id_col = detect_id_column(df=df, id_col=args.id_col)

    agg = aggregate_compounds(
        df=df,
        id_col=id_col,
        method=args.aggregate_method,
        trimmed_frac=args.trimmed_frac,
    )
    # num_cols = agg.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = numeric_feature_columns(df)

    X_all = l2_normalise(X=agg[num_cols].to_numpy())
    ids_all = agg[id_col].astype(str).tolist()

    n_input = int(len(ids_all))

    # Choose clustering path
    chosen_algo = None
    labels: Optional[np.ndarray] = None
    keep_mask: Optional[np.ndarray] = None
    dropped_count = 0
    k_if_kmeans = -1
    noise_handling = "n/a"
    silhouette_overall = float("nan")

    if args.clusterer in {"auto", "hdbscan"}:
        # Try HDBSCAN
        labels_h = try_hdbscan(
            X=X_all,
            min_cluster_size=(None if args.hdbscan_min_cluster_size <= 0 else args.hdbscan_min_cluster_size),
            min_samples=(None if args.hdbscan_min_samples <= 0 else args.hdbscan_min_samples),
        )
        if labels_h is not None:
            lbls, keep_mask = handle_noise(X=X_all, labels=labels_h, strategy=args.hdbscan_noise)
            # Quality checks
            uniq, counts = np.unique(lbls, return_counts=True)
            singleton_frac = float((counts == 1).sum()) / float(counts.sum())

            try:
                sil_overall, sil_samples = cosine_silhouette_scores(X=(X_all if keep_mask is None else X_all[keep_mask, :]),
                                                                    labels=lbls)
            except SystemExit:
                sil_overall, sil_samples = float("nan"), np.full(lbls.shape, np.nan)

            ok = True
            if np.isnan(sil_overall) or (sil_overall < float(args.hdbscan_min_silhouette)):
                ok = False
            if singleton_frac > float(args.max_singleton_frac):
                ok = False

            if ok:
                labels = lbls
                chosen_algo = "hdbscan"
                silhouette_overall = sil_overall
                noise_handling = args.hdbscan_noise
                if keep_mask is not None:
                    dropped_count = int((~keep_mask).sum())
            # else: fall through to kmeans
        # elif: fall through to kmeans

    if (labels is None) and (args.clusterer in {"auto", "kmeans"}):
        # KMeans path (auto-k or fixed)
        from sklearn.cluster import KMeans
        if int(args.n_clusters) == -1:
            labels_k, k_chosen = auto_kmeans(
                X=X_all,
                random_seed=args.random_seed,
                min_k=args.auto_min_clusters,
                max_k=args.auto_max_clusters,
                sample_size=args.silhouette_sample_size,
            )
            labels = labels_k
            k_if_kmeans = int(k_chosen)
        else:
            k_fixed = max(2, min(int(args.n_clusters), X_all.shape[0]))
            km = KMeans(n_clusters=k_fixed, random_state=args.random_seed, n_init="auto")
            labels = km.fit_predict(X=X_all)
            k_if_kmeans = int(k_fixed)
        chosen_algo = "kmeans"
        keep_mask = None  # we used all rows
        noise_handling = "n/a"
        # Overall silhouette on full data
        try:
            sil_overall, _ = cosine_silhouette_scores(X=X_all, labels=labels)
        except SystemExit:
            sil_overall = float("nan")
        silhouette_overall = sil_overall

    if labels is None:
        raise SystemExit("Clustering failed in all modes.")

    # If 'drop' strategy was used, subset data and ids
    if keep_mask is not None:
        X = X_all[keep_mask, :]
        ids = [cid for cid, keep in zip(ids_all, keep_mask) if bool(keep)]
    else:
        X = X_all
        ids = ids_all

    # Reindex cluster ids to 0..C-1 and build Cluster_#### names
    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(sorted(uniq))}
    cl_ids = np.array([remap[int(l)] for l in labels], dtype=int)
    cluster_names = {new: f"C_{new + 1:02d}" for new in sorted(remap.values())}
    moa = [cluster_names[int(c)] for c in cl_ids]

    # Per-sample silhouette (for per-cluster mean)
    try:
        _, sil_samples = cosine_silhouette_scores(X=X, labels=cl_ids)
    except SystemExit:
        sil_samples = np.full(shape=(X.shape[0],), fill_value=np.nan, dtype=float)

    # Centroids for nearest-centroid cosine
    centroids = []
    for k in sorted(cluster_names.keys()):
        idx = np.where(cl_ids == k)[0]
        if idx.size == 0:
            centroids.append(np.zeros(X.shape[1], dtype=float))
            continue
        c = X[idx, :].mean(axis=0)
        nrm = np.linalg.norm(c)
        centroids.append((c / nrm) if nrm > 0 else c)
    C = np.vstack(centroids) if centroids else np.zeros((0, X.shape[1]), dtype=float)

    # Pairwise centroid cosine (avoid self)
    if C.shape[0] >= 2:
        M = C @ C.T
        np.fill_diagonal(M, -np.inf)
        nearest_idx = np.argmax(M, axis=1)
        nearest_cos = M[np.arange(M.shape[0]), nearest_idx]
    else:
        nearest_idx = np.full(shape=(C.shape[0],), fill_value=-1, dtype=int)
        nearest_cos = np.full(shape=(C.shape[0],), fill_value=np.nan, dtype=float)

    # Build cluster stats table
    cluster_rows: List[Dict[str, object]] = []
    for k in sorted(cluster_names.keys()):
        idx = np.where(cl_ids == k)[0]
        n_members = int(idx.size)
        sil = float(np.nan) if (sil_samples is None or n_members == 0) else float(np.nanmean(sil_samples[idx]))
        near_id = int(nearest_idx[k]) if (C.shape[0] >= 2) else -1
        near_cos = float(nearest_cos[k]) if (C.shape[0] >= 2) else float("nan")
        cluster_rows.append({
            "cluster_id": k,
            "moa": cluster_names[k],
            "n_members": n_members,
            "silhouette_cosine": sil,
            "nearest_cluster_id": near_id,
            "nearest_centroid_cosine": near_cos,
        })
    clusters_df = pd.DataFrame(cluster_rows)

    # Anchors table
    anchors_df = pd.DataFrame({id_col: ids, "moa": moa, "cluster_id": cl_ids})
    anchors_df = anchors_df[[id_col, "moa", "cluster_id"]]
    anchors_df = overlay_given_labels(
                                anchors=anchors_df,
                                labels_tsv=args.labels_tsv,
                                id_col=id_col,
                                labels_id_col=args.labels_id_col,
                                labels_label_col=args.labels_label_col,
                            )


    # Summary row
    summary = pd.DataFrame([{
        "algorithm": chosen_algo,
        "clusterer_requested": args.clusterer,
        "k_if_kmeans": k_if_kmeans,
        "num_clusters": int(len(cluster_names)),
        "n_samples_input": n_input,
        "n_samples_used": int(X.shape[0]),
        "dropped_count": int(dropped_count),
        "noise_handling": noise_handling,
        "silhouette_cosine": float(silhouette_overall),
        "aggregate_method": args.aggregate_method,
        "trimmed_frac": float(args.trimmed_frac),
        "auto_min_clusters": int(args.auto_min_clusters),
        "auto_max_clusters": int(args.auto_max_clusters),
        "n_clusters_requested": int(args.n_clusters),
        "hdbscan_min_cluster_size": int(args.hdbscan_min_cluster_size),
        "hdbscan_min_samples": int(args.hdbscan_min_samples),
        "hdbscan_min_silhouette": float(args.hdbscan_min_silhouette),
        "max_singleton_frac": float(args.max_singleton_frac),
        "random_seed": int(args.random_seed),
    }])

    # Write outputs (TSV only)
    Path(args.out_anchors_tsv).parent.mkdir(parents=True, exist_ok=True)
    anchors_df.to_csv(args.out_anchors_tsv, sep="\t", index=False)
    clusters_df.to_csv(args.out_clusters_tsv, sep="\t", index=False)
    summary.to_csv(args.out_summary_tsv, sep="\t", index=False)


if __name__ == "__main__":
    main()
