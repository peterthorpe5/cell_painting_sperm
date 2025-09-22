#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""2D MOA map (PCA/UMAP) with compounds, centroids, and regions.

This script projects compound embeddings and MOA centroids to 2D and renders:
1) A static pdf (always).
2) A best-effort interactive Plotly HTML (attempted every run; skipped gracefully
   if Plotly or file writing is unavailable).

Data sources (in order of preference):
- --moa_dir: reads `<moa_dir>/compound_embeddings.tsv` to ensure parity with
  the embeddings used in your MoA scoring step.
- --embeddings_tsv: if --moa_dir is not provided, aggregates the raw embeddings.

Centroids are rebuilt from `--anchors_tsv` using the same options you used in
scoring (median/mean, optional sub-centroids via k-means, optional shrinkage).
Compounds are coloured either by provided predictions (`--predictions_tsv`) or
by nearest centroid (cosine/CSLS).

Outputs
-------
<out_prefix>.pdf   Static figure (always).
<out_prefix>.html  Interactive Plotly figure (best effort).

Example
-------
python plot_moa_centroids_2d.py \
  --moa_dir path/to/moa_out \
  --anchors_tsv path/to/pseudo_anchors.tsv \
  --assignment predictions \
  --predictions_tsv path/to/moa_out/compound_predictions.tsv \
  --projection umap \
  --n_centroids_per_moa 1 \
  --centroid_method median \
  --adaptive_shrinkage --adaptive_shrinkage_c 0.5 --adaptive_shrinkage_max 0.3 \
  --out_prefix path/to/moa_out/moa_map \
  --random_seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import textwrap
from collections import Counter



# --------------------------------------------------------------------------- #
# Math utilities
# --------------------------------------------------------------------------- #

def l2_normalise(*, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation.

    Parameters
    ----------
    X
        Array of shape (n_samples, n_features).
    eps
        Small stabiliser to avoid division by zero.

    Returns
    -------
    np.ndarray
        L2-normalised array with the same shape as `X`.
    """
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n


def trimmed_mean(*, X: np.ndarray, trim_frac: float = 0.1) -> np.ndarray:
    """Per-feature trimmed mean.

    Parameters
    ----------
    X
        Array of shape (n_rows, n_features).
    trim_frac
        Fraction to trim from each tail per feature (0..0.5).

    Returns
    -------
    np.ndarray
        Vector of length n_features containing the trimmed mean.
    """
    if X.shape[0] == 1 or trim_frac <= 0:
        return X.mean(axis=0)
    lo = int(np.floor(trim_frac * X.shape[0]))
    hi = int(np.ceil((1 - trim_frac) * X.shape[0]))
    Xs = np.sort(X, axis=0)
    return Xs[lo:hi, :].mean(axis=0)


def geometric_median(*, X: np.ndarray, max_iter: int = 256, tol: float = 1e-6) -> np.ndarray:
    """Geometric median via Weiszfeld's algorithm.

    Parameters
    ----------
    X
        Array of shape (n_rows, n_features).
    max_iter
        Maximum number of iterations.
    tol
        Convergence tolerance on the update step.

    Returns
    -------
    np.ndarray
        Vector of length n_features representing the geometric median.
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


# --------------------------------------------------------------------------- #
# Embeddings / centroids
# --------------------------------------------------------------------------- #

def detect_id_column(*, df: pd.DataFrame, id_col: Optional[str]) -> str:
    """Validate or detect the identifier column.

    Parameters
    ----------
    df
        Input DataFrame.
    id_col
        Explicit identifier column name; if None, try common candidates.

    Returns
    -------
    str
        Resolved identifier column name.

    Raises
    ------
    ValueError
        If no suitable identifier column is found.
    """
    if id_col is not None:
        if id_col in df.columns:
            return id_col
        raise ValueError(f"Identifier column '{id_col}' not found.")
    for c in ["cpd_id", "compound_id", "Compound", "compound", "QueryID", "id"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect an identifier column.")


def aggregate_compounds(
    *,
    df: pd.DataFrame,
    id_col: str,
    method: str = "median",
    trimmed_frac: float = 0.1,
) -> pd.DataFrame:
    """Aggregate replicate rows per compound into one embedding.

    Parameters
    ----------
    df
        DataFrame containing `id_col` and numeric feature columns.
    id_col
        Identifier column for grouping.
    method
        One of {"median", "mean", "trimmed_mean", "geometric_median"}.
    trimmed_frac
        Tail fraction for `trimmed_mean`.

    Returns
    -------
    pd.DataFrame
        One row per compound with the aggregated numeric columns.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows: List[Dict[str, float]] = []
    for cid, sub in df.groupby(id_col, sort=False):
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
    out = pd.DataFrame(rows).rename(columns={"__id__": id_col})
    return out[[id_col] + num_cols]


def build_moa_centroids(
    *,
    embeddings: pd.DataFrame,
    anchors: pd.DataFrame,
    id_col: str,
    moa_col: str = "moa",
    n_centroids_per_moa: int = 1,
    centroid_method: str = "median",
    centroid_shrinkage: float = 0.0,
    min_members_per_moa: int = 1,
    skip_tiny_moas: bool = False,
    adaptive_shrinkage: bool = False,
    adaptive_shrinkage_c: float = 0.5,
    adaptive_shrinkage_max: float = 0.3,
    random_seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Build one or more centroids per MOA.

    Parameters
    ----------
    embeddings
        Aggregated compound embeddings (one row per id).
    anchors
        Table with columns [id_col, moa_col] for labelled anchor compounds.
    id_col
        Identifier column name.
    moa_col
        MOA label column in `anchors`.
    n_centroids_per_moa
        If >1, run k-means within each MOA to create sub-centroids.
    centroid_method
        "median" or "mean" (used when n_centroids_per_moa <= 1).
    centroid_shrinkage
        Baseline shrinkage towards the global mean (0..1).
    min_members_per_moa
        Minimum labelled members to form a centroid.
    skip_tiny_moas
        If True, MOAs with < min_members_per_moa are skipped entirely.
    adaptive_shrinkage
        If True, add `min(adaptive_shrinkage_max, adaptive_shrinkage_c / n_members)`
        to the baseline shrinkage.
    adaptive_shrinkage_c
        Constant `C` for the size-aware shrinkage term.
    adaptive_shrinkage_max
        Maximum extra shrinkage due to the size-aware term.
    random_seed
        Random seed for sub-clustering.

    Returns
    -------
    tuple
        (summary_df, P, centroid_moas):
        - summary_df: per-centroid metadata (MOA, size, method, shrinkage).
        - P: centroid matrix (n_centroids, d), L2-normalised.
        - centroid_moas: MOA label per centroid.
    """
    id_idx = {cid: i for i, cid in enumerate(embeddings[id_col].tolist())}
    num_cols = embeddings.select_dtypes(include=[np.number]).columns.tolist()
    X_all = l2_normalise(X=embeddings[num_cols].to_numpy())

    gmean = X_all.mean(axis=0)
    gmean = gmean / np.linalg.norm(gmean) if np.linalg.norm(gmean) > 0 else gmean

    labelled = anchors[[id_col, moa_col]].dropna().copy()
    labelled[id_col] = labelled[id_col].astype(str)
    labelled = labelled[labelled[id_col].isin(id_idx.keys())]
    moa_groups = labelled.groupby(moa_col)

    def eff_alpha(n_members: int) -> float:
        alpha = float(centroid_shrinkage)
        if adaptive_shrinkage and n_members > 0:
            alpha += min(float(adaptive_shrinkage_max), float(adaptive_shrinkage_c) / float(n_members))
        return float(min(1.0, max(0.0, alpha)))

    rng = np.random.RandomState(random_seed)
    P_list: List[np.ndarray] = []
    centroid_moas: List[str] = []
    summary_rows: List[Dict[str, object]] = []

    for moa, sub in moa_groups:
        idxs = [id_idx[c] for c in sub[id_col].tolist()]
        X_m = X_all[idxs, :]
        n_m = int(X_m.shape[0])

        if n_m < int(min_members_per_moa) and bool(skip_tiny_moas):
            summary_rows.append({"moa": moa, "centroid_index": -1, "n_members": n_m,
                                 "method": "skipped_tiny", "shrinkage_effective": 0.0})
            continue

        if n_centroids_per_moa <= 1 or X_m.shape[0] <= 2:
            proto = np.median(X_m, axis=0) if centroid_method == "median" else np.mean(X_m, axis=0)
            a = eff_alpha(n_m)
            if a > 0:
                proto = (1 - a) * proto + a * gmean
            proto = proto / np.linalg.norm(proto) if np.linalg.norm(proto) > 0 else proto
            P_list.append(proto)
            centroid_moas.append(str(moa))
            summary_rows.append({"moa": moa, "centroid_index": 0, "n_members": n_m,
                                 "method": centroid_method, "shrinkage_effective": float(a)})
        else:
            try:
                from sklearn.cluster import KMeans
                k = min(n_centroids_per_moa, X_m.shape[0])
                km = KMeans(n_clusters=k, random_state=rng.randint(0, 10**6), n_init="auto")
                labels = km.fit_predict(X_m)
                for j in range(k):
                    sel = X_m[labels == j, :]
                    if sel.shape[0] == 0:
                        continue
                    n_sub = int(sel.shape[0])
                    if n_sub < int(min_members_per_moa) and bool(skip_tiny_moas):
                        summary_rows.append({"moa": moa, "centroid_index": j, "n_members": n_sub,
                                             "method": "skipped_tiny_subcluster", "shrinkage_effective": 0.0})
                        continue
                    proto = np.median(sel, axis=0) if centroid_method == "median" else np.mean(sel, axis=0)
                    a = eff_alpha(n_sub)
                    if a > 0:
                        proto = (1 - a) * proto + a * gmean
                    proto = proto / np.linalg.norm(proto) if np.linalg.norm(proto) > 0 else proto
                    P_list.append(proto)
                    centroid_moas.append(str(moa))
                    summary_rows.append({"moa": moa, "centroid_index": j, "n_members": n_sub,
                                         "method": f"kmeans/{centroid_method}", "shrinkage_effective": float(a)})
            except Exception:
                proto = np.median(X_m, axis=0) if centroid_method == "median" else np.mean(X_m, axis=0)
                a = eff_alpha(n_m)
                if a > 0:
                    proto = (1 - a) * proto + a * gmean
                proto = proto / np.linalg.norm(proto) if np.linalg.norm(proto) > 0 else proto
                P_list.append(proto)
                centroid_moas.append(str(moa))
                summary_rows.append({"moa": moa, "centroid_index": 0, "n_members": n_m,
                                     "method": f"{centroid_method}(fallback_no_kmeans)",
                                     "shrinkage_effective": float(a)})

    P = np.vstack(P_list) if P_list else np.zeros((0, X_all.shape[1]), dtype=float)
    return pd.DataFrame(summary_rows), P, centroid_moas


def cosine_scores(*, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix.

    Parameters
    ----------
    Q
        Query matrix of shape (n_queries, d), L2-normalised.
    P
        Prototype/centroid matrix of shape (n_centroids, d), L2-normalised.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_queries, n_centroids).
    """
    if Q.size == 0 or P.size == 0:
        return np.zeros((Q.shape[0], P.shape[0]), dtype=float)
    return Q @ P.T


def csls_scores(*, Q: np.ndarray, P: np.ndarray, k: int = 10) -> np.ndarray:
    """Cross-domain similarity local scaling (CSLS).

    CSLS(q, p) = 2*cos(q, p) - r_q - r_p, where r_q and r_p are the average
    top-k cosine similarities for each query and prototype respectively.

    Parameters
    ----------
    Q
        Query matrix (n_queries, d), L2-normalised.
    P
        Prototype/centroid matrix (n_centroids, d), L2-normalised.
    k
        Neighbourhood size used for local scaling.

    Returns
    -------
    np.ndarray
        CSLS matrix of shape (n_queries, n_centroids).
    """
    if Q.size == 0 or P.size == 0:
        return np.zeros((Q.shape[0], P.shape[0]), dtype=float)
    S = Q @ P.T
    kq = min(k, P.shape[0])
    rp = min(k, Q.shape[0])
    if kq <= 0 or rp <= 0:
        return S.copy()
    part_q = np.partition(S, kth=S.shape[1] - kq, axis=1)[:, -kq:]
    r_q = part_q.mean(axis=1, keepdims=True)
    part_p = np.partition(S, kth=S.shape[0] - rp, axis=0)[-rp:, :]
    r_p = part_p.mean(axis=0, keepdims=True)
    return 2.0 * S - r_q - r_p


# --------------------------------------------------------------------------- #
# Projection and plotting
# --------------------------------------------------------------------------- #

def project_2d(*, X: np.ndarray, method: str = "umap", random_seed: int = 0) -> np.ndarray:
    """Project to 2D via UMAP (preferred) or PCA fallback.

    Parameters
    ----------
    X
        Matrix of shape (n_samples, d).
    method
        "umap" or "pca".
    random_seed
        Random seed used by the projector.

    Returns
    -------
    np.ndarray
        2D coordinates of shape (n_samples, 2).
    """
    method = method.lower().strip()
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_seed)
            return reducer.fit_transform(X)
        except Exception:
            pass
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=random_seed).fit_transform(X)


def compute_convex_hulls(*, xy: np.ndarray, labels: Sequence[str], min_points: int = 3) -> Dict[str, np.ndarray]:
    """Compute per-label convex hull vertices in 2D (best effort).

    Parameters
    ----------
    xy
        2D coordinates (n_points, 2).
    labels
        Label for each point (length n_points).
    min_points
        Minimum number of points to attempt a hull.

    Returns
    -------
    dict
        Mapping label -> hull vertices array (m, 2). Empty dict if SciPy is not available.
    """
    try:
        from scipy.spatial import ConvexHull  # type: ignore
    except Exception:
        return {}
    hulls: Dict[str, np.ndarray] = {}
    labels = np.asarray(labels)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if idx.size < min_points:
            continue
        pts = xy[idx, :]
        try:
            hull = ConvexHull(pts)
            hulls[str(lab)] = pts[hull.vertices, :]
        except Exception:
            continue
    return hulls



def plot_static(
    *,
    xy_comp: np.ndarray,
    xy_centroids: np.ndarray,
    comp_labels: Sequence[str],
    centroid_labels: Sequence[str],
    ids: Sequence[str],
    highlight_ids: Optional[Iterable[str]],
    out_path: Union[str, Path],
    title: str,
    label_truncate: int = 12,
    label_fontsize: float = 5.0,
    label_topk: int = 0,
    label_mode: str = "centroid",
) -> None:


    """Render a static matplotlib plot (saved as pdf).

    Parameters
    ----------
    xy_comp
        2D coordinates of compounds (n_compounds, 2).
    xy_centroids
        2D coordinates of centroids (n_centroids, 2).
    comp_labels
        Label (MOA) per compound.
    centroid_labels
        MOA label per centroid.
    ids
        cpd_id per compound (used for highlighting).
    highlight_ids
        Iterable of cpd_ids to annotate on the plot.
    out_path
        Output file path for the pdf figure.
    title
        Plot title.

    Returns
    -------
    None
    """
    comp_labels = np.asarray(comp_labels)
    highlight_set = set(highlight_ids or [])

    uniq = np.unique(comp_labels)
    cmap = plt.get_cmap("tab20")
    colour_map = {lab: cmap(i % 20) for i, lab in enumerate(uniq)}
    # Decide which MOAs to label and how they appear
    # decide which MOAs get text labels
    moa_sizes = Counter(comp_labels)
    label_keep = pick_labelled_moas(moa_names=list(uniq), 
                                    moa_sizes=moa_sizes, topk=label_topk)


    hulls = compute_convex_hulls(xy=xy_comp, labels=comp_labels, min_points=3)

    fig, ax = plt.subplots(figsize=(10, 8))
    for lab in uniq:
        idx = comp_labels == lab
        ax.scatter(xy_comp[idx, 0], xy_comp[idx, 1], s=10, alpha=0.6,
                   label=str(lab), c=[colour_map[lab]], edgecolors="none")

    for lab, verts in hulls.items():
        ax.fill(verts[:, 0], verts[:, 1], alpha=0.08, color=colour_map.get(lab, (0.8, 0.8, 0.8)))

    for (x, y), lab in zip(xy_centroids, centroid_labels):
        ax.scatter([x], [y], s=180, c=[colour_map.get(lab, "k")],
                edgecolors="black", linewidths=1.2, marker="o", zorder=5)
        # show text only if mode allows and MOA is in selected set
        show_text = (label_mode == "centroid") and (lab in labelled_moas)
        text_disp = truncate_label(str(lab), label_truncate) if show_text else ""
        if text_disp:
            ax.text(x, y, f"  {text_disp}", fontsize=label_fontsize,
                    weight="bold", va="center", zorder=6)


    if highlight_set:
        for (x, y), cid in zip(xy_comp, ids):
            if cid in highlight_set:
                ax.scatter([x], [y], s=60, facecolors="none", edgecolors="black", linewidths=1.0, zorder=7)
                ax.text(x, y, f" {cid}", fontsize=8, va="bottom", zorder=8)

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5, frameon=False)
    ax.grid(True, alpha=0.25)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)




def truncate_label(label: str, max_chars: int) -> str:
    """
    Return a truncated label for display, preserving full text for hover.

    Parameters
    ----------
    label : str
        The original MOA label.
    max_chars : int
        Maximum number of characters to show before ellipsis.

    Returns
    -------
    str
        Possibly truncated label; empty string if input is falsy.
    """
    if not label:
        return ""
    if max_chars is None or max_chars <= 0 or len(label) <= max_chars:
        return label
    return label[:max_chars].rstrip() + "…"


def pick_labelled_moas(
    *,
    moa_names: list[str],
    moa_sizes: dict[str, int],
    topk: int
) -> set[str]:
    """
    Decide which MOAs should receive visible text labels.

    Parameters
    ----------
    moa_names : list[str]
        Ordered list of MOA names used in plotting.
    moa_sizes : dict[str, int]
        Mapping MOA -> number of member compounds (or cluster size).
    topk : int
        Label only the top-k MOAs by size. If 0, label all.

    Returns
    -------
    set[str]
        Set of MOA names to label.
    """
    if topk is None or topk <= 0:
        return set(moa_names)
    ranked = sorted(moa_names, key=lambda m: moa_sizes.get(m, 0), reverse=True)
    return set(ranked[:topk])



def try_plot_interactive(
    *,
    xy_comp: np.ndarray,
    xy_centroids: np.ndarray,
    comp_labels: Sequence[str],
    centroid_labels: Sequence[str],
    ids: Sequence[str],
    out_html: Union[str, Path],
    title: str,
    label_truncate: int = 17,
    label_topk: int = 0,
    label_mode: str = "centroid",
) -> bool:


    """Write an interactive Plotly HTML (best effort).

    Parameters
    ----------
    xy_comp
        2D coordinates of compounds (n_compounds, 2).
    xy_centroids
        2D coordinates of centroids (n_centroids, 2).
    comp_labels
        Label (MOA) per compound.
    centroid_labels
        MOA label per centroid.
    ids
        cpd_id per compound (for hover).
    out_html
        Output path for the HTML file.
    title
        Plot title.

    Returns
    -------
    bool
        True if the HTML was written successfully, False otherwise.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except Exception:
        print("[WARN] Plotly not installed; skipping interactive HTML.")
        return False

    comp_labels = np.asarray(comp_labels)
    uniq = np.unique(comp_labels)
    colour_map = {lab: px.colors.qualitative.Dark24[i % 24] for i, lab in enumerate(uniq)}
    # Which MOAs get text labels?
    moa_sizes = Counter(comp_labels)
    labelled_moas = pick_labelled_moas(
        moa_names=list(set(comp_labels)),
        moa_sizes=moa_sizes,
        topk=label_topk,
    )

    centroid_text = []
    centroid_custom = []
    for lab in centroid_labels:
        centroid_custom.append(str(lab))  # full label for hover
        if label_mode == "centroid" and (lab in label_keep):
            centroid_text.append(truncate_label(str(lab), label_truncate))
        else:
            centroid_text.append("")  # hide text



    fig = go.Figure()


    for lab in uniq:
        idx = comp_labels == lab
        short = truncate_label(str(lab), label_truncate)
        fig.add_trace(
            go.Scattergl(
                x=xy_comp[idx, 0], y=xy_comp[idx, 1],
                mode="markers",
                marker=dict(size=5, color=colour_map[lab], opacity=0.7),
                name=short,                      # truncated name in legend
                showlegend=(lab in label_keep),  # only top-K in legend
                text=[f"{i}" for i in np.asarray(ids)[idx]],
                hovertemplate="cpd_id=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}"
                            f"<extra>{lab}</extra>",  # full label in hover
            )
        )


    fig.add_trace(
        go.Scatter(
            x=xy_centroids[:, 0], y=xy_centroids[:, 1],
            mode="markers+text" if any(centroid_text) else "markers",
            marker=dict(size=14, color="black", line=dict(width=1, color="white")),
            text=centroid_text,
            customdata=centroid_custom,
            textposition="middle right",
            name="centroids",
            hovertemplate="centroid=%{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )


    hulls = compute_convex_hulls(xy=xy_comp, labels=comp_labels, min_points=3)
    for lab, verts in hulls.items():
        fig.add_trace(
            go.Scatter(
                x=np.r_[verts[:, 0], verts[0, 0]],
                y=np.r_[verts[:, 1], verts[0, 1]],
                mode="lines",
                fill="toself",
                fillcolor=colour_map.get(lab, "rgba(200,200,200,0.2)"),
                line=dict(width=1, color=colour_map.get(lab, "rgba(50,50,50,0.6)")),
                name=f"{lab} region",
                hoverinfo="skip",
                opacity=0.15,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        template="plotly_white",
        legend=dict(itemsizing="constant"),
        hovermode="closest",
    )

    try:
        out_html = Path(out_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        print(f"[OK] Wrote interactive HTML: {out_html}")
        return True
    except Exception as exc:
        print(f"[WARN] Could not write interactive HTML ({exc}); continuing.")
        return False


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def main() -> None:
    """Parse arguments, load inputs, rebuild centroids, project, and plot.

    Returns
    -------
    None
    """
    p = argparse.ArgumentParser(description="2D MOA centroid map (UMAP/PCA, static + best-effort interactive).")

    # Prefer --moa_dir to reuse aggregated embeddings from scoring
    p.add_argument("--moa_dir", type=str, help="MoA output directory containing compound_embeddings.tsv.")
    p.add_argument("--embeddings_tsv", type=str, help="TSV with embeddings (well- or compound-level).")
    p.add_argument("--anchors_tsv", type=str, required=True, help="TSV with anchors: id + MOA (for centroids).")
    p.add_argument("--predictions_tsv", type=str, help="Optional predictions TSV to colour by top_moa.")
    p.add_argument("--id_col", type=str, default="cpd_id", help="Identifier column name.")
    p.add_argument("--moa_col", type=str, default="moa", help="MOA column in anchors or predictions.")

    # Aggregation (only used if we must aggregate from raw embeddings)
    p.add_argument("--aggregate_method", type=str, default="median",
                   choices=["median", "mean", "trimmed_mean", "geometric_median"],
                   help="Replicate aggregation if embeddings are well-level.")
    p.add_argument("--trimmed_frac", type=float, default=0.1, help="Trim fraction for trimmed_mean.")

    # Centroid build (mirror scoring)
    p.add_argument("--n_centroids_per_moa", type=int, default=1, help="Sub-centroids per MOA (k-means if >1).")
    p.add_argument("--centroid_method", type=str, default="median", choices=["median", "mean"],
                   help="Centroid estimator when n_centroids_per_moa<=1.")
    p.add_argument("--centroid_shrinkage", type=float, default=0.0, help="Shrinkage towards global mean (0..1).")
    p.add_argument("--min_members_per_moa", type=int, default=1, help="Minimum labelled members for a centroid.")
    p.add_argument("--skip_tiny_moas", action="store_true", help="Skip MOAs with < min_members_per_moa.")
    p.add_argument("--adaptive_shrinkage", action="store_true", help="Enable size-aware shrinkage.")
    p.add_argument("--adaptive_shrinkage_c", type=float, default=0.5, help="C constant for size-aware term.")
    p.add_argument("--adaptive_shrinkage_max", type=float, default=0.3, help="Max extra shrinkage.")

    # Assignment (for colouring)
    p.add_argument("--assignment", type=str, default="predictions", choices=["predictions", "cosine", "csls"],
                   help="How to choose MOA for colouring (default: predictions if provided).")
    p.add_argument("--csls_k", type=int, default=-1, help="CSLS k; -1 => auto≈√n_centroids clipped [5,50].")

    # Projection + outputs
    p.add_argument("--projection", type=str, default="umap", choices=["umap", "pca"], help="2D projection method.")
    p.add_argument("--highlight_ids", type=str, default="", help="Comma-separated cpd_ids to annotate.")
    p.add_argument("--out_prefix", type=str, default="moa_map", help="Prefix for outputs (.pdf and .html).")
    p.add_argument("--random_seed", type=int, default=0, help="Random seed.")

    # plots: make the labels less ... invasive
    p.add_argument(
                    "--label_truncate",
                    type=int,
                    default=17,
                    help="Maximum characters to display for MOA labels on the plot (default: 17).",
                )
    p.add_argument(
        "--label_fontsize",
        type=float,
        default=6.0,
        help="Font size for centroid text labels on the static PDF (default: 9).",
    )
    p.add_argument(
        "--label_topk",
        type=int,
        default=0,
        help="Only label the top-K MOAs by membership size. 0 means label all (default: 0).",
    )
    p.add_argument(
        "--label_mode",
        type=str,
        default="centroid",
        choices=["centroid", "none"],
        help="Label mode: 'centroid' shows text at centroids; 'none' hides text labels (hover still shows full).",
    )

    args = p.parse_args()

    # ---------- Load embeddings (prefer MoA dir) ----------
    if args.moa_dir:
        moa_dir = Path(args.moa_dir)
        emb_path = moa_dir / "compound_embeddings.tsv"
        if not emb_path.exists():
            raise SystemExit(f"compound_embeddings.tsv not found in {moa_dir}")
        agg = pd.read_csv(emb_path, sep="\t")
        id_col = detect_id_column(df=agg, id_col=args.id_col)
        num_cols = agg.select_dtypes(include=[np.number]).columns.tolist()
    else:
        if not args.embeddings_tsv:
            raise SystemExit("Provide --moa_dir OR --embeddings_tsv.")
        df = pd.read_csv(args.embeddings_tsv, sep="\t")
        id_col = detect_id_column(df=df, id_col=args.id_col)
        if df.groupby(id_col).size().max() > 1:
            agg = aggregate_compounds(df=df, id_col=id_col,
                                      method=args.aggregate_method, trimmed_frac=args.trimmed_frac)
        else:
            agg = df.copy()
        num_cols = agg.select_dtypes(include=[np.number]).columns.tolist()

    X = l2_normalise(X=agg[num_cols].to_numpy())
    ids = agg[id_col].astype(str).tolist()

    # ---------- Build centroids ----------
    anchors = pd.read_csv(args.anchors_tsv, sep="\t")
    _, P, centroid_moas = build_moa_centroids(
        embeddings=agg,
        anchors=anchors,
        id_col=id_col,
        moa_col=args.moa_col,
        n_centroids_per_moa=args.n_centroids_per_moa,
        centroid_method=args.centroid_method,
        centroid_shrinkage=args.centroid_shrinkage,
        min_members_per_moa=args.min_members_per_moa,
        skip_tiny_moas=args.skip_tiny_moas,
        adaptive_shrinkage=args.adaptive_shrinkage,
        adaptive_shrinkage_c=args.adaptive_shrinkage_c,
        adaptive_shrinkage_max=args.adaptive_shrinkage_max,
        random_seed=args.random_seed,
    )

    # ---------- Colouring labels ----------
    if args.assignment == "predictions" and args.predictions_tsv:
        pred = pd.read_csv(args.predictions_tsv, sep="\t")
        moa_col_pred = args.moa_col if args.moa_col in pred.columns else ("top_moa" if "top_moa" in pred.columns else None)
        if moa_col_pred is None:
            raise SystemExit("Predictions TSV must contain --moa_col or 'top_moa'.")
        top_map = dict(pred[[id_col, moa_col_pred]].astype({id_col: str}).itertuples(index=False, name=None))
        comp_labels = [top_map.get(cid, "Unassigned") for cid in ids]
    else:
        if P.shape[0] == 0:
            comp_labels = ["Unassigned"] * X.shape[0]
        else:
            S_cos = cosine_scores(Q=X, P=P)
            if args.assignment == "csls":
                k_eff = int(np.clip(int(np.sqrt(P.shape[0])), 5, 50)) if int(args.csls_k) <= 0 else int(args.csls_k)
                S = csls_scores(Q=X, P=P, k=k_eff)
            else:
                S = S_cos
            j_max = np.argmax(S, axis=1)
            comp_labels = [centroid_moas[j] for j in j_max]

    # ---------- 2D projection (compounds + centroids together) ----------
    X_all = X if P.shape[0] == 0 else np.vstack([X, P])
    xy_all = project_2d(X=X_all, method=args.projection, random_seed=args.random_seed)
    if P.shape[0] == 0:
        xy_comp = xy_all
        xy_centroids = np.zeros((0, 2))
    else:
        xy_comp = xy_all[: X.shape[0], :]
        xy_centroids = xy_all[X.shape[0]:, :]

    # ---------- Write outputs ----------
    out_pdf = Path(f"{args.out_prefix}.pdf")
    out_html = Path(f"{args.out_prefix}.html")
    highlight_ids = [s for s in args.highlight_ids.split(",") if s] if args.highlight_ids else []

    plot_static(
        xy_comp=xy_comp,
        xy_centroids=xy_centroids,
        comp_labels=comp_labels,
        centroid_labels=centroid_moas,
        ids=ids,
        highlight_ids=highlight_ids,
        out_path=out_pdf,
        title=f"MOA map ({args.projection.upper()})",
        label_truncate=args.label_truncate,
        label_fontsize=args.label_fontsize,
        label_topk=args.label_topk,
        label_mode=args.label_mode,
    )
    print(f"[OK] Wrote static figure: {out_pdf}")

    _ = try_plot_interactive(
        xy_comp=xy_comp,
        xy_centroids=xy_centroids,
        comp_labels=comp_labels,
        centroid_labels=centroid_moas,
        ids=ids,
        out_html=out_html,
        title=f"MOA map ({args.projection.upper()})",
        label_truncate=args.label_truncate,
        label_topk=args.label_topk,
        label_mode=args.label_mode,
    )


if __name__ == "__main__":
    main()
