#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Overlap Between Nearest Neighbour (NN) Lists with Auto-k, Tie-Aware k, and RBO
--------------------------------------------------------------------------------------

Compares NN lists from different runs (e.g., pre-CLIPn feature-space vs post-CLIPn).
Supports:
- LONG tables: query/neighbour rows, optional rank or distance
- WIDE tables: one row per query with NN1..NNk columns

Features
--------
- Auto-k: use `--k auto` to detect each file's typical neighbour count (median per query),
  then compare runs using the minimum across the pair.
- Tie-aware k: `--include_ties_at_k` includes all neighbours tied at the k-th distance
  (LONG tables with a distance column only). Jaccard is computed on these expanded sets.
- Rank-Biased Overlap (RBO): `--with_rbo` computes RBO at depth k using parameter `p`
  (default 0.9), emphasising top ranks.

Outputs (TSV; no comma-separated files)
---------------------------------------
- overlaps_per_item.tsv   : per-query Jaccard (and optional RBO) vs baseline
- overlaps_summary.tsv    : summary stats per run (mean/median/IQR/std)

Usage example
-------------
python knn_overlap_eval.py \
  --baseline_nn_tsv ../E150_L148/post_knn/STB_vs_mitotox_integrate_all_E150_L148_nearest_neighbours.tsv \
  --run_nn_tsv STB_vs_mitotox_integrate_all_E100_L20_nearest_neighbours.tsv \
               STB_vs_mitotox_integrate_all_E120_L10_nearest_neighbours.tsv \
               STB_vs_mitotox_integrate_all_E150_L148_nearest_neighbours.tsv \
               STB_vs_mitotox_integrate_all_E150_L60_nearest_neighbours.tsv \
               STB_vs_mitotox_integrate_all_E150_L79_nearest_neighbours.tsv \
  --out_item_tsv overlaps_per_item.tsv \
  --out_summary_tsv overlaps_summary.tsv \
  --k auto \
  --include_ties_at_k \
  --with_rbo --rbo_p 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------- utilities ---------------------------------- #

def detect_column(*, df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first present column name from candidates (case-sensitive).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    candidates : Iterable[str]
        Candidate column names to try.

    Returns
    -------
    Optional[str]
        The detected column name, or None if none match.
    """
    for name in candidates:
        if name in df.columns:
            return name
    return None


def is_wide_nn_table(*, df: pd.DataFrame, nn_prefix: str = "NN") -> bool:
    """
    Heuristically decide if a table is 'wide' (NN1, NN2, ... columns).

    Parameters
    ----------
    df : pd.DataFrame
        Candidate DataFrame.
    nn_prefix : str, optional
        Prefix to look for, by default 'NN'.

    Returns
    -------
    bool
        True if it looks like a wide NN table, else False.
    """
    return any(c.startswith(nn_prefix) for c in df.columns)


def wide_nn_columns(*, df: pd.DataFrame, nn_prefix: str = "NN") -> List[str]:
    """
    Return ordered NN columns like NN1..NNk (sorted by numeric suffix if present).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with wide NN columns.
    nn_prefix : str, optional
        Prefix for NN columns, by default 'NN'.

    Returns
    -------
    List[str]
        Ordered list of NN columns.
    """
    nn_cols = [c for c in df.columns if c.startswith(nn_prefix)]

    def key(c: str) -> Tuple[int, str]:
        suf = c[len(nn_prefix):]
        try:
            return (int(suf), "")
        except ValueError:
            return (10**9, c)

    return sorted(nn_cols, key=key)


# --------------------------- NN builders ---------------------------------- #

def build_long_lists(
    *,
    df: pd.DataFrame,
    query_col: str,
    neighbour_col: str,
    rank_col: Optional[str],
    distance_col: Optional[str],
) -> Tuple[Dict[str, List[str]], Dict[str, Optional[List[float]]]]:
    """
    Build ordered neighbour lists per query from a LONG-form NN table, and
    optionally aligned distance lists (if a distance column is present).

    Sort precedence within each query:
    1) rank ascending (if present), else
    2) distance ascending (if present), else
    3) file order.

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, Optional[List[float]]]]
        (query -> ordered neighbour IDs, query -> aligned distances or None)
    """
    use_cols = [query_col, neighbour_col]
    if rank_col and rank_col not in use_cols:
        use_cols.append(rank_col)
    if distance_col and distance_col not in use_cols:
        use_cols.append(distance_col)

    work = df.loc[:, use_cols].copy()
    work[query_col] = work[query_col].astype(str)
    work[neighbour_col] = work[neighbour_col].astype(str)

    if rank_col is not None:
        work = work.sort_values([query_col, rank_col], ascending=[True, True])
    elif distance_col is not None:
        work = work.sort_values([query_col, distance_col], ascending=[True, True])

    lists: Dict[str, List[str]] = {}
    dists: Dict[str, Optional[List[float]]] = {}

    for q, sub in work.groupby(query_col, sort=False):
        # drop self if present
        sub = sub.loc[sub[neighbour_col] != q]
        neigh = sub[neighbour_col].tolist()
        lists[str(q)] = neigh
        if distance_col is not None:
            dists[str(q)] = sub[distance_col].astype(float).tolist()
        else:
            dists[str(q)] = None

    return lists, dists


def build_wide_lists(
    *,
    df: pd.DataFrame,
    id_col: str,
    nn_cols: Sequence[str],
) -> Dict[str, List[str]]:
    """
    Build ordered neighbour lists per query from a WIDE table (NN1..NNk).

    Parameters
    ----------
    df : pd.DataFrame
        Wide-form NN table.
    id_col : str
        Identifier column for the query.
    nn_cols : Sequence[str]
        Ordered NN columns.

    Returns
    -------
    Dict[str, List[str]]
        Mapping query -> ordered neighbour IDs.
    """
    keep = [id_col] + list(nn_cols)
    work = df.loc[:, [c for c in keep if c in df.columns]].copy()
    work[id_col] = work[id_col].astype(str)

    lists: Dict[str, List[str]] = {}
    for _, row in work.iterrows():
        q = str(row[id_col])
        neigh: List[str] = []
        for c in nn_cols:
            if c in row and pd.notna(row[c]):
                v = str(row[c])
                if v != q:
                    neigh.append(v)
        lists[q] = neigh
    return lists


# ------------------------- loaders & detection ----------------------------- #

def load_nn_file(
    *,
    path: str,
    nn_prefix: str = "NN",
) -> Tuple[str, Dict[str, List[str]], Dict[str, Optional[List[float]]], int]:
    """
    Load a NN TSV with auto-detection of LONG vs WIDE schema.

    Returns
    -------
    Tuple[str, Dict[str, List[str]], Dict[str, Optional[List[float]]], int]
        (run_name, query->list(neighbours), query->list(distances or None), detected_k_file)
    """
    df = pd.read_csv(path, sep="\t")
    run_name = Path(path).stem.replace("_nearest_neighbours", "")

    if is_wide_nn_table(df=df, nn_prefix=nn_prefix):
        nn_cols = wide_nn_columns(df=df, nn_prefix=nn_prefix)
        id_candidates = [c for c in df.columns if c not in nn_cols]
        if not id_candidates:
            raise ValueError(f"Could not find an ID column in wide table: {path}")
        id_col = id_candidates[0]
        lists = build_wide_lists(df=df, id_col=id_col, nn_cols=nn_cols)
        dists: Dict[str, Optional[List[float]]] = {q: None for q in lists.keys()}
    else:
        q_candidates = ["QueryID", "cpd_id", "query", "Query", "Compound", "compound", "source", "Query_cpd_id"]
        n_candidates = ["NeighbourID", "neighbour_id", "neighbour", "Neighbour", "neighbor", "Neighbor", "target", "Neighbour_cpd_id"]
        r_candidates = ["rank", "Rank", "nn_rank", "k", "order", "position"]
        d_candidates = ["distance", "Distance", "cosine_distance", "euclidean_distance"]

        qcol = detect_column(df=df, candidates=q_candidates)
        ncol = detect_column(df=df, candidates=n_candidates)
        rcol = detect_column(df=df, candidates=r_candidates)
        dcol = detect_column(df=df, candidates=d_candidates)
        if qcol is None or ncol is None:
            raise ValueError(f"Could not detect required columns in {path}. Columns: {list(df.columns)}")

        lists, dists = build_long_lists(
            df=df, query_col=qcol, neighbour_col=ncol, rank_col=rcol, distance_col=dcol
        )

    lengths = [len(v) for v in lists.values()]
    k_file = int(np.median(lengths)) if lengths else 0
    return run_name, lists, dists, k_file


# ------------------------------ metrics ----------------------------------- #

def jaccard_from_sets(*, a_set: set, b_set: set) -> float:
    """
    Compute Jaccard index between two sets.

    Parameters
    ----------
    a_set : set
        First set.
    b_set : set
        Second set.

    Returns
    -------
    float
        Jaccard index in [0, 1].
    """
    if not a_set and not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return 0.0 if union == 0 else inter / union


def rbo_at_depth(
    *,
    list_a: List[str],
    list_b: List[str],
    p: float = 0.9,
    depth: Optional[int] = None,
) -> float:
    """
    Compute Rank-Biased Overlap (RBO) between two ranked lists at a given depth.

    This implementation uses the finite-depth RBO:
        RBO(p, d) = (1 - p) * sum_{i=1..d} ( |A_1..i ∩ B_1..i| / i ) * p^{i-1}

    Parameters
    ----------
    list_a : List[str]
        First ranked list.
    list_b : List[str]
        Second ranked list.
    p : float, optional
        Top-weightedness parameter in (0, 1); higher emphasises top ranks,
        by default 0.9.
    depth : Optional[int], optional
        Depth to evaluate (k_eff). If None, uses min(len(a), len(b)).

    Returns
    -------
    float
        RBO score in [0, 1].
    """
    if not list_a or not list_b:
        return 0.0
    if depth is None:
        depth = min(len(list_a), len(list_b))
    depth = max(0, min(depth, len(list_a), len(list_b)))
    if depth == 0:
        return 0.0

    seen_a: set = set()
    seen_b: set = set()
    overlap = 0
    weighted_sum = 0.0

    for i in range(1, depth + 1):
        a_i = list_a[i - 1]
        b_i = list_b[i - 1]

        if a_i not in seen_a:
            seen_a.add(a_i)
            if a_i in seen_b:
                overlap += 1
        if b_i not in seen_b:
            seen_b.add(b_i)
            if b_i in seen_a and b_i != a_i:
                overlap += 1

        agreement = overlap / i
        weighted_sum += agreement * (p ** (i - 1))

    return (1.0 - p) * weighted_sum


# ------------------------------ evaluation -------------------------------- #

def select_set_with_ties(
    *,
    ordered_ids: List[str],
    ordered_dists: Optional[List[float]],
    k_depth: int,
    include_ties: bool,
) -> set:
    """
    Select a set of neighbour IDs up to depth k, with optional tie expansion.

    Parameters
    ----------
    ordered_ids : List[str]
        Ordered neighbour IDs for a query.
    ordered_dists : Optional[List[float]]
        Distances aligned to ordered_ids (ascending; smaller is closer). If None,
        ties cannot be detected and simple top-k is used.
    k_depth : int
        Target depth k.
    include_ties : bool
        If True and distances are available, include all neighbours whose distance
        equals the k-th distance (tie-aware).

    Returns
    -------
    set
        Set of neighbour IDs selected for Jaccard.
    """
    if k_depth <= 0 or not ordered_ids:
        return set()
    k_eff = min(k_depth, len(ordered_ids))
    base_ids = ordered_ids[:k_eff]
    if not include_ties or ordered_dists is None:
        return set(base_ids)

    # Tie-aware expansion
    kth_dist = ordered_dists[k_eff - 1]
    expanded = [nid for nid, d in zip(ordered_ids, ordered_dists) if d <= kth_dist]
    return set(expanded)


def jaccard_top_k_sets(
    *,
    a_ids: List[str],
    b_ids: List[str],
    k_depth: int,
    include_ties: bool,
    a_dists: Optional[List[float]] = None,
    b_dists: Optional[List[float]] = None,
) -> Tuple[float, int, int]:
    """
    Compute Jaccard between (possibly tie-expanded) top-k neighbour sets.

    Returns
    -------
    Tuple[float, int, int]
        (jaccard, size_a, size_b), where size_* are the sizes of the
        sets actually used (may exceed k if ties included).
    """
    a_set = select_set_with_ties(
        ordered_ids=a_ids, ordered_dists=a_dists, k_depth=k_depth, include_ties=include_ties
    )
    b_set = select_set_with_ties(
        ordered_ids=b_ids, ordered_dists=b_dists, k_depth=k_depth, include_ties=include_ties
    )
    return jaccard_from_sets(a_set=a_set, b_set=b_set), len(a_set), len(b_set)


def evaluate_vs_baseline(
    *,
    baseline_nn_tsv: str,
    run_nn_tsvs: List[str],
    out_item_tsv: str,
    out_summary_tsv: str,
    k: str = "auto",
    include_ties_at_k: bool = False,
    with_rbo: bool = False,
    rbo_p: float = 0.9,
) -> None:
    """
    Compare NN lists versus a baseline, with auto-k, tie-aware k, and optional RBO.

    Parameters
    ----------
    baseline_nn_tsv : str
        Path to baseline NN TSV.
    run_nn_tsvs : List[str]
        Paths to one or more run NN TSVs to compare.
    out_item_tsv : str
        Output TSV path for per-query overlaps.
    out_summary_tsv : str
        Output TSV path for per-run summaries.
    k : str, optional
        'auto' to detect k per file and use the minimum per comparison,
        or an integer as a string (e.g., '10'), by default 'auto'.
    include_ties_at_k : bool, optional
        If True, include all neighbours tied at the k-th distance (when available),
        by default False.
    with_rbo : bool, optional
        If True, compute Rank-Biased Overlap (RBO) at depth k, by default False.
    rbo_p : float, optional
        RBO top-weight parameter in (0, 1), by default 0.9.
    """
    base_run, base_lists, base_dists, base_k = load_nn_file(path=baseline_nn_tsv)

    # Decide fixed vs auto
    if isinstance(k, str) and k.strip().lower() == "auto":
        k_fixed: Optional[int] = None
    else:
        try:
            k_fixed = int(k)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"--k must be 'auto' or an integer; got {k!r}") from exc

    per_item_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for run_path in run_nn_tsvs:
        run_name, run_lists, run_dists, run_k = load_nn_file(path=run_path)

        shared = sorted(set(base_lists.keys()) & set(run_lists.keys()))
        if not shared:
            summary_rows.append(
                {"run": run_name, "baseline": base_run, "k": "NA",
                 "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan,
                 "std": np.nan, "n": 0}
            )
            continue

        # Choose comparison depth
        k_common = min(base_k, run_k) if k_fixed is None else k_fixed
        k_common = max(0, k_common)

        j_vals: List[float] = []
        rbo_vals: List[float] = []
        sizes_a: List[int] = []
        sizes_b: List[int] = []

        for q in shared:
            a_ids = base_lists[q]
            b_ids = run_lists[q]
            a_d = base_dists.get(q) if base_dists else None
            b_d = run_dists.get(q) if run_dists else None

            j, sz_a, sz_b = jaccard_top_k_sets(
                a_ids=a_ids, b_ids=b_ids, k_depth=k_common, include_ties=include_ties_at_k,
                a_dists=a_d, b_dists=b_d
            )
            j_vals.append(j)
            sizes_a.append(sz_a)
            sizes_b.append(sz_b)

            row = {
                "run": run_name,
                "baseline": base_run,
                "query_id": q,
                "k": k_common,
                "jaccard": j,
                "size_a": sz_a,
                "size_b": sz_b,
            }

            if with_rbo:
                rbo_val = rbo_at_depth(list_a=a_ids, list_b=b_ids, p=rbo_p, depth=k_common)
                rbo_vals.append(rbo_val)
                row["rbo"] = rbo_val

            per_item_rows.append(row)

        s = pd.Series(j_vals, dtype=float)
        summary = {
            "run": run_name,
            "baseline": base_run,
            "k": k_common,
            "mean": float(s.mean()),
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "n": int(len(s)),
        }

        if with_rbo and rbo_vals:
            sr = pd.Series(rbo_vals, dtype=float)
            summary.update({
                "mean_rbo": float(sr.mean()),
                "median_rbo": float(sr.median()),
                "q25_rbo": float(sr.quantile(0.25)),
                "q75_rbo": float(sr.quantile(0.75)),
                "std_rbo": float(sr.std(ddof=1)) if len(sr) > 1 else 0.0,
            })

        summary_rows.append(summary)

    pd.DataFrame(per_item_rows).to_csv(out_item_tsv, sep="\t", index=False)
    pd.DataFrame(summary_rows).to_csv(out_summary_tsv, sep="\t", index=False)


# --------------------------------- CLI ------------------------------------ #

def main() -> None:
    """Parse arguments and run the NN-list overlap evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Jaccard (tie-aware optional) and RBO overlap between NN lists versus a baseline (TSV I/O)."
    )
    parser.add_argument("--baseline_nn_tsv", type=str, required=True, help="Path to baseline NN TSV (long or wide).")
    parser.add_argument("--run_nn_tsv", type=str, nargs="+", required=True, help="One or more run NN TSVs (long or wide).")
    parser.add_argument("--out_item_tsv", type=str, required=True, help="Output TSV for per-query overlaps.")
    parser.add_argument("--out_summary_tsv", type=str, required=True, help="Output TSV for per-run summaries.")
    parser.add_argument("--k", type=str, default="auto", help="Neighbourhood size: integer or 'auto' (default: auto).")
    parser.add_argument("--include_ties_at_k", action="store_true", help="Include all neighbours tied at the k-th distance (LONG tables with distance).")
    parser.add_argument("--with_rbo", action="store_true", help="Also compute Rank-Biased Overlap (RBO) at depth k.")
    parser.add_argument("--rbo_p", type=float, default=0.9, help="RBO top-weight parameter p in (0,1), default 0.9.")
    args = parser.parse_args()

    evaluate_vs_baseline(
        baseline_nn_tsv=args.baseline_nn_tsv,
        run_nn_tsvs=args.run_nn_tsv,
        out_item_tsv=args.out_item_tsv,
        out_summary_tsv=args.out_summary_tsv,
        k=args.k,
        include_ties_at_k=args.include_ties_at_k,
        with_rbo=args.with_rbo,
        rbo_p=args.rbo_p,
    )


if __name__ == "__main__":
    main()
