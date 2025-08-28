#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarise CLIPn Nearest Neighbours and Annotate Output
------------------------------------------------------

This script identifies the top-N nearest neighbour compounds for specified
targets based on:
  1) UMAP 2D projections (using **cosine distance** in UMAP space), and
  2) CLIPn latent-space nearest neighbour (NN) results.

It supports merging compound-level annotations from one or two metadata files.

Features
--------
- Computes top-N nearest neighbours by **cosine distance** in UMAP space.
- Extracts top-N nearest neighbours from CLIPn NN output.
- Merges compound annotations from a primary metadata TSV/CSV and an optional
  secondary TSV/CSV (auto-detected delimiter).
- Outputs a combined summary to TSV and Excel (.xlsx).

Typical usage
-------------
python summarise_mcp_neighbours.py \
    --folder path/to/clipn/output \
    --metadata compound_annotations.tsv \
    --extra_metadata extra_annotations.csv \
    --top_n 10
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


def load_metadata(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load compound metadata from a given annotation file (TSV/CSV auto-detected).

    Parameters
    ----------
    path : str or None
        Path to metadata file.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with metadata, or None if file not found.
    """
    if path and os.path.isfile(path):
        print(f"[INFO] Found compound metadata file: {path}")
        try:
            meta = pd.read_csv(path, sep=None, engine="python")  # auto-detect delimiter
        except Exception:
            meta = pd.read_csv(path)  # fallback (,)
        meta.columns = [c.strip() for c in meta.columns]
        return meta
    print("[WARNING] No compound metadata file found.")
    return None


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute cosine distance (1 - cosine similarity) between two vectors.

    Parameters
    ----------
    a : numpy.ndarray
        First vector.
    b : numpy.ndarray
        Second vector.
    eps : float
        Small epsilon to avoid division by zero.

    Returns
    -------
    float
        Cosine distance in [0, 2].
    """
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return 1.0 - (num / den)


def find_nearest_umap(
    df: pd.DataFrame,
    target_id: str,
    top_n: int,
    max_dist: Optional[float] = None,
) -> pd.DataFrame:
    """
    Find top-N nearest neighbours in UMAP 2D space using cosine distance.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'UMAP1', 'UMAP2', and 'cpd_id'.
    target_id : str
        Compound ID for which neighbours will be found.
    top_n : int
        Number of nearest neighbours to retrieve.
    max_dist : float, optional
        Distance threshold; if provided, neighbours with distance > max_dist
        are discarded.

    Returns
    -------
    pandas.DataFrame
        Columns: ['cpd_id', 'nearest_cpd_id', 'distance_metric_UMAP', 'source'].
    """
    # Normalise case/whitespace for robust matching
    df = df.copy()
    df["cpd_id"] = df["cpd_id"].astype(str).str.upper().str.strip()

    target_row = df.loc[df["cpd_id"] == target_id.upper()]
    if target_row.empty:
        print(f"[WARNING] Target compound '{target_id}' not found in UMAP coordinates")
        return pd.DataFrame(columns=["cpd_id", "nearest_cpd_id", "distance_metric_UMAP", "source"])

    target_vec = target_row[["UMAP1", "UMAP2"]].values[0]
    coords = df[["UMAP1", "UMAP2"]].values

    # Cosine distance to all points
    # Vectorised implementation for speed
    norms = np.linalg.norm(coords, axis=1)
    t_norm = np.linalg.norm(target_vec)
    dots = coords @ target_vec
    dists = 1.0 - (dots / (norms * t_norm + 1e-12))

    df["distance_metric_UMAP"] = dists
    if max_dist is not None:
        df = df[df["distance_metric_UMAP"] <= max_dist]

    # Sort by distance, drop duplicates by compound, exclude self, take top-N
    df = df.sort_values("distance_metric_UMAP").drop_duplicates(subset=["cpd_id"], keep="first")
    nearest = df[df["cpd_id"] != target_id.upper()].head(top_n).copy()
    nearest["source"] = "UMAP"
    nearest = nearest.rename(columns={"cpd_id": "nearest_cpd_id"})
    nearest["cpd_id"] = target_id.upper()

    return nearest[["cpd_id", "nearest_cpd_id", "distance_metric_UMAP", "source"]]


def normalise_id_column(
    df: pd.DataFrame,
    *,
    candidates: List[str],
    new_name: str = "cpd_id",
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Find a plausible compound identifier column and normalise it to 'cpd_id'.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to inspect.
    candidates : list[str]
        Candidate column names to try (case-insensitive).
    new_name : str
        Name to assign to the normalised ID column.

    Returns
    -------
    (df, id_col) : tuple[pandas.DataFrame, Optional[str]]
        Updated DataFrame (with 'cpd_id' if found) and the matched column name,
        or (original df, None) if not found.
    """
    upper_map = {c.upper(): c for c in df.columns}
    hit = next((upper_map[c.upper()] for c in candidates if c.upper() in upper_map), None)
    if hit is None:
        return df, None
    out = df.copy()
    out[new_name] = out[hit].astype(str).str.upper().str.strip()
    out = out.drop_duplicates(subset=[new_name])
    return out, hit


def find_nearest_from_nn(
    df: pd.DataFrame,
    target_id: str,
    top_n: int,
    max_dist: Optional[float] = None,
) -> pd.DataFrame:
    """
    Retrieve top-N nearest neighbours from CLIPn-generated NN results.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain: 'cpd_id', 'neighbour_id', 'distance'.
    target_id : str
        Target compound ID.
    top_n : int
        Number of neighbours to return.
    max_dist : float, optional
        Distance threshold; if provided, neighbours with distance > max_dist
        are discarded.

    Returns
    -------
    pandas.DataFrame
        Columns: ['cpd_id', 'nearest_cpd_id', 'distance_metric_NN', 'source'].
    """
    df = df.copy()
    for col in ("cpd_id", "neighbour_id"):
        df[col] = df[col].astype(str).str.upper().str.strip()
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    target_rows = df.loc[df["cpd_id"] == target_id.upper()]
    if target_rows.empty:
        print(f"[WARNING] Target compound '{target_id}' not found in nearest neighbour data")
        return pd.DataFrame(columns=["cpd_id", "nearest_cpd_id", "distance_metric_NN", "source"])

    if max_dist is not None:
        target_rows = target_rows[target_rows["distance"] <= max_dist]

    target_rows = target_rows.sort_values("distance", ascending=True).drop_duplicates(subset=["neighbour_id"], keep="first")
    top_hits = target_rows.head(top_n).copy()
    top_hits["source"] = "NN"
    top_hits = top_hits.rename(columns={"neighbour_id": "nearest_cpd_id", "distance": "distance_metric_NN"})

    return top_hits[["cpd_id", "nearest_cpd_id", "distance_metric_NN", "source"]]


def summarise_neighbours(
    folder: str,
    targets: List[str],
    top_n: int = 15,
    metadata_file: Optional[str] = None,
    max_dist: Optional[float] = None,
    extra_metadata: Optional[str] = None,
    nn_path: Optional[str] = None,
    umap_path: Optional[str] = None,
) -> None:
    """
    Summarise nearest neighbours for target compounds and merge annotations.

    Parameters
    ----------
    folder : str
        Path to base CLIPn output folder.
    targets : list[str]
        Compound IDs to analyse.
    top_n : int, optional
        Number of neighbours to retrieve (default: 15).
    metadata_file : str, optional
        Path to primary compound metadata file (TSV/CSV).
    max_dist : float, optional
        Maximum distance threshold for filtering neighbours.
    extra_metadata : str, optional
        Path to secondary metadata file (TSV/CSV).
    nn_path : str, optional
        Optional override path for nearest_neighbours.tsv.
    umap_path : str, optional
        Optional override path for UMAP coordinates TSV.

    Returns
    -------
    None
        Writes merged summary TSV and Excel files beside the CLIPn outputs.
    """
    # Resolve default input paths if not overridden
    nn_path = nn_path or os.path.join(folder, "post_clipn", "post_analysis_script", "nearest_neighbours.tsv")
    umap_path = umap_path or os.path.join(
        folder, "post_clipn", "UMAP_kNone", "cpd_type", "clipn_umap_coordinates_cosine_n15_d0.1.tsv"
    )

    if not os.path.isfile(nn_path):
        raise FileNotFoundError(f"Nearest neighbours file not found: {nn_path}")
    if not os.path.isfile(umap_path):
        raise FileNotFoundError(f"UMAP coordinate file not found: {umap_path}")

    # Load metadata (primary)
    meta = load_metadata(metadata_file)
    if meta is not None:
        if "cpd_id" not in meta.columns:
            for cand in [
                "COMPOUND_NAME", "Compound", "compound", "compound_id",
                "CPD_ID", "cpd", "Compound Name",
            ]:
                if cand in meta.columns:
                    meta = meta.rename(columns={cand: "cpd_id"})
                    break
        if "cpd_id" in meta.columns:
            meta["cpd_id"] = meta["cpd_id"].astype(str).str.upper().str.strip()
            meta = meta.drop_duplicates(subset=["cpd_id"])
        else:
            print("[WARNING] Metadata has no cpd_id-like column; skipping metadata merge.")
            meta = None

    # Load extra metadata if provided
    extra_meta = None
    if extra_metadata and os.path.isfile(extra_metadata):
        print(f"[INFO] Found extra annotation file: {extra_metadata}")
        try:
            extra_meta = pd.read_csv(extra_metadata, sep=None, engine="python")
        except Exception:
            extra_meta = pd.read_csv(extra_metadata)
        extra_meta, id_col = normalise_id_column(
            extra_meta,
            candidates=[
                "COMPOUND_NAME", "Compound Name", "compound_name",
                "cpd_id", "CPD_ID", "Compound", "compound", "compound_id",
                "SAMPLE", "SAMPLE_ID", "Molecule", "MOLECULE_NAME",
            ],
            new_name="cpd_id",
        )
        if id_col is None:
            print("[WARNING] Extra metadata has no obvious compound ID column; skipping merge with extra metadata.")
            extra_meta = None
    else:
        print("[INFO] No extra annotation file provided or file not found.")

    # Read inputs
    nn_df = pd.read_csv(nn_path, sep="\t")
    umap_df = pd.read_csv(umap_path, sep="\t")

    # Accept alternative headers for NN table
    nn_required = {"cpd_id", "neighbour_id", "distance"}
    if not nn_required.issubset(nn_df.columns):
        alt_map = {}
        if {"QueryID", "NeighbourID"}.issubset(nn_df.columns):
            alt_map.update({"QueryID": "cpd_id", "NeighbourID": "neighbour_id"})
        if "Distance" in nn_df.columns and "distance" not in nn_df.columns:
            alt_map["Distance"] = "distance"
        if alt_map:
            nn_df = nn_df.rename(columns=alt_map)
    if not nn_required.issubset(nn_df.columns):
        raise ValueError("nearest_neighbours.tsv must contain columns: cpd_id, neighbour_id, distance")

    nn_df["distance"] = pd.to_numeric(nn_df["distance"], errors="coerce")

    # UMAP header normalisation
    if "UMAP1" not in umap_df.columns or "UMAP2" not in umap_df.columns:
        rename_umap = {}
        if "UMAP_1" in umap_df.columns:
            rename_umap["UMAP_1"] = "UMAP1"
        if "UMAP_2" in umap_df.columns:
            rename_umap["UMAP_2"] = "UMAP2"
        if rename_umap:
            umap_df = umap_df.rename(columns=rename_umap)
    for col in ("UMAP1", "UMAP2", "cpd_id"):
        if col not in umap_df.columns:
            raise ValueError(f"UMAP coordinates file must contain '{col}'")

    # Normalise IDs
    umap_df["cpd_id"] = umap_df["cpd_id"].astype(str).str.upper().str.strip()
    if meta is not None:
        meta = meta.copy()
        meta["cpd_id"] = meta["cpd_id"].astype(str).str.upper().str.strip()

    # Process each target
    all_summaries: List[pd.DataFrame] = []
    for target in targets:
        print(f"\n[INFO] Processing target: {target}")
        t = str(target).upper().strip()

        top_nn = find_nearest_from_nn(nn_df, t, top_n, max_dist=max_dist)
        top_umap = find_nearest_umap(umap_df, t, top_n, max_dist=max_dist)

        combined = pd.concat([top_nn, top_umap], ignore_index=True)
        combined = combined.drop_duplicates(subset=["cpd_id", "nearest_cpd_id", "source"], keep="first")

        # Merge primary metadata
        if meta is not None and not combined.empty:
            combined = combined.merge(
                meta, left_on="nearest_cpd_id", right_on="cpd_id",
                how="left", suffixes=("", "_meta")
            )
            combined.drop(columns=["cpd_id_meta"], inplace=True, errors="ignore")

        # Merge extra metadata (if available)
        if extra_meta is not None and not combined.empty:
            combined = combined.merge(
                extra_meta, left_on="nearest_cpd_id", right_on="cpd_id",
                how="left", suffixes=("", "_extra")
            )
            combined.drop(columns=["cpd_id_extra"], inplace=True, errors="ignore")

        all_summaries.append(combined)

    final_summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    base_folder = os.path.basename(os.path.abspath(folder.rstrip("/")))
    out_path = os.path.join(folder, f"{base_folder}_summary_neighbours.tsv")
    excel_out_path = os.path.splitext(out_path)[0] + ".xlsx"

    if final_summary.empty:
        print("[WARNING] No neighbours found for the provided targets.")
        return

    # Write outputs (TSV + Excel)
    final_summary.to_csv(out_path, sep="\t", index=False)
    # Use openpyxl (available in your environment) to write XLSX
    final_summary.to_excel(excel_out_path, index=False, engine="openpyxl")

    print(f"[INFO] TSV summary written to: {out_path}")
    print(f"[INFO] Excel summary written to: {excel_out_path}")
    print("\n[INFO] Preview:")
    print(final_summary.head())
    print("[INFO] Done.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise neighbours of target compounds from CLIPn outputs.")
    parser.add_argument("--folder", required=True, help="Path to pipeline output folder")
    parser.add_argument(
        "--target", nargs="+",
        default=[
            "MCP09", "MCP05",
            "DDD02387619", "DDD02443214", "DDD02454019", "DDD02454403",
            "DDD02591200", "DDD02591362", "DDD02941115",
            "DDD02941193", "DDD02947912", "DDD02947919", "DDD02948915",
            "DDD02948916", "DDD02948926", "DDD02952619", "DDD02952620",
            "DDD02955130", "DDD02958365",
        ],
        help="List of target compound IDs (quote names containing spaces).",
    )
    parser.add_argument("--top_n", type=int, default=15, help="Top N neighbours to extract (default: 15).")
    parser.add_argument("--metadata", type=str, default=None, help="Path to primary compound metadata TSV/CSV.")
    parser.add_argument("--max_distance", type=float, default=None, help="Optional maximum distance threshold.")
    parser.add_argument("--extra_metadata", type=str, default=None, help="Path to secondary metadata TSV/CSV.")
    parser.add_argument("--nn_path", type=str, default=None, help="Optional override path for nearest_neighbours.tsv.")
    parser.add_argument("--umap_path", type=str, default=None, help="Optional override path for UMAP coords TSV.")

    args = parser.parse_args()
    summarise_neighbours(
        folder=args.folder,
        targets=args.target,
        top_n=args.top_n,
        metadata_file=args.metadata,
        max_dist=args.max_distance,
        extra_metadata=args.extra_metadata,
        nn_path=args.nn_path,
        umap_path=args.umap_path,
    )
