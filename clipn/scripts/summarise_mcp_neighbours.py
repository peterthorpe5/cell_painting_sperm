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
import glob
import argparse
import os
from typing import Optional, Tuple, List, Iterable
from pathlib import Path
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


def path_contains_component(path: str, name: str) -> bool:
    """
    Return True if 'path' has a component exactly equal to 'name' (case-insensitive).

    Parameters
    ----------
    path : str
        Filesystem path to inspect.
    name : str
        Component name to look for (e.g., 'lib').

    Returns
    -------
    bool
        True if any component equals 'name' ignoring case, otherwise False.
    """
    return any(part.lower() == name.lower() for part in Path(path).parts)



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




def _pick_latest(paths: Iterable[str]) -> Optional[str]:
    """
    Return the most-recently modified path from an iterable (or None if empty).

    Parameters
    ----------
    paths : Iterable[str]
        Candidate file paths.

    Returns
    -------
    Optional[str]
        Path with the latest modification time, or None if no valid files.
    """
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def autodiscover_path(root: str, patterns: list[str]) -> Optional[str]:
    """
    Search recursively under 'root' for any of the glob patterns and return the
    most recently modified match, excluding any hit within a 'lib' component.

    Parameters
    ----------
    root : str
        Root directory to search.
    patterns : list[str]
        Glob patterns relative to root.

    Returns
    -------
    Optional[str]
        Best matching path, or None if nothing matched.
    """
    hits: list[str] = []
    for pat in patterns:
        hits.extend(glob.glob(os.path.join(root, pat), recursive=True))
    # exclude anything inside a /lib/ component
    hits = [p for p in hits if not path_contains_component(p, "lib")]
    return _pick_latest(hits)


def find_nearest_from_nn(
    df: pd.DataFrame,
    target_id: str,
    top_n: int,
    max_dist: Optional[float] = None,
    *,
    source_label: str = "NN",
    distance_col: str = "distance_metric_NN",
) -> pd.DataFrame:
    """
    Retrieve top-N nearest neighbours from a nearest_neighbours.tsv-style table.

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
    source_label : str, optional
        Label to use in the 'source' column (default: 'NN').
    distance_col : str, optional
        Name of the output distance column (default: 'distance_metric_NN').

    Returns
    -------
    pandas.DataFrame
        Columns: ['cpd_id', 'nearest_cpd_id', <distance_col>, 'source'].
    """
    df = df.copy()
    for col in ("cpd_id", "neighbour_id"):
        df[col] = df[col].astype(str).str.upper().str.strip()
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    target_rows = df.loc[df["cpd_id"] == target_id.upper()]
    if target_rows.empty:
        print(f"[WARNING] Target compound '{target_id}' not found in nearest neighbour data")
        return pd.DataFrame(columns=["cpd_id", "nearest_cpd_id", distance_col, "source"])

    if max_dist is not None:
        target_rows = target_rows[target_rows["distance"] <= max_dist]

    target_rows = target_rows.sort_values("distance", ascending=True).drop_duplicates(subset=["neighbour_id"], keep="first")
    top_hits = target_rows.head(top_n).copy()
    top_hits["source"] = source_label
    top_hits = top_hits.rename(columns={"neighbour_id": "nearest_cpd_id", "distance": distance_col})

    return top_hits[["cpd_id", "nearest_cpd_id", distance_col, "source"]]



def summarise_neighbours(
    folder: str,
    targets: List[str],
    top_n: int = 15,
    metadata_file: Optional[str] = None,
    max_dist: Optional[float] = None,
    extra_metadata: Optional[str] = None,
    nn_path: Optional[str] = None,
    umap_path: Optional[str] = None,
    include_umap: bool = False,
    raw_nn_mode: str = "auto",
    raw_nn_path: Optional[str] = None,
) -> None:
    """
    Summarise nearest neighbours for target compounds and merge annotations.

    If include_umap is False (default)

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

    include_umap : bool, optional
        If True, include UMAP-based nearest neighbours. Default is False.

    If include_umap is False (default), only CLIPn NN results are used.
    Raw-feature baseline NN is controlled by 'raw_nn_mode':
      - 'auto' (default): include if a raw NN file is found.
      - 'on': attempt to include; warn if not found.
      - 'off': never include.

    Returns
    -------
    None
        Writes merged summary TSV and Excel files beside the CLIPn outputs.
    """
    # Resolve default input paths if not overridden
    # Skip folders that include a 'lib' path component
    if path_contains_component(folder, "lib"):
        print(f"[INFO] Skipping folder because it contains a 'lib' path component: {folder}")
        return

    nn_path = nn_path or os.path.join(folder, "post_clipn", "post_analysis_script", "nearest_neighbours.tsv")
    umap_path = umap_path or os.path.join(
        folder, "post_clipn", "UMAP_kNone", "cpd_type", "clipn_umap_coordinates_cosine_n15_d0.1.tsv"
    )

    # Try defaults; if missing, auto-discover; otherwise skip folder gracefully
    if not os.path.isfile(nn_path):
        print(f"[WARNING] Nearest neighbours not found at default: {nn_path}")
        nn_auto = autodiscover_path(
            folder,
            patterns=[
                "**/post_analysis_script/nearest_neighbours.tsv",
                "**/nearest_neighbours.tsv",
                "**/post_knn/nearest_neighbours.tsv",
            ],
        )
        if nn_auto:
            print(f"[INFO] Auto-discovered nearest neighbours: {nn_auto}")
            nn_path = nn_auto
        else:
            print(f"[INFO] Skipping folder (no nearest_neighbours.tsv under: {folder})")
            return
    # Only attempt UMAP discovery if explicitly requested
    if include_umap:
        if not os.path.isfile(umap_path):
            print(f"[WARNING] UMAP coordinates not found at default: {umap_path}")
            umap_auto = autodiscover_path(
                folder,
                patterns=[
                    "**/clipn_umap_coordinates_*cosine*.tsv",
                    "**/clipn_umap_coordinates_*.tsv",
                ],
            )
            if umap_auto:
                print(f"[INFO] Auto-discovered UMAP coordinates: {umap_auto}")
                umap_path = umap_auto
            else:
                print(f"[INFO] Skipping UMAP (no coords under: {folder})")
                include_umap = False



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


    # Optional UMAP
    umap_df = None
    if include_umap:
        umap_df = pd.read_csv(umap_path, sep="\t")

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
        #if meta is not None:
        #    meta = meta.copy()
        #    meta["cpd_id"] = meta["cpd_id"].astype(str).str.upper().str.strip()

    raw_nn_df = None
    if raw_nn_mode != "off":
        # If explicit path was not provided, try to auto-discover
        candidate_path = raw_nn_path
        if not candidate_path:
            # Common locations used by your pipeline – adjust if needed
            candidates = [
                "**/post_raw_nn/nearest_neighbours.tsv",
                "**/post_baseline_nn/nearest_neighbours.tsv",
                "**/nearest_neighbours_raw.tsv",
                "**/raw_nn/nearest_neighbours.tsv",
                "**/post_knn/nearest_neighbours.tsv",  # often used in your runs
            ]
            auto = autodiscover_path(folder, patterns=candidates)
            if auto:
                print(f"[INFO] Auto-discovered raw-feature NN: {auto}")
                candidate_path = auto

        if candidate_path and os.path.isfile(candidate_path):
            raw_nn_df = pd.read_csv(candidate_path, sep="\t")
            # Align headers if needed
            rn_required = {"cpd_id", "neighbour_id", "distance"}
            if not rn_required.issubset(raw_nn_df.columns):
                alt_map = {}
                if {"QueryID", "NeighbourID"}.issubset(raw_nn_df.columns):
                    alt_map.update({"QueryID": "cpd_id", "NeighbourID": "neighbour_id"})
                if "Distance" in raw_nn_df.columns and "distance" not in raw_nn_df.columns:
                    alt_map["Distance"] = "distance"
                if alt_map:
                    raw_nn_df = raw_nn_df.rename(columns=alt_map)
            if not rn_required.issubset(raw_nn_df.columns):
                raise ValueError("Raw-NN file must contain columns: cpd_id, neighbour_id, distance")
        else:
            if raw_nn_mode == "on":
                print("[WARNING] raw_nn_mode='on' but no raw-feature NN file was found.")
            else:
                print("[INFO] No raw-feature NN file found; continuing without it.")


    # Process each target
    all_summaries: List[pd.DataFrame] = []
    for target in targets:
        print(f"\n[INFO] Processing target: {target}")
        t = str(target).upper().strip()

        top_nn = find_nearest_from_nn(nn_df, t, top_n, max_dist=max_dist)
        combined = top_nn.copy()
        if include_umap and umap_df is not None:
            top_umap = find_nearest_umap(umap_df, t, top_n, max_dist=max_dist)
            combined = pd.concat([combined, top_umap], ignore_index=True)

            # Optional raw-feature baseline (auto/on)
        if raw_nn_df is not None:
            top_raw = find_nearest_from_nn(
                df=raw_nn_df, target_id=t, top_n=top_n, max_dist=max_dist,
                source_label="NN_RAW", distance_col="distance_metric_RAW"
            )
            combined = pd.concat([combined, top_raw], ignore_index=True)


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
            "DDD02387619", "DDD02443214", "DDD02454019", 
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
    # In argparse
    parser.add_argument(
        "--include_umap",
        action="store_true",
        help="Include UMAP nearest neighbour data (default: off)."
    )
    parser.add_argument(
        "--raw_nn_mode",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Include raw-feature baseline NN. "
            "'auto' (default): include if a file is found; "
            "'on': try to include (warn if missing); "
            "'off': do not include."
        ),
    )
    parser.add_argument(
        "--raw_nn_path",
        type=str,
        default=None,
        help="Optional override path to raw-feature nearest_neighbours.tsv."
    )



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
        include_umap=args.include_umap,
        raw_nn_mode=args.raw_nn_mode,
        raw_nn_path=args.raw_nn_path,)
