"""
Summarise CLIPn Nearest Neighbours and Annotate Output

This script identifies the top-N nearest neighbour compounds for specified target compounds
based on both UMAP 2D projections and CLIPn latent-space nearest neighbour analysis.
It supports merging compound-level annotations from one or more metadata files.

Features
--------
- Computes top-N nearest neighbours by cosine distance in UMAP space.
- Extracts top-N nearest neighbours from CLIPn NN output.
- Merges compound annotations from a primary metadata TSV and an optional secondary CSV.
- Outputs results to both TSV and Excel (.xlsx) formats.

Typical usage example:
----------------------
python summarise_mcp_neighbours.py \
    --folder path/to/clipn/output \
    --metadata compound_annotations.tsv \
    --extra_metadata extra_annotations.csv \
    --top_n 10
"""

import argparse
import os
import pandas as pd
import numpy as np
import openpyxl

def load_metadata(path):
    """
    Load compound metadata from a given annotation file.

    Parameters
    ----------
    path : str
        Path to metadata TSV file.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with metadata or None if file not found.
    """
    if path and os.path.isfile(path):
        print(f"[INFO] Found compound metadata file: {path}")
        try:
            meta = pd.read_csv(path, sep=None, engine="python")  # auto-detect , or \t
        except Exception:
            meta = pd.read_csv(path)  # fallback (comma)
        meta.columns = [c.strip() for c in meta.columns]
        return meta
    print("[WARNING] No compound metadata file found.")
    return None


def find_nearest_umap(df, target_id, top_n, max_dist=None):
    """
    Find top-N nearest neighbours in UMAP 2D space using  distance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'UMAP1', 'UMAP2', and 'cpd_id' columns.
    target_id : str
        Compound ID for which neighbours will be found.
    top_n : int
        Number of nearest neighbours to retrieve.
    max_dist : float, optional
        Optional maximum distance threshold.

    Returns
    -------
    pd.DataFrame
        DataFrame with nearest neighbour IDs, distances, and source label.
    """

    coords = df[["UMAP1", "UMAP2"]].values
    target_row = df[df["cpd_id"].str.upper() == target_id.upper()]
    if target_row.empty:
        print(f"[WARNING] Target compound '{target_id}' not found in UMAP coordinates")
        return pd.DataFrame()
    target_vec = target_row[["UMAP1", "UMAP2"]].values[0]
    dists = np.linalg.norm(coords - target_vec, axis=1)
    df = df.copy()
    df["distance_metric_UMAP"] = dists
    if max_dist is not None:
        df = df[df["distance_metric_UMAP"] <= max_dist]

    # sort by distance then keep one row per compound
    df = df.sort_values("distance_metric_UMAP")
    df = df.drop_duplicates(subset=["cpd_id"], keep="first")
    # drop self explicitly, then take top-N unique compounds
    nearest = df[df["cpd_id"].str.upper() != target_id.upper()].head(top_n).copy()
    nearest["source"] = "UMAP"
    nearest = nearest.rename(columns={"cpd_id": "nearest_cpd_id"})
    nearest["cpd_id"] = target_id.upper()
    return nearest[["cpd_id", "nearest_cpd_id", "distance_metric_UMAP", "source"]]



def find_nearest_from_nn(df, target_id, top_n, max_dist=None):
    """
    Retrieve top-N nearest neighbours from CLIPn-generated NN results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'cpd_id', 'neighbour_id', and 'distance'.
    target_id : str
        Target compound ID for which neighbours will be identified.
    top_n : int
        Number of nearest neighbours to return.
    max_dist : float, optional
        Optional maximum distance threshold for filtering.

    Returns
    -------
    pd.DataFrame
        Filtered and renamed DataFrame of nearest neighbours.
    """

    df = df.copy()
    df[["cpd_id", "neighbour_id"]] = df[["cpd_id", "neighbour_id"]].apply(lambda x: x.str.upper())
    target_rows = df[df["cpd_id"] == target_id.upper()]
    if target_rows.empty:
        print(f"[WARNING] Target compound '{target_id}' not found in nearest neighbour data")
        return pd.DataFrame()
    if max_dist is not None:
        target_rows = target_rows[target_rows["distance"] <= max_dist]

    # sort by distance and keep one row per neighbour compound
    target_rows = target_rows.sort_values("distance", ascending=True)
    target_rows = target_rows.drop_duplicates(subset=["neighbour_id"], keep="first")

    top_hits = target_rows.head(top_n).copy()
    top_hits["source"] = "NN"
    top_hits = top_hits.rename(columns={"neighbour_id": "nearest_cpd_id", "distance": "distance_metric_NN"})
    return top_hits[["cpd_id", "nearest_cpd_id", "distance_metric_NN", "source"]]


def summarise_neighbours(folder, targets, top_n=15, metadata_file=None, max_dist=None, extra_metadata=None):
    """
    Summarise nearest neighbours for a list of target compounds and merge annotations.

    Parameters
    ----------
    folder : str
        Path to the base CLIPn output folder.
    targets : list of str
        List of compound IDs to analyse.
    top_n : int, optional
        Number of top neighbours to retrieve (default is 5).
    metadata_file : str, optional
        Path to the primary compound metadata file (TSV).
    max_dist : float, optional
        Optional maximum distance cutoff for filtering.
    extra_metadata : str, optional
        Path to a second metadata file (CSV) to merge on 'COMPOUND_NAME'.

    Returns
    -------
    None
        Writes merged summary results to TSV and Excel files.
    """


    extra_meta = None
    if extra_metadata and os.path.isfile(extra_metadata):
        print(f"[INFO] Found extra annotation file: {extra_metadata}")
        extra_meta = pd.read_csv(extra_metadata)
        extra_meta["COMPOUND_NAME"] = extra_meta["COMPOUND_NAME"].str.upper()
    else:
        print("[INFO] No extra annotation file provided or file not found.")

    nn_path = os.path.join(folder, "post_clipn", "post_analysis_script", "nearest_neighbours.tsv")
    umap_path = os.path.join(folder, "post_clipn", "UMAP_kNone", "cpd_type", "clipn_umap_coordinates_cosine_n15_d0.1.tsv")
    meta = load_metadata(metadata_file)
    # --- normalise metadata ID column to 'cpd_id' ---
    if meta is not None:
        if "cpd_id" not in meta.columns:
            for cand in [
                "COMPOUND_NAME", "Compound", "compound", "compound_id",
                "CPD_ID", "cpd", "Compound Name"
            ]:
                if cand in meta.columns:
                    meta = meta.rename(columns={cand: "cpd_id"})
                    break
        if "cpd_id" in meta.columns:
            meta["cpd_id"] = meta["cpd_id"].astype(str).str.upper().str.strip()
            # Ensure compound-level metadata to avoid join fan-out
            meta = meta.drop_duplicates(subset=["cpd_id"])
        else:
            print("[WARNING] Metadata has no cpd_id-like column; skipping metadata merge.")
            meta = None


    nn_df = pd.read_csv(nn_path, sep="\t")
    umap_df = pd.read_csv(umap_path, sep="\t")

    umap_df["cpd_id"] = umap_df["cpd_id"].str.upper()
    if meta is not None:
        meta = meta.copy()
        meta["cpd_id"] = meta["cpd_id"].str.upper()

    all_summaries = []

    for target in targets:
        print(f"\n[INFO] Processing target: {target}")
        target_upper = target.upper()
        top_nn = find_nearest_from_nn(nn_df, target_upper, top_n, max_dist=max_dist)
        top_umap = find_nearest_umap(umap_df, target_upper, top_n, max_dist=max_dist)

        combined = pd.concat([top_nn, top_umap], ignore_index=True)
        # keep unique neighbour per source to avoid accidental duplicates
        combined = combined.drop_duplicates(subset=["cpd_id", "nearest_cpd_id", "source"], keep="first")

        if meta is not None and not combined.empty:
            combined = combined.merge(meta, left_on="nearest_cpd_id", right_on="cpd_id", how="left", suffixes=("", "_meta"))
            combined.drop(columns=["cpd_id_meta"], inplace=True, errors="ignore")

        if extra_meta is not None and not combined.empty:
            combined = combined.merge(extra_meta, left_on="nearest_cpd_id", right_on="COMPOUND_NAME", how="left", suffixes=("", "_extra"))


        all_summaries.append(combined)

    final_summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    base_folder = os.path.basename(os.path.abspath(folder.rstrip("/")))
    out_path = os.path.join(folder, f"{base_folder}_summary_neighbours.tsv")
    excel_out_path = os.path.splitext(out_path)[0] + ".xlsx"
    final_summary.to_excel(excel_out_path, index=False, engine='xlsxwriter')

    print(f"[INFO] Excel summary written to: {excel_out_path}")

    final_summary.to_csv(out_path, sep="\t", index=False)

    print("\n[INFO] Combined neighbour summary:")
    print(final_summary.head())
    print(f"\n[INFO] Full summary written to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise neighbours of target compounds from CLIPn outputs")
    parser.add_argument("--folder", required=True, help="Path to pipeline output folder")
    parser.add_argument("--target", nargs="+", default=["MCP09", "MCP05",
                                                        'DDD02387619', 'DDD02443214', 'DDD02454019', 'DDD02454403', 
                                                        'DDD02459457', 'DDD02487111', 'DDD02487311', 'DDD02589868', 
                                                        'DDD02591200', 'DDD02591362', 'DDD02941115', 
                                                        'DDD02941193', 'DDD02947912', 'DDD02947919', 'DDD02948915', 
                                                        'DDD02948916', 'DDD02948926', 'DDD02952619', 'DDD02952620', 
                                                        'DDD02955130', 'DDD02958365'], help="List of target compound IDs")
    parser.add_argument("--top_n", type=int, default=15, 
                        help="Top N neighbours to extract")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Path to compound metadata TSV file")
    parser.add_argument("--max_distance", type=float, default=None, 
                        help="Optional maximum distance threshold for filtering")
    parser.add_argument("--extra_metadata", type=str, default=None, 
                        help="Optional second metadata CSV file with COMPOUND_NAME to merge")

    args = parser.parse_args()

    summarise_neighbours(args.folder, args.target, args.top_n, args.metadata, 
                         args.max_distance, args.extra_metadata)

