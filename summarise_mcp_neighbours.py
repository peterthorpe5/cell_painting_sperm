import argparse
import os
import pandas as pd
import numpy as np

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
        meta = pd.read_csv(path, sep="\t")
        return meta
    print("[WARNING] No compound metadata file found.")
    return None

def find_nearest_umap(df, target_id, top_n, max_dist=None):
    coords = df[["UMAP1", "UMAP2"]].values
    target_row = df[df["cpd_id"].str.upper() == target_id.upper()]
    if target_row.empty:
        print(f"[WARNING] Target compound '{target_id}' not found in UMAP coordinates")
        return pd.DataFrame()
    target_vec = target_row[["UMAP1", "UMAP2"]].values[0]
    dists = np.linalg.norm(coords - target_vec, axis=1)
    df = df.copy()
    df["UMAP_distance"] = dists
    if max_dist is not None:
        df = df[df["UMAP_distance"] <= max_dist]
    nearest = df.sort_values("UMAP_distance").iloc[1:top_n + 1]  # skip self
    return nearest

def find_nearest_from_nn(df, target_id, top_n, max_dist=None):
    df = df.copy()
    df[["cpd_id", "neighbour_id"]] = df[["cpd_id", "neighbour_id"]].apply(lambda x: x.str.upper())
    target_rows = df[df["cpd_id"] == target_id.upper()]
    if target_rows.empty:
        print(f"[WARNING] Target compound '{target_id}' not found in nearest neighbour data")
        return pd.DataFrame()
    if max_dist is not None:
        target_rows = target_rows[target_rows["distance"] <= max_dist]
    top_hits = target_rows.sort_values("distance", ascending=True).head(top_n)
    return top_hits

def summarise_neighbours(folder, targets, top_n=5, metadata_file=None, max_dist=None):
    nn_path = os.path.join(folder, "post_clipn", "post_analysis_script", "nearest_neighbours.tsv")
    umap_path = os.path.join(folder, "post_clipn", "UMAP_kNone", "cpd_type", "clipn_umap_coordinates_euclidean_n15_d0.1.tsv")
    meta = load_metadata(metadata_file)

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

        if not top_nn.empty:
            top_nn["source"] = "NN"
            top_nn = top_nn.rename(columns={"neighbour_id": "nearest_cpd_id", "distance": "distance_metric"})
            top_nn = top_nn[["cpd_id", "nearest_cpd_id", "distance_metric", "source"]]
        if not top_umap.empty:
            top_umap["source"] = "UMAP"
            top_umap = top_umap.rename(columns={"cpd_id": "nearest_cpd_id", "UMAP_distance": "distance_metric"})
            top_umap["cpd_id"] = target_upper
            top_umap = top_umap[["cpd_id", "nearest_cpd_id", "distance_metric", "source"]]

        combined = pd.concat([top_nn, top_umap], ignore_index=True)
        if meta is not None and not combined.empty:
            combined = combined.merge(meta, left_on="nearest_cpd_id", right_on="cpd_id", how="left", suffixes=("", "_meta"))
            combined.drop(columns=["cpd_id_meta"], inplace=True, errors="ignore")
        all_summaries.append(combined)

    final_summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    out_path = os.path.join(folder, "summarised_neighbours.tsv")
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
    parser.add_argument("--top_n", type=int, default=5, help="Top N neighbours to extract")
    parser.add_argument("--metadata", type=str, default=None, help="Path to compound metadata TSV file")
    parser.add_argument("--max_distance", type=float, default=None, help="Optional maximum distance threshold for filtering")
    args = parser.parse_args()

    summarise_neighbours(args.folder, args.target, args.top_n, args.metadata, args.max_distance)
