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

def find_nearest_umap(df, target_id, top_n):
    """
    Find top N closest compounds to target_id based on UMAP coordinates.
    """
    coords = df[["UMAP1", "UMAP2"]].values
    target_row = df[df["cpd_id"].str.upper() == target_id.upper()]
    if target_row.empty:
        raise ValueError(f"Target compound '{target_id}' not found in UMAP coordinates")
    target_vec = target_row[["UMAP1", "UMAP2"]].values[0]
    dists = np.linalg.norm(coords - target_vec, axis=1)
    df = df.copy()
    df["UMAP_distance"] = dists
    nearest = df.sort_values("UMAP_distance").iloc[1:top_n + 1]  # skip self
    return nearest

def find_nearest_from_nn(df, target_id, top_n):
    """
    Return top N nearest entries to target_id from NN distance matrix.
    """
    df = df.copy()
    df[["cpd_id", "neighbour_id"]] = df[["cpd_id", "neighbour_id"]].apply(lambda x: x.str.upper())
    target_rows = df[df["cpd_id"] == target_id.upper()]
    if target_rows.empty:
        raise ValueError(f"Target compound '{target_id}' not found in nearest neighbour data")
    top_hits = target_rows.sort_values("distance", ascending=True).head(top_n)
    return top_hits

def summarise_neighbours(folder, target="MCP09", top_n=5, metadata_file=None):
    """
    Summarise nearest compounds to a given target based on NN and UMAP distance.

    Parameters
    ----------
    folder : str
        Path to the CLIPn pipeline output root.
    target : str
        Target compound ID to find neighbours for.
    top_n : int
        Number of top neighbours to extract.
    metadata_file : str or None
        Optional path to metadata annotation file.
    """
    nn_path = os.path.join(folder, "post_clipn", "post_analysis_script", "nearest_neighbours.tsv")
    umap_path = os.path.join(folder, "post_clipn", "UMAP_kNone", "cpd_type", "clipn_umap_coordinates_euclidean_n15_d0.1.tsv")
    meta = load_metadata(metadata_file)

    nn_df = pd.read_csv(nn_path, sep="\t")
    umap_df = pd.read_csv(umap_path, sep="\t")

    # Standardise cpd_id casing
    umap_df["cpd_id"] = umap_df["cpd_id"].str.upper()
    target = target.upper()

    # Find neighbours
    top_nn = find_nearest_from_nn(nn_df, target, top_n)
    top_umap = find_nearest_umap(umap_df, target, top_n)

    if meta is not None:
        meta = meta.copy()
        meta["cpd_id"] = meta["cpd_id"].str.upper()

        top_nn = top_nn.merge(meta, left_on="neighbour_id", right_on="cpd_id", how="left", suffixes=("", "_meta"))
        top_umap = top_umap.merge(meta, on="cpd_id", how="left")

    print("\nTop Neighbours (Nearest Neighbours TSV):")
    print(top_nn[["neighbour_id", "distance"] + [col for col in meta.columns if col != "cpd_id"]] if meta is not None else top_nn)

    print("\nTop Neighbours (UMAP Distance):")
    print(top_umap[["cpd_id", "UMAP_distance"] + [col for col in meta.columns if col != "cpd_id"]] if meta is not None else top_umap)

    out_path = os.path.join(folder, f"nearest_neighbour_summary_{target}.tsv")
    with open(out_path, "w") as out:
        out.write("# Top Neighbours (Nearest Neighbours TSV)\n")
        top_nn.to_csv(out, sep="\t", index=False)
        out.write("\n# Top Neighbours (UMAP Distance)\n")
        top_umap.to_csv(out, sep="\t", index=False)

    print(f"\n[INFO] Summary written to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise neighbours of a target compound from CLIPn outputs")
    parser.add_argument("--folder", required=True, help="Path to pipeline output folder")
    parser.add_argument("--target", default="MCP09", help="Target compound ID (default=MCP09)")
    parser.add_argument("--top_n", type=int, default=5, help="Top N neighbours to extract")
    parser.add_argument("--metadata", type=str, default=None, help="Path to compound metadata TSV file")
    args = parser.parse_args()

    summarise_neighbours(args.folder, args.target, args.top_n, args.metadata)
