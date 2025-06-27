#!/usr/bin/env python3
"""
SHAP-based Feature Attribution for Cell Painting Nearest Neighbour Similarity
----------------------------------------------------------------------------

This script identifies which Cell Painting features drive the similarity
between query compounds and their nearest neighbours in latent space.

Inputs:
    --features: Well-level feature file (TSV) or file-of-files with column 'path'.
    --nn_file: Tab-separated file with columns: 'query_id', 'neighbour_id'.
    --query_id: Comma-separated list or filename, each line a compound ID.
    --output_dir: Output directory for results.

Outputs:
    - Top N SHAP features per query (TSV).
    - SHAP summary plot per query (PDF).

Example usage:
    python shap_explain_nn_similarity.py \
        --features dataset_paths.txt \
        --nn_file nearest_neighbours.tsv \
        --query_id DDD02387619,DDD02948916,DDD02955130,DDD02958365 \
        --output_dir ./shap_results \
        --n_neighbors 5 \
        --n_top_features 10

Author: Your Name, 2024
"""

import os
import argparse
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import logging


def setup_logger(log_path):
    """
    Set up a logger that writes to file and stdout.
    Args:
        log_path (str): Path to log file.
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger('shap_nn_similarity')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    # File handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_feature_files(list_file, logger):
    """
    Load and concatenate all well-level feature files listed in a file.
    Args:
        list_file (str): Path to a CSV/TSV with column 'path' or a single TSV file.
        logger (logging.Logger): Logger.
    Returns:
        pd.DataFrame: Concatenated well-level data.
    """
    logger.info(f"Attempting to load feature data from {list_file}")
    ext = os.path.splitext(list_file)[-1]
    # Try to load as file-of-files
    try:
        df_list = pd.read_csv(list_file, sep=None, engine="python")
        if "path" in df_list.columns:
            logger.info(f"File-of-files detected with {len(df_list)} files.")
            dfs = []
            for path in df_list["path"]:
                logger.info(f"Reading feature file: {path}")
                dfs.append(pd.read_csv(path, sep="\t"))
            # Harmonise columns
            common_cols = set(dfs[0].columns)
            for d in dfs[1:]:
                common_cols &= set(d.columns)
            dfs = [d[list(common_cols)] for d in dfs]
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined features shape: {combined.shape}")
            return combined
    except Exception as e:
        logger.warning(f"Could not load as file-of-files: {e} — will try as single file.")

    # Else: try loading as a single TSV
    try:
        df = pd.read_csv(list_file, sep="\t")
        logger.info(f"Loaded single feature file, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Could not load feature file: {e}")
        raise


def parse_query_ids(query_id_arg, logger):
    """
    Parse query IDs from a comma-separated string or text file.
    Args:
        query_id_arg (str): Comma-separated string or filename.
        logger (logging.Logger): Logger.
    Returns:
        list: List of query IDs as strings.
    """
    if os.path.isfile(query_id_arg):
        logger.info(f"Parsing query IDs from file: {query_id_arg}")
        with open(query_id_arg, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Found {len(ids)} query IDs in file.")
        return ids
    # Otherwise, treat as comma-separated string
    ids = [x.strip() for x in query_id_arg.split(',') if x.strip()]
    logger.info(f"Parsed {len(ids)} query IDs from string.")
    return ids

def load_data(features_file, nn_file, query_id, n_neighbors, logger):
    """
    Load well-level data and subset for query and its N nearest neighbours.
    Args:
        features_file (str): Path to features file or file-of-files.
        nn_file (str): Path to nearest neighbours file.
        query_id (str): Query compound ID.
        n_neighbors (int): Number of neighbours to use.
        logger (logging.Logger): Logger.
    Returns:
        pd.DataFrame: Features for query and NNs, with 'target' column.
    """
    features = load_feature_files(features_file, logger)
    nn = pd.read_csv(nn_file, sep="\t")
    logger.info(f"Loaded NN file ({nn.shape[0]} rows) from {nn_file}")

    # Handle different column names for queries
    nn_query_col = 'query_id' if 'query_id' in nn.columns else 'cpd_id'
    nn_neigh_col = 'neighbour_id' if 'neighbour_id' in nn.columns else 'nn_id'

    neighbours = nn.loc[nn[nn_query_col].astype(str) == str(query_id), nn_neigh_col].astype(str).unique()
    if len(neighbours) == 0:
        logger.error(f"No neighbours found for query {query_id} in {nn_file}")
        raise ValueError(f"No neighbours found for query {query_id} in {nn_file}")
    neighbours = neighbours[:n_neighbors]
    ids_of_interest = [query_id] + list(neighbours)
    logger.info(f"Query {query_id}: {len(neighbours)} neighbours: {neighbours}")

    features = features[features['cpd_id'].astype(str).isin(ids_of_interest)].copy()
    if features.empty:
        logger.error(f"No matching wells found for query {query_id} or its NNs in feature file.")
        raise ValueError(f"No matching wells for query {query_id} or NNs.")
    features['target'] = (features['cpd_id'].astype(str) == query_id).astype(int)
    logger.info(f"Subset features shape for query and NNs: {features.shape}")
    return features


def run_shap(features, n_top_features, output_dir, query_id, logger):
    """
    Fit model, run SHAP, and output summary.
    Args:
        features (pd.DataFrame): Data for query and NNs, with 'target' column.
        n_top_features (int): Number of top features to output.
        output_dir (str): Output directory.
        query_id (str): Query compound ID.
        logger (logging.Logger): Logger.
    Outputs:
        - TSV of top SHAP features.
        - PDF SHAP summary plot.
    """
    logger.info(f"Running SHAP for query {query_id}...")
    non_feature_cols = {'cpd_id', 'target', 'Dataset', 'Library', 'Plate_Metadata', 'Well_Metadata', 'query_id', 'neighbour_id'}
    # Only numeric columns, and not metadata
    feature_cols = [c for c in features.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(features[c])]
    X = features[feature_cols]
    y = features['target']

    if X.shape[1] == 0:
        logger.error("No feature columns detected for SHAP analysis.")
        raise ValueError("No feature columns detected for SHAP analysis.")

    logger.info(f"Fitting RandomForestClassifier on {X.shape[0]} samples, {X.shape[1]} features.")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    logger.info("Model fit complete. Running SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP values for 'query' class (label 1)
    shap_vals_query = shap_values[1]
    feature_importance = np.abs(shap_vals_query).mean(axis=0)
    top_idx = np.argsort(feature_importance)[::-1][:n_top_features]
    top_features = X.columns[top_idx]
    top_importance = feature_importance[top_idx]

    out_tsv = os.path.join(output_dir, f"{query_id}_top_shap_features.tsv")
    pd.DataFrame({'feature': top_features, 'mean_abs_shap': top_importance}).to_csv(out_tsv, sep="\t", index=False)
    logger.info(f"Wrote top {n_top_features} SHAP features for {query_id} to {out_tsv}")

    # SHAP summary plot
    out_pdf = os.path.join(output_dir, f"{query_id}_shap_summary.pdf")
    logger.info(f"Generating SHAP summary plot for {query_id}")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_query, X, feature_names=X.columns, show=False, max_display=n_top_features)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    logger.info(f"Wrote SHAP summary plot for {query_id} to {out_pdf}")


def main():
    """
    Main function to run SHAP-based feature attribution for NN similarity.
    """
    parser = argparse.ArgumentParser(description="SHAP explain NN similarity in Cell Painting data.")
    parser.add_argument('--features', required=True, help="Well-level feature TSV or file-of-files.")
    parser.add_argument('--nn_file', required=True, help="Nearest neighbours TSV.")
    parser.add_argument('--query_id', required=False,
                        default="DDD02387619,DDD02948916,DDD02955130,DDD02958365",
                        help="Comma-separated list or filename with query IDs.")
    parser.add_argument('--output_dir', required=True, help="Output directory.")
    parser.add_argument('--n_neighbors', type=int, default=5, help="Number of neighbours to explain.")
    parser.add_argument('--n_top_features', type=int, default=10, help="Top N SHAP features to output.")
    parser.add_argument('--log_file', default="shap_explain.log", help="Log file name.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger = setup_logger(log_path)
    logger.info(f"Script arguments: {vars(args)}")

    query_ids = parse_query_ids(args.query_id, logger)
    logger.info(f"Running SHAP analysis for {len(query_ids)} query compound(s): {query_ids}")

    for query_id in query_ids:
        logger.info(f"===== Analysing: {query_id} =====")
        try:
            features = load_data(args.features, args.nn_file, query_id, args.n_neighbors, logger)
            run_shap(features, args.n_top_features, args.output_dir, query_id, logger)
            logger.info(f"SHAP analysis complete for {query_id}. Results written to {args.output_dir}")
        except Exception as e:
            logger.error(f"Failed for {query_id}: {e}")

    logger.info("All analyses complete.")

if __name__ == "__main__":
    main()
