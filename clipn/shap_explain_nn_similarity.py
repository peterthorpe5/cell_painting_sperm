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

#!/usr/bin/env python3
"""
SHAP-based Feature Attribution for Cell Painting Nearest Neighbour Similarity

[...as before...]
"""

import os
import argparse
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import logging

def setup_logger(log_file):
    """
    Set up logging to file and stdout.

    Args:
        log_file (str): Path to log file.

    Returns:
        logging.Logger: Logger object.
    """
    logger = logging.getLogger("shap_explain")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def run_shap(features, n_top_features, output_dir, query_id, logger):
    """
    Fit model, run SHAP, and output summary.

    Args:
        features (pd.DataFrame): Data for query and NNs, with 'target' column.
        n_top_features (int): Number of top features to output.
        output_dir (str): Output directory.
        query_id (str): Query compound ID.
        logger (logging.Logger): Logger object.

    Outputs:
        - TSV of top SHAP features.
        - PDF SHAP summary plot.
    """
    non_feature_cols = ['cpd_id', 'target', 'Dataset', 'Library', 'Plate_Metadata', 'Well_Metadata']
    feature_cols = [c for c in features.columns if c not in non_feature_cols and features[c].dtype in [np.float32, np.float64, np.int64, np.int32]]

    logger.info(f"Number of feature columns available for SHAP: {len(feature_cols)}")
    logger.debug(f"Feature columns: {feature_cols}")

    X = features[feature_cols]
    y = features['target']
    logger.info(f"Shape of X (features): {X.shape}")
    logger.info(f"Value counts for y (target):\n{y.value_counts(dropna=False)}")

    if X.shape[1] == 0:
        logger.error("No feature columns detected for SHAP analysis. Exiting.")
        raise ValueError("No feature columns detected for SHAP analysis.")

    if len(np.unique(y)) != 2:
        logger.error(f"Target variable y must have exactly 2 classes (query vs NN). Found: {np.unique(y)}")
        raise ValueError("Target variable y must have exactly 2 classes.")

    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        logger.info("RandomForestClassifier model fit successfully.")
    except Exception as e:
        logger.error(f"RandomForestClassifier fit failed: {e}")
        raise

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        logger.info("SHAP values computed successfully.")
    except Exception as e:
        logger.error(f"SHAP value computation failed: {e}")
        raise

    # SHAP values for 'query' class (label 1)
    shap_vals_query = shap_values[1]
    feature_importance = np.abs(shap_vals_query).mean(axis=0)
    n_nonzero = (feature_importance > 0).sum()
    logger.info(f"Number of features with nonzero mean_abs_shap: {n_nonzero}")

    top_idx = np.argsort(feature_importance)[::-1][:n_top_features]
    top_features = X.columns[top_idx]
    top_importance = feature_importance[top_idx]

    logger.info(f"Top {n_top_features} features (by mean_abs_shap): {top_features.tolist()}")
    out_tsv = os.path.join(output_dir, f"{query_id}_top_shap_features.tsv")
    pd.DataFrame({'feature': top_features, 'mean_abs_shap': top_importance}).to_csv(out_tsv, sep="\t", index=False)
    logger.info(f"Wrote top SHAP features TSV: {out_tsv}")

    # Optionally: also output all features, sorted
    all_out_tsv = os.path.join(output_dir, f"{query_id}_all_shap_features.tsv")
    df_all = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': feature_importance})
    df_all = df_all.sort_values('mean_abs_shap', ascending=False)
    df_all.to_csv(all_out_tsv, sep="\t", index=False)
    logger.info(f"Wrote all SHAP features TSV: {all_out_tsv}")

    # Plot
    plt.figure(figsize=(10, 6))
    try:
        shap.summary_plot(shap_vals_query, X, feature_names=X.columns, show=False, max_display=n_top_features)
        plt.tight_layout()
        out_pdf = os.path.join(output_dir, f"{query_id}_shap_summary.pdf")
        plt.savefig(out_pdf)
        plt.close()
        logger.info(f"Wrote SHAP summary plot PDF: {out_pdf}")
    except Exception as e:
        logger.error(f"Failed to generate SHAP summary plot: {e}")

def load_feature_files(list_file, logger):
    """
    Load and concatenate all well-level feature files listed in a file.

    Args:
        list_file (str): Path to a CSV/TSV with column 'path' or a single TSV file.
        logger (logging.Logger): Logger.

    Returns:
        pd.DataFrame: Concatenated well-level data.
    """
    ext = os.path.splitext(list_file)[-1]
    if ext in [".csv", ".tsv", ".txt"]:
        try:
            df_list = pd.read_csv(list_file, sep=None, engine="python")
            if "path" in df_list.columns:
                logger.info(f"Loading list of {len(df_list)} feature files as input.")
                dfs = [pd.read_csv(p, sep="\t") for p in df_list["path"]]
                # Harmonise columns
                common_cols = set(dfs[0].columns)
                for d in dfs[1:]:
                    common_cols &= set(d.columns)
                dfs = [d[list(common_cols)] for d in dfs]
                df = pd.concat(dfs, ignore_index=True)
                logger.info(f"Concatenated feature dataframe shape: {df.shape}")
                return df
        except Exception as e:
            logger.warning(f"Failed to load as list-of-files, will try as single file. Error: {e}")

    # Else: try loading as a single TSV
    df = pd.read_csv(list_file, sep="\t")
    logger.info(f"Loaded single feature file: {list_file}, shape: {df.shape}")
    return df

def load_data(features_file, nn_file, query_id, n_neighbors, logger):
    """
    Load well-level data and subset wells for the query and its N nearest neighbours.

    Args:
        features_file (str): Path to well-level feature file or list-of-files TSV/CSV.
        nn_file (str): Path to nearest neighbours file (TSV).
        query_id (str): Query compound ID.
        n_neighbors (int): Number of neighbours to use.
        logger (logging.Logger): Logger.

    Returns:
        pd.DataFrame: Feature data for query and NNs, with 'target' column.
    """
    features = load_feature_files(features_file, logger)
    logger.info(f"Input features file shape: {features.shape}")
    nn = pd.read_csv(nn_file, sep="\t")
    logger.info(f"NN table shape: {nn.shape}")

    neighbours = nn.loc[nn['query_id'].astype(str) == str(query_id), 'neighbour_id'].astype(str).unique()
    logger.info(f"Found {len(neighbours)} neighbours for query {query_id}: {neighbours}")

    if len(neighbours) == 0:
        logger.error(f"No neighbours found for query {query_id} in {nn_file}")
        raise ValueError(f"No neighbours found for query {query_id} in {nn_file}")
    neighbours = neighbours[:n_neighbors]
    ids_of_interest = [query_id] + list(neighbours)

    features_subset = features[features['cpd_id'].astype(str).isin(ids_of_interest)].copy()
    logger.info(f"Subset features shape (query + NNs): {features_subset.shape}")

    features_subset['target'] = (features_subset['cpd_id'].astype(str) == query_id).astype(int)
    logger.info(f"Target class distribution in subset:\n{features_subset['target'].value_counts(dropna=False)}")

    return features_subset

def parse_query_ids(query_id_arg):
    """
    Parse query IDs from a comma-separated string or text file.

    Args:
        query_id_arg (str): Comma-separated string or filename.

    Returns:
        list: List of query IDs as strings.
    """
    if os.path.isfile(query_id_arg):
        with open(query_id_arg, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        return ids
    return [x.strip() for x in query_id_arg.split(',') if x.strip()]

def main():
    """
    Main function to run SHAP-based feature attribution for NN similarity.
    """
    parser = argparse.ArgumentParser(description="SHAP explain NN similarity in Cell Painting data.")
    parser.add_argument('--features', required=True, help="Well-level feature TSV or list of files.")
    parser.add_argument('--nn_file', required=True, help="Nearest neighbours TSV.")
    parser.add_argument('--query_id', required=False,
                        default="DDD02387619,DDD02948916,DDD02955130,DDD02958365",
                        help="Comma-separated list or filename with query IDs.")
    parser.add_argument('--output_dir', required=True, help="Output directory.")
    parser.add_argument('--n_neighbors', type=int, default=5, help="Number of neighbours to explain.")
    parser.add_argument('--n_top_features', type=int, default=10, help="Top N SHAP features to output.")
    parser.add_argument('--log_file', default="shap_explain.log", help="Log file path.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, args.log_file))
    logger.info(f"Arguments: {args}")

    query_ids = parse_query_ids(args.query_id)
    logger.info(f"Running SHAP analysis for {len(query_ids)} query compound(s): {query_ids}")

    for query_id in query_ids:
        logger.info(f"===== Analysing query: {query_id} =====")
        try:
            features = load_data(args.features, args.nn_file, query_id, args.n_neighbors, logger)
            run_shap(features, args.n_top_features, args.output_dir, query_id, logger)
            logger.info(f"SHAP analysis complete for {query_id}. Results written to {args.output_dir}")
        except Exception as e:
            logger.error(f"Failed for {query_id}: {e}")

if __name__ == "__main__":
    main()

