#!/usr/bin/env python3
"""
SHAP-based Feature Attribution for Cell Painting Nearest Neighbour Similarity
----------------------------------------------------------------------------

This script identifies which Cell Painting features drive the similarity
between query compounds and their nearest neighbours in latent space.

To explain which Cell Painting features distinguish a query compound from its nearest neighbours, 
using SHAP (SHapley Additive exPlanations) and a machine learning model.

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

--n_neighbors 5
For each query compound, the script collects the feature data for that compound plus its 5 
nearest neighbours (i.e. a total of 6 sets of well data: 1 query + 5 NNs, but if you have multiple wells per compound, 
all wells are included for each compound).

--n_top_features 10
It then uses a Random Forest model and SHAP to rank all features by their mean absolute SHAP value—i.e., 
how much each feature contributes to distinguishing the query compound from its NNs.
The output is the top 10 features (with the highest mean absolute SHAP values) that most strongly 
separate the query from its neighbours.
        

Machine Learning Model Fitting & SHAP Value Computation
(removes metadata from this): Chooses a model depending on sample size:
If there are fewer than 30 wells, it uses logistic regression. Otherwise, it uses a random forest.
Fits the model to distinguish the query wells (target=1) from neighbour wells (target=0).
Computes SHAP values for every feature, for every well.
These values quantify how much each feature pushes the prediction towards "query" or "NN" for each well.
Why SHAP?
a robust way to say: “for this sample, feature X contributed Y amount to it being class 1 vs class 0”.

SHAP values assign to each feature, for each sample, a number showing how much that feature 
contributed to the model’s prediction (compared to the “average” prediction if that feature was absent).

Using SHAP to find feature dirivng similarity: by default SHAP find the features drving the difference
indirect inference, not a direct test of similarity ... sort by the mean SHAP and lowest values first. 

"""

import os
import argparse
import pandas as pd
import numpy as np
import shap
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


def plot_shap_summary(X, shap_values, feature_names, output_file, logger, n_top_features=10):
    """
    Generate a SHAP summary plot and save to output_file. Handles binary classification.
    """
    try:
        # If list (as in binary classification), pick class 1
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_array = np.asarray(shap_values[1])
            else:
                shap_array = np.asarray(shap_values[0])
        else:
            shap_array = np.asarray(shap_values)

        shap_array = np.squeeze(shap_array)
        logger.info(f"shap_array type: {type(shap_array)}, shape: {getattr(shap_array, 'shape', None)}")
        if shap_array.ndim != 2:
            logger.error(f"SHAP array shape is not 2D after squeeze: {shap_array.shape}")
            return

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_array, X.values, feature_names=feature_names,
            show=False, max_display=n_top_features, plot_type="bar"
        )
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Wrote SHAP summary plot: {output_file}")
    except Exception as e:
        logger.error(f"Could not generate SHAP plot: {e}")


def plot_shap_waterfall(X, shap_values, feature_names, output_file, logger, sample_ind=0, max_display=10):
    """
    Generate a SHAP waterfall plot for a single sample (e.g., well) and save as PDF.
    
    Args:
        X (pd.DataFrame): Feature data.
        shap_values (np.ndarray): SHAP values array.
        feature_names (list): Feature names.
        output_file (str): Output PDF path.
        logger (logging.Logger): Logger.
        sample_ind (int): Index of the sample to plot.
        max_display (int): Max number of features to show.
    """
    try:
        plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[sample_ind],
                base_values=0,  # base value can be taken from SHAP explainer if desired
                data=X.iloc[sample_ind].values,
                feature_names=feature_names
            ),
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Wrote SHAP waterfall plot: {output_file}")
    except Exception as e:
        logger.error(f"Could not generate SHAP waterfall plot: {e}")


def plot_feature_importance_bar(features, importance, output_file, title, logger):
    """
    Plot a horizontal bar plot of feature importances and save as PDF.
    
    Args:
        features (pd.Index or list): Feature names.
        importance (np.array or list): Importances (e.g. mean abs SHAP).
        output_file (str): Path to output PDF.
        title (str): Plot title.
        logger (logging.Logger): Logger for messages.
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 0.5 * len(features) + 2))
        y = np.arange(len(features))
        plt.barh(y, importance[::-1], align='center')
        plt.yticks(y, [f for f in features[::-1]])
        plt.xlabel("Mean absolute SHAP value")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Wrote feature importance bar plot: {output_file}")
    except Exception as e:
        logger.error(f"Could not generate bar plot: {e}")


def plot_shap_summary_all(X, shap_values, feature_names, output_prefix, logger, n_top_features=10):
    """
    Generate both SHAP summary bar and beeswarm (dot) plots for feature attribution.
    Args:
        X (pd.DataFrame): Feature data (n_samples, n_features).
        shap_values (np.ndarray): SHAP values (n_samples, n_features).
        feature_names (list): List of feature names.
        output_prefix (str): Output file path prefix (without extension).
        logger (logging.Logger): Logger for messages.
        n_top_features (int): Number of features to plot.
    """
    try:
        # --- Bar chart (mean |SHAP| values) ---
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X.values, feature_names=feature_names,
            show=False, max_display=n_top_features, plot_type="bar"
        )
        plt.tight_layout()
        bar_path = f"{output_prefix}_bar.pdf"
        plt.savefig(bar_path)
        plt.close()
        logger.info(f"Wrote SHAP summary bar plot: {bar_path}")

        # --- Beeswarm (dot) chart (classic SHAP summary plot) ---
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X.values, feature_names=feature_names,
            show=False, max_display=n_top_features, plot_type="dot"
        )
        plt.tight_layout()
        beeswarm_path = f"{output_prefix}_beeswarm.pdf"
        plt.savefig(beeswarm_path)
        plt.close()
        logger.info(f"Wrote SHAP summary beeswarm plot: {beeswarm_path}")
    except Exception as e:
        logger.error(f"Could not generate SHAP summary plots: {e}")


def plot_shap_heatmap(X, shap_values, feature_names, output_file, logger, max_display=20):
    """
    Generate a SHAP heatmap and save as PDF.

    Args:
        X (pd.DataFrame): Feature data (samples × features).
        shap_values (np.ndarray): SHAP values (samples × features).
        feature_names (list): List of feature names.
        output_file (str): Path to save the PDF.
        logger (logging.Logger): Logger object.
        max_display (int): Maximum number of features to display in the heatmap.
    """
    try:
        plt.figure(figsize=(max_display, min(X.shape[0], 40) / 2 + 4))
        shap.plots.heatmap(
            shap.Explanation(
                values=shap_values,
                data=X.values,
                feature_names=feature_names
            ),
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Wrote SHAP heatmap plot: {output_file}")
    except Exception as e:
        logger.error(f"Could not generate SHAP heatmap plot: {e}")



def run_shap(features, n_top_features, output_dir, query_id, logger, small_sample_threshold=30):
    """
    Fit model (logistic regression for small sample, else random forest), run SHAP, and output summary.

    Args:
        features (pd.DataFrame): Data for query and NNs, with 'target' column.
        n_top_features (int): Number of top features to output.
        output_dir (str): Output directory.
        query_id (str): Query compound ID.
        logger (logging.Logger): Logger for messages.
        small_sample_threshold (int): Use logistic regression if n_samples < this.

    Outputs:
        - Top N SHAP features per query (most different, TSV).
        - N features with lowest mean absolute SHAP (most similar, TSV).
        - SHAP summary plot per query (PDF).
    """
    non_feature_cols = ['cpd_id', 'target', 'Dataset', 'Library', 'Plate_Metadata', 'Well_Metadata']
    X = features[[c for c in features.columns if c not in non_feature_cols and features[c].dtype in [np.float32, np.float64, np.int64, np.int32]]]
    y = features['target']

    logger.info(f"Shape of X (features): {X.shape}")
    logger.info(f"Value counts for y (target):\n{y.value_counts()}")

    if X.shape[1] == 0:
        logger.error("No feature columns detected for SHAP analysis.")
        return

    if y.value_counts().min() < 2:
        logger.warning(f"Very few samples in one class (target):\n{y.value_counts()}")
        logger.warning("SHAP results may be unstable. At least a few query and NN wells are recommended.")

    logger.info(f"{(X.var(axis=0) > 0).sum()} features have variance > 0 in this batch")
    logger.info("Variance of features (top 20):\n" + str(X.var(axis=0).sort_values(ascending=False).head(20)))

    try:
        if X.shape[0] < small_sample_threshold:
            logger.info(f"Sample size ({X.shape[0]}) < {small_sample_threshold}. Using logistic regression.")
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X, y)
            logger.info("LogisticRegression model fit successfully.")
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X).values
        else:
            logger.info(f"Sample size ({X.shape[0]}) >= {small_sample_threshold}. Using random forest.")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            logger.info("RandomForestClassifier model fit successfully.")
            explainer = shap.TreeExplainer(model)
            shap_values_rf = explainer.shap_values(X)
            if isinstance(shap_values_rf, list):
                shap_values = shap_values_rf[1]
            else:
                shap_values = shap_values_rf
        logger.info("SHAP values computed successfully.")
    except Exception as e:
        logger.error(f"SHAP model/explainer failed: {e}")
        logger.error(traceback.format_exc())
        return

    try:
        # Always flatten and check shape before any feature ranking or plotting
        # A wole load of PAINFUL bug fixing. 
        # --- handling for 2D SHAP values ---
        shap_arr = np.asarray(shap_values)
        shap_arr = np.squeeze(shap_arr)
        # If shap_arr is a list (shouldn't be here, but just in case)
        if isinstance(shap_arr, list):
            if len(shap_arr) == 2:  # Binary
                shap_arr = np.asarray(shap_arr[1])  # Use class 1
            else:
                shap_arr = np.asarray(shap_arr[0])  # Fallback
        # If we have 3D (n_samples, n_features, 2), select class 1
        if shap_arr.ndim == 3 and shap_arr.shape[2] == 2:
            logger.warning(f"SHAP array shape {shap_arr.shape} is 3D; selecting class 1 along last axis.")
            shap_arr = shap_arr[:, :, 1]
        # Now should be (n_samples, n_features)
        if shap_arr.ndim != 2 or shap_arr.shape[1] != X.shape[1]:
            logger.error(
                f"SHAP array shape after robust handling: {shap_arr.shape}, feature matrix: {X.shape}. "
                "Skipping feature ranking and plot."
            )
            return



        feature_importance = np.abs(shap_arr).mean(axis=0)
        top_idx = np.argsort(feature_importance)[::-1][:n_top_features]
        top_features = X.columns[top_idx]
        top_importance = feature_importance[top_idx]

        lowest_idx = np.argsort(feature_importance)[:n_top_features]
        lowest_features = X.columns[lowest_idx]
        lowest_importance = feature_importance[lowest_idx]

        X_top = X.iloc[:, top_idx]
        X_lowest = X.iloc[:, lowest_idx]
        shap_top = shap_arr[:, top_idx]
        shap_lowest = shap_arr[:, lowest_idx]
        top_features = X.columns[top_idx]
        lowest_features = X.columns[lowest_idx]

        # Plot for top SHAP features (driving difference)
        output_prefix_top = os.path.join(output_dir, f"{query_id}_shap_summary_top")
        plot_shap_summary_all(X_top, shap_top, top_features, output_prefix_top, logger, n_top_features=n_top_features)

        # Plot for lowest SHAP features (most similar)
        output_prefix_lowest = os.path.join(output_dir, f"{query_id}_shap_summary_similar")
        plot_shap_summary_all(X_lowest, shap_lowest, lowest_features, output_prefix_lowest, logger, n_top_features=n_top_features)

        logger.info(f"Number of features with nonzero mean_abs_shap: {(feature_importance > 0).sum()}")
        logger.info(f"Top {n_top_features} features (by mean_abs_shap): {list(top_features)}")

        out_tsv = os.path.join(output_dir, f"{query_id}_top_shap_features_driving_difference.tsv")
        pd.DataFrame({'feature': top_features, 'mean_abs_shap': top_importance}).to_csv(out_tsv, sep="\t", index=False)
        logger.info(f"Wrote top SHAP features TSV: {out_tsv}")

        out_tsv_sim = os.path.join(output_dir, f"{query_id}_most_similar_features.tsv")
        pd.DataFrame({'feature': lowest_features, 'mean_abs_shap': lowest_importance}).to_csv(out_tsv_sim, sep="\t", index=False)
        logger.info(f"Wrote most similar features TSV: {out_tsv_sim}")

        # out_pdf = os.path.join(output_dir, f"{query_id}_shap_summary.pdf")
        # plot_shap_summary(X, shap_arr, X.columns, out_pdf, logger, n_top_features=n_top_features)
        output_prefix = os.path.join(output_dir, f"{query_id}_shap_summary")
        plot_shap_summary_all(X, shap_arr, X.columns, output_prefix, logger, n_top_features=n_top_features)

        # Plot for “most different” (top features)
        plot_feature_importance_bar(
            top_features,
            top_importance,
            os.path.join(output_dir, f"{query_id}_top_shap_features_bar.pdf"),
            title=f"{query_id}: Features Driving Most Difference",
            logger=logger,
        )

        # Plot for “most similar” (lowest features)
        plot_feature_importance_bar(
            lowest_features,
            lowest_importance,
            os.path.join(output_dir, f"{query_id}_most_similar_features_bar.pdf"),
            title=f"{query_id}: Features Driving Most Similarity",
            logger=logger,)
        
        # After plot_shap_summary_all calls
        try:
            # Find the median query well (closest to the median feature vector)
            query_X = X[features['target'] == 1]
            if query_X.shape[0] > 1:
                median_values = query_X.median(axis=0)
                distances = np.linalg.norm(query_X.values - median_values.values, axis=1)
                median_query_pos = distances.argmin()
                median_query_idx = query_X.index[median_query_pos]
            else:
                median_query_idx = query_X.index[0]

            waterfall_file = os.path.join(output_dir, f"{query_id}_shap_waterfall_query_median.pdf")
            plot_shap_waterfall(
                X, shap_arr, X.columns, waterfall_file, logger,
                sample_ind=median_query_idx, max_display=n_top_features
            )
        except Exception as e:
            logger.warning(f"Could not plot median waterfall plot for query {query_id}: {e}")

        heatmap_file = os.path.join(output_dir, f"{query_id}_shap_heatmap.pdf")
        try:
            plot_shap_heatmap(
                X,
                shap_arr,
                X.columns,
                heatmap_file,
                logger,
                max_display=n_top_features
            )
        except Exception as e:
            logger.warning(f"Could not plot SHAP heatmap for query {query_id}: {e}")


    except Exception as e:
        logger.error(f"Feature extraction, TSV writing, or plotting failed: {e}")
        logger.error(traceback.format_exc())


 

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
        logger (logging.Logger): Logger for messages.

    Returns:
        pd.DataFrame: Feature data for query and NNs, with 'target' column.
    """
    features = load_feature_files(features_file, logger)
    logger.info(f"Input features file shape: {features.shape}")

    nn = pd.read_csv(nn_file, sep="\t")
    logger.info(f"NN table shape: {nn.shape}")
    logger.info(f"NN table columns: {nn.columns.tolist()}")

    query_col = None
    for candidate in ['query_id', 'cpd_id']:
        if candidate in nn.columns:
            query_col = candidate
            break
    if not query_col:
        logger.error(f"No suitable query column found in NN table (expected 'query_id' or 'cpd_id'). Columns are: {nn.columns.tolist()}")
        raise ValueError("Nearest neighbour file must have a 'query_id' or 'cpd_id' column.")

    neighbours = nn.loc[nn[query_col].astype(str) == str(query_id), 'neighbour_id'].astype(str).unique()
    logger.info(f"Found {len(neighbours)} neighbours for query {query_id} using column '{query_col}': {neighbours}")

    if len(neighbours) == 0:
        logger.error(f"No neighbours found for query {query_id} in {nn_file} (using column '{query_col}')")
        raise ValueError(f"No neighbours found for query {query_id} in {nn_file} (using column '{query_col}')")

    neighbours = neighbours[:n_neighbors]
    ids_of_interest = [query_id] + list(neighbours)

    features = features[features['cpd_id'].astype(str).isin(ids_of_interest)].copy()
    features['target'] = (features['cpd_id'].astype(str) == query_id).astype(int)
    logger.info(f"Subset feature dataframe shape: {features.shape}")
    logger.info(f"Query wells: {features[features['target'] == 1].shape[0]}; NN wells: {features[features['target'] == 0].shape[0]}")
    return features

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
    logger.info("SHAP (SHapley Additive exPlanations) quantifies each feature's contribution to the model's classification, with higher mean absolute SHAP values indicating features most responsible for distinguishing the query compound from its nearest neighbours.")

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
