#!/usr/bin/env python
# coding: utf-8

"""
Sperm Cell Painting Data Analysis with CLIPn Clustering and UMAP
-----------------------------------------------------------------
This script:
    - Merges Mitotox_assay and STB datasets.
    - Filters only shared numerical features.
    - Handles missing values carefully.
    - Runs CLIPn clustering.
    - Performs dimensionality reduction via UMAP.
    - Saves latent representations for later interrogation.

Includes extensive logging, detailed documentation, and examples.

Command-line arguments:
    --latent_dim: Dimensionality of latent space for clustering.
    --lr: Learning rate for CLIPn.
    --epoch: Number of training epochs.
    --stb: List of STB data files.
    --experiment: List of Mitotox_assay (experiment) data files.

CLIPn Output:
    - Z: dictionary of latent representations from all datasets, indexed numerically.
    - `.csv` files mapping numerical dataset indices and labels back to original names.
    - A `.csv` file mapping numerical labels back to their original names.
    - Latent representation files saved with `cpd_id` as row index.

    
The automated hyperparameter optimisation process in this script uses Bayesian 
Optimisation to fine-tune the CLIPn model’s key parameters: latent dimension, 
learning rate, and number of epochs. If no pre-optimized parameters are provided, 
the script runs multiple trials (n_trials=20 by default) to find the best 
combination of these hyperparameters based on the model’s training loss. 
Once the optimisation is complete, the best parameters are saved to a 
JSON file inside the output directory, which is dynamically named based on 
the optimized values (e.g., clipn_ldim10_lr0.0001_epoch500/best_hyperparameters.json).
 If the user wishes to skip retraining, they can pass the --use_optimized_params 
 argument with the path to this JSON file, allowing the script to 
 load the best parameters, initialize the model, and proceed directly 
 to generating latent representations. This approach significantly 
 speeds up analysis by avoiding redundant training while ensuring that 
 the most effective hyperparameters are consistently used

"""

import os
import sys
import json
import time
import argparse
import re
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import umap.umap_ as umap
from pathlib import Path
from clipn import CLIPn
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
# so we keep the index .. fingers crossed!
from sklearn import set_config
set_config(transform_output="pandas")

from scipy.spatial.distance import cdist
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import optuna

#  Step 1: Command-Line Arguments
parser = argparse.ArgumentParser(description="Perform CLIPn clustering and UMAP on SCP data.")

parser.add_argument("--latent_dim", 
                    type=int, 
                    default=20, 
                    help="Dimensionality of latent space (default: 20)")
parser.add_argument("--lr", type=float, 
                    default=1e-5, 
                    help="Learning rate for CLIPn (default: 1e-5)")
parser.add_argument("--epoch", type=int, 
                    default=200, 
                    help="Number of training epochs (default: 200)")
parser.add_argument("--impute", 
                    type=lambda x: (str(x).lower() == 'true'), 
                    default=False, 
                    help="Perform imputation for missing values (default: True)")
parser.add_argument("--impute-method", 
                    type=str, 
                    choices=["median", "knn"], 
                    default="knn", 
                    help="Imputation method: 'median' (default) or 'knn'.")
parser.add_argument("--knn-neighbors", 
                    type=int, 
                    default=5, 
                    help="Number of neighbors for KNN imputation (default: 5).")
parser.add_argument("--experiment_name", 
                    type=str, 
                    default="test", 
                    help="Name of the experiment (default: inferred from input filenames)")
# Default STB files
default_stb_files = [
    "data/STB_NPSCDD0003971_05092024_normalised.csv",
    "data/STB_NPSCDD0003972_05092024_normalised.csv",
    "data/STB_NPSCDD000400_05092024_normalised.csv",
    "data/STB_NPSCDD000401_05092024_normalised.csv",
    "data/STB_NPSCDD0004034_13022025_normalised.csv"
]
# Default experiment files
default_experiment_files = [
    "data/Mitotox_assay_NPSCDD0003999_25102024_normalised.csv",
    "data/Mitotox_assay_NPSCDD0004023_25102024_normalised.csv"
]
#parser.add_argument("--stb", nargs="+", default=default_stb_files,
                    #help="List of STB dataset files (default: predefined STB files)")
#parser.add_argument("--experiment", nargs="+", default=default_experiment_files,
                   # help="List of Experiment dataset files (default: predefined experiment files)")

# Modify argument parsing to allow only STB or only Experiment runs
parser.add_argument("--stb", nargs="*", default=None,  # Allow passing multiple files or none
                    help="List of STB dataset files. If omitted, default STB files are used.")

parser.add_argument("--experiment", nargs="*", default=None,  # Allow passing multiple files or none
                    help="List of Experiment dataset files. If omitted, default experiment files are used.")


parser.add_argument("--use_optimized_params",
                    type=str,
                    default=None,
                    help="Path to JSON file containing optimized hyperparameters. If provided, training is skipped.")
args = parser.parse_args()


##################################################################
# functions
def objective(trial):
    """
    Bayesian optimisation objective function for CLIPn.

    This function tunes `latent_dim`, `lr`, and `epochs` dynamically to minimize
    the final training loss.

    Parameters
    ----------
    trial : optuna.trial.Trial
        An Optuna trial object for hyperparameter tuning.

    Returns
    -------
    float
        The final loss after training CLIPn with the suggested hyperparameters.
    """
    latent_dim = trial.suggest_int("latent_dim", 10, 40, step=10)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    epochs = trial.suggest_int("epochs", 100, 500, step=50)

    logger.info(f"Trying CLIPn with latent_dim={latent_dim}, lr={lr:.6f}, epochs={epochs}")

    # Train CLIPn model
    clipn_model = CLIPn(X, y, latent_dim=latent_dim)
    loss = clipn_model.fit(X, y, lr=lr, epochs=epochs)

    final_loss = loss[-1]  # Get final loss
    logger.info(f"Final loss for this run: {final_loss:.6f}")

    return final_loss


def optimize_clipn(n_trials=20):
    """
    Runs Bayesian optimisation for CLIPn hyperparameter tuning.

    This function optimizes `latent_dim`, `lr`, and `epochs` using Optuna
    to find the best combination that minimizes the final loss.

    Parameters
    ----------
    n_trials : int, optional
        The number of trials for optimisation (default is 20).

    Returns
    -------
    dict
        The best hyperparameters found (latent_dim, lr, epochs).
    """
    logger.info(f"Starting Bayesian optimisation with {n_trials} trials.")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_loss = study.best_value

    logger.info(f"Best hyperparameters found: {best_params} with final loss {best_loss:.6f}")
    return best_params

def group_and_filter_data(df):
    """
    Groups data by `cpd_id` and `Library`, then drops unwanted columns.

    Parameters
    ----------
    df : pd.DataFrame
        The imputed dataset to process.

    Returns
    -------
    pd.DataFrame
        The grouped and cleaned DataFrame.
    """
    if df is None or df.empty:
        return df  # Return as-is if empty or None

    # Ensure 'cpd_id' and 'Library' are in the MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex DataFrame with ['cpd_id', 'Library', 'cpd_type'].")

    # Group by `cpd_id` and `Library` (taking the mean for numeric values)
    df = df.groupby(["cpd_id", "Library"], as_index=True).mean()

    # Drop columns matching `filter_pattern`
    columns_to_drop = [col for col in df.columns if filter_pattern.search(col)]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    return df


def decode_clipn_predictions(predicted_labels, predicted_cpd_ids,
                             cpd_type_encoder, cpd_id_encoder):
    """
    Decode numeric predictions back to original 'cpd_type' and 'cpd_id' labels.

    Parameters
    ----------
    predicted_labels : np.ndarray or list
        Predicted numeric `cpd_type` values (from CLIPn or similar models).
    predicted_cpd_ids : np.ndarray or list
        Predicted numeric `cpd_id` values (optional, if used for clustering).

    cpd_type_encoder : LabelEncoder
        Fitted LabelEncoder used to encode 'cpd_type'.

    cpd_id_encoder : LabelEncoder
        Fitted LabelEncoder used to encode 'cpd_id'.

    Returns
    -------
    tuple
        - original_labels : np.ndarray
            Decoded `cpd_type` labels.
        - original_cpd_ids : np.ndarray
            Decoded `cpd_id` labels.
    """
    original_labels = cpd_type_encoder.inverse_transform(predicted_labels)
    original_cpd_ids = cpd_id_encoder.inverse_transform(predicted_cpd_ids)
    return original_labels, original_cpd_ids


def ensure_multiindex(df, required_levels=("cpd_id", "Library", "cpd_type"), logger=None, dataset_name="dataset"):
    """
    Ensures the given DataFrame has a MultiIndex on required levels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be checked or modified.

    required_levels : tuple of str, optional
        Columns to use as MultiIndex (default: ('cpd_id', 'Library', 'cpd_type')).

    logger : logging.Logger or None, optional
        Logger instance for info/warning messages. If None, prints will be used.

    dataset_name : str, optional
        Name of the dataset (used for logging).

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex applied if needed.
    """
    if df is None or df.empty:
        if logger:
            logger.warning(f"{dataset_name}: DataFrame is empty or None, skipping MultiIndex restoration.")
        return df

    if isinstance(df.index, pd.MultiIndex) and all(level in df.index.names for level in required_levels):
        if logger:
            logger.info(f"{dataset_name}: MultiIndex already present.")
        return df  # Already has correct MultiIndex

    # Check that all required levels exist as columns
    missing_cols = set(required_levels) - set(df.columns)
    if missing_cols:
        msg = f"{dataset_name}: Cannot restore MultiIndex, missing columns: {missing_cols}"
        if logger:
            logger.error(msg)
        else:
            print("ERROR:", msg)
        return df  # Return unchanged to prevent crash

    # Set MultiIndex
    df = df.set_index(list(required_levels))
    if logger:
        logger.info(f"{dataset_name}: MultiIndex restored using columns {required_levels}")
    return df



def compute_pairwise_distances(latent_df):
    """
    Compute the pairwise Euclidean distance between compounds in latent space.

    Parameters:
    -----------
    latent_df : pd.DataFrame
        DataFrame containing latent representations indexed by `cpd_id`.

    Returns:
    --------
    pd.DataFrame
        Distance matrix with `cpd_id` as row and column labels.
    """
    dist_matrix = cdist(latent_df.values, latent_df.values, metric="euclidean")
    dist_df = pd.DataFrame(dist_matrix, index=latent_df.index, columns=latent_df.index)
    return dist_df


def generate_similarity_summary(dist_df):
    """
    Generate a summary of the closest and farthest compounds.

    Parameters:
    -----------
    dist_df : pd.DataFrame
        Pairwise distance matrix with `cpd_id` as row and column labels.

    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with closest and farthest compounds.
    """
    closest_compounds = dist_df.replace(0, np.nan).idxmin(axis=1)  # Ignore self-comparison
    farthest_compounds = dist_df.idxmax(axis=1)

    summary_df = pd.DataFrame({
        "Compound": dist_df.index,  # Preserve cpd_id
        "Closest Compound": closest_compounds,
        "Distance to Closest": dist_df.min(axis=1),
        "Farthest Compound": farthest_compounds,
        "Distance to Farthest": dist_df.max(axis=1)
    })

    return summary_df


def plot_umap_coloured_by_experiment(umap_df, output_file, color_map=None):
    """
    Generates a UMAP visualization coloured by experiment vs. STB.

    Parameters
    ----------
    umap_df : pd.DataFrame
        DataFrame containing UMAP coordinates with a MultiIndex (`cpd_id`, `Library`).
    output_file : str
        Path to save the UMAP plot.
    color_map : dict, optional
        A dictionary mapping dataset types (e.g., "Experiment", "STB") to colors.
        Default is {"Experiment": "red", "STB": "blue"}.

    Returns
    -------
    None
    """
    try:
        logger.info("Generating UMAP visualization highlighting Experiment vs. STB data.")

        # Default color map if none provided
        if color_map is None:
            color_map = {"Experiment": "red", "STB": "blue"}

        # Ensure 'Library' exists in MultiIndex
        if "Library" not in umap_df.index.names:
            logger.warning("Warning: 'Library' not found in MultiIndex! Attempting to use column instead.")
            if "Library" in umap_df.columns:
                umap_df = umap_df.set_index("Library")
            else:
                logger.error("Error: 'Library' column not found! Skipping UMAP experiment coloring.")
                return

        # Map colors based on 'Library'
        dataset_labels = umap_df.index.get_level_values("Library")  # Extract library info
        dataset_colors = [color_map.get(label, "gray") for label in dataset_labels]  # Assign colors

        # Create scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], s=5, alpha=0.7, c=dataset_colors)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title("UMAP Visualization: Experiment (Red) vs. STB (Blue)")

        # Save plot
        plt.savefig(output_file, dpi=300)
        plt.close()
        logger.info(f"UMAP visualization (experiment vs. STB) saved as '{output_file}'.")

    except Exception as e:
        logger.error(f"Error generating UMAP experiment visualization: {e}. Continuing script execution.")


def restore_encoded_labels(encoded_series, encoder):
    """
    Restores original categorical labels from an encoded Series using the fitted LabelEncoder.

    Parameters
    ----------
    encoded_series : pd.Series or np.ndarray
        Encoded numeric labels to be restored.

    encoder : sklearn.preprocessing.LabelEncoder
        The fitted LabelEncoder used during encoding.

    Returns
    -------
    np.ndarray
        Array of original categorical labels.
    """
    if encoder is None:
        raise ValueError("Encoder must be provided and fitted.")
    return encoder.inverse_transform(encoded_series)



def plot_distance_heatmap(dist_df, output_path):
    """
    Generate and save a heatmap of pairwise compound distances.

    Parameters:
    -----------
    dist_df : pd.DataFrame
        Pairwise distance matrix with `cpd_id` as row and column labels.
    output_path : str
        Path to save the heatmap PDF file.
    """
    plt.figure(figsize=(12, 10))
    htmap = sns.clustermap(dist_df, cmap="viridis", method="ward",
                           figsize=(12, 10),
                           xticklabels=True,
                           yticklabels=True)

    # Rotate labels for better readability
    plt.setp(htmap.ax_heatmap.get_xticklabels(), rotation=90, fontsize=4)
    plt.setp(htmap.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)

    plt.title("Pairwise Distance Heatmap of Compounds")
    plt.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close()

def plot_dendrogram(dist_df, output_path):
    """
    Generate and save a dendrogram based on hierarchical clustering.

    Parameters:
    -----------
    dist_df : pd.DataFrame
        Pairwise distance matrix with `cpd_id` as row and column labels.
    output_path : str
        Path to save the dendrogram PDF file.
    """
    linkage_matrix = linkage(squareform(dist_df), method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=list(dist_df.index), leaf_rotation=90, leaf_font_size=8)
    plt.title("Hierarchical Clustering of Compounds")
    plt.xlabel("Compound")
    plt.ylabel("Distance")

    plt.savefig(output_path)
    plt.close()


def reconstruct_combined_latent_df(Z, dataset_index_map, index_lookup):
    """
    Reconstructs a combined DataFrame of latent representations using original MultiIndex.

    Parameters
    ----------
    Z : dict
        Dictionary containing CLIPn latent representations, with integer keys (dataset labels).
    
    dataset_index_map : dict
        Dictionary mapping integer dataset indices (from CLIPn) to dataset names (e.g., {0: "experiment", 1: "stb"}).

    index_lookup : dict
        Dictionary mapping dataset names to their original MultiIndex (e.g., {"experiment": index_df, "stb": index_df}).

    Returns
    -------
    pd.DataFrame
        Combined latent representation DataFrame with correct MultiIndex restored.
    """
    latent_frames = []

    for index_id, dataset_name in dataset_index_map.items():
        if dataset_name not in index_lookup:
            raise ValueError(f"Missing index for dataset '{dataset_name}'")

        latent_array = Z[index_id]
        dataset_index = index_lookup[dataset_name]

        if latent_array.shape[0] != len(dataset_index):
            raise ValueError(
                f"Mismatch: latent array for '{dataset_name}' has shape {latent_array.shape[0]}, "
                f"but index has length {len(dataset_index)}"
            )

        df = pd.DataFrame(latent_array, index=dataset_index)
        df["dataset"] = dataset_name  # Optional: helpful for visualisation
        latent_frames.append(df)

    combined_latent_df = pd.concat(latent_frames)
    return combined_latent_df



def impute_missing_values(experiment_data, stb_data, impute_method="median", knn_neighbors=5):
    """
    Perform missing value imputation while preserving MultiIndex (cpd_id, Library, cpd_type).

    Parameters
    ----------
    experiment_data : pd.DataFrame or None
        Experiment dataset with MultiIndex (`cpd_id`, `Library`, `cpd_type`).

    stb_data : pd.DataFrame or None
        STB dataset with MultiIndex (`cpd_id`, `Library`, `cpd_type`).

    impute_method : str, optional
        Imputation method: "median" (default) or "knn".

    knn_neighbors : int, optional
        Number of neighbours for KNN imputation (default: 5).

    Returns
    -------
    tuple
        - experiment_data_imputed : pd.DataFrame or None
        - stb_data_imputed : pd.DataFrame or None
        - stb_labels : np.array
        - stb_cpd_id_map : dict
    """
    logger.info(f"Performing imputation using {impute_method} strategy.")

    # Choose and configure imputer
    if impute_method == "median":
        imputer = SimpleImputer(strategy="median")
    elif impute_method == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
    else:
        raise ValueError("Invalid imputation method. Choose 'median' or 'knn'.")

    # Enable pandas output to preserve index and column names
    imputer.set_output(transform="pandas")

    # Helper to apply imputation to a DataFrame
    def impute_dataframe(df):
        if df is None or df.empty:
            return df
        numeric_df = df.select_dtypes(include=[np.number])
        return imputer.fit_transform(numeric_df)

    # Apply imputation
    experiment_data_imputed = impute_dataframe(experiment_data)
    stb_data_imputed = impute_dataframe(stb_data)

    logger.info(f"Imputation complete. Experiment shape: {experiment_data_imputed.shape if experiment_data_imputed is not None else 'None'}, "
                f"STB shape: {stb_data_imputed.shape if stb_data_imputed is not None else 'None'}")

    # Encode STB labels if available
    if stb_data is not None and "cpd_type" in stb_data.index.names:
        try:
            label_encoder = LabelEncoder()
            stb_labels = label_encoder.fit_transform(stb_data.index.get_level_values("cpd_type"))
            stb_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            stb_cpd_id_map = dict(zip(stb_data.index.get_level_values("cpd_id"), stb_labels))
        except Exception as e:
            logger.warning(f"Failed to encode STB labels: {e}")
            stb_labels = np.zeros(stb_data_imputed.shape[0]) if stb_data_imputed is not None else np.array([])
            stb_cpd_id_map = {}
    else:
        stb_labels = np.zeros(stb_data_imputed.shape[0]) if stb_data_imputed is not None else np.array([])
        stb_cpd_id_map = {}
        logger.warning("Warning: No STB labels available!")
    # Restore original non-numeric columns (e.g., cpd_id, Library, cpd_type)
    if experiment_data is not None:
        for col in ["cpd_id", "Library", "cpd_type"]:
            if col in experiment_data.index.names:
                experiment_data_imputed[col] = experiment_data.index.get_level_values(col)
            elif col in experiment_data.columns:
                experiment_data_imputed[col] = experiment_data[col]

    if stb_data is not None:
        for col in ["cpd_id", "Library", "cpd_type"]:
            if col in stb_data.index.names:
                stb_data_imputed[col] = stb_data.index.get_level_values(col)
            elif col in stb_data.columns:
                stb_data_imputed[col] = stb_data[col]
    # Log presence of restored columns
    logger.debug(f"experiment_data_imputed columns: {experiment_data_imputed.columns.tolist()}")
    logger.debug(f"stb_data_imputed columns: {stb_data_imputed.columns.tolist()}")
    return experiment_data_imputed, stb_data_imputed, stb_labels, stb_cpd_id_map


def process_common_columns(df1, df2, step="before"):
    """
    Identify and apply common numerical columns between two datasets.

    Parameters:
    -----------
    df1 : pd.DataFrame or None
        First dataset (e.g., experiment data).
    df2 : pd.DataFrame or None
        Second dataset (e.g., STB data).
    step : str, optional
        Whether this is executed "before" or "after" imputation (default: "before").

    Returns:
    --------
    df1_filtered : pd.DataFrame or None
        First dataset filtered to keep only common columns.
    df2_filtered : pd.DataFrame or None
        Second dataset filtered to keep only common columns.
    common_columns : Index
        The set of common columns between the two datasets.
    """
    if df1 is not None and df2 is not None:
        common_columns = df1.columns.intersection(df2.columns)
    elif df1 is not None:
        common_columns = df1.columns
    elif df2 is not None:
        common_columns = df2.columns
    else:
        raise ValueError(f"Error: No valid numerical data available at step '{step}'!")

    logger.info(f"Common numerical columns {step} imputation: {list(common_columns)}")

    # Filter datasets to keep only common columns
    df1_filtered = df1[common_columns] if df1 is not None else None
    df2_filtered = df2[common_columns] if df2 is not None else None

    return df1_filtered, df2_filtered, common_columns


def encode_cpd_data(dataframes, encode_labels=False):
    """
    Applies MultiIndex and optionally encodes 'cpd_id' and 'cpd_type' for ML.

    Parameters
    ----------
    dataframes : dict
        Dictionary of dataset name to DataFrame.
    encode_labels : bool, optional
        If True, returns encoded labels and mappings. Default is False.

    Returns
    -------
    dict
        Dictionary with structure:
        {
            "dataset_name": {
                "data": DataFrame,
                "cpd_type_encoded": np.ndarray,
                "cpd_type_mapping": dict,
                "cpd_id_mapping": dict
            }
        }
        If encode_labels is False, only returns {"dataset_name": DataFrame}.
    """
    from sklearn.preprocessing import LabelEncoder

    results = {}

    for name, df in dataframes.items():
        if df is None or df.empty:
            continue

        # Ensure proper index
        if {"cpd_id", "Library", "cpd_type"}.issubset(df.columns):
            df = df.set_index(["cpd_id", "Library", "cpd_type"])
        elif not isinstance(df.index, pd.MultiIndex):
            raise ValueError(f"{name} is missing MultiIndex or required columns.")

        output = {"data": df}

        if encode_labels:
            # Encode cpd_type
            le_type = LabelEncoder()
            cpd_type_encoded = le_type.fit_transform(df.index.get_level_values("cpd_type"))
            cpd_type_mapping = dict(zip(le_type.classes_, le_type.transform(le_type.classes_)))

            # Encode cpd_id
            le_id = LabelEncoder()
            cpd_id_encoded = le_id.fit_transform(df.index.get_level_values("cpd_id"))
            cpd_id_mapping = dict(zip(le_id.classes_, le_id.transform(le_id.classes_)))
            # Add to DataFrame (with restored MultiIndex)
            df = df.copy()
            df["cpd_type_encoded"] = cpd_type_encoded
            df["cpd_id_encoded"] = cpd_id_encoded
            output.update({
                "data": df,
                "cpd_type_encoded": cpd_type_encoded,
                "cpd_type_mapping": cpd_type_mapping,
                "cpd_type_encoder": le_type,
                "cpd_id_mapping": cpd_id_mapping,
                "cpd_id_encoder": le_id
            })
        results[name] = output
    return results



def prepare_data_for_clipn(experiment_data_imputed, experiment_labels, experiment_label_mapping,
                           stb_data_imputed, stb_labels, stb_label_mapping):
    """
    Prepare data for CLIPn clustering by encoding datasets, removing non-numeric columns,
    and structuring inputs for training.

    Parameters
    ----------
    experiment_data_imputed : pd.DataFrame or None
        The imputed experimental dataset.
    experiment_labels : np.array
        Encoded labels for experiment compounds.
    experiment_label_mapping : dict
        Mapping of encoded labels to original experiment cpd_type.
    stb_data_imputed : pd.DataFrame or None
        The imputed STB dataset.
    stb_labels : np.array
        Encoded labels for STB compounds.
    stb_label_mapping : dict
        Mapping of encoded labels to original STB cpd_type.

    Returns
    -------
    tuple
        X (dict): Dictionary containing dataset arrays for CLIPn.
        y (dict): Dictionary of corresponding labels.
        label_mappings (dict): Mapping of dataset indices to original labels.
    """
    X, y, label_mappings = {}, {}, {}

    # Ensure at least one dataset exists
    dataset_names = []
    if experiment_data_imputed is not None and not experiment_data_imputed.empty:
        dataset_names.append("experiment_assay_combined")
    if stb_data_imputed is not None and not stb_data_imputed.empty:
        dataset_names.append("STB_combined")

    if not dataset_names:
        logger.error("No valid datasets available for CLIPn analysis.")
        raise ValueError("Error: No valid datasets available for CLIPn analysis.")

    # Encode dataset names
    dataset_encoder = LabelEncoder()
    dataset_indices = dataset_encoder.fit_transform(dataset_names)
    dataset_mapping = dict(zip(dataset_indices, dataset_names))

    logger.info(f"Dataset Mapping: {dataset_mapping}")

    # Define non-numeric columns to drop before passing to CLIPn
    non_numeric_cols = ["cpd_id", "Library", "cpd_type"]

    # Process experiment data
    if experiment_data_imputed is not None and not experiment_data_imputed.empty:
        experiment_data_imputed = experiment_data_imputed.drop(columns=[col for col in non_numeric_cols if col in experiment_data_imputed], errors="ignore")
        
        exp_index = dataset_encoder.transform(["experiment_assay_combined"])[0]
        X[exp_index] = experiment_data_imputed.values
        y[exp_index] = experiment_labels
        label_mappings[exp_index] = experiment_label_mapping

        logger.info(f"  Added Experiment Data to X with shape: {experiment_data_imputed.shape}")
    else:
        logger.warning(" No valid experiment data for CLIPn.")

    # Process STB data
    if stb_data_imputed is not None and not stb_data_imputed.empty:
        stb_data_imputed = stb_data_imputed.drop(columns=[col for col in non_numeric_cols if col in stb_data_imputed], errors="ignore")

        stb_index = dataset_encoder.transform(["STB_combined"])[0]
        X[stb_index] = stb_data_imputed.values
        y[stb_index] = stb_labels
        label_mappings[stb_index] = stb_label_mapping

        logger.info(f"  Added STB Data to X with shape: {stb_data_imputed.shape}")
    else:
        logger.warning(" No valid STB data for CLIPn.")

    # Debugging: Log dataset keys before passing to CLIPn
    logger.info(f" X dataset keys before CLIPn: {list(X.keys())}")
    logger.info(f" y dataset keys before CLIPn: {list(y.keys())}")

    # Ensure at least one dataset is available
    if not X:
        logger.error(" No valid datasets available for CLIPn analysis. Aborting!")
        raise ValueError("Error: No valid datasets available for CLIPn analysis.")

    logger.info(" Datasets successfully structured for CLIPn.")
    logger.info(f" Final dataset shapes being passed to CLIPn: { {k: v.shape for k, v in X.items()} }")

    return X, y, label_mappings, dataset_mapping


def run_clipn(X, y, output_folder, args):
    """
    Runs CLIPn clustering with optional hyperparameter optimization.

    Parameters
    ----------
    X : dict
        Dictionary containing dataset arrays for CLIPn.
    y : dict
        Dictionary of corresponding labels.
    output_folder : str
        Directory to save CLIPn output files.
    args : argparse.Namespace
        Command-line arguments, including latent_dim, learning rate, and epoch count.

    Returns
    -------
    dict
        Dictionary containing latent representations from CLIPn.
    """
    hyperparam_file = os.path.join(output_folder, "best_hyperparameters.json")

    # Check if optimized hyperparameters should be used
    if args.use_optimized_params:
        try:
            logger.info(f"Loading optimized hyperparameters from {args.use_optimized_params}")
            with open(args.use_optimized_params, "r") as f:
                best_params = json.load(f)
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load optimized parameters: {e}")
            raise ValueError("Invalid or missing hyperparameter JSON file.")

        # Update args with loaded parameters
        args.latent_dim = best_params["latent_dim"]
        args.lr = best_params["lr"]
        args.epoch = best_params["epochs"]

        logger.info(f"Using pre-trained parameters: latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")

        # Initialize model and directly run prediction
        clipn_model = CLIPn(X, y, latent_dim=args.latent_dim)
        logger.info("Skipping training. Generating latent representations using pre-trained parameters.")
        Z = clipn_model.predict(X)

    else:
        # Run Hyperparameter Optimization
        logger.info("Running Hyperparameter Optimization")
        best_params = optimize_clipn(n_trials=20)  # Bayesian Optimization

        # Save optimized parameters
        with open(hyperparam_file, "w") as f:
            json.dump(best_params, f, indent=4)

        logger.info(f"Optimized hyperparameters saved to {hyperparam_file}")

        # Update args with best parameters
        args.latent_dim = best_params["latent_dim"]
        args.lr = best_params["lr"]
        args.epoch = best_params["epochs"]

        logger.info(f"Using optimized parameters: latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")

        # Train the model with the optimized parameters
        logger.info(f"Running CLIPn with optimized latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")
        clipn_model = CLIPn(X, y, latent_dim=args.latent_dim)
        logger.info("Fitting CLIPn model...")
        loss = clipn_model.fit(X, y, lr=args.lr, epochs=args.epoch)
        logger.info(f"CLIPn training completed. Final loss: {loss[-1]:.6f}")

        # Generate latent representations
        logger.info("Generating latent representations.")
        Z = clipn_model.predict(X)

    return Z


def generate_umap(combined_latent_df, output_folder, umap_plot_file, args, 
                  n_neighbors=15, num_clusters=10, add_labels=False
):
    """
    Generates UMAP embeddings, performs KMeans clustering, and saves the results.

    Parameters
    ----------
    combined_latent_df : pd.DataFrame
        Latent representations indexed by (cpd_id, Library, cpd_type).
    
    output_folder : str
        Path to save the UMAP outputs.

    umap_plot_file : str
        Full file path to save the UMAP plot (PDF).

    args : argparse.Namespace
        Parsed command-line arguments containing hyperparameters (latent_dim, lr, epoch).

    n_neighbors : int, optional
        Number of neighbors for UMAP (default: 15).
    
    num_clusters : int, optional
        Number of clusters for KMeans clustering (default: 10).

    add_labels : bool, optional
        Whether to label points with `cpd_id` on the UMAP plot (default: False).

    Returns
    -------
    pd.DataFrame
        DataFrame containing UMAP coordinates and cluster labels, indexed by (cpd_id, Library, cpd_type).
    """

    logger.info(f"Generating UMAP visualization with n_neighbors={n_neighbors} and {num_clusters} clusters.")

    # Perform UMAP dimensionality reduction
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)

    latent_umap = umap_model.fit_transform(combined_latent_df.drop(columns=["dataset"], errors="ignore"))


    # Create DataFrame with MultiIndex
    umap_df = pd.DataFrame(latent_umap, columns=["UMAP1", "UMAP2"], index=combined_latent_df.index)

    # Perform clustering
    logger.info(f"Running KMeans clustering with {num_clusters} clusters.")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_umap)
    
    # Add cluster labels to DataFrame
    umap_df["Cluster"] = cluster_labels

    # Save UMAP results
    umap_file = os.path.join(output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_UMAP.csv")
    umap_df.to_csv(umap_file)
    logger.info(f"UMAP coordinates saved to '{umap_file}'.")

    # Generate and save UMAP plot with cluster colors
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], alpha=0.7, s=5, c=cluster_labels, cmap="tab10")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("CLIPn UMAP Visualization (coloured by Cluster)")
    plt.colorbar(scatter, label="Cluster ID")

    # Add `cpd_id` labels if enabled
    if add_labels:
        logger.info("Adding `cpd_id` labels to UMAP plot.")
        for (cpd_id, _), (x, y) in zip(umap_df.index, latent_umap):
            plt.text(x, y, str(cpd_id), fontsize=6, alpha=0.7)

    # Adjust filename if labels are included
    if add_labels:
        umap_plot_file = umap_plot_file.replace(".pdf", "_labeled.pdf")

    # Save plot
    plt.savefig(umap_plot_file, dpi=1200)
    plt.close()

    logger.info(f"UMAP visualization with clusters saved to '{umap_plot_file}'.")

    return umap_df


# Ensure index backup is not empty and restore MultiIndex properly
def restore_multiindex(imputed_df, index_backup, dataset_name):
    """
    Restores the MultiIndex after imputation by aligning and joining with the backup.

    Parameters
    ----------
    imputed_df : pd.DataFrame
        The imputed dataset where the MultiIndex needs to be restored.
    index_backup : pd.DataFrame
        Backup of the original MultiIndex (must contain 'cpd_id', 'Library', 'cpd_type').
    dataset_name : str
        Name of the dataset for logging (e.g., "experiment", "stb").

    Returns
    -------
    pd.DataFrame
        The dataset with the MultiIndex properly restored.
    """
    if index_backup is not None and imputed_df is not None:
        # Ensure index_backup is a DataFrame with the necessary columns
        if isinstance(index_backup, pd.MultiIndex):
            index_backup = index_backup.to_frame(index=False)

        # Ensure the backup contains the required columns
        required_cols = {"cpd_id", "Library", "cpd_type"}
        missing_cols = required_cols - set(index_backup.columns)
        if missing_cols:
            logger.error(f"Missing columns in index_backup for {dataset_name}: {missing_cols}")
            return imputed_df  # Return unchanged if we can't restore properly

        # Ensure alignment: Trim to the number of available rows
        common_rows = min(len(imputed_df), len(index_backup))
        if common_rows == 0:
            logger.warning(f"⚠️ No overlapping rows found between imputation and original index for {dataset_name}_data_imputed!")
        else:
            # Trim both DataFrames to ensure correct row count
            index_backup = index_backup.iloc[:common_rows].reset_index(drop=True)
            imputed_df = imputed_df.reset_index(drop=True)

            # Join safely, ensuring all three index columns are restored
            imputed_df = index_backup.join(imputed_df)
            imputed_df = imputed_df.set_index(["cpd_id", "Library", "cpd_type"])

            logger.info(f"Successfully restored MultiIndex for {dataset_name}_data_imputed. Final shape: {imputed_df.shape}")

        return imputed_df
    else:
        logger.error(f"Failed to restore MultiIndex for {dataset_name}_data_imputed! Check backup index.")
        return imputed_df  # Return unchanged if restoration fails




#################################################################
#################################################################
#################################################################
if sys.version_info[:1] != (3,):
    # e.g. sys.version_info(major=3, minor=9, micro=7,
    # releaselevel='final', serial=0)
    # break the program
    print ("currently using:", sys.version_info,
           "  version of python")
    raise ImportError("Python 3.x is required")
    print ("did you activate the virtual environment?")
    print ("this is to deal with module imports")
    sys.exit(1)

VERSION = "cell painting: clipn intergration: v0.0.1"
if "--version" in sys.argv:
    print(VERSION)
    sys.exit(1)

##################################################################
#  Step 2: Setup Output Directory
# TO DO: needs to change as we train for best params, so this doesnt work. 

# Define the main output directory for this experiment
# Determine the experiment name from the argument or infer from file names
##################################################################
# Log file name includes the experiment name


# **1. Define Experiment Name First**
if args.experiment_name:
    experiment_name = args.experiment_name
else:
    # Default: extract from first experiment file (assuming structured as 'experiment_assay_...')
    experiment_name = os.path.basename(args.experiment[0]).split("_")[0]  

# Ensure experiment_name is defined from command-line arguments
experiment_name = args.experiment_name if hasattr(args, "experiment_name") else "test"

# **2. Create Main Output Folder Before Logging**
main_output_folder = f"{experiment_name}_clipn_output"
os.makedirs(main_output_folder, exist_ok=True)

##################################################################
# Determine the log file name (including the experiment prefix)
log_filename = os.path.join(main_output_folder, f"{experiment_name}_clipn_intergration.log")

# **Proper Logger Initialization**
logger = logging.getLogger(f"{experiment_name}: {time.asctime()}")
logger.setLevel(logging.DEBUG)  # Capture all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Stream Handler (stderr)
err_handler = logging.StreamHandler(sys.stderr)
err_formatter = logging.Formatter('%(levelname)s: %(message)s')
err_handler.setFormatter(err_formatter)
logger.addHandler(err_handler)

# File Handler (Logfile)
try:
    logstream = open(log_filename, 'w')
    err_handler_file = logging.StreamHandler(logstream)
    err_handler_file.setFormatter(err_formatter)
    err_handler_file.setLevel(logging.INFO)  # Logfile should always be verbose
    logger.addHandler(err_handler_file)
except Exception as e:
    print(f"Could not open {log_filename} for logging: {e}", file=sys.stderr)
    sys.exit(1)

# **4. Now we can use `logger.info()` because `logger` is initialized**
logger.info(f"Using experiment name: {experiment_name}")

# **System & Command-line Information for Reproducibility**
logger.info(f"Python Version: {sys.version_info}")
logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
logger.info(f"Experiment Name: {experiment_name}")
logger.info(f"STB Datasets: {args.stb}")
logger.info(f"Experiment Datasets: {args.experiment}")
logger.info(f"Using Logfile: {log_filename}")
logger.info(f"Logging initialized at {time.asctime()}")

##################################################################
#  Step 3: Setup Per-Run Output Folder for Different Hyperparameters
output_folder = os.path.join(main_output_folder, "parameters_log")
os.makedirs(output_folder, exist_ok=True)

# **5. Reinitialize logging inside per-run folder**
log_filename = os.path.join(output_folder, "clipn_integration.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)


logger.info(f"Starting SCP data analysis using CLIPn with latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")
logger.info(f"STB datasets: {args.stb}")
logger.info(f"Experiment datasets: {args.experiment}")


logger.info(f"Starting SCP data analysis using CLIPn with latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")

# Default STB files (used only if no --stb argument is given)
default_stb_files = [
    "data/STB_NPSCDD0003971_05092024_normalised.csv",
    "data/STB_NPSCDD0003972_05092024_normalised.csv",
    "data/STB_NPSCDD000400_05092024_normalised.csv",
    "data/STB_NPSCDD000401_05092024_normalised.csv",
    "data/STB_NPSCDD0004034_13022025_normalised.csv"
]

# Default Experiment files (used only if no --experiment argument is given)
default_experiment_files = [
    "data/Mitotox_assay_NPSCDD0003999_25102024_normalised.csv",
    "data/Mitotox_assay_NPSCDD0004023_25102024_normalised.csv"
]


# Determine dataset selection based on command-line input
stb_files = args.stb if args.stb else default_stb_files  
experiment_files = args.experiment if args.experiment else []  # No default experiment files

# Ensure that at least one dataset is provided
if not stb_files and not experiment_files:
    parser.error("At least one dataset (STB or experiment) must be provided.")

# Log the dataset choices
if stb_files and experiment_files:
    logger.info("Running with both STB and Experiment datasets.")
elif stb_files:
    logger.info("Running with only STB dataset.")
elif experiment_files:
    logger.info("Running with only Experiment dataset.")

# Load and merge datasets (only if they exist)
stb_dfs = [pd.read_csv(f) for f in stb_files] if stb_files else []
experiment_dfs = [pd.read_csv(f) for f in experiment_files] if experiment_files else []


# Only concatenate if there are datasets provided
stb_data = pd.concat(stb_dfs, axis=0, ignore_index=True) if stb_files else None
experiment_data = pd.concat(experiment_dfs, axis=0, ignore_index=True) if experiment_files else None

# Log the first few rows of the data before processing
if experiment_data is not None:
    logger.info("First few rows of experiment_data:\n" + experiment_data.head().to_string())
if stb_data is not None:
    logger.info("First few rows of stb_data:\n" + stb_data.head().to_string())



# Ensure 'Library' column exists and assign based on dataset origin
if "Library" not in experiment_data.columns:
    logger.warning("'Library' column is missing in experiment_data. Assigning 'Experiment'.")
    experiment_data["Library"] = "Experiment"

if "Library" not in stb_data.columns:
    logger.warning("'Library' column is missing in stb_data. Assigning 'STB'.")
    stb_data["Library"] = "STB"


# Multiindex
# Ensure 'cpd_id', 'Library', and 'cpd_type' exist before setting as index
if {"cpd_id", "Library", "cpd_type"}.issubset(experiment_data.columns):
    experiment_data = experiment_data.set_index(["cpd_id", "Library", "cpd_type"])

if {"cpd_id", "Library", "cpd_type"}.issubset(stb_data.columns):
    stb_data = stb_data.set_index(["cpd_id", "Library", "cpd_type"])



# Check if MultiIndex contains these columns
if isinstance(experiment_data.index, pd.MultiIndex):
    index_levels = experiment_data.index.names  # Get MultiIndex level names
else:
    index_levels = []

if isinstance(stb_data.index, pd.MultiIndex):
    stb_index_levels = stb_data.index.names  # Get MultiIndex level names
else:
    stb_index_levels = []

logger.info(f"Experiment MultiIndex levels: {index_levels}")
logger.info(f"STB MultiIndex levels: {stb_index_levels}")



# Log the first few rows after extracting numerical features
if experiment_data is not None:
    logger.info("First few rows of experiment_data:\n" + experiment_data.head().to_string())
if stb_data is not None:
    logger.info("First few rows of stb_data:\n" + stb_data.head().to_string())


# Drop columns that are entirely NaN in either dataset BEFORE imputation

if "cpd_id" not in experiment_data.columns:
    experiment_data = experiment_data.reset_index()

if "cpd_id" not in stb_data.columns:
    stb_data = stb_data.reset_index()


# Debug: Print actual column names before calling dropna
logger.info(f"Columns in experiment_data before dropping NaNs: {list(experiment_data.columns)}")
logger.info(f"Columns in STB_data before dropping NaNs: {list(stb_data.columns)}")

# Check if protected columns exist before applying dropna
missing_experiment_cols = [col for col in ["cpd_id", "cpd_type", "Library"] if col not in experiment_data.columns]
missing_stb_cols = [col for col in ["cpd_id", "cpd_type", "Library"] if col not in stb_data.columns]

if missing_experiment_cols:
    logger.warning(f"Missing columns in experiment_data: {missing_experiment_cols}")

if missing_stb_cols:
    logger.warning(f"Missing columns in stb_data: {missing_stb_cols}")


logger.info("Current MultiIndex levels:", experiment_data.index.names)
logger.info("Current MultiIndex levels (STB):", stb_data.index.names)

# Only apply dropna if all required columns exist
if not missing_experiment_cols:
    experiment_data = experiment_data.dropna(subset=["cpd_id", "cpd_type", "Library"], how="any")

if not missing_stb_cols:
    stb_data = stb_data.dropna(subset=["cpd_id", "cpd_type", "Library"], how="any")


protected_columns = ["cpd_id", "cpd_type", "Library"]


logger.info("Columns before dropna:", experiment_data.columns)

if experiment_data is not None:
    experiment_data = experiment_data.dropna(axis=1, how="all").dropna(axis=0, how="all")
    experiment_data = experiment_data.dropna(subset=protected_columns, how="any")  # Ensure IDs are not lost

if stb_data is not None:
    stb_data = stb_data.dropna(axis=1, how="all").dropna(axis=0, how="all")
    stb_data = stb_data.dropna(subset=protected_columns, how="any")  # Ensure IDs are not lost



# remove unexpected empty rows too (remove the NaNs for these)
if experiment_data is not None:
    experiment_data = experiment_data.dropna(axis=0, how='all')

if stb_data is not None:
    stb_data = stb_data.dropna(axis=0, how='all')


original_experiment_columns = experiment_data.columns if experiment_data is not None else []
original_stb_columns = stb_data.columns if stb_data is not None else []


dropped_experiment_cols = set(original_experiment_columns) - set(experiment_data.columns)
dropped_stb_cols = set(original_stb_columns) - set(stb_data.columns)
logger.info(f"Dropped experiment columns: {dropped_experiment_cols}")
logger.info(f"Dropped STB columns: {dropped_stb_cols}")

# Identify initial common columns BEFORE imputation
# retain only common columns
experiment_data, stb_data, common_columns_before = process_common_columns(experiment_data, stb_data, step="before")
logger.info(f"Common numerical columns BEFORE imputation: {len(common_columns_before)}")


####################
# Perform imputation
# Ensure numerical datasets
experiment_data_imputed, stb_data_imputed = None, None  
experiment_data_imputed, stb_data_imputed, stb_labels, \
    stb_cpd_id_map = impute_missing_values(experiment_data, stb_data, 
                                           impute_method=args.impute_method, 
                                           knn_neighbors=args.knn_neighbors)

# Also restore non-numeric columns into experiment_data and stb_data
# so that encode_cpd_data() doesn't complain
experiment_data = experiment_data_imputed.copy()
stb_data = stb_data_imputed.copy()


# Step 2: Now it's safe to check for missing columns
missing_exp_cols = {"cpd_id", "Library", "cpd_type"} - set(experiment_data_imputed.columns)
missing_stb_cols = {"cpd_id", "Library", "cpd_type"} - set(stb_data_imputed.columns)

if missing_exp_cols:
    raise ValueError(f"Missing restored columns in experiment_data_imputed: {missing_exp_cols}")
if missing_stb_cols:
    raise ValueError(f"Missing restored columns in stb_data_imputed: {missing_stb_cols}")


# Log the first few rows after imputation
if experiment_data_imputed is not None:
    logger.info("First few rows of experiment_data_imputed:\n" + experiment_data_imputed.head().to_string())
if stb_data_imputed is not None:
    logger.info("First few rows of stb_data_imputed:\n" + stb_data_imputed.head().to_string())




# Step 3: Identify common columns AFTER imputation
# retain only those coloumns
experiment_data_imputed, stb_data_imputed, \
    common_columns_after = process_common_columns(experiment_data_imputed, 
                                                  stb_data_imputed, step="after")


logger.info(f"Common numerical columns AFTER imputation: {len(common_columns_after)}")
columns_lost = set(common_columns_before) - set(common_columns_after)
logger.info(f"Columns lost during imputation: {len(columns_lost)}")


columns_lost = set(common_columns_before) - set(common_columns_after)
logger.info(f"Columns lost during imputation: {len(columns_lost)}")
logger.info(f"Lost columns: {columns_lost}")  # Add this line




#   Check if experiment_data_imputed is None before accessing .shape
if experiment_data_imputed is not None:
    logger.info(f"Experiment data shape after imputation: {experiment_data_imputed.shape}")
else:
    logger.info("Experiment data is None after imputation.")

#   Check if stb_data_imputed is None before accessing .shape
if stb_data_imputed is not None:
    logger.info(f"STB data shape after imputation: {stb_data_imputed.shape}")
else:
    logger.info("STB data is None after imputation.")


# Save MultiIndex from imputed data BEFORE passing to CLIPn (for later reconstruction)
experiment_index_backup = experiment_data_imputed.index if experiment_data_imputed is not None else None
stb_index_backup = stb_data_imputed.index if stb_data_imputed is not None else None

# Check if MultiIndex still exists after imputation
logger.info(f"Experiment MultiIndex after imputation: {experiment_data_imputed.index.names}")
logger.info(f"STB MultiIndex after imputation: {stb_data_imputed.index.names}")

# Check if cpd_id and Library are present in columns (if MultiIndex was reset)
# Sanity check: ensure 'cpd_id' and 'Library' are still present after imputation
missing_cols_experiment = {"cpd_id", "Library"} - set(experiment_data_imputed.columns)
missing_cols_stb = {"cpd_id", "Library"} - set(stb_data_imputed.columns)

if missing_cols_experiment:
    logger.warning(f"Experiment data is missing: {missing_cols_experiment}")

if missing_cols_stb:
    logger.warning(f"STB data is missing: {missing_cols_stb}")



#################################################
# encode the non numeric, but important cols. 
# Create dataset labels
dataset_labels = {0: "experiment Assay", 1: "STB"}

# Encode cpd_type and cpd_id for both experiment and STB datasets
dataframes = {
    "experiment": experiment_data_imputed,
    "stb": stb_data_imputed
}


encoded_results = encode_cpd_data(dataframes, encode_labels=True)

# Experiment data
experiment_data = encoded_results["experiment"]["data"]
experiment_labels = encoded_results["experiment"]["cpd_type_encoded"]
experiment_label_mapping = encoded_results["experiment"]["cpd_type_mapping"]
experiment_cpd_id_map = encoded_results["experiment"]["cpd_id_mapping"]
experiment_label_encoder = encoded_results["experiment"]["cpd_type_encoder"]
experiment_id_encoder = encoded_results["experiment"]["cpd_id_encoder"]

# STB data
stb_data = encoded_results["stb"]["data"]
stb_labels = encoded_results["stb"]["cpd_type_encoded"]
stb_label_mapping = encoded_results["stb"]["cpd_type_mapping"]
stb_cpd_id_map = encoded_results["stb"]["cpd_id_mapping"]
stb_label_encoder = encoded_results["stb"]["cpd_type_encoder"]
stb_id_encoder = encoded_results["stb"]["cpd_id_encoder"]


logger.info(f"Encoded Experiment Labels: {experiment_labels[:5]}")
logger.info(f"Encoded STB Labels: {stb_labels[:5]}")
logger.info(f"Experiment cpd_id mapping size: {len(experiment_cpd_id_map)}")
logger.info(f"STB cpd_id mapping size: {len(stb_cpd_id_map)}")
logger.info(f"Experiment label mapping: {experiment_label_mapping}")
logger.info(f"STB label mapping: {stb_label_mapping}")

# Restore non-numeric metadata columns if needed
for col in ["cpd_id", "Library", "cpd_type"]:
    if col not in experiment_data.columns and col in experiment_data_imputed.columns:
        experiment_data[col] = experiment_data_imputed[col]
    if col not in stb_data.columns and col in stb_data_imputed.columns:
        stb_data[col] = stb_data_imputed[col]


required_columns = {"cpd_id", "Library", "cpd_type"}
missing_exp = required_columns - set(experiment_data.columns)
missing_stb = required_columns - set(stb_data.columns)

if missing_exp:
    logger.error(f"Missing required columns in experiment dataset: {missing_exp}")
if missing_stb:
    logger.error(f"Missing required columns in STB dataset: {missing_stb}")

if missing_exp or missing_stb:
    raise ValueError("Critical columns are missing after preprocessing. Check logs above.")



# Define datasets to process
datasets = {
    "experiment": experiment_data_imputed,
    "stb": stb_data_imputed
}

# Ensure at least one dataset exists before proceeding
if experiment_data_imputed is None and stb_data_imputed is None:
    logger.error("No valid datasets available for CLIPn analysis.")
    raise ValueError("Error: No valid datasets available for CLIPn analysis.")

# Assign numerical indices for dataset names ONLY if required for CLIPn
if experiment_data_imputed is not None and not experiment_data_imputed.empty:
    experiment_data_imputed["dataset_label"] = 0  # Experiment → 0
if stb_data_imputed is not None and not stb_data_imputed.empty:
    stb_data_imputed["dataset_label"] = 1  # STB → 1

logger.info("Dataset labels assigned: Experiment → 0, STB → 1")

# Log the updated DataFrames
if experiment_data_imputed is not None:
    logger.info(f"First few rows of experiment_data_imputed:\n{experiment_data_imputed.head()}")
if stb_data_imputed is not None:
    logger.info(f"First few rows of stb_data_imputed:\n{stb_data_imputed.head()}")



# Define the pattern for columns to drop
filter_pattern = re.compile(
    r"Source_Plate_Barcode|COMPOUND_NUMBER|Notes|Seahorse_alert|Treatment|Number|"
    r"Child|Paren|Location_[X,Y,Z]|ZernikePhase|Euler|Plate|Well|Field|"
    r"Center_Z|Center_X|Center_Y|no_|fn_"
)

# Reassign the updated datasets
experiment_data_imputed = datasets["experiment"]
stb_data_imputed = datasets["stb"]



# Apply grouping and filtering to both datasets
# Ensure 'cpd_id', 'Library', and 'cpd_type' exist before setting MultiIndex
required_cols = {"cpd_id", "Library", "cpd_type"}

if not required_cols.issubset(experiment_data_imputed.columns):
    raise ValueError(f"Missing columns in experiment_data_imputed: {required_cols - set(experiment_data_imputed.columns)}")

if not required_cols.issubset(stb_data_imputed.columns):
    raise ValueError(f"Missing columns in stb_data_imputed: {required_cols - set(stb_data_imputed.columns)}")

# Set MultiIndex
experiment_data_imputed = ensure_multiindex(experiment_data_imputed, logger=logger, dataset_name="experiment_data_imputed")
stb_data_imputed = ensure_multiindex(stb_data_imputed, logger=logger, dataset_name="stb_data_imputed")


logger.info(f"After setting MultiIndex: Experiment Data Index: {experiment_data_imputed.index.names}")
logger.info(f"After setting MultiIndex: STB Data Index: {stb_data_imputed.index.names}")

logger.info(f"Before grouping: STB Data Index: {stb_data_imputed.index.names}")
logger.info(f"Before grouping: Experiment Data Index: {experiment_data_imputed.index.names}")



experiment_data_imputed = group_and_filter_data(experiment_data_imputed)
stb_data_imputed = group_and_filter_data(stb_data_imputed)

# Keep a backup of the grouped index (after grouping by cpd_id and Library)
experiment_index_backup = experiment_data_imputed.index
stb_index_backup = stb_data_imputed.index


# Log results
logger.info("Grouped and filtered experiment data shape: {}".format(experiment_data_imputed.shape if experiment_data_imputed is not None else "None"))
logger.info("Grouped and filtered STB data shape: {}".format(stb_data_imputed.shape if stb_data_imputed is not None else "None"))


#######################################################
# Initialize empty dictionaries for CLIPn input
# 

X, y, label_mappings, dataset_mapping = prepare_data_for_clipn(
                                experiment_data_imputed=experiment_data_imputed,
                                experiment_labels=experiment_labels,
                                experiment_label_mapping=experiment_label_mapping,
                                stb_data_imputed=stb_data_imputed,
                                stb_labels=stb_labels,
                                stb_label_mapping=stb_label_mapping
)


logger.info(" Datasets successfully structured for CLIPn.")
logger.info(f" Final dataset shapes being passed to CLIPn: { {k: v.shape for k, v in X.items()} }")



########################################################
# CLIPn clustering with hyper optimisation
logger.info(f"Running CLIPn")

# Define hyperparameter output path

logger.info(f"Arguments passed to run_clipn: latent_dim={args.latent_dim}, lr={args.lr}, epoch={args.epoch}, use_optimized_params={args.use_optimized_params}")

Z = run_clipn(X, y, output_folder, args)


# mk new dir for new params. 
output_folder = os.path.join(main_output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}")
os.makedirs(output_folder, exist_ok=True)

# Save latent representations
# Convert numerical dataset names back to string keys and make values serialisable
Z_named = {str(k): v.tolist() for k, v in Z.items()}  # Convert keys to strings and values to lists

np.savez(os.path.join(output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_latent_representations.npz"), **Z_named)

# Convert numerical dataset names back to original names
Z_named = {str(k): v.tolist() for k, v in Z.items()}  # Convert keys to strings and values to lists

# Save latent representations in NPZ format
np.savez(os.path.join(output_folder, "CLIPn_latent_representations.npz"), **Z_named)
logger.info("Latent representations saved successfully.")



# Store your MultiIndex backups from before CLIPn:
index_lookup = {
    "experiment_assay_combined": experiment_index_backup,
    "STB_combined": stb_index_backup
}

combined_latent_df = reconstruct_combined_latent_df(Z, dataset_mapping, index_lookup)



# Perform UMAP
logger.info("Generating UMAP visualization.")
# Define the plot filename outside the function
umap_plot_file = os.path.join(output_folder, "clipn_ldim_UMAP.pdf")

# Generate UMAP without labels
umap_df = generate_umap(combined_latent_df, output_folder, umap_plot_file, args, add_labels=False)
# Generate UMAP with labels
umap_plot_file = os.path.join(output_folder, "clipn_ldim_UMAP_labels.pdf")
umap_df = generate_umap(combined_latent_df, output_folder, umap_plot_file, args, add_labels=True)


# === Summarise UMAP clusters by cpd_type and cpd_id ===
logger.info("Generating UMAP cluster summary.")

try:
    # Check required columns exist
    if "Cluster" not in umap_df.columns:
        raise ValueError("Cluster column not found in UMAP DataFrame.")

    # Reset index to access cpd_id and cpd_type
    umap_reset = umap_df.reset_index()

    # Group and summarise
    cluster_summary = (
        umap_reset.groupby("Cluster")
        .agg({
            "cpd_type": lambda x: list(sorted(set(x))),
            "cpd_id": lambda x: list(sorted(set(x)))
        })
        .rename(columns={
            "cpd_type": "cpd_types_in_cluster",
            "cpd_id": "cpd_ids_in_cluster"
        })
        .reset_index()
    )

    # Save to file
    cluster_summary_file = os.path.join(output_folder, "umap_cluster_summary.csv")
    cluster_summary.to_csv(cluster_summary_file, index=False)
    logger.info(f"UMAP cluster summary saved to '{cluster_summary_file}'.")

except Exception as e:
    logger.error(f"Failed to generate UMAP cluster summary: {e}")


# Save each dataset's latent representations separately with the correct prefix
for dataset, values in Z_named.items():
    df = pd.DataFrame(values)
    df.to_csv(os.path.join(output_folder, f"latent_representations_{dataset}.csv"), index=False)

logger.info(f"Latent representations saved successfully in {output_folder}/")

# Save index and label mappings
pd.Series({"experiment_assay_combined": 0, "STB_combined": 1}).to_csv(
    os.path.join(output_folder, f"dataset_index_mapping.csv")
)
pd.DataFrame(label_mappings).to_csv(
    os.path.join(output_folder, f"label_mappings.csv")
)

logger.info(f"Index and label mappings saved successfully in {output_folder}/")
logger.info("Index and label mappings saved.")


# Ensure `combined_latent_df` retains the MultiIndex before saving
combined_output_file = os.path.join(output_folder, "CLIPn_latent_representations_with_cpd_id.csv")

# Ensure MultiIndex is preserved
if isinstance(combined_latent_df.index, pd.MultiIndex):
    combined_latent_df.to_csv(combined_output_file)
else:
    logger.warning("MultiIndex is missing! Attempting to restore it before saving.")
    if {"cpd_id", "Library", "cpd_type"}.issubset(combined_latent_df.columns):
        combined_latent_df = combined_latent_df.set_index(["cpd_id", "Library", "cpd_type"])
    combined_latent_df.to_csv(combined_output_file)

logger.info(f"Latent representations saved successfully with MultiIndex to {combined_output_file}.")


# Log the combined latent representation DataFrame
if combined_latent_df is not None:
    logger.info("First few rows of combined_latent_df (final merged latent space):\n" + combined_latent_df.head().to_string())



# UMAP per experiment

# Define the UMAP output file before calling function
umap_experiment_plot_file = os.path.join(output_folder, "UMAP_experiment_vs_stb.pdf")

# Call the function using the UMAP DataFrame (ensuring MultiIndex)
plot_umap_coloured_by_experiment(umap_df, umap_experiment_plot_file)


###########
# Generate Summary of Closest & Farthest Compounds
# Compute pairwise distances **before** using `dist_df`
logger.info("Computing pairwise compound distances at the (cpd_id, Library) level without collapsing across Libraries.")

try:
    # Drop non-numeric columns (e.g. dataset name annotations, strings)
    numeric_latent_df = combined_latent_df.select_dtypes(include=[np.number])

    # Compute pairwise Euclidean distances
    dist_df = compute_pairwise_distances(numeric_latent_df)

    # Restore full MultiIndex
    dist_df.index = combined_latent_df.index
    dist_df.columns = combined_latent_df.index

    logger.info(f"Distance matrix shape: {dist_df.shape}")

    # Save distance matrix to CSV (MultiIndex as compound identifiers)
    distance_matrix_file = os.path.join(output_folder, "pairwise_compound_distances.csv")
    dist_df.to_csv(distance_matrix_file)
    logger.info(f"Pairwise distance matrix saved to '{distance_matrix_file}'.")

except Exception as e:
    logger.error(f"Error computing pairwise distances: {e}")


# Save distance matrix
distance_matrix_file = os.path.join(output_folder, "pairwise_compound_distances.csv")
dist_df.to_csv(distance_matrix_file)
logger.info(f"Pairwise distance matrix saved to '{distance_matrix_file}'.")

# **Now, generate similarity summary**
summary_df = generate_similarity_summary(dist_df)

# Ensure `cpd_id` is correctly assigned
summary_df["Compound"] = summary_df["Compound"].astype(str)
summary_df["Closest Compound"] = summary_df["Closest Compound"].astype(str)
summary_df["Farthest Compound"] = summary_df["Farthest Compound"].astype(str)

summary_file = os.path.join(output_folder, "compound_similarity_summary.csv")
summary_df.to_csv(summary_file, index=False)
logger.info(f"Compound similarity summary saved to '{summary_file}'.")



#  **Generate and Save Heatmap**
heatmap_file = os.path.join(output_folder, "compound_distance_heatmap.pdf")
plot_distance_heatmap(dist_df, heatmap_file)
logger.info(f"Pairwise distance heatmap saved to '{heatmap_file}'.")

# **Generate and Save Dendrogram**
dendrogram_file = os.path.join(output_folder, "compound_clustering_dendrogram.pdf")
plot_dendrogram(dist_df, dendrogram_file)
logger.info(f"Hierarchical clustering dendrogram saved to '{dendrogram_file}'.")

logger.info("Intergration step finished")

