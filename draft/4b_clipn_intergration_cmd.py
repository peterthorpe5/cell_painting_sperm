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



def restore_cpd_id(imputed_df, cpd_id_map):
    """
    Restore the 'cpd_id' and 'Library' columns after data processing.
    
    Parameters:
    -----------
    imputed_df : pd.DataFrame
        The DataFrame with transformed numerical features.
    cpd_id_map : dict
        A dictionary mapping original row indices to 'cpd_id' values.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with 'cpd_id' restored as an index.
    """
    if imputed_df is None or imputed_df.empty:
        logger.warning("Imputed data is empty. Returning unchanged.")
        return imputed_df

    if cpd_id_map is None:
        logger.error("cpd_id_map is missing! Cannot restore 'cpd_id'. Returning unchanged.")
        return imputed_df

    # Ensure that the imputed DataFrame has the correct number of rows
    expected_rows = len(imputed_df)
    actual_mapped_rows = min(expected_rows, len(cpd_id_map))

    # Assign `cpd_id` based on the stored mapping, filling missing values with "Unknown"
    restored_cpd_ids = [cpd_id_map.get(i, f"Unknown_{i}") for i in range(expected_rows)]

    # Set `cpd_id` as the index
    imputed_df.index = restored_cpd_ids

    # Log any mismatches
    missing_count = sum(1 for cpd in restored_cpd_ids if cpd.startswith("Unknown"))
    if missing_count > 0:
        logger.warning(f"Warning: {missing_count} missing 'cpd_id' values were replaced with 'Unknown_X'.")

    return imputed_df



def restore_non_numeric(imputed_df, mappings):
    """
    Restore the 'cpd_id' and 'Library' columns after imputation.

    Parameters:
    -----------
    imputed_df : pd.DataFrame
        The DataFrame with numeric values imputed.
    mappings : dict
        A dictionary mapping row indices to {'cpd_id': value, 'Library': value}.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with 'cpd_id' and 'Library' restored.
    """
    if mappings is None:
        logger.warning("No mappings found. Returning data unchanged.")
        return imputed_df

    # Convert mappings back to DataFrame
    non_numeric_df = pd.DataFrame.from_dict(mappings, orient="index")
    
    # Ensure both DataFrames have the same index
    if len(non_numeric_df) != len(imputed_df):
        logger.error("Mismatch in row counts after imputation. Check data integrity.")
        sys.exit(1)

    return pd.concat([non_numeric_df, imputed_df], axis=1)


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


def plot_umap_colored_by_experiment(umap_df, output_file, color_map=None):
    """
    Generates a UMAP visualization colored by experiment vs. STB.

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


def reconstruct_combined_latent_df(Z, experiment_data_imputed, stb_data_imputed):
    """
    Reconstructs a DataFrame of latent representations with MultiIndex.

    Parameters
    ----------
    Z : dict
        Dictionary containing CLIPn latent representations.

    experiment_data_imputed : pd.DataFrame
        Imputed experiment dataset with MultiIndex (cpd_id, Library, cpd_type).

    stb_data_imputed : pd.DataFrame
        Imputed STB dataset with MultiIndex (cpd_id, Library, cpd_type).

    Returns
    -------
    pd.DataFrame
        Combined latent representations indexed by (cpd_id, Library, cpd_type).
    """

    # Convert experiment latent representations
    experiment_latent_df = pd.DataFrame(Z[0], index=experiment_data_imputed.index)

    # Convert STB latent representations
    stb_latent_df = pd.DataFrame(Z[1], index=stb_data_imputed.index)

    # Concatenate both datasets
    combined_latent_df = pd.concat([experiment_latent_df, stb_latent_df])

    return combined_latent_df


def impute_missing_values(experiment_data, stb_data, impute_method="median", knn_neighbors=5):
    """
    Perform missing value imputation while preserving MultiIndex (cpd_id, Library).

    Parameters:
    -----------
    experiment_data : pd.DataFrame or None
        Experiment dataset with `cpd_id` and `Library` as MultiIndex.
    stb_data : pd.DataFrame or None
        STB dataset with `cpd_id` and `Library` as MultiIndex.
    impute_method : str, optional
        Imputation method: "median" (default) or "knn".
    knn_neighbors : int, optional
        Number of neighbors for KNN imputation (default is 5).

    Returns:
    --------
    tuple:
        - experiment_data_imputed : pd.DataFrame or None
        - stb_data_imputed : pd.DataFrame or None
        - stb_labels : np.array
        - stb_cpd_id_map : dict
    """
    logger.info(f"Performing imputation using {impute_method} strategy.")

    # Choose imputer based on method
    if impute_method == "median":
        imputer = SimpleImputer(strategy="median")
    elif impute_method == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
    else:
        raise ValueError("Invalid imputation method. Choose 'median' or 'knn'.")

    # Function to apply imputation and restore MultiIndex
    def impute_dataframe(df):
        if df is None or df.empty:
            return df
        # Backup original index
        original_index = df.index
        # Extract numeric columns
        numeric_df = df.select_dtypes(include=[np.number])  
        # Perform imputation
        imputed_array = imputer.fit_transform(numeric_df)
        # Convert back to DataFrame
        imputed_df = pd.DataFrame(imputed_array, index=original_index, columns=numeric_df.columns)
        # Restore MultiIndex if it was originally present
        if isinstance(original_index, pd.MultiIndex):
            imputed_df.index = original_index  # Explicitly reapply MultiIndex
        return imputed_df

    # Apply imputation to each dataset
    experiment_data_imputed = impute_dataframe(experiment_data)
    stb_data_imputed = impute_dataframe(stb_data)
    logger.info(f"Imputation complete. Experiment shape: {experiment_data_imputed.shape if experiment_data_imputed is not None else 'None'}, "
                f"STB shape: {stb_data_imputed.shape if stb_data_imputed is not None else 'None'}")
    # Handle STB labels
    if stb_data is not None and "cpd_type" in stb_data.columns:
        label_encoder = LabelEncoder()
        stb_labels = label_encoder.fit_transform(stb_data["cpd_type"])
        stb_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # Store mapping of encoded label to cpd_id
        if "cpd_id" in stb_data.index.get_level_values(0):
            stb_cpd_id_map = dict(zip(stb_data.index.get_level_values(0), stb_labels))
        else:
            stb_cpd_id_map = {}
            logger.warning("Warning: 'cpd_id' is missing from STB data index!")
    else:
        stb_labels = np.zeros(stb_data_imputed.shape[0]) if stb_data_imputed is not None else np.array([])
        stb_label_mapping = {"unknown": 0}
        stb_cpd_id_map = {}
        logger.warning("Warning: No STB labels available!")
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


def encode_cpd_data(dataframes, encode=False):
    """
    Ensures 'cpd_id', 'Library', and 'cpd_type' are set as a MultiIndex
    and optionally encodes them for machine learning.

    Parameters
    ----------
    dataframes : dict
        Dictionary where keys are dataset names and values are DataFrames.
    encode : bool, optional
        If True, encodes 'cpd_id' and 'cpd_type' numerically (default is False).

    Returns
    -------
    dict
        Dictionary of processed DataFrames with MultiIndex applied.
    """
    processed_dfs = {}

    for name, df in dataframes.items():
        if df is not None and not df.empty:
            # Ensure 'cpd_id', 'Library', and 'cpd_type' exist before indexing
            if {"cpd_id", "Library", "cpd_type"}.issubset(df.columns):
                df = df.set_index(["cpd_id", "Library", "cpd_type"])
            else:
                raise ValueError(f"Missing required columns in {name} dataset: {df.columns}")

            # Encode 'cpd_id' and 'cpd_type' if requested
            if encode:
                df = df.copy()  # Ensure we don't modify the original dataframe
                df["cpd_id_encoded"] = LabelEncoder().fit_transform(df.index.get_level_values("cpd_id"))
                df["cpd_type_encoded"] = LabelEncoder().fit_transform(df.index.get_level_values("cpd_type"))

                # Restore MultiIndex after encoding
                df = df.reset_index().set_index(["cpd_id", "Library", "cpd_type"])

            processed_dfs[name] = df

    return processed_dfs



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

    return X, y, label_mappings


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
    latent_umap = umap_model.fit_transform(combined_latent_df)

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
    plt.title("CLIPn UMAP Visualization (Colored by Cluster)")
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
# **3. Initialize Logger BEFORE Calling logger.info()**

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


# **Drop columns that are entirely NaN in either dataset BEFORE imputation**
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


# Backup the full MultiIndex as a DataFrame

# ✅ Ensure full MultiIndex is backed up as a proper DataFrame
if experiment_data is not None:
    experiment_index_backup = experiment_data.index.to_frame(index=False)  # Converts MultiIndex to DataFrame
    experiment_index_backup.columns = ["cpd_id", "Library", "cpd_type"]  # Explicitly name columns

if stb_data is not None:
    stb_index_backup = stb_data.index.to_frame(index=False)  # Converts MultiIndex to DataFrame
    stb_index_backup.columns = ["cpd_id", "Library", "cpd_type"]  # Explicitly name columns

# ✅ Confirm backup is correct
logger.info(f"Backed up index before imputation. Experiment index shape: {experiment_index_backup.shape}")
logger.info(f"Backed up index before imputation. STB index shape: {stb_index_backup.shape}")
logger.info(f"Experiment index backup preview:\n{experiment_index_backup.head()}")
logger.info(f"STB index backup preview:\n{stb_index_backup.head()}")

# 🚨 Check if backup contains the required columns
missing_exp_cols = {"cpd_id", "Library", "cpd_type"} - set(experiment_index_backup.columns)
if missing_exp_cols:
    logger.error(f"Missing columns in index_backup for experiment: {missing_exp_cols}")
else:
    logger.info(" experiment_index_backup contains all required columns.")

missing_stb_cols = {"cpd_id", "Library", "cpd_type"} - set(stb_index_backup.columns)
if missing_stb_cols:
    logger.error(f" Missing columns in index_backup for stb: {missing_stb_cols}")
else:
    logger.info(" stb_index_backup contains all required columns.")


logger.info(f"Backed up index before imputation. Experiment index shape: {experiment_index_backup.shape if experiment_index_backup is not None else 'None'}")
logger.info(f"Backed up index before imputation. STB index shape: {stb_index_backup.shape if stb_index_backup is not None else 'None'}")

logger.info(f"Experiment index backup preview:\n{experiment_index_backup.head() if experiment_index_backup is not None else 'None'}")
logger.info(f"STB index backup preview:\n{stb_index_backup.head() if stb_index_backup is not None else 'None'}")

####################
# Perform imputation
# Ensure numerical datasets
experiment_data_imputed, stb_data_imputed = None, None  
experiment_data_imputed, stb_data_imputed, stb_labels, \
    stb_cpd_id_map = impute_missing_values(experiment_data, stb_data, 
                                           impute_method=args.impute_method, 
                                           knn_neighbors=args.knn_neighbors)

# Restore the original MultiIndex after imputation
# Ensure index backup is not empty

# Restore MultiIndex for both datasets
experiment_data_imputed = restore_multiindex(experiment_data_imputed, experiment_index_backup, "experiment")
stb_data_imputed = restore_multiindex(stb_data_imputed, stb_index_backup, "stb")




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


# Ensure original data exists
if experiment_data is not None and stb_data is not None:

    # Restore 'cpd_id', 'Library', and 'cpd_type' before setting MultiIndex
    experiment_data_imputed = experiment_data_imputed.copy()
    stb_data_imputed = stb_data_imputed.copy()

    for col in ["cpd_id", "Library", "cpd_type"]:
        if col in experiment_data.index.names:
            experiment_data_imputed[col] = experiment_data.index.get_level_values(col)
        elif col in experiment_data.columns:
            experiment_data_imputed[col] = experiment_data[col]
        else:
            logger.warning(f"Warning: {col} not found in experiment_data!")

        if col in stb_data.index.names:
            stb_data_imputed[col] = stb_data.index.get_level_values(col)
        elif col in stb_data.columns:
            stb_data_imputed[col] = stb_data[col]
        else:
            logger.warning(f"Warning: {col} not found in stb_data!")

    logger.info("Restored 'cpd_id', 'Library', and 'cpd_type' to imputed datasets.")

# Now check if columns exist before setting MultiIndex
missing_exp_cols = {"cpd_id", "Library", "cpd_type"} - set(experiment_data_imputed.columns)
missing_stb_cols = {"cpd_id", "Library", "cpd_type"} - set(stb_data_imputed.columns)

if missing_exp_cols:
    raise ValueError(f"Missing restored columns in experiment_data_imputed: {missing_exp_cols}")

if missing_stb_cols:
    raise ValueError(f"Missing restored columns in stb_data_imputed: {missing_stb_cols}")

# Now, we can safely set MultiIndex
experiment_data_imputed = experiment_data_imputed.set_index(["cpd_id", "Library", "cpd_type"])
stb_data_imputed = stb_data_imputed.set_index(["cpd_id", "Library", "cpd_type"])

logger.info(f"Successfully restored MultiIndex: {experiment_data_imputed.index.names}")
logger.info(f"Successfully restored MultiIndex: {stb_data_imputed.index.names}")



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


# Check if MultiIndex still exists after imputation
logger.info(f"Experiment MultiIndex after imputation: {experiment_data_imputed.index.names}")
logger.info(f"STB MultiIndex after imputation: {stb_data_imputed.index.names}")

# Check if cpd_id and Library are present in columns (if MultiIndex was reset)
missing_cols_experiment = {"cpd_id", "Library"} - set(experiment_data_imputed.columns)
missing_cols_stb = {"cpd_id", "Library"} - set(stb_data_imputed.columns)

if missing_cols_experiment:
    logger.warning(f"Experiment data is missing: {missing_cols_experiment}")

if missing_cols_stb:
    logger.warning(f"STB data is missing: {missing_cols_stb}")



# If imputation removed MultiIndex, restore it!

# Ensure MultiIndex is restored
if not isinstance(experiment_data_imputed.index, pd.MultiIndex):
    experiment_data_imputed = experiment_data_imputed.set_index(["cpd_id", "Library", "cpd_type"])
    logger.info("Restored MultiIndex for experiment_data_imputed.")

if not isinstance(stb_data_imputed.index, pd.MultiIndex):
    stb_data_imputed = stb_data_imputed.set_index(["cpd_id", "Library", "cpd_type"])
    logger.info("Restored MultiIndex for stb_data_imputed.")

# Debugging AFTER restoring MultiIndex
logger.info(f"After restoring - Experiment index levels: {experiment_data_imputed.index.names}")
logger.info(f"After restoring - STB index levels: {stb_data_imputed.index.names}")

# Create dataset labels
dataset_labels = {0: "experiment Assay", 1: "STB"}

# Initialize dictionaries to store mappings between LabelEncoder values and cpd_id
experiment_cpd_id_map = {}
stb_cpd_id_map = {}



required_columns = {"cpd_id", "Library", "cpd_type"}
missing_exp = required_columns - set(experiment_data.columns)
missing_stb = required_columns - set(stb_data.columns)

if missing_exp:
    logger.error(f"Missing required columns in experiment dataset: {missing_exp}")
if missing_stb:
    logger.error(f"Missing required columns in STB dataset: {missing_stb}")

if missing_exp or missing_stb:
    raise ValueError("Critical columns are missing after preprocessing. Check logs above.")


# Handle labels (assuming 'cpd_type' exists)
# Labelencoder as cmp_id and type have to be numeric. 
# Prepare DataFrames for encoding
dataframes = {
    "experiment": experiment_data,
    "stb": stb_data
}

# Apply MultiIndex without encoding
processed_data = encode_cpd_data(dataframes, encode=False)

# If encoding is required (for ML models), apply encoding
encoded_data = encode_cpd_data(dataframes, encode=True)

# Extract processed datasets
experiment_data = processed_data.get("experiment")
stb_data = processed_data.get("stb")

# Extract encoded datasets if needed
experiment_data_encoded = encoded_data.get("experiment")
stb_data_encoded = encoded_data.get("stb")

#  Ensure MultiIndex is not lost
logger.info(f"After encoding - Experiment index levels: {experiment_data.index.names}")
logger.info(f"After encoding - STB index levels: {stb_data.index.names}")

# Debugging logs
logger.info(f"Experiment DataFrame after MultiIndexing:\n{experiment_data.head()}")
logger.info(f"STB DataFrame after MultiIndexing:\n{stb_data.head()}")

# Handle encoded labels for downstream ML tasks
if experiment_data_encoded is not None:
    experiment_labels = experiment_data_encoded["cpd_type_encoded"].values
    experiment_cpd_id_map = dict(zip(experiment_data_encoded.index.get_level_values("cpd_id"),
                                     experiment_data_encoded["cpd_id_encoded"]))
    experiment_label_mapping = dict(zip(experiment_data_encoded.index.get_level_values("cpd_type"),
                                        experiment_data_encoded["cpd_type_encoded"]))
else:
    experiment_labels = np.array([])
    experiment_cpd_id_map = {}
    experiment_label_mapping = {}

if stb_data_encoded is not None:
    stb_labels = stb_data_encoded["cpd_type_encoded"].values
    stb_cpd_id_map = dict(zip(stb_data_encoded.index.get_level_values("cpd_id"),
                              stb_data_encoded["cpd_id_encoded"]))
    stb_label_mapping = dict(zip(stb_data_encoded.index.get_level_values("cpd_type"),
                                 stb_data_encoded["cpd_type_encoded"]))
else:
    stb_labels = np.array([])
    stb_cpd_id_map = {}
    stb_label_mapping = {}

# Log outputs
logger.info(f"Encoded Experiment Labels: {experiment_labels[:5] if len(experiment_labels) > 5 else experiment_labels}")
logger.info(f"Encoded STB Labels: {stb_labels[:5] if len(stb_labels) > 5 else stb_labels}")
logger.info(f"Experiment cpd_id mapping size: {len(experiment_cpd_id_map)}")
logger.info(f"STB cpd_id mapping size: {len(stb_cpd_id_map)}")
logger.info(f"Experiment label mapping: {experiment_label_mapping}")
logger.info(f"STB label mapping: {stb_label_mapping}")



logger.info(f"Experiment label mapping: {experiment_label_mapping}")
logger.info(f"Experiment cpd_id mapping size: {len(experiment_cpd_id_map)}")

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



# If imputation removed MultiIndex, restore it!
if {"cpd_id", "Library", "cpd_type"}.issubset(experiment_data.columns):
    experiment_data_imputed = restore_non_numeric(experiment_data_imputed, experiment_data)
    experiment_data_imputed = experiment_data_imputed.set_index(["cpd_id", "Library", "cpd_type"])
    logger.info("Restored MultiIndex after imputation for experiment_data_imputed.")
else:
    raise ValueError("Critical columns missing BEFORE imputation! Check previous steps.")

if {"cpd_id", "Library", "cpd_type"}.issubset(stb_data.columns):
    stb_data_imputed = restore_non_numeric(stb_data_imputed, stb_data)
    stb_data_imputed = stb_data_imputed.set_index(["cpd_id", "Library", "cpd_type"])
    logger.info("Restored MultiIndex after imputation for stb_data_imputed.")
else:
    raise ValueError("Critical columns missing BEFORE imputation! Check previous steps.")



# Apply grouping and filtering to both datasets
# Ensure 'cpd_id', 'Library', and 'cpd_type' exist before setting MultiIndex
required_cols = {"cpd_id", "Library", "cpd_type"}

if not required_cols.issubset(experiment_data_imputed.columns):
    raise ValueError(f"Missing columns in experiment_data_imputed: {required_cols - set(experiment_data_imputed.columns)}")

if not required_cols.issubset(stb_data_imputed.columns):
    raise ValueError(f"Missing columns in stb_data_imputed: {required_cols - set(stb_data_imputed.columns)}")

# Set MultiIndex
experiment_data_imputed = experiment_data_imputed.set_index(["cpd_id", "Library", "cpd_type"])
stb_data_imputed = stb_data_imputed.set_index(["cpd_id", "Library", "cpd_type"])

logger.info(f"After setting MultiIndex: Experiment Data Index: {experiment_data_imputed.index.names}")
logger.info(f"After setting MultiIndex: STB Data Index: {stb_data_imputed.index.names}")

logger.info(f"Before grouping: STB Data Index: {stb_data_imputed.index.names}")
logger.info(f"Before grouping: Experiment Data Index: {experiment_data_imputed.index.names}")



experiment_data_imputed = group_and_filter_data(experiment_data_imputed)
stb_data_imputed = group_and_filter_data(stb_data_imputed)

# Log results
logger.info("Grouped and filtered experiment data shape: {}".format(experiment_data_imputed.shape if experiment_data_imputed is not None else "None"))
logger.info("Grouped and filtered STB data shape: {}".format(stb_data_imputed.shape if stb_data_imputed is not None else "None"))


#######################################################
# Initialize empty dictionaries for CLIPn input
# 

X, y, label_mappings = prepare_data_for_clipn(
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
np.savez(os.path.join(output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_latent_representations.npz"), **Z_named)

# Convert numerical dataset names back to original names
Z_named = {str(k): v.tolist() for k, v in Z.items()}  # Convert keys to strings and values to lists

# Save latent representations in NPZ format
np.savez(os.path.join(output_folder, "CLIPn_latent_representations.npz"), **Z_named)
logger.info("Latent representations saved successfully.")


# Perform UMAP
logger.info("Generating UMAP visualization.")
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
latent_umap = umap_model.fit_transform(np.vstack([Z[0], Z[1]]))


combined_latent_df = reconstruct_combined_latent_df(Z, experiment_data_imputed, stb_data_imputed)


# Save UMAP results
# Ensure UMAP results keep original MultiIndex (cpd_id, Library, cpd_type)

# Define the plot filename outside the function
umap_plot_file = os.path.join(output_folder, "clipn_ldim_UMAP.pdf")

# Generate UMAP without labels
umap_df = generate_umap(combined_latent_df, output_folder, umap_plot_file, args, add_labels=False)
# Generate UMAP with labels
umap_plot_file = os.path.join(output_folder, "clipn_ldim_UMAP_labels.pdf")
umap_df = generate_umap(combined_latent_df, output_folder, umap_plot_file, args, add_labels=True)


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
if isinstance(combined_latent_df.index, pd.MultiIndex):
    combined_latent_df.to_csv(os.path.join(output_folder, "CLIPn_latent_representations_with_cpd_id.csv"))
    logger.info(f"Latent representations saved successfully with MultiIndex to {combined_output_file}.")
else:
    logger.warning("Warning: MultiIndex is missing! Attempting to restore it before saving.")
    combined_latent_df.reset_index().to_csv(os.path.join(output_folder, "CLIPn_latent_representations_with_cpd_id.csv"))

    # If MultiIndex was lost, reset and restore
    if "cpd_id" in combined_latent_df.columns and "Library" in combined_latent_df.columns:
        combined_latent_df = combined_latent_df.set_index(["cpd_id", "Library"])
        combined_latent_df.to_csv(os.path.join(output_folder, "CLIPn_latent_representations_with_cpd_id.csv"))
        logger.info(f"Restored MultiIndex and saved latent representations to {combined_output_file}.")
    else:
        logger.error("Error: Could not restore MultiIndex. `cpd_id` and `Library` missing from columns!")


# Log the combined latent representation DataFrame
if combined_latent_df is not None:
    logger.info("First few rows of combined_latent_df (final merged latent space):\n" + combined_latent_df.head().to_string())



# UMAP per experiment

# Define the UMAP output file before calling function
umap_experiment_plot_file = os.path.join(output_folder, "UMAP_experiment_vs_stb.pdf")

# Call the function using the UMAP DataFrame (ensuring MultiIndex)
plot_umap_colored_by_experiment(umap_df, umap_experiment_plot_file)


###########
# Generate Summary of Closest & Farthest Compounds
# Compute pairwise distances **before** using `dist_df`
logger.info("Computing pairwise compound distances...")

# Drop 'Library' before computing distances (if it exists)
dist_df = compute_pairwise_distances(combined_latent_df.drop(columns=["Library"], errors="ignore"))

# Ensure index and columns are correctly assigned
dist_df.index = combined_latent_df.index  
dist_df.columns = combined_latent_df.index  

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


