#!/usr/bin/env python
# coding: utf-8

"""
library of data processing modules. 
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
logger = logging.getLogger(__name__)




##################################################################
# functions
def objective(trial, X, y):
    """
    Objective function for Optuna hyperparameter tuning of CLIPn.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.
    X : dict
        Dictionary of dataset inputs (e.g., {0: stb_data, 1: experiment_data}).
    y : dict
        Dictionary of dataset labels matching the structure of `X`.

    Returns
    -------
    float
        Validation loss or score to be minimised.
    """
    latent_dim = trial.suggest_int("latent_dim", 10, 60)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 200, 500)

    logger.info(f"Trying CLIPn with latent_dim={latent_dim}, lr={lr:.6f}, epochs={epochs}")
    for dataset_id in X:
        logger.info(f"Dataset {dataset_id}: X shape = {X[dataset_id].shape}, y shape = {y[dataset_id].shape}")


    clipn_model = CLIPn(X, y, latent_dim=latent_dim)
    loss = clipn_model.fit(X, y, lr=lr, epochs=epochs)
    # Return final loss if it's a list or array, otherwise assume scalar
    if isinstance(loss, (list, tuple, np.ndarray)):
        return loss[-1]
    else:
        return loss


def optimise_clipn(X, y, n_trials=40):
    """
    Runs Optuna Bayesian optimisation to tune CLIPn hyperparameters.

    Parameters
    ----------
    X : dict
        Dictionary of dataset inputs.

    y : dict
        Dictionary of dataset labels.

    n_trials : int, optional
        Number of optimisation trials (default: 20).

    Returns
    -------
    dict
        Best hyperparameter set found.
    """
    logger.info("Starting Bayesian optimisation with %d trials.", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    logger.info(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params



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
    # Define columns to drop based on known noise patterns
    filter_cols = df.columns.str.contains(
        r"Source_Plate_Barcode|COMPOUND_NUMBER|Notes|Seahorse_alert|Treatment|Number|"
        r"Child|Paren|Location_[XYZ]|ZernikePhase|Euler|Plate|Well|Field|Center_[XYZ]|"
        r"no_|fn_"
    )
    df = df.loc[:, ~filter_cols]
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


def align_features_and_labels(X, y):
    """
    Ensures that features and labels for each dataset ID are aligned in length.

    Parameters
    ----------
    X : dict
        Feature arrays for each dataset index.
    y : dict
        Label arrays for each dataset index.

    Returns
    -------
    tuple
        Aligned versions of X and y.
    """
    X_aligned, y_aligned = {}, {}

    for k in X:
        x_len = X[k].shape[0]
        y_len = len(y[k])
        if x_len != y_len:
            logger.warning(f"Dataset {k}: Length mismatch (X: {x_len}, y: {y_len}). Truncating to min length.")
            min_len = min(x_len, y_len)
            X_aligned[k] = X[k][:min_len]
            y_aligned[k] = y[k][:min_len]
        else:
            X_aligned[k] = X[k]
            y_aligned[k] = y[k]

    return X_aligned, y_aligned


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
        best_params = optimise_clipn(X,y, n_trials=40)  # Bayesian Optimization

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

