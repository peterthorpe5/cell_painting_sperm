#!/usr/bin/env python3
# coding: utf-8

"""
Run CLIPn Integration on Cell Painting Data
-------------------------------------------

This script:
- Loads and merges multiple reference and query datasets.
- Harmonises column features across datasets.
- Encodes labels for compatibility with CLIPn.
- Runs CLIPn integration analysis (either train on references or integrate all).
- Decodes labels post-analysis, restoring original annotations.
- Outputs results, including latent representations and similarity matrices.

Command-line arguments:
-----------------------
    --datasets_csv      : Path to CSV listing dataset names and paths.
    --out               : Directory to save outputs.
    --experiment        : Experiment name for file naming.
    --mode              : Operation mode ('reference_only' or 'integrate_all').
    --clipn_param       : Optional CLIPn parameter for tuning (e.g., number of epochs).
    --latent_dim        : Dimensionality of latent space (default: 20).
    --lr                : Learning rate for CLIPn (default: 1e-5).
    --epoch             : Number of training epochs (default: 300).
    --save_model        : If set, save the trained CLIPn model after training.
    --load_model        : Path to a previously saved CLIPn model to load and reuse.
    --scaling_mode      : Scaling mode for features ('all', 'per_plate', 'none'; default: 'all').
    --scaling_method    : Scaling method ('robust' or 'standard'; default: 'robust').
    --skip_standardise  : If set, skip feature standardisation (default: False).
    --reference_names   : Comma-separated list of dataset names to use as references (only in 'reference_only' mode).
    --annotation_file   : Path to a CSV file with compound annotations (optional).
    --debug             : If set, enable debug-level logging.


Logging:
--------
Logs detailed info and debug-level outputs.
Logging is configured to output to both console and file.
Logging is set up to capture:
- Python version and command-line arguments.
- Experiment name.
- Dataset loading and harmonisation steps.
- Feature scaling and encoding steps.
- CLIPn training and integration steps.
Logging is done using the `logging` module, with separate handlers for console and file output.

"""

import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import glob
from clipn.model import CLIPn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn import set_config
import csv
import torch
import torch.serialization
from cell_painting.process_data import (

        prepare_data_for_clipn_from_df,
        run_clipn_simple,
        standardise_metadata_columns,
        project_query_to_latent
)


set_config(transform_output="pandas")
torch.serialization.add_safe_globals([CLIPn])

def setup_logging(out_dir, experiment):
    """Configure logging with stream and file handlers."""
    log_dir = Path(out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"{experiment}_clipn.log"

    logger = logging.getLogger("clipn_logger")
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Experiment Name: {experiment}")

    return logger


def detect_csv_delimiter(csv_path):
    """Detect the delimiter of a CSV file."""
    with open(csv_path, 'r', newline='') as csvfile:
        sample = csvfile.read(2048)
        csvfile.seek(0)
        if ',' in sample and '\t' not in sample:
            return ','
        elif '\t' in sample and ',' not in sample:
            return '\t'
        else:
            # default to comma if both are found or none
            return ','


def ensure_library_column(df, filepath, logger, value=None):
    """
    Add a 'Library' column to df using the filename if missing (or use a provided value).

    Parameters
    ----------
    df : pd.DataFrame
        Metadata dataframe.
    filepath : str
        File path for the source file.
    logger : logging.Logger
        Logger for status messages.
    value : str, optional
        Explicit value for the Library column. If not provided, uses filename.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Library' column ensured.
    """
    if "Library" not in df.columns:
        if value is not None:
            base_library = value
        else:
            base_library = os.path.splitext(os.path.basename(filepath))[0]
        df["Library"] = base_library
        logger.info(f"'Library' column not found. Set to: {base_library}")
    return df


def scale_features(df, feature_cols, plate_col=None, mode='all', method='robust', logger=None):
    """
    Scale features globally or per plate, using the specified method.

    Args:
        df (pd.DataFrame): DataFrame with features and metadata.
        feature_cols (list): Names of feature columns to scale.
        plate_col (str or None): Plate column name (required if mode='per_plate').
        mode (str): 'all', 'per_plate', or 'none'.
        method (str): 'robust' or 'standard'.
        logger (logging.Logger): Logger for status messages.

    Returns:
        pd.DataFrame: DataFrame with scaled features.
    """
    logger = logger or logging.getLogger("scaling")
    scaler_cls = RobustScaler if method == 'robust' else StandardScaler

    if mode == 'none':
        logger.info("No scaling applied.")
        return df

    df_scaled = df.copy()
    if mode == 'all':
        scaler = scaler_cls()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
        logger.info(f"Scaled all features together using {method} scaler.")
    elif mode == 'per_plate':
        if plate_col is None or plate_col not in df.columns:
            raise ValueError("plate_col must be provided for per_plate scaling.")
        for plate, group_idx in df.groupby(plate_col).groups.items():
            scaler = scaler_cls()
            idx = list(group_idx)
            df_scaled.loc[idx, feature_cols] = scaler.fit_transform(df.loc[idx, feature_cols])
        logger.info(f"Scaled features per plate using {method} scaler.")
    else:
        logger.warning(f"Unknown scaling mode: {mode}. No scaling applied.")
    return df_scaled


def ensure_plate_well_metadata(decoded_df: pd.DataFrame, metadata_source: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Ensure Plate_Metadata and Well_Metadata are attached to the decoded DataFrame.

    Parameters
    ----------
    decoded_df : pd.DataFrame
        DataFrame from the decoded latent space.
    metadata_source : pd.DataFrame
        Original combined metadata including Plate_Metadata and Well_Metadata.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with Plate_Metadata and Well_Metadata columns.
    """
    required_cols = ["Plate_Metadata", "Well_Metadata"]
    if all(col in decoded_df.columns for col in required_cols):
        logger.debug("Plate_Metadata and Well_Metadata already present in decoded_df.")
        return decoded_df

    logger.debug("Re-attaching Plate_Metadata and Well_Metadata to decoded_df.")
    meta_cols_extended = ["Dataset", "Sample"] + required_cols
    lookup_df = metadata_source[meta_cols_extended].drop_duplicates()

    merged_df = pd.merge(
        decoded_df,
        lookup_df,
        on=["Dataset", "Sample"],
        how="left",
        validate="many_to_one"
    )

    return merged_df

def merge_annotations(latent_df_or_path, annotation_file: str, output_prefix: str, logger: logging.Logger) -> None:
    """
    Merge compound annotations into the CLIPn latent output.

    Parameters
    ----------
    latent_df_or_path : str or pd.DataFrame
        Path to CLIPn latent space output (TSV) or a DataFrame.
    annotation_file : str
        Path to annotation file with compound information (TSV).
    output_prefix : str
        Base path prefix for output files (no extension).
    logger : logging.Logger
        Logger instance.
    """
    try:
        if isinstance(latent_df_or_path, str):
            latent_df = pd.read_csv(latent_df_or_path, sep='\t')
        else:
            latent_df = latent_df_or_path.copy()

        annot_df = pd.read_csv(annotation_file, sep=',')

        if "Plate_Metadata" not in annot_df.columns and "Plate" in annot_df.columns:
            annot_df["Plate_Metadata"] = annot_df["Plate"]
        if "Well_Metadata" not in annot_df.columns and "Well" in annot_df.columns:
            annot_df["Well_Metadata"] = annot_df["Well"]

        logger.info("Merging annotations on keys: Plate_Metadata, Well_Metadata")
        logger.info(f"Latent columns: {latent_df.columns.tolist()}")
        logger.info(f"Annotation columns: {annot_df.columns.tolist()}")
        logger.info(f"Latent shape: {latent_df.shape}, Annotation shape: {annot_df.shape}")

        if "Plate_Metadata" not in latent_df.columns or "Well_Metadata" not in latent_df.columns:
            logger.warning("Plate_Metadata or Well_Metadata missing in latent data — merge skipped.")
            return

        merged = pd.merge(
            latent_df,
            annot_df,
            on=["Plate_Metadata", "Well_Metadata"],
            how="left",
            validate="many_to_one"
        )

        logger.info(f"Merged shape: {merged.shape}")
        n_merged = merged["cpd_id"].notna().sum()
        logger.info(f"Successfully merged rows with non-null cpd_id: {n_merged}")

        merged_tsv = f"{output_prefix}_latent_with_annotations.tsv"
        merged_csv = f"{output_prefix}_latent_with_annotations.csv"

        merged.to_csv(merged_tsv, sep='\t', index=False)
        merged.to_csv(merged_csv, index=False)

        logger.info(f"Merged annotation saved to:\n- {merged_tsv}\n- {merged_csv}")

    except Exception as e:
        logger.warning(f"Annotation merging failed: {e}")

def aggregate_latent_per_compound(
    df,
    group_col="cpd_id",
    latent_cols=None,
    method="median"
):
    """
    Aggregate image-level latent vectors to a single vector per compound.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image-level latent vectors and compound identifiers.
    group_col : str
        Name of the column identifying compounds (default: "cpd_id").
    latent_cols : list of str or None
        List of latent space column names. If None, uses all integer-named columns.
    method : str
        Aggregation method ("median", "mean", "min", "max"; default: "median").

    Returns
    -------
    pandas.DataFrame
        Aggregated DataFrame with one row per compound and collapsed latent features.
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame.")

    # Auto-detect latent columns if not provided (pure integer column names)
    if latent_cols is None:
        latent_cols = [col for col in decoded_df.columns if (isinstance(col, int)) or (isinstance(col, str) and col.isdigit())]

        if not latent_cols:
            raise ValueError("No integer-named latent columns found.")

    # Ensure numeric and sorted order for columns
    latent_cols = sorted(latent_cols, key=int)

    # Group and aggregate
    aggfunc = method if method in ["mean", "median", "min", "max"] else "median"
    aggregated = df.groupby(group_col)[latent_cols].agg(aggfunc).reset_index()
    return aggregated


# this is the problem function when we loose cpd_id
# I hate this function. 
def load_single_dataset(name, path, logger, metadata_cols):
    """
    Load a single dataset, standardise metadata, and wrap it with a MultiIndex.

    Parameters
    ----------
    name : str
        Dataset name used to label the MultiIndex.
    path : str
        Path to the input CSV/TSV file.
    logger : logging.Logger
        Logger instance for status and error reporting.
    metadata_cols : list of str
        List of required metadata column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with harmonised metadata and a MultiIndex ('Dataset', 'Sample').

    Raises
    ------
    ValueError
        If any mandatory metadata column is missing after standardisation.
    """
    delimiter = detect_csv_delimiter(path)

    # Load with NO index, avoid dropping metadata
    df = pd.read_csv(path, delimiter=delimiter, index_col=None)

    if logger:
        logger.debug(f"[{name}] Columns after initial load: {df.columns.tolist()}")

    if df.index.name in metadata_cols:
        promoted_col = df.index.name
        df[promoted_col] = df.index
        df.index.name = None
        logger.warning(f"[{name}] Promoted index '{promoted_col}' to column to preserve metadata.")

    # Ensure Library column is present before standardisation
    df = ensure_library_column(df, path, logger, value=name)

    # Standardise column names (e.g., trimming whitespace)
    df = standardise_metadata_columns(df, logger=logger, dataset_name=name)

    # Check required metadata columns
    missing_cols = [col for col in metadata_cols if col not in df.columns]
    if missing_cols:
        for col in missing_cols:
            logger.error(f"[{name}] Mandatory column '{col}' missing after standardisation.")
        raise ValueError(f"[{name}] Mandatory column(s) {missing_cols} missing after standardisation.")

    # Reset index to ensure clean MultiIndex
    df = df.reset_index(drop=True)

    # Wrap with MultiIndex
    df.index = pd.MultiIndex.from_frame(
        pd.DataFrame({"Dataset": name, "Sample": range(len(df))})
    )

    if logger:
        logger.debug(f"[{name}] Final columns: {df.columns.tolist()}")
        logger.debug(f"[{name}] Final shape: {df.shape}")
        logger.debug(f"[{name}] Final index names: {df.index.names}")

    return df



def harmonise_numeric_columns(dataframes, logger):
    numeric_cols_sets = [set(df.select_dtypes(include=[np.number]).columns) for df in dataframes.values()]
    common_cols = sorted(set.intersection(*numeric_cols_sets))
    logger.info(f"Harmonised numeric columns across datasets: {len(common_cols)}")

    metadata_cols = ["cpd_id", "cpd_type", "Library"]
    for name, df in dataframes.items():
        numeric_df = df[common_cols]
        metadata_df = df[metadata_cols]
        df_harmonised = pd.concat([numeric_df, metadata_df], axis=1)
        assert df_harmonised.index.equals(df.index), f"Index mismatch after harmonisation in '{name}'."
        dataframes[name] = df_harmonised
        logger.debug(f"[{name}] Harmonisation successful, final columns: {df_harmonised.columns.tolist()}")

    return dataframes, common_cols




def apply_harmonisation(dataframes, common_cols, metadata_cols, logger):
    """Subset numeric columns and explicitly preserve metadata."""
    harmonised_dataframes = {}
    for name, df in dataframes.items():
        numeric_df = df[common_cols]
        metadata_df = df[metadata_cols]

        harmonised_df = pd.concat([numeric_df, metadata_df], axis=1)

        assert metadata_df.index.equals(harmonised_df.index), f"Metadata indices misaligned for '{name}'"
        logger.info(f"Sanity check passed for '{name}'.")

        harmonised_dataframes[name] = harmonised_df

    return harmonised_dataframes


def load_and_harmonise_datasets(datasets_csv, logger, mode=None):
    delimiter = detect_csv_delimiter(datasets_csv)
    datasets_df = pd.read_csv(datasets_csv, delimiter=delimiter)
    dataset_paths = datasets_df.set_index('dataset')['path'].to_dict()

    metadata_cols = ["cpd_id", "cpd_type", "Library"]
    dataframes = {}

    logger.info("Loading datasets individually")
    for name, path in dataset_paths.items():
        try:
            dataframes[name] = load_single_dataset(name, path, logger, metadata_cols)
        except ValueError as e:
            logger.error(f"Loading dataset '{name}' failed: {e}")
            raise

    return harmonise_numeric_columns(dataframes, logger)


def standardise_numeric_columns_preserving_metadata(df: pd.DataFrame, meta_columns: list[str]) -> pd.DataFrame:
    """
    Standardise numeric columns in a MultiIndex DataFrame per dataset,
    preserving metadata columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with MultiIndex (Dataset, Sample).
    meta_columns : list of str
        List of columns to exclude from standardisation.

    Returns
    -------
    pd.DataFrame
        Standardised DataFrame with numeric features scaled per dataset
        PER DATASET!!!!!!!
        and metadata columns preserved.
    """
    if not isinstance(df.index, pd.MultiIndex) or 'Dataset' not in df.index.names:
        raise ValueError("Input DataFrame must have a MultiIndex with level 'Dataset'.")

    scaled_frames = []
    for dataset, group in df.groupby(level="Dataset"):
        meta = group[meta_columns]
        numeric = group.drop(columns=meta_columns)

        scaler = StandardScaler()
        numeric_scaled = pd.DataFrame(
            scaler.fit_transform(numeric),
            columns=numeric.columns,
            index=group.index
        )

        scaled_group = pd.concat([numeric_scaled, meta], axis=1)
        scaled_frames.append(scaled_group)

    df_scaled_all = pd.concat(scaled_frames).sort_index()
    return df_scaled_all


def decode_labels(df, encoders, logger):
    """
    Decode categorical columns in a DataFrame to original labels using LabelEncoders.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose columns may be label-encoded.
    encoders : dict
        Mapping of column names to fitted LabelEncoder objects.
    logger : logging.Logger
        Logger for progress and warnings.

    Returns
    -------
    pandas.DataFrame
        DataFrame with decoded columns where possible.
    """
    for col, le in encoders.items():
        if col not in df.columns:
            logger.warning(f"decode_labels: Column '{col}' not found in DataFrame. Skipping.")
            continue

        if df[col].isna().all():
            logger.warning(f"decode_labels: Column '{col}' is all-NaN. Skipping decode.")
            continue

        # Only attempt decoding if dtype is integer (not string/categorical)
        if not np.issubdtype(df[col].dtype, np.integer):
            logger.info(f"decode_labels: Column '{col}' is not integer-encoded. Skipping decode.")
            continue

        try:
            mask_notna = df[col].notna()
            decoded_vals = df[col].copy()
            decoded_vals.loc[mask_notna] = le.inverse_transform(df.loc[mask_notna, col].astype(int))
            df[col] = decoded_vals
            logger.info(f"decode_labels: Decoded column '{col}'.")
        except Exception as e:
            logger.warning(
                f"decode_labels: Could not decode column '{col}': {e}. "
                "May be due to unseen labels, type errors, or missing encoder classes."
            )
    return df


def encode_labels(df, logger):
    """
    Encode categorical columns (excluding 'cpd_id') using LabelEncoder.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to encode.
    logger : logging.Logger
        Logger for debug information.

    Returns
    -------
    df : pd.DataFrame
        Encoded DataFrame.
    encoders : dict
        Mapping of column names to LabelEncoders.
    """
    encoders = {}
    skip_columns = {"cpd_id"}  # Do not encode cpd_id
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col in skip_columns:
            logger.debug(f"Skipping encoding for {col}")
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        logger.debug(f"Encoded column {col}")
    return df, encoders


def extend_model_encoders(model, new_keys, reference_key, logger):
    """
    Extend CLIPn model's encoder mapping for new datasets using a reference encoder.

    Parameters
    ----------
    model : CLIPn
        The trained CLIPn model object.
    new_keys : iterable
        Keys of new datasets to be projected.
    reference_key : int
        Key of the reference dataset to copy the encoder from.
    logger : logging.Logger
        Logger instance for debug messages.
    """
    for new_key in new_keys:
        model.model.encoders[new_key] = model.model.encoders[reference_key]
        logger.debug(f"Assigned encoder for dataset key {new_key} using reference encoder {reference_key}")



def run_clipn_integration(df, logger, clipn_param, output_path, experiment, mode, 
                          latent_dim, lr, epochs, skip_standardise=False):
    """
    Train CLIPn model on input data and return latent space.

    Parameters
    ----------
    df : pd.DataFrame
        Combined input dataframe (MultiIndex: Dataset, Sample).
    logger : logging.Logger
        Logging instance.
    clipn_param : str
        Optional parameter for tuning.
    output_path : str or Path
        Directory to save latent data.
    latent_dim : int
        Dimensionality of latent space.
    lr : float
        Learning rate.
    epochs : int
        Training epochs.

    Returns
    -------
    latent_df : pd.DataFrame
        Combined latent DataFrame with MultiIndex.
    cpd_ids : dict
        Dictionary of compound IDs by original dataset name.
    model : CLIPn
        Trained CLIPn model instance.
    dataset_key_mapping : dict
        Mapping of integer dataset index to original dataset name.
    """
    logger.info(f"Running CLIPn integration with param: {clipn_param}")
    meta_cols = ["cpd_id", "cpd_type", "Library"]
    
    logger.info("==== DEBUG: Columns in combined_df ====")
    logger.info(df.columns.tolist())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("==== DEBUG: Numeric columns in combined_df ====")
    logger.info(numeric_cols)
    logger.info(f"Combined DataFrame shape: {df.shape}")

    logger.info("==== DEBUG: First few rows of combined_df ====")
    logger.info("\n" + str(df.head()))

    # If using 'common_cols' from harmonisation, log that too
    try:
        logger.info("==== DEBUG: Common numeric columns after harmonisation ====")
        logger.info(common_cols)
    except Exception:
        pass

    if not numeric_cols:
        logger.error(
            "No numeric feature columns found for scaling after harmonisation. "
            "Check your input files and feature harmonisation step. "
            "Possible causes: no overlap of features, all numeric columns are NaN, or wrong column dtypes."
        )
        raise ValueError("No numeric columns available for scaling in combined_df.")
    # --- END DEBUG BLOCK --


    data_dict, label_dict, label_mappings, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(df_scaled)

    latent_dict, model, loss = run_clipn_simple(data_dict, label_dict, latent_dim=latent_dim, lr=lr, epochs=epochs)
    if isinstance(loss, (list, np.ndarray)):
        logger.info(f"CLIPn final loss: {loss[-1]:.6f}")
    else:
        logger.info(f"CLIPn loss: {loss}")


    latent_frames = []
    for i, latent in latent_dict.items():
        name = dataset_key_mapping[i]
        df_latent = pd.DataFrame(latent)
        df_latent.index = pd.MultiIndex.from_product([[name], range(len(df_latent))], names=["Dataset", "Sample"])
        latent_frames.append(df_latent)

    latent_combined = pd.concat(latent_frames)

    latent_file = Path(output_path) / f"{experiment}_{mode}_CLIPn_latent_representations.npz"
    # Save using string keys to comply with np.savez requirements
    latent_dict_str_keys = {str(k): v for k, v in latent_dict.items()}
    np.savez(latent_file, **latent_dict_str_keys)

    logger.info(f"Latent representations saved to: {latent_file}")

    latent_file_id = Path(output_path) / f"{experiment}_{mode}_CLIPn_latent_representations_cpd_id.npz"
    # Convert latent space and cpd_ids into savable format
    latent_dict_str_keys = {str(k): v for k, v in latent_dict.items()}

    # Also store cpd_ids dictionary as arrays
    cpd_ids_array = {f"cpd_ids_{k}": np.array(v) for k, v in cpd_ids.items()}

    # Combine and save
    np.savez(latent_file_id, **latent_dict_str_keys, **cpd_ids_array)

    post_clipn_dir = Path(args.out) / "post_clipn"
    post_clipn_dir.mkdir(parents=True, exist_ok=True)

    latent_file = post_clipn_dir / f"{experiment}_{mode}_CLIPn_latent_representations.npz"
    np.savez(latent_file, **latent_dict_str_keys)



    return latent_combined, cpd_ids, model, dataset_key_mapping



def main(args):
    """Main function to execute CLIPn integration pipeline."""
    logger = setup_logging(args.out, args.experiment)

    logger.info("Starting CLIPn integration pipeline")
    mapping_dir = Path(args.out) / "post_clipn"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    post_clipn_dir = Path(args.out) / "post_clipn"
    post_clipn_dir.mkdir(parents=True, exist_ok=True)


    dataframes, common_cols = load_and_harmonise_datasets(args.datasets_csv, logger, mode=args.mode)
    logger.info("NOTE this script does not perfom any DMSO normalisation, you must do this before running this script. if you want it..")

    # Sanity check here:
    for name, df in dataframes.items():
        missing_meta = [col for col in ["cpd_id", "cpd_type", "Library"] if col not in df.columns]
        if missing_meta:
            raise ValueError(f"Sanity check failed after harmonisation for '{name}': Missing {missing_meta}")
        logger.info(f"Sanity check passed for '{name}'.")
        all_have_multiindex = all(isinstance(df.index, pd.MultiIndex) for df in dataframes.values())
        combined_df = pd.concat(dataframes.values())
        for name, df in dataframes.items():
            if not isinstance(df.index, pd.MultiIndex):
                logger.warning(f"[{name}] Expected MultiIndex but found {type(df.index)}")



        logger.debug(f"Columns at this stage, combined: {combined_df.columns.tolist()}")
    
    # -------------------------------------------------------------
    # After merging and harmonising all DataFrames into combined_df:
    # -------------------------------------------------------------

    # Define your metadata columns (do not scale these!)
    meta_columns = ["cpd_id", "cpd_type", "Plate_Metadata", "Well_Metadata", "Library"]
    # Ensure metadata columns are present in combined_df
    for col in meta_columns:
        if col not in combined_df.columns:
            raise ValueError(f"Metadata column '{col}' not found in combined DataFrame after harmonisation.")
    logger.info(f"Metadata columns present in combined DataFrame: {meta_columns}")
    logger.info(f"Combined DataFrame shape after harmonisation: {combined_df.shape}")

    # Determine numeric feature columns to scale
    feature_cols = [
        col for col in combined_df.columns
        if col not in meta_columns and pd.api.types.is_numeric_dtype(combined_df[col])
    ]

    # Perform scaling if requested (default: robust scaling of all data together)
    # Perform scaling unless skip_standardise is set
    if args.skip_standardise:
        logger.info("Skipping feature scaling (--skip_standardise set).")
    else:
        if args.scaling_mode != 'none':
            logger.info(f"Scaling numeric features using mode='{args.scaling_mode}', method='{args.scaling_method}'")
            combined_df = scale_features(
                combined_df,
                feature_cols=feature_cols,
                plate_col="Plate_Metadata",
                mode=args.scaling_mode,
                method=args.scaling_method,
                logger=logger
            )
            logger.info(f"Scaled columns: {feature_cols}")
            logger.info(f"Scaled combined DataFrame shape: {combined_df.shape}")
        else:
            logger.info("No scaling applied to numeric features (--scaling_mode=none)")


    # -------------------------------------------------------------
    #  Proceed to label encoding
    # -------------------------------------------------------------
    logger.info("Encoding categorical labels for CLIPn compatibility")

    combined_df, encoders = encode_labels(combined_df, logger)
    metadata_df = decode_labels(combined_df.copy(), encoders, logger)[["cpd_id", "cpd_type", "Library"]]
    metadata_df = metadata_df.reset_index()


    if args.mode == "reference_only":
        # Define exactly which dataset to train on
        reference_names = args.reference_names
        logger.info(f"Using reference datasets {reference_names} for training, projecting others onto shared latent space.")


        if not set(reference_names).issubset(dataframes):
            raise ValueError(f"Reference dataset(s) {reference_names} not found in input datasets.")


        # All others will be treated as projection targets
        query_names = [name for name in dataframes if name not in reference_names]


        logger.info(f"CLIPn training on: {reference_names} ({len(reference_names)} datasets)")
        logger.info(f"CLIPn projecting onto reference latent space: {query_names} ({len(query_names)} datasets)")
        logger.debug(f"All dataset names: {list(dataframes.keys())}")



        reference_df = combined_df.loc[reference_names]
        query_df = combined_df.loc[query_names]


        logger.info(f"Training CLIPn on references: {reference_names}")

        # -------------------------------------------------------------
        #  loading pre-trained model or training new one
        # -------------------------------------------------------------
        # Check if we are loading a pre-trained model
        if args.load_model and args.mode != "integrate_all":
            logger.warning("Loading pre-trained model is only supported in 'integrate_all' mode. "
                           "Switching to 'integrate_all' mode for loading.")
            args.mode = "integrate_all"

        # Check for model loading
        if args.load_model:
            model_files = glob.glob(args.load_model)
            if not model_files:
                raise FileNotFoundError(f"No model files matched pattern: {args.load_model}")
            model_path = model_files[0]
            logger.info(f"Loading pre-trained CLIPn model from cmd line: {model_path}")
            logger.info(f"Loading pre-trained CLIPn model from actual file: {model_path}")
            model = torch.load(model_path, weights_only=False)

            # Still need to prepare data (e.g. scaling and projection input)
            meta_columns = ["cpd_id", "cpd_type", "Plate_Metadata", "Well_Metadata", "Library"]
            feature_cols = [col for col in reference_df.columns if col not in meta_columns and pd.api.types.is_numeric_dtype(reference_df[col])]
            
            if args.skip_standardise:
                logger.info("Skipping feature scaling (--skip_standardise set).")
            else:
                if args.scaling_mode != 'none':
                    logger.info(f"Scaling numeric features using mode='{args.scaling_mode}', method='{args.scaling_method}'")
                    combined_df = scale_features(
                        combined_df,
                        feature_cols=feature_cols,
                        plate_col="Plate_Metadata",
                        mode=args.scaling_mode,
                        method=args.scaling_method,
                        logger=logger
                    )
                    logger.info(f"Scaled columns: {feature_cols}")
                    logger.info(f"Scaled combined DataFrame shape: {combined_df.shape}")
                else:
                    logger.info("No scaling applied to numeric features (--scaling_mode=none)")

                
            df_scaled = scale_features(
                reference_df,
                feature_cols=feature_cols,
                plate_col="Plate_Metadata",
                mode=args.scaling_mode,
                method=args.scaling_method,
                logger=logger)
            logger.info(f"Scaled columns: {feature_cols}")

            logger.info(f"Scaled reference data shape: {df_scaled.shape}")
            
            # Prepare data for CLIPn
            logger.info("Preparing data for CLIPn integration from scaled DataFrame")
            # Prepare data for CLIPn
            data_dict, label_dict, label_mappings, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(df_scaled)



            # Generate latent space without re-training
            latent_dict = model.predict(data_dict)

            latent_frames = []
            for i, latent in latent_dict.items():
                name = dataset_key_mapping[i]
                df_latent = pd.DataFrame(latent)
                df_latent.index = pd.MultiIndex.from_product([[name], range(len(df_latent))], names=["Dataset", "Sample"])
                latent_frames.append(df_latent)

            latent_df = pd.concat(latent_frames)

        else:
            # Train as before
            logger.info("Training new CLIPn model on reference datasets")
            logger.debug(f"Reference DataFrame shape: {reference_df.shape}")
            latent_df, cpd_ids, model, dataset_key_mapping = run_clipn_integration(reference_df,
                                                                                    logger,
                                                                                    args.clipn_param,
                                                                                    args.out,
                                                                                    args.experiment,
                                                                                    args.mode,
                                                                                    args.latent_dim,
                                                                                    args.lr,
                                                                                    args.epoch,
                                                                                    skip_standardise=args.skip_standardise
                                                                                    )
            
            if args.save_model:
                model_path = Path(args.out) / f"{args.experiment}_clipn_model.pt"
                torch.save(model, model_path)
                logger.info(f"Trained CLIPn model saved to: {model_path}")



        latent_training_df = latent_df.reset_index()
        latent_training_df = latent_training_df[latent_training_df['Dataset'].isin(reference_names)]

        # nice coding guys!
        logger.debug(f"Model encoders keys after training: {list(model.model.encoders.keys())}")



        training_metadata_df = metadata_df[metadata_df['Dataset'].isin(reference_names)]

        # Add cpd_id from cpd_ids dict explicitly
        latent_training_df = latent_training_df.merge(training_metadata_df, on=["Dataset", "Sample"], how="left")

       
        training_output_path = Path(args.out) / "training"
        training_output_path.mkdir(parents=True, exist_ok=True)
        # Debug output
        logger.debug(f"cpd_ids keys: {list(cpd_ids.keys())}")
        logger.debug(f"Example row: {latent_training_df.iloc[0].to_dict()}")
        assert all(name in cpd_ids for name in latent_training_df["Dataset"].unique()), "Missing cpd_id mappings for some datasets"

        # Correctly inject cpd_id using the known dictionary
        latent_training_df["cpd_id"] = latent_training_df.apply(
            lambda row: cpd_ids.get(row["Dataset"], [None])[row["Sample"]]
            if row["Sample"] < len(cpd_ids.get(row["Dataset"], [])) else None,
            axis=1
        )

        latent_training_df.to_csv(training_output_path / "training_only_latent.csv", index=False)

        logger.debug("First 10 cpd_id values:\n%s", latent_training_df["cpd_id"].head(10).to_string(index=False))
        logger.debug("Unique cpd_id values (first 10): %s", latent_training_df["cpd_id"].unique()[:10])


        if not query_df.empty:
            logger.info(f"Projecting query datasets onto reference latent space: {query_names}")
            logger.debug(f"Query DataFrame shape: {query_df.shape}")

            # Prepare query data
            query_data_dict, _, _, query_cpd_ids, query_key_map = prepare_data_for_clipn_from_df(query_df)


            # Extend dataset_key_mapping to include query datasets
            max_existing_key = max(dataset_key_mapping.keys(), default=-1)
            new_keys = range(max_existing_key + 1, max_existing_key + 1 + len(query_names))

            for new_key, name in zip(new_keys, query_names):
                dataset_key_mapping[new_key] = name
            
            # Extend model.encoders with identity mappings for query datasets
            try:
                reference_encoder_key = next(
                    k for k, v in dataset_key_mapping.items()
                    if v in reference_names and k in model.model.encoders
                )
            except StopIteration:
                logger.error("No valid reference_encoder_key found. None of the reference datasets matched trained encoders.")
                raise


            extend_model_encoders(model, new_keys, reference_encoder_key, logger)



            # Invert after adding new keys
            dataset_key_mapping_inv = {v: k for k, v in dataset_key_mapping.items()}

            query_groups = query_df.groupby(level="Dataset")
            query_data_dict_corrected = {
                dataset_key_mapping_inv[name]: group.droplevel("Dataset").drop(columns=["cpd_id", "cpd_type", "Library"]).values
                for name, group in query_groups
                if name in dataset_key_mapping_inv
            }



            # Predict using model
            projected_dict = model.predict(query_data_dict_corrected)
            if not projected_dict:
                logger.warning("model.predict() returned an empty dictionary. Check dataset keys and input formatting.")
            else:
                logger.debug(f"Projected {len(projected_dict)} datasets into latent space: {list(projected_dict.keys())}")

                        # Rebuild latent_query_df and update query_cpd_ids
            projected_frames = []
            query_cpd_ids = {}
            # Rebuild latent_query_df and update query_cpd_ids
            for i, latent in projected_dict.items():
                name = dataset_key_mapping[i]
                df_proj = pd.DataFrame(latent)
                df_proj.index = pd.MultiIndex.from_product([[name], range(len(df_proj))], names=["Dataset", "Sample"])
                projected_frames.append(df_proj)

                query_cpd_ids[name] = query_df.loc[name]["cpd_id"].tolist()

            latent_query_df = pd.concat(projected_frames)


            # Now to reset index and assign cpd_id
            latent_query_df = latent_query_df.reset_index()
            latent_query_df["cpd_id"] = latent_query_df.apply(
                lambda row: query_cpd_ids.get(row["Dataset"], [None])[row["Sample"]],
                axis=1
            )

            logger.debug(f"Number of unique cpd_id in query: {latent_query_df['cpd_id'].nunique()}")

            # Save file
            query_output_path = Path(args.out) / "query_only" / f"{args.experiment}_query_only_latent.csv"
            query_output_path.parent.mkdir(parents=True, exist_ok=True)
            latent_query_df.to_csv(query_output_path, index=False)
            logger.info(f"Query-only latent data saved to {query_output_path}")
            logger.info(f"Total query samples projected: {latent_query_df.shape[0]}")

            # Add to final merged latent
            latent_df = pd.concat([latent_df, latent_query_df.set_index(["Dataset", "Sample"])])
            cpd_ids.update(query_cpd_ids)

    else:
        # run on all data, not projected onto the reference ...
        # but the reference is included in all the training
        logger.info("Training and integrating CLIPn on all datasets")

        if args.load_model:
            from clipn.model import CLIPn  
            torch.serialization.add_safe_globals([CLIPn])
            logger.info(f"Loading pre-trained CLIPn model from: {args.load_model}")
            model_files = glob.glob(args.load_model)
            if not model_files:
                raise FileNotFoundError(f"No model files matched pattern: {args.load_model}")
            model_path = model_files[0]

            logger.info(f"Loading pre-trained CLIPn model from: {model_path}")
            model = torch.load(model_path, weights_only=False)

            # Standardise and prepare input data
            df_scaled = standardise_numeric_columns_preserving_metadata(combined_df, meta_columns=["cpd_id", "cpd_type", "Library"])
            data_dict, _, _, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(df_scaled)

            # Predict latent space using loaded model
            latent_dict = model.predict(data_dict)

            latent_frames = []
            for i, latent in latent_dict.items():
                name = dataset_key_mapping[i]
                df_latent = pd.DataFrame(latent)
                df_latent.index = pd.MultiIndex.from_product([[name], range(len(df_latent))], names=["Dataset", "Sample"])
                latent_frames.append(df_latent)

            latent_df = pd.concat(latent_frames)

        else:
            logger.info("Training and integrating CLIPn on all datasets")
            latent_df, cpd_ids, model, \
                dataset_key_mapping = run_clipn_integration(combined_df, 
                                                    logger, 
                                                    args.clipn_param, 
                                                    args.out, 
                                                    args.experiment,
                                                    args.mode,
                                                    args.latent_dim, 
                                                    args.lr, args.epoch)
            
            if args.save_model:
                model_path = Path(args.out) / f"{args.experiment}_clipn_model.pt"
                torch.save(model, model_path)
                logger.info(f"Trained CLIPn model saved to: {model_path}")

    # Reset index and join metadata
    latent_df = latent_df.reset_index()
    latent_df = pd.merge(latent_df, metadata_df, on=["Dataset", "Sample"], how="left")

    # Decode labels
    decoded_df = decode_labels(latent_df.copy(), encoders, logger)

    # Clean up any duplicate cpd_id columns (after merge)
    if "cpd_id_x" in decoded_df.columns or "cpd_id_y" in decoded_df.columns:
        decoded_df["cpd_id"] = (
            decoded_df.get("cpd_id_x", pd.Series(dtype=object))
            .combine_first(decoded_df.get("cpd_id_y", pd.Series(dtype=object)))
            .combine_first(decoded_df.get("cpd_id", pd.Series(dtype=object)))
        )
        cols_to_drop = [col for col in ["cpd_id_x", "cpd_id_y"] if col in decoded_df.columns]
        decoded_df = decoded_df.drop(columns=cols_to_drop)

    # Drop rows missing cpd_id (optional, but usually sensible!)
    n_before = decoded_df.shape[0]
    decoded_df = decoded_df[decoded_df["cpd_id"].notna()]
    n_after = decoded_df.shape[0]
    if n_before != n_after:
        logger.warning(f"Dropped {n_before - n_after} rows with missing cpd_id after decoding/merge.")

    # Save decoded (image-level) file in both locations
    main_decoded_path = Path(args.out) / f"{args.experiment}_decoded.csv"
    decoded_df.to_csv(main_decoded_path, index=False)
    logger.info(f"Decoded data saved to {main_decoded_path}")

    post_clipn_dir = Path(args.out) / "post_clipn"
    post_clipn_dir.mkdir(parents=True, exist_ok=True)
    post_clipn_decoded_path = post_clipn_dir / f"{args.experiment}_decoded.csv"
    decoded_df.to_csv(post_clipn_decoded_path, index=False)
    logger.info(f"Decoded data saved to {post_clipn_decoded_path}")

    # AGGREGATE TO COMPOUND-LEVEL HERE
    if getattr(args, "aggregate_method", None):
        latent_cols = [col for col in decoded_df.columns if (isinstance(col, int)) or (isinstance(col, str) and col.isdigit())]
        if not latent_cols:
            logger.error("No latent feature columns found for aggregation. Check column names.")
            raise ValueError("No latent feature columns found for aggregation.")
        df_compound = (
            decoded_df
            .groupby("cpd_id")[latent_cols]
            .agg(args.aggregate_method)
            .reset_index()
        )
        # Add cpd_type/Library by first-observed value (if present)
        for col in ["cpd_type", "Library"]:
            if col in decoded_df.columns:
                first_vals = decoded_df.groupby("cpd_id")[col].first().reset_index()
                df_compound = pd.merge(df_compound, first_vals, on="cpd_id", how="left")
        agg_path = post_clipn_dir / f"{args.experiment}_CLIPn_latent_aggregated_{args.aggregate_method}.tsv"
        df_compound.to_csv(agg_path, sep="\t", index=False)
        logger.info(f"Aggregated latent space saved to: {agg_path}")

    # Save Plate/Well lookup if present
    if "Plate_Metadata" in decoded_df.columns and "Well_Metadata" in decoded_df.columns:
        plate_well_df = decoded_df[["Dataset", "Sample", "cpd_id", "Plate_Metadata", "Well_Metadata"]].copy()
        plate_well_file = post_clipn_dir / f"{args.experiment}_latent_plate_well_lookup.tsv"
        plate_well_df.to_csv(plate_well_file, sep="\t", index=False)
        logger.info(f"Saved Plate/Well metadata to: {plate_well_file}")
    else:
        logger.warning("Plate_Metadata or Well_Metadata missing in decoded output — skipping plate/well export.")

    # ---- Annotation merge (optional) ----
    if args.annotations:
        logger.info(f"Merging annotations from: {args.annotations}")
        annot_merge_df = decoded_df.copy()
        # If missing Plate_Metadata/Well_Metadata, reconstruct from combined_df
        for col in ["Plate_Metadata", "Well_Metadata"]:
            if col not in annot_merge_df.columns and col in combined_df.columns:
                annot_merge_df = pd.merge(
                    annot_merge_df,
                    combined_df[["Dataset", "Sample", col]].drop_duplicates(),
                    on=["Dataset", "Sample"],
                    how="left"
                )
        merge_annotations(
            latent_df_or_path=annot_merge_df,
            annotation_file=args.annotations,
            output_prefix=str(post_clipn_dir / args.experiment),
            logger=logger
        )

    # ---- Generate combined label mapping from decoded data ----
    try:
        label_mapping_combined = decoded_df[["Dataset", "Sample", "cpd_type"]].copy()
        label_mapping_wide = (
            label_mapping_combined
            .pivot(index="Sample", columns="Dataset", values="cpd_type")
            .T
        )
        label_mapping_wide.to_csv(post_clipn_dir / "label_mappings.tsv", sep="\t")
        logger.info("Saved label mappings to post_clipn/label_mappings.tsv")
    except Exception as e:
        logger.warning(f"Failed to generate label mapping CSV: {e}")

    # ---- Save label encoder mappings ----
    try:
        mapping_dir = Path(args.out)
        mapping_dir.mkdir(parents=True, exist_ok=True)
        for column, encoder in encoders.items():
            mapping_path = mapping_dir / f"label_mapping_{column}.csv"
            mapping_df = pd.DataFrame({
                column: encoder.classes_,
                f"{column}_encoded": range(len(encoder.classes_))
            })
            mapping_df.to_csv(mapping_path, index=False)
            logger.info(f"Saved label mapping for {column} to {mapping_path}")
        logger.info("clipn integration completed. If this ran .. there is a god")
    except Exception as e:
        logger.warning(f"Failed to save label encoder mappings: {e}")

    logger.info(f"Columns at this stage, encoded: {combined_df.columns.tolist()}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIPn Integration.")
    parser.add_argument("--datasets_csv", required=True, help="CSV listing dataset names and paths.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--experiment", required=True, help="Experiment name.")
    
    parser.add_argument('--scaling_mode', choices=['all', 'per_plate', 'none'], default='all',
            help="How to scale features: 'all' (default, scale all data together), 'per_plate' (scale within each plate), or 'none' (no scaling).")
    parser.add_argument('--scaling_method', choices=['robust', 'standard'], default='robust',
            help="Scaler to use: 'robust' (default) or 'standard'.")
    parser.add_argument("--mode", choices=['reference_only', 'integrate_all'], required=True,
                        help="Mode of CLIPn operation.")
    parser.add_argument("--clipn_param", type=str, default="default",
                        help="Optional CLIPn parameter for tuning (e.g., number of epochs).")
    parser.add_argument("--latent_dim", type=int, default=20,
                        help="Dimensionality of latent space (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for CLIPn (default: 1e-5)")
    parser.add_argument("--epoch", type=int, default=500,
                        help="Number of training epochs (default: 500)")
    parser.add_argument("--save_model",
                        action="store_true",
                        help="If set, save the trained CLIPn model after training.")

    parser.add_argument("--load_model",
                        type=str,
                        default=None,
                        help="Path to a previously saved CLIPn model (for skipping training and directly projecting query datasets).")

   
    parser.add_argument("--reference_names", nargs='+', default=["reference1", "reference2"],
                    help="List of dataset names to use for training the CLIPn model.")
    parser.add_argument(
                        '--aggregate_method',
                        choices=['median', 'mean', 'min', 'max'],
                        default='median',
                        help='How to aggregate image-level latent space to compound-level (default: median).'
                    )
    parser.add_argument("--skip_standardise", action="store_true", 
                        help="Skip standardising numeric columns if already scaled.")

    parser.add_argument("--annotations",
                        type=str,
                        default=None,
                        help="Optional annotation file (TSV) to merge using Plate_Metadata and Well_Metadata.")


    args = parser.parse_args()
    main(args)