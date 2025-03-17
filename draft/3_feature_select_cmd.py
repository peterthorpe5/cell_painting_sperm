#!/usr/bin/env python
# coding: utf-8

"""
Sperm Cell Painting Data Feature Selection & Cleaning
-----------------------------------------------------
This script processes and cleans Sperm Cell Painting (SCP) data by:
    - Importing annotation files for all libraries included on plates.
    - Handling missing values using median or KNN imputation.
    - Performing feature selection based on variance thresholding.
    - Filtering out highly correlated features based on a user-defined threshold.
    - Preparing data for downstream analysis and visualisation.
    - Logging all operations and key data snapshots for easier debugging.

Command-Line Arguments:
-----------------------
    --experiment_name          : (str) Name of the experiment. Used as a prefix for output files. Default: "SCP".
    --input_dir                : (str) Path to the directory containing input CSV files.
                                 Default: "/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/IXM_data/MCP09_ext_3751/Processed_data/".
    --output_dir               : (str) Path to the directory where outputs will be saved.
                                 Default: "{experiment_name}_feature_selection".
    --impute_method            : (str) Method for imputing missing values. Options: "median", "knn". Default: "knn".
    --knn_neighbors            : (int) Number of neighbors for KNN imputation. Default: 5.
    --correlation_threshold    : (float) Threshold for removing highly correlated features. Default: 0.99.
    --annotation_file          : (str) Path to the annotation CSV file.
                                 Default: "/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/04022025/KVP_4Plates_04022025.csv".

Example Usage:
--------------
    python feature_selection.py --experiment_name "TestExp" \
                                --input_dir "/path/to/input/" \
                                --output_dir "/path/to/output/" \
                                --impute_method "knn" \
                                --knn_neighbors 5 \
                                --correlation_threshold 0.98 \
                                --annotation_file "/path/to/annotations.csv"

Logging:
--------
The script logs key details, including system information, command-line arguments, dataset shapes, and processing steps.
The log file is saved in the output directory as "{experiment_name}_feature_selection.log".

"""


import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold


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

VERSION = "cell painting: feature selection: v0.0.1"
if "--version" in sys.argv:
    print(VERSION)
    sys.exit(1)

# Configure logging
logger = logging.getLogger("feature_selection")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def variance_threshold_selector(data: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Filters columns based on low variance.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing numerical features.
    threshold : float, optional
        Variance threshold for feature selection (default is 0.05).

    Returns
    -------
    pd.DataFrame
        DataFrame with features above the specified variance threshold.
    """
    logger.debug("Applying variance threshold selector.")
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    selected_data = data.iloc[:, selector.get_support(indices=True)]
    logger.debug(f"Selected {selected_data.shape[1]} features out of {data.shape[1]} after variance filtering.")
    return selected_data


def csv_parser(file: Path) -> pd.DataFrame:
    """
    Reads normalised CSV files and annotates organelle information.

    Parameters
    ----------
    file : Path
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame or None
        Parsed and formatted DataFrame if valid, otherwise None.
    """
    organelle_map = {
        'acrosome': 'acrosome',
        'nucleus': 'nucleus',
        'mitochondria': 'mitochondria'
    }
    # Detect organelle type from filename
    organelle = next((name for key, name in organelle_map.items() if key in file.stem.lower()), None)
    if organelle is None:
        # Log the ignored file
        logger.warning(f"Ignoring file: {file.stem} (does not contain a known organelle type).")
        return None  # Skip processing this file
    logger.debug(f"Processing organelle file: {file}")
    df = pd.read_csv(file)
    if df.empty:
        logger.warning(f"File {file} is empty. Skipping processing.")
        return None  # Skip empty files
    
    logger.debug(f"Sample data from {file.name}:\n{df.head()}")
    # Assign filename metadata and set the appropriate index
    df = (df.assign(fn=file.stem)
          .rename(columns=lambda x: x.replace('Cy5', 'AR'))
          .set_index(['Plate_Metadata', 'Well_Metadata'], drop=False)  # Ensure index is not lost
          .add_suffix(f"_{organelle}"))
    return df

def impute_per_plate(group, imputer):
    """
    Applies imputation within each Plate_Metadata group separately.
    
    Parameters
    ----------
    group : pd.DataFrame
        Data subset for a specific plate.

    Returns
    -------
    pd.DataFrame
        Imputed data for the plate.
    """
    numeric_cols = group.select_dtypes(include=[np.number]).columns  # Select only numeric columns
    if numeric_cols.empty:
        logger.warning(f"No numeric columns found for plate: {group['Plate_Metadata'].iloc[0]}. Skipping imputation.")
        return group  # Skip if no numeric columns

    # Apply imputation only to numeric columns
    group[numeric_cols] = imputer.fit_transform(group[numeric_cols])
    return group


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection for SCP data.")

    parser.add_argument("--experiment_name", 
                        type=str, 
                        default="SCP", 
                        help="Name of the experiment (default: SCP)")

    parser.add_argument("--input_dir", 
                        type=str, 
                        default="/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/04022025/Normalised/", 
                        help="Path to input directory containing CSV files (default: predefined path)")

    parser.add_argument("--output_dir", 
                        type=str, 
                        default="{experiment}_feature_selection", 
                        help="Path to output directory (default: {experiment}_feature_selection)")

    parser.add_argument("--impute_method", 
                        type=str, 
                        choices=["median", "knn"], 
                        default="knn", 
                        help="Method for imputing missing values (default: knn)")

    parser.add_argument("--knn_neighbors", 
                        type=int, 
                        default=5, 
                        help="Number of neighbors for KNN imputation (default: 5)")

    parser.add_argument("--correlation_threshold", 
                        type=float, 
                        default=0.99, 
                        help="Threshold for correlation-based feature selection (default: 0.99)")

    parser.add_argument("--annotation_file", 
                        type=str, 
                        default="/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/04022025/KVP_4Plates_04022025.csv", 
                        help="Path to annotation CSV file (default: predefined path)")



    args = parser.parse_args()

    # Extract the experiment name before using it in logging
    experiment_name = args.experiment_name

    # Load annotation data
    annotation_path = Path(args.annotation_file)
    # **System & Command-line Information for Reproducibility**
    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Experiment Name: {experiment_name}")
    logger.info("Loading annotation data.")

    ddu = pd.read_csv(annotation_path, index_col=0)  # Ignores the first unnamed column
    ddu.columns = ddu.columns.str.strip().str.replace(" ", "_")  # Normalize column names

    # Ensure 'Plate_Metadata' exists
    if 'Plate_Metadata' not in ddu.columns and 'Plate' in ddu.columns:
        ddu.rename(columns={'Plate': 'Plate_Metadata'}, inplace=True)

    # Rename 'Well' to 'Well_Metadata' to match expected column names
    if 'Well_Metadata' not in ddu.columns and 'Well' in ddu.columns:
        ddu.rename(columns={'Well': 'Well_Metadata'}, inplace=True)

    # Ensure that renaming worked
    print(f"Columns after renaming: {ddu.columns.tolist()}")

    # Now setting the index should work
    ddu.set_index(['Plate_Metadata', 'Well_Metadata'], inplace=True)
    
    print("WARNING HARD CODED - change!!")
    # Standardise plate metadata
    plate_patterns = {
        'NPSCDD000401': '20241129_NPSCDD000401_STB',
        'NPSCDD000400': '20241129_NPSCDD000400_STB',
        'NPSCDD0003971': 'NPSCDD0003971_05092024',
        'NPSCDD0003972': 'NPSCDD0003972_05092024'
    }
    ddu.reset_index(inplace=True)
    for pattern, replacement in plate_patterns.items():
        ddu.loc[ddu['Plate_Metadata'].str.contains(pattern, na=False), 'Plate_Metadata'] = replacement
    ddu.set_index(['Plate_Metadata', 'Well_Metadata'], inplace=True)
    
    logger.info(f"Annotation data loaded with shape: {ddu.shape}")
    logger.debug(f"Annotation data sample:\n{ddu.head()}")

    # Load and parse SCP data
    input_directory = Path(args.input_dir)
    print("....WARNING hard code -  change.....")
    input_files = list(input_directory.glob('NPSCDD000400*csv*'))
    logger.info(f"Loading SCP data from {len(input_files)} files.")
    parsed_dfs = [csv_parser(file) for file in input_files]

    parsed_dfs = [df for df in parsed_dfs if df is not None]  # Remove ignored files

    # Check if we have enough files to join
    if len(parsed_dfs) == 0:
        logger.error("No valid organelle data files found! Check input directory or file naming conventions.")
        sys.exit(1)
    elif len(parsed_dfs) < 3:
        logger.warning(f"Only {len(parsed_dfs)} organelle data files found. Expected 3 (acrosome, nucleus, mitochondria).")

    # Merge datasets safely
    # Merge organelle data first
    # Merge organelle data efficiently
    df = pd.concat(parsed_dfs, axis=1)  # Instead of multiple join operations

    # Merge annotation data
    df = df.join(ddu, how='inner').reset_index()

    # Defragment DataFrame after heavy modifications
    df = df.copy()
 

    # Ensure no self-referencing metadata rows
    # Set the correct index first (avoiding reprocessing later)
    df = df.query('~Plate_Metadata.str.contains("Plate_Metadata")')
    df.set_index(['Source_Plate_Barcode', 'Source_well', 'name', 'Plate_Metadata', 'Well_Metadata', 'cpd_id', 'cpd_type'], inplace=True)
    logger.info(f"Merged data shape: {df.shape}")

    
    # clean data prior to imputation, no inf or only NaN values
    # Check for Inf values
    # Select only numeric columns before checking for infinities
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # replace inf values with NaN
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)


    # Drop columns that contain only NaN values
    num_cols = df.select_dtypes(include=[np.number]).columns
    nan_counts = df[num_cols].isna().sum()
    nan_only_cols = nan_counts[nan_counts == df.shape[0]].index.tolist()

    if nan_only_cols:
        logger.warning(f"These numeric columns contain only NaNs and will be dropped: {nan_only_cols}")
        df.drop(columns=nan_only_cols, inplace=True)

    # Choose imputer based on argument
    if args.impute_method == "knn":
        imputer = KNNImputer(n_neighbors=args.knn_neighbors)
    else:
        imputer = SimpleImputer(strategy=args.impute_method)

    if "Plate_Metadata" not in df.columns:
        df = df.reset_index()  # Avoid modifying in place (prevents fragmentation)
        df = df.copy()  # Defragment DataFrame


    # Apply imputation per plate
    if "Plate_Metadata" in df.columns:
        df = df.groupby("Plate_Metadata", group_keys=False, observed=True).apply(lambda g: impute_per_plate(g.drop(columns=["Plate_Metadata"]), imputer))

        logger.info(f"Missing values imputed using {args.impute_method} method per plate.")
    else:
        logger.warning("Plate_Metadata column not found in DataFrame. Applying imputation to entire dataset.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        logger.info(f"Missing values imputed using {args.impute_method} method on full dataset.")


    # Handle non-numeric data explicitly
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Non-numeric columns found: {non_numeric_cols.tolist()}")
        df2 = df.drop(columns=non_numeric_cols)
    else:
        df2 = df
    
    # Correlation-based feature selection
    logger.info("Computing correlation matrix.")
    cm = df2.corr().abs()
    upper_tri = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > args.correlation_threshold)]
    logger.debug(f"Dropping {len(to_drop)} correlated features.")
    df3 = df2.drop(columns=to_drop)
    
    # Variance thresholding
    final_df = variance_threshold_selector(df3)
    logger.info(f"Final data shape after feature selection: {final_df.shape}")
    logger.info(f"Final selected features: {final_df.shape[1]} columns out of {df3.shape[1]}. {df3.shape[1] - final_df.shape[1]} features dropped.")

    # Ensure output directory exists
    # Format the output directory properly
    output_dir = Path(args.output_dir.format(experiment=args.experiment_name)).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)  # Create if not exists



    
    # Save cleaned data
    output_file = output_dir / f"{args.experiment_name}_feature_select.csv"
    final_df.to_csv(output_file)
    logger.info(f"Cleaned data saved successfully to {output_file}.")
    logger.info(f"Final selected features: {final_df.shape[1]} columns out of {df3.shape[1]}")
    print(f"Feature selection and cleaning completed. Data saved to {output_file}.")
