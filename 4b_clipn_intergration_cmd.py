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
from cell_painting.plot import plot_dendrogram, plot_umap_coloured_by_experiment, \
    plot_distance_heatmap, generate_umap
from cell_painting.process_data import objective, optimise_clipn, group_and_filter_data, \
    decode_clipn_predictions, ensure_multiindex, \
        compute_pairwise_distances, generate_similarity_summary, restore_encoded_labels, \
        reconstruct_combined_latent_df, impute_missing_values, process_common_columns, \
        encode_cpd_data, prepare_data_for_clipn, run_clipn, restore_multiindex

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


# Create intermediate_files directory inside the main output folder
intermediate_folder = os.path.join(main_output_folder, "intermediate_files")
os.makedirs(intermediate_folder, exist_ok=True)




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


# Define filenames
imputed_experiment_path = os.path.join(intermediate_folder, "experiment_data_imputed.csv")
imputed_stb_path = os.path.join(intermediate_folder, "stb_data_imputed.csv")

# Save imputed datasets with MultiIndex (e.g. cpd_id, Library, cpd_type)
try:
    logger.info("Saving imputed experiment and STB datasets with MultiIndex.")

    experiment_data_imputed.to_csv(imputed_experiment_path)
    stb_data_imputed.to_csv(imputed_stb_path)

    logger.info(f"Imputed experiment data saved to: {imputed_experiment_path}")
    logger.info(f"Imputed STB data saved to: {imputed_stb_path}")

except Exception as e:
    logger.error(f"Error saving imputed datasets: {e}")


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




# Reassign the updated datasets
experiment_data_imputed = datasets["experiment"]
stb_data_imputed = datasets["stb"]



# Define output paths
encoded_experiment_path = os.path.join(intermediate_folder, "experiment_data_encoded_imputed.csv")
encoded_stb_path = os.path.join(intermediate_folder, "stb_data_encoded_imputed.csv")

# Save encoded datasets
try:
    logger.info("Saving label-encoded + imputed experiment and STB datasets with MultiIndex.")

    experiment_data.to_csv(encoded_experiment_path)
    stb_data.to_csv(encoded_stb_path)

    logger.info(f"Encoded experiment data saved to: {encoded_experiment_path}")
    logger.info(f"Encoded STB data saved to: {encoded_stb_path}")

except Exception as e:
    logger.error(f"Error saving encoded datasets: {e}")


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



# Define output paths for grouped, imputed, and encoded data
grouped_experiment_path = os.path.join(intermediate_folder, "experiment_data_grouped_encoded_imputed_final.csv")
grouped_stb_path = os.path.join(intermediate_folder, "stb_data_grouped_encoded_imputed_final.csv")

try:
    logger.info("Saving grouped, imputed, and label-encoded experiment and STB datasets.")

    experiment_data_imputed.to_csv(grouped_experiment_path)
    stb_data_imputed.to_csv(grouped_stb_path)

    logger.info(f"Grouped experiment data saved to: {grouped_experiment_path}")
    logger.info(f"Grouped STB data saved to: {grouped_stb_path}")

except Exception as e:
    logger.error(f"Error saving grouped, encoded datasets: {e}")



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


# Create plot_file directory inside the main output folder
plot_files = os.path.join(output_folder, "plot_files")
os.makedirs(plot_files, exist_ok=True)

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
umap_plot_file = os.path.join(plot_files, "clipn_ldim_UMAP.pdf")

# Generate UMAP without labels
umap_df = generate_umap(combined_latent_df, output_folder, umap_plot_file, args, add_labels=False)
# Generate UMAP with labels
umap_plot_file = os.path.join(plot_files, "clipn_ldim_UMAP_labels.pdf")
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
umap_experiment_plot_file = os.path.join(plot_files, "UMAP_experiment_vs_stb.pdf")

# Call the function using the UMAP DataFrame (ensuring MultiIndex)
plot_umap_coloured_by_experiment(umap_df, umap_experiment_plot_file)


###########
# Generate Summary of Closest & Farthest Compounds
# Compute pairwise distances **before** using `dist_df`
logger.info("Computing pairwise compound distances at the (cpd_id, Library) level without collapsing across Libraries.")

logger.info("Computing pairwise compound distances at the (cpd_id, Library) level without collapsing across Libraries.")

try:
    # Step 1: Drop non-numeric columns (e.g. annotations)
    numeric_latent_df = combined_latent_df.select_dtypes(include=[np.number])

    # Step 2: Compute pairwise distances
    dist_df = compute_pairwise_distances(numeric_latent_df)

    # Step 3: Assign MultiIndex (same as original latent data)
    dist_df.index = combined_latent_df.index
    dist_df.columns = combined_latent_df.index

    logger.info(f"Distance matrix shape: {dist_df.shape}")

    # Step 4: Ensure square and symmetric matrix
    if not dist_df.shape[0] == dist_df.shape[1]:
        raise ValueError("Computed distance matrix is not square!")

    if not (dist_df.columns.equals(dist_df.index)):
        logger.warning("Distance matrix is not symmetric — enforcing symmetry.")
        dist_df = (dist_df + dist_df.T) / 2

    # Step 5: Flatten MultiIndex for clean CSV export
    dist_df.index = ['__'.join(map(str, idx)) for idx in dist_df.index]
    dist_df.columns = ['__'.join(map(str, idx)) for idx in dist_df.columns]

    # Step 6: Save to CSV
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
heatmap_file = os.path.join(plot_files, "compound_distance_heatmap.pdf")
plot_distance_heatmap(dist_df, heatmap_file)
logger.info(f"Pairwise distance heatmap saved to '{heatmap_file}'.")

# **Generate and Save Dendrogram**
dendrogram_file = os.path.join(plot_files, "compound_clustering_dendrogram.pdf")
plot_dendrogram(dist_df, dendrogram_file, label_fontsize=4)

logger.info(f"Hierarchical clustering dendrogram saved to '{dendrogram_file}'.")

logger.info("Intergration step finished")

