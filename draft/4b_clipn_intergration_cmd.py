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


# Extract numerical features ... retain non numeric too ...
# Store non-numeric columns separately to reattach later
non_numeric_cols = ['cpd_id', 'Library']  # Explicitly list columns to keep
if experiment_data is not None:
    experiment_non_numeric = experiment_data[non_numeric_cols]
    experiment_numeric = experiment_data.select_dtypes(include=[np.number])

if stb_data is not None:
    stb_non_numeric = stb_data[non_numeric_cols]
    stb_numeric = stb_data.select_dtypes(include=[np.number])


# Log the first few rows after extracting numerical features
if experiment_numeric is not None:
    logger.info("First few rows of experiment_numeric:\n" + experiment_numeric.head().to_string())
if stb_numeric is not None:
    logger.info("First few rows of stb_numeric:\n" + stb_numeric.head().to_string())


# Create cpd_id Mapping Dictionary ---
if experiment_data is not None and 'cpd_id' in experiment_data.columns:
    experiment_cpd_id_map = dict(enumerate(experiment_data['cpd_id']))  # Store index -> cpd_id mapping
else:
    experiment_cpd_id_map = None  # Handle missing case gracefully
    logger.warning("Warning: 'cpd_id' column is missing from experiment data!")


# Store a direct mapping from the original indices to cpd_id
# Create cpd_id Mapping Dictionary ---
if experiment_data is not None and 'cpd_id' in experiment_data.columns:
    experiment_cpd_id_map = dict(enumerate(experiment_data['cpd_id']))  # Store index -> cpd_id mapping
else:
    experiment_cpd_id_map = None
    logger.warning("Warning: 'cpd_id' column is missing from experiment data!")

stb_cpd_id_map = stb_data['cpd_id'].copy()


# **Drop columns that are entirely NaN in either dataset BEFORE imputation**
# Drop columns that are entirely NaN in either dataset BEFORE imputation
if experiment_numeric is not None:
    experiment_numeric = experiment_numeric.dropna(axis=1, how='all')

if stb_numeric is not None:
    stb_numeric = stb_numeric.dropna(axis=1, how='all')


# Identify initial common columns BEFORE imputation
if experiment_numeric is not None and stb_numeric is not None:
    common_columns_before = experiment_numeric.columns.intersection(stb_numeric.columns)
elif experiment_numeric is not None:
    common_columns_before = experiment_numeric.columns  # No STB data, use only experiment columns
elif stb_numeric is not None:
    common_columns_before = stb_numeric.columns  # No experiment data, use only STB columns
else:
    raise ValueError("Error: No valid numerical data available!")

logger.info(f"Common numerical columns BEFORE imputation: {list(common_columns_before)}")



# Retain only common columns
if experiment_numeric is not None:
    experiment_numeric = experiment_numeric[common_columns_before]

if stb_numeric is not None:
    stb_numeric = stb_numeric[common_columns_before]

# Step 1: Ensure numerical datasets exist
experiment_numeric_imputed, stb_numeric_imputed = None, None  

# Drop columns that are entirely NaN before imputation
if experiment_numeric is not None:
    experiment_numeric = experiment_numeric.dropna(axis=1, how='all')
if stb_numeric is not None:
    stb_numeric = stb_numeric.dropna(axis=1, how='all')

# Identify initial common columns BEFORE imputation
if experiment_numeric is not None and stb_numeric is not None:
    common_columns_before = experiment_numeric.columns.intersection(stb_numeric.columns)
elif experiment_numeric is not None:
    common_columns_before = experiment_numeric.columns
elif stb_numeric is not None:
    common_columns_before = stb_numeric.columns
else:
    raise ValueError("Error: No valid numerical data available!")

logger.info(f"Common numerical columns BEFORE imputation: {list(common_columns_before)}")

# Retain only common columns
if experiment_numeric is not None:
    experiment_numeric = experiment_numeric[common_columns_before]
if stb_numeric is not None:
    stb_numeric = stb_numeric[common_columns_before]

# Step 2: Perform imputation
if args.impute:
    logger.info(f"Performing imputation for missing values using {args.impute_method} strategy.")

    if args.impute_method == "median":
        imputer = SimpleImputer(strategy="median")
    elif args.impute_method == "knn":
        imputer = KNNImputer(n_neighbors=args.knn_neighbors)

    # Apply imputation only if data exists
    if experiment_numeric is not None and not experiment_numeric.empty:
        experiment_numeric_imputed = pd.DataFrame(imputer.fit_transform(experiment_numeric), 
                                                  columns=common_columns_before)
    if stb_numeric is not None and not stb_numeric.empty:
        stb_numeric_imputed = pd.DataFrame(imputer.fit_transform(stb_numeric), 
                                           columns=common_columns_before)

    logger.info(f"Imputation complete. Experiment shape: {experiment_numeric_imputed.shape if experiment_numeric_imputed is not None else 'None'}, "
                f"STB shape: {stb_numeric_imputed.shape if stb_numeric_imputed is not None else 'None'}")

else:
    logger.info("Skipping imputation as per command-line argument.")
    experiment_numeric_imputed = experiment_numeric if experiment_numeric is not None else None
    stb_numeric_imputed = stb_numeric if stb_numeric is not None else None

if stb_data is not None and 'cpd_type' in stb_data.columns:
    label_encoder = LabelEncoder()
    stb_labels = label_encoder.fit_transform(stb_data['cpd_type'])
    stb_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Store mapping of encoded label to cpd_id
    if 'cpd_id' in stb_data.columns:
        stb_cpd_id_map = {i: cpd_id for i, cpd_id in enumerate(stb_data['cpd_id'].values)}
    else:
        stb_cpd_id_map = {}
        logger.warning("Warning: 'cpd_id' column is missing from STB data!")

else:
    # Ensure stb_labels is always assigned a valid array
    if stb_numeric_imputed is not None:
        stb_labels = np.zeros(stb_numeric_imputed.shape[0])
    else:
        stb_labels = np.array([])  # Ensure an empty array instead of unassigned variable
        logger.warning("Warning: No STB labels available!")

    stb_label_mapping = {"unknown": 0}
    stb_cpd_id_map = {}

# Debugging: Confirm label assignment
logger.info(f"STB label mapping: {stb_label_mapping}")
logger.info(f"STB cpd_id mapping size: {len(stb_cpd_id_map)}")
logger.info(f"STB labels assigned: {len(stb_labels)}")

# Log the first few rows after imputation
if experiment_numeric_imputed is not None:
    logger.info("First few rows of experiment_numeric_imputed:\n" + experiment_numeric_imputed.head().to_string())
if stb_numeric_imputed is not None:
    logger.info("First few rows of stb_numeric_imputed:\n" + stb_numeric_imputed.head().to_string())

# Step 3: Identify common columns AFTER imputation
if experiment_numeric_imputed is not None and stb_numeric_imputed is not None:
    common_columns_after = experiment_numeric_imputed.columns.intersection(stb_numeric_imputed.columns)
elif experiment_numeric_imputed is not None:
    common_columns_after = experiment_numeric_imputed.columns
elif stb_numeric_imputed is not None:
    common_columns_after = stb_numeric_imputed.columns
else:
    raise ValueError("Error: No numerical data available after imputation!")

logger.info(f"Common numerical columns AFTER imputation: {list(common_columns_after)}")
columns_lost = set(common_columns_before) - set(common_columns_after)
logger.info(f"Columns lost during imputation: {list(columns_lost)}")




# Step 4: Ensure both datasets retain only common columns AFTER imputation
if experiment_numeric_imputed is not None:
    experiment_numeric_imputed = experiment_numeric_imputed[common_columns_after]
if stb_numeric_imputed is not None:
    stb_numeric_imputed = stb_numeric_imputed[common_columns_after]



# Ensure both datasets retain only these common columns AFTER imputation
if experiment_numeric_imputed is not None:
    experiment_numeric_imputed = experiment_numeric_imputed[common_columns_after]

if stb_numeric_imputed is not None:
    stb_numeric_imputed = stb_numeric_imputed[common_columns_after]

logger.info(f"Common numerical columns AFTER imputation: {list(common_columns_after)}")

#   Check if experiment_numeric_imputed is None before accessing .shape
if experiment_numeric_imputed is not None:
    logger.info(f"Experiment data shape after imputation: {experiment_numeric_imputed.shape}")
else:
    logger.info("Experiment data is None after imputation.")

#   Check if stb_numeric_imputed is None before accessing .shape
if stb_numeric_imputed is not None:
    logger.info(f"STB data shape after imputation: {stb_numeric_imputed.shape}")
else:
    logger.info("STB data is None after imputation.")


# Create dataset labels
dataset_labels = {0: "experiment Assay", 1: "STB"}

# Initialize dictionaries to store mappings between LabelEncoder values and cpd_id
experiment_cpd_id_map = {}
stb_cpd_id_map = {}

# Handle labels (assuming 'cpd_type' exists)
if experiment_data is not None and 'cpd_type' in experiment_data.columns:
    label_encoder = LabelEncoder()
    experiment_labels = label_encoder.fit_transform(experiment_data['cpd_type'])
    experiment_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Store mapping of encoded label to cpd_id
    if 'cpd_id' in experiment_data.columns:
        experiment_cpd_id_map = {i: cpd_id for i, cpd_id in enumerate(experiment_data['cpd_id'].values)}
    else:
        experiment_cpd_id_map = {}
        logger.warning("Warning: 'cpd_id' column is missing from experiment data!")

else:
    # Handle NoneType properly before creating labels
    if experiment_numeric_imputed is not None:
        experiment_labels = np.zeros(experiment_numeric_imputed.shape[0])
    else:
        experiment_labels = np.array([])  # Empty array if no data exists

    experiment_label_mapping = {"unknown": 0}
    experiment_cpd_id_map = {}

logger.info(f"Experiment label mapping: {experiment_label_mapping}")
logger.info(f"Experiment cpd_id mapping size: {len(experiment_cpd_id_map)}")

# Convert dataset names to numerical indices using LabelEncoder
dataset_names = []
if experiment_numeric_imputed is not None and len(experiment_numeric_imputed) > 0:
    dataset_names.append("experiment_assay_combined")
if stb_numeric_imputed is not None and len(stb_numeric_imputed) > 0:
    dataset_names.append("STB_combined")

# Ensure at least one dataset exists before proceeding
if not dataset_names:
    logger.error("No valid datasets available for CLIPn analysis.")
    raise ValueError("Error: No valid datasets available for CLIPn analysis.")

dataset_encoder = LabelEncoder()
dataset_indices = dataset_encoder.fit_transform(dataset_names)

dataset_mapping = dict(zip(dataset_indices, dataset_names))

logger.info(f"Dataset Mapping: {dataset_mapping}")


# Ensure data is aggregated at the compound level
logger.info("Grouping by cpd_id and Library...")
logger.info("Grouping by cpd_id and Library with filtering of unnecessary columns...")

# Define datasets to process
datasets = {
    "experiment": experiment_numeric_imputed,
    "stb": stb_numeric_imputed
}

# Define the pattern for columns to drop
filter_pattern = 'Source_Plate_Barcode|COMPOUND_NUMBER|Notes|Seahorse_alert|Treatment|Number|Child|Paren|Location_[X,Y,Z]|ZernikePhase|Euler|Plate|Well|Field|Center_Z|Center_X|Center_Y|no_|fn_'

for dataset_name, dataset in datasets.items():
    if dataset is not None:
        logger.info(f"Processing {dataset_name} dataset...")
        logger.info(f"Before filtering & aggregation: {dataset.shape}")

        # Ensure `cpd_id` and `Library` exist before filtering
        required_cols = ['cpd_id', 'Library']
        if not all(col in dataset.columns for col in required_cols):
            logger.error(f"Missing required columns in {dataset_name}! Available columns: {list(dataset.columns)}")
            sys.exit(1)

        # Preserve non-numeric columns before dropping unnecessary ones
        dataset_non_numeric = dataset[required_cols]  
        
        # Identify columns to drop (if they exist)
        filter_cols = dataset.columns.str.contains(filter_pattern, regex=True)
        columns_to_drop = dataset.columns[filter_cols]

        if len(columns_to_drop) > 0:
            logger.info(f"Dropping {len(columns_to_drop)} unnecessary columns: {list(columns_to_drop)}")
            dataset_numeric = dataset.drop(columns=columns_to_drop, errors='ignore')  # Drop only if they exist
        else:
            logger.info("No unnecessary columns found for removal.")
            dataset_numeric = dataset.copy()  # If nothing to drop, retain all

        # Ensure only numeric columns are aggregated
        numeric_cols = dataset_numeric.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            logger.error(f"No numeric columns found in {dataset_name} after filtering!")
            sys.exit(1)

        # Perform safe aggregation: group only numeric features, then merge back non-numeric columns
        aggregated_data = dataset_numeric.groupby(required_cols)[numeric_cols].mean().reset_index()

        # Merge back non-numeric columns to ensure `cpd_id` and `Library` are retained
        aggregated_data = dataset_non_numeric.merge(aggregated_data, on=required_cols, how='right')

        # Assign back the processed dataset
        datasets[dataset_name] = aggregated_data

        # Log after aggregation
        logger.info(f"After filtering & aggregation: {aggregated_data.shape}")
    else:
        logger.warning(f"{dataset_name} dataset is None, skipping aggregation.")

# Reassign the updated datasets
experiment_numeric_imputed = datasets["experiment"]
stb_numeric_imputed = datasets["stb"]




#######################################################
# Initialize empty dictionaries for CLIPn input
# 
X, y, label_mappings = {}, {}, {}

# Ensure dataset_encoder exists before using transform
if dataset_names:
    dataset_encoder = LabelEncoder()
    dataset_indices = dataset_encoder.fit_transform(dataset_names)
    dataset_mapping = dict(zip(dataset_indices, dataset_names))
else:
    raise ValueError("Error: No valid datasets available for CLIPn analysis.")

logger.info(f"Dataset Mapping: {dataset_mapping}")

# Add datasets dynamically if they exist
if experiment_numeric_imputed is not None and not experiment_numeric_imputed.empty:
    exp_index = dataset_encoder.transform(["experiment_assay_combined"])[0]
    X[exp_index] = experiment_numeric_imputed.values
    y[exp_index] = experiment_labels
    label_mappings[exp_index] = experiment_label_mapping
    logger.info(f"  Added Experiment Data to X with shape: {experiment_numeric_imputed.shape}")
else:
    logger.warning(" No valid experiment data for CLIPn.")

if stb_numeric_imputed is not None and not stb_numeric_imputed.empty:
    stb_index = dataset_encoder.transform(["STB_combined"])[0]
    X[stb_index] = stb_numeric_imputed.values
    y[stb_index] = stb_labels
    label_mappings[stb_index] = stb_label_mapping
    logger.info(f"  Added STB Data to X with shape: {stb_numeric_imputed.shape}")
else:
    logger.warning(" No valid STB data for CLIPn.")

# Debugging: Log dataset keys before passing to CLIPn
logger.info(f" X dataset keys before CLIPn: {list(X.keys())}")
logger.info(f" y dataset keys before CLIPn: {list(y.keys())}")

# Ensure that at least one dataset is available
if not X:
    logger.error(" No valid datasets available for CLIPn analysis. Aborting!")
    raise ValueError("Error: No valid datasets available for CLIPn analysis.")

logger.info(" Datasets successfully structured for CLIPn.")
logger.info(f" Final dataset shapes being passed to CLIPn: { {k: v.shape for k, v in X.items()} }")



########################################################
# CLIPn clustering with hyper optimisation
logger.info(f"Running CLIPn")

# Define hyperparameter output path
hyperparam_file = os.path.join(output_folder, "best_hyperparameters.json")

if args.use_optimized_params:
    try:
        logger.info(f"Loading optimized hyperparameters from {args.use_optimized_params}")
        with open(args.use_optimized_params, "r") as f:
            best_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load optimized parameters: {e}")
        raise ValueError("Invalid or missing hyperparameter JSON file.")

if args.use_optimized_params:
    # Load pre-trained parameters and skip training
    logger.info(f"Loading optimized hyperparameters from {args.use_optimized_params}")
    with open(args.use_optimized_params, "r") as f:
        best_params = json.load(f)

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
    # Run Hyperparameter Optimisation
    logger.info("Running Hyperparameter Optimisation")
    best_params = optimize_clipn(n_trials=20)  # Bayesian Optimisation

    # Save optimized parameters
    with open(hyperparam_file, "w") as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"optimized hyperparameters saved to {hyperparam_file}")

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


# mk new dir for new params. 
output_folder = os.path.join(main_output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}")
os.makedirs(output_folder, exist_ok=True)


# Convert numerical dataset names back to string names
Z_named = {str(k): v for k, v in Z.items()}  # Ensure all keys are strings
# Save latent representations
np.savez(os.path.join(output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_latent_representations.npz"), **Z_named)


# Define the output folder dynamically based on hyperparameters
# Ensure all outputs go into the correct location


# Convert numerical dataset names back to original names
Z_named = {str(k): v.tolist() for k, v in Z.items()}  # Convert keys to strings and values to lists

# Save latent representations in NPZ format
np.savez(os.path.join(output_folder, "CLIPn_latent_representations.npz"), **Z_named)
logger.info("Latent representations saved successfully.")


# Perform UMAP
logger.info("Generating UMAP visualization.")
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
latent_umap = umap_model.fit_transform(np.vstack([Z[0], Z[1]]))

# Save UMAP results
umap_df = pd.DataFrame(latent_umap, columns=["UMAP1", "UMAP2"])
umap_df.to_csv(os.path.join(output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_UMAP.csv"), index=False)

# Generate and save UMAP plot
plt.figure(figsize=(8, 6))
plt.scatter(latent_umap[:, 0], latent_umap[:, 1], alpha=0.7, s=5, c="blue")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("CLIPn UMAP Visualization")
plt.savefig(os.path.join(output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_UMAP.pdf"), dpi=600)
plt.close()

logger.info(f"UMAP visualization saved to '{output_folder}/clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_UMAP.pdf'.")



# Perform KMeans clustering to assign colors to different clusters
num_clusters = 10  # Adjust this based on expected number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(latent_umap)

plt.figure(figsize=(12, 8))
plt.scatter(latent_umap[:, 0], latent_umap[:, 1], s=5, alpha=0.7, c=cluster_labels, cmap="tab10")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Visualization with Cluster Labels")

umap_clustered_plot_file = os.path.join(output_folder, f"clipn_latent_dim_UMAP_clusters.pdf")
plt.savefig(umap_clustered_plot_file, dpi=600)
plt.close()
logger.info(f"UMAP visualization with clusters saved as '{umap_clustered_plot_file}'.")



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

# Convert latent representations to DataFrame (only if available)
experiment_latent_df, stb_latent_df = None, None

if 0 in Z:  # Check if experiment data exists in Z
    experiment_latent_df = pd.DataFrame(Z[0])
    if experiment_cpd_id_map is not None:
        experiment_latent_df.index = [experiment_cpd_id_map.get(i, f"Unknown_{i}") for i in range(len(experiment_latent_df))]
    else:
        logger.warning("Warning: experiment_cpd_id_map is None, using default index.")

if 1 in Z:  # Check if STB data exists in Z
    stb_latent_df = pd.DataFrame(Z[1])
    if stb_cpd_id_map is not None:
        stb_latent_df.index = [stb_cpd_id_map.get(i, f"Unknown_{i}") for i in range(len(stb_latent_df))]
    else:
        logger.warning("Warning: stb_cpd_id_map is None, using default index.")


# Log the latent representations before UMAP
if experiment_latent_df is not None:
    logger.info("First few rows of experiment_latent_df (latent representations):\n" + experiment_latent_df.head().to_string())
if stb_latent_df is not None:
    logger.info("First few rows of stb_latent_df (latent representations):\n" + stb_latent_df.head().to_string())



logger.info("Expected experiment_cpd_id_map length:", len(experiment_cpd_id_map))
logger.info("Expected stb_cpd_id_map length:", len(stb_cpd_id_map))
logger.info("Observed experiment_latent_df index length:", len(experiment_latent_df.index))
logger.info("Observed stb_latent_df index length:", len(stb_latent_df.index))


# Ensure at least one DataFrame exists before concatenation
if experiment_latent_df is not None and stb_latent_df is not None:
    combined_latent_df = pd.concat([experiment_latent_df, stb_latent_df])
elif experiment_latent_df is not None:
    combined_latent_df = experiment_latent_df
elif stb_latent_df is not None:
    combined_latent_df = stb_latent_df
else:
    raise ValueError("Error: No latent representations available!")

logger.info("Latent representations processed successfully.")

# Log the combined latent representation DataFrame
if combined_latent_df is not None:
    logger.info("First few rows of combined_latent_df (final merged latent space):\n" + combined_latent_df.head().to_string())


# Ensure at least one DataFrame is available before concatenation
if experiment_latent_df is not None and stb_latent_df is not None:
    combined_latent_df = pd.concat([experiment_latent_df, stb_latent_df])
elif experiment_latent_df is not None:
    combined_latent_df = experiment_latent_df
elif stb_latent_df is not None:
    combined_latent_df = stb_latent_df
else:
    raise ValueError("Error: No latent representations available to save!")

# Save the combined file if there is data
combined_output_file = os.path.join(output_folder, "CLIPn_latent_representations_with_cpd_id.csv")
combined_latent_df.to_csv(combined_output_file)

logger.info(f"Latent representations saved successfully with cpd_id as index to {combined_output_file}.")


#####################################################################
# Perform UMAP dimensionality reduction
logger.info("Generating UMAP visualization with cpd_id labels.")
umap_model = umap.UMAP(n_neighbors=25, min_dist=0.1, n_components=2, random_state=42)
latent_umap = umap_model.fit_transform(combined_latent_df)

# Extract `cpd_id` for labeling
cpd_id_list = list(combined_latent_df.index)

# Save UMAP coordinates
umap_df = pd.DataFrame(latent_umap, index=combined_latent_df.index, columns=["UMAP1", "UMAP2"])
umap_file = os.path.join(output_folder, "UMAP_coordinates.csv")
umap_df.to_csv(umap_file)
logger.info(f"UMAP coordinates saved to '{umap_file}'.")

# -----------------------------------------------
# UMAP 1: Scatter Plot with `cpd_id` Labels
# -----------------------------------------------
plt.figure(figsize=(12, 8))
plt.scatter(latent_umap[:, 0], latent_umap[:, 1], s=5, alpha=0.7, c="blue")  # Fix: Set explicit color

# Add small text labels for `cpd_id`
for i, cpd in enumerate(cpd_id_list):
    plt.text(latent_umap[i, 0], latent_umap[i, 1], str(cpd), fontsize=6, alpha=0.7)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Visualization with cpd_id Labels")

umap_plot_file = os.path.join(output_folder, "UMAP_latent_visualization_cpd_id.pdf")
plt.savefig(umap_plot_file, dpi=300)
plt.close()
logger.info(f"UMAP visualization saved as '{umap_plot_file}'.")

# -----------------------------------------------
# UMAP 2: Colour-coded by Dataset (Experiment vs. STB)
# -----------------------------------------------
logger.info("Generating UMAP visualization highlighting Experiment vs. STB data.")

# Convert experiment_cpd_id_map.values to a set to avoid TypeError
experiment_cpd_id_set = set(experiment_cpd_id_map.values()) 


# Create a colour list based on dataset membership
dataset_labels = ["Experiment" if cpd in experiment_cpd_id_set else "STB" for cpd in cpd_id_list]

# Convert to colour-mapped values
dataset_colors = ["red" if label == "Experiment" else "blue" for label in dataset_labels]

plt.figure(figsize=(12, 8))
plt.scatter(latent_umap[:, 0], latent_umap[:, 1], s=5, alpha=0.7, c=dataset_colors)  # Fix: c now correctly maps colors

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Visualization: Experiment (Red) vs. STB (Blue)")

umap_colored_plot_file = os.path.join(output_folder, "UMAP_experiment_vs_stb.pdf")
plt.savefig(umap_colored_plot_file, dpi=300)
plt.close()
logger.info(f"UMAP visualization (experiment vs. STB) saved as '{umap_colored_plot_file}'.")


###########
# Generate Summary of Closest & Farthest Compounds



# Load the latent representation CSV
latent_df = combined_latent_df

# Compute pairwise Euclidean distances
dist_matrix = cdist(latent_df.values, latent_df.values, metric="euclidean")

# Convert to DataFrame for easy analysis
dist_df = pd.DataFrame(dist_matrix, index=latent_df.index, columns=latent_df.index)

# Save full distance matrix for further analysis
dist_df.to_csv(os.path.join(output_folder, "pairwise_compound_distances.csv"))  



print("Pairwise distance matrix saved as 'pairwise_compound_distances.csv'.")


# Find closest compounds (excluding self-comparison)
closest_compounds = dist_df.replace(0, np.nan).idxmin(axis=1)

# Find farthest compounds
farthest_compounds = dist_df.idxmax(axis=1)

# Create a summary DataFrame
summary_df = pd.DataFrame({
    "Closest Compound": closest_compounds,
    "Distance to Closest": dist_df.min(axis=1),
    "Farthest Compound": farthest_compounds,
    "Distance to Farthest": dist_df.max(axis=1)
})



# Save summary file
summary_file = os.path.join(output_folder, "compound_similarity_summary.csv")
summary_df.to_csv(summary_file)
logger.info(f"Compound similarity summary saved to '{summary_file}'.")


###########
# Generate a clustered heatmap
plt.figure(figsize=(12, 10))
sns.clustermap(dist_df, cmap="viridis", method="ward", figsize=(12, 10))
plt.title("Pairwise Distance Heatmap of Compounds")

heatmap_file = os.path.join(output_folder, "compound_distance_heatmap.pdf")
plt.savefig(heatmap_file)
plt.close()
logger.info(f"Pairwise distance heatmap saved to '{heatmap_file}'.")


###########
# Perform hierarchical clustering
linkage_matrix = linkage(squareform(dist_matrix), method="ward")

# Create and save dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=latent_df.index, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical Clustering of Compounds")
plt.xlabel("Compound")
plt.ylabel("Distance")

dendrogram_file = os.path.join(output_folder, "compound_clustering_dendrogram.pdf")
plt.savefig(dendrogram_file)
plt.close()
logger.info(f"Hierarchical clustering dendrogram saved to '{dendrogram_file}'.")

# Save linkage matrix for further reference
linkage_matrix_file = os.path.join(output_folder, "compound_clustering_linkage_matrix.csv")
np.savetxt(linkage_matrix_file, linkage_matrix, delimiter="\t")
logger.info(f"Linkage matrix saved to '{linkage_matrix_file}'.")

# Final completion message
logger.info("SCP data analysis with CLIPn completed successfully!")

