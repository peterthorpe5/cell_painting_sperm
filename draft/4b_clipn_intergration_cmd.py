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
 If the user wishes to skip retraining, they can pass the --use_optimised_params 
 argument with the path to this JSON file, allowing the script to 
 load the best parameters, initialize the model, and proceed directly 
 to generating latent representations. This approach significantly 
 speeds up analysis by avoiding redundant training while ensuring that 
 the most effective hyperparameters are consistently used

"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import umap.umap_ as umap
from pathlib import Path
from clipn import CLIPn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
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

parser.add_argument("--stb", nargs="+", default=default_stb_files,
                    help="List of STB dataset files (default: predefined STB files)")

parser.add_argument("--experiment", nargs="+", default=default_experiment_files,
                    help="List of Experiment dataset files (default: predefined experiment files)")

parser.add_argument("--use_optimised_params",
                    type=str,
                    default=None,
                    help="Path to JSON file containing optimised hyperparameters. If provided, training is skipped.")
args = parser.parse_args()


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
    latent_dim = trial.suggest_int("latent_dim", 10, 100, step=10)
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



##################################################################
#  Step 2: Setup Output Directory
# TO DO: needs to change as we train for best params, so this doesnt work. 
output_folder = f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}"
os.makedirs(output_folder, exist_ok=True)

#  Step 3: Setup Logging
log_filename = os.path.join(output_folder, "clipn_integration.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
logger.info(f"Starting SCP data analysis using CLIPn with latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")
logger.info(f"STB datasets: {args.stb}")
logger.info(f"Experiment datasets: {args.experiment}")

# Load and merge datasets
stb_dfs = [pd.read_csv(f) for f in args.stb]
experiment_dfs = [pd.read_csv(f) for f in args.experiment]

stb_data = pd.concat(stb_dfs, axis=0, ignore_index=True)
experiment_data = pd.concat(experiment_dfs, axis=0, ignore_index=True)

# Extract numerical features
stb_numeric = stb_data.select_dtypes(include=[np.number])
experiment_numeric = experiment_data.select_dtypes(include=[np.number])



#  Create cpd_id Mapping Dictionary ---
if 'cpd_id' in experiment_data.columns:
    experiment_cpd_id_map = dict(enumerate(experiment_data['cpd_id']))  # Store index -> cpd_id mapping
else:
    raise KeyError("Error: 'cpd_id' column is missing from experiment data!")

# Store a direct mapping from the original indices to cpd_id
experiment_cpd_id_map = experiment_data['cpd_id'].copy()
stb_cpd_id_map = stb_data['cpd_id'].copy()


# Extract numerical features
experiment_numeric = experiment_data.select_dtypes(include=[np.number])
stb_numeric = stb_data.select_dtypes(include=[np.number])

# **Drop columns that are entirely NaN in either dataset BEFORE imputation**
experiment_numeric = experiment_numeric.dropna(axis=1, how='all')
stb_numeric = stb_numeric.dropna(axis=1, how='all')

# Identify initial common columns BEFORE imputation
common_columns_before = experiment_numeric.columns.intersection(stb_numeric.columns)
logger.info(f"Common numerical columns BEFORE imputation: {list(common_columns_before)}")

# Retain only common columns
experiment_numeric = experiment_numeric[common_columns_before]
stb_numeric = stb_numeric[common_columns_before]

# **Handle missing values with median imputation**
imputer = SimpleImputer(strategy="median")
experiment_numeric_imputed = pd.DataFrame(imputer.fit_transform(experiment_numeric), columns=common_columns_before)
stb_numeric_imputed = pd.DataFrame(imputer.fit_transform(stb_numeric), columns=common_columns_before)

# **Identify columns lost during imputation**
common_columns_after = experiment_numeric_imputed.columns.intersection(stb_numeric_imputed.columns)
columns_lost = set(common_columns_before) - set(common_columns_after)
logger.info(f"Columns lost during imputation: {list(columns_lost)}")



# Ensure both datasets retain only these common columns AFTER imputation
experiment_numeric_imputed = experiment_numeric_imputed[common_columns_after]
stb_numeric_imputed = stb_numeric_imputed[common_columns_after]

logger.info(f"Common numerical columns AFTER imputation: {list(common_columns_after)}")
logger.info(f"experiment data shape after imputation: {experiment_numeric_imputed.shape}")
logger.info(f"STB data shape after imputation: {stb_numeric_imputed.shape}")

# Create dataset labels
dataset_labels = {0: "experiment Assay", 1: "STB"}

# Initialize dictionaries to store mappings between LabelEncoder values and cpd_id
experiment_cpd_id_map = {}
stb_cpd_id_map = {}

# **Handle labels (assuming 'cpd_type' exists)**
if 'cpd_type' in experiment_data.columns:
    label_encoder = LabelEncoder()
    experiment_labels = label_encoder.fit_transform(experiment_data['cpd_type'])
    experiment_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    # Store mapping of encoded label to cpd_id
    experiment_cpd_id_map = {i: cpd_id for i, cpd_id in enumerate(experiment_data['cpd_id'].values)}

else:
    experiment_labels = np.zeros(experiment_numeric_imputed.shape[0])
    experiment_label_mapping = {"unknown": 0}
    experiment_cpd_id_map = {i: None for i in range(len(experiment_numeric_imputed))}  # Empty mapping

if 'cpd_type' in stb_data.columns:
    label_encoder = LabelEncoder()
    stb_labels = label_encoder.fit_transform(stb_data['cpd_type'])
    stb_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    # Store mapping of encoded label to cpd_id
    stb_cpd_id_map = {i: cpd_id for i, cpd_id in enumerate(stb_data['cpd_id'].values)}

else:
    stb_labels = np.zeros(stb_numeric_imputed.shape[0])
    stb_label_mapping = {"unknown": 0}
    stb_cpd_id_map = {i: None for i in range(len(stb_numeric_imputed))}  # Empty mapping


# **Convert dataset names to numerical indices using LabelEncoder**
dataset_names = ["experiment_assay_combined", "STB_combined"]
dataset_encoder = LabelEncoder()
dataset_indices = dataset_encoder.fit_transform(dataset_names)

dataset_mapping = dict(zip(dataset_indices, dataset_names))
X = {dataset_indices[0]: experiment_numeric_imputed.values, dataset_indices[1]: stb_numeric_imputed.values}
y = {dataset_indices[0]: experiment_labels, dataset_indices[1]: stb_labels}
label_mappings = {dataset_indices[0]: experiment_label_mapping, dataset_indices[1]: stb_label_mapping}

logger.info("Datasets successfully structured for CLIPn.")
logger.info(f"Final dataset shapes being passed to CLIPn: { {k: v.shape for k, v in X.items()} }")



########################################################
# CLIPn clustering with hyper optimisation
logger.info(f"Running CLIPn")

# Define hyperparameter output path
hyperparam_file = os.path.join(output_folder, "best_hyperparameters.json")

if args.use_optimised_params:
    # Load pre-trained parameters and skip training
    logger.info(f"Loading optimised hyperparameters from {args.use_optimised_params}")
    with open(args.use_optimised_params, "r") as f:
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

    # Save optimised parameters
    with open(hyperparam_file, "w") as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Optimised hyperparameters saved to {hyperparam_file}")

    # Update args with best parameters
    args.latent_dim = best_params["latent_dim"]
    args.lr = best_params["lr"]
    args.epoch = best_params["epochs"]

    logger.info(f"Using optimised parameters: latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")

    # Train the model with the optimised parameters
    logger.info(f"Running CLIPn with optimised latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")
    clipn_model = CLIPn(X, y, latent_dim=args.latent_dim)
    logger.info("Fitting CLIPn model...")
    loss = clipn_model.fit(X, y, lr=args.lr, epochs=args.epoch)
    logger.info(f"CLIPn training completed. Final loss: {loss[-1]:.6f}")

    # Generate latent representations
    logger.info("Generating latent representations.")
    Z = clipn_model.predict(X)


# mk new dir for new params. 
output_folder = f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}"
os.makedirs(output_folder, exist_ok=True)

# Convert numerical dataset names back to string names
Z_named = {str(k): v for k, v in Z.items()}  # Ensure all keys are strings
# Save latent representations
np.savez(os.path.join(output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_latent_representations.npz"), **Z_named)


# Convert numerical dataset names back to original names
Z_named = {str(k): v.tolist() for k, v in Z.items()}  # Convert keys to strings and values to lists

# Save latent representations in NPZ format
np.savez("CLIPn_latent_representations.npz", **Z_named)
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
plt.savefig(os.path.join(output_folder, f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_UMAP.pdf"), dpi=300)
plt.close()

logger.info(f"UMAP visualization saved to '{output_folder}/clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_UMAP.pdf'.")

# Define the output folder dynamically based on hyperparameters
output_folder = f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}"
Path(output_folder).mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

# Save each dataset's latent representations separately with the correct prefix
for dataset, values in Z_named.items():
    df = pd.DataFrame(values)
    df.to_csv(os.path.join(output_folder, f"{output_folder}_latent_representations_{dataset}.csv"), index=False)

logger.info(f"Latent representations saved successfully in {output_folder}/")

# Save index and label mappings
pd.Series({"experiment_assay_combined": 0, "STB_combined": 1}).to_csv(
    os.path.join(output_folder, f"{output_folder}_dataset_index_mapping.csv")
)
pd.DataFrame(label_mappings).to_csv(
    os.path.join(output_folder, f"{output_folder}_label_mappings.csv")
)

logger.info(f"Index and label mappings saved successfully in {output_folder}/")

logger.info("Index and label mappings saved.")


# Convert latent representations to DataFrame and restore cpd_id
experiment_latent_df = pd.DataFrame(Z[0])
stb_latent_df = pd.DataFrame(Z[1])

# Restore original cpd_id values
experiment_latent_df.index = [experiment_cpd_id_map.get(i, f"Unknown_{i}") for i in range(len(experiment_latent_df))]
stb_latent_df.index = [stb_cpd_id_map.get(i, f"Unknown_{i}") for i in range(len(stb_latent_df))]



# Save combined file with cpd_id as index
combined_latent_df = pd.concat([experiment_latent_df, stb_latent_df])
combined_latent_df.to_csv(os.path.join(output_folder, f"{output_folder}_CLIPn_latent_representations_with_cpd_id.csv"))

logger.info("Latent representations saved successfully with cpd_id as index.")




#####################################################################
# Perform UMAP dimensionality reduction
logger.info("Generating UMAP visualization with cpd_id labels.")
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
latent_umap = umap_model.fit_transform(combined_latent_df)

# Extract `cpd_id` for labeling
cpd_id_list = list(combined_latent_df.index)

# Save UMAP coordinates
umap_df = pd.DataFrame(latent_umap, index=combined_latent_df.index, columns=["UMAP1", "UMAP2"])
umap_file = os.path.join(output_folder, f"{output_folder}_UMAP_coordinates.csv")
umap_df.to_csv(umap_file)
logger.info(f"UMAP coordinates saved to '{umap_file}'.")

# -----------------------------------------------
# UMAP 1: Scatter Plot with `cpd_id` Labels
# -----------------------------------------------
plt.figure(figsize=(12, 8))
plt.scatter(latent_umap[:, 0], latent_umap[:, 1], s=5, alpha=0.7, cmap="viridis")

# Add small text labels for `cpd_id`
for i, cpd in enumerate(cpd_id_list):
    plt.text(latent_umap[i, 0], latent_umap[i, 1], str(cpd), fontsize=6, alpha=0.7)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Visualization with cpd_id Labels")

umap_plot_file = os.path.join(output_folder, f"{output_folder}_UMAP_latent_visualization_cpd_id.pdf")
plt.savefig(umap_plot_file, dpi=300)
plt.close()
logger.info(f"UMAP visualization saved as '{umap_plot_file}'.")

# -----------------------------------------------
# UMAP 2: Colour-coded by Dataset (Experiment vs. STB)
# -----------------------------------------------
logger.info("Generating UMAP visualization highlighting Experiment vs. STB data.")

# Create a colour list based on dataset membership
dataset_labels = []
for cpd in combined_latent_df.index:
    if cpd in experiment_cpd_id_map.values:
        dataset_labels.append("Experiment")  # Colour these differently
    else:
        dataset_labels.append("STB")

# Convert to colour-mapped values
dataset_colors = ["red" if label == "Experiment" else "blue" for label in dataset_labels]

plt.figure(figsize=(12, 8))
plt.scatter(latent_umap[:, 0], latent_umap[:, 1], s=5, alpha=0.7, c=dataset_colors)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Visualization: Experiment (Red) vs. STB (Blue)")

umap_colored_plot_file = os.path.join(output_folder, f"{output_folder}_UMAP_experiment_vs_stb.pdf")
plt.savefig(umap_colored_plot_file, dpi=300)
plt.close()
logger.info(f"UMAP visualization (experiment vs. STB) saved as '{umap_colored_plot_file}'.")


##################################################################################
# Compute Pairwise Distances
logger.info("Computing pairwise distances.")

# Compute distance matrices
dist_matrix = cdist(combined_latent_df.values, combined_latent_df.values, metric="euclidean")

# Convert to DataFrame
dist_df = pd.DataFrame(dist_matrix, index=combined_latent_df.index, columns=combined_latent_df.index)

# Save full distance matrix
dist_matrix_file = os.path.join(output_folder, f"{output_folder}_pairwise_compound_distances.csv")
dist_df.to_csv(dist_matrix_file)
logger.info(f"Pairwise distance matrix saved to '{dist_matrix_file}'.")


###########
# Generate Summary of Closest & Farthest Compounds



# Load the latent representation CSV
latent_df = combined_latent_df

# Compute pairwise Euclidean distances
dist_matrix = cdist(latent_df.values, latent_df.values, metric="euclidean")

# Convert to DataFrame for easy analysis
dist_df = pd.DataFrame(dist_matrix, index=latent_df.index, columns=latent_df.index)

# Save full distance matrix for further analysis
dist_df.to_csv("pairwise_compound_distances.csv")

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
summary_file = os.path.join(output_folder, f"{output_folder}_compound_similarity_summary.csv")
summary_df.to_csv(summary_file)
logger.info(f"Compound similarity summary saved to '{summary_file}'.")


###########
# Generate a clustered heatmap
plt.figure(figsize=(12, 10))
sns.clustermap(dist_df, cmap="viridis", method="ward", figsize=(12, 10))
plt.title("Pairwise Distance Heatmap of Compounds")

heatmap_file = os.path.join(output_folder, f"{output_folder}_compound_distance_heatmap.pdf")
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

dendrogram_file = os.path.join(output_folder, f"{output_folder}_compound_clustering_dendrogram.pdf")
plt.savefig(dendrogram_file)
plt.close()
logger.info(f"Hierarchical clustering dendrogram saved to '{dendrogram_file}'.")

# Save linkage matrix for further reference
linkage_matrix_file = os.path.join(output_folder, f"{output_folder}_compound_clustering_linkage_matrix.csv")
np.savetxt(linkage_matrix_file, linkage_matrix, delimiter="\t")
logger.info(f"Linkage matrix saved to '{linkage_matrix_file}'.")

# Final completion message
logger.info("SCP data analysis with CLIPn completed successfully!")

