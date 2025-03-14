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

"""

import os
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

#  Step 1: Command-Line Arguments
parser = argparse.ArgumentParser(description="Perform CLIPn clustering and UMAP on SCP data.")

parser.add_argument("--latent_dim", type=int, default=20, help="Dimensionality of latent space (default: 20)")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for CLIPn (default: 1e-5)")
parser.add_argument("--epoch", type=int, default=200, help="Number of training epochs (default: 200)")

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

args = parser.parse_args()


######################################
#  Step 2: Setup Output Directory
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

# Drop columns entirely NaN before imputation
stb_numeric = stb_numeric.dropna(axis=1, how='all')
experiment_numeric = experiment_numeric.dropna(axis=1, how='all')

# Retain only common columns
common_columns = stb_numeric.columns.intersection(experiment_numeric.columns)
stb_numeric = stb_numeric[common_columns]
experiment_numeric = experiment_numeric[common_columns]

# Handle missing values with median imputation
imputer = SimpleImputer(strategy="median")
stb_numeric_imputed = pd.DataFrame(imputer.fit_transform(stb_numeric), columns=common_columns)
experiment_numeric_imputed = pd.DataFrame(imputer.fit_transform(experiment_numeric), columns=common_columns)

# CLIPn clustering
logger.info(f"Running CLIPn with latent_dim={args.latent_dim}, lr={args.lr}, epochs={args.epoch}")
clipn_model = CLIPn(
    {0: stb_numeric_imputed.values, 
     1: experiment_numeric_imputed.values},
    {0: np.zeros(len(stb_numeric_imputed)), 
     1: np.ones(len(experiment_numeric_imputed))},
    latent_dim=args.latent_dim
)
clipn_model.fit({0: stb_numeric_imputed.values, 
                 1: experiment_numeric_imputed.values},
                {0: np.zeros(len(stb_numeric_imputed)), 
                 1: np.ones(len(experiment_numeric_imputed))},
                lr=args.lr, epochs=args.epoch)

# Extract latent representations
logger.info("Generating latent representations.")
Z = clipn_model.predict({0: stb_numeric_imputed.values, 
                         1: experiment_numeric_imputed.values})

# Save latent representations
np.savez(os.path.join(output_folder, 
                      f"clipn_ldim{args.latent_dim}_lr{args.lr}_epoch{args.epoch}_latent_representations.npz"), **Z)

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

# Final message
logger.info("SCP data analysis with CLIPn completed successfully!")




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

# Generate UMAP scatter plot
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

# Find closest compounds (excluding self-comparison)
closest_compounds = dist_df.replace(0, np.nan).idxmin(axis=1)

# Find farthest compounds
farthest_compounds = dist_df.idxmax(axis=1)

# Create summary DataFrame
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
logger.info("SCP data analysis with CLIPn completed successfully! 🚀")

