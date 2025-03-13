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

CLIPn Input:
    - X: dictionary of numeric feature matrices indexed numerically.
    - y: dictionary of corresponding labels indexed numerically.
    - latent_dim: the dimensionality of latent space for clustering.

CLIPn Output:
    - Z: dictionary of latent representations from all datasets, indexed numerically.
    - `.csv` files mapping numerical dataset indices and labels back to original names.
    - A `.csv` file mapping numerical labels back to their original names.
    - Latent representation files saved with `cpd_id` as row index.

"""

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
import fastcluster
import ace_tools as tools
from scipy.cluster.hierarchy import linkage

# Configure logging
logging.basicConfig(
    filename='clipn_integration.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

logger.info("Starting SCP data analysis using CLIPn.")

# Paths to datasets
data_files = [
    'data/Mitotox_assay_NPSCDD0003999_25102024_normalised.csv',
    'data/Mitotox_assay_NPSCDD0004023_25102024_normalised.csv',
    'data/STB_NPSCDD0003971_05092024_normalised.csv',
    'data/STB_NPSCDD0003972_05092024_normalised.csv',
    'data/STB_NPSCDD000400_05092024_normalised.csv',
    'data/STB_NPSCDD000401_05092024_normalised.csv',
    'data/STB_NPSCDD0004034_13022025_normalised.csv'
]

# Separate Mitotox and STB files
mitotox_files = [f for f in data_files if "Mitotox_assay" in f]
stb_files = [f for f in data_files if "STB" in f]

# Load and merge Mitotox datasets
mitotox_dfs = [pd.read_csv(f) for f in mitotox_files]
stb_dfs = [pd.read_csv(f) for f in stb_files]

# Concatenate datasets
mitotox_data = pd.concat(mitotox_dfs, axis=0, ignore_index=True)
stb_data = pd.concat(stb_dfs, axis=0, ignore_index=True)

#  Create cpd_id Mapping Dictionary ---
if 'cpd_id' in mitotox_data.columns:
    mitotox_cpd_id_map = dict(enumerate(mitotox_data['cpd_id']))  # Store index -> cpd_id mapping
else:
    raise KeyError("Error: 'cpd_id' column is missing from Mitotox data!")

# Store a direct mapping from the original indices to cpd_id
mitotox_cpd_id_map = mitotox_data['cpd_id'].copy()
stb_cpd_id_map = stb_data['cpd_id'].copy()


# Extract numerical features
mitotox_numeric = mitotox_data.select_dtypes(include=[np.number])
stb_numeric = stb_data.select_dtypes(include=[np.number])

# **Drop columns that are entirely NaN in either dataset BEFORE imputation**
mitotox_numeric = mitotox_numeric.dropna(axis=1, how='all')
stb_numeric = stb_numeric.dropna(axis=1, how='all')

# Identify initial common columns BEFORE imputation
common_columns_before = mitotox_numeric.columns.intersection(stb_numeric.columns)
logger.info(f"Common numerical columns BEFORE imputation: {list(common_columns_before)}")

# Retain only common columns
mitotox_numeric = mitotox_numeric[common_columns_before]
stb_numeric = stb_numeric[common_columns_before]

# **Handle missing values with median imputation**
imputer = SimpleImputer(strategy="median")
mitotox_numeric_imputed = pd.DataFrame(imputer.fit_transform(mitotox_numeric), columns=common_columns_before)
stb_numeric_imputed = pd.DataFrame(imputer.fit_transform(stb_numeric), columns=common_columns_before)

# **Identify columns lost during imputation**
common_columns_after = mitotox_numeric_imputed.columns.intersection(stb_numeric_imputed.columns)
columns_lost = set(common_columns_before) - set(common_columns_after)
logger.info(f"Columns lost during imputation: {list(columns_lost)}")



# Ensure both datasets retain only these common columns AFTER imputation
mitotox_numeric_imputed = mitotox_numeric_imputed[common_columns_after]
stb_numeric_imputed = stb_numeric_imputed[common_columns_after]

logger.info(f"Common numerical columns AFTER imputation: {list(common_columns_after)}")
logger.info(f"Mitotox data shape after imputation: {mitotox_numeric_imputed.shape}")
logger.info(f"STB data shape after imputation: {stb_numeric_imputed.shape}")





# Create dataset labels
dataset_labels = {0: "Mitotox Assay", 1: "STB"}

# Initialize dictionaries to store mappings between LabelEncoder values and cpd_id
mitotox_cpd_id_map = {}
stb_cpd_id_map = {}

# **Handle labels (assuming 'cpd_type' exists)**
if 'cpd_type' in mitotox_data.columns:
    label_encoder = LabelEncoder()
    mitotox_labels = label_encoder.fit_transform(mitotox_data['cpd_type'])
    mitotox_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    # Store mapping of encoded label to cpd_id
    mitotox_cpd_id_map = {i: cpd_id for i, cpd_id in enumerate(mitotox_data['cpd_id'].values)}

else:
    mitotox_labels = np.zeros(mitotox_numeric_imputed.shape[0])
    mitotox_label_mapping = {"unknown": 0}
    mitotox_cpd_id_map = {i: None for i in range(len(mitotox_numeric_imputed))}  # Empty mapping

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
dataset_names = ["Mitotox_assay_combined", "STB_combined"]
dataset_encoder = LabelEncoder()
dataset_indices = dataset_encoder.fit_transform(dataset_names)

dataset_mapping = dict(zip(dataset_indices, dataset_names))
X = {dataset_indices[0]: mitotox_numeric_imputed.values, dataset_indices[1]: stb_numeric_imputed.values}
y = {dataset_indices[0]: mitotox_labels, dataset_indices[1]: stb_labels}
label_mappings = {dataset_indices[0]: mitotox_label_mapping, dataset_indices[1]: stb_label_mapping}

logger.info("Datasets successfully structured for CLIPn.")
logger.info(f"Final dataset shapes being passed to CLIPn: { {k: v.shape for k, v in X.items()} }")

# **CLIPn clustering**
latent_dim = 20
logger.info(f"Running CLIPn with latent dimension: {latent_dim}")

clipn_model = CLIPn(X, y, latent_dim=latent_dim)
logger.info("Fitting CLIPn model...")
clipn_model.fit(X, y, lr=1e-5, epochs=200)

# **Extract latent representations**
logger.info("Generating latent representations.")
Z = clipn_model.predict(X)



# Convert numerical dataset names back to original names
Z_named = {str(k): v.tolist() for k, v in Z.items()}  # Convert keys to strings and values to lists

# Save latent representations in NPZ format
np.savez("CLIPn_latent_representations.npz", **Z_named)
logger.info("Latent representations saved successfully.")

# Save each dataset's latent representations separately
for dataset, values in Z_named.items():
    df = pd.DataFrame(values)
    df.to_csv(f"CLIPn_latent_representations_{dataset}.csv", index=False)

logger.info("Latent representations saved successfully as CSV files.")

# Save index and label mappings
pd.Series({"Mitotox_assay_combined": 0, "STB_combined": 1}).to_csv("dataset_index_mapping.csv")
pd.DataFrame(label_mappings).to_csv("label_mappings.csv")

logger.info("Index and label mappings saved.")


# Convert latent representations to DataFrame and restore cpd_id
mitotox_latent_df = pd.DataFrame(Z[0])
stb_latent_df = pd.DataFrame(Z[1])

# Restore original cpd_id values
mitotox_latent_df.index = [mitotox_cpd_id_map[i] for i in range(len(mitotox_latent_df))]
stb_latent_df.index = [stb_cpd_id_map[i] for i in range(len(stb_latent_df))]

# Save combined file with cpd_id as index
combined_latent_df = pd.concat([mitotox_latent_df, stb_latent_df])
combined_latent_df.to_csv("CLIPn_latent_representations_with_cpd_id.csv")

logger.info("Latent representations saved successfully with cpd_id as index.")



##########################################################
### **UMAP Visualization of CLIPn Latent Representations**
logger.info("Generating UMAP visualization.")

# Define plot colors
colors = ["blue", "red"]

# Plot UMAP of latent representations
plt.figure(figsize=(8, 6))
for i, (dataset, values) in enumerate(Z.items()):
    z_embed = umap.UMAP().fit_transform(values)  # Perform UMAP dimensionality reduction
    plt.scatter(
        z_embed[:, 0], z_embed[:, 1], 
        c=y[dataset], cmap="tab10", label=dataset, alpha=0.7, s=10
    )

plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("CLIPn UMAP Latent Representations")
plt.legend()
plt.grid(True)
plt.savefig("CLIPn_UMAP_latent_visualization.pdf")
plt.show()

logger.info("UMAP visualization saved as 'CLIPn_UMAP_latent_visualization_original.pdf'.")
logger.info("SCP data analysis completed successfully.")



# UMAP Visualization
umap_model = umap.UMAP()
Z_umap = np.vstack([Z[0], Z[1]])
y_umap = np.concatenate([y[0], y[1]])

plt.figure(figsize=(8, 6))
scatter = plt.scatter(Z_umap[:, 0], Z_umap[:, 1], c=y_umap, cmap="tab10", s=10, alpha=0.75)
plt.colorbar(label="Cluster")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("CLIPn UMAP Visualization with Cluster Labels")
plt.savefig("CLIPn_UMAP_latent_visualization_corrected_labels.pdf")
logger.info("UMAP visualization saved with corrected labels.")



logger.info("SCP data analysis completed successfully.")



#####################################################################
# Perform UMAP dimensionality reduction
logger.info("Generating UMAP visualization with cpd_id labels.")
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
latent_umap = umap_model.fit_transform(combined_latent_df)

# Extract `cpd_id` for labeling
cpd_id_list = list(combined_latent_df.index)

# Create UMAP scatter plot with `cpd_id` labels
plt.figure(figsize=(12, 8))
plt.scatter(latent_umap[:, 0], latent_umap[:, 1], s=5, alpha=0.7, cmap="viridis")

# Add small text labels for `cpd_id`
for i, cpd in enumerate(cpd_id_list):
    plt.text(latent_umap[i, 0], latent_umap[i, 1], str(cpd), fontsize=6, alpha=0.7)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Visualization with cpd_id Labels")
plt.savefig("CLIPn_UMAP_latent_visualization_cpd_id.pdf", dpi=300)
logger.info("UMAP visualization with cpd_id labels saved.")
plt.show()



##################################################################################
# Compute Pairwise Distances

# Ensure we correctly map the original cpd_id back to Z
logger.info("Reconstructing cpd_id mappings for latent representations.")

# Reconstruct original cpd_id mappings for Mitotox and STB
mitotox_latent_df = pd.DataFrame(Z[0], index=[mitotox_cpd_id_map[i] for i in range(len(Z[0]))])
stb_latent_df = pd.DataFrame(Z[1], index=[stb_cpd_id_map[i] for i in range(len(Z[1]))])

# Save latent representations with proper cpd_id
mitotox_latent_df.to_csv("CLIPn_latent_representations_mitotox.csv")
stb_latent_df.to_csv("CLIPn_latent_representations_stb.csv")

# Merge both datasets into a single CSV
combined_latent_df = pd.concat([mitotox_latent_df, stb_latent_df])
combined_latent_df.to_csv("CLIPn_latent_representations_combined.csv")

logger.info("Latent representations successfully saved with cpd_id as index.")


# Compute Pairwise Distances using cpd_id as index
logger.info("Computing pairwise distances.")

mitotox_distances = cdist(mitotox_latent_df.values, mitotox_latent_df.values, metric="euclidean")
stb_distances = cdist(stb_latent_df.values, stb_latent_df.values, metric="euclidean")
inter_dataset_distances = cdist(mitotox_latent_df.values, stb_latent_df.values, metric="euclidean")

# Compute average distances
avg_mitotox_dist = np.mean(mitotox_distances)
avg_stb_dist = np.mean(stb_distances)
avg_inter_dist = np.mean(inter_dataset_distances)

logger.info(f"Average Intra-Mitotox Distance: {avg_mitotox_dist}")
logger.info(f"Average Intra-STB Distance: {avg_stb_dist}")
logger.info(f"Average Inter-Dataset Distance: {avg_inter_dist}")

# Save distances to CSV
distance_df = pd.DataFrame({
    "cpd_id": list(mitotox_latent_df.index) + list(stb_latent_df.index),
    "Average Intra-Mitotox Distance": [avg_mitotox_dist] * len(mitotox_latent_df) + [None] * len(stb_latent_df),
    "Average Intra-STB Distance": [None] * len(mitotox_latent_df) + [avg_stb_dist] * len(stb_latent_df),
    "Average Inter-Dataset Distance": [avg_inter_dist] * (len(mitotox_latent_df) + len(stb_latent_df))
})
distance_df.to_csv("cpd_id_pairwise_distances.csv", index=False)
logger.info("Pairwise distances saved to 'cpd_id_pairwise_distances.csv'.")





###########
# Generate UMAP Visualization with `cpd_id` Labels

logger.info("Generating UMAP visualization with cpd_id labels.")

# Perform UMAP dimensionality reduction
latent_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(combined_latent_df)

# Convert to DataFrame for saving
umap_df = pd.DataFrame(latent_umap, index=combined_latent_df.index, columns=["UMAP1", "UMAP2"])
umap_df.to_csv("CLIPn_UMAP_coordinates.csv")

logger.info("UMAP coordinates saved with cpd_id.")

# Plot UMAP
plt.figure(figsize=(12, 8))
scatter = plt.scatter(latent_umap[:, 0], latent_umap[:, 1], c="blue", alpha=0.7, s=5)

# Add cpd_id labels to each point (very small text)
for i, txt in enumerate(combined_latent_df.index):
    plt.annotate(txt, (latent_umap[i, 0], latent_umap[i, 1]), fontsize=3, alpha=0.6)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("CLIPn UMAP Visualization with cpd_id Labels")
plt.savefig("CLIPn_UMAP_latent_visualization_cpd_id.pdf", dpi=300)
plt.show()

logger.info("UMAP visualization with cpd_id labels saved.")





logger.info(f"Shape of data passed to UMAP: {combined_latent_df.shape}")



# Load the latent representation CSV
latent_df = pd.read_csv("CLIPn_latent_representations_combined.csv", index_col=0)

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

# Save results
summary_df.to_csv("compound_similarity_summary.csv")

print("Compound similarity summary saved as 'compound_similarity_summary.csv'.")




# Generate a clustered heatmap
plt.figure(figsize=(12, 10))
sns.clustermap(dist_df, cmap="viridis", method="ward", figsize=(12, 10))
plt.title("Pairwise Distance Heatmap of Compounds")
plt.savefig("compound_distance_heatmap.pdf")

# Perform hierarchical clustering
linkage_matrix = linkage(dist_matrix, method="ward")

# Create a dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=latent_df.index, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical Clustering of Compounds")
plt.xlabel("Compound")
plt.ylabel("Distance")
plt.savefig("compound_clustering_dendrogram.pdf")

# Save linkage matrix for further reference
np.savetxt("compound_clustering_linkage_matrix.csv", linkage_matrix, delimiter="\t")

# Display final outputs

tools.display_dataframe_to_user(name="Pairwise Compound Distances", dataframe=dist_df)