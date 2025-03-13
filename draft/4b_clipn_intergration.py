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


# **Handle labels (assuming 'cpd_type' exists)**
if 'cpd_type' in mitotox_data.columns:
    label_encoder = LabelEncoder()
    mitotox_labels = label_encoder.fit_transform(mitotox_data['cpd_type'])
    mitotox_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
else:
    mitotox_labels = np.zeros(mitotox_numeric_imputed.shape[0])
    mitotox_label_mapping = {"unknown": 0}

if 'cpd_type' in stb_data.columns:
    label_encoder = LabelEncoder()
    stb_labels = label_encoder.fit_transform(stb_data['cpd_type'])
    stb_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
else:
    stb_labels = np.zeros(stb_numeric_imputed.shape[0])
    stb_label_mapping = {"unknown": 0}

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
latent_dim = 10
logger.info(f"Running CLIPn with latent dimension: {latent_dim}")

clipn_model = CLIPn(X, y, latent_dim=latent_dim)
logger.info("Fitting CLIPn model...")
clipn_model.fit(X, y, lr=1e-5, epochs=100)

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
