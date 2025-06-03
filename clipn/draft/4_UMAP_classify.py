#!/usr/bin/env python
# coding: utf-8

"""
Downstream Data Analysis for Sperm Cell Painting Data
---------------------------------------------------
This script performs downstream analysis of sperm cell painting (SCP) data, including:
    - Loading and processing SCP data from multiple plates.
    - Data cleaning and filtering (e.g., handling missing values, feature selection).
    - Exploratory data analysis, including correlation and clustering.
    - Dimensionality reduction using UMAP.
    - Clustering using KMeans and hierarchical clustering.
    - Network graph analysis based on cosine and correlation distances.
    - Feature profile visualizations and heatmaps.
    - Visualizing data overlays based on chemotypes.
    - Computing pairwise distances in UMAP space for interrogation.
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP

# Configure logging
logging.basicConfig(
    filename='classification.log', 
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("Starting classification analysis.")

# Define file paths to normalized SCP datasets
data_files = [
    '/uod/npsc/Lab_Book/BMGF/NHCP/SCP/Mitotox_assay/Normalised/Mitotox_assay_NPSCDD0003999_25102024_normalised.csv',
    '/uod/npsc/Lab_Book/BMGF/NHCP/SCP/Mitotox_assay/Normalised/Mitotox_assay_NPSCDD0004023_25102024_normalised.csv',
    '/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/04022025/Normalised/STB_NPSCDD0003971_05092024_normalised.csv',
    '/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/04022025/Normalised/STB_NPSCDD0003972_05092024_normalised.csv',
    '/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/04022025/Normalised/STB_NPSCDD000400_05092024_normalised.csv',
    '/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/04022025/Normalised/STB_NPSCDD000401_05092024_normalised.csv',
    '/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/13022025/Normalised/STB_NPSCDD0004034_13022025_normalised.csv'
]

logger.info("Loading SCP datasets.")
sp1 = pd.concat([pd.read_csv(file) for file in data_files], ignore_index=True)
logger.info(f"Data loaded with shape: {sp1.shape}")

# Drop unnecessary columns
logger.info("Dropping unnecessary columns.")
sp1 = sp1.drop(columns=['Source_Plate_Barcode', 'Source_well', 'Destination_Concentration'])

# Ensure compound IDs are treated as strings
sp1['cpd_id'] = sp1['cpd_id'].astype(str)

# Remove empty or NaN compound IDs
logger.info("Filtering out empty or NaN compound IDs.")
sp1 = sp1.query('~cpd_id.str.contains("empty|nan")')

# Assign library labels based on metadata
logger.info("Assigning library labels.")
sp1.loc[sp1['cpd_type'].str.contains('positive controls', na=False), 'Library'] = 'STB'
sp1.loc[sp1['cpd_type'].str.contains('negative control \(DMSO\)', na=False), 'Library'] = 'control'
sp1.loc[sp1['cpd_id'].str.contains('MCP09|DDD', na=False), 'Library'] = 'MCP09'
sp1.loc[sp1['Plate_Metadata'].str.contains('20241129_NPSCDD000401_STB|20241129_NPSCDD000400_STB|NPSCDD0003971_05092024|NPSCDD0003972_05092024', na=False), 'Library'] = 'STB'

# Aggregate data by compound ID and library, taking the mean of feature values
logger.info("Aggregating data by compound ID and library.")
numeric_columns = sp1.select_dtypes(include=[np.number]).columns
sp1 = sp1.groupby(['cpd_id', 'Library'])[numeric_columns].mean()

# Remove features with high proportions of missing values
missing_threshold = 0.05
missing_proportions = sp1.isnull().mean()
features_to_drop = missing_proportions[missing_proportions > missing_threshold].index
sp1 = sp1.drop(columns=features_to_drop)
logger.info(f"Dropped {len(features_to_drop)} features with >{missing_threshold * 100}% missing values.")

# Fill remaining NaN values with 0
sp1 = sp1.fillna(0)

# Log that standardization has already been performed
logger.info("Skipping standardization step, as it was completed in the normalisation script.")

# Save processed data
sp1.to_csv("processed_classification_data.csv")
logger.info("Processed classification data saved.")

# ## Feature Correlation Analysis
logger.info("Computing pairwise correlation distances.")
correlation_distances = pdist(sp1, metric='correlation')
correlation_matrix = pd.DataFrame(squareform(correlation_distances), index=sp1.index, columns=sp1.index)

# Generate heatmap to visualize correlations
sns.clustermap(correlation_matrix, cmap='coolwarm', figsize=(10, 10))
plt.title("Feature Correlation Heatmap")
plt.savefig("feature_correlation_heatmap.pdf")
plt.show()

# ## Dimensionality Reduction with UMAP
logger.info("Performing UMAP dimensionality reduction.")
umap_model = UMAP(random_state=123, min_dist=0.001, n_neighbors=8, metric='correlation', n_components=3)
sp1_umap = umap_model.fit_transform(sp1)

umap_df = pd.DataFrame(sp1_umap, columns=['UMAP1', 'UMAP2', 'UMAP3'], index=sp1.index)
logger.info("UMAP transformation completed.")

# Compute pairwise Euclidean distances in UMAP space
logger.info("Computing pairwise Euclidean distances in UMAP space.")
umap_distances = pd.DataFrame(squareform(pdist(umap_df, metric='euclidean')), index=sp1.index, columns=sp1.index)
umap_distances.to_csv("umap_pairwise_distances.csv")
logger.info("Saved UMAP pairwise distances to umap_pairwise_distances.csv")

# Visualize UMAP results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=umap_df['UMAP1'], y=umap_df['UMAP2'], hue=[lib for _, lib in sp1.index], palette='tab10')
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP Projection of Feature Space")
plt.legend(title='Library', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("umap_projection.pdf")
plt.show()

logger.info("Classification analysis completed successfully.")
