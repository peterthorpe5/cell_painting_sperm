# MOA Prediction Pipeline  
Unified Analysis for CLIPn Latent Space and CellProfiler Features

This repository provides a complete, modular pipeline for clustering, scoring, and visualising compounds using numerical embeddings derived from either:

- **CLIPn latent space outputs**, or  
- **CellProfiler processed morphological features** (after variance/correlation filtering)

The workflow is fully automatic:  
it detects feature columns, preserves important metadata, aggregates replicate wells, performs clustering, scores compounds against mode-of-action (MOA) centroids, conducts permutation testing, and generates high-quality UMAP/PCA plots.

All outputs are tab-separated.  
All Python code uses PEP-8 docstrings and UK English spelling.


---

## Contents

- `embedding_utils.py`  
  Shared utilities for loading data, detecting metadata, selecting numerical features, replicates aggregation, L2 normalisation, PCA/UMAP projection, and logging.

- `make_pseudo_anchors.py`  
  Simple KMeans-based pseudo-anchor generator with optional automatic k selection (via silhouette score).

- `make_pseudo_anchors_kmeans.py`  
  Bootstrap consensus clustering to create more stable pseudo-anchors, suitable for noisy or heterogeneous datasets.

- `centroid_moa_scoring.py`  
  Core scoring engine: builds MOA centroids, computes cosine/CSLS similarities, assigns predicted MOA, and performs permutation tests.

- `plot_moa_centroids_2d.py`  
  Generates PCA/UMAP two-dimensional projections of compounds and MOA centroids, with optional colouring by metadata.

- `MAKE_MOA_prediction.sh`  
  End-to-end wrapper script calling the full pipeline. Customisable for desktops, HPC clusters, or CI/CD.

---

## Key Features

### Automatic Data Handling
- Detects metadata columns such as  
  `cpd_id`, `Plate_Metadata`, `Well_Metadata`, `Library`, `Dataset`, `Sample`
- Identifies *all* numeric feature columns automatically.
- Supports both CLIPn embeddings (`0`, `1`, `2`, …) and CellProfiler features (`AreaShape_…`, `Texture_…`, etc.).
- Replicates aggregated using median or mean.
- L2-normalisation optional but included by default.

### Flexible Clustering
- Standard KMeans or  
- **Bootstrap consensus clustering** with adjustable:
  - number of bootstrap replicates  
  - candidate k values  
  - subsampling fraction  
- Robust to noise and varying cluster densities.

### Centroid Construction
- Median or mean MOA centroids
- Optional **sub-centroids** (multiple centroids per MOA)
- Optional **adaptive shrinkage** towards global mean to stabilise small groups

### Scoring
- Cosine similarity (default)  
- CSLS similarity (optionally enabled)  
- Full scoring matrix saved for downstream analysis

### Permutation Testing
- Per-compound enrichment statistics
- Null distribution saved in TSV format
- P-values computed using empirical null

### Visualisation
- PCA and UMAP projections
- Optional metadata-based colouring (e.g., `cpd_type`, `Library`)
- Centroids labelled directly on the map
- Optional HTML interactive plots (if Plotly installed)

---

## Requirements

Python 3.9+ recommended.

### Python Packages


numpy
pandas
scikit-learn
umap-learn
matplotlib
plotly (optional)

### Optional (UMAP)

umap-learn


---

## Typical Workflow

### 1. Prepare Embeddings  
Your `TSV` should contain:

- a compound identifier column (default: `cpd_id`)
- one or more metadata columns (optional)
- only **numeric features** for embeddings

Both CLIPn latent space and CellProfiler features are supported.

### 2. Run the Pipeline

Edit the top section of:



MAKE_MOA_prediction.sh


Then run:

```bash
bash MAKE_MOA_prediction.sh

```



This will create an output directory (default: moa_output) containing:

pseudo_anchors.tsv

compound_predictions.tsv

centroids_summary.tsv

raw_scores.tsv

permutation_pvalues.tsv

null_distribution.tsv

moa_map_pca.pdf

moa_map_umap.pdf
