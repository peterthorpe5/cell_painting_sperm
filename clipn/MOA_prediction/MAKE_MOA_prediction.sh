#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# MOA prediction pipeline
# ---------------------------------------------------------------
# This wrapper runs the full analysis:
#   1. Generate pseudo-anchors (simple or bootstrap version)
#   2. Score compounds against MOA centroids
#   3. Produce PCA / UMAP 2D projection plots
#
# All inputs/outputs are tab-separated.
# ================================================================

# -------------------------------
# User parameters (edit as needed)
# -------------------------------
EMB_TSV=""                # Input embeddings (CLIPn or CellProfiler TSV)
ID_COL="cpd_id"           # Identifier column
MOA_COL="pseudo_moa"      # MOA column for anchors
OUT_DIR="moa_output"      # Output directory

# Methods
AGG_METHOD="median"       # replicate aggregation: median or mean
CENTROID_METHOD="median"  # median or mean
N_SUBCENTROIDS=1          # number of sub-centroids per MOA (use 1 for simple)

# Clustering
USE_BOOTSTRAP=false       # if true, use the bootstrap KMeans anchor script
K_CANDIDATES="8,12,16,24,32"
N_BOOTSTRAP=50
SUBSAMPLE=0.8

# Similarity scoring
USE_CSLS=false            # if true, use CSLS instead of cosine
CSLS_K=10

# Projection
PROJECTION="both"         # umap | pca | both
COLOUR_BY=""              # optional metadata column

# Other
RANDOM_SEED=0


# -------------------------------
# Create output directory
# -------------------------------
mkdir -p "${OUT_DIR}"

ANCHORS="${OUT_DIR}/pseudo_anchors.tsv"
ANCHORS_SUMMARY="${OUT_DIR}/anchors_summary.tsv"
ANCHORS_CLUSTERS="${OUT_DIR}/anchors_clusters.tsv"
KSEL="${OUT_DIR}/k_selection.tsv"


# ================================================================
# 1. Generate pseudo-anchors
# ================================================================

if [ "${USE_BOOTSTRAP}" = true ]; then
    echo "Running bootstrap consensus clustering for pseudo-anchors..."
    python3 analysis_pipeline/make_pseudo_anchors_kmeans.py \
        --embeddings_tsv "${EMB_TSV}" \
        --out_anchors_tsv "${ANCHORS}" \
        --out_summary_tsv "${ANCHORS_SUMMARY}" \
        --out_clusters_tsv "${ANCHORS_CLUSTERS}" \
        --out_k_selection_tsv "${KSEL}" \
        --id_col "${ID_COL}" \
        --k_candidates "${K_CANDIDATES}" \
        --n_bootstrap "${N_BOOTSTRAP}" \
        --subsample "${SUBSAMPLE}" \
        --aggregate_method "${AGG_METHOD}" \
        --random_seed "${RANDOM_SEED}"
else
    echo "Running simple KMeans pseudo-anchor generation..."
    python3 analysis_pipeline/make_pseudo_anchors.py \
        --embeddings_tsv "${EMB_TSV}" \
        --out_anchors_tsv "${ANCHORS}" \
        --id_col "${ID_COL}" \
        --aggregate_method "${AGG_METHOD}" \
        --auto_k \
        --k_min 8 \
        --k_max 64 \
        --random_seed "${RANDOM_SEED}"
fi


# ================================================================
# 2. Score compounds against MOA centroids
# ================================================================

echo "Running MOA scoring + permutation testing..."

python3 analysis_pipeline/centroid_moa_scoring.py \
    --embeddings_tsv "${EMB_TSV}" \
    --anchors_tsv "${ANCHORS}" \
    --out_dir "${OUT_DIR}" \
    --id_col "${ID_COL}" \
    --moa_col "${MOA_COL}" \
    --aggregate_method "${AGG_METHOD}" \
    --centroid_method "${CENTROID_METHOD}" \
    --n_subcentroids "${N_SUBCENTROIDS}" \
    $( [ "${USE_CSLS}" = true ] && echo "--use_csls" ) \
    --csls_k "${CSLS_K}" \
    --n_perm 1000 \
    --random_seed "${RANDOM_SEED}"


# ================================================================
# 3. 2D projection plots
# ================================================================

echo "Generating PCA/UMAP plots..."

PLOT_PREFIX="${OUT_DIR}/moa_map"

python3 analysis_pipeline/plot_moa_centroids_2d.py \
    --embeddings_tsv "${EMB_TSV}" \
    --anchors_tsv "${ANCHORS}" \
    --out_prefix "${PLOT_PREFIX}" \
    --id_col "${ID_COL}" \
    --moa_col "${MOA_COL}" \
    --projection "${PROJECTION}" \
    $( [ -n "${COLOUR_BY}" ] && echo "--colour_by ${COLOUR_BY}" ) \
    --random_seed "${RANDOM_SEED}"

echo "Done."

