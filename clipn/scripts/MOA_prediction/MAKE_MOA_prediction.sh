#!/usr/bin/env bash
set -euo pipefail

# ---- paths & names ---------------------------------------------------------
POST_DIR="./STB_vs_mitotox_integrate_all_E150_L148/post_clipn"
EMB_TSV="${POST_DIR}/STB_vs_mitotox_integrate_all_E150_L148_decoded.tsv"

ANCHORS_TSV="pseudo_anchors.tsv"
ANCHORS_SUMMARY_TSV="anchors_summary.tsv"
ANCHORS_CLUSTERS_TSV="anchors_clusters_summary.tsv"
ANCHORS_KSEL_TSV="anchors_pseudo_k_selection.tsv"

MOA_OUT="moa_centroid_E150_L148_unsup"
MOA_KSEL_TSV="${MOA_OUT}/per_moa_k_selection.tsv"

mkdir -p "${MOA_OUT}"

# A) Make pseudo-anchors by clustering compounds (K-means; bootstrap auto-k)
python make_pseudo_anchors.py \
  --embeddings_tsv "${EMB_TSV}" \
  --out_anchors_tsv "${ANCHORS_TSV}" \
  --out_summary_tsv "${ANCHORS_SUMMARY_TSV}" \
  --out_clusters_tsv "${ANCHORS_CLUSTERS_TSV}" \
  --id_col cpd_id \
  --aggregate_method median \
  --clusterer kmeans \
  --n_clusters -1 \
  --auto_min_clusters 8 \
  --auto_max_clusters 64 \
  --random_seed 42 \
  --bootstrap_k_main \
  --k_candidates_main "8,12,16,24,32" \
  --n_bootstrap_main 100 \
  --subsample_main 0.8 \
  --stability_metric_main consensus_silhouette \
  --consensus_linkage_main average \
  --consensus_pac_limits "0.1,0.9" \
  --out_k_selection_tsv "${ANCHORS_KSEL_TSV}"

# B) Score compounds (cosine + CSLS; auto primary; adaptive shrinkage)
#    Enable per-MOA auto-k via bootstrap inside MOA:
python centroid_moa_scoring.py \
  --embeddings_tsv "${EMB_TSV}" \
  --anchors_tsv "${ANCHORS_TSV}" \
  --out_dir "${MOA_OUT}" \
  --id_col cpd_id \
  --moa_col moa_label \
  --aggregate_method median \
  --centroid_method median \
  --centroid_shrinkage 0.0 \
  --use_csls \
  --csls_k -1 \
  --primary_score auto \
  --auto_margin_threshold 0.02 \
  --moa_score_agg max \
  --adaptive_shrinkage \
  --adaptive_shrinkage_c 0.5 \
  --adaptive_shrinkage_max 0.3 \
  --annotate_anchors \
  --random_seed 0 \
  --n_permutations 1000 \
  --n_centroids_per_moa -1 \
  --bootstrap_k_moa \
  --k_candidates_moa "1,2,3,4" \
  --n_bootstrap_moa 50 \
  --subsample_moa 0.8 \
  --stability_metric_moa consensus_silhouette \
  --consensus_linkage_moa average \
  --consensus_pac_limits_moa "0.1,0.9" \
  --out_moa_k_selection_tsv "${MOA_KSEL_TSV}"

# C) Plot for visualisation
python plot_moa_centroids_2d.py \
  --moa_dir "${MOA_OUT}" \
  --anchors_tsv "${ANCHORS_TSV}" \
  --assignment predictions \
  --predictions_tsv "${MOA_OUT}/compound_predictions.tsv" \
  --projection umap \
  --n_centroids_per_moa 1 \
  --centroid_method median \
  --adaptive_shrinkage \
  --adaptive_shrinkage_c 0.5 \
  --adaptive_shrinkage_max 0.3 \
  --out_prefix "${MOA_OUT}/moa_map" \
  --random_seed 0
