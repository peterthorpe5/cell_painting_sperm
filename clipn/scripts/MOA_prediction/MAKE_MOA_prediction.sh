

# A) Make pseudo-anchors by clustering compounds (K-means; auto-k ≈ √n)
python make_pseudo_anchors.py \
  --embeddings_tsv ./STB_vs_mitotox_integrate_all_E150_L148/post_clipn/STB_vs_mitotox_integrate_all_E150_L148_decoded.tsv \
  --out_anchors_tsv pseudo_anchors.tsv \
  --out_summary_tsv anchors_summary.tsv \
  --out_clusters_tsv anchors_clusters_summary.tsv \
  --id_col cpd_id \
  --aggregate_method median \
  --clusterer kmeans \
  --n_clusters -1 \
  --auto_min_clusters 8 \
  --auto_max_clusters 64 \
  --random_seed 42

# B) Score compounds (cosine + CSLS on; auto primary; adaptive shrinkage for small-n)
python prototype_moa_scoring.py \
  --embeddings_tsv ./STB_vs_mitotox_integrate_all_E150_L148/post_clipn/STB_vs_mitotox_integrate_all_E150_L148_decoded.tsv \
  --anchors_tsv pseudo_anchors.tsv \
  --out_dir moa_centroid_E150_L148_unsup \
  --id_col cpd_id \
  --aggregate_method median \
  --n_centroids_per_moa 1 \
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
  --random_seed 0

  # c) plot for visualization 
  python plot_moa_centroids_2d.py \
      --moa_dir "$MOA_OUT" \
      --anchors_tsv "${POST_DIR}/pseudo_anchors.tsv" \
      --assignment predictions \
      --predictions_tsv "$MOA_OUT/compound_predictions.tsv" \
      --projection umap \
      --n_centroids_per_moa 1 \
      --centroid_method median \
      --adaptive_shrinkage --adaptive_shrinkage_c 0.5 --adaptive_shrinkage_max 0.3 \
      --out_prefix "$MOA_OUT/moa_map" \
      --random_seed 0
