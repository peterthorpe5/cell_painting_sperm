# A) Make pseudo-anchors by clustering compounds (k-means; 30 clusters as a start)
python make_pseudo_anchors.py \
  --embeddings_tsv ./STB_vs_mitotox_integrate_all_E150_L148/post_clipn/STB_vs_mitotox_integrate_all_E150_L148_decoded.tsv \
  --out_anchors_tsv pseudo_anchors.tsv \
  --id_col cpd_id \
  --aggregate_method median \
  --clusterer kmeans \
  --n_clusters 30

# B) Score compounds against those prototypes (cosine + CSLS, robust aggregation)
python prototype_moa_scoring.py \
  --embeddings_tsv ./STB_vs_mitotox_integrate_all_E150_L148/post_clipn/STB_vs_mitotox_integrate_all_E150_L148_decoded.tsv \
  --anchors_tsv pseudo_anchors.tsv \
  --out_dir moa_proto_E150_L148_unsup \
  --id_col cpd_id \
  --aggregate_method median \
  --n_prototypes_per_moa 1 \
  --use_csls --csls_k 10
