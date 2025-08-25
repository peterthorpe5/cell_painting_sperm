#!/bin/bash
#SBATCH -J clipn                            # Job name
#SBATCH --partition=gpu                     # Partition/queue
#SBATCH --mem=270GB                          # Memory


cd /home/pthorpe/scratch/laura//home/pthorpe/scratch/laura/all_plates_together

set -Eeuo pipefail
IFS=$'\n\t'


#python split.py --output2 HGTx_LSS02_2.tsv --output1 HGTx_LSS02_1.tsv --input HGTx_LSS02_imputed_per_object.tsv  --sort


############################################
# Optional: conda env
############################################
# source ~/.bashrc
# conda activate clipn

############################################
# Paths & config
############################################
BASE_DIR="/home/pthorpe/scratch/laura/all_plates_together"
DATASET_CSV="${BASE_DIR}/datasets.txt"
METADATA="/home/pthorpe/scratch/laura/metadata/merged_metadata.csv.tsv"
EXTRA_METADATA="/mnt/shared/projects/uosa/Pete_Thorpe/2025_cell_painting_sperm/STB_annotation/24CPSC-Map.csv"

# Scripts
CLIPN_HOME="$HOME/apps/cell_painting_sperm/clipn"
RUN_CLIPN_SCRIPT="${CLIPN_HOME}/run_clipn.py"
UMAP_SCRIPT="${CLIPN_HOME}/scripts/UMAP_output_per_function.py"
ANALYSIS_SCRIPT="${CLIPN_HOME}/analyse_clipn_results.py"
PLOT_SCRIPT="${CLIPN_HOME}/plot_clipn_results.py"
SUMMARISE_NEIGHBOURS="${CLIPN_HOME}/scripts/summarise_mcp_neighbours.py"
IDENTIFY_ACROSOME="${CLIPN_HOME}/identify_acrosome_abnormalities.py"
EXPLAIN_FEATURES="${CLIPN_HOME}/explain_feature_driven_results.py"
SHAP_SCRIPT="${CLIPN_HOME}/shap_explain_nn_similarity.py"

# Run grid
MODES=("integrate_all")              # or ("reference_only" "integrate_all")
EPOCHS=(50 100 200)
LATENT_DIMS=(20)

# UMAP params
MIN_DISTS=(0.1)
COLOUR_BY=("Library" "Dataset" "cpd_type")
CLUSTERS=("None" 15)                 # "None" means “no clustering”

# Compounds to highlight / use as references downstream
COMPOUNDS=(Antimycine "Gossy Pol" Rotenone "Cytochalasin D" CCCP ketoconazole atorvastatin simvastatin)

############################################
# Helpers
############################################
section () { echo -e "\n==== $* ====\n"; }
run () { echo "+ $*"; "$@"; }

############################################
# Kick off
############################################
cd "$BASE_DIR"

for MODE in "${MODES[@]}"; do
  for EPOCH in "${EPOCHS[@]}"; do
    for LATENT_DIM in "${LATENT_DIMS[@]}"; do

      EXP_NAME="HGTx_all_${MODE}_E${EPOCH}_L${LATENT_DIM}"
      OUT_DIR="${BASE_DIR}/${EXP_NAME}"
      POST_DIR="${OUT_DIR}/post_clipn"
      LATENT_FILE="${EXP_NAME}_CLIPn_latent_aggregated_median.tsv"

      section "Running CLIPn: ${EXP_NAME}"
      mkdir -p "$OUT_DIR" "$POST_DIR"

      # Train/integrate CLIPn
      run python "$RUN_CLIPN_SCRIPT" \
        --datasets_csv "$DATASET_CSV" \
        --out "$OUT_DIR" \
        --experiment "$EXP_NAME" \
        --mode "$MODE" \
        --epoch "$EPOCH" \
        --latent_dim "$LATENT_DIM" \
        --save_model \
        --aggregate_method median \
        --scaling_mode all \
        --scaling_method robust

      # UMAP sweeps
      section "UMAP outputs for ${EXP_NAME}"
      for DIST in "${MIN_DISTS[@]}"; do
        for CLUSTER_COUNT in "${CLUSTERS[@]}"; do
          for COLOUR in "${COLOUR_BY[@]}"; do

            if [[ "$CLUSTER_COUNT" == "None" ]]; then
              CLUSTER_LABEL="kNone"
              CLUSTER_ARGS=()
            else
              CLUSTER_LABEL="k${CLUSTER_COUNT}"
              CLUSTER_ARGS=(--num_clusters "$CLUSTER_COUNT")
            fi

            OUTDIR="${POST_DIR}/UMAP_${CLUSTER_LABEL}"
            mkdir -p "$OUTDIR"

            run python "$UMAP_SCRIPT" \
              --input "${POST_DIR}/${LATENT_FILE}" \
              --add_labels \
              --output_dir "$OUTDIR" \
              --umap_min_dist "$DIST" \
              --colour_by "$COLOUR" \
              --compound_metadata "$METADATA" \
              --highlight_list "${COMPOUNDS[@]}" \
              "${CLUSTER_ARGS[@]}"
          done
        done
      done

      # Post-analysis (nearest neighbours, overlap, etc.)
      section "CLIPn post analysis for ${EXP_NAME}"
      run python "$ANALYSIS_SCRIPT" \
        --latent_csv "${POST_DIR}/${LATENT_FILE}" \
        --output_dir "${POST_DIR}/post_analysis_script" \
        --reference_ids "${COMPOUNDS[@]}" \
	--dataset_col Library \
 	--nn_metric cosine \
  	--n_neighbours 100 \
  	--network \
  	--threshold 0.2 \
	  --network_max_edges_per_node 10

      # Cosine UMAP & interactive plots
      section "Cosine UMAP plots for ${EXP_NAME}"
        run python "$PLOT_SCRIPT" \
        --latent_csv "${POST_DIR}/${LATENT_FILE}" \
        --plots "${POST_DIR}/cosine_UMAP" \
        --embedding umap \
        --umap_metric cosine \
        --umap_n_neighbors 15 \
        --umap_min_dist 0.1 \
        --colour_by cpd_type \
        --tooltip_columns cpd_id Library Plate_Metadata Well_Metadata cpd_type

      run python "$PLOT_SCRIPT" \
        --latent_csv "${POST_DIR}/${LATENT_FILE}" \
        --plots "${POST_DIR}/topo_mapper" \
        --embedding topo \
        --mapper_lens pca \
        --mapper_n_cubes 8 \
        --mapper_overlap 0.5 \
        --mapper_cluster dbscan \
        --mapper_eps 1.0 \
        --mapper_min_samples 2 \
        --interactive_topo \
        --tooltip_columns cpd_id Library Plate_Metadata Well_Metadata cpd_type


      echo "=== Finished $EXP_NAME ==="
      echo ""

      # Summarise neighbours at run level
      run python "$SUMMARISE_NEIGHBOURS" \
        --folder "$OUT_DIR" \
        --top_n 100 \
        --target "${COMPOUNDS[@]}" \
        --metadata "$METADATA"
        
      # Extra post-analysis if directory exists
      if [[ -d "${POST_DIR}/post_analysis_script" ]]; then
        pushd "${POST_DIR}/post_analysis_script" >/dev/null

        #run python "$IDENTIFY_ACROSOME" \
         # --ungrouped_list "$DATASET_CSV"

        #run python "$EXPLAIN_FEATURES" \
         # --output_dir nn_with_n \
         # --ungrouped_list "$DATASET_CSV" \
         # --query_ids "${COMPOUNDS[@]}" \
         # --nn_file nearest_neighbours.tsv

        #run python "$SHAP_SCRIPT" \
          #--nn_file nearest_neighbours.tsv \
          #--features "$DATASET_CSV" \
          #--query_ids "${COMPOUNDS[@]}" \
          #--output_dir shap_analysis_real \
          #--n_neighbors 10

        popd >/dev/null
      fi

    done
  done
done

############################################
# Batch: summarise neighbours for all subfolders
############################################
section "Summarising neighbours across all ${BASE_DIR} subdirectories"
for FOLDER in "${BASE_DIR}"/*; do
  if [[ -d "$FOLDER" ]]; then
    echo ">>> Processing: $FOLDER"
    run python "$SUMMARISE_NEIGHBOURS" \
      --folder "$FOLDER" \
      --top_n 30 \
      --metadata "$METADATA" \
      --extra_metadata "$EXTRA_METADATA" \
      --target "${COMPOUNDS[@]}"
    echo ""
  fi
done

echo "=== All neighbour summaries attempted ==="
