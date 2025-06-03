#!/bin/bash
#SBATCH -J clipn   #jobname
#SBATCH --partition=gpu
#SBATCH --mem=24GB


cd $HOME/scratch/2025_STB/2025_cell_painting_sperm

conda activate clipn


# Set the base directory where outputs will be written
BASE_DIR=~/scratch/2025_STB/2025_cell_painting_sperm
DATASET_CSV=dataset.txt
MODES=("reference_only" "integrate_all")
EPOCHS=(100 250 300 500 1000)
LATENT_DIMS=(10 20 50 100)
REFERENCE_DATASET="STBV1"

# Full paths to your scripts
RUN_CLIPN_SCRIPT=$HOME/apps/cell_painting_sperm/run_clipn.py
UMAP_SCRIPT=$HOME/apps/cell_painting_sperm/scripts/UMAP_output_per_function.py
ANALYSIS_SCRIPT=$HOME/apps/cell_painting_sperm/analyse_clipn_results.py
PLOT_SCRIPT=$HOME/apps/cell_painting_sperm/plot_clipn_results.py

for MODE in "${MODES[@]}"; do
  for EPOCH in "${EPOCHS[@]}"; do
    for LATENT_DIM in "${LATENT_DIMS[@]}"; do

      EXP_NAME="SelleckChem_vs_stb_${MODE}_E${EPOCH}_L${LATENT_DIM}"
      OUT_DIR="${BASE_DIR}/${EXP_NAME}"
      POST_DIR="${OUT_DIR}/post_clipn"
      LATENT_FILE="${EXP_NAME}_CLIPn_latent_representations_with_cpd_id.tsv"

      echo "=== Running CLIPn: $EXP_NAME ==="

      python "$RUN_CLIPN_SCRIPT" \
        --datasets_csv "$DATASET_CSV" \
        --out "$OUT_DIR" \
        --experiment "$EXP_NAME" \
        --mode "$MODE" \
        --epoch "$EPOCH" \
        --latent_dim "$LATENT_DIM" \
        --annotations STBV1_and_2_10uM_10032024.csv

      echo "--- Running UMAP output ---"
      MIN_DISTS=(0.1 0.2 0.3 0.4 0.5 0.6)
      COLOUR_BY=("Library" "Dataset" "cpd_type")
      CLUSTERS=("None" 15 20 25 30)

      for DIST in "${MIN_DISTS[@]}"; do
        for CLUSTER_COUNT in "${CLUSTERS[@]}"; do
          for COLOUR in "${COLOUR_BY[@]}"; do

            # Label to tag output files/folders
            if [[ "$CLUSTER_COUNT" == "None" ]]; then
              CLUSTER_LABEL="kNone"
              CLUSTER_ARGS=()
            else
              CLUSTER_LABEL="k${CLUSTER_COUNT}"
              CLUSTER_ARGS=(--num_clusters "$CLUSTER_COUNT")
            fi

            # Custom output directory that includes cluster setting
            OUTDIR="${POST_DIR}/UMAP_${CLUSTER_LABEL}"

            python "$UMAP_SCRIPT" \
              --input "${POST_DIR}/${LATENT_FILE}" \
              --add_labels \
              --output_dir "$OUTDIR" \
              --umap_min_dist "$DIST" \
              --colour_by "$COLOUR" \
              "${CLUSTER_ARGS[@]}"

          done
        done
      done


      echo "--- Running CLIPn post analysis ---"
      python "$ANALYSIS_SCRIPT" \
        --latent_csv "${POST_DIR}/${LATENT_FILE}" \
        --output_dir "${POST_DIR}/post_analysis_script" \
        --reference_dataset "$REFERENCE_DATASET"

      echo "--- Running cosine UMAP plots ---"
      python "$PLOT_SCRIPT" \
        --latent_csv "${POST_DIR}/${LATENT_FILE}" \
        --plots "${POST_DIR}/cosine_UMAP" \
        --umap_metric cosine \
	      --include_heatmap

      echo "=== Finished $EXP_NAME ==="
      echo ""

    done
  done
done
