#!/bin/bash
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
        --latent_dim "$LATENT_DIM"

      echo "--- Running UMAP output ---"
      python "$UMAP_SCRIPT" \
        --input "${POST_DIR}/${LATENT_FILE}" \
        --add_labels \
        --output_dir "${POST_DIR}/UMAP" \
        --colour_by Library

      python "$UMAP_SCRIPT" \
        --input "${POST_DIR}/${LATENT_FILE}" \
        --add_labels \
        --output_dir "${POST_DIR}/UMAP" \
        --colour_by Dataset

      python "$UMAP_SCRIPT" \
        --input "${POST_DIR}/${LATENT_FILE}" \
        --add_labels \
        --output_dir "${POST_DIR}/UMAP" \
        --colour_by cpd_type




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
