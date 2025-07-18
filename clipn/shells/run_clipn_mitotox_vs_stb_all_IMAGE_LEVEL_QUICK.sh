#!/bin/bash
#SBATCH -J STB_mtox_ref_clipn   #jobname
#SBATCH --partition=gpu
#SBATCH --mem=20GB


cd $HOME/scratch/2025_STB/2025_cell_painting_sperm

conda activate clipn


# Set the base directory where outputs will be written
BASE_DIR=~/scratch/image_level
DATASET_CSV=/mnt/shared/projects/uosa/Pete_Thorpe/2025_cell_painting_sperm/image_level_data/datasets/dataset_M_S_S_Image_level.csv
#MODES=("reference_only" "integrate_all")
MODES=("reference_only")
EPOCHS=(100)
LATENT_DIMS=(20)
REFERENCE_DATASET="STB"

# Full paths to your scripts
RUN_CLIPN_SCRIPT=$HOME/apps/cell_painting_sperm/clipn/run_clipn.py
UMAP_SCRIPT=$HOME/apps/cell_painting_sperm/clipn/scripts/UMAP_output_per_function.py
ANALYSIS_SCRIPT=$HOME/apps/cell_painting_sperm/clipn/analyse_clipn_results.py
PLOT_SCRIPT=$HOME/apps/cell_painting_sperm/clipn/plot_clipn_results.py

for MODE in "${MODES[@]}"; do
  for EPOCH in "${EPOCHS[@]}"; do
    for LATENT_DIM in "${LATENT_DIMS[@]}"; do

      EXP_NAME="Image_level_Sellle_STB_vs_mitotox_${MODE}_E${EPOCH}_L${LATENT_DIM}"
      OUT_DIR="${BASE_DIR}/${EXP_NAME}"
      POST_DIR="${OUT_DIR}/post_clipn"
      LATENT_FILE="${EXP_NAME}_CLIPn_latent_aggregated_median.tsv"

      echo "=== Running CLIPn: $EXP_NAME ==="

      python "$RUN_CLIPN_SCRIPT" \
        --datasets_csv "$DATASET_CSV" \
        --out "$OUT_DIR" \
        --experiment "$EXP_NAME" \
        --mode "$MODE" \
        --epoch "$EPOCH" \
        --latent_dim "$LATENT_DIM" \
        --save_model \
        --aggregate_method median \
        --skip_standardise

      echo "--- Running UMAP output ---"
      MIN_DISTS=(0.1)
      COLOUR_BY=("Library" "Dataset" "cpd_type")
      CLUSTERS=("None", 15)

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
              "${CLUSTER_ARGS[@]}" --compound_metadata /home/pthorpe/scratch/2025_STB/2025_cell_painting_sperm/STB_annotation/combined_references_with_annotations.tsv

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
        --compound_metadata  \
        /home/pthorpe/scratch/2025_STB/2025_cell_painting_sperm/STB_annotation/combined_references_with_annotations.tsv \
        --interactive

      echo "=== Finished $EXP_NAME ==="
      echo ""

      python "$HOME/apps/cell_painting_sperm/clipn/scripts/summarise_mcp_neighbours.py" \
        --folder "$OUT_DIR" \
        --top_n 100 \
        --metadata /home/pthorpe/scratch/2025_STB/2025_cell_painting_sperm/STB_annotation/combined_references_with_annotations.tsv

    done
  done
done


# Set base directory and metadata file
METADATA="/mnt/shared/projects/uosa/Pete_Thorpe/2025_cell_painting_sperm/STB_annotation/combined_references_with_annotations.tsv"
SCRIPT="$HOME/apps/cell_painting_sperm/clipn/scripts/summarise_mcp_neighbours.py"

# Loop through all subdirectories
for FOLDER in "${BASE_DIR}"/*; do
    if [[ -d "$FOLDER" ]]; then
        echo ">>> Processing: $FOLDER"
        python "$SCRIPT" --folder "$FOLDER" --top_n 30 --metadata "$METADATA"  --extra_metadata /mnt/shared/projects/uosa/Pete_Thorpe/2025_cell_painting_sperm/STB_annotation/24CPSC-Map.csv
        echo ""
    fi
done

echo "=== All neighbour summaries attempted ==="
