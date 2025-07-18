#!/bin/bash
cd /mnt/shared/projects/uosa/Pete_Thorpe/2025_cell_painting_sperm/shells

# Set the path to your comparison script
SCRIPT="/home/pthorpe/apps/cell_painting_sperm/scripts/compare_clipn_results.py"

# Base directory containing CLIPn output folders
BASE_DIR="/mnt/shared/projects/uosa/Pete_Thorpe/2025_cell_painting_sperm/"

# Compound filter
COMPOUND_ID="MCP|DDU"

# Preferred latent dimension to compare on
LATENT_DIM=20

# List of baseline prefixes to compare against
#BASELINES=(
 #   "STB_vs_mitotox_reference_only"
  #  "STB_vs_mitotox_integrate_all"
   # "SelleckChem_vs_stb_reference_only"
   # "SelleckChem_vs_stb_integrate_all"
#)

BASELINES=("STB_vs_mitotox_reference_only")

# Loop over baseline prefixes
for PREFIX in "${BASELINES[@]}"; do
    echo "Running comparison for baseline: $PREFIX (L${LATENT_DIM})"
    python "$SCRIPT" \
        --base_dir "$BASE_DIR" \
        --compound_id "$COMPOUND_ID" \
        --baseline_prefix "$PREFIX" \
        --preferred_latent "$LATENT_DIM" \
        --plot_heatmap   --output_dir results_${PREFIX}
    echo ""
done
