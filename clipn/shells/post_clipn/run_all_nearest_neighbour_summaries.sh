#!/bin/bash
#SBATCH -J summarise_mcp
#SBATCH --mem=16GB

# Activate environment
conda activate clipn

# Set base directory and metadata file
BASE_DIR="/home/pthorpe/scratch/image_level"
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
