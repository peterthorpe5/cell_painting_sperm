#!/bin/bash

cd /Users/PThorpe001/Desktop/cell_painting/STB

set -e

# conda activate pytorch_anomaly

# List of folder names
FOLDERS=(NPSCDD000401 NPSCDD0003971 NPSCDD0003972 NPSCDD000400)
PLATES=(NPSCDD000401 NPSCDD0003971 NPSCDD0003972 NPSCDD000400)


# Absolute path to the script and metadata file

METADATA="KVP_4Plates_04022025.csv"




# Step 1: Aggregate CellProfiler data for each plate
for FOLDER in "${FOLDERS[@]}"; do
    echo "Processing $FOLDER..."
    python /Users/PThorpe001/github_repos/cell_painting_sperm/clipn/impute_missing_data_feature_select_image_level.py \
        --input_dir "$FOLDER" \
        --output_file "$FOLDER/image_level.tsv" \
        --metadata_file "$METADATA"
    if [[ $? -ne 0 ]]; then
        echo "Aggregation failed for $FOLDER. Exiting." >&2
        exit 1
    fi
done

# Step 2: Merge all per-plate well-level profiles
MERGED_OUT="STB_all_plates_image_level.tsv"
MERGED_LOG="image_level_merge_profiles.log"

echo "Merging all per-plate profiles into $MERGED_OUT..."
python /Users/PThorpe001/github_repos/cell_painting_sperm/anomaly_detection/merge_cellprofiler_well_level_profiles.py \
    --input_files $(for FOLDER in "${PLATES[@]}"; do echo -n "$FOLDER/image_level.tsv "; done) \
    --output_file "$MERGED_OUT" 
if [[ $? -ne 0 ]]; then
    echo "Merging failed. See $MERGED_LOG for details." >&2
    exit 1
fi

echo "Merged profile shape:"
head -2 "$MERGED_OUT" | tail -1 | awk -F'\t' '{print NF " columns"}'

# Step 3: Prepare the anomaly detection input
ANOMALY_OUTDIR="STB_data_split_image_level"
echo "Running anomaly detection data prep..."
python /Users/PThorpe001/github_repos/cell_painting_sperm/anomaly_detection/prepare_anomaly_detection_data.py \
    --input_file "$MERGED_OUT" \
    --output_dir "$ANOMALY_OUTDIR" \
    --control_label DMSO \
    --experiment STB \
    --zscore_method median \
    --impute knn \
    --scale_per_plate \
    --scale_method auto \
    --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
    --na_cutoff 0.05 --corr_threshold 0.9 --unique_cutoff 0.01 --freq_cut 0.05

if [[ $? -ne 0 ]]; then
    echo "Anomaly detection prep failed." >&2
    exit 1
fi

echo "All done! Outputs in $ANOMALY_OUTDIR"
