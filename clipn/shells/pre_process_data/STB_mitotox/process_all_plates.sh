#!/bin/env bash

##$ -adds l_hard gpu 1
##$ -adds l_hard cuda.0.name 'NVIDIA A40'
#$ -j y
#$ -N process_all_plates
#$ -l h_vmem=200G
#$ -cwd


cd /home/pthorpe/scratch/laura

set -Eeuo pipefail
IFS=$'\n\t'
mkdir -p logs

# Base paths
BASE="/home/pthorpe/scratch/laura"
METADATA_DIR="${BASE}/metadata"

# Tool/script path
PY_SCRIPT="${HOME}/apps/cell_painting_sperm/clipn/impute_missing_data_feature_select_object_level.py"

# Parameters (adjust as needed)
CORR="0.98"
VAR="0.01"
SCALE_METHOD="none"
MERGE_KEYS="Plate_Metadata,Well_Metadata"

echo "Host: $(hostname)"


# conda activate clipn

# Iterate over all dataset folders that match HGTx-LSS*
for FOLDER in "${BASE}"/HGTx-LSS*/ ; do
  [[ -d "${FOLDER}" ]] || continue

  DATASET="$(basename "${FOLDER}")"            # e.g. HGTx-LSS02
  LIBRARY="${DATASET}"                         # matches folder name (with hyphen)
  METADATA_FILE="${METADATA_DIR}/${DATASET}-KVP.csv"

  # Verify metadata exists and is non-empty
  if [[ ! -s "${METADATA_FILE}" ]]; then
    echo "Error: Missing metadata for ${DATASET}: ${METADATA_FILE}" >&2
    exit 1
  fi

  # Build output filename (underscores, dataset-specific)
  stem="${DATASET//-/_}"                       # HGTx_LSS02
  OUT="${FOLDER%/}/${stem}_v${VAR}_c${CORR}_no_scale_imputed.tsv"

  if [[ -s "${OUT}" ]]; then
    echo "Skipping ${DATASET} (output already exists): ${OUT}"
    continue
  fi

  echo "Processing ${DATASET}"
  echo "  input_dir:      ${FOLDER%/}"
  echo "  metadata_file:  ${METADATA_FILE}"
  echo "  library:        ${LIBRARY}"
  echo "  output_file:    ${OUT}"

  python "${PY_SCRIPT}" \
    --input_dir "${FOLDER%/}" \
    --output_file "${OUT}" \
    --metadata_file "${METADATA_FILE}" \
    --library "${LIBRARY}" \
    --correlation_threshold "${CORR}" \
    --variance_threshold "${VAR}" \
    --scale_per_plate \
    --merge_keys "${MERGE_KEYS}" \
    --impute median \
    --per_object_output \
    --aggregate_per_well \
    --scale_method "${SCALE_METHOD}"

  echo "Done: ${DATASET}"
done

echo "All datasets processed."


echo "merge the plates"


python ~/apps/cell_painting_sperm/clipn/merge_cellprofiler_plates.py --input_files \
     ./HGTx-LSS02/HGTx_LSS02_v0.01_c0.98_no_scale_imputed_per_object.tsv \
     ./HGTx-LSS04/HGTx_LSS04_v0.01_c0.98_no_scale_imputed_per_object.tsv \
     ./HGTx-LSS03/HGTx_LSS03_v0.01_c0.98_no_scale_imputed_per_object.tsv --output_file HGTx-LSS0_234_v0.01_c0.98_no_scale_imputed_per_object.tsv

python split.py --output2 all_plates_together/HGTx_2.tsv --output1 all_plates_together/HGTx_1.tsv --input HGTx-LSS0_234_v0.01_c0.98_no_scale_imputed_per_object.tsv  --sort
