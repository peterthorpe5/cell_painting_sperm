#!/usr/bin/env python
# coding: utf-8

"""
New Sperm Painting Processing - IXM
-----------------------------------
This script processes sperm painting data from IXM, including:
    - Filtering potentially poor-quality wells using a random forest model.
    - Normalisation of cell-level features using DMSO median and MAD.
    - Plate-level standardisation of features prior to merging data.
    - Saving processed data for downstream analysis.
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
import psutil
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    filename='normalisation.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("Normalisation script started.")

# Check available system memory
available_memory = psutil.virtual_memory().available
logger.info(f"Available memory: {available_memory / (1024 ** 3):.2f} GB")


def compute_mad(array: np.ndarray) -> float:
    """
    Compute the Median Absolute Deviation (MAD) of an array.

    Parameters
    ----------
    array : np.ndarray
        Input numerical array.

    Returns
    -------
    float
        MAD value of the input array.
    """
    array = np.nan_to_num(array)
    return np.median(np.abs(array - np.median(array))) * 1.4826


# Define directories
input_directory = Path('/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/IXM_data/MCP09_ext_3751/Omero_raw/')
output_directory = Path('/uod/npsc/homes/pthorpe001/test_run')

# Load metadata
logger.info("Loading metadata file.")
metadata_df = pd.read_csv(input_directory / 'MyExpt_test_Image.csv.gz')
logger.info(f"Metadata loaded with shape: {metadata_df.shape}")
logger.debug(f"Metadata sample:\n{metadata_df.head()}")

# Load random forest model safely with compatibility check
logger.info("Attempting to load random forest model for quality filtering.")
try:
    random_forest_model = joblib.load('/uod/npsc/data/BMG/Sperm_painting/image_QC_rf_model.sav')
    model_loaded = True
    logger.info("Random forest model loaded successfully.")
except Exception as e:
    random_forest_model = None
    model_loaded = False
    logger.warning(f"Failed to load random forest model due to: {e}. Proceeding without filtering.")

# Apply filtering only if model loaded successfully
if model_loaded:
    filtered_metadata = (
        metadata_df
        .filter(regex='^ImageQ.*|(Field|Well|Plate).*Metadata')
        .set_index(['Plate_Metadata', 'Well_Metadata', 'Field_Metadata'])
    )

    filter_mask = random_forest_model.predict(filtered_metadata) == 'ok'
    logger.info(f"Filtered out {np.sum(~filter_mask)} wells identified as poor quality.")
    metadata_df = metadata_df[filter_mask].reset_index(drop=True)
else:
    logger.info("Skipping well-quality filtering due to model incompatibility.")

# Normalise and save data
cell_painting_files = [
    'MyExpt_test_FilteredNuclei.csv.gz',
    'MyExpt_test_Mitochondria.csv.gz',
    'MyExpt_test_Acrosome.csv.gz'
]
screen_identifier = 'NPSCDD003740_05032025'

for file_name in cell_painting_files:
    file_path = input_directory / file_name
    if not file_path.exists():
        logger.error(f"Missing input file: {file_path}. Skipping.")
        continue

    logger.info(f"Processing file: {file_name}")
    input_df = pd.read_csv(file_path)
    organelle_key = file_name.lower().split('_')[2].split('.')[0]
    organelle_map = {'filterednuclei': 'nucleus', 'mitochondria': 'mitochondria', 'acrosome': 'acrosome'}
    organelle_name = organelle_map.get(organelle_key, organelle_key)

    logger.info(f"Organelle {organelle_name} data loaded with shape: {input_df.shape}")
    logger.debug(f"Organelle {organelle_name} data sample:\n{input_df.head()}")

    combined_df = (
        input_df
        .merge(metadata_df[['ImageNumber', 'Plate_Metadata', 'Well_Metadata']], on='ImageNumber')
        .set_index(['Plate_Metadata', 'Well_Metadata'])
    )

    dmso_wells = combined_df.query('Well_Metadata.str.contains("23")', engine='python')
    if dmso_wells.empty:
        logger.warning("No DMSO control wells found. Skipping normalisation for this file.")
        continue

    plate_median = dmso_wells.groupby(level='Plate_Metadata').median()
    plate_mad = dmso_wells.groupby(level='Plate_Metadata').agg(compute_mad)

    # Handle MAD values close to zero
    plate_mad.replace(to_replace=0, value=np.nan, inplace=True)

    normalised_df = combined_df.subtract(plate_median).div(plate_mad).fillna(0)

    # Standardisation per plate
    # Removes systematic biases unique to individual plates (batch effects)
    # but Might introduce artificial differences, complicating downstream analysis
    # if genuine plate-level differences exist.
    logger.info("Applying StandardScaler per plate.")
    standardised_dfs = []
    for plate_id, plate_df in normalised_df.groupby(level='Plate_Metadata'):
        scaler = StandardScaler()
        standardised_plate_df = pd.DataFrame(
            scaler.fit_transform(plate_df),
            index=plate_df.index,
            columns=plate_df.columns
        )
        standardised_dfs.append(standardised_plate_df)
        logger.info(f"Standardised plate {plate_id} with shape: {plate_df.shape}")
        logger.debug(f"Standardised plate {plate_id} sample:\n{standardised_plate_df.head()}")

    final_df = pd.concat(standardised_dfs)

    if final_df.isnull().values.any():
        logger.warning(f"NaN values detected in final dataframe for {organelle_name}. Check input data and normalisation steps.")

    output_file_path = output_directory / f"{screen_identifier}_{organelle_name}.csv"
    final_df.to_csv(output_file_path)
    logger.info(f"Saved normalised data to {output_file_path}")

    # Log memory usage
    memory_used = psutil.virtual_memory().used / (1024 ** 3)
    logger.info(f"Memory usage after processing {file_name}: {memory_used:.2f} GB")

logger.info("Normalisation completed successfully.")
print("Normalisation completed successfully. Processed data saved.")
