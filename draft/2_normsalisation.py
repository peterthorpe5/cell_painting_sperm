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


Overview
This script processes Cell Painting feature data by performing multi-level normalisation. 
The goal is to remove unwanted technical variations while preserving meaningful 
biological differences. The normalisation process consists of the following key steps:


Cell Painting Data Normalisation Pipeline
1) Cell-Level Normalisation
Method: Dividing by Median Absolute Deviation (MAD) of control wells
Purpose: Adjusts features at the individual cell level using control wells.

2) Plate-Level Normalisation
Method: Subtracting plate-specific median and dividing by plate-specific MAD
Purpose: Accounts for plate-specific biases by standardising features within each plate.

3) Control-Well Standard Deviation Scaling
Method: Dividing by the standard deviation of control wells (DMSO) per plate
Purpose: Ensures feature values are scaled relative to their plate's control wells.

4) Per-Plate Standardisation using StandardScaler
Method: Subtracting per-plate mean and dividing by per-plate standard deviation
Purpose: Final adjustment to remove remaining batch effects.
Each step helps to ensure that features are comparable within and across plates
 while minimising unwanted variability.

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
    # Construct the full file path
    file_path = input_directory / file_name

    # Check if the input file exists; if not, log an error and skip this file
    if not file_path.exists():
        logger.error(f"Missing input file: {file_path}. Skipping.")
        continue

    # Log that processing has started for the current file
    logger.info(f"Processing file: {file_name}")

    # Load the CellProfiler feature data into a DataFrame
    input_df = pd.read_csv(file_path)

    # Extract the organelle name from the filename 
    # (e.g., 'FilteredNuclei' from 'MyExpt_test_FilteredNuclei.csv.gz')
    organelle_key = file_name.lower().split('_')[2].split('.')[0]

    # Map the extracted organelle name to a standardised version
    organelle_map = {'filterednuclei': 'nucleus', 'mitochondria': 'mitochondria', 'acrosome': 'acrosome'}
    organelle_name = organelle_map.get(organelle_key, organelle_key)  # Default to extracted name if not in map

    # Log organelle data loading details
    logger.info(f"Organelle {organelle_name} data loaded with shape: {input_df.shape}")
    logger.debug(f"Organelle {organelle_name} data sample:\n{input_df.head()}")

    # Merge the feature data with metadata, keeping only necessary columns and setting the index
    combined_df = (
        input_df
        .merge(metadata_df[['ImageNumber', 'Plate_Metadata', 'Well_Metadata']], on='ImageNumber')
        .set_index(['Plate_Metadata', 'Well_Metadata'])  # Organise data by plate and well
    )

    # Identify DMSO control wells (wells containing "23" in their metadata)
    dmso_wells = combined_df.query('Well_Metadata.str.contains("23")', engine='python')

    # If no control wells are found, log a warning and skip normalisation for this file
    if dmso_wells.empty:
        logger.warning("No DMSO control wells found. Skipping normalisation for this file.")
        continue

    # Compute the median feature values for DMSO control wells per plate
    plate_median = dmso_wells.groupby(level='Plate_Metadata').median()

    # Compute the Median Absolute Deviation (MAD) for DMSO control wells per plate
    plate_mad = dmso_wells.groupby(level='Plate_Metadata').agg(compute_mad)


    # Handle MAD values close to zero
    plate_mad.replace(to_replace=0, value=np.nan, inplace=True)

    normalised_df = combined_df.subtract(plate_median).div(plate_mad).fillna(0)

    #############################################################
    # important to normalise to the std of the controls per plate
    # Compute standard deviation of DMSO control wells per plate

    dmso_std = (
        normalised_df
        .query('Well_Metadata.str.contains("23")', 
               engine='python')  # Filter for control wells (DMSO)
        .groupby(level='Plate_Metadata')  # Group by plate
        .std()  # Compute standard deviation per feature per plate
    )

    # Apply this standard deviation to normalised_df before StandardScaler
    # This divides each feature's value by the corresponding standard deviation 
    # from the DMSO wells in that plate.
    # The operation is done plate-wise, so each plate's data is divided only by 
    # its own control standard deviation.
    normalised_df = normalised_df.div(dmso_std)


    #############################################################
    # Standardisation per plate
    # Removes systematic biases unique to individual plates (batch effects)
    # but Might introduce artificial differences, complicating downstream analysis
    # if genuine plate-level differences exist.
    logger.info("Applying StandardScaler per plate.")
    standardised_dfs = []
    # Loop through each plate in the normalised dataset
    for plate_id, plate_df in normalised_df.groupby(level='Plate_Metadata'):
        
        # Initialise a StandardScaler instance for per-plate normalisation
        # StandardScaler transforms numerical data so that each feature (column) 
        # has a mean of 0 and a standard deviation of 1. This is done to ensure 
        # that all features contribute equally to downstream analyses, preventing 
        # features with larger numerical ranges from dominating those with smaller ranges.
        # removes systematic differences between plates while maintaining relative 
        # differences within each plate, making the dataset more comparable across 
        # experimental batches. This step is crucial for reducing plate-to-plate variability
        scaler = StandardScaler()

        # Apply StandardScaler transformation to standardise the features
        # This ensures that each feature within a plate has a mean of 0 and a standard deviation of 1
        standardised_plate_df = pd.DataFrame(
            scaler.fit_transform(plate_df),  # Standardisation of the features
            index=plate_df.index,  # Preserve the original multi-index structure
            columns=plate_df.columns  # Maintain original feature names
        )

        # Append the standardised DataFrame for this plate to the list
        standardised_dfs.append(standardised_plate_df)

        # Log the standardisation process for the current plate
        logger.info(f"Standardised plate {plate_id} with shape: {plate_df.shape}")
        logger.debug(f"Standardised plate {plate_id} sample:\n{standardised_plate_df.head()}")

    # Concatenate all standardised plates into a single DataFrame
    final_df = pd.concat(standardised_dfs)

    # Check for any NaN values in the final dataset and log a warning if found
    if final_df.isnull().values.any():
        logger.warning(f"NaN values detected in final dataframe for {organelle_name}. "
                       "Check input data and normalisation steps.")

    # Define the output file path for saving the processed data
    output_file_path = output_directory / f"{screen_identifier}_{organelle_name}.csv"

    # Save the final standardised dataset as a CSV file
    final_df.to_csv(output_file_path)

    # Log the successful saving of the processed data
    logger.info(f"Saved normalised data to {output_file_path}")

    # Log memory usage
    memory_used = psutil.virtual_memory().used / (1024 ** 3)
    logger.info(f"Memory usage after processing {file_name}: {memory_used:.2f} GB")

logger.info("Normalisation completed successfully.")
print("Normalisation completed successfully. Processed data saved.")
