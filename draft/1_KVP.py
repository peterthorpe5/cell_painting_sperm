#!/usr/bin/env python
# coding: utf-8

"""
Generate CSV Files for OMERO Annotation
---------------------------------------
This script generates CSV files for annotating data in OMERO.

Key Features:
- Extracts source and destination plate barcodes from ECHO transfer files.
- Loads and integrates STB annotation data.
- Generates a 384-well dataframe mapping transferred and control compounds.
- Outputs the annotated dataset to CSV.
"""

import pandas as pd
import os
import logging
from itertools import product
import string

# Configure logging
logging.basicConfig(
    filename='kvp.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

logger.info("Script started.")

def extract_barcodes(echo_file: str) -> tuple:
    """
    Extract barcodes from the header of an ECHO transfer file using column headers.

    Parameters
    ----------
    echo_file : str
        Path to the ECHO transfer file.

    Returns
    -------
    tuple
        Contains destination plate and source plate barcodes.
    """
    logger.debug(f"Extracting barcodes from {echo_file}")
    df = pd.read_csv(echo_file, nrows=1)
    if not {'Source Plate Barcode', 'Destination Plate Barcode'}.issubset(df.columns):
        logger.error(f"Required columns not found in file {echo_file}")
        raise ValueError(f"Incorrect file format for {echo_file}")
    source_plate = df.at[0, 'Source Plate Barcode']
    destination_plate = df.at[0, 'Destination Plate Barcode']
    logger.debug(f"Extracted: Dest={destination_plate}, Source={source_plate}")
    return destination_plate, source_plate

def process_echo_file(echo_file: str) -> pd.DataFrame:
    """
    Read ECHO transfer CSV and extract barcode data.

    Parameters
    ----------
    echo_file : str
        Path to the ECHO transfer CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing processed barcode data.
    """
    df = pd.read_csv(echo_file, skip_blank_lines=False)
    dest_plate, source_plate = extract_barcodes(echo_file)
    df['Source_Plate_Barcode'] = source_plate
    df['Destination_Plate_Barcode'] = dest_plate
    return df

def create_well_grid(input_dict: dict) -> pd.DataFrame:
    """
    Generate Cartesian product for well mapping.

    Parameters
    ----------
    input_dict : dict
        Dictionary with keys as column names and values as iterables for Cartesian product.

    Returns
    -------
    pd.DataFrame
        DataFrame with all possible well mappings.
    """
    combinations = product(*input_dict.values())
    return pd.DataFrame.from_records(combinations, columns=input_dict.keys())

# Load STB annotation data
try:
    stb = pd.read_csv('/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/13022025/22EP0009.csv')
    stb.rename(columns={'OLPTID': 'Source_Plate_Barcode', 'well_number': 'Source_well'}, inplace=True)
    logger.info(f"Loaded STB data ({len(stb)} rows).")
except FileNotFoundError as e:
    logger.error(f"STB annotation file not found: {e}")
    raise

# Locate and process ECHO transfer files
input_folder = '/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/13022025/Echo_logs/'
input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
               if f.startswith('Transfer') and f.endswith('.csv')]

if not input_files:
    logger.error("No ECHO files found.")
    raise FileNotFoundError("No ECHO transfer files located.")

logger.info(f"Processing {len(input_files)} ECHO files.")
echo_transfer_df = pd.concat([process_echo_file(f) for f in input_files])

echo_transfer_df.rename(columns={
    "Source Plate Barcode": "Source_Plate_Barcode",
    "Source Well": "Source_well",
    "Destination Plate Barcode": "Destination_Plate_Barcode",
    "Destination Well": "well_number"
}, inplace=True)

# Remove duplicated columns explicitly
echo_transfer_df = echo_transfer_df.loc[:, ~echo_transfer_df.columns.duplicated()]
logger.debug(f"Columns after processing: {echo_transfer_df.columns.tolist()}")

required_columns = ['Destination_Plate_Barcode', 'well_number', 'Source_Plate_Barcode', 'Source_well']
missing_cols = [col for col in required_columns if col not in echo_transfer_df.columns]
if missing_cols:
    logger.error(f"Missing required columns: {missing_cols}")
    raise KeyError(f"Missing required columns: {missing_cols}")

plate_barcodes = echo_transfer_df['Destination_Plate_Barcode'].dropna().unique()
letters = string.ascii_uppercase[:16]
numbers = range(1, 25)
well_numbers = [f"{x}{str(y).zfill(2)}" for x, y in product(letters, numbers)]

# Merge datasets to create final dataframe
ddf = (create_well_grid({'Destination_Plate_Barcode': plate_barcodes, 'well_number': well_numbers})
       .merge(echo_transfer_df, how='left', on=['Destination_Plate_Barcode', 'well_number'], suffixes=('', '_echo'))
       .merge(stb, how='left', on=['Source_Plate_Barcode', 'Source_well'], suffixes=('', '_stb'))
       .fillna({'name': 'empty'})
       .rename(columns={'Destination_Plate_Barcode': 'Plate', 'well_number': 'Well'}))

if ddf.empty:
    logger.warning("Final dataset is empty. Check input files and merging logic.")
else:
    ddf.to_csv('STB_kvp.csv', index=False)
    logger.info("OMERO annotation file saved successfully.")

logger.info("Script completed successfully.")
