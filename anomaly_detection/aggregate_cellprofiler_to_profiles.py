#!/usr/bin/env python3
# coding: utf-8

"""
aggregate_cellprofiler_to_profiles.py

Aggregate multiple CellProfiler CSVs (one per channel/compartment) into a single wide-format
well-level profile using pycytominer. Optionally merges in a metadata file and harmonises compound type labels.

Requires:
    pandas, pycytominer

Example usage:
    python aggregate_cellprofiler_to_profiles.py \
        --input_dir ./raw_cp/ \
        --output_file cellprofiler_well_profiles.tsv \
        --metadata_file plate_map.csv \
        --merge_keys Plate,Well
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from pycytominer.aggregate import aggregate
from anomaly_detection.library import  (harmonise_cpd_type_column, 
                                        harmonise_metadata_columns,
                                        harmonise_plate_well_columns)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Aggregate CellProfiler CSVs to well-level profile.")
    parser.add_argument('--input_dir', required=True, help='Directory with CellProfiler CSV files.')
    parser.add_argument('--output_file', required=True, help='Output file (TSV or CSV).')
    parser.add_argument('--metadata_file', default=None, help='Optional metadata file to merge.')
    parser.add_argument('--merge_keys', default='Plate,Well',
                        help='Comma-separated list of metadata columns for merging (default: Plate,Well).')
    parser.add_argument('--sep', default='\t', help='Delimiter for output file (default: tab).')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set logging level (default: INFO)')
    return parser.parse_args()

def setup_logging(log_level="INFO"):
    """
    Set up logging to console.
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("agg_logger")


def main():
    """
    Main workflow for aggregation.
    """
    args = parse_args()
    logger = setup_logging(args.log_level)
    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")

    # Find all CSVs in the directory
    input_dir = Path(args.input_dir)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {args.input_dir}")
        return

    logger.info(f"Found {len(csv_files)} CellProfiler CSV files: {[f.name for f in csv_files]}")

    # Read and concatenate all CSVs
    dataframes = [pd.read_csv(f) for f in csv_files]
    cp_df = pd.concat(dataframes, axis=0, ignore_index=True)
    logger.info(f"Concatenated DataFrame shape: {cp_df.shape}")

    # Defensive cleaning: replace bad strings with NaN
    # bad data has been found :(
    bad_strings = ['#NAME?', '#VALUE!', '#DIV/0!', 'N/A', 'NA', '', ' ']
    cp_df.replace(bad_strings, np.nan, inplace=True)
    logger.info("Replaced known bad strings in numeric columns with NaN.")

    # Harmonise plate/well columns before aggregation
    cp_df, plate_col, well_col = harmonise_plate_well_columns(
        cp_df,
        logger=logger,
        desired_plate="Plate_Metadata",
        desired_well="Well_Metadata"
    )
    meta_cols = set([plate_col, well_col, "ImageNumber", "ObjectNumber"])
    feature_cols = [c for c in cp_df.columns if c not in meta_cols]
    logger.info(f"Using plate column: {plate_col}, well column: {well_col}")
    logger.debug(f"Using {len(feature_cols)} feature columns for aggregation.")

    # Aggregate to well-level using pycytominer
    logger.info("Aggregating to well-level profiles (using mean for each well)...")
    agg_df = aggregate(
        population_df=cp_df,
        strata=[plate_col, well_col],
        features=feature_cols,
        operation="mean"
    )
    logger.info(f"Aggregated DataFrame shape: {agg_df.shape}")

    # Merge metadata if provided
    if args.metadata_file is not None:
        logger.info(f"Loading and harmonising metadata file: {args.metadata_file}")
        meta_df = pd.read_csv(args.metadata_file)
        meta_df = harmonise_metadata_columns(meta_df, logger=logger)
        # Harmonise metadata table columns as well
        meta_df, meta_plate_col, meta_well_col = harmonise_plate_well_columns(
            meta_df,
            logger=logger,
            desired_plate=plate_col,
            desired_well=well_col
        )
        keys = [plate_col, well_col]
        logger.info(f"Merging on keys: {keys}")
        # Defensive check for merge keys
        for k in keys:
            if k not in agg_df.columns:
                logger.error(f"Aggregated data is missing merge key: {k}")
                return
            if k not in meta_df.columns:
                logger.error(f"Metadata is missing merge key: {k}")
                return

        agg_df = agg_df.merge(meta_df, how="left", on=keys)
        logger.info(f"Shape after metadata merge: {agg_df.shape}")

        # After merging metadata, check for essential columns
        required_cols = ["cpd_id", "cpd_type", plate_col, well_col]
        if "Library" in meta_df.columns:
            required_cols.append("Library")
        missing_cols = [col for col in required_cols if col not in agg_df.columns]
        if missing_cols:
            logger.warning(
                f"Missing expected metadata columns in output: {missing_cols}.\n"
                "Check your metadata file and merge_keys settings."
            )
        else:
            logger.info("All required metadata columns are present in the final output.")

    # Always harmonise cpd_type (rename and create clean version)
    agg_df = harmonise_cpd_type_column(
        agg_df,
        id_col="cpd_id",
        cpd_type_col="cpd_type",
        logger=logger
    )

    # Output as TSV or CSV
    out_path = Path(args.output_file)
    agg_df.to_csv(out_path, sep=args.sep, index=False)
    logger.info(f"Saved merged profiles to {out_path}")
    logger.info(f"Output file shape: {agg_df.shape}")
    logger.info("Aggregation complete.")



if __name__ == "__main__":
    main()
