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
import pandas as pd
from pycytominer.aggregate import aggregate

def harmonise_cpd_type_column(df, id_col="cpd_id", cpd_type_col="cpd_type", logger=None):
    """
    Harmonise compound type labelling, renaming the original and creating a cleaned 'cpd_type' column.
    """
    # Rename original cpd_type column
    if cpd_type_col in df.columns:
        df = df.rename(columns={cpd_type_col: "cpd_type_raw"})
        if logger:
            logger.info(f"Renamed original '{cpd_type_col}' column to 'cpd_type_raw'.")

    # Fill with blanks if missing
    cpd_type_vals = df.get("cpd_type_raw", pd.Series([""]*len(df))).fillna("").str.lower()
    cpd_id_vals = df.get(id_col, pd.Series([""]*len(df))).fillna("").str.strip().str.upper()

    new_types = []
    for orig, cid in zip(cpd_type_vals, cpd_id_vals):
        if "positive control" in orig:
            new_types.append("positive control")
        elif "negative control" in orig or cid == "DMSO":
            new_types.append("DMSO")
        elif "compound" in orig:
            new_types.append("compound")
        else:
            new_types.append("compound")
    df["cpd_type"] = new_types

    if logger:
        logger.info("Harmonised compound type column added as 'cpd_type'.")

    return df

def harmonise_metadata_columns(df, logger=None):
    """
    Harmonise column names in a metadata DataFrame to ensure compatibility
    with downstream merging, regardless of their original names.
    """
    rename_dict = {}
    col_map = {col.lower(): col for col in df.columns}

    # Plate harmonisation
    plate_candidates = ["plate_metadata", "plate"]
    for platename in plate_candidates:
        if platename in col_map:
            rename_dict[col_map[platename]] = "Plate_Metadata"
            if logger:
                logger.info(f"Renaming '{col_map[platename]}' to 'Plate_Metadata'")
            break

    # Well harmonisation
    well_candidates = ["well_metadata", "well"]
    for wellname in well_candidates:
        if wellname in col_map:
            rename_dict[col_map[wellname]] = "Well_Metadata"
            if logger:
                logger.info(f"Renaming '{col_map[wellname]}' to 'Well_Metadata'")
            break

    # cpd_id harmonisation (including many possible variants)
    cpd_candidates = [
        "cpd_id", "compound_id", "comp_id", "compound", "compud_id",
        "compund_id", "compid", "comp", "compoundid"
    ]
    for cpdname in cpd_candidates:
        if cpdname in col_map:
            rename_dict[col_map[cpdname]] = "cpd_id"
            if logger:
                logger.info(f"Renaming '{col_map[cpdname]}' to 'cpd_id'")
            break

    # cpd_type harmonisation (similar variants possible)
    cpd_type_candidates = ["cpd_type", "compound_type", "type"]
    for cpdtype in cpd_type_candidates:
        if cpdtype in col_map:
            rename_dict[col_map[cpdtype]] = "cpd_type"
            if logger:
                logger.info(f"Renaming '{col_map[cpdtype]}' to 'cpd_type'")
            break

    # Library harmonisation (optional)
    library_candidates = ["library", "lib", "collection"]
    for lib in library_candidates:
        if lib in col_map:
            rename_dict[col_map[lib]] = "Library"
            if logger:
                logger.info(f"Renaming '{col_map[lib]}' to 'Library'")
            break

    if logger and not rename_dict:
        logger.info("No metadata columns required renaming.")

    return df.rename(columns=rename_dict)

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

    # Aggregate to well-level using pycytominer
    logger.info("Aggregating to well-level profiles (using mean for each well)...")

    meta_cols = set([k.strip() for k in args.merge_keys.split(',')] + ["ImageNumber", "ObjectNumber"])
    feature_cols = [c for c in cp_df.columns if c not in meta_cols]
    logger.debug(f"Using {len(feature_cols)} feature columns for aggregation.")
    logger.debug(f"First five feature columns: {feature_cols[:5]}")

    agg_df = aggregate(
        population_df=cp_df,
        strata=[k.strip() for k in args.merge_keys.split(',')],
        features=feature_cols,
        operation="mean"
    )

    logger.info(f"Aggregated DataFrame shape: {agg_df.shape}")

    # Merge metadata if provided
    if args.metadata_file is not None:
        logger.info(f"Loading and harmonising metadata file: {args.metadata_file}")
        meta_df = pd.read_csv(args.metadata_file)
        meta_df = harmonise_metadata_columns(meta_df, logger=logger)

        keys = ["Plate_Metadata", "Well_Metadata"]
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
        required_cols = ["cpd_id", "cpd_type", "Plate_Metadata", "Well_Metadata"]
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
