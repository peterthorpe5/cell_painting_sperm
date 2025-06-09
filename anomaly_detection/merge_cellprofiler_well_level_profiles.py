#!/usr/bin/env python3
# coding: utf-8

"""
merge_cellprofiler_profiles.py

Merge multiple CellProfiler well-level profile files into a single dataset for downstream analysis.

- Accepts an arbitrary number of TSV or CSV files as input.
- Validates that all files have identical columns.
- Preserves metadata and logs all merging steps and any issues.
- Outputs a merged file in tab-separated format.

Usage
-----
python merge_cellprofiler_profiles.py \
    --input_files plate1_profiles.tsv plate2_profiles.tsv plate3_profiles.tsv \
    --output_file merged_profiles.tsv
"""

import argparse
import sys
import logging
import os
import pandas as pd

def parse_args():
    """
    Parse command-line arguments for merging CellProfiler profile files.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Merge CellProfiler profile files (TSV/CSV) with logging.")
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='Input files (TSV/CSV) to merge.')
    parser.add_argument('--output_file', required=True,
                        help='Output merged file (TSV).')
    return parser.parse_args()

def setup_logging():
    """
    Set up logging to file and console.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    logger = logging.getLogger("merge_logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("merge_cellprofiler_profiles.log", mode='w')
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def main():
    """
    Main workflow for merging CellProfiler well-level profile files.
    """
    args = parse_args()
    logger = setup_logging()

    logger.info(f"Input files: {args.input_files}")
    merged_df = None
    columns_ref = None

    for i, file in enumerate(args.input_files):
        ext = os.path.splitext(file)[-1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        try:
            df = pd.read_csv(file, sep=sep)
            logger.info(f"Loaded '{file}' shape: {df.shape}")
        except Exception as e:
            logger.error(f"Could not read file '{file}': {e}")
            sys.exit(1)

        if columns_ref is None:
            columns_ref = df.columns.tolist()
            merged_df = df
        else:
            if not df.columns.tolist() == columns_ref:
                logger.error(f"Column mismatch in '{file}'. Columns differ from previous files.")
                logger.error(f"Expected columns: {columns_ref}")
                logger.error(f"Found columns: {df.columns.tolist()}")
                sys.exit(1)
            merged_df = pd.concat([merged_df, df], ignore_index=True)

    logger.info(f"Final merged DataFrame shape: {merged_df.shape}")
    merged_df.to_csv(args.output_file, sep="\t", index=False)
    logger.info(f"Merged output saved to: {args.output_file}")

if __name__ == "__main__":
    main()
