#!/usr/bin/env python3
"""
merge_cellprofiler_by_intersection.py

Robustly merge multiple CellProfiler per-object tables by
- dropping all-NA columns in each file,
- taking the intersection of all remaining columns,
- reordering columns for consistent merge.

Outputs merged table, logs all drops, and saves a summary.

Usage:
    python merge_cellprofiler_by_intersection.py \
        --input_files file1.tsv file2.tsv ... \
        --output_file merged.tsv \
        --metadata_cols cpd_id,cpd_type,Plate_Metadata,Well_Metadata
"""

import argparse
import logging
import os
import sys
import pandas as pd

def parse_args():
    """
    Parse command-line arguments for the merge process.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge CellProfiler tables by intersecting columns."
    )
    parser.add_argument('--input_files', nargs='+', required=True, help='Input files (TSV/CSV)')
    parser.add_argument('--output_file', required=True, help='Output merged file (TSV)')
    parser.add_argument('--metadata_cols', default="cpd_id,cpd_type,Plate_Metadata,Well_Metadata", help='Comma-separated metadata columns')
    parser.add_argument('--log_level', default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set logging level')
    return parser.parse_args()

def setup_logging(log_level="INFO"):
    """
    Set up logging to file and console.

    Parameters
    ----------
    log_level : str
        Logging level ("DEBUG", "INFO", etc.)

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger("merge_logger")
    logger.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler("merge_intersection.log", mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    return logger

def robust_read_table(file_path, logger):
    """
    Read a TSV/CSV file into a pandas DataFrame, drop all-NA columns.

    Parameters
    ----------
    file_path : str
        Path to file.
    logger : logging.Logger
        Logger for logging.

    Returns
    -------
    pandas.DataFrame
        DataFrame with all-NA columns dropped.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    sep = "\t" if ext in [".tsv", ".txt"] else ","
    try:
        df = pd.read_csv(file_path, sep=sep)
        # Drop possible index col
        if "Unnamed: 0" in df.columns:
            logger.info(f"Dropping index column 'Unnamed: 0' in '{file_path}'.")
            df = df.drop(columns=["Unnamed: 0"])
        # Drop all-NA columns
        na_cols = df.columns[df.isna().all()].tolist()
        if na_cols:
            logger.info(f"Dropping {len(na_cols)} all-NA columns in '{file_path}': {na_cols}")
            df = df.drop(columns=na_cols)
        logger.info(f"Loaded '{file_path}' shape after NA drop: {df.shape}")
        return df
    except Exception as exc:
        logger.error(f"Could not read '{file_path}': {exc}")
        return None

def main():
    """
    Merge workflow: load files, drop NA columns, intersect, reorder, merge.
    """
    args = parse_args()
    logger = setup_logging(args.log_level)
    logger.info(f"Input files: {args.input_files}")
    logger.info(f"Output file: {args.output_file}")

    metadata_cols = [c.strip() for c in args.metadata_cols.split(",")]
    dfs = []
    columns_per_file = []
    summary = []

    # Load and clean files
    for file_path in args.input_files:
        df = robust_read_table(file_path, logger)
        if df is None:
            logger.error(f"Skipping file due to read error: {file_path}")
            continue
        missing = [c for c in metadata_cols if c not in df.columns]
        if missing:
            logger.error(f"File '{file_path}' missing required metadata: {missing}")
            summary.append({'file': file_path, 'status': 'failed', 'missing_metadata': missing, 'columns': list(df.columns)})
            continue
        columns_per_file.append(set(df.columns))
        dfs.append((file_path, df))
        summary.append({'file': file_path, 'status': 'ok', 'missing_metadata': [], 'columns': list(df.columns)})

    if not dfs:
        logger.error("No files with all metadata present. Aborting.")
        sys.exit(1)

    # Intersect columns
    common_cols = set.intersection(*columns_per_file)
    logger.info(f"{len(common_cols)} columns in intersection across all files.")
    missing_meta = [c for c in metadata_cols if c not in common_cols]
    if missing_meta:
        logger.error(f"Metadata columns missing from intersection: {missing_meta}")
        sys.exit(1)

    # Reorder and align all DataFrames to the same columns
    ordered_cols = metadata_cols + [c for c in common_cols if c not in metadata_cols]
    merged_dfs = []
    for file_path, df in dfs:
        # Reorder columns
        aligned = df.loc[:, ordered_cols]
        merged_dfs.append(aligned)

    merged = pd.concat(merged_dfs, axis=0, ignore_index=True)
    logger.info(f"Merged DataFrame shape: {merged.shape}")

    # Drop duplicates if present
    if merged.columns.duplicated().any():
        dup_cols = merged.columns[merged.columns.duplicated()].tolist()
        logger.warning(f"Dropping duplicate columns: {dup_cols}")
        merged = merged.loc[:, ~merged.columns.duplicated()]

    merged.to_csv(args.output_file, sep="\t", index=False)
    logger.info(f"Saved merged output to: {args.output_file}")

    summary_file = os.path.splitext(args.output_file)[0] + "_merge_summary.tsv"
    pd.DataFrame(summary).to_csv(summary_file, sep="\t", index=False)
    logger.info(f"Per-file column summary saved to: {summary_file}")
    logger.info("Merge complete.")

if __name__ == "__main__":
    main()
