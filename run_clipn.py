#!/usr/bin/env python3
# coding: utf-8

"""
Run CLIPn Integration on Cell Painting Data
-------------------------------------------

This script:
- Loads and merges multiple reference and query datasets.
- Harmonises column features across datasets.
- Encodes labels for compatibility with CLIPn.
- Runs CLIPn integration analysis (either train on references or integrate all).
- Decodes labels post-analysis, restoring original annotations.
- Outputs results, including latent representations and similarity matrices.

Command-line arguments:
-----------------------
    --datasets_csv      : Path to CSV listing dataset names and paths.
    --out               : Directory to save outputs.
    --experiment        : Experiment name for file naming.
    --mode              : Operation mode ('reference_only' or 'integrate_all').

Logging:
--------
Logs detailed info and debug-level outputs.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import set_config
from cell_painting.run_cplin import (
    load_and_harmonise_datasets,
    encode_labels,
    decode_labels,
    run_clipn_placeholder
    )

set_config(transform_output="pandas")


def setup_logging(out_dir, experiment):
    """Configure logging with stream and file handlers."""
    log_dir = Path(out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"{experiment}_clipn.log"

    logger = logging.getLogger("clipn_logger")
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Experiment Name: {experiment}")

    return logger



def main(args):
    """Main function to execute CLIPn integration pipeline."""
    logger = setup_logging(args.out, args.experiment)

    dataframes, common_cols = load_and_harmonise_datasets(args.datasets_csv, logger)
    combined_df = pd.concat(dataframes.values(), keys=dataframes.keys(), names=['Dataset', 'Sample'])

    combined_df, encoders = encode_labels(combined_df, logger)

    if args.mode == "reference_only":
        reference_names = [name for name in dataframes if 'reference' in name.lower()]
        reference_df = combined_df.loc[reference_names]
        logger.info(f"Training CLIPn on references: {reference_names}")
        latent_df = run_clipn_placeholder(reference_df, logger)
    else:
        logger.info("Training and integrating CLIPn on all datasets")
        latent_df = run_clipn_placeholder(combined_df, logger)

    encoded_path = Path(args.out) / f"{args.experiment}_encoded.csv"
    combined_df.to_csv(encoded_path)
    logger.info(f"Encoded data saved to {encoded_path}")

    decoded_df = decode_labels(latent_df.copy(), encoders, logger)
    decoded_path = Path(args.out) / f"{args.experiment}_decoded.csv"
    decoded_df.to_csv(decoded_path)
    logger.info(f"Decoded data saved to {decoded_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIPn Integration.")
    parser.add_argument("--datasets_csv", required=True, help="CSV listing dataset names and paths.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--experiment", required=True, help="Experiment name.")
    parser.add_argument("--mode", choices=['reference_only', 'integrate_all'], required=True,
                        help="Mode of CLIPn operation.")

    args = parser.parse_args()
    main(args)
