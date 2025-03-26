import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import set_config
import csv

logger = logging.getLogger(__name__)




def load_and_harmonise_datasets(datasets_csv, logger):
    """Load datasets from CSV and harmonise columns."""
    datasets_df = pd.read_csv(datasets_csv)
    dataset_paths = datasets_df.set_index('dataset')['path'].to_dict()

    dataframes = {}
    all_cols = set()

    logger.info("Loading datasets")
    for name, path in dataset_paths.items():
        df = pd.read_csv(path, index_col=0)
        dataframes[name] = df
        all_cols.update(df.columns)
        logger.debug(f"Loaded {name}: shape {df.shape}")

    common_cols = sorted(list(all_cols.intersection(*[set(df.columns) for df in dataframes.values()])))
    logger.info(f"Harmonised columns count: {len(common_cols)}")

    for name in dataframes:
        dataframes[name] = dataframes[name][common_cols]
        logger.debug(f"Harmonised {name}: shape {dataframes[name].shape}")

    return dataframes, common_cols


def encode_labels(df, logger):
    """Encode categorical columns and return encoders for decoding later."""
    encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        logger.debug(f"Encoded column {col}")
    return df, encoders


def decode_labels(df, encoders, logger):
    """Decode categorical columns to original labels."""
    for col, le in encoders.items():
        df[col] = le.inverse_transform(df[col])
        logger.debug(f"Decoded column {col}")
    return df


def run_clipn_placeholder(df, logger):
    """Placeholder function for running CLIPn integration analysis."""
    logger.info("Running CLIPn analysis (placeholder)")
    # Real CLIPn integration logic should be implemented here.
    return df
