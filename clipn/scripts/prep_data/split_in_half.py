#!/usr/bin/env python3
# coding: utf-8
"""
split_table_in_half.py

Split a large table (TSV/CSV) into two files so that no group (e.g., cpd_id)
is split across outputs. Optionally sort outputs by the key.

Usage:
    python split_table_in_half.py \
      --input input.tsv \
      --output1 part1.tsv \
      --output2 part2.tsv \
      --sep '\t' \
      --key cpd_id \
      --sort

Author: Pete Thorpe, 2025
"""

from __future__ import annotations
import logging
import argparse
from typing import Iterable
import sys
import os
import warnings

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a table into two parts without splitting keys across files."
    )
    parser.add_argument("--input", required=True, help="Input TSV/CSV")
    parser.add_argument("--output1", required=True, help="First output file")
    parser.add_argument("--output2", required=True, help="Second output file")
    parser.add_argument("--sep", default="\t", help="Input/output delimiter (default: tab)")
    parser.add_argument(
        "--key",
        default="cpd_id",
        help="Column name used to group rows (default: cpd_id)",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort each output by the key column before writing",
    )
    parser.add_argument(
        "--drop-missing-key",
        action="store_true",
        help="Drop rows with missing/empty key (default: keep them and assign together)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()



def setup_logging(log_level="INFO"):
    """
    Set up logging to console.

    Parameters
    ----------
    log_level : str
        Logging level as a string ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns
    -------
    logging.Logger
        Configured logger object.
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("merge_logger")

def _assign_groups_balanced(
    keys: Iterable[str],
    sizes: pd.Series,
) -> set[str]:
    """
    Greedy bin-packing of groups into two parts by total row count.
    Returns the set of keys assigned to part A (the rest go to part B).
    """
    # Sort groups largest-first for better balance
    order = list(keys)
    order.sort(key=lambda k: sizes[k], reverse=True)

    a_size = 0
    b_size = 0
    a_keys: set[str] = set()

    for k in order:
        if a_size <= b_size:
            a_keys.add(k)
            a_size += sizes[k]
        else:
            b_size += sizes[k]

    return a_keys


def main() -> None:
    args = parse_args()

    logger = setup_logging(args.log_level)
    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Arguments: {' '.join(sys.argv)}")


    df = pd.read_csv(args.input, sep=args.sep)
    logger.info(f"Read {len(df):,} rows from {args.input}")

    if args.key not in df.columns:
        raise ValueError(f"Key column '{args.key}' not found in the input.")

    # Normalize key: strip whitespace; treat empty strings as NA
    df[args.key] = df[args.key].astype("string").str.strip()
    missing_mask = df[args.key].isna() | (df[args.key] == "")



    # Normalize key: strip whitespace; treat empty strings as NA
    if args.key not in df.columns:
        raise ValueError(f"Key column '{args.key}' not found in the input.")

    df[args.key] = df[args.key].astype("string").str.strip()
    missing_mask = df[args.key].isna() | (df[args.key] == "")

    if args.drop_missing_key:
        df = df.loc[~missing_mask].copy()
    else:
        # Keep them, but ensure they are grouped together under a sentinel key
        df.loc[missing_mask, args.key] = "__MISSING_KEY__"

    # Compute group sizes and balanced assignment
    group_sizes = df.groupby(args.key, dropna=False).size()
    unique_keys = group_sizes.index.tolist()

    # If missing key group exists, keep it as a normal group in the balancing
    a_keys = _assign_groups_balanced(unique_keys, group_sizes)

    # Split
    mask_a = df[args.key].isin(a_keys)
    part_a = df.loc[mask_a].copy()
    part_b = df.loc[~mask_a].copy()

    # Optional sort by key
    if args.sort:
        part_a = part_a.sort_values(by=[args.key], kind="mergesort")
        part_b = part_b.sort_values(by=[args.key], kind="mergesort")

    # If we kept missing keys, write them as-is (with sentinel value).
    # If you’d prefer to blank them back out, uncomment:
    # for frame in (part_a, part_b):
    #     frame.loc[frame[args.key] == "__MISSING_KEY__", args.key] = pd.NA

    # Write
    part_a.to_csv(args.output1, sep=args.sep, index=False)
    part_b.to_csv(args.output2, sep=args.sep, index=False)

    # Quick report
    a_rows, b_rows = len(part_a), len(part_b)
    a_groups = part_a[args.key].nunique(dropna=False)
    b_groups = part_b[args.key].nunique(dropna=False)
    print(
        f"Done. Part A: {a_rows:,} rows, {a_groups:,} groups | "
        f"Part B: {b_rows:,} rows, {b_groups:,} groups"
    )
    logger.info(
        f"Done. Part A: {a_rows:,} rows, {a_groups:,} groups | "
        f"Part B: {b_rows:,} rows, {b_groups:,} groups"
    )


if __name__ == "__main__":
    main()
