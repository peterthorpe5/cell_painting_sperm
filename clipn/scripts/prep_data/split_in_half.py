#!/usr/bin/env python3
"""
split_table_in_half.py

Split a large table (TSV/CSV) into two files of approximately equal size.
Both output files will have the header row preserved.

Usage:
    python split_table_in_half.py --input input.tsv --output1 part1.tsv --output2 part2.tsv --sep '\t'

Arguments:
    --input      : Path to the input file (TSV or CSV).
    --output1    : Path to first output file.
    --output2    : Path to second output file.
    --sep        : Delimiter (default: tab).

Author: Pete Thorpe, 2025
"""

import pandas as pd
import argparse

def split_table(input_file, output1, output2, sep='\t'):
    """
    Split the input table into two approximately equal parts, preserving header in both outputs.

    Parameters
    ----------
    input_file : str
        Path to the input file.
    output1 : str
        Path to the first output file.
    output2 : str
        Path to the second output file.
    sep : str
        Delimiter for reading/writing the files.

    Returns
    -------
    None
    """
    # Read input
    df = pd.read_csv(input_file, sep=sep)
    n = len(df)
    mid = n // 2
    # First half
    df.iloc[:mid, :].to_csv(output1, sep=sep, index=False)
    # Second half
    df.iloc[mid:, :].to_csv(output2, sep=sep, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a table in half with headers in both output files.")
    parser.add_argument('--input', required=True, help='Input table file (TSV/CSV)')
    parser.add_argument('--output1', required=True, help='First output file')
    parser.add_argument('--output2', required=True, help='Second output file')
    parser.add_argument('--sep', default='\t', help='Delimiter (default: tab)')
    args = parser.parse_args()
    split_table(args.input, args.output1, args.output2, sep=args.sep)
