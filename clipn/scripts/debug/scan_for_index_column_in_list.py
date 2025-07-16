#!/usr/bin/env python3
"""
Scan Listed Files for Index Column Issues
-----------------------------------------
Given a list file (CSV/TSV with a column 'path'), checks each file for a suspicious
leading index column (e.g. 'Unnamed: 0' or a column of sequential integers).

Usage:
    python scan_for_index_column_in_list.py list_of_files.tsv

Outputs a report to stdout.
"""
import pandas as pd
import sys
import os

def looks_like_index_column(col):
    """Check if a column looks like a sequential integer index."""
    if col.name in ["Unnamed: 0", "", None]:
        return True
    try:
        vals = pd.to_numeric(col, errors="coerce")
        if vals.isnull().any():
            return False
        starts_at_zero = (vals.iloc[0] == 0)
        starts_at_one = (vals.iloc[0] == 1)
        if starts_at_zero or starts_at_one:
            return (vals == range(vals.iloc[0], vals.iloc[0] + len(vals))).all()
    except Exception:
        pass
    return False

def scan_file(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python", nrows=10000)
        first_col = df.columns[0]
        col_data = df.iloc[:, 0]
        if looks_like_index_column(col_data) or str(first_col).startswith("Unnamed"):
            print(f"[SUSPECT] {os.path.basename(path)}: first column is '{first_col}' (possible index column)")
        else:
            print(f"[OK]      {os.path.basename(path)}: first column is '{first_col}'")
    except Exception as e:
        print(f"[ERROR]   {os.path.basename(path)}: {e}")

def main(list_file):
    df_list = pd.read_csv(list_file, sep=None, engine="python")
    if "path" not in df_list.columns:
        print("ERROR: File-of-files must have a column named 'path'")
        sys.exit(1)
    print(f"Scanning {len(df_list)} files listed in {list_file}")
    for path in df_list["path"]:
        scan_file(path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scan_for_index_column_in_list.py list_of_files.tsv")
        sys.exit(1)
    main(sys.argv[1])
