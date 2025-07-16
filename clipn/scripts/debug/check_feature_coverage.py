#!/usr/bin/env python3
"""
Check Feature Coverage Across Multiple Files
-------------------------------------------
Given a file-of-files (CSV/TSV with a column 'path'), reads each file and logs:
- The set of column names in each file.
- A summary of all columns seen, and which files lack which columns.
- Optionally writes a coverage report to disk.

Usage:
    python check_feature_coverage.py file_of_files.tsv [output_log.txt]

Example:
    python check_feature_coverage.py dataset_M_S_S_Image_level.csv feature_coverage.log
"""
import pandas as pd
import sys
import os

def log(msg, fh=None):
    print(msg)
    if fh:
        print(msg, file=fh)

def main(list_file, log_file=None):
    # Read file-of-files
    df_list = pd.read_csv(list_file, sep=None, engine="python")
    if "path" not in df_list.columns:
        print("ERROR: File-of-files must have a column named 'path'")
        sys.exit(1)

    # Open log file if needed
    fh = open(log_file, "w") if log_file else None

    # Track column sets per file
    file_to_columns = {}
    for path in df_list["path"]:
        try:
            df = pd.read_csv(path, sep=None, engine="python", nrows=10)  # Only need header
            cols = list(df.columns)
            file_to_columns[os.path.basename(path)] = cols
            log(f"[OK] {os.path.basename(path)}: {len(cols)} columns", fh)
        except Exception as e:
            file_to_columns[os.path.basename(path)] = None
            log(f"[ERROR] {os.path.basename(path)}: {e}", fh)

    # Build set of all columns
    all_columns = set()
    for cols in file_to_columns.values():
        if cols:
            all_columns.update(cols)
    all_columns = sorted(all_columns)

    log("\n==== Summary of Feature Coverage ====", fh)
    log(f"Total files: {len(file_to_columns)}", fh)
    log(f"Total unique columns: {len(all_columns)}", fh)
    log("\nColumn\tPresent_in_files\tMissing_in_files", fh)

    # For each column, which files have/miss it?
    for col in all_columns:
        present = [fname for fname, cols in file_to_columns.items() if cols and col in cols]
        missing = [fname for fname, cols in file_to_columns.items() if cols and col not in cols]
        log(f"{col}\t{len(present)}/{len(file_to_columns)}\t{', '.join(missing) if missing else ''}", fh)

    log("\n==== Per-file missing columns (metadata+features) ====", fh)
    for fname, cols in file_to_columns.items():
        if cols:
            missing = [col for col in all_columns if col not in cols]
            if missing:
                log(f"{fname} missing {len(missing)} columns: {', '.join(missing)}", fh)
            else:
                log(f"{fname} has all columns.", fh)
        else:
            log(f"{fname}: could not be read.", fh)

    if fh:
        fh.close()
        print(f"Log written to {log_file}")

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python check_feature_coverage.py file_of_files.tsv [output_log.txt]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
