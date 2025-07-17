#!/usr/bin/env python3
"""
Merge multiple compound annotation files on Plate_Metadata and cpd_id.

Inputs:
    - ./mitotox/mitotox.tsv
    - SelleckChem_10uM_22_07_2024/selechem_metadata.tsv
    - ./STB_final/STBV2_10uM_10032024.csv
    - ./STB_final/STBV2_10uM_10032024.csv

Output:
    - merged_annotations.tsv

Assumes columns:
    - Plate or Plate_Metadata (will be harmonised)
    - cpd_id (required)

"""

import pandas as pd

def robust_read(path):
    # Detect delimiter
    with open(path, "r") as f:
        header = f.readline()
        if "\t" in header:
            sep = "\t"
        elif "," in header:
            sep = ","
        else:
            sep = "\t"
    return pd.read_csv(path, sep=sep)

def harmonise_cols(df):
    # Rename columns if needed
    if "Plate" in df.columns and "Plate_Metadata" not in df.columns:
        df = df.rename(columns={"Plate": "Plate_Metadata"})
    if "Well" in df.columns and "Well_Metadata" not in df.columns:
        df = df.rename(columns={"Well": "Well_Metadata"})
    return df

def main():
    files = [
        "./mitotox/mitotox.tsv",
        "SelleckChem_10uM_22_07_2024/selechem_metadata.tsv",
        "./STB_final/STBV2_10uM_10032024.csv",
        "./STB_final/STBV2_10uM_10032024.csv"
    ]

    merged = None
    for i, file in enumerate(files):
        df = robust_read(file)
        df = harmonise_cols(df)
        # Only keep relevant columns for merging
        keep_cols = [c for c in df.columns if c in ["Plate_Metadata", "Well_Metadata", "cpd_id"] or c.lower().startswith("cpd") or c.lower().startswith("plate")]
        df = df[keep_cols]
        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=["Plate_Metadata", "cpd_id"], how="outer", suffixes=("", f"_{i+1}"))
    # Drop duplicate rows if present
    merged = merged.drop_duplicates()
    merged.to_csv("merged_annotations.tsv", sep="\t", index=False)
    print(f"Merged annotation file written: merged_annotations.tsv (shape: {merged.shape})")

if __name__ == "__main__":
    main()
