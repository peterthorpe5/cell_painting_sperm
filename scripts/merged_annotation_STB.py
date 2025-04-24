#!/usr/bin/env python3
"""
Merge Cell Painting Reference Data with Compound Annotations
------------------------------------------------------------

This script merges grouped Cell Painting reference datasets (by mean)
with an annotation file based on Plate and Well positions.

It handles:
- Loading reference1 and reference2 (TSV with Plate_Metadata and Well_Metadata)
- Loading annotation CSV (comma-separated, with Plate and Well)
- Merging using Plate_Metadata and Well_Metadata vs Plate and Well
- Renaming annotation cpd_id to annotation_cpd_id to avoid collision

Output:
- A merged TSV file in the same directory as reference1 with suffix '_with_annotations.tsv'

Usage:
    python merge_annotations_plate_well.py --reference1 path.tsv --reference2 path.tsv --annotation path.csv --output_dir path
"""

import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_reference_data(path):
    df = pd.read_csv(path, sep='\t')
    logger.info(f"Loaded reference: {path} with shape {df.shape}")
    return df

def load_annotation_data(path):
    df = pd.read_csv(path)
    logger.info(f"Loaded annotation CSV: {path} with shape {df.shape}")

    if "cpd_id" in df.columns:
        df = df.rename(columns={"cpd_id": "annotation_cpd_id"})

    # Add Plate_Metadata and Well_Metadata to annotation for merging
    if "Plate" in df.columns and "Well" in df.columns:
        df["Plate_Metadata"] = df["Plate"].astype(str)
        df["Well_Metadata"] = df["Well"].astype(str)
    else:
        raise ValueError("Annotation file must contain 'Plate' and 'Well' columns.")

    return df

def merge_and_save(ref_df, annot_df, label, output_dir):
    merged = pd.merge(ref_df, annot_df, on=["Plate_Metadata", "Well_Metadata"], how="left")
    logger.info(f"Merged DataFrame for {label}: shape {merged.shape}")

    output_path = Path(output_dir) / f"{label}_with_annotations.tsv"
    merged.to_csv(output_path, sep='\t', index=False)
    logger.info(f"Saved merged output to: {output_path}")

def main(args):
    ref1_df = load_reference_data(args.reference1)
    ref2_df = load_reference_data(args.reference2)
    annot_df = load_annotation_data(args.annotation)

    merge_and_save(ref1_df, annot_df, label="reference1", output_dir=args.output_dir)
    merge_and_save(ref2_df, annot_df, label="reference2", output_dir=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Plate+Well annotations into reference data")
    parser.add_argument("--reference1", required=True, help="Path to first reference TSV")
    parser.add_argument("--reference2", required=True, help="Path to second reference TSV")
    parser.add_argument("--annotation", required=True, help="Path to annotation CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to save merged outputs")
    args = parser.parse_args()
    main(args)
