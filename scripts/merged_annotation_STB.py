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

SAFE_REFERENCE_COLUMNS = ["Plate_Metadata", "Well_Metadata", "cpd_id", "Library"]

SAFE_ANNOTATION_COLUMNS = [
    "annotation_cpd_id", "name", "published_phenotypes", "publish own other",
    "published_target", "published_vivo vitro", "published_in_vivo model",
    "published_in_vitro model", "published cell_model", "pubchem", "pubmed ids",
    "DMSO solubility info", "info", "well_number_y", "library", "smiles",
    "pubchem_cid", "cpd_information", "pubchem_url", "cpd_type"
]

def load_reference_data(path):
    """
    Load and filter a reference dataset.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep='\t')
    logger.info(f"Loaded reference: {path} with shape {df.shape}")
    return df

def load_annotation_data(path):
    """
    Load and prepare the annotation dataset.
    """
    df = pd.read_csv(path)
    logger.info(f"Loaded annotation CSV: {path} with shape {df.shape}")

    if "cpd_id" in df.columns:
        df = df.rename(columns={"cpd_id": "annotation_cpd_id"})

    if "Plate" not in df.columns or "Well" not in df.columns:
        raise ValueError("Annotation file must contain 'Plate' and 'Well' columns.")

    df["Plate_Metadata"] = df["Plate"].astype(str)
    df["Well_Metadata"] = df["Well"].astype(str)

    # Pad wells to A01, A02 format
    df["Well_Metadata"] = df["Well_Metadata"].apply(
        lambda x: f"A{int(x[1:]):02d}" if x.startswith('A') and x[1:].isdigit() else x
    )

    return df

def fix_reference_well_metadata(df):
    """
    Ensure the reference Well_Metadata matches A01, A02 format.
    """
    df["Well_Metadata"] = df["Well_Metadata"].astype(str)
    df["Well_Metadata"] = df["Well_Metadata"].apply(
        lambda x: f"A{int(x[1:]):02d}" if x.startswith('A') and x[1:].isdigit() else x
    )
    return df

def filter_reference_columns(df):
    """
    Keep only safe columns from the reference dataset.
    """
    cols_to_keep = [col for col in SAFE_REFERENCE_COLUMNS if col in df.columns]
    return df[cols_to_keep]

def merge_references(ref1_df, ref2_df, annot_df):
    """
    Merge reference datasets with annotation data.
    """
    combined_df = pd.concat([ref1_df, ref2_df], ignore_index=True)
    logger.info(f"Combined reference data shape before filtering: {combined_df.shape}")

    combined_df = filter_reference_columns(combined_df)
    logger.info(f"Reference data shape after filtering: {combined_df.shape}")

    annot_cols = [col for col in SAFE_ANNOTATION_COLUMNS if col in annot_df.columns]
    annot_df = annot_df[["Plate_Metadata", "Well_Metadata"] + annot_cols]

    merged = pd.merge(
        combined_df,
        annot_df,
        on=["Plate_Metadata", "Well_Metadata"],
        how="left"
    )

    logger.info(f"Merged output shape: {merged.shape}")
    return merged

def main(args):
    ref1_df = load_reference_data(args.reference1)
    ref2_df = load_reference_data(args.reference2)
    annot_df = load_annotation_data(args.annotation)

    ref1_df = fix_reference_well_metadata(ref1_df)
    ref2_df = fix_reference_well_metadata(ref2_df)

    merged_df = merge_references(ref1_df, ref2_df, annot_df)

    output_path = Path("combined_references_with_annotations.tsv")
    merged_df.to_csv(output_path, sep='\t', index=False)
    logger.info(f"Saved clean merged output to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safely merge annotations into reference data.")
    parser.add_argument("--reference1", required=True, help="Path to first reference file.")
    parser.add_argument("--reference2", required=True, help="Path to second reference file.")
    parser.add_argument("--annotation", required=True, help="Path to annotation CSV.")
    args = parser.parse_args()
    main(args)
