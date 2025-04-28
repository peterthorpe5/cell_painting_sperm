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
    """Load a reference dataset from CSV or TSV."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep='\t')
    logger.info(f"Loaded reference: {path} with shape {df.shape}")
    return df

def load_annotation_data(path):
    """Load the annotation dataset."""
    df = pd.read_csv(path)
    logger.info(f"Loaded annotation CSV: {path} with shape {df.shape}")

    if "cpd_id" in df.columns:
        df = df.rename(columns={"cpd_id": "annotation_cpd_id"})

    if "Plate" in df.columns and "Well" in df.columns:
        df["Plate_Metadata"] = df["Plate"].astype(str)
        df["Well_Metadata"] = df["Well"].astype(str)
    else:
        raise ValueError("Annotation file must contain 'Plate' and 'Well' columns.")

    return df

def fix_well_metadata_format(df, col_name="Well_Metadata"):
    """Fix Well_Metadata to ensure 2 digits, e.g., 'A01' instead of 'A1'."""
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str)
        df[col_name] = df[col_name].apply(
            lambda x: x if len(x) == 3 else f"{x[0]}{x[1:].zfill(2)}"
            if len(x) >= 2 and x[0].isalpha() and x[1:].isdigit() else x
        )
    return df

def debug_merge(ref1_df, ref2_df, annot_df):
    """Merge reference dataframes with annotation dataframe and debug."""
    combined_df = pd.concat([ref1_df, ref2_df], ignore_index=True)
    logger.info(f"Combined reference data shape: {combined_df.shape}")

    if "Plate_Metadata" not in combined_df.columns or "Well_Metadata" not in combined_df.columns:
        raise ValueError("Reference data must contain 'Plate_Metadata' and 'Well_Metadata' columns.")

    logger.info("\nUnique Plate_Metadata examples (Reference):")
    print(combined_df['Plate_Metadata'].astype(str).dropna().unique()[:10])

    logger.info("\nUnique Well_Metadata examples (Reference):")
    print(combined_df['Well_Metadata'].astype(str).dropna().unique()[:10])

    logger.info("\nUnique Plate_Metadata examples (Annotation):")
    print(annot_df['Plate_Metadata'].astype(str).dropna().unique()[:10])

    logger.info("\nUnique Well_Metadata examples (Annotation):")
    print(annot_df['Well_Metadata'].astype(str).dropna().unique()[:10])

    combined_df = fix_well_metadata_format(combined_df)
    annot_df = fix_well_metadata_format(annot_df)

    desired_annot_cols = [
        "annotation_cpd_id", "name", "published_phenotypes", "publish own other",
        "published_target", "published_vivo vitro", "published_in_vivo model",
        "published_in_vitro model", "published cell_model", "pubchem", "pubmed ids",
        "DMSO solubility info", "info", "well_number_y", "library", "smiles",
        "pubchem_cid", "cpd_information", "pubchem_url", "cpd_type"
    ]

    annot_keep_cols = [col for col in desired_annot_cols if col in annot_df.columns]
    if not annot_keep_cols:
        logger.warning("None of the desired annotation columns found!")

    merged = pd.merge(
        combined_df,
        annot_df[["Plate_Metadata", "Well_Metadata"] + annot_keep_cols],
        on=["Plate_Metadata", "Well_Metadata"],
        how="left"
    )

    logger.info("Attempting merge now...")
    logger.info(f"Merged shape: {merged.shape}")

    final_keep_cols = [col for col in annot_keep_cols if col in merged.columns]
    if not final_keep_cols:
        logger.warning("None of the desired annotation columns exist after merge!")
        n_annotated = 0
    else:
        n_annotated = merged[final_keep_cols].notna().any(axis=1).sum()

    logger.info(f"Rows with at least one annotation field filled: {n_annotated} / {merged.shape[0]}")

    logger.info("\nExamples of rows missing annotation (first 5):")
    print(merged.loc[merged[final_keep_cols].isna().all(axis=1), ["Plate_Metadata", "Well_Metadata"] + final_keep_cols].head())

    logger.info("\nExamples of rows with successful annotation (first 5):")
    print(merged.loc[merged[final_keep_cols].notna().any(axis=1), ["Plate_Metadata", "Well_Metadata"] + final_keep_cols].head())

    output_path = Path("combined_references_with_annotations.tsv")
    merged.to_csv(output_path, sep='\t', index=False)
    logger.info(f"Saved merged output to: {output_path}")

def main(args):
    ref1_df = load_reference_data(args.reference1)
    ref2_df = load_reference_data(args.reference2)
    annot_df = load_annotation_data(args.annotation)

    debug_merge(ref1_df, ref2_df, annot_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Plate+Well annotations into reference data")
    parser.add_argument("--reference1", required=True, help="Path to first reference file (CSV or TSV)")
    parser.add_argument("--reference2", required=True, help="Path to second reference file (CSV or TSV)")
    parser.add_argument("--annotation", required=True, help="Path to annotation CSV")
    args = parser.parse_args()
    main(args)
