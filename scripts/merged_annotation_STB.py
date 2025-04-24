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
    """
    Load a reference dataset from a TSV file.

    Parameters
    ----------
    path : str
        Path to the reference TSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    df = pd.read_csv(path, sep='\t')
    logger.info(f"Loaded reference: {path} with shape {df.shape}")
    return df

def load_annotation_data(path):
    """
    Load and prepare the annotation data.

    Parameters
    ----------
    path : str
        Path to the annotation CSV file.

    Returns
    -------
    pd.DataFrame
        Processed annotation DataFrame with Plate_Metadata and Well_Metadata.
    """
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

def merge_and_save_combined(ref1_df, ref2_df, annot_df, output_dir):
    """
    Merge combined reference datasets with annotation data and save as a single file.

    Parameters
    ----------
    ref1_df : pd.DataFrame
        First reference DataFrame.
    ref2_df : pd.DataFrame
        Second reference DataFrame.
    annot_df : pd.DataFrame
        Annotation DataFrame with Plate_Metadata and Well_Metadata.
    output_dir : str or Path
        Directory where the merged file will be saved.
    """
    combined_df = pd.concat([ref1_df, ref2_df], axis=0).reset_index(drop=True)
    logger.info(f"Combined reference data shape: {combined_df.shape}")

    desired_annot_cols = [
        "annotation_cpd_id", "name", "published_phenotypes", "publish own other",
        "published_target", "published_vivo vitro", "published_in_vivo model",
        "published_in_vitro model", "published cell_model", "pubchem", "pubmed ids",
        "DMSO solubility info", "info", "well_number_y", "library", "smiles",
        "pubchem_cid", "cpd_information", "pubchem_url", "cpd_type"
    ]
    annot_df = annot_df[[col for col in desired_annot_cols if col in annot_df.columns] + ["Plate_Metadata", "Well_Metadata"]]

    merged = pd.merge(combined_df, annot_df, on=["Plate_Metadata", "Well_Metadata"], how="left")
    logger.info(f"Merged DataFrame shape: {merged.shape}")

    output_path = Path(output_dir) / "combined_references_with_annotations.tsv"
    merged.to_csv(output_path, sep='\t', index=False)
    logger.info(f"Saved merged output to: {output_path}")

def main(args):
    """
    Main function to execute the merge script.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    ref1_df = load_reference_data(args.reference1)
    ref2_df = load_reference_data(args.reference2)
    annot_df = load_annotation_data(args.annotation)

    merge_and_save_combined(ref1_df, ref2_df, annot_df, output_dir=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Plate+Well annotations into reference data")
    parser.add_argument("--reference1", required=True, help="Path to first reference TSV")
    parser.add_argument("--reference2", required=True, help="Path to second reference TSV")
    parser.add_argument("--annotation", required=True, help="Path to annotation CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to save merged outputs")
    args = parser.parse_args()
    main(args)
