#!/usr/bin/env python3
"""
Merge and average multiple Cell Painting feature-selected datasets by `cpd_id`.

Each dataset is assumed to be a tab-delimited file with common columns including `cpd_id`.
The script concatenates all files, groups by `cpd_id`, and averages the numeric feature columns.

Usage:
    - Update the `input_files` dictionary below with your batch name and full path.
    - Run the script.

hard coded paths for now.
"""

import pandas as pd
import os


def load_and_tag_dataset(file_path: str, dataset_name: str) -> pd.DataFrame:
    """
    Load a dataset and tag it with the dataset name.

    Parameters:
        file_path (str): Path to the TSV file.
        dataset_name (str): Identifier for the dataset.

    Returns:
        pd.DataFrame: Loaded DataFrame with dataset name added.
    """
    df = pd.read_csv(file_path, sep="\t")
    df["Dataset"] = dataset_name
    return df


def merge_and_average_by_cpd_id(file_dict: dict) -> pd.DataFrame:
    """
    Merge multiple datasets and average by `cpd_id`.

    Parameters:
        file_dict (dict): Mapping of dataset name to file path.

    Returns:
        pd.DataFrame: Averaged DataFrame grouped by `cpd_id`.
    """
    all_dfs = []

    for dataset_name, file_path in file_dict.items():
        df = load_and_tag_dataset(file_path, dataset_name)
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    if "cpd_id" not in merged_df.columns:
        raise ValueError("Expected column 'cpd_id' not found in merged data.")

    # Separate metadata from features
    non_numeric_cols = merged_df.select_dtypes(include=["object"]).columns.tolist()
    non_numeric_cols = [col for col in non_numeric_cols if col != "cpd_id"]
    numeric_cols = merged_df.select_dtypes(include=["number"]).columns.tolist()

    grouped = merged_df.groupby("cpd_id")[numeric_cols].mean().reset_index()

    # Optionally preserve the first value of non-numeric columns per cpd_id
    metadata = merged_df.groupby("cpd_id")[non_numeric_cols].first().reset_index()
    result = pd.merge(grouped, metadata, on="cpd_id", how="left")

    return result


if __name__ == "__main__":
    # Update with your actual paths
    input_files = {
        "B1": "/home/pthorpe/scratch/2025_STB/2025_cell_painting_sperm/SelleckChem_10uM_22_07_2024/Batch1/normalised/imputed/SelleckChem_10uM_22_07_2024_B1_imputed_grouped_filtered_feature_selected.tsv",
        "B2": "/home/pthorpe/scratch/2025_STB/2025_cell_painting_sperm/SelleckChem_10uM_22_07_2024/Batch2/normalised/imputed/SelleckChem_10uM_22_07_2024_B2_imputed_grouped_filtered_feature_selected.tsv",
        "B3": "/home/pthorpe/scratch/2025_STB/2025_cell_painting_sperm/SelleckChem_10uM_22_07_2024/Batch3/normalised/imputed/SelleckChem_10uM_22_07_2024_B3_imputed_grouped_filtered_feature_selected.tsv",
        "B4": "/home/pthorpe/scratch/2025_STB/2025_cell_painting_sperm/SelleckChem_10uM_22_07_2024/Batch4/normalised/imputed/SelleckChem_10uM_22_07_2024_B4_imputed_grouped_filtered_feature_selected.tsv",
        "B5": "/home/pthorpe/scratch/2025_STB/2025_cell_painting_sperm/SelleckChem_10uM_22_07_2024/Batch5/normalised/imputed/SelleckChem_10uM_22_07_2024_B5_imputed_grouped_filtered_feature_selected.tsv",
        "B6": "/home/pthorpe/scratch/2025_STB/2025_cell_painting_sperm/SelleckChem_10uM_22_07_2024/Batch6/normalised/imputed/SelleckChem_10uM_22_07_2024_B6_imputed_grouped_filtered_feature_selected.tsv",
    }

    averaged_df = merge_and_average_by_cpd_id(input_files)
    output_path = "merged_SelleckChem_averaged_by_cpd_id_imputed_feature_selected.tsv"
    averaged_df.to_csv(output_path, sep="\t", index=False)
    print(f"Merged and averaged file saved to: {output_path}")
