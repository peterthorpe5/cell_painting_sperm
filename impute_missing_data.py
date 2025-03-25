#!/usr/bin/env python
# coding: utf-8

"""
Sperm Cell Painting Data Analysis with CLIPn Clustering and UMAP
-----------------------------------------------------------------
Imputation step..


more doc to come once drafted more. 

Logging also needed. 

"""



#!/usr/bin/env python
# impute_missing_data.py

import argparse
import os
import pandas as pd
from process_data import (
    load_datasets_from_folderlist,
    group_and_filter_data,
    impute_missing_values,
    encode_cpd_data
)

def parse_args():
    parser = argparse.ArgumentParser(description="Impute and encode multiple datasets for CLIPn analysis.")
    parser.add_argument("--dataset-list", required=True,
                        help="Path to dataset list file (each line: dataset_name folder_path)")
    parser.add_argument("--output-folder", required=True, help="Folder to save imputed and encoded files")
    parser.add_argument("--exclude-substring", default=None,
                        help="Substring of CSV filenames to exclude (optional)")
    parser.add_argument("--impute-method", choices=["median", "knn"], default="median",
                        help="Imputation method to use")
    parser.add_argument("--knn-neighbors", type=int, default=5, help="Neighbors for KNN imputation")
    parser.add_argument("--encode-labels", action="store_true", help="Whether to encode labels")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    datasets = load_datasets_from_folderlist(args.dataset_list, args.exclude_substring)

    processed_data = {}
    logger.info(f"Python Version: {sys.version_info}")
    logger.info(f"Command-line Arguments: {' '.join(sys.argv)}")
    logger.info(f"Using Logfile: {log_filename}")
    logger.info(f"Logging initialized at {time.asctime()}")

    for name, df in datasets.items():
        print(f"Processing dataset: {name}")

        # Group and filter data
        grouped_df = group_and_filter_data(df)

        # Impute missing values
        imputed_df, _, _, _ = impute_missing_values(grouped_df, None,
                                                    impute_method=args.impute_method,
                                                    knn_neighbors=args.knn_neighbors)

        # Optionally encode labels
        if args.encode_labels:
            encoded_data = encode_cpd_data({name: imputed_df}, encode_labels=True)
            final_df = encoded_data[name]["data"]
        else:
            final_df = imputed_df

        # Save the processed data
        output_path = os.path.join(args.output_folder, f"{name}_grouped_imputed_encoded.csv")
        final_df.to_csv(output_path)
        print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main()
