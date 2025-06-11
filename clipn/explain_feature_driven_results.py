#!/usr/bin/env python3
"""
Feature Attribution and Statistical Comparison for Cell Painting Nearest Neighbours
----------------------------------------------------------------------------------

This script:
- Identifies features/compartments most different between a query compound and DMSO controls.
- Identifies features/compartments that explain the similarity to nearest neighbour compounds.
- Performs non-parametric tests (Mann–Whitney U or KS) at the well level.
- Automatically infers compartments from feature names if no mapping file is provided.
- Outputs top N features and groups for each comparison, with BH-corrected p-values.
- Provides extensive logging.


python explain_feature_driven_results.py \
    --ungrouped_list dataset_paths.txt \
    --query_ids queries.txt \
    --nn_file nearest_neighbours.tsv \
    --output_dir ./nn_analysis_results \
    --nn_per_query 10 \
    --top_features 5

"""



#!/usr/bin/env python3
"""
Feature Attribution and Statistical Comparison for Cell Painting Nearest Neighbours
----------------------------------------------------------------------------------

For each query compound, identifies features and compartments:
- Most different between the query and DMSO (using well-level data, robust to non-normality)
- Most similar between query and its nearest neighbours (as listed in NN table)
- Outputs top N features/compartments per query

Inputs:
- List of well-level (ungrouped) files (via CSV/TSV)
- Query IDs (via CLI or file)
- Nearest neighbour table (query, neighbour, distance)

Author: [Your name]
"""

import argparse
import os
import pandas as pd
import numpy as np
import logging
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, ks_2samp
from collections import defaultdict
import warnings

def setup_logger(log_file):
    """
    Configure logging to file and stdout.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("feature_attribution_logger")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    if not logger.hasHandlers():
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def load_ungrouped_files(list_file):
    """
    Load and concatenate all ungrouped (well-level) feature files from a list.

    Args:
        list_file (str): Path to dataset list file (CSV/TSV) with 'path' column.

    Returns:
        pd.DataFrame: Concatenated well-level feature data.
    """
    if list_file.endswith(".csv"):
        df_list = pd.read_csv(list_file)
    else:
        df_list = pd.read_csv(list_file, sep=None, engine='python')
    assert "path" in df_list.columns, "Dataset list file must contain a 'path' column"
    dfs = []
    for p in df_list['path']:
        tmp = pd.read_csv(p, sep="\t")
        dfs.append(tmp)
    # Harmonise columns (intersection)
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    dfs = [d[list(common_cols)] for d in dfs]
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def load_groupings(group_file):
    """
    Load feature-to-group mapping from file (CSV/TSV: columns = ['feature','group']).

    Args:
        group_file (str): Path to the feature-group file.

    Returns:
        dict: Mapping from group to feature list.
    """
    df = pd.read_csv(group_file, sep=None, engine='python')
    group_map = {}
    for group, subdf in df.groupby('group'):
        group_map[group] = subdf['feature'].tolist()
    return group_map

def infer_compartments(feature_cols):
    """
    Infer compartments from feature names by taking the final underscore-delimited token.

    Args:
        feature_cols (list): List of feature column names.

    Returns:
        dict: Mapping of compartment name to feature list.
    """
    group_map = defaultdict(list)
    for feat in feature_cols:
        tokens = feat.split("_")
        if len(tokens) > 1:
            compartment = tokens[-1].lower()
        else:
            compartment = "unknown"
        group_map[compartment].append(feat)
    return dict(group_map)

def get_well_level(df, cpd_id, cpd_id_col='cpd_id'):
    """
    Extract all rows for a given compound id (well-level).

    Args:
        df (pd.DataFrame): DataFrame of well-level data.
        cpd_id (str): Compound ID of interest.
        cpd_id_col (str): Name of column for compound ID.

    Returns:
        pd.DataFrame: Subset for the compound.
    """
    mask = df[cpd_id_col].astype(str).str.upper() == str(cpd_id).upper()
    return df[mask]

def get_wells_for_dmso(df, cpd_type_col='cpd_type', dmso_label='DMSO'):
    """
    Extract all DMSO wells (robust to capitalisation and variants).

    Args:
        df (pd.DataFrame): DataFrame of well-level data.
        cpd_type_col (str): Column for compound type.
        dmso_label (str): String label for DMSO.

    Returns:
        pd.DataFrame: Subset for DMSO controls.
    """
    mask = df[cpd_type_col].astype(str).str.upper().str.contains(dmso_label.upper())
    return df[mask]

def compare_distributions(df1, df2, feature_cols, test='mw'):
    """
    Compare distributions of each feature between two well sets. Returns stats and p-values.

    Args:
        df1 (pd.DataFrame): DataFrame for set 1 (e.g., query).
        df2 (pd.DataFrame): DataFrame for set 2 (e.g., DMSO or NN).
        feature_cols (list): List of feature columns.
        test (str): 'mw' for Mann–Whitney U, 'ks' for Kolmogorov–Smirnov.

    Returns:
        pd.DataFrame: Results with stat, raw_pvalue, abs_median_diff, med_query, med_comp.
    """
    stats = []
    for feat in feature_cols:
        x1 = df1[feat].dropna()
        x2 = df2[feat].dropna()
        if len(x1) < 2 or len(x2) < 2:
            stat, p = np.nan, np.nan
        else:
            if test == 'mw':
                try:
                    stat, p = mannwhitneyu(x1, x2, alternative='two-sided')
                except Exception as e:
                    stat, p = np.nan, np.nan
            else:
                try:
                    stat, p = ks_2samp(x1, x2)
                except Exception as e:
                    stat, p = np.nan, np.nan
        med1 = np.median(x1) if len(x1) else np.nan
        med2 = np.median(x2) if len(x2) else np.nan
        abs_diff = np.abs(med1 - med2)
        stats.append({'feature': feat, 'stat': stat, 'raw_pvalue': p,
                      'abs_median_diff': abs_diff, 'med_query': med1, 'med_comp': med2})
    return pd.DataFrame(stats)

def group_feature_stats(feat_stats, group_map):
    """
    Summarise feature stats by group (mean abs_median_diff and min p-value per group).

    Args:
        feat_stats (pd.DataFrame): Per-feature stats.
        group_map (dict): Mapping from group to feature list.

    Returns:
        pd.DataFrame: Grouped stats by compartment.
    """
    grouped = []
    for group, feats in group_map.items():
        group_df = feat_stats[feat_stats['feature'].isin(feats)]
        if group_df.empty:
            continue
        abs_diff = group_df['abs_median_diff'].mean()
        min_p = group_df['raw_pvalue'].min()
        grouped.append({'group': group, 'mean_abs_median_diff': abs_diff,
                        'min_raw_pvalue': min_p})
    return pd.DataFrame(grouped)

def parse_query_ids(query_ids_arg):
    """
    Parse query IDs from comma-separated string, file, or list.

    Args:
        query_ids_arg (str or list): Comma-separated string or filename.

    Returns:
        list: List of query IDs as strings.
    """
    if isinstance(query_ids_arg, list):
        return query_ids_arg
    if os.path.isfile(query_ids_arg):
        with open(query_ids_arg, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        return ids
    # Otherwise, treat as comma-separated string
    return [x.strip() for x in query_ids_arg.split(',') if x.strip()]

def load_nn_table(nn_file, query_id, nn_per_query=10):
    """
    Load nearest neighbours for a given query from NN table.

    Args:
        nn_file (str): Path to nearest_neighbours.tsv.
        query_id (str): Query cpd_id.
        nn_per_query (int): Number of NNs to use.

    Returns:
        list: List of NN cpd_ids (strings).
    """
    nn_df = pd.read_csv(nn_file, sep=None, engine='python')
    mask = nn_df['cpd_id'].astype(str).str.upper() == str(query_id).upper()
    neighbours = nn_df[mask].sort_values('distance')['neighbour_id'].astype(str).tolist()
    return neighbours[:nn_per_query]

def main():
    """
    Main script entry point for batch feature attribution and comparison to DMSO for multiple queries.
    """
    parser = argparse.ArgumentParser(description="Statistical attribution for cell painting NNs vs DMSO (batch mode)")
    parser.add_argument('--ungrouped_list', required=True, help="CSV/TSV with column 'path' for ungrouped feature files")
    parser.add_argument('--query_ids', required=True, default=[
                        "DDD02387619", "DDD02948916'
                        "DDD02955130", "DDD02958365"], help="Comma-separated, one-per-line, or file with query IDs")
    parser.add_argument('--nn_file', required=True, help="Nearest neighbours TSV (cpd_id, neighbour_id, distance)")
    parser.add_argument('--cpd_type_col', default='cpd_type', help="Column for compound type")
    parser.add_argument('--cpd_id_col', default='cpd_id', help="Column for compound ID")
    parser.add_argument('--dmso_label', default='DMSO', help="Control label (case-insensitive, substring match)")
    parser.add_argument('--feature_group_file', help="Feature-group mapping (CSV/TSV, optional)")
    parser.add_argument('--output_dir', required=True, help="Directory for output tables")
    parser.add_argument('--nn_per_query', type=int, default=10, help="Number of nearest neighbours to compare per query")
    parser.add_argument('--top_features', type=int, default=10, help="Number of top features/groups to report per comparison")
    parser.add_argument('--test', default='mw', choices=['mw', 'ks'], help="Statistical test: 'mw' or 'ks'")
    parser.add_argument('--log_file', default="feature_attribution.log", help="Log file name")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, args.log_file))
    logger.info("Starting batch NN feature attribution analysis.")
    logger.info(f"Arguments: {args}")

    logger.info("Loading all well-level (ungrouped) data files...")
    df = load_ungrouped_files(args.ungrouped_list)
    meta_cols = [args.cpd_id_col, args.cpd_type_col, 'Library', 'Plate_Metadata', 'Well_Metadata', 'Dataset']
    feature_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    logger.info(f"Detected {len(feature_cols)} features for analysis: {feature_cols[:10]} ...")

    # DMSO wells (all files)
    logger.info("Extracting DMSO wells (control distribution)...")
    dmso_df = get_wells_for_dmso(df, cpd_type_col=args.cpd_type_col, dmso_label=args.dmso_label)
    logger.info(f"Found {dmso_df.shape[0]} DMSO wells.")


    # Suppress openpyxl warnings
    warnings.simplefilter("ignore")

    # Feature grouping: use provided file or infer compartments
    if args.feature_group_file:
        group_map = load_groupings(args.feature_group_file)
        logger.info("Loaded feature groupings from file.")
    else:
        group_map = infer_compartments(feature_cols)
        logger.info(f"Inferred {len(group_map)} compartments: {list(group_map.keys())}")

    # Query IDs
    query_ids = parse_query_ids(args.query_ids)
    logger.info(f"Processing {len(query_ids)} query compounds: {query_ids}")

    for query_id in query_ids:
        logger.info(f"\n=== Analysing query: {query_id} ===")
        # Query wells
        query_df = get_well_level(df, query_id, cpd_id_col=args.cpd_id_col)
        if query_df.empty:
            logger.warning(f"No wells found for query {query_id}, skipping.")
            continue

        # Query vs DMSO
        logger.info(f"Comparing query compound {query_id} to DMSO controls...")
        q_dmso_stats = compare_distributions(query_df, dmso_df, feature_cols, test=args.test)
        _, q_dmso_stats['pvalue_bh'], _, _ = multipletests(q_dmso_stats['raw_pvalue'], method='fdr_bh')
        q_dmso_stats_sorted = q_dmso_stats.sort_values('abs_median_diff', ascending=False).head(args.top_features)
        out_tsv = os.path.join(args.output_dir, f"{query_id}_vs_DMSO_top_features.tsv")
        q_dmso_stats_sorted.to_csv(out_tsv, sep="\t", index=False)
        out_xlsx = out_tsv.replace(".tsv", ".xlsx")
        q_dmso_stats_sorted.to_excel(out_xlsx, index=False)
        logger.info(f"Saved top features distinguishing {query_id} from DMSO: {q_dmso_stats_sorted['feature'].tolist()}")



        if group_map:
            q_dmso_group_stats = group_feature_stats(q_dmso_stats, group_map)
            q_dmso_group_stats = q_dmso_group_stats.sort_values('mean_abs_median_diff', ascending=False).head(args.top_features)
            out_tsv = os.path.join(args.output_dir, f"{query_id}_vs_DMSO_top_groups.tsv")
            q_dmso_group_stats.to_csv(out_tsv, sep="\t", index=False)
            out_xlsx = out_tsv.replace(".tsv", ".xlsx")
            q_dmso_group_stats.to_excel(out_xlsx, index=False)
            logger.info(f"Saved top compartments distinguishing {query_id} from DMSO: {q_dmso_group_stats['group'].tolist()}")



        # Nearest neighbours (from NN table)
        nn_ids = load_nn_table(args.nn_file, query_id, nn_per_query=args.nn_per_query)
        logger.info(f"Loaded {len(nn_ids)} nearest neighbours for {query_id}: {nn_ids}")

        for nn_id in nn_ids:
            nn_df = get_well_level(df, nn_id, cpd_id_col=args.cpd_id_col)
            if nn_df.empty:
                logger.warning(f"No wells found for NN {nn_id}, skipping.")
                continue                        
            logger.info(f"Comparing {query_id} to NN {nn_id}...")
            q_nn_stats = compare_distributions(query_df, nn_df, feature_cols, test=args.test)
            _, q_nn_stats['pvalue_bh'], _, _ = multipletests(q_nn_stats['raw_pvalue'], method='fdr_bh')
            q_nn_stats_sorted = q_nn_stats.sort_values('abs_median_diff').head(args.top_features)
            out_tsv = os.path.join(args.output_dir, f"{query_id}_vs_{nn_id}_top_features.tsv")
            q_nn_stats_sorted.to_csv(out_tsv, sep="\t", index=False)
            out_xlsx = out_tsv.replace(".tsv", ".xlsx")
            q_nn_stats_sorted.to_excel(out_xlsx, index=False)
            logger.info(f"Saved top features explaining similarity between {query_id} and {nn_id}: {q_nn_stats_sorted['feature'].tolist()}")

            if group_map:
                
                q_nn_group_stats = group_feature_stats(q_nn_stats, group_map)
                q_nn_group_stats = q_nn_group_stats.sort_values('mean_abs_median_diff').head(args.top_features)
                out_tsv = os.path.join(args.output_dir, f"{query_id}_vs_{nn_id}_top_groups.tsv")
                q_nn_group_stats.to_csv(out_tsv, sep="\t", index=False)
                out_xlsx = out_tsv.replace(".tsv", ".xlsx")
                q_nn_group_stats.to_excel(out_xlsx, index=False)
                logger.info(f"Saved top compartments explaining similarity between {query_id} and {nn_id}: {q_nn_group_stats['group'].tolist()}")


if __name__ == "__main__":
    main()
