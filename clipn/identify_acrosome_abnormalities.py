#!/usr/bin/env python3
"""
Batch comparison: All compounds vs. DMSO—Acrosome Features

- Loads well-level feature tables from a file list (CSV/TSV with column 'path')
- For each compound, compares its wells to DMSO wells using Mann–Whitney U test.
- Results for all features and acrosome features (by group) with FDR correction.
- Outputs both full and significant results to output_dir and output_dir/significant.

Usage:
    python batch_acrosome_vs_dmso.py --ungrouped_list dataset_paths.txt [--output_dir acrosome_vs_DMSO]
"""

import argparse
import os
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
from collections import defaultdict
import logging
import warnings
from scipy.stats import mannwhitneyu, ks_2samp, wasserstein_distance
import numpy as np


def setup_logger(log_file):
    """
    Configure a logger for both file and console output.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("acrosome_vs_dmso_logger")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def reorder_and_write(df, filename, col_order, logger, filetype="tsv"):
    """
    Ensure output DataFrame has all desired columns, in order.
    Any missing columns will be filled with NaN.

    Args:
        df (pd.DataFrame): Data to write.
        filename (str): Output file path.
        col_order (list): Desired column order.
        logger (logging.Logger): Logger for messages.
        filetype (str): 'tsv' or 'excel'.
    """
    # Only include columns that are present, in order. Add missing as NaN.
    for col in col_order:
        if col not in df.columns:
            df[col] = np.nan
    cols_in_order = [col for col in col_order if col in df.columns]
    # Append any others at the end
    remaining = [c for c in df.columns if c not in cols_in_order]
    df_out = df[cols_in_order + remaining]
    if filetype == "tsv":
        df_out.to_csv(filename, sep="\t", index=False)
    elif filetype == "excel":
        df_out.to_excel(filename, index=False)
    logger.debug(f"Wrote {filename} with columns: {df_out.columns.tolist()}")
    

def ensure_columns(df, required_cols, fill_value="missing"):
    """
    Ensure all columns in required_cols exist in the DataFrame,
    adding with fill_value if missing.

    Args:
        df (pd.DataFrame): Input DataFrame.
        required_cols (list): List of required column names.
        fill_value (Any): Value to fill for missing columns.

    Returns:
        pd.DataFrame: DataFrame with all required columns.
    """
    for col in required_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df

def load_ungrouped_files(list_file, logger, required_cols=None):
    """
    Load and concatenate all well-level feature files listed in a file,
    ensuring essential columns are present.

    Args:
        list_file (str): Path to dataset list (CSV/TSV) with column 'path'.
        logger (logging.Logger): Logger for messages.
        required_cols (list): Columns to ensure are present in each file.

    Returns:
        pd.DataFrame: Concatenated well-level data.
    """
    if required_cols is None:
        required_cols = ["cpd_id", "cpd_type"]
    logger.info(f"Reading file-of-files: {list_file}")
    df_list = pd.read_csv(list_file, sep=None, engine='python')
    if "path" not in df_list.columns:
        logger.error("Input list file must have a column named 'path'.")
        raise ValueError("Missing 'path' column in file-of-files.")
    dfs = []
    for path in df_list['path']:
        logger.info(f"Reading well-level data: {path}")
        tmp = pd.read_csv(path, sep="\t")
        tmp = ensure_columns(tmp, required_cols, fill_value="missing")
        dfs.append(tmp)
    # Harmonise columns (intersection only)
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    if not common_cols:
        logger.error("No common columns found across input files.")
        raise ValueError("No common columns in well-level files.")
    logger.info(f"{len(common_cols)} columns are common to all files.")
    dfs = [d[list(common_cols)] for d in dfs]
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined well-level data shape: {combined.shape}")
    return combined



def infer_acrosome_features(feature_cols, logger):
    """
    Identify features containing 'acrosome' (case-insensitive).

    Args:
        feature_cols (list): List of feature column names.
        logger (logging.Logger): Logger.

    Returns:
        list: Feature names containing 'acrosome'.
    """
    acrosome_feats = [f for f in feature_cols if "acrosome" in f.lower()]
    logger.info(f"Found {len(acrosome_feats)} acrosome features: {acrosome_feats[:5]} ...")
    return acrosome_feats


def infer_compartments(feature_cols, logger):
    """
    Infer compartments from feature names using the last underscore-separated token.

    Args:
        feature_cols (list): Feature column names.
        logger (logging.Logger): Logger.

    Returns:
        dict: Mapping from compartment to features.
    """
    group_map = defaultdict(list)
    for feat in feature_cols:
        tokens = feat.split("_")
        if len(tokens) > 1:
            compartment = tokens[-1].lower()
        else:
            compartment = "unknown"
        group_map[compartment].append(feat)
    logger.info(f"Inferred compartments: {list(group_map.keys())}")
    return dict(group_map)



def compare_distributions(df1, df2, feature_cols,logger, test='mw'):
    """
    Compare distributions of each feature between two well sets using non-parametric
    statistical tests and Earth Mover's Distance (EMD/Wasserstein).

    Args:
        df1 (pd.DataFrame): DataFrame for group 1 (e.g., query).
        df2 (pd.DataFrame): DataFrame for group 2 (e.g., DMSO/NN).
        feature_cols (list): List of feature column names.
        logger (logging.Logger): Logger for diagnostic output.
        test (str): Statistical test to use ('mw' for Mann–Whitney U, 'ks' for Kolmogorov–Smirnov).

    Returns:
        pd.DataFrame: Results with stat, raw_pvalue, abs_median_diff, emd, med_query, med_comp.
    """
    stats = []
    for feat in feature_cols:
        x1 = df1[feat].dropna()
        x2 = df2[feat].dropna()
        if len(x1) < 2 or len(x2) < 2:
            stat, p, emd = np.nan, np.nan, np.nan
            logger.debug(f"Skipping {feat}: not enough values in one or both groups.")
        else:
            # Choose test
            if test == 'mw':
                try:
                    stat, p = mannwhitneyu(x1, x2, alternative='two-sided')
                except Exception as e:
                    logger.warning(f"Mann–Whitney failed for {feat}: {e}")
                    stat, p = np.nan, np.nan
            else:
                try:
                    stat, p = ks_2samp(x1, x2)
                except Exception as e:
                    logger.warning(f"KS test failed for {feat}: {e}")
                    stat, p = np.nan, np.nan
            # EMD/Wasserstein
            try:
                emd = wasserstein_distance(x1, x2)
            except Exception as e:
                logger.warning(f"EMD failed for {feat}: {e}")
                emd = np.nan
        med1 = np.median(x1) if len(x1) else np.nan
        med2 = np.median(x2) if len(x2) else np.nan
        abs_diff = np.abs(med1 - med2)
        stats.append({
            'feature': feat,
            'stat': stat,
            'raw_pvalue': p,
            'abs_median_diff': abs_diff,
            'emd': emd,
            'med_query': med1,
            'med_comp': med2
        })
    logger.info("Completed comparison for all features with EMD.")
    return pd.DataFrame(stats)

def group_feature_stats(feat_stats, group_map, fdr_alpha=0.05, logger=None):
    """
    Summarise per-feature stats by group, including size and FDR counts.

    Args:
        feat_stats (pd.DataFrame): Per-feature stats with FDR.
        group_map (dict): Mapping from group to feature list.
        fdr_alpha (float): FDR threshold.
        logger (logging.Logger): Logger.

    Returns:
        pd.DataFrame: Per-group stats.
    """
    grouped = []
    for group, feats in group_map.items():
        group_df = feat_stats[feat_stats['feature'].isin(feats)]
        if group_df.empty:
            continue
        abs_diff = group_df['abs_median_diff'].mean()
        min_p = group_df['raw_pvalue'].min()
        mean_emd = group_df['emd'].mean() if 'emd' in group_df.columns else np.nan
        min_fdr = group_df['pvalue_bh'].min() if 'pvalue_bh' in group_df.columns else np.nan
        n_features = len(group_df)
        n_sig_fdr = (group_df['pvalue_bh'] <= fdr_alpha).sum() if 'pvalue_bh' in group_df.columns else np.nan
        grouped.append({
            'group': group,
            'mean_abs_median_diff': abs_diff,
            'min_raw_pvalue': min_p,
            'mean_emd': mean_emd,
            'min_pvalue_bh': min_fdr,
            'n_features_in_group': n_features,
            'n_features_sig_fdr': n_sig_fdr
        })
    if logger:
        logger.debug(f"Summarised {len(grouped)} feature groups.")
    return pd.DataFrame(grouped)


def main():
    """
    Main function to batch compare all compounds to DMSO using well-level features.
    """
    parser = argparse.ArgumentParser(description="Batch compare all compounds to DMSO for acrosome features.")
    parser.add_argument('--ungrouped_list', required=True, help="CSV/TSV with column 'path' for well-level files")
    parser.add_argument('--cpd_id_col', default="cpd_id", help="Compound ID column")
    parser.add_argument('--cpd_type_col', default="cpd_type", help="Compound type column")
    parser.add_argument('--dmso_label', default="DMSO", help="Label for DMSO")
    parser.add_argument('--output_dir', default="acrosome_vs_DMSO", help="Output folder")
    parser.add_argument('--top_features', type=int, default=10, help="Number of top features to report")
    parser.add_argument('--fdr_alpha', type=float, default=0.05, help="FDR threshold for significance")
    parser.add_argument('--log_file', default="acrosome_vs_dmso.log", help="Log file name")
    args = parser.parse_args()


    
    standard_col_order = ["feature", "stat", "raw_pvalue", "abs_median_diff",
                         "emd", "med_query", "med_comp", "pvalue_bh",
                         "n_query_wells", "n_dmso_wells"]


    standard_col_order_group = [
        "group",
        "mean_abs_median_diff",
        "min_raw_pvalue",
        "mean_emd",
        "min_pvalue_bh",
        "n_features_in_group",
        "n_features_sig_fdr",
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    sig_dir = os.path.join(args.output_dir, "significant")
    os.makedirs(sig_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, args.log_file))
    warnings.simplefilter("ignore")

    logger.info("Starting batch analysis: all compounds vs. DMSO.")
    df = load_ungrouped_files(args.ungrouped_list, logger)
    feature_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in [args.cpd_id_col, args.cpd_type_col]
    ]

    acrosome_feats = infer_acrosome_features(feature_cols, logger)
    group_map = infer_compartments(feature_cols, logger)

    # Find DMSO wells (search all columns for DMSO label)
    dmso_mask = df.apply(lambda row: row.astype(str).str.upper().str.contains(args.dmso_label.upper()).any(), axis=1)
    dmso_df = df[dmso_mask]
    logger.info(f"Found {dmso_df.shape[0]} DMSO wells.")

    # All non-DMSO compounds
    compound_ids = df.loc[~dmso_mask, args.cpd_id_col].dropna().unique().tolist()
    logger.info(f"Found {len(compound_ids)} non-DMSO compounds for analysis.")

    for cpd_id in compound_ids:
        logger.info(f"==== Analysing compound: {cpd_id} ====")
        query_df = df[(df[args.cpd_id_col] == cpd_id) & (~dmso_mask)]
        if query_df.empty:
            logger.warning(f"No wells found for {cpd_id}, skipping.")
            continue

        # All features
        logger.info(f"Comparing {cpd_id} to DMSO across all features.")
        all_stats = compare_distributions(query_df, dmso_df, feature_cols, logger)
        n_query_wells = query_df.shape[0]
        n_dmso_wells = dmso_df.shape[0]
        all_stats['n_query_wells'] = n_query_wells
        all_stats['n_dmso_wells'] = n_dmso_wells

        # Always assign pvalue_bh before slicing!
        reject, pvals_bh, _, _ = multipletests(all_stats['raw_pvalue'], method='fdr_bh')
        all_stats['pvalue_bh'] = pvals_bh
        all_stats = all_stats.sort_values('abs_median_diff', ascending=False)
        all_tsv = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_top_features.tsv")
        all_xlsx = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_top_features.xlsx")
        top_all_stats = all_stats.head(args.top_features).copy()

        reorder_and_write(top_all_stats, all_tsv, standard_col_order, logger, "tsv")
        reorder_and_write(top_all_stats, all_xlsx, standard_col_order, logger, "excel")

        logger.info(f"Saved top feature stats for {cpd_id}: {all_tsv}")

        # Significant only
        sig_stats = all_stats[all_stats['pvalue_bh'] <= args.fdr_alpha].copy()
        if not sig_stats.empty:
            sig_tsv = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_significant_features.tsv")
            sig_xlsx = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_significant_features.xlsx")
            reorder_and_write(sig_stats, sig_tsv, standard_col_order, logger, "tsv")
            reorder_and_write(sig_stats, sig_xlsx, standard_col_order, logger, "excel")


            logger.info(f"Saved significant features for {cpd_id}: {sig_tsv}")

        # Acrosome features
        if acrosome_feats:
            logger.info(f"Comparing {cpd_id} to DMSO across acrosome features.")
            acro_stats = compare_distributions(query_df, dmso_df, acrosome_feats, logger)
            n_query_wells = query_df.shape[0]
            n_dmso_wells = dmso_df.shape[0]
            acro_stats['n_query_wells'] = n_query_wells
            acro_stats['n_dmso_wells'] = n_dmso_wells



            reject, pvals_bh, _, _ = multipletests(acro_stats['raw_pvalue'], method='fdr_bh')
            acro_stats['pvalue_bh'] = pvals_bh
            acro_stats = acro_stats.sort_values('abs_median_diff', ascending=False)
            acro_tsv = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_top_features.tsv")
            acro_xlsx = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_top_features.xlsx")
            top_acro_stats = acro_stats.head(args.top_features).copy()
            reorder_and_write(top_acro_stats, acro_tsv, standard_col_order, logger, "tsv")
            reorder_and_write(top_acro_stats, acro_xlsx, standard_col_order, logger, "excel")



            logger.info(f"Saved acrosome feature stats for {cpd_id}: {acro_tsv}")

            sig_acro = acro_stats[acro_stats['pvalue_bh'] <= args.fdr_alpha].copy()
            if not sig_acro.empty:
                sig_acro_tsv = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_features.tsv")
                sig_acro_xlsx = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_features.xlsx")
                reorder_and_write(sig_acro, sig_acro_tsv, standard_col_order, logger, "tsv")
                reorder_and_write(sig_acro, sig_acro_xlsx, standard_col_order, logger, "excel")


                logger.info(f"Saved significant acrosome features for {cpd_id}: {sig_acro_tsv}")

        # Acrosome group
        if "acrosome" in group_map:
            logger.info(f"Summarising group stats for acrosome ({cpd_id}).")
            acro_group = {"acrosome": group_map["acrosome"]}
            acro_group_stats = group_feature_stats(all_stats, acro_group, logger=logger)

            acro_grp_tsv = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_group.tsv")
            acro_grp_xlsx = os.path.join(args.output_dir, f"{cpd_id}_vs_DMSO_acrosome_group.xlsx")
            # CORRECT — use group col order
            reorder_and_write(acro_group_stats, acro_grp_tsv, standard_col_order_group, logger, "tsv")
            reorder_and_write(acro_group_stats, acro_grp_xlsx, standard_col_order_group, logger, "excel")
            logger.info(f"Saved acrosome group stats for {cpd_id}: {acro_grp_tsv}")

            # Significant only (min_raw_pvalue threshold)
            sig_grp = acro_group_stats[acro_group_stats['min_raw_pvalue'] <= args.fdr_alpha].copy()
            if not sig_grp.empty:
                sig_grp_tsv = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_group.tsv")
                sig_grp_xlsx = os.path.join(sig_dir, f"{cpd_id}_vs_DMSO_acrosome_significant_group.xlsx")
                reorder_and_write(sig_grp, sig_grp_tsv, standard_col_order_group, logger, "tsv")
                reorder_and_write(sig_grp, sig_grp_xlsx, standard_col_order_group, logger, "excel")
                logger.info(f"Saved significant acrosome group stats for {cpd_id}: {sig_grp_tsv}")


    logger.info("Batch comparison complete.")


if __name__ == "__main__":
    main()
