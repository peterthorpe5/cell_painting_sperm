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


import argparse
import os
import pandas as pd
import numpy as np
import logging
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, ks_2samp
from collections import defaultdict
import warnings
from scipy.stats import mannwhitneyu, ks_2samp, wasserstein_distance
import numpy as np


def setup_logger(log_file):
    """
    Configure logging to both file and stdout.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("feature_attribution_logger")
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
        # Ensure cpd_type is present
        tmp = ensure_column(tmp, "cpd_type", "missing")
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


def get_wells_for_dmso(df, dmso_label='DMSO'):
    """
    Extract all DMSO wells (robust: returns any row where any column contains the string 'DMSO', ignoring case).

    Args:
        df (pd.DataFrame): DataFrame of well-level data.
        dmso_label (str): String label for DMSO.

    Returns:
        pd.DataFrame: Subset for DMSO controls.
    """
    mask = df.apply(lambda row: row.astype(str).str.upper().str.contains(dmso_label.upper()).any(), axis=1)
    return df[mask]




def compare_distributions(df1, df2, feature_cols, test='mw', logger=None):
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


def ensure_column(df, colname, fill_value="missing"):
    """
    Ensure the specified column exists in the DataFrame.
    If missing, add it with the given fill value.

    Args:
        df (pd.DataFrame): Input DataFrame.
        colname (str): Column name to ensure.
        fill_value (Any): Value to fill if column is missing.

    Returns:
        pd.DataFrame: DataFrame with the column present.
    """
    if colname not in df.columns:
        df[colname] = fill_value
    return df


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
    parser = argparse.ArgumentParser(
        description="Statistical attribution for cell painting NNs vs DMSO (batch mode)"
    )
    parser.add_argument('--ungrouped_list', required=True,
                        help="CSV/TSV with column 'path' for ungrouped feature files")
    parser.add_argument('--query_ids', required=False,
                        default="DDD02387619,DDD02948916,DDD02955130,DDD02958365",
                        help="Comma-separated string, one-per-line file, or filename with query IDs")
    parser.add_argument('--nn_file', required=True,
                        help="Nearest neighbours TSV (cpd_id, neighbour_id, distance)")
    parser.add_argument('--cpd_type_col', default='cpd_type',
                        help="Column for compound type")
    parser.add_argument('--cpd_id_col', default='cpd_id',
                        help="Column for compound ID")
    parser.add_argument('--dmso_label', default='DMSO',
                        help="Control label (case-insensitive, substring match)")
    parser.add_argument('--feature_group_file',
                        help="Feature-group mapping (CSV/TSV, optional)")
    parser.add_argument('--output_dir', required=True,
                        help="Directory for output tables")
    parser.add_argument('--nn_per_query', type=int, default=10,
                        help="Number of nearest neighbours to compare per query")
    parser.add_argument('--top_features', type=int, default=10,
                        help="Number of top features/groups to report per comparison")
    parser.add_argument('--test', default='mw', choices=['mw', 'ks'],
                        help="Statistical test: 'mw' or 'ks'")
    parser.add_argument('--log_file', default="feature_attribution.log",
                        help="Log file name")
    args = parser.parse_args()

    standard_col_order = [
    "feature", "stat", "raw_pvalue", "abs_median_diff", "emd", "med_query", "med_comp", "pvalue_bh"]


    standard_col_order_group = [
    "group",
    "mean_abs_median_diff",
    "min_raw_pvalue",
    "mean_emd",
    "min_pvalue_bh",
    "n_features_sig_fdr",
    "n_features_in_group"]


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
    dmso_df = get_wells_for_dmso(df, dmso_label=args.dmso_label)
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

        n_query_wells = query_df.shape[0]
        n_dmso_wells = dmso_df.shape[0]

        # Query vs DMSO
        logger.info(f"Comparing query compound {query_id} to DMSO controls...")
        q_dmso_stats = compare_distributions(query_df, dmso_df, feature_cols, test=args.test, logger=logger)
        # Always assign FDR correction to the full result table!
        if not q_dmso_stats.empty and q_dmso_stats['raw_pvalue'].notna().any():
            _, q_dmso_stats['pvalue_bh'], _, _ = multipletests(
                q_dmso_stats['raw_pvalue'].fillna(1), method='fdr_bh'
            )
        else:
            q_dmso_stats['pvalue_bh'] = np.nan

        q_dmso_stats['n_query_wells'] = n_query_wells
        q_dmso_stats['n_dmso_wells'] = n_dmso_wells

        q_dmso_stats = q_dmso_stats.sort_values('abs_median_diff', ascending=False)
        top_dmso = q_dmso_stats.head(args.top_features).copy()
        out_tsv = os.path.join(args.output_dir, f"{query_id}_vs_DMSO_top_features.tsv")
        out_xlsx = out_tsv.replace(".tsv", ".xlsx")
        reorder_and_write(top_dmso, out_tsv, standard_col_order + ["n_query_wells", "n_dmso_wells"], logger, "tsv")
        reorder_and_write(top_dmso, out_xlsx, standard_col_order + ["n_query_wells", "n_dmso_wells"], logger, "excel")

        logger.info(f"Saved top features distinguishing {query_id} from DMSO: {top_dmso['feature'].tolist()}")

        # Groups (compartments)
        if group_map:
            q_dmso_group_stats = group_feature_stats(q_dmso_stats, group_map, logger=logger)
            q_dmso_group_stats = q_dmso_group_stats.sort_values('mean_abs_median_diff', ascending=False)
            top_grp = q_dmso_group_stats.head(args.top_features).copy()
            out_tsv = os.path.join(args.output_dir, f"{query_id}_vs_DMSO_top_groups.tsv")
            out_xlsx = out_tsv.replace(".tsv", ".xlsx")
            
            q_dmso_group_stats['n_query_wells'] = n_query_wells
            q_dmso_group_stats['n_dmso_wells'] = n_dmso_wells
            reorder_and_write(top_grp, out_tsv, standard_col_order_group + ["n_query_wells", "n_dmso_wells"], logger, "tsv")
            reorder_and_write(top_grp, out_xlsx, standard_col_order_group + ["n_query_wells", "n_dmso_wells"], logger, "excel")

            logger.info(f"Saved top compartments distinguishing {query_id} from DMSO: {top_grp['group'].tolist()}")

        # Nearest neighbours (from NN table)
        nn_ids = load_nn_table(args.nn_file, query_id, nn_per_query=args.nn_per_query)
        logger.info(f"Loaded {len(nn_ids)} nearest neighbours for {query_id}: {nn_ids}")

        for nn_id in nn_ids:
            nn_df = get_well_level(df, nn_id, cpd_id_col=args.cpd_id_col)
            if nn_df.empty:
                logger.warning(f"No wells found for NN {nn_id}, skipping.")
                continue
            n_nn_wells = nn_df.shape[0]
            logger.info(f"Comparing {query_id} to NN {nn_id}...")
            q_nn_stats = compare_distributions(query_df, nn_df, feature_cols, test=args.test, logger=logger)
            if not q_nn_stats.empty and q_nn_stats['raw_pvalue'].notna().any():
                _, q_nn_stats['pvalue_bh'], _, _ = multipletests(
                    q_nn_stats['raw_pvalue'].fillna(1), method='fdr_bh'
                )
            else:
                q_nn_stats['pvalue_bh'] = np.nan

            q_nn_stats = q_nn_stats.sort_values('abs_median_diff')
            top_nn = q_nn_stats.head(args.top_features).copy()
            out_tsv = os.path.join(args.output_dir, f"{query_id}_vs_{nn_id}_top_features.tsv")
            out_xlsx = out_tsv.replace(".tsv", ".xlsx")

            q_nn_stats['n_query_wells'] = n_query_wells
            q_nn_stats['n_nn_wells'] = n_nn_wells
            reorder_and_write(top_nn, out_tsv, standard_col_order + ["n_query_wells", "n_nn_wells"], logger, "tsv")
            reorder_and_write(top_nn, out_xlsx, standard_col_order + ["n_query_wells", "n_nn_wells"], logger, "excel")

            logger.info(f"Saved top features explaining similarity between {query_id} and {nn_id}: {top_nn['feature'].tolist()}")

            if group_map:
                q_nn_group_stats = group_feature_stats(q_nn_stats, group_map, logger=logger)
                q_nn_group_stats = q_nn_group_stats.sort_values('mean_abs_median_diff')
                top_grp = q_nn_group_stats.head(args.top_features).copy()
                out_tsv = os.path.join(args.output_dir, f"{query_id}_vs_{nn_id}_top_groups.tsv")
                out_xlsx = out_tsv.replace(".tsv", ".xlsx")

                q_nn_group_stats['n_query_wells'] = n_query_wells
                q_nn_group_stats['n_nn_wells'] = n_nn_wells
                reorder_and_write(top_grp, out_tsv, standard_col_order_group + ["n_query_wells", "n_nn_wells"], logger, "tsv")
                reorder_and_write(top_grp, out_xlsx, standard_col_order_group + ["n_query_wells", "n_nn_wells"], logger, "excel")

                logger.info(f"Saved top compartments explaining similarity between {query_id} and {nn_id}: {top_grp['group'].tolist()}")

    logger.info("Feature attribution and statistical comparison completed.")

    print("Total rows:", df.shape[0])
    print("DMSO rows:", get_wells_for_dmso(df).shape[0])

    get_wells_for_dmso(df).to_csv("all_dmso_rows.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
