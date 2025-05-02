"""
compare_mcp_class_across_runs.py

Compares nearest neighbour (NN) and UMAP neighbours of cpd_id classes (e.g. MCP09/DDU)
across multiple CLIPn run folders. Computes Jaccard similarity within each run (NN vs UMAP),
and across runs (baseline vs other setups), with optional visualisation.
"""

import os
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from glob import glob


def extract_run_metadata(run_folder):
    """Extract mode, epoch, latent dimension, and metric from folder name.

    Args:
        run_folder (str): Folder name containing encoding details (e.g. '..._E300_L10').

    Returns:
        tuple: (mode (str), epoch (int), latent_dim (int), metric (str))"""
    mode = 'reference_only' if 'reference_only' in run_folder else 'integrate_all'
    latent_match = re.search(r'_L(\d+)', run_folder)
    epoch_match = re.search(r'_E(\d+)', run_folder)
    latent_dim = int(latent_match.group(1)) if latent_match else None
    epoch = int(epoch_match.group(1)) if epoch_match else None
    metric = 'cosine' if 'cosine' in run_folder.lower() else 'euclidean'
    return mode, epoch, latent_dim, metric



def pivot_neighbour_table(df):
    """
    Convert long-format summary file to wide format with NearestNeighbours and UMAPNeighbours columns.

    Args:
        df (pd.DataFrame): Raw long-format dataframe with 'cpd_id', 'nearest_cpd_id', and 'source'.

    Returns:
        pd.DataFrame: Pivoted dataframe with 'cpd_id', 'NearestNeighbours', and 'UMAPNeighbours'.
    """
    df = df[df['nearest_cpd_id'].notna() & df['source'].isin(['NN', 'UMAP'])].copy()
    grouped = (
        df.groupby(['cpd_id', 'source'])['nearest_cpd_id']
          .apply(lambda x: ','.join(x.astype(str)))
          .unstack(fill_value='')
          .reset_index()
    )
    grouped = grouped.rename(columns={
        'NN': 'NearestNeighbours',
        'UMAP': 'UMAPNeighbours'
    })
    return grouped

def jaccard_similarity(list1, list2):
    """Calculate Jaccard similarity between two lists.
        Args:
        list1 (list): First list of neighbours.
        list2 (list): Second list of neighbours.

    Returns:
        float: Jaccard similarity score."""
    set1, set2 = set(list1), set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0


def load_and_tag_all_neighbour_summaries(base_dir, cpd_id_pattern, suffix_filter=None):
    """Load and annotate all *_summary_neighbours.tsv files with metadata from folder names.

    Args:
        base_dir (str): Root directory containing CLIPn run subfolders.
        cpd_id_pattern (str): Regex pattern to filter cpd_id IDs (e.g. 'MCP|DDU').

    Returns:
        tuple: (DataFrame with all summaries, list of processed folder names)
        """
    all_summaries = []
    cpd_id_names = set()
    valid_folders = []
    
    for run_folder in glob(os.path.join(base_dir, '*')):
        if not os.path.isdir(run_folder):
            continue
        if suffix_filter and not run_folder.endswith(suffix_filter):
            continue

        if not os.path.isdir(run_folder):
            continue
        folder_name = os.path.basename(run_folder)
        valid_folders.append(folder_name)
        metadata = extract_run_metadata(run_folder)
        summary_files = (glob(os.path.join(run_folder, 'post_clipn', '*_summary_neighbours.tsv')) +
                        glob(os.path.join(run_folder, '*_summary_neighbours.tsv'))
                        )
        if not summary_files:
            print(f"Warning: No summary file found in {run_folder}")
            continue
        for summary_file in summary_files:
            try:
                print(f"Reading: {summary_file}")
                df = pd.read_csv(summary_file, sep='\t')
                match_df = df[df['cpd_id'].str.contains(cpd_id_pattern, regex=True)].copy()
                found = sorted(match_df['cpd_id'].unique())
                if found:
                    print(f"  Matched cpd_ids: {found}")
                cpd_id_names.update(found)

                # Pivot and add metadata
                match_df = pivot_neighbour_table(match_df)
                match_df['RunFolder'] = folder_name
                match_df['Mode'], match_df['Epoch'], match_df['LatentDim'], match_df['Metric'] = metadata
                all_summaries.append(match_df)


            except Exception as e:
                print(f"Warning: Failed to process {summary_file}: {e}")
                continue
    if not all_summaries:
        raise ValueError("No matching neighbour summary files or cpd_ids found.")
    print(f"\nTotal unique cpd_ids matched: {len(cpd_id_names)}")
    return pd.concat(all_summaries, ignore_index=True), valid_folders


def filter_for_baseline_reference(df, baseline_prefix, preferred_latent):
    """Filter rows matching the baseline prefix and latent dimension.
        Args:
        df (pd.DataFrame): Input dataframe.
        baseline_prefix (str): Folder prefix (e.g. 'STB_vs_mitotox_reference_only').
        preferred_latent (int): Latent dimension to filter on.

    Returns:
        pd.DataFrame: Filtered dataframe with baseline reference rows.
    """
    return df[(df['RunFolder'].str.startswith(baseline_prefix)) & (df['LatentDim'] == preferred_latent)].copy()


def build_baseline_lookup(baseline_df):
    """Build a dictionary mapping cpd_id to its top 5 neighbours from baseline.
        Args:
        baseline_df (pd.DataFrame): Filtered dataframe with only baseline rows.

    Returns:
        dict: {cpd_id_id: [neighbour1, neighbour2, ...]}
    """
    lookup = {}
    for _, row in baseline_df.iterrows():
        lookup[row['cpd_id']] = row['NearestNeighbours'].split(',')[:5]
    return lookup


def generate_output_suffix(baseline_prefix, latent_dim):
    """Generate safe output suffix from prefix and latent dimension.
    
    Args:
        baseline_prefix (str): Baseline prefix (e.g. 'STB_vs_mitotox_reference_only').
        latent_dim (int): Latent dimension value.

    Returns:
        str: Safe output string like 'STB_vs_mitotox_reference_only_L20'
    """
    clean_prefix = re.sub(r'[^a-zA-Z0-9]+', '_', baseline_prefix.strip())
    return f"{clean_prefix}_L{latent_dim}"


def compare_within_run(df):
    """Calculate NN vs UMAP overlap per run using Jaccard similarity.
    Args:
        df (pd.DataFrame): Input dataframe of cpd_id summaries.

    Returns:
        pd.DataFrame: Summary table with Jaccard scores for NN vs UMAP.
    """
    overlap_results = []

    for _, row in df.iterrows():

        nn_list = row.get('NearestNeighbours', '')
        umap_list = row.get('UMAPNeighbours', '')

        nn_list = nn_list.split(',')[:5] if isinstance(nn_list, str) else []
        umap_list = umap_list.split(',')[:5] if isinstance(umap_list, str) else []

        if not nn_list and not umap_list:
            continue


        jaccard = jaccard_similarity(nn_list, umap_list)
        overlap_results.append({
            'cpd_id': row['cpd_id'],
            'RunFolder': row['RunFolder'],
            'Mode': row['Mode'],
            'Epoch': row['Epoch'],
            'LatentDim': row['LatentDim'],
            'Metric': row['Metric'],
            'Jaccard_NN_vs_UMAP': jaccard
        })
    return pd.DataFrame(overlap_results)


def compare_across_runs(df, baseline_lookup, baseline_prefix):
    """Compare each cpd_id's top 5 NN to baseline using Jaccard similarity.
    
    Args:
        df (pd.DataFrame): Full combined dataframe of all runs.
        baseline_lookup (dict): Dictionary from build_baseline_lookup().

    Returns:
        pd.DataFrame: Summary of Jaccard overlap for each cpd_id across runs.
    """
    comparisons = []
    for _, row in df[~df['RunFolder'].str.startswith(baseline_prefix)].iterrows():

        cpd_id = row['cpd_id']
        run_name = row['RunFolder']
        if cpd_id not in baseline_lookup:
            continue
        jaccard = jaccard_similarity(baseline_lookup[cpd_id], row['NearestNeighbours'].split(',')[:5])
        comparisons.append({
            'cpd_id': cpd_id,
            'CompareRun': run_name,
            'Jaccard_with_baseline': jaccard
        })
    return pd.DataFrame(comparisons)


def plot_jaccard_heatmap(df, index_col, output_file):
    """
    Plot Jaccard similarity heatmap from input table and save as PDF.

    Args:
        df (pd.DataFrame): Jaccard similarity table.
        index_col (str): Column name to use as y-axis (usually 'cpd_id').
        output_file (str): Path to PDF output.
    """
    if 'CompareRun' not in df.columns:
        if 'RunFolder' in df.columns:
            df = df.assign(CompareRun=df['RunFolder'])
        else:
            raise ValueError("DataFrame must contain 'CompareRun' or 'RunFolder' column for plotting.")

    pivot = df.pivot(index=index_col, columns='CompareRun', values='Jaccard_with_baseline')
    plt.figure(figsize=(12, max(4, 0.4 * len(pivot))))
    sns.heatmap(pivot, annot=True, cmap='viridis', cbar_kws={'label': 'Jaccard Similarity'})
    plt.title('Jaccard Similarity Heatmap')
    plt.xticks(rotation=90, fontsize=4)
    plt.yticks(fontsize=6)
    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()
    print(f"Saved heatmap to: {output_file}")




def main():
    """Entry point for argument parsing and full comparison analysis.
    Parses arguments, loads input files, performs within-run and across-run comparisons,
    and optionally plots results as heatmaps.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True, help='Folder containing CLIPn run subfolders')
    parser.add_argument('--compound_id', default='MCP05|DDD0', help='Regex to match cpd_id names')
    parser.add_argument('--baseline_prefix', required=True, help='Folder name prefix for selecting baseline group')
    parser.add_argument('--preferred_latent', type=int, default=20, help='Latent dim to use for baseline group')
    parser.add_argument('--plot_heatmap', action='store_true', help='Plot heatmaps for both comparisons')
    parser.add_argument('--output_dir', default='comparison_results', help='Output directory for result files')
    parser.add_argument('--suffix_filter', default=None,
                    help='Optional suffix to restrict run folders (e.g., "_mitotox")')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_df, _ = load_and_tag_all_neighbour_summaries(args.base_dir, args.compound_id, args.suffix_filter)
    suffix = generate_output_suffix(args.baseline_prefix, args.preferred_latent)

    all_df.to_csv(os.path.join(args.output_dir, f'combined_neighbours_{suffix}.tsv'), sep='\t', index=False)

    within_df = compare_within_run(all_df)
    within_df.to_csv(os.path.join(args.output_dir, f'within_run_overlap_summary_{suffix}.tsv'), sep='\t', index=False)

    baseline_df = filter_for_baseline_reference(all_df, args.baseline_prefix, args.preferred_latent)
    baseline_lookup = build_baseline_lookup(baseline_df)

    across_df = compare_across_runs(all_df, baseline_lookup, args.baseline_prefix)
    across_df.to_csv(os.path.join(args.output_dir,f'across_runs_overlap_summary_{suffix}.tsv'), sep='\t', index=False)

    if args.plot_heatmap:
        plot_jaccard_heatmap(within_df.rename(columns={'Jaccard_NN_vs_UMAP': 'Jaccard_with_baseline'})
                    .assign(CompareRun=lambda x: x['RunFolder']), 
                    'cpd_id',
                    os.path.join(args.output_dir, f'heatmap_within_run_{suffix}.pdf')
                )

        plot_jaccard_heatmap(across_df, 'cpd_id', os.path.join(args.output_dir, f'heatmap_across_runs_{suffix}.pdf'))

if __name__ == '__main__':
    main()