"""
compare_mcp_class_across_runs.py

Compares nearest neighbour (NN) and UMAP neighbours of compound classes (e.g. MCP09/DDU)
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


def load_and_tag_all_neighbour_summaries(base_dir, compound_pattern):
    """Load and annotate all *_summary_neighbours.tsv files with metadata from folder names.

    Args:
        base_dir (str): Root directory containing CLIPn run subfolders.
        compound_pattern (str): Regex pattern to filter compound IDs (e.g. 'MCP|DDU').

    Returns:
        tuple: (DataFrame with all summaries, list of processed folder names)
        """
    all_summaries = []
    compound_names = set()
    valid_folders = []
    for run_folder in glob(os.path.join(base_dir, '*')):
        if not os.path.isdir(run_folder):
            continue
        folder_name = os.path.basename(run_folder)
        valid_folders.append(folder_name)
        metadata = extract_run_metadata(run_folder)
        summary_files = glob(os.path.join(run_folder, 'post_clipn', '*_summary_neighbours.tsv'))
        if not summary_files:
            print(f"Warning: No summary file found in {run_folder}")
            continue
        for summary_file in summary_files:
            try:
                print(f"Reading: {summary_file}")
                df = pd.read_csv(summary_file, sep='\t')
                match_df = df[df['Compound'].str.contains(compound_pattern, regex=True)]
                found = sorted(match_df['Compound'].unique())
                if found:
                    print(f"  Matched compounds: {found}")
                compound_names.update(found)
                match_df['RunFolder'] = folder_name
                match_df['Mode'], match_df['Epoch'], match_df['LatentDim'], match_df['Metric'] = metadata
                all_summaries.append(match_df)
            except Exception as e:
                print(f"Warning: Failed to process {summary_file}: {e}")
                continue
    if not all_summaries:
        raise ValueError("No matching neighbour summary files or compounds found.")
    print(f"\nTotal unique compounds matched: {len(compound_names)}")
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
    """Build a dictionary mapping compound to its top 5 neighbours from baseline.
        Args:
        baseline_df (pd.DataFrame): Filtered dataframe with only baseline rows.

    Returns:
        dict: {compound_id: [neighbour1, neighbour2, ...]}
    """
    lookup = {}
    for _, row in baseline_df.iterrows():
        lookup[row['Compound']] = row['NearestNeighbours'].split(',')[:5]
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
        df (pd.DataFrame): Input dataframe of compound summaries.

    Returns:
        pd.DataFrame: Summary table with Jaccard scores for NN vs UMAP.
    """
    overlap_results = []
    for _, row in df.iterrows():
        nn_list = row['NearestNeighbours'].split(',')[:5]
        umap_list = row['UMAPNeighbours'].split(',')[:5]
        jaccard = jaccard_similarity(nn_list, umap_list)
        overlap_results.append({
            'Compound': row['Compound'],
            'RunFolder': row['RunFolder'],
            'Mode': row['Mode'],
            'Epoch': row['Epoch'],
            'LatentDim': row['LatentDim'],
            'Metric': row['Metric'],
            'Jaccard_NN_vs_UMAP': jaccard
        })
    return pd.DataFrame(overlap_results)


def compare_across_runs(df, baseline_lookup):
    """Compare each compound's top 5 NN to baseline using Jaccard similarity.
    
    Args:
        df (pd.DataFrame): Full combined dataframe of all runs.
        baseline_lookup (dict): Dictionary from build_baseline_lookup().

    Returns:
        pd.DataFrame: Summary of Jaccard overlap for each compound across runs.
    """
    comparisons = []
    for _, row in df.iterrows():
        compound = row['Compound']
        run_name = row['RunFolder']
        if compound not in baseline_lookup:
            continue
        jaccard = jaccard_similarity(baseline_lookup[compound], row['NearestNeighbours'].split(',')[:5])
        comparisons.append({
            'Compound': compound,
            'CompareRun': run_name,
            'Jaccard_with_baseline': jaccard
        })
    return pd.DataFrame(comparisons)


def plot_jaccard_heatmap(df, index_col, output_file):
    """Plot Jaccard similarity heatmap from input table and save as PDF.
    Args:
        df (pd.DataFrame): Jaccard similarity table.
        index_col (str): Column name to use as y-axis (usually 'Compound').
        output_file (str): Path to PDF output.
        
    """
    pivot = df.pivot(index=index_col, columns='CompareRun', values='Jaccard_with_baseline')
    plt.figure(figsize=(12, max(4, 0.4 * len(pivot))))
    sns.heatmap(pivot, annot=True, cmap='viridis', cbar_kws={'label': 'Jaccard Similarity'})
    plt.title('Jaccard Similarity Heatmap')
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
    parser.add_argument('--compound_id', default='MCP|DDU', help='Regex to match compound names')
    parser.add_argument('--baseline_prefix', required=True, help='Folder name prefix for selecting baseline group')
    parser.add_argument('--preferred_latent', type=int, default=20, help='Latent dim to use for baseline group')
    parser.add_argument('--plot_heatmap', action='store_true', help='Plot heatmaps for both comparisons')
    args = parser.parse_args()

    all_df, _ = load_and_tag_all_neighbour_summaries(args.base_dir, args.compound_id)
    suffix = generate_output_suffix(args.baseline_prefix, args.preferred_latent)

    all_df.to_csv(f'combined_neighbours_{suffix}.tsv', sep='\t', index=False)

    within_df = compare_within_run(all_df)
    within_df.to_csv(f'within_run_overlap_summary_{suffix}.tsv', sep='\t', index=False)

    baseline_df = filter_for_baseline_reference(all_df, args.baseline_prefix, args.preferred_latent)
    baseline_lookup = build_baseline_lookup(baseline_df)

    across_df = compare_across_runs(all_df, baseline_lookup)
    across_df.to_csv(f'across_runs_overlap_summary_{suffix}.tsv', sep='\t', index=False)

    if args.plot_heatmap:
        plot_jaccard_heatmap(within_df.rename(columns={'Jaccard_NN_vs_UMAP': 'Jaccard_with_baseline'}), 'Compound', f'heatmap_within_run_{suffix}.pdf')
        plot_jaccard_heatmap(across_df, 'Compound', f'heatmap_across_runs_{suffix}.pdf')

if __name__ == '__main__':
    main()