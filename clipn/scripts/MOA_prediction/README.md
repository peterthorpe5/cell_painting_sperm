# make_pseudo_anchors.py

Create **pseudo-MOA anchors** by clustering compound embeddings, with optional **bootstrap-based k selection** for stable, reproducible cluster counts.  
These anchors can be used by [`centroid_moa_scoring.py`](./centroid_moa_scoring.py) to build centroids and score MOA for all compounds (including unlabeled ones).

---

## Why pseudo-anchors?

Most compounds have no MOA labels. We cluster the embedding space to produce **coherent groups** that act like “pseudo-labels” (anchors). These anchors:

- Provide **structure** to the embedding space.
- Enable **prototype/centroid MOA scoring** downstream.
- Allow mixing of **real labels** and **pseudo labels** (semi-supervised).

---

## What the script does

1. **Load embeddings** (TSV) and select numeric feature columns.
2. **Optionally aggregate** replicate rows per compound (`median`, `mean`, etc.).
3. **Cluster compounds**:
   - **Fixed k**, or
   - **Auto k** with optional **bootstrap stability** (consensus silhouette / PAC).
4. **Write outputs**: anchors table, cluster diagnostics, and optional k-selection stability table.

---

## Inputs

| Argument | Description |
|----------|-------------|
| `--embeddings_tsv` | TSV file with at least one ID column (e.g., `cpd_id`) and numeric feature columns (embedding dimensions). |
| `--id_col` | Identifier column name (default: `cpd_id`). The script auto-detects if not provided. |
| `--aggregate_method` | How to combine replicate rows per compound. Choices: `median` (default), `mean`, `trimmed_mean`, `geometric_median`. |
| `--trimmed_frac` | Trim fraction for `trimmed_mean` (default: `0.1`). |
| `--labels_tsv` | Optional TSV with user-provided labels to overlay (does not affect clustering). |
| `--labels_id_col` | ID column name in labels file (default: `cpd_id`). |
| `--labels_label_col` | Column in labels file containing label values (default: `label`). |

---

## Outputs

- **`--out_anchors_tsv`**  
  Table mapping each compound to a pseudo-MOA cluster:  


cpd_id pseudo_moa cluster_id
ABC0001 CL_001 1
ABC0002 CL_001 1
ABC0003 CL_002 2


- **`--out_summary_tsv`**  
One-row run summary (algorithm, chosen k, silhouette score, etc.).

- **`--out_clusters_tsv`**  
Per-cluster diagnostics (size, average silhouette, nearest-cluster cosine).

- **`--out_k_selection_tsv`** *(only when bootstrapping is enabled)*  
Per-k stability metrics and the **chosen k**.

> All files are tab-separated (TSV), never CSV.

---

## Clustering options

### K-means

- **Fixed k:**  
`--clusterer kmeans --n_clusters 32`

- **Auto k (no bootstrap):**  
`--clusterer kmeans --n_clusters -1`  
Uses a heuristic ≈√N within `--auto_min_clusters` and `--auto_max_clusters`.

### Auto k with bootstrap (recommended)

Add these flags:

- `--bootstrap_k_main` (activates bootstrapping)
- `--k_candidates_main "8,12,16,24,32"` (candidate k values)
- `--n_bootstrap_main 100` (number of bootstrap replicates)
- `--subsample_main 0.8` (fraction of rows per replicate)
- `--stability_metric_main consensus_silhouette` (metric: `consensus_silhouette` \| `pac` \| `mean_ari`)
- `--consensus_linkage_main average` (hierarchical linkage for consensus)
- `--consensus_pac_limits "0.1,0.9"` (low/high cut for PAC metric)

The script writes the chosen k and full diagnostics to `--out_k_selection_tsv`.

---

## Stability metrics explained (no maths)

- **Consensus silhouette (higher = better):**  
Measures how well the **consensus partition** separates clusters based on the agreement matrix from all bootstraps.

- **PAC (proportion of ambiguous clustering, lower = better):**  
Measures the fraction of pairwise similarities that are neither clearly clustered together nor clearly apart.

- **Mean ARI (higher = better):**  
Average Adjusted Rand Index across bootstrap runs; measures how similar different bootstrap clusterings are.

---

## Typical usage

### 1. Bootstrap auto-k (stable default)
```bash
python make_pseudo_anchors.py \
--embeddings_tsv ./EXP_decoded.tsv \
--out_anchors_tsv pseudo_anchors.tsv \
--out_summary_tsv anchors_summary.tsv \
--out_clusters_tsv anchors_clusters_summary.tsv \
--out_k_selection_tsv anchors_pseudo_k_selection.tsv \
--id_col cpd_id \
--aggregate_method median \
--clusterer kmeans \
--n_clusters -1 \
--auto_min_clusters 12 \
--auto_max_clusters 64 \
--bootstrap_k_main \
--k_candidates_main "8,12,16,24,32" \
--n_bootstrap_main 100 \
--subsample_main 0.8 \
--stability_metric_main consensus_silhouette \
--consensus_linkage_main average \
--consensus_pac_limits "0.1,0.9" \
--random_seed 42

```


2. Fixed k (fastest)


```bash
python make_pseudo_anchors.py \
  --embeddings_tsv ./EXP_decoded.tsv \
  --out_anchors_tsv pseudo_anchors.tsv \
  --id_col cpd_id \
  --aggregate_method median \
  --clusterer kmeans \
  --n_clusters 32 \
  --random_seed 42

```

Performance tips

For very large datasets:

Limit candidate k values ("12,16,24") and reduce --n_bootstrap_main (e.g., 25–50).

Use smaller --subsample_main (e.g., 0.6–0.8).

Ensure enough RAM to hold the full co-clustering matrix (size ≈ N×N for consensus).


Downstream

The resulting pseudo_anchors.tsv can be passed directly into:

```bash

python centroid_moa_scoring.py \
  --embeddings_tsv ./EXP_decoded.tsv \
  --anchors_tsv pseudo_anchors.tsv \
  --moa_col moa_final \
  --out_dir ./moa_scoring_results

```



# centroid_moa_scoring.py

Score **mode of action (MOA)** for compounds by building centroids (prototypes) from labelled or **pseudo-labelled** anchors and comparing every compound’s embedding to these centroids.  
Supports **adaptive shrinkage**, **CSLS hubness correction**, and **permutation-based FDR**.

---

## Why use this script?

- After you cluster compounds into **pseudo-anchors** (using [`make_pseudo_anchors.py`](./make_pseudo_anchors.py)) or if you already have MOA-labelled compounds, this script:
  - Builds **centroids** (one or more per MOA).
  - Scores every compound against these centroids using **cosine similarity** or **CSLS**.
  - Outputs **predicted MOA** per compound and confidence metrics.
  - Provides **p/q-values** for MOA assignment using permutation testing.

---

## Workflow overview

1. **Input embeddings** (TSV with compound features).
2. **Load anchors** (real or pseudo MOA labels).
3. **Aggregate replicates** (median, trimmed mean, etc.).
4. **Build centroids per MOA**  
   - Optionally split each MOA into **sub-centroids** (K-means or bootstrapped auto-k).
5. **Score compounds**  
   - Cosine similarity or **CSLS** (corrects for hubness).
   - Choose best MOA and report margins.
6. **(Optional) Permutation FDR**  
   - Shuffle labels to estimate p/q-values.

---

## Inputs

| Argument | Description |
|----------|-------------|
| `--embeddings_tsv` | TSV file with embeddings (well or compound level). |
| `--anchors_tsv` | TSV with at least `cpd_id` and a MOA column (e.g., from `make_pseudo_anchors.py`). |
| `--moa_col` | Column in anchors file to use as MOA label (default: `moa_final`). |
| `--id_col` | Compound identifier column (default: `cpd_id`). |
| `--aggregate_method` | How to combine replicates: `median` (default), `mean`, `trimmed_mean`, `geometric_median`. |
| `--trimmed_frac` | Trim fraction for `trimmed_mean` (default: `0.1`). |

---

## Centroid building options

- `--n_centroids_per_moa`  
  Fixed number of centroids per MOA (default: `1`).  
  Set to `-1` to enable **auto-k per MOA** (silhouette/bootstrapped).

- `--centroid_method`  
  How to build each centroid:  
  - `median` (default)
  - `mean`
  - `geometric_median`

- `--centroid_shrinkage`  
  Global shrinkage factor for centroids (default: `0.0`).  
  Reduces overfitting for small clusters.

- `--adaptive_shrinkage`  
  Enable adaptive shrinkage: shrink more when MOA group size is small.  
  - `--adaptive_shrinkage_c` (default `0.5`)  
  - `--adaptive_shrinkage_max` (default `0.3`)

---

## Scoring options

- `--use_csls`  
  Apply **Cross-Domain Similarity Local Scaling** (CSLS) to correct hubness.

- `--csls_k`  
  Neighbourhood size for CSLS. Set `-1` to auto-pick ≈√number_of_centroids.

- `--primary_score`  
  Which similarity to report:
  - `auto` (default: cosine unless CSLS is used)
  - `cosine`
  - `csls`

- `--auto_margin_threshold`  
  Minimum margin between top-1 and top-2 MOA scores to call a confident assignment (default: `0.02`).

- `--moa_score_agg`  
  How to combine scores if multiple centroids exist per MOA:
  - `max` (default)
  - `mean`

---

## Permutation FDR

- `--n_permutations`  
  Number of label-shuffle permutations for empirical p/q-values (default: `0` disables FDR).

---

## Output files

All outputs go to `--out_dir` (must exist or will be created):

| File | Description |
|------|-------------|
| `compound_predictions.tsv` | Main table: per compound MOA prediction, score, margin, p/q-values (if permutations used). |
| `centroids.tsv` | Centroid coordinates and metadata per MOA. |
| `centroids_summary.tsv` | Per-MOA summary: number of compounds, chosen sub-k (if auto), silhouette stats. |
| `score_matrix.tsv` | Full compound × centroid similarity matrix. |
| `permutation_fdr.tsv` *(optional)* | Empirical FDR results if `--n_permutations > 0`. |

---

## Typical usage

### 1. Basic MOA scoring (single centroid per MOA)
```bash
python centroid_moa_scoring.py \
  --embeddings_tsv ./EXP_decoded.tsv \
  --anchors_tsv pseudo_anchors.tsv \
  --out_dir moa_centroid_results \
  --id_col cpd_id \
  --aggregate_method median \
  --centroid_method median \
  --centroid_shrinkage 0.0 \
  --use_csls \
  --csls_k -1 \
  --primary_score auto \
  --auto_margin_threshold 0.02 \
  --moa_score_agg max \
  --adaptive_shrinkage \
  --adaptive_shrinkage_c 0.5 \
  --adaptive_shrinkage_max 0.3 \
  --random_seed 0 \
  --n_permutations 200

```

2. Auto-select sub-centroids per MOA


```
python centroid_moa_scoring.py \
  --embeddings_tsv ./EXP_decoded.tsv \
  --anchors_tsv pseudo_anchors.tsv \
  --out_dir moa_centroid_auto_subk \
  --id_col cpd_id \
  --aggregate_method median \
  --n_centroids_per_moa -1 \
  --centroid_method median \
  --use_csls \
  --adaptive_shrinkage \
  --random_seed 42

```

Key concepts (high-level)

Centroid
A representative embedding for a MOA group (median or mean of compound embeddings).

Sub-centroids (auto-k)
If a MOA is internally heterogeneous, the script can split it into multiple sub-clusters for better prototypes.

CSLS
Corrects hubness: normalises cosine similarity using local density, reducing bias toward dense regions.

Adaptive shrinkage
Pulls small centroids toward the global mean to reduce overfitting.

Permutation FDR
Shuffles MOA labels to estimate how often your scores could occur by chance, giving empirical p/q-values.

Downstream

The outputs can be visualised using plot_moa_centroids_2d.py
 for UMAP or other 2D projections:

```bash
 python plot_moa_centroids_2d.py \
  --moa_dir ./moa_centroid_results \
  --anchors_tsv pseudo_anchors.tsv \
  --assignment predictions \
  --predictions_tsv ./moa_centroid_results/compound_predictions.tsv \
  --projection umap \
  --adaptive_shrinkage \
  --adaptive_shrinkage_c 0.5 \
  --adaptive_shrinkage_max 0.3

```

# plot_moa_centroids_2d.py

Visualise **MOA centroids** and **compound predictions** in 2D (e.g., UMAP).  
Designed to work with the outputs of [`centroid_moa_scoring.py`](./centroid_moa_scoring.py) and the anchor file from [`make_pseudo_anchors.py`](./make_pseudo_anchors.py).

---

## Why use this script?

- Quickly **see how compounds cluster** by predicted MOA in a low-dimensional space.
- Overlay **MOA centroids** and highlight selected compounds.
- Support both **labelled** and **pseudo-labelled** data.
- Generate **publication-ready static plots** and optional interactive plots.

---

## Workflow overview

1. **Load embeddings / predictions** from `centroid_moa_scoring.py`.
2. **Embed into 2D** using UMAP or other methods.
3. **Overlay centroids** (prototypes).
4. **Colour points** by MOA, library, or any metadata.
5. **Optionally highlight compounds of interest**.

---

## Inputs

| Argument | Description |
|----------|-------------|
| `--moa_dir` | Output directory from `centroid_moa_scoring.py` (must contain `compound_predictions.tsv` and `centroids.tsv`). |
| `--anchors_tsv` | Anchors table used for scoring (e.g., from `make_pseudo_anchors.py`). |
| `--assignment` | What to colour points by: `predictions` (default), `anchors`, or `moa_col` if provided. |
| `--predictions_tsv` | Predictions table to use (default: `${moa_dir}/compound_predictions.tsv`). |
| `--projection` | Dimensionality reduction method: `umap` (default) or `pca`. |
| `--id_col` | Identifier column (default: `cpd_id`). |
| `--moa_col` | MOA column name if you want to override what to colour by (default: `moa_final`). |

---

## Centroid & shrinkage options

- `--n_centroids_per_moa`  
  Number of centroids to plot per MOA (default: `1`).

- `--centroid_method`  
  How centroids were built (`median`, `mean`, etc.) — should match `centroid_moa_scoring.py`.

- `--adaptive_shrinkage`  
  Apply the same adaptive shrinkage logic as in scoring when plotting.
  - `--adaptive_shrinkage_c` (default `0.5`)
  - `--adaptive_shrinkage_max` (default `0.3`)

---

## UMAP / projection options

- `--umap_metric`  
  Distance metric for UMAP (default: `cosine`).

- `--umap_n_neighbors`  
  Number of neighbours for UMAP (default: `15`).

- `--umap_min_dist`  
  Minimum distance for UMAP layout (default: `0.1`).

- `--random_seed`  
  Fix for reproducible layouts.

---

## Plot output options

- `--out_prefix`  
  Prefix for all outputs (e.g., `moa_map` → creates `moa_map.pdf`, `moa_map.png`, …).

- `--highlight_list`  
  One or more compound IDs to highlight.

- `--colour_by`  
  Column to colour points by (default: predicted MOA or whatever is set with `--assignment`).

- `--tooltip_columns`  
  Columns to include in interactive tooltips (for HTML plots).

---

## Outputs

- `<prefix>_scatter.pdf` / `<prefix>_scatter.png` — static 2D scatter plot of compounds coloured by chosen assignment.
- `<prefix>_centroids.pdf` — centroids plotted on top of the embedding.
- `<prefix>_interactive.html` — optional interactive plot (if supported by environment).
- Logs written to `<prefix>.log`.

---

## Typical usage

### 1. Plot predictions with centroids (UMAP)

```bash
python plot_moa_centroids_2d.py \
  --moa_dir ./moa_centroid_results \
  --anchors_tsv pseudo_anchors.tsv \
  --assignment predictions \
  --predictions_tsv ./moa_centroid_results/compound_predictions.tsv \
  --projection umap \
  --n_centroids_per_moa 1 \
  --centroid_method median \
  --adaptive_shrinkage \
  --adaptive_shrinkage_c 0.5 \
  --adaptive_shrinkage_max 0.3 \
  --out_prefix ./moa_centroid_results/moa_map \
  --random_seed 42

```

2. Highlight specific compounds

```bash

python plot_moa_centroids_2d.py \
  --moa_dir ./moa_centroid_results \
  --anchors_tsv pseudo_anchors.tsv \
  --assignment predictions \
  --projection umap \
  --highlight_list ABC02955130 Rotenone CCCP \
  --out_prefix ./moa_centroid_results/moa_map_highlight
```