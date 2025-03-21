# cell_painting_sperm


## 4b_clipn_intergration_cmd.py – CLIPn Integration and Compound Latent Space Analysis

This script performs an integrated Cell Painting analysis using the CLIPn model to embed experimental and reference (STB) compound profiles into a shared latent space. It supports a complete end-to-end workflow including preprocessing, imputation, encoding, CLIPn training or inference, and latent space visualisation and distance analysis.

---

### Features

- Preprocesses and imputes experimental and STB datasets
- Encodes categorical metadata for CLIPn
- Runs or loads CLIPn latent embeddings
- Visualises latent space via UMAP
- Computes compound similarity distances
- Outputs fully traceable, encoded, imputed, and grouped datasets
- Saves latent representations, UMAP plots, and distance summaries
- Supports reproducible runs with optional loading of pretrained parameters

---

### Workflow Summary

#### 1. Load Input Data
- Loads two datasets: **experimental** data and **STB (reference)** compounds.
- Ensures both share a consistent set of numerical features.

#### 2. Preprocessing
- Aligns both datasets by intersecting common numeric feature columns.
- Removes non-numeric or unwanted columns using a regex pattern if defined.
- Optionally drops metadata columns before analysis.

#### 3. Imputation
- Missing values are filled using the mean strategy (or other strategies if implemented).
- Imputed datasets retain MultiIndex: `cpd_id`, `Library`, `cpd_type`.

#### 4. Label Encoding
- `cpd_id` and `cpd_type` are encoded into numeric values.
- Encoding maps and label arrays are stored separately.
- Useful for supervised contrastive learning in CLIPn.

#### 5. CLIPn Embedding
- CLIPn is either trained from scratch or loaded using saved optimal parameters.
- Latent embeddings are generated for both datasets.
- Parameters include latent dimension, learning rate, and training epochs.

#### 6. Latent Space Reconstruction
- Latent representations are converted back into DataFrames with the original MultiIndex.
- These are used for plotting, clustering, and similarity analysis.

#### 7. Visualisation (UMAP)
- UMAP plots are generated to visualise compound distributions.
- Supports:
  - Labelled plots
  - Experiment vs STB colouring
  - Highlighting specific compound types (e.g. toxic controls)

#### 8. Pairwise Similarity Matrix
- Computes Euclidean distances between all latent profiles.
- Supports averaging across replicates (e.g., by `cpd_id`, `Library`) or per-instance comparisons.
- Outputs a full square distance matrix in CSV format.

#### 9. Compound Similarity Summary
- Identifies closest and farthest compounds for each entry.
- Outputs a CSV summarising compound-wise nearest/farthest neighbours.

#### 10. Saving Intermediate Files
- Saves:
  - Imputed and label-encoded data
  - Grouped imputed data
  - Latent representations
  - All encoding and index mappings
  - UMAP plots and distance matrices

---

### Command-Line Options

These are set at the top of the script or passed dynamically via argument parser (if added later):

| Option                  | Type     | Description                                                                 |
|-------------------------|----------|-----------------------------------------------------------------------------|
| `--latent-dim`          | `int`    | Size of latent space (default: 100)                                        |
| `--lr`                  | `float`  | Learning rate for CLIPn training (default: 1e-4 or optimised value)         |
| `--epoch`               | `int`    | Number of training epochs (default: 200 or from optimised config)          |
| `--use-optimised-params`| `str`    | Path to JSON file with pretrained best parameters                          |
| `--skip-training`       | `flag`   | If set, skips CLIPn training and uses saved weights                        |
| `--output-folder`       | `str`    | Folder to store all outputs (latent data, plots, matrices, summaries)      |

> Note: These options are usually configured at the top of the script or injected from an external wrapper.

---

### Output Files

All outputs are saved under the specified `output_folder`, organised by CLIPn parameter values (latent dim, LR, epoch):

#### Latent Space
- `clipn_latent_representations.npz` – CLIPn latent embeddings
- `clipn_latent_metadata.pkl` – MultiIndex and label mapping

#### UMAP Visualisations
- `clipn_ldim_UMAP_labels_labeled.pdf` – UMAP with compound labels
- `UMAP_experiment_vs_stb.pdf` – UMAP coloured by dataset origin
- `clipn_ldim*_UMAP.csv` – Coordinates of each compound in 2D UMAP space

#### Distance Analysis
- `pairwise_compound_distances.csv` – Full distance matrix
- `compound_similarity_summary.csv` – Nearest and farthest neighbour summary

#### Intermediate Data
- `intermediate_files/experiment_data_imputed.csv`
- `intermediate_files/stb_data_imputed.csv`
- `intermediate_files/experiment_data_encoded.csv`
- `intermediate_files/stb_data_encoded.csv`
- `intermediate_files/experiment_grouped.csv`
- `intermediate_files/stb_grouped.csv`

#### Logs
- `clipn_log.txt` – Run log with parameters, progress and issues

---

### Usage Notes

- Ensure both datasets use the same feature extraction pipeline (e.g. CellProfiler).
- Use `gt` or `cpd_type` labels to identify controls or known clusters.
- Label mappings are saved to enable interpretation of latent space clustering.

---

### Dependencies

- `pandas`, `numpy`, `scikit-learn`, `umap-learn`, `matplotlib`, `seaborn`, `torch`, `cdist` (from `scipy`), and `clipn` (custom or local module).

---

### Suggested Downstream Analyses

- Similarity network clustering (`5_compound_similarity_network.py`)
- Toxic compound filtering via distance thresholds
- Latent space analysis for compound mechanism of action
- Integration with transcriptomic or proteomic embeddings (if available)

