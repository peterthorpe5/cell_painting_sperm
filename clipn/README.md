# Cell Painting Sperm: CLIPn Integration

This repository provides a pipeline for preprocessing Cell Painting data and performing integrative latent space analysis using [CLIPn](https://github.com/momeara/CLIPn). It consists of two main scripts:

- `impute_missing_data_feature_select.py` – imputes missing values, filters features, and prepares clean metadata-rich datasets.
- `run_clipn.py` – trains or applies a CLIPn model to one or more datasets and outputs latent embeddings, similarity data, and optional annotated results.

---


conda create -n clipn python=3.10

conda activate clipn

pip install -r requirements.txt

then you may need to run this:  

pip install --upgrade --no-deps \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install --upgrade onnx==1.14.1 onnxruntime==1.16.3 onnxscript==0.1.0

pip uninstall -y onnxscript
pip install --upgrade onnx==1.14.1 onnxruntime==1.16.3



conda install -y -c conda-forge --force-reinstall "numpy=1.26.*" "pandas=2.0.*"


## 1. Preprocessing: `impute_missing_data_feature_select.py`

### Description

Cleans and preprocesses Cell Painting assay data:
- Handles multiple CSV inputs
- Performs KNN or median imputation
- Applies correlation and variance-based feature filtering
- Optionally merges annotation files
- Outputs both grouped (averaged) and ungrouped datasets
- Logs all stages with dataset shape summaries

### Usage

```bash
python impute_missing_data_feature_select.py \
  --input_file_list files.txt \
  --experiment MyExperiment \
  --out processed_data \
  --impute knn \
  --knn_neighbors 5 \
  --correlation_threshold 0.98 \
  --annotation_file annotations.csv
```

### Key Arguments

| Argument                  | Description                                                           |
|---------------------------|-----------------------------------------------------------------------|
| `--input`                 | Folder or CSV file to process                                        |
| `--input_file_list`       | Text file listing input CSV files                                    |
| `--experiment`            | Name of the experiment                                                |
| `--out`                   | Output directory (default: `processed`)                              |
| `--impute`                | Imputation method: `median` or `knn`                                 |
| `--knn_neighbors`         | Number of neighbours for KNN (default: 5)                            |
| `--correlation_threshold` | Threshold to remove highly correlated features (default: 0.99)       |
| `--annotation_file`       | Optional CSV file with compound annotations                          |
| `--force_averaging`       | Force averaging even for single input files                          |

---

## 2. Latent Embedding: `run_clipn.py`

### Description

Trains and applies the CLIPn model across datasets. Supports:

- Reference-only training (projecting query datasets)
- Full integration of all datasets
- Label encoding and restoration
- Metadata and annotation merging
- Output of latent space representations in multiple formats

### Usage Example

```bash
python run_clipn.py \
  --datasets_csv datasets.csv \
  --out results \
  --experiment MyExperiment \
  --mode reference_only \
  --reference_names reference1 reference2 \
  --latent_dim 50 \
  --epoch 300 \
  --lr 1e-5 \
  --save_model \
  --annotations annotations.tsv
```

### Key Arguments

| Argument             | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `--datasets_csv`     | CSV listing dataset names and file paths                                   |
| `--out`              | Output directory                                                            |
| `--experiment`       | Name for the experiment; used in filenames and logs                         |
| `--mode`             | Operation mode: `reference_only` or `integrate_all`                         |
| `--reference_names`  | Names of datasets to use as reference (only with `reference_only` mode)     |
| `--latent_dim`       | Latent space dimensionality (default: 20)                                   |
| `--epoch`            | Number of training epochs (default: 500)                                    |
| `--lr`               | Learning rate (default: 1e-5)                                                |
| `--save_model`       | Save the trained CLIPn model for reuse                                      |
| `--load_model`       | Path (or glob pattern) to load a pretrained model                           |
| `--annotations`      | Optional annotation TSV file to merge using `Plate_Metadata` and `Well_Metadata` |

---

## Outputs

- CLIPn latent representations (`.npz`, `.csv`, `.tsv`)
- Separate folders for `training/` and `query_only/` if in `reference_only` mode
- Annotated latent files with compound metadata
- Plate and Well lookup tables
- Label encoding mappings per categorical field

---

## Notes

- All scripts include extensive logging of parameters, input files, and output status
- Automatically detects CSV/TSV delimiters
- Preserves and restores all key metadata columns (`cpd_id`, `cpd_type`, `Library`, `Plate_Metadata`, `Well_Metadata`)
- Designed for integration into high-throughput Cell Painting analysis pipelines

---

## Citation

If this software contributes to your research, please cite the original CLIPn preprint:  
NO LINK YET... 

---

## Funding

This project was funded by the **Melinda and Bill Gates Foundation**.

---

## Author

Developed by [Peter Thorpe](https://github.com/peterthorpe5)  
University of Dundee  
Bioinformatics and Cell Painting Integration
