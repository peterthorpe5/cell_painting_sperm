# prepare_anomaly_detection_data.py

Prepare Cell Painting well-level features for Anomaly Detection Screening (ZaritskyLab, 2024).

This script harmonises, imputes, scales, feature-selects, and splits well-level Cell Painting data for input into anomaly detection pipelines. It supports robust, reproducible preprocessing for single- or multi-plate experiments.

## Features

- Metadata harmonisation for robust merging.
- Imputation of missing values (`median` or `KNN`, default: `knn`).
- Per-plate feature scaling to reduce batch effects (`standard`, `robust`, or `auto`, default: `auto`).
- Z-scoring using mean/std or median/MAD.
- Feature selection via pycytominer (removes low-variance, highly-correlated, or high-NA features).
- Train/Validation/Test splitting of controls per plate (fractions configurable).
- Aligned output columns for PyTorch/ML compatibility.
- Logging and output of per-feature normality statistics.
- Outputs:
  - `train_controls.tsv`
  - `val_controls.tsv`
  - `test_controls.tsv`
  - `treatments.tsv`
  - `feature_normality.tsv`
  - `all_preprocessed.tsv` (fully preprocessed dataset, not split)

## Requirements

- Python 3.7+
- `pandas`
- `numpy`
- `scikit-learn`
- `pycytominer`
- `scipy`

Install requirements (example):

```bash
pip install pandas numpy scikit-learn pycytominer scipy
```



## Usage



```bash
python prepare_anomaly_detection_data.py \
  --input_file cellprofiler_well_profiles.tsv \
  --output_dir prepped/ \
  --control_label DMSO \
  --zscore_method mean \
  --impute knn \
  --scale_per_plate \
  --scale_method auto \
  --train_frac 0.6 --val_frac 0.2 --test_frac 0.2 \
  --na_cutoff 0.05 --corr_threshold 0.9 --unique_cutoff 0.01 --freq_cut 0.05

```


##  Key Arguments
--input_file: Tab- or comma-separated file with harmonised well-level features and metadata.

--output_dir: Output directory (created if not present).

--control_label: Value in cpd_type that defines negative controls (e.g. DMSO).

--scale_per_plate: Scale features separately for each plate (recommended for batch correction).

--scale_method: Scaling method (standard, robust, auto, or none). auto chooses based on feature normality.

--impute: Imputation method for missing values (median, knn, or none).

--zscore_method: Standardisation method (mean for normal data, median for robust scaling).

--train_frac, --val_frac, --test_frac: Proportions for splitting controls. Must sum to 1.0.

--plate_col, --well_col, --cpd_type_col: Specify column names for flexibility.

See python prepare_anomaly_detection_data.py --help for all options.

##  Output Files
train_controls.tsv, val_controls.tsv, test_controls.tsv: Split control wells for training, validation, and testing.

treatments.tsv: All wells not labelled as controls.

feature_normality.tsv: Per-feature results from the Shapiro-Wilk normality test.

all_preprocessed.tsv: Entire harmonised, imputed, scaled, and feature-selected dataset before splitting.

##  Notes
Z-scoring with --zscore_method median is more robust to outliers.

Auto-scaling uses the Shapiro-Wilk test: if >80% of features are normal, uses standard scaler; otherwise, uses robust scaler.

All outputs are tab-delimited.

##  Citation
If you use this workflow, please cite the relevant methods and the ZaritskyLab Anomaly Detection Screening preprint (2024).