#!/usr/bin/env python3
# coding: utf-8

"""
Run CLIPn Integration on Cell Painting Data
-------------------------------------------

This script:
- Loads and merges multiple reference and query datasets.
- Harmonises column features across datasets.
- Encodes labels for compatibility with CLIPn.
- Runs CLIPn integration analysis (either train on references or integrate all).
- Decodes labels post-analysis, restoring original annotations.
- Outputs results, including latent representations and similarity matrices.

All outputs are tab-separated (TSV). No comma-separated outputs are written.

Command-line arguments:
-----------------------
    --datasets_csv      : Path to TSV/CSV listing dataset names and paths
                          with columns: 'dataset' and 'path'.
    --out               : Directory to save outputs.
    --experiment        : Experiment name for file naming.
    --mode              : Operation mode ('reference_only' or 'integrate_all').
    --clipn_param       : Optional CLIPn parameter for logging only.
    --latent_dim        : Dimensionality of latent space (default: 20).
    --lr                : Learning rate for CLIPn (default: 1e-5).
    --epoch             : Number of training epochs (default: 500).
    --save_model        : If set, save the trained CLIPn model after training.
    --load_model        : Path (or glob) to a previously saved CLIPn model to load.
    --scaling_mode      : 'all', 'per_plate', or 'none' (default: 'all').
    --scaling_method    : 'robust' or 'standard' (default: 'robust').
    --skip_standardise  : If set, skip feature scaling.
    --reference_names   : Space-separated list of dataset names to use as
                          references (only used in 'reference_only' mode).
    --aggregate_method  : Aggregate image-level latent to compound-level
                          using 'median' (default), 'mean', 'min', or 'max'.
    --annotations       : Optional annotation file (TSV) to merge using
                          Plate_Metadata and Well_Metadata.
    --impute            : Missing data imputation method: 'none' (default) or 'knn'
    --impute_k          : Number of neighbours for KNN imputation (default: 50).
    --no_plot_loss      : If set, disable plotting and saving the training loss curve and TSV.

"""

from __future__ import annotations

import argparse
import glob
import logging
import os  # must be before 'import torch'
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# One known-good set; pick CPU/GPU build to match your cluster
#pip install --upgrade --no-deps \
#  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
#pip install --upgrade onnx==1.14.1 onnxruntime==1.16.3 onnxscript==0.1.0
#
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Callable
import re
import numpy as np
import pandas as pd
from typing import Sequence

import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import psutil
import concurrent.futures
import torch
import torch.serialization
import gzip
from clipn.model import CLIPn
import math
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn import set_config
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


from cell_painting.process_data import (
    prepare_data_for_clipn_from_df,
    run_clipn_simple,
    standardise_metadata_columns,)

from cell_painting.clipn_lib import (
    save_training_loss,
    register_clipn_for_pickle,
    save_training_loss,
    run_training_diagnostics,
    extend_model_encoders,
    run_clipn_integration)

from cell_painting.AI import (
    torch_load_compat,
    configure_torch_performance,
    precision_at_k,
    plot_and_save_precision_curves,
    dataset_mixing_entropy,
    plot_and_save_entropy,
    compute_and_save_silhouette,
    save_latent_variance_report,
    wbd_ratio_per_compound)

from cell_painting.knn_lib import (
    build_knn_index,
    save_knn_outputs,
    simple_knn_qc,
    run_knn_analysis,
    aggregate_for_knn
    )

from cell_painting.dataframe_lib import (
    extract_latent_and_meta,
    detect_csv_delimiter,
    mode_nonnull,
    aggregate_latent_from_decoded,
    exclude_technical_features,
    ensure_library_column,
    load_and_harmonise_datasets,
    harmonise_numeric_columns,
    safe_to_csv,
    load_single_dataset,
    encode_labels,
    decode_labels
)

from cell_painting.general import (
    log_memory_usage,
    read_table_auto,
    aggregate_latent_per_compound,
    clean_nonfinite_features,
    clean_and_impute_features,
    clean_and_impute_features_knn,
    merge_annotations

)

from cell_painting.dataframe_process import (
    scale_features,
    mode_strict)

# Global timer (for memory log timestamps)
_SCRIPT_START_TIME = time.time()

# Make sklearn return DataFrames
set_config(transform_output="pandas")


def setup_logging(out_dir: str | Path, experiment: str) -> logging.Logger:
    """
    Configure logging with stream (stderr) and file handlers.

    Parameters
    ----------
    out_dir : str | Path
        Output directory for logs.
    experiment : str
        Experiment name; used for the log filename.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_dir = Path(out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"{experiment}_clipn.log"

    logger = logging.getLogger("clipn_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_filename, mode="w")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info("Python Version: %s", sys.version_info)
    logger.info("Command-line Arguments: %s", " ".join(sys.argv))
    logger.info("Experiment Name: %s", experiment)

    return logger



# Technical, non-biological columns that must never be used as features
TECHNICAL_FEATURE_BLOCKLIST = {"ImageNumber","Number_Object_Number","ObjectNumber","TableNumber"}



def apply_threads(n: int, logger):
    """
    Set BLAS/OpenMP and PyTorch threads to exactly 'n'.

    Parameters
    ----------
    n : int
        Thread count requested on the command line.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    int
        The thread count actually set.
    """
    n = max(1, int(n))

    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"
    ):
        os.environ[var] = str(n)

    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n // 2))
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        logger.info("GPU detected; set CPU-thread env vars for BLAS/pandas, "
                    "but PyTorch compute will mainly run on GPU.")
    else:
        logger.info("CPU backend: using %d threads.", n)


    logger.info("CPU threads set to %d (from --cpu_threads)", n)
    logger.info("only change threads if you are using Torch cpu backend")
    return n



# =====
# Main
# =====

def main(args: argparse.Namespace) -> None:
    """
    Execute CLIPn integration pipeline from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    logger = setup_logging(out_dir=args.out, experiment=args.experiment)
    configure_torch_performance(logger)
    logger.info("Starting CLIPn integration pipeline")
    logger.info("PyTorch Version: %s", getattr(torch, "__version__", "unknown"))


    # Threading: simple and explicit
    _CLIPN_THREADS = apply_threads(args.cpu_threads, logger)



    register_clipn_for_pickle()
    post_clipn_dir = Path(args.out) / "post_clipn"
    post_clipn_dir.mkdir(parents=True, exist_ok=True)

    plot_loss = not args.no_plot_loss


    # Load + harmonise - lots of logging here
    dataframes, common_cols = load_and_harmonise_datasets(
        datasets_csv=args.datasets_csv,
        logger=logger,
        mode=args.mode,
        audit_dir=Path(args.out) / "feature_audit",
    )
    logger.info("Loaded and harmonised %d datasets from %s", len(dataframes), args.datasets_csv)


    # Per-dataset sanity checks
    for name, df in dataframes.items():
        missing_meta = [c for c in ["cpd_id", "cpd_type", "Library"] if c not in df.columns]
        if missing_meta:
            raise ValueError(
                f"Sanity check failed after harmonisation for '{name}': missing {missing_meta}"
            )

    bad_index = [
        (name, type(df.index), df.index.names)
        for name, df in dataframes.items()
        if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["Dataset", "Sample"]
    ]
    if bad_index:
        details = "; ".join([f"{n} -> {t} names={names}" for n, t, names in bad_index])
        raise ValueError(f"Expected MultiIndex ['Dataset','Sample'] for all datasets; got: {details}")

    logger.info("Per-dataset sanity checks passed.")

    # Concatenate (deterministic order), normalise index names, sort for determinism
    combined_df = pd.concat(
        objs=[dataframes[name] for name in dataframes.keys()],
        axis=0,
        sort=False,
        copy=False,
    )
    if list(combined_df.index.names) != ["Dataset", "Sample"]:
        combined_df.index = combined_df.index.set_names(names=["Dataset", "Sample"])
    combined_df = combined_df.sort_index()

    dupe_count = combined_df.index.duplicated().sum()
    if dupe_count:
        logger.warning("Found %d duplicate (Dataset, Sample) index rows after concat.", dupe_count)

    logger.debug("Columns at this stage, combined: %s", combined_df.columns.tolist())
    log_memory_usage(logger=logger, prefix="[After loading datasets] ")

    # Metadata columns (never scale these)
    meta_columns = ["cpd_id", "cpd_type", "Plate_Metadata", "Well_Metadata", "Library"]
    for col in meta_columns:
        if col not in combined_df.columns:
            raise ValueError(f"Metadata column '{col}' not found in combined DataFrame after harmonisation.")
    logger.info("Metadata columns present in combined DataFrame: %s", meta_columns)
    logger.info("Combined DataFrame shape after harmonisation: %s", combined_df.shape)

    # Identify feature columns
    feature_cols = [
        col for col in combined_df.columns
        if col not in meta_columns and pd.api.types.is_numeric_dtype(combined_df[col])
    ]
    # drop technicals from the features we’re going to scale/project
    feature_cols = exclude_technical_features(feature_cols, logger)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after harmonisation. Check feature overlap and dtypes.")

    # Clean obvious non-finite values before any scaling
    combined_df = clean_nonfinite_features(
        df=combined_df,
        feature_cols=feature_cols,
        logger=logger,
        label="pre-scaling",
    )


    # Optional clean + impute (pre-scaling)
    if args.impute == "median":
        combined_df, dropped_feats = clean_and_impute_features(
            df=combined_df,
            feature_cols=feature_cols,
            logger=logger,
            groupby_cols=["Dataset", "Plate_Metadata"] if "Plate_Metadata" in combined_df.columns else ["Dataset"],
            max_nan_col_frac=0.30,
            max_nan_row_frac=0.80,
        )
        if dropped_feats:
            feature_cols = [c for c in feature_cols if c not in dropped_feats]

    elif args.impute == "knn":
        combined_df, dropped_feats = clean_and_impute_features_knn(
            df=combined_df,
            feature_cols=feature_cols,
            logger=logger,
            groupby_cols=["Dataset", "Plate_Metadata"] if "Plate_Metadata" in combined_df.columns else ["Dataset"],
            max_nan_col_frac=0.30,
            max_nan_row_frac=0.80,
            n_neighbors=args.impute_knn_k,
        )
        if dropped_feats:
            feature_cols = [c for c in feature_cols if c not in dropped_feats]

    else:
        logger.info("Imputation disabled (--impute none); skipping NaN drop/impute step.")
        dropped_feats = []



    # Hard guard
    if combined_df.shape[0] == 0:
        raise ValueError(
            "No rows left after NaN handling. Loosen thresholds (max_nan_col_frac/max_nan_row_frac) "
            "or inspect inputs for pervasive missingness."
        )



    # Optional scaling
    if args.skip_standardise:
        logger.info("Skipping feature scaling (--skip_standardise set).")
        df_scaled_all = combined_df
    else:
        logger.info("Scaling numeric features using mode='%s', method='%s'", args.scaling_mode, args.scaling_method)
        df_scaled_all = scale_features(
            df=combined_df,
            feature_cols=feature_cols,
            plate_col="Plate_Metadata",
            mode=args.scaling_mode,
            method=args.scaling_method,
            logger=logger,
        )
        logger.info("Scaled combined DataFrame shape: %s", df_scaled_all.shape)
        log_memory_usage(logger, prefix="[After scaling] ")
    # Clean again in case scaling produced NaN/Inf (e.g. zero-variance issues)
    df_scaled_all = clean_nonfinite_features(
        df=df_scaled_all,
        feature_cols=feature_cols,
        logger=logger,
        label="post-scaling",
    )

    # Final guard: never let technical counters into modelling
    df_scaled_all = df_scaled_all.drop(columns=[c for c in TECHNICAL_FEATURE_BLOCKLIST if c in df_scaled_all.columns], errors="ignore")
    

    feature_audit_dir = Path(args.out) / "feature_audit"
    feature_audit_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(feature_cols, name="feature").to_csv(
        feature_audit_dir / "features_used_after_imputation.tsv", sep="\t", index=False
    )
    logger.info("Final feature column count after cleaning: %d", len(feature_cols))

    # If imputation is off, fail fast if NaNs remain in features
    if args.impute == "none" and df_scaled_all[feature_cols].isna().any().any():
        n_missing = int(df_scaled_all[feature_cols].isna().sum().sum())
        raise ValueError(
            f"{n_missing} missing values remain in feature columns with --impute none. "
            "Enable imputation (--impute median|knn) or pre-clean inputs."
        )

    # ===== Optional: k-NN baseline on the pre-CLIPn feature space =====
    if args.knn_only or args.knn_also:
        logger.info("Running k-NN baseline (pre-CLIPn) with metric='%s', level='%s', k=%d",
                    args.knn_metric, args.knn_level, args.knn_k)

        # Build the matrix at the chosen granularity
        X_knn, meta_knn = aggregate_for_knn(
            df=df_scaled_all,
            feature_cols=feature_cols,
            level=args.knn_level,
            logger=logger,
        )

        # Compute neighbours
        knn_df = run_knn_analysis(
            X=X_knn,
            meta=meta_knn,
            k=args.knn_k,
            metric=args.knn_metric,
            logger=logger,
        )

        # Simple QC summary
        qc_df = simple_knn_qc(
            knn_df=knn_df,
            logger=logger,
        )

        # Save
        knn_dir = Path(args.out) / args.knn_out_subdir
        save_knn_outputs(
            knn_df=knn_df,
            qc_df=qc_df,
            X=X_knn,
            meta=meta_knn,
            out_dir=knn_dir,
            experiment=args.experiment,
            save_full_matrix=args.knn_save_full_matrix,
            metric=args.knn_metric,
            logger=logger,
        )


        if args.knn_only:
            logger.info("k-NN baseline completed; exiting early (--knn_only set).")
            return



    # Encode labels (cpd_id is explicitly not encoded)
    # ============================================================
    # Encode ONLY cpd_type for training labels; exclude other metadata
    # ============================================================

    meta_cols = ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]

    # Model features: strictly numeric, excluding metadata
    feature_cols_model = [
        c for c in df_scaled_all.columns
        if (c not in meta_cols) and pd.api.types.is_numeric_dtype(df_scaled_all[c])
    ]
    if not feature_cols_model:
        raise ValueError("No numeric feature columns for CLIPn after excluding metadata.")

    df_features = df_scaled_all.loc[:, feature_cols_model].copy()

    # Encode cpd_type (required for CLIPn training labels)
    if "cpd_type" not in df_scaled_all.columns:
        raise ValueError("Column 'cpd_type' is required but missing.")
    le_cpd = LabelEncoder()
    cpd_type_codes = le_cpd.fit_transform(df_scaled_all["cpd_type"].astype("string").fillna("NA"))

    # Build df_encoded: numeric features + 'cpd_type' (+ 'cpd_id' for bookkeeping)
    df_encoded = df_features.copy()
    df_encoded["cpd_type"] = cpd_type_codes.astype(np.int32)
    df_encoded["cpd_id"] = df_scaled_all["cpd_id"].astype("string").fillna("NA").astype("category")

    # Invariants: forbid other metadata in training matrix; ensure cpd_id is non-numeric
    bad_cols = set(df_encoded.columns) & {"Library", "Plate_Metadata", "Well_Metadata"}
    if bad_cols:
        raise ValueError(f"Unexpected metadata leaked into training matrix: {sorted(bad_cols)}")
    assert not pd.api.types.is_numeric_dtype(df_encoded["cpd_id"]), "cpd_id must be non-numeric."

    # Features actually used for the model (all numeric except the label)
    features_for_model = [
        c for c in df_encoded.select_dtypes(include=[np.number]).columns
        if c != "cpd_type"
    ]

    # computed this earlier:
    features_for_model = [
        c for c in df_encoded.select_dtypes(include=[np.number]).columns
        if c != "cpd_type"
    ]
    logger.info(
        "Training matrix validated: %d rows, %d feature cols + 'cpd_type' label (plus 'cpd_id' for bookkeeping).",
        df_encoded.shape[0], len(features_for_model)
    )


    # Optional: write the columns you really train on (exclude cpd_id)
    train_cols_path = Path(args.out) / "clipn_training_columns.tsv"
    pd.Series([*features_for_model, "cpd_type"], name="column").to_csv(
                train_cols_path, sep="\t", index=False)
    logger.info("Wrote final list of training columns to %s", train_cols_path)  

    # Expose encoder mapping for downstream decode/exports
    encoders = {"cpd_type": le_cpd}

    logger.info(
        "Prepared CLIPn inputs: features=%d, rows=%d; encoded 'cpd_type' with %d classes.",
        len(feature_cols_model), df_encoded.shape[0], len(le_cpd.classes_)
    )
    log_memory_usage(logger=logger, prefix="[After cpd_type-only encoding] ")

    # (Optional) save cpd_type mapping
    mapping_df = pd.DataFrame({
        "cpd_type": le_cpd.classes_,
        "cpd_type_encoded": np.arange(len(le_cpd.classes_), dtype=int),
    })
    safe_to_csv(
        df=mapping_df,
        path=(Path(args.out) / "label_mapping_cpd_type.tsv"),
        sep="\t",
        logger=logger,
    )

    # Report training input columns
    logger.info("CLIPn training DataFrame shape: %s", df_encoded.shape)
    logger.info("Training columns: %d total", df_encoded.shape[1])
    logger.info("First 10 training columns: %s", df_encoded.columns[:10].tolist())

    # dump all column names to a TSV for inspection
    all_cols_path = Path(args.out) / "clipn_all_columns.tsv"
    pd.Series(df_encoded.columns, name="column").to_csv(
        all_cols_path, sep="\t", index=False, header=True
    )
    logger.info("Wrote full list of columns to %s", all_cols_path)
    # Sanity: training matrix must be (features + exactly one label column 'cpd_type')

    bad_cols = set(df_encoded.columns) & {"Library", "Plate_Metadata", "Well_Metadata"}
    assert not bad_cols, f"Unexpected metadata in training matrix: {sorted(bad_cols)}"

    _non_numeric = [
        c for c in df_encoded.columns
        if c not in {"cpd_type", "cpd_id"} and not pd.api.types.is_numeric_dtype(df_encoded[c])
    ]


    assert not _non_numeric, f"Non-numeric feature columns present: {_non_numeric}"

    logger.info(
    "Training matrix validated: %d rows, %d feature cols + 'cpd_type' label.",
    df_encoded.shape[0], len(features_for_model)
)


    assert pd.api.types.is_integer_dtype(df_encoded["cpd_type"]) or pd.api.types.is_numeric_dtype(df_encoded["cpd_type"]), \
        f"'cpd_type' should be numeric-encoded; got {df_encoded['cpd_type'].dtype}"

    meta_cols = ["cpd_id", "cpd_type", "Plate_Metadata", "Well_Metadata", "Library"]
    decoded_meta_df = (
        df_scaled_all
        .reset_index()
        .loc[:, [c for c in ["Dataset", "Sample"] + meta_cols if c in df_scaled_all.columns]]
        .copy()
    )


    # =========================
    # Mode: reference-only flow
    # =========================
    if args.mode == "reference_only":
        reference_names = args.reference_names
        logger.info("Using reference datasets %s for training; projecting all others.", reference_names)

        # Check that reference names exist in the MultiIndex level
        ds_index_values = set(df_encoded.index.get_level_values("Dataset"))
        missing_refs = [n for n in reference_names if n not in ds_index_values]
        if missing_refs:
            raise ValueError(f"Reference dataset(s) not found in combined_df index: {missing_refs}")

        query_names = [name for name in dataframes if name not in reference_names]
        logger.info("Training on: %s; projecting: %s", reference_names, query_names)

        reference_df = df_encoded.loc[reference_names]
        query_df = df_encoded.loc[query_names] if query_names else pd.DataFrame()

        # Load model path usage policy:
        if args.load_model and args.mode != "integrate_all":
            logger.warning(
                "Loading a pre-trained model is typically used with 'integrate_all'. "
                "Proceeding with 'reference_only' but ensuring encoders align."
            )

        if args.load_model:
            model_files = glob.glob(args.load_model)
            if not model_files:
                raise FileNotFoundError(f"No model files matched pattern: {args.load_model}")
            model_path = model_files[0]
            logger.info("Loading pre-trained CLIPn model from: %s", model_path)
            model = torch_load_compat(model_path=model_path, weights_only=False)


            # Prepare and predict training references
            data_dict, _, _, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(reference_df)
            latent_dict = model.predict(data_dict)

            latent_frames = []
            for i, latent in latent_dict.items():
                name = dataset_key_mapping[i]
                df_latent = pd.DataFrame(latent)
                df_latent.index = pd.MultiIndex.from_product(
                    [[name], range(len(df_latent))], names=["Dataset", "Sample"]
                )
                latent_frames.append(df_latent)
            latent_df = pd.concat(latent_frames)

        else:
            # Train a new model on references
            logger.info("Training new CLIPn model on reference datasets")
            latent_df, cpd_ids, model, dataset_key_mapping = run_clipn_integration(
                df=df_encoded,
                logger=logger,
                clipn_param=args.clipn_param,
                output_path=args.out,
                experiment=args.experiment,
                mode=args.mode,
                latent_dim=args.latent_dim,
                lr=args.lr,
                epochs=args.epoch,
                skip_standardise=args.skip_standardise,
                plot_loss=plot_loss, 
            )
            if args.save_model:
                model_path = Path(args.out) / f"{args.experiment}_clipn_model.pt"
                torch.save(obj=model, f=model_path)
                logger.info("Trained CLIPn model saved to: %s", model_path)

        log_memory_usage(logger, prefix="[After CLIPn training] ")

        # Attach metadata for training references
        latent_training_df = latent_df.reset_index()
        training_metadata_df = decoded_meta_df[decoded_meta_df["Dataset"].isin(reference_names)]
        latent_training_df = latent_training_df.merge(
            right=training_metadata_df,
            on=["Dataset", "Sample"],
            how="left",
        )

        # Inject cpd_id from cpd_ids dict for safety (overrides if present)
        assert all(name in cpd_ids for name in latent_training_df["Dataset"].unique()), \
            "Missing cpd_id mappings for some datasets."
        latent_training_df["cpd_id"] = latent_training_df.apply(
            func=lambda row: cpd_ids.get(row["Dataset"], [None])[row["Sample"]]
            if row["Sample"] < len(cpd_ids.get(row["Dataset"], [])) else None,
            axis=1,
        )
        training_output_path = Path(args.out) / "training"
        training_output_path.mkdir(parents=True, exist_ok=True)

        safe_to_csv(df=latent_training_df,
                    path=training_output_path / "training_only_latent.tsv",
                    sep="\t",
                    logger=logger,)

        logger.debug("First 10 cpd_id values:\n%s", latent_training_df["cpd_id"].head(10).to_string(index=False))
        logger.debug("Unique cpd_id values (first 10): %s", latent_training_df["cpd_id"].unique()[:10])

        # Project queries (if any)
        if not query_df.empty:
            logger.info("Projecting query datasets onto reference latent space: %s", query_names)
            # Extend dataset_key_mapping to include query datasets
            max_existing_key = max(dataset_key_mapping.keys(), default=-1)
            new_keys = list(range(max_existing_key + 1, max_existing_key + 1 + len(query_names)))
            if len(new_keys) != len(query_names):
                raise ValueError("Internal error: key/name length mismatch while extending dataset_key_mapping.")
            for new_key, name in zip(new_keys, query_names):
                dataset_key_mapping[new_key] = name


            # Identify a reference encoder to copy from
            try:
                reference_encoder_key = next(
                    k for k, v in dataset_key_mapping.items()
                    if v in reference_names and k in model.model.encoders
                )
            except StopIteration as exc:
                logger.error(
                    "No valid reference_encoder_key found. "
                    "None of the references matched trained encoders."
                )
                raise exc

            extend_model_encoders(model=model, new_keys=new_keys, reference_key=reference_encoder_key, logger=logger)

            # Build model input for queries (drop metadata cols that the model should not see)
            dataset_key_mapping_inv = {v: k for k, v in dataset_key_mapping.items()}
            query_groups = query_df.groupby(level="Dataset", sort=False)
            cols_to_drop = [
                c for c in ["cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"]
                if c in query_df.columns
            ]
            query_data_dict_corrected = {
                dataset_key_mapping_inv[name]: group.droplevel("Dataset").drop(columns=cols_to_drop).values
                for name, group in query_groups if name in dataset_key_mapping_inv
            }

            projected_dict = model.predict(query_data_dict_corrected)
            if not projected_dict:
                logger.warning("model.predict() returned an empty dictionary. Check dataset keys and inputs.")
            else:
                logger.debug("Projected %d datasets into latent space: %s", len(projected_dict), list(projected_dict.keys()))

            projected_frames = []
            query_cpd_ids: Dict[str, List[str]] = {}
            for i, latent in projected_dict.items():
                name = dataset_key_mapping[i]
                df_proj = pd.DataFrame(latent)
                df_proj.index = pd.MultiIndex.from_product(
                    [[name], range(len(df_proj))], names=["Dataset", "Sample"]
                )
                projected_frames.append(df_proj)
                # recover cpd_id from original query_df by ordered Sample
                query_cpd_ids[name] = query_df.loc[name]["cpd_id"].tolist()

            latent_query_df = pd.concat(projected_frames).reset_index()
            latent_query_df["cpd_id"] = latent_query_df.apply(
                func=lambda row: query_cpd_ids.get(row["Dataset"], [None])[row["Sample"]],
                axis=1,
            )

            query_output_path = Path(args.out) / "query_only" / f"{args.experiment}_query_only_latent.tsv"
            query_output_path.parent.mkdir(parents=True, exist_ok=True)
            safe_to_csv(df=latent_query_df,
                        path=query_output_path,
                        sep="\t",
                        logger=logger,)
            logger.info("Query-only latent data saved to %s", query_output_path)

            # Merge training + query for downstream combined decode/outputs
            latent_df = pd.concat(
                [latent_df, latent_query_df.set_index(["Dataset", "Sample"])],
                axis=0,
                sort=False,
            )
            cpd_ids.update(query_cpd_ids)

    # ======================
    # Mode: integrate-all flow
    # ======================
    else:
        logger.info("Training and integrating CLIPn on all datasets")
        if args.load_model:
            model_files = glob.glob(args.load_model)
            if not model_files:
                raise FileNotFoundError(f"No model files matched pattern: {args.load_model}")
            model_path = model_files[0]
            logger.info("Loading pre-trained CLIPn model from: %s", model_path)
            model = torch_load_compat(model_path=model_path, weights_only=False)



            # Prepare and predict latent with loaded model
            data_dict, _, _, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(df_encoded)
            latent_dict = model.predict(data_dict)

            latent_frames = []
            for i, latent in latent_dict.items():
                name = dataset_key_mapping[i]
                df_latent = pd.DataFrame(latent)
                df_latent.index = pd.MultiIndex.from_product(
                    [[name], range(len(df_latent))], names=["Dataset", "Sample"]
                )
                latent_frames.append(df_latent)
            latent_df = pd.concat(latent_frames)

        else:
            latent_df, cpd_ids, model, dataset_key_mapping = run_clipn_integration(
                df=df_encoded,
                logger=logger,
                clipn_param=args.clipn_param,
                output_path=args.out,
                experiment=args.experiment,
                mode=args.mode,
                latent_dim=args.latent_dim,
                lr=args.lr,
                epochs=args.epoch,
                skip_standardise=args.skip_standardise,
                plot_loss=plot_loss,
            )
            if args.save_model:
                model_path = Path(args.out) / f"{args.experiment}_clipn_model.pt"
                torch.save(obj=model, f=model_path)
                logger.info("Trained CLIPn model saved to: %s", model_path)

    # =========================
    # Decode + persist artefacts
    # =========================
    latent_df = latent_df.reset_index()
    latent_df = pd.merge(
        left=latent_df,
        right=decoded_meta_df,
        on=["Dataset", "Sample"],
        how="left",
    )

    decoded_df = decode_labels(df=latent_df.copy(), encoders=encoders, logger=logger)

    # Clean up duplicate cpd_id columns if any
    if "cpd_id_x" in decoded_df.columns or "cpd_id_y" in decoded_df.columns:
        decoded_df["cpd_id"] = (
            decoded_df.get("cpd_id_x", pd.Series(dtype=object))
            .combine_first(decoded_df.get("cpd_id_y", pd.Series(dtype=object)))
            .combine_first(decoded_df.get("cpd_id", pd.Series(dtype=object)))
        )
        decoded_df = decoded_df.drop(columns=[c for c in ["cpd_id_x", "cpd_id_y"] if c in decoded_df.columns])

    # Drop rows missing cpd_id (sensible default)
    n_before = decoded_df.shape[0]
    decoded_df = decoded_df[decoded_df["cpd_id"].notna()]
    n_after = decoded_df.shape[0]
    if n_before != n_after:
        logger.warning("Dropped %d rows with missing cpd_id after decoding/merge.", n_before - n_after)

    # Persist decoded outputs (TSV only)
    main_decoded_path = Path(args.out) / f"{args.experiment}_decoded.tsv"
    safe_to_csv(df=decoded_df,
                path=main_decoded_path,
                sep="\t",
                logger=logger,)
    logger.info("Decoded data saved to %s", main_decoded_path)
    if not args.no_diagnostics:
        run_training_diagnostics(
                decoded_df=decoded_df,
                out_dir=Path(args.out),
                experiment=args.experiment,
                mode=args.mode,
                level=args.diag_level,
                k_nn=args.diag_k,
                metric=args.diag_metric,
                logger=logger,
            )
        logger.info("Diagnostics completed.")


    post_decoded_path = post_clipn_dir / f"{args.experiment}_decoded.tsv"
    safe_to_csv(df=decoded_df,
                path=post_decoded_path,
                sep="\t",
                logger=logger,)
    logger.info("Decoded data saved to %s", post_decoded_path)


    # Optional compound-level aggregation (from decoded table; categoricals by mode)
    if getattr(args, "aggregate_method", None):
        try:
            df_compound = aggregate_latent_from_decoded(
                decoded_df=decoded_df,
                aggregate=args.aggregate_method,
                logger=logger,
            )
        except Exception:
            logger.exception("Failed to aggregate latent space from decoded table.")
            raise

        agg_path = post_clipn_dir / f"{args.experiment}_CLIPn_latent_aggregated_{args.aggregate_method}.tsv"
        safe_to_csv(
            df=df_compound,
            path=agg_path,
            sep="\t",
            logger=logger,
        )
        logger.info("Aggregated latent space saved to: %s", agg_path)


    # Plate/Well lookup (if present)
    if {"Plate_Metadata", "Well_Metadata"}.issubset(decoded_df.columns):
        plate_well_df = decoded_df[["Dataset", "Sample", "cpd_id", "Plate_Metadata", "Well_Metadata"]].copy()
        plate_well_file = post_clipn_dir / f"{args.experiment}_latent_plate_well_lookup.tsv"
        safe_to_csv(df=plate_well_df,
                    path=plate_well_file,
                    sep="\t",
                    logger=logger,)
        logger.info("Saved Plate/Well metadata to: %s", plate_well_file)
    else:
        logger.warning("Plate_Metadata or Well_Metadata missing in decoded output — skipping plate/well export.")

    # Optional annotation merge
    if args.annotations:
        logger.info("Merging annotations from: %s", args.annotations)
        annot_merge_df = decoded_df.copy()
        # Reconstruct Plate/Well if needed from combined_df (kept in-memory as df_encoded index)
        merge_annotations(
            latent_df_or_path=annot_merge_df,
            annotation_file=args.annotations,
            output_prefix=str(post_clipn_dir / args.experiment),
            logger=logger,
        )

    # Label encoder mappings (TSV)
    try:
        mapping_dir = Path(args.out)
        mapping_dir.mkdir(parents=True, exist_ok=True)
        for column, encoder in encoders.items():
            mapping_path = mapping_dir / f"label_mapping_{column}.tsv"
            mapping_df = pd.DataFrame({column: encoder.classes_, f"{column}_encoded": range(len(encoder.classes_))})
            safe_to_csv(df=mapping_df,
                        path=mapping_path.with_suffix(".tsv"),
                        sep="\t",
                        logger=logger,)
            logger.info("Saved label mapping for %s to %s", column, mapping_path)
        logger.info("CLIPn integration completed.")
        log_memory_usage(logger, prefix="[Mostly finished] ")
    except Exception as exc:
        logger.warning("Failed to save label encoder mappings: %s", exc)

    logger.info("Columns at this stage, encoded: %s", df_encoded.columns.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIPn Integration.")
    parser.add_argument("--datasets_csv", required=True, help="TSV/CSV with columns: 'dataset', 'path'.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--experiment", required=True, help="Experiment name.")
    parser.add_argument(
        "--scaling_mode",
        choices=["all", "per_plate", "none"],
        default="all",
        help="How to scale features.",
    )
    parser.add_argument(
        "--scaling_method",
        choices=["robust", "standard"],
        default="robust",
        help="Scaler to use.",
    )
    parser.add_argument(
        "--mode",
        choices=["reference_only", "integrate_all"],
        required=True,
        help="CLIPn operation mode.",
    )
    parser.add_argument(
        "--clipn_param",
        type=str,
        default="default",
        help="Optional CLIPn parameter for logging only.",
    )
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent space dimensionality.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--epoch", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--save_model", action="store_true", help="Save trained CLIPn model.")
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Path or glob to a saved CLIPn model (.pt).",
    )
    parser.add_argument(
        "--reference_names",
        nargs="+",
        default=["reference1", "reference2"],
        help="Datasets to use for training in reference_only mode.",
    )
    parser.add_argument(
        "--aggregate_method",
        choices=["median", "mean", "min", "max"],
        default="median",
        help="Aggregate image-level latent to compound-level.",
    )
    parser.add_argument(
        "--skip_standardise",
        action="store_true",
        help="Skip standardising numeric columns if already scaled.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Optional annotation TSV to merge using Plate/Well.",
    )
    parser.add_argument("--cpu_threads",
                        type=int,
                        default=1,
                        help="Number of CPU threads to use (default: 1)."
                    )


    parser.add_argument(
        "--knn_k",
        type=int,
        default=10,
        help="Number of neighbours per entity.",
    )
    parser.add_argument(
        "--knn_metric",
        choices=["cosine", "euclidean", "correlation"],
        default="cosine",
        help="Distance metric for k-NN.",
    )
    parser.add_argument(
        "--knn_level",
        choices=["compound", "well", "image"],
        default="compound",
        help="Granularity for k-NN.",
    )
    parser.add_argument(
        "--knn_save_full_matrix",
        action="store_true",
        help="Also save the full pairwise distance matrix (guarded to small n).",
    )
    parser.add_argument(
        "--knn_out_subdir",
        type=str,
        default="post_knn",
        help="Subdirectory name for k-NN outputs inside --out.",
    )
    knn_group = parser.add_mutually_exclusive_group()
    knn_group.add_argument(
                        "--knn_only",
                        action="store_true",
                        help="Run k-NN on the pre-CLIPn feature space and exit early."
                    )
    knn_group.add_argument(
                            "--knn_also",
                            action="store_true",
                            help="Run k-NN baseline first, then continue to CLIPn."
                        )
    parser.add_argument(
    "--impute",
    choices=["median", "knn", "none"],
    default="none",
    help="Impute missing values before scaling/modeling. "
         "'median' = per-group median , 'knn' = KNNImputer, "
         "'none' = skip imputation (default)."
    )
    parser.add_argument(
        "--impute_knn_k",
        type=int,
        default=50,
        help="Number of neighbours for KNN imputation (used when --impute knn)."
    )
    parser.add_argument(
                        "--no_plot_loss",
                        action="store_true",
                        help="Disable plotting and saving the training loss curve and TSV.",
                    )

    parser.add_argument(
        "--no_diagnostics",
        action="store_true",
        help="Disable post-training diagnostics (Precision@k, mixing entropy, silhouette, variance, WBDR).",
    )
    parser.add_argument(
        "--diag_level",
        choices=["compound", "image"],
        default="compound",
        help="Granularity for diagnostics ('compound' aggregates by cpd_id).",
    )
    parser.add_argument(
        "--diag_k",
        type=int,
        default=15,
        help="Neighbourhood size k for diagnostics.",
    )
    parser.add_argument(
        "--diag_metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric for diagnostics.",
    )


    main(parser.parse_args())
