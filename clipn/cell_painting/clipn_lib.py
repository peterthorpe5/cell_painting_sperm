#!/usr/bin/env python3
# coding: utf-8



from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import set_config
import csv
set_config(transform_output="pandas")


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
from sklearn import set_config
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

logger = logging.getLogger(__name__)


def register_clipn_for_pickle() -> None:
    """
    Best-effort registration of CLIPn for Torch's safe unpickling.

    On older PyTorch versions the 'add_safe_globals' helper does not exist.
    In that case this function becomes a no-op and loading still works
    provided the CLIPn class is importable.
    """
    try:
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([CLIPn])
    except Exception:
        # Optional hardening only; safe to ignore on older Torch.
        pass

def save_training_loss(
    *,
    loss_values: Sequence[float] | np.ndarray | "torch.Tensor",
    out_dir: str | Path,
    experiment: str,
    mode: str,
    logger: logging.Logger,
    expected_epochs: int | None = None,
    aggregate: str = "last",          
) -> tuple[Path, Path]:
    """
    Plot and save CLIPn training loss per epoch.

    - If loss is per-step and expected_epochs is provided, collapse to per-epoch.
    - Writes TSV and a PDF line plot.
    """
    def _to_float_list(x):
        if hasattr(x, "detach"):  # torch.Tensor
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return [float(v) for v in x]

    # 1) coerce + clean
    try:
        vals = _to_float_list(loss_values)
    except Exception:
        vals = [float(getattr(v, "item", lambda: v)()) if hasattr(v, "item") else float(v) for v in loss_values]
    vals = [v for v in vals if np.isfinite(v)]

    # 2) collapse per-step -> per-epoch if applicable
    if expected_epochs and expected_epochs > 0 and len(vals) != expected_epochs:
        if len(vals) % expected_epochs == 0:
            steps_per_epoch = len(vals) // expected_epochs
            collapsed = []
            for e in range(expected_epochs):
                block = vals[e*steps_per_epoch:(e+1)*steps_per_epoch]
                collapsed.append(block[-1] if aggregate == "last" else float(np.mean(block)))
            logger.info(
                "Collapsed per-step loss (%d points, %d/epoch) to per-epoch (%d points) using '%s'.",
                len(vals), steps_per_epoch, len(collapsed), aggregate
            )
            vals = collapsed
        else:
            logger.warning(
                "expected_epochs=%d but len(loss)=%d not divisible; leaving as-is (x-axis will reflect steps).",
                expected_epochs, len(vals)
            )

    # 3) write TSV
    epochs = list(range(1, len(vals) + 1))
    post_dir = Path(out_dir) / "post_clipn"
    post_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = post_dir / f"{experiment}_{mode}_clipn_training_loss.tsv"
    pd.DataFrame({"epoch": epochs, "loss": vals}).to_csv(tsv_path, sep="\t", index=False)

    # 4) plot PDF (or touch empty if <2 points)
    pdf_path = post_dir / f"{experiment}_{mode}_clipn_training_loss.pdf"
    if len(vals) < 2:
        logger.warning("Training loss has < 2 valid points; wrote TSV only -> %s", tsv_path)
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(pdf_path): pass
        except Exception:
            pass
        return tsv_path, pdf_path

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(epochs, vals)
    ax.set_title(f"CLIPn training loss — {experiment} [{mode}]")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)

    logger.info("Saved training loss TSV -> %s", tsv_path)
    logger.info("Saved training loss PDF -> %s", pdf_path)
    return tsv_path, pdf_path



def run_training_diagnostics(
    *,
    decoded_df: pd.DataFrame,
    out_dir: Path,
    experiment: str,
    mode: str,
    level: str = "compound",
    k_nn: int = 15,
    metric: str = "cosine",
    logger: logging.Logger,
) -> None:
    """
    Run post-training diagnostics on the latent space and save TSV/PDF outputs.

    Parameters
    ----------
    decoded_df : pandas.DataFrame
        Decoded latent table (must include latent dims and 'Dataset';
        ideally also 'cpd_id' and 'cpd_type').
    out_dir : pathlib.Path
        Base output directory.
    experiment : str
        Experiment name for file naming.
    mode : str
        'reference_only' or 'integrate_all'.
    level : str
        'compound' (default) or 'image'.
    k_nn : int
        Neighbourhood size for diagnostics.
    metric : str
        'cosine' (default) or 'euclidean'.
    logger : logging.Logger
        Logger instance.
    """
    diag_dir = Path(out_dir) / "training_diagnostics"
    X, meta, _ = extract_latent_and_meta(
        decoded_df=decoded_df,
        level=level,
        aggregate="median",
        logger=logger,
    )
    nn_idx, _ = build_knn_index(X=X, k=k_nn, metric=metric, logger=logger)

    if "cpd_id" in meta.columns:
        p_curve = precision_at_k(labels=meta["cpd_id"], nn_indices=nn_idx)
        plot_and_save_precision_curves(
            out_dir=diag_dir, experiment=experiment, mode=mode,
            label_name="cpd_id", prec_curve=p_curve, logger=logger,
        )
    if "cpd_type" in meta.columns:
        p_curve = precision_at_k(labels=meta["cpd_type"], nn_indices=nn_idx)
        plot_and_save_precision_curves(
            out_dir=diag_dir, experiment=experiment, mode=mode,
            label_name="cpd_type", prec_curve=p_curve, logger=logger,
        )

    if "Dataset" in meta.columns and meta["Dataset"].notna().any():
        ent = dataset_mixing_entropy(datasets=meta["Dataset"], nn_indices=nn_idx)
        plot_and_save_entropy(
            ent=ent, out_dir=diag_dir, experiment=experiment, mode=mode,
            num_datasets=int(meta["Dataset"].nunique(dropna=True)), logger=logger,
        )

    if "cpd_type" in meta.columns:
        compute_and_save_silhouette(
            X=X, labels=meta["cpd_type"], metric=metric,
            out_dir=diag_dir, experiment=experiment, mode=mode,
            label_name="cpd_type", logger=logger,
        )
    if "Dataset" in meta.columns:
        compute_and_save_silhouette(
            X=X, labels=meta["Dataset"], metric=metric,
            out_dir=diag_dir, experiment=experiment, mode=mode,
            label_name="Dataset", logger=logger,
        )

    save_latent_variance_report(
        X=X, out_dir=diag_dir, experiment=experiment, mode=mode, eps=1e-6, logger=logger,
    )

    if "cpd_id" in meta.columns:
        wbd_ratio_per_compound(
            X=X, meta=meta, k=k_nn, metric=metric,
            out_dir=diag_dir, experiment=experiment, mode=mode, logger=logger,
        )

    logger.info("Training diagnostics completed. Outputs in %s", diag_dir)


# ====================
# CLIPn core functions
# ====================

def extend_model_encoders(
    model: CLIPn,
    new_keys: Iterable[int],
    reference_key: int,
    logger: logging.Logger,
) -> None:
    """
    Extend CLIPn model's encoder mapping for new datasets using a reference encoder.

    Parameters
    ----------
    model : CLIPn
        Trained CLIPn model object.
    new_keys : Iterable[int]
        Keys of new datasets to be projected.
    reference_key : int
        Key of the reference dataset to copy the encoder from.
    logger : logging.Logger
        Logger instance.
    """
    for new_key in new_keys:
        model.model.encoders[new_key] = model.model.encoders[reference_key]
        logger.debug("Assigned encoder for dataset key %s using reference encoder %s", new_key, reference_key)


def prepare_data_for_clipn_from_df(df):
    """
    Prepares input data for CLIPn training from a MultiIndex DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with MultiIndex (Dataset, Sample). Should contain numeric features and metadata.

    Returns
    -------
    tuple
        data_dict : dict
            Dictionary mapping integer ID to feature matrix (np.ndarray).
        label_dict : dict
            Dictionary mapping integer ID to label vector (np.ndarray).
        label_mappings : dict
            Dictionary mapping dataset name to {label_id: label_name}.
        cpd_ids : dict
            Dictionary mapping dataset name to list of compound IDs.
        dataset_key_mapping : dict
            Dictionary mapping integer keys to original dataset names.
    """
    from collections import defaultdict

    data_dict = {}
    label_dict = {}
    label_mappings = {}
    cpd_ids = {}

    # Ensure it's a MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex DataFrame with levels ['Dataset', 'Sample']")

    dataset_keys = df.index.get_level_values('Dataset').unique()

    for dataset_name in dataset_keys:
        dataset_df = df.loc[dataset_name]

        # Get feature columns (exclude metadata)
        feature_cols = dataset_df.select_dtypes(include=[np.number]).columns.tolist()
        meta_cols = ['cpd_id', 'cpd_type', 'Library']
        feature_cols = [col for col in feature_cols if col not in meta_cols]

        X = dataset_df[feature_cols].to_numpy()
        y = dataset_df['cpd_type'].astype(str).to_numpy()
        ids = dataset_df['cpd_id'].astype(str).tolist()

        unique_labels = sorted(set(y))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_encoded = np.array([label_map[label] for label in y])

        data_dict[dataset_name] = X
        label_dict[dataset_name] = y_encoded
        label_mappings[dataset_name] = {v: k for k, v in label_map.items()}
        cpd_ids[dataset_name] = ids

    # Reindex with integers for CLIPn compatibility
    indexed_data_dict = {i: data_dict[k] for i, k in enumerate(data_dict)}
    indexed_label_dict = {i: label_dict[k] for i, k in enumerate(label_dict)}
    dataset_key_mapping = {i: k for i, k in enumerate(data_dict)}

    return indexed_data_dict, indexed_label_dict, label_mappings, cpd_ids, dataset_key_mapping



def run_clipn_integration(
    df: pd.DataFrame,
    logger: logging.Logger,
    clipn_param: str,
    output_path: str | Path,
    experiment: str,
    mode: str,
    latent_dim: int,
    lr: float,
    epochs: int,
    skip_standardise: bool = False,
    plot_loss: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], CLIPn, Dict[int, str]]:
    """
    Train a CLIPn model on the provided DataFrame and return latent representations.

    Parameters
    ----------
    df : pd.DataFrame
        Combined input DataFrame with MultiIndex (Dataset, Sample).
    logger : logging.Logger
        Logger instance.
    clipn_param : str
        Optional parameter for logging (no functional effect here).
    output_path : str | Path
        Directory to save latent arrays.
    experiment : str
        Experiment name.
    mode : str
        Operation mode (for filename context).
    latent_dim : int
        Dimensionality of the latent space.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    skip_standardise : bool
        Unused here (kept for API compatibility).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, list[str]], CLIPn, dict[int, str]]
        Combined latent DataFrame (MultiIndex),
        dictionary of cpd_ids per dataset,
        trained CLIPn model,
        dataset key mapping.
    """
    logger.info("Running CLIPn integration with param: %s", clipn_param)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("Numeric feature column count: %d", len(numeric_cols))
    logger.info("Combined DataFrame shape: %s", df.shape)
    logger.debug("Head of combined DataFrame:\n%s", df.head())

    if not numeric_cols:
        logger.error(
            "No numeric feature columns found after harmonisation. "
            "Possible causes: no overlap of features, all numeric columns are NaN, or wrong dtypes."
        )
        raise ValueError("No numeric columns available for CLIPn.")

    data_dict, label_dict, label_mappings, cpd_ids, dataset_key_mapping = prepare_data_for_clipn_from_df(df)
    latent_dict, model, loss = run_clipn_simple(
        data_dict,
        label_dict,
        latent_dim=latent_dim,
        lr=lr,
        epochs=epochs,
    )
    logger.info("CLIPn training completed.")
    # Save loss curve (TSV + PNG) if available
    if plot_loss and loss is not None:
        try:
            save_training_loss(
                loss_values=loss,
                out_dir=output_path,
                experiment=experiment,
                mode=mode,
                logger=logger,
                expected_epochs=epochs, 
                aggregate="last",        
                
            )
        except Exception as exc:
            logger.warning("Failed to save/plot training loss: %s", exc)


    if isinstance(loss, (list, np.ndarray)):
        logger.info("CLIPn final loss: %.6f", loss[-1])
    else:
        logger.info("CLIPn loss: %s", loss)

    latent_frames = []
    for i, latent in latent_dict.items():
        name = dataset_key_mapping[i]
        df_latent = pd.DataFrame(latent)
        df_latent.index = pd.MultiIndex.from_product(
            [[name], range(len(df_latent))], names=["Dataset", "Sample"]
        )
        latent_frames.append(df_latent)

    latent_combined = pd.concat(latent_frames)

    # Save latent as NPZ (with string keys)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    latent_file = output_path / f"{experiment}_{mode}_CLIPn_latent_representations.npz"
    latent_dict_str_keys = {str(k): v for k, v in latent_dict.items()}
    np.savez(file=latent_file, **latent_dict_str_keys)
    logger.info("Latent representations saved to: %s", latent_file)

    latent_file_id = output_path / f"{experiment}_{mode}_CLIPn_latent_representations_cpd_id.npz"
    cpd_ids_array = {f"cpd_ids_{k}": np.array(v) for k, v in cpd_ids.items()}
    np.savez(file=latent_file_id, **latent_dict_str_keys, **cpd_ids_array)

    post_clipn_dir = output_path / "post_clipn"
    post_clipn_dir.mkdir(parents=True, exist_ok=True)
    post_latent_file = post_clipn_dir / f"{experiment}_{mode}_CLIPn_latent_representations.npz"
    np.savez(file=post_latent_file, **latent_dict_str_keys)

    return latent_combined, cpd_ids, model, dataset_key_mapping

