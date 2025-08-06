#!/usr/bin/env python3
# coding: utf-8

"""
shap_feature_removal.py

Iterative SHAP-based feature removal for Cell Painting median per-well data,
modelled on ZaritskyLab Anomaly Detection workflow (2024).

- Trains a PyTorch autoencoder on DMSO (control) wells only.
- Computes anomaly (reconstruction) scores for all wells.
- Fits a RandomForest regressor to map features to anomaly scores.
- Runs SHAP feature attribution.
- Removes features with negative mean SHAP impact, logs changes.
- Repeats until no more features to remove or max rounds reached.
- Outputs filtered feature table with original metadata columns reattached, plus a log of feature removal rounds.

Requires:
    pandas, numpy, torch, scikit-learn, shap, psutil

Author: Pete Thorpe, 2025
"""

import argparse
import logging
import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
import shap

# --------- CONFIG ---------
DEFAULT_METADATA_COLS = [
    "cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"
]
# --------------------------


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Iterative SHAP-based feature removal for Cell Painting median per-well data.")
    parser.add_argument('--input_file', required=True, help="Input median-per-well table (TSV/CSV, with metadata columns).")
    parser.add_argument('--output_file', required=True, help="Output file for filtered data (TSV).")
    parser.add_argument('--control_label', required=True, help="Label in cpd_type indicating DMSO control (e.g., DMSO).")
    parser.add_argument('--metadata_cols', default="cpd_id,cpd_type,Library,Plate_Metadata,Well_Metadata", help="Comma-separated metadata columns.")
    parser.add_argument('--max_rounds', type=int, default=10, help="Maximum number of feature removal rounds.")
    parser.add_argument('--min_features', type=int, default=10, help="Stop if fewer than this many features remain.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--log_level', default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="Logging level.")
    parser.add_argument('--shap_sample_size', type=int, default=500, help="Number of samples to use for SHAP (speedup for large data).")
    parser.add_argument('--remove_threshold', type=float, default=0.0, help="Remove features with mean SHAP impact below this value (default: 0.0 = remove only negative impact).")
    return parser.parse_args()


def setup_logging(log_level="INFO"):
    """
    Set up logging.

    Parameters
    ----------
    log_level : str
        Logging level.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("shap_logger")


def log_memory_usage(logger, prefix="", extra_msg=None):
    """
    Log current and peak memory usage (RAM).

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    prefix : str
        Prefix for log message.
    extra_msg : str or None
        Additional message.
    """
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_gb = mem_bytes / (1024 ** 3)
    try:
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Linux":
            peak_gb = peak_rss / (1024 ** 2)
        else:
            peak_gb = peak_rss / (1024 ** 3)
    except Exception:
        peak_gb = None
    msg = f"{prefix} Memory: {mem_gb:.2f} GB"
    if peak_gb:
        msg += f" (peak {peak_gb:.2f} GB)"
    if extra_msg:
        msg += f" | {extra_msg}"
    logger.info(msg)


class Autoencoder(nn.Module):
    """
    Simple feed-forward autoencoder for anomaly detection.
    """
    def __init__(self, n_features, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, max(32, latent_dim * 2)),
            nn.ReLU(),
            nn.Linear(max(32, latent_dim * 2), latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(32, latent_dim * 2)),
            nn.ReLU(),
            nn.Linear(max(32, latent_dim * 2), n_features)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


def train_autoencoder(X, n_epochs=50, batch_size=64, latent_dim=16, lr=1e-3, seed=42, logger=None, device="cpu"):
    """
    Train a simple PyTorch autoencoder on control data.

    Parameters
    ----------
    X : np.ndarray
        Training data (controls only).
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    latent_dim : int
        Size of latent layer.
    lr : float
        Learning rate.
    seed : int
        Random seed.
    logger : logging.Logger or None
        Logger.
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    Autoencoder
        Trained autoencoder model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Autoencoder(X.shape[1], latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(n_epochs):
        losses = []
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            optimiser.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_X)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        if logger and (epoch % 10 == 0 or epoch == n_epochs - 1):
            logger.info(f"Epoch {epoch+1}/{n_epochs}, loss={np.mean(losses):.4e}")
    return model


def compute_recon_error(model, X, device="cpu"):
    """
    Compute reconstruction error for all samples.

    Parameters
    ----------
    model : Autoencoder
        Trained autoencoder.
    X : np.ndarray
        Data.
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    np.ndarray
        Reconstruction error (MSE) per sample.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
        X_recon = model(X_tensor).cpu().numpy()
        err = np.mean((X_recon - X) ** 2, axis=1)
    return err


def select_features_shap(X, y, feature_names, sample_size=500, remove_threshold=0.0, random_state=42, logger=None):
    """
    Use SHAP to attribute anomaly scores to features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Anomaly scores.
    feature_names : list of str
        Feature names.
    sample_size : int
        Subsample for SHAP (for speed).
    remove_threshold : float
        Remove features with mean SHAP value below this threshold (default: <0.0).
    random_state : int
        Random seed.
    logger : logging.Logger
        Logger.

    Returns
    -------
    list
        List of features to remove.
    dict
        SHAP value means (feature: value).
    """
    np.random.seed(random_state)
    if sample_size < len(X):
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_shap = X[idx]
        y_shap = y[idx]
    else:
        X_shap = X
        y_shap = y

    logger.info("Fitting RandomForestRegressor for SHAP explanation.")
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=random_state)
    rf.fit(X_shap, y_shap)
    explainer = shap.Explainer(rf, X_shap)
    shap_values = explainer(X_shap)
    # For regression, shap_values are [n_samples, n_features]
    shap_means = np.abs(shap_values.values).mean(axis=0)
    shap_signed = shap_values.values.mean(axis=0)
    shap_feature_means = dict(zip(feature_names, shap_signed))
    # Remove features with mean SHAP < remove_threshold
    features_to_remove = [f for f, v in shap_feature_means.items() if v < remove_threshold]
    logger.info(f"SHAP analysis: {len(features_to_remove)} features flagged for removal (threshold={remove_threshold}).")
    # Log top and bottom features
    top5 = sorted(shap_feature_means.items(), key=lambda x: -abs(x[1]))[:5]
    bot5 = sorted(shap_feature_means.items(), key=lambda x: x[1])[:5]
    logger.info("Top 5 features (|mean SHAP|): " + ', '.join([f"{k}:{v:.2e}" for k, v in top5]))
    logger.info("Lowest 5 features (mean SHAP): " + ', '.join([f"{k}:{v:.2e}" for k, v in bot5]))
    return features_to_remove, shap_feature_means


def main():
    """
    Main workflow for iterative SHAP-based feature removal.
    """
    args = parse_args()
    logger = setup_logging(args.log_level)
    logger.info(f"Arguments: {' '.join(sys.argv)}")
    start_time = time.time()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # 1. Load data
    logger.info(f"Loading input file: {args.input_file}")
    sep = "\t" if args.input_file.endswith(".tsv") else ","
    df = pd.read_csv(args.input_file, sep=sep)
    logger.info(f"Loaded data shape: {df.shape}")
    log_memory_usage(logger, prefix="Loaded data")

    # 2. Separate metadata/features
    metadata_cols = [col.strip() for col in args.metadata_cols.split(",") if col.strip() in df.columns]
    missing_metadata = [col for col in args.metadata_cols.split(",") if col.strip() not in df.columns]
    if missing_metadata:
        logger.warning(f"Metadata columns not found: {missing_metadata}")
    feature_cols = [col for col in df.columns if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])]
    logger.info(f"{len(metadata_cols)} metadata columns, {len(feature_cols)} feature columns.")

    # 3. Main loop: iterative SHAP feature removal
    round_idx = 1
    features_removed_total = []
    removed_per_round = []
    stop_reason = ""
    prev_n_features = len(feature_cols)
    df_current = df.copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    while round_idx <= args.max_rounds:
        logger.info(f"=== Round {round_idx} ===")
        # Reload features for this round
        X = df_current[feature_cols].values.astype(np.float32)
        # Identify control wells
        mask_ctrl = df_current["cpd_type"].str.lower() == args.control_label.lower()
        X_ctrl = X[mask_ctrl.values]
        if X_ctrl.shape[0] < 10:
            logger.error("Fewer than 10 control wells found! Check control_label and data.")
            sys.exit(1)
        # --- 1. Train autoencoder on control wells
        logger.info(f"Training autoencoder on {X_ctrl.shape[0]} control wells with {len(feature_cols)} features.")
        ae = train_autoencoder(
            X_ctrl,
            n_epochs=30,
            batch_size=64,
            latent_dim=min(16, max(4, len(feature_cols)//8)),
            lr=1e-3,
            seed=args.random_seed,
            logger=logger,
            device=device
        )
        # --- 2. Compute anomaly (reconstruction) scores for all
        recon_err = compute_recon_error(ae, X, device=device)
        df_current["anomaly_score"] = recon_err
        logger.info(f"Anomaly scores: min={recon_err.min():.3e}, max={recon_err.max():.3e}, mean={recon_err.mean():.3e}")

        # --- 3. SHAP feature attribution via RandomForest
        features_to_remove, shap_feature_means = select_features_shap(
            X,
            recon_err,
            feature_cols,
            sample_size=args.shap_sample_size,
            remove_threshold=args.remove_threshold,
            random_state=args.random_seed,
            logger=logger
        )
        # --- 4. Remove flagged features
        if not features_to_remove:
            stop_reason = "No more features with negative SHAP impact."
            break
        features_removed_total += features_to_remove
        removed_per_round.append((round_idx, features_to_remove))
        feature_cols = [f for f in feature_cols if f not in features_to_remove]
        logger.info(f"Removed {len(features_to_remove)} features this round; {len(feature_cols)} features remain.")
        log_memory_usage(logger, prefix=f"Round {round_idx}")
        # Drop from dataframe
        df_current = df_current.drop(columns=features_to_remove)
        if len(feature_cols) < args.min_features:
            stop_reason = f"Fewer than {args.min_features} features remain."
            break
        if len(feature_cols) == prev_n_features:
            stop_reason = "No features removed in this round."
            break
        prev_n_features = len(feature_cols)
        round_idx += 1

    elapsed = time.time() - start_time
    logger.info(f"Completed after {round_idx} rounds. Reason for stopping: {stop_reason}")
    logger.info(f"Total features removed: {len(features_removed_total)}")
    logger.info(f"Final number of features: {len(feature_cols)}")
    logger.info(f"Elapsed time: {elapsed/60:.2f} min")

    # Output logs (summary)
    log_df = pd.DataFrame(
        [
            {"round": r, "features_removed": f, "n_removed": len(f)}
            for r, f in removed_per_round
        ]
    )
    log_file = os.path.splitext(args.output_file)[0] + "_feature_removal_log.tsv"
    log_df.to_csv(log_file, sep="\t", index=False)
    logger.info(f"Feature removal log written to: {log_file}")

    # Output filtered data
    keep_cols = metadata_cols + feature_cols
    output_df = df_current[keep_cols]
    output_df.to_csv(args.output_file, sep="\t", index=False)
    logger.info(f"Filtered data written to: {args.output_file} (shape: {output_df.shape})")
    log_memory_usage(logger, prefix="END")


if __name__ == "__main__":
    main()
