#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shap_feature_removal.py

Iterative SHAP-based feature removal for Cell Painting median per-well data,
modelled on the ZaritskyLab Anomaly Detection workflow (2024).

Core loop per round:
  1) Train a PyTorch autoencoder on DMSO (control) wells only.
  2) Compute anomaly (reconstruction) scores for all wells.
  3) Fit a RandomForest regressor mapping features -> anomaly scores.
  4) Run SHAP feature attribution.
  5) Remove features with NEGATIVE mean SHAP impact (or below a user threshold).
  6) Repeat until nothing more to remove, min-features reached, or max rounds hit.

Design notes (minimal but robust):
  • Input is a single, per-well table (your upstream variance/correlation filtering already ran).
  • Metadata columns are preserved; only numeric, non-metadata columns are candidates for removal.
  • NaNs are filled column-wise using CONTROL medians to stabilise AE training; the same
    medians are used to fill all rows for AE/SHAP in each round (deterministic).
  • Optional per-plate scaling for the AE/SHAP stage (OFF by default; enable with --scale_for_ae).
  • Logs include per-round SHAP means, removed features, and memory snapshots.
  • All outputs are TSV (never comma-separated).

Requirements:
    pandas, numpy, torch, scikit-learn, shap, psutil

Author: Pete Thorpe, 2025
"""

from __future__ import annotations

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
from sklearn.preprocessing import RobustScaler, StandardScaler
import shap


# --------- DEFAULT METADATA COLUMNS ---------
DEFAULT_METADATA_COLS = [
    "cpd_id", "cpd_type", "Library", "Plate_Metadata", "Well_Metadata"
]
# --------------------------------------------


# ==========================
# CLI & LOGGING
# ==========================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Iterative SHAP-based feature removal for Cell Painting median per-well data."
    )
    parser.add_argument(
        "--input_file", required=True,
        help="Input median-per-well table (TSV/CSV; .gz ok) containing metadata + numeric features."
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Output TSV (or .tsv.gz) for filtered data (metadata + surviving features)."
    )
    parser.add_argument(
        "--control_label", required=True,
        help="Label in cpd_type identifying DMSO controls (e.g., DMSO or dmso)."
    )
    parser.add_argument(
        "--metadata_cols", default=",".join(DEFAULT_METADATA_COLS),
        help=f"Comma-separated metadata columns. Default: {','.join(DEFAULT_METADATA_COLS)}"
    )
    parser.add_argument(
        "--max_rounds", type=int, default=10,
        help="Maximum number of feature removal rounds (default: 10)."
    )
    parser.add_argument(
        "--min_features", type=int, default=10,
        help="Stop if fewer than this many features remain (default: 10)."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)."
    )
    parser.add_argument(
        "--shap_sample_size", type=int, default=800,
        help="Subsample size for SHAP to speed up on large data (default: 800)."
    )
    parser.add_argument(
        "--remove_threshold", type=float, default=0.0,
        help="Remove features whose MEAN signed SHAP impact is < threshold (default: 0.0)."
    )
    parser.add_argument(
        "--min_abs_shap", type=float, default=0.0,
        help="Optional absolute SHAP filter: require |mean SHAP| >= this to consider removal/keeping (default: 0.0=off)."
    )
    parser.add_argument(
        "--protect_features", type=lambda s: [x.strip() for x in s.split(",")] if s else [],
        default=[],
        help="Comma-separated feature names to NEVER remove."
    )
    parser.add_argument(
        "--n_estimators", type=int, default=300,
        help="Number of trees in RandomForest for SHAP explanation (default: 300)."
    )
    parser.add_argument(
        "--ae_epochs", type=int, default=30,
        help="Autoencoder training epochs per round (default: 30)."
    )
    parser.add_argument(
        "--ae_batch_size", type=int, default=128,
        help="Autoencoder batch size (default: 128)."
    )
    parser.add_argument(
        "--ae_latent_dim", type=int, default=32,
        help="Autoencoder latent dimension (default: 32)."
    )
    parser.add_argument(
        "--scale_for_ae", choices=["none", "robust", "standard"], default="none",
        help="Optional per-plate scaling for AE/SHAP stage (default: none)."
    )
    parser.add_argument(
        "--plate_col", default="Plate_Metadata",
        help="Plate column name (used if --scale_for_ae != none). Default: Plate_Metadata."
    )
    parser.add_argument(
        "--selection_rule",
        choices=["negative_mean", "top_abs_quantile"],
        default="negative_mean",
        help=("Feature removal rule. "
            "'negative_mean' (default): remove features with mean signed SHAP < remove_threshold. "
            "'top_abs_quantile': keep the top fraction by absolute mean SHAP and remove the rest.")
        )

    parser.add_argument(
        "--keep_top_abs_quantile",
        type=float,
        default=0.5,
        help=("When --selection_rule=top_abs_quantile, keep this top fraction (0<q≤1) by |mean SHAP| "
            "(default: 0.5).")
    )

    parser.add_argument(
        "--write_round_stats", action="store_true",
        help="If set, write per-round SHAP means and removed features as TSVs alongside the output."
    )
    return parser.parse_args()


def setup_logging(log_level: str = "INFO") -> logging.Logger:
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
    return logging.getLogger("shap_iter")


def log_memory_usage(logger: logging.Logger, prefix: str = "", extra_msg: str | None = None) -> None:
    """
    Log current and peak memory usage.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    prefix : str
        Prefix for the log message.
    extra_msg : str or None
        Additional message to append.
    """
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    peak_gb = None
    try:
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Linux":
            peak_gb = peak_rss / (1024 ** 2)
        else:
            peak_gb = peak_rss / (1024 ** 3)
    except Exception:
        pass
    msg = f"{prefix} Memory: {mem_gb:.2f} GB"
    if peak_gb is not None:
        msg += f" (peak {peak_gb:.2f} GB)"
    if extra_msg:
        msg += f" | {extra_msg}"
    logger.info(msg)


# ==========================
# MODELLING UTILITIES
# ==========================

def set_deterministic(seed: int = 42) -> None:
    """
    Seed NumPy and PyTorch for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Autoencoder(nn.Module):
    """
    Simple feed-forward autoencoder for anomaly detection.
    """
    def __init__(self, n_features: int, latent_dim: int = 32) -> None:
        super().__init__()
        hidden = max(64, min(512, latent_dim * 4))
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch.

        Returns
        -------
        torch.Tensor
            Reconstructed batch.
        """
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(
    X_ctrl: np.ndarray,
    *,
    n_epochs: int = 30,
    batch_size: int = 128,
    latent_dim: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    logger: logging.Logger | None = None
) -> Autoencoder:
    """
    Train an autoencoder on control wells only.

    Parameters
    ----------
    X_ctrl : numpy.ndarray
        Control data (rows = controls, cols = features).
    n_epochs : int
        Number of epochs.
    batch_size : int
        Batch size.
    latent_dim : int
        Bottleneck size.
    lr : float
        Learning rate.
    device : str
        'cpu' or 'cuda'.
    logger : logging.Logger or None
        Logger instance.

    Returns
    -------
    Autoencoder
        Trained model.
    """
    model = Autoencoder(X_ctrl.shape[1], latent_dim=latent_dim).to(device)
    ds = TensorDataset(torch.from_numpy(X_ctrl.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for epoch in range(n_epochs):
        losses: list[float] = []
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = crit(recon, xb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if logger and (epoch % 10 == 0 or epoch == n_epochs - 1):
            logger.info("AE epoch %d/%d, loss=%.4e", epoch + 1, n_epochs, np.mean(losses))
    return model


def compute_recon_error(model: Autoencoder, X: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    """
    Compute per-sample reconstruction error (MSE).

    Parameters
    ----------
    model : Autoencoder
        Trained autoencoder.
    X : numpy.ndarray
        Data to score.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    numpy.ndarray
        Reconstruction errors per row.
    """
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X.astype(np.float32)).to(device)
        recon = model(xt).cpu().numpy()
    return np.mean((recon - X) ** 2, axis=1)


def platewise_scale_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    plate_col: str,
    method: str = "robust"
) -> np.ndarray:
    """
    Optionally scale features per plate for the AE/SHAP stage.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    feature_cols : list of str
        Feature names to scale.
    plate_col : str
        Plate metadata column.
    method : {'robust', 'standard'}
        Scaling method.

    Returns
    -------
    numpy.ndarray
        Scaled matrix aligned to df.index.
    """
    X_scaled = np.zeros((df.shape[0], len(feature_cols)), dtype=np.float32)
    for _, idx in df.groupby(plate_col, sort=False).indices.items():
        block = df.loc[idx, feature_cols].to_numpy(dtype=np.float32, copy=False)
        if method == "standard":
            scaler = StandardScaler(with_mean=True, with_std=True)
        else:
            scaler = RobustScaler(with_centering=True, with_scaling=True)
        X_scaled[idx, :] = scaler.fit_transform(block).astype(np.float32)
    return X_scaled


def build_shap_table(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_estimators: int = 300,
    sample_size: int = 800,
    seed: int = 42,
    logger: logging.Logger | None = None
) -> pd.DataFrame:
    """
    Compute SHAP attributions of anomaly scores to features via RandomForest.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Anomaly scores.
    feature_names : list of str
        Feature names aligned to X columns.
    n_estimators : int
        Number of trees for the RandomForest.
    sample_size : int
        Subsample size for SHAP (speed vs variance trade-off).
    seed : int
        Random seed.
    logger : logging.Logger or None
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        SHAP summary: columns = ['feature', 'shap_abs_mean', 'shap_signed_mean'].
    """
    rng = np.random.default_rng(seed)
    if sample_size < X.shape[0]:
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(Xs, ys)

    # Prefer TreeExplainer for speed/stability on tree models
    try:
        expl = shap.TreeExplainer(rf)
        vals = expl.shap_values(Xs)  # array [n_samples, n_features]
    except Exception:
        expl = shap.Explainer(rf, Xs)
        vals = expl(Xs).values

    abs_mean = np.abs(vals).mean(axis=0)
    signed_mean = vals.mean(axis=0)
    summary = pd.DataFrame({
        "feature": feature_names,
        "shap_abs_mean": abs_mean,
        "shap_signed_mean": signed_mean
    }).sort_values("shap_abs_mean", ascending=False)

    if logger:
        top = ", ".join(f"{r.feature}:{r.shap_signed_mean:.2e}" for _, r in summary.head(5).iterrows())
        bot = ", ".join(f"{r.feature}:{r.shap_signed_mean:.2e}" for _, r in summary.tail(5).iterrows())
        logger.info("SHAP computed: top-5 by |mean| → %s", top)
        logger.debug("SHAP tail-5: %s", bot)
    return summary


# ==========================
# MAIN ORCHESTRATION
# ==========================

def main() -> None:
    """
    Run the iterative SHAP-based feature removal workflow.
    """
    args = parse_args()
    logger = setup_logging(args.log_level)
    set_deterministic(args.random_seed)
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # --- Load input (TSV preferred; CSV accepted) ---
    logger.info("Loading input file: %s", args.input_file)
    sep_guess = "\t" if args.input_file.lower().endswith((".tsv", ".tsv.gz", ".txt")) else None
    df = pd.read_csv(args.input_file, sep=sep_guess)
    logger.info("Loaded shape: %s", df.shape)
    log_memory_usage(logger, prefix="Loaded")

    # --- Split metadata / features ---
    meta_cols = [c.strip() for c in args.metadata_cols.split(",") if c.strip() in df.columns]
    miss_meta = [c.strip() for c in args.metadata_cols.split(",") if c.strip() not in df.columns]
    if miss_meta:
        logger.warning("Metadata columns not found: %s", miss_meta)
    if "cpd_type" not in df.columns:
        logger.error("Required metadata column 'cpd_type' not found.")
        sys.exit(1)

    # Candidate numeric features
    feat_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    if len(feat_cols) < args.min_features:
        logger.error("Not enough numeric features to proceed (found %d).", len(feat_cols))
        sys.exit(1)
    logger.info("Metadata cols: %d | Feature cols: %d", len(meta_cols), len(feat_cols))

    # Track state across rounds
    df_current = df.copy()
    features_removed_total: list[str] = []
    removed_per_round: list[dict] = []
    stop_reason = ""

    # Round loop
    for round_idx in range(1, args.max_rounds + 1):
        logger.info("=== Round %d ===", round_idx)

        # Refresh current feature list
        feat_cols = [c for c in df_current.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df_current[c])]
        if len(feat_cols) < args.min_features:
            stop_reason = f"Fewer than {args.min_features} features remain."
            break

        # Build matrices & control mask
        X_df = df_current[feat_cols]
        ctrl_mask = df_current["cpd_type"].astype(str).str.lower().eq(args.control_label.lower())
        n_ctrl = int(ctrl_mask.sum())
        if n_ctrl < 10:
            logger.error("Fewer than 10 control wells found (found %d). Check --control_label and data.", n_ctrl)
            sys.exit(1)

        # Fill NaNs using CONTROL medians (deterministic each round)
        ctrl_medians = X_df.loc[ctrl_mask, :].median(numeric_only=True)
        X_filled = X_df.fillna(ctrl_medians).astype(np.float32)

        # Optional per-plate scaling for AE/SHAP (OFF by default)
        if args.scale_for_ae != "none":
            plate_col = args.plate_col
            if plate_col not in df_current.columns:
                logger.error("--scale_for_ae requested but plate column '%s' not present.", plate_col)
                sys.exit(1)
            method = "standard" if args.scale_for_ae == "standard" else "robust"
            X_mat = platewise_scale_matrix(
                df_current.assign(**{c: X_filled[c] for c in feat_cols}),
                feat_cols,
                plate_col=plate_col,
                method=method
            )
        else:
            X_mat = X_filled.to_numpy(dtype=np.float32, copy=False)

        # Controls matrix for AE training
        X_ctrl = X_mat[ctrl_mask.to_numpy()]
        logger.info("Training AE on %d controls with %d features.", X_ctrl.shape[0], X_ctrl.shape[1])

        # Train AE and compute reconstruction errors
        ae = train_autoencoder(
            X_ctrl,
            n_epochs=args.ae_epochs,
            batch_size=args.ae_batch_size,
            latent_dim=args.ae_latent_dim,
            device=device,
            logger=logger
        )
        recon_err = compute_recon_error(ae, X_mat, device=device)
        df_current["anomaly_score"] = recon_err
        logger.info("Anomaly: min=%.3e | mean=%.3e | max=%.3e", recon_err.min(), recon_err.mean(), recon_err.max())

        # SHAP attributions for anomaly scores
        shap_tbl = build_shap_table(
            X_mat, recon_err, feat_cols,
            n_estimators=args.n_estimators,
            sample_size=args.shap_sample_size,
            seed=args.random_seed,
            logger=logger
        )

        # Decide removal set according to selection_rule.
        #  - Always honour protected features (never remove them).
        prot = set(args.protect_features or [])

        if args.selection_rule == "negative_mean":
            # Optional absolute filter before applying sign rule
            if args.min_abs_shap > 0.0:
                consider = shap_tbl.loc[
                    shap_tbl["shap_signed_mean"].abs() >= float(args.min_abs_shap)
                ]
            else:
                consider = shap_tbl

            to_remove = consider.loc[
                consider["shap_signed_mean"] < float(args.remove_threshold), "feature"
            ].tolist()
            to_remove = [f for f in to_remove if f not in prot]

        elif args.selection_rule == "top_abs_quantile":
            q = float(args.keep_top_abs_quantile)
            if not (0.0 < q <= 1.0):
                raise ValueError("--keep_top_abs_quantile must be in (0, 1].")

            # Rank by |mean SHAP| (desc), keep top q, remove the rest (except protected).
            ranked = shap_tbl.sort_values("shap_abs_mean", ascending=False)
            k = max(1, int(np.ceil(q * ranked.shape[0])))
            keep_set = set(ranked.head(k)["feature"].tolist()) | prot

            current_feats = [
                c for c in df_current.columns
                if c not in meta_cols and pd.api.types.is_numeric_dtype(df_current[c])
            ]
            to_remove = [f for f in current_feats if f not in keep_set]

        else:
            raise ValueError(f"Unknown selection_rule: {args.selection_rule}")

        logger.info(
            "Flagged for removal this round: %d features "
            "(selection_rule=%s; remove_threshold=%.3g; min_abs_shap=%.3g; keep_q=%s).",
            len(to_remove),
            args.selection_rule,
            args.remove_threshold,
            args.min_abs_shap,
            f"{args.keep_top_abs_quantile:.2f}" if args.selection_rule == "top_abs_quantile" else "1.00",
        )

        # Honour protected features
        prot = set(args.protect_features or [])
        to_remove = [f for f in to_remove if f not in prot]

        logger.info("Flagged for removal this round: %d features (threshold=%.3g; min_abs=%.3g).",
                    len(to_remove), args.remove_threshold, args.min_abs_shap)

        # Write per-round stats if requested
        if args.write_round_stats:
            base = os.path.splitext(args.output_file)[0]
            shap_out = f"{base}.round{round_idx}_shap.tsv"
            rem_out = f"{base}.round{round_idx}_removed.tsv"
            shap_tbl.to_csv(shap_out, sep="\t", index=False)
            pd.DataFrame({"feature": to_remove}).to_csv(rem_out, sep="\t", index=False)
            logger.info("Wrote round %d SHAP stats → %s ; removed → %s", round_idx, shap_out, rem_out)

        # Stop if nothing to remove
        if len(to_remove) == 0:
            stop_reason = "No features with negative mean SHAP impact (or below threshold)."
            # Clean up helper column
            df_current = df_current.drop(columns=["anomaly_score"])
            break

        # Apply removal
        features_removed_total.extend(to_remove)
        removed_per_round.append({
            "round": round_idx,
            "n_removed": len(to_remove),
            "features_removed": ";".join(to_remove)
        })

        # Update dataframe
        df_current = df_current.drop(columns=to_remove)
        # Drop helper
        df_current = df_current.drop(columns=["anomaly_score"])

        log_memory_usage(logger, prefix=f"After round {round_idx}")

        # Early stop if few features left
        n_feat_left = sum(c not in meta_cols and pd.api.types.is_numeric_dtype(df_current[c]) for c in df_current.columns)
        if n_feat_left < args.min_features:
            stop_reason = f"Fewer than {args.min_features} features remain."
            break

    # --- Finalise outputs ---
    elapsed = time.time() - start_time
    logger.info("Completed. Reason for stopping: %s", stop_reason or "max rounds reached")
    logger.info("Total features removed: %d", len(features_removed_total))

    # Build filtered output (metadata + surviving features)
    final_meta = [c for c in meta_cols if c in df_current.columns]
    final_feats = [c for c in df_current.columns if c not in final_meta and pd.api.types.is_numeric_dtype(df_current[c])]
    output_df = df_current[final_meta + final_feats].copy()

    # Ensure TSV output (never commas)
    out_path = args.output_file
    out_lower = out_path.lower()
    if out_lower.endswith(".tsv.gz"):
        output_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    else:
        if not out_lower.endswith(".tsv"):
            base, _ = os.path.splitext(out_path)
            out_path = f"{base}.tsv"
        output_df.to_csv(out_path, sep="\t", index=False)
    logger.info("Filtered data written to: %s (shape: %s)", out_path, output_df.shape)

    # Removal log
    log_df = pd.DataFrame(removed_per_round, columns=["round", "n_removed", "features_removed"])
    log_file = os.path.splitext(out_path)[0] + "_feature_removal_log.tsv"
    log_df.to_csv(log_file, sep="\t", index=False)
    logger.info("Feature removal log written to: %s", log_file)

    log_memory_usage(logger, prefix="END", extra_msg=f"Elapsed {elapsed/60:.2f} min")


if __name__ == "__main__":
    main()
