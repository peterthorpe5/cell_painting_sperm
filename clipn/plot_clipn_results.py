#!/usr/bin/env python3
# coding: utf-8
"""
Visualise CLIPn Latents with Topological Graphs, UMAP, and PHATE.

Default behaviour:
- Build a **topological graph** (Mapper) from the latent features.
- Save a **PDF** of the graph and **TSV** nodes/edges.
- Optionally also compute UMAP/PHATE embeddings and export PDFs + TSV coords.

Inputs
------
--latent_csv       : TSV with latent features (digit-named columns by default) + metadata
--plots            : Output directory
--latent_prefix    : Optional prefix for latent columns (else uses digit-named cols)
--colour_by        : Metadata column for colouring nodes/points (default: Dataset)
--embedding        : Which embeddings to run: topo | umap | phate | all (default: topo)

Mapper / Topological graph options
----------------------------------
--mapper_lens        : Lens for Mapper: pca | umap | identity (default: pca)
--mapper_n_cubes     : Number of hypercubes for Mapper cover (default: 15)
--mapper_overlap     : Fractional overlap for cover in [0,1) (default: 0.4)
--mapper_cluster     : Clustering algorithm for bins: dbscan | hdbscan (default: dbscan)
--mapper_eps         : DBSCAN eps (default: 0.5)
--mapper_min_samples : DBSCAN min_samples (default: 5)
--knn_k              : If Mapper unavailable, k for k-NN graph fallback (default: 10)

UMAP / PHATE
------------
--umap_metric        : cosine | euclidean (default: cosine)
--umap_n_neighbors   : UMAP n_neighbours (default: 40)
--umap_min_dist      : UMAP min_dist (default: 0.25)
--phate_knn          : PHATE k-NN (default: 15)

Outputs (all under --plots)
---------------------------
- topo_graph.pdf
- topo_nodes.tsv / topo_edges.tsv
- umap.pdf (+ umap_coords.tsv) if requested/available
- phate.pdf (+ phate_coords.tsv) if requested/available
- plot.log
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

set_config(transform_output="pandas")

# Optional libs, handled gracefully
try:
    import kmapper as km  # KeplerMapper
    KMAP_AVAILABLE = True
except Exception:
    KMAP_AVAILABLE = False

try:
    import hdbscan  # type: ignore
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    import phate  # type: ignore
    PHATE_AVAILABLE = True
except Exception:
    PHATE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except Exception as exc:
    raise RuntimeError("matplotlib is required for PDF outputs.") from exc


# =========
# Logging
# =========

def setup_logging(*, output_dir: str | Path) -> logging.Logger:
    """
    Configure console and file logging.

    Parameters
    ----------
    output_dir : str | Path
        Directory where plot.log will be written.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "plot.log"

    logger = logging.getLogger("clipn_plot")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setLevel(logging.INFO)
    stream.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    fileh = logging.FileHandler(filename=log_path, mode="w", encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(stream)
    logger.addHandler(fileh)
    logger.info("Logging to %s", log_path)
    return logger


# =========
# Helpers
# =========

def validate_columns(*, df: pd.DataFrame, required: Iterable[str], logger: logging.Logger) -> None:
    """
    Validate required columns exist in df.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    required : Iterable[str]
        Required column names.
    logger : logging.Logger
        Logger for error reporting.

    Raises
    ------
    ValueError
        If any columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")


def clean_merge_artifacts(*, df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Drop merge artefacts (e.g. *_x, *_y, 'index') and reorder metadata first.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing merge artefacts.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with metadata columns front-loaded.
    """
    drop_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y") or c == "index"]
    if drop_cols:
        logger.info("Dropping merge artefacts: %s", drop_cols)
        df = df.drop(columns=drop_cols, errors="ignore")

    meta = [c for c in ["cpd_id", "cpd_type", "Library", "Dataset", "Sample"] if c in df.columns]
    ordered = df[meta + [c for c in df.columns if c not in meta]]
    return ordered


def select_latent(*, df: pd.DataFrame, prefix: Optional[str], logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select latent numeric columns (digit-named by default).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with latent features + metadata.
    prefix : str | None
        If provided, use columns starting with this prefix. Otherwise digit-named.

    Returns
    -------
    (X, cols) : (pd.DataFrame, list[str])
        Feature matrix and column names (copy, numeric).
    """
    if prefix:
        cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix) and pd.api.types.is_numeric_dtype(df[c])]
    else:
        cols = [c for c in df.columns if isinstance(c, str) and c.isdigit() and pd.api.types.is_numeric_dtype(df[c])]

    if not cols:
        logger.error("No latent feature columns found (prefix=%s).", prefix)
        raise ValueError("No latent feature columns found. Check column names and --latent_prefix.")

    X = df[cols].copy()
    n_nans = int(X.isna().sum().sum())
    if n_nans:
        logger.warning("Latent matrix contains %d NaNs; filling with 0.", n_nans)
        X = X.fillna(value=0)
    logger.info("Selected %d latent columns; feature matrix shape=%s", len(cols), tuple(X.shape))
    return X, cols


def write_tsv(*, df: pd.DataFrame, path: str | Path, logger: logging.Logger, index: bool = False) -> None:
    """
    Write DataFrame to TSV (never commas).

    Parameters
    ----------
    df : pd.DataFrame
        Frame to write.
    path : str | Path
        Output path.
    logger : logging.Logger
        Logger instance.
    index : bool
        Whether to write index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_or_buf=path, sep="\t", index=index)
    logger.info("Wrote %s rows to %s", len(df), path)


# =========================
# Topological graph (Mapper)
# =========================

def _mapper_lens_array(*, X: np.ndarray, method: str, logger: logging.Logger) -> np.ndarray:
    """
    Compute lens for Mapper.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    method : str
        'pca', 'umap', or 'identity'.
    logger : logging.Logger
        Logger.

    Returns
    -------
    np.ndarray
        Lens array for KeplerMapper.
    """
    method = method.lower()
    if method == "pca":
        logger.info("Mapper lens: PCA(2)")
        lens = PCA(n_components=2, random_state=42).fit_transform(X)
        return lens
    if method == "umap" and UMAP_AVAILABLE:
        logger.info("Mapper lens: UMAP(2)")
        reducer = umap.UMAP(n_neighbors=40, min_dist=0.25, metric="cosine", random_state=42)
        lens = reducer.fit_transform(X)
        return lens
    if method == "identity":
        logger.info("Mapper lens: identity (raw features)")
        return X
    logger.warning("Mapper lens '%s' unavailable; falling back to PCA(2).", method)
    return PCA(n_components=2, random_state=42).fit_transform(X)


def build_topological_graph(
    *,
    X: pd.DataFrame,
    df_meta: pd.DataFrame,
    out_dir: Path,
    colour_by: str,
    mapper_lens: str,
    n_cubes: int,
    overlap: float,
    cluster_alg: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    knn_k: int,
    logger: logging.Logger,
) -> None:
    """
    Build a topological graph (Mapper if available; else k-NN fallback) and save:
    - topo_nodes.tsv
    - topo_edges.tsv
    - topo_graph.pdf

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric).
    df_meta : pd.DataFrame
        DataFrame including 'cpd_id' and metadata for colouring.
    out_dir : Path
        Output directory.
    colour_by : str
        Column for colour mapping (if present).
    mapper_lens : str
        'pca' | 'umap' | 'identity'.
    n_cubes : int
        Cover hypercubes.
    overlap : float
        Cover overlap fraction in [0, 1).
    cluster_alg : str
        'dbscan' | 'hdbscan'.
    dbscan_eps : float
        DBSCAN eps (if used).
    dbscan_min_samples : int
        DBSCAN min_samples (if used).
    knn_k : int
        Fallback: k for k-NN graph.
    logger : logging.Logger
        Logger instance.
    """
    nodes_path = out_dir / "topo_nodes.tsv"
    edges_path = out_dir / "topo_edges.tsv"
    pdf_path = out_dir / "topo_graph.pdf"

    # Attempt KeplerMapper
    if KMAP_AVAILABLE:
        logger.info("Building Mapper graph using KeplerMapper.")
        mapper = km.KeplerMapper(verbose=0)

        lens = _mapper_lens_array(X=X.values, method=mapper_lens, logger=logger)

        # choose clusterer
        if cluster_alg.lower() == "hdbscan" and HDBSCAN_AVAILABLE:
            logger.info("Mapper clustering: HDBSCAN")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, dbscan_min_samples))
        else:
            from sklearn.cluster import DBSCAN
            logger.info("Mapper clustering: DBSCAN(eps=%.3f, min_samples=%d)", dbscan_eps, dbscan_min_samples)
            clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)

        graph = mapper.map(
            lens=lens,
            X=X.values,
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap),
            clusterer=clusterer,
        )

        # Extract nodes/edges
        # graph['nodes'] is dict: node_id -> list of sample indices
        node_rows = []
        for nid, members in graph["nodes"].items():
            members = list(members)
            # Compute a colour label summary (majority) if possible
            colour_val = None
            if colour_by in df_meta.columns:
                vals = df_meta.iloc[members][colour_by].astype(str)
                if len(vals):
                    colour_val = vals.value_counts().idxmax()
            node_rows.append(
                {
                    "node_id": str(nid),
                    "size": int(len(members)),
                    "colour_value": colour_val if colour_val is not None else "",
                    "members": ";".join(df_meta.iloc[members]["cpd_id"].astype(str).tolist()),
                }
            )

        edge_rows = []
        for a, nbrs in graph["links"].items():
            for b in nbrs:
                if int(a) < int(b):  # undirected unique
                    edge_rows.append({"source": str(a), "target": str(b)})

        nodes_df = pd.DataFrame(node_rows)
        edges_df = pd.DataFrame(edge_rows)
        write_tsv(df=nodes_df, path=nodes_path, logger=logger, index=False)
        write_tsv(df=edges_df, path=edges_path, logger=logger, index=False)

        # Draw PDF with matplotlib
        _draw_graph_pdf(
            nodes=nodes_df,
            edges=edges_df,
            output_pdf=pdf_path,
            logger=logger,
        )
        return

    # Fallback: build a k-NN graph that behaves like a topology sketch
    logger.warning("KeplerMapper not available; building k-NN graph fallback (k=%d).", knn_k)
    n = len(X)
    k = min(max(2, knn_k), max(2, n - 1))  # at least 2, exclude self later

    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X.values)
    dist, idx = nn.kneighbors(X.values)

    # Build edges (undirected, dedup)
    edges = set()
    for i in range(n):
        for j in idx[i][1:]:  # skip self
            a, b = sorted((int(i), int(j)))
            edges.add((a, b))

    # Collapse to node level as Mapper-like: each original sample is a node
    nodes_df = pd.DataFrame(
        {
            "node_id": [str(i) for i in range(n)],
            "size": 1,
            "colour_value": df_meta[colour_by].astype(str).tolist() if colour_by in df_meta.columns else [""] * n,
            "members": df_meta["cpd_id"].astype(str).tolist(),
        }
    )
    edges_df = pd.DataFrame([{"source": str(a), "target": str(b)} for (a, b) in sorted(edges)])

    write_tsv(df=nodes_df, path=nodes_path, logger=logger, index=False)
    write_tsv(df=edges_df, path=edges_path, logger=logger, index=False)

    _draw_graph_pdf(
        nodes=nodes_df,
        edges=edges_df,
        output_pdf=pdf_path,
        logger=logger,
    )


def _draw_graph_pdf(*, nodes: pd.DataFrame, edges: pd.DataFrame, output_pdf: Path, logger: logging.Logger) -> None:
    """
    Render a simple graph PDF with matplotlib.

    Parameters
    ----------
    nodes : pd.DataFrame
        Columns: node_id, size, colour_value (optional), members (semicolon-separated).
    edges : pd.DataFrame
        Columns: source, target.
    output_pdf : Path
        Output PDF path.
    logger : logging.Logger
        Logger instance.
    """
    # Build adjacency
    node_ids = nodes["node_id"].tolist()
    id_to_ix = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    adj = [[] for _ in range(n)]
    for _, row in edges.iterrows():
        a = id_to_ix.get(str(row["source"]))
        b = id_to_ix.get(str(row["target"]))
        if a is None or b is None:
            continue
        adj[a].append(b)
        adj[b].append(a)

    # Positions: simple Fruchterman-Reingold via networkx-like layout (manual)
    # To avoid hard dependency on networkx, use a basic deterministic layout.
    # For larger graphs, consider installing networkx for nicer layouts.
    rng = np.random.default_rng(seed=42)
    pos = rng.normal(loc=0.0, scale=1.0, size=(n, 2))

    # One or two smoothing passes by averaging neighbours
    for _ in range(50):
        new_pos = pos.copy()
        for i in range(n):
            if adj[i]:
                new_pos[i] = 0.5 * pos[i] + 0.5 * np.mean(pos[adj[i]], axis=0)
        pos = new_pos

    # Colour mapping by 'colour_value' categorical
    colours = None
    if "colour_value" in nodes.columns and nodes["colour_value"].notna().any():
        categories = nodes["colour_value"].astype(str).fillna("").tolist()
        cats = sorted(set(categories))
        cmap = mpl.colormaps.get("tab20", mpl.colormaps["tab20"])
        colour_map = {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
        colours = [colour_map.get(c, (0.5, 0.5, 0.5, 1.0)) for c in categories]
    sizes = nodes["size"].astype(float).clip(lower=1.0).values
    sizes = 20.0 * np.log1p(sizes)  # perceptual scaling

    fig, ax = plt.subplots(figsize=(9, 7))
    # edges
    for _, row in edges.iterrows():
        a = id_to_ix[str(row["source"])]
        b = id_to_ix[str(row["target"])]
        ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], lw=0.4, c="0.8", zorder=1)
    # nodes
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=colours if colours is not None else "0.3", alpha=0.95, zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Topological graph")
    fig.tight_layout()
    fig.savefig(output_pdf, dpi=300)
    plt.close(fig)
    logger.info("Saved topological graph PDF to %s", output_pdf)


# ==========
# UMAP / PHATE
# ==========

def run_umap(
    *,
    X: pd.DataFrame,
    df_meta: pd.DataFrame,
    out_dir: Path,
    colour_by: str,
    metric: str,
    n_neighbors: int,
    min_dist: float,
    logger: logging.Logger,
) -> None:
    """
    Compute UMAP (if available) and save PDF + coords TSV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    df_meta : pd.DataFrame
        Metadata (must include 'cpd_id').
    out_dir : Path
        Output directory.
    colour_by : str
        Column for colouring if present.
    metric : str
        'cosine' | 'euclidean'.
    n_neighbors : int
        UMAP n_neighbours.
    min_dist : float
        UMAP min_dist.
    logger : logging.Logger
        Logger instance.
    """
    if not UMAP_AVAILABLE:
        logger.warning("UMAP not available; skipping.")
        return

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    emb = reducer.fit_transform(X.values)
    coords = pd.DataFrame({"cpd_id": df_meta["cpd_id"].astype(str), "UMAP1": emb[:, 0], "UMAP2": emb[:, 1]})
    write_tsv(df=coords, path=out_dir / "umap_coords.tsv", logger=logger, index=False)

    # simple PDF
    fig, ax = plt.subplots(figsize=(8, 6))
    if colour_by in df_meta.columns:
        # build categorical colour map
        cats = df_meta[colour_by].astype(str).tolist()
        uniq = sorted(set(cats))
        cmap = mpl.colormaps.get("tab20", mpl.colormaps["tab20"])
        colour_map = {c: cmap(i % cmap.N) for i, c in enumerate(uniq)}
        colours = [colour_map[c] for c in cats]
        ax.scatter(coords["UMAP1"], coords["UMAP2"], s=6, c=colours, lw=0, alpha=0.95)
        ax.set_title(f"UMAP ({metric}) coloured by {colour_by}")
    else:
        ax.scatter(coords["UMAP1"], coords["UMAP2"], s=6, c="0.3", lw=0, alpha=0.95)
        ax.set_title(f"UMAP ({metric})")

    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "umap.pdf", dpi=300)
    plt.close(fig)
    logger.info("Saved UMAP PDF + coords.")


def run_phate(
    *,
    X: pd.DataFrame,
    df_meta: pd.DataFrame,
    out_dir: Path,
    colour_by: str,
    knn: int,
    logger: logging.Logger,
) -> None:
    """
    Compute PHATE (if available) and save PDF + coords TSV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    df_meta : pd.DataFrame
        Metadata (must include 'cpd_id').
    out_dir : Path
        Output directory.
    colour_by : str
        Column for colouring if present.
    knn : int
        k-NN parameter for PHATE.
    logger : logging.Logger
        Logger instance.
    """
    if not PHATE_AVAILABLE:
        logger.warning("PHATE not available; skipping.")
        return

    ph = phate.PHATE(k=knn, random_state=42)
    emb = ph.fit_transform(X.values)
    coords = pd.DataFrame({"cpd_id": df_meta["cpd_id"].astype(str), "PHATE1": emb[:, 0], "PHATE2": emb[:, 1]})
    write_tsv(df=coords, path=out_dir / "phate_coords.tsv", logger=logger, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    if colour_by in df_meta.columns:
        cats = df_meta[colour_by].astype(str).tolist()
        uniq = sorted(set(cats))
        cmap = mpl.colormaps.get("tab20", mpl.colormaps["tab20"])
        colour_map = {c: cmap(i % cmap.N) for i, c in enumerate(uniq)}
        colours = [colour_map[c] for c in cats]
        ax.scatter(coords["PHATE1"], coords["PHATE2"], s=6, c=colours, lw=0, alpha=0.95)
        ax.set_title(f"PHATE coloured by {colour_by}")
    else:
        ax.scatter(coords["PHATE1"], coords["PHATE2"], s=6, c="0.3", lw=0, alpha=0.95)
        ax.set_title("PHATE")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "phate.pdf", dpi=300)
    plt.close(fig)
    logger.info("Saved PHATE PDF + coords.")


# =====
# Main
# =====

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description="CLIPn latent visualisation with topology-first graphs.")
    p.add_argument("--latent_csv", required=True, help="TSV with latent features + metadata.")
    p.add_argument("--plots", required=True, help="Output directory for plots/TSVs.")
    p.add_argument("--latent_prefix", default=None, help="Prefix for latent columns (default: use digit-named).")
    p.add_argument("--colour_by", default="Dataset", help="Metadata column to colour by (default: Dataset).")
    p.add_argument("--embedding", choices=["topo", "umap", "phate", "all"], default="topo",
                   help="Which outputs to compute (default: topo).")

    # Mapper/topo
    p.add_argument("--mapper_lens", choices=["pca", "umap", "identity"], default="pca", help="Lens for Mapper (default: pca).")
    p.add_argument("--mapper_n_cubes", type=int, default=15, help="Mapper cover n_cubes (default: 15).")
    p.add_argument("--mapper_overlap", type=float, default=0.4, help="Mapper cover overlap fraction (default: 0.4).")
    p.add_argument("--mapper_cluster", choices=["dbscan", "hdbscan"], default="dbscan", help="Clusterer in bins (default: dbscan).")
    p.add_argument("--mapper_eps", type=float, default=0.5, help="DBSCAN eps (default: 0.5).")
    p.add_argument("--mapper_min_samples", type=int, default=5, help="DBSCAN min_samples (default: 5).")
    p.add_argument("--knn_k", type=int, default=10, help="k for k-NN fallback when Mapper is unavailable (default: 10).")

    # UMAP / PHATE params
    p.add_argument("--umap_metric", default="cosine", help="UMAP metric (default: cosine).")
    p.add_argument("--umap_n_neighbors", type=int, default=40, help="UMAP n_neighbours (default: 40).")
    p.add_argument("--umap_min_dist", type=float, default=0.25, help="UMAP min_dist (default: 0.25).")
    p.add_argument("--phate_knn", type=int, default=15, help="PHATE k-NN (default: 15).")

    return p.parse_args()


def main() -> None:
    """
    Entry point for plotting script.
    """
    args = parse_args()
    outdir = Path(args.plots)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir=outdir)

    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("Args: %s", vars(args))

    logger.info("Reading latent TSV: %s", args.latent_csv)
    df = pd.read_csv(filepath_or_buffer=args.latent_csv, sep="\t")

    validate_columns(df=df, required=["cpd_id"], logger=logger)
    df = clean_merge_artifacts(df=df, logger=logger)

    # Pick latent features
    X, latent_cols = select_latent(df=df, prefix=args.latent_prefix, logger=logger)

    # Metadata view for colouring
    meta_cols = [c for c in ["cpd_id", "cpd_type", "Library", "Dataset", "Sample"] if c in df.columns]
    df_meta = df[meta_cols].copy()

    # Topological graph (default)
    if args.embedding in ("topo", "all"):
        build_topological_graph(
            X=X,
            df_meta=df_meta,
            out_dir=outdir,
            colour_by=args.colour_by,
            mapper_lens=args.mapper_lens,
            n_cubes=args.mapper_n_cubes,
            overlap=args.mapper_overlap,
            cluster_alg=args.mapper_cluster,
            dbscan_eps=args.mapper_eps,
            dbscan_min_samples=args.mapper_min_samples,
            knn_k=args.knn_k,
            logger=logger,
        )

    # UMAP
    if args.embedding in ("umap", "all"):
        run_umap(
            X=X,
            df_meta=df_meta,
            out_dir=outdir,
            colour_by=args.colour_by,
            metric=args.umap_metric,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            logger=logger,
        )

    # PHATE
    if args.embedding in ("phate", "all"):
        run_phate(
            X=X,
            df_meta=df_meta,
            out_dir=outdir,
            colour_by=args.colour_by,
            knn=args.phate_knn,
            logger=logger,
        )

    logger.info("Plotting complete.")


if __name__ == "__main__":
    main()
