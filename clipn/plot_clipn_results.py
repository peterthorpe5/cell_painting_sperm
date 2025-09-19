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


--mapper_lens: pca is fast & stable; umap can separate non-linear structure (slower).

--mapper_n_cubes: number of bins across the lens; small data (~50 pts): 6–12; larger (1k+): 10–20.

--mapper_overlap: 0.3–0.6; more overlap → more node connectivity (and potentially hairier graphs).

--mapper_cluster: dbscan is fine; hdbscan gives nicer clusters if available.

--mapper_eps (DBSCAN radius): if you get too many tiny nodes, increase eps; if one giant node, decrease eps.

--mapper_min_samples: 2–10; higher = stricter clusters (fewer nodes).

If you ever see 0 edges, try raising overlap or eps, or increasing --mapper_n_cubes.



"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import re
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


def _html_escape(text: str) -> str:
    """
    Escape minimal HTML entities for safe insertion into an HTML string.

    Parameters
    ----------
    text : str
        Raw text to escape.

    Returns
    -------
    str
        HTML-escaped text.
    """
    import html
    return html.escape(text, quote=True)


def _inject_html_legend(*, html_text: str, colour_map: dict[str, str],
                        counts: dict[str, int]) -> str:
    """
    Inject a floating legend (category → colour swatch) into a PyVis HTML.

    Parameters
    ----------
    html_text : str
        The original HTML produced by PyVis.
    colour_map : dict[str, str]
        Mapping from category string to hex colour (e.g., '#1f77b4').
    counts : dict[str, int]
        Mapping from category string to its node count for display.

    Returns
    -------
    str
        Modified HTML with an appended legend container before </body>.
    """
    # Build legend items (sorted by category for determinism)
    items = []
    for cat in sorted(colour_map.keys()):
        colour = colour_map[cat]
        label = cat if cat else "(blank)"
        n = counts.get(cat, 0)
        items.append(
            f'<div class="legend-row">'
            f'  <span class="swatch" style="background:{_html_escape(colour)};"></span>'
            f'  <span class="label">{_html_escape(label)}'
            f'  <span class="count">(n={n})</span></span>'
            f'</div>'
        )

    legend_html = f"""
<!-- Injected legend -->
<style>
  .legend-container {{
    position: fixed;
    top: 12px;
    right: 12px;
    max-height: 70vh;
    overflow: auto;
    background: rgba(255,255,255,0.95);
    border: 1px solid #ddd;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    border-radius: 10px;
    padding: 10px 12px;
    z-index: 9999;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
                 Arial, "Noto Sans", "Liberation Sans", sans-serif;
    font-size: 13px;
    color: #222;
  }}
  .legend-title {{
    font-weight: 600;
    margin-bottom: 8px;
  }}
  .legend-row {{
    display: flex;
    align-items: center;
    margin: 4px 0;
    gap: 8px;
    line-height: 1.25;
    white-space: nowrap;
  }}
  .swatch {{
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid rgba(0,0,0,0.15);
    flex-shrink: 0;
  }}
  .label .count {{
    color: #666;
    margin-left: 6px;
    font-weight: 400;
  }}
</style>
<div class="legend-container">
  <div class="legend-title">Legend: node colour by category</div>
  {''.join(items)}
</div>
"""
    # Insert just before </body> (fallback: append if tag missing)
    lower = html_text.lower()
    idx = lower.rfind("</body>")
    if idx == -1:
        return html_text + legend_html
    return html_text[:idx] + legend_html + html_text[idx:]



def _categorical_palette(n: int) -> list[str]:
    """
    Return a deterministic list of n distinct CSS hex colours.

    Parameters
    ----------
    n : int
        Number of colours required.

    Returns
    -------
    list[str]
        Hex colour strings (e.g., '#1f77b4').

    Notes
    -----
    Uses a fixed qualitative palette and repeats deterministically if more
    categories are requested than the base palette length. This ensures
    stable colouring across runs.
    """
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
    ]
    if n <= len(base):
        return base[:n]
    # Extend deterministically without randomness
    out: list[str] = []
    i: int = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out


def _draw_graph_html_pyvis(
    *,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_html: Path,
    logger: logging.Logger
) -> None:
    """
    Write an interactive network HTML using PyVis (fallback if KeplerMapper is unavailable),
    with coloured and sized nodes and an injected legend.

    Parameters
    ----------
    nodes : pd.DataFrame
        Node table with columns:
        - 'node_id' : str, node identifier
        - 'size' : int, number of members in the node (used for node sizing)
        - 'colour_value' : str, categorical value for colouring (e.g., Dataset)
        - 'members' : str, semicolon-separated cpd_id list for tooltip
    edges : pd.DataFrame
        Edge table with columns: 'source', 'target'.
    output_html : Path
        Destination HTML file path.
    logger : logging.Logger
        Logger for status messages.

    Behaviour
    ---------
    - Colours nodes by the categorical 'colour_value'.
    - Scales node size by 'size' via the PyVis 'value' attribute.
    - Adds a floating legend (category → colour, with counts).
    - Includes informative tooltips with node id, category, and members.
    """
    from pyvis.network import Network

    df = nodes.copy()
    if "colour_value" not in df.columns:
        df["colour_value"] = ""

    # Normalise types and missing values
    df["colour_value"] = df["colour_value"].fillna("").astype(str)
    df["members"] = df.get("members", "").fillna("").astype(str)
    df["size"] = df.get("size", 1).fillna(1).astype(int)
    df["node_id"] = df.get("node_id").astype(str)

    # Build categorical colour map and counts
    categories = sorted(df["colour_value"].unique().tolist())
    palette = _categorical_palette(n=len(categories))
    colour_map = dict(zip(categories, palette))
    cat_counts = df["colour_value"].value_counts(dropna=False).to_dict()

    net = Network(
        height="800px",
        width="100%",
        directed=False,
        notebook=False,
        bgcolor="#ffffff",
        font_color="#000000",
    )
    net.toggle_physics(True)

    # Add nodes
    for _, row in df.iterrows():
        label = row["node_id"]
        cat = row["colour_value"]
        members = row["members"]
        value = int(row["size"])
        colour = colour_map.get(cat, "#999999")

        title = (
            f"<b>Node:</b> {label}"
            f"<br><b>Colour:</b> {cat if cat else '(blank)'}"
            f"<br><b>Members:</b> { _html_escape(members) }"
        )

        net.add_node(
            n_id=label,
            label=label,
            title=title,
            color=colour,
            value=value,
        )

    # Add edges
    for _, row in edges.iterrows():
        net.add_edge(
            str(row["source"]),
            str(row["target"])
        )

    # Write to a temporary file first, then inject legend HTML
    output_html.parent.mkdir(parents=True, exist_ok=True)
    tmp_html = output_html.with_suffix(".tmp.html")
    net.write_html(path=str(tmp_html))

    html_text = tmp_html.read_text(encoding="utf-8")
    html_with_legend = _inject_html_legend(
        html_text=html_text,
        colour_map=colour_map,
        counts=cat_counts,
    )
    output_html.write_text(html_with_legend, encoding="utf-8")

    # Best-effort clean-up
    try:
        tmp_html.unlink(missing_ok=True)
    except Exception:
        pass

    logger.info("Saved interactive topology HTML (with legend) to %s", output_html)




def _majority_label_and_purity(
    *, labels: pd.Series
) -> tuple[str, float]:
    """
    Compute the majority label and its purity.

    Parameters
    ----------
    labels : pd.Series
        Categorical labels for samples in a node.

    Returns
    -------
    tuple[str, float]
        (majority_label, purity_fraction in [0, 1])
    """
    if labels.empty:
        return ("", 0.0)
    counts = labels.astype(str).value_counts(dropna=False)
    maj = counts.idxmax()
    purity = counts.max() / counts.sum()
    return (str(maj), float(purity))


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
    Select latent numeric columns (digit-named by default) and drop metadata-like names.

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
        raw_cols = [c for c in df.columns
                    if isinstance(c, str) and c.startswith(prefix) and pd.api.types.is_numeric_dtype(df[c])]
    else:
        raw_cols = [c for c in df.columns
                    if isinstance(c, str) and c.isdigit() and pd.api.types.is_numeric_dtype(df[c])]

    if not raw_cols:
        logger.error("No latent feature columns found (prefix=%s).", prefix)
        raise ValueError("No latent feature columns found. Check column names and --latent_prefix.")

    # Drop any banned/metadata-like names defensively
    cols = [c for c in raw_cols if not _is_metadata_like(c)]
    dropped = [c for c in raw_cols if c not in cols]
    if dropped:
        logger.info("Excluded %d metadata/housekeeping columns from latent set (first few: %s)",
                    len(dropped), ", ".join(map(str, dropped[:10])))

    if not cols:
        logger.error("All candidate columns were excluded as metadata/housekeeping.")
        raise ValueError("No usable latent features remain after exclusions.")

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


def _build_tooltips_array(*, df_meta: pd.DataFrame, tooltip_cols: list[str]) -> np.ndarray:
    """
    Build HTML-safe tooltips from selected metadata columns.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata for each sample (rows align with X).
    tooltip_cols : list[str]
        Columns to show in the tooltip.

    Returns
    -------
    np.ndarray
        Array of HTML strings, one per row in df_meta.
    """
    cols = [c for c in tooltip_cols if c in df_meta.columns]
    if not cols:
        cols = ["cpd_id"]
    rows = []
    for _, r in df_meta[cols].astype(str).iterrows():
        parts = [f"<b>{c}</b>: {r[c]}" for c in cols]
        rows.append("<br>".join(parts))
    return np.array(rows, dtype=object)



# Columns we never want to treat as features (even if numeric)
BANNED_FEATURES_EXACT = {
    "ImageNumber",
    "Number_Object_Number",
    "ObjectNumber",
    "TableNumber",
}

# Heuristics for metadata/housekeeping columns (case-insensitive)
BANNED_FEATURES_REGEX = re.compile(
    r"""(?ix)
        ( ^metadata($|_)         # Metadata*, *_Metadata
        | _metadata$
        | ^filename_             # FileName_*
        | ^pathname_             # PathName_*
        | ^url_                  # URL_*
        | ^parent_               # Parent_*
        | ^children_             # Children_*
        | (^|_)imagenumber$      # ImageNumber
        | ^number_object_number$ # Number_Object_Number
        | ^objectnumber$         # ObjectNumber
        | ^tablenumber$          # TableNumber
        )
    """
)

def _is_metadata_like(col: str) -> bool:
    """
    Return True if a column name is metadata/housekeeping and must not be used as a feature.
    """
    cname = str(col)
    if cname in BANNED_FEATURES_EXACT:
        return True
    return bool(BANNED_FEATURES_REGEX.search(cname.lower()))


def _draw_graph_html_pyvis(
    *,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_html: Path,
    logger: logging.Logger
) -> None:
    """
    Write an interactive network HTML using pyvis (fallback if KeplerMapper isn't available).

    Parameters
    ----------
    nodes : pd.DataFrame
        Columns: node_id, size, colour_value (optional), members (semicolon-separated).
    edges : pd.DataFrame
        Columns: source, target.
    output_html : Path
        Output HTML file path.
    logger : logging.Logger
        Logger.
    """
    try:
        from pyvis.network import Network
    except Exception as exc:
        logger.warning("pyvis not available; cannot create interactive topology HTML (%s).", exc)
        return

    net = Network(height="800px", width="100%", directed=False, notebook=False)
    net.barnes_hut()  # nice physics defaults

    # Add nodes with title tooltips
    for _, row in nodes.iterrows():
        label = str(row.get("node_id", ""))
        members = str(row.get("members", ""))
        colour = row.get("colour_value", "")
        title = f"<b>Node:</b> {label}<br><b>Colour:</b> {colour}<br><b>Members:</b> {members}"
        net.add_node(n_id=label, label=label, title=title)

    # Add edges
    for _, row in edges.iterrows():
        net.add_edge(str(row["source"]), str(row["target"]))

    output_html.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_html))
    logger.info("Saved interactive topology HTML to %s", output_html)


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
    interactive: bool,
    tooltip_cols: list[str],
    logger: logging.Logger,
) -> None:
    """
    Build a topological graph (Mapper if available; else k-NN fallback) and save:
    - topo_nodes.tsv
    - topo_edges.tsv
    - topo_graph.pdf
    - topo_graph.html (if interactive requested and deps available)
    """
    nodes_path = out_dir / "topo_nodes.tsv"
    edges_path = out_dir / "topo_edges.tsv"
    pdf_path   = out_dir / "topo_graph.pdf"
    html_path  = out_dir / "topo_graph.html"

    # KeplerMapper branch
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


        edge_rows = []
       
        # Deduplicate undirected edges using string node IDs
        edge_keys = set()
        # Nodes/edges TSV
        node_rows = []
        for nid, members in graph["nodes"].items():
            members = list(members)
            colour_val = None
            if colour_by in df_meta.columns:
                vals = df_meta.iloc[members][colour_by].astype(str)
                if len(vals):
                    colour_val = vals.value_counts().idxmax()
            colour_by_resolved = resolve_meta_column(
                df=df_meta, requested=colour_by, fallbacks=["Library", "Dataset"], logger=logger
                        )
            maj, purity = _majority_label_and_purity(labels=df_meta.iloc[members][colour_by_resolved])
            node_rows.append(
                {
                    "node_id": str(nid),
                    "size": int(len(members)),
                    "colour_value": maj,
                    "purity": purity,
                    "members": ";".join(df_meta.iloc[members]["cpd_id"].astype(str).tolist()),
                }
                            )


        # Deduplicate undirected edges using string node IDs
        edge_keys = set()
        for a, nbrs in graph["links"].items():
            sa = str(a)
            for b in nbrs:
                sb = str(b)
                if sa == sb:
                    continue  # no self-loops
                key = tuple(sorted((sa, sb)))  # canonical undirected edge
                edge_keys.add(key)

        valid_nodes = {str(nid) for nid in graph["nodes"].keys()}
        edge_rows = [
            {"source": s, "target": t}
            for (s, t) in sorted(edge_keys)
            if (s in valid_nodes and t in valid_nodes)
        ]

        nodes_df = pd.DataFrame(node_rows)
        edges_df = pd.DataFrame(edge_rows)
        logger.info("Mapper produced %d nodes and %d edges.", len(nodes_df), len(edges_df))

        write_tsv(df=nodes_df, path=nodes_path, logger=logger, index=False)
        write_tsv(df=edges_df, path=edges_path, logger=logger, index=False)


        # PDF (static)
        _draw_graph_pdf(nodes=nodes_df, edges=edges_df, output_pdf=pdf_path, logger=logger)


        # HTML (interactive) via KeplerMapper
        if interactive:
            logger.info("Building interactive HTML visualisation with KeplerMapper.")
            try:
                tooltips = _build_tooltips_array(df_meta=df_meta, tooltip_cols=tooltip_cols)
                tooltip_keep = [c for c in tooltip_columns if c in df_meta.columns]
                if len(tooltip_keep) != len(tooltip_columns):
                    missing = sorted(set(tooltip_columns) - set(tooltip_keep))
                    logger.warning("Dropping missing tooltip columns: %s", ", ".join(missing))
                tooltips = tooltip_keep

                tooltips = list(map(str, tooltips))  # ensure list[str], not ndarray/DataFrame

                # --- Point 1: ensure contiguous numpy arrays and 1-D colour vector ---
                X_arr = np.asarray(X.values if hasattr(X, "values") else X)
                X_arr = np.ascontiguousarray(X_arr)
                lens_arr = np.asarray(lens)
                lens_arr = np.ascontiguousarray(lens_arr)

                # KeplerMapper likes a 1-D colour array; take the first lens component if 2-D
                if lens_arr.ndim > 1:
                    color_vec = np.ascontiguousarray(lens_arr[:, 0]).ravel()
                else:
                    color_vec = np.ascontiguousarray(lens_arr).ravel()

                # --- Point 2: try color_function, fall back to color_values if TypeError ---
                try:
                    mapper.visualize(
                        graph,
                        path_html=str(html_path),
                        title="Topological graph (Mapper)",
                        X=X_arr,
                        lens=lens_arr,
                        color_function=color_vec,
                        custom_tooltips=tooltips,
                    )
                except TypeError:
                    # Some versions expect 'color_values' instead of 'color_function'
                    mapper.visualize(
                        graph,
                        path_html=str(html_path),
                        title="Topological graph (Mapper)",
                        X=X_arr,
                        lens=lens_arr,
                        color_values=color_vec,
                        custom_tooltips=tooltips,
                    )
                logger.info("Saved interactive topology HTML to %s", html_path)
            except Exception as exc:
                logger.warning("KeplerMapper HTML visualisation failed, falling back to PyVis: %s", exc)
                _draw_graph_html_pyvis(
                    nodes=nodes_df, edges=edges_df,
                    output_html=html_path, logger=logger
                )


        return

    # Fallback: k-NN graph
    logger.warning("KeplerMapper not available; building k-NN graph fallback (k=%d).", knn_k)
    n = len(X)
    k = min(max(2, knn_k), max(2, n - 1))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X.values)
    _, idx = nn.kneighbors(X.values)

    edges = set()
    for i in range(n):
        for j in idx[i][1:]:
            a, b = sorted((int(i), int(j)))
            edges.add((a, b))

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

    _draw_graph_pdf(nodes=nodes_df, edges=edges_df, output_pdf=pdf_path, logger=logger)

    if interactive:
        _draw_graph_html_pyvis(nodes=nodes_df, edges=edges_df, output_html=html_path, logger=logger)


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
    # Gracefully skip if nothing to draw
    if nodes is None or nodes.empty or ("node_id" not in nodes.columns):
        logger.warning("No topo nodes to draw; skipping PDF: %s", output_pdf)
        return
    if edges is None or ("source" not in edges.columns) or ("target" not in edges.columns):
        logger.warning("No topo edges to draw; skipping PDF: %s", output_pdf)
        return
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


def resolve_meta_column(*, df: pd.DataFrame, requested: str,
                        fallbacks: list[str], logger: logging.Logger) -> str:
    """
    Return a real column present in df for a requested metadata field (case-insensitive),
    trying fallbacks if needed.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    req = (requested or "").lower()
    if req in cols_lower:
        return cols_lower[req]
    for name in fallbacks:
        if name.lower() in cols_lower:
            real = cols_lower[name.lower()]
            logger.warning("Column '%s' not found; using '%s' instead.", requested, real)
            return real
    raise KeyError(f"None of {[requested, *fallbacks]} present; have: {list(df.columns)}")


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
    p.add_argument("--colour_by", default="Library", help="Metadata column to colour by (default: Library).")
    p.add_argument("--embedding", choices=["topo", "umap", "phate", "all"], default="topo",
                   help="Which outputs to compute (default: topo).")

    # Mapper/topo
    p.add_argument("--mapper_lens", choices=["pca", "umap", "identity"], default="pca", help="Lens for Mapper (default: pca).")
    p.add_argument("--mapper_n_cubes", type=int, default=12, help="Mapper cover n_cubes (default: 12).")
    p.add_argument("--mapper_overlap", type=float, default=0.4, help="Mapper cover overlap fraction (default: 0.4).")
    p.add_argument("--mapper_cluster", choices=["dbscan", "hdbscan"], default="dbscan", help="Clusterer in bins (default: dbscan).")
    p.add_argument("--mapper_eps", type=float, default=0.8, help="DBSCAN eps (default: 0.8).")
    p.add_argument("--mapper_min_samples", type=int, default=3, help="DBSCAN min_samples (default: 3).")
    p.add_argument("--knn_k", type=int, default=10, help="k for k-NN fallback when Mapper is unavailable (default: 10).")
    p.add_argument("--interactive_topo", action="store_true",
                   help="Write an interactive HTML for the topological graph.")
    p.add_argument("--tooltip_columns", nargs="+",
                   default=["cpd_id", "cpd_type", "Library"],
                   help="Columns to include in node/sample tooltips for interactive topo HTML.")


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
            interactive=args.interactive_topo,
            tooltip_cols=args.tooltip_columns,
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
