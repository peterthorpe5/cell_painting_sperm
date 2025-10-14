#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phenotype-wise enrichment of significant features from acrosome-vs-DMSO TSVs.

Overview
--------
Given a directory containing files like:
    <CompoundName>_vs_DMSO_significant_features.tsv

and a phenotype table with columns:
    name   published_phenotypes

(where multiple phenotypes are separated by ';'), this script:

1) Parses compound -> set of significant features from each TSV.
2) Normalises and aligns compounds to published phenotypes (case-insensitive).
3) Builds a compound × feature presence/absence matrix.
4) For each phenotype, performs one-sided Fisher's exact tests to find features
   enriched among compounds annotated with that phenotype versus all others.
5) Adjusts p-values across all tests with Benjamini–Hochberg (FDR).
6) Writes several tab-separated outputs for downstream analysis or plotting.

Outputs (all TSV, tab-separated)
--------------------------------
- compound_feature_matrix.tsv
    Binary matrix (rows = compounds, columns = features).
- phenotype_assignment.tsv
    Normalised mapping of compound to one-or-more phenotypes (one row per pair).
- feature_enrichment_by_phenotype.tsv
    Long table with counts, odds ratios, raw p-values, BH q-values.
- top_enriched_features_by_phenotype.tsv
    Filtered view by q-value and minimum support.
- upset_compound_feature.tsv
    Duplicate of compound_feature_matrix.tsv (explicitly named for UpSet tools).
- upset_phenotype_feature.tsv
    Phenotype × feature binary matrix (feature present in ≥1 compound with that
    phenotype).

Notes
-----
- Only the presence/absence of significance in each compound's TSV is used
  (no weighting by effect size).
- If a compound appears in the phenotype file but has no TSV (or vice versa),
  it is safely handled; tests only use compounds with at least one parsed TSV.

Usage
-----
python phenotype_feature_enrichment.py \
    --sig_dir "/cluster/gjb_lab/pthorpe001/2025_jason_cell_painting/STB_vs_mitotox_image_level/ACROSOME_SCREEN/results_with_meta/STB_V1/significant" \
    --phenotype_tsv "/path/to/phenotypes.tsv" \
    --phenotype_col "published_phenotypes" \
    --name_col "name" \
    --min_support 2 \
    --qvalue_alpha 0.05 \
    --no_plots

All outputs are written into --out_dir (defaults to <sig_dir>/../summary).
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

try:
    # Fallback to statsmodels if desired, but BH is easy to implement.
    import statsmodels.stats.multitest as smm
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False

try:
    import matplotlib.pyplot as plt  # Only used if plots requested
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

try:
    from upsetplot import UpSet, from_indicators
    HAS_UPSETPLOT = True
except Exception:  # pragma: no cover
    HAS_UPSETPLOT = False


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Perform Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    pvals : numpy.ndarray
        Array of raw p-values.

    Returns
    -------
    numpy.ndarray
        Array of BH-adjusted q-values (same shape as pvals).
    """
    if HAS_STATSMODELS:
        _, qvals = smm.fdrcorrection(pvals, alpha=0.05, method="indep")
        return qvals
    # Simple manual BH as a fallback
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty_like(p)
    ranked[order] = np.arange(1, n + 1)
    q = p * n / ranked
    # Ensure monotonicity
    q_sorted = q[order]
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = q_sorted
    return out


def _normalise_name(x: str) -> str:
    """
    Normalise a compound name for case-insensitive matching.

    Parameters
    ----------
    x : str
        Original name.

    Returns
    -------
    str
        Normalised name (lower-cased, single spaces).
    """
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def _parse_compound_from_filename(path: Path) -> str:
    """
    Extract the compound name from a filename like:
    '<Compound>_vs_DMSO_significant_features.tsv'

    This function is case-insensitive for the suffix.

    Parameters
    ----------
    path : pathlib.Path
        Path to the TSV file.

    Returns
    -------
    str
        Parsed compound name (original case retained where possible).
    """
    stem = path.stem  # without extension
    # Remove an optional duplicate suffix patterns and tolerate spaces
    pat = re.compile(r"_vs[_\s]*DMSO_significant_features$", flags=re.IGNORECASE)
    name = re.sub(pat, "", stem).strip()
    return name


def _read_significant_features(sig_file: Path) -> Set[str]:
    """
    Read a single significant-features TSV and return the set of features.

    Parameters
    ----------
    sig_file : pathlib.Path
        Path to the significant-features TSV.

    Returns
    -------
    set[str]
        Set of feature names present in the 'feature' column.
    """
    df = pd.read_csv(sig_file, sep="\t", dtype=str)
    if "feature" not in df.columns:
        raise ValueError(f"'feature' column not found in {sig_file}")
    feats = set(df["feature"].astype(str).str.strip())
    return feats


def _explode_phenotypes(df: pd.DataFrame, name_col: str, phenotype_col: str) -> pd.DataFrame:
    """
    Split multi-phenotype strings into one row per (compound, phenotype).

    Parameters
    ----------
    df : pandas.DataFrame
        Phenotype table with at least the columns in name_col and phenotype_col.
    name_col : str
        Column containing compound names.
    phenotype_col : str
        Column containing ';' separated phenotypes.

    Returns
    -------
    pandas.DataFrame
        Two-column DataFrame with columns ['name', 'phenotype'].
    """
    sub = df[[name_col, phenotype_col]].copy()
    sub[name_col] = sub[name_col].astype(str).str.strip()
    sub[phenotype_col] = sub[phenotype_col].fillna("").astype(str)

    rows: List[Tuple[str, str]] = []
    for _, r in sub.iterrows():
        nm = r[name_col].strip()
        phenos = [p.strip() for p in r[phenotype_col].split(";") if p.strip()]
        if not phenos:
            rows.append((nm, "NA"))
        else:
            for p in phenos:
                rows.append((nm, p))
    out = pd.DataFrame(rows, columns=["name", "phenotype"]).drop_duplicates()
    return out


def _prepare_features_by_phenotype(
    *,
    ph_feat: pd.DataFrame,
    phenotypes_keep: List[str] | None,
    min_feature_phenotypes: int,
) -> pd.DataFrame:
    """
    Build a features × phenotypes boolean matrix and filter it.

    Parameters
    ----------
    ph_feat : pandas.DataFrame
        Phenotype × feature 0/1 matrix as written earlier in the script.
    phenotypes_keep : list[str] or None
        Optional explicit subset of phenotype names to include.
    min_feature_phenotypes : int
        Keep only features present in at least this many phenotypes.

    Returns
    -------
    pandas.DataFrame
        Boolean DataFrame with rows = features, columns = phenotypes.
    """
    if phenotypes_keep:
        present = [p for p in phenotypes_keep if p in ph_feat.index]
        missing = [p for p in phenotypes_keep if p not in ph_feat.index]
        if missing:
            logging.warning("Requested phenotypes missing and will be ignored: %s", ", ".join(missing))
        if present:
            ph_feat = ph_feat.loc[present]

    feat_by_ph = ph_feat.T.astype(bool)

    if int(min_feature_phenotypes) > 1:
        mask = feat_by_ph.sum(axis=1) >= int(min_feature_phenotypes)
        feat_by_ph = feat_by_ph.loc[mask]

    return feat_by_ph


def _write_upset_intersections(
    *,
    feat_by_ph: pd.DataFrame,
    out_tsv: Path,
) -> None:
    """
    Write phenotype-set intersection sizes to a TSV (tab-separated).

    Parameters
    ----------
    feat_by_ph : pandas.DataFrame
        Features × phenotypes boolean matrix.
    out_tsv : pathlib.Path
        Output TSV path.
    """
    rows: list[tuple[str, str]] = []
    for feature, row in feat_by_ph.iterrows():
        combo = tuple(sorted(row.index[row.values]))
        if combo:
            rows.append((";".join(combo), feature))

    if not rows:
        pd.DataFrame({"phenotype_set": [], "n_features": []}).to_csv(out_tsv, sep="\t", index=False)
        return

    df = pd.DataFrame(rows, columns=["phenotype_set", "feature"])
    out = (
        df.groupby("phenotype_set", as_index=False)
        .agg(n_features=("feature", "size"))
        .sort_values("n_features", ascending=False)
    )
    out.to_csv(out_tsv, sep="\t", index=False)


def run(
    sig_dir: Path,
    phenotype_tsv: Path,
    out_dir: Path,
    name_col: str,
    phenotype_col: str,
    min_support: int,
    qvalue_alpha: float,
    make_plots: bool,
    plot_all: bool,
    make_upset: bool,                  
    min_feature_phenotypes: int,       
    max_upset_sets: int,               
    phenotypes_keep: List[str] | None, 
) -> None:
    """
    Execute the phenotype-feature enrichment workflow.

    Parameters
    ----------
    sig_dir : pathlib.Path
        Directory containing '*_vs_DMSO_significant_features.tsv' files.
    phenotype_tsv : pathlib.Path
        TSV file with columns [name_col, phenotype_col].
    out_dir : pathlib.Path
        Output directory for all result files.
    name_col : str
        Column in phenotype_tsv containing compound names.
    phenotype_col : str
        Column in phenotype_tsv containing ';' separated phenotypes.
    min_support : int
        Minimum number of compounds with phenotype that must have a feature
        for the feature to appear in the 'top' table.
    qvalue_alpha : float
        BH FDR threshold used to define 'enriched' features.
    make_plots : bool
        If True, generate simple bar plots for top features per phenotype.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Reading phenotype table: %s", phenotype_tsv)
    ph = pd.read_csv(phenotype_tsv, sep="\t", dtype=str)
    if name_col not in ph.columns or phenotype_col not in ph.columns:
        raise ValueError(f"Phenotype TSV must contain '{name_col}' and '{phenotype_col}'.")

    ph_pairs = _explode_phenotypes(ph, name_col=name_col, phenotype_col=phenotype_col)
    # Normalise keys for joining
    ph_pairs["name_norm"] = ph_pairs["name"].map(_normalise_name)

    # Parse all compound feature sets
    sig_files = sorted(sig_dir.glob("*_vs_DMSO_significant_features.tsv"))
    logging.info("Found %d significant-feature TSVs.", len(sig_files))

    comp_to_feats: Dict[str, Set[str]] = {}
    comp_to_display: Dict[str, str] = {}

    for f in sig_files:
        comp_display = _parse_compound_from_filename(f)
        feats = _read_significant_features(f)
        comp_norm = _normalise_name(comp_display)
        comp_to_feats[comp_norm] = feats
        comp_to_display[comp_norm] = comp_display

    # Intersect compounds present in both phenotype mapping and TSVs
    comp_norm_in_both = sorted(set(comp_to_feats.keys()) & set(ph_pairs["name_norm"]))
    if not comp_norm_in_both:
        raise RuntimeError("No overlapping compounds between TSVs and phenotype table after normalisation.")

    logging.info("Overlapping compounds for analysis: %d", len(comp_norm_in_both))

    # Build compound × feature matrix
    all_features = sorted({f for c in comp_norm_in_both for f in comp_to_feats[c]})
    comp_feat = pd.DataFrame(0, index=comp_norm_in_both, columns=all_features, dtype=int)
    for c in comp_norm_in_both:
        comp_feat.loc[c, list(comp_to_feats[c])] = 1

    # Keep a display name index
    comp_feat.index = [comp_to_display[c] for c in comp_feat.index]
    comp_feat.index.name = "compound"

    # Write matrices (TSV)
    comp_feat_out = out_dir / "compound_feature_matrix.tsv"
    comp_feat.to_csv(comp_feat_out, sep="\t")
    (out_dir / "upset_compound_feature.tsv").write_text(comp_feat.to_csv(sep="\t"), encoding="utf-8")

    # Optional UpSet plot of phenotype–feature overlaps
    if make_upset:
        if not HAS_UPSETPLOT:
            logging.warning("upsetplot not available; install with 'pip install upsetplot'. Skipping UpSet.")
        else:
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            feat_by_ph = _prepare_features_by_phenotype(
                ph_feat=ph_feat,
                phenotypes_keep=phenotypes_keep,
                min_feature_phenotypes=int(min_feature_phenotypes),
            )

            # Always write the filtered matrix for reproducibility
            (plots_dir / "features_by_phenotype.filtered.tsv").write_text(
                feat_by_ph.astype(int).to_csv(sep="\t"), encoding="utf-8"
            )

            # Intersections table (useful alongside UpSet)
            _write_upset_intersections(
                feat_by_ph=feat_by_ph,
                out_tsv=plots_dir / "upset_intersections.tsv",
            )

            if feat_by_ph.empty:
                logging.info("UpSet skipped: no features passed filtering (min_feature_phenotypes=%d).",
                             int(min_feature_phenotypes))
            else:
                # Build indicators and plot
                indicators = from_indicators(
                    data=feat_by_ph.astype(bool),
                    data_name="features",
                )
                plt.figure(figsize=(12, 7))
                UpSet(
                    indicators,
                    subset_size="count",
                    show_counts=True,
                    sort_by="cardinality",
                    sort_categories_by="degree",
                    max_num=int(max_upset_sets),
                ).plot()
                plt.tight_layout()
                plt.savefig(plots_dir / "upset_phenotype_features.pdf")
                plt.close()



    # Phenotype assignments restricted to overlapping compounds
    ph_use = ph_pairs.loc[ph_pairs["name_norm"].isin(set(_normalise_name(x) for x in comp_feat.index))].copy()
    # Replace with display name casing
    norm_to_display = { _normalise_name(x): x for x in comp_feat.index }
    ph_use["compound"] = ph_use["name_norm"].map(norm_to_display)
    ph_use = ph_use[["compound", "phenotype"]].drop_duplicates().sort_values(["compound", "phenotype"])
    ph_use.to_csv(out_dir / "phenotype_assignment.tsv", sep="\t", index=False)

    # Phenotype × feature (present in ≥ 1 compound with that phenotype)
    phenos = sorted(ph_use["phenotype"].unique())
    ph_feat = pd.DataFrame(0, index=phenos, columns=all_features, dtype=int)
    for phen in phenos:
        comps = ph_use.loc[ph_use["phenotype"] == phen, "compound"].unique().tolist()
        if comps:
            ph_feat.loc[phen] = (comp_feat.loc[comps].sum(axis=0) > 0).astype(int)
    ph_feat.index.name = "phenotype"
    ph_feat.to_csv(out_dir / "upset_phenotype_feature.tsv", sep="\t")

    # Enrichment tests (one-sided Fisher, greater)
    rows = []
    comp_set_all = set(comp_feat.index)
    for phen in phenos:
        comp_with = set(ph_use.loc[ph_use["phenotype"] == phen, "compound"])
        if not comp_with:
            continue
        comp_without = comp_set_all - comp_with
        n_with = len(comp_with)
        n_without = len(comp_without)

        for feat in all_features:
            a = int(comp_feat.loc[list(comp_with), feat].sum())          # with phenotype & with feature
            b = n_with - a                                               # with phenotype & without feature
            c = int(comp_feat.loc[list(comp_without), feat].sum())       # without phenotype & with feature
            d = n_without - c                                            # without phenotype & without feature
            # To be robust if any side is empty
            if (a + b == 0) or (c + d == 0):
                pval = 1.0
                odds = np.nan
            else:
                odds, pval = fisher_exact([[a, b], [c, d]], alternative="greater")

            rows.append({
                "phenotype": phen,
                "feature": feat,
                "n_compounds_with_phenotype": n_with,
                "n_compounds_without_phenotype": n_without,
                "support_with_phenotype": a,
                "support_without_phenotype": c,
                "odds_ratio": odds,
                "p_value": pval,
            })

    enr = pd.DataFrame(rows)
    if enr.empty:
        raise RuntimeError("No tests performed; check inputs.")

    enr["q_value"] = _bh_fdr(enr["p_value"].to_numpy())
    enr = enr.sort_values(["q_value", "p_value", "phenotype", "feature"]).reset_index(drop=True)
    enr.to_csv(out_dir / "feature_enrichment_by_phenotype.tsv", sep="\t", index=False)

    # Top enriched per-phenotype
    top = enr.loc[
        (enr["q_value"] <= qvalue_alpha) &
        (enr["support_with_phenotype"] >= int(min_support))
    ].copy()
    top["frequency_in_phenotype"] = top["support_with_phenotype"] / top["n_compounds_with_phenotype"]
    top = top.sort_values(["phenotype", "q_value", "frequency_in_phenotype", "odds_ratio"], ascending=[True, True, False, False])
    top.to_csv(out_dir / "top_enriched_features_by_phenotype.tsv", sep="\t", index=False)

    logging.info("Saved outputs into: %s", out_dir)

    # Optional quick plots (bar charts), per phenotype
    if make_plots:
        if not HAS_MPL:
            logging.warning("matplotlib not available; skipping plots.")
            return

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Log if the strict 'top' table is empty
        if top.empty:
            logging.info(
                "No plots generated under current thresholds: "
                "top table empty (q_value ≤ %.3g and support ≥ %d).",
                qvalue_alpha, int(min_support)
            )

        # Decide what to plot: strict 'top' or fallback to best by p-value
        to_plot = {}
        if not top.empty:
            for phen in phenos:
                sub = top.loc[top["phenotype"] == phen].head(20)
                if not sub.empty:
                    to_plot[phen] = sub
        elif plot_all:
            logging.info("Using fallback plotting: top 20 by raw p-value per phenotype.")
            # build best-by-p (unfiltered except phenotype membership)
            for phen in phenos:
                sub = enr.loc[enr["phenotype"] == phen].sort_values("p_value").head(20)
                if not sub.empty:
                    to_plot[phen] = sub

        if not to_plot:
            logging.info("No per-phenotype plot candidates found.")
            return

        for phen, sub in to_plot.items():
            # Replace infinities for plotting
            sub = sub.copy()
            sub["odds_ratio"] = sub["odds_ratio"].replace([np.inf, -np.inf], np.nan)
            sub["odds_ratio"] = sub["odds_ratio"].fillna(sub["odds_ratio"].max())

            plt.figure(figsize=(10, 5))
            plt.barh(sub["feature"], sub["odds_ratio"])
            plt.xlabel("Odds ratio")
            plt.ylabel("Feature")
            plt.title(f"{phen} — enriched features")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            safe_phen = re.sub(r"[^A-Za-z0-9]+", "_", phen)
            plt.savefig(plots_dir / f"{safe_phen}_top_features.pdf", dpi=200)
            plt.close()



def main() -> None:
    """
    Parse command-line arguments and run the workflow.
    """
    parser = argparse.ArgumentParser(
        description="Enrichment of significant features by published sperm phenotypes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sig_dir", type=str, required=True, help="Directory with '*_vs_DMSO_significant_features.tsv'.")
    parser.add_argument("--phenotype_tsv", type=str, required=True, help="TSV with columns for compound name and phenotypes.")
    parser.add_argument("--name_col", type=str, default="name", help="Column holding compound names.")
    parser.add_argument("--phenotype_col", type=str, default="published_phenotypes", help="Column holding ';'-separated phenotypes.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <sig_dir>/../summary).")
    parser.add_argument("--min_support", type=int, default=2, help="Minimum #compounds-with-phenotype supporting a feature for 'top' table.")
    parser.add_argument("--qvalue_alpha", type=float, default=0.05, help="BH FDR threshold for calling enrichment.")
    parser.add_argument("--no_plots", action="store_true", help="If set, do not generate quick bar plots.")
    parser.add_argument("--plot_all", action="store_true",
                       help="If no features pass thresholds, plot top 20 by p-value per phenotype.")
    parser.add_argument("--make_upset", action="store_true",
                    help="Generate an UpSet plot from the phenotype×feature matrix.")
    parser.add_argument("--min_feature_phenotypes", type=int, default=2,
                        help="Keep features present in at least this many phenotypes for the UpSet.")
    parser.add_argument("--max_upset_sets", type=int, default=12,
                        help="Maximum number of phenotype sets shown in the UpSet intersections.")
    parser.add_argument("--phenotypes_keep", type=str, nargs="*", default=None,
                        help="Optional list of phenotype names to include in the UpSet (others dropped).")


    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")

    sig_dir = Path(args.sig_dir).resolve()
    phenotype_tsv = Path(args.phenotype_tsv).resolve()
    if args.out_dir is None:
        out_dir = (sig_dir / ".." / "summary").resolve()
    else:
        out_dir = Path(args.out_dir).resolve()

    run(
        sig_dir=sig_dir,
        phenotype_tsv=phenotype_tsv,
        out_dir=out_dir,
        name_col=args.name_col,
        phenotype_col=args.phenotype_col,
        min_support=int(args.min_support),
        qvalue_alpha=float(args.qvalue_alpha),
        make_plots=(not args.no_plots),
        plot_all=bool(args.plot_all),
        make_upset=bool(args.make_upset),                          
        min_feature_phenotypes=int(args.min_feature_phenotypes),   
        max_upset_sets=int(args.max_upset_sets),                   
        phenotypes_keep=args.phenotypes_keep,                      
    )


if __name__ == "__main__":
    main()
