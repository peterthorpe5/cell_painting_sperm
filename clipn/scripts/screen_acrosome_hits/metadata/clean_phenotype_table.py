#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean and standardise a compound→phenotype TSV for downstream enrichment.

Overview
--------
- Reads a TSV with columns for compound name and published phenotypes.
- Splits multi-phenotype cells on ';'.
- Normalises spacing/case, fixes common typos/variants, and canonicalises labels.
- De-duplicates (compound, phenotype) pairs.
- Writes a cleaned table and small QC reports (value counts, duplicates).

All outputs are tab-separated.

Example
-------
python clean_phenotype_table.py \
    --in_tsv "/path/to/published_sperm_phenotypes.tsv" \
    --name_col "name" \
    --phenotype_col "published_phenotypes" \
    --out_clean "/path/to/published_sperm_phenotypes.cleaned.tsv" \
    --out_counts "/path/to/phenotype_value_counts.tsv" \
    --out_dupes "/path/to/name_dupes_report.tsv"

Notes
-----
- You can extend CANONICAL_MAP and CANONICAL_PHRASES below to suit your dataset.
- If a row has an empty phenotype, 'NA' is used.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# Common typo fixes / exact normalisations (match on lower-cased string)
CANONICAL_MAP: Dict[str, str] = {
    "spermicidial": "Spermicidal",
    "spermicidal": "Spermicidal",
    "immotile sperm": "Immotile sperm",
    "intereferes with ejaculation": "Interferes with ejaculation",
    "interferes with ejaculation": "Interferes with ejaculation",
    "interferes with spermatogenesis": "Interferes with spermatogenesis",
    "reducing sperm motility": "Reduces sperm motility",
    "reduction of sperm motility": "Reduces sperm motility",
    "enhances sperm motility": "Enhances sperm motility",
    "increases motility": "Enhances sperm motility",
    "increases hyperactivated sperm": "Enhances hyperactivation",
    "prevents hyperactivation": "Prevents hyperactivation",
    "blocks hyperactivation": "Blocks hyperactivation",
    "shape abnormality": "Shape abnormality",
    "only sperm with x chr": "Enriches X-bearing sperm",
    "causes azoospermia": "Azoospermia",
    "na": "NA",
}

# Phrase-level canonicalisation (regex on lower-cased phenotype)
CANONICAL_PHRASES: List[Tuple[re.Pattern, str]] = [
    # P4 / progesterone / Ca2+ responses
    (re.compile(r"\bblocks?\s+progesterone\s+induced\s+acrosome\s+reaction\b"), "Blocks P4-induced acrosome reaction"),
    (re.compile(r"\bblocks?\s+progesterone\s+induced\s*ca2\+\s*response\b"), "Blocks P4-evoked Ca2+ response"),
    (re.compile(r"\bp4\s*evoked\s*ca2\+\s*response\b"), "P4-evoked Ca2+ response"),
    (re.compile(r"\bevokes?\s*ca2\+\s*signaling?"), "Evokes Ca2+ signalling"),

    # ZP3 / CFTR / sAC etc.
    (re.compile(r"\bblocks?\s+rh?zp3\s+induced\s+acrosome\s+reaction\b"), "Blocks ZP3-induced acrosome reaction"),
    (re.compile(r"\bblocks?\s+sac\b"), "Blocks sAC"),
    (re.compile(r"\bctfrinh-?172\b"), "CFTRinh-172"),

    # General acrosome reaction
    (re.compile(r"\binduces?\s+acrosome\s+reaction\b"), "Induces acrosome reaction"),
    (re.compile(r"\bblocks?\s+acrosome\s+reaction\b"), "Blocks acrosome reaction"),

    # Liquefaction / capacitation
    (re.compile(r"\binterferes?\s+with\s+liquefaction\b"), "Interferes with liquefaction"),
    (re.compile(r"\bblocks?\s+capacitation\b"), "Blocks capacitation"),

    # Hyperactivation
    (re.compile(r"\binhibition\s+of\s+hyperactive\s+motility\b"), "Blocks hyperactivation"),
    (re.compile(r"\bprevents?\s+hyperactivation\b"), "Prevents hyperactivation"),

    # Motility catch-alls
    (re.compile(r"\breduces?\s+sperm\s+motility\b"), "Reduces sperm motility"),
    (re.compile(r"\benhances?\s+sperm\s+motility\b"), "Enhances sperm motility"),

    # Misc
    (re.compile(r"\bblocks?\s*ccl-?20\s+induced\s*ca2\+\s*response\b"), "Blocks CCL20-evoked Ca2+ response"),
    (re.compile(r"\bblocks?\s*onset\s+of\s+hyperactive\s+motility\b"), "Blocks hyperactivation"),
]


def normalise_name(*, x: str) -> str:
    """
    Normalise a compound name (case and spacing).

    Parameters
    ----------
    x : str
        Raw compound name.

    Returns
    -------
    str
        Cleaned compound name.
    """
    y = str(x).strip()
    y = re.sub(r"\s+", " ", y)
    return y


def canonicalise_phenotype(*, x: str) -> str:
    """
    Canonicalise a phenotype label using exact and regex rules.

    Parameters
    ----------
    x : str
        Raw phenotype.

    Returns
    -------
    str
        Canonical phenotype label.
    """
    s = str(x).strip()
    if not s:
        return "NA"

    s_l = s.lower()
    if s_l in CANONICAL_MAP:
        return CANONICAL_MAP[s_l]

    for pat, repl in CANONICAL_PHRASES:
        if pat.search(s_l):
            return repl

    # Title-case with British spelling preference where relevant
    s = re.sub(r"\s+", " ", s)
    return s[0].upper() + s[1:]


def split_multi(*, s: str, sep: str = ";") -> List[str]:
    """
    Split a multi-phenotype string on a separator.

    Parameters
    ----------
    s : str
        Input string possibly containing multiple phenotypes.
    sep : str
        Separator (default ';').

    Returns
    -------
    list[str]
        List of trimmed phenotype tokens (may be empty).
    """
    if s is None:
        return []
    toks = [t.strip() for t in str(s).split(sep)]
    return [t for t in toks if t]


def main() -> None:
    """
    CLI entry-point for phenotype table cleaning.
    """
    parser = argparse.ArgumentParser(
        description="Clean and standardise a compound→phenotype TSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in_tsv", type=str, required=True, help="Input TSV with compound→phenotype columns.")
    parser.add_argument("--name_col", type=str, default="name", help="Column for compound name.")
    parser.add_argument("--phenotype_col", type=str, default="published_phenotypes", help="Column for raw phenotype string(s).")
    parser.add_argument("--out_clean", type=str, required=True, help="Output cleaned TSV (compound, phenotype).")
    parser.add_argument("--out_counts", type=str, required=True, help="Output TSV of phenotype value counts.")
    parser.add_argument("--out_dupes", type=str, required=True, help="Output TSV of duplicate-name diagnostics.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    in_path = Path(args.in_tsv)
    df = pd.read_csv(in_path, sep="\t", dtype=str)

    if args.name_col not in df.columns or args.phenotype_col not in df.columns:
        raise SystemExit(f"Missing required columns: {args.name_col!r}, {args.phenotype_col!r}")

    logging.info("Loaded %d rows from %s", len(df), in_path)

    # Normalise names and explode phenotypes

    df["__name_clean__"] = df[args.name_col].apply(lambda s: normalise_name(x=s))

    df["__phenos_raw__"] = df[args.phenotype_col].fillna("NA")

    rows: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        name = r["__name_clean__"]
        phenos = split_multi(s=r["__phenos_raw__"], sep=";") or ["NA"]
        for p in phenos:
            rows.append((name, canonicalise_phenotype(x=p)))

    clean = pd.DataFrame(rows, columns=["compound", "phenotype"]).drop_duplicates().sort_values(["compound", "phenotype"])
    clean.to_csv(args.out_clean, sep="\t", index=False)
    logging.info("Wrote cleaned table → %s (%d unique pairs)", args.out_clean, len(clean))

    # Counts and duplicate diagnostics
    counts = clean["phenotype"].value_counts(dropna=False).rename_axis("phenotype").reset_index(name="n")
    counts.to_csv(args.out_counts, sep="\t", index=False)
    logging.info("Wrote phenotype value counts → %s", args.out_counts)

    dupes = (
        df.groupby("__name_clean__", as_index=False)
        .agg(n_rows=(args.name_col, "size"), distinct_raw_phenotype_strings=(args.phenotype_col, pd.Series.nunique))
        .rename(columns={"__name_clean__": "compound"})
        .sort_values(["n_rows", "distinct_raw_phenotype_strings"], ascending=False)
    )
    dupes.to_csv(args.out_dupes, sep="\t", index=False)
    logging.info("Wrote duplicate-name diagnostics → %s", args.out_dupes)


if __name__ == "__main__":
    main()
