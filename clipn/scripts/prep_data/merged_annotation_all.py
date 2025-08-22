#!/usr/bin/env python3
# coding: utf-8

"""
Merge multiple compound annotation files on Plate_Metadata, Well_Metadata, and cpd_id,
coalescing duplicate columns for `cpd_type` and `Library`.

This script reads one or more TSV/CSV metadata files, harmonises key columns
(Plate -> Plate_Metadata, Well -> Well_Metadata, library -> Library), and merges them
on the specified keys (default: Plate_Metadata, Well_Metadata, cpd_id). It coalesces
multiple sources of `cpd_type` and `Library` into single columns by taking the first
non-null value in file order, and logs conflicts.

Output is always tab-separated (.tsv), never comma-separated.

Typical usage example
---------------------
python merge_annotations.py \
    --files ./mitotox/mitotox.tsv \
            SelleckChem_10uM_22_07_2024/selechem_metadata.tsv \
            ./STB_final/STBV2_10uM_10032024.csv \
    --output merged_annotations.tsv
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Sequence, List

import pandas as pd


def robust_read(*, path: str) -> pd.DataFrame:
    """
    Read a delimited text file (TSV or CSV), inferring the delimiter from the header.

    Parameters
    ----------
    path : str
        Path to the input metadata file.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame with inferred delimiter.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    if "\t" in header:
        sep = "\t"
    elif "," in header:
        sep = ","
    else:
        sep = "\t"

    df = pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_values=["", "NA", "NaN"]
    )
    # Normalise whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    return df


def harmonise_cols(*, df: pd.DataFrame, fallback_name_to_cpd_id: bool = False) -> pd.DataFrame:
    """
    Harmonise standard metadata column names.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    fallback_name_to_cpd_id : bool, optional
        If True and `cpd_id` is missing, use `name` (if present) as `cpd_id`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with harmonised column names.
    """
    colmap = {}

    # Plate/Well standardisation
    if "Plate" in df.columns and "Plate_Metadata" not in df.columns:
        colmap["Plate"] = "Plate_Metadata"
    if "Well" in df.columns and "Well_Metadata" not in df.columns:
        colmap["Well"] = "Well_Metadata"

    # Library casing normalisation
    if "library" in {c.lower() for c in df.columns}:
        for c in list(df.columns):
            if c.lower() == "library" and c != "Library":
                colmap[c] = "Library"

    # cpd_type normalisation (just ensure consistent exact name)
    if "cpdtype" in {c.lower() for c in df.columns}:
        for c in list(df.columns):
            if c.lower() == "cpdtype" and c != "cpd_type":
                colmap[c] = "cpd_type"

    df = df.rename(columns=colmap)

    # Optional fallback for cpd_id
    if "cpd_id" not in df.columns and fallback_name_to_cpd_id and "name" in df.columns:
        df = df.rename(columns={"name": "cpd_id"})

    return df


def select_relevant_columns(
    *,
    df: pd.DataFrame,
    required_keys: Sequence[str],
    include_well: bool = True
) -> pd.DataFrame:
    """
    Select columns relevant for merging and commonly used annotations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    required_keys : Sequence[str]
        Columns required for merging (e.g., ["Plate_Metadata", "Well_Metadata", "cpd_id"]).
    include_well : bool, optional
        If True, keep `Well_Metadata` when present.

    Returns
    -------
    pandas.DataFrame
        Pruned DataFrame containing required keys, cpd_type, Library, and useful metadata.
    """
    keep = set(required_keys)
    if include_well and "Well_Metadata" in df.columns:
        keep.add("Well_Metadata")

    # Always try to keep these if present
    for target in ("cpd_type", "Library"):
        if target in df.columns:
            keep.add(target)

    # Commonly useful extras
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("cpd") or lc.startswith("plate") or lc in {"library"}:
            keep.add(c)

    pruned = df.loc[:, [c for c in df.columns if c in keep]].copy()
    pruned = pruned.loc[:, ~pruned.columns.duplicated()]
    return pruned


def coalesce_columns(*, df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Coalesce duplicate/suffixed versions of a column into a single target column.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged DataFrame potentially containing suffixed columns (e.g., Library, Library_2).
    target : str
        Base column name to coalesce (e.g., "Library", "cpd_type").

    Returns
    -------
    pandas.DataFrame
        DataFrame with a single `target` column and suffixed variants dropped.
    """
    # Gather exact and suffixed variants (from pandas merge suffixes)
    candidates = [c for c in df.columns if c == target or c.startswith(f"{target}_")]
    # Also sweep up accidental casing duplicates (e.g., 'library')
    candidates += [c for c in df.columns if c.lower() == target.lower() and c not in candidates]
    candidates = list(dict.fromkeys(candidates))  # de-dup, preserve order

    if not candidates:
        df[target] = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
        return df

    coalesced = df[candidates].bfill(axis=1).iloc[:, 0]

    # Best-effort conflict logging
    differing = []
    for idx, row in df[candidates].iterrows():
        vals = [v for v in row.tolist() if pd.notna(v) and v != ""]
        if len(set(vals)) > 1:
            differing.append(idx)
    if differing:
        logging.warning(
            "Detected %d rows with conflicting '%s' values across sources; kept the first non-null in file order.",
            len(differing),
            target,
        )

    df[target] = coalesced

    # Drop variants except the base target
    drop_cols = [c for c in candidates if c != target]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def coalesce_many(*, df: pd.DataFrame, targets: Sequence[str]) -> pd.DataFrame:
    """
    Coalesce several target columns in order.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame after merges.
    targets : Sequence[str]
        Column base names to coalesce (e.g., ["cpd_type", "Library"]).

    Returns
    -------
    pandas.DataFrame
        DataFrame with targets coalesced.
    """
    for t in targets:
        df = coalesce_columns(df=df, target=t)
    return df


def merge_files(
    *,
    files: Sequence[str],
    join_keys: Sequence[str],
    include_well: bool,
    how: str = "outer",
    fallback_name_to_cpd_id: bool = False
) -> pd.DataFrame:
    """
    Merge multiple metadata files on the specified join keys and coalesce target columns.

    This merges on the intersection of requested join keys present in each file,
    so plate/compound-level files without Well_Metadata still merge correctly
    (their annotations broadcast across wells on that plate/compound).

    Parameters
    ----------
    files : Sequence[str]
        List of file paths to merge.
    join_keys : Sequence[str]
        Desired keys (e.g., ["Plate_Metadata", "Well_Metadata", "cpd_id"]).
    include_well : bool
        If True, preserve `Well_Metadata` when present.
    how : str, optional
        Type of join to perform: 'outer', 'inner', 'left', or 'right'. Default is 'outer'.
    fallback_name_to_cpd_id : bool, optional
        If True, use `name` as `cpd_id` when `cpd_id` is absent.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with single `cpd_type` and `Library` columns.
    """
    merged: Optional[pd.DataFrame] = None

    for idx, path in enumerate(files, start=1):
        logging.info("Reading %s", path)
        df = robust_read(path=path)
        df = harmonise_cols(df=df, fallback_name_to_cpd_id=fallback_name_to_cpd_id)
        df = select_relevant_columns(df=df, required_keys=join_keys, include_well=include_well)

        if merged is None:
            merged = df
            continue

        # Determine intersection of join keys actually present in both frames
        left_keys = [k for k in join_keys if k in merged.columns]
        right_keys = [k for k in join_keys if k in df.columns]
        common_keys = [k for k in join_keys if k in left_keys and k in right_keys]

        if not common_keys:
            raise ValueError(
                f"No common join keys between accumulated data and '{path}'. "
                f"Requested keys: {list(join_keys)}; "
                f"left has: {left_keys}; right has: {right_keys}."
            )

        merged = pd.merge(
            left=merged,
            right=df,
            how=how,
            left_on=common_keys,
            right_on=common_keys,
            suffixes=("", f"_{idx}")
        )

        merged = merged.loc[:, ~merged.columns.duplicated()]

    if merged is None:
        raise ValueError("No input files were provided or all inputs were empty.")

    # Coalesce duplicate target columns
    merged = coalesce_many(df=merged, targets=["cpd_type", "Library"])

    # Drop exact duplicate rows if present
    merged = merged.drop_duplicates(ignore_index=True)

    # Reorder columns: keys first, then Library/cpd_type, then the rest
    key_set = list(dict.fromkeys([k for k in join_keys if k in merged.columns]))
    preferred = [c for c in ["Library", "cpd_type"] if c in merged.columns]
    others = [c for c in merged.columns if c not in key_set + preferred]
    merged = merged.loc[:, key_set + preferred + others]

    return merged


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge annotations on Plate_Metadata, Well_Metadata, and cpd_id; coalesce Library and cpd_type."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of input metadata files (TSV/CSV)."
    )
    parser.add_argument(
        "--output",
        required=False,
        default="merged_annotations.tsv",
        help="Path to the output TSV file."
    )
    parser.add_argument(
        "--join_keys",
        nargs="+",
        required=False,
        default=["Plate_Metadata", "Well_Metadata", "cpd_id"],
        help="Join keys (default: Plate_Metadata Well_Metadata cpd_id)."
    )
    parser.add_argument(
        "--include_well",
        action="store_true",
        help="If set, include Well_Metadata in outputs when present."
    )
    parser.add_argument(
        "--how",
        choices=["outer", "inner", "left", "right"],
        default="outer",
        help="Type of join to perform (default: outer)."
    )
    parser.add_argument(
        "--fallback_name_to_cpd_id",
        action="store_true",
        help="If set, use 'name' as 'cpd_id' when 'cpd_id' is absent."
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)."
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.

    Returns
    -------
    None
        Writes the merged TSV to the specified output path.
    """
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s"
    )

    merged = merge_files(
        files=args.files,
        join_keys=args.join_keys,
        include_well=args.include_well,
        how=args.how,
        fallback_name_to_cpd_id=args.fallback_name_to_cpd_id
    )

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not out_path.lower().endswith(".tsv"):
        logging.warning("Changing output extension to .tsv to enforce tab-separated format.")
        out_path = f"{out_path}.tsv"

    merged.to_csv(path_or_buf=out_path, sep="\t", index=False)
    print(f"Merged annotation file written: {out_path} (shape: {tuple(merged.shape)})")


if __name__ == "__main__":
    main()
