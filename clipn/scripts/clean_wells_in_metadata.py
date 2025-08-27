#!/usr/bin/env python3
# coding: utf-8

"""
remove_dmso_when_paired.py

Remove DMSO rows only when a Plate+Well has exactly two entries and one of them
is DMSO while the other is a non-DMSO compound.

Behaviour:
    - Groups rows by Plate and Well.
    - If a group has exactly two rows and contains both:
        (a) a control row (e.g. DMSO), and
        (b) a non-control row,
      it drops the control row.
    - Otherwise, it leaves the group unchanged.

This is useful when plate maps contain paired entries for a well, where one row
is a vehicle control (DMSO) and the second row is the actual compound.

The script is conservative by default: it ONLY applies the rule to pairs of rows
(group size == 2). You may generalise the rule to ANY group with a mix of
control and non-control by passing --drop_when_any_non_control true.

Inputs and outputs are tab-separated to avoid comma-separated outfiles.

Usage example:
    python remove_dmso_when_paired.py \
        --input_path mitotox_plate_map.tsv \
        --output_path mitotox_plate_map.filtered.tsv \
        --plate_col Plate \
        --well_col Well \
        --compound_cols cpd_id COMPOUND_NUMBER \
        --control_names DMSO \
        --drop_when_any_non_control false

"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, List

import pandas as pd


def normalise_str(value: object) -> str:
    """
    Convert a value to a normalised string for comparison.

    Parameters
    ----------
    value : object
        Any value from a DataFrame cell.

    Returns
    -------
    str
        Trimmed, case-insensitive representation; empty string if value is null.
    """
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def row_is_control(row: pd.Series, *, compound_cols: List[str], control_names: List[str]) -> bool:
    """
    Determine whether a row represents a control (e.g. DMSO).

    Parameters
    ----------
    row : pandas.Series
        The DataFrame row to assess.
    compound_cols : list of str
        Column names to inspect in order of preference (e.g. ['cpd_id', 'COMPOUND']).
        The first non-empty value among these will be tested.
    control_names : list of str
        Accepted control names (case-insensitive), e.g. ['DMSO'].

    Returns
    -------
    bool
        True if the row is identified as a control; otherwise False.
    """
    control_set = {normalise_str(c) for c in control_names}
    for col in compound_cols:
        if col in row.index:
            token = normalise_str(row[col])
            if token:
                return token in control_set
    return False


def remove_controls_in_pairs(
    df: pd.DataFrame,
    *,
    plate_col: str,
    well_col: str,
    compound_cols: List[str],
    control_names: List[str],
    drop_when_any_non_control: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Remove control rows in Plate+Well groups according to the specified rule.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    plate_col : str
        Column name for plate identifier (e.g. 'Plate').
    well_col : str
        Column name for well identifier (e.g. 'Well').
    compound_cols : list of str
        Candidate columns that contain compound identifiers; checked in order.
    control_names : list of str
        List of control names (e.g. ['DMSO']), case-insensitive.
    drop_when_any_non_control : bool, optional
        If False (default), drop control rows ONLY when a Plate+Well group has
        exactly two rows and is a control/non-control pair.
        If True, drop control rows whenever a Plate+Well group contains at least
        one control and at least one non-control, regardless of group size.
    logger : logging.Logger or None, optional
        Logger for progress messages.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with the requested control rows removed.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    required = [plate_col, well_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    missing = [c for c in compound_cols if c not in df.columns]
    if len(missing) == len(compound_cols):
        raise ValueError(
            f"None of the compound columns were found: {compound_cols}. "
            "Please provide at least one existing column via --compound_cols."
        )

    # Determine control status per row
    is_ctrl = df.apply(
        lambda r: row_is_control(r, compound_cols=compound_cols, control_names=control_names),
        axis=1,
    )
    df = df.copy()
    df["_is_control"] = is_ctrl

    # Group and decide which to drop
    to_drop_idx = []

    grouped = df.groupby([plate_col, well_col], sort=False, dropna=False)
    for (plate, well), sub in grouped:
        n = len(sub)
        ctrl_count = int(sub["_is_control"].sum())
        non_ctrl_count = n - ctrl_count

        if ctrl_count == 0 or non_ctrl_count == 0:
            # All control or all non-control: leave unchanged
            continue

        if drop_when_any_non_control:
            # Drop all control rows if any non-control rows exist in the group
            to_drop_idx.extend(sub.loc[sub["_is_control"]].index.tolist())
        else:
            # Strict pair rule: only when the group has exactly two rows,
            # and they are split between control and non-control
            if n == 2 and ctrl_count == 1 and non_ctrl_count == 1:
                to_drop_idx.extend(sub.loc[sub["_is_control"]].index.tolist())

    removed = len(to_drop_idx)
    if removed:
        logger.info("Removing %d control rows under the specified rule.", removed)
    else:
        logger.info("No rows met the removal criteria.")

    out = df.drop(index=to_drop_idx).drop(columns=["_is_control"])
    # Preserve original order
    return out.sort_index(kind="stable")


def read_table(path: str, *, sep: str = "\t") -> pd.DataFrame:
    """
    Read a delimited text file into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the input file.
    sep : str, optional
        Field delimiter. Default is tab ('\\t').

    Returns
    -------
    pandas.DataFrame
        Parsed table.
    """
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, na_values=[""])


def write_table(df: pd.DataFrame, path: str, *, sep: str = "\t") -> None:
    """
    Write a DataFrame to a delimited text file.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to write.
    path : str
        Output file path. Compression inferred from file suffix (e.g. '.gz').
    sep : str, optional
        Field delimiter. Default is tab ('\\t').
    """
    df.to_csv(path, sep=sep, index=False)


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with named options only.
    """
    p = argparse.ArgumentParser(
        description=(
            "Remove DMSO rows only when a Plate+Well has exactly two entries "
            "and one is DMSO while the other is not."
        )
    )
    p.add_argument("--input_path", required=True, help="Path to the input TSV.")
    p.add_argument("--output_path", required=True, help="Path to the output TSV (tab-separated).")
    p.add_argument(
        "--plate_col",
        required=False,
        default="Plate",
        help="Column name for plate identifier. Default: Plate",
    )
    p.add_argument(
        "--well_col",
        required=False,
        default="Well",
        help="Column name for well identifier. Default: Well",
    )
    p.add_argument(
        "--compound_cols",
        nargs="+",
        required=False,
        default=["cpd_id", "COMPOUND", "COMPOUND_NUMBER"],
        help=(
            "One or more columns to check for the compound name/ID, in order "
            "of preference. Default: cpd_id COMPOUND COMPOUND_NUMBER"
        ),
    )
    p.add_argument(
        "--control_names",
        nargs="+",
        required=False,
        default=["DMSO"],
        help="Control names to treat as vehicle controls (case-insensitive). Default: DMSO",
    )
    p.add_argument(
        "--drop_when_any_non_control",
        choices=["true", "false"],
        default="false",
        help=(
            "If 'true', drop control rows whenever a group has both control and non-control "
            "rows (regardless of size). If 'false', apply the strict pair rule only. "
            "Default: false"
        ),
    )
    p.add_argument(
        "--sep",
        default="\t",
        help="Input and output delimiter. Default: tab",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    return p


def main() -> None:
    """
    Entry point for command-line execution.
    """
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )
    logger = logging.getLogger("remove_dmso_when_paired")

    df = read_table(args.input_path, sep=args.sep)
    logger.info("Loaded input: %d rows, %d columns.", df.shape[0], df.shape[1])

    out = remove_controls_in_pairs(
        df,
        plate_col=args.plate_col,
        well_col=args.well_col,
        compound_cols=args.compound_cols,
        control_names=args.control_names,
        drop_when_any_non_control=(args.drop_when_any_non_control.lower() == "true"),
        logger=logger,
    )

    write_table(out, args.output_path, sep=args.sep)
    logger.info("Wrote output: %s (%d rows).", args.output_path, out.shape[0])


if __name__ == "__main__":
    main()
