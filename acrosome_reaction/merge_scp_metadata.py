#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone script for merging the three SCP stamped metadata files, harmonising
well identifiers and column names, and producing a single unified table.

This script:

    • Loads three stamping metadata files (CSV or TSV, any delimiter).
    • Normalises column names:
        OLPTID               -> plate_id
        PTODWELLREFERENCE    -> well_id
        OBJDID               -> cpd_id
        PTODCONCVALUE        -> conc
    • Normalises well identifiers (A001 -> A01).
    • Does NOT filter by plate barcode.
    • Outputs one combined tab-separated table.

This is intended for use before merging with CellProfiler Image data.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Well normalisation
# ---------------------------------------------------------------------------

def normalise_well_identifier(
    *,
    well: str,
) -> str:
    """
    Convert well identifiers into CellProfiler-style A01 format.

    Examples
    --------
    A001 -> A01
    A1   -> A01
    H12  -> H12

    Parameters
    ----------
    well : str
        Raw well identifier (any messy format).

    Returns
    -------
    str
        Normalised well identifier.
    """
    if not isinstance(well, str):
        return well

    s = well.strip().upper()
    if not s:
        return s

    row = s[0]
    digits = "".join(ch for ch in s[1:] if ch.isdigit())
    if digits == "":
        return s

    col = int(digits)
    return f"{row}{col:02d}"


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_single_metadata_file(*, path: Path) -> pd.DataFrame:
    """
    Load a single SCP / PTOD metadata file with robust encoding detection.

    Handles CSV/TSV saved in:
        • UTF-8
        • UTF-16 / UTF-16LE / UTF-16BE (common for robot exports)
        • Latin-1 (ISO-8859-1)
        • Unknown delimiter (auto-detected)

    Parameters
    ----------
    path : pathlib.Path
        File path.

    Returns
    -------
    pandas.DataFrame
        Loaded metadata table.
    """
    LOGGER.info("Loading metadata file: %s", path)

    # 1. Try utf-8 with auto delimiter
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except UnicodeDecodeError:
        LOGGER.warning("UTF-8 decode failed for %s; attempting fallback encodings.", path)

    # 2. Try UTF-16 variants
    for enc in ["utf-16", "utf-16le", "utf-16be"]:
        try:
            LOGGER.info("Trying encoding=%s for %s", enc, path)
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue

    # 3. Try Latin-1 (the 'accept anything' fallback)
    try:
        LOGGER.info("Trying encoding=latin-1 for %s", path)
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
    except Exception as exc:
        LOGGER.error("Failed to read %s with UTF-8, UTF-16, or Latin-1.", path)
        raise exc


def harmonise_metadata_columns(
    *,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Harmonise column names for SCP stamp metadata so downstream
    merges are consistent.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw metadata table.

    Returns
    -------
    pandas.DataFrame
        Metadata with:
            plate_id
            well_id
            cpd_id
            conc (optional)
            + all original columns.
    """
    # Standard renaming rules
    rename_map = {
        "OLPTID": "plate_id",
        "PTODWELLREFERENCE": "well_id",
        "OBJDID": "cpd_id",
        "Plate ID": "plate_id",
        "Well": "well_id",
        "Plate_Metadata": "plate_id",
        "Well_Metadata": "well_id",
    }
    df = df.rename(columns=rename_map)

    # Concentration variants
    for col in ["conc", "Concentration", "PTODCONCVALUE"]:
        if col in df.columns:
            df = df.rename(columns={col: "conc"})
            break

    # If well_id missing but Row/Col present
    if "well_id" not in df.columns:
        if {"Row", "Col"}.issubset(df.columns):
            df["well_id"] = [
                f"{str(r).strip().upper()}{int(c):02d}"
                for r, c in zip(df["Row"], df["Col"])
            ]
        else:
            raise KeyError(
                "Cannot determine well_id: need 'PTODWELLREFERENCE' or ('Row','Col')."
            )

    # Normalise well identifiers
    df["well_id"] = df["well_id"].map(
        lambda w: normalise_well_identifier(well=w)
    )

    # plate_id must exist
    if "plate_id" not in df.columns:
        raise KeyError(
            "Cannot determine plate_id: expected OLPTID, 'Plate ID', "
            "'Plate_Metadata', etc."
        )

    return df


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

def merge_three_metadata_files(
    *,
    files: List[Path],
) -> pd.DataFrame:
    """
    Load and merge three metadata files into one harmonised table.
    No filtering is performed on plate barcodes.

    Parameters
    ----------
    files : list of Path
        Paths to the three metadata files.

    Returns
    -------
    pandas.DataFrame
        Unified harmonised metadata table.
    """
    LOGGER.info("Loading and harmonising metadata files.")

    frames = []
    for path in files:
        df_raw = load_single_metadata_file(path=path)
        df_clean = harmonise_metadata_columns(df=df_raw)
        frames.append(df_clean)

    meta = pd.concat(frames, ignore_index=True)
    LOGGER.info("Unified metadata has %d rows, %d columns", *meta.shape)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Merge three SCP metadata files into a harmonised TSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--meta1", required=True, help="Metadata file 1.")
    parser.add_argument("--meta2", required=True, help="Metadata file 2.")
    parser.add_argument("--meta3", required=True, help="Metadata file 3.")
    parser.add_argument("--output", required=True, help="Output TSV file.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entry point for running the metadata merge tool.
    """
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    files = [Path(args.meta1), Path(args.meta2), Path(args.meta3)]

    merged = merge_three_metadata_files(files=files)

    out = Path(args.output)
    merged.to_csv(out, sep="\t", index=False)
    LOGGER.info("Wrote unified metadata to '%s'", out)


if __name__ == "__main__":
    main()

