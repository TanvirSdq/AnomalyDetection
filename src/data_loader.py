"""
data_loader.py — Dataset Loading and Initial Profiling
=======================================================

PURPOSE:
    Provide a single, reusable entry-point for loading the CSV dataset and
    generating a structured data-quality profile.

WHY A SEPARATE MODULE:
    Keeping I/O logic separate from modelling logic means the notebook can
    call load_dataset() once and then hand a clean DataFrame to every
    subsequent step.  It also makes it easy to swap the CSV for a different
    source later.

INPUTS:
    A file-system path to a UTF-8 CSV file.

OUTPUTS:
    A pandas DataFrame (all rows, original dtypes) plus a plain-dict
    profile summary that can be printed or logged.

ASSUMPTIONS:
    • The CSV has a header row on line 0.
    • All numeric values use '.' as the decimal separator.
    • The file fits comfortably in RAM (< a few hundred MB).

LIMITATIONS:
    • Does not stream very large files — use chunked_load() for that.
    • Does not parse date/time columns automatically; timestamps stay as
      numeric seconds (the Time column in this dataset).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public constants — column roles inferred from domain knowledge and schema
# ---------------------------------------------------------------------------

# These are metadata / identifier columns that should NOT be used as model
# features.  They are preserved in the DataFrame so that alerts can reference
# them, but excluded from numeric feature matrices.
METADATA_COLS: list[str] = ["Time", "subject id", "condition", "SSSQ"]

# The column that uniquely identifies individual subjects.
SUBJECT_COL: str = "subject id"

# The column that records elapsed time in seconds (not a wall-clock datetime).
TIME_COL: str = "Time"

# The column recording the experimental condition label.
CONDITION_COL: str = "condition"

# Subjective stress self-report score (integer 1–5 Likert-style scale).
SSSQ_COL: str = "SSSQ"


# ---------------------------------------------------------------------------
# Core loading function
# ---------------------------------------------------------------------------

def load_dataset(
    csv_path: str | Path,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the CSV dataset and return it as a DataFrame.

    ALGORITHM:
        1. Resolve the path and confirm the file exists.
        2. Read the CSV with pandas.  If max_rows is set, only that many rows
           are loaded (useful for fast iteration during development).
        3. Return the DataFrame with original dtypes.

    Parameters
    ----------
    csv_path : str or Path
        Absolute or project-relative path to the CSV file.
    max_rows : int, optional
        If provided, read only the first ``max_rows`` rows.  Set to None to
        load the full file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    ValueError
        If the CSV is empty or cannot be parsed.
    """
    csv_path = Path(csv_path)

    # ------------------------------------------------------------------ #
    # Step 1 — Existence check                                            #
    # ------------------------------------------------------------------ #
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Please verify the path and make sure the dummy_data/ folder "
            "is present relative to the project root."
        )

    # ------------------------------------------------------------------ #
    # Step 2 — Load CSV                                                   #
    # ------------------------------------------------------------------ #
    df = pd.read_csv(csv_path, nrows=max_rows)

    # ------------------------------------------------------------------ #
    # Step 3 — Sanity checks                                              #
    # ------------------------------------------------------------------ #
    if df.empty:
        raise ValueError(f"The CSV at {csv_path} loaded as an empty DataFrame.")

    if df.shape[1] < 2:
        raise ValueError(
            f"The CSV at {csv_path} has only {df.shape[1]} column(s). "
            "Expected a multi-column dataset."
        )

    return df


# ---------------------------------------------------------------------------
# Chunked loader (for very large files)
# ---------------------------------------------------------------------------

def chunked_load(
    csv_path: str | Path,
    chunk_size: int = 50_000,
    max_chunks: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a large CSV in chunks and concatenate the result.

    WHEN TO USE:
        When the file is too large to fit in RAM.  For the current
        test_sample.csv (~13 MB, ~12 K rows) the simple load_dataset() is
        more than adequate, so this function is provided as a precaution.

    Parameters
    ----------
    csv_path : str or Path
    chunk_size : int
        Number of rows per chunk.
    max_chunks : int, optional
        Stop after this many chunks (useful for quick sanity checks).

    Returns
    -------
    pd.DataFrame
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    chunks = []
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        chunks.append(chunk)
        if max_chunks is not None and i + 1 >= max_chunks:
            break

    if not chunks:
        raise ValueError("No data was read from the CSV.")

    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Data quality profile
# ---------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a structured data-quality profile from a loaded DataFrame.

    ALGORITHM:
        Iterates once over the columns to collect:
        • Shape, dtypes, missing value counts.
        • Numeric summary statistics.
        • Near-constant column detection (std < threshold).
        • Infinite value detection.
        • Duplicate row count.

    DESIGN DECISION:
        Returns a plain dict rather than printing directly so that the
        notebook can format it, save it, and reference individual fields
        programmatically.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile.

    Returns
    -------
    dict
        A nested dictionary with keys described below.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- Missing values ---
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    # --- Infinite values (numeric columns only) ---
    inf_counts = np.isinf(df[numeric_cols].replace([np.inf, -np.inf], np.nan)).sum()

    # --- Near-constant columns ---
    # A column whose standard deviation is less than NEAR_CONST_THRESHOLD
    # relative to its mean (or absolute) is flagged as near-constant.
    # These columns carry almost no discriminative information.
    NEAR_CONST_THRESHOLD = 0.001
    near_constant = [
        col for col in numeric_cols
        if df[col].std() < NEAR_CONST_THRESHOLD
    ]

    # --- Duplicate rows ---
    n_duplicate_rows = int(df.duplicated().sum())

    profile = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_counts": missing_counts.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "total_missing_cells": int(missing_counts.sum()),
        "inf_counts": inf_counts.to_dict(),
        "near_constant_columns": near_constant,
        "n_duplicate_rows": n_duplicate_rows,
        # Summary statistics for numeric columns (returns a nested dict)
        "numeric_summary": df[numeric_cols].describe().to_dict(),
        # Unique value counts for categorical columns
        "categorical_unique_counts": {
            col: int(df[col].nunique()) for col in categorical_cols
        },
        "categorical_value_counts": {
            col: df[col].value_counts().to_dict() for col in categorical_cols
        },
    }

    return profile


def print_profile(profile: dict) -> None:
    """
    Pretty-print the most important fields from profile_dataset().

    DESIGN DECISION:
        This function is for interactive notebook use only.  The profile dict
        is the authoritative source; this is just a convenience display.
    """
    print("=" * 60)
    print(f"DATASET PROFILE")
    print("=" * 60)
    print(f"  Rows      : {profile['n_rows']:,}")
    print(f"  Columns   : {profile['n_cols']}")
    print(f"  Numeric   : {len(profile['numeric_columns'])}")
    print(f"  Categorical: {len(profile['categorical_columns'])}")
    print()

    print(f"  Total missing cells : {profile['total_missing_cells']}")
    missing_nonzero = {
        k: v for k, v in profile["missing_counts"].items() if v > 0
    }
    if missing_nonzero:
        print("  Columns with missing values:")
        for col, cnt in missing_nonzero.items():
            pct = profile["missing_pct"][col]
            print(f"    {col}: {cnt} ({pct}%)")
    else:
        print("  No missing values detected.")
    print()

    inf_nonzero = {k: v for k, v in profile["inf_counts"].items() if v > 0}
    if inf_nonzero:
        print("  Columns with infinite values:", inf_nonzero)
    else:
        print("  No infinite values detected.")
    print()

    if profile["near_constant_columns"]:
        print("  Near-constant columns (low discriminative power):")
        for col in profile["near_constant_columns"]:
            print(f"    {col}")
    else:
        print("  No near-constant columns detected.")
    print()

    print(f"  Duplicate rows: {profile['n_duplicate_rows']}")
    print()

    print("  Categorical column value counts:")
    for col, vc in profile["categorical_value_counts"].items():
        print(f"    {col}: {vc}")
    print("=" * 60)
