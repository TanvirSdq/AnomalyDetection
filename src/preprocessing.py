"""
preprocessing.py — Reusable Preprocessing Pipeline
====================================================

PURPOSE:
    Transform the raw DataFrame into a clean, scaled numeric feature matrix
    suitable for anomaly-detection algorithms.

WHY A SEPARATE MODULE:
    Preprocessing steps (e.g. fitting a scaler) must be done on training data
    only and then applied identically to validation and test data.  Keeping
    them in one class ensures the fitted transformations are stored and
    reapplied correctly — avoiding the most common source of data leakage.

ALGORITHM OVERVIEW:
    1. Separate metadata columns (IDs, timestamps, labels) from features.
    2. Remove near-constant or redundant columns.
    3. Handle any missing or infinite values.
    4. Scale features using RobustScaler (preferred over StandardScaler for
       data with potential outliers — it uses the median and IQR rather than
       the mean and std, so extreme values don't distort the scaling).
    5. Return the scaled matrix together with the original metadata so that
       alerts can still reference subject, time, and condition.

INPUTS:
    Raw pandas DataFrame from data_loader.load_dataset().

OUTPUTS:
    • Scaled numpy array (X_scaled) for model input.
    • DataFrame of preserved metadata columns.
    • List of selected feature column names.
    • Fitted preprocessing state (for reuse at inference time).

ASSUMPTIONS:
    • The metadata columns are already known (defined in data_loader.py).
    • All remaining numeric columns are candidate features.

LIMITATIONS:
    • RobustScaler normalizes the distribution shape but does not guarantee
      Gaussian input.  Algorithms that assume normality (e.g. Mahalanobis
      distance) may need additional transforms.
    • Near-constant detection uses a fixed absolute std threshold; this may
      need adjustment for different datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Import metadata column definitions from our data_loader
from src.data_loader import METADATA_COLS, SUBJECT_COL, TIME_COL, CONDITION_COL

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Columns with standard deviation below this value are dropped.
# These carry negligible discriminative signal and can destabilize scalers.
NEAR_CONST_STD_THRESHOLD: float = 0.001

# Minimum number of training rows required before the pipeline will fit.
MIN_TRAIN_ROWS: int = 10


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------

class AnomalyPreprocessor:
    """
    Stateful preprocessing pipeline for the astronaut health monitoring system.

    DESIGN PATTERN:
        The fit() method learns from training data (computes medians, scales,
        etc.) and stores the results in instance attributes.
        The transform() method applies the stored transformations to any
        DataFrame — including validation, test, or live inference data.
        This pattern mirrors scikit-learn's Transformer API and prevents
        data leakage.

    Attributes
    ----------
    feature_cols_ : list[str]
        Column names selected as model features (set after fit).
    scaler_ : RobustScaler
        Fitted scaler (set after fit).
    dropped_cols_ : list[str]
        Columns dropped for being near-constant (set after fit).
    is_fitted_ : bool
        Whether fit() has been called successfully.
    """

    def __init__(
        self,
        extra_drop_cols: Optional[list[str]] = None,
        near_const_threshold: float = NEAR_CONST_STD_THRESHOLD,
    ):
        """
        Parameters
        ----------
        extra_drop_cols : list of str, optional
            Additional column names to exclude beyond the standard
            METADATA_COLS.  Use this to remove domain-redundant engineered
            features (e.g. LOG/BOXCOX/SQRT transforms of already-included
            features).
        near_const_threshold : float
            Columns with std below this value are dropped.
        """
        self.extra_drop_cols = extra_drop_cols or []
        self.near_const_threshold = near_const_threshold

        # These are set by fit()
        self.feature_cols_: list[str] = []
        self.scaler_: Optional[RobustScaler] = None
        self.dropped_cols_: list[str] = []
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------ #
    # fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> "AnomalyPreprocessor":
        """
        Learn preprocessing parameters from training data.

        WORKFLOW:
            1. Determine which columns are features (numeric, not metadata,
               not redundant).
            2. Drop near-constant columns.
            3. Replace any remaining infinities with NaN, then median-impute.
            4. Fit a RobustScaler on the training feature matrix.

        IMPORTANT — LEAKAGE RULE:
            Call fit() ONLY on training data.  Applying fit() to test data
            would allow information from the test set to influence the scaler,
            making the anomaly scores artificially look better.

        Parameters
        ----------
        df : pd.DataFrame
            Training observations (rows × columns).

        Returns
        -------
        self (for method chaining)
        """
        if len(df) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Training set has only {len(df)} rows. "
                f"At least {MIN_TRAIN_ROWS} rows are required."
            )

        # ---- Step 1: Identify candidate feature columns ----
        all_exclude = set(METADATA_COLS) | set(self.extra_drop_cols)
        candidate_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in all_exclude
        ]

        # ---- Step 2: Drop near-constant columns ----
        # Why: A column where every value is essentially the same adds no
        # discriminative information and can cause numerical instability in
        # some algorithms (e.g. division by near-zero variance).
        self.dropped_cols_ = [
            col for col in candidate_cols
            if df[col].std() < self.near_const_threshold
        ]
        self.feature_cols_ = [
            col for col in candidate_cols if col not in self.dropped_cols_
        ]

        if not self.feature_cols_:
            raise ValueError(
                "No feature columns remain after removing metadata and "
                "near-constant columns.  Check that the DataFrame contains "
                "numeric columns beyond the metadata set."
            )

        # ---- Step 3: Extract feature matrix and handle bad values ----
        X = self._extract_and_clean(df, is_fitting=True)

        # ---- Step 4: Fit the RobustScaler ----
        # RobustScaler uses the median (center) and IQR (scale) rather than
        # mean/std.  This means a handful of extreme readings will not
        # stretch the scale and make normal readings appear compressed.
        self.scaler_ = RobustScaler()
        self.scaler_.fit(X)

        # Store training-set column medians for later median imputation
        self._train_medians = pd.DataFrame(X, columns=self.feature_cols_).median()

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------ #
    # transform                                                           #
    # ------------------------------------------------------------------ #

    def transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Apply fitted transformations to a DataFrame.

        WORKFLOW:
            1. Extract the same feature columns selected during fit().
            2. Clean infinities and impute missing values using
               TRAINING medians (not test medians — leakage prevention).
            3. Apply the fitted RobustScaler.

        Parameters
        ----------
        df : pd.DataFrame
            Any DataFrame with the same columns as the training set.

        Returns
        -------
        X_scaled : np.ndarray, shape (n_rows, n_features)
            The scaled feature matrix ready for model input.
        meta_df : pd.DataFrame
            Metadata columns (Time, subject id, condition, SSSQ) plus the
            DataFrame index, so that scores can be joined back to original rows.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Preprocessor has not been fitted yet. Call fit() first."
            )

        # Check that all required columns are present
        missing_in_input = [
            col for col in self.feature_cols_ if col not in df.columns
        ]
        if missing_in_input:
            raise ValueError(
                f"Input DataFrame is missing feature columns that were "
                f"present during training: {missing_in_input}"
            )

        X = self._extract_and_clean(df, is_fitting=False)
        X_scaled = self.scaler_.transform(X)

        # Preserve available metadata columns
        available_meta = [c for c in METADATA_COLS if c in df.columns]
        meta_df = df[available_meta].reset_index(drop=True)

        return X_scaled, meta_df

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Convenience method: fit on df, then transform df."""
        self.fit(df)
        return self.transform(df)

    # ------------------------------------------------------------------ #
    # Internal helper                                                     #
    # ------------------------------------------------------------------ #

    def _extract_and_clean(
        self, df: pd.DataFrame, is_fitting: bool
    ) -> np.ndarray:
        """
        Extract feature columns and handle bad values.

        Bad values addressed:
        • np.inf / -np.inf — replaced with NaN.
        • NaN — filled with training-set column medians (set during fit).
          During fit itself, actual column medians of the training data are
          used before the stored medians exist.

        DESIGN DECISION — Use median imputation:
            The median is robust to the same extreme values that constitute
            anomalies.  Mean imputation would be pulled toward outliers,
            potentially normalizing or masking the very patterns we want to
            detect.

        WHY NOT DELETE ROWS WITH BAD VALUES:
            Per Section 8 of the project context, extreme observations must
            NOT be silently deleted — they may be the anomalies we are
            trying to detect.
        """
        X_raw = df[self.feature_cols_].copy()

        # Replace infinities with NaN so they can be consistently imputed
        X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)

        if is_fitting:
            # During fit: impute with the column medians of this training batch
            col_medians = X_raw.median()
            X_raw.fillna(col_medians, inplace=True)
        else:
            # During transform: use the medians computed during fit
            # This is the leakage-prevention step — test data is imputed with
            # training statistics, not test statistics.
            X_raw.fillna(self._train_medians, inplace=True)

        return X_raw.to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """
        Serialize the fitted preprocessor to disk using joblib.

        ALGORITHM:
            joblib.dump serializes the Python object (including all fitted
            attributes such as scaler_ and feature_cols_) to a binary file.
            This is more efficient than pickle for numpy arrays.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. 'artifacts/preprocessor.joblib').
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyPreprocessor":
        """
        Load a previously saved preprocessor from disk.

        Parameters
        ----------
        path : str or Path
            Path to the .joblib file produced by save().

        Returns
        -------
        AnomalyPreprocessor
            The loaded, fitted preprocessor.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found at: {path}"
            )
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is {type(obj)}, expected AnomalyPreprocessor."
            )
        return obj


# ---------------------------------------------------------------------------
# Utility: chronological train/test split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    time_col: str = TIME_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into training and test sets chronologically.

    WHY CHRONOLOGICAL (NOT RANDOM):
        In time-series data, a random split would allow the model to be
        trained on observations that occurred AFTER the test observations.
        This is data leakage: the model would have seen the "future" during
        training, making its test performance unrealistically optimistic.

        Chronological splitting ensures the training set always precedes the
        test set in time — matching the intended deployment scenario where
        the system learns from historical data and evaluates on new data.

    ALGORITHM:
        1. Sort the DataFrame by the time column.
        2. Take the first (train_fraction × n) rows as training.
        3. Take the remainder as the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset, which should be approximately time-ordered.
    train_fraction : float
        Proportion of rows to use for training (default 0.70 = 70 %).
    time_col : str
        Name of the timestamp column to sort by.

    Returns
    -------
    df_train : pd.DataFrame
    df_test  : pd.DataFrame
    """
    if not (0 < train_fraction < 1):
        raise ValueError(
            f"train_fraction must be between 0 and 1, got {train_fraction}."
        )

    # Sort by time to guarantee chronological order
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n_train = int(len(df_sorted) * train_fraction)

    df_train = df_sorted.iloc[:n_train].copy()
    df_test = df_sorted.iloc[n_train:].copy()

    return df_train, df_test
