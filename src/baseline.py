"""
baseline.py — Personalized Statistical Anomaly Baseline
=========================================================

PURPOSE:
    Provide a transparent, interpretable anomaly-detection baseline using
    robust statistics (median + MAD) computed on the training data.

WHY A BASELINE FIRST:
    Advanced models are only valuable if they outperform a simpler reference.
    A well-implemented statistical baseline:
    • Is easy to explain and debug.
    • Provides per-feature anomaly signals (not just a black-box score).
    • Serves as a sanity check: if the baseline misses obvious anomalies,
      the data or preprocessing may be wrong.
    • Sets a performance floor against which IsolationForest and the
      Autoencoder are compared.

ALGORITHM — Robust Z-Score (MAD-based):
    Given a feature column x:
        median_x = median(x_train)
        MAD_x    = median(|x_train − median_x|)
        robust_z = |x − median_x| / (1.4826 × MAD_x)
    The constant 1.4826 makes MAD comparable to the standard deviation
    under a Gaussian distribution.  A robust_z > threshold (e.g. 3.5) flags
    an observation as a per-feature outlier.

    If MAD_x ≈ 0 (constant-valued feature), we fall back to a small epsilon
    to avoid division by zero.

    The overall sample anomaly score is the MAX robust Z across all features.
    This captures "the worst single-feature deviation" and is intuitive to
    a human reviewer.

INPUTS:
    Scaled numpy feature matrix from AnomalyPreprocessor.transform().
    (Using the pre-scaled matrix ensures comparability across features.)

OUTPUTS:
    Per-sample anomaly scores (array of floats) and binary alert flags.

ASSUMPTIONS:
    • The feature matrix has already been cleaned (no NaN or inf).
    • Each row represents one observation.

LIMITATIONS:
    • Treats features independently — does not detect unusual COMBINATIONS
      of otherwise normal-looking feature values (multivariate anomalies).
      IsolationForest addresses this limitation.
    • MAD can be zero for near-constant features; epsilon prevents division
      by zero but the score becomes unreliable for such features.
    • Without a subject identifier, a single global baseline is used.
      This may flag activity-related changes (stress vs. rest) as anomalous
      depending on how the training/test split is defined.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# The robust Z-score threshold above which a feature is considered anomalous.
# Literature commonly uses 3.5 for MAD-based outlier detection.
DEFAULT_THRESHOLD: float = 3.5

# Small constant to prevent division by zero when MAD ≈ 0.
MAD_EPSILON: float = 1e-6

# MAD normalisation constant (makes MAD ≈ std under Gaussian).
MAD_SCALE: float = 1.4826


# ---------------------------------------------------------------------------
# RobustStatBaseline class
# ---------------------------------------------------------------------------

class RobustStatBaseline:
    """
    Feature-wise robust statistical anomaly baseline.

    Attributes
    ----------
    medians_ : np.ndarray, shape (n_features,)
        Median of each feature, computed on training data.
    mads_ : np.ndarray, shape (n_features,)
        Median Absolute Deviation of each feature, with epsilon floor.
    feature_names_ : list[str]
        Names of the features in the order they appear in X.
    threshold_ : float
        Robust Z-score threshold for flagging anomalies.
    is_fitted_ : bool
        Whether fit() has been called.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        """
        Parameters
        ----------
        threshold : float
            Robust Z-score above which a sample is flagged.
            A value of 3.5 corresponds roughly to 3.5 standard deviations
            from the median under Gaussian assumptions — capturing the most
            extreme ~0.05 % of normal observations.
        """
        self.threshold = threshold

        # Set after fit()
        self.medians_: Optional[np.ndarray] = None
        self.mads_: Optional[np.ndarray] = None
        self.feature_names_: list[str] = []
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------ #
    # fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(
        self, X: np.ndarray, feature_names: Optional[list[str]] = None
    ) -> "RobustStatBaseline":
        """
        Compute per-feature medians and MADs from training data.

        ALGORITHM:
            For each feature column j in the training matrix X:
                1. Compute median_j = median(X[:, j])
                2. Compute MAD_j   = median(|X[:, j] − median_j|)
                3. Apply epsilon floor: MAD_j = max(MAD_j, MAD_EPSILON)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training feature matrix (already scaled by AnomalyPreprocessor).
        feature_names : list of str, optional
            Names for each feature column (for interpretable output).

        Returns
        -------
        self
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if len(X) < 2:
            raise ValueError("At least 2 training samples are required.")

        n_features = X.shape[1]

        # Step 1: Compute medians
        self.medians_ = np.median(X, axis=0)  # shape (n_features,)

        # Step 2: Compute MAD for each feature
        # |x - median| for every observation, then take the median of those
        abs_deviations = np.abs(X - self.medians_[np.newaxis, :])
        self.mads_ = np.median(abs_deviations, axis=0)  # shape (n_features,)

        # Step 3: Apply epsilon floor to avoid division by zero
        self.mads_ = np.maximum(self.mads_, MAD_EPSILON)

        # Store feature names (use indices if not provided)
        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError(
                    f"feature_names has {len(feature_names)} entries, "
                    f"but X has {n_features} columns."
                )
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------ #
    # score_samples                                                       #
    # ------------------------------------------------------------------ #

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the per-sample anomaly score.

        ALGORITHM:
            For each sample (row) i and feature j:
                robust_z[i, j] = |X[i, j] − median_j| / (MAD_SCALE × MAD_j)
            sample_score[i] = max(robust_z[i, :])

            Interpretation:
            • A score of 1.0 means the worst-feature deviation equals 1 MAD.
            • A score of 3.5 means it is 3.5 MADs from the median —
              the chosen anomaly threshold.
            • Higher scores indicate stronger anomalies.

        DESIGN CHOICE — max vs. mean aggregation:
            Using the maximum single-feature Z means an observation is flagged
            if ANY one feature is extreme, even if all others are normal.
            A mean-based aggregate would dilute a single strong signal.
            The maximum is more appropriate for health monitoring where a
            single abnormal reading warrants attention.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix (same preprocessing as training).

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Anomaly score per sample.  Higher = more anomalous.
        """
        self._check_fitted()
        if X.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"X has {X.shape[1]} features; expected {len(self.feature_names_)}."
            )

        # Compute robust Z-score for every (sample, feature) cell
        robust_z = np.abs(X - self.medians_[np.newaxis, :]) / (
            MAD_SCALE * self.mads_[np.newaxis, :]
        )

        # The overall score is the worst single-feature deviation
        return robust_z.max(axis=1)

    # ------------------------------------------------------------------ #
    # feature_scores                                                      #
    # ------------------------------------------------------------------ #

    def feature_scores(self, X: np.ndarray) -> pd.DataFrame:
        """
        Return a DataFrame of robust Z-scores for every (sample, feature).

        PURPOSE:
            This is the primary interpretability tool.  When a sample is
            flagged as anomalous, this function reveals WHICH features drove
            the score, and by how much.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        pd.DataFrame, shape (n_samples, n_features)
            Rows = observations, columns = features, values = robust Z-scores.
        """
        self._check_fitted()
        robust_z = np.abs(X - self.medians_[np.newaxis, :]) / (
            MAD_SCALE * self.mads_[np.newaxis, :]
        )
        return pd.DataFrame(robust_z, columns=self.feature_names_)

    # ------------------------------------------------------------------ #
    # predict                                                             #
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary anomaly labels: 1 = anomaly, 0 = normal.

        THRESHOLD:
            A sample is anomalous if its score >= self.threshold.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        labels : np.ndarray of int, shape (n_samples,)
        """
        scores = self.score_samples(X)
        return (scores >= self.threshold).astype(int)

    # ------------------------------------------------------------------ #
    # top_contributing_features                                           #
    # ------------------------------------------------------------------ #

    def top_contributing_features(
        self, X: np.ndarray, n_top: int = 5
    ) -> pd.DataFrame:
        """
        For each sample, identify the top-N features with the highest
        robust Z-score.

        PURPOSE:
            When generating an alert, this function provides a concise
            explanation of what is unusual about a specific observation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        n_top : int
            Number of top features to return per sample.

        Returns
        -------
        pd.DataFrame with columns ['sample_idx', 'rank', 'feature', 'score']
        """
        feat_scores = self.feature_scores(X)
        records = []
        for i, row in feat_scores.iterrows():
            top = row.nlargest(n_top)
            for rank, (feat, score) in enumerate(top.items(), start=1):
                records.append({
                    "sample_idx": i,
                    "rank": rank,
                    "feature": feat,
                    "score": round(float(score), 4),
                })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Save the fitted baseline to disk."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted baseline.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"RobustStatBaseline saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "RobustStatBaseline":
        """Load a previously saved baseline from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline artifact not found at: {path}")
        return joblib.load(path)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "RobustStatBaseline has not been fitted. Call fit() first."
            )
