"""
anomaly_detector.py — Classical and Neural Anomaly Detection Models
====================================================================

PURPOSE:
    Implement the multi-model anomaly-detection layer:
    1. IsolationForest   — classical multivariate unsupervised detection.
    2. SimpleAutoencoder — neural reconstruction-based detection.
    3. SafetyRuleChecker — domain-informed rule-based alerts.
    4. EnsembleScorer    — combines calibrated scores from all detectors.

WHY MULTIPLE MODELS:
    No single algorithm detects all types of anomalies equally well.
    • IsolationForest is good at multivariate outliers but ignores time.
    • The Autoencoder learns a compressed "normal" representation; anomalies
      create high reconstruction error.
    • Safety rules catch extreme values even when learned models lack
      sufficient training data.
    Combining complementary signals produces a more robust alert system.

SCORE DIRECTION CONVENTION:
    All detectors in this module are normalised so that:
        HIGH score  →  MORE anomalous
        LOW score   →  more normal
    This is necessary before combining scores; some algorithms (e.g.
    IsolationForest) natively output negative scores for anomalies.

INPUTS:
    Pre-scaled numpy arrays from AnomalyPreprocessor.transform().
    Raw DataFrame rows for safety-rule evaluation.

OUTPUTS:
    Per-sample anomaly scores (float arrays, range roughly 0–1 after
    calibration) and binary alert flags.

LIMITATIONS:
    • The Autoencoder requires enough normal training observations to learn
      a useful "normal" manifold.  With < ~500 rows the network may
      overfit or underfit.
    • IsolationForest contamination parameter defaults to 'auto' — this
      assumes roughly 10 % of training data is anomalous.  Adjust if
      domain knowledge suggests a different prevalence.
    • Safety rules are ILLUSTRATIVE engineering thresholds, not validated
      clinical standards.  See Section 18.4 of the project context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# 1. IsolationForestDetector
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """
    Isolation Forest anomaly detector.

    HOW ISOLATION FOREST WORKS:
        Random decision trees are built by repeatedly choosing a random
        feature and a random split value within its range.  Anomalies —
        observations that are very different from the majority — require
        fewer splits to isolate.  The anomaly score is the average path
        length across all trees: SHORT path = ANOMALOUS.

        Intuition: imagine sorting coloured marbles into bins at random.
        Normal marbles cluster together and are hard to separate; an oddly
        coloured marble ends up isolated quickly.

    SCORE CONVENTION:
        sklearn's decision_function() returns negative values for anomalies.
        We negate and min-max normalise to the [0, 1] range so that
        1.0 = most anomalous.

    Attributes
    ----------
    model_ : IsolationForest
        The fitted sklearn model.
    score_scaler_ : MinMaxScaler
        Fitted on training scores to normalise outputs to [0, 1].
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Number of isolation trees.  100 is a good default; larger values
            improve stability but increase training time linearly.
        contamination : float or 'auto'
            Expected proportion of anomalies in training data.  'auto' lets
            sklearn set it based on the original IsolationForest paper.
            If you have domain knowledge (e.g. 5 % anomaly rate), set 0.05.
        random_state : int
            Seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        self.model_: Optional[IsolationForest] = None
        self.score_scaler_: Optional[MinMaxScaler] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Train the IsolationForest on normal (training) observations.

        TRAINING PHILOSOPHY:
            Ideally, training data should contain predominantly normal
            (non-anomalous) observations.  In unsupervised settings we assume
            most observations are normal — a reasonable assumption for healthy
            astronaut data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,          # use all available CPU cores
        )
        self.model_.fit(X)

        # Compute training scores and fit a MinMaxScaler on them.
        # WHY: Different runs can produce decision_function values in
        # different ranges.  Normalising to [0, 1] makes scores comparable
        # across models and across time.
        raw_scores = -self.model_.decision_function(X)  # negate: high = anomalous
        self.score_scaler_ = MinMaxScaler()
        self.score_scaler_.fit(raw_scores.reshape(-1, 1))

        self.is_fitted_ = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return normalised anomaly scores in [0, 1].

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray, shape (n_samples,)
            0 = most normal, 1 = most anomalous.
        """
        self._check_fitted()
        raw = -self.model_.decision_function(X)
        # Clip to the range seen during training to avoid extrapolation artefacts
        scaled = self.score_scaler_.transform(raw.reshape(-1, 1)).flatten()
        return np.clip(scaled, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.6) -> np.ndarray:
        """Return 1 (anomaly) / 0 (normal) based on score threshold."""
        return (self.score_samples(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"IsolationForestDetector saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestDetector":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"IsolationForest artifact not found at: {path}")
        return joblib.load(path)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before scoring.")


# ---------------------------------------------------------------------------
# 2. SimpleAutoencoder
# ---------------------------------------------------------------------------

class SimpleAutoencoder:
    """
    Feed-forward autoencoder for reconstruction-based anomaly detection.

    HOW AN AUTOENCODER WORKS:
        An autoencoder is a neural network split into two parts:
        • Encoder: compresses the input into a small "latent" representation.
        • Decoder: reconstructs the original input from the latent vector.
        Trained on normal observations, the network learns to reconstruct
        normal patterns well.  When shown an anomalous input, the compressed
        representation cannot capture the unusual pattern, so the
        RECONSTRUCTION ERROR is high.

        Anomaly score = mean squared error between original and reconstruction.

    ARCHITECTURE (this dataset, ~64 features after preprocessing):
        Input (n_features)
            → Dense(32, ReLU)      ← encoder hidden layer
            → Dense(8, ReLU)       ← bottleneck (compressed representation)
            → Dense(32, ReLU)      ← decoder hidden layer
            → Dense(n_features, Linear)  ← reconstruction

    WHY THIS SIZE:
        The bottleneck dimension (8) is small enough to force the model to
        learn a compressed "normal" manifold but large enough to capture
        physiological patterns.  Too small = underfitting; too large = the
        model can memorise individual rows (no generalisation).

    LIMITATIONS:
        • Requires PyTorch or falls back to a numpy-based approximation
          if neither PyTorch nor TensorFlow is available.
        • GPU is used automatically if available; CPU is the fallback.
        • With ~11 k rows the model trains in seconds on CPU.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (32, 8),
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        hidden_dims : tuple of int
            Sizes of the encoder hidden layers (decoder mirrors these in
            reverse).  Default (32, 8) → bottleneck of size 8.
        epochs : int
            Training epochs.  50 is a good starting point for this dataset.
        batch_size : int
            Mini-batch size.  Larger = faster per-epoch but noisier gradients.
        learning_rate : float
            Adam optimizer step size.  1e-3 is the standard default.
        random_state : int
            Seed for weight initialisation reproducibility.
        """
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

        self._model = None  # filled by fit()
        self._train_errors = None  # distribution of training reconstruction errors
        self.score_scaler_: Optional[MinMaxScaler] = None
        self.is_fitted_: bool = False
        self._backend: str = "none"

    def fit(self, X: np.ndarray) -> "SimpleAutoencoder":
        """
        Train the autoencoder on representative (normal) observations.

        BACKEND DETECTION:
            We try PyTorch first (lighter weight for CPU).
            If unavailable we fall back to a sklearn PCA-based approximation
            (less powerful but dependency-free and always available).

        TRAINING PROCEDURE (PyTorch backend):
            1. Build encoder and decoder as sequential linear layers.
            2. Optimise with Adam (adaptive learning rate — robust default).
            3. Loss = Mean Squared Error between input and reconstruction.
            4. Train for `epochs` passes over the training data.
            5. Store training reconstruction errors for score normalisation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        np.random.seed(self.random_state)

        # --- Try PyTorch backend ---
        try:
            import torch
            self._fit_pytorch(X, torch)
            self._backend = "pytorch"
        except ImportError:
            # --- Fallback: PCA-based approximation ---
            print(
                "[Autoencoder] PyTorch not available. "
                "Using PCA-based reconstruction approximation instead.\n"
                "This is a valid fallback — PCA reconstruction error is "
                "a linear analogue of the autoencoder reconstruction error."
            )
            self._fit_pca(X)
            self._backend = "pca"

        self.is_fitted_ = True
        return self

    def _fit_pytorch(self, X: np.ndarray, torch) -> None:
        """PyTorch autoencoder training."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)

        n_features = X.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build encoder layers
        enc_layers = []
        prev_dim = n_features
        for dim in self.hidden_dims:
            enc_layers += [nn.Linear(prev_dim, dim), nn.ReLU()]
            prev_dim = dim

        # Build decoder layers (mirror of encoder, without final activation)
        dec_layers = []
        for dim in reversed(self.hidden_dims[:-1]):
            dec_layers += [nn.Linear(prev_dim, dim), nn.ReLU()]
            prev_dim = dim
        dec_layers.append(nn.Linear(prev_dim, n_features))

        self._model = nn.Sequential(*enc_layers, *dec_layers).to(device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate
        )
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(device)

        self._model.train()
        for epoch in range(self.epochs):
            # Mini-batch training
            idx = np.random.permutation(len(X))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(X), self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                batch = X_tensor[batch_idx]

                optimizer.zero_grad()
                reconstructed = self._model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Autoencoder epoch {epoch+1}/{self.epochs} "
                      f"| loss: {epoch_loss/n_batches:.6f}")

        # Store training reconstruction errors for normalisation
        self._model.eval()
        with torch.no_grad():
            X_recon = self._model(X_tensor).cpu().numpy()
        self._train_errors = np.mean((X - X_recon) ** 2, axis=1)
        self._device = device

    def _fit_pca(self, X: np.ndarray) -> None:
        """
        PCA-based reconstruction approximation (fallback when no PyTorch).

        ALGORITHM:
            PCA finds the directions of maximum variance in the data.
            We keep the top K components (K = bottleneck size).
            Projecting into K dimensions and back gives a reconstruction.
            Points that lie far from the K-dimensional hyperplane have high
            reconstruction error — analogous to autoencoder behaviour.
        """
        from sklearn.decomposition import PCA

        bottleneck = self.hidden_dims[-1]  # use bottleneck size as n_components
        n_components = min(bottleneck, X.shape[1], X.shape[0] - 1)
        self._model = PCA(n_components=n_components, random_state=self.random_state)
        self._model.fit(X)

        X_recon = self._model.inverse_transform(self._model.transform(X))
        self._train_errors = np.mean((X - X_recon) ** 2, axis=1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute normalised reconstruction-error anomaly scores.

        High score = large reconstruction error = more anomalous.

        Returns
        -------
        np.ndarray, shape (n_samples,), values in [0, 1] (approx.)
        """
        self._check_fitted()

        if self._backend == "pytorch":
            import torch
            self._model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self._device)
                X_recon = self._model(X_tensor).cpu().numpy()
        else:
            # PCA fallback
            X_recon = self._model.inverse_transform(self._model.transform(X))

        errors = np.mean((X - X_recon) ** 2, axis=1)

        # Normalise using training error statistics
        # We use the 99th percentile of training errors as the scale factor
        # so that 1.0 ≈ "as anomalous as the worst training sample".
        train_99 = np.percentile(self._train_errors, 99)
        if train_99 < 1e-10:
            return np.zeros(len(X))
        return np.clip(errors / train_99, 0.0, None)  # can exceed 1.0 for strong anomalies

    def predict(self, X: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """Return 1 (anomaly) / 0 (normal). Threshold is on normalised score."""
        return (self.score_samples(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"SimpleAutoencoder saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleAutoencoder":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Autoencoder artifact not found at: {path}")
        return joblib.load(path)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before scoring.")


# ---------------------------------------------------------------------------
# 3. SafetyRuleChecker
# ---------------------------------------------------------------------------

class SafetyRuleChecker:
    """
    Domain-informed physiological safety rule layer.

    PURPOSE:
        Provide a deterministic, explainable safety net that alerts on
        extreme measurements even when learned models have insufficient
        training data to recognise the pattern.

    DESIGN:
        Rules are defined as named, configurable dictionaries.
        Each rule specifies:
        • 'feature'       : column name in the raw DataFrame.
        • 'low' / 'high'  : threshold boundaries (values outside trigger alert).
        • 'severity'      : 'informational', 'review', or 'high_priority'.
        • 'description'   : human-readable explanation.
        • 'source'        : justification for the threshold.

    IMPORTANT DISCLAIMER:
        All thresholds below are ILLUSTRATIVE ENGINEERING RULES for prototype
        development.  They are NOT validated clinical thresholds or NASA
        astronaut health standards.  Any operational use requires review by
        qualified medical personnel.

    ASSUMPTIONS:
        • The raw DataFrame has the columns referenced by each rule.
        • Heart rate (HR) is in beats per minute.
        • RR intervals (MEAN_RR) are in milliseconds.
    """

    # Illustrative rules — configurable via the constructor
    DEFAULT_RULES: list[dict] = [
        {
            "name": "HR_extreme_high",
            "feature": "HR",
            "high": 130.0,
            "low": None,
            "severity": "high_priority",
            "description": "Heart rate exceeds 130 bpm — illustrative engineering rule",
            "source": "Project context Section 18.4 (engineering example, not clinical standard)",
        },
        {
            "name": "HR_extreme_low",
            "feature": "HR",
            "low": 35.0,
            "high": None,
            "severity": "high_priority",
            "description": "Heart rate below 35 bpm — extreme bradycardia range",
            "source": "Illustrative engineering rule — requires clinical validation",
        },
        {
            "name": "HR_elevated",
            "feature": "HR",
            "high": 100.0,
            "low": None,
            "severity": "review",
            "description": "Heart rate above 100 bpm — tachycardia range",
            "source": "Illustrative engineering rule — requires clinical validation",
        },
        {
            "name": "RMSSD_very_low",
            "feature": "RMSSD",
            "low": 5.0,
            "high": None,
            "severity": "review",
            "description": "RMSSD below 5 ms — very low HRV, possible autonomic stress",
            "source": "Illustrative engineering rule — RMSSD context-dependent",
        },
    ]

    def __init__(self, rules: Optional[list[dict]] = None):
        """
        Parameters
        ----------
        rules : list of rule dicts, optional
            Custom rule set.  If None, DEFAULT_RULES is used.
        """
        self.rules = rules if rules is not None else self.DEFAULT_RULES

    def check(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all rules against a DataFrame of raw (unscaled) observations.

        ALGORITHM:
            For each rule, for each row:
            1. Check if the relevant feature column exists.
            2. Check if the value violates the rule's threshold(s).
            3. Collect all triggered alerts as a DataFrame.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw (unscaled) DataFrame — safety rules use physiological units.

        Returns
        -------
        pd.DataFrame
            Rows = triggered alerts with columns:
            ['row_index', 'feature', 'value', 'rule_name',
             'severity', 'description']
        """
        alerts = []
        for rule in self.rules:
            col = rule["feature"]
            if col not in df_raw.columns:
                continue  # Skip rules for unavailable features

            values = df_raw[col]
            mask = pd.Series(False, index=df_raw.index)

            if rule.get("high") is not None:
                mask = mask | (values > rule["high"])
            if rule.get("low") is not None:
                mask = mask | (values < rule["low"])

            triggered_rows = df_raw[mask]
            for idx, row in triggered_rows.iterrows():
                alerts.append({
                    "row_index": idx,
                    "feature": col,
                    "value": round(float(row[col]), 4),
                    "rule_name": rule["name"],
                    "severity": rule["severity"],
                    "description": rule["description"],
                })

        return pd.DataFrame(alerts) if alerts else pd.DataFrame(
            columns=["row_index", "feature", "value", "rule_name",
                     "severity", "description"]
        )


# ---------------------------------------------------------------------------
# 4. EnsembleScorer
# ---------------------------------------------------------------------------

class EnsembleScorer:
    """
    Combine calibrated anomaly scores from multiple detectors.

    PURPOSE:
        Produce a single, interpretable composite anomaly score by
        averaging the calibrated outputs of the statistical baseline,
        IsolationForest, and Autoencoder.

    CALIBRATION STRATEGY:
        Before averaging, each score array is clipped to [0, 1] using
        MinMaxScaler fitted on the TRAINING scores.  This ensures that
        all detectors contribute equally to the composite score, regardless
        of their native output range.

    DESIGN DECISION — Why average (not max or weighted sum)?
        • Max selects the most alarmed detector; one bad detector inflates
          all scores.
        • Weighted sum requires choosing weights, which introduces bias.
        • Simple average provides a balanced, robust consensus signal when
          detectors have been individually calibrated.

    ALERT TIERS:
        composite_score >= 0.75 → high_priority
        composite_score >= 0.50 → review
        otherwise              → informational

    LIMITATIONS:
        If one detector is miscalibrated, the composite score is affected.
        Always inspect individual scores alongside the composite.
    """

    # Alert tier thresholds (configurable)
    HIGH_PRIORITY_THRESHOLD: float = 0.75
    REVIEW_THRESHOLD: float = 0.50

    def __init__(self, weights: Optional[dict[str, float]] = None):
        """
        Parameters
        ----------
        weights : dict, optional
            Mapping of detector name → weight.  If None, equal weights are used.
            Example: {'baseline': 1.0, 'iforest': 1.5, 'autoencoder': 1.0}
        """
        self.weights = weights  # None = equal weighting

    def combine(
        self,
        scores: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Combine detector scores into a composite score and alert tier.

        ALGORITHM:
            1. Clip each score array to [0, 1].
            2. If weights are specified, apply them before averaging.
            3. Compute the weighted (or simple) average.
            4. Assign alert tiers based on threshold comparison.

        Parameters
        ----------
        scores : dict mapping detector_name (str) → np.ndarray of scores
            All arrays must have the same length (one value per observation).
            Scores should already be in [0, 1] range.

        Returns
        -------
        pd.DataFrame with columns:
            ['baseline_score', 'iforest_score', 'autoencoder_score',
             'composite_score', 'alert_tier']
        """
        if not scores:
            raise ValueError("scores dict is empty — no detector outputs provided.")

        # Ensure all arrays have the same length
        lengths = {name: len(arr) for name, arr in scores.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Score arrays have different lengths: {lengths}")

        n = list(lengths.values())[0]

        # Clip all scores to [0, 1]
        clipped = {
            name: np.clip(arr, 0.0, 1.0) for name, arr in scores.items()
        }

        # Compute weighted average
        if self.weights:
            total_weight = sum(
                self.weights.get(name, 1.0) for name in clipped
            )
            composite = sum(
                clipped[name] * self.weights.get(name, 1.0)
                for name in clipped
            ) / total_weight
        else:
            # Simple average
            composite = np.mean(list(clipped.values()), axis=0)

        # Assign alert tiers
        alert_tiers = np.where(
            composite >= self.HIGH_PRIORITY_THRESHOLD, "high_priority",
            np.where(composite >= self.REVIEW_THRESHOLD, "review", "informational")
        )

        # Build output DataFrame
        result = pd.DataFrame(clipped)
        result.columns = [f"{name}_score" for name in clipped.keys()]
        result["composite_score"] = composite
        result["alert_tier"] = alert_tiers

        return result

    def build_alert_records(
        self,
        ensemble_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        feature_explanations: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Produce structured alert records ready for dashboard integration.

        Each record represents one observation and follows the output schema
        described in Section 13 of the project context.

        Parameters
        ----------
        ensemble_df : pd.DataFrame
            Output of combine().
        meta_df : pd.DataFrame
            Metadata columns (Time, subject id, condition, SSSQ) aligned
            row-for-row with ensemble_df.
        feature_explanations : pd.DataFrame, optional
            Top contributing features from RobustStatBaseline for each row.

        Returns
        -------
        pd.DataFrame of alert records.
        """
        records = ensemble_df.copy()

        # Attach metadata if available
        for col in meta_df.columns:
            records[col] = meta_df[col].values

        # Attach alert explanation
        records["explanation"] = records.apply(
            lambda row: (
                f"Composite anomaly score {row['composite_score']:.3f} — "
                f"alert tier: {row['alert_tier']}"
            ),
            axis=1,
        )

        return records
