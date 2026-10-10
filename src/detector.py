"""
detector.py — Multivariate Anomaly Detection & Explainability Engine
KUET_CHAYAPOTH | NASA Space Apps 2026
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.baseline import PersonalBaseline


class MultivariateAnomalyDetector:
    """
    Unsupervised multivariate anomaly detector using Isolation Forest,
    with directional physiological explainability.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42
    ):
        self.features = features or ['HR', 'RMSSD', 'SDRR', 'MEAN_RR']
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, condition_filter: Optional[str] = 'baseline') -> "MultivariateAnomalyDetector":
        """
        Fits the Isolation Forest strictly on baseline/nominal data.
        """
        train_df = df[df['condition'] == condition_filter] if condition_filter and 'condition' in df.columns else df
        X = train_df[self.features].dropna()
        self.model.fit(X)
        self.is_fitted = True
        return self

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """
        Computes anomaly score normalized to [0, 1] where:
        Higher value -> More anomalous
        Lower value  -> Nominal / Inlier
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        # decision_function returns negative for anomalies, positive for inliers
        raw_scores = self.model.decision_function(df[self.features])
        # Invert and normalize into roughly [0, 1]
        norm_scores = 1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-6)
        return norm_scores

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns binary labels: 1 for anomaly, 0 for normal.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        raw_pred = self.model.predict(df[self.features])
        return np.where(raw_pred == -1, 1, 0)

    def explain_sample(
        self,
        sample: pd.Series,
        baseline: PersonalBaseline,
        sig_threshold: float = 2.0
    ) -> Dict:
        """
        Decomposes an anomaly into directional physiological explanations ('Why Now?').
        """
        attributions = []
        state_type = "UNKNOWN"
        hr_up = False
        rmssd_down = False
        hr_down = False
        rmssd_up = False

        for col, metric in baseline.metrics.items():
            if col not in sample:
                continue
            val = float(sample[col])
            z_score = (val - metric.median) / (1.4826 * metric.mad + 1e-6)
            direction = "ELEVATED" if z_score > 0 else "DEPRESSED"

            if col == 'HR':
                if z_score > 1.5:
                    hr_up = True
                elif z_score < -1.5:
                    hr_down = True
            elif col in ['RMSSD', 'SDRR']:
                if z_score < -1.5:
                    rmssd_down = True
                elif z_score > 1.5:
                    rmssd_up = True

            if abs(z_score) >= sig_threshold:
                pct_diff = ((val - metric.median) / (metric.median + 1e-6)) * 100.0
                attributions.append({
                    "signal": col,
                    "value": round(val, 2),
                    "baseline_median": round(metric.median, 2),
                    "robust_z": round(z_score, 2),
                    "direction": direction,
                    "pct_change": f"{pct_diff:+.1f}%"
                })

        # Physiological State Classification
        if hr_up and rmssd_down:
            state_type = "ACUTE_PHYSIOLOGICAL_STRAIN"  # Sympathetic activation / cardiac deconditioning
        elif hr_down and rmssd_up:
            state_type = "PARASYMPATHETIC_RECOVERY"    # Meditation / Deep rest
        elif hr_up:
            state_type = "TACHYCARDIA_ACTIVE_LOAD"
        elif hr_down:
            state_type = "BRADYCARDIA_EXTREME_REST"
        else:
            state_type = "ATYPICAL_MULTI_SIGNAL_DEVIATION"

        return {
            "state_classification": state_type,
            "contributing_signals": attributions,
            "signal_count": len(attributions)
        }
