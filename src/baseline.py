"""
baseline.py — Personalized Digital Baseline Engine
KUET_CHAYAPOTH | NASA Space Apps 2026
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class BaselineMetric:
    feature: str
    median: float
    mad: float
    q25: float
    q75: float


class PersonalBaseline:
    """
    Computes and maintains an astronaut's personalized physiological baseline
    using robust non-parametric statistics (Median & Median Absolute Deviation).
    """

    def __init__(self, features: Optional[List[str]] = None, eps: float = 1e-6):
        self.features = features or ['HR', 'RMSSD', 'SDRR', 'MEAN_RR']
        self.eps = eps
        self.metrics: Dict[str, BaselineMetric] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, condition_filter: Optional[str] = 'baseline') -> "PersonalBaseline":
        """
        Fits personal baseline statistics on nominal/resting data.
        """
        ref_df = df[df['condition'] == condition_filter] if condition_filter and 'condition' in df.columns else df

        if len(ref_df) == 0:
            raise ValueError(f"No records found for condition: {condition_filter}")

        self.metrics = {}
        for col in self.features:
            if col not in ref_df.columns:
                continue
            series = ref_df[col].dropna()
            med = float(series.median())
            mad = float((series - med).abs().median())
            # Guard against zero MAD for discrete or constant sensors
            if mad == 0.0:
                mad = float(series.std()) if series.std() > 0 else self.eps

            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))

            self.metrics[col] = BaselineMetric(
                feature=col,
                median=med,
                mad=mad,
                q25=q25,
                q75=q75
            )

        self.is_fitted = True
        return self

    def robust_z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates directional robust z-scores (number of MADs from median) for each feature.
        Positive: value > baseline median
        Negative: value < baseline median
        """
        if not self.is_fitted:
            raise RuntimeError("Baseline has not been fitted yet.")

        z_df = pd.DataFrame(index=df.index)
        for col, metric in self.metrics.items():
            if col in df.columns:
                z_df[f"{col}_zscore"] = (df[col] - metric.median) / (1.4826 * metric.mad + self.eps)
        return z_df

    def get_summary(self) -> pd.DataFrame:
        """Returns baseline reference table."""
        if not self.is_fitted:
            raise RuntimeError("Baseline has not been fitted yet.")
        records = [
            {
                "Feature": m.feature,
                "Median": m.median,
                "MAD": m.mad,
                "IQR (Q25-Q75)": f"[{m.q25:.2f}, {m.q75:.2f}]"
            }
            for m in self.metrics.values()
        ]
        return pd.DataFrame(records)
