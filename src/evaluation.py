"""
evaluation.py — Quantitative Verification & Metrics Engine
KUET_CHAYAPOTH | NASA Space Apps 2026
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_detector(
    df: pd.DataFrame,
    raw_col: str = 'anomaly_raw',
    verified_col: str = 'verified_anomaly',
    ground_truth_condition: str = 'stress'
) -> pd.DataFrame:
    """
    Evaluates detector performance against WESAD's ground-truth condition.
    Compares raw point-in-time anomalies vs temporally verified anomalies.
    """
    # Binary ground truth: 1 if stress condition, 0 otherwise
    y_true = (df['condition'] == ground_truth_condition).astype(int).to_numpy()

    results = []

    for name, col in [("Raw Isolation Forest", raw_col), ("Temporally Verified", verified_col)]:
        y_pred = df[col].astype(int).to_numpy()

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # False alarm rate on baseline + meditation (relax)
        non_stress_mask = (y_true == 0)
        false_alarms = int(np.sum((y_pred == 1) & non_stress_mask))
        total_non_stress = int(np.sum(non_stress_mask))
        far = float(false_alarms / total_non_stress) if total_non_stress > 0 else 0.0

        results.append({
            "Pipeline Stage": name,
            "Precision (Stress)": f"{prec * 100:.1f}%",
            "Recall (Stress)": f"{rec * 100:.1f}%",
            "F1-Score": f"{f1:.3f}",
            "False Alarms": false_alarms,
            "False Alarm Rate": f"{far * 100:.1f}%"
        })

    return pd.DataFrame(results)
