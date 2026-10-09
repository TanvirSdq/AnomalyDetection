"""
evaluation.py — Evaluation Utilities and Synthetic Anomaly Tests
================================================================

PURPOSE:
    Provide reusable evaluation functions that work with or without labeled
    ground truth:
    1. Distribution-based score analysis (no labels required).
    2. Synthetic anomaly injection and detection testing.
    3. Consistency and stability checks.

WHY SYNTHETIC ANOMALY TESTING:
    The dataset does not contain verified labeled anomalies.  We cannot
    compute true precision/recall without ground truth labels.
    Instead, we INJECT known anomalies (by modifying feature values far
    outside the normal range) and check whether each detector flags them.
    This validates that the pipeline logic is correct even without real
    anomaly labels.

LIMITATIONS:
    Synthetic anomaly tests are ENGINEERING VALIDATION, not clinical
    accuracy proof.  A detector that finds injected anomalies may still
    miss subtle real anomalies that differ from the injection pattern.

INPUTS:
    Score arrays from detectors, raw DataFrames for injection.

OUTPUTS:
    Structured evaluation summaries (dicts and DataFrames).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Score distribution analysis (no labels needed)
# ---------------------------------------------------------------------------

def analyse_score_distribution(
    scores: np.ndarray,
    detector_name: str = "detector",
    alert_threshold: Optional[float] = None,
) -> dict:
    """
    Summarise the distribution of anomaly scores from a single detector.

    WHY:
        Without ground-truth labels we cannot compute precision or recall.
        Instead we report the distribution shape, which reveals whether the
        detector is producing sensible spread vs. degenerate all-zero or
        all-one outputs.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples,)
    detector_name : str
    alert_threshold : float, optional
        If provided, also compute the alert rate (fraction above threshold).

    Returns
    -------
    dict with summary statistics.
    """
    summary = {
        "detector": detector_name,
        "n_samples": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "max": float(np.max(scores)),
        "n_finite": int(np.isfinite(scores).sum()),
        "n_nan": int(np.isnan(scores).sum()),
    }
    if alert_threshold is not None:
        n_alerts = int((scores >= alert_threshold).sum())
        summary["alert_threshold"] = float(alert_threshold)
        summary["n_alerts"] = n_alerts
        summary["alert_rate_pct"] = round(n_alerts / len(scores) * 100, 2)

    return summary


def print_score_summary(summary: dict) -> None:
    """Pretty-print a score distribution summary dict."""
    print(f"\n--- Score Distribution: {summary['detector']} ---")
    print(f"  N samples  : {summary['n_samples']:,}")
    print(f"  Mean ± Std : {summary['mean']:.4f} ± {summary['std']:.4f}")
    print(f"  Min / Max  : {summary['min']:.4f} / {summary['max']:.4f}")
    print(f"  Percentiles: p25={summary['p25']:.4f}  "
          f"p50={summary['median']:.4f}  "
          f"p75={summary['p75']:.4f}  "
          f"p95={summary['p95']:.4f}  "
          f"p99={summary['p99']:.4f}")
    if "alert_rate_pct" in summary:
        print(f"  Threshold  : {summary['alert_threshold']:.4f}")
        print(f"  Alerts     : {summary['n_alerts']:,} ({summary['alert_rate_pct']}%)")


# ---------------------------------------------------------------------------
# Synthetic anomaly injection
# ---------------------------------------------------------------------------

def inject_synthetic_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str],
    anomaly_type: str = "extreme_value",
    n_anomalies: int = 50,
    random_state: int = 42,
    **kwargs,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Inject synthetic anomalies into a copy of the DataFrame.

    PURPOSE:
        Create a test dataset with KNOWN anomaly locations so we can verify
        that each detector correctly identifies them.  The original DataFrame
        is never modified.

    ANOMALY TYPES SUPPORTED:
        'extreme_value'   — Shift selected features to 5× their training range.
        'sudden_change'   — Apply a large spike to one randomly selected feature.
        'unusual_combo'   — Flip the sign of a subset of features simultaneously.
        'missing_values'  — Set a feature to its column mean (should look normal).

    DESIGN DECISION:
        Anomalies are injected into randomly selected ROWS, not in any
        temporal cluster.  This tests per-row detection without temporal logic.

    IMPORTANT:
        Always clearly label injected rows.  Never mix them with genuine
        data for reporting purposes.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset (not modified — a copy is returned).
    feature_cols : list[str]
        Feature column names available for injection.
    anomaly_type : str
        One of: 'extreme_value', 'sudden_change', 'unusual_combo'.
    n_anomalies : int
        Number of rows to modify.
    random_state : int
        Reproducibility seed.
    **kwargs :
        anomaly_features : list[str]
            Specific features to modify (default: random subset of size 3).
        shift_multiplier : float
            For 'extreme_value' — how many std units to shift.  Default 8.

    Returns
    -------
    df_injected : pd.DataFrame
        Copy of df with modified rows.  Original column structure preserved.
    labels : np.ndarray of int, shape (n_rows,)
        1 = injected anomaly, 0 = genuine observation.
        These are the ground-truth labels for evaluating detection accuracy.
    """
    rng = np.random.RandomState(random_state)
    df_injected = df.copy().reset_index(drop=True)
    labels = np.zeros(len(df_injected), dtype=int)

    n_anomalies = min(n_anomalies, len(df_injected))
    anomaly_indices = rng.choice(len(df_injected), size=n_anomalies, replace=False)
    labels[anomaly_indices] = 1

    # Choose which features to modify
    n_mod_features = min(3, len(feature_cols))
    anomaly_features = kwargs.get("anomaly_features", None)
    if anomaly_features is None:
        anomaly_features = list(rng.choice(feature_cols, size=n_mod_features, replace=False))

    shift_multiplier = kwargs.get("shift_multiplier", 8.0)

    if anomaly_type == "extreme_value":
        # Shift selected features to far outside normal range
        for feat in anomaly_features:
            if feat not in df_injected.columns:
                continue
            col_std = df_injected[feat].std()
            col_mean = df_injected[feat].mean()
            df_injected.loc[anomaly_indices, feat] = (
                col_mean + shift_multiplier * col_std
            )

    elif anomaly_type == "sudden_change":
        # Random spike in a single feature per anomalous row
        for idx in anomaly_indices:
            feat = rng.choice(anomaly_features)
            if feat not in df_injected.columns:
                continue
            col_std = df_injected[feat].std()
            col_mean = df_injected[feat].mean()
            spike_sign = rng.choice([-1, 1])
            df_injected.loc[idx, feat] = (
                col_mean + spike_sign * shift_multiplier * col_std
            )

    elif anomaly_type == "unusual_combo":
        # Set one group of features to very high values AND another to very low
        half = len(anomaly_features) // 2
        high_feats = anomaly_features[:half] if half > 0 else anomaly_features[:1]
        low_feats = anomaly_features[half:] if half > 0 else anomaly_features[:1]
        for feat in high_feats:
            if feat not in df_injected.columns:
                continue
            col_std = df_injected[feat].std()
            col_mean = df_injected[feat].mean()
            df_injected.loc[anomaly_indices, feat] = col_mean + shift_multiplier * col_std
        for feat in low_feats:
            if feat not in df_injected.columns:
                continue
            col_std = df_injected[feat].std()
            col_mean = df_injected[feat].mean()
            df_injected.loc[anomaly_indices, feat] = col_mean - shift_multiplier * col_std

    else:
        raise ValueError(
            f"Unknown anomaly_type '{anomaly_type}'. "
            "Choose from: 'extreme_value', 'sudden_change', 'unusual_combo'."
        )

    return df_injected, labels


# ---------------------------------------------------------------------------
# Detection performance on labeled test set
# ---------------------------------------------------------------------------

def evaluate_detection(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    detector_name: str = "detector",
) -> dict:
    """
    Compute detection metrics when ground-truth binary labels are available.

    WHEN TO USE:
        Only valid on synthetic anomaly test sets where labels are known.
        Do NOT use this function on the original unlabelled dataset and
        report the numbers as clinical accuracy — that would be misleading.

    METRICS:
        • Precision = true anomalies among flagged rows / total flagged.
          "Of all the alerts, how many were real?"
        • Recall    = flagged real anomalies / total real anomalies.
          "Of all real anomalies, how many did we catch?"
        • F1        = harmonic mean of precision and recall.
        • Detection rate on synthetic anomalies = fraction of injected
          anomalies that received a score above the threshold.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples,)
        Anomaly scores (higher = more anomalous).
    labels : np.ndarray of int, shape (n_samples,)
        Ground-truth labels: 1 = anomaly, 0 = normal.
    threshold : float
        Score threshold above which a sample is flagged as anomalous.
    detector_name : str

    Returns
    -------
    dict with precision, recall, f1, alert_rate, detection_rate.
    """
    predicted = (scores >= threshold).astype(int)
    true_positives = int(((predicted == 1) & (labels == 1)).sum())
    false_positives = int(((predicted == 1) & (labels == 0)).sum())
    false_negatives = int(((predicted == 0) & (labels == 1)).sum())

    n_anomalies = int(labels.sum())
    n_flagged = int(predicted.sum())

    precision = true_positives / n_flagged if n_flagged > 0 else 0.0
    recall = true_positives / n_anomalies if n_anomalies > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "detector": detector_name,
        "threshold": threshold,
        "n_samples": len(scores),
        "n_injected_anomalies": n_anomalies,
        "n_flagged": n_flagged,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "alert_rate_pct": round(n_flagged / len(scores) * 100, 2),
        "detection_rate_pct": round(recall * 100, 2),
    }


def print_detection_results(results: dict) -> None:
    """Pretty-print detection results."""
    print(f"\n--- Detection Results: {results['detector']} ---")
    print(f"  Threshold          : {results['threshold']:.4f}")
    print(f"  Injected anomalies : {results['n_injected_anomalies']}")
    print(f"  Flagged            : {results['n_flagged']}")
    print(f"  True positives     : {results['true_positives']}")
    print(f"  False positives    : {results['false_positives']}")
    print(f"  False negatives    : {results['false_negatives']}")
    print(f"  Precision          : {results['precision']:.4f}")
    print(f"  Recall             : {results['recall']:.4f}")
    print(f"  F1 score           : {results['f1']:.4f}")
    print(f"  Detection rate     : {results['detection_rate_pct']}%")
    print(f"  Alert rate         : {results['alert_rate_pct']}%")


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------

def score_stability_check(
    score_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_bootstrap: int = 10,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> dict:
    """
    Estimate score stability across bootstrap subsamples.

    PURPOSE:
        A good anomaly detector should produce consistent scores when applied
        to slightly different subsets of the same data.  High variance
        suggests the model is unstable and may not generalise well.

    ALGORITHM:
        1. Draw n_bootstrap random subsamples (with replacement) of the data.
        2. Compute scores on each subsample.
        3. Compute the standard deviation of each sample's score across
           bootstrap iterations (not possible without consistent indices,
           so we measure the std of mean scores across subsamples instead).

    Parameters
    ----------
    score_fn : callable
        A function that takes an np.ndarray and returns an np.ndarray of scores.
    X : np.ndarray
    n_bootstrap : int
    sample_fraction : float
    random_state : int

    Returns
    -------
    dict with mean, std of score distributions across bootstrap runs.
    """
    rng = np.random.RandomState(random_state)
    mean_scores = []
    for _ in range(n_bootstrap):
        n_sample = int(len(X) * sample_fraction)
        idx = rng.choice(len(X), size=n_sample, replace=True)
        scores = score_fn(X[idx])
        mean_scores.append(np.mean(scores))

    return {
        "n_bootstrap": n_bootstrap,
        "mean_of_means": float(np.mean(mean_scores)),
        "std_of_means": float(np.std(mean_scores)),
        "min_mean": float(np.min(mean_scores)),
        "max_mean": float(np.max(mean_scores)),
        "cv_pct": round(
            np.std(mean_scores) / np.mean(mean_scores) * 100
            if np.mean(mean_scores) > 0 else 0.0,
            2
        ),
    }
