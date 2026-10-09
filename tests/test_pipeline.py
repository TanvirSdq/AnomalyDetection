"""
test_pipeline.py — Unit and Integration Tests for the Anomaly Detection Pipeline
==================================================================================

PURPOSE:
    Verify that every component of the pipeline handles both normal inputs
    and edge cases correctly, following the requirements in Section 9.4 of
    the project context.

HOW TO RUN:
    From the project root directory:
        python -m pytest tests/test_pipeline.py -v

WHAT IS TESTED:
    1. Dataset loading — valid file, missing file, empty file.
    2. Data profiling — structure of the returned dict.
    3. Preprocessing — fit/transform, leakage prevention, edge cases.
    4. RobustStatBaseline — fit, score, predict, persistence.
    5. IsolationForestDetector — fit, score, predict.
    6. SimpleAutoencoder (PCA fallback) — fit, score.
    7. SafetyRuleChecker — rule triggering and non-triggering.
    8. EnsembleScorer — score combination and alert tiers.
    9. Synthetic anomaly injection — injection and detection.
    10. Chronological split — ordering and proportions.

NOTE ON ASSERTIONS:
    Where exact numeric values are not predictable (unsupervised models have
    no guaranteed output magnitude), we test:
    • Shape and dtype of outputs.
    • Monotonicity (anomalous samples should score higher than normal ones
      when injected anomalies are extreme enough).
    • Finiteness (no NaN or inf in scores).
    • Boundary conditions (empty DataFrame, single row, etc.).
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure src/ is importable when running tests from the project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset, profile_dataset, METADATA_COLS
from src.preprocessing import AnomalyPreprocessor, chronological_split
from src.baseline import RobustStatBaseline
from src.anomaly_detector import (
    IsolationForestDetector,
    SimpleAutoencoder,
    SafetyRuleChecker,
    EnsembleScorer,
)
from src.evaluation import (
    inject_synthetic_anomalies,
    evaluate_detection,
    analyse_score_distribution,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable test data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df():
    """
    Create a small synthetic DataFrame that mimics the schema of test_sample.csv.
    Using synthetic data here means tests run without requiring the actual CSV.
    """
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "Time": np.linspace(0, 100, n),
        "HR": rng.normal(70, 5, n),
        "MEAN_RR": rng.normal(860, 50, n),
        "RMSSD": rng.normal(30, 8, n),
        "SDRR": rng.normal(50, 15, n),
        "LF": rng.normal(1000, 200, n),
        "HF": rng.normal(500, 100, n),
        "LF_HF": rng.normal(2.0, 0.5, n),
        "subject id": np.ones(n, dtype=int) * 2,
        "condition": np.where(np.arange(n) < 140, "baseline", "stress"),
        "SSSQ": rng.randint(3, 6, n),
    })
    return df


@pytest.fixture(scope="module")
def fitted_preprocessor(sample_df):
    """Return a preprocessor fitted on the first 140 rows of sample_df."""
    train_df = sample_df.iloc[:140]
    prep = AnomalyPreprocessor()
    prep.fit(train_df)
    return prep


@pytest.fixture(scope="module")
def X_train(sample_df, fitted_preprocessor):
    train_df = sample_df.iloc[:140]
    X, _ = fitted_preprocessor.transform(train_df)
    return X


@pytest.fixture(scope="module")
def X_test(sample_df, fitted_preprocessor):
    test_df = sample_df.iloc[140:]
    X, _ = fitted_preprocessor.transform(test_df)
    return X


# ---------------------------------------------------------------------------
# 1. Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoader:

    def test_load_valid_csv(self, tmp_path, sample_df):
        """Loading a properly formed CSV should return a non-empty DataFrame."""
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_dataset(csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_df)
        assert list(df.columns) == list(sample_df.columns)

    def test_load_missing_file_raises(self, tmp_path):
        """A missing file must raise FileNotFoundError, not a silent failure."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_dataset(tmp_path / "does_not_exist.csv")

    def test_load_empty_csv_raises(self, tmp_path):
        """An empty CSV (header only) should raise ValueError."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("col1,col2\n")  # header only, no data rows
        with pytest.raises(ValueError, match="empty"):
            load_dataset(csv_path)

    def test_load_max_rows(self, tmp_path, sample_df):
        """max_rows parameter should limit the number of loaded rows."""
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_dataset(csv_path, max_rows=50)
        assert len(df) == 50

    def test_profile_structure(self, sample_df):
        """profile_dataset() must return a dict with the expected keys."""
        profile = profile_dataset(sample_df)
        required_keys = [
            "n_rows", "n_cols", "column_names", "dtypes",
            "numeric_columns", "categorical_columns",
            "missing_counts", "total_missing_cells",
            "n_duplicate_rows", "numeric_summary",
        ]
        for key in required_keys:
            assert key in profile, f"Missing key in profile: {key}"
        assert profile["n_rows"] == len(sample_df)
        assert profile["total_missing_cells"] == 0


# ---------------------------------------------------------------------------
# 2. Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:

    def test_fit_transform_shape(self, sample_df):
        """Output shape must be (n_rows, n_features) after fit_transform."""
        prep = AnomalyPreprocessor()
        X, meta = prep.fit_transform(sample_df.iloc[:140])
        assert X.ndim == 2
        assert X.shape[0] == 140
        assert X.shape[1] > 0
        assert isinstance(meta, pd.DataFrame)

    def test_no_nan_in_output(self, sample_df):
        """Output scaled matrix must contain no NaN values."""
        prep = AnomalyPreprocessor()
        X, _ = prep.fit_transform(sample_df.iloc[:140])
        assert not np.isnan(X).any(), "NaN values found in preprocessed output"

    def test_no_inf_in_output(self, sample_df):
        """Output scaled matrix must contain no infinite values."""
        prep = AnomalyPreprocessor()
        X, _ = prep.fit_transform(sample_df.iloc[:140])
        assert not np.isinf(X).any(), "Infinite values found in preprocessed output"

    def test_transform_before_fit_raises(self, sample_df):
        """Calling transform() without fitting must raise RuntimeError."""
        prep = AnomalyPreprocessor()
        with pytest.raises(RuntimeError, match="fitted"):
            prep.transform(sample_df)

    def test_consistent_feature_count(self, sample_df):
        """Training and test transforms must produce the same number of features."""
        prep = AnomalyPreprocessor()
        X_train, _ = prep.fit_transform(sample_df.iloc[:140])
        X_test, _ = prep.transform(sample_df.iloc[140:])
        assert X_train.shape[1] == X_test.shape[1]

    def test_save_and_load(self, tmp_path, fitted_preprocessor, sample_df):
        """A saved and reloaded preprocessor must produce identical outputs."""
        save_path = tmp_path / "prep.joblib"
        fitted_preprocessor.save(save_path)
        loaded = AnomalyPreprocessor.load(save_path)
        X_orig, _ = fitted_preprocessor.transform(sample_df.iloc[:5])
        X_loaded, _ = loaded.transform(sample_df.iloc[:5])
        np.testing.assert_array_almost_equal(X_orig, X_loaded)

    def test_insufficient_rows_raises(self):
        """fit() with fewer than MIN_TRAIN_ROWS rows must raise ValueError."""
        from src.preprocessing import MIN_TRAIN_ROWS
        tiny_df = pd.DataFrame({
            "Time": [1.0] * 3,
            "HR": [70.0] * 3,
            "subject id": [1] * 3,
            "condition": ["baseline"] * 3,
            "SSSQ": [5] * 3,
        })
        prep = AnomalyPreprocessor()
        with pytest.raises(ValueError, match="rows"):
            prep.fit(tiny_df)

    def test_chronological_split_ordering(self, sample_df):
        """Train set must contain earlier time values than test set."""
        train, test = chronological_split(sample_df, train_fraction=0.7)
        assert train["Time"].max() <= test["Time"].min()

    def test_chronological_split_proportions(self, sample_df):
        """Split proportions must approximately match the requested fraction."""
        train, test = chronological_split(sample_df, train_fraction=0.7)
        total = len(train) + len(test)
        assert total == len(sample_df)
        assert abs(len(train) / total - 0.7) < 0.02  # within 2%


# ---------------------------------------------------------------------------
# 3. Robust statistical baseline tests
# ---------------------------------------------------------------------------

class TestRobustStatBaseline:

    def test_fit_and_score_shape(self, X_train, X_test):
        """score_samples must return an array with length == n_test."""
        baseline = RobustStatBaseline()
        baseline.fit(X_train)
        scores = baseline.score_samples(X_test)
        assert scores.shape == (len(X_test),)

    def test_scores_finite(self, X_train, X_test):
        """All scores must be finite (no NaN or inf)."""
        baseline = RobustStatBaseline()
        baseline.fit(X_train)
        scores = baseline.score_samples(X_test)
        assert np.isfinite(scores).all(), "Non-finite baseline scores detected"

    def test_scores_nonneg(self, X_train, X_test):
        """Robust Z-scores are absolute values and must be non-negative."""
        baseline = RobustStatBaseline()
        baseline.fit(X_train)
        scores = baseline.score_samples(X_test)
        assert (scores >= 0).all()

    def test_extreme_injection_scores_higher(self, X_train):
        """Injected extreme values must score higher than normal values."""
        baseline = RobustStatBaseline()
        baseline.fit(X_train)

        X_normal = X_train[:20].copy()
        X_extreme = X_train[:20].copy()
        X_extreme[:, 0] += 100  # extreme shift in first feature

        scores_normal = baseline.score_samples(X_normal)
        scores_extreme = baseline.score_samples(X_extreme)

        assert scores_extreme.mean() > scores_normal.mean()

    def test_predict_binary(self, X_train, X_test):
        """predict() must return only 0 and 1 values."""
        baseline = RobustStatBaseline()
        baseline.fit(X_train)
        preds = baseline.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_save_load(self, tmp_path, X_train, X_test):
        """Save/load cycle must produce identical scores."""
        baseline = RobustStatBaseline()
        baseline.fit(X_train)
        scores_before = baseline.score_samples(X_test)

        save_path = tmp_path / "baseline.joblib"
        baseline.save(save_path)
        loaded = RobustStatBaseline.load(save_path)
        scores_after = loaded.score_samples(X_test)

        np.testing.assert_array_almost_equal(scores_before, scores_after)

    def test_unfitted_raises(self, X_test):
        """Calling score_samples on unfitted baseline must raise RuntimeError."""
        baseline = RobustStatBaseline()
        with pytest.raises(RuntimeError, match="fitted"):
            baseline.score_samples(X_test)


# ---------------------------------------------------------------------------
# 4. IsolationForest tests
# ---------------------------------------------------------------------------

class TestIsolationForest:

    def test_fit_score_shape(self, X_train, X_test):
        detector = IsolationForestDetector(n_estimators=50, random_state=0)
        detector.fit(X_train)
        scores = detector.score_samples(X_test)
        assert scores.shape == (len(X_test),)

    def test_scores_in_range(self, X_train, X_test):
        """Normalised IsolationForest scores should be in [0, 1]."""
        detector = IsolationForestDetector(n_estimators=50, random_state=0)
        detector.fit(X_train)
        scores = detector.score_samples(X_test)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_scores_finite(self, X_train, X_test):
        detector = IsolationForestDetector(n_estimators=50, random_state=0)
        detector.fit(X_train)
        scores = detector.score_samples(X_test)
        assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# 5. Autoencoder (PCA fallback) tests
# ---------------------------------------------------------------------------

class TestAutoencoder:

    def test_fit_score_shape(self, X_train, X_test):
        ae = SimpleAutoencoder(hidden_dims=(4, 2), epochs=5, random_state=0)
        ae.fit(X_train)
        scores = ae.score_samples(X_test)
        assert scores.shape == (len(X_test),)

    def test_scores_finite(self, X_train, X_test):
        ae = SimpleAutoencoder(hidden_dims=(4, 2), epochs=5, random_state=0)
        ae.fit(X_train)
        scores = ae.score_samples(X_test)
        assert np.isfinite(scores).all()

    def test_extreme_injection_scores_higher(self, X_train):
        ae = SimpleAutoencoder(hidden_dims=(4, 2), epochs=10, random_state=0)
        ae.fit(X_train)

        X_normal = X_train[:20].copy()
        X_extreme = X_train[:20].copy()
        X_extreme += 50  # far outside training distribution

        scores_normal = ae.score_samples(X_normal)
        scores_extreme = ae.score_samples(X_extreme)
        assert scores_extreme.mean() > scores_normal.mean()


# ---------------------------------------------------------------------------
# 6. Safety rule checker tests
# ---------------------------------------------------------------------------

class TestSafetyRuleChecker:

    def test_hr_above_threshold_triggers(self):
        """A row with HR > 130 should trigger the HR_extreme_high rule."""
        df = pd.DataFrame({"HR": [70.0, 135.0, 68.0], "RMSSD": [30.0, 25.0, 28.0]})
        checker = SafetyRuleChecker()
        alerts = checker.check(df)
        triggered_rules = alerts["rule_name"].tolist()
        assert "HR_extreme_high" in triggered_rules

    def test_normal_hr_no_high_priority(self):
        """Normal HR values should not trigger high-priority rules."""
        df = pd.DataFrame({"HR": [65.0, 72.0, 68.0], "RMSSD": [30.0, 25.0, 28.0]})
        checker = SafetyRuleChecker()
        alerts = checker.check(df)
        high_priority = alerts[alerts["severity"] == "high_priority"]
        assert len(high_priority) == 0

    def test_missing_column_skipped(self):
        """Rules for unavailable columns should be silently skipped."""
        df = pd.DataFrame({"HR": [70.0, 72.0]})  # RMSSD missing
        checker = SafetyRuleChecker()
        alerts = checker.check(df)
        # Should not raise, just skip RMSSD rule
        assert isinstance(alerts, pd.DataFrame)

    def test_returns_dataframe(self):
        """check() must always return a DataFrame, even with no alerts."""
        df = pd.DataFrame({"HR": [70.0], "RMSSD": [30.0]})
        checker = SafetyRuleChecker()
        result = checker.check(df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# 7. Ensemble scorer tests
# ---------------------------------------------------------------------------

class TestEnsembleScorer:

    def test_combine_output_shape(self):
        n = 100
        scores = {
            "baseline": np.random.rand(n),
            "iforest": np.random.rand(n),
        }
        scorer = EnsembleScorer()
        result = scorer.combine(scores)
        assert len(result) == n
        assert "composite_score" in result.columns
        assert "alert_tier" in result.columns

    def test_alert_tiers_valid(self):
        n = 100
        scores = {"baseline": np.random.rand(n)}
        scorer = EnsembleScorer()
        result = scorer.combine(scores)
        valid_tiers = {"informational", "review", "high_priority"}
        assert set(result["alert_tier"].unique()).issubset(valid_tiers)

    def test_all_high_scores_trigger_high_priority(self):
        n = 10
        scores = {"baseline": np.ones(n)}  # all scores = 1.0
        scorer = EnsembleScorer()
        result = scorer.combine(scores)
        assert (result["alert_tier"] == "high_priority").all()

    def test_all_zero_scores_informational(self):
        n = 10
        scores = {"baseline": np.zeros(n)}
        scorer = EnsembleScorer()
        result = scorer.combine(scores)
        assert (result["alert_tier"] == "informational").all()

    def test_mismatched_lengths_raises(self):
        scores = {"a": np.ones(10), "b": np.ones(20)}
        scorer = EnsembleScorer()
        with pytest.raises(ValueError, match="different lengths"):
            scorer.combine(scores)


# ---------------------------------------------------------------------------
# 8. Synthetic anomaly injection + detection tests
# ---------------------------------------------------------------------------

class TestSyntheticAnomalyInjection:

    def test_injection_labels_correct_count(self, sample_df, fitted_preprocessor):
        """Exactly n_anomalies rows should be labeled as anomalous."""
        df_injected, labels = inject_synthetic_anomalies(
            sample_df, feature_cols=fitted_preprocessor.feature_cols_,
            n_anomalies=30, anomaly_type="extreme_value", random_state=0
        )
        assert labels.sum() == 30
        assert len(labels) == len(sample_df)

    def test_injection_does_not_modify_original(self, sample_df, fitted_preprocessor):
        """inject_synthetic_anomalies must not modify the original DataFrame."""
        original_hr = sample_df["HR"].values.copy()
        _ = inject_synthetic_anomalies(
            sample_df, feature_cols=fitted_preprocessor.feature_cols_,
            n_anomalies=30, anomaly_type="extreme_value", random_state=0
        )
        np.testing.assert_array_equal(sample_df["HR"].values, original_hr)

    def test_baseline_detects_extreme_injection(self, sample_df, fitted_preprocessor, X_train):
        """Baseline should detect extreme-value injections with detection rate > 50%."""
        df_injected, labels = inject_synthetic_anomalies(
            sample_df, feature_cols=fitted_preprocessor.feature_cols_,
            n_anomalies=50, anomaly_type="extreme_value", random_state=0,
            shift_multiplier=10.0
        )
        X_inj, _ = fitted_preprocessor.transform(df_injected)

        baseline = RobustStatBaseline()
        baseline.fit(X_train)
        scores = baseline.score_samples(X_inj)

        results = evaluate_detection(scores, labels, threshold=3.5, detector_name="baseline")
        # With a 10× shift, detection rate should be high
        assert results["detection_rate_pct"] > 50.0, (
            f"Baseline detection rate too low: {results['detection_rate_pct']}%"
        )

    def test_evaluate_detection_structure(self, sample_df, fitted_preprocessor, X_train):
        """evaluate_detection must return all required keys."""
        df_injected, labels = inject_synthetic_anomalies(
            sample_df, feature_cols=fitted_preprocessor.feature_cols_,
            n_anomalies=30, random_state=0
        )
        X_inj, _ = fitted_preprocessor.transform(df_injected)
        baseline = RobustStatBaseline()
        baseline.fit(X_train)
        scores = baseline.score_samples(X_inj)
        results = evaluate_detection(scores, labels, threshold=3.5)
        required_keys = [
            "precision", "recall", "f1", "detection_rate_pct",
            "alert_rate_pct", "true_positives"
        ]
        for key in required_keys:
            assert key in results


# ---------------------------------------------------------------------------
# 9. Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_row_scoring(self, fitted_preprocessor, X_train):
        """Pipeline must handle a single-row input without errors."""
        single_row = X_train[[0]]
        baseline = RobustStatBaseline()
        baseline.fit(X_train)
        scores = baseline.score_samples(single_row)
        assert len(scores) == 1
        assert np.isfinite(scores[0])

    def test_score_distribution_summary(self, X_train):
        """analyse_score_distribution must return a complete summary."""
        baseline = RobustStatBaseline()
        baseline.fit(X_train)
        scores = baseline.score_samples(X_train)
        summary = analyse_score_distribution(scores, "baseline", alert_threshold=3.5)
        assert summary["n_samples"] == len(X_train)
        assert "alert_rate_pct" in summary
        assert "mean" in summary
