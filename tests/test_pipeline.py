"""
test_pipeline.py — Verification Tests for KUET_CHAYAPOTH Modular Pipeline
"""

import numpy as np
import pandas as pd
import pytest
from src.baseline import PersonalBaseline
from src.detector import MultivariateAnomalyDetector
from src.temporal import TemporalVerifier
from src.protocols import NASAProtocolEngine
from src.evaluation import evaluate_detector


@pytest.fixture
def mock_dataset():
    """Generates a synthetic dataset with baseline, stress, and recovery segments."""
    np.random.seed(42)
    n_base = 200
    n_stress = 100
    n_meditation = 100

    # Baseline (Normal)
    df_base = pd.DataFrame({
        "HR": np.random.normal(70, 2, n_base),
        "RMSSD": np.random.normal(20, 2, n_base),
        "SDRR": np.random.normal(60, 5, n_base),
        "MEAN_RR": np.random.normal(850, 20, n_base),
        "condition": "baseline",
        "Time": np.linspace(0, 100, n_base)
    })

    # Stress (Elevated HR, Depressed RMSSD)
    df_stress = pd.DataFrame({
        "HR": np.random.normal(85, 3, n_stress),
        "RMSSD": np.random.normal(12, 1, n_stress),
        "SDRR": np.random.normal(45, 3, n_stress),
        "MEAN_RR": np.random.normal(710, 15, n_stress),
        "condition": "stress",
        "Time": np.linspace(101, 150, n_stress)
    })

    # Meditation (Depressed HR, Elevated RMSSD)
    df_med = pd.DataFrame({
        "HR": np.random.normal(60, 2, n_meditation),
        "RMSSD": np.random.normal(30, 2, n_meditation),
        "SDRR": np.random.normal(75, 4, n_meditation),
        "MEAN_RR": np.random.normal(1000, 25, n_meditation),
        "condition": "meditation",
        "Time": np.linspace(151, 200, n_meditation)
    })

    return pd.concat([df_base, df_stress, df_med], ignore_index=True)


def test_personal_baseline(mock_dataset):
    baseline = PersonalBaseline(features=['HR', 'RMSSD', 'SDRR', 'MEAN_RR'])
    baseline.fit(mock_dataset, condition_filter='baseline')

    assert baseline.is_fitted
    assert 'HR' in baseline.metrics
    assert 68.0 < baseline.metrics['HR'].median < 72.0
    assert baseline.metrics['HR'].mad > 0.0

    z_df = baseline.robust_z_scores(mock_dataset)
    assert 'HR_zscore' in z_df.columns
    assert len(z_df) == len(mock_dataset)


def test_multivariate_detector(mock_dataset):
    baseline = PersonalBaseline().fit(mock_dataset)
    detector = MultivariateAnomalyDetector(contamination=0.05, n_estimators=50, random_state=42)
    detector.fit(mock_dataset, condition_filter='baseline')

    assert detector.is_fitted
    scores = detector.score_samples(mock_dataset)
    assert len(scores) == len(mock_dataset)
    assert 0.0 <= scores.min() and scores.max() <= 1.0

    preds = detector.predict(mock_dataset)
    assert set(np.unique(preds)).issubset({0, 1})

    # Check explainability on a known stress sample
    stress_sample = mock_dataset[mock_dataset['condition'] == 'stress'].iloc[0]
    explanation = detector.explain_sample(stress_sample, baseline)

    assert "state_classification" in explanation
    assert explanation["state_classification"] == "ACUTE_PHYSIOLOGICAL_STRAIN"
    assert len(explanation["contributing_signals"]) > 0


def test_temporal_verifier(mock_dataset):
    verifier = TemporalVerifier(window_size=10, min_anomalous=8)

    raw_anomalies = np.zeros(100)
    # Inject isolated 2-sample spike (should be filtered out)
    raw_anomalies[10:12] = 1
    # Inject sustained 15-sample event (should be verified)
    raw_anomalies[50:65] = 1

    verified = verifier.verify(raw_anomalies)

    # Spike was filtered
    assert verified.iloc[10] == 0
    assert verified.iloc[11] == 0

    # Sustained event was verified
    assert verified.iloc[57] == 1

    episodes = verifier.extract_episodes(verified, raw_anomalies)
    assert len(episodes) >= 1
    assert episodes[0].sample_count >= 8


def test_nasa_protocol_engine():
    rec_strain = NASAProtocolEngine.get_recommendation("ACUTE_PHYSIOLOGICAL_STRAIN")
    assert rec_strain.code == "FS-CARDIO-02"
    assert rec_strain.urgency == "ELEVATED"

    rec_recovery = NASAProtocolEngine.get_recommendation("PARASYMPATHETIC_RECOVERY")
    assert rec_recovery.code == "FS-RECOVERY-01"
    assert rec_recovery.urgency == "ROUTINE"


def test_evaluation_metrics(mock_dataset):
    mock_dataset['anomaly_raw'] = np.where(mock_dataset['condition'] == 'stress', 1, 0)
    mock_dataset['verified_anomaly'] = mock_dataset['anomaly_raw']

    res = evaluate_detector(mock_dataset)
    assert isinstance(res, pd.DataFrame)
    assert "F1-Score" in res.columns
    assert len(res) == 2
