# Astronaut Health Monitor — Anomaly Detection System

> **NASA Space Apps Challenge 2026 (Test Prototype)**  
> *Create Health Monitoring Software for Astronauts on Space Missions*

---

## ⚠️ Disclaimer

This is a **research and demonstration prototype**, not a certified medical device.  
All alerts are signals for human review — not diagnoses.  
All physiological safety thresholds are illustrative engineering examples, not validated clinical standards.

---

## Overview

A personalized, unsupervised anomaly-detection system for astronaut physiological (HRV) data. The system learns what "normal" looks like for a specific subject from historical observations, then identifies unusual deviations for human review.

**Pipeline:**
1. Load and profile the HRV dataset
2. Chronological train/test split (no data leakage)
3. Robust statistical baseline (MAD Z-score)
4. Isolation Forest (multivariate anomaly detection)
5. Autoencoder (neural reconstruction, or PCA fallback)
6. Physiological safety rules (domain-informed)
7. Ensemble composite scoring and alert prioritization
8. Synthetic anomaly injection for pipeline validation

---

## Project Structure

```
Anomaly Detection/
├── Project context.md           ← Project requirements and specifications
├── special_instruction.txt      ← Developer instructions
├── dummy_data/
│   └── test_sample.csv          ← HRV dataset (1 subject, 4 conditions, ~12k rows)
├── notebooks/
│   └── astronaut_health_monitor.ipynb   ← PRIMARY DELIVERABLE
├── src/
│   ├── __init__.py
│   ├── data_loader.py           ← CSV loading and dataset profiling
│   ├── preprocessing.py         ← Stateful feature preprocessing (RobustScaler)
│   ├── baseline.py              ← MAD-based robust statistical baseline
│   ├── anomaly_detector.py      ← IsolationForest, Autoencoder, SafetyRules, Ensemble
│   ├── evaluation.py            ← Score analysis, synthetic injection, detection metrics
│   └── visualization.py         ← All plotting functions
├── tests/
│   └── test_pipeline.py         ← 42 unit and integration tests
├── artifacts/                   ← Saved models and evaluation results (generated)
├── requirements.txt
├── build_notebook.py            ← Script to regenerate the notebook
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter notebook (primary demo)

```bash
jupyter notebook notebooks/astronaut_health_monitor.ipynb
```

Or open directly in VS Code / JupyterLab. Run all cells top-to-bottom.

### 3. Run tests

```bash
python -m pytest tests/test_pipeline.py -v
```
All 42 tests should pass.

### 4. Regenerate the notebook (if needed)

```bash
python build_notebook.py
```

---

## Dataset

`dummy_data/test_sample.csv` — Synthetic/representative HRV (Heart Rate Variability) data:
- **Rows:** 11,973
- **Columns:** 66 (HRV features + metadata)
- **Subjects:** 1 (Subject ID = 2)
- **Conditions:** baseline (4,617 rows), meditation (3,284), stress (2,580), amusement (1,492)
- **Time range:** ~7s to ~100s (seconds elapsed)
- **Missing values:** None
- **Key features:** HR, MEAN_RR, RMSSD, SDRR, LF, HF, LF_HF, SD1, SD2, pNN25, pNN50

---

## Model Architecture

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| Statistical baseline | MAD robust Z-score | Per-feature extreme detection |
| IsolationForest | Random partitioning trees | Multivariate outlier detection |
| Autoencoder | Neural reconstruction (PCA fallback) | Pattern anomaly detection |
| Safety rules | Configurable thresholds | Domain-informed hard limits |
| Ensemble | Normalized average | Composite alert scoring |

---

## Output Schema

Each inference observation produces a structured record:

```json
{
  "subject_id": "2",
  "timestamp_s": 75.3,
  "condition": "meditation",
  "anomaly_score": 0.612,
  "alert_status": "review",
  "baseline_score": 0.45,
  "iforest_score": 0.63,
  "autoencoder_score": 0.79,
  "explanation": "Composite anomaly score 0.612 — alert tier: review",
  "data_quality_warning": null,
  "model_version": "v1.0-prototype",
  "disclaimer": "Research prototype. Not a medical device."
}
```

---

## Alert Tiers

| Composite Score | Tier | Meaning |
|----------------|------|---------|
| ≥ 0.75 | 🔴 high_priority | Strong anomaly signal — requires prompt review |
| 0.50–0.75 | 🟡 review | Unusual pattern — warrants closer inspection |
| < 0.50 | 🔵 informational | Minor deviation — track but no immediate action |

---

## Test Results

42/42 tests pass:
- Data loading (valid, missing, empty files)
- Preprocessing (shape, NaN-free, leakage prevention, save/load)
- Baseline (scores, feature contributions, persistence)
- IsolationForest (scores, range, finiteness)
- Autoencoder (scores, anomaly sensitivity)
- Safety rules (trigger/non-trigger, missing columns)
- Ensemble (combination, alert tiers)
- Synthetic injection (label accuracy, detection rate > 50%)
- Edge cases (single row, score distribution)

---

## Neural Autoencoder Note

The autoencoder uses a **PCA-based fallback** because PyTorch is not installed.  
To use the full neural autoencoder, install PyTorch:

```bash
pip install torch
```

The `SimpleAutoencoder` class automatically detects PyTorch and switches to the neural backend.
