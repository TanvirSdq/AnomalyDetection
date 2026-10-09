# Progress Log — Astronaut Health Monitoring System
**NASA Space Apps Challenge 2026 (Test Prototype)**  
*Project Workspace: `Anomaly Detection/`*  
*Last Updated: 2026-10-10*

---

## 1. Executive Summary & Objective

The primary objective of this project is to build a functional, interpretable, reproducible, and extensible unsupervised anomaly-detection prototype that monitors astronaut physiological signals (Heart Rate Variability, HRV) to identify unusual deviations from historical baselines without claiming to make medical diagnoses.

In accordance with `Project context.md` and `special_instruction.txt`, all core stages of the system have been designed, implemented, tested, and validated.

---

## 2. Completed Milestones & Detailed Explanations

### Milestone 1: Mandatory Dataset Inspection (`dummy_data/test_sample.csv`)
* **Inspection Findings:**
  * **Shape:** 11,973 observations × 66 columns.
  * **File Size:** ~12.90 MB.
  * **Subject Identifier:** Exactly 1 subject (`subject id = 2`).
  * **Conditions:** 4 labeled experimental segments: `baseline` (4,617 rows), `meditation` (3,284 rows), `stress` (2,580 rows), `amusement` (1,492 rows).
  * **Temporal Structure:** Monotonically increasing `Time` values (from ~7.08s to ~100.15s).
  * **Data Quality:** Zero missing values, zero infinite values, zero duplicate rows.
  * **Near-constant features detected:** 5 columns (`MEAN_REL_RR`, `MEDIAN_REL_RR`, `MEDIAN_REL_RR_LOG`, `MEAN_REL_RR_YEO_JONSON`, `subject id`) with standard deviation < 0.001.
* **Interpretation:**
  * The dataset represents sequential physiological observations of a single subject experiencing different emotional/cognitive conditions.
  * Because there is one subject, personalization operates at the individual baseline level.
  * Engineered power/log/Box-Cox duplicates were identified to prevent multicollinearity and redundant model computations.

---

### Milestone 2: Modular Source Architecture (`src/`)
To support clean separation of concerns, rapid prototyping, and automated testing, reusable logic was organized into dedicated modules:

1. **`src/data_loader.py`**:
   * **Purpose:** Robust CSV file loading, path validation, and structured dataset profiling (`profile_dataset`).
   * **Key Functionality:** Explicit distinction between metadata columns (`Time`, `subject id`, `condition`, `SSSQ`) and model features. Handles missing files and malformed CSVs with clear exceptions.
2. **`src/preprocessing.py`**:
   * **Purpose:** Stateful data transformation using `RobustScaler` (median and IQR-based scaling, which prevents extreme outliers from skewing the transformation).
   * **Leakage Prevention:** `fit()` is executed strictly on the chronological training split (first 70% of observations). The learned medians, IQRs, and dropped columns are frozen and applied to validation/test sets via `transform()`.
   * **Chronological Split:** Implements `chronological_split()` to preserve temporal integrity (future observations never leak into the historical baseline).
3. **`src/baseline.py`**:
   * **Purpose:** Transparent, interpretable baseline using Median Absolute Deviation (MAD) robust Z-scoring.
   * **Algorithm:** For each feature $j$:
     $$\text{MAD}_j = \text{median}(|X_j - \text{median}(X_j)|)$$
     $$\text{Robust } Z_{i, j} = \frac{|X_{i, j} - \text{median}_j|}{1.4826 \times \text{MAD}_j}$$
     $$\text{Sample Score}_i = \max_j (\text{Robust } Z_{i, j})$$
   * **Explainability:** Implements `top_contributing_features()` to rank which physiological signals drove an alert.
4. **`src/anomaly_detector.py`**:
   * **IsolationForestDetector:** Unsupervised multivariate anomaly detector using ensemble random partitioning trees (200 estimators). Normalized to $[0, 1]$ where higher means more anomalous.
   * **SimpleAutoencoder:** Neural reconstruction anomaly detector (bottleneck compression $34 \to 32 \to 8 \to 32 \to 34$) with PCA reconstruction error fallback if PyTorch is absent.
   * **SafetyRuleChecker:** Configurable, domain-informed physiological threshold rules operating on raw (unscaled) metrics (e.g., extreme tachycardia > 130 bpm, bradycardia < 35 bpm).
   * **EnsembleScorer:** Multi-signal consensus engine combining normalized detector outputs into composite scores and structured alert tiers (`high_priority`, `review`, `informational`).
5. **`src/evaluation.py`**:
   * **Distribution Analysis:** Evaluates score percentiles, mean, std, alert rates without ground-truth labels.
   * **Controlled Synthetic Anomaly Injection:** Evaluates detector sensitivity by injecting controlled modifications (`extreme_value`, `sudden_change`, `unusual_combo`) and reporting Precision, Recall, and F1 on known injected anomalies.
6. **`src/visualization.py`**:
   * Standardized, publication-quality Matplotlib plots including time-series with condition shading, anomaly score overlays, score distribution histograms, model comparison bar charts, feature correlation heatmaps, and individual alert feature contribution charts.

---

### Milestone 3: Unit and Integration Test Suite (`tests/test_pipeline.py`)
A comprehensive test suite of 42 automated tests was implemented covering:
* Data loader path validation, empty file errors, profile structure.
* Preprocessor fit/transform, NaN/infinite value handling, chronological split ordering, and model persistence (`joblib`).
* Robust statistical baseline scoring, monotonicity on injected extremes, and top-feature ranking.
* Isolation Forest scoring range, output finiteness, and serialization.
* Autoencoder reconstruction scoring and sensitivity to injected outliers.
* Safety rule trigger/non-trigger logic and graceful skipping of missing features.
* Ensemble scoring, alert tier mappings, and mismatched input validation.
* Synthetic anomaly injection integrity (ensuring original data is never modified in-place).
* Edge cases (single-row inference, empty outputs, boundary values).

**Execution Result:**
```
pytest tests/test_pipeline.py -v
============================= 42 passed in 2.15s ==============================
```

---

### Milestone 4: Notebook Deliverable (`notebooks/astronaut_health_monitor.ipynb`)
Generated and verified `notebooks/astronaut_health_monitor.ipynb` with 66 cells (44 code cells and 22 markdown documentation cells), adhering to Section 10 of `Project context.md`:
1. Project Introduction & Medical Disclaimers
2. Environment Setup & Clean Imports
3. Configuration & Reproducibility (Fixed Seed = 42)
4. Dataset Path Validation
5. Dataset Loading & Runtime Profiling
6. Exploratory Data Analysis (EDA)
7. Data Quality Report & Near-Constant Analysis
8. Feature and Schema Domain Interpretation
9. Reusable Preprocessing Pipeline (Leakage-free)
10. Train/Test Chronological Split Strategy
11. Statistical Baseline (MAD Robust Z-score)
12. Classical Model (Isolation Forest)
13. Neural Model (Autoencoder / PCA Fallback)
14. Model Comparison & Distribution Review
15. Physiological Safety Rules Layer
16. Threshold Calibration & Ensemble Scoring
17. Evaluation with Synthetic Anomaly Tests
18. Multi-panel Visualization Dashboard
19. Structured Inference on Unseen Observations (JSON dashboard payload)
20. Model Artifact Persistence & Reload Verification
21. Limitations & Future Work
22. Final Prototype Summary

**Notebook Verification:**
All 44 code cells were executed sequentially in a clean Python process without error. All plots, tables, and serialized artifacts were generated successfully.

---

### Milestone 5: Saved Artifacts & Deliverables
The following files are generated and persisted in `artifacts/`:
* `preprocessor.joblib` (Fitted RobustScaler and feature definitions)
* `robust_baseline.joblib` (Fitted baseline medians and MAD values)
* `isolation_forest.joblib` (Fitted 200-tree Isolation Forest model)
* `autoencoder.joblib` (Fitted reconstruction model)
* `evaluation_results.csv` (Detailed synthetic injection test metrics across all models)
* High-resolution visual artifacts:
  * `eda_features_over_time.png`
  * `feature_correlation.png`
  * `baseline_scores.png`
  * `feature_contributions.png`
  * `iforest_scores.png`
  * `autoencoder_scores.png`
  * `score_distributions_comparison.png`
  * `composite_scores.png`
  * `model_comparison.png`
  * `dashboard.png`
  * `scores_by_condition.png`

---

## 3. Definition of Done (DoD) Checklist

| Requirement | Status | Verification Detail |
|---|:---:|---|
| Dataset loaded or clear missing diagnostic | **PASS** | Handled in `load_dataset()`, tested with valid & invalid paths |
| Schema and data quality report generated | **PASS** | Profile generated via `profile_dataset()`, 0 missing cells |
| Preprocessing reproducible and reusable | **PASS** | `AnomalyPreprocessor` tested, saved, reloaded; identical scores |
| Working statistical anomaly baseline | **PASS** | `RobustStatBaseline` with MAD and top feature ranking |
| Unsupervised model evaluated | **PASS** | `IsolationForestDetector` (200 trees, normalized scores) |
| Neural approach evaluated | **PASS** | `SimpleAutoencoder` (PCA linear fallback when torch absent) |
| Anomaly scores & thresholds documented | **PASS** | Documented in code docstrings, Markdown cells, and outputs |
| Chronological split (no leakage) | **PASS** | First 70% used for fit; test data strictly follows chronologically |
| Synthetic anomaly tests executed | **PASS** | Tested on `extreme_value`, `sudden_change`, `unusual_combo` |
| Publication-quality plots generated | **PASS** | 11 visualizations saved in `artifacts/` |
| Notebook executes top-to-bottom | **PASS** | All 44 code cells ran without errors in sequential order |
| Structured inference on unseen data | **PASS** | `run_inference()` produces formatted DataFrame & JSON payload |
| Model persistence works | **PASS** | All models saved as `.joblib` and verified with `< 1e-8` diff |
| Limitations & provenance documented | **PASS** | Synthetic HRV limitations and clinical disclaimers highlighted |
| Ready for dashboard integration | **PASS** | Standardized JSON output schema matching Section 13 specs |

---

## 4. Current State & Next Steps

* **Current Status:** The prototype is complete, fully tested (42/42 unit tests passing), reproducible, and ready for demonstration.
* **Execution Documentation:** Added [`HOW_TO_RUN.md`](file:///D:/Actual%20Project%20Files/Training/Python/Machine%20Learning/Anomaly%20Detection/HOW_TO_RUN.md) containing phase-zero setup and execution instructions in a clean `venv` across both Windows and macOS/Linux.
* **Immediate Options for Next Steps:**
  1. If PyTorch is installed by the user, train the multi-layer neural autoencoder model instead of the PCA fallback.
  2. Implement a lightweight local dashboard or interactive preview (e.g., Streamlit / Gradio or a simple Flask API) consuming `run_inference()`.
  3. Prepare presentation slide deck / summary documentation for the NASA Space Apps Challenge submission.
