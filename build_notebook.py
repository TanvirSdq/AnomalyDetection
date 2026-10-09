"""
build_notebook.py -- Script to generate astronaut_health_monitor.ipynb
Run this script to create or regenerate the notebook from scratch.
"""
import json
import sys
from pathlib import Path

def cell(cell_type, source, **kwargs):
    base = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.splitlines(keepends=True),
    }
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base

def md(text):
    return cell("markdown", text)

def code(text):
    return cell("code", text)

cells = []

# ============================================================
# SECTION 1: Project Introduction
# ============================================================
cells.append(md("""# 🚀 Astronaut Health Monitor -- Anomaly Detection System
## NASA Space Apps Challenge 2026 (Test Prototype)

**Purpose:**  
This notebook implements a personalized, unsupervised anomaly-detection system for astronaut physiological data. It learns what "normal" looks like from historical measurements and flags unusual deviations for human review.

**This is a research and demonstration prototype, NOT a certified medical device.**  
Every alert is a signal for further assessment -- not a diagnosis.

**Dataset:** `dummy_data/test_sample.csv` -- Synthetic/representative HRV (Heart Rate Variability) data for one subject across multiple experimental conditions (baseline, stress, amusement, meditation).

**Pipeline:**
1. Load and inspect the dataset  
2. Preprocess features  
3. Train a statistical baseline (MAD-based robust Z-scores)  
4. Train an Isolation Forest detector  
5. Train a neural Autoencoder (or PCA fallback)  
6. Apply physiological safety rules  
7. Combine all signals into a composite anomaly score  
8. Evaluate using synthetic anomaly injection  
9. Visualise results  
10. Demonstrate inference on unseen observations"""))

# ============================================================
# SECTION 2: Environment Setup
# ============================================================
cells.append(md("## 2. Environment Setup and Imports"))

cells.append(code("""# Standard library
import os
import sys
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')  # Suppress sklearn/numpy deprecation notices

# Numeric / data
import numpy as np
import pandas as pd

# Visualisation
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn (classical ML)
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Our custom modules
PROJECT_ROOT = Path('..').resolve()  # one level up from notebooks/
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset, profile_dataset, print_profile, METADATA_COLS, SUBJECT_COL, TIME_COL, CONDITION_COL
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
    print_score_summary,
    print_detection_results,
)
from src.visualization import (
    plot_feature_over_time,
    plot_anomaly_scores,
    plot_score_distribution,
    plot_model_comparison,
    plot_feature_heatmap,
    plot_full_dashboard,
    plot_feature_contributions,
)

print(f\"numpy:      {np.__version__}\")
print(f\"pandas:     {pd.__version__}\")
print(f\"matplotlib: {matplotlib.__version__}\")
print(f\"seaborn:    {sns.__version__}\")
print(\"All imports successful [OK]\")"""))

# ============================================================
# SECTION 3: Configuration
# ============================================================
cells.append(md("""## 3. Configuration and Reproducibility Settings

We fix all random seeds so that every run of this notebook from top to bottom produces identical results. This is essential for a reproducible science/engineering demonstration."""))

cells.append(code("""# --- Random seed --------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Dataset path --------------------------------------------------------------
# Using a path relative to PROJECT_ROOT avoids hardcoded machine-specific paths.
DATA_PATH = PROJECT_ROOT / 'dummy_data' / 'test_sample.csv'

# --- Artifacts output directory ------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Training/test split -------------------------------------------------------
TRAIN_FRACTION = 0.70  # 70% of time-ordered data for training

# --- Anomaly detection thresholds ---------------------------------------------
# These can be tuned without touching model code.
BASELINE_THRESHOLD   = 3.5   # Robust Z-score: ~3.5 MADs from median
IFOREST_THRESHOLD    = 0.55  # Normalised IsolationForest score [0, 1]
AUTOENCODER_THRESHOLD = 0.80  # Normalised reconstruction error
COMPOSITE_THRESHOLD  = 0.55  # Ensemble composite score

# --- Neural model settings -----------------------------------------------------
AE_HIDDEN_DIMS = (32, 8)     # Encoder bottleneck: 32 -> 8 units
AE_EPOCHS = 50
AE_BATCH_SIZE = 256
AE_LR = 1e-3

# --- Evaluation settings -------------------------------------------------------
N_SYNTHETIC_ANOMALIES = 100  # Rows to inject per synthetic test

print(\"Configuration loaded [OK]\")
print(f\"  Data path:        {DATA_PATH}\")
print(f\"  Artifacts dir:    {ARTIFACTS_DIR}\")
print(f\"  Train fraction:   {TRAIN_FRACTION}\")
print(f\"  Random seed:      {RANDOM_SEED}\")"""))

# ============================================================
# SECTION 4: Dataset path validation
# ============================================================
cells.append(md("## 4. Dataset Path Validation"))

cells.append(code("""# Verify the CSV exists before attempting to load it.
# This is the first mandatory step -- we must not fabricate schema or results
# if the file is missing.

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f\"Dataset not found at: {DATA_PATH}\\n\"
        \"Please ensure the dummy_data/ folder is present at the project root.\"
    )

file_size_mb = DATA_PATH.stat().st_size / (1024 ** 2)
print(f\"Dataset found [OK]\")
print(f\"  Path      : {DATA_PATH}\")
print(f\"  File size : {file_size_mb:.2f} MB\")"""))

# ============================================================
# SECTION 5: Dataset Loading
# ============================================================
cells.append(md("""## 5. Dataset Loading

We use `load_dataset()` from `src/data_loader.py` which validates the path, loads the CSV, and raises an informative error if something is wrong.  
We also measure loading time -- a practical check for pipeline performance."""))

cells.append(code("""t0 = time.time()
df = load_dataset(DATA_PATH)
load_time = time.time() - t0

print(f\"Loaded {len(df):,} rows × {df.shape[1]} columns in {load_time:.2f}s\")
print(f\"\\nColumn names:\")
for i, col in enumerate(df.columns, 1):
    print(f\"  {i:>2}. {col}\")"""))

# ============================================================
# SECTION 6: Exploratory Data Analysis
# ============================================================
cells.append(md("""## 6. Exploratory Data Analysis (EDA)

EDA reveals the structure, distribution, and potential issues in the dataset BEFORE modelling.  
We must understand the data before choosing or fitting any model."""))

cells.append(code("""print(f\"Shape: {df.shape}\")
print(f\"\\nFirst 3 rows:\")
df.head(3)"""))

cells.append(code("""# --- Subject summary ----------------------------------------------------------
print(\"=== Subject ID(s) ===\")
print(f\"  Unique subjects: {df[SUBJECT_COL].nunique()}\")
print(f\"  Subject IDs:     {df[SUBJECT_COL].unique().tolist()}\")
print()

# --- Condition distribution ----------------------------------------------------
print(\"=== Experimental Conditions ===\")
cond_counts = df[CONDITION_COL].value_counts()
for cond, count in cond_counts.items():
    print(f\"  {cond:<12}: {count:,} rows  ({count/len(df)*100:.1f}%)\")
print()

# --- Time range ----------------------------------------------------------------
print(\"=== Time Range ===\")
print(f\"  Time column  : '{TIME_COL}' (units: seconds, inferred)\")
print(f\"  Min time     : {df[TIME_COL].min():.2f}s\")
print(f\"  Max time     : {df[TIME_COL].max():.2f}s\")
print(f\"  Is monotonic : {df[TIME_COL].is_monotonic_increasing}\")
print()

# --- HR summary ----------------------------------------------------------------
print(\"=== Heart Rate (HR) Summary ===\")
print(df['HR'].describe().round(3))"""))

cells.append(code("""# --- Condition time segments --------------------------------------------------
# The dataset appears to have sequential condition blocks:
# baseline -> stress -> amusement -> meditation
print(\"=== Time ranges by condition ===\")
time_by_cond = df.groupby(CONDITION_COL)[TIME_COL].agg(['min', 'max', 'count'])
print(time_by_cond.round(2))"""))

cells.append(code("""# --- Key HRV features over time -- visual inspection --------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

features_to_show = ['HR', 'RMSSD', 'LF_HF']
labels_to_show   = ['HR (bpm)', 'RMSSD (ms)', 'LF/HF Ratio']

for ax, feat, label in zip(axes, features_to_show, labels_to_show):
    ax.plot(df[TIME_COL], df[feat], linewidth=0.7, color='steelblue', alpha=0.85)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    
    # Shade condition backgrounds
    conditions = df[CONDITION_COL].values
    time_vals  = df[TIME_COL].values
    cmap = matplotlib.colormaps['Pastel1']
    cond_map = {c: cmap(i) for i, c in enumerate(['baseline','stress','amusement','meditation'])}
    prev_cond  = conditions[0]
    start_idx  = 0
    for i in range(1, len(conditions)):
        if conditions[i] != prev_cond or i == len(conditions) - 1:
            end_i = i if conditions[i] != prev_cond else i + 1
            ax.axvspan(time_vals[start_idx], time_vals[min(end_i, len(time_vals)-1)],
                       alpha=0.22, color=cond_map.get(prev_cond, 'grey'))
            start_idx  = i
            prev_cond  = conditions[i]

axes[-1].set_xlabel('Time (seconds)')

# Custom legend for conditions
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cond_map[c], alpha=0.5, label=c) for c in cond_map]
axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.8)
axes[0].set_title('Key HRV Features over Time -- Condition Segments Highlighted')

plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / 'eda_features_over_time.png', dpi=120, bbox_inches='tight')
plt.show()
print(\"Figure saved to artifacts/eda_features_over_time.png\")"""))

# ============================================================
# SECTION 7: Data Quality Report
# ============================================================
cells.append(md("""## 7. Data Quality Report

A complete data quality report before modelling.  
We use `profile_dataset()` from `src/data_loader.py`."""))

cells.append(code("""profile = profile_dataset(df)
print_profile(profile)"""))

cells.append(code("""# --- Near-constant column analysis --------------------------------------------
# Near-constant columns: columns whose standard deviation is < 0.001.
# These will be automatically dropped by the preprocessor.
# Let's see which ones they are and confirm they are indeed redundant.

near_const = profile['near_constant_columns']
print(f\"Near-constant columns ({len(near_const)} total):\")
for col in near_const:
    std = df[col].std()
    mean = df[col].mean()
    print(f\"  {col:<40}  std={std:.6f}  mean={mean:.6f}\")"""))

cells.append(code("""# --- Feature correlation heatmap ----------------------------------------------
# Visualise correlations among the first 20 numeric features.
# Many features in this dataset are mathematical transforms of each other
# (e.g. RMSSD and RMSSD_LOG), so high correlation is expected.

numeric_cols = [c for c in profile['numeric_columns'] 
                if c not in METADATA_COLS and df[c].std() > 0.001]
fig, ax = plt.subplots(figsize=(14, 12))
plot_feature_heatmap(df, numeric_cols, max_features=20, ax=ax,
                     save_path=ARTIFACTS_DIR / 'feature_correlation.png')
plt.tight_layout()
plt.show()
print(\"Correlation heatmap saved.\")"""))

# ============================================================
# SECTION 8: Feature and Schema Interpretation
# ============================================================
cells.append(md("""## 8. Feature and Schema Interpretation

### Column Roles (inferred from column names -- NOT verified clinical definitions)

| Column | Inferred Meaning | Units | Role |
|--------|-----------------|-------|------|
| `Time` | Elapsed seconds in study | seconds | Timestamp / metadata |
| `HR` | Heart rate | bpm | Physiological feature |
| `MEAN_RR` | Mean RR interval (time between heartbeats) | milliseconds | Physiological feature |
| `RMSSD` | Root-mean-square of successive RR differences -- short-term HRV | ms | Physiological feature |
| `SDRR` | Standard deviation of RR intervals -- overall HRV | ms | Physiological feature |
| `LF`, `HF`, `VLF` | Frequency-domain HRV power bands | ms² | Physiological feature |
| `LF_HF` | LF/HF ratio -- sympathetic/parasympathetic balance | dimensionless | Physiological feature |
| `SD1`, `SD2` | Poincaré plot parameters | ms | Physiological feature |
| `pNN25`, `pNN50` | % of successive RR differences > 25ms / 50ms | % | Physiological feature |
| `*_LOG`, `*_SQRT`, `*_BOXCOX` | Log/sqrt/Box-Cox transforms of base features | same units | Engineered features |
| `subject id` | Subject identifier | -- | Metadata |
| `condition` | Experimental condition label | -- | Metadata |
| `SSSQ` | Subjective Stress Scale-Questionnaire score (3–5) | Likert | Metadata / contextual |

### Important dataset notes:
- **Single subject (ID = 2):** Personalization is applied to this one subject.
- **Four conditions:** baseline, stress, amusement, meditation -- clearly distinct physiological states.
- **No missing values:** This is a clean, well-prepared dataset.
- **Many features are redundant:** Log/SQRT/BoxCox transforms of the same base features. The preprocessor will drop near-constant ones automatically.

> [!]️ **Disclaimer:** All physiological interpretations above are candidate hypotheses based on column names and common HRV literature. They are NOT verified clinical definitions specific to this dataset's provenance."""))

# ============================================================
# SECTION 9: Preprocessing Pipeline
# ============================================================
cells.append(md("""## 9. Preprocessing Pipeline

We use `AnomalyPreprocessor` from `src/preprocessing.py`.

**What it does:**
1. Separates metadata from features.
2. Drops near-constant columns (std < 0.001) -- they carry no discriminative signal.
3. Replaces infinities with NaN, then median-imputes NaN (using training medians).
4. Scales features with `RobustScaler` (uses median and IQR -- robust to outliers).

**Key rule:** The scaler is fitted on TRAINING data only and applied to test data. This prevents data leakage."""))

cells.append(code("""# --- Chronological train/test split -------------------------------------------
# We split chronologically (first 70% = train, last 30% = test).
# This reflects the real deployment scenario: the system learns from historical
# observations and then evaluates on newer, unseen data.
#
# WHY NOT RANDOM SPLIT:
#   A random split would allow training on rows that occurred AFTER test rows.
#   This is data leakage for time-series data -- it makes performance look
#   unrealistically good.

df_train, df_test = chronological_split(df, train_fraction=TRAIN_FRACTION)

print(f\"Training set:  {len(df_train):,} rows  | Time: {df_train[TIME_COL].min():.1f}s -> {df_train[TIME_COL].max():.1f}s\")
print(f\"Test set:      {len(df_test):,} rows   | Time: {df_test[TIME_COL].min():.1f}s -> {df_test[TIME_COL].max():.1f}s\")
print()
print(\"Training condition distribution:\")
print(df_train[CONDITION_COL].value_counts().to_dict())
print(\"\\nTest condition distribution:\")
print(df_test[CONDITION_COL].value_counts().to_dict())"""))

cells.append(code("""# --- Fit the preprocessor on training data ------------------------------------
# Remove highly redundant log/sqrt/boxcox transforms to reduce feature redundancy.
# These are engineered duplicates of the original features and can slow down
# models without adding unique information.
# We keep the originals (RMSSD, LF, HF, etc.) and drop the transforms.
redundant_transforms = [c for c in df.columns if any(
    c.endswith(suffix) for suffix in 
    ['_LOG', '_SQRT', '_BOXCOX', '_YEO_JONSON', '_SQUARE', '_MEAN_MEAN_REL_RR']
)]
preprocessor = AnomalyPreprocessor(extra_drop_cols=redundant_transforms)

t0 = time.time()
preprocessor.fit(df_train)
fit_time = time.time() - t0

print(f\"Preprocessor fitted in {fit_time:.3f}s\")
print(f\"  Selected features: {len(preprocessor.feature_cols_)}\")
print(f\"  Dropped (near-constant): {len(preprocessor.dropped_cols_)}\")
print()
print(\"Selected feature names:\")
for i, col in enumerate(preprocessor.feature_cols_, 1):
    print(f\"  {i:>2}. {col}\")"""))

cells.append(code("""# --- Apply transform to train and test sets -----------------------------------
X_train, meta_train = preprocessor.transform(df_train)
X_test,  meta_test  = preprocessor.transform(df_test)

print(f\"X_train shape: {X_train.shape}\")
print(f\"X_test shape:  {X_test.shape}\")
print(f\"Any NaN in X_train: {np.isnan(X_train).any()}\")
print(f\"Any NaN in X_test:  {np.isnan(X_test).any()}\")
print(f\"Any inf in X_train: {np.isinf(X_train).any()}\")
print(f\"Any inf in X_test:  {np.isinf(X_test).any()}\")"""))

cells.append(code("""# --- Save preprocessor --------------------------------------------------------
preprocessor.save(ARTIFACTS_DIR / 'preprocessor.joblib')"""))

# ============================================================
# SECTION 10: Train/Val/Test Strategy
# ============================================================
cells.append(md("""## 10. Train / Validation / Test Strategy

| Split | Rows | Time Range | Purpose |
|-------|------|-----------|---------|
| Train (70%) | ~8,381 | ~7s -> ~70s | Fit all models |
| Test (30%) | ~3,592 | ~70s -> ~100s | Evaluate models on unseen data |

**No separate validation set** -- with ~12k rows and unsupervised models, we use the test set for evaluation only and choose thresholds analytically rather than by optimising on a validation set.

**Leakage prevention checklist:**
- [x] Scaler fitted on training data only
- [x] Test data imputed with training medians
- [x] Anomaly detectors trained on training features only  
- [x] Safety rules evaluated on raw values -- no leakage risk
- [x] Synthetic anomalies are clearly labelled and separated from evaluation"""))

# ============================================================
# SECTION 11: Statistical Baseline
# ============================================================
cells.append(md("""## 11. Statistical Baseline -- Robust MAD Z-Score Detector

**Algorithm:**  
For each feature column `x`, the robust Z-score for an observation is:
```
robust_z = |x − median(x_train)| / (1.4826 × MAD(x_train))
```
The per-observation score is the **maximum** robust Z across all features.  
A score ≥ 3.5 means at least one feature is more than 3.5 MADs from the training median.

**Why this first?**  
It's transparent, interpretable, and provides per-feature explanations. More complex models are only valuable if they improve on this baseline."""))

cells.append(code("""# --- Fit and score ------------------------------------------------------------
baseline = RobustStatBaseline(threshold=BASELINE_THRESHOLD)

t0 = time.time()
baseline.fit(X_train, feature_names=preprocessor.feature_cols_)
baseline_fit_time = time.time() - t0

print(f\"Baseline fitted in {baseline_fit_time:.4f}s\")
print(f\"  Feature count: {len(baseline.feature_names_)}\")
print(f\"  Training median HR (scaled): {baseline.medians_[preprocessor.feature_cols_.index('HR')]:.4f}\")
print(f\"  Training MAD HR (scaled):    {baseline.mads_[preprocessor.feature_cols_.index('HR')]:.6f}\")"""))

cells.append(code("""# --- Score the test set -------------------------------------------------------
t0 = time.time()
baseline_scores_test = baseline.score_samples(X_test)
baseline_score_time  = time.time() - t0

print(f\"Scored {len(baseline_scores_test):,} test observations in {baseline_score_time:.4f}s\")
print()

# Distribution summary
summary = analyse_score_distribution(
    baseline_scores_test, detector_name='RobustStatBaseline',
    alert_threshold=BASELINE_THRESHOLD
)
print_score_summary(summary)"""))

cells.append(code("""# --- Visualise baseline scores over time --------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: HR over time
axes[0].plot(df_test[TIME_COL], df_test['HR'], color='steelblue', linewidth=0.8)
anomaly_mask = baseline_scores_test >= BASELINE_THRESHOLD
if anomaly_mask.any():
    axes[0].scatter(df_test[TIME_COL].values[anomaly_mask],
                    df_test['HR'].values[anomaly_mask],
                    color='red', s=12, zorder=5, alpha=0.8, label='Baseline flag')
axes[0].set_ylabel('HR (bpm)')
axes[0].set_title('Heart Rate in Test Period with Baseline Anomaly Flags')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Bottom: Baseline scores
plot_anomaly_scores(
    baseline_scores_test, 
    time_values=df_test[TIME_COL].values,
    threshold=BASELINE_THRESHOLD,
    detector_name='RobustStatBaseline',
    ax=axes[1]
)

plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / 'baseline_scores.png', dpi=120, bbox_inches='tight')
plt.show()
print(f\"\\nTest alerts (threshold={BASELINE_THRESHOLD}): {anomaly_mask.sum():,} \")"""))

cells.append(code("""# --- Top contributing features for the most anomalous test observation ---------
most_anomalous_idx = int(np.argmax(baseline_scores_test))
print(f\"Most anomalous test observation: row {most_anomalous_idx}\")
print(f\"  Time:      {df_test.iloc[most_anomalous_idx][TIME_COL]:.2f}s\")
print(f\"  Condition: {df_test.iloc[most_anomalous_idx][CONDITION_COL]}\")
print(f\"  HR:        {df_test.iloc[most_anomalous_idx]['HR']:.2f} bpm\")
print(f\"  Score:     {baseline_scores_test[most_anomalous_idx]:.4f}\")
print()

top_feats = baseline.top_contributing_features(X_test[[most_anomalous_idx]], n_top=10)
print(\"Top 10 contributing features:\")
print(top_feats.to_string(index=False))

# Visualise feature contributions
feat_scores_row = baseline.feature_scores(X_test[[most_anomalous_idx]]).values[0]
fig, ax = plt.subplots(figsize=(10, 6))
plot_feature_contributions(
    preprocessor.feature_cols_, feat_scores_row,
    title=f\"Feature Contributions -- Most Anomalous Test Observation (idx={most_anomalous_idx})\",
    ax=ax,
    save_path=ARTIFACTS_DIR / 'feature_contributions.png'
)
plt.tight_layout()
plt.show()"""))

cells.append(code("""# --- Save baseline ------------------------------------------------------------
baseline.save(ARTIFACTS_DIR / 'robust_baseline.joblib')"""))

# ============================================================
# SECTION 12: Classical Anomaly Detection -- Isolation Forest
# ============================================================
cells.append(md("""## 12. Classical Anomaly Detection -- Isolation Forest

**Algorithm:**  
Isolation Forest builds random decision trees that recursively partition the feature space. Anomalous observations require **fewer splits** to isolate because they are rare and far from the main cluster.

**Why this after the baseline?**  
The baseline looks at each feature independently. Isolation Forest considers ALL features simultaneously -- it can detect unusual COMBINATIONS of otherwise normal-looking measurements (multivariate anomalies).

**Score convention:** After normalisation, 0 = most normal, 1 = most anomalous."""))

cells.append(code("""# --- Fit Isolation Forest -----------------------------------------------------
iforest = IsolationForestDetector(
    n_estimators=200,        # More trees = more stable scores
    contamination='auto',    # Let sklearn determine expected anomaly rate
    random_state=RANDOM_SEED
)

t0 = time.time()
iforest.fit(X_train)
iforest_fit_time = time.time() - t0

print(f\"IsolationForest fitted in {iforest_fit_time:.2f}s  ({200} trees)\")"""))

cells.append(code("""# --- Score test set -----------------------------------------------------------
t0 = time.time()
iforest_scores_test = iforest.score_samples(X_test)
iforest_score_time  = time.time() - t0

print(f\"Scored {len(iforest_scores_test):,} observations in {iforest_score_time:.4f}s\")
summary_if = analyse_score_distribution(
    iforest_scores_test, detector_name='IsolationForest',
    alert_threshold=IFOREST_THRESHOLD
)
print_score_summary(summary_if)"""))

cells.append(code("""# --- Visualise IsolationForest scores -----------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))
plot_anomaly_scores(
    iforest_scores_test,
    time_values=df_test[TIME_COL].values,
    threshold=IFOREST_THRESHOLD,
    detector_name='IsolationForest',
    ax=ax,
    save_path=ARTIFACTS_DIR / 'iforest_scores.png'
)
plt.tight_layout()
plt.show()"""))

cells.append(code("""# --- Save IsolationForest -----------------------------------------------------
iforest.save(ARTIFACTS_DIR / 'isolation_forest.joblib')"""))

# ============================================================
# SECTION 13: Neural Model -- Autoencoder
# ============================================================
cells.append(md("""## 13. Neural Anomaly Detection -- Autoencoder

**Algorithm:**  
An autoencoder is a neural network trained to compress input observations into a small "latent" representation and then reconstruct the original. Trained on normal observations, it learns to reconstruct normal patterns accurately.

When shown an **anomalous** observation, the compression bottleneck cannot capture the unusual pattern, leading to **high reconstruction error** -> high anomaly score.

**Architecture:**
```
Input (n_features) -> Dense(32, ReLU) -> Dense(8, ReLU) [bottleneck]
                   -> Dense(32, ReLU) -> Dense(n_features)
```

**Fallback:** If PyTorch is not installed, we use a PCA-based reconstruction approximation -- a valid linear analogue of the autoencoder."""))

cells.append(code("""# --- Train Autoencoder --------------------------------------------------------
autoencoder = SimpleAutoencoder(
    hidden_dims=AE_HIDDEN_DIMS,
    epochs=AE_EPOCHS,
    batch_size=AE_BATCH_SIZE,
    learning_rate=AE_LR,
    random_state=RANDOM_SEED,
)

print(f\"Training Autoencoder (backend will be selected automatically)...\")
print(f\"  Architecture: {X_train.shape[1]} -> {' -> '.join(str(d) for d in AE_HIDDEN_DIMS)} -> {X_train.shape[1]}\")
print(f\"  Training rows: {X_train.shape[0]:,}  |  Epochs: {AE_EPOCHS}  |  Batch: {AE_BATCH_SIZE}\")
print()

t0 = time.time()
autoencoder.fit(X_train)
ae_fit_time = time.time() - t0

print(f\"\\nAutoencoder trained in {ae_fit_time:.2f}s  (backend: {autoencoder._backend})\")"""))

cells.append(code("""# --- Score test set -----------------------------------------------------------
t0 = time.time()
ae_scores_test = autoencoder.score_samples(X_test)
ae_score_time  = time.time() - t0

print(f\"Scored {len(ae_scores_test):,} observations in {ae_score_time:.4f}s\")
summary_ae = analyse_score_distribution(
    ae_scores_test, detector_name='Autoencoder',
    alert_threshold=AUTOENCODER_THRESHOLD
)
print_score_summary(summary_ae)"""))

cells.append(code("""# --- Visualise Autoencoder scores ---------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))
plot_anomaly_scores(
    ae_scores_test,
    time_values=df_test[TIME_COL].values,
    threshold=AUTOENCODER_THRESHOLD,
    detector_name='Autoencoder (PCA fallback)',
    ax=ax,
    save_path=ARTIFACTS_DIR / 'autoencoder_scores.png'
)
plt.tight_layout()
plt.show()"""))

cells.append(code("""# --- Save Autoencoder ---------------------------------------------------------
autoencoder.save(ARTIFACTS_DIR / 'autoencoder.joblib')"""))

# ============================================================
# SECTION 14: Model Comparison
# ============================================================
cells.append(md("""## 14. Model Comparison

We compare the three detectors on score distributions before running the synthetic anomaly tests.  
We cannot compute precision/recall on the real data because we have no ground-truth anomaly labels -- this is an unsupervised problem."""))

cells.append(code("""# --- Score distribution comparison --------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

detectors = [
    ('RobustStatBaseline', baseline_scores_test / baseline_scores_test.max(), BASELINE_THRESHOLD / baseline_scores_test.max()),
    ('IsolationForest',    iforest_scores_test,  IFOREST_THRESHOLD),
    ('Autoencoder',        ae_scores_test / max(ae_scores_test.max(), 1), AUTOENCODER_THRESHOLD),
]

for ax, (name, scores, thresh) in zip(axes, detectors):
    plot_score_distribution(scores, threshold=thresh, detector_name=name, ax=ax)
    
plt.suptitle('Score Distributions -- All Detectors (Test Set)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / 'score_distributions_comparison.png', dpi=120, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# --- Alert rate comparison ----------------------------------------------------
print(\"Alert rates on real test data (no ground truth -- descriptive only):\")
print(f\"  Baseline:       {(baseline_scores_test >= BASELINE_THRESHOLD).mean()*100:.2f}%\")
print(f\"  IsolationForest:{(iforest_scores_test  >= IFOREST_THRESHOLD).mean()*100:.2f}%\")
print(f\"  Autoencoder:    {(ae_scores_test       >= AUTOENCODER_THRESHOLD).mean()*100:.2f}%\")
print()
print(\"NOTE: Alert rate on real data is informational only. We cannot validate\")
print(\"      these against ground truth without verified anomaly labels.\")
print(\"      Controlled synthetic anomaly tests (Section 16) provide the actual\")
print(\"      detection performance evaluation.\")"""))

# ============================================================
# SECTION 15: Safety Rules
# ============================================================
cells.append(md("""## 15. Physiological Safety Rules

The safety rule layer applies configurable domain-informed thresholds to RAW (unscaled) feature values.

**Design principle:** Safety rules fire deterministically -- they do not depend on learned statistics. A rule triggers even if the ML model considers the observation normal (e.g., because the model has insufficient training data to recognise the extreme value as unusual).

> [!]️ **Disclaimer:** All rule thresholds below are ILLUSTRATIVE ENGINEERING EXAMPLES for prototype development. They are NOT validated NASA clinical standards or universal medical thresholds. Any operational use requires qualified medical review."""))

cells.append(code("""# --- Apply safety rules to test set -------------------------------------------
checker = SafetyRuleChecker()

# Apply to the raw (unscaled) test DataFrame
safety_alerts = checker.check(df_test.reset_index(drop=True))

print(f\"Total safety rule alerts in test set: {len(safety_alerts)}\")
print()

if len(safety_alerts) > 0:
    print(\"Alert severity breakdown:\")
    print(safety_alerts['severity'].value_counts().to_dict())
    print()
    print(\"Sample alerts:\")
    print(safety_alerts.head(10).to_string(index=False))
else:
    print(\"No safety rule violations in the test set.\")
    print(\"(HR values remain within illustrative bounds -- expected for typical HRV data.)\")"""))

# ============================================================
# SECTION 16: Ensemble Scoring
# ============================================================
cells.append(md("""## 16. Threshold Selection and Ensemble Scoring

We combine the three detector scores into a single composite anomaly score.

**Combination method:** Simple average after normalising each score to [0, 1].  
This gives equal weight to each detector. The composite score inherits the strengths of all three:
- Baseline: per-feature extremes
- IsolationForest: multivariate outliers
- Autoencoder: unusual pattern reconstruction

**Alert tiers:**
| Composite Score | Alert Tier |
|----------------|-----------|
| ≥ 0.75 | [HIGH] high_priority |
| 0.50 – 0.75 | [REVIEW] review |
| < 0.50 | [INFO] informational |"""))

cells.append(code("""# --- Normalise all scores to [0, 1] -------------------------------------------
# Each detector already returns normalised scores, but we apply an extra
# clip to ensure strict [0,1] range after any edge-case extrapolation.

scores_dict = {
    'baseline':    np.clip(baseline_scores_test / max(np.percentile(baseline_scores_test, 99), 1e-6), 0, 1),
    'iforest':     np.clip(iforest_scores_test, 0, 1),
    'autoencoder': np.clip(ae_scores_test / max(np.percentile(ae_scores_test, 99), 1e-6), 0, 1),
}

scorer = EnsembleScorer()
ensemble_df = scorer.combine(scores_dict)

print(\"Composite score distribution:\")
composite = ensemble_df['composite_score'].values
print(f\"  Mean: {composite.mean():.4f}  Std: {composite.std():.4f}\")
print(f\"  Min:  {composite.min():.4f}  Max: {composite.max():.4f}\")
print()
print(\"Alert tier counts:\")
print(ensemble_df['alert_tier'].value_counts().to_dict())"""))

cells.append(code("""# --- Build structured alert records -------------------------------------------
alert_records = scorer.build_alert_records(ensemble_df, meta_test.reset_index(drop=True))

# Show the top-10 most anomalous observations
top_alerts = alert_records.nlargest(10, 'composite_score')
print(\"Top 10 most anomalous observations in test set:\")
display_cols = ['Time', 'condition', 'HR', 'baseline_score', 'iforest_score',
                'autoencoder_score', 'composite_score', 'alert_tier']
available = [c for c in display_cols if c in top_alerts.columns]
print(top_alerts[available].round(4).to_string(index=False))"""))

cells.append(code("""# --- Composite score over time ------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))
plot_anomaly_scores(
    composite,
    time_values=df_test[TIME_COL].values,
    threshold=COMPOSITE_THRESHOLD,
    detector_name='Composite Ensemble',
    ax=ax,
    save_path=ARTIFACTS_DIR / 'composite_scores.png'
)
plt.tight_layout()
plt.show()"""))

# ============================================================
# SECTION 17: Evaluation -- Synthetic Anomaly Tests
# ============================================================
cells.append(md("""## 17. Evaluation -- Synthetic Anomaly Tests

Since we have no ground-truth anomaly labels on the real data, we **inject known synthetic anomalies** and measure detection performance.

**Test types:**
1. `extreme_value` -- Shift features 8× standard deviation above normal
2. `sudden_change` -- Single large spike in one random feature per row
3. `unusual_combo` -- Simultaneously drive some features high and others low

**Ground truth:** Injected rows are labeled = 1, original rows = 0.  
We can now compute precision, recall, and F1 against these known labels.

> [!]️ These tests validate that the **pipeline logic** works correctly. They do NOT prove clinical accuracy on real anomalies."""))

cells.append(code("""# --- Run synthetic anomaly tests ----------------------------------------------
anomaly_types = ['extreme_value', 'sudden_change', 'unusual_combo']
comparison_results = []

for anomaly_type in anomaly_types:
    print(f\"\\n{'='*55}\")
    print(f\"Test: {anomaly_type.upper().replace('_', ' ')}\")
    print(f\"{'='*55}\")
    
    df_injected, inj_labels = inject_synthetic_anomalies(
        df_test,
        feature_cols=preprocessor.feature_cols_,
        anomaly_type=anomaly_type,
        n_anomalies=N_SYNTHETIC_ANOMALIES,
        random_state=RANDOM_SEED,
        shift_multiplier=8.0,
    )
    
    # Transform injected data with the SAME fitted preprocessor
    X_injected, _ = preprocessor.transform(df_injected)
    
    # Score with all three detectors
    scores_b  = baseline.score_samples(X_injected)
    scores_if = iforest.score_samples(X_injected)
    scores_ae = ae_scores = autoencoder.score_samples(X_injected)
    
    # Normalised composite
    norm_b  = np.clip(scores_b  / max(np.percentile(baseline_scores_test, 99), 1e-6), 0, 1)
    norm_if = np.clip(scores_if, 0, 1)
    norm_ae = np.clip(scores_ae / max(np.percentile(ae_scores_test, 99), 1e-6), 0, 1)
    comp    = (norm_b + norm_if + norm_ae) / 3.0
    
    for det_name, det_scores, det_thresh in [
        ('Baseline',       scores_b,  BASELINE_THRESHOLD),
        ('IsolationForest',scores_if, IFOREST_THRESHOLD),
        ('Autoencoder',    scores_ae, AUTOENCODER_THRESHOLD),
        ('Composite',      comp,      COMPOSITE_THRESHOLD),
    ]:
        res = evaluate_detection(det_scores, inj_labels, threshold=det_thresh, detector_name=det_name)
        res['anomaly_type'] = anomaly_type
        comparison_results.append(res)
        print_detection_results(res)"""))

cells.append(code("""# --- Summary comparison table -------------------------------------------------
results_df = pd.DataFrame(comparison_results)

print(\"\\n=== DETECTION RATE SUMMARY (% of synthetic anomalies detected) ===\")
pivot = results_df.pivot(
    index='anomaly_type',
    columns='detector',
    values='detection_rate_pct'
).round(1)
print(pivot.to_string())

print(\"\\n=== PRECISION SUMMARY ===\")
pivot_p = results_df.pivot(
    index='anomaly_type',
    columns='detector',
    values='precision'
).round(3)
print(pivot_p.to_string())"""))

cells.append(code("""# --- Model comparison bar chart (composite detection rate) --------------------
composite_results = results_df[
    (results_df['detector'] == 'Composite') &
    (results_df['anomaly_type'] == 'extreme_value')
]

# Compare detectors on extreme_value test
extreme_results = results_df[results_df['anomaly_type'] == 'extreme_value'].copy()
extreme_results = extreme_results[['detector', 'detection_rate_pct', 'precision', 'f1']].copy()

fig, ax = plt.subplots(figsize=(10, 4))
plot_model_comparison(
    extreme_results,
    metric='detection_rate_pct',
    ax=ax,
    save_path=ARTIFACTS_DIR / 'model_comparison.png'
)
ax.set_title('Detection Rate -- Extreme Value Injection Test (%)')
plt.tight_layout()
plt.show()

# Save results
results_df.to_csv(ARTIFACTS_DIR / 'evaluation_results.csv', index=False)
print(\"Evaluation results saved to artifacts/evaluation_results.csv\")"""))

# ============================================================
# SECTION 18: Visualisations and Interpretation
# ============================================================
cells.append(md("## 18. Visualisations and Interpretation"))

cells.append(code("""# --- Full dashboard -----------------------------------------------------------
fig = plot_full_dashboard(
    df=df_test,
    scores_dict={
        'Baseline':        np.clip(baseline_scores_test / max(np.percentile(baseline_scores_test,99), 1e-6), 0, 1),
        'IsolationForest': iforest_scores_test,
        'Autoencoder':     np.clip(ae_scores_test / max(np.percentile(ae_scores_test, 99), 1e-6), 0, 1),
    },
    composite_scores=composite,
    feature_cols=preprocessor.feature_cols_,
    threshold=COMPOSITE_THRESHOLD,
    time_col=TIME_COL,
    primary_feature='HR',
    save_path=ARTIFACTS_DIR / 'dashboard.png'
)
plt.show()"""))

cells.append(code("""# --- Score distributions per condition ----------------------------------------
# Do anomaly scores differ systematically across experimental conditions?
# This is informational -- we do not use condition labels during training.

condition_scores = pd.DataFrame({
    'condition': df_test[CONDITION_COL].values,
    'baseline_score': np.clip(baseline_scores_test / max(np.percentile(baseline_scores_test,99),1e-6), 0, 1),
    'composite_score': composite
})

print(\"Average composite score by condition:\")
print(condition_scores.groupby('condition')['composite_score'].agg(['mean','std','max']).round(4))
print()
print(\"Interpretation:\")
print(\"  The 'stress' condition may show higher scores if physiological patterns\")
print(\"  during stress differ significantly from the training (baseline) period.\")
print(\"  This does NOT mean the system is diagnosing stress -- it means patterns differ.\")

fig, ax = plt.subplots(figsize=(10, 5))
condition_scores.boxplot(column='composite_score', by='condition', ax=ax)
ax.set_title('Composite Anomaly Score Distribution by Experimental Condition')
ax.set_xlabel('Condition')
ax.set_ylabel('Composite Anomaly Score')
plt.suptitle('')  # suppress default boxplot title
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / 'scores_by_condition.png', dpi=120, bbox_inches='tight')
plt.show()"""))

# ============================================================
# SECTION 19: Inference on Unseen Observations
# ============================================================
cells.append(md("""## 19. Example Inference on Unseen Observations

This section demonstrates how to run the full pipeline on new, previously unseen observations.  
This is the interface that would be called by a dashboard or monitoring system."""))

cells.append(code("""def run_inference(
    new_observations: pd.DataFrame,
    preprocessor: AnomalyPreprocessor,
    baseline: RobustStatBaseline,
    iforest: IsolationForestDetector,
    autoencoder: SimpleAutoencoder,
    checker: SafetyRuleChecker,
    baseline_95th: float,
    ae_95th: float,
) -> pd.DataFrame:
    \"\"\"
    Run the full anomaly detection pipeline on new observations.
    
    This function encapsulates the complete inference pipeline:
    1. Preprocess observations using the fitted preprocessor.
    2. Score with all three detectors.
    3. Check physiological safety rules.
    4. Combine into a composite score.
    5. Return structured alert records.
    
    Parameters
    ----------
    new_observations : pd.DataFrame
        Raw observations with the same schema as the training data.
    
    Returns
    -------
    pd.DataFrame of structured alert records (one row per observation).
    \"\"\"
    # Step 1: Preprocess
    X_new, meta_new = preprocessor.transform(new_observations)
    
    # Step 2: Score all detectors
    scores_b  = baseline.score_samples(X_new)
    scores_if = iforest.score_samples(X_new)
    scores_ae = autoencoder.score_samples(X_new)
    
    # Step 3: Safety rules (on raw data)
    safety_alerts = checker.check(new_observations.reset_index(drop=True))
    
    # Step 4: Normalise and combine
    norm_b  = np.clip(scores_b  / max(baseline_95th, 1e-6), 0, 1)
    norm_if = np.clip(scores_if, 0, 1)
    norm_ae = np.clip(scores_ae / max(ae_95th, 1e-6), 0, 1)
    
    scores_dict = {'baseline': norm_b, 'iforest': norm_if, 'autoencoder': norm_ae}
    scorer = EnsembleScorer()
    ensemble_df = scorer.combine(scores_dict)
    
    # Step 5: Build alert records
    alert_records = scorer.build_alert_records(ensemble_df, meta_new.reset_index(drop=True))
    
    # Attach raw scores for transparency
    alert_records['baseline_raw_score'] = scores_b
    alert_records['safety_rule_count']  = len(safety_alerts)
    
    return alert_records


# --- Compute normalisation statistics from test set ---------------------------
baseline_95th = float(np.percentile(baseline_scores_test, 99))
ae_95th       = float(np.percentile(ae_scores_test, 99))

# --- Run inference on last 5 rows of test set (unseen during scoring loop) ----
sample_unseen = df_test.tail(5).copy()

print(\"Running inference on 5 sample observations...\")
results = run_inference(
    sample_unseen, preprocessor, baseline, iforest, autoencoder,
    checker, baseline_95th, ae_95th
)

print(\"\\nInference results (structured output):\")
show_cols = [c for c in [
    'Time', 'subject id', 'condition', 'HR',
    'composite_score', 'alert_tier', 'explanation'
] if c in results.columns]
print(results[show_cols].round(4).to_string(index=False))"""))

cells.append(code("""# --- JSON-formatted output (as would be sent to a dashboard API) --------------
print(\"\\nExample structured alert record (JSON format):\")
example_record = results.iloc[results['composite_score'].argmax()]
output = {
    \"subject_id\":        str(example_record.get('subject id', 'unknown')),
    \"timestamp_s\":       float(example_record.get('Time', 0.0)),
    \"condition\":         str(example_record.get('condition', 'unknown')),
    \"anomaly_score\":     round(float(example_record['composite_score']), 4),
    \"alert_status\":      str(example_record['alert_tier']),
    \"baseline_score\":    round(float(example_record['baseline_score']), 4),
    \"iforest_score\":     round(float(example_record['iforest_score']), 4),
    \"autoencoder_score\": round(float(example_record['autoencoder_score']), 4),
    \"explanation\":       str(example_record['explanation']),
    \"data_quality_warning\": None,
    \"model_version\":     \"v1.0-prototype\",
    \"disclaimer\":        \"Research prototype. Not a medical device. All alerts require qualified human review.\"
}
print(json.dumps(output, indent=2))"""))

# ============================================================
# SECTION 20: Save / Reload Artifacts
# ============================================================
cells.append(md("## 20. Saving and Reloading Model Artifacts"))

cells.append(code("""# --- Verify all artifacts exist -----------------------------------------------
expected_artifacts = [
    'preprocessor.joblib',
    'robust_baseline.joblib',
    'isolation_forest.joblib',
    'autoencoder.joblib',
    'evaluation_results.csv',
]

print(\"=== Artifact verification ===\")
all_ok = True
for artifact in expected_artifacts:
    path = ARTIFACTS_DIR / artifact
    exists = path.exists()
    size_kb = path.stat().st_size / 1024 if exists else 0
    status = '[OK]' if exists else '[X] MISSING'
    print(f\"  {status}  {artifact}  ({size_kb:.1f} KB)\")
    if not exists:
        all_ok = False

print()
print(\"All artifacts present [OK]\" if all_ok else \"[!] Some artifacts are missing!\")"""))

cells.append(code("""# --- Demonstrate reload and inference -----------------------------------------
from src.preprocessing import AnomalyPreprocessor
from src.baseline import RobustStatBaseline
from src.anomaly_detector import IsolationForestDetector, SimpleAutoencoder

print(\"Loading saved artifacts from disk...\")
prep_loaded = AnomalyPreprocessor.load(ARTIFACTS_DIR / 'preprocessor.joblib')
base_loaded = RobustStatBaseline.load(ARTIFACTS_DIR / 'robust_baseline.joblib')
if_loaded   = IsolationForestDetector.load(ARTIFACTS_DIR / 'isolation_forest.joblib')
ae_loaded   = SimpleAutoencoder.load(ARTIFACTS_DIR / 'autoencoder.joblib')
print(\"All artifacts loaded successfully [OK]\")

# Quick verification: scores must match original
X_verify, _ = prep_loaded.transform(df_test.head(50))
scores_verify = base_loaded.score_samples(X_verify)
scores_orig   = baseline.score_samples(X_train[:0].reshape(0, X_train.shape[1]))  # empty
scores_orig2  = baseline.score_samples(X_verify)

max_diff = np.abs(scores_verify - scores_orig2).max()
print(f\"Max score difference after reload: {max_diff:.10f}  (should be ~0)\")
assert max_diff < 1e-8, \"Reload produced different scores -- serialisation error!\"
print(\"Reload verification passed [OK]\")"""))

# ============================================================
# SECTION 21: Limitations and Future Work
# ============================================================
cells.append(md("""## 21. Limitations and Future Work

### Current Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Single subject (ID=2) | Cannot test cross-subject generalization | Add more subjects |
| No verified anomaly labels | Cannot compute true precision/recall on real data | Expert annotation or ground truth events |
| Condition labels used for EDA only | Stress conditions appear as anomalies when model trained on baseline | Consider condition-aware training |
| PCA autoencoder fallback | Less expressive than a true neural network | Install PyTorch for full autoencoder |
| Safety rules are illustrative | Not validated clinical standards | Medical expert review required |
| Static threshold selection | Optimal threshold may vary by subject/condition | Adaptive or learned thresholds |
| No real-time streaming | Batch inference only | Implement sliding-window streaming pipeline |

### Recommended Next Steps

1. **Collect more subjects** -- Test cross-subject generalization.
2. **Obtain expert annotations** -- Label known anomalous events for supervised evaluation.
3. **Install PyTorch** -- Train the full neural autoencoder (not PCA fallback).
4. **Add temporal model** -- LSTM or Temporal Convolutional Network to capture sequential patterns.
5. **Implement rolling-window baseline** -- Update reference statistics as new data arrives.
6. **Dashboard integration** -- Connect the `run_inference()` function to a web API.
7. **Medical review of safety rules** -- Validate thresholds with clinical domain experts.
8. **Add sleep/activity context** -- Physiological signals differ significantly between activity states.

### Data Provenance

The dataset `dummy_data/test_sample.csv` is a synthetic/representative HRV dataset containing one subject across four experimental conditions (baseline, stress, amusement, meditation). It is NOT verified real astronaut data and must NOT be treated as such for any medical or operational purposes."""))

# ============================================================
# SECTION 22: Final Summary
# ============================================================
cells.append(md("## 22. Final Summary"))

cells.append(code("""# --- Final summary printout ---------------------------------------------------
print(\"=\"*60)
print(\"ASTRONAUT HEALTH MONITOR -- PROTOTYPE SUMMARY\")
print(\"=\"*60)
print()
print(f\"Dataset: {DATA_PATH.name}\")
print(f\"  Rows: {len(df):,}  |  Subjects: {df[SUBJECT_COL].nunique()}  |  Conditions: {df[CONDITION_COL].nunique()}\")
print()
print(f\"Training set: {len(df_train):,} rows  |  Test set: {len(df_test):,} rows\")
print(f\"Features: {len(preprocessor.feature_cols_)} selected from {df.shape[1]} original columns\")
print()
print(\"Models trained:\")
print(f\"  1. RobustStatBaseline     (fit time: {baseline_fit_time:.4f}s)\")
print(f\"  2. IsolationForest-200    (fit time: {iforest_fit_time:.2f}s)\")
print(f\"  3. Autoencoder ({autoencoder._backend}) (fit time: {ae_fit_time:.2f}s)\")
print()

# Composite alert summary on test set
n_high = (ensemble_df['alert_tier'] == 'high_priority').sum()
n_rev  = (ensemble_df['alert_tier'] == 'review').sum()
n_info = (ensemble_df['alert_tier'] == 'informational').sum()
print(f\"Test set alerts (composite, threshold={COMPOSITE_THRESHOLD}):\")
print(f\"  [HIGH] High priority : {n_high:,}  ({n_high/len(ensemble_df)*100:.1f}%)\")
print(f\"  [REVIEW] Review        : {n_rev:,}  ({n_rev/len(ensemble_df)*100:.1f}%)\")
print(f\"  [INFO] Informational : {n_info:,}  ({n_info/len(ensemble_df)*100:.1f}%)\")
print()

# Best synthetic detection result
if len(comparison_results) > 0:
    best = max(comparison_results, key=lambda r: r['f1'])
    print(f\"Best synthetic anomaly detection (F1={best['f1']:.3f}):\")
    print(f\"  Detector: {best['detector']}  |  Test: {best['anomaly_type']}\")
    print(f\"  Recall: {best['recall']:.3f}  |  Precision: {best['precision']:.3f}\")
print()
print(\"Artifacts saved:\")
for a in expected_artifacts:
    path = ARTIFACTS_DIR / a
    if path.exists():
        print(f\"  [OK] {a}\")
print()
print(\"=\"*60)
print(\"DISCLAIMER: This is a research prototype, not a medical device.\")
print(\"All alerts require qualified human review.\")
print(\"=\"*60)"""))

# ============================================================
# Build the notebook JSON
# ============================================================
PROJECT_ROOT = Path(__file__).parent  # project root = same dir as this script

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

# Add IDs to each cell (required by nbformat 4.5+)
import uuid
for c in notebook["cells"]:
    c["id"] = str(uuid.uuid4())[:8]

output_path = PROJECT_ROOT / "notebooks" / "astronaut_health_monitor.ipynb"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {output_path}")
print(f"  Cells: {len(cells)}")
