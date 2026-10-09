# NASA Space Apps Challenge 2026 (Test Prototype) — Astronaut Health Monitoring System

## 1. Project Overview

**Project type:** Astronaut health monitoring and personalized anomaly detection (Test Prototype before actual implementation)
**Primary language:** Python
**Primary development environment:** Jupyter Notebook (`.ipynb`), preferably Google Colab-compatible
**Input dataset:** `dummy_data/test_sample.csv`
**Primary objective:** Build a functional, interpretable, extensible health-monitoring engine that learns an individual's typical physiological behavior from historical data and identifies unusual deviations without requiring labeled examples of every possible abnormal condition.

This project is being developed for the NASA Space Apps Challenge 2026, under the *Create Health Monitoring Software for Astronauts on Space Missions* challenge.

The system should help identify unusual physiological patterns and provide understandable information that supports human review. It must not claim to diagnose medical conditions or replace professional medical judgment.

The implementation must prioritize a working, demonstrable submission over unnecessary complexity.

---

## 2. Developer Context

The primary developer has a background in software engineering, system architecture, and UI/UX design. Python scripts can be executed, but machine learning, deep learning, statistical anomaly detection, and transformer architectures are new learning areas.

Therefore, the implementation must satisfy two goals simultaneously:

1. Deliver a functional prototype within the hackathon deadline.
2. Explain the underlying algorithms and implementation decisions clearly enough that the developer can understand, debug, demonstrate, and defend the system.

Do not assume advanced ML knowledge.

Explain unfamiliar terminology when it first appears. Prefer understandable implementations, explicit intermediate results, meaningful variable names, and modular architecture over clever but opaque code.

Do not sacrifice correctness, reproducibility, or explainability merely to make the project appear sophisticated.

---

## 3. Core Objective

Build a personalized, primarily unsupervised anomaly-detection system.

The system should learn patterns from historical observations and calculate how unusual subsequent observations are relative to those learned patterns.

The intended pipeline is:

1. Load and inspect the supplied CSV.
2. Determine the dataset's schema and measurement semantics.
3. Validate and preprocess the observations.
4. Identify individual subjects, timestamps, and physiological features when available.
5. Establish appropriate historical baselines.
6. Train or fit an anomaly-detection model using the available historical observations.
7. Score previously unseen observations.
8. Identify and rank potentially anomalous observations.
9. Explain which features contributed to each alert when the selected method supports that explanation.
10. Visualize the baseline, observations, anomaly scores, and flagged intervals.
11. Evaluate the prototype using suitable validation procedures.
12. Save reusable preprocessing artifacts, fitted models, evaluation results, and example predictions where appropriate.

The system must distinguish between an unusual measurement and a confirmed health problem. Every alert is a signal for further assessment, not a diagnosis.

---

## 4. Mandatory First Step: Inspect the Dataset

**Do not design the final model before inspecting `dummy_data/test_sample.csv`.**

Begin by verifying that the file exists and loading a manageable sample.

Create an initial dataset profile containing:

- Number of rows and columns.
- Column names and inferred data types.
- Example records.
- Missing-value counts and percentages.
- Duplicate rows and potential duplicate observations.
- Numeric and categorical features.
- Candidate subject identifiers.
- Candidate timestamps and temporal resolution.
- Measurement units, when documented.
- Constant or nearly constant columns.
- Invalid, infinite, or implausible values requiring investigation.
- Potential target, label, metadata, and identifier columns.
- Whether observations are independent records or sequential time-series measurements.

Use pandas for CSV inspection. If the CSV is large, use chunked reading or a representative sample rather than loading everything into memory unnecessarily.

Never silently guess column meanings or physiological units. Infer candidate meanings from column names only as a hypothesis, and clearly identify anything that requires confirmation.

If the sample does not establish what the columns represent, explain what information is missing and proceed only with a defensible generic analysis.

### Dataset limitations

The supplied file is named `dummy_data/test_sample.csv`. It may contain synthetic, anonymized, incomplete, or representative data.

Do not describe it as real astronaut data unless its provenance establishes that fact.

If the file is missing or unreadable, stop before inventing its schema. Provide a useful diagnostic and request the file or a valid path.

---

## 5. Model Architecture: Build a Strong, Modular Engine

Implement a model-comparison pipeline instead of assuming that one advanced architecture is automatically best.

The final architecture must be selected according to the actual dataset, its size, temporal structure, available computing resources, and validation results.

### 5.1 Baseline: Personalized Statistical Anomaly Detection

Implement a transparent baseline before introducing complex models.

Where appropriate, calculate subject-specific robust statistics such as:

- Median.
- Median absolute deviation (MAD).
- Robust standardized deviations.
- Feature-specific anomaly thresholds.

Account for near-zero dispersion, missing observations, insufficient historical samples, and extreme outliers.

This baseline serves as a reference for evaluating more sophisticated methods.

Do not treat fixed physiological thresholds as universal medical standards. Any physiological interpretation requires suitable domain evidence.

### 5.2 Classical Unsupervised Models

Evaluate one or more suitable methods, such as:

- Isolation Forest.
- Local Outlier Factor, when its training and inference behavior fits the pipeline.
- Robust covariance or other statistical methods when their assumptions are appropriate.

Avoid including every available algorithm merely to increase the apparent sophistication of the project.

Compare methods using consistent preprocessing and validation where possible.

### 5.3 Neural Anomaly Detection

Evaluate a neural approach if the dataset supports it.

Potential architectures include:

**A. Autoencoder**

Train a neural network to reconstruct typical observations. Reconstruction error can serve as an anomaly score.

An autoencoder is a reasonable first neural baseline for numeric tabular data when sufficient representative observations are available.

**B. Temporal Autoencoder or Sequence Model**

If observations form meaningful time series, evaluate a model that learns temporal patterns rather than treating every row as independent.

Possible approaches include recurrent models or temporal convolutional networks.

**C. Transformer-Based Time-Series Model**

Consider a transformer only if the dataset contains sufficient sequential information and the computational cost is justified.

Possible approaches include a transformer encoder trained for reconstruction or a suitable forecasting-based anomaly detector.

A transformer is not mandatory. It must not be selected solely because it sounds advanced.

If there is insufficient data to train a transformer reliably, explain the limitation and implement a simpler validated alternative.

### 5.4 Model Selection

Compare suitable candidates on:

- Detection of deliberately introduced synthetic anomalies.
- False-positive behavior.
- Consistency across subjects where applicable.
- Sensitivity to missing or noisy data.
- Stability under different training samples.
- Computational cost.
- Interpretability.
- Suitability for real-time or near-real-time inference.

Synthetic anomaly injection is a development test, not proof of clinical accuracy.

Do not invent performance metrics or claim that a model is medically validated.

If labeled ground truth is unavailable, explicitly state that conventional precision, recall, F1-score, and medical sensitivity cannot be established reliably from unlabeled data alone.

---

## 6. Personalization and Historical Baselines

The intended system is personalized whenever the data supports personalization.

If a reliable subject identifier exists:

- Separate observations by subject for baseline estimation.
- Avoid accidentally mixing one subject's historical measurements into another subject's baseline.
- Consider subject-specific models or a shared model combined with subject-specific calibration.
- Define a minimum amount of historical data before enabling personalized alerts.
- Provide a fallback or insufficient-history status for new subjects.
- Prevent evaluation leakage between training and test periods.

If no subject identifier exists, implement a clearly labeled global or dataset-level baseline and document that true person-specific monitoring cannot yet be demonstrated.

Do not assume that each person requires a separate neural network. Choose between separate models and shared models with personalized calibration based on evidence.

---

## 7. Temporal Analysis

If timestamps and repeated measurements are available, preserve temporal information.

Investigate:

- Sampling intervals.
- Missing time intervals.
- Irregular sampling.
- Short-term changes.
- Rolling statistics.
- Trends and deviations from recent history.
- Delayed or persistent anomalies.
- Relationships among measurements over time.

Do not treat sequential observations as independent when the intended model depends on temporal relationships.

Use chronological training, validation, and test splits wherever appropriate.

A model must not use future observations to construct a baseline for an earlier test observation.

If timestamps are unavailable, do not fabricate temporal structure. Use a tabular approach and document the limitation.

---

## 8. Data Preprocessing Requirements

Build preprocessing as a reusable, testable module.

Address the following where relevant:

- Numeric type conversion.
- Missing-value handling.
- Duplicate detection.
- Invalid and infinite values.
- Categorical encoding.
- Feature selection.
- Feature scaling.
- Subject grouping.
- Timestamp parsing.
- Irregular time intervals.
- Outlier handling during preprocessing.

Important rules:

1. Fit scalers, imputers, encoders, and other learned preprocessing steps using training data only.
2. Apply the same fitted transformations during validation, testing, and inference.
3. Do not automatically delete extreme observations merely because they look unusual; they may be the anomalies the system is intended to detect.
4. Preserve identifiers and timestamps as metadata when they are not appropriate model inputs.
5. Prevent identifier leakage and target leakage.
6. Keep the original records available for interpreting alerts.
7. Document each transformation and why it is needed.

Do not assume that every numeric column represents a meaningful physiological measurement.

---

## 9. Evaluation and Testing

Evaluation is mandatory.

### 9.1 Data Leakage Prevention

Prefer chronological evaluation when the dataset contains time-series data.

For personalized monitoring, establish the baseline from historical observations and evaluate against later observations.

Where the dataset contains multiple subjects, assess whether evaluation should also hold out entire subjects to test generalization to unseen individuals.

Choose the split according to the intended deployment scenario.

### 9.2 Synthetic Anomaly Tests

Create controlled test cases based on the actual schema, when possible.

Examples include:

- A feature value shifted far from its historical distribution.
- A sudden change in a previously stable measurement.
- An unusual combination of otherwise plausible measurements.
- Missing or corrupted measurements.
- A sustained change over several observations.

Use these tests to verify that the pipeline behaves as expected.

Clearly distinguish injected test cases from genuine observations.

### 9.3 Metrics

Use metrics that are appropriate to the available ground truth.

When reliable labels exist, consider precision, recall, F1-score, ROC-AUC, and precision-recall AUC where applicable.

When labels do not exist, report appropriate alternatives such as:

- Anomaly score distributions.
- Alert rates.
- Score stability.
- Results of controlled synthetic anomaly tests.
- Runtime and memory usage.
- Qualitative review of flagged observations.

An anomaly score is not automatically a probability that a medical event will occur.

### 9.4 Required Tests

Test at least:

- Successful dataset loading.
- Missing or malformed input files.
- Missing values.
- Constant-valued features.
- Insufficient historical data.
- Empty datasets.
- Single-row datasets.
- Previously unseen observations.
- Consistent preprocessing during inference.
- Model save/load behavior when persistence is implemented.
- Prediction output schema and finite scores.

Fail gracefully with actionable error messages.

---

## 10. Required Notebook Deliverable

**The primary deliverable must be a fully executable Jupyter Notebook.**

Create:

`notebooks/astronaut_health_monitor.ipynb`

The notebook must be organized into numbered Markdown and code sections.

Recommended structure:

1. Project introduction and objectives.
2. Environment setup and imports.
3. Configuration and reproducibility settings.
4. Dataset path validation.
5. Dataset loading.
6. Exploratory data analysis.
7. Data quality report.
8. Feature and schema interpretation.
9. Preprocessing pipeline.
10. Train/validation/test strategy.
11. Statistical baseline.
12. Classical anomaly-detection models.
13. Neural model, if justified by the dataset.
14. Model comparison.
15. Threshold selection and calibration.
16. Evaluation and synthetic anomaly tests.
17. Visualizations and interpretation.
18. Example inference on unseen observations.
19. Saving and reloading model artifacts.
20. Limitations and future work.
21. Final summary.

Every section must explain its purpose.

The notebook must execute from top to bottom in a clean environment, without relying on variables created manually in earlier interactive sessions.

Include clear plots with labels, legends, units when known, and useful titles.

Use fixed random seeds where practical.

Avoid installing unnecessary libraries or downloading large pretrained models without justification.

### Notebook quality requirements

- No placeholder implementations presented as finished features.
- No hidden dependencies on local files that are not documented.
- No unexplained magic constants.
- No silently swallowed exceptions.
- No fabricated experimental results.
- No claims that training succeeded unless it actually ran.
- No code that requires a GPU unless an appropriate fallback exists or the GPU requirement is explicitly documented.

If a section cannot be completed because the dataset is missing or its schema is unknown, explain that limitation and provide a safe, executable diagnostic rather than fabricating results.

---

## 11. Project Structure

Use a simple modular structure that can evolve into an integrated application.

Recommended layout:

```
astronaut-health-monitor/
├── PROJECT_CONTEXT.md
├── dummy_data/
│   └── test_sample.csv
├── notebooks/
│   └── astronaut_health_monitor.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── baseline.py
│   ├── anomaly_detector.py
│   ├── evaluation.py
│   └── visualization.py
├── tests/
│   └── test_pipeline.py
├── artifacts/
│   └── .gitkeep
├── requirements.txt
└── README.md
```

The notebook should be the primary development and demonstration surface.

Extract reusable logic into `src/` modules when doing so improves maintainability and enables testing.

Do not create empty modules simply to match the proposed structure. Implement only the files needed by the current working prototype.

Use relative paths derived from a documented project root or an explicit configurable data path. Do not hardcode machine-specific absolute paths.

Keep generated model artifacts separate from source code and raw datasets.

---

## 12. Runtime and Resource Constraints

The prototype must run on an ordinary CPU-based development environment when feasible.

Optimize for rapid experimentation and a successful hackathon demonstration.

Requirements:

- Start with a representative subset of the dataset.
- Measure data-loading and training time.
- Avoid loading large files repeatedly.
- Use efficient numerical operations rather than unnecessary Python loops.
- Keep neural models appropriately small for the available data.
- Avoid expensive hyperparameter searches.
- Avoid large pretrained models unless they solve a demonstrated problem.
- Allow configuration of sample size, model parameters, and random seeds.
- Expand to the full dataset only after the pipeline works correctly.

A compact, validated model is preferable to a large model that cannot finish training before submission.

---

## 13. Dashboard and Integration Readiness

The project may later be integrated with a frontend dashboard.

Design the inference interface to return structured results rather than unformatted console messages.

A prediction record should include applicable fields such as:

- Subject identifier.
- Timestamp.
- Anomaly score.
- Alert status.
- Relevant feature deviations.
- Data-quality warnings.
- Model identifier or version.
- Explanation of insufficient history when applicable.

For example, a result could conceptually look like:

```
{
    "subject_id": "sample_subject_01",
    "timestamp": "2026-01-01T12:00:00",
    "anomaly_score": 0.91,
    "alert_status": "review",
    "explanation": "Unusual deviation from the learned baseline",
    "data_quality_warning": null
}
```

This is an illustrative output schema, not a claim that the supplied dataset contains these fields.

The implementation must define and document the actual score direction, normalization, and threshold semantics. Do not assume that higher scores always indicate anomalies across every algorithm.

Keep the ML engine independent of the dashboard framework.

---

## 14. Medical Safety and Responsible Interpretation

This is a research and demonstration prototype, not a certified medical device.

The system must:

- Avoid diagnosing diseases.
- Avoid asserting that an anomaly proves an astronaut is unhealthy.
- Distinguish unusual observations from data-quality errors.
- Account for activity, sleep, environmental changes, and other context when available.
- Communicate uncertainty.
- Avoid unsupported universal physiological thresholds.
- Document dataset provenance and limitations.
- Avoid treating synthetic observations as genuine astronaut measurements.
- Recommend qualified human review rather than automated medical decisions.

Any operational medical threshold or clinical interpretation requires appropriate domain validation.

---

## 15. Coding Model Instructions

Act as a senior ML engineer, Python developer, and patient technical mentor.

Follow this implementation order:

1. Inspect the existing project files and supplied CSV.
2. Report the schema and identify the most important unknowns.
3. Propose the smallest viable architecture based on the actual data.
4. Create the notebook and implement the first working baseline.
5. Execute and verify the code where execution tools are available.
6. Add evaluation and synthetic anomaly tests.
7. Compare suitable models.
8. Add a neural or transformer-based approach only when justified.
9. Refactor reusable logic into modules as needed.
10. Document execution, outputs, limitations, and next steps.

### Strict implementation rules

- Do not invent dataset columns or fabricate experimental outputs.
- Do not generate a massive codebase before verifying the first working pipeline.
- Do not claim that code was executed or tested if it was not.
- Do not introduce a transformer solely to make the project sound advanced.
- Do not remove the baseline just because a neural model is available.
- Do not use random train/test splits for temporally dependent data without a clear justification.
- Do not leak test data into preprocessing or training.
- Do not silently change the input dataset.
- Do not assume a GPU is available.
- Do not hide errors that prevent valid inference.
- Do not use unlabeled anomaly scores as proof of medical accuracy.

When presenting implementation changes, identify the affected files and explain how to run and verify them.

When teaching, explain each important code section in plain English, including:

- What it does.
- Why it is necessary.
- What its inputs and outputs are.
- Which assumptions it makes.
- What could go wrong.
- How to verify that it worked.

Prioritize a correct, demonstrable system over architectural perfection.

---

## 16. Definition of Done

The initial prototype is considered complete when:

- [ ] The supplied CSV is loaded successfully, or a clear missing-file diagnostic is provided.
- [ ] The schema and data-quality report are generated.
- [ ] Preprocessing is reproducible and reusable.
- [ ] A working statistical anomaly baseline exists.
- [ ] At least one suitable unsupervised model is evaluated when the data supports it.
- [ ] A neural approach is evaluated only when justified.
- [ ] Anomaly scores and thresholds are documented.
- [ ] Evaluation avoids data leakage.
- [ ] Synthetic anomaly tests run successfully where feasible.
- [ ] Relevant plots are generated.
- [ ] The notebook executes from top to bottom.
- [ ] Example inference works on unseen observations.
- [ ] Model persistence works if included.
- [ ] Limitations and data provenance are documented.
- [ ] The developer can explain the major components.
- [ ] The prototype is ready to integrate with the team's presentation or dashboard.

The initial goal is a reliable working demonstration, not a clinically validated astronaut-monitoring system.

---

## 17. Immediate Mission

Begin now.

**First action:** locate and inspect `dummy_data/test_sample.csv`.

**Second action:** create `notebooks/astronaut_health_monitor.ipynb`.

**Third action:** implement and run the smallest valid data-inspection and baseline-detection pipeline.

Do not spend the first several hours on theory, architecture diagrams, dependency installation, or transformer research.

Build something small that runs. Verify it. Understand it. Improve it.

**Success criterion:** a working, explainable, reproducible anomaly-detection prototype that can be demonstrated honestly within the hackathon deadline.

---

## 18. Advanced Algorithm Requirements, Code Documentation, and Physiological Safety Rules

### 18.1 Mandatory Algorithm and Workflow Comments

Every major algorithm, preprocessing stage, training procedure, inference pipeline, and evaluation method must contain meaningful code comments.

Comments must explain **the algorithm's reasoning and workflow**, not merely restate what the next line of code does.

For every important component, document:

- **Purpose:** What problem does this component solve?
- **Algorithm:** Which method is being used, and how does it work conceptually?
- **Workflow:** What happens to the data, step by step?
- **Inputs and outputs:** What enters and leaves the component?
- **Assumptions:** What conditions must hold for the method to be valid?
- **Limitations:** What can cause incorrect results?
- **Design decisions:** Why was this method chosen instead of a plausible alternative?

Include explanatory Markdown cells in the notebook alongside inline comments in the implementation.

For complex algorithms, include concise pseudocode or mathematical intuition where useful. Explain unfamiliar ML terminology in beginner-friendly language.

Do not clutter every line with redundant comments. Prioritize algorithmic reasoning, non-obvious transformations, critical decisions, and error-handling logic.

### 18.2 Use Advanced, Complementary Anomaly-Detection Methods

Do not rely exclusively on mean, standard deviation, or fixed per-feature thresholds as the final anomaly-detection solution.

Implement a layered detection architecture using multiple complementary methods when the available dataset supports them.

Evaluate suitable combinations of:

1. **Robust statistical detection:** Median, median absolute deviation, robust scaling, and other appropriate statistical methods. Use these as transparent baselines and supporting signals, not as the sole final engine.
2. **Isolation Forest:** Detect observations that are unusually easy to isolate from the learned distribution.
3. **Multivariate anomaly detection:** Identify unusual combinations of physiological measurements that may appear normal when each feature is examined independently.
4. **Neural reconstruction models:** Evaluate an autoencoder trained to reconstruct representative observations, using reconstruction error as one anomaly signal.
5. **Temporal anomaly detection:** Where timestamps and sequential observations are available, detect unusual trends, transitions, persistence, and deviations from learned temporal patterns.
6. **Transformer-based detection:** Evaluate a transformer encoder or another suitable attention-based temporal architecture if sufficient sequential training data and computational resources are available.

Do not automatically activate every algorithm. Select models based on dataset characteristics, validation results, runtime, and complementary detection capabilities.

### 18.3 Multi-Signal Anomaly Scoring

Where justified, combine complementary anomaly signals into a documented scoring or decision system.

A potential architecture is:

- A statistical detector identifies extreme feature-level deviations.
- A multivariate detector identifies unusual combinations of features.
- A temporal detector identifies unexpected changes over time.
- A neural detector identifies observations poorly represented by learned normal patterns.
- A decision layer combines calibrated signals and assigns an alert priority.

Before combining scores, normalize or calibrate them appropriately. Different algorithms can produce scores with different ranges, directions, and meanings.

Do not simply add raw model scores together.

Document the combination method, calibration procedure, alert threshold, and trade-offs between false alarms and missed anomalies.

If there is insufficient data to train a reliable ensemble, use the strongest validated combination available and explicitly document which components remain experimental.

### 18.4 Physiological Safety Rules and Domain-Informed Alerts

Incorporate documented physiological knowledge as a separate, configurable safety-rule layer alongside learned anomaly detection.

The purpose of this layer is to recognize potentially important conditions that a model might overlook, particularly when historical training data is limited.

Examples of rule categories include:

- Sustained extreme heart-rate measurements.
- Unusual changes from an individual's established baseline.
- Abnormal respiratory-rate patterns, when reliable measurements and appropriate reference ranges are available.
- Persistent oxygen-saturation deviations, when applicable and supported by suitable domain evidence.
- Sudden or sustained changes across multiple physiological signals.
- Sensor faults, missing measurements, and implausible values.

**Illustrative development rule:** a heart rate above 130 beats per minute sustained for at least 60 minutes during confirmed rest may be configured as a high-priority review condition for testing. This is an example engineering rule, not a universal clinical threshold or a validated NASA astronaut protocol.

Before using this rule operationally, verify its appropriateness with authoritative clinical or mission-specific guidance. Confirm that heart-rate units, sampling frequency, duration, and the definition of resting activity are reliable. Do not activate a resting-heart-rate rule if the available data cannot establish that the subject was resting.

Implement safety rules as named, configurable functions or configuration entries rather than scattered magic numbers.

Each rule must document:

- Its source or justification.
- Required input signals and units.
- Activation conditions.
- Duration or persistence requirements.
- Missing-data behavior.
- Severity and recommended review action.
- Known limitations and validation status.

Keep safety-rule alerts separate from ML anomaly scores. A documented rule can trigger an alert even when the learned model considers the observation typical.

Avoid presenting an unverified rule as a medical diagnosis or established astronaut health standard.

### 18.5 Persistence, Duration, and Context Awareness

Do not interpret every isolated unusual reading as an equally serious event.

Where sampling frequency and timestamps permit, evaluate:

- Whether a deviation persists over time.
- How quickly the signal changes.
- Whether multiple measurements deviate simultaneously.
- Whether the subject is resting, exercising, sleeping, or performing another known activity.
- Whether the sensor is reliable and the observation is complete.
- Whether the signal returns to its historical baseline.

For duration-based rules, use elapsed timestamps rather than assuming that a fixed number of rows always represents a fixed duration.

Handle irregular sampling, missing intervals, duplicated timestamps, and stale sensor values explicitly. Do not infer uninterrupted persistence across a large data gap.

Use contextual information only when it is actually available and sufficiently reliable.

### 18.6 Alert Prioritization

Where appropriate, implement a transparent alert-prioritization layer with distinct categories, such as:

- **Informational:** Minor or isolated deviation worth tracking.
- **Review:** Unusual behavior requiring closer inspection.
- **High priority:** A documented safety rule or sufficiently concerning validated signal has been triggered.

The categories must be configurable and grounded in explicit rules. Do not translate arbitrary anomaly-score percentiles directly into medical severity.

Every alert should include an explanation of the evidence that triggered it, such as the affected feature, observed value, historical comparison, persistence duration, model signal, or rule identifier.

If the data is insufficient to determine severity, return an uncertainty or insufficient-data status rather than manufacturing confidence.

### 18.7 Advanced Does Not Mean Unvalidated

The goal is a robust, high-quality anomaly-detection engine, not the largest possible neural network.

Use advanced algorithms when they provide measurable benefits. Compare them against simpler baselines, test their failure cases, and retain only justified complexity.

Never claim that a model is accurate, clinically validated, or superior without supporting evaluation.

Where ground-truth labels are unavailable, distinguish clearly between controlled synthetic tests, unsupervised evaluation, and real-world clinical validation.

The final system should combine learned patterns, temporal reasoning where supported, documented domain-informed rules, and explainable alert generation—while clearly separating experimental capabilities from validated safety requirements.