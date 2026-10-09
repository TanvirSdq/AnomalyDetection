# Astronaut Health Monitoring (Anomaly Detection)

**NASA Space Apps 2026**

## Core Concept
A personalized anomaly detection system that learns an astronaut's unique physiological baseline and identifies meaningful deviations. We prioritize explainability and temporal verification over generic thresholds or opaque "health scores."

## Technical Approach

- **Layer 1: Personalized Statistical Baseline**
  Calculates dynamic baselines using rolling averages and robust statistics (e.g., MAD, z-scores) rather than fixed universal thresholds.
- **Layer 2: Multivariate Anomaly Detection**
  Uses **Isolation Forest** to identify observations that deviate from the learned historical norm across multiple physiological signals simultaneously (e.g., HR, RMSSD, SDRR).
- **Layer 3: Temporal Verification**
  Applies rolling windows and persistence checks to filter out single-point noise. Alerts are only triggered for sustained, meaningful transitions.

## Key Differentiator
**Explainability & Context:** Alerts answer "Why Now?" by explicitly stating which signals deviated, by how much, and when the deviation started, assisting in protocol-linked decisions without making unsupported medical diagnoses.
