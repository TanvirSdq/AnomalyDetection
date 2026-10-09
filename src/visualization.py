"""
visualization.py — Plotting Utilities for Anomaly Detection Results
====================================================================

PURPOSE:
    Provide a collection of clearly labelled, reusable plotting functions
    that cover all visualization requirements from Section 10 of the
    project context.

DESIGN PRINCIPLES:
    • Every plot includes a title, axis labels, units where known, and
      a legend where applicable.
    • All functions accept a matplotlib `ax` parameter so they can be
      embedded in multi-panel figures in the notebook.
    • Functions save to disk only when explicitly requested (save_path arg).
    • Colors follow a consistent palette: blue = normal, red = anomaly,
      orange = threshold, green = baseline.

LIMITATIONS:
    • All plots are static (matplotlib).  Interactive dashboards would
      require a frontend framework (Plotly, Bokeh, etc.).
    • HRV feature names contain technical abbreviations; axis labels use
      short names where possible and longer descriptions in the title.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# Use non-interactive backend by default so plots don't block in scripts
# The notebook will override this with %matplotlib inline
matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["figure.figsize"] = (14, 5)
matplotlib.rcParams["font.size"] = 11

# ---------------------------------------------------------------------------
# Color palette constants
# ---------------------------------------------------------------------------
COLOR_NORMAL = "#2196F3"       # blue
COLOR_ANOMALY = "#F44336"      # red
COLOR_THRESHOLD = "#FF9800"    # orange
COLOR_BASELINE = "#4CAF50"     # green
COLOR_COMPOSITE = "#9C27B0"    # purple


# ---------------------------------------------------------------------------
# 1. Time-series feature plot with anomaly overlay
# ---------------------------------------------------------------------------

def plot_feature_over_time(
    df: pd.DataFrame,
    feature: str,
    scores: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    time_col: str = "Time",
    condition_col: str = "condition",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Plot one physiological feature over time, shading anomalous rows.

    WHAT THIS SHOWS:
        • Feature values on the primary y-axis.
        • Condition labels as background shading.
        • If scores + threshold provided: rows above threshold highlighted in red.

    Parameters
    ----------
    df : pd.DataFrame
        Contains time_col and feature columns (and optionally condition_col).
    feature : str
        Column name to plot on y-axis.
    scores : np.ndarray, optional
        Anomaly scores aligned row-for-row with df.
    threshold : float, optional
        Anomaly score threshold — rows above this are highlighted.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.get_figure()

    x = df[time_col].values if time_col in df.columns else np.arange(len(df))
    y = df[feature].values

    # --- Background condition shading ---
    if condition_col in df.columns:
        conditions = df[condition_col].values
        unique_conditions = pd.unique(conditions)
        cmap = plt.cm.get_cmap("Pastel1", len(unique_conditions))
        cond_colors = {c: cmap(i) for i, c in enumerate(unique_conditions)}
        prev_cond = conditions[0]
        start_idx = 0
        for i in range(1, len(conditions)):
            if conditions[i] != prev_cond or i == len(conditions) - 1:
                end_idx = i if conditions[i] != prev_cond else i + 1
                ax.axvspan(
                    x[start_idx], x[min(end_idx, len(x)-1)],
                    alpha=0.18, color=cond_colors[prev_cond], label=f"_{prev_cond}"
                )
                start_idx = i
                prev_cond = conditions[i]
        # Add condition legend entries manually
        for cond, color in cond_colors.items():
            ax.fill_between([], [], color=color, alpha=0.5, label=f"Condition: {cond}")

    # --- Main feature line ---
    ax.plot(x, y, color=COLOR_NORMAL, linewidth=0.8, alpha=0.85, label=feature)

    # --- Anomaly highlights ---
    if scores is not None and threshold is not None:
        anomaly_mask = scores >= threshold
        if anomaly_mask.any():
            ax.scatter(
                x[anomaly_mask], y[anomaly_mask],
                color=COLOR_ANOMALY, s=12, zorder=5, label="Flagged anomaly",
                alpha=0.8
            )

    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel(feature)
    ax.set_title(title or f"{feature} over time")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# 2. Anomaly score time-series
# ---------------------------------------------------------------------------

def plot_anomaly_scores(
    scores: np.ndarray,
    time_values: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    detector_name: str = "detector",
    condition_col: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Plot anomaly scores over time with threshold line and alert markers.

    WHAT THIS SHOWS:
        • Score value at each time step.
        • Orange dashed horizontal threshold line (if provided).
        • Red markers above threshold.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.get_figure()

    x = time_values if time_values is not None else np.arange(len(scores))

    # Score line
    ax.plot(x, scores, color=COLOR_COMPOSITE, linewidth=0.9, alpha=0.85,
            label=f"{detector_name} score")

    # Threshold line
    if threshold is not None:
        ax.axhline(
            y=threshold, color=COLOR_THRESHOLD, linestyle="--",
            linewidth=1.5, label=f"Alert threshold ({threshold:.3f})"
        )
        # Shade alert zone
        ax.fill_between(
            x, threshold, scores,
            where=(scores >= threshold),
            color=COLOR_ANOMALY, alpha=0.25, label="Alert zone"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(f"Anomaly Score over Time — {detector_name}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# 3. Score distribution histogram
# ---------------------------------------------------------------------------

def plot_score_distribution(
    scores: np.ndarray,
    threshold: Optional[float] = None,
    detector_name: str = "detector",
    labels: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Histogram of anomaly scores.

    WHAT THIS SHOWS:
        • Distribution shape — a good detector should have most scores
          clustered near 0 (normal), with a small tail of high scores.
        • If ground-truth labels provided: overlay normal vs. anomalous.
        • Threshold marker showing the decision boundary.

    INTERPRETATION:
        Bimodal distributions (two peaks) suggest clear separation between
        normal and anomalous observations — a good sign.
        Unimodal distributions suggest the detector is not discriminating.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    bins = 60

    if labels is not None:
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        ax.hist(normal_scores, bins=bins, color=COLOR_NORMAL, alpha=0.6,
                label=f"Normal (n={len(normal_scores)})", density=True)
        ax.hist(anomaly_scores, bins=bins, color=COLOR_ANOMALY, alpha=0.6,
                label=f"Injected anomaly (n={len(anomaly_scores)})", density=True)
    else:
        ax.hist(scores, bins=bins, color=COLOR_COMPOSITE, alpha=0.7,
                label="Score distribution", density=True)

    if threshold is not None:
        ax.axvline(x=threshold, color=COLOR_THRESHOLD, linestyle="--",
                   linewidth=2, label=f"Threshold: {threshold:.3f}")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {detector_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# 4. Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "detection_rate_pct",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Bar chart comparing multiple detectors on a chosen evaluation metric.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Each row is a detector; must contain columns 'detector' and `metric`.
    metric : str
        Column name of the metric to plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    colors = [COLOR_BASELINE, COLOR_NORMAL, COLOR_COMPOSITE,
              COLOR_ANOMALY, COLOR_THRESHOLD]
    n = len(comparison_df)
    bar_colors = colors[:n] + [COLOR_COMPOSITE] * max(0, n - len(colors))

    bars = ax.bar(
        comparison_df["detector"], comparison_df[metric],
        color=bar_colors, alpha=0.8, edgecolor="white"
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.5,
            f"{height:.1f}",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_xlabel("Detector")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(comparison_df[metric].max() * 1.2, 10))

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# 5. Feature correlation heatmap
# ---------------------------------------------------------------------------

def plot_feature_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_features: int = 20,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Correlation heatmap of selected features.

    PURPOSE:
        Reveals highly correlated feature pairs.  Many features in this
        dataset are mathematical transformations of each other (e.g.
        RMSSD and RMSSD_LOG), which can cause redundancy.
        Understanding the correlation structure helps justify feature
        selection decisions.
    """
    cols = feature_cols[:max_features]
    corr = df[cols].corr()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        corr, ax=ax, cmap="coolwarm", center=0,
        vmin=-1, vmax=1, square=True,
        xticklabels=cols, yticklabels=cols,
        cbar_kws={"shrink": 0.6},
        linewidths=0.3,
    )
    ax.set_title("Feature Correlation Heatmap (Pearson r)")
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# 6. Composite dashboard figure
# ---------------------------------------------------------------------------

def plot_full_dashboard(
    df: pd.DataFrame,
    scores_dict: dict[str, np.ndarray],
    composite_scores: np.ndarray,
    feature_cols: list[str],
    threshold: float,
    time_col: str = "Time",
    primary_feature: str = "HR",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Produce a multi-panel dashboard summarising all detector results.

    PANELS:
        Row 1: Primary feature (HR) over time with anomaly highlights.
        Row 2: Composite anomaly score over time.
        Row 3: Score distribution comparison (one subplot per detector).
    """
    n_detectors = len(scores_dict)
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, n_detectors, figure=fig, hspace=0.45, wspace=0.35)

    time_vals = df[time_col].values if time_col in df.columns else np.arange(len(df))

    # --- Row 1: Primary feature over time ---
    ax_feat = fig.add_subplot(gs[0, :])  # spans all columns
    composite_flag = composite_scores >= threshold
    ax_feat.plot(time_vals, df[primary_feature].values,
                 color=COLOR_NORMAL, linewidth=0.8, label=primary_feature)
    if composite_flag.any():
        ax_feat.scatter(
            time_vals[composite_flag],
            df[primary_feature].values[composite_flag],
            color=COLOR_ANOMALY, s=12, zorder=5,
            alpha=0.8, label="Composite anomaly flag"
        )
    ax_feat.set_xlabel("Time (s)")
    ax_feat.set_ylabel("HR (bpm)")
    ax_feat.set_title(f"{primary_feature} over Time with Composite Anomaly Flags")
    ax_feat.legend(fontsize=9)
    ax_feat.grid(True, alpha=0.3)

    # --- Row 2: Composite score over time ---
    ax_comp = fig.add_subplot(gs[1, :])
    ax_comp.plot(time_vals, composite_scores,
                 color=COLOR_COMPOSITE, linewidth=0.9, label="Composite score")
    ax_comp.axhline(
        y=threshold, color=COLOR_THRESHOLD, linestyle="--",
        linewidth=1.5, label=f"Threshold {threshold:.3f}"
    )
    ax_comp.fill_between(
        time_vals, threshold, composite_scores,
        where=(composite_scores >= threshold),
        color=COLOR_ANOMALY, alpha=0.25
    )
    ax_comp.set_xlabel("Time (s)")
    ax_comp.set_ylabel("Composite Score")
    ax_comp.set_title("Composite Anomaly Score over Time")
    ax_comp.legend(fontsize=9)
    ax_comp.grid(True, alpha=0.3)

    # --- Row 3: Per-detector score distribution ---
    for i, (det_name, det_scores) in enumerate(scores_dict.items()):
        ax_hist = fig.add_subplot(gs[2, i])
        ax_hist.hist(det_scores, bins=50, color=COLOR_COMPOSITE,
                     alpha=0.7, density=True, edgecolor="white")
        ax_hist.axvline(x=threshold, color=COLOR_THRESHOLD, linestyle="--",
                        linewidth=1.5)
        ax_hist.set_xlabel("Score")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"{det_name}\nScore Distribution", fontsize=10)
        ax_hist.grid(True, alpha=0.3)

    fig.suptitle(
        "Astronaut Health Monitor — Anomaly Detection Dashboard",
        fontsize=14, fontweight="bold", y=1.01
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"Dashboard saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 7. Feature contribution bar chart (for a single observation)
# ---------------------------------------------------------------------------

def plot_feature_contributions(
    feature_names: list[str],
    scores: np.ndarray,
    title: str = "Feature Contributions to Anomaly Score",
    top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Horizontal bar chart of per-feature robust Z-scores for one observation.

    PURPOSE:
        When a sample is flagged as anomalous, this chart explains WHICH
        features drove the score, helping a human reviewer decide whether
        the alert deserves attention.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Sort by score descending, keep top_n
    sorted_idx = np.argsort(scores)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]

    bar_colors = [
        COLOR_ANOMALY if s >= 3.5 else COLOR_BASELINE
        for s in sorted_scores
    ]

    ax.barh(
        range(len(sorted_names)), sorted_scores,
        color=bar_colors, alpha=0.8, edgecolor="white"
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=3.5, color=COLOR_THRESHOLD, linestyle="--",
               linewidth=1.5, label="Threshold (3.5 MADs)")
    ax.set_xlabel("Robust Z-Score (MADs from median)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")

    return ax
