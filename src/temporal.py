"""
temporal.py — Temporal Verification & Changepoint Detection
KUET_CHAYAPOTH | NASA Space Apps 2026
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd


@dataclass
class AnomalyEpisode:
    episode_id: int
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    duration_seconds: float
    sample_count: int
    peak_score: float
    mean_score: float


class TemporalVerifier:
    """
    Applies temporal persistence filtering to eliminate transient sensor noise
    and only alert on sustained physiological transitions.
    """

    def __init__(self, window_size: int = 30, min_anomalous: int = 24):
        """
        window_size: Number of consecutive samples in rolling check.
        min_anomalous: Minimum anomalous samples required within window (e.g. 80%).
        """
        self.window_size = window_size
        self.min_anomalous = min_anomalous

    def verify(self, raw_anomalies: np.ndarray | pd.Series) -> pd.Series:
        """
        Returns boolean series of temporally verified anomalies.
        """
        series = pd.Series(raw_anomalies).fillna(0).astype(int)
        rolling_count = series.rolling(window=self.window_size, min_periods=1).sum()
        verified = (rolling_count >= self.min_anomalous).astype(int)
        return verified

    def extract_episodes(
        self,
        verified_mask: pd.Series,
        scores: np.ndarray,
        time_series: Optional[pd.Series] = None
    ) -> List[AnomalyEpisode]:
        """
        Segments continuous verified anomaly alerts into discrete episodes with timestamps.
        """
        episodes: List[AnomalyEpisode] = []
        is_in_episode = False
        start_idx = 0
        episode_counter = 1

        verified_arr = verified_mask.to_numpy()
        n = len(verified_arr)

        for i in range(n):
            if verified_arr[i] == 1 and not is_in_episode:
                is_in_episode = True
                start_idx = i
            elif verified_arr[i] == 0 and is_in_episode:
                is_in_episode = False
                end_idx = i - 1
                episodes.append(self._build_episode(episode_counter, start_idx, end_idx, scores, time_series))
                episode_counter += 1

        if is_in_episode:
            end_idx = n - 1
            episodes.append(self._build_episode(episode_counter, start_idx, end_idx, scores, time_series))

        return episodes

    def _build_episode(
        self,
        ep_id: int,
        start_idx: int,
        end_idx: int,
        scores: np.ndarray,
        time_series: Optional[pd.Series]
    ) -> AnomalyEpisode:
        sub_scores = scores[start_idx : end_idx + 1]
        t_start = float(time_series.iloc[start_idx]) if time_series is not None else float(start_idx)
        t_end = float(time_series.iloc[end_idx]) if time_series is not None else float(end_idx)
        duration = max(0.0, t_end - t_start)

        return AnomalyEpisode(
            episode_id=ep_id,
            start_index=start_idx,
            end_index=end_idx,
            start_time=t_start,
            end_time=t_end,
            duration_seconds=duration,
            sample_count=end_idx - start_idx + 1,
            peak_score=float(np.max(sub_scores)),
            mean_score=float(np.mean(sub_scores))
        )
