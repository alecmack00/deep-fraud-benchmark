"""
Purged Group TimeSeries Cross-Validation Engine.
Prevents lookahead and identity overlap across chronological fold boundaries.
"""

from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from src.utils.logger import get_logger

logger = get_logger("tuning.validation")


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Purged Group TimeSeries Cross-Validator for transaction streams.

    Enforces two strict leakage guarantees:
      1. Boundary Purging: Drops all transactions within `purge_window_seconds`
         prior to the validation fold start to eliminate continuous state/leakage.
      2. Group Purging: Any entity (e.g., card_id) appearing in the validation fold
         is purged from the candidate training set to eliminate identity memorization.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window_seconds: float = 86400.0,  # 24 hours buffer in seconds (TransactionDT unit)
        min_train_ratio: float = 0.4,
        purge_buffer_seconds: float | None = None,
    ):
        self.n_splits = n_splits
        self.purge_window_seconds = float(
            purge_buffer_seconds
            if purge_buffer_seconds is not None
            else purge_window_seconds
        )
        self.min_train_ratio = float(min_train_ratio)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | pd.Series | None = None,
        timestamps: np.ndarray | pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train and validation indices respecting time boundaries and identity purging.
        """
        n_samples = len(X)
        times = (
            np.arange(n_samples, dtype=np.float64)
            if timestamps is None
            else np.asarray(timestamps, dtype=np.float64)
        )
        group_arr = (
            np.zeros(n_samples, dtype=np.int32)
            if groups is None
            else np.asarray(groups)
        )

        t_min, t_max = times[0], times[-1]
        total_span = t_max - t_min

        train_start_ratio = self.min_train_ratio
        val_step_ratio = (1.0 - train_start_ratio) / self.n_splits

        for fold in range(self.n_splits):
            val_start_ratio = train_start_ratio + fold * val_step_ratio
            val_end_ratio = val_start_ratio + val_step_ratio

            t_val_start = t_min + val_start_ratio * total_span
            t_val_end = t_min + val_end_ratio * total_span

            # 1. Isolate validation indices
            val_mask = (times >= t_val_start) & (
                times < t_val_end if fold < self.n_splits - 1 else times <= t_val_end
            )
            val_indices = np.where(val_mask)[0]

            if len(val_indices) == 0:
                logger.warning(f"CV Fold {fold + 1}: Empty validation window. Skipping.")
                continue

            # 2. Base candidate train: strictly prior to validation window
            candidate_train_mask = times < t_val_start

            # 3. Temporal boundary buffer purge
            t_buffer_start = t_val_start - self.purge_window_seconds
            temporal_buffer_mask = (times >= t_buffer_start) & (times < t_val_start)

            # 4. Group entity purge: drop all historical entries for entities in val set
            val_entities = set(group_arr[val_indices])
            entity_overlap_mask = np.isin(group_arr, list(val_entities)) & candidate_train_mask

            # Combined purge mask
            to_purge_mask = temporal_buffer_mask | entity_overlap_mask
            purged_train_mask = candidate_train_mask & (~to_purge_mask)
            train_indices = np.where(purged_train_mask)[0]

            n_purged = np.sum(to_purge_mask)
            logger.info(
                f"CV Fold {fold + 1}/{self.n_splits}: "
                f"Train={len(train_indices)} (Purged={n_purged} rows [Buffer+Entities]), "
                f"Val={len(val_indices)}"
            )

            yield train_indices, val_indices
