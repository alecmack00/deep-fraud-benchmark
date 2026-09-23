"""
Sequence Tensor Generator and PyTorch Dataset Builder.
Constructs chronological sliding windows of length L=20 per entity (card_id),
with zero padding, boolean padding masks, and cross-split context lookback.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.logger import get_logger

logger = get_logger("data.dataset_builder")


class TransactionSequenceDataset(Dataset):
    """
    PyTorch Dataset yielding:
        x_seq: [L, D] float32 tensor
        y_target: [1] float32 binary label of the final transaction in the window
        padding_mask: [L] bool tensor (True where padded / invalid)
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        meta_df: pd.DataFrame,
        window_length: int = 20,
        min_history: int = 1,
        entity_col: str = "card_id",
        time_col: str = "TransactionDT",
        context_data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | None = None,
    ):
        if min_history < 1:
            raise ValueError(f"min_history must be >= 1, got {min_history}")

        self.window_length = window_length
        self.min_history = min_history
        self.feature_dim = features.shape[1]

        # Combine historical context data (without labels for lookback) if provided
        if context_data is not None:
            ctx_features, ctx_targets, ctx_meta = context_data
            self.features = np.vstack([ctx_features, features]).astype(np.float32)
            self.targets = np.concatenate([ctx_targets, targets]).astype(np.float32)
            all_meta = pd.concat([ctx_meta, meta_df], ignore_index=True)
            eval_offset = len(ctx_features)
        else:
            self.features = features.astype(np.float32)
            self.targets = targets.astype(np.float32)
            all_meta = meta_df.reset_index(drop=True)
            eval_offset = 0

        eval_len = len(features)

        # Slices stored as (window_indices_array, target_idx)
        self.windows: list[tuple[np.ndarray, int]] = []
        self._build_sliding_windows(
            all_meta=all_meta,
            eval_offset=eval_offset,
            eval_len=eval_len,
            entity_col=entity_col,
            time_col=time_col,
        )

    def _build_sliding_windows(
        self,
        all_meta: pd.DataFrame,
        eval_offset: int,
        eval_len: int,
        entity_col: str,
        time_col: str,
    ) -> None:
        logger.info(
            f"Indexing transaction sequences (L={self.window_length}, min_history={self.min_history})..."
        )
        df_idx = pd.DataFrame(
            {
                "orig_idx": np.arange(len(self.features), dtype=np.int32),
                "entity": all_meta[entity_col].values,
                "time": all_meta[time_col].values,
            }
        ).sort_values(by=["entity", "time"])

        entity_history = (
            df_idx.groupby("entity", sort=False)["orig_idx"].apply(np.array).to_dict()
        )

        idx_to_entity_pos: dict[int, tuple[object, int]] = {}
        for e, hist in entity_history.items():
            for pos, idx_val in enumerate(hist):
                idx_to_entity_pos[int(idx_val)] = (e, pos)

        windows_list: list[tuple[np.ndarray, int]] = []
        for j in range(eval_len):
            target_idx = eval_offset + j
            if target_idx not in idx_to_entity_pos:
                windows_list.append(
                    (np.array([target_idx], dtype=np.int32), target_idx)
                )
                continue

            e, pos = idx_to_entity_pos[target_idx]
            hist = entity_history[e]
            start_pos = max(0, pos - self.window_length + 1)
            window_indices = hist[start_pos : pos + 1]

            if len(window_indices) < self.min_history:
                window_indices = np.array([target_idx], dtype=np.int32)

            windows_list.append((window_indices, target_idx))

        self.windows = windows_list
        logger.info(
            f"Constructed {len(self.windows)} sequence windows strictly aligned 1-to-1 with evaluation rows across {len(entity_history)} entities."
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window_indices, target_idx = self.windows[idx]
        seq_len = len(window_indices)
        pad_len = self.window_length - seq_len

        event_feats = self.features[window_indices]

        if pad_len > 0:
            pad_block = np.zeros((pad_len, self.feature_dim), dtype=np.float32)
            seq_feats = np.vstack([pad_block, event_feats])
            pad_mask = np.array([True] * pad_len + [False] * seq_len, dtype=bool)
        else:
            seq_feats = event_feats
            pad_mask = np.zeros(self.window_length, dtype=bool)

        target_val = np.array([self.targets[target_idx]], dtype=np.float32)

        return (
            torch.from_numpy(seq_feats),
            torch.from_numpy(target_val),
            torch.from_numpy(pad_mask),
        )


def collate_sequence_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function combining batch items into:
        x_seq: [B, L, D]
        y_target: [B, 1]
        padding_mask: [B, L]
    """
    x_batch = torch.stack([item[0] for item in batch], dim=0)
    y_batch = torch.stack([item[1] for item in batch], dim=0)
    mask_batch = torch.stack([item[2] for item in batch], dim=0)
    return x_batch, y_batch, mask_batch


def build_dataloaders(
    train_data: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    dev_data: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    test_data: tuple[np.ndarray, np.ndarray, pd.DataFrame] | None = None,
    window_length: int = 20,
    min_history: int = 1,
    batch_size: int = 128,
    num_workers: int = 0,
    entity_col: str = "card_id",
    time_col: str = "TransactionDT",
    use_context_history: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Create PyTorch DataLoaders for Train, Dev, and Test splits with optional
    cross-split historical context lookback.
    """
    # Detect entity col fallback if given entity_col is not present
    actual_entity_col = entity_col
    meta_cols = train_data[2].columns
    if actual_entity_col not in meta_cols:
        for candidate in ["card_id", "card1", "TransactionID", meta_cols[0]]:
            if candidate in meta_cols:
                actual_entity_col = candidate
                break

    train_dataset = TransactionSequenceDataset(
        features=train_data[0],
        targets=train_data[1],
        meta_df=train_data[2],
        window_length=window_length,
        min_history=min_history,
        entity_col=actual_entity_col,
        time_col=time_col,
    )

    dev_context = train_data if use_context_history else None
    dev_dataset = TransactionSequenceDataset(
        features=dev_data[0],
        targets=dev_data[1],
        meta_df=dev_data[2],
        window_length=window_length,
        min_history=min_history,
        entity_col=actual_entity_col,
        time_col=time_col,
        context_data=dev_context,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequence_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = None
    if test_data is not None:
        if use_context_history:
            combined_ctx_feats = np.vstack([train_data[0], dev_data[0]])
            combined_ctx_targets = np.concatenate([train_data[1], dev_data[1]])
            combined_ctx_meta = pd.concat(
                [train_data[2], dev_data[2]], ignore_index=True
            )
            test_context = (
                combined_ctx_feats,
                combined_ctx_targets,
                combined_ctx_meta,
            )
        else:
            test_context = None

        test_dataset = TransactionSequenceDataset(
            features=test_data[0],
            targets=test_data[1],
            meta_df=test_data[2],
            window_length=window_length,
            min_history=min_history,
            entity_col=actual_entity_col,
            time_col=time_col,
            context_data=test_context,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sequence_batch,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, dev_loader, test_loader
