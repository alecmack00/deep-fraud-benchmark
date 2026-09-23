"""
Debugging script for evaluation tensor integrity on MPS/CUDA/CPU.
Run with: .venv/bin/python scripts/debug_eval_mps.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import torch

from src.data.dataset_builder import build_dataloaders
from src.data.preprocessor import run_preprocessing
from src.deep_models.lstm_network import BiLSTMFraudModel
from src.deep_models.trainer import DeepSequenceTrainer
from src.eda_baselines.clustering import KMeansClusterFeatureGenerator
from src.eda_baselines.decomposition import PCARepresentationLearner


def run_debug():
    print("--- 1. Loading Preprocessing & Data ---")
    preprocessor, paths = run_preprocessing()
    train_df = pl.read_parquet(paths["train"])
    dev_df = pl.read_parquet(paths["dev"])

    X_train, y_train, meta_train = preprocessor.transform(train_df)
    X_dev, y_dev, meta_dev = preprocessor.transform(dev_df)

    print(f"Total positive labels in Dev: {np.sum(y_dev)} / {len(y_dev)}")

    pca = PCARepresentationLearner.load()
    kmeans = KMeansClusterFeatureGenerator.load()

    X_train_pca = pca.transform(X_train)
    X_dev_pca = pca.transform(X_dev)

    X_train_aug = kmeans.augment_features(X_train, X_train_pca)
    X_dev_aug = kmeans.augment_features(X_dev, X_dev_pca)

    _train_loader, dev_loader, _ = build_dataloaders(
        train_data=(X_train_aug, y_train, meta_train),
        dev_data=(X_dev_aug, y_dev, meta_dev),
        window_length=10,
        batch_size=64,
    )

    bilstm = BiLSTMFraudModel(input_dim=X_train_aug.shape[1], hidden_size=64)
    trainer = DeepSequenceTrainer(
        model=bilstm,
        model_name="BiLSTM_Sequence",
        max_epochs=1,
    )

    print(f"\n--- 2. Auditing Batches on Device: {trainer.device} ---")
    bilstm.eval()
    all_raw_labels = []

    with torch.no_grad():
        for b_idx, (x_seq, y_target, padding_mask) in enumerate(dev_loader):
            raw_uniques = np.unique(y_target.numpy())
            all_raw_labels.extend(y_target.flatten().tolist())

            # Synchronous transfer (avoids MPS async buffer hazard)
            x_dev = x_seq.to(trainer.device)
            p_dev = padding_mask.to(trainer.device)
            y_dev_tensor = y_target.to(trainer.device)

            if trainer.device.type == "mps":
                torch.mps.synchronize()

            before_forward = np.unique(y_dev_tensor.cpu().numpy())
            _logits = bilstm(x_dev, p_dev)

            if trainer.device.type == "mps":
                torch.mps.synchronize()

            after_forward = np.unique(y_dev_tensor.cpu().numpy())

            if not np.array_equal(before_forward, after_forward):
                print(f"🚨 HAZARD: Batch {b_idx} mutated during forward pass!")
                print(f"   Before: {before_forward} -> After: {after_forward}")
            else:
                print(
                    f"Batch {b_idx:02d} | CPU: {raw_uniques} | MPS after forward: {after_forward}"
                )

    print(
        f"\nOverall Dev Loader label distribution: {np.unique(all_raw_labels, return_counts=True)}"
    )


if __name__ == "__main__":
    run_debug()
