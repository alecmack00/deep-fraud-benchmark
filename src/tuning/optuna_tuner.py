"""
Bayesian Hyperparameter Optimization using Optuna with TPE and MedianPruner.
Optimizes XGBoost, LSTM, and Transformer architectures to maximize Dev PR-AUC.
"""

import gc
from typing import Any

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from src.deep_models.lstm_network import BiLSTMFraudModel
from src.deep_models.trainer import DeepSequenceTrainer
from src.deep_models.transformer_encoder import TransformerEncoderFraudModel
from src.eda_baselines.tree_classifiers import XGBoostFraudClassifier
from src.utils.logger import get_logger, timer

logger = get_logger("tuning.optuna_tuner")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cleanup_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


class OptunaHyperparameterTuner:
    """
    Bayesian optimization engine maximizing PR-AUC on the Dev split.
    """

    def __init__(
        self,
        n_trials: int = 20,
        seed: int = 42,
    ):
        self.n_trials = n_trials
        self.seed = seed
        self.sampler = TPESampler(seed=seed)
        self.pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)

    @timer
    def tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
    ) -> dict[str, Any]:
        logger.info(f"Starting Optuna sweep for XGBoost ({self.n_trials} trials)...")

        def objective(trial: optuna.Trial) -> float:
            max_depth = trial.suggest_int("max_depth", 4, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            scale_pos_weight = trial.suggest_float("scale_pos_weight", 10.0, 35.0)

            model = XGBoostFraudClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                scale_pos_weight=scale_pos_weight,
                n_estimators=300,
                early_stopping_rounds=20,
                random_state=self.seed,
            )
            model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)])

            preds_dev = model.predict_proba(X_dev)[:, 1]
            if len(np.unique(y_dev)) < 2:
                return 0.0
            pr_auc = average_precision_score(y_dev, preds_dev)
            return float(pr_auc)

        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            study_name="xgb_pr_auc_optimization",
        )
        study.optimize(objective, n_trials=self.n_trials)

        logger.info(f"XGBoost Tuning Complete! Best PR-AUC: {study.best_value:.4f}")
        return {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "study": study,
        }

    @timer
    def tune_lstm(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        input_dim: int,
        epochs_per_trial: int = 5,
    ) -> dict[str, Any]:
        logger.info(f"Starting Optuna sweep for LSTM ({self.n_trials} trials)...")

        def objective(trial: optuna.Trial) -> float:
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.1, 0.4)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

            model = BiLSTMFraudModel(
                input_dim=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
            trainer = DeepSequenceTrainer(
                model=model,
                model_name=f"optuna_lstm_trial_{trial.number}",
                lr=lr,
                max_epochs=epochs_per_trial,
                early_stopping_patience=2,
            )

            try:
                result = trainer.fit(train_loader, dev_loader, trial=trial)
                return result["best_dev_pr_auc"]
            finally:
                _cleanup_memory(trainer.device)
                del model
                del trainer

        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            study_name="lstm_pr_auc_optimization",
        )
        study.optimize(objective, n_trials=self.n_trials)

        logger.info(f"LSTM Tuning Complete! Best PR-AUC: {study.best_value:.4f}")
        return {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "study": study,
        }

    @timer
    def tune_transformer(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        input_dim: int,
        epochs_per_trial: int = 5,
    ) -> dict[str, Any]:
        logger.info(
            f"Starting Optuna sweep for Transformer ({self.n_trials} trials)..."
        )

        def objective(trial: optuna.Trial) -> float:
            d_model = trial.suggest_categorical("d_model", [64, 128])
            nhead_options = [2, 4] if d_model == 64 else [2, 4, 8]
            nhead = trial.suggest_categorical("nhead", nhead_options)
            num_layers = trial.suggest_int("num_layers", 2, 4)
            dim_feedforward = trial.suggest_categorical("dim_feedforward", [256, 512])
            dropout = trial.suggest_float("dropout", 0.05, 0.3)
            lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)

            model = TransformerEncoderFraudModel(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            trainer = DeepSequenceTrainer(
                model=model,
                model_name=f"optuna_transformer_trial_{trial.number}",
                lr=lr,
                max_epochs=epochs_per_trial,
                early_stopping_patience=2,
            )

            try:
                result = trainer.fit(train_loader, dev_loader, trial=trial)
                return result["best_dev_pr_auc"]
            finally:
                _cleanup_memory(trainer.device)
                del model
                del trainer

        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            study_name="transformer_pr_auc_optimization",
        )
        study.optimize(objective, n_trials=self.n_trials)

        logger.info(f"Transformer Tuning Complete! Best PR-AUC: {study.best_value:.4f}")
        return {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "study": study,
        }
