"""
Supervised Baseline Classifiers for Tabular Fraud Detection.
Standardized Scikit-Learn API for XGBoost, Random Forest, and Calibrated Linear SVM.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from src.utils.logger import get_logger, timer

logger = get_logger("eda_baselines.classifiers")


class XGBoostFraudClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost Classifier using histogram tree method, early stopping on Dev PR-AUC,
    and scale_pos_weight handling positive class imbalance.
    """

    def __init__(
        self,
        tree_method: str = "hist",
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float | str | None = "auto",
        early_stopping_rounds: int = 30,
        eval_metric: str = "aucpr",
        random_state: int = 42,
    ):
        self.tree_method = tree_method
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state

        self.model_: xgb.XGBClassifier | None = None

    @timer
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: list | tuple | None = None,
    ) -> "XGBoostFraudClassifier":
        # Ensure eval_set is a list of (X, y) tuples as required by XGBoost >= 2.0
        if eval_set is not None and isinstance(eval_set, tuple):
            if len(eval_set) == 2 and not isinstance(eval_set[0], (tuple, list)):
                eval_set = [eval_set]
            else:
                eval_set = list(eval_set)

        # Dynamic calculation of scale_pos_weight if auto or None; fallback to 27.5
        if self.scale_pos_weight is None or self.scale_pos_weight == "auto":
            n_pos = int(np.sum(y_train == 1))
            n_neg = int(np.sum(y_train == 0))
            pos_weight = (n_neg / max(1, n_pos)) if n_pos > 0 else 27.5
        else:
            pos_weight = float(self.scale_pos_weight)

        logger.info(
            f"Training XGBoost (scale_pos_weight={pos_weight:.2f}, tree_method={self.tree_method})..."
        )
        self.model_ = xgb.XGBClassifier(
            tree_method=self.tree_method,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=pos_weight,
            early_stopping_rounds=(
                self.early_stopping_rounds if eval_set is not None else None
            ),
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model_.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict_proba!")
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict!")
        return self.model_.predict(X)


class RandomForestFraudClassifier(BaseEstimator, ClassifierMixin):
    """
    Random Forest Classifier using balanced subsample bagging for class imbalance.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 15,
        class_weight: str = "balanced_subsample",
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model_: RandomForestClassifier | None = None

    @timer
    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> "RandomForestFraudClassifier":
        logger.info(
            f"Training Random Forest ({self.n_estimators} trees, max_depth={self.max_depth})..."
        )
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model_.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict_proba!")
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict!")
        return self.model_.predict(X)


class CalibratedLinearSVM(BaseEstimator, ClassifierMixin):
    """
    Scalable Linear Support Vector Machine (SGDClassifier with loss='hinge')
    calibrated via Platt scaling (CalibratedClassifierCV sigmoid).
    Internal cross-validation (cv=3) is applied strictly on X_train to prevent
    leakage into validation or test partitions.
    """

    def __init__(
        self,
        loss: str = "hinge",
        penalty: str = "l2",
        alpha: float = 0.0001,
        max_iter: int = 2000,
        calibration_method: str = "sigmoid",
        random_state: int = 42,
    ):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.calibration_method = calibration_method
        self.random_state = random_state

        self.model_: Any | None = None

    @timer
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CalibratedLinearSVM":
        logger.info(
            f"Training Calibrated Linear SVM (loss={self.loss}, penalty={self.penalty})..."
        )
        base_estimator = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=self.random_state,
        )
        # Platt scaling via CalibratedClassifierCV
        self.model_ = CalibratedClassifierCV(
            estimator=base_estimator,
            method=self.calibration_method,
            cv=3,
        )
        self.model_.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict_proba!")
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict!")
        return self.model_.predict(X)


def save_baseline_model(
    model: Any, model_name: str, checkpoint_dir: str = "models/checkpoints"
) -> str:
    path = Path(checkpoint_dir) / f"{model_name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved {model_name} model checkpoint to {path}")
    return str(path)


def load_baseline_model(
    model_name: str, checkpoint_dir: str = "models/checkpoints"
) -> Any:
    path = Path(checkpoint_dir) / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    return joblib.load(path)
