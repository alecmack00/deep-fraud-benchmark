"""
Experiment Tracking and Operational Logging with MLflow.
Logs statistical imbalance-aware metrics, latency benchmarks, and evaluation curves.
"""

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger("tracking.mlflow_logger")


def _sanitize_params(params: dict[str, Any], max_len: int = 450) -> dict[str, Any]:
    """
    Sanitizes parameters for MLflow logging by serializing nested types
    and truncating values exceeding MLflow's parameter length limit.
    """
    sanitized: dict[str, Any] = {}
    for k, v in params.items():
        key_str = str(k)[:250]
        if isinstance(v, (dict, list, tuple, set)):
            val_str = json.dumps(v, default=str)
            if len(val_str) > max_len:
                val_str = val_str[: max_len - 3] + "..."
            sanitized[key_str] = val_str
        elif isinstance(v, (int, float, bool, np.integer, np.floating)):
            str_repr = str(v)
            if len(str_repr) > max_len:
                sanitized[key_str] = str_repr[: max_len - 3] + "..."
            else:
                sanitized[key_str] = v
        else:
            val_str = str(v)
            if len(val_str) > max_len:
                val_str = val_str[: max_len - 3] + "..."
            sanitized[key_str] = val_str
    return sanitized


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    criterion: str = "f1",
    fn_cost: float = 100.0,
    fp_cost: float = 10.0,
) -> float:
    """
    Sweeps decision thresholds in [0.01, 0.99] to find optimal threshold
    maximizing F1 or minimizing expected business/financial loss.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float32)

    if len(np.unique(y_true)) < 2:
        return 0.50

    thresholds = np.linspace(0.01, 0.99, 99)
    best_score = -1.0 if criterion == "f1" else float("inf")
    best_th = 0.50

    for th in thresholds:
        preds = (y_pred_proba >= th).astype(np.int32)
        if criterion == "f1":
            score = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
            if score > best_score:
                best_score = score
                best_th = float(th)
        elif criterion == "cost":
            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            _tn, fp, fn, _tp = cm.ravel()
            cost = fn * fn_cost + fp * fp_cost
            if cost < best_score:
                best_score = cost
                best_th = float(th)

    return float(best_th)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    optimal_threshold: float | None = None,
) -> dict[str, float]:
    """
    Computes imbalance-aware evaluation metrics:
    PR-AUC, ROC-AUC, Macro F1, Positive Class F1 (at threshold and optimal threshold),
    Recall at 95% Precision, Brier Score.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float32)

    if len(y_true) == 0:
        logger.warning("Empty evaluation arrays provided to calculate_metrics.")
        return {
            "pr_auc": 0.0,
            "roc_auc": 0.5,
            "macro_f1": 0.0,
            "fraud_f1": 0.0,
            "fraud_f1_opt": 0.0,
            "fraud_f1_50": 0.0,
            "optimal_threshold": 0.5,
            "recall_at_95_precision": 0.0,
            "brier_score": 0.0,
        }

    unique_classes = np.unique(y_true)
    y_pred_bin_default = (y_pred_proba >= threshold).astype(np.int32)
    macro_f1 = float(
        f1_score(y_true, y_pred_bin_default, average="macro", zero_division=0)
    )
    pos_f1_50 = float(
        f1_score(y_true, y_pred_bin_default, pos_label=1, zero_division=0)
    )
    brier = float(brier_score_loss(y_true, y_pred_proba))

    opt_th = (
        optimal_threshold
        if optimal_threshold is not None
        else find_optimal_threshold(y_true, y_pred_proba)
    )
    y_pred_bin_opt = (y_pred_proba >= opt_th).astype(np.int32)
    pos_f1_opt = float(f1_score(y_true, y_pred_bin_opt, pos_label=1, zero_division=0))

    if len(unique_classes) < 2 or 1 not in unique_classes:
        logger.warning(
            "Evaluation batch contains fewer than 2 classes or no positive fraud cases. "
            "Returning fallback metrics."
        )
        return {
            "pr_auc": 0.0,
            "roc_auc": 0.5,
            "macro_f1": macro_f1,
            "fraud_f1": pos_f1_opt,
            "fraud_f1_opt": pos_f1_opt,
            "fraud_f1_50": pos_f1_50,
            "optimal_threshold": float(opt_th),
            "recall_at_95_precision": 0.0,
            "brier_score": brier,
        }

    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))

    # Calculate Recall at 95% Precision
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)
    idx_p95 = np.where(precisions >= 0.95)[0]
    recall_at_p95 = float(np.max(recalls[idx_p95])) if len(idx_p95) > 0 else 0.0

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "macro_f1": macro_f1,
        "fraud_f1": pos_f1_opt,
        "fraud_f1_opt": pos_f1_opt,
        "fraud_f1_50": pos_f1_50,
        "optimal_threshold": float(opt_th),
        "recall_at_95_precision": recall_at_p95,
        "brier_score": brier,
    }


def compute_financial_loss(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
    fn_cost: float = 500.0,  # Cost of missed fraud
    fp_cost: float = 20.0,  # Cost of customer friction / manual review
) -> dict[str, float]:
    """
    Computes financial expected loss for an operating threshold.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = (y_pred_proba >= threshold).astype(np.int32)

    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "total_financial_loss": 0.0,
            "cost_per_transaction": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fn * fn_cost) + (fp * fp_cost)
    cost_per_tx = total_cost / max(1, len(y_true))

    return {
        "threshold": float(threshold),
        "total_financial_loss": float(total_cost),
        "cost_per_transaction": float(cost_per_tx),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


class BenchmarkMLflowTracker:
    """
    Manages experiment runs, metrics logging, artifact rendering, and model logging.
    """

    def __init__(
        self,
        experiment_name: str = "Enterprise-Fraud-Sequence-Benchmark",
        tracking_uri: str | None = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = (
            tracking_uri
            if tracking_uri is not None
            else os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(
            f"MLflow initialized with experiment '{experiment_name}' at {self.tracking_uri}"
        )

    def log_run(
        self,
        model_name: str,
        params: dict[str, Any],
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        optimal_threshold: float | None = None,
        operational_metrics: dict[str, float] | None = None,
        artifacts_dir: str = "models/artifacts",
        model: Any | None = None,
        input_example: Any | None = None,
        signature: Any | None = None,
        registered_model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Log complete run with statistical metrics, operational profiling, curve plots,
        and optional model registry logging with input/output signature inference.
        """
        artifacts_path = Path(artifacts_dir) / model_name
        artifacts_path.mkdir(parents=True, exist_ok=True)

        metrics = calculate_metrics(
            y_true, y_pred_proba, optimal_threshold=optimal_threshold
        )
        if operational_metrics:
            metrics.update(operational_metrics)

        with mlflow.start_run(run_name=model_name) as run:
            # MLflow strictly limits parameter string length to 500 characters
            sanitized_params = _sanitize_params(params)
            mlflow.log_params(sanitized_params)
            mlflow.log_metrics(metrics)

            # Generate and save Precision-Recall curve
            pr_fig = self._plot_precision_recall(
                y_true, y_pred_proba, model_name, metrics["pr_auc"]
            )
            try:
                pr_path = artifacts_path / "precision_recall_curve.png"
                pr_fig.savefig(pr_path, bbox_inches="tight", dpi=150)
                mlflow.log_artifact(str(pr_path))
            finally:
                plt.close(pr_fig)

            # Generate and save ROC curve
            roc_fig = self._plot_roc(
                y_true, y_pred_proba, model_name, metrics["roc_auc"]
            )
            try:
                roc_path = artifacts_path / "roc_curve.png"
                roc_fig.savefig(roc_path, bbox_inches="tight", dpi=150)
                mlflow.log_artifact(str(roc_path))
            finally:
                plt.close(roc_fig)

            # Generate and save Calibration curve
            calib_fig = self._plot_calibration(
                y_true, y_pred_proba, model_name, metrics["brier_score"]
            )
            try:
                calib_path = artifacts_path / "calibration_curve.png"
                calib_fig.savefig(calib_path, bbox_inches="tight", dpi=150)
                mlflow.log_artifact(str(calib_path))
            finally:
                plt.close(calib_fig)

            # Save summary json (preserves full un-truncated parameters)
            summary = {
                "model_name": model_name,
                "metrics": metrics,
                "params": params,
                "run_id": run.info.run_id,
            }
            summary_path = artifacts_path / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_artifact(str(summary_path))

            # Save test predictions npz for interactive dashboard analysis
            pred_path = artifacts_path / "test_predictions.npz"
            np.savez(pred_path, y_true=y_true, y_pred_proba=y_pred_proba)
            mlflow.log_artifact(str(pred_path))

            # Optional model logging with signature inference and Model Registry
            if model is not None:
                self._log_model_artifact(
                    model=model,
                    model_name=model_name,
                    input_example=input_example,
                    signature=signature,
                    registered_model_name=registered_model_name,
                )

            logger.info(
                f"Logged MLflow run for {model_name}: "
                f"PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['fraud_f1']:.4f}"
            )
            return summary

    def _log_model_artifact(
        self,
        model: Any,
        model_name: str,
        input_example: Any | None = None,
        signature: Any | None = None,
        registered_model_name: str | None = None,
    ) -> None:
        """
        Logs a fitted Scikit-Learn/XGBoost or PyTorch model with schema signature and registry.
        """
        sample_ex = None
        if input_example is not None:
            sample_ex = (
                input_example[:2]
                if hasattr(input_example, "__len__") and len(input_example) > 2
                else input_example
            )

        if signature is None and sample_ex is not None:
            try:
                from mlflow.models import infer_signature

                if hasattr(model, "predict_proba"):
                    sample_pred = model.predict_proba(sample_ex)
                elif hasattr(model, "predict"):
                    sample_pred = model.predict(sample_ex)
                elif callable(model):
                    import torch

                    if isinstance(sample_ex, np.ndarray):
                        sample_t = torch.from_numpy(sample_ex)
                        sample_pred = model(sample_t)
                    else:
                        sample_pred = model(sample_ex)
                    if hasattr(sample_pred, "detach"):
                        sample_pred = sample_pred.detach().cpu().numpy()
                else:
                    sample_pred = None

                if sample_pred is not None:
                    signature = infer_signature(sample_ex, sample_pred)
            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                logger.warning(
                    f"Could not infer schema signature for {model_name}: {e}"
                )

        try:
            from torch import nn

            is_torch = isinstance(model, nn.Module)
        except ImportError:
            is_torch = False

        if is_torch:
            import mlflow.pytorch

            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                signature=signature,
                input_example=sample_ex,
                serialization_format="pickle",
                registered_model_name=registered_model_name,
            )
        else:
            import mlflow.sklearn

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=sample_ex,
                serialization_format="cloudpickle",
                registered_model_name=registered_model_name,
            )
        logger.info(
            f"Successfully logged model artifact for {model_name} "
            f"(registered_model_name={registered_model_name})"
        )

    def _plot_precision_recall(
        self, y_true: np.ndarray, y_pred: np.ndarray, title: str, pr_auc: float
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 5))
        if len(np.unique(y_true)) < 2:
            ax.text(
                0.5,
                0.5,
                "Precision-Recall Curve Undefined\n(Single class present in y_true)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )
        else:
            p, r, _ = precision_recall_curve(y_true, y_pred)
            ax.plot(r, p, color="#2563eb", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
            base_rate = float(np.mean(y_true))
            ax.axhline(
                base_rate,
                color="gray",
                linestyle="--",
                label=f"Base rate ({base_rate:.3f})",
            )
            ax.legend(loc="upper right")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{title} - Precision-Recall Curve")
        ax.grid(True, alpha=0.3)
        return fig

    def _plot_roc(
        self, y_true: np.ndarray, y_pred: np.ndarray, title: str, roc_auc: float
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 5))
        if len(np.unique(y_true)) < 2:
            ax.text(
                0.5,
                0.5,
                "ROC Curve Undefined\n(Single class present in y_true)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )
        else:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            ax.plot(
                fpr,
                tpr,
                color="#10b981",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.4f})",
            )
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
            ax.legend(loc="lower right")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title} - ROC Curve")
        ax.grid(True, alpha=0.3)
        return fig

    def _plot_calibration(
        self, y_true: np.ndarray, y_pred: np.ndarray, title: str, brier: float
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 5))
        if len(np.unique(y_true)) < 2:
            ax.text(
                0.5,
                0.5,
                "Calibration Curve Undefined\n(Single class present in y_true)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )
        else:
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
            ax.plot(
                prob_pred,
                prob_true,
                marker="o",
                lw=2,
                color="#8b5cf6",
                label=f"Reliability (Brier={brier:.4f})",
            )
            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="gray",
                label="Perfect calibration",
            )
            ax.legend(loc="upper left")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"{title} - Calibration Curve")
        ax.grid(True, alpha=0.3)
        return fig
