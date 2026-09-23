"""
Tests for Imbalance-Aware Evaluation Metrics and Financial Loss Modeling.
"""

import numpy as np

from src.tracking.mlflow_logger import calculate_metrics, compute_financial_loss


def test_calculate_metrics_imbalance_aware():
    np.random.seed(42)
    # 5% fraud imbalance
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # Model that predicts high probability for true positives
    y_pred = np.array([0.9, 0.8, 0.1, 0.2, 0.05, 0.15, 0.3, 0.1, 0.25, 0.05])

    metrics = calculate_metrics(y_true, y_pred, threshold=0.5)

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "fraud_f1" in metrics
    assert "brier_score" in metrics
    assert "recall_at_95_precision" in metrics

    # PR-AUC and ROC-AUC should be high for this separable toy data
    assert metrics["pr_auc"] > 0.8
    assert metrics["roc_auc"] > 0.8
    assert metrics["brier_score"] < 0.1
    assert metrics["fraud_f1"] == 1.0


def test_compute_financial_loss():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.1])  # 1 TP, 1 FN, 1 FP, 1 TN at th=0.5

    loss_dict = compute_financial_loss(
        y_true=y_true,
        y_pred_proba=y_pred,
        threshold=0.5,
        fn_cost=500.0,
        fp_cost=25.0,
    )

    assert loss_dict["tp"] == 1
    assert loss_dict["fn"] == 1
    assert loss_dict["fp"] == 1
    assert loss_dict["tn"] == 1

    expected_cost = 1 * 500.0 + 1 * 25.0  # 525.0
    assert loss_dict["total_financial_loss"] == expected_cost
    assert loss_dict["cost_per_transaction"] == expected_cost / 4.0


def test_mlflow_logger_run(tmp_path):
    from pathlib import Path

    from src.tracking.mlflow_logger import BenchmarkMLflowTracker

    db_path = tmp_path / "test_mlflow.db"
    tracker = BenchmarkMLflowTracker(
        experiment_name="Test-Experiment",
        tracking_uri=f"sqlite:///{db_path}",
    )
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.85, 0.05, 0.2])

    summary = tracker.log_run(
        model_name="TestModel",
        params={"learning_rate": 0.01},
        y_true=y_true,
        y_pred_proba=y_pred,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert "metrics" in summary
    assert "pr_auc" in summary["metrics"]
    assert Path(
        tmp_path / "artifacts" / "TestModel" / "precision_recall_curve.png"
    ).exists()
    assert Path(tmp_path / "artifacts" / "TestModel" / "roc_curve.png").exists()
    assert Path(tmp_path / "artifacts" / "TestModel" / "calibration_curve.png").exists()
    assert Path(tmp_path / "artifacts" / "TestModel" / "summary.json").exists()
    assert Path(tmp_path / "artifacts" / "TestModel" / "test_predictions.npz").exists()


def test_calculate_metrics_edge_cases():
    # 1. All zeros (no fraud positives)
    y_zeros = np.zeros(20, dtype=np.int32)
    y_pred = np.linspace(0.01, 0.4, 20)
    metrics_zero = calculate_metrics(y_zeros, y_pred)
    assert metrics_zero["pr_auc"] == 0.0
    assert metrics_zero["roc_auc"] == 0.5
    assert metrics_zero["fraud_f1"] == 0.0
    assert metrics_zero["recall_at_95_precision"] == 0.0

    # 2. All ones
    y_ones = np.ones(20, dtype=np.int32)
    metrics_one = calculate_metrics(y_ones, y_pred)
    assert metrics_one["pr_auc"] == 0.0
    assert metrics_one["roc_auc"] == 0.5
    assert metrics_one["recall_at_95_precision"] == 0.0

    # 3. Empty input arrays
    metrics_empty = calculate_metrics(np.array([]), np.array([]))
    assert metrics_empty["pr_auc"] == 0.0
    assert metrics_empty["roc_auc"] == 0.5
    assert metrics_empty["fraud_f1"] == 0.0


def test_parameter_sanitization_and_truncation(tmp_path):
    from pathlib import Path

    from src.tracking.mlflow_logger import BenchmarkMLflowTracker, _sanitize_params

    # Direct test of _sanitize_params
    long_val = "feature_name_" * 50  # ~650 chars
    nested_cfg = {"deep": {"layers": [128, 64, 32], "name": "complex_arch"}}
    params = {
        "learning_rate": 0.001,
        "long_feature_list": long_val,
        "nested_cfg": nested_cfg,
    }
    sanitized = _sanitize_params(params, max_len=450)
    assert len(str(sanitized["long_feature_list"])) <= 450
    assert str(sanitized["long_feature_list"]).endswith("...")
    assert isinstance(sanitized["nested_cfg"], str)
    assert len(sanitized["nested_cfg"]) <= 450
    assert sanitized["learning_rate"] == 0.001

    # End-to-end logging test with oversized parameters
    db_path = tmp_path / "test_trunc.db"
    tracker = BenchmarkMLflowTracker(
        experiment_name="Truncation-Experiment",
        tracking_uri=f"sqlite:///{db_path}",
    )
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

    summary = tracker.log_run(
        model_name="TruncationModel",
        params=params,
        y_true=y_true,
        y_pred_proba=y_pred,
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    # Full un-truncated params must be preserved in summary artifact
    assert summary["params"]["long_feature_list"] == long_val
    assert Path(tmp_path / "artifacts" / "TruncationModel" / "summary.json").exists()


def test_mlflow_model_logging_and_registry(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    from src.tracking.mlflow_logger import BenchmarkMLflowTracker

    db_path = tmp_path / "test_registry.db"
    tracker = BenchmarkMLflowTracker(
        experiment_name="Registry-Experiment",
        tracking_uri=f"sqlite:///{db_path}",
    )

    X = np.random.randn(30, 4)
    y = np.random.randint(0, 2, size=30)
    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]

    summary = tracker.log_run(
        model_name="RFWithRegistry",
        params={"n_estimators": 10},
        y_true=y,
        y_pred_proba=y_pred,
        artifacts_dir=str(tmp_path / "artifacts"),
        model=clf,
        input_example=X[:2],
        registered_model_name="FraudRandomForestRegistry",
    )
    assert summary["model_name"] == "RFWithRegistry"
    assert "metrics" in summary


def test_mlflow_single_class_curves(tmp_path):
    from pathlib import Path

    from src.tracking.mlflow_logger import BenchmarkMLflowTracker

    db_path = tmp_path / "test_single_class.db"
    tracker = BenchmarkMLflowTracker(
        experiment_name="SingleClass-Experiment",
        tracking_uri=f"sqlite:///{db_path}",
    )

    y_true = np.zeros(25, dtype=np.int32)
    y_pred = np.linspace(0.01, 0.3, 25)

    summary = tracker.log_run(
        model_name="SingleClassModel",
        params={"lr": 0.01},
        y_true=y_true,
        y_pred_proba=y_pred,
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    assert summary["metrics"]["pr_auc"] == 0.0
    assert summary["metrics"]["roc_auc"] == 0.5
    assert Path(
        tmp_path / "artifacts" / "SingleClassModel" / "precision_recall_curve.png"
    ).exists()
    assert Path(tmp_path / "artifacts" / "SingleClassModel" / "roc_curve.png").exists()
    assert Path(
        tmp_path / "artifacts" / "SingleClassModel" / "calibration_curve.png"
    ).exists()
