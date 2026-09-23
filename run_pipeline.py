"""
Unified CLI Pipeline Runner for Enterprise Fraud Detection & Sequence Benchmark.
Executes ingestion, zero-leakage preprocessing, classical baselines, deep sequence models,
and metric logging.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

from src.data.dataset_builder import build_dataloaders
from src.data.ingestion import run_ingestion
from src.data.preprocessor import run_preprocessing
from src.deep_models.lstm_network import BiLSTMFraudModel
from src.deep_models.trainer import DeepSequenceTrainer
from src.deep_models.transformer_encoder import TransformerEncoderFraudModel
from src.eda_baselines.clustering import KMeansClusterFeatureGenerator
from src.eda_baselines.decomposition import PCARepresentationLearner
from src.eda_baselines.tree_classifiers import (
    CalibratedLinearSVM,
    RandomForestFraudClassifier,
    XGBoostFraudClassifier,
    save_baseline_model,
)
from src.tracking.mlflow_logger import BenchmarkMLflowTracker
from src.tuning.optuna_tuner import OptunaHyperparameterTuner
from src.utils.config_parser import load_data_config
from src.utils.logger import get_logger, set_seed

logger = get_logger("pipeline_runner")


def run_full_pipeline(
    quick_mode: bool = False, skip_deep: bool = False, run_tune: bool = False
):
    set_seed(42)
    load_data_config()

    logger.info("=" * 70)
    logger.info("STAGE 1: DATA INGESTION")
    logger.info("=" * 70)
    sample_records = 15000 if quick_mode else 60000
    df = run_ingestion(quick_sample_records=sample_records)
    logger.info(f"Ingested dataset size: {df.shape[0]} rows x {df.shape[1]} columns")

    logger.info("=" * 70)
    logger.info("STAGE 2: LEAKAGE-FREE TEMPORAL SPLITTING & PREPROCESSING")
    logger.info("=" * 70)
    preprocessor, paths = run_preprocessing()

    train_df = pl.read_parquet(paths["train"])
    dev_df = pl.read_parquet(paths["dev"])
    test_df = pl.read_parquet(paths["test"])

    # Transform splits using strictly train-fitted preprocessor
    X_train, y_train, meta_train = preprocessor.transform(train_df)
    X_dev, y_dev, meta_dev = preprocessor.transform(dev_df)
    X_test, y_test, _meta_test = preprocessor.transform(test_df)

    logger.info(
        f"Preprocessed Feature Matrix: Train={X_train.shape}, Dev={X_dev.shape}, Test={X_test.shape}"
    )
    logger.info(
        f"Train Fraud Rate: {np.mean(y_train) * 100:.2f}%, Dev Fraud Rate: {np.mean(y_dev) * 100:.2f}%, Test Fraud Rate: {np.mean(y_test) * 100:.2f}%"
    )

    logger.info("=" * 70)
    logger.info("STAGE 3: UNSUPERVISED REPRESENTATION DISCOVERY (PCA & K-MEANS)")
    logger.info("=" * 70)
    # 1. PCA fit strictly on Train
    pca = PCARepresentationLearner(variance_threshold=0.90, random_state=42)
    pca.fit(X_train)
    pca.save()

    X_train_pca = pca.transform(X_train)
    X_dev_pca = pca.transform(X_dev)
    X_test_pca = pca.transform(X_test)

    # 2. MiniBatch K-Means fit strictly on Train PCA projections
    kmeans = KMeansClusterFeatureGenerator(n_clusters=8, random_state=42)
    kmeans.fit(X_train_pca)
    kmeans.save()

    # Augment tabular features with centroid Euclidean distance vectors
    X_train_aug = kmeans.augment_features(X_train, X_train_pca)
    X_dev_aug = kmeans.augment_features(X_dev, X_dev_pca)
    kmeans.augment_features(X_test, X_test_pca)

    # Save 3D coordinates for dashboard visualization
    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    n_vis = min(2000, len(X_dev_pca))
    coords_3d = X_dev_pca[:n_vis, :3]
    labels_3d = y_dev[:n_vis]
    clusters_3d = kmeans.predict_clusters(X_dev_pca[:n_vis])
    np.savez(
        "models/artifacts/latent_space_dev.npz",
        coords=coords_3d,
        labels=labels_3d,
        clusters=clusters_3d,
        centroids=kmeans.cluster_centers_[:, :3],
        scree_var=pca.explained_variance_ratio_,
        cum_var=pca.cumulative_variance_ratio_,
    )
    logger.info("Saved latent space coordinates for dashboard.")

    logger.info("=" * 70)
    logger.info("STAGE 4: CLASSICAL BASELINE CLASSIFIERS")
    logger.info("=" * 70)
    tracker = BenchmarkMLflowTracker()
    results = {}

    # 1. XGBoost
    t0 = time.perf_counter()
    xgb_model = XGBoostFraudClassifier(
        max_depth=5 if quick_mode else 6,
        n_estimators=100 if quick_mode else 400,
        learning_rate=0.05,
        early_stopping_rounds=20,
    )
    xgb_model.fit(X_train_aug, y_train, eval_set=[(X_dev_aug, y_dev)])
    xgb_train_dur = time.perf_counter() - t0
    save_baseline_model(xgb_model, "xgboost_baseline")

    # Latency profiling
    t_lat0 = time.perf_counter()
    for _ in range(100):
        _ = xgb_model.predict_proba(X_dev_aug[:1])
    xgb_lat1 = ((time.perf_counter() - t_lat0) / 100.0) * 1000.0

    xgb_dev_probs = xgb_model.predict_proba(X_dev_aug)[:, 1]
    results["XGBoost"] = tracker.log_run(
        model_name="XGBoost",
        params={"model": "XGBoost", "max_depth": 6, "n_estimators": 400},
        y_true=y_dev,
        y_pred_proba=xgb_dev_probs,
        operational_metrics={
            "train_duration_sec": xgb_train_dur,
            "latency_b1_ms": xgb_lat1,
        },
        model=xgb_model,
        input_example=X_dev_aug[:2],
        registered_model_name="XGBoost_Baseline",
    )

    # 2. Random Forest
    t0 = time.perf_counter()
    rf_model = RandomForestFraudClassifier(
        n_estimators=50 if quick_mode else 200,
        max_depth=12,
        class_weight="balanced_subsample",
    )
    rf_model.fit(X_train_aug, y_train)
    rf_train_dur = time.perf_counter() - t0
    save_baseline_model(rf_model, "random_forest_baseline")

    t_lat0 = time.perf_counter()
    for _ in range(50):
        _ = rf_model.predict_proba(X_dev_aug[:1])
    rf_lat1 = ((time.perf_counter() - t_lat0) / 50.0) * 1000.0

    rf_dev_probs = rf_model.predict_proba(X_dev_aug)[:, 1]
    results["RandomForest"] = tracker.log_run(
        model_name="RandomForest",
        params={"model": "RandomForest", "n_estimators": 200, "max_depth": 12},
        y_true=y_dev,
        y_pred_proba=rf_dev_probs,
        operational_metrics={
            "train_duration_sec": rf_train_dur,
            "latency_b1_ms": rf_lat1,
        },
        model=rf_model,
        input_example=X_dev_aug[:2],
        registered_model_name="RandomForest_Baseline",
    )

    # 3. Calibrated Linear SVM
    t0 = time.perf_counter()
    svm_model = CalibratedLinearSVM(max_iter=1000 if quick_mode else 2000)
    svm_model.fit(X_train_aug, y_train)
    svm_train_dur = time.perf_counter() - t0
    save_baseline_model(svm_model, "linear_svm_baseline")

    t_lat0 = time.perf_counter()
    for _ in range(100):
        _ = svm_model.predict_proba(X_dev_aug[:1])
    svm_lat1 = ((time.perf_counter() - t_lat0) / 100.0) * 1000.0

    svm_dev_probs = svm_model.predict_proba(X_dev_aug)[:, 1]
    results["CalibratedSVM"] = tracker.log_run(
        model_name="CalibratedSVM",
        params={"model": "CalibratedLinearSVM", "loss": "hinge"},
        y_true=y_dev,
        y_pred_proba=svm_dev_probs,
        operational_metrics={
            "train_duration_sec": svm_train_dur,
            "latency_b1_ms": svm_lat1,
        },
        model=svm_model,
        input_example=X_dev_aug[:2],
        registered_model_name="CalibratedSVM_Baseline",
    )

    if not skip_deep:
        logger.info("=" * 70)
        logger.info("STAGE 5: DEEP SEQUENCE MODELING (BILSTM & TRANSFORMER)")
        logger.info("=" * 70)

        # Build sequence datasets
        window_L = 10 if quick_mode else 20
        train_loader, dev_loader, _ = build_dataloaders(
            train_data=(X_train_aug, y_train, meta_train),
            dev_data=(X_dev_aug, y_dev, meta_dev),
            window_length=window_L,
            batch_size=64 if quick_mode else 128,
        )

        feat_dim = X_train_aug.shape[1]

        # 4. Bidirectional LSTM
        bilstm = BiLSTMFraudModel(
            input_dim=feat_dim, hidden_size=64 if quick_mode else 128
        )
        lstm_trainer = DeepSequenceTrainer(
            model=bilstm,
            model_name="BiLSTM_Sequence",
            max_epochs=3 if quick_mode else 12,
            early_stopping_patience=3,
        )
        lstm_fit_res = lstm_trainer.fit(train_loader, dev_loader)
        lstm_lat = lstm_trainer.profile_inference_latency(
            seq_length=window_L, feature_dim=feat_dim
        )

        # Dev predictions
        bilstm.eval()
        lstm_preds = []
        with torch.no_grad():
            for x, _, m in dev_loader:
                p = bilstm.predict_proba(
                    x.to(lstm_trainer.device), m.to(lstm_trainer.device)
                )
                lstm_preds.extend(p.cpu().reshape(-1).numpy())
        lstm_preds_arr = np.array(lstm_preds)

        results["BiLSTM"] = tracker.log_run(
            model_name="BiLSTM",
            params={"hidden_size": 128, "num_layers": 2, "window_length": window_L},
            y_true=y_dev,
            y_pred_proba=lstm_preds_arr,
            operational_metrics={
                "train_duration_sec": lstm_fit_res["training_duration_sec"],
                "latency_b1_ms": lstm_lat["latency_b1_ms_per_sample"],
                "checkpoint_mb": lstm_lat["checkpoint_mb"],
            },
            model=bilstm,
            registered_model_name="BiLSTM_Sequence",
        )

        # 5. Transformer Encoder
        transformer = TransformerEncoderFraudModel(
            input_dim=feat_dim,
            d_model=64 if quick_mode else 128,
            nhead=4,
            num_layers=2 if quick_mode else 3,
        )
        tx_trainer = DeepSequenceTrainer(
            model=transformer,
            model_name="Transformer_Sequence",
            max_epochs=3 if quick_mode else 12,
            early_stopping_patience=3,
        )
        tx_fit_res = tx_trainer.fit(train_loader, dev_loader)
        tx_lat = tx_trainer.profile_inference_latency(
            seq_length=window_L, feature_dim=feat_dim
        )

        transformer.eval()
        tx_preds = []
        with torch.no_grad():
            for x, _, m in dev_loader:
                p = transformer.predict_proba(
                    x.to(tx_trainer.device), m.to(tx_trainer.device)
                )
                tx_preds.extend(p.cpu().reshape(-1).numpy())
        tx_preds_arr = np.array(tx_preds)

        results["Transformer"] = tracker.log_run(
            model_name="Transformer",
            params={
                "d_model": 128,
                "nhead": 4,
                "num_layers": 3,
                "window_length": window_L,
            },
            y_true=y_dev,
            y_pred_proba=tx_preds_arr,
            operational_metrics={
                "train_duration_sec": tx_fit_res["training_duration_sec"],
                "latency_b1_ms": tx_lat["latency_b1_ms_per_sample"],
                "checkpoint_mb": tx_lat["checkpoint_mb"],
            },
            model=transformer,
            registered_model_name="Transformer_Sequence",
        )

    if run_tune:
        logger.info("=" * 70)
        logger.info("STAGE 6: OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 70)
        tuner = OptunaHyperparameterTuner(n_trials=5 if quick_mode else 20)
        xgb_tune_res = tuner.tune_xgboost(X_train_aug, y_train, X_dev_aug, y_dev)
        logger.info(f"Tuned XGBoost Best PR-AUC: {xgb_tune_res['best_value']:.4f}")

    # Save summary dataframe for dashboard
    leaderboard_records = []
    for model_name, data in results.items():
        m = data["metrics"]
        leaderboard_records.append(
            {
                "Model": model_name,
                "PR-AUC": m.get("pr_auc", 0.0),
                "ROC-AUC": m.get("roc_auc", 0.0),
                "F1 (Fraud)": m.get("fraud_f1", 0.0),
                "Recall @ 95% Prec": m.get("recall_at_95_precision", 0.0),
                "Brier Score": m.get("brier_score", 0.0),
                "Training Time (s)": m.get("train_duration_sec", 0.0),
                "Latency (ms/sample)": m.get("latency_b1_ms", 0.0),
                "Model Size (MB)": m.get("checkpoint_mb", 1.5),
            }
        )

    df_leaderboard = pd.DataFrame(leaderboard_records)
    df_leaderboard.to_csv("models/artifacts/leaderboard.csv", index=False)
    logger.info("\n" + df_leaderboard.to_string(index=False))

    logger.info("=" * 70)
    logger.info("BENCHMARK PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Fraud Detection & Sequence Benchmark"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "ingest", "preprocess", "eda", "tune", "dashboard"],
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with small sample for rapid verification",
    )
    parser.add_argument(
        "--skip-deep",
        action="store_true",
        help="Skip deep sequence models (LSTM, Transformer)",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Run Bayesian optimization sweep"
    )
    args = parser.parse_args()

    if args.stage == "dashboard":
        import subprocess

        subprocess.run(
            ["streamlit", "run", "dashboard/app.py", "--server.port", "8501"],
            check=False,
        )
    elif args.stage == "all":
        run_full_pipeline(
            quick_mode=args.quick, skip_deep=args.skip_deep, run_tune=args.tune
        )
    elif args.stage == "ingest":
        run_ingestion(quick_sample_records=20000 if args.quick else None)
    elif args.stage == "preprocess":
        run_preprocessing()
    else:
        run_full_pipeline(quick_mode=args.quick)


if __name__ == "__main__":
    main()
