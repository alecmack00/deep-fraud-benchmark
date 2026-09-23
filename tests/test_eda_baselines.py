"""
Tests for Dimensionality Reduction (PCA), Clustering (MiniBatchKMeans), and Classical Classifiers.
"""

import numpy as np
import pytest

from src.eda_baselines.clustering import KMeansClusterFeatureGenerator
from src.eda_baselines.decomposition import PCARepresentationLearner
from src.eda_baselines.tree_classifiers import (
    CalibratedLinearSVM,
    RandomForestFraudClassifier,
    XGBoostFraudClassifier,
)


@pytest.fixture
def dummy_train_dev_data():
    np.random.seed(42)
    n_train = 300
    n_dev = 100
    n_features = 20

    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    # 5% fraud imbalance
    y_train = (np.random.uniform(0, 1, size=n_train) < 0.05).astype(int)

    X_dev = np.random.randn(n_dev, n_features).astype(np.float32)
    y_dev = (np.random.uniform(0, 1, size=n_dev) < 0.05).astype(int)

    return X_train, y_train, X_dev, y_dev


def test_pca_monotonicity_and_variance(dummy_train_dev_data):
    X_train, _, X_dev, _ = dummy_train_dev_data
    pca = PCARepresentationLearner(variance_threshold=0.85)
    pca.fit(X_train)

    exp_var = pca.explained_variance_ratio_
    # Monotonically non-increasing
    for i in range(len(exp_var) - 1):
        assert (
            exp_var[i] >= exp_var[i + 1] - 1e-6
        ), "Explained variance must be non-increasing!"

    # All non-negative
    assert (exp_var >= 0.0).all()

    # Transform test
    dev_proj = pca.transform(X_dev)
    assert dev_proj.shape == (X_dev.shape[0], pca.n_components_)

    # 3D coordinates extraction
    coords_3d = pca.extract_3d_coordinates(X_dev)
    assert coords_3d.shape == (X_dev.shape[0], 3)


def test_kmeans_distances_and_augmentation(dummy_train_dev_data):
    X_train, _, X_dev, _ = dummy_train_dev_data
    kmeans = KMeansClusterFeatureGenerator(n_clusters=6)
    kmeans.fit(X_train)

    # Distances to 6 centroids
    dists = kmeans.transform(X_dev)
    assert dists.shape == (X_dev.shape[0], 6)
    assert (dists >= 0.0).all(), "Euclidean distances must be non-negative!"

    # Augmentation check
    augmented = kmeans.augment_features(X_dev, X_dev)
    assert augmented.shape == (X_dev.shape[0], X_dev.shape[1] + 6)


def test_tree_classifiers_predict_proba(dummy_train_dev_data):
    X_train, y_train, X_dev, _ = dummy_train_dev_data

    classifiers = [
        ("XGBoost", XGBoostFraudClassifier(n_estimators=20, max_depth=3)),
        ("RandomForest", RandomForestFraudClassifier(n_estimators=10, max_depth=5)),
        ("CalibratedLinearSVM", CalibratedLinearSVM(max_iter=200)),
    ]

    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_dev)

        assert probs.shape == (
            X_dev.shape[0],
            2,
        ), f"{name} predict_proba shape mismatch"
        assert (probs >= 0.0).all() and (
            probs <= 1.0
        ).all(), f"{name} probabilities outside [0, 1]"
        row_sums = np.sum(probs, axis=1)
        assert np.allclose(
            row_sums, 1.0, atol=1e-4
        ), f"{name} probability rows must sum to 1.0"


def test_xgboost_dynamic_scale_pos_weight():
    """
    Verifies that XGBoostFraudClassifier properly handles 'auto', None, explicit float,
    and falls back to 27.5 when no positive samples exist.
    """
    X = np.random.randn(50, 4).astype(np.float32)
    # 5 positives, 45 negatives -> ratio 9.0
    y = np.array([1] * 5 + [0] * 45)

    # 1. 'auto'
    clf_auto = XGBoostFraudClassifier(n_estimators=5, scale_pos_weight="auto")
    clf_auto.fit(X, y)
    assert clf_auto.model_ is not None
    assert clf_auto.model_.get_params()["scale_pos_weight"] == 9.0

    # 2. None
    clf_none = XGBoostFraudClassifier(n_estimators=5, scale_pos_weight=None)
    clf_none.fit(X, y)
    assert clf_none.model_.get_params()["scale_pos_weight"] == 9.0

    # 3. Explicit float
    clf_float = XGBoostFraudClassifier(n_estimators=5, scale_pos_weight=15.5)
    clf_float.fit(X, y)
    assert clf_float.model_.get_params()["scale_pos_weight"] == 15.5

    # 4. Fallback on 0 positives
    y_zeros = np.zeros(50, dtype=int)
    clf_fallback = XGBoostFraudClassifier(n_estimators=5, scale_pos_weight="auto")
    clf_fallback.fit(X, y_zeros)
    assert clf_fallback.model_.get_params()["scale_pos_weight"] == 27.5


def test_calibrated_linear_svm_hinge_loss():
    """
    Verifies that CalibratedLinearSVM uses true Linear SVM (loss='hinge')
    and Platt calibration via CalibratedClassifierCV.
    """
    X = np.random.randn(60, 5).astype(np.float32)
    y = np.array([1] * 10 + [0] * 50)

    svm = CalibratedLinearSVM(loss="hinge", max_iter=200)
    assert svm.loss == "hinge"
    svm.fit(X, y)

    # Base estimator in CalibratedClassifierCV calibrated_classifiers_
    base_est = svm.model_.estimator
    assert base_est.loss == "hinge"

    probs = svm.predict_proba(X)
    assert probs.shape == (60, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)


def test_pca_threshold_fallback_no_zero_bug():
    """
    Verifies that when cumulative variance threshold is not met within n_eval,
    PCARepresentationLearner does not fall back to 0 (the old np.argmax bug on all-False arrays).
    """
    X = np.random.randn(100, 20).astype(np.float32)
    # Require 99.999% variance but limit target evaluation to 5 components
    pca = PCARepresentationLearner(variance_threshold=0.99999, n_components=5)
    pca.fit(X)

    assert pca.n_components_ == 5
    assert len(pca.cumulative_variance_ratio_) == 5
    projections = pca.transform(X)
    assert projections.shape == (100, 5)


def test_kmeans_augmentation_memory_and_mismatch(dummy_train_dev_data):
    """
    Verifies that augment_features pre-allocates a C-contiguous float32 buffer
    and raises ValueError when row counts do not match.
    """
    X_train, _, X_dev, _ = dummy_train_dev_data
    kmeans = KMeansClusterFeatureGenerator(n_clusters=4)
    kmeans.fit(X_train)

    # Contiguity and dtype test
    aug = kmeans.augment_features(X_dev, X_dev)
    assert aug.flags.c_contiguous
    assert aug.dtype == np.float32
    assert aug.shape == (X_dev.shape[0], X_dev.shape[1] + 4)

    # Row mismatch test
    with pytest.raises(ValueError, match="Row dimension mismatch"):
        kmeans.augment_features(X_dev[:10], X_dev[:20])


def test_xgboost_eval_set_formats(dummy_train_dev_data):
    """
    Verifies that XGBoostFraudClassifier gracefully handles both single tuple (X_dev, y_dev)
    and list of tuples [(X_dev, y_dev)] without ValueError in XGBoost >= 2.0.
    """
    X_train, y_train, X_dev, y_dev = dummy_train_dev_data

    # 1. Single tuple format
    clf_tuple = XGBoostFraudClassifier(n_estimators=5, early_stopping_rounds=2)
    clf_tuple.fit(X_train, y_train, eval_set=(X_dev, y_dev))
    probs_t = clf_tuple.predict_proba(X_dev)
    assert probs_t.shape == (X_dev.shape[0], 2)

    # 2. List of tuples format
    clf_list = XGBoostFraudClassifier(n_estimators=5, early_stopping_rounds=2)
    clf_list.fit(X_train, y_train, eval_set=[(X_dev, y_dev)])
    probs_l = clf_list.predict_proba(X_dev)
    assert probs_l.shape == (X_dev.shape[0], 2)
