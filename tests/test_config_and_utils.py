"""
Tests for YAML config parsing, Pydantic validation schemas, and logging utilities.
"""

import logging

import pytest
from pydantic import ValidationError

from src.utils.config_parser import (
    ClassicalModelsConfig,
    DataConfig,
    DeepModelsConfig,
    SplitRatiosConfig,
    load_classical_config,
    load_data_config,
    load_deep_config,
)
from src.utils.logger import get_logger, set_seed


def test_load_data_config_valid():
    """Verify load_data_config parses and validates configs/data_config.yaml."""
    cfg = load_data_config()
    assert isinstance(cfg, DataConfig)
    assert cfg.split_ratios.train == 0.98
    assert cfg.split_ratios.dev == 0.01
    assert cfg.split_ratios.test == 0.01
    assert cfg.cross_validation.purge_buffer_seconds == 86400.0
    assert cfg.cross_validation.purge_buffer_hours == 24.0
    assert cfg.sequence.min_history == 1


def test_split_ratios_sum_validator():
    """Verify SplitRatiosConfig enforces that split ratios sum to 1.0."""
    # Valid split
    valid = SplitRatiosConfig(train=0.80, dev=0.10, test=0.10)
    assert valid.train + valid.dev + valid.test == 1.0

    # Invalid split: exceeds 1.0
    with pytest.raises(ValidationError) as excinfo:
        SplitRatiosConfig(train=0.98, dev=0.05, test=0.05)
    assert "Split ratios must sum to 1.0" in str(excinfo.value)

    # Invalid split: under 1.0
    with pytest.raises(ValidationError) as excinfo:
        SplitRatiosConfig(train=0.50, dev=0.20, test=0.20)
    assert "Split ratios must sum to 1.0" in str(excinfo.value)


def test_load_classical_config_validation():
    """Verify load_classical_config returns a validated ClassicalModelsConfig instance."""
    cfg = load_classical_config()
    assert isinstance(cfg, ClassicalModelsConfig)

    # Typed sub-models
    assert cfg.tree_classifiers.xgboost.tree_method == "hist"
    assert cfg.tree_classifiers.xgboost.scale_pos_weight == "auto"
    assert cfg.tree_classifiers.linear_svm.loss == "hinge"
    assert cfg.tree_classifiers.random_forest.class_weight == "balanced_subsample"
    assert cfg.decomposition.pca.variance_threshold == 0.90
    assert cfg.clustering.minibatch_kmeans.n_clusters == 8

    # Ensure validation fails on schema error/bad values
    with pytest.raises(ValidationError):
        ClassicalModelsConfig.model_validate(
            {"tree_classifiers": {"xgboost": {"max_depth": "not_an_int"}}}
        )


def test_load_deep_config_validation():
    """Verify load_deep_config returns a validated DeepModelsConfig instance."""
    cfg = load_deep_config()
    assert isinstance(cfg, DeepModelsConfig)

    # Typed sub-models
    assert cfg.lstm.hidden_size == 128
    assert cfg.lstm.bidirectional is True
    assert cfg.transformer.d_model == 128
    assert cfg.transformer.activation == "gelu"
    assert cfg.loss.type == "focal"
    assert cfg.loss.pos_weight is None
    assert cfg.training["learning_rate"] == 0.0005

    # Ensure validation fails on schema error/bad values
    with pytest.raises(ValidationError):
        DeepModelsConfig.model_validate({"lstm": {"hidden_size": "not_an_int"}})


def test_set_seed_and_logger_no_duplicate_handlers():
    """Verify deterministic set_seed executes cleanly and get_logger avoids duplicate handlers."""
    set_seed(42)

    logger1 = get_logger("TestLogger")
    initial_stream_handlers = [
        h for h in logger1.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(initial_stream_handlers) == 1

    # Call get_logger again with the same name
    logger2 = get_logger("TestLogger")
    stream_handlers_after = [
        h for h in logger2.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(stream_handlers_after) == 1
    assert logger1 is logger2
    assert logger1.propagate is False
