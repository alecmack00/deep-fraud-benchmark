"""
Tests for Deep Sequence Architectures (BiLSTM and Transformer Encoder) and Focal Loss.
"""

import numpy as np
import pytest
import torch

from src.deep_models.lstm_network import BiLSTMFraudModel
from src.deep_models.trainer import BinaryFocalLoss
from src.deep_models.transformer_encoder import TransformerEncoderFraudModel


@pytest.fixture
def dummy_batch():
    B, L, D = 8, 20, 32
    torch.manual_seed(42)
    x = torch.randn(B, L, D)
    mask = torch.zeros(B, L, dtype=torch.bool)
    # Simulate left-padding for the first 10 steps in half the batch
    mask[:4, :10] = True
    targets = torch.randint(0, 2, (B, 1)).float()
    return x, mask, targets


def test_bilstm_output_shape_and_grad(dummy_batch):
    x, mask, targets = dummy_batch
    B, _L, D = x.shape

    model = BiLSTMFraudModel(input_dim=D, hidden_size=64, num_layers=2)
    logits = model(x, padding_mask=mask)

    assert logits.shape == (B, 1)
    assert not torch.isnan(logits).any()

    # Check backprop gradient flow
    loss_fn = BinaryFocalLoss()
    loss = loss_fn(logits, targets)
    loss.backward()

    # Check that model weights received valid gradients
    assert model.input_proj.weight.grad is not None
    assert not torch.isnan(model.input_proj.weight.grad).any()


def test_transformer_output_shape_and_grad(dummy_batch):
    x, mask, targets = dummy_batch
    B, _L, D = x.shape

    model = TransformerEncoderFraudModel(
        input_dim=D,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
    )
    logits = model(x, padding_mask=mask)

    assert logits.shape == (B, 1)
    assert not torch.isnan(logits).any()

    # Predict proba between 0 and 1
    probs = model.predict_proba(x, padding_mask=mask)
    assert probs.shape == (B, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Check backprop gradient flow
    loss_fn = BinaryFocalLoss()
    loss = loss_fn(logits, targets)
    loss.backward()

    assert model.feature_proj.weight.grad is not None
    assert not torch.isnan(model.feature_proj.weight.grad).any()


def test_focal_loss_numerical_stability():
    """
    Guarantees Focal Loss does not explode or return NaN even with extreme logits (+/- 100).
    """
    loss_fn = BinaryFocalLoss(alpha=0.75, gamma=2.0)

    extreme_logits = torch.tensor([[-100.0], [100.0], [0.0], [-50.0], [50.0]])
    targets = torch.tensor([[0.0], [1.0], [1.0], [1.0], [0.0]])

    loss = loss_fn(extreme_logits, targets)
    assert not torch.isnan(loss).item()
    assert not torch.isinf(loss).item()
    assert loss.item() >= 0.0


def test_transformer_cls_token_presence():
    """
    Verifies that the learnable [CLS] token is present and appropriately shaped.
    """
    model = TransformerEncoderFraudModel(input_dim=16, d_model=32)
    assert hasattr(model, "cls_token")
    assert model.cls_token.shape == (1, 1, 32)


def test_optuna_tuner_step():
    from src.tuning.optuna_tuner import OptunaHyperparameterTuner

    np.random.seed(42)
    X_train = np.random.randn(80, 8).astype(np.float32)
    y_train = (np.random.rand(80) > 0.8).astype(int)
    X_dev = np.random.randn(20, 8).astype(np.float32)
    y_dev = (np.random.rand(20) > 0.8).astype(int)

    tuner = OptunaHyperparameterTuner(n_trials=2, seed=42)
    res = tuner.tune_xgboost(X_train, y_train, X_dev, y_dev)
    assert "best_value" in res
    assert "best_params" in res
    assert "max_depth" in res["best_params"]

    # Test single-class y_dev guard
    y_dev_zeros = np.zeros(20, dtype=int)
    res_zero = tuner.tune_xgboost(X_train, y_train, X_dev, y_dev_zeros)
    assert res_zero["best_value"] == 0.0


def test_optuna_tune_lstm_and_cleanup():
    from torch.utils.data import DataLoader, TensorDataset

    from src.tuning.optuna_tuner import OptunaHyperparameterTuner, _cleanup_memory

    # 1. Test _cleanup_memory on CPU and MPS (if available)
    _cleanup_memory(torch.device("cpu"))
    if torch.backends.mps.is_available():
        _cleanup_memory(torch.device("mps"))

    # 2. Test tune_lstm with miniature loaders
    x_train = torch.randn(16, 5, 8)
    y_train = torch.randint(0, 2, (16, 1)).float()
    mask_train = torch.zeros(16, 5, dtype=torch.bool)
    train_ds = TensorDataset(x_train, y_train, mask_train)
    train_loader = DataLoader(train_ds, batch_size=4)

    x_dev = torch.randn(8, 5, 8)
    y_dev = torch.randint(0, 2, (8, 1)).float()
    mask_dev = torch.zeros(8, 5, dtype=torch.bool)
    dev_ds = TensorDataset(x_dev, y_dev, mask_dev)
    dev_loader = DataLoader(dev_ds, batch_size=4)

    tuner = OptunaHyperparameterTuner(n_trials=2, seed=42)
    res = tuner.tune_lstm(
        train_loader=train_loader,
        dev_loader=dev_loader,
        input_dim=8,
        epochs_per_trial=2,
    )
    assert "best_value" in res
    assert "best_params" in res
    assert "hidden_dim" in res["best_params"]


def test_loss_function_decoupling_and_config():
    """
    Verifies that get_loss_function decouples focal loss from pos_weight
    so no redundant >80x double-weighting can occur.
    """
    from torch import nn

    from src.deep_models.trainer import get_loss_function
    from src.utils.config_parser import LossConfig

    # 1. Focal loss ignores pos_weight
    focal_fn = get_loss_function(
        loss_type="focal", alpha=0.75, gamma=2.0, pos_weight=27.5
    )
    assert isinstance(focal_fn, BinaryFocalLoss)
    assert focal_fn.alpha == 0.75
    assert focal_fn.gamma == 2.0
    assert not hasattr(focal_fn, "pos_weight")

    # 2. Weighted BCE uses pos_weight
    bce_fn = get_loss_function(loss_type="weighted_bce", pos_weight=27.5)
    assert isinstance(bce_fn, nn.BCEWithLogitsLoss)
    assert bce_fn.pos_weight is not None
    assert torch.isclose(bce_fn.pos_weight, torch.tensor([27.5])).item()

    # 3. LossConfig pydantic model defaults
    cfg = LossConfig()
    assert cfg.type == "focal"
    assert cfg.pos_weight is None
    assert cfg.focal_alpha == 0.75


def test_trainer_amp_guard():
    """
    Verifies that DeepSequenceTrainer only enables AMP when running on CUDA,
    safely bypassing AMP on CPU and Apple Silicon MPS.
    """
    from src.deep_models.trainer import DeepSequenceTrainer

    model = BiLSTMFraudModel(input_dim=8, hidden_size=16)

    # 1. On CPU, AMP must always be False
    cpu_trainer = DeepSequenceTrainer(
        model=model,
        device_str="cpu",
        use_amp=True,
    )
    assert cpu_trainer.use_amp is False

    # 2. If MPS is available, AMP must also be safely False
    if torch.backends.mps.is_available():
        mps_trainer = DeepSequenceTrainer(
            model=model,
            device_str="mps",
            use_amp=True,
        )
        assert mps_trainer.use_amp is False


def test_sequence_attention_pooling_gradient_isolation_and_fp16():
    """
    Verifies that SequenceAttentionPooling:
    1. Prevents FP16 overflow / NaNs when masking out padded positions.
    2. Completely isolates padded positions from receiving backprop gradients.
    """
    from src.deep_models.lstm_network import SequenceAttentionPooling

    pooling = SequenceAttentionPooling(hidden_dim=16)

    # 1. Test FP16 stability
    x_fp16 = torch.randn(2, 6, 16, dtype=torch.float16)
    mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, False, False, False, False],
        ]
    )
    # Forward in fp16
    pooling_fp16 = SequenceAttentionPooling(hidden_dim=16).to(torch.float16)
    pooled_fp16 = pooling_fp16(x_fp16, mask=mask)
    assert not torch.isnan(pooled_fp16).any()
    assert not torch.isinf(pooled_fp16).any()

    # 2. Test gradient isolation on padded positions
    x = torch.randn(2, 6, 16, requires_grad=True)
    pooled = pooling(x, mask=mask)
    loss = pooled.sum()
    loss.backward()

    # Padded steps (indices 0, 1, 2 for row 0; indices 0, 1 for row 1) must have grad 0.0
    assert torch.equal(x.grad[0, :3], torch.zeros_like(x.grad[0, :3]))
    assert torch.equal(x.grad[1, :2], torch.zeros_like(x.grad[1, :2]))
    # Valid steps must receive valid non-zero gradients
    assert (x.grad[0, 3:].abs() > 0.0).any()
    assert (x.grad[1, 2:].abs() > 0.0).any()


def test_binary_focal_loss_1d_target_broadcasting_guard():
    """
    Verifies that BinaryFocalLoss handles 1D targets [B] with 2D logits [B, 1]
    without expanding into a [B, B] loss matrix or raising dimension mismatch errors.
    """
    loss_fn = BinaryFocalLoss()
    logits = torch.randn(12, 1)
    targets_1d = torch.randint(0, 2, (12,)).float()

    # Must evaluate without error and compute scalar mean
    loss = loss_fn(logits, targets_1d)
    assert loss.ndim == 0
    assert not torch.isnan(loss)

    # Sum reduction check
    loss_sum = BinaryFocalLoss(reduction="sum")(logits, targets_1d)
    assert loss_sum.ndim == 0
    assert torch.isclose(loss_sum, loss * 12)


def test_trainer_evaluate_single_sample_batch():
    """
    Verifies that DeepSequenceTrainer.evaluate() handles batches of size 1
    without raising TypeError: iteration over a 0-d array.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from src.deep_models.trainer import DeepSequenceTrainer

    model = BiLSTMFraudModel(input_dim=8, hidden_size=16)
    trainer = DeepSequenceTrainer(model=model, device_str="cpu", use_amp=False)

    # Exactly 1 sample in dataset
    x_single = torch.randn(1, 10, 8)
    y_single = torch.tensor([1.0])
    mask_single = torch.zeros(1, 10, dtype=torch.bool)

    dataset = TensorDataset(x_single, y_single, mask_single)
    loader = DataLoader(dataset, batch_size=1)

    metrics = trainer.evaluate(loader)
    assert "pr_auc" in metrics
    assert "loss" in metrics
    assert not np.isnan(metrics["loss"])
