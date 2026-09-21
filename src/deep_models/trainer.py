"""
PyTorch Deep Learning Training Pipeline with Focal Loss, AMP, and Early Stopping on PR-AUC.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.utils.logger import get_logger, timer

logger = get_logger("deep_models.trainer")


class BinaryFocalLoss(nn.Module):
    """
    Numerically stable Binary Focal Loss operating on logits:
    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where alpha_t = alpha for y=1, (1-alpha) for y=0.
    """

    def __init__(
        self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Guarantee shape alignment to prevent [B, 1] vs [B] broadcasting
        targets = targets.view_as(logits).float()

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        # p_t is the probability of the true class
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        # alpha_t is class weighting
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(
    loss_type: str = "focal",
    alpha: float = 0.75,
    gamma: float = 2.0,
    pos_weight: float | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    if loss_type == "focal":
        if pos_weight is not None:
            logger.debug(
                "pos_weight is ignored when loss_type is 'focal' to prevent redundant >80x double-weighting."
            )
        return BinaryFocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == "weighted_bce":
        effective_pw = 27.5 if pos_weight is None else pos_weight
        pw_tensor = torch.tensor([effective_pw], dtype=torch.float32)
        if device is not None:
            pw_tensor = pw_tensor.to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    else:
        return nn.BCEWithLogitsLoss()


def get_default_device(preference: str = "auto") -> torch.device:
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif (
        preference in ["auto", "mps"]
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    elif preference == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DeepSequenceTrainer:
    """
    End-to-end trainer for LSTM and Transformer sequence architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "sequence_model",
        lr: float = 0.0005,
        weight_decay: float = 1e-4,
        loss_type: str = "focal",
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        pos_weight: float | None = None,
        max_epochs: int = 20,
        early_stopping_patience: int = 5,
        grad_clip_norm: float = 1.0,
        checkpoint_dir: str = "models/checkpoints",
        device_str: str = "auto",
        use_amp: bool = True,
    ):
        self.device = get_default_device(device_str)
        self.model = model.to(self.device)
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = early_stopping_patience
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / f"{model_name}.pt"

        self.loss_fn = get_loss_function(
            loss_type=loss_type,
            alpha=focal_alpha,
            gamma=focal_gamma,
            pos_weight=pos_weight,
            device=self.device,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        # Automatic Mixed Precision
        # CUDA natively supports GradScaler and torch.autocast.
        # On Apple Silicon (MPS), GradScaler is unavailable and fp16 autocast in older PyTorch
        # versions can cause unstable fallbacks for recurrent/attention primitives.
        # Therefore, AMP is strictly enabled on CUDA.
        self.use_amp = bool(use_amp and (self.device.type == "cuda"))
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if use_amp and self.device.type == "mps":
            logger.info(
                "AMP requested with MPS device: safely disabling AMP on Apple Silicon to prevent operator fallbacks."
            )

    @timer
    def fit(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader,
    ) -> dict[str, Any]:
        """
        Train sequence model with early stopping on Dev PR-AUC.
        """
        logger.info(
            f"Starting training for {self.model_name} on {self.device} (AMP={self.use_amp})..."
        )
        best_pr_auc = -1.0
        epochs_no_improve = 0
        history = {
            "train_loss": [],
            "dev_loss": [],
            "dev_pr_auc": [],
            "dev_roc_auc": [],
        }
        t0_train = time.perf_counter()

        for epoch in range(1, self.max_epochs + 1):
            # Training Phase
            self.model.train()
            train_losses = []

            for x_seq, y_target, padding_mask in train_loader:
                x_seq = x_seq.to(self.device)
                y_target = y_target.to(self.device)
                padding_mask = padding_mask.to(self.device)
                if self.device.type == "mps":
                    torch.mps.synchronize()

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.autocast(device_type="cuda"):
                        logits = self.model(x_seq, padding_mask)
                        loss = self.loss_fn(logits, y_target)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(x_seq, padding_mask)
                    loss = self.loss_fn(logits, y_target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                    self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            # Dev Evaluation Phase
            dev_metrics = self.evaluate(dev_loader)
            dev_pr_auc = dev_metrics["pr_auc"]
            dev_roc_auc = dev_metrics["roc_auc"]
            dev_loss = dev_metrics["loss"]

            self.scheduler.step(dev_pr_auc)

            history["train_loss"].append(avg_train_loss)
            history["dev_loss"].append(dev_loss)
            history["dev_pr_auc"].append(dev_pr_auc)
            history["dev_roc_auc"].append(dev_roc_auc)

            logger.info(
                f"Epoch {epoch:02d}/{self.max_epochs:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Dev Loss: {dev_loss:.4f} | "
                f"Dev PR-AUC: {dev_pr_auc:.4f} | "
                f"Dev ROC-AUC: {dev_roc_auc:.4f}"
            )

            # Early stopping check on Dev PR-AUC
            if dev_pr_auc > best_pr_auc:
                best_pr_auc = dev_pr_auc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                logger.info(
                    f"Saved new best model checkpoint to {self.checkpoint_path} (PR-AUC: {best_pr_auc:.4f})"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(
                        f"Early stopping triggered after {epoch} epochs. Best Dev PR-AUC: {best_pr_auc:.4f}"
                    )
                    break

        total_training_time = time.perf_counter() - t0_train

        # Load best checkpoint
        if self.checkpoint_path.exists():
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )

        return {
            "best_dev_pr_auc": best_pr_auc,
            "total_epochs": epoch,
            "training_duration_sec": total_training_time,
            "history": history,
            "checkpoint_path": str(self.checkpoint_path),
        }

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate model on DataLoader and return loss, PR-AUC, and ROC-AUC.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        val_losses = []

        with torch.no_grad():
            for x_seq, y_target, padding_mask in dataloader:
                # Capture host targets directly to prevent async DMA hazards on unified memory (MPS)
                all_targets.extend(y_target.reshape(-1).numpy())

                x_seq = x_seq.to(self.device)
                y_target = y_target.to(self.device)
                padding_mask = padding_mask.to(self.device)
                if self.device.type == "mps":
                    torch.mps.synchronize()

                if self.use_amp:
                    with torch.autocast(device_type="cuda"):
                        logits = self.model(x_seq, padding_mask)
                        loss = self.loss_fn(logits, y_target)
                else:
                    logits = self.model(x_seq, padding_mask)
                    loss = self.loss_fn(logits, y_target)

                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                if self.device.type == "mps":
                    torch.mps.synchronize()

                # Use reshape(-1) to safely flatten even if batch size is 1
                all_preds.extend(probs.cpu().reshape(-1).numpy())

        preds_arr = np.array(all_preds, dtype=np.float32)
        targets_arr = np.array(all_targets, dtype=np.float32)

        # Imbalance-aware metric computation with single-class guards
        if len(np.unique(targets_arr)) > 1 and not np.isnan(preds_arr).any():
            pr_auc = float(average_precision_score(targets_arr, preds_arr))
            roc_auc = float(roc_auc_score(targets_arr, preds_arr))
        else:
            pr_auc = 0.0
            roc_auc = 0.5

        avg_loss = float(np.mean(val_losses)) if val_losses else 0.0

        return {
            "loss": avg_loss,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
        }

    def profile_inference_latency(
        self,
        seq_length: int = 20,
        feature_dim: int | None = None,
        num_reps: int = 100,
    ) -> dict[str, float]:
        """
        Measure inference latency in ms/sample for batch size 1 and batch size 64.
        """
        self.model.eval()
        d = feature_dim or getattr(self.model, "input_dim", 32)
        latencies = {}

        for batch_size in [1, 64]:
            dummy_x = torch.randn(batch_size, seq_length, d, device=self.device)
            dummy_mask = torch.zeros(
                batch_size, seq_length, dtype=torch.bool, device=self.device
            )

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_x, dummy_mask)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_reps):
                    _ = self.model(dummy_x, dummy_mask)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - t0
            ms_per_sample = (elapsed / (num_reps * batch_size)) * 1000.0
            latencies[f"latency_b{batch_size}_ms_per_sample"] = ms_per_sample

        # Model checkpoint footprint
        if self.checkpoint_path.exists():
            size_mb = self.checkpoint_path.stat().st_size / (1024 * 1024)
            latencies["checkpoint_mb"] = size_mb
        else:
            latencies["checkpoint_mb"] = 0.0

        return latencies
