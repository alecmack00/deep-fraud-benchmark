"""
Principal Component Analysis (PCA) for Latent Representation Discovery.
Fitted strictly on standardized training data.
Computes explained variance ratios and selects k components capturing >= 90% variance.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA

from src.utils.logger import get_logger, timer

logger = get_logger("eda_baselines.decomposition")


class PCARepresentationLearner:
    """
    Learns low-dimensional orthogonal representations of standardized transaction features.
    Strictly fit on training features.
    """

    def __init__(
        self,
        variance_threshold: float = 0.90,
        n_components: int | None = None,
        batch_size: int = 4096,
        random_state: int = 42,
    ):
        self.variance_threshold = variance_threshold
        self.n_components_target = n_components
        self.batch_size = batch_size
        self.random_state = random_state

        self.pca_: IncrementalPCA | None = None
        self.n_components_: int = 0
        self.explained_variance_ratio_: np.ndarray = np.array([])
        self.cumulative_variance_ratio_: np.ndarray = np.array([])

    @timer
    def fit(self, X_train: np.ndarray) -> "PCARepresentationLearner":
        """
        Fit IncrementalPCA on training data and determine minimum components for variance_threshold.
        Uses chunked mini-batches for memory efficiency on big tabular data.
        """
        n_samples, n_features = X_train.shape
        max_possible = min(n_samples, n_features)

        logger.info(
            f"Fitting IncrementalPCA across {n_features} features (batch_size={self.batch_size})..."
        )

        n_eval = (
            max_possible
            if self.n_components_target is None
            else min(max_possible, self.n_components_target)
        )

        # Ensure batch_size is at least n_eval if specified
        batch_size = (
            max(self.batch_size, n_eval) if self.batch_size is not None else None
        )

        self.pca_ = IncrementalPCA(
            n_components=n_eval,
            batch_size=batch_size,
        )
        self.pca_.fit(X_train)

        cum_var = np.cumsum(self.pca_.explained_variance_ratio_)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        self.cumulative_variance_ratio_ = cum_var

        if self.n_components_target is not None:
            self.n_components_ = min(self.n_components_target, max_possible)
        else:
            valid_indices = np.where(cum_var >= self.variance_threshold)[0]
            if len(valid_indices) > 0:
                self.n_components_ = int(valid_indices[0] + 1)
            else:
                self.n_components_ = n_eval
            self.n_components_ = min(max(3, self.n_components_), max_possible)

        selected_idx = min(
            self.n_components_ - 1, len(self.cumulative_variance_ratio_) - 1
        )
        logger.info(
            f"Selected k={self.n_components_} components explaining "
            f"{self.cumulative_variance_ratio_[selected_idx] * 100:.2f}% cumulative variance "
            f"(threshold={self.variance_threshold * 100:.1f}%)."
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project standardized features into PCA latent space.
        Slices to the optimal k components.
        """
        if self.pca_ is None:
            raise RuntimeError(
                "PCARepresentationLearner must be fit before calling transform!"
            )
        projections = self.pca_.transform(X)
        return projections[:, : self.n_components_].astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)

    def extract_3d_coordinates(self, X: np.ndarray) -> np.ndarray:
        """
        Extract the first 3 principal component coordinates for 3D scatter plots.
        """
        projections = self.transform(X)
        return projections[:, :3]

    def save(self, file_path: str = "models/checkpoints/pca_model.joblib") -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved PCA model to {file_path}")

    @staticmethod
    def load(
        file_path: str = "models/checkpoints/pca_model.joblib",
    ) -> "PCARepresentationLearner":
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PCA artifact not found at {file_path}")
        return joblib.load(file_path)
