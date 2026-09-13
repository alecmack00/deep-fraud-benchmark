"""
MiniBatch K-Means Clustering on PCA Projections.
Segments behavioral spending profiles and generates Euclidean distance feature vectors.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.utils.logger import get_logger, timer

logger = get_logger("eda_baselines.clustering")


class KMeansClusterFeatureGenerator:
    """
    Fits MiniBatchKMeans on PCA projections and generates distance vectors to centroids.
    Fit strictly on training representations.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        batch_size: int = 2048,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state

        self.kmeans_: MiniBatchKMeans | None = None
        self.cluster_centers_: np.ndarray | None = None

    @timer
    def fit(self, X_pca_train: np.ndarray) -> "KMeansClusterFeatureGenerator":
        """
        Fit MiniBatchKMeans strictly on PCA-transformed training data.
        """
        logger.info(
            f"Fitting MiniBatchKMeans with K={self.n_clusters} clusters (batch_size={self.batch_size})..."
        )
        self.kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_init="auto",
        )
        self.kmeans_.fit(X_pca_train)
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        logger.info("MiniBatchKMeans fit complete.")
        return self

    def transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance vectors from each sample to all K cluster centroids:
        d(x, mu_k) = ||x - mu_k||_2
        Returns:
            distances: [N, K] float32 array
        """
        if self.kmeans_ is None or self.cluster_centers_ is None:
            raise RuntimeError(
                "KMeansClusterFeatureGenerator must be fitted before transform!"
            )

        # sklearn transform() returns Euclidean distances to each centroid
        distances = self.kmeans_.transform(X_pca).astype(np.float32)
        return distances

    def predict_clusters(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Assign each sample to its nearest centroid cluster index.
        """
        if self.kmeans_ is None:
            raise RuntimeError(
                "KMeansClusterFeatureGenerator must be fitted before predicting clusters!"
            )
        return self.kmeans_.predict(X_pca)

    def augment_features(self, X_tabular: np.ndarray, X_pca: np.ndarray) -> np.ndarray:
        """
        Append the K Euclidean distance features to the baseline tabular feature set.
        Pre-allocates C-contiguous output array to prevent memory spikes and fragmentation.
        """
        if len(X_tabular) != len(X_pca):
            raise ValueError(
                f"Row dimension mismatch: len(X_tabular)={len(X_tabular)} != len(X_pca)={len(X_pca)}"
            )

        distances = self.transform(X_pca)
        n_samples, n_tab_cols = X_tabular.shape
        n_dist_cols = distances.shape[1]

        augmented = np.empty(
            (n_samples, n_tab_cols + n_dist_cols), dtype=np.float32, order="C"
        )
        augmented[:, :n_tab_cols] = X_tabular
        augmented[:, n_tab_cols:] = distances
        return augmented

    def save(self, file_path: str = "models/checkpoints/kmeans_model.joblib") -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved KMeans model to {file_path}")

    @staticmethod
    def load(
        file_path: str = "models/checkpoints/kmeans_model.joblib",
    ) -> "KMeansClusterFeatureGenerator":
        if not Path(file_path).exists():
            raise FileNotFoundError(f"KMeans artifact not found at {file_path}")
        return joblib.load(file_path)
