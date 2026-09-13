"""
Leakage-Free Preprocessing and Out-of-Time Splitting Pipeline.
Strictly fits all stateful transformations (imputers, scalers, correlation filters)
exclusively on the training split.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils.config_parser import load_data_config
from src.utils.logger import get_logger, timer

logger = get_logger("data.preprocessor")


def temporal_train_dev_test_split(
    df: pl.DataFrame,
    time_col: str = "TransactionDT",
    train_ratio: float = 0.98,
    dev_ratio: float = 0.01,
    test_ratio: float = 0.01,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Strict out-of-time chronological splitting.
    Transactions are sorted by time_col.
    Train: [t_0, t_1] (first train_ratio %)
    Dev:   (t_1, t_2] (intermediate dev_ratio %)
    Test:  (t_2, t_3] (strictly held-out future test_ratio %)
    """
    total_ratio = train_ratio + dev_ratio + test_ratio
    assert np.isclose(total_ratio, 1.0), f"Ratios must sum to 1.0, got {total_ratio}"

    sorted_df = df.sort(time_col)
    n_total = sorted_df.shape[0]

    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    # Ensure non-empty dev and test splits even on small sample runs
    if n_dev == 0 and n_total >= 10:
        n_dev = max(1, int(n_total * 0.1))
        n_train = n_total - 2 * n_dev

    train_df = sorted_df.slice(0, n_train)
    dev_df = sorted_df.slice(n_train, n_dev)
    test_df = sorted_df.slice(n_train + n_dev, n_total - (n_train + n_dev))

    logger.info(
        f"Temporal Partition: Train={train_df.shape[0]} rows "
        f"({train_df[time_col].min()} -> {train_df[time_col].max()}), "
        f"Dev={dev_df.shape[0]} rows "
        f"({dev_df[time_col].min()} -> {dev_df[time_col].max()}), "
        f"Test={test_df.shape[0]} rows "
        f"({test_df[time_col].min()} -> {test_df[time_col].max()})"
    )

    # Verification of strict no-lookahead boundary
    assert (
        train_df[time_col].max() <= dev_df[time_col].min()
    ), "Leakage detected: Train max DT > Dev min DT!"
    assert (
        dev_df[time_col].max() <= test_df[time_col].min()
    ), "Leakage detected: Dev max DT > Test min DT!"

    return train_df, dev_df, test_df


def compute_cyclical_features(
    df: pl.DataFrame, time_col: str = "TransactionDT"
) -> pl.DataFrame:
    """
    Add cyclical day-of-week and hour-of-day features:
    sin/cos(2*pi*hour/24), sin/cos(2*pi*day/7)
    TransactionDT is elapsed seconds from reference point.
    """
    # 86400 seconds per day, 3600 seconds per hour
    seconds_in_day = 86400.0
    seconds_in_hour = 3600.0
    days_in_week = 7.0

    hour_expr = (pl.col(time_col) % seconds_in_day) / seconds_in_hour
    day_expr = (pl.col(time_col) / seconds_in_day) % days_in_week

    two_pi = 2.0 * np.pi

    sin_hour = (hour_expr * (two_pi / 24.0)).sin().cast(pl.Float32).alias("hour_sin")
    cos_hour = (hour_expr * (two_pi / 24.0)).cos().cast(pl.Float32).alias("hour_cos")
    sin_day = (day_expr * (two_pi / 7.0)).sin().cast(pl.Float32).alias("day_sin")
    cos_day = (day_expr * (two_pi / 7.0)).cos().cast(pl.Float32).alias("day_cos")

    return df.with_columns([sin_hour, cos_hour, sin_day, cos_day])


class HighCorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Prunes numerical features exceeding correlation_threshold on the train set.
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold
        self.drop_features_: list[str] = []
        self.keep_features_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        corr_matrix = X.corr(method="pearson").abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.drop_features_ = [
            col for col in upper.columns if any(upper[col] > self.threshold)
        ]
        self.keep_features_ = [
            col for col in X.columns if col not in self.drop_features_
        ]
        logger.info(
            f"HighCorrelationFilter: Pruned {len(self.drop_features_)} redundant features (threshold={self.threshold}). Retained {len(self.keep_features_)}."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.keep_features_]


class FraudDataPreprocessor:
    """
    Stateful preprocessor that guarantees 100% zero-leakage transforms.
    Fitted strictly on training data.
    """

    def __init__(
        self,
        target_col: str = "isFraud",
        time_col: str = "TransactionDT",
        entity_col: str = "card_id",
        correlation_threshold: float = 0.90,
    ):
        self.target_col = target_col
        self.time_col = time_col
        self.entity_col = entity_col
        self.correlation_threshold = correlation_threshold

        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.metadata_cols: list[str] = [
            "TransactionID",
            self.time_col,
            self.entity_col,
        ]

        self.corr_filter = HighCorrelationFilter(threshold=self.correlation_threshold)
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_encoder: OrdinalEncoder | None = None
        self.cat_categories_: dict[str, list[str]] = {}
        self.feature_names_out_: list[str] = []
        self.is_fitted: bool = False

    def _identify_column_types(self, df: pd.DataFrame) -> None:
        exclude_cols = set(self.metadata_cols + [self.target_col])
        candidate_cols = [c for c in df.columns if c not in exclude_cols]

        self.num_cols = []
        self.cat_cols = []
        for col in candidate_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.num_cols.append(col)
            else:
                self.cat_cols.append(col)

        logger.info(
            f"Identified {len(self.num_cols)} numerical features and {len(self.cat_cols)} categorical features."
        )

    @timer
    def fit(self, train_df_pl: pl.DataFrame) -> "FraudDataPreprocessor":
        """
        Fit imputers, correlation filter, and scaler strictly on the train split.
        """
        logger.info("Fitting FraudDataPreprocessor strictly on Train split...")
        train_df_pl = compute_cyclical_features(train_df_pl, time_col=self.time_col)
        df_train = train_df_pl.to_pandas()

        self._identify_column_types(df_train)

        # 1. High correlation pruning on raw continuous numerics (excluding cyclical engineered signals)
        cyclical_cols = [
            c
            for c in ["hour_sin", "hour_cos", "day_sin", "day_cos"]
            if c in self.num_cols
        ]
        raw_num_cols = [c for c in self.num_cols if c not in cyclical_cols]

        df_num = df_train[raw_num_cols]
        self.corr_filter.fit(df_num)
        pruned_num_cols = self.corr_filter.keep_features_
        # Always retain cyclical features as domain signal
        self.num_cols = pruned_num_cols + cyclical_cols

        # 2. Fit median imputer on pruned numerics + cyclical features
        self.num_imputer.fit(df_train[self.num_cols].to_numpy())

        # 3. Fit scaler on imputed numerics
        imputed_num = self.num_imputer.transform(df_train[self.num_cols].to_numpy())
        self.scaler.fit(imputed_num)

        # 4. Fit vectorized OrdinalEncoder with unknown handling (no category collision)
        if self.cat_cols:
            df_cat = df_train[self.cat_cols].fillna("MISSING").astype(str)
            self.cat_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            )
            self.cat_encoder.fit(df_cat)
            self.cat_categories_ = {
                col: list(cats)
                for col, cats in zip(self.cat_cols, self.cat_encoder.categories_)
            }
        else:
            self.cat_encoder = None
            self.cat_categories_ = {}

        # Build feature output ordering
        self.feature_names_out_ = [f"num_{c}" for c in self.num_cols] + [
            f"cat_{c}" for c in self.cat_cols
        ]
        self.is_fitted = True
        logger.info(
            f"Preprocessor fitted successfully. Total feature dimensionality: {len(self.feature_names_out_)}"
        )
        return self

    @timer
    def transform(
        self,
        df_pl: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
        """
        Apply preprocessor transforms to any split (Train, Dev, Test).
        Returns:
            X: Standardized and encoded feature matrix [N, D] as float32 np.ndarray
            y: Binary target array [N] or None
            meta_df: DataFrame containing TransactionID, TransactionDT, card_id
        """
        if not self.is_fitted:
            raise RuntimeError("FraudDataPreprocessor must be fitted before transform!")

        df_pl = compute_cyclical_features(df_pl, time_col=self.time_col)
        df = df_pl.to_pandas()

        # Metadata extraction
        meta_dict = {}
        for col in self.metadata_cols:
            meta_dict[col] = df[col] if col in df.columns else np.nan
        meta_df = pd.DataFrame(meta_dict)

        # Target extraction
        y = (
            df[self.target_col].to_numpy(dtype=np.float32)
            if self.target_col in df.columns
            else None
        )

        # Numerical transform: impute medians -> scale
        num_raw = df[self.num_cols].to_numpy()
        num_imputed = self.num_imputer.transform(num_raw)
        num_scaled = self.scaler.transform(num_imputed).astype(np.float32)

        # Categorical transform: vectorized OrdinalEncoder with normalized non-colliding scaling
        if self.cat_cols and self.cat_encoder is not None:
            df_cat = df[self.cat_cols].fillna("MISSING").astype(str)
            encoded = self.cat_encoder.transform(df_cat)
            # Map unknown (-1.0) to 0.0, and map known categories (0 .. K-1) to (1 .. K) / (K + 1)
            cat_dim_scales = np.array(
                [
                    max(1.0, float(len(cats) + 1.0))
                    for cats in self.cat_encoder.categories_
                ],
                dtype=np.float32,
            )
            cat_matrix = ((encoded + 1.0) / cat_dim_scales).astype(np.float32)
            X = np.hstack([num_scaled, cat_matrix]).astype(np.float32)
        else:
            X = num_scaled

        return X, y, meta_df

    def fit_transform(
        self,
        train_df_pl: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
        self.fit(train_df_pl)
        return self.transform(train_df_pl)

    def save(self, file_path: str = "models/checkpoints/preprocessor.joblib") -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved preprocessor artifact to {file_path}")

    @staticmethod
    def load(
        file_path: str = "models/checkpoints/preprocessor.joblib",
    ) -> "FraudDataPreprocessor":
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Preprocessor file not found at {file_path}")
        return joblib.load(file_path)


def run_preprocessing(
    config_path: str = "configs/data_config.yaml",
) -> tuple[FraudDataPreprocessor, dict[str, str]]:
    """
    Orchestrates out-of-time partition and leakage-free preprocessing.
    Persists train/dev/test split parquets and preprocessor.joblib.
    """
    cfg = load_data_config(config_path)

    merged_parquet = cfg.processed_paths.merged_parquet
    if not Path(merged_parquet).exists():
        raise FileNotFoundError(
            f"Processed transactions not found at {merged_parquet}. Run ingestion first."
        )

    logger.info(f"Loading merged transactions from {merged_parquet}...")
    df = pl.read_parquet(merged_parquet)

    # 1. Temporal Partitioning (default 98% Train, 1% Dev, 1% Test)
    train_df, dev_df, test_df = temporal_train_dev_test_split(
        df=df,
        time_col=cfg.schema.time_col,
        train_ratio=cfg.split_ratios.train,
        dev_ratio=cfg.split_ratios.dev,
        test_ratio=cfg.split_ratios.test,
    )

    # Save partitioned splits
    train_df.write_parquet(cfg.processed_paths.train_parquet, compression="zstd")
    dev_df.write_parquet(cfg.processed_paths.dev_parquet, compression="zstd")
    test_df.write_parquet(cfg.processed_paths.test_parquet, compression="zstd")

    # 2. Fit preprocessor strictly on train
    preprocessor = FraudDataPreprocessor(
        target_col=cfg.schema.target_col,
        time_col=cfg.schema.time_col,
        entity_col=cfg.schema.synthesized_entity_col,
        correlation_threshold=cfg.feature_engineering.correlation_threshold,
    )
    preprocessor.fit(train_df)
    preprocessor.save(cfg.processed_paths.preprocessor_artifact)

    output_paths = {
        "train": cfg.processed_paths.train_parquet,
        "dev": cfg.processed_paths.dev_parquet,
        "test": cfg.processed_paths.test_parquet,
        "preprocessor": cfg.processed_paths.preprocessor_artifact,
    }

    return preprocessor, output_paths
