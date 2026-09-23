"""
YAML configuration parser with Pydantic validation.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class RawPathsConfig(BaseModel):
    transaction_csv: str = "data/raw/train_transaction.csv"
    identity_csv: str = "data/raw/train_identity.csv"
    paysim_csv: str = "data/raw/PS_2017364724410115.csv"


class ProcessedPathsConfig(BaseModel):
    merged_parquet: str = "data/processed/merged_transactions.parquet"
    train_parquet: str = "data/processed/train_transactions.parquet"
    dev_parquet: str = "data/processed/dev_transactions.parquet"
    test_parquet: str = "data/processed/test_transactions.parquet"
    sequence_tensors_dir: str = "data/processed/sequences"
    preprocessor_artifact: str = "models/checkpoints/preprocessor.joblib"


class SchemaConfig(BaseModel):
    target_col: str = "isFraud"
    time_col: str = "TransactionDT"
    join_key: str = "TransactionID"
    entity_cols: list[str] = Field(
        default_factory=lambda: ["card1", "card2", "card3", "card4", "addr1", "D1"]
    )
    synthesized_entity_col: str = "card_id"


class SplitRatiosConfig(BaseModel):
    train: float = 0.98
    dev: float = 0.01
    test: float = 0.01

    @model_validator(mode="after")
    def validate_ratios_sum(self) -> "SplitRatiosConfig":
        total = self.train + self.dev + self.test
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
        return self


class FeatureEngineeringConfig(BaseModel):
    correlation_threshold: float = 0.90
    cyclical_encoding: bool = True
    missing_categorical_token: str = "MISSING"


class SequenceConfig(BaseModel):
    window_length: int = 20
    stride: int = 1
    min_history: int = 1


class CrossValidationConfig(BaseModel):
    n_splits: int = 5
    purge_buffer_seconds: float = 86400.0
    purge_buffer_hours: float = 24.0

    @model_validator(mode="before")
    @classmethod
    def sync_purge_buffer(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "purge_buffer_seconds" in data and "purge_buffer_hours" not in data:
                data["purge_buffer_hours"] = (
                    float(data["purge_buffer_seconds"]) / 3600.0
                )
            elif "purge_buffer_hours" in data and "purge_buffer_seconds" not in data:
                data["purge_buffer_seconds"] = (
                    float(data["purge_buffer_hours"]) * 3600.0
                )
        return data


class DataConfig(BaseModel):
    model_config = {"populate_by_name": True}

    raw_paths: RawPathsConfig = Field(default_factory=RawPathsConfig)
    processed_paths: ProcessedPathsConfig = Field(default_factory=ProcessedPathsConfig)
    data_schema: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    split_ratios: SplitRatiosConfig = Field(default_factory=SplitRatiosConfig)
    cross_validation: CrossValidationConfig = Field(
        default_factory=CrossValidationConfig
    )
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)

    @property
    def schema(self) -> SchemaConfig:
        return self.data_schema


class PCAConfig(BaseModel):
    variance_threshold: float = 0.90
    n_components: int | None = None
    batch_size: int = 4096
    random_state: int = 42


class KMeansConfig(BaseModel):
    n_clusters: int = 8
    batch_size: int = 2048
    random_state: int = 42
    compute_distances: bool = True


class XGBoostConfig(BaseModel):
    tree_method: str = "hist"
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float | str | None = "auto"
    early_stopping_rounds: int = 30
    eval_metric: str = "aucpr"
    random_state: int = 42


class RandomForestConfig(BaseModel):
    n_estimators: int = 300
    max_depth: int = 15
    class_weight: str = "balanced_subsample"
    min_samples_split: int = 10
    min_samples_leaf: int = 4
    n_jobs: int = -1
    random_state: int = 42


class LinearSVMConfig(BaseModel):
    loss: str = "hinge"
    penalty: str = "l2"
    alpha: float = 0.0001
    max_iter: int = 2000
    calibration_method: str = "sigmoid"
    random_state: int = 42


class DecompositionConfig(BaseModel):
    pca: PCAConfig = Field(default_factory=PCAConfig)


class ClusteringConfig(BaseModel):
    minibatch_kmeans: KMeansConfig = Field(default_factory=KMeansConfig)


class TreeClassifiersConfig(BaseModel):
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    linear_svm: LinearSVMConfig = Field(default_factory=LinearSVMConfig)


class ClassicalModelsConfig(BaseModel):
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    tree_classifiers: TreeClassifiersConfig = Field(
        default_factory=TreeClassifiersConfig
    )


class LSTMConfig(BaseModel):
    input_dim: int | None = None
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.2
    head_hidden_dim: int = 64


class TransformerConfig(BaseModel):
    input_dim: int | None = None
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"


class LossConfig(BaseModel):
    type: str = "focal"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    pos_weight: float | None = None


class DeepModelsConfig(BaseModel):
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: dict[str, Any] = Field(default_factory=dict)


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_config(file_path: str = "configs/data_config.yaml") -> DataConfig:
    raw = load_yaml(file_path)
    return DataConfig.model_validate(raw)


def load_classical_config(
    file_path: str = "configs/classical_models.yaml",
) -> ClassicalModelsConfig:
    raw = load_yaml(file_path)
    return ClassicalModelsConfig.model_validate(raw)


def load_deep_config(file_path: str = "configs/deep_models.yaml") -> DeepModelsConfig:
    raw = load_yaml(file_path)
    return DeepModelsConfig.model_validate(raw)
