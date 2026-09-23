"""
Tests for Data Ingestion, Leakage Prevention, and Temporal Cross-Validation.
"""

import numpy as np
import polars as pl
import pytest

from src.data.ingestion import generate_synthetic_benchmark, synthesize_card_id
from src.data.preprocessor import (
    FraudDataPreprocessor,
    compute_cyclical_features,
    temporal_train_dev_test_split,
)
from src.tuning.validation import PurgedGroupTimeSeriesSplit


@pytest.fixture
def sample_polars_df():
    np.random.seed(42)
    n = 1000
    times = np.sort(np.random.uniform(1000, 100000, size=n))
    return pl.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": times.astype(np.int32),
            "isFraud": (np.random.uniform(0, 1, size=n) < 0.035).astype(np.int32),
            "TransactionAmt": np.random.uniform(10, 500, size=n).astype(np.float32),
            "card1": np.random.randint(100, 200, size=n),
            "card2": np.random.randint(10, 50, size=n).astype(np.float32),
            "card3": 150.0,
            "card4": ["visa"] * n,
            "addr1": 300.0,
            "D1": np.random.randint(0, 100, size=n).astype(np.float32),
            "C1": np.random.poisson(2, size=n).astype(np.float32),
            "V1": np.random.normal(0, 1, size=n).astype(np.float32),
        }
    )


def test_synthesize_card_id(sample_polars_df):
    lf = sample_polars_df.lazy()
    lf_with_card = synthesize_card_id(lf)
    df_out = lf_with_card.collect()
    assert "card_id" in df_out.columns
    assert df_out["card_id"].dtype == pl.String
    # Check that it's composed of separated components
    first_val = df_out["card_id"][0]
    assert "_" in first_val


def test_strict_temporal_split_order(sample_polars_df):
    """
    Guarantees no future transactions leak into earlier splits.
    """
    train_df, dev_df, test_df = temporal_train_dev_test_split(
        sample_polars_df,
        time_col="TransactionDT",
        train_ratio=0.80,
        dev_ratio=0.10,
        test_ratio=0.10,
    )

    max_train_t = train_df["TransactionDT"].max()
    min_dev_t = dev_df["TransactionDT"].min()
    max_dev_t = dev_df["TransactionDT"].max()
    min_test_t = test_df["TransactionDT"].min()

    assert (
        max_train_t <= min_dev_t
    ), f"Train max ({max_train_t}) > Dev min ({min_dev_t})"
    assert max_dev_t <= min_test_t, f"Dev max ({max_dev_t}) > Test min ({min_test_t})"


def test_cyclical_encoding(sample_polars_df):
    df_cyc = compute_cyclical_features(sample_polars_df, time_col="TransactionDT")
    for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
        assert col in df_cyc.columns
        vals = df_cyc[col].to_numpy()
        assert np.all(vals >= -1.0001) and np.all(vals <= 1.0001)


def test_preprocessor_no_leakage(sample_polars_df):
    """
    Guarantees preprocessor is fitted strictly on train split and transforms dev without error.
    """
    train_df, dev_df, test_df = temporal_train_dev_test_split(
        sample_polars_df,
        time_col="TransactionDT",
        train_ratio=0.80,
        dev_ratio=0.10,
        test_ratio=0.10,
    )

    preprocessor = FraudDataPreprocessor(
        target_col="isFraud",
        time_col="TransactionDT",
        entity_col="card1",
        correlation_threshold=0.90,
    )

    preprocessor.fit(train_df)
    assert preprocessor.is_fitted

    X_train, _y_train, _meta_train = preprocessor.transform(train_df)
    X_dev, _y_dev, _meta_dev = preprocessor.transform(dev_df)
    X_test, _y_test, _meta_test = preprocessor.transform(test_df)

    assert X_train.shape[1] == X_dev.shape[1] == X_test.shape[1]
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_dev).any()
    assert not np.isnan(X_test).any()
    # Check that train scaler mean was applied to transform dev
    assert np.allclose(np.mean(X_train[:, :2], axis=0), 0.0, atol=1e-1)


def test_purged_group_time_series_split():
    """
    Verifies that overlapping boundary entities in the purge buffer are excluded from train folds.
    """
    n = 200
    times = np.arange(n, dtype=float) * 3600.0  # hourly steps
    # card_id 'card_A' appears around the fold boundary
    groups = np.array(["normal"] * n)
    groups[80:110] = "card_A"

    cv = PurgedGroupTimeSeriesSplit(
        n_splits=3, purge_window_seconds=86400.0
    )  # 24h buffer
    X = np.zeros((n, 2))

    for train_idx, val_idx in cv.split(X, groups=groups, timestamps=times):
        # Val start time
        val_start_time = times[val_idx[0]]
        # Buffer window start
        buf_start_time = val_start_time - 86400.0

        val_groups = set(groups[val_idx])

        # 1. Complete group isolation: No entity in val should appear anywhere in candidate train
        full_entity_overlap = val_groups.intersection(set(groups[train_idx]))
        assert (
            len(full_entity_overlap) == 0
        ), f"Group leakage: entities in val also found in train fold: {full_entity_overlap}"

        # 2. Complete boundary buffer purge: No transaction in buffer window [val_start - buffer, val_start)
        buffer_violations = np.sum(
            (times[train_idx] >= buf_start_time) & (times[train_idx] < val_start_time)
        )
        assert (
            buffer_violations == 0
        ), f"Boundary leakage: {buffer_violations} transactions found in temporal buffer"


def test_transaction_sequence_dataset_and_loader(sample_polars_df):
    from src.data.dataset_builder import (
        TransactionSequenceDataset,
        build_dataloaders,
    )

    preprocessor = FraudDataPreprocessor(
        target_col="isFraud",
        time_col="TransactionDT",
        entity_col="card1",
    )
    preprocessor.fit(sample_polars_df)
    X, y, meta = preprocessor.transform(sample_polars_df)

    dataset = TransactionSequenceDataset(
        features=X,
        targets=y,
        meta_df=meta,
        window_length=10,
        entity_col="card1",
        time_col="TransactionDT",
    )
    assert len(dataset) == len(X)
    x_seq, y_target, mask = dataset[0]
    assert x_seq.shape == (10, X.shape[1])
    assert y_target.shape == (1,)
    assert mask.shape == (10,)

    # Test dataloader builder
    train_loader, _dev_loader, _test_loader = build_dataloaders(
        train_data=(X, y, meta),
        dev_data=(X[:50], y[:50], meta.iloc[:50]),
        test_data=(X[:50], y[:50], meta.iloc[:50]),
        window_length=10,
        batch_size=16,
    )
    for batch_x, batch_y, batch_mask in train_loader:
        assert batch_x.shape[1:] == (10, X.shape[1])
        assert batch_y.shape[1:] == (1,)
        assert batch_mask.shape[1:] == (10,)
        break


def test_synthetic_benchmark_generation(tmp_path):
    trans_path, id_path = generate_synthetic_benchmark(
        num_records=200,
        num_entities=20,
        fraud_rate=0.05,
        output_dir=str(tmp_path),
    )
    df_trans = pl.read_csv(trans_path)
    df_id = pl.read_csv(id_path)

    assert df_trans.shape[0] == 200
    assert "isFraud" in df_trans.columns
    assert "TransactionDT" in df_trans.columns
    assert "card1" in df_trans.columns
    assert df_id.shape[0] > 0
    assert "TransactionID" in df_id.columns


def test_single_transaction_entity_padding():
    """
    Guarantees that an entity with only 1 transaction still gets shaped into (20, D)
    via left zero-padding, with the transaction features placed at the final timestep.
    """
    import pandas as pd

    from src.data.dataset_builder import TransactionSequenceDataset

    D = 8
    # 1 transaction for card_lonely, 3 transactions for card_regular
    features = np.ones((4, D), dtype=np.float32)
    features[0, :] = 42.0  # card_lonely's single transaction
    targets = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    meta_df = pd.DataFrame(
        {
            "card_id": ["card_lonely", "card_regular", "card_regular", "card_regular"],
            "TransactionDT": [100.0, 10.0, 20.0, 30.0],
        }
    )

    dataset = TransactionSequenceDataset(
        features=features,
        targets=targets,
        meta_df=meta_df,
        window_length=20,
        min_history=1,
        entity_col="card_id",
        time_col="TransactionDT",
    )

    assert len(dataset) == 4

    # Find the window corresponding to card_lonely (target_idx 0)
    lonely_item = None
    for i in range(len(dataset)):
        x_seq, y_val, pad_mask = dataset[i]
        if y_val.item() == 1.0:
            lonely_item = (x_seq, y_val, pad_mask)
            break

    assert lonely_item is not None
    x_seq, y_val, pad_mask = lonely_item

    # Verify shape is strictly (20, D)
    assert x_seq.shape == (20, D)
    assert pad_mask.shape == (20,)

    # First 19 timesteps must be zero-padded and masked as True
    assert (pad_mask[:19] == True).all()
    assert (x_seq[:19] == 0.0).all()

    # The 20th timestep (index 19) must be the valid transaction (False mask)
    assert pad_mask[19].item() is False
    assert (x_seq[19].numpy() == 42.0).all()


def test_cross_validation_config_units():
    """
    Verifies that CrossValidationConfig correctly synchronizes purge_buffer_seconds
    and purge_buffer_hours, and PurgedGroupTimeSeriesSplit accepts purge_buffer_seconds.
    """
    from src.utils.config_parser import CrossValidationConfig

    # 1. Config with seconds converts to hours
    cfg1 = CrossValidationConfig(purge_buffer_seconds=86400.0)
    assert cfg1.purge_buffer_hours == 24.0
    assert cfg1.purge_buffer_seconds == 86400.0

    # 2. Config with hours converts to seconds
    cfg2 = CrossValidationConfig(purge_buffer_hours=12.0)
    assert cfg2.purge_buffer_seconds == 43200.0

    # 3. PurgedGroupTimeSeriesSplit purge_buffer_seconds argument
    cv = PurgedGroupTimeSeriesSplit(n_splits=3, purge_buffer_seconds=3600.0)
    assert cv.purge_window_seconds == 3600.0


def test_synthesize_card_id_null_safety():
    """
    Verifies pl.concat_str handles null and missing values cleanly without producing nulls.
    """
    lf = pl.LazyFrame(
        {
            "card1": [1000, None, 2000],
            "card2": [None, 200.0, None],
            "card3": [150.0, None, 150.0],
            "card4": [None, "mastercard", None],
            "addr1": [None, 300.0, None],
            "D1": [14.0, None, None],
        }
    )
    res = synthesize_card_id(lf).collect()
    assert "card_id" in res.columns
    assert res["card_id"].null_count() == 0
    # First row check
    assert res["card_id"][0] == "1000_0.0_150.0_UNK_0.0_14.0"
    # Second row check
    assert res["card_id"][1] == "0_200.0_0.0_mastercard_300.0_0.0"


def test_preprocessor_cyclical_features_retained_and_no_category_collision():
    """
    Verifies that cyclical features are never dropped by the correlation filter,
    and unknown categories do not collide with known categories.
    """
    # Create synthetic dataframe where hour_sin and a dummy col are perfectly collinear
    n = 200
    times = np.arange(n, dtype=float) * 3600.0
    hour_sin_val = np.sin(2.0 * np.pi * ((times % 86400.0) / 3600.0) / 24.0)

    train_pl = pl.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": times,
            "card_id": ["C1"] * n,
            "isFraud": [0] * n,
            "cat_col": ["alpha"] * 100 + ["beta"] * 100,
            "collinear_num": hour_sin_val,  # perfectly correlated with hour_sin
            "other_num": np.random.randn(n),
        }
    )

    test_pl = pl.DataFrame(
        {
            "TransactionID": np.arange(n, n + 10),
            "TransactionDT": times[:10] + 100000,
            "card_id": ["C1"] * 10,
            "isFraud": [0] * 10,
            "cat_col": ["unknown_gamma"] * 10,  # brand new category
            "collinear_num": hour_sin_val[:10],
            "other_num": np.random.randn(10),
        }
    )

    preprocessor = FraudDataPreprocessor(
        target_col="isFraud",
        time_col="TransactionDT",
        entity_col="card_id",
        correlation_threshold=0.85,
    )
    preprocessor.fit(train_pl)

    # Cyclical features must be present
    for cyc in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
        assert cyc in preprocessor.num_cols

    X_train, _, _ = preprocessor.transform(train_pl)
    X_test, _, _ = preprocessor.transform(test_pl)

    # Check that unknown_gamma mapped to 0.0, while known alpha/beta mapped to > 0.0
    cat_idx = preprocessor.feature_names_out_.index("cat_cat_col")
    test_cat_vals = X_test[:, cat_idx]
    train_cat_vals = X_train[:, cat_idx]

    # Unknown category maps to 0.0
    assert np.allclose(test_cat_vals, 0.0)
    # Known categories map to non-zero values
    assert np.all(train_cat_vals > 0.0)


def test_dataset_builder_cross_split_context_lookback():
    """
    Verifies that Dev and Test sequences look back into Train/Dev history
    so a cardholder active across splits is not artificially zero-padded.
    """
    import pandas as pd

    from src.data.dataset_builder import build_dataloaders

    D = 4
    # Entity 'card_A' has 10 transactions in train, and 1 transaction in dev
    X_train = np.ones((10, D), dtype=np.float32)
    y_train = np.zeros(10, dtype=np.float32)
    meta_train = pd.DataFrame(
        {"card_id": ["card_A"] * 10, "TransactionDT": np.arange(10, dtype=float)}
    )

    X_dev = np.full((1, D), 99.0, dtype=np.float32)
    y_dev = np.ones(1, dtype=np.float32)
    meta_dev = pd.DataFrame({"card_id": ["card_A"], "TransactionDT": [11.0]})

    _train_loader, dev_loader, _ = build_dataloaders(
        train_data=(X_train, y_train, meta_train),
        dev_data=(X_dev, y_dev, meta_dev),
        window_length=5,
        min_history=1,
        batch_size=1,
        use_context_history=True,
    )

    # Dev loader should have exactly 1 item
    assert len(dev_loader.dataset) == 1

    for x_seq, y_val, pad_mask in dev_loader:
        assert x_seq.shape == (1, 5, D)
        assert y_val.item() == 1.0
        # Because card_A had 10 transactions in train + 1 in dev, the window of length 5
        # is fully populated from history: 0 padded steps!
        assert pad_mask.sum().item() == 0
        # Final step must be the dev transaction
        assert (x_seq[0, -1].numpy() == 99.0).all()
        # Earlier steps must be the train transactions (ones)
        assert (x_seq[0, :-1].numpy() == 1.0).all()
