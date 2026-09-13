"""
Data Ingestion Pipeline for IEEE-CIS and PaySim Fraud Benchmark.
Uses Polars with lazy evaluation (scan_csv) for memory-efficient joins and downcasting.
"""

from pathlib import Path

import numpy as np
import polars as pl

from src.utils.logger import get_logger, timer

logger = get_logger("data.ingestion")


def downcast_polars_schema(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Downcast Float64 to Float32 and Int64 to Int32/UInt32 to optimize memory footprint.
    """
    schema = lf.collect_schema()
    downcast_exprs = []

    for col, dtype in schema.items():
        if dtype == pl.Float64:
            downcast_exprs.append(pl.col(col).cast(pl.Float32))
        elif dtype == pl.Int64:
            # Check if column is non-negative like IDs or counts
            if any(
                term in col.lower() for term in ["id", "dt", "count", "card", "addr"]
            ):
                downcast_exprs.append(pl.col(col).cast(pl.UInt32))
            else:
                downcast_exprs.append(pl.col(col).cast(pl.Int32))
        elif dtype == pl.String:
            # If low cardinality string column, keep or cast to Categorical when collecting
            pass

    if downcast_exprs:
        lf = lf.with_columns(downcast_exprs)
    return lf


def synthesize_card_id(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Synthesize composite card_id:
    card1 + '_' + card2 + '_' + card3 + '_' + card4 + '_' + addr1 + '_' + D1
    Uses pl.concat_str for native null safety and high throughput.
    """
    expr = pl.concat_str(
        [
            pl.col("card1").fill_null(0).cast(pl.String),
            pl.col("card2").fill_null(0).cast(pl.String),
            pl.col("card3").fill_null(0).cast(pl.String),
            pl.col("card4").fill_null("UNK").cast(pl.String),
            pl.col("addr1").fill_null(0).cast(pl.String),
            pl.col("D1").fill_null(0).cast(pl.String),
        ],
        separator="_",
    ).alias("card_id")

    return lf.with_columns(expr)


@timer
def ingest_ieee_cis(
    trans_path: str,
    id_path: str | None,
    output_path: str,
) -> pl.DataFrame:
    """
    Lazy ingestion and joining of IEEE-CIS transaction and identity tables via Polars.
    """
    logger.info(f"Scanning transactions CSV: {trans_path}")
    trans_lf = pl.scan_csv(trans_path)
    trans_lf = downcast_polars_schema(trans_lf)

    if id_path and Path(id_path).exists():
        logger.info(f"Scanning identity CSV: {id_path}")
        id_lf = pl.scan_csv(id_path)
        id_lf = downcast_polars_schema(id_lf)
        merged_lf = trans_lf.join(id_lf, on="TransactionID", how="left")
    else:
        logger.info(
            "Identity table not provided or not found; proceeding with transactions table only."
        )
        merged_lf = trans_lf

    # Synthesize composite entity card_id if necessary columns exist
    schema = merged_lf.collect_schema()
    if "card1" in schema and "D1" in schema:
        logger.info("Synthesizing composite card_id...")
        merged_lf = synthesize_card_id(merged_lf)
    elif "card1" in schema:
        merged_lf = merged_lf.with_columns(
            pl.col("card1").cast(pl.String).alias("card_id")
        )
    else:
        # Fallback entity
        merged_lf = merged_lf.with_columns(pl.lit("entity_0").alias("card_id"))

    # Strictly sort chronologically by TransactionDT
    if "TransactionDT" in schema:
        merged_lf = merged_lf.sort("TransactionDT")

    logger.info("Executing lazy query plan and collecting materialized DataFrame...")
    df = merged_lf.collect()

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    logger.info(f"Persisted {df.shape[0]} rows x {df.shape[1]} cols to {output_path}")

    return df


@timer
def ingest_paysim(
    paysim_path: str,
    output_path: str,
) -> pl.DataFrame:
    """
    Ingest PaySim CSV as a drop-in substitute using step as time, nameOrig as entity.
    """
    logger.info(f"Scanning PaySim CSV: {paysim_path}")
    lf = pl.scan_csv(paysim_path)
    lf = downcast_polars_schema(lf)

    lf = lf.with_columns(
        [
            pl.col("step").cast(pl.UInt32).alias("TransactionDT"),
            pl.col("nameOrig").cast(pl.String).alias("card_id"),
            pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("TransactionID"),
        ]
    )

    if "isFraud" not in lf.collect_schema():
        raise ValueError("PaySim CSV missing 'isFraud' target column.")

    lf = lf.sort("TransactionDT")
    df = lf.collect()

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    logger.info(
        f"Persisted PaySim {df.shape[0]} rows x {df.shape[1]} cols to {output_path}"
    )
    return df


def generate_synthetic_benchmark(
    num_records: int = 50000,
    num_entities: int = 2500,
    fraud_rate: float = 0.035,
    output_dir: str = "data/raw",
    seed: int = 42,
) -> tuple[str, str]:
    """
    High-fidelity synthetic IEEE-CIS generator preserving:
    - ~3.5% positive fraud imbalance
    - Temporal sorting across TransactionDT
    - Multi-transaction card_id entity histories
    - Numerical V-features, C-features, D-features, and categorical ProductCD/card4
    """
    logger.info(
        f"Generating synthetic benchmark: {num_records} rows across {num_entities} entities..."
    )
    rng = np.random.default_rng(seed)

    raw_path = Path(output_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    trans_file = raw_path / "train_transaction.csv"
    id_file = raw_path / "train_identity.csv"

    # Entities
    card1_pool = rng.integers(1000, 19999, size=num_entities)
    card2_pool = rng.integers(100, 600, size=num_entities)
    card3_pool = rng.choice([150, 185, 144, 106], size=num_entities)
    card4_pool = rng.choice(
        ["visa", "mastercard", "discover", "american express"], size=num_entities
    )
    addr1_pool = rng.integers(100, 500, size=num_entities)
    d1_pool = rng.integers(0, 600, size=num_entities)

    entity_idx = rng.choice(num_entities, size=num_records)
    # Temporal timestamps: strictly increasing sequence with jitter
    base_times = np.sort(rng.uniform(86400, 86400 * 180, size=num_records))
    transaction_ids = np.arange(3000000, 3000000 + num_records, dtype=np.int32)

    # Base fraud status per entity with sporadic bursts
    entity_fraud_propensity = rng.beta(0.3, 8.0, size=num_entities)
    tx_fraud_prob = entity_fraud_propensity[entity_idx] * 0.4
    is_fraud = (
        rng.uniform(0, 1, size=num_records) < (tx_fraud_prob + (fraud_rate * 0.3))
    ).astype(np.int8)

    # Amounts: log-normal distribution, higher for fraud
    amt = np.exp(rng.normal(3.8, 1.2, size=num_records))
    amt[is_fraud == 1] *= rng.uniform(1.5, 4.0, size=int(np.sum(is_fraud == 1)))
    amt = np.round(amt, 2)

    product_cd = rng.choice(
        ["W", "C", "R", "H", "S"], size=num_records, p=[0.74, 0.12, 0.07, 0.05, 0.02]
    )

    data_trans = {
        "TransactionID": transaction_ids,
        "isFraud": is_fraud,
        "TransactionDT": base_times.astype(np.int32),
        "TransactionAmt": amt.astype(np.float32),
        "ProductCD": product_cd,
        "card1": card1_pool[entity_idx].astype(np.int32),
        "card2": card2_pool[entity_idx].astype(np.float32),
        "card3": card3_pool[entity_idx].astype(np.float32),
        "card4": card4_pool[entity_idx],
        "card6": rng.choice(["debit", "credit"], size=num_records, p=[0.75, 0.25]),
        "addr1": addr1_pool[entity_idx].astype(np.float32),
        "addr2": rng.choice([87, 60, 96], size=num_records).astype(np.float32),
        "dist1": rng.exponential(15.0, size=num_records).astype(np.float32),
        "D1": d1_pool[entity_idx].astype(np.float32),
        "D2": rng.integers(0, 500, size=num_records).astype(np.float32),
        "D15": rng.integers(0, 600, size=num_records).astype(np.float32),
    }

    # Add core C-features (counts) and V-features (engineered transaction features)
    for c in range(1, 15):
        data_trans[f"C{c}"] = rng.poisson(lam=1.5, size=num_records).astype(np.float32)

    for v in range(1, 41):
        data_trans[f"V{v}"] = rng.standard_normal(size=num_records).astype(np.float32)

    df_trans = pl.DataFrame(data_trans)
    df_trans.write_csv(trans_file)
    logger.info(
        f"Wrote synthetic transaction table to {trans_file} ({df_trans.shape[0]} rows)"
    )

    # Identity table (~25% of transactions have identity metadata)
    id_mask = rng.uniform(0, 1, size=num_records) < 0.25
    id_tx_ids = transaction_ids[id_mask]
    num_ids = len(id_tx_ids)

    data_id = {
        "TransactionID": id_tx_ids,
        "id_01": rng.uniform(-100, 0, size=num_ids).astype(np.float32),
        "id_02": rng.integers(1000, 900000, size=num_ids).astype(np.float32),
        "DeviceType": rng.choice(["desktop", "mobile"], size=num_ids, p=[0.55, 0.45]),
        "DeviceInfo": rng.choice(
            ["iOS Device", "Windows", "MacOS", "SM-G950F", "Trident/7.0"], size=num_ids
        ),
    }
    for i in range(3, 12):
        data_id[f"id_{i:02d}"] = rng.standard_normal(size=num_ids).astype(np.float32)

    df_id = pl.DataFrame(data_id)
    df_id.write_csv(id_file)
    logger.info(f"Wrote synthetic identity table to {id_file} ({df_id.shape[0]} rows)")

    return str(trans_file), str(id_file)


def run_ingestion(
    config_path: str = "configs/data_config.yaml",
    quick_sample_records: int | None = None,
) -> pl.DataFrame:
    """
    Main entrypoint for dataset ingestion. Checks for raw IEEE-CIS or PaySim files;
    generates synthetic benchmark data if absent.
    """
    from src.utils.config_parser import load_data_config

    cfg = load_data_config(config_path)

    trans_csv = Path(cfg.raw_paths.transaction_csv)
    id_csv = Path(cfg.raw_paths.identity_csv)
    paysim_csv = Path(cfg.raw_paths.paysim_csv)
    out_parquet = cfg.processed_paths.merged_parquet

    if trans_csv.exists():
        logger.info(f"Found existing IEEE-CIS raw data at {trans_csv}")
        id_path = str(id_csv) if id_csv.exists() else None
        return ingest_ieee_cis(str(trans_csv), id_path, out_parquet)

    elif paysim_csv.exists():
        logger.info(f"Found existing PaySim raw data at {paysim_csv}")
        return ingest_paysim(str(paysim_csv), out_parquet)

    else:
        logger.warning(
            f"Raw data not detected at {trans_csv}. Generating synthetic benchmark dataset..."
        )
        n_records = quick_sample_records or 60000
        t_path, i_path = generate_synthetic_benchmark(
            num_records=n_records,
            num_entities=max(500, n_records // 20),
            fraud_rate=0.035,
            output_dir=str(trans_csv.parent),
        )
        return ingest_ieee_cis(t_path, i_path, out_parquet)
