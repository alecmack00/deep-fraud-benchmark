"""
Data Acquisition Utility for IEEE-CIS Fraud Detection Benchmark.
Downloads the official Kaggle dataset if kaggle credentials are configured,
or generates a high-fidelity synthetic benchmark dataset (>= 200,000 transactions).
"""

import argparse
import subprocess
from pathlib import Path

from src.data.ingestion import generate_synthetic_benchmark
from src.utils.logger import get_logger

logger = get_logger("data.download")


def download_or_generate_dataset(
    output_dir: str = "data/raw",
    target_records: int = 200000,
    force_synthetic: bool = False,
) -> tuple[str, str]:
    """
    Ensures raw transaction and identity CSVs are present.
    Attempts Kaggle CLI download first; falls back to synthetic generation if unavailable.
    """
    raw_path = Path(output_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    trans_csv = raw_path / "train_transaction.csv"
    id_csv = raw_path / "train_identity.csv"

    if not force_synthetic:
        # Check if Kaggle CLI is installed and configured
        try:
            logger.info("Checking Kaggle CLI availability...")
            result = subprocess.run(
                [
                    "kaggle",
                    "competitions",
                    "download",
                    "-c",
                    "ieee-fraud-detection",
                    "-p",
                    str(raw_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info(
                    "Successfully downloaded IEEE-CIS competition archive via Kaggle API."
                )
                zip_file = raw_path / "ieee-fraud-detection.zip"
                if zip_file.exists():
                    subprocess.run(
                        ["unzip", "-o", str(zip_file), "-d", str(raw_path)], check=True
                    )
                    logger.info("Unpacked official IEEE-CIS competition files.")
                    if trans_csv.exists() and id_csv.exists():
                        return str(trans_csv), str(id_csv)
            else:
                logger.warning(
                    f"Kaggle CLI download returned code {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
                )
        except FileNotFoundError:
            logger.info("Kaggle CLI not installed or credentials not configured.")

    # Fallback / Benchmark Mode: High-fidelity generation of >= 200,000 transactions
    logger.info(
        f"Proceeding in Benchmark Generation Mode: synthesizing {target_records:,} transactions across {target_records // 20:,} entities..."
    )
    return generate_synthetic_benchmark(
        num_records=target_records,
        num_entities=max(5000, target_records // 20),
        fraud_rate=0.035,
        output_dir=str(raw_path),
        seed=42,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download or generate IEEE-CIS benchmark dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for raw CSVs",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=200000,
        help="Number of records to generate if Kaggle is unavailable (default: 200000)",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Skip Kaggle CLI attempt and generate synthetic benchmark directly",
    )
    args = parser.parse_args()

    t_path, i_path = download_or_generate_dataset(
        output_dir=args.output_dir,
        target_records=args.records,
        force_synthetic=args.force_synthetic,
    )
    logger.info(f"Raw data ready: {t_path}, {i_path}")


if __name__ == "__main__":
    main()
