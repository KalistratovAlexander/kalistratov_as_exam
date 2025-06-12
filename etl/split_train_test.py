#!/usr/bin/env python3
"""split_train_test.py

Stratified split of the *processed* breast-cancer dataset into **train**,
**validation**, and **test** partitions.

By default:
* **70 %**  → train
* **15 %**  → validation
* **15 %**  → test

Outputs three Parquet files under `processed/` (or custom path via CLI):
* `train.parquet`
* `val.parquet`
* `test.parquet`

Designed to be called directly or via Airflow.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
SEED = 42

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame, train_size: float, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "diagnosis" not in df.columns:
        raise KeyError("Target column 'diagnosis' missing in processed dataset")

    test_size = 1.0 - (train_size + val_size)
    if test_size <= 0:
        raise ValueError("train_size + val_size must be < 1.0")

    # First split train+val vs test
    df_tv, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["diagnosis"],
        random_state=SEED,
    )

    # Split train vs val
    val_fraction_within_tv = val_size / (train_size + val_size)
    df_train, df_val = train_test_split(
        df_tv,
        test_size=val_fraction_within_tv,
        stratify=df_tv["diagnosis"],
        random_state=SEED,
    )

    logger.info(
        "Split sizes — train: %s, val: %s, test: %s",
        len(df_train),
        len(df_val),
        len(df_test),
    )
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits from processed dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(os.getenv("PROCESSED_DATA_PATH", "processed/data.parquet")),
        help="Path to processed Parquet file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(os.getenv("SPLIT_OUTPUT_DIR", "processed")),
        help="Directory to write train/val/test Parquet files",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=float(os.getenv("TRAIN_SIZE", 0.70)),
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=float(os.getenv("VAL_SIZE", 0.15)),
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    )


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    logger.info("Reading processed dataset from %s", args.input)
    df = pd.read_parquet(args.input)

    df_train, df_val, df_test = stratified_split(df, args.train_size, args.val_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(args.output_dir / "train.parquet", index=False)
    df_val.to_parquet(args.output_dir / "val.parquet", index=False)
    df_test.to_parquet(args.output_dir / "test.parquet", index=False)
    logger.info("Parquet files written to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
