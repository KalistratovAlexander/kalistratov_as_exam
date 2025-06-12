#!/usr/bin/env python3
"""preprocess.py

Clean and preprocess the raw Breast Cancer Diagnostic CSV.

Key fixes (2025-06-12)
---------------------
* **Target mapping** now supports original UCI labels `'M'`/`'B'` as well as
  `'malignant'`/`'benign'` and numeric `1`/`0`.  This prevents *NaN* values that
  previously broke the train/val/test split.

Pipeline steps
--------------
1. Load raw CSV produced by `fetch_data.py`.
2. Drop redundant columns (e.g. `id`).
3. Rename feature columns to *snake_case*.
4. Encode `diagnosis` → 1 (malignant) / 0 (benign).
5. Impute missing values (median).
6. Scale features (`StandardScaler`).
7. Save processed data to Parquet.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
SEED = 42

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_snake_case(name: str) -> str:
    import re

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace(" ", "_").replace("-", "_").lower()


# ---------------------------------------------------------------------------
# Core preprocessing routine
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting preprocessing …")

    # ---- Drop ID column if exists ----
    maybe_id_cols = [c for c in df.columns if c.lower() == "id"]
    if maybe_id_cols:
        logger.debug("Dropping columns: %s", maybe_id_cols)
        df = df.drop(columns=maybe_id_cols)

    # ---- Standardize column names ----
    df = df.rename(columns={c: _to_snake_case(c) for c in df.columns})

    # ---- Encode target ----
    if "diagnosis" not in df.columns:
        raise KeyError("Target column 'diagnosis' not found after renaming.")

    target_map = {
        "malignant": 1,
        "benign": 0,
        "M": 1,
        "B": 0,
        1: 1,
        0: 0,
    }
    df["diagnosis"] = df["diagnosis"].map(target_map)
    if df["diagnosis"].isna().any():
        n_bad = df["diagnosis"].isna().sum()
        logger.warning("%s rows with unknown target labels will be dropped.", n_bad)
        df = df.dropna(subset=["diagnosis"])

    # ---- Split features/target ----
    X = df.drop(columns=["diagnosis"])
    y = df[["diagnosis"]]

    # ---- Impute missing values ----
    imputer = SimpleImputer(strategy="median", random_state=SEED)
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # ---- Scale numeric features ----
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)

    processed = pd.concat([X_scaled, y], axis=1)
    logger.info("Finished preprocessing: %s rows × %s columns", *processed.shape)
    return processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean & normalize raw breast-cancer dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, default=Path(os.getenv("RAW_DATA_PATH", "raw/breast_cancer.csv")), help="Path to raw CSV file")
    parser.add_argument("--output", "-o", type=Path, default=Path(os.getenv("PROCESSED_DATA_PATH", "processed/data.parquet")), help="Destination Parquet path")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity")
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s — %(levelname)s — %(name)s — %(message)s")


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    logger.info("Loading raw data from %s", args.input)
    df_raw = pd.read_csv(args.input)

    df_processed = preprocess(df_raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing processed data to %s", args.output)
    df_processed.to_parquet(args.output, index=False)
    logger.info("Done. File size: %.1f KB", args.output.stat().st_size / 1024)


if __name__ == "__main__":
    main()