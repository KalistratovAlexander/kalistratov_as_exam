#!/usr/bin/env python3
"""fetch_data.py

Download the **Breast Cancer Wisconsin (Diagnostic)** dataset directly from the
UCI Machine Learning Repository using the `ucimlrepo` package and perform a
concise initial analysis.

This module is callable both as a **stand-alone script** and from an Airflow
`PythonOperator`.  It writes the raw CSV to a configurable location, prints a
brief summary, and logs all events in a structured format.

Usage (CLI)
-----------
::

   pip install ucimlrepo pandas
   python etl/fetch_data.py --output raw/breast_cancer.csv --log-level INFO

Environment overrides (for Airflow):
* ``RAW_DATA_PATH`` – default output path
* ``LOG_LEVEL``     – default log level
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
from ucimlrepo import fetch_ucirepo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def fetch_dataset() -> pd.DataFrame:
    """Fetch dataset via *ucimlrepo* (ID = 17) and return a combined DataFrame.

    Returns
    -------
    pandas.DataFrame
        569 rows × 32 columns (30 features + ID column if present + target).
    """
    logger.info("Fetching dataset via ucimlrepo (id=17)…")
    dataset = fetch_ucirepo(id=17)

    X: pd.DataFrame = dataset.data.features
    y: pd.DataFrame = dataset.data.targets

    # Ensure the target column has the canonical name
    if y.shape[1] != 1:
        raise ValueError(
            f"Expected exactly one target column, got {y.shape[1]}: {list(y.columns)}"
        )
    y = y.rename(columns={y.columns[0]: "diagnosis"})

    df = pd.concat([X, y], axis=1)
    logger.debug("Combined DataFrame columns: %s", list(df.columns))
    return df


def quick_analysis(df: pd.DataFrame) -> None:
    """Log a concise, human-readable summary of *df* for sanity checking."""
    logger.info("Dataset shape: %s rows × %s columns", *df.shape)

    with pd.option_context("display.max_columns", 5, "display.width", 100):
        logger.info("First rows:\n%s", df.head(3).to_string(index=False))

    class_counts = df["diagnosis"].value_counts().to_dict()
    logger.info("Class distribution: %s", class_counts)

    missing = df.isna().sum()
    if missing.any():
        logger.warning("Missing values detected:\n%s", missing[missing > 0])
    else:
        logger.info("No missing values detected.")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download UCI Breast Cancer Diagnostic dataset and save as CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(os.getenv("RAW_DATA_PATH", "raw/breast_cancer.csv")),
        help="Destination CSV path (created if not existing)",
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

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_dataset()
    quick_analysis(df)

    logger.info("Saving CSV to %s", out_path)
    df.to_csv(out_path, index=False)
    size_kb = out_path.stat().st_size / 1024
    logger.info("Done — written %.1f KB", size_kb)


if __name__ == "__main__":
    main()