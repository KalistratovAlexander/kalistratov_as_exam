#!/usr/bin/env python3
"""evaluate_model.py

Compute final test-set metrics for the fitted Logistic Regression model.

Inputs
------
* `model.pkl`      – pickled model from *train_model.py*
* `test.parquet`   – hold-out split from *split_train_test.py*

Outputs
-------
* `test_metrics.json` – Accuracy, Precision, Recall, F1, ROC-AUC (optional)
* (optional) `classification_report.csv`

The script exits with a **non-zero** status code if any required input is
missing, to allow Airflow to mark the task as *failed*.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_xy(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    if "diagnosis" not in df.columns:
        raise KeyError("Target column 'diagnosis' missing in %s" % parquet_path)
    X = df.drop(columns=["diagnosis"]).values
    y = df["diagnosis"].values
    return X, y


def compute_metrics(y_true, y_pred, y_prob=None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Only one class present in y_true
            metrics["roc_auc"] = None
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate logistic regression on test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", "artifacts/model.pkl")),
        help="Path to pickled model",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path(os.getenv("TEST_PATH", "processed/test.parquet")),
        help="Path to test parquet",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path(os.getenv("METRICS_OUTPUT", "artifacts/test_metrics.json")),
        help="Where to write JSON metrics",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path(os.getenv("REPORT_OUTPUT", "artifacts/classification_report.csv")),
        help="Optional CSV classification report",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if not args.model.exists():
        logger.error("Model file not found: %s", args.model)
        raise SystemExit(1)
    if not args.test.exists():
        logger.error("Test parquet not found: %s", args.test)
        raise SystemExit(1)

    logger.info("Loading model from %s", args.model)
    model = joblib.load(args.model)

    logger.info("Loading test set from %s", args.test)
    X_test, y_test = load_xy(args.test)

    logger.info("Predicting …")
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC if model supports `predict_proba`
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    logger.info(
        "Test metrics — Acc: {accuracy:.3f}  Prec: {precision:.3f}  Rec: {recall:.3f}  F1: {f1:.3f}".format(**metrics)
    )

    # Save JSON
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics JSON written to %s", args.metrics_output)

    # Optional classification report CSV
    if args.report_output:
        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(args.report_output, index=True)
        logger.info("Classification report CSV saved to %s", args.report_output)

    # Confusion matrix logged for debugging
    cm = confusion_matrix(y_test, y_pred)
    logger.debug("Confusion matrix:\n%s", cm)


if __name__ == "__main__":
    main()
