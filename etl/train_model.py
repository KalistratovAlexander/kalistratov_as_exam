#!/usr/bin/env python3
"""train_model.py

Train a **Logistic Regression** classifier on the processed Breast-Cancer data.

Assumes that *split_train_test.py* (or an equivalent step) has already saved
stratified `train.parquet` and `val.parquet` files with **scaled numeric
features** and a binary `diagnosis` target (0 = benign, 1 = malignant).

Outputs
-------
* `model.pkl` – the fitted `LogisticRegression` (pickled via joblib)
* (Optional) `val_metrics.json` – accuracy, precision, recall, F1 on the
  validation set (useful for monitoring)  
  *(calculated here for convenience; final test metrics are produced in
  `evaluate_model.py`).*

CLI usage
~~~~~~~~~
::

   python etl/train_model.py \
       --train processed/train.parquet \
       --val processed/val.parquet \
       --model-output artifacts/model.pkl \
       --metrics-output artifacts/val_metrics.json

Environment variable fallbacks are provided for Airflow.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

logger = logging.getLogger(__name__)
SEED = 42

# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def load_xy(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    if "diagnosis" not in df.columns:
        raise KeyError("Target column 'diagnosis' missing in dataset: %s" % parquet_path)
    X = df.drop(columns=["diagnosis"]).values
    y = df["diagnosis"].values
    return X, y


def train_logreg(X_train, y_train) -> LogisticRegression:
    """Fit Logistic Regression with cross-validated C hyper-parameter."""
    logger.info("Beginning grid-search CV …")
    base_est = LogisticRegression(
        penalty="l2",
        solver="saga",
        class_weight="balanced",
        random_state=SEED,
        max_iter=1000,
    )
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    grid = GridSearchCV(
        estimator=base_est,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    logger.info("Best C: %s (mean CV F1 = %.4f)", grid.best_params_["C"], grid.best_score_)
    return grid.best_estimator_


def evaluate(model, X_val, y_val) -> dict[str, float]:
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
    }
    logger.info(
        "Validation metrics — Acc: {accuracy:.3f}  Prec: {precision:.3f}  Rec: {recall:.3f}  F1: {f1:.3f}".format(**metrics)
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression on breast-cancer data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path(os.getenv("TRAIN_PATH", "processed/train.parquet")),
        help="Path to training parquet file",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path(os.getenv("VAL_PATH", "processed/val.parquet")),
        help="Path to validation parquet file",
    )
    parser.add_argument(
        "--model-output",
        "-m",
        type=Path,
        default=Path(os.getenv("MODEL_OUTPUT", "artifacts/model.pkl")),
        help="Destination for the pickled model",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path(os.getenv("VAL_METRICS_OUTPUT", "artifacts/val_metrics.json")),
        help="Optional path to save validation metrics as JSON",
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

    logger.info("Loading train & validation splits …")
    X_train, y_train = load_xy(args.train)
    X_val, y_val = load_xy(args.val)

    model = train_logreg(X_train, y_train)
    val_metrics = evaluate(model, X_val, y_val)

    # Ensure output dir exists
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    logger.info("Model saved to %s (%.1f KB)", args.model_output, args.model_output.stat().st_size / 1024)

    # Save validation metrics JSON (optional)
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)
        logger.info("Validation metrics written to %s", args.metrics_output)


if __name__ == "__main__":
    main()
