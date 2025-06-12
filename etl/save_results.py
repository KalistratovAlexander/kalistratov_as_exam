#!/usr/bin/env python3
"""save_results.py

Collect trained-model artefacts (model pickle + metrics files) and move/copy
them into a *results/* directory or any destination path specified via CLI.

This script is intentionally simple: it does **not** upload to cloud (handled
later in Stage 4).  It ensures that downstream consumers (dashboards,
researchers) find a single, clean folder with the latest run outputs.

Behaviour
---------
* Copies (or moves) artefacts, preserving timestamps and file permissions.
* Optionally timestamps the filenames (e.g. `20250612T1430_model.pkl`).
* Generates a small `manifest.json` summarising what was saved.

Usage example
~~~~~~~~~~~~~
::

   python etl/save_results.py \
       --model artifacts/model.pkl \
       --metrics artifacts/test_metrics.json \
       --report artifacts/classification_report.csv \
       --dest results/latest --timestamp
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def copy_with_optional_ts(src: Path, dest_dir: Path, timestamp: bool) -> Path:
    """Copy *src* into *dest_dir*; return destination path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if timestamp:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        dest_name = f"{ts}_{src.name}"
    else:
        dest_name = src.name
    dest_path = dest_dir / dest_name
    shutil.copy2(src, dest_path)
    logger.info("Copied %s → %s", src, dest_path)
    return dest_path


def save_results(model: Path, metrics: Path, report: Path | None, dest_dir: Path, timestamp: bool) -> None:
    saved_files: dict[str, str] = {}

    for tag, path in [("model", model), ("metrics", metrics), ("report", report)]:
        if path is None:
            continue
        if not path.exists():
            logger.warning("%s file not found, skipping: %s", tag.capitalize(), path)
            continue
        saved = copy_with_optional_ts(path, dest_dir, timestamp)
        saved_files[tag] = str(saved.relative_to(dest_dir))

    # Write manifest
    manifest_path = dest_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(saved_files, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy artefacts (model + metrics) to results directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=Path(os.getenv("MODEL_PATH", "artifacts/model.pkl")), help="Path to model pickle")
    parser.add_argument("--metrics", type=Path, default=Path(os.getenv("METRICS_OUTPUT", "artifacts/test_metrics.json")), help="Path to metrics JSON")
    parser.add_argument("--report", type=Path, default=Path(os.getenv("REPORT_OUTPUT", "artifacts/classification_report.csv")), help="Optional classification report CSV")
    parser.add_argument("--dest", "-d", type=Path, default=Path(os.getenv("RESULTS_DIR", "results")), help="Destination directory")
    parser.add_argument("--timestamp", action="store_true", help="Prefix files with a datetime stamp (YYYYMMDDThhmmss)")
    parser.add_argument("--move", action="store_true", help="Move instead of copy (removes source files)")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity")
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s — %(levelname)s — %(name)s — %(message)s")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if args.move:
        global shutil
        original_copy2 = shutil.copy2

        def move_and_log(src, dst, *, follow_symlinks=True):
            shutil.move(src, dst)
            return dst

        shutil.copy2 = move_and_log  # type: ignore

    save_results(args.model, args.metrics, args.report, args.dest, args.timestamp)


if __name__ == "__main__":
    main()
