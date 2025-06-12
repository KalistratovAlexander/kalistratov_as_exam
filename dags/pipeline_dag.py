"""
pipeline_dag.py

Airflow DAG: автоматизированный ETL-пайплайн + обучение LogisticRegression
для датасета Breast Cancer Wisconsin (Diagnostic).

• Логи всех задач — в  <project_root>/logs/
• Запуск по расписанию — раз в сутки (@daily).  Можно переопределить переменной
  окружения PIPELINE_SCHEDULE или вручную изменить schedule_interval.
"""

from __future__ import annotations

import os
import pathlib
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ─────────────────────────────────────────────────────────────────────────────
# PYTHONPATH: чтобы Airflow нашёл локальный пакет `etl`
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Логи Airflow → <project_root>/logs  (если не переопределено извне)
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("AIRFLOW__LOGGING__BASE_LOG_FOLDER", str(LOG_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# Импорт шагов пайплайна
# ─────────────────────────────────────────────────────────────────────────────
from etl import (  # noqa: E402
    fetch_data,
    preprocess,
    split_train_test,
    train_model,
    evaluate_model,
    save_results,
)

# ─────────────────────────────────────────────────────────────────────────────
# Аргументы DAG
# ─────────────────────────────────────────────────────────────────────────────
default_args = {
    "owner": "data_engineer",
    "email": ["alerts@example.com"],
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}

SCHEDULE = os.getenv("PIPELINE_SCHEDULE", "@daily")  # измените на None для «ручного» запуска

with DAG(
    dag_id="breast_cancer_pipeline",
    description="End-to-end ETL + LogReg training for Breast Cancer Diagnosis",
    default_args=default_args,
    start_date=datetime(2025, 6, 12),
    schedule="@daily",
    catchup=False,
    tags=["ml", "etl", "breast_cancer"],
) as dag:

    # ───────────── 1. Скачивание датасета ─────────────
    fetch = PythonOperator(
        task_id="fetch_data",
        python_callable=lambda: fetch_data.main([]),
        retries=3,
        retry_delay=timedelta(minutes=2),
    )

    # ───────────── 2. Предобработка ─────────────
    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=lambda: preprocess.main([]),
    )

    # ───────────── 3. Сплит ─────────────
    split = PythonOperator(
        task_id="split_train_test",
        python_callable=lambda: split_train_test.main([]),
    )

    # ───────────── 4. Обучение ─────────────
    train = PythonOperator(
        task_id="train_model",
        python_callable=lambda: train_model.main([]),
        execution_timeout=timedelta(minutes=10),
    )

    # ───────────── 5. Оценка ─────────────
    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=lambda: evaluate_model.main([]),
    )

    # ───────────── 6. Сохранение результатов ─────────────
    save = PythonOperator(
        task_id="save_results",
        python_callable=lambda: save_results.main([]),
        trigger_rule="all_success",
    )

    # ───────────── Зависимости ─────────────
    fetch >> preprocess_task >> split >> train >> evaluate >> save