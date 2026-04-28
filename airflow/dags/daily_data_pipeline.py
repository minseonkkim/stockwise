"""Airflow DAG — 매일 장 마감 후 데이터 수집 파이프라인.

스케줄: 매일 오후 6시 ET (UTC 23:00)
태스크 순서:
    1. sync_universe    — S&P 500 종목 목록 동기화
    2. collect_data     — 증분 OHLCV 수집
    3. validate_data    — 데이터 품질 검증
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "trade",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="daily_data_pipeline",
    default_args=default_args,
    description="S&P 500 일별 OHLCV 수집 파이프라인",
    schedule_interval="0 23 * * 1-5",  # UTC 23:00 = ET 18:00 (월~금)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["phase1", "data-collection"],
    max_active_runs=1,
) as dag:

    # ── Task 1: S&P 500 종목 목록 동기화 ──────────────────────
    def _sync_universe(**ctx) -> None:
        from src.data_collection import sync_universe_to_db

        counts = sync_universe_to_db()
        ctx["ti"].xcom_push(key="sync_counts", value=counts)

    sync_universe = PythonOperator(
        task_id="sync_universe",
        python_callable=_sync_universe,
    )

    # ── Task 2: 증분 데이터 수집 ──────────────────────────────
    def _collect_data(**ctx) -> None:
        from src.data_collection import collect_incremental, get_active_tickers

        tickers = get_active_tickers()
        counts = collect_incremental(tickers)
        ctx["ti"].xcom_push(key="collect_counts", value=counts)

    collect_data = PythonOperator(
        task_id="collect_data",
        python_callable=_collect_data,
    )

    # ── Task 3: 데이터 품질 검증 ──────────────────────────────
    def _validate_data(**ctx) -> None:
        from src.data_collection import get_active_tickers, validate_all

        tickers = get_active_tickers()
        result = validate_all(tickers, lookback_days=3)  # 최근 3일만 검증
        if result["total_alerts"] > 0:
            import logging
            logging.warning(f"Validation alerts: {result['total_alerts']}")

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=_validate_data,
    )

    # ── 의존성 ────────────────────────────────────────────────
    sync_universe >> collect_data >> validate_data
