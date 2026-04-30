"""Celery 일별 파이프라인 — Phase 5 자동화 태스크.

매일 오후 6시(ET, UTC-5 기준 23:00 UTC) 자동 실행:
    1. 증분 데이터 수집
    2. 피처 빌드
    3. 예측 실행 (캐시 갱신)
    4. 신호 저장 (로그)
"""
from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from src.config.settings import settings

app = Celery(
    "stockwise",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

app.conf.timezone = "UTC"

# 매일 23:00 UTC (ET 18:00) 자동 실행
app.conf.beat_schedule = {
    "daily-pipeline": {
        "task": "src.tasks.pipeline.daily_pipeline",
        "schedule": crontab(hour=23, minute=0),
    },
}


@app.task(name="src.tasks.pipeline.daily_pipeline", bind=True, max_retries=3)
def daily_pipeline(self):
    """일별 ML 파이프라인: 수집 → 피처 → 예측 → 저장."""
    from src.utils import logger

    try:
        # 1. 증분 데이터 수집
        logger.info("[Pipeline] Step 1: 증분 데이터 수집")
        from src.data_collection import collect_incremental, get_active_tickers
        collect_incremental(get_active_tickers())

        # 2. 피처 빌드
        logger.info("[Pipeline] Step 2: 피처 빌드")
        from src.features.pipeline import run_feature_pipeline
        run_feature_pipeline()

        # 3. 예측 캐시 초기화 → 재생성
        logger.info("[Pipeline] Step 3: 예측 실행 (캐시 갱신)")
        from src.api.deps import get_predictions_df
        get_predictions_df.cache_clear()
        predictions = get_predictions_df()
        latest = predictions["date"].max()
        buy_count = (predictions[predictions["date"] == latest]["signal"] == "buy").sum()

        # 4. 결과 로깅
        logger.info(
            f"[Pipeline] Step 4 완료 — 날짜: {latest.date()}, "
            f"매수 신호: {buy_count}개"
        )
        return {
            "status": "ok",
            "date": str(latest.date()),
            "buy_signals": int(buy_count),
        }

    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * 10)  # 10분 후 재시도
