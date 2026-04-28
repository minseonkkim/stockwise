"""데이터 품질 검증 모듈.

Phase 1 완료 기준의 핵심:
- 누락 거래일 탐지
- 가격 이상치 탐지 (전일 대비 ±50% 이상 변동)
- 0 / 음수 값 탐지
- 검증 결과 DB 기록 + Slack 알림(선택)
"""
import json
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sqlalchemy import select, text

from src.config import settings
from src.database import DailyPrice, ValidationAlert, get_session
from src.utils import logger

# 미국 주요 공휴일 (고정, 간단 처리 — 실제 운영 시 pandas_market_calendars 사용 권장)
_KNOWN_HOLIDAYS_2024_2025 = {
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19),
    date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
    date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28),
    date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
}


# ────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────

def validate_ticker(ticker: str, lookback_days: int = 30) -> list[dict]:
    """
    단일 종목의 최근 데이터를 검증하고 이상 항목을 반환한다.

    Returns:
        알림 목록 [{"type": str, "message": str, "date": date}]
    """
    alerts = []
    with get_session() as session:
        df = _load_recent(session, ticker, lookback_days)

    if df.empty:
        alerts.append({
            "type": "no_data",
            "message": f"{ticker}: 최근 {lookback_days}일 데이터 없음",
            "date": date.today(),
        })
        return alerts

    alerts.extend(_check_missing_days(ticker, df))
    alerts.extend(_check_price_outliers(ticker, df))
    alerts.extend(_check_zero_values(ticker, df))

    return alerts


def validate_all(tickers: list[str], lookback_days: int = 30) -> dict:
    """전 종목 검증 후 DB 저장 및 Slack 알림."""
    all_alerts = []

    for ticker in tickers:
        try:
            alerts = validate_ticker(ticker, lookback_days)
            all_alerts.extend(alerts)
        except Exception as e:
            logger.warning(f"Validation error for {ticker}: {e}")

    _save_alerts(all_alerts)

    if all_alerts:
        _notify_slack(all_alerts)
        logger.warning(f"Validation found {len(all_alerts)} alerts")
    else:
        logger.info("Validation passed — no alerts")

    return {"total_alerts": len(all_alerts), "alerts": all_alerts}


# ────────────────────────────────────────────
# Checks
# ────────────────────────────────────────────

def _check_missing_days(ticker: str, df: pd.DataFrame) -> list[dict]:
    """예상 거래일 대비 누락 날짜 탐지."""
    if df.empty:
        return []

    alerts = []
    min_date = df["date"].min()
    max_date = df["date"].max()
    existing = set(df["date"].tolist())

    expected = _get_trading_days(min_date, max_date)
    missing = [d for d in expected if d not in existing]

    if missing:
        alerts.append({
            "type": "missing_days",
            "message": f"{ticker}: {len(missing)}개 거래일 누락 — {missing[:5]}{'...' if len(missing) > 5 else ''}",
            "date": date.today(),
        })
    return alerts


def _check_price_outliers(ticker: str, df: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    """전일 대비 ±50% 이상 변동 탐지."""
    if len(df) < 2:
        return []

    alerts = []
    df = df.sort_values("date")
    df["pct_change"] = df["close"].pct_change().abs()
    outliers = df[df["pct_change"] > threshold]

    for _, row in outliers.iterrows():
        alerts.append({
            "type": "outlier",
            "message": (
                f"{ticker} [{row['date']}]: 전일 대비 "
                f"{row['pct_change']:.1%} 가격 변동 (종가 {row['close']:.2f})"
            ),
            "date": row["date"],
        })
    return alerts


def _check_zero_values(ticker: str, df: pd.DataFrame) -> list[dict]:
    """0 또는 음수 가격 / 거래량 탐지."""
    alerts = []
    price_cols = ["open", "high", "low", "close", "adj_close"]

    bad_price = df[(df[price_cols] <= 0).any(axis=1)]
    for _, row in bad_price.iterrows():
        alerts.append({
            "type": "zero_price",
            "message": f"{ticker} [{row['date']}]: 0 또는 음수 가격 감지",
            "date": row["date"],
        })

    bad_volume = df[df["volume"] < 0]
    for _, row in bad_volume.iterrows():
        alerts.append({
            "type": "negative_volume",
            "message": f"{ticker} [{row['date']}]: 음수 거래량 감지 ({row['volume']})",
            "date": row["date"],
        })

    return alerts


# ────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────

def _load_recent(session, ticker: str, lookback_days: int) -> pd.DataFrame:
    since = date.today() - timedelta(days=lookback_days)
    rows = session.execute(
        text(
            "SELECT date, open, high, low, close, adj_close, volume "
            "FROM daily_prices WHERE ticker = :ticker AND date >= :since "
            "ORDER BY date"
        ),
        {"ticker": ticker, "since": since},
    ).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "adj_close", "volume"])


def _get_trading_days(start: date, end: date) -> list[date]:
    """토/일/공휴일을 제외한 예상 거래일 목록."""
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5 and cur not in _KNOWN_HOLIDAYS_2024_2025:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _save_alerts(alerts: list[dict]) -> None:
    if not alerts:
        return
    with get_session() as session:
        for a in alerts:
            session.add(
                ValidationAlert(
                    ticker=a.get("ticker"),
                    alert_date=a["date"],
                    alert_type=a["type"],
                    message=a["message"],
                )
            )


def _notify_slack(alerts: list[dict]) -> None:
    if not settings.SLACK_WEBHOOK_URL:
        return

    lines = [f"*[데이터 검증 알림]* {len(alerts)}건\n"]
    for a in alerts[:10]:
        lines.append(f"• `{a['type']}` {a['message']}")
    if len(alerts) > 10:
        lines.append(f"... 외 {len(alerts) - 10}건")

    payload = {"text": "\n".join(lines)}
    try:
        resp = requests.post(settings.SLACK_WEBHOOK_URL, json=payload, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Slack notification failed: {e}")
