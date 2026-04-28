"""yfinance 기반 일별 OHLCV 수집기.

Phase 1 핵심 모듈.
- 배치 단위로 yfinance 호출 (rate limit 준수)
- 결측치 / 상장폐지 종목 예외 처리
- DB upsert (중복 적재 방지)
- 수집 이력 DataCollectionLog 기록
"""
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.database import DailyPrice, DataCollectionLog, get_session
from src.utils import logger


# ────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────

def collect_all(
    tickers: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    tickers 전체를 배치 단위로 수집해 DB에 저장한다.

    Args:
        tickers: 수집할 티커 목록
        start_date: "YYYY-MM-DD" (기본값: settings.START_DATE)
        end_date:   "YYYY-MM-DD" (기본값: 오늘)

    Returns:
        {"success": int, "failed": int, "skipped": int}
    """
    start_date = start_date or settings.START_DATE
    end_date = end_date or date.today().strftime("%Y-%m-%d")

    counts = {"success": 0, "failed": 0, "skipped": 0}
    batches = _make_batches(tickers, settings.BATCH_SIZE)

    for i, batch in enumerate(batches, 1):
        logger.info(f"Batch {i}/{len(batches)}: {len(batch)} tickers")
        batch_counts = _collect_batch(batch, start_date, end_date)
        for k in counts:
            counts[k] += batch_counts[k]
        time.sleep(settings.YFINANCE_DELAY)

    logger.info(
        f"Collection complete — success={counts['success']}, "
        f"failed={counts['failed']}, skipped={counts['skipped']}"
    )
    return counts


def collect_incremental(tickers: list[str]) -> dict:
    """
    각 종목의 마지막 저장 날짜 다음 날부터 오늘까지 증분 수집한다.
    일별 Airflow DAG에서 호출.
    """
    end_date = date.today().strftime("%Y-%m-%d")
    counts = {"success": 0, "failed": 0, "skipped": 0}

    with get_session() as session:
        last_dates = _get_last_dates(session, tickers)

    batches = _make_batches(tickers, settings.BATCH_SIZE)

    for i, batch in enumerate(batches, 1):
        logger.info(f"Incremental batch {i}/{len(batches)}")
        for ticker in batch:
            last = last_dates.get(ticker)
            if last:
                start = (last + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start = settings.START_DATE

            if start > end_date:
                counts["skipped"] += 1
                continue

            result = _collect_single(ticker, start, end_date)
            counts[result["status"]] += 1

        time.sleep(settings.YFINANCE_DELAY)

    return counts


# ────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────

def _collect_batch(batch: list[str], start_date: str, end_date: str) -> dict:
    counts = {"success": 0, "failed": 0, "skipped": 0}
    for ticker in batch:
        result = _collect_single(ticker, start_date, end_date)
        counts[result["status"]] += 1
    return counts


def _collect_single(ticker: str, start_date: str, end_date: str) -> dict:
    t0 = time.time()
    try:
        df = _download(ticker, start_date, end_date)

        if df is None or df.empty:
            _log_collection(ticker, "skipped", 0, 0, "Empty data returned", time.time() - t0)
            return {"status": "skipped"}

        df = _normalize(df, ticker)
        inserted, updated = _upsert(df, ticker)
        _log_collection(ticker, "success", inserted, updated, None, time.time() - t0)
        logger.debug(f"{ticker}: inserted={inserted}, updated={updated}")
        return {"status": "success"}

    except Exception as e:
        msg = str(e)
        logger.warning(f"{ticker} failed: {msg}")
        _log_collection(ticker, "failed", 0, 0, msg[:500], time.time() - t0)
        return {"status": "failed"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _download(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """yfinance 다운로드 (재시도 포함)."""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, auto_adjust=False)

    if df is None or df.empty:
        return None

    return df


def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance 컬럼명을 DB 컬럼명으로 정규화."""
    df = df.copy()
    df.index = pd.to_datetime(df.index).date

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # 필요한 컬럼만 유지
    required = ["open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # adj_close가 없는 경우 close로 대체
        if "adj_close" in missing and "close" in df.columns:
            df["adj_close"] = df["close"]
            missing.remove("adj_close")
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    df = df[required].dropna(subset=["open", "high", "low", "close"])
    df["ticker"] = ticker
    df.index.name = "date"
    df = df.reset_index()

    # 음수 가격 / 0 거래량 필터링
    df = df[
        (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
    ]
    df["volume"] = df["volume"].fillna(0).astype("int64")

    return df


def _upsert(df: pd.DataFrame, ticker: str) -> tuple[int, int]:
    """daily_prices에 upsert. (ticker, date) 충돌 시 UPDATE."""
    records = df.to_dict(orient="records")
    inserted = 0
    updated = 0

    with get_session() as session:
        for record in records:
            stmt = (
                pg_insert(DailyPrice)
                .values(**record)
                .on_conflict_do_update(
                    constraint="uq_daily_prices_ticker_date",
                    set_={
                        "open": record["open"],
                        "high": record["high"],
                        "low": record["low"],
                        "close": record["close"],
                        "adj_close": record["adj_close"],
                        "volume": record["volume"],
                    },
                )
            )
            result = session.execute(stmt)
            # rowcount: 1=inserted or 2=updated (PostgreSQL upsert 규칙)
            if result.rowcount == 1:
                inserted += 1
            else:
                updated += 1

    return inserted, updated


def _log_collection(
    ticker: str,
    status: str,
    inserted: int,
    updated: int,
    error: Optional[str],
    duration: float,
) -> None:
    with get_session() as session:
        session.add(
            DataCollectionLog(
                ticker=ticker,
                collection_date=date.today(),
                status=status,
                rows_inserted=inserted,
                rows_updated=updated,
                error_message=error,
                duration_seconds=round(duration, 3),
            )
        )


def _get_last_dates(session, tickers: list[str]) -> dict[str, Optional[date]]:
    """각 티커의 DB 최신 날짜를 반환."""
    result = {}
    if not tickers:
        return result

    rows = session.execute(
        text(
            "SELECT ticker, MAX(date) as last_date FROM daily_prices "
            "WHERE ticker = ANY(:tickers) GROUP BY ticker"
        ),
        {"tickers": tickers},
    ).fetchall()

    for row in rows:
        result[row.ticker] = row.last_date

    return result


def _make_batches(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]
