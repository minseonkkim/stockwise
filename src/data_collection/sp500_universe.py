"""S&P 500 종목 유니버스 관리.

Wikipedia에서 현재 편입 종목 목록을 가져오고 DB에 동기화한다.
"""
from datetime import date
from typing import Optional

import pandas as pd
import requests
from sqlalchemy import select, update
from tenacity import retry, stop_after_attempt, wait_exponential

from src.database import Stock, get_session
from src.utils import logger

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_sp500_tickers() -> pd.DataFrame:
    """Wikipedia에서 S&P 500 종목 목록을 가져온다."""
    resp = requests.get(WIKIPEDIA_URL, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(resp.text, header=0)
    df = tables[0]

    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
            "Date added": "added_to_sp500",
        }
    )
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df = df[["ticker", "name", "sector", "industry", "added_to_sp500"]].copy()
    logger.info(f"Fetched {len(df)} S&P 500 tickers from Wikipedia")
    return df


def sync_universe_to_db(df: Optional[pd.DataFrame] = None) -> dict:
    """
    Wikipedia 종목 목록을 DB stocks 테이블과 동기화한다.

    Returns:
        {"inserted": int, "updated": int, "deactivated": int}
    """
    if df is None:
        df = fetch_sp500_tickers()

    current_tickers = set(df["ticker"].tolist())
    counts = {"inserted": 0, "updated": 0, "deactivated": 0}

    with get_session() as session:
        # 기존 활성 종목 조회
        existing = {
            row.ticker: row
            for row in session.scalars(select(Stock).where(Stock.is_active == True)).all()
        }

        # 신규 or 업데이트
        for _, row in df.iterrows():
            ticker = row["ticker"]
            added = _parse_date(row.get("added_to_sp500"))

            if ticker in existing:
                stock = existing[ticker]
                stock.name = row["name"]
                stock.sector = row.get("sector")
                stock.industry = row.get("industry")
                counts["updated"] += 1
            else:
                session.add(
                    Stock(
                        ticker=ticker,
                        name=row["name"],
                        sector=row.get("sector"),
                        industry=row.get("industry"),
                        is_active=True,
                        added_to_sp500=added,
                    )
                )
                counts["inserted"] += 1

        # 편출된 종목 비활성화
        for ticker, stock in existing.items():
            if ticker not in current_tickers:
                stock.is_active = False
                stock.removed_from_sp500 = date.today()
                counts["deactivated"] += 1

    logger.info(
        f"Universe sync complete — inserted={counts['inserted']}, "
        f"updated={counts['updated']}, deactivated={counts['deactivated']}"
    )
    return counts


def get_active_tickers() -> list[str]:
    """DB에서 현재 활성 S&P 500 티커 목록을 반환한다."""
    with get_session() as session:
        rows = session.scalars(
            select(Stock.ticker).where(Stock.is_active == True).order_by(Stock.ticker)
        ).all()
    return list(rows)


def _parse_date(value) -> Optional[date]:
    if pd.isna(value) or value is None:
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None
