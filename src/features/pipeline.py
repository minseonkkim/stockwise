"""피처 파이프라인 — 전 종목 피처 계산 후 parquet 저장.

실행 순서:
    1. DB에서 OHLCV 로드 (종목별)
    2. 차트 기법 피처 + 수치 피처 계산
    3. 타깃 레이블 추가
    4. 전체 합치기 → data/features/features_all.parquet
"""
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.database import get_session
from src.features.chart_features import CHART_FEATURE_COLS, compute_chart_features
from src.features.quant_features import QUANT_FEATURE_COLS, compute_quant_features
from src.features.target import add_target
from src.utils import logger

FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

ALL_FEATURE_COLS = CHART_FEATURE_COLS + QUANT_FEATURE_COLS


def build_features_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """단일 종목 OHLCV → 피처 DataFrame."""
    df = compute_chart_features(df)
    df = compute_quant_features(df)
    df = add_target(df, threshold=threshold)
    df["ticker"] = ticker
    return df


def build_all_features(
    tickers: list[str] | None = None,
    batch_size: int = 50,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """
    전 종목 피처를 계산하고 합친 DataFrame을 반환한다.

    Args:
        tickers: None이면 DB의 전체 활성 종목
        batch_size: DB 조회 배치 크기
        threshold: 타깃 레이블 수익률 기준 (기본 0.02 = +2%)
    """
    if tickers is None:
        tickers = _get_all_tickers()

    logger.info(f"Building features for {len(tickers)} tickers (target threshold={threshold:.1%})")
    results = []
    failed  = []

    for i, ticker in enumerate(tickers, 1):
        try:
            df = _load_ohlcv(ticker)
            if df is None or len(df) < 70:  # 최소 60일 필요
                logger.debug(f"{ticker}: insufficient data ({len(df) if df is not None else 0} rows)")
                continue

            feat_df = build_features_for_ticker(ticker, df, threshold=threshold)
            results.append(feat_df)

            if i % 50 == 0:
                logger.info(f"  {i}/{len(tickers)} tickers processed")

        except Exception as e:
            logger.warning(f"{ticker} feature error: {e}")
            failed.append(ticker)

    if not results:
        raise RuntimeError("No features computed — check DB connection and data")

    combined = pd.concat(results, ignore_index=True)
    logger.info(f"Feature build complete: {len(combined):,} rows, {len(results)} tickers, {len(failed)} failed")

    if failed:
        logger.warning(f"Failed tickers: {failed[:10]}{'...' if len(failed) > 10 else ''}")

    return combined


def save_features(df: pd.DataFrame) -> Path:
    """전체 피처를 parquet으로 저장."""
    path = FEATURES_DIR / "features_all.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Saved {len(df):,} rows to {path}")
    return path


def load_features() -> pd.DataFrame:
    """저장된 피처 parquet 로드."""
    path = FEATURES_DIR / "features_all.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run build_features.py first: {path}")
    return pd.read_parquet(path)


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _load_ohlcv(ticker: str) -> pd.DataFrame | None:
    with get_session() as session:
        rows = session.execute(
            text(
                "SELECT date, open, high, low, close, adj_close, volume "
                "FROM daily_prices WHERE ticker = :ticker ORDER BY date"
            ),
            {"ticker": ticker},
        ).fetchall()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _get_all_tickers() -> list[str]:
    with get_session() as session:
        rows = session.execute(
            text("SELECT ticker FROM stocks WHERE is_active = TRUE ORDER BY ticker")
        ).fetchall()
    return [r[0] for r in rows]
