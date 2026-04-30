"""섹터 상대 강도 피처 — 종목의 섹터 내 상대 성과를 수치화.

추가 피처 (3개):
    sector_return_20d   : 동일 섹터 종목들의 동등 가중 20일 수익률 평균
    sector_rel_strength : 개별 종목 return_20d / sector_return_20d
    sector_rank_pct     : 섹터 내 return_20d 백분위 순위 (0=최하, 1=최상)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import logger

SECTOR_FEATURE_COLS = [
    "sector_return_20d",
    "sector_rel_strength",
    "sector_rank_pct",
]

FEATURES_DIR = Path("data/features")


def fetch_sector_map() -> pd.DataFrame:
    """S&P 500 섹터 매핑 로드.

    우선순위:
      1. data/features/sector_map.csv (캐시)
      2. DB stocks 테이블
      3. Wikipedia (온라인)

    Returns:
        DataFrame with columns [ticker, sector]
    """
    cache_path = FEATURES_DIR / "sector_map.csv"
    if cache_path.exists():
        logger.info(f"섹터 매핑 로드 (캐시): {cache_path}")
        return pd.read_csv(cache_path)

    # DB에서 시도
    try:
        from sqlalchemy import text
        from src.database import get_session
        with get_session() as session:
            rows = session.execute(
                text("SELECT ticker, sector FROM stocks WHERE is_active = TRUE AND sector IS NOT NULL")
            ).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["ticker", "sector"])
            _save_sector_map(df, cache_path)
            logger.info(f"섹터 매핑 로드 (DB): {len(df)}개 종목")
            return df
    except Exception as e:
        logger.warning(f"DB에서 섹터 매핑 로드 실패 ({e}), Wikipedia로 시도")

    # Wikipedia fallback
    from src.data_collection.sp500_universe import fetch_sp500_tickers
    df = fetch_sp500_tickers()[["ticker", "sector"]].dropna()
    _save_sector_map(df, cache_path)
    logger.info(f"섹터 매핑 로드 (Wikipedia): {len(df)}개 종목")
    return df


def _save_sector_map(df: pd.DataFrame, path: Path) -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def add_sector_features(
    df: pd.DataFrame,
    sector_map: pd.DataFrame,
) -> pd.DataFrame:
    """features_all DataFrame에 섹터 상대 강도 피처 3개를 추가.

    Args:
        df: features_all DataFrame. 반드시 [ticker, date, return_20d] 포함.
        sector_map: [ticker, sector] DataFrame.
    """
    if "return_20d" not in df.columns:
        logger.warning("return_20d 컬럼 없음 — 섹터 피처 건너뜀")
        return df

    df = df.copy()
    df = df.merge(sector_map[["ticker", "sector"]], on="ticker", how="left")

    # 섹터 정보 없는 종목은 전체 평균으로 대체
    unknown_mask = df["sector"].isna()
    if unknown_mask.any():
        logger.debug(f"섹터 정보 없는 종목 {unknown_mask.sum():,}행 → 'Unknown' 처리")
        df.loc[unknown_mask, "sector"] = "Unknown"

    # ── 날짜별 섹터 평균 return_20d ──────────────────────────────
    sector_avg = (
        df.groupby(["date", "sector"])["return_20d"]
        .mean()
        .rename("sector_return_20d")
        .reset_index()
    )
    df = df.merge(sector_avg, on=["date", "sector"], how="left")

    # ── 상대 강도: 개별 / 섹터 평균 ─────────────────────────────
    sector_ret = df["sector_return_20d"]
    df["sector_rel_strength"] = df["return_20d"] / sector_ret.where(sector_ret != 0).abs()
    df["sector_rel_strength"] = df["sector_rel_strength"].clip(-5, 5)  # 극단값 제한

    # ── 섹터 내 백분위 순위 ───────────────────────────────────────
    df["sector_rank_pct"] = (
        df.groupby(["date", "sector"])["return_20d"]
        .rank(pct=True, method="average")
    )

    df = df.drop(columns=["sector"])

    added = [c for c in SECTOR_FEATURE_COLS if c in df.columns]
    logger.info(f"섹터 피처 {len(added)}개 추가: {added}")
    return df
