"""피처 현황 엔드포인트.

GET /api/features/{ticker}    종목별 최신 피처 값
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

FEATURES_DIR = Path("data/features")

router = APIRouter()


@lru_cache(maxsize=1)
def _load_features_all() -> pd.DataFrame:
    path = FEATURES_DIR / "features_all.parquet"
    if not path.exists():
        raise FileNotFoundError("build_features.py 를 먼저 실행하세요.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@router.get("/{ticker}")
def get_ticker_features(ticker: str):
    """종목의 가장 최근 날짜 피처 값을 반환."""
    try:
        df = _load_features_all()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    ticker = ticker.upper()
    sub = df[df["ticker"] == ticker]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"ticker '{ticker}' not found")

    latest = sub.sort_values("date").iloc[-1]
    feature_dict = {
        col: (None if pd.isna(v) else round(float(v), 6) if isinstance(v, float) else v)
        for col, v in latest.items()
        if col not in ("ticker",)
    }
    feature_dict["date"] = str(latest["date"].date())
    feature_dict["ticker"] = ticker

    return feature_dict
