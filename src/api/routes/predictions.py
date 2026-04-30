"""예측 신호 엔드포인트.

GET /api/predictions              오늘(최신 날짜)의 전 종목 신호
GET /api/predictions/{ticker}     특정 종목 예측 히스토리
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from src.api.deps import N_TOP, get_predictions_df

router = APIRouter()


@router.get("")
def get_today_predictions(n_top: Annotated[int, Query(ge=1, le=200)] = N_TOP):
    """
    최신 날짜의 전 종목 예측 신호를 prob 순위 기준으로 반환.

    - signal = "buy"  : prob 상위 n_top 개 종목
    - signal = "hold" : 나머지
    """
    try:
        df = get_predictions_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    latest_date = df["date"].max()
    today = df[df["date"] == latest_date].copy()

    # n_top이 기본값(N_TOP)과 다르면 신호 재계산
    if n_top != N_TOP:
        today = today.sort_values("prob", ascending=False)
        today["rank"] = range(1, len(today) + 1)
        today["signal"] = today["rank"].apply(lambda r: "buy" if r <= n_top else "hold")

    today = today.sort_values("rank")

    rows = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "ticker": row.ticker,
            "prob": round(float(row.prob), 4),
            "signal": row.signal,
            "rank": int(row.rank),
        }
        for row in today.itertuples()
    ]

    return {
        "date": latest_date.strftime("%Y-%m-%d"),
        "n_top": n_top,
        "total": len(rows),
        "predictions": rows,
    }


@router.get("/{ticker}")
def get_ticker_predictions(ticker: str, limit: Annotated[int, Query(ge=1, le=2000)] = 500):
    """특정 종목의 예측 히스토리 (최근 limit 행)."""
    try:
        df = get_predictions_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    ticker = ticker.upper()
    sub = df[df["ticker"] == ticker].sort_values("date", ascending=False).head(limit)

    if sub.empty:
        raise HTTPException(status_code=404, detail=f"ticker '{ticker}' not found in test split")

    rows = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "ticker": row.ticker,
            "prob": round(float(row.prob), 4),
            "signal": row.signal,
            "rank": int(row.rank),
        }
        for row in sub.itertuples()
    ]

    return {"ticker": ticker, "count": len(rows), "history": rows}
