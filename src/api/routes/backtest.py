"""백테스트 성과 엔드포인트.

GET /api/backtest/summary    성과 지표 요약 (JSON)
GET /api/backtest/equity     일별 누적 수익률 시계열
GET /api/backtest/weekly     주간 수익률 테이블
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

REPORTS_DIR = Path("data/reports")

router = APIRouter()


def _load_json(name: str) -> dict:
    path = REPORTS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"run_backtest.py 를 먼저 실행하세요. ({path})",
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/summary")
def get_backtest_summary():
    """백테스트 성과 지표 요약."""
    return _load_json("backtest_result")


@router.get("/equity")
def get_equity_curve():
    """일별 누적 수익률 시계열 (차트용)."""
    path = REPORTS_DIR / "backtest_daily_returns.csv"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="run_backtest.py 를 먼저 실행하세요.",
        )
    df = pd.read_csv(path, parse_dates=[0])
    df.columns = ["date", "ret"]
    df = df.dropna()

    # 누적 수익률
    df["equity"] = (1 + df["ret"]).cumprod()

    rows = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "daily_return": round(float(row.ret), 6),
            "equity": round(float(row.equity), 6),
        }
        for row in df.itertuples()
    ]
    return {"count": len(rows), "equity_curve": rows}


@router.get("/weekly")
def get_weekly_returns():
    """주간 수익률 테이블."""
    path = REPORTS_DIR / "backtest_weekly.csv"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="run_backtest.py 를 먼저 실행하세요.",
        )
    df = pd.read_csv(path)
    return {"count": len(df), "weekly": df.to_dict(orient="records")}
