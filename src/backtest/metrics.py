"""백테스트 성과 지표 계산.

PLAN.md Phase 4 목표:
  - CAGR : S&P 500 초과
  - 샤프  : 1.0 이상
  - MDD   : 20% 이내
  - 승률  : 50% 이상
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BacktestMetrics:
    """일별 수익률 시리즈로부터 성과 지표 계산."""

    # Phase 4 목표 기준
    TARGET_SHARPE = 1.0
    TARGET_MDD    = -0.20
    TARGET_WIN_RATE = 0.50

    def __init__(self, returns: pd.Series, periods_per_year: int = 252):
        self.returns = returns.dropna()
        self.periods_per_year = periods_per_year

    # ─────────────────────────────────────────────
    # 핵심 지표
    # ─────────────────────────────────────────────

    def cagr(self) -> float:
        """연 복리 수익률 (CAGR)."""
        n_years = len(self.returns) / self.periods_per_year
        if n_years <= 0:
            return 0.0
        total = (1 + self.returns).prod()
        return float(total ** (1 / n_years) - 1)

    def annualized_volatility(self) -> float:
        """연화 변동성."""
        return float(self.returns.std() * np.sqrt(self.periods_per_year))

    def sharpe_ratio(self, risk_free: float = 0.04) -> float:
        """샤프 비율 (무위험 수익률 기본 4%)."""
        excess = self.cagr() - risk_free
        vol = self.annualized_volatility()
        return float(excess / vol) if vol > 0 else 0.0

    def max_drawdown(self) -> float:
        """최대 낙폭 (MDD). 음수 반환."""
        equity = (1 + self.returns).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1
        return float(drawdown.min())

    def win_rate(self) -> float:
        """승률 — 양수 수익률 비율."""
        return float((self.returns > 0).mean())

    def calmar_ratio(self) -> float:
        """칼마 비율 = CAGR / |MDD|."""
        mdd = abs(self.max_drawdown())
        return float(self.cagr() / mdd) if mdd > 0 else 0.0

    def total_return(self) -> float:
        """누적 총 수익률."""
        return float((1 + self.returns).prod() - 1)

    # ─────────────────────────────────────────────
    # 요약
    # ─────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "total_return":   round(self.total_return(), 4),
            "cagr":           round(self.cagr(), 4),
            "volatility":     round(self.annualized_volatility(), 4),
            "sharpe_ratio":   round(self.sharpe_ratio(), 4),
            "max_drawdown":   round(self.max_drawdown(), 4),
            "win_rate":       round(self.win_rate(), 4),
            "calmar_ratio":   round(self.calmar_ratio(), 4),
            "n_days":         len(self.returns),
        }

    def check_phase4_targets(self, benchmark_cagr: float = 0.0) -> dict:
        """Phase 4 목표 달성 여부 체크."""
        s = self.summary()
        checks = {
            "cagr_vs_benchmark": {
                "value": s["cagr"],
                "benchmark": round(benchmark_cagr, 4),
                "pass": s["cagr"] > benchmark_cagr,
            },
            "sharpe_gte_1": {
                "value": s["sharpe_ratio"],
                "target": self.TARGET_SHARPE,
                "pass": s["sharpe_ratio"] >= self.TARGET_SHARPE,
            },
            "mdd_within_20pct": {
                "value": s["max_drawdown"],
                "target": self.TARGET_MDD,
                "pass": s["max_drawdown"] >= self.TARGET_MDD,
            },
            "win_rate_gte_50pct": {
                "value": s["win_rate"],
                "target": self.TARGET_WIN_RATE,
                "pass": s["win_rate"] >= self.TARGET_WIN_RATE,
            },
        }
        return checks
