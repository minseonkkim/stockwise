"""벡터화 포트폴리오 백테스트 시뮬레이터.

전략:
  - 매주 첫 거래일에 예측 확률 상위 N개 종목 선택
  - 동일 비중 (Equal Weight)
  - 5거래일 보유 후 다음 주에 리밸런싱
  - 거래비용: 0.1% 매수 + 0.1% 매도 = 0.2% 왕복
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import logger


class VectorizedBacktest:
    """벡터화 방식 주간 리밸런싱 포트폴리오 시뮬레이터."""

    def __init__(
        self,
        initial_capital: float = 100_000,
        n_top: int = 20,
        transaction_cost: float = 0.001,   # 단방향 0.1%
        min_stocks: int = 5,               # 리밸런싱 날 최소 종목 수
    ):
        self.initial_capital = initial_capital
        self.n_top = n_top
        self.transaction_cost = transaction_cost
        self.min_stocks = min_stocks

        # 결과 저장
        self.weekly_results: pd.DataFrame | None = None
        self.portfolio_returns: pd.Series | None = None
        self.holdings_history: list[dict] = []

    def run(self, pred_df: pd.DataFrame) -> pd.Series:
        """백테스트 실행.

        Args:
            pred_df: [ticker, date, prob, actual_return_5d] 컬럼 포함 DataFrame.

        Returns:
            일별 포트폴리오 수익률 pd.Series (index=date).
        """
        pred_df = pred_df.copy()
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df = pred_df.sort_values("date")

        # 주간 그룹: 각 주의 첫 거래일만 사용 (리밸런싱 기준일)
        pred_df["week"] = pred_df["date"].dt.to_period("W-SUN")
        rebal_dates = (
            pred_df.groupby("week")["date"].min().reset_index(name="rebal_date")
        )

        rows = []
        prev_tickers: set[str] = set()

        for _, row in rebal_dates.iterrows():
            rebal_date = row["rebal_date"]
            day_df = pred_df[pred_df["date"] == rebal_date].copy()

            if len(day_df) < self.min_stocks:
                continue

            # 예측 확률 상위 N개 선택
            top_df = day_df.nlargest(self.n_top, "prob")
            selected = set(top_df["ticker"].tolist())

            # 거래비용 계산 (변경된 포지션에만 적용)
            changed = len(selected.symmetric_difference(prev_tickers))
            turnover_rate = changed / (2 * self.n_top)           # 0~1
            total_cost = 2 * self.transaction_cost * turnover_rate  # 왕복

            # 포트폴리오 수익률 (동일 비중 평균)
            gross_return = top_df["actual_return_5d"].mean()
            net_return = gross_return - total_cost

            rows.append({
                "date": rebal_date,
                "week": row["week"],
                "gross_return": gross_return,
                "net_return": net_return,
                "transaction_cost": total_cost,
                "turnover_rate": turnover_rate,
                "n_stocks": len(top_df),
                "avg_prob": top_df["prob"].mean(),
                "tickers": ",".join(sorted(selected)),
            })

            self.holdings_history.append({
                "date": rebal_date,
                "tickers": list(selected),
                "avg_prob": top_df["prob"].mean(),
            })

            prev_tickers = selected

        self.weekly_results = pd.DataFrame(rows).set_index("date")

        # 일별 수익률로 변환 (각 주의 수익을 5거래일에 분배)
        self.portfolio_returns = self._to_daily_returns(pred_df, self.weekly_results)

        logger.info(
            f"백테스트 완료: {len(self.weekly_results)}주 | "
            f"평균 주간 수익률: {self.weekly_results['net_return'].mean():.4f}"
        )
        return self.portfolio_returns

    def _to_daily_returns(
        self,
        pred_df: pd.DataFrame,
        weekly: pd.DataFrame,
    ) -> pd.Series:
        """주간 수익률 → 일별 균등 분배 수익률 변환."""
        # 실제 거래일 목록 (test 기간)
        all_dates = pred_df["date"].drop_duplicates().sort_values().reset_index(drop=True)

        daily_returns = pd.Series(0.0, index=all_dates, name="strategy")

        for rebal_date, w_row in weekly.iterrows():
            # 해당 주의 리밸런싱일부터 5거래일 인덱스
            idx = all_dates.searchsorted(rebal_date)
            period_dates = all_dates.iloc[idx : idx + 5]
            if len(period_dates) == 0:
                continue
            # 5거래일에 균등 분배
            per_day = w_row["net_return"] / len(period_dates)
            daily_returns.loc[period_dates] += per_day

        return daily_returns

    def get_equity_curve(self) -> pd.Series:
        """누적 자본 곡선 반환."""
        if self.portfolio_returns is None:
            raise RuntimeError("run()을 먼저 실행하세요.")
        equity = self.initial_capital * (1 + self.portfolio_returns).cumprod()
        equity.name = "equity"
        return equity
