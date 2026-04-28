"""백테스트 결과 리포트 생성 — 콘솔 출력 + Quantstats HTML."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.backtest.metrics import BacktestMetrics
from src.utils import logger

REPORTS_DIR = Path("data/reports")


class BacktestReporter:
    """백테스트 성과 리포트 생성기."""

    def __init__(
        self,
        strategy_returns: pd.Series,
        benchmark_ticker: str = "SPY",
    ):
        self.strategy_returns = strategy_returns
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_returns: pd.Series | None = None

    def fetch_benchmark(self) -> pd.Series:
        """yfinance로 벤치마크(SPY) 일별 수익률 다운로드."""
        start = self.strategy_returns.index.min()
        end   = self.strategy_returns.index.max()

        logger.info(f"벤치마크 다운로드: {self.benchmark_ticker} ({start.date()} ~ {end.date()})")
        raw = yf.download(
            self.benchmark_ticker,
            start=start,
            end=end + pd.Timedelta(days=1),
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            logger.warning("벤치마크 데이터 없음 — 0 수익률로 대체")
            self.benchmark_returns = pd.Series(0.0, index=self.strategy_returns.index)
        else:
            closes = raw["Close"]
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            closes.index = pd.to_datetime(closes.index)
            self.benchmark_returns = closes.pct_change().dropna()
            self.benchmark_returns.name = self.benchmark_ticker

        return self.benchmark_returns

    def print_console_report(self) -> None:
        """콘솔 성과 요약 출력."""
        strat_m = BacktestMetrics(self.strategy_returns)
        s = strat_m.summary()

        bench_cagr = 0.0
        bench_s = {}
        if self.benchmark_returns is not None:
            bench_m = BacktestMetrics(self.benchmark_returns)
            bench_s = bench_m.summary()
            bench_cagr = bench_s.get("cagr", 0.0)

        checks = strat_m.check_phase4_targets(bench_cagr)

        width = 58
        print("=" * width)
        print(" Phase 4 백테스트 결과")
        print("=" * width)
        print(f"  기간     : {self.strategy_returns.index.min().date()} ~ "
              f"{self.strategy_returns.index.max().date()}")
        print(f"  거래일수  : {s['n_days']:,}일")
        print()

        header = f"  {'지표':<20} {'전략':>10}  {'SPY':>10}  {'기준':<12}"
        print(header)
        print("-" * width)

        rows = [
            ("총 수익률",      f"{s['total_return']:+.2%}",
             f"{bench_s.get('total_return', 0):+.2%}" if bench_s else "-",    "-"),
            ("CAGR",           f"{s['cagr']:+.2%}",
             f"{bench_cagr:+.2%}" if bench_s else "-",
             f">{bench_cagr:.1%}",
             checks["cagr_vs_benchmark"]["pass"]),
            ("연화 변동성",     f"{s['volatility']:.2%}",
             f"{bench_s.get('volatility', 0):.2%}" if bench_s else "-",       "-"),
            ("샤프 비율",      f"{s['sharpe_ratio']:.3f}",
             f"{bench_s.get('sharpe_ratio', 0):.3f}" if bench_s else "-",
             ">= 1.0",
             checks["sharpe_gte_1"]["pass"]),
            ("최대 낙폭(MDD)", f"{s['max_drawdown']:.2%}",
             f"{bench_s.get('max_drawdown', 0):.2%}" if bench_s else "-",
             ">= -20%",
             checks["mdd_within_20pct"]["pass"]),
            ("승률",           f"{s['win_rate']:.2%}",
             f"{bench_s.get('win_rate', 0):.2%}" if bench_s else "-",
             ">= 50%",
             checks["win_rate_gte_50pct"]["pass"]),
            ("칼마 비율",      f"{s['calmar_ratio']:.3f}",
             f"{bench_s.get('calmar_ratio', 0):.3f}" if bench_s else "-",     "-"),
        ]

        for r in rows:
            name, strat_val, bench_val = r[0], r[1], r[2]
            target = r[3] if len(r) > 3 else ""
            passed = r[4] if len(r) > 4 else None
            status = ("PASS" if passed else "FAIL") if passed is not None else "    "
            print(f"  {name:<20} {strat_val:>10}  {bench_val:>10}  {target:<12} {status}")

        print("=" * width)
        all_pass = all(v["pass"] for v in checks.values())
        print(f"  Phase 4 목표: {'전체 달성!' if all_pass else '일부 미달 (아래 참고)'}")
        print("=" * width)

    def save_json(self, name: str = "backtest_result") -> Path:
        """성과 지표 JSON 저장."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        strat_m = BacktestMetrics(self.strategy_returns)
        result = {
            "strategy": strat_m.summary(),
            "benchmark": (
                BacktestMetrics(self.benchmark_returns).summary()
                if self.benchmark_returns is not None else {}
            ),
            "phase4_checks": strat_m.check_phase4_targets(
                BacktestMetrics(self.benchmark_returns).cagr()
                if self.benchmark_returns is not None else 0.0
            ),
        }
        path = REPORTS_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"백테스트 결과 저장 -> {path}")
        return path

    def save_quantstats_report(self, name: str = "backtest_qs") -> Path | None:
        """Quantstats HTML 리포트 저장."""
        try:
            import quantstats as qs

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = REPORTS_DIR / f"{name}.html"

            qs.extend_pandas()
            strat = self.strategy_returns.copy()
            strat.index = pd.to_datetime(strat.index)

            bench = None
            if self.benchmark_returns is not None:
                bench = self.benchmark_returns.copy()
                bench.index = pd.to_datetime(bench.index)
                # 날짜 정렬
                common = strat.index.intersection(bench.index)
                strat = strat.loc[common]
                bench = bench.loc[common]

            qs.reports.html(
                strat,
                benchmark=bench,
                output=str(path),
                title="ML Quant Strategy vs SPY",
                download_filename=str(path),
            )
            logger.info(f"Quantstats HTML 리포트 저장 -> {path}")
            return path
        except Exception as e:
            logger.warning(f"Quantstats 리포트 생성 실패 (무시): {e}")
            return None
