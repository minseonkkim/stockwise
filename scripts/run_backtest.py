"""Phase 4: 백테스트 실행 CLI.

실행 예:
  python scripts/run_backtest.py                    # 기본 (상위 20개, 주간 리밸런싱)
  python scripts/run_backtest.py --n-top 30         # 상위 30개 종목
  python scripts/run_backtest.py --no-qs            # Quantstats HTML 생략
  python scripts/run_backtest.py --split valid      # 검증 기간 백테스트
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.predictor import generate_predictions
from src.backtest.simulator import VectorizedBacktest
from src.backtest.reporter import BacktestReporter
from src.utils import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4: 백테스트")
    p.add_argument("--model-name",    default="xgb_model",  help="모델 이름")
    p.add_argument("--split",         default="test",        help="test / valid")
    p.add_argument("--n-top",   type=int,   default=20,      help="상위 종목 수 (기본 20)")
    p.add_argument("--cost",    type=float, default=0.001,   help="단방향 거래비용 (기본 0.1%%)")
    p.add_argument("--capital", type=float, default=100_000, help="초기 자본 (기본 $100,000)")
    p.add_argument("--no-qs",  action="store_true",          help="Quantstats HTML 리포트 생략")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Phase 4: 백테스트 시작")
    logger.info("=" * 60)
    logger.info(f"  전략: 상위 {args.n_top}개 종목 / 주간 리밸런싱 / 거래비용 {args.cost:.1%}")
    logger.info(f"  자본: ${args.capital:,.0f} | 데이터: {args.split}")

    # ─── 1. 예측 생성 ─────────────────────────────────────────
    logger.info("[1/4] 모델 예측 생성 중...")
    pred_df = generate_predictions(model_name=args.model_name, split=args.split)

    logger.info(f"  예측 건수: {len(pred_df):,} | "
                f"평균 확률: {pred_df['prob'].mean():.4f} | "
                f"중앙값: {pred_df['prob'].median():.4f}")

    # ─── 2. 백테스트 시뮬레이션 ───────────────────────────────
    logger.info("[2/4] 포트폴리오 시뮬레이션 중...")
    bt = VectorizedBacktest(
        initial_capital=args.capital,
        n_top=args.n_top,
        transaction_cost=args.cost,
    )
    daily_returns = bt.run(pred_df)

    # 주간 요약
    wr = bt.weekly_results
    logger.info(f"  리밸런싱 횟수: {len(wr)}회")
    logger.info(f"  평균 주간 수익률(세전): {wr['gross_return'].mean():+.4f}")
    logger.info(f"  평균 주간 수익률(세후): {wr['net_return'].mean():+.4f}")
    logger.info(f"  평균 턴오버: {wr['turnover_rate'].mean():.2%}")

    # ─── 3. 성과 분석 ─────────────────────────────────────────
    logger.info("[3/4] 성과 분석 중...")
    reporter = BacktestReporter(daily_returns)
    reporter.fetch_benchmark()

    reporter.print_console_report()
    reporter.save_json("backtest_result")

    # ─── 4. 상세 리포트 저장 ──────────────────────────────────
    logger.info("[4/4] 리포트 저장 중...")

    # 주간 결과 CSV
    from pathlib import Path
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    wr_path = reports_dir / "backtest_weekly.csv"
    bt.weekly_results.to_csv(wr_path)
    logger.info(f"  주간 수익률 -> {wr_path}")

    # 일별 수익률 CSV
    dr_path = reports_dir / "backtest_daily_returns.csv"
    daily_returns.to_csv(dr_path, header=True)
    logger.info(f"  일별 수익률 -> {dr_path}")

    if not args.no_qs:
        reporter.save_quantstats_report("backtest_qs")

    logger.info("=" * 60)
    logger.info("Phase 4 완료")
    logger.info("  결과: data/reports/backtest_result.json")
    logger.info("  주간: data/reports/backtest_weekly.csv")
    if not args.no_qs:
        logger.info("  HTML: data/reports/backtest_qs.html")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
