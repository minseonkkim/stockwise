import { fetchBacktestSummary, fetchEquityCurve } from "@/lib/api";
import StatsCard from "@/components/StatsCard";
import EquityChart from "@/components/EquityChart";

export const revalidate = 3600;

function fmt(v: number, type: "pct" | "num") {
  if (type === "pct") return `${(v * 100).toFixed(2)}%`;
  return v.toFixed(3);
}

export default async function BacktestPage() {
  let summary: Awaited<ReturnType<typeof fetchBacktestSummary>> | null = null;
  let equity: Awaited<ReturnType<typeof fetchEquityCurve>> | null = null;
  let error: string | null = null;

  try {
    [summary, equity] = await Promise.all([fetchBacktestSummary(), fetchEquityCurve()]);
  } catch (e) {
    error = e instanceof Error ? e.message : "API 연결 오류";
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">백테스트 리포트</h1>

      {error ? (
        <div className="bg-red-900/20 border border-red-800 rounded-xl p-6">
          <p className="text-red-400">{error}</p>
          <p className="text-xs text-slate-500 mt-2">
            <code>python scripts/run_backtest.py</code> 와{" "}
            <code>python scripts/run_api.py</code> 를 실행했는지 확인하세요.
          </p>
        </div>
      ) : summary ? (
        <>
          {/* 성과 지표 */}
          <section className="mb-8">
            <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-4">
              핵심 성과 지표
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
              <StatsCard
                label="총 수익률"
                value={fmt(summary.strategy.total_return, "pct")}
              />
              <StatsCard
                label="CAGR"
                value={fmt(summary.strategy.cagr, "pct")}
                sub={`SPY: ${fmt(summary.benchmark?.cagr ?? 0, "pct")}`}
                pass={summary.phase4_checks?.cagr_vs_benchmark?.pass}
              />
              <StatsCard
                label="샤프 비율"
                value={fmt(summary.strategy.sharpe_ratio, "num")}
                sub="목표: ≥ 1.0"
                pass={summary.phase4_checks?.sharpe_gte_1?.pass}
              />
              <StatsCard
                label="최대 낙폭 (MDD)"
                value={fmt(summary.strategy.max_drawdown, "pct")}
                sub="목표: ≥ -20%"
                pass={summary.phase4_checks?.mdd_within_20pct?.pass}
              />
              <StatsCard
                label="승률"
                value={fmt(summary.strategy.win_rate, "pct")}
                sub="목표: ≥ 50%"
                pass={summary.phase4_checks?.win_rate_gte_50pct?.pass}
              />
              <StatsCard
                label="연화 변동성"
                value={fmt(summary.strategy.volatility, "pct")}
              />
              <StatsCard
                label="칼마 비율"
                value={fmt(summary.strategy.calmar_ratio, "num")}
              />
              <StatsCard
                label="거래일 수"
                value={summary.strategy.n_days.toLocaleString()}
                sub="일"
              />
            </div>
          </section>

          {/* 수익 곡선 */}
          {equity && equity.equity_curve.length > 0 && (
            <section>
              <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-4">
                누적 수익 곡선 (equity curve)
              </h2>
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                <EquityChart data={equity.equity_curve} />
              </div>
            </section>
          )}
        </>
      ) : null}
    </div>
  );
}
