import Link from "next/link";
import { fetchTickerHistory, fetchFeatures, type Prediction } from "@/lib/api";
import SignalBadge from "@/components/SignalBadge";
import ProbChart from "@/components/ProbChart";

export const revalidate = 300;

interface Props {
  params: { ticker: string };
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export default async function StockPage({ params }: Props) {
  const ticker = params.ticker.toUpperCase();
  let history: Prediction[] = [];
  let features: Record<string, unknown> = {};
  let error: string | null = null;

  try {
    const [histRes, featRes] = await Promise.allSettled([
      fetchTickerHistory(ticker),
      fetchFeatures(ticker),
    ]);
    if (histRes.status === "fulfilled") history = histRes.value.history;
    if (featRes.status === "fulfilled") features = featRes.value;
    if (histRes.status === "rejected") throw new Error(histRes.reason?.message);
  } catch (e) {
    error = e instanceof Error ? e.message : "API 연결 오류";
  }

  const latest = history[0];

  return (
    <div>
      {/* 헤더 */}
      <div className="flex items-center gap-3 mb-6">
        <Link href="/" className="text-slate-500 hover:text-white transition-colors text-sm">
          ← 대시보드
        </Link>
        <span className="text-slate-700">|</span>
        <h1 className="text-2xl font-bold">{ticker}</h1>
        {latest && <SignalBadge signal={latest.signal} />}
      </div>

      {error ? (
        <div className="bg-red-900/20 border border-red-800 rounded-xl p-6">
          <p className="text-red-400">{error}</p>
        </div>
      ) : (
        <>
          {/* 최신 예측 요약 */}
          {latest && (
            <div className="grid grid-cols-3 gap-4 mb-8">
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                <p className="text-xs text-slate-400 mb-1">기준일</p>
                <p className="text-lg font-semibold">{latest.date}</p>
              </div>
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                <p className="text-xs text-slate-400 mb-1">예측 확률 (5일 후 +2% 이상)</p>
                <p className="text-lg font-semibold text-emerald-400">{pct(latest.prob)}</p>
              </div>
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                <p className="text-xs text-slate-400 mb-1">전체 순위</p>
                <p className="text-lg font-semibold">#{latest.rank}</p>
              </div>
            </div>
          )}

          {/* 확률 추이 차트 */}
          {history.length > 0 && (
            <section className="mb-8">
              <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-4">
                예측 확률 추이 (↑ = BUY 신호)
              </h2>
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                <ProbChart data={[...history].reverse()} ticker={ticker} />
              </div>
            </section>
          )}

          {/* 피처 현황 */}
          {Object.keys(features).length > 0 && (
            <section className="mb-8">
              <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-4">
                최신 피처 현황
              </h2>
              <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 bg-slate-800/80">
                      <th className="text-left px-4 py-2 text-slate-400 font-medium">피처</th>
                      <th className="text-right px-4 py-2 text-slate-400 font-medium">값</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(features)
                      .filter(([k]) => !["ticker"].includes(k))
                      .map(([k, v]) => (
                        <tr key={k} className="border-b border-slate-700/50 hover:bg-slate-700/20">
                          <td className="px-4 py-2 text-slate-300 font-mono text-xs">{k}</td>
                          <td className="px-4 py-2 text-right tabular-nums text-slate-200 text-xs">
                            {v === null ? "—" : String(v)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* 예측 히스토리 테이블 */}
          {history.length > 0 && (
            <section>
              <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-4">
                예측 히스토리 (최근 {Math.min(history.length, 50)}일)
              </h2>
              <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 bg-slate-800/80">
                      <th className="text-left px-4 py-2 text-slate-400 font-medium">날짜</th>
                      <th className="text-right px-4 py-2 text-slate-400 font-medium">확률</th>
                      <th className="text-center px-4 py-2 text-slate-400 font-medium">신호</th>
                      <th className="text-right px-4 py-2 text-slate-400 font-medium">순위</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.slice(0, 50).map((p) => (
                      <tr
                        key={p.date}
                        className="border-b border-slate-700/50 hover:bg-slate-700/20"
                      >
                        <td className="px-4 py-2 text-slate-300">{p.date}</td>
                        <td className="px-4 py-2 text-right tabular-nums text-slate-200">
                          {pct(p.prob)}
                        </td>
                        <td className="px-4 py-2 text-center">
                          <SignalBadge signal={p.signal} />
                        </td>
                        <td className="px-4 py-2 text-right text-slate-400">#{p.rank}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
