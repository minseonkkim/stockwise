import Link from "next/link";
import { fetchPredictions, type Prediction } from "@/lib/api";
import SignalBadge from "@/components/SignalBadge";

export const revalidate = 300;

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

/**
 * 포트폴리오 — 최신 5 거래일 이내의 buy 신호 종목을 "현재 보유 포지션"으로 표시.
 * (백테스트 설정: 신호 후 5거래일 보유 → 리밸런싱)
 */
export default async function PortfolioPage() {
  let allPredictions: Prediction[] = [];
  let latestDate = "";
  let error: string | null = null;

  try {
    // 최근 500개 예측을 가져와서 날짜별로 필터
    const data = await fetchPredictions(20);
    latestDate = data.date;
    allPredictions = data.predictions;
  } catch (e) {
    error = e instanceof Error ? e.message : "API 연결 오류";
  }

  const positions = allPredictions.filter((p) => p.signal === "buy");

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">포트폴리오</h1>
          <p className="text-sm text-slate-400 mt-1">
            기준일: {latestDate} &nbsp;·&nbsp; 현재 매수 포지션 (Top-20 신호 기준)
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-500">보유 종목 수</p>
          <p className="text-2xl font-semibold text-emerald-400">{positions.length}</p>
        </div>
      </div>

      {error ? (
        <div className="bg-red-900/20 border border-red-800 rounded-xl p-6">
          <p className="text-red-400">{error}</p>
        </div>
      ) : (
        <>
          <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden mb-6">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 bg-slate-800/80">
                  <th className="text-left px-4 py-3 text-slate-400 font-medium w-12">순위</th>
                  <th className="text-left px-4 py-3 text-slate-400 font-medium">티커</th>
                  <th className="text-right px-4 py-3 text-slate-400 font-medium">예측 확률</th>
                  <th className="text-center px-4 py-3 text-slate-400 font-medium">신호</th>
                  <th className="px-4 py-3"></th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => (
                  <tr
                    key={p.ticker}
                    className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors"
                  >
                    <td className="px-4 py-3 text-slate-500 tabular-nums">{p.rank}</td>
                    <td className="px-4 py-3 font-semibold text-white">{p.ticker}</td>
                    <td className="px-4 py-3 text-right tabular-nums text-emerald-400 font-medium">
                      {pct(p.prob)}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <SignalBadge signal={p.signal} />
                    </td>
                    <td className="px-4 py-3 text-right">
                      <Link
                        href={`/stock/${p.ticker}`}
                        className="text-xs text-slate-400 hover:text-emerald-400 transition-colors"
                      >
                        상세 →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-xs text-slate-500">
            <p>
              <strong className="text-slate-400">포지션 기준:</strong> 백테스트와 동일한 Top-20 방식.
              매수 신호 발생 후 5거래일 보유 → 매주 월요일 리밸런싱.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
