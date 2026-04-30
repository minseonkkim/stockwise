import Link from "next/link";
import { fetchPredictions, type Prediction } from "@/lib/api";
import SignalBadge from "@/components/SignalBadge";

export const revalidate = 300;

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export default async function DashboardPage() {
  let data: Awaited<ReturnType<typeof fetchPredictions>> | null = null;
  let error: string | null = null;

  try {
    data = await fetchPredictions(20);
  } catch (e) {
    error = e instanceof Error ? e.message : "API 연결 오류";
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">오늘의 매수 신호</h1>
          {data && (
            <p className="text-sm text-slate-400 mt-1">
              기준일: {data.date} &nbsp;·&nbsp; 상위 {data.n_top}개 종목 (prob 순위 기준)
            </p>
          )}
        </div>
        {data && (
          <div className="text-right">
            <p className="text-xs text-slate-500">전체 종목</p>
            <p className="text-xl font-semibold text-emerald-400">{data.total}</p>
          </div>
        )}
      </div>

      {error ? (
        <ApiError message={error} />
      ) : data ? (
        <BuySignalsTable predictions={data.predictions} />
      ) : null}
    </div>
  );
}

function BuySignalsTable({ predictions }: { predictions: Prediction[] }) {
  const buys = predictions.filter((p) => p.signal === "buy");

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
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
          {buys.map((p) => (
            <tr
              key={p.ticker}
              className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors"
            >
              <td className="px-4 py-3 text-slate-500 tabular-nums">{p.rank}</td>
              <td className="px-4 py-3 font-semibold text-white">{p.ticker}</td>
              <td className="px-4 py-3 text-right tabular-nums">
                <span className="text-emerald-400 font-medium">
                  {pct(p.prob)}
                </span>
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
  );
}

function ApiError({ message }: { message: string }) {
  return (
    <div className="bg-red-900/20 border border-red-800 rounded-xl p-6 text-center">
      <p className="text-red-400 font-medium mb-2">API 연결 실패</p>
      <p className="text-sm text-red-300/70">{message}</p>
      <p className="text-xs text-slate-500 mt-4">
        FastAPI 서버가 실행 중인지 확인하세요:{" "}
        <code className="text-slate-400">python scripts/run_api.py</code>
      </p>
    </div>
  );
}
