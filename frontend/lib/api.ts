const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { next: { revalidate: 300 } });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── 타입 정의 ─────────────────────────────────────────────────

export interface Prediction {
  date: string;
  ticker: string;
  prob: number;
  signal: "buy" | "hold";
  rank: number;
}

export interface PredictionsResponse {
  date: string;
  n_top: number;
  total: number;
  predictions: Prediction[];
}

export interface TickerHistoryResponse {
  ticker: string;
  count: number;
  history: Prediction[];
}

export interface BacktestSummary {
  strategy: {
    total_return: number;
    cagr: number;
    volatility: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    calmar_ratio: number;
    n_days: number;
  };
  benchmark: {
    total_return?: number;
    cagr?: number;
    volatility?: number;
    sharpe_ratio?: number;
    max_drawdown?: number;
    win_rate?: number;
  };
  phase4_checks: Record<string, { value: number; target?: number; benchmark?: number; pass: boolean }>;
}

export interface EquityPoint {
  date: string;
  daily_return: number;
  equity: number;
}

export interface EquityResponse {
  count: number;
  equity_curve: EquityPoint[];
}

// ── API 함수 ──────────────────────────────────────────────────

export const fetchPredictions = (nTop = 20) =>
  get<PredictionsResponse>(`/api/predictions?n_top=${nTop}`);

export const fetchTickerHistory = (ticker: string) =>
  get<TickerHistoryResponse>(`/api/predictions/${ticker}`);

export const fetchBacktestSummary = () =>
  get<BacktestSummary>("/api/backtest/summary");

export const fetchEquityCurve = () =>
  get<EquityResponse>("/api/backtest/equity");

export const fetchFeatures = (ticker: string) =>
  get<Record<string, unknown>>(`/api/features/${ticker}`);
