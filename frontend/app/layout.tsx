import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "StockWise — ML 퀀트 투자 플랫폼",
  description: "XGBoost 기반 S&P 500 매수 신호 대시보드",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body className="min-h-screen bg-slate-900 text-slate-100">
        <nav className="border-b border-slate-700 bg-slate-800/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 h-14 flex items-center gap-6">
            <Link href="/" className="font-bold text-lg text-emerald-400 tracking-tight">
              StockWise
            </Link>
            <Link href="/" className="text-sm text-slate-300 hover:text-white transition-colors">
              대시보드
            </Link>
            <Link href="/backtest" className="text-sm text-slate-300 hover:text-white transition-colors">
              백테스트
            </Link>
            <Link href="/portfolio" className="text-sm text-slate-300 hover:text-white transition-colors">
              포트폴리오
            </Link>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
