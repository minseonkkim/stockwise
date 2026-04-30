"use client";

import { useEffect, useRef } from "react";
import type { Prediction } from "@/lib/api";

interface Props {
  data: Prediction[];
  ticker: string;
}

export default function ProbChart({ data, ticker }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || data.length === 0) return;

    let chart: ReturnType<typeof createChart> | null = null;

    import("lightweight-charts").then(({ createChart, ColorType }) => {
      if (!ref.current) return;

      chart = createChart(ref.current, {
        layout: {
          background: { type: ColorType.Solid, color: "#1e293b" },
          textColor: "#94a3b8",
        },
        grid: {
          vertLines: { color: "#334155" },
          horzLines: { color: "#334155" },
        },
        width: ref.current.clientWidth,
        height: 280,
        timeScale: { borderColor: "#334155" },
        rightPriceScale: { borderColor: "#334155" },
      });

      const series = chart.addLineSeries({
        color: "#6366f1",
        lineWidth: 2,
        priceFormat: { type: "percent", precision: 1 },
        priceLineVisible: false,
      });

      const sorted = [...data].sort((a, b) => a.date.localeCompare(b.date));
      series.setData(
        sorted.map((p) => ({
          time: p.date as `${number}-${number}-${number}`,
          value: p.prob * 100,
        }))
      );

      // 매수 신호에 마커 추가
      const markers = sorted
        .filter((p) => p.signal === "buy")
        .map((p) => ({
          time: p.date as `${number}-${number}-${number}`,
          position: "belowBar" as const,
          color: "#10b981",
          shape: "arrowUp" as const,
          text: "BUY",
        }));
      series.setMarkers(markers);

      chart.timeScale().fitContent();
    });

    return () => {
      chart?.remove();
    };
  }, [data, ticker]);

  return <div ref={ref} className="w-full rounded-lg overflow-hidden" />;
}
