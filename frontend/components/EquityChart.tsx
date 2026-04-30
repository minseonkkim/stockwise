"use client";

import { useEffect, useRef } from "react";
import type { EquityPoint } from "@/lib/api";

interface Props {
  data: EquityPoint[];
}

export default function EquityChart({ data }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || data.length === 0) return;

    let chart: ReturnType<typeof createChart> | null = null;

    import("lightweight-charts").then(({ createChart, ColorType, LineStyle }) => {
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
        height: 320,
        timeScale: { borderColor: "#334155" },
        rightPriceScale: { borderColor: "#334155" },
      });

      const series = chart.addLineSeries({
        color: "#10b981",
        lineWidth: 2,
        priceLineVisible: false,
      });

      series.setData(
        data.map((p) => ({
          time: p.date as `${number}-${number}-${number}`,
          value: p.equity,
        }))
      );

      chart.timeScale().fitContent();
    });

    return () => {
      chart?.remove();
    };
  }, [data]);

  return <div ref={ref} className="w-full rounded-lg overflow-hidden" />;
}
