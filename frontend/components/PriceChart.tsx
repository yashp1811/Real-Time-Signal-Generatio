"use client";
import { useEffect, useRef } from "react";
import {
  createChart,
  CandlestickSeries,
  createSeriesMarkers,
  ISeriesApi,
  Time,
} from "lightweight-charts";
import { OHLCBar, SignalResult } from "@/lib/types";

interface Props {
  instrument: string;
  bars: OHLCBar[];
  signal?: SignalResult;
}

type MarkerShape = "arrowUp" | "arrowDown" | "circle";
type MarkerPos   = "aboveBar" | "belowBar" | "inBar";

const DIR_MARKER: Record<string, { shape: MarkerShape; color: string; position: MarkerPos }> = {
  up:   { shape: "arrowUp",   color: "#22c55e", position: "belowBar" },
  down: { shape: "arrowDown", color: "#ef4444", position: "aboveBar" },
  flat: { shape: "circle",    color: "#f59e0b", position: "inBar"    },
};

export default function PriceChart({ instrument, bars, signal }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<ReturnType<typeof createChart> | null>(null);
  const seriesRef    = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!containerRef.current || bars.length === 0) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
      seriesRef.current = null;
    }

    const isMobile = containerRef.current.clientWidth < 480;
    const chart = createChart(containerRef.current, {
      width:  containerRef.current.clientWidth,
      height: isMobile ? 240 : 340,
      layout: {
        background: { color: "#1a2035" },
        textColor:  "#94a3b8",
      },
      grid: {
        vertLines: { color: "#2a3550" },
        horzLines: { color: "#2a3550" },
      },
      rightPriceScale: { borderColor: "#2a3550" },
      timeScale:       { borderColor: "#2a3550", timeVisible: true },
    });

    // v5 API: chart.addSeries(SeriesPlugin, options)
    const series = chart.addSeries(CandlestickSeries, {
      upColor:         "#22c55e",
      downColor:       "#ef4444",
      borderUpColor:   "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor:     "#22c55e",
      wickDownColor:   "#ef4444",
    });

    series.setData(bars.map(b => ({ ...b, time: b.time as Time })));

    // Add signal marker via createSeriesMarkers (v5 API)
    if (signal?.direction && DIR_MARKER[signal.direction]) {
      const m      = DIR_MARKER[signal.direction];
      const bar    = bars[bars.length - 1];
      createSeriesMarkers(series, [
        {
          time:     bar.time as Time,
          position: m.position,
          color:    m.color,
          shape:    m.shape,
          text:     `${signal.direction.toUpperCase()} ${((signal.confidence ?? 0) * 100).toFixed(0)}%`,
          size:     2,
        },
      ]);
    }

    chart.timeScale().fitContent();
    chartRef.current  = chart;
    seriesRef.current = series;

    const ro = new ResizeObserver(() => {
      if (containerRef.current && chartRef.current) {
        const w = containerRef.current.clientWidth;
        chartRef.current.applyOptions({
          width:  w,
          height: w < 480 ? 240 : 340,
        });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chartRef.current  = null;
      seriesRef.current = null;
      chart.remove();
    };
  }, [bars, signal]);

  const DIR_COLOR: Record<string, string> = { up: "#22c55e", down: "#ef4444", flat: "#f59e0b" };
  const dirColor = signal?.direction ? DIR_COLOR[signal.direction] : undefined;

  function fmtDate(iso?: string) {
    if (!iso) return "";
    const [y, m, d] = iso.split("-").map(Number);
    return new Date(y, m - 1, d).toLocaleDateString("en-US", {
      weekday: "short", month: "short", day: "numeric", year: "numeric",
    });
  }

  return (
    <div>
      {/* Title row */}
      <div className="flex items-start justify-between mb-3 gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-bold text-lg leading-tight">
            {instrument} — Price History
          </h3>
          {signal?.prediction_date && (
            <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
              Data up to <span style={{ color: "var(--text-primary)" }}>{fmtDate(signal.prediction_date)}</span>
            </p>
          )}
        </div>

        {/* Prediction badge */}
        {signal?.direction && signal?.next_trading_day && (
          <div
            className="flex-shrink-0 rounded-xl px-3 py-2 text-center"
            style={{ background: (dirColor ?? "#3b82f6") + "22", border: `1px solid ${dirColor ?? "#3b82f6"}44` }}
          >
            <div className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>
              Predicting
            </div>
            <div className="text-sm font-black" style={{ color: dirColor }}>
              {signal.direction === "up" ? "↑ UP" : signal.direction === "down" ? "↓ DOWN" : "→ FLAT"}
              {" "}
              <span className="text-xs font-normal">{((signal.confidence ?? 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="text-xs font-semibold mt-0.5" style={{ color: dirColor }}>
              {fmtDate(signal.next_trading_day)}
            </div>
          </div>
        )}
      </div>

      {bars.length === 0 ? (
        <div
          className="flex items-center justify-center h-[240px] sm:h-[340px] text-sm"
          style={{ color: "var(--text-muted)" }}
        >
          {instrument} — loading chart data…
        </div>
      ) : (
        <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />
      )}
    </div>
  );
}
