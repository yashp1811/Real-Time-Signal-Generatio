"use client";
import { SignalResult } from "@/lib/types";

const DIR_STYLES: Record<string, { label: string; icon: string; color: string; bg: string }> = {
  up:   { label: "UP",   icon: "↑", color: "#22c55e", bg: "rgba(34,197,94,0.12)"  }, 
  down: { label: "DOWN", icon: "↓", color: "#ef4444", bg: "rgba(239,68,68,0.12)"  },
  flat: { label: "FLAT", icon: "→", color: "#f59e0b", bg: "rgba(245,158,11,0.12)" },
};

function fmtDate(iso?: string): string {
  if (!iso) return "";
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("en-US", {
    weekday: "short", month: "short", day: "numeric",
  });
}

function fmtPrice(p?: number): string {
  if (p == null) return "";
  return p < 10 ? p.toFixed(4) : p.toFixed(2);
}

interface Props {
  signal: SignalResult;
  selected: boolean;
  onClick: () => void;
}

export default function SignalCard({ signal, selected, onClick }: Props) {
  const isLoading = signal.status === "loading";
  const isError   = signal.status === "error";
  const isIdle    = signal.status === "idle";
  const dir       = signal.direction ? DIR_STYLES[signal.direction] : null;

  return (
    <div
      onClick={onClick}
      className="cursor-pointer rounded-xl p-4 transition-all duration-200"
      style={{
        background: selected
          ? "rgba(59,130,246,0.15)"
          : isError
          ? "rgba(239,68,68,0.10)"
          : dir
          ? dir.bg
          : "var(--bg-card)",
        border: `1px solid ${
          selected
            ? "#3b82f6"
            : isError
            ? "#ef4444"
            : dir
            ? dir.color + "55"
            : "var(--border)"
        }`,
      }}
    >
      {/* Row 1: instrument + last price */}
      <div className="flex items-center justify-between mb-2">
        <span className="font-bold text-sm" style={{ color: "var(--text-muted)" }}>
          {signal.instrument}
        </span>
        {signal.last_price != null && (
          <span className="text-xs font-mono font-semibold" style={{ color: "var(--text-primary)" }}>
            {fmtPrice(signal.last_price)}
          </span>
        )}
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center gap-2 py-1">
          <div className="animate-spin w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full" />
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>Generating…</span>
        </div>
      )}

      {/* Error */}
      {isError && (
        <div className="py-1">
          <div className="text-xs font-bold text-red-400">ERROR</div>
          <div className="text-xs mt-0.5 text-red-300 truncate max-w-[140px]" title={signal.error}>
            {signal.error}
          </div>
        </div>
      )}

      {/* Idle */}
      {isIdle && (
        <div className="text-xs py-1" style={{ color: "var(--text-muted)" }}>
          Click to view chart
        </div>
      )}

      {/* Signal result */}
      {!isLoading && !isError && !isIdle && dir && (
        <div>
          {/* Direction badge — biggest element */}
          <div className="text-xl sm:text-2xl font-black leading-tight" style={{ color: dir.color }}>
            {dir.icon} {dir.label}
          </div>

          {/* Confidence */}
          <div className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
            Confidence:{" "}
            <span className="font-bold" style={{ color: dir.color }}>
              {((signal.confidence ?? 0) * 100).toFixed(1)}%
            </span>
          </div>

          {/* Prediction target day — highlighted */}
          {signal.next_trading_day && (
            <div
              className="mt-2 px-2 py-1 rounded text-xs font-semibold text-center"
              style={{ background: dir.color + "22", color: dir.color }}
            >
              Predicting {fmtDate(signal.next_trading_day)}
            </div>
          )}

          {/* Data used */}
          {signal.prediction_date && (
            <div className="text-xs mt-1 text-center" style={{ color: "var(--text-muted)" }}>
              based on data to {fmtDate(signal.prediction_date)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
