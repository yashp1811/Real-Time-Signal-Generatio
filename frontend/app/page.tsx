"use client";
import { useState, useCallback, useRef, useEffect } from "react";
import dynamic from "next/dynamic";
import SignalCard from "@/components/SignalCard";
import { SignalResult, OHLCBar } from "@/lib/types";

const PriceChart = dynamic(() => import("@/components/PriceChart"), { ssr: false });

const API = "http://localhost:8000";
const ALL_INSTRUMENTS = ["EURUSD", "GBPUSD", "AAPL", "MSFT", "GLD", "USO", "SPX", "NDX"];

type Phase = "idle" | "initializing" | "streaming" | "done" | "error";

export default function Dashboard() {
  const [selected,    setSelected]    = useState<Set<string>>(new Set(ALL_INSTRUMENTS));
  const [signals,     setSignals]     = useState<Record<string, SignalResult>>({});
  const [activeInst,  setActiveInst]  = useState<string>("EURUSD");
  const [chartData,   setChartData]   = useState<Record<string, OHLCBar[]>>({});
  const [phase,       setPhase]       = useState<Phase>("idle");
  const [statusMsg,   setStatusMsg]   = useState("Click 'Load & Generate' to begin");
  const esRef = useRef<EventSource | null>(null);

  const completed = Object.values(signals).filter(s => s.status !== "loading").length;
  const total     = Object.keys(signals).length;
  const chartBars = chartData[activeInst] ?? [];

  // ── Fetch a single chart ───────────────────────────────────────────────────
  const fetchChart = useCallback(async (name: string) => {
    try {
      const res = await fetch(`${API}/instruments/${name}/history?limit=250`);
      if (!res.ok) return;
      const data = await res.json();
      setChartData(prev => ({ ...prev, [name]: data.data ?? [] }));
    } catch { /* ignore */ }
  }, []);

  // Refresh active chart when instrument selection changes
  useEffect(() => {
    if (chartData[activeInst]) return;
    // Only fetch if we have it in the backend (loaded phase)
    if (phase !== "idle" && phase !== "error") fetchChart(activeInst);
  }, [activeInst, phase, chartData, fetchChart]);

  // ── Core: stream signals for a list of instruments ────────────────────────
  const startStreaming = useCallback((names: string[]) => {
    esRef.current?.close();

    // Set all to loading
    const init: Record<string, SignalResult> = {};
    names.forEach(n => { init[n] = { instrument: n, status: "loading" }; });
    setSignals(init);
    setPhase("streaming");
    setStatusMsg(`Generating signals for ${names.length} instruments…`);

    const es = new EventSource(`${API}/signals/stream?instruments=${names.join(",")}`);
    esRef.current = es;
    let count = 0;

    es.onmessage = (ev) => {
      const data: SignalResult & { type?: string } = JSON.parse(ev.data);

      if (data.type === "done") {
        es.close();
        setPhase("done");
        setStatusMsg(`Done — ${count}/${names.length} signals generated`);
        return;
      }

      count++;
      setSignals(prev => ({ ...prev, [data.instrument]: data }));

      const ntd = data.next_trading_day ?? "";
      setStatusMsg(
        `${count}/${names.length} — ${data.instrument}: ${(data.direction ?? "error").toUpperCase()}` +
        (ntd ? ` → predicts ${ntd}` : "")
      );

      // Refresh chart for this instrument so the marker appears
      fetchChart(data.instrument);
    };

    es.onerror = () => {
      es.close();
      setPhase("error");
      setStatusMsg("SSE error — check backend on port 8000");
    };
  }, [fetchChart]);

  // ── Load live data + auto-generate ────────────────────────────────────────
  const handleLoadAndGenerate = async () => {
    setPhase("initializing");
    setStatusMsg("Fetching live data from Yahoo Finance…");
    setSignals({});
    setChartData({});

    try {
      const res  = await fetch(`${API}/instruments/autoload?instruments=ALL`, { method: "POST" });
      const data = await res.json();
      if (data.loaded.length === 0) throw new Error("No instruments loaded");

      setStatusMsg(`Loaded ${data.loaded.length} instruments — fetching charts & generating signals…`);

      const names = (data.loaded as string[]).filter(n => selected.has(n));

      // Pre-fetch all charts in parallel, then auto-start streaming
      await Promise.all(data.loaded.map((n: string) => fetchChart(n)));

      // Auto-start signal generation immediately
      startStreaming(names.length > 0 ? names : data.loaded);

      if (Object.keys(data.errors ?? {}).length > 0) {
        console.warn("Load errors:", data.errors);
      }
    } catch (err) {
      setPhase("error");
      setStatusMsg(`Failed: ${err}`);
    }
  };

  // ── Re-generate with current selection ────────────────────────────────────
  const handleRegenerate = () => {
    const names = ALL_INSTRUMENTS.filter(n => selected.has(n));
    if (names.length > 0) startStreaming(names);
  };

  const toggleInst = (name: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  };

  const isLoading   = phase === "initializing" || phase === "streaming";
  const hasLoaded   = phase === "done" || phase === "streaming";

  return (
    <div className="min-h-screen p-4 md:p-6" style={{ background: "var(--bg-primary)" }}>
      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <header className="flex flex-wrap items-center justify-between gap-3 mb-6">
        <div>
          <h1 className="text-2xl font-black tracking-tight">Signal Dashboard</h1>
          <p className="text-sm mt-0.5" style={{ color: "var(--text-muted)" }}>
            LGBM + Transformer Ensemble · Live Yahoo Finance · No Holidays
          </p>
        </div>
        <div
          className="flex items-center gap-2 text-sm px-3 py-1.5 rounded-full"
          style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
        >
          <span className={`w-2 h-2 rounded-full ${
            phase === "streaming"    ? "bg-green-400 animate-pulse" :
            phase === "done"         ? "bg-green-400" :
            phase === "initializing" ? "bg-yellow-400 animate-pulse" :
            phase === "error"        ? "bg-red-400" : "bg-gray-500"
          }`} />
          <span style={{ color: "var(--text-muted)" }}>
            {phase === "streaming"    ? "Generating…" :
             phase === "done"         ? "Ready" :
             phase === "initializing" ? "Loading" :
             phase === "error"        ? "Error" : "Idle"}
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* ── Left panel ──────────────────────────────────────────────────── */}
        <div className="lg:col-span-1 space-y-4">
          <div
            className="rounded-xl p-4"
            style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
          >
            <div className="text-xs font-bold uppercase mb-3" style={{ color: "var(--text-muted)" }}>
              Instruments
            </div>
            <div className="grid grid-cols-2 gap-2 mb-4">
              {ALL_INSTRUMENTS.map(name => (
                <label key={name} className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={selected.has(name)}
                    onChange={() => toggleInst(name)}
                    disabled={isLoading}
                    className="accent-blue-500"
                  />
                  <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                    {name}
                  </span>
                </label>
              ))}
            </div>

            {/* Load + auto-generate */}
            {(phase === "idle" || phase === "error") && (
              <button
                onClick={handleLoadAndGenerate}
                className="w-full py-2.5 rounded-lg font-bold text-sm"
                style={{ background: "#3b82f6", color: "white", cursor: "pointer" }}
              >
                Load &amp; Generate Signals
              </button>
            )}

            {/* Re-generate (after first load) */}
            {(phase === "done") && (
              <button
                onClick={handleRegenerate}
                disabled={selected.size === 0}
                className="w-full py-2.5 rounded-lg font-bold text-sm"
                style={{ background: "#22c55e", color: "white", cursor: "pointer" }}
              >
                Regenerate Signals
              </button>
            )}

            {/* Loading state */}
            {isLoading && (
              <div
                className="w-full py-2.5 rounded-lg font-bold text-sm text-center"
                style={{ background: "#1e293b", color: "#64748b" }}
              >
                {phase === "initializing" ? "Loading Yahoo Finance…" : "Streaming…"}
              </div>
            )}
          </div>

          {/* Status */}
          <div
            className="rounded-xl p-4 text-sm"
            style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
          >
            <div className="text-xs font-bold uppercase mb-2" style={{ color: "var(--text-muted)" }}>
              Status
            </div>
            <div style={{ color: "var(--text-primary)" }}>{statusMsg}</div>

            {total > 0 && phase === "streaming" && (
              <div className="mt-3">
                <div className="flex justify-between text-xs mb-1" style={{ color: "var(--text-muted)" }}>
                  <span>Progress</span><span>{completed}/{total}</span>
                </div>
                <div className="w-full h-1.5 rounded-full" style={{ background: "var(--border)" }}>
                  <div
                    className="h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${(completed / total) * 100}%`, background: "#3b82f6" }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Chart panel ─────────────────────────────────────────────────── */}
        <div
          className="lg:col-span-2 rounded-xl p-4"
          style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
        >
          <PriceChart
            instrument={activeInst}
            bars={chartBars}
            signal={signals[activeInst]}
          />
        </div>
      </div>

      {/* ── Signal cards ─────────────────────────────────────────────────────── */}
      {total > 0 && (
        <div className="mt-4">
          <div className="text-xs font-bold uppercase mb-3" style={{ color: "var(--text-muted)" }}>
            Signals — click a card to view chart
          </div>
          <div className="signal-card-grid">
            {ALL_INSTRUMENTS.filter(n => signals[n]).map(n => (
              <SignalCard
                key={n}
                signal={signals[n]}
                selected={n === activeInst}
                onClick={() => setActiveInst(n)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
