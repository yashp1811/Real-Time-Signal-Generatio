"""
Benchmark script — measures p50/p95/p99 latency and demonstrates that
concurrent batch generation is faster than a sequential loop for the same
batch of instruments within a single request.

Usage:
  1. Start the API:  uvicorn main:app --host 0.0.0.0 --port 8000
  2. Run benchmark:  python benchmark.py

What it measures
----------------
The spec asks for a demonstration that concurrent signal generation across N
instruments is faster than processing them one-by-one in a sequential loop.

  mode=concurrent  ->  all N instruments dispatched to ThreadPoolExecutor
                       simultaneously; total time ≈ max(individual times)
  mode=sequential  ->  instruments processed one-by-one in a loop;
                       total time ≈ sum(individual times)

Both modes are tested via the same /signals/generate endpoint using the
?mode= parameter added for this benchmark.  The server runs on the same
machine (loopback), so network latency is negligible.
"""

from __future__ import annotations

import asyncio
import csv
import statistics
import time
from pathlib import Path

import aiohttp

BASE_URL    = "http://127.0.0.1:8000"
DATA_DIR    = Path(__file__).parent.parent / "data"
INSTRUMENTS = ["EURUSD", "GBPUSD", "AAPL", "MSFT", "GLD", "USO", "SPX", "NDX"]
N_WARMUP    = 3     # warm-up requests (not counted)
N_TIMED     = 20    # timed requests per mode


# ── helpers ──────────────────────────────────────────────────────────────────

def read_csv_as_dicts(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "date":   row.get("date") or row.get("Date"),
                "open":   float(row.get("open")   or row.get("Open")   or 0),
                "high":   float(row.get("high")   or row.get("High")   or 0),
                "low":    float(row.get("low")    or row.get("Low")    or 0),
                "close":  float(row.get("close")  or row.get("Close")  or 0),
                "volume": float(row.get("volume") or row.get("Volume") or 0),
            })
    return rows


def pct(data: list[float], p: float) -> float:
    s = sorted(data)
    return s[max(0, int(len(s) * p / 100) - 1)]


# ── load instruments ─────────────────────────────────────────────────────────

async def load_all(session: aiohttp.ClientSession) -> list[str]:
    """
    Load instruments for benchmarking.
    Strategy:
      1. Try POST /instruments/autoload (fetches live data from Yahoo Finance).
      2. If autoload loads nothing or fails, fall back to local CSV files in DATA_DIR.
    """
    # ── Attempt 1: autoload via Yahoo Finance ─────────────────────────────────
    try:
        async with session.post(
            f"{BASE_URL}/instruments/autoload?instruments=ALL"
        ) as resp:
            result = await resp.json()
        loaded = result.get("loaded", [])
        if loaded:
            print(f"  Autoload (Yahoo Finance): {loaded}")
            if result.get("errors"):
                print(f"  Autoload errors: {result['errors']}")
            return loaded
        print("  Autoload returned 0 instruments — falling back to local CSVs")
    except Exception as exc:
        print(f"  Autoload failed ({exc}) — falling back to local CSVs")

    # ── Attempt 2: local CSV files in Real_Time_Signal_Generation/data/ ───────
    instruments = []
    for name in INSTRUMENTS:
        csv_path = DATA_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"  SKIP {name}: not found at {csv_path}")
            continue
        data = read_csv_as_dicts(csv_path)
        instruments.append({"name": name, "data": data[-500:]})

    if not instruments:
        return []

    async with session.post(
        f"{BASE_URL}/instruments/load", json={"instruments": instruments}
    ) as resp:
        result = await resp.json()
    print(f"  Loaded from CSV: {result['loaded']}")
    if result.get("errors"):
        print(f"  Load errors: {result['errors']}")
    return result["loaded"]


# ── single timed request ─────────────────────────────────────────────────────

async def timed_request(
    session: aiohttp.ClientSession,
    instruments: list[str],
    mode: str,
) -> tuple[float, float]:
    """Returns (client-side latency ms, server-reported batch_latency_ms)."""
    payload = {"instruments": instruments, "mode": mode}
    t0 = time.perf_counter()
    async with session.post(f"{BASE_URL}/signals/generate", json=payload) as resp:
        data = await resp.json()
    client_ms = (time.perf_counter() - t0) * 1000
    server_ms = data.get("batch_latency_ms", client_ms)
    return client_ms, server_ms


# ── run N requests (serial at client) ────────────────────────────────────────

async def run_series(
    session: aiohttp.ClientSession,
    instruments: list[str],
    mode: str,
    n: int,
) -> tuple[list[float], list[float]]:
    client_lats, server_lats = [], []
    for _ in range(n):
        c, s = await timed_request(session, instruments, mode)
        client_lats.append(c)
        server_lats.append(s)
    return client_lats, server_lats


# ── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # Health check
        async with session.get(f"{BASE_URL}/health") as resp:
            h = await resp.json()
        print(f"\nServer: {h['status']} | models={h['models_loaded']} | "
              f"registered={h['instruments_registered']}")

        # Load instruments
        print("\nLoading instruments ...")
        active = await load_all(session)
        if not active:
            print(
                "No instruments loaded.\n"
                "  • Autoload requires internet access (Yahoo Finance).\n"
                f"  • CSV fallback looks for files in: {DATA_DIR}\n"
                "    Place EURUSD.csv, AAPL.csv, … there to run offline."
            )
            return

        print(f"\nBatch size: {len(active)} instruments  |  {N_TIMED} timed requests each")
        print("=" * 65)

        # ── Warm-up ──────────────────────────────────────────────────────────
        print(f"\nWarming up ({N_WARMUP} requests, not counted) ...")
        for mode in ("concurrent", "sequential"):
            for _ in range(N_WARMUP):
                await timed_request(session, active, mode)
        print("  Done.")

        # ── Concurrent mode ──────────────────────────────────────────────────
        print(f"\n[concurrent]  ThreadPoolExecutor — all {len(active)} instruments in parallel")
        print(f"  ({N_TIMED} requests, sent one-at-a-time so each measures a cold batch)")
        c_cli, c_srv = await run_series(session, active, "concurrent", N_TIMED)
        print(f"  server-side  p50={pct(c_srv,50):6.1f}ms  "
              f"p95={pct(c_srv,95):6.1f}ms  p99={pct(c_srv,99):6.1f}ms  "
              f"mean={statistics.mean(c_srv):6.1f}ms")
        print(f"  client-side  p50={pct(c_cli,50):6.1f}ms  "
              f"p95={pct(c_cli,95):6.1f}ms  p99={pct(c_cli,99):6.1f}ms")

        # ── Sequential mode ──────────────────────────────────────────────────
        print(f"\n[sequential]  naive loop — {len(active)} instruments one-by-one")
        s_cli, s_srv = await run_series(session, active, "sequential", N_TIMED)
        print(f"  server-side  p50={pct(s_srv,50):6.1f}ms  "
              f"p95={pct(s_srv,95):6.1f}ms  p99={pct(s_srv,99):6.1f}ms  "
              f"mean={statistics.mean(s_srv):6.1f}ms")
        print(f"  client-side  p50={pct(s_cli,50):6.1f}ms  "
              f"p95={pct(s_cli,95):6.1f}ms  p99={pct(s_cli,99):6.1f}ms")

        # ── Summary ──────────────────────────────────────────────────────────
        speedup_p50  = pct(s_srv, 50)  / pct(c_srv, 50)
        speedup_mean = statistics.mean(s_srv) / statistics.mean(c_srv)
        tput_con = N_TIMED / (sum(c_cli) / 1000)
        tput_seq = N_TIMED / (sum(s_cli) / 1000)

        print(f"\n{'=' * 65}")
        print(f"  Batch size         : {len(active)} instruments")
        print(f"  Speedup (p50)      : {speedup_p50:.1f}x  "
              f"(sequential {pct(s_srv,50):.1f}ms -> concurrent {pct(c_srv,50):.1f}ms)")
        print(f"  Speedup (mean)     : {speedup_mean:.1f}x")
        print(f"  Throughput conc.   : {tput_con:.1f} req/s")
        print(f"  Throughput seq.    : {tput_seq:.1f} req/s")
        print("=" * 65)
        print()
        print("Hardware note: record CPU model + core count for README.")
        print("Concurrency level for this run: 8 worker threads (ThreadPoolExecutor).")


if __name__ == "__main__":
    asyncio.run(main())
