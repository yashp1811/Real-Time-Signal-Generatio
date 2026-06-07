"""
Part 2 — Concurrent FastAPI Backend
Signal Generation Service v2

Endpoints:
  POST /instruments/load        register instruments + historical data
  POST /signals/generate        generate signals for a batch (concurrent)
  GET  /signals/{job_id}        retrieve a previous generation result
  GET  /health                  readiness check

Concurrency mechanism:
  Signal inference is CPU-bound (NumPy + PyTorch arithmetic).  asyncio alone
  cannot parallelize CPU work — it only interleaves coroutines on one thread.
  Each instrument is dispatched to a ThreadPoolExecutor via
  asyncio.run_in_executor(); NumPy and PyTorch release the GIL during their
  core computations, so threads run truly in parallel rather than taking turns.
  asyncio.gather() then awaits all threads concurrently so the batch completes
  in max(individual latencies) rather than sum(individual latencies).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from predictor import Predictor
from schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    LoadRequest,
    LoadResponse,
    SignalResult,
)

# Path to the CSV data files (fallback when Yahoo Finance is unavailable)
DATA_DIR = Path(__file__).parent.parent / "data"

# Yahoo Finance ticker symbols for each instrument
TICKERS: Dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "SPX":    "^GSPC",
    "NDX":    "^NDX",
    "GLD":    "GLD",
    "USO":    "USO",
    "AAPL":   "AAPL",
    "MSFT":   "MSFT",
}

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Real-Time Signal Generation Service",
    version="2.0",
    description="Concurrent FastAPI backend for directional signal generation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (loaded once at startup) ────────────────────────────────────

predictor = Predictor()
executor  = ThreadPoolExecutor(max_workers=8)   # 8 threads ≈ 8 instruments in parallel

# In-memory stores (use Redis in production)
registry: Dict[str, dict] = {}    # name → {raw, tab, seq}
jobs:     Dict[str, GenerateResponse] = {}


# ── Lifecycle ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    predictor.load_models()
    print("[startup] Signal service ready.")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=len(predictor.models),
        instruments_registered=len(registry),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _fetch_live_df(name: str) -> pd.DataFrame:
    """Fetch 3 years of daily OHLCV from Yahoo Finance; fall back to local CSV."""
    ticker_sym = TICKERS.get(name, name)
    try:
        raw = yf.Ticker(ticker_sym).history(period="3y", interval="1d", auto_adjust=True)
        if raw.empty:
            raise ValueError("Empty response from Yahoo Finance")
        # yfinance returns timezone-aware index — strip tz for consistency
        raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
        raw.index.name = "date"
        raw.columns = [c.lower() for c in raw.columns]
        # Keep only OHLCV columns
        raw = raw[["open", "high", "low", "close", "volume"]].dropna()
        return raw
    except Exception as yf_err:
        # Fall back to local CSV
        csv_path = DATA_DIR / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Yahoo Finance failed ({yf_err}) and no local CSV at {csv_path}"
            )
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        df.columns = [c.lower() for c in df.columns]
        return df


def _register(name: str, df: pd.DataFrame) -> None:
    """Build features and store in registry."""
    tab_df   = predictor.build_tab_features(df)
    seq_df   = predictor.build_seq_features_df(df)
    tab_feat = [f for f in predictor.tab_feat if f in tab_df.columns]
    tab_vals = tab_df[tab_feat].values
    cut      = max(int(len(tab_vals) * 0.8), 30)
    mu_t     = tab_vals[:cut].mean(axis=0)
    std_t    = tab_vals[:cut].std(axis=0) + 1e-8
    registry[name] = {"raw": df, "tab": tab_df, "seq": seq_df, "mu_t": mu_t, "std_t": std_t}


@app.post("/instruments/autoload", response_model=LoadResponse, tags=["instruments"])
async def autoload_instruments(
    instruments: str = Query("ALL", description="Comma-separated names, or ALL")
) -> LoadResponse:
    """
    Fetch live OHLCV data from Yahoo Finance and register instruments.
    Falls back to local CSV files when Yahoo Finance is unavailable.
    """
    all_names = ["AAPL", "EURUSD", "GBPUSD", "GLD", "MSFT", "NDX", "SPX", "USO"]
    names = all_names if instruments.upper() == "ALL" else [
        n.strip().upper() for n in instruments.split(",") if n.strip()
    ]

    loaded: list[str] = []
    errors: dict[str, str] = {}

    loop = asyncio.get_event_loop()
    for name in names:
        try:
            df = await loop.run_in_executor(executor, _fetch_live_df, name)
            _register(name, df)
            loaded.append(name)
            print(f"[autoload] {name}: {len(df)} bars, last={df.index[-1].date()}")
        except Exception as exc:
            errors[name] = str(exc)
            print(f"[autoload] {name} FAILED: {exc}")

    return LoadResponse(loaded=loaded, errors=errors)


@app.post("/instruments/load", response_model=LoadResponse, tags=["instruments"])
async def load_instruments(req: LoadRequest) -> LoadResponse:
    """
    Register one or more instruments with their OHLCV history.
    Features are computed once and cached; subsequent /signals/generate calls
    use the cached features directly.
    """
    loaded: list[str] = []
    errors: dict[str, str] = {}

    for inst in req.instruments:
        try:
            rows = [r.model_dump() for r in inst.data]
            df   = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df.columns = [c.lower() for c in df.columns]
            for col in ("open", "high", "low", "close"):
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            tab_df = predictor.build_tab_features(df)
            seq_df = predictor.build_seq_features_df(df)

            # Compute tabular normalization stats from the first 80% of data
            # (approximates training-set stats; LGBM was trained on normalized features)
            tab_feat = [f for f in predictor.tab_feat if f in tab_df.columns]
            tab_vals = tab_df[tab_feat].values
            cut      = max(int(len(tab_vals) * 0.8), 30)
            mu_t     = tab_vals[:cut].mean(axis=0)
            std_t    = tab_vals[:cut].std(axis=0) + 1e-8

            name = inst.name.upper()
            registry[name] = {
                "raw": df, "tab": tab_df, "seq": seq_df,
                "mu_t": mu_t, "std_t": std_t,
            }
            loaded.append(name)
        except Exception as exc:
            errors[inst.name] = str(exc)

    return LoadResponse(loaded=loaded, errors=errors)


@app.post("/signals/generate", response_model=GenerateResponse, tags=["signals"])
async def generate_signals(req: GenerateRequest) -> GenerateResponse:
    """
    Generate directional signals for a batch of instruments concurrently.

    Each instrument's inference runs in a separate thread (ThreadPoolExecutor)
    so a batch of 8 instruments takes ≈ max(8 individual times) instead of
    ≈ sum(8 individual times).  Per-instrument errors are caught and returned
    as error entries — the rest of the batch always completes.
    """
    names = [n.upper() for n in req.instruments]

    # Input validation
    if not names:
        raise HTTPException(status_code=422, detail="instruments list is empty")

    if req.fail_on_unknown:
        unknown = [n for n in names if n not in registry]
        if unknown:
            raise HTTPException(
                status_code=404, detail=f"Instruments not loaded: {unknown}"
            )

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    t0     = time.perf_counter()
    loop   = asyncio.get_event_loop()

    if req.mode == "sequential":
        # Sequential processing — for benchmarking comparison only
        raw_results = []
        for name in names:
            try:
                raw_results.append(predictor.predict_one(name, registry))
            except Exception as exc:
                raw_results.append(exc)
    else:
        # Concurrent: dispatch all instruments to thread pool simultaneously
        # Each call runs in a separate thread; NumPy + PyTorch release the GIL
        # so threads execute in parallel rather than taking turns.
        tasks = [
            loop.run_in_executor(executor, predictor.predict_one, name, registry)
            for name in names
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    signals: list[SignalResult] = []
    for name, result in zip(names, raw_results):
        if isinstance(result, Exception):
            signals.append(SignalResult(
                instrument=name,
                status="error",
                error=str(result),
            ))
        else:
            signals.append(result)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    response = GenerateResponse(
        job_id           = job_id,
        status           = "completed",
        generated_at     = datetime.now(timezone.utc).isoformat(),
        batch_latency_ms = elapsed_ms,
        signals          = signals,
    )

    jobs[job_id] = response   # store for GET retrieval
    return response


@app.get("/signals/stream", tags=["signals"])
async def stream_signals(
    instruments: str = Query(..., description="Comma-separated instrument names e.g. EURUSD,AAPL"),
) -> StreamingResponse:
    """
    SSE endpoint — emits one JSON event per instrument as each finishes.

    Choice of SSE over WebSocket: generation is server-to-client only (client
    sends one trigger, server pushes N results). SSE is the correct primitive
    for unidirectional push — a WebSocket would add bidirectional overhead
    without benefit.  EventSource is natively supported in all modern browsers
    and works over plain HTTP/1.1 without a protocol upgrade.
    """
    names = [n.strip().upper() for n in instruments.split(",") if n.strip()]
    if not names:
        raise HTTPException(status_code=422, detail="No instruments specified")

    async def event_generator():
        loop = asyncio.get_event_loop()

        async def safe_predict(name: str) -> SignalResult:
            try:
                return await loop.run_in_executor(
                    executor, predictor.predict_one, name, registry
                )
            except Exception as exc:
                return SignalResult(instrument=name, status="error", error=str(exc))

        tasks = [asyncio.create_task(safe_predict(n)) for n in names]

        for coro in asyncio.as_completed(tasks):
            result: SignalResult = await coro
            yield f"data: {result.model_dump_json()}\n\n"

        yield 'data: {"type":"done"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/signals/{job_id}", response_model=GenerateResponse, tags=["signals"])
async def get_job(job_id: str) -> GenerateResponse:
    """Retrieve the result of a previous /signals/generate call by job ID."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return jobs[job_id]


@app.get("/instruments/{name}/history", tags=["instruments"])
async def get_history(name: str, limit: int = 200) -> dict:
    """Return recent OHLCV bars for charting (most recent `limit` bars)."""
    name = name.upper()
    if name not in registry:
        raise HTTPException(status_code=404, detail=f"Instrument '{name}' not loaded")
    raw = registry[name]["raw"].tail(limit)
    records = [
        {
            "time":  str(date.date()),
            "open":  float(row["open"]),
            "high":  float(row["high"]),
            "low":   float(row["low"]),
            "close": float(row["close"]),
        }
        for date, row in raw.iterrows()
    ]
    return {"instrument": name, "data": records}
