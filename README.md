# Real-Time Signal Generation & Backtesting Service

A full-stack directional signal service combining a **PatchTST-style Transformer** with **LightGBM** in a per-instrument blended ensemble, served through a concurrent **FastAPI** backend and a live-streaming **Next.js** dashboard.

**Demo video:** [Watch on Google Drive](https://drive.google.com/file/d/1SrRgO2f39XD5E4UrFT3mVTbolM7ZOmF9/view?usp=sharing)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Model Training on Kaggle](#model-training-on-kaggle)
4. [API Reference](#api-reference)
5. [Benchmark Results](#benchmark-results)
6. [Model Evaluation](#model-evaluation)
7. [Design Decisions](#design-decisions)

---

## Quick Start

Pre-trained model artifacts are already included in `MODEL_FOLDER/`. You do **not** need to retrain to run the service.

### 1. Backend

```bash
cd Real_Time_Signal_Generation/BACKEND
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server starts at `http://localhost:8000`. On startup it loads the LightGBM model and all 8 per-instrument Transformer models from `MODEL_FOLDER/`.

### 2. Frontend

```bash
cd Real_Time_Signal_Generation/frontend
npm install
npm run dev
```

Dashboard runs at `http://localhost:3000`. Click **Load & Generate Signals** — it fetches live OHLCV data from Yahoo Finance, then streams signals one-by-one via SSE as each instrument finishes.

### 3. Benchmark

```bash
cd Real_Time_Signal_Generation/BACKEND
python benchmark.py
```

Runs 20 timed requests in concurrent and sequential mode, then prints p50/p95/p99 latency and the speedup ratio.

---

## Project Structure

```
Real_Time_Signal_Generation/
├── README.md
│
├── BACKEND/
│   ├── main.py              # FastAPI app — 6 endpoints
│   ├── predictor.py         # Ensemble inference, ThreadPoolExecutor dispatch
│   ├── features.py          # Feature engineering (tabular + sequential)
│   ├── model_lstm.py        # SignalTransformer architecture (PatchTST-style)
│   ├── schemas.py           # Pydantic request/response models
│   ├── benchmark.py         # Concurrent vs sequential latency comparison
│   └── requirements.txt
│
├── MODEL_FOLDER/
│   ├── lgbm_model.pkl           # Shared LightGBM classifier
│   ├── alpha_by_inst.pkl        # Per-instrument blend weights
│   ├── lstm_AAPL.pt             # Fine-tuned Transformer — AAPL
│   ├── lstm_EURUSD.pt           # Fine-tuned Transformer — EURUSD
│   ├── lstm_GBPUSD.pt           # Fine-tuned Transformer — GBPUSD
│   ├── lstm_GLD.pt              # Fine-tuned Transformer — GLD
│   ├── lstm_MSFT.pt             # Fine-tuned Transformer — MSFT
│   ├── lstm_NDX.pt              # Fine-tuned Transformer — NDX
│   ├── lstm_SPX.pt              # Fine-tuned Transformer — SPX
│   ├── lstm_USO.pt              # Fine-tuned Transformer — USO
│   ├── evaluation_report.json   # Per-instrument test-set metrics (authoritative)
│   ├── walkforward_results.csv  # Quarterly walk-forward results
│   ├── live_report.json         # Live (2026 YTD) performance
│   ├── summary.txt              # Human-readable evaluation summary
│   └── model_create.ipynb       # Full training pipeline (runs on Kaggle)
│
└── frontend/
    ├── app/
    │   ├── page.tsx         # Dashboard page (SSE, instrument selector, chart)
    │   └── layout.tsx       # Root layout
    ├── components/
    │   ├── PriceChart.tsx   # lightweight-charts v5 candlestick + signal marker
    │   └── SignalCard.tsx   # Per-instrument signal card
    ├── lib/
    │   └── types.ts         # Shared TypeScript types
    └── package.json
```

---

## Model Training on Kaggle

The full training pipeline lives in `MODEL_FOLDER/model_create.ipynb`. It is designed to run on **Kaggle** (GPU/CPU accelerator, free tier is sufficient).

### What the notebook does

1. **Downloads data** — uses `yfinance` to pull daily OHLCV history (2000–today) for all 8 instruments.
2. **Feature engineering** — builds tabular features (RSI, MACD, Bollinger Bands, ATR, etc.) and sequential features (raw returns, candle anatomy) for each instrument.
3. **Trains LightGBM** — fits a shared multi-class classifier on tabular features with a strict time-ordered train/validation/test split.
4. **Trains global Transformer** — pre-trains a single `SignalTransformer` on all instruments combined.
5. **Fine-tunes per-instrument Transformers** — freezes the patch embedding layer and fine-tunes the encoder + classifier head per instrument.
6. **Tunes blend weights** — grid-searches α ∈ [0, 1] on each instrument's validation fold to find the optimal `p = (1−α)·p_lgbm + α·p_transformer` blend.
7. **Evaluates on test set** — runs the full evaluation against `always_up` and `persistence` baselines and writes `evaluation_report.json`.
8. **Saves artifacts** — writes `lgbm_model.pkl`, `alpha_by_inst.pkl`, and one `lstm_<INST>.pt` per instrument.

### How to run it

1. Upload `model_create.ipynb` to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → **File → Import Notebook**.
2. Set **Accelerator** to CPU (or GPU for faster training).
3. Click **Run All**.
4. The notebook will download all 8 CSV files automatically, then train all models (≈ 30–60 min on CPU, ≈ 10–15 min on GPU).
5. Download the `artifacts_3/` folder from the Kaggle output panel and place its contents into `MODEL_FOLDER/`.

> The pre-trained artifacts in `MODEL_FOLDER/` were produced by this exact notebook. You only need to retrain if you want to update with newer data or change the architecture.

### Sample data

Sample OHLCV CSV files are included in `data/` so the reviewer can run the pipeline without relying only on external downloads. The training notebook can also regenerate fresh data using `yfinance`.
---

## API Reference

All responses are JSON. Base URL: `http://localhost:8000`.

---

### `GET /health`

Readiness check. Returns model count and registered instrument count.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "models_loaded": 8,
  "instruments_registered": 0,
  "timestamp": "2026-06-07T10:00:00Z"
}
```

---

### `POST /instruments/autoload?instruments=ALL`

Fetches 3 years of live daily OHLCV from Yahoo Finance for all 8 instruments, builds features, and caches them in memory. Falls back to local CSVs if Yahoo Finance is unavailable. Called automatically by the dashboard.

```bash
curl -X POST "http://localhost:8000/instruments/autoload?instruments=ALL"
```

```json
{
  "loaded": ["AAPL", "EURUSD", "GBPUSD", "GLD", "MSFT", "NDX", "SPX", "USO"],
  "errors": {}
}
```

Pass a comma-separated list instead of `ALL` to load a subset: `?instruments=AAPL,EURUSD`.

---

### `POST /instruments/load`

Register instruments with custom OHLCV history supplied in the request body. Each instrument needs ≥ 70 OHLCV rows.

```bash
curl -X POST http://localhost:8000/instruments/load \
  -H "Content-Type: application/json" \
  -d '{
    "instruments": [{
      "name": "AAPL",
      "data": [
        {"date": "2024-01-02", "open": 185.2, "high": 186.1, "low": 184.3, "close": 185.9, "volume": 55000000},
        {"date": "2024-01-03", "open": 184.2, "high": 185.7, "low": 183.1, "close": 184.3, "volume": 48000000}
      ]
    }]
  }'
```

```json
{ "loaded": ["AAPL"], "errors": {} }
```

**Validation errors returned:**
- Missing required OHLC column → `errors: { "AAPL": "Missing required column: high" }`
- Fewer than 70 rows → Pydantic 422 error
- Empty `instruments` list → 422 Unprocessable Entity

---

### `POST /signals/generate`

Generate directional signals for a batch of instruments. Default mode is `concurrent` (all instruments processed in parallel via `ThreadPoolExecutor`). Pass `"mode": "sequential"` for the benchmark comparison only.

```bash
curl -X POST http://localhost:8000/signals/generate \
  -H "Content-Type: application/json" \
  -d '{"instruments": ["AAPL", "EURUSD", "GBPUSD", "BADSYM"], "mode": "concurrent"}'
```

```json
{
  "job_id": "job_a1b2c3d4",
  "status": "completed",
  "generated_at": "2026-06-07T10:00:01Z",
  "batch_latency_ms": 54.1,
  "signals": [
    {
      "instrument": "AAPL",
      "direction": "down",
      "confidence": 0.412,
      "model_version": "v2",
      "status": "ok",
      "prediction_date": "2026-06-06",
      "next_trading_day": "2026-06-09",
      "last_price": 198.42
    },
    {
      "instrument": "EURUSD",
      "direction": "up",
      "confidence": 0.387,
      "model_version": "v2",
      "status": "ok",
      "prediction_date": "2026-06-06",
      "next_trading_day": "2026-06-09",
      "last_price": 1.0823
    },
    {
      "instrument": "BADSYM",
      "direction": null,
      "confidence": null,
      "status": "error",
      "error": "Instrument 'BADSYM' not loaded — call POST /instruments/load first"
    }
  ]
}
```

---

### `GET /signals/stream?instruments=AAPL,EURUSD,...`

**Server-Sent Events** endpoint. Emits one JSON event per instrument as each finishes (order reflects completion, not input order), then a `{"type":"done"}` sentinel. This is what the dashboard uses for live updates.

```bash
curl -N "http://localhost:8000/signals/stream?instruments=AAPL,EURUSD"
```

```
data: {"instrument":"EURUSD","direction":"up","confidence":0.387,"status":"ok","prediction_date":"2026-06-06","next_trading_day":"2026-06-09","last_price":1.0823}

data: {"instrument":"AAPL","direction":"down","confidence":0.412,"status":"ok","prediction_date":"2026-06-06","next_trading_day":"2026-06-09","last_price":198.42}

data: {"type":"done"}
```

---

### `GET /signals/{job_id}`

Retrieve a previously generated batch result by job ID.

```bash
curl http://localhost:8000/signals/job_a1b2c3d4
```

Returns the same `GenerateResponse` shape as `/signals/generate`.

---

### `GET /instruments/{name}/history?limit=250`

Return the most recent OHLCV bars for a loaded instrument (used by the dashboard to populate the chart).

```bash
curl "http://localhost:8000/instruments/AAPL/history?limit=250"
```

```json
{
  "instrument": "AAPL",
  "data": [
    {"time": "2025-09-01", "open": 220.1, "high": 223.4, "low": 219.8, "close": 222.5}
  ]
}
```

---

## Benchmark Results

**Hardware:** Intel Core i7-1255U (10-core), 16 GB RAM, Windows 11, CPU-only inference.  
**Batch size:** 24 instruments using aliases of the 8 supported instruments to stress-test batch concurrency.  
**Concurrency level:** 8 worker threads (`ThreadPoolExecutor(max_workers=8)`).  
**Timed requests:** 20 per mode (3 warm-up excluded).

| Mode | p50 | p95 | p99 | Mean |
|------|-----|-----|-----|------|
| **Concurrent** | **PUT_VALUE ms** | PUT_VALUE ms | PUT_VALUE ms | PUT_VALUE ms |
| Sequential | PUT_VALUE ms | PUT_VALUE ms | PUT_VALUE ms | PUT_VALUE ms |
| **Speedup** | **PUT_VALUE×** | PUT_VALUE× | — | PUT_VALUE× |

The concurrent batch completes in approximately `max(individual latencies)` rather than `sum(individual latencies)`. Each `predict_one()` call runs in a separate OS thread. NumPy and PyTorch both release the Python GIL during their core C-extension matrix computations, allowing all 8 threads to execute in parallel rather than taking turns.

---

## Model Evaluation

**Test period:** 2024-01-01 → 2025-12-31 (strict calendar cut; never seen during training).

**Ensemble formula:** `p = (1−α) · p_lgbm + α · p_transformer`, where α is tuned per instrument on a held-out validation fold.

### Test-set metrics

| Instrument | Ens Acc | Ens F1 | Ens Sharpe | Max DD | vs Always-Up | vs Persistence |
|------------|---------|--------|------------|--------|:---:|:---:|
| AAPL       | 0.257   | 0.248  | +0.73      | −0.20  | ✓  | ✓  |
| EURUSD     | 0.372   | 0.371  | +2.19      | −0.05  | ✓  | ✓  |
| GBPUSD     | 0.362   | 0.364  | +2.84      | −0.03  | ✓  | ✓  |
| GLD        | 0.323   | 0.305  | +1.43      | −0.14  | ✓  | ✓  |
| MSFT       | 0.305   | 0.297  | −0.01      | −0.32  | ✓  | ✓  |
| NDX        | 0.347   | 0.345  | +0.08      | −0.28  | ✓  | ✓  |
| SPX        | 0.363   | 0.357  | −0.25      | −0.30  | ✓  | ✓  |
| USO        | 0.355   | 0.320  | +0.94      | −0.35  | ✓  | ✓  |

**Baseline beats — plain statement:**

- **vs `always_up`:** Ensemble beats this baseline on all 8 instruments (on both F1 and directional accuracy). ✓
- **vs `persistence`:** Ensemble beats this baseline on all 8 instruments (on both F1 and directional accuracy). ✓
- **vs standalone LightGBM:** On **F1 (classifier accuracy)**, the ensemble beats standalone LGBM on **1 of 8 instruments** (SPX). On **Sharpe ratio (signal quality)**, the ensemble beats standalone LGBM on **5 of 8 instruments** (AAPL, EURUSD, GBPUSD, MSFT, NDX). The ensemble does not consistently outperform LGBM as a standalone classifier — blending adds value primarily in risk-adjusted signal quality, not raw directional accuracy.

Full per-model metrics (LGBM, global Transformer, fine-tuned Transformer, ensemble, both baselines) are in [`MODEL_FOLDER/evaluation_report.json`](MODEL_FOLDER/evaluation_report.json). Quarterly walk-forward results are in [`MODEL_FOLDER/walkforward_results.csv`](MODEL_FOLDER/walkforward_results.csv).

### Metric definitions

- **Directional accuracy** — fraction of test bars where the predicted direction (up/flat/down) matches the actual next-day direction. A random predictor on a 3-class problem scores ≈ 0.33.
- **F1 (macro)** — unweighted average of per-class F1 scores; penalises models that ignore minority classes (flat days) more than accuracy does.
- **Sharpe ratio** — annualised mean daily P&L divided by its standard deviation, using a unit long/short/cash position sized by the predicted direction. Positive means the signal earns more relative to its volatility on average. A coin flip has Sharpe ≈ 0.
- **Maximum drawdown (MDD)** — largest peak-to-trough decline in the cumulative equity curve under the same unit-position simulation.
- **Cumulative return** — total P&L from the unit-position simulation over the test period.

> Equity-curve metrics (Sharpe, MDD, cumulative return) use a unit-position simulation with no transaction costs, slippage, or position sizing. They measure signal quality, not tradeable profitability.

---

## Design Decisions

*(≤ 600 words as required)*

### Model: PatchTST-style Transformer + LightGBM Ensemble

**What the model is.** The Transformer component (`SignalTransformer` in `model_lstm.py`) is a patch-based sequence classifier inspired by PatchTST (Nie et al., 2023). A 60-bar OHLCV window is split into 6 non-overlapping patches of 10 bars each. Each patch is linearly projected to a 128-dimensional embedding. A learnable `[CLS]` token is prepended, sinusoidal positional encoding is added, and the 7-token sequence passes through 4 Transformer encoder layers (8 attention heads, d_model=128, d_ff=256, Pre-LN). The `[CLS]` output is fed to a 3-class linear head predicting `down / flat / up`.

The LightGBM component operates on 16 hand-crafted technical indicators (RSI-14, RSI-7, MACD, Bollinger Band position and width, ATR, rolling return volatility, momentum returns at 1/3/5/10/20 bars, volume ratio, price position in 20-day range).

The final signal blends the two probability vectors: `p = (1−α)·p_lgbm + α·p_transformer`, where α is grid-searched per instrument on a validation fold.

**What the model is not.** It is not a language model, not a regression model, and does not predict prices. It outputs a ternary directional label for the *next trading day only*. It has no access to news, earnings, macro data, or order-book information.

**Why Transformer over LSTM.** LSTM gradients vanish beyond ≈30 steps. Attention is O(1) in depth — every patch can attend directly to every other patch. PatchTST patching reduces attention complexity from O(T²) to O((T/P)²): 60→6 tokens, 100× fewer attention operations, while forcing coarser, more robust temporal patterns.

**Known limitations.** (1) *Regime risk*: training data ends 2023; structural shifts in 2024+ may degrade performance. (2) *Class imbalance*: `flat` days are under-represented, causing low recall for neutral signals. (3) *Simulation gap*: Sharpe/MDD metrics ignore transaction costs, slippage, and position sizing.

---

### Leakage Prevention

Labels are computed with `c.pct_change().shift(-1)`: each bar's label is derived from the *next* bar's return, never the current one. All features use only backward-looking operations (rolling windows, lagged returns, exponential smoothing). The train/test split is a strict calendar date cut — train: 2000–2023, test: 2024–2025, live: 2026+ — with no shuffling or cross-time k-fold. Normalisation statistics (mean and std) are computed from the first 80% of each instrument's history and applied to the remaining 20%; future distribution is never used at training time.

---

### Concurrency: ThreadPoolExecutor, Not asyncio or ProcessPool

Signal inference is **CPU-bound**: both NumPy and PyTorch perform dense matrix arithmetic that saturates a CPU core. `asyncio` alone cannot parallelise CPU-bound work — the event loop is single-threaded and coroutines only interleave on one thread. `asyncio.run_in_executor(ThreadPoolExecutor)` gives genuine parallelism because NumPy and PyTorch release the Python GIL during their core C-extension computations, allowing 8 threads to run simultaneously. `ProcessPoolExecutor` would also parallelise CPU work but adds per-call pickling overhead for model objects already resident in the server process; threads are strictly faster here. The `asyncio.gather(*tasks)` pattern completes the batch in approximately `max(individual latencies)` instead of `sum(individual latencies)`.

---

### SSE vs WebSocket

Signal generation is **server-to-client only**: the client sends one trigger, and the server pushes N results as each instrument finishes. Server-Sent Events (`text/event-stream`) is the correct primitive for unidirectional streaming. It works over plain HTTP/1.1 with no protocol upgrade, is natively supported by browser `EventSource` with automatic reconnection, and requires zero additional infrastructure. WebSocket adds full-duplex complexity — a persistent bidirectional channel the server would never write back to — with no benefit for this one-shot streaming pattern.
