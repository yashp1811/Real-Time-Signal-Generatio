from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class OHLCVRow(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class InstrumentPayload(BaseModel):
    name: str = Field(..., description="Instrument identifier, e.g. EURUSD")
    data: List[OHLCVRow] = Field(
        ..., min_length=70, description="OHLCV rows sorted oldest → newest (≥70 required)"
    )


class LoadRequest(BaseModel):
    instruments: List[InstrumentPayload] = Field(..., min_length=1)


class LoadResponse(BaseModel):
    loaded: List[str]
    errors: Dict[str, str] = {}


class GenerateRequest(BaseModel):
    instruments: List[str] = Field(..., min_length=1)
    fail_on_unknown: bool = Field(
        False, description="If True, raise 404 when any instrument is not loaded"
    )
    mode: str = Field(
        "concurrent",
        description="'concurrent' (default) or 'sequential' — sequential is for benchmarking only"
    )


class SignalResult(BaseModel):
    instrument: str
    direction: Optional[str] = None       # "up" | "down" | "flat"
    confidence: Optional[float] = None    # max blended probability for winning class
    model_version: Optional[str] = None
    status: str                            # "ok" | "error"
    error: Optional[str] = None
    prediction_date: Optional[str] = None   # date of last bar used for prediction
    next_trading_day: Optional[str] = None  # next business day (the day being predicted)
    last_price: Optional[float] = None      # closing price of last bar


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    generated_at: str
    batch_latency_ms: float
    signals: List[SignalResult]


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    instruments_registered: int
    timestamp: str
