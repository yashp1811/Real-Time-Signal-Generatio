"""
Feature engineering for the directional signal model.

All features are computed from past data only — no future information leaks in.
Each feature window looks backward from the current bar, never forward.
"""

import numpy as np
import pandas as pd


# ── label thresholds ─────────────────────────────────────────────────────────
UP_THRESHOLD   =  0.001   # +0.1 % → "up"
DOWN_THRESHOLD = -0.001   # -0.1 % → "down"
# returns in [-0.1%, +0.1%] → "flat"

LABEL_MAP = {1: "up", 0: "flat", -1: "down"}


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with columns [open, high, low, close, volume] and a DatetimeIndex.
    Returns: DataFrame of features + 'label' column, with NaN rows dropped.

    Label is the *next* bar's direction, computed from the *next* close relative to
    the *current* close.  Because we shift(-1) to get the next return, the last row
    of the dataset naturally has no label and is dropped — this prevents look-ahead
    since we never include any information from bar t+1 in the features of bar t.
    """
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df.get("volume", pd.Series(np.nan, index=df.index))

    # ── returns ───────────────────────────────────────────────────────────────
    for n in [1, 3, 5, 10, 20]:
        out[f"ret_{n}"] = c.pct_change(n)

    # ── momentum / trend ──────────────────────────────────────────────────────
    out["rsi_14"]  = _rsi(c, 14)
    out["rsi_7"]   = _rsi(c, 7)

    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd  = ema12 - ema26
    signal = _ema(macd, 9)
    out["macd"]        = macd / c           # normalised by price
    out["macd_signal"] = signal / c
    out["macd_hist"]   = (macd - signal) / c

    # ── Bollinger Bands (20-period) ───────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out["bb_pos"]   = (c - bb_mid) / (2 * bb_std)   # -1…+1 within bands
    out["bb_width"] = (2 * bb_std) / bb_mid          # normalised width

    # ── volatility ───────────────────────────────────────────────────────────
    out["atr_14_norm"] = _atr(h, l, c, 14) / c
    out["vol_5"]  = c.pct_change().rolling(5).std()
    out["vol_20"] = c.pct_change().rolling(20).std()

    # ── volume (if available) ─────────────────────────────────────────────────
    if not v.isna().all() and (v != 0).any():
        vol_ma20 = v.rolling(20).mean()
        out["vol_ratio"] = v / vol_ma20.replace(0, np.nan)
    else:
        out["vol_ratio"] = np.nan

    # ── price position within recent range ───────────────────────────────────
    high_20 = h.rolling(20).max()
    low_20  = l.rolling(20).min()
    rng = (high_20 - low_20).replace(0, np.nan)
    out["price_pos_20"] = (c - low_20) / rng  # 0…1

    # ── label: next bar's direction ───────────────────────────────────────────
    next_ret = c.pct_change().shift(-1)   # shift(-1) → next bar's return
    out["label"] = 0
    out.loc[next_ret >  UP_THRESHOLD,   "label"] = 1
    out.loc[next_ret <  DOWN_THRESHOLD, "label"] = -1

    # drop last row (no label); drop columns that are entirely NaN
    # (e.g. vol_ratio for instruments with no volume data), then drop remaining NaN rows
    out = out.iloc[:-1]
    out = out.dropna(axis=1, how="all")
    out = out.dropna()

    return out


FEATURE_COLS = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "rsi_14", "rsi_7",
    "macd", "macd_signal", "macd_hist",
    "bb_pos", "bb_width",
    "atr_14_norm", "vol_5", "vol_20",
    "vol_ratio",
    "price_pos_20",
]


# ── Sequence features + volatility-adjusted labels (for LSTM) ────────────────

def build_seq_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw, sequential feature set designed for the CNN-BiLSTM.

    Key differences from build_features():
    1.  Emphasises raw price-action inputs (returns, candle body, intraday range)
        so the CNN can learn its own pattern detectors rather than receiving
        pre-summarised indicators.
    2.  Uses VOLATILITY-ADJUSTED labels instead of a fixed ±0.1% threshold.
        Label = +1 if next_return > +0.5*rolling_std(20),
                -1 if next_return < -0.5*rolling_std(20), else 0.
        This makes 'up'/'down' mean the same thing across calm and turbulent
        regimes — a 0.1% move in a 2% daily-vol environment is noise,
        while the same move in a 0.3% daily-vol environment is a signal.
    """
    out = pd.DataFrame(index=df.index)
    c   = df["close"]
    h   = df["high"]
    l   = df["low"]
    o   = df["open"]
    v   = df.get("volume", pd.Series(np.nan, index=df.index))

    # ── raw returns (primary sequential signal) ───────────────────────────────
    daily_ret = c.pct_change()
    for n in [1, 2, 3, 5, 10]:
        out[f"ret_{n}"] = c.pct_change(n)

    # ── candle anatomy (captures bar structure the LSTM can sequence) ─────────
    hl   = (h - l).replace(0, np.nan)
    out["hl_ratio"] = hl / c                          # intraday range / price
    out["co_ratio"] = (c - o) / hl                    # body position: -1..+1

    # ── momentum oscillators (normalised) ─────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    out["rsi"] = (100 - 100 / (1 + rs)) / 50 - 1     # rescaled to -1..+1

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - sig) / c               # normalised by price

    # ── Bollinger Band position ───────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std().replace(0, np.nan)
    out["bb_pos"]   = (c - bb_mid) / (2 * bb_std)
    out["bb_width"] = (2 * bb_std) / bb_mid

    # ── realised vol (regime signal) ──────────────────────────────────────────
    out["vol_5"]  = daily_ret.rolling(5).std()
    out["vol_20"] = daily_ret.rolling(20).std()

    # ── volume ratio (if available) ───────────────────────────────────────────
    if not v.isna().all() and (v != 0).any():
        vol_ma = v.rolling(20).mean().replace(0, np.nan)
        out["vol_ratio"] = v / vol_ma
    # no else — column simply absent; make_windows handles missing cols

    # ── VOLATILITY-ADJUSTED LABEL ─────────────────────────────────────────────
    roll_std  = daily_ret.rolling(20).std()            # 20-day realised vol
    threshold = 0.5 * roll_std                         # adaptive threshold
    next_ret  = daily_ret.shift(-1)                    # next bar's return (target)

    out["label"] = 0
    out.loc[next_ret >  threshold, "label"] =  1
    out.loc[next_ret < -threshold, "label"] = -1

    # drop last row (no next-bar label) + NaN-only columns + any remaining NaN rows
    out = out.iloc[:-1]
    out = out.dropna(axis=1, how="all")
    out = out.dropna()
    return out


SEQ_FEATURE_COLS = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    "hl_ratio", "co_ratio",
    "rsi", "macd_hist",
    "bb_pos", "bb_width",
    "vol_5", "vol_20",
    "vol_ratio",   # included if present; intersection handles missing columns
]
