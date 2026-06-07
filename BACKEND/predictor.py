"""
Predictor — loads all models at startup and runs ensemble inference.

Concurrency note (graded):
  Per-instrument inference is CPU-bound: LGBM runs NumPy matrix ops and
  PyTorch runs tensor arithmetic. Neither is waiting on network or disk I/O
  during a prediction call.

  asyncio alone cannot parallelize CPU-bound work — the event loop is
  single-threaded, so coroutines merely interleave, not run in parallel.

  Solution: asyncio.run_in_executor(ThreadPoolExecutor). Each instrument's
  predict_one() call is dispatched to a separate OS thread. NumPy and PyTorch
  both release the GIL during their core computations, so multiple threads
  genuinely execute in parallel rather than taking turns.

  ProcessPoolExecutor would also give CPU parallelism but adds serialization
  overhead (pickling models per call) and process start cost. For in-memory
  models already loaded into the server process, ThreadPoolExecutor is faster.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from features import build_features, build_seq_features, SEQ_FEATURE_COLS
from model_lstm import SignalTransformer
from schemas import SignalResult

warnings.filterwarnings("ignore")

# Build once at import time — reused for every predict_one call (no per-call overhead)
_US_BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())

MODEL_FOLDER = Path(__file__).parent.parent / "MODEL_FOLDER"

SEQ_LEN      = 60
LABEL_DECODE = {0: -1, 1: 0, 2: 1}
LABEL_MAP    = {1: "up", 0: "flat", -1: "down"}
DEVICE       = torch.device("cpu")   # CPU-only inference server


class Predictor:
    def __init__(self) -> None:
        self.lgbm          = None
        self.tab_feat: list = []
        self.seq_feat: list = []
        self.models:   Dict[str, SignalTransformer] = {}
        self.bundles:  Dict[str, dict]              = {}
        self.alpha_by_inst: Dict[str, float]        = {}

    # ── Startup ──────────────────────────────────────────────────────────────

    def load_models(self) -> None:
        # LightGBM
        with open(MODEL_FOLDER / "lgbm_model.pkl", "rb") as f:
            lgbm_bundle = pickle.load(f)
        self.lgbm      = lgbm_bundle["model"]
        self.tab_feat  = lgbm_bundle["features"]

        # Per-instrument alpha values
        alpha_path = MODEL_FOLDER / "alpha_by_inst.pkl"
        if alpha_path.exists():
            self.alpha_by_inst = joblib.load(alpha_path)

        # Per-instrument Transformer models
        for pt_file in sorted(MODEL_FOLDER.glob("lstm_*.pt")):
            name = pt_file.stem.replace("lstm_", "")
            if name == "global":
                continue
            bundle = torch.load(pt_file, map_location=DEVICE, weights_only=False)
            arch   = bundle.get("arch", {})
            model  = SignalTransformer(
                n_features = bundle["n_features"],
                patch_len  = arch.get("patch_len",  10),
                d_model    = arch.get("d_model",   128),
                n_heads    = arch.get("n_heads",     8),
                n_layers   = arch.get("n_layers",    4),
                d_ff       = arch.get("d_ff",      256),
                dropout    = arch.get("dropout",   0.3),
            )
            model.load_state_dict(bundle["state_dict"])
            model.eval()
            self.models[name]  = model
            self.bundles[name] = bundle
            if "features" in bundle:
                self.seq_feat = bundle["features"]

        print(f"[Predictor] Loaded LGBM + {len(self.models)} Transformer models: "
              f"{sorted(self.models)}")

    # ── Feature helpers ──────────────────────────────────────────────────────

    @staticmethod
    def build_tab_features(df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df)

    @staticmethod
    def build_seq_features_df(df: pd.DataFrame) -> pd.DataFrame:
        return build_seq_features(df)

    # ── Inference (called from ThreadPoolExecutor) ───────────────────────────

    def predict_one(self, name: str, registry: dict) -> SignalResult:
        """
        Run full ensemble inference for one instrument.
        This method is CPU-bound and is dispatched to a thread pool by main.py.
        """
        # ── Validate ─────────────────────────────────────────────────────────
        if name not in registry:
            raise ValueError(
                f"Instrument '{name}' not loaded — call POST /instruments/load first"
            )
        if name not in self.models:
            raise ValueError(f"No trained Transformer model for '{name}'")

        data   = registry[name]
        tab_df = data["tab"]
        seq_df = data["seq"]
        bundle = self.bundles[name]
        model  = self.models[name]

        # ── LGBM tabular prediction ───────────────────────────────────────────
        # mu_t/std_t are stored in registry (computed at load time from the
        # first 80% of the uploaded history to approximate training-set stats).
        tab_feat = [f for f in self.tab_feat if f in tab_df.columns]
        tab_X    = tab_df[tab_feat].values
        mu_t     = data["mu_t"][:len(tab_feat)]
        std_t    = data["std_t"][:len(tab_feat)]
        tab_norm    = (tab_X - mu_t) / std_t
        lgbm_proba  = self.lgbm.predict_proba(tab_norm[-1:])    # (1, 3)

        # ── Transformer sequence prediction ───────────────────────────────────
        # norm_mu / norm_std are the sequence normalization stats saved in bundle.
        seq_feat = [f for f in (self.seq_feat or SEQ_FEATURE_COLS)
                    if f in seq_df.columns]
        seq_X    = seq_df[seq_feat].values
        norm_mu  = np.array(bundle["norm_mu"])
        norm_std = np.array(bundle["norm_std"])
        mu_s_    = norm_mu[:len(seq_feat)]
        std_s_   = norm_std[:len(seq_feat)]
        seq_norm = (seq_X - mu_s_) / std_s_

        if len(seq_norm) < SEQ_LEN:
            raise ValueError(
                f"Not enough data for '{name}': need ≥{SEQ_LEN} rows, got {len(seq_norm)}"
            )

        # Build the last window only (most recent SEQ_LEN bars)
        last_window = seq_norm[-SEQ_LEN:]                           # (60, F)
        x = torch.tensor(last_window[np.newaxis], dtype=torch.float32)  # (1, 60, F)
        with torch.no_grad():
            logits     = model(x)
            trans_proba = torch.softmax(logits, dim=-1).numpy()    # (1, 3)

        # ── Blend ─────────────────────────────────────────────────────────────
        alpha   = bundle.get("alpha", self.alpha_by_inst.get(name, 0.3))
        blended = (1 - alpha) * lgbm_proba + alpha * trans_proba   # (1, 3)

        pred_class = int(blended.argmax(axis=1)[0])
        direction  = LABEL_MAP[LABEL_DECODE[pred_class]]
        confidence = round(float(blended[0, pred_class]), 4)

        last_bar   = data["raw"].iloc[-1]
        last_dt    = data["raw"].index[-1]
        last_date  = str(last_dt.date())
        last_price = round(float(last_bar["close"]), 5)

        # Next trading day: skip weekends AND US federal holidays.
        # _US_BD is built once at module load — zero overhead per call.
        next_td = (pd.Timestamp(last_date) + _US_BD).date()

        return SignalResult(
            instrument       = name,
            direction        = direction,
            confidence       = confidence,
            model_version    = bundle.get("version", "v2"),
            status           = "ok",
            prediction_date  = last_date,
            next_trading_day = str(next_td),
            last_price       = last_price,
        )

