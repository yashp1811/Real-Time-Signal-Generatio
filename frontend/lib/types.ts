export type Direction = "up" | "down" | "flat";
export type SignalStatus = "ok" | "error" | "loading" | "idle";

export interface SignalResult {
  instrument: string;
  direction?: Direction;
  confidence?: number;
  model_version?: string;
  status: SignalStatus | string;
  error?: string;
  prediction_date?: string;    // last bar date (data used for signal)
  next_trading_day?: string;   // actual day being predicted (skips weekends + holidays)
  last_price?: number;         // closing price of last bar
}

export interface OHLCBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
}
