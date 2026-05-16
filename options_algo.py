"""
Supertrend Intraday Backtesting Strategy — v4.0 (Greeks / Options Edition)
─────────────────────────────────────────────────────────────────────────────
ADDITIONS vs v3.0
  • Greeks-based Option P&L approximation per trade (Delta, Gamma, Theta, Vega)
  • Dynamic Delta update:  new_delta = old_delta + gamma * ΔS
  • Vega effect via India VIX CSV  (VIX_CSV path in CONFIG)
  • Break-even ΔS estimation per trade
  • Per-trade Greek attribution breakdown
  • Full HTML report (replaces Excel) — all summaries, charts, option analytics

Install: pip install pandas plotly pytz numpy yfinance scipy
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pytz, os, json, math
from datetime import time as dtime
from scipy.optimize import brentq

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATA_SOURCE      = "csv"
BASE_CSV         = "Book1min.csv"
HTF_CSV          = "Book1.csv"
VIX_CSV          = None               # path to India VIX daily CSV, or None

YF_TICKER        = "^NSEI"
YF_BASE_INTERVAL = "1m"
YF_HTF_INTERVAL  = "15m"
YF_PERIOD        = "7d"
YF_START         = None
YF_END           = None
YF_VIX_TICKER    = "^INDIAVIX"       # used when VIX_CSV is None

SYMBOL           = "NIFTY"

# ── Supertrend ─────────────────────────────────────────────────────────────────
ST_PERIOD         = 14
ST_MULTIPLIER     = 4.0
HTF_ST_PERIOD     = 10
HTF_ST_MULTIPLIER = 3.0

# ── Risk ───────────────────────────────────────────────────────────────────────
HARD_SL_ENABLED  = True
HARD_SL_PCT      = 0.25
EXIT_USE_HTF_ST  = False
CLOSE_BASED_EXIT = True

# ── Filters ────────────────────────────────────────────────────────────────────
TRADE_WINDOW_ENABLED = True
TRADE_WINDOWS = [
    (dtime(9, 30), dtime(11, 30)),
    (dtime(14,  0), dtime(14, 45)),
]
CONFIRM_CANDLES            = 10
MIN_GAP_ENABLED            = True
MIN_GAP_PCT                = 0.20
VOLUME_FILTER_ENABLED      = False
VOLUME_MULTIPLIER          = 1.5
VOLUME_LOOKBACK            = 25
MAX_TRADES_PER_DAY_ENABLED = True
MAX_TRADES_PER_DAY         = 5

# ── Market ─────────────────────────────────────────────────────────────────────
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 15)
TRADING_HOURS_PER_DAY = 6.25
IST          = pytz.timezone("Asia/Kolkata")
CHART_DAYS   = 5

# ══════════════════════════════════════════════════════════════════════════════
#  GREEKS PARAMETERS  ← USER INPUTS
# ══════════════════════════════════════════════════════════════════════════════

OPTION_LOT_SIZE   = 65       # NIFTY lot size
OPTION_DELTA      = 0.20     # e.g. 0.40 for slightly ITM CE / 0.20 for OTM
OPTION_GAMMA      = 0.0005   # typical NIFTY gamma for ATM
OPTION_THETA      = -10.53   # daily theta (negative for long options)
OPTION_VEGA       = 5.0      # per 1% change in IV; typical ATM NIFTY value
OPTION_DIRECTION  = "long"   # "long"  → buying options; "short" → selling

# Whether to use dynamic delta (updated per ΔS step for better accuracy)
USE_DYNAMIC_DELTA = True

LOW_POINTS_THRESHOLD = 20

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fmt(ts):
    return ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "—"

def mins_to_hhmm(m):
    m = int(round(m))
    return f"{m // 60}H:{m % 60:02d}M"

# ═══════════════════════════════════════════════════════════════════════════════
#  SUPERTREND
# ═══════════════════════════════════════════════════════════════════════════════

def compute_supertrend(df, period, multiplier):
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n     = len(df)

    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:]  - close[:-1])))

    atr = np.full(n, np.nan)
    if n > period:
        atr[period] = tr[:period].mean()
        for j in range(period + 1, n):
            atr[j] = (atr[j-1] * (period - 1) + tr[j-1]) / period

    hl2         = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper = np.full(n, np.nan); lower = np.full(n, np.nan)
    bull  = np.full(n, True, dtype=bool)

    for j in range(period, n):
        if np.isnan(upper[j-1]):
            upper[j] = upper_basic[j]; lower[j] = lower_basic[j]
            bull[j]  = close[j] >= lower[j]; continue
        upper[j] = (upper_basic[j]
                    if upper_basic[j] < upper[j-1] or close[j-1] > upper[j-1]
                    else upper[j-1])
        lower[j] = (lower_basic[j]
                    if lower_basic[j] > lower[j-1] or close[j-1] < lower[j-1]
                    else lower[j-1])
        bull[j] = (close[j] >= lower[j]) if bull[j-1] else (close[j] > upper[j])

    df = df.copy()
    df["ST_val"]       = np.where(bull, lower, upper)
    df["ST_bull"]      = bull
    df["ST_direction"] = np.where(bull, "Bullish", "Bearish")
    return df

# ═══════════════════════════════════════════════════════════════════════════════
#  GREEKS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_option_pnl(entry_price_S, exit_price_S, holding_hours,
                       trade_direction="long",          # ← "long" or "short" underlying
                       vix_at_entry=None, vix_at_exit=None,
                       delta=None, gamma=None, theta_daily=None, vega=None,
                       lot_size=None, opt_direction=None):
    """
    Returns a dict of all Greeks effects and final P&L for one trade.

    KEY DESIGN DECISIONS (fixes vs v4.0):
    ──────────────────────────────────────────────────────────────────────────
    FIX 1 — Delta sign is now trade-direction-aware:
      • Long underlying trade  → buying CE  → delta is POSITIVE (+OPTION_DELTA)
      • Short underlying trade → buying PE  → delta is NEGATIVE (-OPTION_DELTA)
      User sets OPTION_DELTA as a positive magnitude (e.g. 0.40); the sign is
      applied here automatically based on which way the underlying trade goes.
      OPTION_DIRECTION="long"/"short" still controls buyer vs seller of the option.

    FIX 2 — No gamma double-count with dynamic delta:
      When USE_DYNAMIC_DELTA=True the gamma curvature is ALREADY embedded in the
      delta path (each micro-step uses an updated delta that reflects gamma).
      Adding a separate gamma_effect on top was double-counting.
      Correct approach:
        • Dynamic delta ON  → gamma_effect = 0 (already in delta path)
        • Dynamic delta OFF → gamma_effect = ½ · Γ · ΔS²  (static Taylor term)

    FIX 3 — Break-even solved with the correct signed delta.
    ──────────────────────────────────────────────────────────────────────────
    OPTION_DIRECTION semantics (unchanged):
      "long"  → option buyer  (pay premium → P&L = +option_change)
      "short" → option seller (receive premium → P&L = -option_change)
    """
    abs_delta  = abs(delta  if delta  is not None else OPTION_DELTA)
    gamma      = gamma      if gamma      is not None else OPTION_GAMMA
    theta_daily= theta_daily if theta_daily is not None else OPTION_THETA
    vega       = vega       if vega       is not None else OPTION_VEGA
    lot_size   = lot_size   if lot_size   is not None else OPTION_LOT_SIZE
    opt_direction = opt_direction or OPTION_DIRECTION

    # FIX 1: delta sign follows underlying trade direction
    #   Long  trade → CE  → positive delta
    #   Short trade → PE  → negative delta
    signed_delta = abs_delta if trade_direction == "long" else -abs_delta

    dS_total     = exit_price_S - entry_price_S   # positive for long win, negative for short win
    theta_per_hr = theta_daily / TRADING_HOURS_PER_DAY
    theta_effect = theta_per_hr * holding_hours   # always negative (time decay cost)

    # Vega effect
    vega_effect = 0.0
    vix_change  = 0.0
    if vix_at_entry is not None and vix_at_exit is not None and vix_at_entry > 0:
        vix_change  = vix_at_exit - vix_at_entry
        vega_effect = vega * vix_change
        # For PE (short trade), rising VIX still helps long option buyer
        # vega is always positive for long options regardless of CE/PE

    # FIX 2: Dynamic delta with NO separate gamma term
    steps     = 10
    step_size = dS_total / steps
    cur_delta = signed_delta
    delta_effect_dyn = 0.0

    for _ in range(steps):
        delta_effect_dyn += cur_delta * step_size
        if USE_DYNAMIC_DELTA:
            # For CE: gamma > 0 → delta grows as price rises  (positive feedback)
            # For PE: gamma > 0 → delta becomes less negative as price falls (also positive feedback)
            # Both CE and PE benefit from gamma convexity → gamma always added positively
            cur_delta = cur_delta + gamma * step_size

    # gamma_effect only needed for STATIC delta mode (not double-counted)
    gamma_effect = 0.0 if USE_DYNAMIC_DELTA else 0.5 * gamma * (dS_total ** 2)

    # Static gamma for reference/display
    gamma_effect_static = 0.5 * gamma * (dS_total ** 2)

    # Total option price change per unit (option premium change)
    option_change = delta_effect_dyn + gamma_effect + theta_effect + vega_effect
    final_delta   = cur_delta

    # FIX 3: option buyer/seller sign (this is about the option contract, not underlying direction)
    sign = 1 if opt_direction == "long" else -1
    pnl_per_lot = option_change * sign
    total_pnl   = pnl_per_lot * lot_size

    # Break-even ΔS using FIX 1 signed delta
    const_term   = theta_effect + vega_effect
    breakeven_dS = _solve_breakeven(signed_delta, gamma, const_term)

    return {
        "dS"                  : round(dS_total, 4),
        "holding_hours"       : round(holding_hours, 4),
        "delta_entry"         : round(signed_delta, 6),
        "delta_exit"          : round(final_delta, 6),
        "gamma"               : round(gamma, 6),
        "theta_daily"         : round(theta_daily, 4),
        "theta_per_hour"      : round(theta_per_hr, 4),
        "vega"                : round(vega, 4),
        "vix_at_entry"        : round(vix_at_entry, 2) if vix_at_entry else None,
        "vix_at_exit"         : round(vix_at_exit, 2) if vix_at_exit else None,
        "vix_change"          : round(vix_change, 4),
        "delta_effect"        : round(delta_effect_dyn, 4),
        "gamma_effect"        : round(gamma_effect, 4),
        "gamma_effect_static" : round(gamma_effect_static, 4),
        "theta_effect"        : round(theta_effect, 4),
        "vega_effect"         : round(vega_effect, 4),
        "option_change"       : round(option_change, 4),
        "pnl_per_lot"         : round(pnl_per_lot, 4),
        "total_pnl"           : round(total_pnl, 2),
        "lot_size"            : lot_size,
        "opt_direction"       : opt_direction,
        "breakeven_dS"        : breakeven_dS,
        "trade_direction"     : trade_direction,
    }


def _solve_breakeven(delta, gamma, const_term):
    """Solve delta*x + 0.5*gamma*x^2 + const_term = 0 for x."""
    if gamma == 0:
        if delta == 0:
            return None
        return round(-const_term / delta, 4)
    try:
        def f(x): return delta * x + 0.5 * gamma * x * x + const_term
        # Search in range [-5000, 5000] (NIFTY points)
        root = brentq(f, -5000, 5000)
        return round(root, 4)
    except Exception:
        # Try simple quadratic
        try:
            disc = delta**2 - 4 * (0.5 * gamma) * const_term
            if disc < 0:
                return None
            r1 = (-delta + math.sqrt(disc)) / (2 * 0.5 * gamma)
            r2 = (-delta - math.sqrt(disc)) / (2 * 0.5 * gamma)
            candidates = [r for r in [r1, r2] if abs(r) < 10000]
            return round(min(candidates, key=abs), 4) if candidates else None
        except Exception:
            return None

# ═══════════════════════════════════════════════════════════════════════════════
#  CSV / YFINANCE LOADERS  (same as v3)
# ═══════════════════════════════════════════════════════════════════════════════

_DT_CANDS = ["datetime","date","time","timestamp","date/time","date time"]
_O_CANDS  = ["open","o"]
_H_CANDS  = ["high","h"]
_L_CANDS  = ["low","l"]
_C_CANDS  = ["close","ltp","last","c"]
_V_CANDS  = ["volume","vol","v"]

def _find_col(columns, candidates):
    lmap = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lmap:
            return lmap[cand]
    return None

def load_csv(path, label="CSV"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n  ❌  File not found: '{path}'\n")
    print(f"\n  Loading {label}: {path} …")
    df = pd.read_csv(path)
    dt_col = _find_col(df.columns, _DT_CANDS)
    o_col  = _find_col(df.columns, _O_CANDS)
    h_col  = _find_col(df.columns, _H_CANDS)
    l_col  = _find_col(df.columns, _L_CANDS)
    c_col  = _find_col(df.columns, _C_CANDS)
    v_col  = _find_col(df.columns, _V_CANDS)
    rename = {dt_col: "__dt", o_col: "Open", h_col: "High",
              l_col: "Low",   c_col: "Close"}
    if v_col: rename[v_col] = "Volume"
    df = df[list(rename.keys())].rename(columns=rename)
    formats_to_try = ["%d-%m-%Y %H:%M","%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M",
                      "%d-%m-%Y %H:%M:%S","%m/%d/%Y %H:%M","%Y-%m-%dT%H:%M:%S"]
    parsed = None
    for fmt_ in formats_to_try:
        attempt = pd.to_datetime(df["__dt"], format=fmt_, errors="coerce")
        if attempt.notna().mean() > 0.95:
            parsed = attempt; break
    if parsed is None:
        parsed = pd.to_datetime(df["__dt"], errors="coerce")
    df["__dt"] = parsed
    df = df.dropna(subset=["__dt"]).set_index("__dt")
    df.index.name = "Datetime"
    if df.index.tz is None:
        df.index = df.index.tz_localize(IST, ambiguous="infer", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert(IST)
    df.sort_index(inplace=True)
    for col in ["Open","High","Low","Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df.get("Volume", 0), errors="coerce").fillna(0)
    df.dropna(subset=["Open","High","Low","Close"], inplace=True)
    tod = df.index.time
    df  = df[(tod >= MARKET_OPEN) & (tod <= MARKET_CLOSE)].copy()
    print(f"    Clean rows: {len(df)}  |  {fmt(df.index[0])} → {fmt(df.index[-1])}")
    return df

def _normalise_yf_df(raw):
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).strip().title() for c in raw.columns]
    if "Adj Close" in raw.columns and "Close" not in raw.columns:
        raw = raw.rename(columns={"Adj Close": "Close"})
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in raw.columns]
    df = raw[keep].copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df

def load_yfinance(ticker, interval, label="yfinance"):
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("\n  ❌  yfinance not installed. Run: pip install yfinance\n")
    print(f"\n  Downloading {label}: ticker={ticker}  interval={interval} …")
    dl_kwargs = dict(tickers=ticker, interval=interval, auto_adjust=True, progress=False)
    if YF_START is not None:
        dl_kwargs["start"] = YF_START
        if YF_END: dl_kwargs["end"] = YF_END
    else:
        dl_kwargs["period"] = YF_PERIOD
    raw = yf.download(**dl_kwargs)
    if raw is None or raw.empty:
        raise ValueError(f"\n  ❌  yfinance returned no data for ticker='{ticker}' interval='{interval}'.\n")
    df = _normalise_yf_df(raw)
    df.index.name = "Datetime"
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")
    df.index = df.index.tz_convert(IST)
    df.sort_index(inplace=True)
    for col in ["Open","High","Low","Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df.dropna(subset=["Open","High","Low","Close"], inplace=True)
    tod = df.index.time
    df  = df[(tod >= MARKET_OPEN) & (tod <= MARKET_CLOSE)].copy()
    if df.empty:
        raise ValueError(f"\n  ❌  No data left after market-hours filter for {ticker}.\n")
    print(f"    Clean rows: {len(df)}  |  {fmt(df.index[0])} → {fmt(df.index[-1])}")
    return df

def load_vix():
    """Load VIX data: from CSV or yfinance. Returns Series indexed by date."""
    if VIX_CSV and os.path.exists(VIX_CSV):
        print(f"\n  Loading VIX from CSV: {VIX_CSV}")
        dv = pd.read_csv(VIX_CSV)
        dt_col = _find_col(dv.columns, _DT_CANDS)
        c_col  = _find_col(dv.columns, _C_CANDS)
        if dt_col and c_col:
            dv[dt_col] = pd.to_datetime(dv[dt_col], errors="coerce")
            dv = dv.dropna(subset=[dt_col]).set_index(dt_col)
            dv.index = dv.index.normalize()
            return dv[c_col].rename("VIX")
    # Try yfinance
    try:
        import yfinance as yf
        print(f"\n  Downloading VIX from yfinance: {YF_VIX_TICKER}")
        raw = yf.download(YF_VIX_TICKER, period=YF_PERIOD if YF_START is None else None,
                          start=YF_START, end=YF_END, auto_adjust=True, progress=False)
        if raw is not None and not raw.empty:
            raw = _normalise_yf_df(raw)
            raw.index = raw.index.normalize()
            if raw.index.tz is not None:
                raw.index = raw.index.tz_convert(IST).normalize()
            return raw["Close"].rename("VIX")
    except Exception as e:
        print(f"  ⚠  VIX download failed: {e}")
    print("  ⚠  No VIX data — Vega effect will be 0.")
    return None

def fetch_data():
    src = DATA_SOURCE.strip().lower()
    if src == "csv":
        df = load_csv(BASE_CSV, label="Base TF CSV")
        htf_source = HTF_CSV
    elif src == "yfinance":
        df = load_yfinance(YF_TICKER, YF_BASE_INTERVAL, label="Base TF (yfinance)")
        htf_source = YF_HTF_INTERVAL
    else:
        raise ValueError(f"\n  ❌  Unknown DATA_SOURCE='{DATA_SOURCE}'\n")

    df = compute_supertrend(df, ST_PERIOD, ST_MULTIPLIER)
    df.dropna(inplace=True)

    htf_enabled = False
    if htf_source is not None:
        try:
            if src == "csv":
                df_htf = load_csv(htf_source, label="HTF CSV")
            else:
                df_htf = load_yfinance(YF_TICKER, YF_HTF_INTERVAL, label="HTF (yfinance)")
            df_htf = compute_supertrend(df_htf, HTF_ST_PERIOD, HTF_ST_MULTIPLIER)
            df_htf.dropna(inplace=True)
            htf_bull = df_htf["ST_bull"].reindex(df.index, method="ffill")
            htf_val  = df_htf["ST_val"].reindex(df.index, method="ffill")
            if htf_bull.notna().mean() >= 0.5:
                df["HTF_bull"] = htf_bull.ffill()
                df["HTF_val"]  = htf_val.ffill()
                df.dropna(subset=["HTF_bull","HTF_val"], inplace=True)
                htf_enabled = True
                print(f"  ✅  HTF ENABLED — ST({HTF_ST_PERIOD},{HTF_ST_MULTIPLIER})")
        except FileNotFoundError:
            print("  ⚠  HTF file not found — HTF filter DISABLED.")

    if not htf_enabled:
        df["HTF_bull"] = True
        df["HTF_val"]  = df["ST_val"]

    vix_series = load_vix()
    return df, htf_enabled, vix_series

# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE  (with option P&L per trade)
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(df, htf_enabled, vix_series):
    trades = []

    in_trade = False; direction = None; entry_price = None
    entry_time = None; peak = None; hard_sl_price = None
    flip_candle = -999

    closes    = df["Close"].values.astype(float)
    highs     = df["High"].values.astype(float)
    lows      = df["Low"].values.astype(float)
    st_vals   = df["ST_val"].values.astype(float)
    htf_vals  = df["HTF_val"].values.astype(float)
    st_bulls  = df["ST_bull"].values.astype(bool)
    htf_bulls = df["HTF_bull"].values.astype(bool)
    volumes   = df["Volume"].values.astype(float)
    times     = df.index

    exit_st   = htf_vals if (EXIT_USE_HTF_ST and htf_enabled) else st_vals
    vol_avg   = pd.Series(volumes).rolling(VOLUME_LOOKBACK).mean().values
    daily_trades = {}

    def in_window(t):
        if not TRADE_WINDOW_ENABLED: return True
        return any(s <= t <= e for s, e in TRADE_WINDOWS)

    def get_vix(ts):
        if vix_series is None: return None
        day = ts.normalize() if hasattr(ts, "normalize") else pd.Timestamp(ts).normalize()
        try:
            return float(vix_series.asof(day))
        except Exception:
            return None

    def record(dir_, e_time, e_price, x_time, x_price, pk, sl_at_exit, reason, hard_sl):
        pnl = round(((x_price - e_price) / e_price * 100) if dir_ == "long"
                    else ((e_price - x_price) / e_price * 100), 4)
        hold_mins = int((x_time - e_time).total_seconds() // 60)
        hold_str  = f"{hold_mins // 60}H:{hold_mins % 60:02d}M"
        holding_hours = hold_mins / 60.0

        # ── Greeks / Option P&L ───────────────────────────────────────────
        vix_e = get_vix(e_time)
        vix_x = get_vix(x_time)
        opt = compute_option_pnl(
            entry_price_S  = e_price,
            exit_price_S   = x_price,
            holding_hours  = holding_hours,
            trade_direction= dir_,          # "long" or "short" underlying → sets CE/PE delta sign
            vix_at_entry   = vix_e,
            vix_at_exit    = vix_x,
        )

        trades.append({
            "Date"            : e_time.strftime("%Y-%m-%d"),
            "Direction"       : "Long  ↑" if dir_ == "long" else "Short ↓",
            "Entry Time"      : fmt(e_time),
            "Entry Price"     : round(e_price, 2),
            "Hard SL Price"   : round(hard_sl, 2) if hard_sl else "OFF",
            "Exit ST Line"    : round(sl_at_exit, 2),
            "Exit Time"       : fmt(x_time),
            "Exit Price"      : round(x_price, 2),
            "Peak"            : round(pk, 2),
            "Holding Mins"    : hold_mins,
            "Holding Time"    : hold_str,
            "Points Captured" : round(abs(x_price - e_price), 4),
            "P&L %"           : pnl,
            "Exit Reason"     : reason,
            "Result"          : "WIN" if pnl > 0 else "LOSS",
            # Option fields
            "Opt_dS"          : opt["dS"],
            "Opt_DeltaEntry"  : opt["delta_entry"],
            "Opt_DeltaExit"   : opt["delta_exit"],
            "Opt_Gamma"       : opt["gamma"],
            "Opt_Theta_daily" : opt["theta_daily"],
            "Opt_Vega"        : opt["vega"],
            "Opt_VIX_Entry"   : opt["vix_at_entry"],
            "Opt_VIX_Exit"    : opt["vix_at_exit"],
            "Opt_VIX_Chg"     : opt["vix_change"],
            "Opt_DeltaEffect" : opt["delta_effect"],
            "Opt_GammaEffect" : opt["gamma_effect"],
            "Opt_ThetaEffect" : opt["theta_effect"],
            "Opt_VegaEffect"  : opt["vega_effect"],
            "Opt_Change"      : opt["option_change"],
            "Opt_PnL_PerLot"  : opt["pnl_per_lot"],
            "Opt_PnL_Total"   : opt["total_pnl"],
            "Opt_BreakevenDS" : opt["breakeven_dS"],
            "Opt_LotSize"     : opt["lot_size"],
        })

    def open_trade(dir_, price, time_, high_, low_, date_key):
        nonlocal in_trade, direction, entry_price, entry_time, peak, hard_sl_price
        in_trade      = True; direction = dir_; entry_price = price
        entry_time    = time_; peak = high_ if dir_ == "long" else low_
        hard_sl_price = (round(price * (1 - HARD_SL_PCT/100), 4) if dir_ == "long"
                         else round(price * (1 + HARD_SL_PCT/100), 4)) if HARD_SL_ENABLED else None
        daily_trades[date_key] = daily_trades.get(date_key, 0) + 1

    for i in range(1, len(df)):
        c_close    = closes[i]; c_high = highs[i]; c_low = lows[i]
        c_time     = times[i];  c_tod  = c_time.time(); c_st = st_vals[i]
        c_exit_st  = exit_st[i]
        c_bull     = st_bulls[i]; prev_bull = st_bulls[i-1]
        c_htf_bull = htf_bulls[i]; date_key = c_time.date()
        prev_close = closes[i-1]

        flipped_bull = (not prev_bull) and c_bull
        flipped_bear = prev_bull and (not c_bull)
        if flipped_bull or flipped_bear: flip_candle = i

        # EOD
        if in_trade and c_tod >= MARKET_CLOSE:
            pk_f = max(peak, c_high) if direction == "long" else min(peak, c_low)
            record(direction, entry_time, entry_price, c_time, c_close,
                   pk_f, c_exit_st, "EOD Exit", hard_sl_price)
            in_trade = False; continue

        if c_tod < MARKET_OPEN or c_tod >= MARKET_CLOSE: continue

        if in_trade:
            if direction == "long":
                if c_high > peak: peak = c_high
                hard_hit = HARD_SL_ENABLED and hard_sl_price and c_low <= hard_sl_price
                st_hit   = (c_close < c_exit_st) or flipped_bear if CLOSE_BASED_EXIT \
                           else c_low <= c_exit_st or flipped_bear
                if hard_hit or st_hit:
                    if hard_hit:   reason = f"Hard SL ({HARD_SL_PCT}%)"; exit_px = hard_sl_price
                    elif (CLOSE_BASED_EXIT and c_close < c_exit_st) or \
                         (not CLOSE_BASED_EXIT and c_low <= c_exit_st):
                        reason = "HTF ST Exit" if (EXIT_USE_HTF_ST and htf_enabled) else "ST Stop Loss"
                        exit_px = c_exit_st
                    else:
                        reason = "ST Flip Bear"; exit_px = c_exit_st
                    record("long", entry_time, entry_price, c_time, round(exit_px,2),
                           peak, c_exit_st, reason, hard_sl_price)
                    in_trade = False
                    if (not hard_hit and not c_bull and
                            (not htf_enabled or not c_htf_bull) and
                            in_window(c_tod) and
                            daily_trades.get(date_key,0) < MAX_TRADES_PER_DAY and
                            c_close < prev_close):
                        open_trade("short", c_close, c_time, c_high, c_low, date_key)
            elif direction == "short":
                if c_low < peak: peak = c_low
                hard_hit = HARD_SL_ENABLED and hard_sl_price and c_high >= hard_sl_price
                st_hit   = (c_close > c_exit_st) or flipped_bull if CLOSE_BASED_EXIT \
                           else c_high >= c_exit_st or flipped_bull
                if hard_hit or st_hit:
                    if hard_hit:   reason = f"Hard SL ({HARD_SL_PCT}%)"; exit_px = hard_sl_price
                    elif (CLOSE_BASED_EXIT and c_close > c_exit_st) or \
                         (not CLOSE_BASED_EXIT and c_high >= c_exit_st):
                        reason = "HTF ST Exit" if (EXIT_USE_HTF_ST and htf_enabled) else "ST Stop Loss"
                        exit_px = c_exit_st
                    else:
                        reason = "ST Flip Bull"; exit_px = c_exit_st
                    record("short", entry_time, entry_price, c_time, round(exit_px,2),
                           peak, c_exit_st, reason, hard_sl_price)
                    in_trade = False
                    if (not hard_hit and c_bull and
                            (not htf_enabled or c_htf_bull) and
                            in_window(c_tod) and
                            daily_trades.get(date_key,0) < MAX_TRADES_PER_DAY and
                            c_close > prev_close):
                        open_trade("long", c_close, c_time, c_high, c_low, date_key)
            continue

        # Entry filters
        if not in_window(c_tod):                                                continue
        if (i - flip_candle) < CONFIRM_CANDLES:                                continue
        if MIN_GAP_ENABLED and abs(c_close-c_st)/c_close*100 < MIN_GAP_PCT:    continue
        if (VOLUME_FILTER_ENABLED and not np.isnan(vol_avg[i])
                and vol_avg[i] > 0
                and volumes[i] < VOLUME_MULTIPLIER * vol_avg[i]):               continue
        if (MAX_TRADES_PER_DAY_ENABLED and
                daily_trades.get(date_key,0) >= MAX_TRADES_PER_DAY):           continue

        if c_bull and (not htf_enabled or c_htf_bull):
            if c_close > prev_close:
                open_trade("long",  c_close, c_time, c_high, c_low, date_key)
        elif not c_bull and (not htf_enabled or not c_htf_bull):
            if c_close < prev_close:
                open_trade("short", c_close, c_time, c_high, c_low, date_key)

    return trades

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_daily_summary(trades):
    if not trades: return pd.DataFrame()
    df_t = pd.DataFrame(trades); rows = []
    for date, grp in df_t.groupby("Date"):
        total = len(grp); wins = (grp["P&L %"] > 0).sum(); pnl = grp["P&L %"].sum()
        wp = grp.loc[grp["P&L %"] > 0,  "Points Captured"].sum()
        lp = grp.loc[grp["P&L %"] <= 0, "Points Captured"].sum()
        opt_pnl = grp["Opt_PnL_Total"].sum()
        rows.append({
            "Date"           : date,
            "Total Trades"   : total,
            "Longs"          : grp["Direction"].str.contains("Long").sum(),
            "Shorts"         : grp["Direction"].str.contains("Short").sum(),
            "Winners"        : wins,
            "Losers"         : total - wins,
            "Win Rate %"     : round(wins / total * 100, 1),
            "Total P&L %"    : round(pnl, 4),
            "Best Trade %"   : round(grp["P&L %"].max(), 4),
            "Worst Trade %"  : round(grp["P&L %"].min(), 4),
            "Points Captured": round(wp, 2),
            "Points Lost"    : round(lp, 2),
            "Net Points"     : round(wp - lp, 2),
            "Opt P&L (₹)"    : round(opt_pnl, 2),
            "Avg Hold Time"  : mins_to_hhmm(grp["Holding Mins"].mean()),
            "Day Result"     : "✅ Profit" if pnl > 0 else "❌ Loss" if pnl < 0 else "⚖ Flat",
        })
    return pd.DataFrame(rows)

def build_time_slot_summary(trades):
    if not trades: return pd.DataFrame()
    SLOTS = [
        ("09:15","09:30"),("09:30","10:00"),("10:00","10:30"),("10:30","11:00"),
        ("11:00","11:30"),("11:30","12:00"),("12:00","12:30"),("12:30","13:00"),
        ("13:00","13:30"),("13:30","14:00"),("14:00","14:30"),("14:30","15:00"),
        ("15:00","15:15"),
    ]
    df_t = pd.DataFrame(trades)
    df_t["Entry_tod"] = pd.to_datetime(df_t["Entry Time"], format="%Y-%m-%d %H:%M").dt.strftime("%H:%M")
    rows = []
    for s, e in SLOTS:
        grp = df_t[(df_t["Entry_tod"] >= s) & (df_t["Entry_tod"] < e)]
        if grp.empty:
            rows.append({"Time Slot":f"{s}–{e}","Trades":0,"Winners":0,"Losers":0,
                         "Win Rate %":"—","Total P&L %":0.0,"Avg P&L %":0.0,
                         "Best %":0.0,"Worst %":0.0,"Opt P&L (₹)":0.0,"Verdict":"—"})
            continue
        total = len(grp); wins = (grp["P&L %"] > 0).sum()
        pnl = grp["P&L %"].sum(); wr = wins / total * 100
        rows.append({
            "Time Slot"  : f"{s}–{e}",
            "Trades"     : total,
            "Winners"    : wins,
            "Losers"     : total - wins,
            "Win Rate %" : round(wr, 1),
            "Total P&L %": round(pnl, 4),
            "Avg P&L %"  : round(grp["P&L %"].mean(), 4),
            "Best %"     : round(grp["P&L %"].max(), 4),
            "Worst %"    : round(grp["P&L %"].min(), 4),
            "Opt P&L (₹)": round(grp["Opt_PnL_Total"].sum(), 2),
            "Verdict"    : "🟢 Trade" if pnl > 0 and wr >= 50 else "🔴 Avoid",
        })
    return pd.DataFrame(rows)

def build_option_summary(trades):
    """Build comprehensive option analytics summary."""
    if not trades: return {}
    df_t = pd.DataFrame(trades)

    total_opt_pnl  = df_t["Opt_PnL_Total"].sum()
    win_opt_pnl    = df_t.loc[df_t["Opt_PnL_Total"] > 0, "Opt_PnL_Total"].sum()
    loss_opt_pnl   = df_t.loc[df_t["Opt_PnL_Total"] <= 0, "Opt_PnL_Total"].sum()
    opt_wins       = (df_t["Opt_PnL_Total"] > 0).sum()
    opt_losses     = (df_t["Opt_PnL_Total"] <= 0).sum()
    best_opt_trade = df_t.loc[df_t["Opt_PnL_Total"].idxmax()]
    worst_opt_trade= df_t.loc[df_t["Opt_PnL_Total"].idxmin()]

    avg_delta_eff  = df_t["Opt_DeltaEffect"].mean()
    avg_gamma_eff  = df_t["Opt_GammaEffect"].mean()
    avg_theta_eff  = df_t["Opt_ThetaEffect"].mean()
    avg_vega_eff   = df_t["Opt_VegaEffect"].mean()
    total_theta    = df_t["Opt_ThetaEffect"].sum() * OPTION_LOT_SIZE
    total_delta_c  = df_t["Opt_DeltaEffect"].sum() * OPTION_LOT_SIZE
    total_gamma_c  = df_t["Opt_GammaEffect"].sum() * OPTION_LOT_SIZE
    total_vega_c   = df_t["Opt_VegaEffect"].sum() * OPTION_LOT_SIZE

    avg_be = df_t["Opt_BreakevenDS"].dropna().mean() if df_t["Opt_BreakevenDS"].notna().any() else None

    return {
        "df_t"             : df_t,
        "total_opt_pnl"    : round(total_opt_pnl, 2),
        "win_opt_pnl"      : round(win_opt_pnl, 2),
        "loss_opt_pnl"     : round(loss_opt_pnl, 2),
        "opt_wins"         : int(opt_wins),
        "opt_losses"       : int(opt_losses),
        "opt_win_rate"     : round(opt_wins / len(df_t) * 100, 1),
        "best_opt_trade"   : best_opt_trade,
        "worst_opt_trade"  : worst_opt_trade,
        "avg_delta_eff"    : round(avg_delta_eff, 4),
        "avg_gamma_eff"    : round(avg_gamma_eff, 4),
        "avg_theta_eff"    : round(avg_theta_eff, 4),
        "avg_vega_eff"     : round(avg_vega_eff, 4),
        "total_theta_cost" : round(total_theta, 2),
        "total_delta_contrib": round(total_delta_c, 2),
        "total_gamma_contrib": round(total_gamma_c, 2),
        "total_vega_contrib" : round(total_vega_c, 2),
        "avg_breakeven_dS" : round(avg_be, 2) if avg_be else None,
        "total_trades"     : len(df_t),
    }

def build_advanced_option_metrics(trades):
    if not trades: return ""
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(trades)
    if df.empty or "Opt_PnL_Total" not in df.columns: return ""
    
    SEP = "=" * 69
    DASH = "-" * 69
    lines = [
        "", SEP,
        "  OPTION P/L ANALYTICS METRICS FRAMEWORK",
        SEP
    ]
    
    # helper for safe division
    def safe_div(a, b): return a / b if b != 0 else 0.0
    def format_money(v): return f"₹{v:,.2f}" if not pd.isna(v) else "N/A"

    returns = df["Opt_PnL_Total"]
    wins = df[returns > 0]
    losses = df[returns < 0]
    
    # ── SEC 3: BASIC PERFORMANCE ────────────────────────
    net_pnl = returns.sum()
    avg_trade = returns.mean()
    med_trade = returns.median()
    win_rate = safe_div(len(wins), len(returns))
    avg_win = wins["Opt_PnL_Total"].mean() if len(wins) > 0 else 0
    avg_loss = losses["Opt_PnL_Total"].mean() if len(losses) > 0 else 0
    rr_ratio = abs(safe_div(avg_win, avg_loss))
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    gross_profit = wins["Opt_PnL_Total"].sum()
    gross_loss = abs(losses["Opt_PnL_Total"].sum())
    pf = safe_div(gross_profit, gross_loss)
    
    lines += [
        "  SECTION 3 — BASIC PERFORMANCE METRICS", DASH,
        f"  Total Net P/L        : {format_money(net_pnl)}",
        f"  Average Trade P/L    : {format_money(avg_trade)}",
        f"  Median Trade P/L     : {format_money(med_trade)}",
        f"  Win Rate             : {win_rate*100:.2f}%",
        f"  Average Winner       : {format_money(avg_win)}",
        f"  Average Loser        : {format_money(avg_loss)}",
        f"  Risk Reward Ratio    : {rr_ratio:.2f}",
        f"  Expectancy           : {format_money(expectancy)}",
        f"  Profit Factor        : {pf:.2f}",
        SEP
    ]
    
    # ── SEC 4: RISK ─────────────────────────────────────
    cum_equity = returns.cumsum()
    peak = cum_equity.cummax()
    drawdown = peak - cum_equity
    mdd = drawdown.max()
    mdd_pct = (mdd / peak.max()) * 100 if peak.max() > 0 else 0
    ulcer = np.sqrt((drawdown**2).mean())
    std_dev = returns.std()
    downside = losses["Opt_PnL_Total"].std() if len(losses) > 1 else 0
    sharpe = safe_div(returns.mean(), std_dev)
    sortino = safe_div(returns.mean(), downside)
    calmar = safe_div(returns.mean() * 252, mdd)  # simple annualized proxy
    
    lines += [
        "  SECTION 4 — RISK METRICS", DASH,
        f"  Maximum Drawdown (MDD) : {format_money(mdd)}",
        f"  Max Drawdown %       : {mdd_pct:.2f}%",
        f"  Ulcer Index          : {ulcer:.2f}",
        f"  Std Dev of Returns   : {std_dev:.2f}",
        f"  Downside Deviation   : {downside:.2f}",
        f"  Sharpe Ratio         : {sharpe:.2f}",
        f"  Sortino Ratio        : {sortino:.2f}",
        f"  Calmar Ratio (Proxy) : {calmar:.4f}",
        SEP
    ]
    
    # ── SEC 5: GREEKS EXPOSURE ──────────────────────────
    lot = df["Opt_LotSize"].iloc[0] if len(df)>0 and "Opt_LotSize" in df.columns else OPTION_LOT_SIZE
    avg_delta_exp = (df["Opt_DeltaEntry"].abs() * lot).mean()
    gamma_exp = (df["Opt_Gamma"] * lot).sum() if "Opt_Gamma" in df.columns else 0
    theta_per_hr = df["Opt_Theta_daily"].iloc[0] / TRADING_HOURS_PER_DAY if "Opt_Theta_daily" in df.columns and len(df)>0 else 0
    vega_exp = (df["Opt_Vega"] * lot).sum() if "Opt_Vega" in df.columns else 0
    
    delta_drift = (df["Opt_DeltaExit"].abs() - df["Opt_DeltaEntry"].abs()).mean() if "Opt_DeltaExit" in df.columns else 0
    gamma_eff_pct = safe_div(df["Opt_GammaEffect"].abs().sum(), df["Opt_Change"].abs().sum()) * 100
    theta_eff_pct = safe_div(df["Opt_ThetaEffect"].abs().sum(), df["Opt_Change"].abs().sum()) * 100
    
    lines += [
        "  SECTION 5 — GREEKS EXPOSURE METRICS", DASH,
        f"  Avg Delta Exposure   : {avg_delta_exp:.4f} per trade",
        f"  Gamma Exposure (GEX) : {gamma_exp:.6f}",
        f"  Theta Exposure / Hr  : {theta_per_hr:.4f}",
        f"  Vega Exposure        : {vega_exp:.4f}",
        f"  Avg Delta Drift      : {delta_drift:.4f}",
        f"  Convexity Contrib %  : {gamma_eff_pct:.2f}% (abs Gamma/abs Total)",
        f"  Time Decay Contrib % : {theta_eff_pct:.2f}% (abs Theta/abs Total)",
        SEP
    ]

    # ── SEC 6: BREAKEVEN ────────────────────────────────
    be = df["Opt_BreakevenDS"].dropna().mean() if "Opt_BreakevenDS" in df.columns else float('nan')
    lines += [
        "  SECTION 6 — BREAK-EVEN ANALYSIS", DASH,
        f"  Avg Req. Underlying Move : {be:.2f} pts" if not np.isnan(be) else "  Avg Req. Underlying Move : N/A",
        SEP
    ]
    
    # ── SEC 7: DISTRIBUTION ─────────────────────────────
    skew = returns.skew()
    kurt = returns.kurtosis()
    left_tail = returns.quantile(0.05) if len(returns) > 0 else 0
    right_tail = returns.quantile(0.95) if len(returns) > 0 else 0
    
    lines += [
        "  SECTION 7 — DISTRIBUTION METRICS", DASH,
        f"  Skewness             : {skew:.2f}",
        f"  Kurtosis             : {kurt:.2f}",
        f"  Left Tail Risk (5%)  : {format_money(left_tail)}",
        f"  Right Tail Gain (95%): {format_money(right_tail)}",
        SEP
    ]
    
    # ── SEC 8: TIME EFFICIENCY ──────────────────────────
    hold_hrs = df["Holding Mins"].sum() / 60.0 if "Holding Mins" in df.columns else 0
    pnl_per_hr = safe_div(net_pnl, hold_hrs)
    theta_eff = safe_div(returns.sum(), (df["Opt_ThetaEffect"].abs().sum() * lot))
    move_eff = safe_div(returns.sum(), df["Opt_dS"].abs().sum())
    gamma_util = safe_div(df["Opt_GammaEffect"].sum(), df["Opt_Change"].sum())

    lines += [
        "  SECTION 8 — TIME EFFICIENCY METRICS", DASH,
        f"  Profit per Hour      : {format_money(pnl_per_hr)}",
        f"  Theta Efficiency     : {theta_eff:.2f} P&L per unit Theta Cost",
        f"  Move Efficiency      : {move_eff:.2f} P&L per Pts Move",
        f"  Gamma Utilization    : {gamma_util:.2f}x (Net Gamma/Net Change)",
        SEP
    ]
    
    # ── SEC 9: CAPITAL EFFICIENCY ───────────────────────
    lines += [
        "  SECTION 9 — CAPITAL EFFICIENCY", DASH,
        "  Return on Capital (ROC) : [Skipped - Requires Initial Capital Data]",
        "  Margin Utilization %    : [Skipped - Requires Margin Data]",
        SEP
    ]
    
    # ── SEC 10: STRATEGY STABILITY ──────────────────────
    roll_sharpe = returns.rolling(20).apply(lambda x: safe_div(x.mean(), x.std())).dropna()
    avg_roll_sharpe = roll_sharpe.mean() if not roll_sharpe.empty else 0
    roll_win = (returns > 0).rolling(20).mean().dropna()
    avg_roll_win = roll_win.mean() * 100 if not roll_win.empty else 0
    
    x_eq = np.arange(len(cum_equity))
    if len(x_eq) > 1:
        slope, intercept = np.polyfit(x_eq, cum_equity, 1)
        r2 = np.corrcoef(x_eq, cum_equity)[0, 1]**2
    else:
        slope, r2 = 0, 0
        
    is_win = returns > 0
    streaks = is_win.ne(is_win.shift()).cumsum()
    cons_wins = is_win.groupby(streaks).sum().max()
    cons_losses = (~is_win).groupby(streaks).sum().max()
    
    lines += [
        "  SECTION 10 — STRATEGY STABILITY METRICS", DASH,
        f"  Rolling 20-Trade Sharpe: {avg_roll_sharpe:.2f}",
        f"  Rolling Win Rate       : {avg_roll_win:.2f}%",
        f"  Equity Curve Slope     : {slope:.2f}",
        f"  Equity Curve R²        : {r2:.2f}",
        f"  Consecutive Wins       : {cons_wins}",
        f"  Consecutive Losses     : {cons_losses}",
        SEP
    ]
    
    # ── SEC 11: SENSITIVITY ANALYSIS ────────────────────
    
    def sim_pnl(ds_mult=1.0, ds_add=0.0, iv_add=0.0, hr_add=0.0):
        s_ds = df["Opt_dS"] * ds_mult + ds_add
        s_g  = df["Opt_Gamma"] if "Opt_Gamma" in df.columns else 0
        s_d  = df["Opt_DeltaEntry"] if "Opt_DeltaEntry" in df.columns else OPTION_DELTA
        
        dt_hrs = (df["Holding Mins"]/60.0) + hr_add
        dt_hrs = dt_hrs.clip(lower=0)
        theta_hr = df["Opt_Theta_daily"] / TRADING_HOURS_PER_DAY if "Opt_Theta_daily" in df.columns else (OPTION_THETA / TRADING_HOURS_PER_DAY)
        s_t = theta_hr * dt_hrs
        
        s_v = 0.0
        if "Opt_Vega" in df.columns and "Opt_VIX_Chg" in df.columns:
            s_v = df["Opt_Vega"] * (df["Opt_VIX_Chg"] + iv_add)
            
        s_opt_change = (s_d * s_ds) + (0.5 * s_g * (s_ds**2)) + s_t + s_v
        
        # P&L per lot, considering option buying vs selling direction
        sign = 1 if OPTION_DIRECTION == "long" else -1
        
        return (s_opt_change * sign * lot).sum()

    lines += [
        "  SECTION 11 — SENSITIVITY ANALYSIS (Avg Trade Impact)", DASH,
        f"  Base Strategy Net P/L : {format_money(net_pnl)}"
    ]
    
    try:
        sim_up10 = sim_pnl(ds_mult=1.1)
        sim_dn10 = sim_pnl(ds_mult=0.9)
        sim_up20 = sim_pnl(ds_mult=1.2)
        sim_dn20 = sim_pnl(ds_mult=0.8)
        sim_up30 = sim_pnl(ds_mult=1.3)
        sim_dn30 = sim_pnl(ds_mult=0.7)
        sim_iv5_up = sim_pnl(iv_add=5.0)
        sim_iv10_up = sim_pnl(iv_add=10.0)
        sim_iv5_dn = sim_pnl(iv_add=-5.0)
        sim_iv10_dn = sim_pnl(iv_add=-10.0)
        sim_hold_1h = sim_pnl(hr_add=1.0)
        sim_hold_m1h = sim_pnl(hr_add=-1.0)

        lines += [
            f"  P/L if ΔS +10%        : {format_money(sim_up10)} (Δ: {format_money(sim_up10-net_pnl)})",
            f"  P/L if ΔS -10%        : {format_money(sim_dn10)} (Δ: {format_money(sim_dn10-net_pnl)})",
            f"  P/L if ΔS +20%        : {format_money(sim_up20)} (Δ: {format_money(sim_up20-net_pnl)})",
            f"  P/L if ΔS -20%        : {format_money(sim_dn20)} (Δ: {format_money(sim_dn20-net_pnl)})",
            f"  P/L if ΔS +30%        : {format_money(sim_up30)} (Δ: {format_money(sim_up30-net_pnl)})",
            f"  P/L if ΔS -30%        : {format_money(sim_dn30)} (Δ: {format_money(sim_dn30-net_pnl)})",
            f"  P/L if IV +5%         : {format_money(sim_iv5_up)} (Δ: {format_money(sim_iv5_up-net_pnl)})",
            f"  P/L if IV -5%         : {format_money(sim_iv5_dn)} (Δ: {format_money(sim_iv5_dn-net_pnl)})",
            f"  P/L if IV +10%        : {format_money(sim_iv10_up)} (Δ: {format_money(sim_iv10_up-net_pnl)})",
            f"  P/L if IV -10%        : {format_money(sim_iv10_dn)} (Δ: {format_money(sim_iv10_dn-net_pnl)})",
            f"  P/L if Hold Time +1h  : {format_money(sim_hold_1h)} (Δ: {format_money(sim_hold_1h-net_pnl)})",
            f"  P/L if Hold Time -1h  : {format_money(sim_hold_m1h)} (Δ: {format_money(sim_hold_m1h-net_pnl)})"
        ]
    except Exception as e:
        lines += [f"  Sensitivity Simulation Error: {e}"]
        
    lines.append(SEP + "\n")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDERS  (used inside HTML)
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_json(fig):
    return json.loads(fig.to_json())

def build_candle_chart(df):
    if CHART_DAYS:
        udays  = sorted(df.index.normalize().unique())
        cutoff = udays[-CHART_DAYS] if len(udays) >= CHART_DAYS else udays[0]
        df_c   = df[df.index.normalize() >= cutoff].copy()
    else:
        df_c = df.copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_c.index, open=df_c["Open"], high=df_c["High"],
        low=df_c["Low"], close=df_c["Close"], name="Price",
        increasing_line_color="#00d4aa", decreasing_line_color="#ff4757"))
    fig.add_trace(go.Scatter(
        x=df_c.index, y=df_c["ST_val"].where(df_c["ST_bull"]),
        name=f"ST Bullish", mode="lines", line=dict(color="#00d4aa", width=2.5)))
    fig.add_trace(go.Scatter(
        x=df_c.index, y=df_c["ST_val"].where(~df_c["ST_bull"]),
        name=f"ST Bearish", mode="lines", line=dict(color="#ff4757", width=2.5)))
    fig.update_layout(
        template="plotly_dark", height=520, xaxis_rangeslider_visible=False,
        margin=dict(l=50,r=20,t=40,b=40), paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
        font=dict(family="'Courier New', monospace", size=11))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"]), dict(bounds=[15.5,9.25], pattern="hour")])
    return _fig_to_json(fig)

def build_pnl_chart(df_t):
    cumulative = df_t["Opt_PnL_Total"].cumsum()
    colors = ["#00d4aa" if v >= 0 else "#ff4757" for v in df_t["Opt_PnL_Total"]]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        subplot_titles=("Option P&L per Trade (₹)", "Cumulative Option P&L (₹)"),
                        vertical_spacing=0.15)
    fig.add_trace(go.Bar(x=list(range(1,len(df_t)+1)), y=df_t["Opt_PnL_Total"],
                         marker_color=colors, name="Per Trade"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1,len(df_t)+1)), y=cumulative,
                             line=dict(color="#ffd700", width=2.5), fill="tozeroy",
                             fillcolor="rgba(255,215,0,0.1)", name="Cumulative"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=500, showlegend=False,
                      margin=dict(l=50,r=20,t=60,b=40),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"))
    return _fig_to_json(fig)

def build_greek_attribution_chart(df_t):
    delta_tot = df_t["Opt_DeltaEffect"].sum() * OPTION_LOT_SIZE
    gamma_tot = df_t["Opt_GammaEffect"].sum() * OPTION_LOT_SIZE
    theta_tot = df_t["Opt_ThetaEffect"].sum() * OPTION_LOT_SIZE
    vega_tot  = df_t["Opt_VegaEffect"].sum() * OPTION_LOT_SIZE

    fig = go.Figure(go.Bar(
        x=["Delta", "Gamma", "Theta", "Vega"],
        y=[delta_tot, gamma_tot, theta_tot, vega_tot],
        marker_color=["#00d4aa","#3d9bff","#ff4757","#ffd700"],
        text=[f"₹{v:,.0f}" for v in [delta_tot, gamma_tot, theta_tot, vega_tot]],
        textposition="outside"))
    fig.update_layout(template="plotly_dark", height=380, title="Greek P&L Attribution (₹, all trades)",
                      margin=dict(l=50,r=20,t=60,b=40),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"))
    return _fig_to_json(fig)

def build_delta_drift_chart(df_t):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,len(df_t)+1)), y=df_t["Opt_DeltaEntry"],
                             name="Delta at Entry", line=dict(color="#00d4aa", width=2)))
    fig.add_trace(go.Scatter(x=list(range(1,len(df_t)+1)), y=df_t["Opt_DeltaExit"],
                             name="Delta at Exit",  line=dict(color="#ffd700", width=2, dash="dot")))
    fig.update_layout(template="plotly_dark", height=350,
                      title="Dynamic Delta: Entry vs Exit (Gamma-adjusted)",
                      margin=dict(l=50,r=20,t=60,b=40),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"),
                      legend=dict(orientation="h", y=1.02))
    return _fig_to_json(fig)

def build_theta_vs_delta_chart(df_t):
    fig = go.Figure()
    colors = ["#00d4aa" if v >= 0 else "#ff4757" for v in df_t["Opt_PnL_Total"]]
    fig.add_trace(go.Scatter(
        x=df_t["Opt_DeltaEffect"] * OPTION_LOT_SIZE,
        y=df_t["Opt_ThetaEffect"] * OPTION_LOT_SIZE,
        mode="markers",
        marker=dict(color=colors, size=10, line=dict(width=1, color="#333")),
        text=[f"Trade {i+1}<br>ΔEffect: ₹{d*OPTION_LOT_SIZE:.0f}<br>ΘEffect: ₹{t*OPTION_LOT_SIZE:.0f}"
              for i, (d, t) in enumerate(zip(df_t["Opt_DeltaEffect"], df_t["Opt_ThetaEffect"]))],
        hovertemplate="%{text}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_dark", height=400,
                      title="Theta Cost vs Delta Gain per Trade (₹) — Green=Win, Red=Loss",
                      xaxis_title="Delta P&L (₹)", yaxis_title="Theta Cost (₹)",
                      margin=dict(l=50,r=20,t=60,b=50),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"))
    return _fig_to_json(fig)

def build_breakeven_chart(df_t):
    be_vals = df_t["Opt_BreakevenDS"].dropna()
    fig = go.Figure(go.Histogram(x=be_vals, nbinsx=15,
                                 marker_color="#3d9bff", opacity=0.8))
    fig.update_layout(template="plotly_dark", height=360,
                      title="Break-even ΔS Distribution (underlying pts needed to overcome Theta)",
                      xaxis_title="ΔS (points)", yaxis_title="Count",
                      margin=dict(l=50,r=20,t=60,b=50),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"))
    return _fig_to_json(fig)

def build_daily_opt_pnl_chart(df_day):
    colors = ["#00d4aa" if v >= 0 else "#ff4757" for v in df_day["Opt P&L (₹)"]]
    fig = go.Figure(go.Bar(x=df_day["Date"], y=df_day["Opt P&L (₹)"],
                           marker_color=colors,
                           text=[f"₹{v:,.0f}" for v in df_day["Opt P&L (₹)"]],
                           textposition="outside"))
    fig.update_layout(template="plotly_dark", height=380,
                      title="Daily Option P&L (₹)",
                      margin=dict(l=50,r=20,t=60,b=80),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"),
                      xaxis_tickangle=-30)
    return _fig_to_json(fig)

def build_vix_chart(df_t):
    has_vix = df_t["Opt_VIX_Entry"].notna().any()
    if not has_vix:
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("India VIX at Entry/Exit", "Vega P&L Effect (₹)"),
                        vertical_spacing=0.15)
    idx = list(range(1, len(df_t)+1))
    fig.add_trace(go.Scatter(x=idx, y=df_t["Opt_VIX_Entry"], name="VIX Entry",
                             line=dict(color="#ffd700", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=df_t["Opt_VIX_Exit"], name="VIX Exit",
                             line=dict(color="#ff9f43", width=2, dash="dot")), row=1, col=1)
    vega_pnl = df_t["Opt_VegaEffect"] * OPTION_LOT_SIZE
    vc = ["#00d4aa" if v >= 0 else "#ff4757" for v in vega_pnl]
    fig.add_trace(go.Bar(x=idx, y=vega_pnl, marker_color=vc, name="Vega P&L"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=480,
                      margin=dict(l=50,r=20,t=60,b=40),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"),
                      legend=dict(orientation="h", y=1.02))
    return _fig_to_json(fig)

# ═══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _row_color(pnl):
    if pnl > 0:  return "win-row"
    if pnl < 0:  return "loss-row"
    return ""

def _badge(txt, cls):
    return f'<span class="badge {cls}">{txt}</span>'

def _fmt_inr(v):
    try:
        v = float(v)
        return f"₹{v:,.2f}"
    except:
        return str(v)

def df_to_html_table(df, id_="", row_color_col=None, pnl_col=None):
    """Convert DataFrame to styled HTML table."""
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = []
    for _, row in df.iterrows():
        cls = ""
        if row_color_col and row_color_col in df.columns:
            val = row[row_color_col]
            if isinstance(val, str):
                cls = "win-row" if "WIN" in val or "Profit" in val or "🟢" in val \
                      else "loss-row" if "LOSS" in val or "Loss" in val or "🔴" in val else ""
            elif isinstance(val, (int, float)):
                cls = "win-row" if val > 0 else "loss-row" if val < 0 else ""
        if pnl_col and pnl_col in df.columns:
            try:
                cls = "win-row" if float(row[pnl_col]) > 0 else "loss-row"
            except: pass

        cells = []
        for c in df.columns:
            v = row[c]
            cell_cls = ""
            if c in ("P&L %", "Total P&L %", "Avg P&L %", "Best %", "Worst %",
                     "Best Trade %", "Worst Trade %", "Opt P&L (₹)", "Net Points"):
                try:
                    fv = float(v)
                    cell_cls = "pos-val" if fv > 0 else "neg-val" if fv < 0 else ""
                except: pass
            cells.append(f'<td class="{cell_cls}">{v}</td>')
        rows_html.append(f'<tr class="{cls}">{"".join(cells)}</tr>')

    id_attr = f'id="{id_}"' if id_ else ""
    return f"""
    <div class="table-wrap">
      <table {id_attr}>
        <thead><tr>{headers}</tr></thead>
        <tbody>{"".join(rows_html)}</tbody>
      </table>
    </div>"""

def export_html(df, trades, df_day, df_ts, htf_enabled, vix_series):
    if not trades:
        print("  ⚠  No trades — HTML report skipped.")
        return
    df_t = pd.DataFrame(trades)
    opt_sum = build_option_summary(trades)

    # Build all chart JSON
    candle_json  = build_candle_chart(df)
    pnl_json     = build_pnl_chart(df_t)
    attr_json    = build_greek_attribution_chart(df_t)
    delta_json   = build_delta_drift_chart(df_t)
    td_json      = build_theta_vs_delta_chart(df_t)
    be_json      = build_breakeven_chart(df_t)
    daily_json   = build_daily_opt_pnl_chart(df_day) if df_day is not None and not df_day.empty else None
    vix_json     = build_vix_chart(df_t)

    # ── Trade log table ───────────────────────────────────────────────────
    tl_cols = ["Date","Direction","Entry Time","Entry Price","Exit Time","Exit Price",
               "Holding Time","Points Captured","P&L %","Exit Reason","Result",
               "Opt_DeltaEffect","Opt_GammaEffect","Opt_ThetaEffect","Opt_VegaEffect",
               "Opt_Change","Opt_PnL_Total","Opt_BreakevenDS"]
    tl_display = df_t[[c for c in tl_cols if c in df_t.columns]].copy()
    tl_display.columns = [c.replace("Opt_","") for c in tl_display.columns]
    tl_html = df_to_html_table(tl_display, id_="trade-log-tbl", row_color_col="Result")

    # ── Daily summary ─────────────────────────────────────────────────────
    day_html = df_to_html_table(df_day, row_color_col="Day Result") if df_day is not None else ""

    # ── Time slot ─────────────────────────────────────────────────────────
    ts_html = df_to_html_table(df_ts, row_color_col="Verdict") if df_ts is not None else ""

    # ── Option trade detail ───────────────────────────────────────────────
    opt_cols = ["Date","Direction","Entry Time","Entry Price","Exit Time","Exit Price",
                "Opt_DeltaEntry","Opt_DeltaExit","Opt_Gamma","Opt_Theta_daily","Opt_Vega",
                "Opt_VIX_Entry","Opt_VIX_Exit","Opt_VIX_Chg",
                "Opt_DeltaEffect","Opt_GammaEffect","Opt_ThetaEffect","Opt_VegaEffect",
                "Opt_Change","Opt_PnL_PerLot","Opt_PnL_Total","Opt_BreakevenDS"]
    opt_tbl = df_t[[c for c in opt_cols if c in df_t.columns]].copy()
    opt_tbl.columns = [c.replace("Opt_","") for c in opt_tbl.columns]
    opt_html = df_to_html_table(opt_tbl, pnl_col="PnL_Total")

    # ── Stat cards ────────────────────────────────────────────────────────
    total   = len(df_t); wins = (df_t["P&L %"] > 0).sum()
    net_pts = df_t.loc[df_t["P&L %"]>0,"Points Captured"].sum() - df_t.loc[df_t["P&L %"]<=0,"Points Captured"].sum()

    def card(label, value, sub="", cls=""):
        return f"""
        <div class="stat-card {cls}">
          <div class="stat-label">{label}</div>
          <div class="stat-value">{value}</div>
          {"<div class='stat-sub'>"+sub+"</div>" if sub else ""}
        </div>"""

    strat_cards = f"""
    {card("Total Trades", total)}
    {card("Win Rate", f"{wins/total*100:.1f}%", f"{wins}W / {total-wins}L")}
    {card("Net Points", f"{net_pts:.2f}", "winning – losing pts")}
    {card("Trading Days", len(df_day) if df_day is not None else 0)}
    """

    opt_total_sign = "pos-card" if opt_sum["total_opt_pnl"] >= 0 else "neg-card"
    opt_cards = f"""
    {card("Total Option P&L", f"₹{opt_sum['total_opt_pnl']:,.2f}", f"{opt_sum['opt_wins']}W / {opt_sum['opt_losses']}L", opt_total_sign)}
    {card("Option Win Rate", f"{opt_sum['opt_win_rate']}%", "trades where option P&L > 0")}
    {card("Total Profit", f"₹{opt_sum['win_opt_pnl']:,.2f}", "from winning option trades", "pos-card")}
    {card("Total Loss", f"₹{opt_sum['loss_opt_pnl']:,.2f}", "from losing option trades", "neg-card")}
    {card("Delta Contribution", f"₹{opt_sum['total_delta_contrib']:,.2f}", "cumulative all trades")}
    {card("Gamma Contribution", f"₹{opt_sum['total_gamma_contrib']:,.2f}", "cumulative all trades")}
    {card("Theta Cost", f"₹{opt_sum['total_theta_cost']:,.2f}", "cumulative time decay paid", "neg-card")}
    {card("Vega Contribution", f"₹{opt_sum['total_vega_contrib']:,.2f}", "via VIX change")}
    {card("Avg Break-even ΔS", f"{opt_sum['avg_breakeven_dS']} pts" if opt_sum['avg_breakeven_dS'] else "—", "underlying pts to beat theta")}
    {card("Lot Size", OPTION_LOT_SIZE, f"Δ={OPTION_DELTA}  Γ={OPTION_GAMMA}")}
    {card("Theta (daily)", OPTION_THETA, "₹ decay per day")}
    {card("Vega", OPTION_VEGA, "₹ per 1-vol-pt change")}
    """

    # ── Greeks params table ───────────────────────────────────────────────
    greeks_info = f"""
    <table class="info-table">
      <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
      <tr><td>Delta (Δ)</td><td>{OPTION_DELTA}</td><td>∂C/∂S — price sensitivity per 1pt move</td></tr>
      <tr><td>Gamma (Γ)</td><td>{OPTION_GAMMA}</td><td>∂²C/∂S² — rate of delta change</td></tr>
      <tr><td>Theta (Θ)</td><td>{OPTION_THETA}/day</td><td>Time decay per day (negative = long option)</td></tr>
      <tr><td>Vega (ν)</td><td>{OPTION_VEGA}/vol-pt</td><td>IV sensitivity per 1% VIX change</td></tr>
      <tr><td>Lot Size</td><td>{OPTION_LOT_SIZE}</td><td>NIFTY standard lot size</td></tr>
      <tr><td>Option Side</td><td>{OPTION_DIRECTION.upper()}</td><td>Long = buying; Short = selling</td></tr>
      <tr><td>Dynamic Delta</td><td>{"ON" if USE_DYNAMIC_DELTA else "OFF"}</td><td>Gamma-adjusted delta per ΔS step</td></tr>
      <tr><td>Trading Hours</td><td>{TRADING_HOURS_PER_DAY}h/day</td><td>IST session length</td></tr>
    </table>
    <div class="formula-box">
      <p class="formula-title">Option Price Change Formula (Taylor Expansion)</p>
      <code>ΔC  ≈  Δ·ΔS  +  ½·Γ·(ΔS)²  +  Θ·Δt  +  ν·Δσ</code>
      <br><code>P&L  =  ΔC  ×  lot_size  (×-1 for short)</code>
      <br><br>
      <code>Break-even:  Δ·x  +  ½·Γ·x²  +  Θ·Δt  +  ν·Δσ  =  0</code>
      <br><code>Dynamic Delta Update:  Δ_new  =  Δ_old  +  Γ·ΔS</code>
    </div>"""

    # ── Chart helpers ─────────────────────────────────────────────────────
    def plot_div(fig_json, div_id):
        if fig_json is None:
            return f'<div class="chart-placeholder">No data available for this chart</div>'
        return f'<div id="{div_id}" class="chart-container"></div>'

    def plot_script(fig_json, div_id):
        if fig_json is None:
            return ""
        return f"Plotly.react('{div_id}', {json.dumps(fig_json['data'])}, {json.dumps(fig_json['layout'])}, {{responsive:true}});\n"

    all_scripts = (
        plot_script(candle_json, "chart-candle") +
        plot_script(pnl_json,   "chart-pnl") +
        plot_script(attr_json,  "chart-attr") +
        plot_script(delta_json, "chart-delta") +
        plot_script(td_json,    "chart-td") +
        plot_script(be_json,    "chart-be") +
        (plot_script(daily_json,"chart-daily") if daily_json else "") +
        (plot_script(vix_json,  "chart-vix") if vix_json else "")
    )

    vix_section = f"""
    <section id="sec-vix">
      <h2>📉 India VIX &amp; Vega Analysis</h2>
      {plot_div(vix_json, "chart-vix")}
    </section>""" if vix_json else ""

    # ── Full HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{SYMBOL} — Supertrend + Options Report v4.0</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0e14; --bg2: #0d1117; --bg3: #161b22; --bg4: #1c2128;
    --border: #21262d; --border2: #30363d;
    --green: #00d4aa; --red: #ff4757; --gold: #ffd700; --blue: #3d9bff;
    --text: #c9d1d9; --text2: #8b949e; --text3: #6e7681;
    --win-bg: rgba(0,212,170,0.07); --loss-bg: rgba(255,71,87,0.07);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Rajdhani', sans-serif;
          font-size: 15px; line-height: 1.5; }}
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg2); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}

  /* ── Header ── */
  header {{ background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            border-bottom: 1px solid var(--border2);
            padding: 28px 36px 20px; }}
  .header-top {{ display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px; }}
  .logo {{ font-family: 'Share Tech Mono', monospace; font-size: 11px; color: var(--text3); letter-spacing: 2px; text-transform: uppercase; }}
  h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: 1px;
        background: linear-gradient(90deg, var(--green), var(--gold));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .meta {{ font-family: 'Share Tech Mono', monospace; font-size: 11px; color: var(--text3); margin-top: 6px; }}

  /* ── Nav ── */
  nav {{ background: var(--bg3); border-bottom: 1px solid var(--border);
         padding: 0 36px; display: flex; gap: 0; overflow-x: auto; }}
  nav a {{ display: inline-block; padding: 12px 18px; color: var(--text2);
           text-decoration: none; font-weight: 600; font-size: 13px;
           letter-spacing: .5px; border-bottom: 2px solid transparent;
           white-space: nowrap; transition: all .2s; }}
  nav a:hover {{ color: var(--green); border-bottom-color: var(--green); }}

  /* ── Main layout ── */
  main {{ padding: 28px 36px; max-width: 1600px; }}
  section {{ margin-bottom: 52px; scroll-margin-top: 60px; }}
  h2 {{ font-size: 1.25rem; font-weight: 700; color: var(--gold); margin-bottom: 18px;
        letter-spacing: .5px; padding-bottom: 8px; border-bottom: 1px solid var(--border2); }}
  h3 {{ font-size: 1rem; font-weight: 600; color: var(--text2); margin: 20px 0 10px; }}

  /* ── Stat cards ── */
  .card-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(185px, 1fr)); gap: 14px; }}
  .stat-card {{ background: var(--bg3); border: 1px solid var(--border); border-radius: 8px;
                padding: 16px 18px; position: relative; overflow: hidden; }}
  .stat-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
                        background: linear-gradient(90deg, var(--green), transparent); }}
  .pos-card::before {{ background: linear-gradient(90deg, var(--green), transparent); }}
  .neg-card::before {{ background: linear-gradient(90deg, var(--red), transparent); }}
  .stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text3); margin-bottom: 6px; }}
  .stat-value {{ font-size: 1.5rem; font-weight: 700; font-family: 'Share Tech Mono', monospace; color: var(--text); }}
  .pos-card .stat-value {{ color: var(--green); }}
  .neg-card .stat-value {{ color: var(--red); }}
  .stat-sub {{ font-size: 11px; color: var(--text3); margin-top: 4px; }}

  /* ── Tables ── */
  .table-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 18px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  thead tr {{ background: var(--bg4); }}
  th {{ padding: 10px 12px; text-align: left; font-weight: 700; font-size: 11px;
        text-transform: uppercase; letter-spacing: .6px; color: var(--text2);
        border-bottom: 1px solid var(--border2); white-space: nowrap; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); white-space: nowrap;
        font-family: 'Share Tech Mono', monospace; font-size: 12px; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,.025); }}
  .win-row td {{ background: var(--win-bg); }}
  .loss-row td {{ background: var(--loss-bg); }}
  .pos-val {{ color: var(--green) !important; font-weight: 700; }}
  .neg-val {{ color: var(--red) !important; font-weight: 700; }}

  /* ── Charts ── */
  .chart-container {{ border-radius: 8px; border: 1px solid var(--border); overflow: hidden;
                      background: var(--bg2); margin-bottom: 18px; }}
  .chart-placeholder {{ background: var(--bg3); border: 1px dashed var(--border2);
                        border-radius: 8px; padding: 40px; text-align: center;
                        color: var(--text3); font-style: italic; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}

  /* ── Formula box ── */
  .formula-box {{ background: var(--bg4); border: 1px solid var(--border2);
                  border-left: 3px solid var(--gold); border-radius: 6px;
                  padding: 16px 20px; margin-top: 16px; }}
  .formula-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
                    color: var(--gold); margin-bottom: 10px; }}
  code {{ font-family: 'Share Tech Mono', monospace; font-size: 13px;
          color: var(--green); display: block; margin: 4px 0; }}

  /* ── Info table ── */
  .info-table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 13px; }}
  .info-table th {{ background: var(--bg4); padding: 9px 14px; text-align: left;
                    color: var(--text2); font-size: 11px; text-transform: uppercase;
                    letter-spacing: .6px; border-bottom: 1px solid var(--border2); }}
  .info-table td {{ padding: 8px 14px; border-bottom: 1px solid var(--border);
                    font-family: 'Share Tech Mono', monospace; font-size: 12.5px; }}
  .info-table tr:hover td {{ background: rgba(255,255,255,.025); }}

  /* ── Badge ── */
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 700; letter-spacing: .5px; }}
  .badge-green {{ background: rgba(0,212,170,.15); color: var(--green); }}
  .badge-red   {{ background: rgba(255,71,87,.15);  color: var(--red); }}

  /* ── Disclaimer ── */
  .disclaimer {{ background: var(--bg3); border: 1px solid var(--border2);
                 border-left: 3px solid var(--red); border-radius: 6px;
                 padding: 14px 18px; margin-bottom: 28px; font-size: 12px; color: var(--text3); }}

  /* ── Footer ── */
  footer {{ padding: 24px 36px; border-top: 1px solid var(--border);
            font-size: 11px; color: var(--text3); text-align: center;
            font-family: 'Share Tech Mono', monospace; letter-spacing: 1px; }}

  @media (max-width: 900px) {{
    main {{ padding: 16px 12px; }}
    .chart-row {{ grid-template-columns: 1fr; }}
    h1 {{ font-size: 1.4rem; }}
  }}
</style>
</head>
<body>

<header>
  <div class="logo">{SYMBOL} · Supertrend v4.0 · Greeks-Based Option P&amp;L Engine</div>
  <div class="header-top">
    <div>
      <h1>⚡ {SYMBOL} — Options Backtest Report</h1>
      <div class="meta">
        ST({ST_PERIOD},{ST_MULTIPLIER}) &nbsp;|&nbsp; HTF: {"ON" if htf_enabled else "OFF"} &nbsp;|&nbsp;
        Lot: {OPTION_LOT_SIZE} &nbsp;|&nbsp; Δ={OPTION_DELTA} Γ={OPTION_GAMMA} Θ={OPTION_THETA}/d ν={OPTION_VEGA} &nbsp;|&nbsp;
        Mode: {OPTION_DIRECTION.upper()} &nbsp;|&nbsp; Dynamic Δ: {"ON" if USE_DYNAMIC_DELTA else "OFF"}
      </div>
    </div>
  </div>
</header>

<nav>
  <a href="#sec-overview">📊 Overview</a>
  <a href="#sec-chart">📈 Chart</a>
  <a href="#sec-options">🎯 Option P&amp;L</a>
  <a href="#sec-greeks">⚗️ Greeks</a>
  <a href="#sec-trades">📋 Trade Log</a>
  <a href="#sec-daily">📅 Daily</a>
  <a href="#sec-timeslot">🕐 Time Slots</a>
  {"<a href='#sec-vix'>📉 VIX/Vega</a>" if vix_json else ""}
  <a href="#sec-model">📐 Model</a>
</nav>

<main>

  <div class="disclaimer">
    ⚠ <strong>DISCLAIMER:</strong> This report uses Greek-approximation (Taylor expansion) for option P&amp;L.
    Greeks are assumed locally constant per trade. Vega is included when VIX data is available.
    Accuracy reduces for large moves, near-expiry 0DTE, deep OTM, or volatility shocks.
    Not financial advice.
  </div>

  <!-- ═══ STRATEGY OVERVIEW ════════════════════════════════════════════════ -->
  <section id="sec-overview">
    <h2>📊 Strategy Overview</h2>
    <div class="card-grid">{strat_cards}</div>

    <h3>Option P&amp;L Summary</h3>
    <div class="card-grid">{opt_cards}</div>
  </section>

  <!-- ═══ CANDLESTICK CHART ════════════════════════════════════════════════ -->
  <section id="sec-chart">
    <h2>📈 Price &amp; Supertrend Chart (Last {CHART_DAYS} Days)</h2>
    {plot_div(candle_json, "chart-candle")}
  </section>

  <!-- ═══ OPTION P&L ══════════════════════════════════════════════════════ -->
  <section id="sec-options">
    <h2>🎯 Option P&amp;L Analysis</h2>
    {plot_div(pnl_json, "chart-pnl")}
    {"" if daily_json is None else plot_div(daily_json, "chart-daily")}
    <h3>Per-Trade Option Detail</h3>
    {opt_html}
  </section>

  <!-- ═══ GREEKS ANALYSIS ══════════════════════════════════════════════════ -->
  <section id="sec-greeks">
    <h2>⚗️ Greeks Behaviour &amp; Attribution</h2>
    <div class="chart-row">
      {plot_div(attr_json, "chart-attr")}
      {plot_div(delta_json, "chart-delta")}
    </div>
    <div class="chart-row">
      {plot_div(td_json, "chart-td")}
      {plot_div(be_json, "chart-be")}
    </div>
  </section>

  <!-- ═══ VIX / VEGA ═══════════════════════════════════════════════════════ -->
  {vix_section}

  <!-- ═══ FULL TRADE LOG ══════════════════════════════════════════════════ -->
  <section id="sec-trades">
    <h2>📋 Complete Trade Log</h2>
    {tl_html}
  </section>

  <!-- ═══ DAILY P&L ════════════════════════════════════════════════════════ -->
  <section id="sec-daily">
    <h2>📅 Daily P&amp;L Breakdown</h2>
    {day_html}
  </section>

  <!-- ═══ TIME SLOT ════════════════════════════════════════════════════════ -->
  <section id="sec-timeslot">
    <h2>🕐 Time Slot P&amp;L Analysis</h2>
    {ts_html}
  </section>

  <!-- ═══ MODEL / GREEKS INFO ══════════════════════════════════════════════ -->
  <section id="sec-model">
    <h2>📐 Model Parameters &amp; Theoretical Foundation</h2>
    {greeks_info}
    <h3>Model Assumptions</h3>
    <table class="info-table">
      <tr><th>#</th><th>Assumption</th><th>Impact</th></tr>
      <tr><td>1</td><td>Greeks locally constant during holding period</td><td>Good for intraday short holds; weakens for >2hr trades</td></tr>
      <tr><td>2</td><td>Implied volatility change captured via VIX (Vega)</td><td>Approximation; actual IV skew not modelled</td></tr>
      <tr><td>3</td><td>Small-to-moderate underlying moves</td><td>Gamma term captures curvature; large gap moves reduce accuracy</td></tr>
      <tr><td>4</td><td>Intraday holding — Theta linear in time</td><td>Accurate for ≤1 day; non-linear for 0DTE near expiry</td></tr>
      <tr><td>5</td><td>Dynamic Delta: Δ_new = Δ_old + Γ·ΔS (10 micro-steps)</td><td>Significantly improves accuracy vs static delta</td></tr>
      <tr><td>6</td><td>Indian session = 6.25 hours/day for Theta normalisation</td><td>Theta per hour = Theta_daily / 6.25</td></tr>
    </table>
  </section>

</main>

<footer>
  {SYMBOL} Supertrend v4.0 — Greeks-Based Intraday Option P&amp;L Engine &nbsp;·&nbsp;
  Δ={OPTION_DELTA} &nbsp; Γ={OPTION_GAMMA} &nbsp; Θ={OPTION_THETA}/d &nbsp; ν={OPTION_VEGA} &nbsp;·&nbsp;
  Lot={OPTION_LOT_SIZE} &nbsp;|&nbsp; {OPTION_DIRECTION.upper()} &nbsp;|&nbsp;
  Dynamic Δ: {"ON" if USE_DYNAMIC_DELTA else "OFF"}
</footer>

<script>
{all_scripts}
</script>
</body>
</html>"""

    fname = f"{SYMBOL}_options_report_v4.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅  HTML Report saved → {fname}")
    return fname

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSOLE PRINT  (condensed + option summary)
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(trades, htf_enabled):
    SEP = "═" * 110; DASH = "─" * 110
    if not trades:
        print("  ⚠  No trades."); return None, None, None

    df_t   = pd.DataFrame(trades)
    df_day = build_daily_summary(trades)
    df_ts  = build_time_slot_summary(trades)
    opt    = build_option_summary(trades)

    total = len(df_t); wins = (df_t["P&L %"] > 0).sum()
    print("\n" + SEP)
    print(f"  {SYMBOL}  ST({ST_PERIOD},{ST_MULTIPLIER})  |  Trades: {total}  |  "
          f"Win Rate: {wins/total*100:.1f}%  |  HTF: {'ON' if htf_enabled else 'OFF'}")
    print(SEP)

    cols = ["Date","Direction","Entry Time","Entry Price","Exit Time","Exit Price",
            "Holding Time","Points Captured","P&L %","Opt_PnL_Total","Exit Reason","Result"]
    print(df_t[[c for c in cols if c in df_t.columns]].to_string(index=False))

    print("\n" + SEP); print("  PER-DAY P&L"); print(DASH)
    print(df_day.to_string(index=False))

    print("\n" + SEP); print("  TIME SLOT ANALYSIS"); print(DASH)
    print(df_ts.to_string(index=False))

    print("\n" + SEP); print("  OPTION P&L SUMMARY"); print(DASH)
    print(f"  Total Option P&L         : ₹{opt['total_opt_pnl']:>12,.2f}")
    print(f"  Total Option Profit      : ₹{opt['win_opt_pnl']:>12,.2f}")
    print(f"  Total Option Loss        : ₹{opt['loss_opt_pnl']:>12,.2f}")
    print(f"  Option Win Rate          : {opt['opt_win_rate']}%")
    print(DASH)
    print(f"  Delta Contribution (all) : ₹{opt['total_delta_contrib']:>12,.2f}")
    print(f"  Gamma Contribution (all) : ₹{opt['total_gamma_contrib']:>12,.2f}")
    print(f"  Theta Cost (all)         : ₹{opt['total_theta_cost']:>12,.2f}")
    print(f"  Vega Contribution (all)  : ₹{opt['total_vega_contrib']:>12,.2f}")
    print(DASH)
    print(f"  Avg Break-even ΔS        : {opt['avg_breakeven_dS']} pts")
    print(f"  Best Option Trade        : ₹{opt['best_opt_trade']['Opt_PnL_Total']:,.2f}  "
          f"({opt['best_opt_trade']['Entry Time']})")
    print(f"  Worst Option Trade       : ₹{opt['worst_opt_trade']['Opt_PnL_Total']:,.2f}  "
          f"({opt['worst_opt_trade']['Entry Time']})")
    print(SEP + "\n")

    adv_metrics_report = build_advanced_option_metrics(trades)
    if adv_metrics_report:
        print(adv_metrics_report)

    return df_t, df_day, df_ts

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  SUPERTREND + OPTIONS BACKTEST v4.0")
    print("  Greeks: Δ={} Γ={} Θ={}/d ν={} | Lot={}".format(
          OPTION_DELTA, OPTION_GAMMA, OPTION_THETA, OPTION_VEGA, OPTION_LOT_SIZE))
    print("═"*60)

    try:
        df, htf_enabled, vix_series = fetch_data()
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(e); return

    trades = run_backtest(df, htf_enabled, vix_series)
    df_t, df_day, df_ts = print_results(trades, htf_enabled)

    print("  Building HTML Report …")
    fname = export_html(df, trades, df_day, df_ts, htf_enabled, vix_series)

    print("\n  ✅  Done.\n")
    return fname

if __name__ == "__main__":
    main()