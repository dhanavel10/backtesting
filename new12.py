"""
Supertrend Intraday Backtesting Strategy — v5.0 (Clean Rebuild)
────────────────────────────────────────────────────────────────
ENTRY CONDITIONS (ALL must be true on the same candle):
  1. ST Bullish  : LTF Supertrend is bullish
     HTF Filter  : If HTF_FILTER_ENABLED=True, HTF Supertrend must ALSO be
                   bullish (long) / bearish (short). Both must agree.
  2. EMA9 Gap    : open > EMA9 + EMA9_ENTRY_GAP  (long)
                   open < EMA9 - EMA9_ENTRY_GAP  (short)
  3. EMA9 Slope  : rolling linear regression slope of EMA9 over last
                   EMA9_SLOPE_CANDLES candles converted to angle (degrees).
                   If angle < EMA9_SLOPE_MIN_DEG (default 10°) → skip, too flat.
                   Long  : slope must be positive AND angle >= threshold
                   Short : slope must be negative AND |angle| >= threshold
  4. Candle Body : |open - close| > CANDLE_BODY_MIN (default 12 pts)
  Entry price    : close of the entry candle

EXIT CONDITIONS (priority order, first triggered wins):
  1. Hard SL     : 0.25% adverse move from entry price
                   exit_px = hard_sl_price
  2. Peak EMA9   : highest EMA9 seen since entry (updated each candle).
                   Long  : close < peak_ema9 - PEAK_EMA9_DROP (default 30 pts) → exit
                   Short : close > peak_ema9 + PEAK_EMA9_DROP → exit
                   exit_px = c_close
  3. EMA9 Slope  : rolling slope of EMA9 over last EMA9_SLOPE_CANDLES candles.
                   Long  : slope turns negative → exit
                   Short : slope turns positive → exit
                   exit_px = c_close
  4. EOD         : forced exit at market close → exit_px = c_close

Install: pip install pandas plotly pytz numpy yfinance scipy
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz, os, json, math
from datetime import time as dtime
from scipy.optimize import brentq

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATA_SOURCE = "yfinance"          # "csv" or "yfinance"
BASE_CSV    = "Book1min.csv"
HTF_CSV     = "Book1.csv"
VIX_CSV     = None

YF_TICKER        = "^NSEI"
YF_BASE_INTERVAL = "1m"
YF_HTF_INTERVAL  = "15m"
YF_PERIOD        = "7d"
YF_START         = None
YF_END           = None
YF_VIX_TICKER    = "^INDIAVIX"

SYMBOL = "NIFTY"

# ── Supertrend (LTF = base timeframe) ─────────────────────────────────────────
ST_PERIOD     = 7
ST_MULTIPLIER = 4.0

# ── HTF Supertrend ─────────────────────────────────────────────────────────────
HTF_FILTER_ENABLED = True    # True  → both LTF+HTF must agree before entry
HTF_ST_PERIOD      = 10
HTF_ST_MULTIPLIER  = 3.0

# ── EMA9 ───────────────────────────────────────────────────────────────────────
EMA9_PERIOD         = 9

# Entry: open must be this many points beyond EMA9
EMA9_ENTRY_GAP      = 10.0   # pts

# Slope filter (entry + exit)
EMA9_SLOPE_CANDLES  = 5      # rolling window for linear regression
EMA9_SLOPE_MIN_DEG  = 10.0   # degrees — slopes flatter than this block entry

# Exit: if price falls this many pts below the highest EMA9 seen since entry
PEAK_EMA9_DROP      = 50.0   # pts

# ── Candle body filter (entry only) ────────────────────────────────────────────
CANDLE_BODY_MIN     = 9.0   # pts  |open - close| > 12

# ── Hard Stop Loss ─────────────────────────────────────────────────────────────
HARD_SL_PCT         = 0.25   # % adverse from entry price

# ── Trade window ───────────────────────────────────────────────────────────────
TRADE_WINDOW_ENABLED = True
TRADE_WINDOWS = [
    (dtime(9, 30), dtime(11, 30)),
    (dtime(14,  0), dtime(14, 45)),
]

# ── Daily trade cap ────────────────────────────────────────────────────────────
MAX_TRADES_PER_DAY_ENABLED = True
MAX_TRADES_PER_DAY         = 5

# ── Market ─────────────────────────────────────────────────────────────────────
MARKET_OPEN           = dtime(9, 15)
MARKET_CLOSE          = dtime(15, 15)
TRADING_HOURS_PER_DAY = 6.25
IST                   = pytz.timezone("Asia/Kolkata")
CHART_DAYS            = 5

# ══════════════════════════════════════════════════════════════════════════════
#  GREEKS PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
OPTION_LOT_SIZE   = 65
OPTION_DELTA      = 0.20
OPTION_GAMMA      = 0.0006
OPTION_THETA      = -10.53
OPTION_VEGA       = 5.0
OPTION_DIRECTION  = "long"   # "long" = buying options
USE_DYNAMIC_DELTA = True

SLIPPAGE_PER_CONTRACT = 50.0
BROKERAGE_PER_TRADE   = 0.0
MAX_LOSS_PER_TRADE    = None
MAX_MARGIN_USED       = None

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fmt(ts):
    return ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "—"

def mins_to_hhmm(m):
    m = int(round(m))
    return f"{m // 60}H:{m % 60:02d}M"

def slope_to_deg(slope):
    """Convert raw EMA slope (pts per candle) to angle in degrees."""
    return math.degrees(math.atan(slope))

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
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
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
    df["ST_val"]  = np.where(bull, lower, upper)
    df["ST_bull"] = bull
    return df

# ═══════════════════════════════════════════════════════════════════════════════
#  EMA9 SLOPE — rolling linear regression over last N candles
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ema9_slope(ema9_series, window):
    """
    Returns a Series of slopes (pts per candle) using linear regression
    over a rolling window. Positive = rising, negative = falling.
    """
    slopes = np.full(len(ema9_series), np.nan)
    vals   = ema9_series.values
    x      = np.arange(window, dtype=float)
    x_mean = x.mean()
    ss_x   = ((x - x_mean) ** 2).sum()

    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean   = y.mean()
        slope    = ((x - x_mean) * (y - y_mean)).sum() / ss_x
        slopes[i] = slope

    return pd.Series(slopes, index=ema9_series.index)

# ═══════════════════════════════════════════════════════════════════════════════
#  GREEKS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_option_pnl(entry_price_S, exit_price_S, holding_hours,
                       trade_direction="long",
                       vix_at_entry=None, vix_at_exit=None):
    abs_delta     = abs(OPTION_DELTA)
    gamma         = OPTION_GAMMA
    theta_daily   = OPTION_THETA
    vega          = OPTION_VEGA
    lot_size      = OPTION_LOT_SIZE
    opt_direction = OPTION_DIRECTION

    signed_delta = abs_delta if trade_direction == "long" else -abs_delta
    dS_total     = exit_price_S - entry_price_S
    theta_per_hr = theta_daily / TRADING_HOURS_PER_DAY
    theta_effect = theta_per_hr * holding_hours

    vega_effect = 0.0
    vix_change  = 0.0
    if vix_at_entry is not None and vix_at_exit is not None and vix_at_entry > 0:
        vix_change_pts    = vix_at_exit - vix_at_entry
        iv_change_decimal = vix_change_pts / 100.0
        vega_effect       = vega * iv_change_decimal
        vix_change        = vix_change_pts

    steps     = 10
    step_size = dS_total / steps if steps else dS_total
    cur_delta = signed_delta
    delta_effect_dyn = 0.0

    for _ in range(steps):
        delta_effect_dyn += cur_delta * step_size
        if USE_DYNAMIC_DELTA:
            cur_delta = cur_delta + gamma * step_size

    gamma_effect        = 0.0 if USE_DYNAMIC_DELTA else 0.5 * gamma * (dS_total ** 2)
    gamma_effect_static = 0.5 * gamma * (dS_total ** 2)
    option_change       = delta_effect_dyn + gamma_effect + theta_effect + vega_effect
    final_delta         = cur_delta

    sign        = 1 if opt_direction == "long" else -1
    pnl_per_lot = option_change * sign
    gross_pnl   = pnl_per_lot * lot_size
    slippage    = SLIPPAGE_PER_CONTRACT + BROKERAGE_PER_TRADE
    total_pnl   = gross_pnl - slippage
    gamma_pnl_total = gamma_effect_static * lot_size

    const_term   = theta_effect + vega_effect
    breakeven_dS = _solve_breakeven(signed_delta, gamma, const_term)

    return {
        "dS": round(dS_total, 4),
        "holding_hours": round(holding_hours, 4),
        "delta_entry": round(signed_delta, 6),
        "delta_exit": round(final_delta, 6),
        "gamma": round(gamma, 6),
        "theta_daily": round(theta_daily, 4),
        "theta_per_hour": round(theta_per_hr, 4),
        "vega": round(vega, 4),
        "vix_at_entry": round(vix_at_entry, 2) if vix_at_entry else None,
        "vix_at_exit": round(vix_at_exit, 2) if vix_at_exit else None,
        "vix_change": round(vix_change, 4),
        "delta_effect": round(delta_effect_dyn, 4),
        "gamma_effect": round(gamma_effect, 4),
        "gamma_effect_static": round(gamma_effect_static, 4),
        "gamma_pnl_total": round(gamma_pnl_total, 4),
        "theta_effect": round(theta_effect, 4),
        "vega_effect": round(vega_effect, 4),
        "option_change": round(option_change, 4),
        "pnl_per_lot": round(pnl_per_lot, 4),
        "gross_pnl": round(gross_pnl, 2),
        "slippage": round(slippage, 2),
        "total_pnl": round(total_pnl, 2),
        "lot_size": lot_size,
        "opt_direction": opt_direction,
        "breakeven_dS": breakeven_dS,
        "trade_direction": trade_direction,
    }


def _solve_breakeven(delta, gamma, const_term):
    if gamma == 0:
        return round(-const_term / delta, 4) if delta != 0 else None
    try:
        a = 0.5 * gamma; b = delta; c = const_term
        disc = b**2 - 4 * a * c
        if disc >= 0:
            r1 = (-b + math.sqrt(disc)) / (2 * a)
            r2 = (-b - math.sqrt(disc)) / (2 * a)
            cands = [r for r in [r1, r2] if abs(r) < 10000]
            if cands:
                return round(min(cands, key=abs), 4)
        root = brentq(lambda x: a*x*x + b*x + c, -5000, 5000)
        return round(root, 4)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
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
        if cand in lmap: return lmap[cand]
    return None

def load_csv(path, label="CSV"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n  ❌  File not found: '{path}'\n")
    print(f"\n  Loading {label}: {path} …")
    df     = pd.read_csv(path)
    dt_col = _find_col(df.columns, _DT_CANDS)
    o_col  = _find_col(df.columns, _O_CANDS)
    h_col  = _find_col(df.columns, _H_CANDS)
    l_col  = _find_col(df.columns, _L_CANDS)
    c_col  = _find_col(df.columns, _C_CANDS)
    v_col  = _find_col(df.columns, _V_CANDS)
    rename = {dt_col:"__dt", o_col:"Open", h_col:"High", l_col:"Low", c_col:"Close"}
    if v_col: rename[v_col] = "Volume"
    df = df[list(rename.keys())].rename(columns=rename)
    parsed = None
    for fmt_ in ["%d-%m-%Y %H:%M","%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M",
                 "%d-%m-%Y %H:%M:%S","%m/%d/%Y %H:%M","%Y-%m-%dT%H:%M:%S"]:
        attempt = pd.to_datetime(df["__dt"], format=fmt_, errors="coerce")
        if attempt.notna().mean() > 0.95: parsed = attempt; break
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
    print(f"    Rows: {len(df)}  |  {fmt(df.index[0])} → {fmt(df.index[-1])}")
    return df

def _normalise_yf_df(raw):
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).strip().title() for c in raw.columns]
    if "Adj Close" in raw.columns and "Close" not in raw.columns:
        raw = raw.rename(columns={"Adj Close":"Close"})
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in raw.columns]
    df = raw[keep].copy()
    if "Volume" not in df.columns: df["Volume"] = 0.0
    return df

def load_yfinance(ticker, interval, label="yfinance"):
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("\n  ❌  yfinance not installed.\n")
    print(f"\n  Downloading {label}: {ticker} {interval} …")
    kw = dict(tickers=ticker, interval=interval, auto_adjust=True, progress=False)
    if YF_START is not None:
        kw["start"] = YF_START
        if YF_END: kw["end"] = YF_END
    else:
        kw["period"] = YF_PERIOD
    raw = yf.download(**kw)
    if raw is None or raw.empty:
        raise ValueError(f"\n  ❌  No data for {ticker} {interval}\n")
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
    if df.empty: raise ValueError(f"\n  ❌  No data after market filter for {ticker}\n")
    print(f"    Rows: {len(df)}  |  {fmt(df.index[0])} → {fmt(df.index[-1])}")
    return df

def load_vix():
    if VIX_CSV and os.path.exists(VIX_CSV):
        print(f"\n  Loading VIX: {VIX_CSV}")
        for sep in ["\t", ","]:
            try:
                dv = pd.read_csv(VIX_CSV, sep=sep)
                if len(dv.columns) >= 2: break
            except Exception: continue
        dt_col = _find_col(dv.columns, _DT_CANDS)
        c_col  = _find_col(dv.columns, _C_CANDS)
        if dt_col and c_col:
            dv[dt_col] = pd.to_datetime(dv[dt_col], errors="coerce")
            dv = dv.dropna(subset=[dt_col]).set_index(dt_col)
            dv.index = dv.index.normalize()
            vix_out = dv[c_col].rename("VIX")
            print(f"    VIX rows: {len(vix_out)}")
            return vix_out
    try:
        import yfinance as yf
        raw = yf.download(YF_VIX_TICKER,
                          period=YF_PERIOD if YF_START is None else None,
                          start=YF_START, end=YF_END,
                          auto_adjust=True, progress=False)
        if raw is not None and not raw.empty:
            raw = _normalise_yf_df(raw)
            raw.index = raw.index.normalize()
            if raw.index.tz is not None:
                raw.index = raw.index.tz_convert(IST).normalize()
            return raw["Close"].rename("VIX")
    except Exception as e:
        print(f"  ⚠  VIX download failed: {e}")
    print("  ⚠  No VIX — Vega effect = 0.")
    return None

# ═══════════════════════════════════════════════════════════════════════════════
#  FETCH & PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_data():
    src = DATA_SOURCE.strip().lower()
    if src == "csv":
        df         = load_csv(BASE_CSV, "Base TF CSV")
        htf_source = HTF_CSV
    elif src == "yfinance":
        df         = load_yfinance(YF_TICKER, YF_BASE_INTERVAL, "Base TF")
        htf_source = YF_HTF_INTERVAL
    else:
        raise ValueError(f"Unknown DATA_SOURCE='{DATA_SOURCE}'")

    # LTF Supertrend
    df = compute_supertrend(df, ST_PERIOD, ST_MULTIPLIER)
    df.dropna(inplace=True)

    # EMA9
    df["EMA9"] = df["Close"].ewm(span=EMA9_PERIOD, adjust=False).mean()

    # EMA9 rolling slope (pts per candle)
    df["EMA9_slope"] = compute_ema9_slope(df["EMA9"], EMA9_SLOPE_CANDLES)
    df.dropna(subset=["EMA9","EMA9_slope"], inplace=True)
    print(f"  ✅  EMA{EMA9_PERIOD} + slope({EMA9_SLOPE_CANDLES} candles) computed")

    # HTF Supertrend
    htf_enabled = False
    if HTF_FILTER_ENABLED and htf_source is not None:
        try:
            df_htf = (load_csv(htf_source, "HTF CSV") if src == "csv"
                      else load_yfinance(YF_TICKER, YF_HTF_INTERVAL, "HTF"))
            df_htf = compute_supertrend(df_htf, HTF_ST_PERIOD, HTF_ST_MULTIPLIER)
            df_htf.dropna(inplace=True)
            htf_bull = df_htf["ST_bull"].reindex(df.index, method="ffill")
            if htf_bull.notna().mean() >= 0.5:
                df["HTF_bull"] = htf_bull.ffill()
                df.dropna(subset=["HTF_bull"], inplace=True)
                htf_enabled = True
                print(f"  ✅  HTF ENABLED — ST({HTF_ST_PERIOD},{HTF_ST_MULTIPLIER})")
        except FileNotFoundError:
            print("  ⚠  HTF file not found — HTF filter DISABLED.")

    if not htf_enabled:
        df["HTF_bull"] = True   # sentinel: passes all HTF checks

    vix_series = load_vix()
    return df, htf_enabled, vix_series

# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE — v5.0
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(df, htf_enabled, vix_series):
    trades = []

    # ── Trade state ────────────────────────────────────────────────────────────
    in_trade      = False
    direction     = None
    entry_price   = None
    entry_time    = None
    hard_sl_price = None
    peak_ema9     = None   # highest EMA9 seen since entry (long) / lowest (short)

    daily_trades  = {}

    # ── Pre-extract arrays for speed ──────────────────────────────────────────
    closes     = df["Close"].values.astype(float)
    opens      = df["Open"].values.astype(float)
    highs      = df["High"].values.astype(float)
    lows       = df["Low"].values.astype(float)
    st_bulls   = df["ST_bull"].values.astype(bool)
    htf_bulls  = df["HTF_bull"].values.astype(bool)
    ema9_vals  = df["EMA9"].values.astype(float)
    slope_vals = df["EMA9_slope"].values.astype(float)
    times      = df.index

    # ── Helpers ────────────────────────────────────────────────────────────────
    def in_window(t):
        if not TRADE_WINDOW_ENABLED: return True
        return any(s <= t <= e for s, e in TRADE_WINDOWS)

    def get_vix(ts):
        if vix_series is None: return None
        day = ts.normalize() if hasattr(ts, "normalize") else pd.Timestamp(ts).normalize()
        try:    return float(vix_series.asof(day))
        except: return None

    def record_trade(dir_, e_time, e_price, x_time, x_price, reason, hsl):
        pnl = round(
            ((x_price - e_price) / e_price * 100) if dir_ == "long"
            else ((e_price - x_price) / e_price * 100), 4)
        hold_mins     = int((x_time - e_time).total_seconds() // 60)
        holding_hours = hold_mins / 60.0
        opt = compute_option_pnl(
            entry_price_S  = e_price,
            exit_price_S   = x_price,
            holding_hours  = holding_hours,
            trade_direction= dir_,
            vix_at_entry   = get_vix(e_time),
            vix_at_exit    = get_vix(x_time),
        )
        trades.append({
            "Date"            : e_time.strftime("%Y-%m-%d"),
            "Direction"       : "Long  ↑" if dir_ == "long" else "Short ↓",
            "Entry Time"      : fmt(e_time),
            "Entry Price"     : round(e_price, 2),
            "Hard SL"         : round(hsl, 2) if hsl else "OFF",
            "Exit Time"       : fmt(x_time),
            "Exit Price"      : round(x_price, 2),
            "Holding Mins"    : hold_mins,
            "Holding Time"    : mins_to_hhmm(hold_mins),
            "Points Captured" : round(abs(x_price - e_price), 4),
            "P&L %"           : pnl,
            "Exit Reason"     : reason,
            "Result"          : "WIN" if pnl > 0 else "LOSS",
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
            "Opt_GammaPnL"    : opt["gamma_pnl_total"],
            "Opt_ThetaEffect" : opt["theta_effect"],
            "Opt_VegaEffect"  : opt["vega_effect"],
            "Opt_Change"      : opt["option_change"],
            "Opt_PnL_PerLot"  : opt["pnl_per_lot"],
            "Opt_GrossPnL"    : opt["gross_pnl"],
            "Opt_Slippage"    : opt["slippage"],
            "Opt_PnL_Total"   : opt["total_pnl"],
            "Opt_BreakevenDS" : opt["breakeven_dS"],
            "Opt_LotSize"     : opt["lot_size"],
        })

    # ── Main loop ──────────────────────────────────────────────────────────────
    for i in range(EMA9_SLOPE_CANDLES, len(df)):
        c_close  = closes[i]
        c_open   = opens[i]
        c_high   = highs[i]
        c_low    = lows[i]
        c_time   = times[i]
        c_tod    = c_time.time()
        c_bull   = st_bulls[i]
        c_htf    = htf_bulls[i]
        c_ema9   = ema9_vals[i]
        c_slope  = slope_vals[i]
        date_key = c_time.date()

        # ── EOD forced exit ───────────────────────────────────────────────────
        if c_tod >= MARKET_CLOSE:
            if in_trade:
                record_trade(direction, entry_time, entry_price,
                             c_time, c_close, "EOD Exit", hard_sl_price)
                in_trade = False
            continue

        if c_tod < MARKET_OPEN:
            continue

        # ── Manage open trade — check exits ───────────────────────────────────
        if in_trade:
            # Update peak EMA9 tracker
            if direction == "long":
                if c_ema9 > peak_ema9: peak_ema9 = c_ema9
            else:
                if c_ema9 < peak_ema9: peak_ema9 = c_ema9

            # -- EXIT 1: Hard SL (0.25%) ──────────────────────────────────────
            if direction == "long":
                hard_hit = c_low <= hard_sl_price
            else:
                hard_hit = c_high >= hard_sl_price

            if hard_hit:
                record_trade(direction, entry_time, entry_price,
                             c_time, hard_sl_price,
                             f"Hard SL ({HARD_SL_PCT}%)", hard_sl_price)
                in_trade = False
                continue

            # -- EXIT 2: Peak EMA9 drop ───────────────────────────────────────
            # Long  : close < highest-EMA9-since-entry - 30 pts
            # Short : close > lowest-EMA9-since-entry  + 30 pts
            if direction == "long":
                peak_ema9_hit = c_close < peak_ema9 - PEAK_EMA9_DROP
            else:
                peak_ema9_hit = c_close > peak_ema9 + PEAK_EMA9_DROP

            if peak_ema9_hit:
                record_trade(direction, entry_time, entry_price,
                             c_time, c_close,
                             f"Peak EMA9 Drop ({PEAK_EMA9_DROP}pts)", hard_sl_price)
                in_trade = False
                continue

            # -- EXIT 3: EMA9 slope reversal ──────────────────────────────────
            # Long  : slope turns negative (any downward tilt)
            # Short : slope turns positive (any upward tilt)
            if not np.isnan(c_slope):
                slope_reversed = (
                    (direction == "long"  and c_slope < -20) or
                    (direction == "short" and c_slope > 20)
                )
                if slope_reversed:
                    slope_deg = round(slope_to_deg(c_slope), 2)
                    record_trade(direction, entry_time, entry_price,
                                 c_time, c_close,
                                 f"EMA9 Slope Reversal ({slope_deg}°)", hard_sl_price)
                    in_trade = False
                    continue

            # No exit triggered — hold trade
            continue

        # ── Entry scan ────────────────────────────────────────────────────────
        if not in_window(c_tod):
            continue
        if MAX_TRADES_PER_DAY_ENABLED and daily_trades.get(date_key, 0) >= MAX_TRADES_PER_DAY:
            continue
        if np.isnan(c_ema9) or np.isnan(c_slope):
            continue

        # ── ENTRY: LONG ───────────────────────────────────────────────────────
        # 1. LTF ST bullish
        # 2. HTF ST bullish (if filter enabled)
        # 3. open > EMA9 + EMA9_ENTRY_GAP
        # 4. slope positive AND angle >= EMA9_SLOPE_MIN_DEG
        # 5. |open - close| > CANDLE_BODY_MIN
        if c_bull and c_htf:
            ema9_gap_ok   = c_open > c_ema9 + EMA9_ENTRY_GAP
            slope_deg     = slope_to_deg(c_slope)
            slope_ok      = c_slope > 0 and slope_deg >= EMA9_SLOPE_MIN_DEG
            body_ok       = abs(c_open - c_close) > CANDLE_BODY_MIN

            if ema9_gap_ok and slope_ok and body_ok:
                entry_price   = c_close
                entry_time    = c_time
                direction     = "long"
                in_trade      = True
                hard_sl_price = round(entry_price * (1 - HARD_SL_PCT / 100), 4)
                peak_ema9     = c_ema9   # initialise peak EMA9 tracker
                daily_trades[date_key] = daily_trades.get(date_key, 0) + 1
                continue

        # ── ENTRY: SHORT ──────────────────────────────────────────────────────
        # 1. LTF ST bearish
        # 2. HTF ST bearish (if filter enabled)
        # 3. open < EMA9 - EMA9_ENTRY_GAP
        # 4. slope negative AND |angle| >= EMA9_SLOPE_MIN_DEG
        # 5. |open - close| > CANDLE_BODY_MIN
        if not c_bull and not c_htf:
            ema9_gap_ok   = c_open < c_ema9 - EMA9_ENTRY_GAP
            slope_deg     = slope_to_deg(c_slope)
            slope_ok      = c_slope < 0 and abs(slope_deg) >= EMA9_SLOPE_MIN_DEG
            body_ok       = abs(c_open - c_close) > CANDLE_BODY_MIN

            if ema9_gap_ok and slope_ok and body_ok:
                entry_price   = c_close
                entry_time    = c_time
                direction     = "short"
                in_trade      = True
                hard_sl_price = round(entry_price * (1 + HARD_SL_PCT / 100), 4)
                peak_ema9     = c_ema9   # initialise peak EMA9 tracker (lowest for short)
                daily_trades[date_key] = daily_trades.get(date_key, 0) + 1

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
        rows.append({
            "Date"            : date,
            "Total Trades"    : total,
            "Longs"           : grp["Direction"].str.contains("Long").sum(),
            "Shorts"          : grp["Direction"].str.contains("Short").sum(),
            "Winners"         : wins,
            "Losers"          : total - wins,
            "Win Rate %"      : round(wins / total * 100, 1),
            "Total P&L %"     : round(pnl, 4),
            "Best Trade %"    : round(grp["P&L %"].max(), 4),
            "Worst Trade %"   : round(grp["P&L %"].min(), 4),
            "Points Captured" : round(wp, 2),
            "Points Lost"     : round(lp, 2),
            "Net Points"      : round(wp - lp, 2),
            "Opt P&L (₹)"     : round(grp["Opt_PnL_Total"].sum(), 2),
            "Avg Hold Time"   : mins_to_hhmm(grp["Holding Mins"].mean()),
            "Day Result"      : "✅ Profit" if pnl > 0 else "❌ Loss" if pnl < 0 else "⚖ Flat",
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
    df_t["Entry_tod"] = pd.to_datetime(df_t["Entry Time"],
                                       format="%Y-%m-%d %H:%M").dt.strftime("%H:%M")
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
            "Time Slot"   : f"{s}–{e}",
            "Trades"      : total,
            "Winners"     : wins,
            "Losers"      : total - wins,
            "Win Rate %"  : round(wr, 1),
            "Total P&L %" : round(pnl, 4),
            "Avg P&L %"   : round(grp["P&L %"].mean(), 4),
            "Best %"      : round(grp["P&L %"].max(), 4),
            "Worst %"     : round(grp["P&L %"].min(), 4),
            "Opt P&L (₹)" : round(grp["Opt_PnL_Total"].sum(), 2),
            "Verdict"     : "🟢 Trade" if pnl > 0 and wr >= 50 else "🔴 Avoid",
        })
    return pd.DataFrame(rows)

def build_option_summary(trades):
    if not trades: return {}
    df_t = pd.DataFrame(trades)
    total_opt_pnl  = df_t["Opt_PnL_Total"].sum()
    win_opt_pnl    = df_t.loc[df_t["Opt_PnL_Total"] > 0, "Opt_PnL_Total"].sum()
    loss_opt_pnl   = df_t.loc[df_t["Opt_PnL_Total"] <= 0, "Opt_PnL_Total"].sum()
    opt_wins       = (df_t["Opt_PnL_Total"] > 0).sum()
    opt_losses     = (df_t["Opt_PnL_Total"] <= 0).sum()
    total_delta_c  = df_t["Opt_DeltaEffect"].sum() * OPTION_LOT_SIZE
    total_gamma_c  = df_t["Opt_GammaPnL"].sum()
    total_theta    = df_t["Opt_ThetaEffect"].sum() * OPTION_LOT_SIZE
    total_vega_c   = df_t["Opt_VegaEffect"].sum() * OPTION_LOT_SIZE
    total_slippage = df_t["Opt_Slippage"].sum()
    be_series      = df_t["Opt_BreakevenDS"].dropna()
    be_positive    = be_series[be_series > 0]
    avg_be         = be_positive.mean() if not be_positive.empty else None
    return {
        "df_t"               : df_t,
        "total_opt_pnl"      : round(total_opt_pnl, 2),
        "win_opt_pnl"        : round(win_opt_pnl, 2),
        "loss_opt_pnl"       : round(loss_opt_pnl, 2),
        "opt_wins"           : int(opt_wins),
        "opt_losses"         : int(opt_losses),
        "opt_win_rate"       : round(opt_wins / len(df_t) * 100, 1),
        "total_theta_cost"   : round(total_theta, 2),
        "total_delta_contrib": round(total_delta_c, 2),
        "total_gamma_contrib": round(total_gamma_c, 2),
        "total_vega_contrib" : round(total_vega_c, 2),
        "total_slippage"     : round(total_slippage, 2),
        "avg_breakeven_dS"   : round(avg_be, 2) if avg_be else None,
        "total_trades"       : len(df_t),
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  ADVANCED METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def build_advanced_option_metrics(trades):
    if not trades: return ""
    df = pd.DataFrame(trades)
    if df.empty or "Opt_PnL_Total" not in df.columns: return ""

    SEP  = "=" * 69
    DASH = "-" * 69
    lines = ["", SEP, "  OPTION P/L ANALYTICS — v5.0", SEP]

    def safe_div(a, b): return a / b if b != 0 else 0.0
    def fm(v): return f"₹{v:,.2f}" if not pd.isna(v) else "N/A"

    returns = df["Opt_PnL_Total"]
    wins    = df[returns > 0]
    losses  = df[returns < 0]

    net_pnl      = returns.sum()
    avg_trade    = returns.mean()
    med_trade    = returns.median()
    win_rate     = safe_div(len(wins), len(returns))
    avg_win      = wins["Opt_PnL_Total"].mean()   if len(wins)   > 0 else 0
    avg_loss     = losses["Opt_PnL_Total"].mean() if len(losses) > 0 else 0
    rr_ratio     = abs(safe_div(avg_win, avg_loss))
    expectancy   = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    gross_profit = wins["Opt_PnL_Total"].sum()
    gross_loss   = abs(losses["Opt_PnL_Total"].sum())
    pf           = safe_div(gross_profit, gross_loss)
    total_slip   = df["Opt_Slippage"].sum()

    lines += [
        "  SECTION 1 — PERFORMANCE METRICS", DASH,
        f"  Total Net P/L        : {fm(net_pnl)}",
        f"  Total Slippage Cost  : {fm(total_slip)}  (₹{SLIPPAGE_PER_CONTRACT}/trade)",
        f"  Average Trade P/L    : {fm(avg_trade)}",
        f"  Median Trade P/L     : {fm(med_trade)}",
        f"  Win Rate             : {win_rate*100:.2f}%",
        f"  Average Winner       : {fm(avg_win)}",
        f"  Average Loser        : {fm(avg_loss)}",
        f"  Risk Reward Ratio    : {rr_ratio:.2f}",
        f"  Expectancy           : {fm(expectancy)}",
        f"  Profit Factor        : {pf:.2f}",
        SEP,
    ]

    equity_curve = returns.cumsum()
    rolling_max  = equity_curve.cummax()
    drawdown     = equity_curve - rolling_max
    max_drawdown = drawdown.min()
    if rolling_max.max() > 0:
        drawdown_pct = (drawdown / rolling_max.replace(0, np.nan)) * 100
        mdd_pct      = drawdown_pct.min()
    else:
        drawdown_pct = pd.Series(np.zeros(len(drawdown)))
        mdd_pct      = 0.0
    ulcer    = np.sqrt((drawdown_pct ** 2).mean())
    std_dev  = returns.std()
    downside = losses["Opt_PnL_Total"].std() if len(losses) > 1 else 0

    capital_used  = (df["Entry Price"] * OPTION_LOT_SIZE
                     if "Entry Price" in df.columns
                     else pd.Series([OPTION_LOT_SIZE * 22000] * len(df)))
    trade_returns = returns / capital_used
    sharpe_raw    = safe_div(trade_returns.mean(), trade_returns.std())
    trading_days  = df["Date"].nunique() if "Date" in df.columns else 1
    tpy           = safe_div(len(df), max(trading_days, 1)) * 250
    sharpe_ann    = sharpe_raw * math.sqrt(max(tpy, 1))
    sortino_raw   = safe_div(trade_returns.mean(),
                             trade_returns[trade_returns < 0].std()
                             if (trade_returns < 0).any() else 1)
    calmar        = safe_div(returns.mean() * 252, abs(max_drawdown))

    lines += [
        "  SECTION 2 — RISK METRICS", DASH,
        f"  Maximum Drawdown (MDD) : {fm(max_drawdown)}",
        f"  Max Drawdown %         : {mdd_pct:.2f}%",
        f"  Ulcer Index            : {ulcer:.4f}",
        f"  Std Dev of Returns     : {std_dev:.2f}",
        f"  Downside Deviation     : {downside:.2f}",
        f"  Sharpe (raw)           : {sharpe_raw:.4f}",
        f"  Sharpe (annualised)    : {sharpe_ann:.4f}",
        f"  Sortino Ratio          : {sortino_raw:.4f}",
        f"  Calmar Ratio           : {calmar:.4f}",
        SEP,
    ]

    lot           = df["Opt_LotSize"].iloc[0] if "Opt_LotSize" in df.columns else OPTION_LOT_SIZE
    avg_delta_exp = (df["Opt_DeltaEntry"].abs() * lot).mean()
    gex           = (df["Opt_Gamma"] * lot).sum()
    theta_per_hr  = OPTION_THETA / TRADING_HOURS_PER_DAY
    tot_gamma_abs = df["Opt_GammaPnL"].abs().sum()
    tot_opt_abs   = (df["Opt_Change"].abs() * lot).sum()
    gamma_util    = safe_div(tot_gamma_abs, tot_opt_abs)
    theta_abs     = (df["Opt_ThetaEffect"].abs() * lot).sum()
    theta_pct     = safe_div(theta_abs, tot_opt_abs) * 100
    delta_drift   = (df["Opt_DeltaExit"].abs() - df["Opt_DeltaEntry"].abs()).mean()

    lines += [
        "  SECTION 3 — GREEKS EXPOSURE", DASH,
        f"  Avg Delta Exposure   : {avg_delta_exp:.4f}",
        f"  Gamma Exposure (GEX) : {gex:.6f}",
        f"  Theta / Hour         : {theta_per_hr:.4f}",
        f"  Avg Delta Drift      : {delta_drift:.4f}",
        f"  Gamma Utilisation    : {gamma_util:.4f}x",
        f"  Theta Contrib %      : {theta_pct:.2f}%",
        SEP,
    ]

    be_all     = df["Opt_BreakevenDS"].dropna()
    be_pos     = be_all[be_all > 0]; be_neg = be_all[be_all < 0]
    be_avg_pos = be_pos.mean() if not be_pos.empty else float("nan")
    be_avg_all = be_all.mean() if not be_all.empty else float("nan")
    pct_needs  = len(be_pos) / len(be_all) * 100 if len(be_all) > 0 else 0
    lines += [
        "  SECTION 4 — BREAK-EVEN", DASH,
        (f"  Avg Move Req (pos) : {be_avg_pos:.2f} pts" if not np.isnan(be_avg_pos)
         else "  Avg Move Req (pos) : N/A"),
        (f"  Avg Move Req (all) : {be_avg_all:.2f} pts" if not np.isnan(be_avg_all)
         else "  Avg Move Req (all) : N/A"),
        f"  Needs move         : {pct_needs:.1f}%  ({len(be_pos)} of {len(be_all)})",
        f"  Already profitable : {100-pct_needs:.1f}%  ({len(be_neg)} of {len(be_all)})",
        SEP,
    ]

    skew       = returns.skew()
    kurt       = returns.kurtosis()
    left_tail  = returns.quantile(0.05)
    right_tail = returns.quantile(0.95)
    hold_hrs   = df["Holding Mins"].sum() / 60.0 if "Holding Mins" in df.columns else 0
    pnl_per_hr = safe_div(net_pnl, hold_hrs)
    lines += [
        "  SECTION 5 — DISTRIBUTION & TIME", DASH,
        f"  Skewness             : {skew:.2f}",
        f"  Kurtosis             : {kurt:.2f}",
        f"  Left Tail (5%)       : {fm(left_tail)}",
        f"  Right Tail (95%)     : {fm(right_tail)}",
        f"  Profit per Hour      : {fm(pnl_per_hr)}",
        SEP,
    ]

    is_win  = returns > 0
    streaks = is_win.ne(is_win.shift()).cumsum()
    cons_w  = is_win.groupby(streaks).sum().max()
    cons_l  = (~is_win).groupby(streaks).sum().max()
    x_eq    = np.arange(len(equity_curve))
    slope_e, _ = np.polyfit(x_eq, equity_curve, 1) if len(x_eq) > 1 else (0, 0)
    r2      = np.corrcoef(x_eq, equity_curve)[0, 1] ** 2 if len(x_eq) > 1 else 0
    lines += [
        "  SECTION 6 — STABILITY", DASH,
        f"  Equity Curve Slope   : {slope_e:.2f}",
        f"  Equity Curve R²      : {r2:.4f}",
        f"  Consecutive Wins     : {cons_w}",
        f"  Consecutive Losses   : {cons_l}",
        SEP,
    ]

    sorted_r  = returns.sort_values(ascending=False)
    n_rem     = min(5, len(sorted_r) - 1)
    trimmed   = sorted_r.iloc[n_rem:].sum()
    drop_pct  = (1 - safe_div(trimmed, net_pnl)) * 100 if net_pnl != 0 else 0
    lines += [
        "  SECTION 7 — TAIL DEPENDENCY", DASH,
        f"  Full P/L              : {fm(net_pnl)}",
        f"  P/L (top {n_rem} removed)  : {fm(trimmed)}",
        f"  Drop %                : {drop_pct:.1f}%",
        f"  Tail Dependent?       : {'⚠ YES' if drop_pct > 30 else '✅ NO'}",
        SEP + "\n",
    ]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHARTS
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
        name="ST Bull", mode="lines", line=dict(color="#00d4aa", width=2)))
    fig.add_trace(go.Scatter(
        x=df_c.index, y=df_c["ST_val"].where(~df_c["ST_bull"]),
        name="ST Bear", mode="lines", line=dict(color="#ff4757", width=2)))
    fig.add_trace(go.Scatter(
        x=df_c.index, y=df_c["EMA9"],
        name="EMA9", mode="lines", line=dict(color="#ffd700", width=1.5, dash="dot")))
    fig.update_layout(
        template="plotly_dark", height=540, xaxis_rangeslider_visible=False,
        margin=dict(l=50,r=20,t=40,b=40),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
        font=dict(family="'Courier New', monospace", size=11))
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat","mon"]),
        dict(bounds=[15.5, 9.25], pattern="hour")])
    return _fig_to_json(fig)

def build_slope_chart(df):
    if CHART_DAYS:
        udays  = sorted(df.index.normalize().unique())
        cutoff = udays[-CHART_DAYS] if len(udays) >= CHART_DAYS else udays[0]
        df_c   = df[df.index.normalize() >= cutoff].copy()
    else:
        df_c = df.copy()

    slope_deg = df_c["EMA9_slope"].apply(
        lambda s: slope_to_deg(s) if not np.isnan(s) else np.nan)
    colors = ["#00d4aa" if v >= 0 else "#ff4757" for v in slope_deg.fillna(0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_c.index, y=slope_deg,
                         marker_color=colors, name="EMA9 Slope (°)"))
    fig.add_hline(y=EMA9_SLOPE_MIN_DEG,  line_dash="dash",
                  line_color="#ffd700", annotation_text=f"+{EMA9_SLOPE_MIN_DEG}° entry threshold")
    fig.add_hline(y=-EMA9_SLOPE_MIN_DEG, line_dash="dash",
                  line_color="#ffd700", annotation_text=f"-{EMA9_SLOPE_MIN_DEG}° entry threshold")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        template="plotly_dark", height=320,
        title=f"EMA9 Slope Angle (°) — {EMA9_SLOPE_CANDLES}-candle rolling regression",
        margin=dict(l=50,r=20,t=60,b=40),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="'Courier New', monospace"))
    return _fig_to_json(fig)

def build_pnl_chart(df_t):
    cumulative = df_t["Opt_PnL_Total"].cumsum()
    colors = ["#00d4aa" if v >= 0 else "#ff4757" for v in df_t["Opt_PnL_Total"]]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        subplot_titles=("Option P&L per Trade (₹)", "Cumulative P&L (₹)"),
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
    gamma_tot = df_t["Opt_GammaPnL"].sum()
    theta_tot = df_t["Opt_ThetaEffect"].sum() * OPTION_LOT_SIZE
    vega_tot  = df_t["Opt_VegaEffect"].sum() * OPTION_LOT_SIZE
    fig = go.Figure(go.Bar(
        x=["Delta","Gamma","Theta","Vega"],
        y=[delta_tot, gamma_tot, theta_tot, vega_tot],
        marker_color=["#00d4aa","#3d9bff","#ff4757","#ffd700"],
        text=[f"₹{v:,.0f}" for v in [delta_tot, gamma_tot, theta_tot, vega_tot]],
        textposition="outside"))
    fig.update_layout(template="plotly_dark", height=380,
                      title="Greek P&L Attribution (₹)",
                      margin=dict(l=50,r=20,t=60,b=40),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"))
    return _fig_to_json(fig)

def build_equity_drawdown_chart(df_t):
    equity   = df_t["Opt_PnL_Total"].cumsum()
    peak_eq  = equity.cummax()
    drawdown = equity - peak_eq
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Equity Curve (₹)", "Drawdown (₹)"),
                        vertical_spacing=0.12)
    idx = list(range(1, len(df_t)+1))
    fig.add_trace(go.Scatter(x=idx, y=equity, name="Equity",
                             line=dict(color="#ffd700", width=2),
                             fill="tozeroy", fillcolor="rgba(255,215,0,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=peak_eq, name="Peak",
                             line=dict(color="#00d4aa", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=drawdown, name="Drawdown",
                             line=dict(color="#ff4757", width=2),
                             fill="tozeroy", fillcolor="rgba(255,71,87,0.15)"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=480,
                      margin=dict(l=50,r=20,t=60,b=40),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"),
                      legend=dict(orientation="h", y=1.02))
    return _fig_to_json(fig)

def build_exit_reason_chart(df_t):
    reason_counts = df_t["Exit Reason"].value_counts()
    fig = go.Figure(go.Bar(
        x=reason_counts.index.tolist(),
        y=reason_counts.values.tolist(),
        marker_color=["#00d4aa","#3d9bff","#ff4757","#ffd700","#ff9f43"][:len(reason_counts)],
        text=reason_counts.values.tolist(), textposition="outside"))
    fig.update_layout(template="plotly_dark", height=360,
                      title="Exit Reason Breakdown",
                      margin=dict(l=50,r=20,t=60,b=80),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"),
                      xaxis_tickangle=-20)
    return _fig_to_json(fig)

def build_daily_opt_pnl_chart(df_day):
    colors = ["#00d4aa" if v >= 0 else "#ff4757" for v in df_day["Opt P&L (₹)"]]
    fig = go.Figure(go.Bar(
        x=df_day["Date"], y=df_day["Opt P&L (₹)"],
        marker_color=colors,
        text=[f"₹{v:,.0f}" for v in df_day["Opt P&L (₹)"]],
        textposition="outside"))
    fig.update_layout(template="plotly_dark", height=380,
                      title="Daily Option P&L — Net after Slippage (₹)",
                      margin=dict(l=50,r=20,t=60,b=80),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"),
                      xaxis_tickangle=-30)
    return _fig_to_json(fig)

# ═══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def df_to_html_table(df, id_="", row_color_col=None, pnl_col=None):
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = []
    for _, row in df.iterrows():
        cls = ""
        if row_color_col and row_color_col in df.columns:
            val = row[row_color_col]
            if isinstance(val, str):
                cls = ("win-row"  if any(k in val for k in ["WIN","Profit","🟢"]) else
                       "loss-row" if any(k in val for k in ["LOSS","Loss","🔴"]) else "")
            elif isinstance(val, (int, float)):
                cls = "win-row" if val > 0 else "loss-row" if val < 0 else ""
        if pnl_col and pnl_col in df.columns:
            try: cls = "win-row" if float(row[pnl_col]) > 0 else "loss-row"
            except: pass

        cells = []
        for c in df.columns:
            v = row[c]; cell_cls = ""
            if c in ("P&L %","Total P&L %","Avg P&L %","Best %","Worst %",
                     "Best Trade %","Worst Trade %","Opt P&L (₹)","Net Points",
                     "PnL_Total","GrossPnL"):
                try:
                    fv = float(v)
                    cell_cls = "pos-val" if fv > 0 else "neg-val" if fv < 0 else ""
                except: pass
            cells.append(f'<td class="{cell_cls}">{v}</td>')
        rows_html.append(f'<tr class="{cls}">{"".join(cells)}</tr>')

    id_attr = f'id="{id_}"' if id_ else ""
    return f"""<div class="table-wrap">
      <table {id_attr}>
        <thead><tr>{headers}</tr></thead>
        <tbody>{"".join(rows_html)}</tbody>
      </table></div>"""

def export_html(df, trades, df_day, df_ts, htf_enabled, vix_series):
    if not trades:
        print("  ⚠  No trades — HTML report skipped."); return
    df_t    = pd.DataFrame(trades)
    opt_sum = build_option_summary(trades)

    candle_json = build_candle_chart(df)
    slope_json  = build_slope_chart(df)
    pnl_json    = build_pnl_chart(df_t)
    attr_json   = build_greek_attribution_chart(df_t)
    eq_dd_json  = build_equity_drawdown_chart(df_t)
    exit_json   = build_exit_reason_chart(df_t)
    daily_json  = (build_daily_opt_pnl_chart(df_day)
                   if df_day is not None and not df_day.empty else None)

    tl_cols = ["Date","Direction","Entry Time","Entry Price","Hard SL",
               "Exit Time","Exit Price","Holding Time","Points Captured",
               "P&L %","Exit Reason","Result",
               "Opt_DeltaEffect","Opt_GammaEffect","Opt_GammaPnL",
               "Opt_ThetaEffect","Opt_VegaEffect","Opt_Change",
               "Opt_GrossPnL","Opt_Slippage","Opt_PnL_Total","Opt_BreakevenDS"]
    tl_display = df_t[[c for c in tl_cols if c in df_t.columns]].copy()
    tl_display.columns = [c.replace("Opt_","") for c in tl_display.columns]
    tl_html  = df_to_html_table(tl_display, id_="trade-log-tbl", row_color_col="Result")
    day_html = df_to_html_table(df_day, row_color_col="Day Result") if df_day is not None else ""
    ts_html  = df_to_html_table(df_ts,  row_color_col="Verdict")   if df_ts  is not None else ""

    total = len(df_t); wins = (df_t["P&L %"] > 0).sum()
    net_pts = (df_t.loc[df_t["P&L %"]>0,"Points Captured"].sum()
             - df_t.loc[df_t["P&L %"]<=0,"Points Captured"].sum())

    def card(label, value, sub="", cls=""):
        return f"""<div class="stat-card {cls}">
          <div class="stat-label">{label}</div>
          <div class="stat-value">{value}</div>
          {"<div class='stat-sub'>"+sub+"</div>" if sub else ""}
        </div>"""

    opt_cls = "pos-card" if opt_sum["total_opt_pnl"] >= 0 else "neg-card"
    strat_cards = f"""
    {card("Total Trades", total)}
    {card("Win Rate", f"{wins/total*100:.1f}%", f"{wins}W / {total-wins}L")}
    {card("Net Points", f"{net_pts:.2f}", "win pts – loss pts")}
    {card("Trading Days", len(df_day) if df_day is not None else 0)}
    {card("HTF Filter", "ON" if htf_enabled else "OFF", f"ST({HTF_ST_PERIOD},{HTF_ST_MULTIPLIER})")}
    {card("EMA9 Gap", f"{EMA9_ENTRY_GAP} pts", "open vs EMA9 at entry")}
    {card("Slope Min", f"{EMA9_SLOPE_MIN_DEG}°", f"{EMA9_SLOPE_CANDLES}-candle regression")}
    {card("Peak EMA9 Drop", f"{PEAK_EMA9_DROP} pts", "exit if close < peak_ema9 - X")}
    {card("Candle Body", f">{CANDLE_BODY_MIN} pts", "|open-close| filter")}
    {card("Hard SL", f"{HARD_SL_PCT}%", "from entry price")}
    """
    opt_cards = f"""
    {card("Total Option P&L", f"₹{opt_sum['total_opt_pnl']:,.2f}", f"{opt_sum['opt_wins']}W / {opt_sum['opt_losses']}L", opt_cls)}
    {card("Option Win Rate", f"{opt_sum['opt_win_rate']}%")}
    {card("Total Profit", f"₹{opt_sum['win_opt_pnl']:,.2f}", "winning trades", "pos-card")}
    {card("Total Loss", f"₹{opt_sum['loss_opt_pnl']:,.2f}", "losing trades", "neg-card")}
    {card("Delta Contrib", f"₹{opt_sum['total_delta_contrib']:,.2f}")}
    {card("Gamma Contrib", f"₹{opt_sum['total_gamma_contrib']:,.2f}")}
    {card("Theta Cost", f"₹{opt_sum['total_theta_cost']:,.2f}", "", "neg-card")}
    {card("Vega Contrib", f"₹{opt_sum['total_vega_contrib']:,.2f}")}
    {card("Slippage", f"₹{opt_sum['total_slippage']:,.2f}", f"₹{SLIPPAGE_PER_CONTRACT}/trade", "neg-card")}
    {card("Avg Break-even ΔS", f"{opt_sum['avg_breakeven_dS']} pts" if opt_sum['avg_breakeven_dS'] else "—")}
    """

    logic_box = f"""<div class="formula-box">
      <p class="formula-title">v5.0 — Entry &amp; Exit Logic</p>
      <code>── ENTRY (all must be true on same candle) ──────────────────────────────</code>
      <code>  1. LTF ST bullish (bearish for short)</code>
      <code>  2. HTF ST bullish (bearish for short)  [HTF_FILTER_ENABLED={HTF_FILTER_ENABLED}]</code>
      <code>  3. Open &gt; EMA{EMA9_PERIOD} + {EMA9_ENTRY_GAP} pts  (long)  |  Open &lt; EMA{EMA9_PERIOD} - {EMA9_ENTRY_GAP} pts  (short)</code>
      <code>  4. EMA{EMA9_PERIOD} slope ({EMA9_SLOPE_CANDLES}-candle regression) angle ≥ {EMA9_SLOPE_MIN_DEG}°  and in correct direction</code>
      <code>  5. |open - close| &gt; {CANDLE_BODY_MIN} pts</code>
      <code>  Entry price = candle close</code>
      <br>
      <code>── EXIT PRIORITY (first triggered wins) ──────────────────────────────────</code>
      <code>  1. Hard SL   : {HARD_SL_PCT}% adverse from entry  →  exit_px = hard_sl_price</code>
      <code>  2. Peak EMA9 : close &lt; highest-EMA9-since-entry - {PEAK_EMA9_DROP} pts  →  exit_px = c_close</code>
      <code>               (short: close &gt; lowest-EMA9-since-entry + {PEAK_EMA9_DROP} pts)</code>
      <code>  3. Slope Rev : EMA9 slope turns negative (long) / positive (short)  →  exit_px = c_close</code>
      <code>  4. EOD       : forced exit at {MARKET_CLOSE}  →  exit_px = c_close</code>
    </div>"""

    def plot_div(fig_json, div_id):
        if fig_json is None:
            return f'<div class="chart-placeholder">No data</div>'
        return f'<div id="{div_id}" class="chart-container"></div>'

    def plot_script(fig_json, div_id):
        if fig_json is None: return ""
        return (f"Plotly.react('{div_id}',{json.dumps(fig_json['data'])},"
                f"{json.dumps(fig_json['layout'])},{{responsive:true}});\n")

    all_scripts = (
        plot_script(candle_json, "chart-candle") +
        plot_script(slope_json,  "chart-slope")  +
        plot_script(pnl_json,    "chart-pnl")    +
        plot_script(attr_json,   "chart-attr")   +
        plot_script(eq_dd_json,  "chart-eqdd")   +
        plot_script(exit_json,   "chart-exit")   +
        (plot_script(daily_json, "chart-daily") if daily_json else "")
    )

    CSS = """
  :root{--bg:#0a0e14;--bg2:#0d1117;--bg3:#161b22;--bg4:#1c2128;
    --border:#21262d;--border2:#30363d;--green:#00d4aa;--red:#ff4757;
    --gold:#ffd700;--blue:#3d9bff;--text:#c9d1d9;--text2:#8b949e;--text3:#6e7681;
    --win-bg:rgba(0,212,170,0.07);--loss-bg:rgba(255,71,87,0.07);}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;font-size:15px;line-height:1.5;}
  ::-webkit-scrollbar{width:6px;height:6px;}
  ::-webkit-scrollbar-track{background:var(--bg2);}
  ::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;}
  header{background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);border-bottom:1px solid var(--border2);padding:28px 36px 20px;}
  .logo{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--text3);letter-spacing:2px;text-transform:uppercase;}
  h1{font-size:2rem;font-weight:700;background:linear-gradient(90deg,var(--green),var(--gold));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .meta{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--text3);margin-top:6px;}
  nav{background:var(--bg3);border-bottom:1px solid var(--border);padding:0 36px;display:flex;overflow-x:auto;}
  nav a{display:inline-block;padding:12px 18px;color:var(--text2);text-decoration:none;font-weight:600;font-size:13px;border-bottom:2px solid transparent;white-space:nowrap;transition:all .2s;}
  nav a:hover{color:var(--green);border-bottom-color:var(--green);}
  main{padding:28px 36px;max-width:1600px;}
  section{margin-bottom:52px;scroll-margin-top:60px;}
  h2{font-size:1.25rem;font-weight:700;color:var(--gold);margin-bottom:18px;padding-bottom:8px;border-bottom:1px solid var(--border2);}
  h3{font-size:1rem;font-weight:600;color:var(--text2);margin:20px 0 10px;}
  .card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:14px;}
  .stat-card{background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:16px 18px;position:relative;overflow:hidden;}
  .stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--green),transparent);}
  .pos-card::before{background:linear-gradient(90deg,var(--green),transparent);}
  .neg-card::before{background:linear-gradient(90deg,var(--red),transparent);}
  .stat-label{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--text3);margin-bottom:6px;}
  .stat-value{font-size:1.5rem;font-weight:700;font-family:'Share Tech Mono',monospace;color:var(--text);}
  .pos-card .stat-value{color:var(--green);}
  .neg-card .stat-value{color:var(--red);}
  .stat-sub{font-size:11px;color:var(--text3);margin-top:4px;}
  .table-wrap{overflow-x:auto;border-radius:8px;border:1px solid var(--border);margin-bottom:18px;}
  table{width:100%;border-collapse:collapse;font-size:12.5px;}
  thead tr{background:var(--bg4);}
  th{padding:10px 12px;text-align:left;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--text2);border-bottom:1px solid var(--border2);white-space:nowrap;}
  td{padding:8px 12px;border-bottom:1px solid var(--border);white-space:nowrap;font-family:'Share Tech Mono',monospace;font-size:12px;}
  tr:last-child td{border-bottom:none;}
  tr:hover td{background:rgba(255,255,255,.025);}
  .win-row td{background:var(--win-bg);}
  .loss-row td{background:var(--loss-bg);}
  .pos-val{color:var(--green)!important;font-weight:700;}
  .neg-val{color:var(--red)!important;font-weight:700;}
  .chart-container{border-radius:8px;border:1px solid var(--border);overflow:hidden;background:var(--bg2);margin-bottom:18px;}
  .chart-placeholder{background:var(--bg3);border:1px dashed var(--border2);border-radius:8px;padding:40px;text-align:center;color:var(--text3);}
  .chart-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
  .formula-box{background:var(--bg4);border:1px solid var(--border2);border-left:3px solid var(--gold);border-radius:6px;padding:16px 20px;margin-top:16px;}
  .formula-title{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--gold);margin-bottom:10px;}
  code{font-family:'Share Tech Mono',monospace;font-size:13px;color:var(--green);display:block;margin:4px 0;}
  .disclaimer{background:var(--bg3);border:1px solid var(--border2);border-left:3px solid var(--red);border-radius:6px;padding:14px 18px;margin-bottom:28px;font-size:12px;color:var(--text3);}
  footer{padding:24px 36px;border-top:1px solid var(--border);font-size:11px;color:var(--text3);text-align:center;font-family:'Share Tech Mono',monospace;}
  @media(max-width:900px){main{padding:16px 12px;}.chart-row{grid-template-columns:1fr;}h1{font-size:1.4rem;}}
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{SYMBOL} — Supertrend + Options Report v5.0</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="logo">{SYMBOL} · Supertrend v5.0 · EMA9 Slope + Peak EMA9 Exit</div>
  <div>
    <h1>⚡ {SYMBOL} — Options Backtest Report v5.0</h1>
    <div class="meta">
      ST({ST_PERIOD},{ST_MULTIPLIER}) LTF &nbsp;|&nbsp;
      HTF ST({HTF_ST_PERIOD},{HTF_ST_MULTIPLIER}): {"ON" if htf_enabled else "OFF"} &nbsp;|&nbsp;
      EMA{EMA9_PERIOD} gap={EMA9_ENTRY_GAP}pts slope≥{EMA9_SLOPE_MIN_DEG}° ({EMA9_SLOPE_CANDLES}c) &nbsp;|&nbsp;
      Body&gt;{CANDLE_BODY_MIN}pts &nbsp;|&nbsp;
      SL={HARD_SL_PCT}% PeakDrop={PEAK_EMA9_DROP}pts &nbsp;|&nbsp;
      Lot={OPTION_LOT_SIZE} Δ={OPTION_DELTA} Γ={OPTION_GAMMA} Θ={OPTION_THETA}/d &nbsp;|&nbsp;
      {OPTION_DIRECTION.upper()} Slip=₹{SLIPPAGE_PER_CONTRACT}
    </div>
  </div>
</header>
<nav>
  <a href="#sec-overview">📊 Overview</a>
  <a href="#sec-logic">📐 Logic</a>
  <a href="#sec-chart">📈 Chart</a>
  <a href="#sec-slope">〰 EMA9 Slope</a>
  <a href="#sec-options">🎯 Option P&amp;L</a>
  <a href="#sec-greeks">⚗️ Greeks</a>
  <a href="#sec-eqdd">📉 Equity/DD</a>
  <a href="#sec-exits">🚪 Exits</a>
  <a href="#sec-trades">📋 Trade Log</a>
  <a href="#sec-daily">📅 Daily</a>
  <a href="#sec-timeslot">🕐 Time Slots</a>
</nav>
<main>
  <div class="disclaimer">⚠ <strong>DISCLAIMER:</strong> Greek approximation via Taylor expansion. Not financial advice.</div>

  <section id="sec-overview">
    <h2>📊 Strategy Overview</h2>
    <div class="card-grid">{strat_cards}</div>
    <h3>Option P&amp;L Summary</h3>
    <div class="card-grid">{opt_cards}</div>
  </section>

  <section id="sec-logic">
    <h2>📐 Entry &amp; Exit Logic</h2>
    {logic_box}
  </section>

  <section id="sec-chart">
    <h2>📈 Price, Supertrend &amp; EMA9 (Last {CHART_DAYS} Days)</h2>
    {plot_div(candle_json, "chart-candle")}
  </section>

  <section id="sec-slope">
    <h2>〰 EMA9 Slope Angle</h2>
    {plot_div(slope_json, "chart-slope")}
  </section>

  <section id="sec-options">
    <h2>🎯 Option P&amp;L Analysis</h2>
    {plot_div(pnl_json, "chart-pnl")}
    {"" if daily_json is None else plot_div(daily_json, "chart-daily")}
  </section>

  <section id="sec-greeks">
    <h2>⚗️ Greeks Attribution</h2>
    {plot_div(attr_json, "chart-attr")}
  </section>

  <section id="sec-eqdd">
    <h2>📉 Equity Curve &amp; Drawdown</h2>
    {plot_div(eq_dd_json, "chart-eqdd")}
  </section>

  <section id="sec-exits">
    <h2>🚪 Exit Reason Breakdown</h2>
    {plot_div(exit_json, "chart-exit")}
  </section>

  <section id="sec-trades">
    <h2>📋 Complete Trade Log</h2>
    {tl_html}
  </section>

  <section id="sec-daily">
    <h2>📅 Daily P&amp;L Breakdown</h2>
    {day_html}
  </section>

  <section id="sec-timeslot">
    <h2>🕐 Time Slot Analysis</h2>
    {ts_html}
  </section>
</main>
<footer>
  {SYMBOL} v5.0 · ST({ST_PERIOD},{ST_MULTIPLIER}) · EMA{EMA9_PERIOD}
  gap={EMA9_ENTRY_GAP}pts slope≥{EMA9_SLOPE_MIN_DEG}° body&gt;{CANDLE_BODY_MIN}pts ·
  SL={HARD_SL_PCT}% PeakDrop={PEAK_EMA9_DROP}pts SlopeRev ·
  {OPTION_DIRECTION.upper()} Lot={OPTION_LOT_SIZE} Slip=₹{SLIPPAGE_PER_CONTRACT}
</footer>
<script>{all_scripts}</script>
</body>
</html>"""

    fname = f"{SYMBOL}_options_report_v5_0.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅  HTML Report → {fname}")
    return fname

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSOLE PRINT
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(trades, htf_enabled):
    SEP  = "═" * 115
    DASH = "─" * 115
    if not trades:
        print("  ⚠  No trades generated."); return None, None, None

    df_t   = pd.DataFrame(trades)
    df_day = build_daily_summary(trades)
    df_ts  = build_time_slot_summary(trades)
    opt    = build_option_summary(trades)
    total  = len(df_t); wins = (df_t["P&L %"] > 0).sum()

    print("\n" + SEP)
    print(f"  {SYMBOL}  ST({ST_PERIOD},{ST_MULTIPLIER})  |  Trades:{total}  "
          f"Win:{wins/total*100:.1f}%  HTF:{'ON' if htf_enabled else 'OFF'}  "
          f"EMA9 gap={EMA9_ENTRY_GAP}pts slope≥{EMA9_SLOPE_MIN_DEG}°  "
          f"body>{CANDLE_BODY_MIN}pts  SL={HARD_SL_PCT}%  PeakDrop={PEAK_EMA9_DROP}pts")
    print(SEP)

    cols = ["Date","Direction","Entry Time","Entry Price","Exit Time","Exit Price",
            "Holding Time","Points Captured","P&L %",
            "Opt_GrossPnL","Opt_Slippage","Opt_PnL_Total","Exit Reason","Result"]
    print(df_t[[c for c in cols if c in df_t.columns]].to_string(index=False))

    print("\n" + SEP); print("  PER-DAY SUMMARY"); print(DASH)
    print(df_day.to_string(index=False))

    print("\n" + SEP); print("  TIME SLOT ANALYSIS"); print(DASH)
    print(df_ts.to_string(index=False))

    print("\n" + SEP); print("  OPTION P&L SUMMARY"); print(DASH)
    print(f"  Total Net P&L      : ₹{opt['total_opt_pnl']:>12,.2f}")
    print(f"  Slippage           : ₹{opt['total_slippage']:>12,.2f}")
    print(f"  Delta Contribution : ₹{opt['total_delta_contrib']:>12,.2f}")
    print(f"  Gamma Contribution : ₹{opt['total_gamma_contrib']:>12,.2f}")
    print(f"  Theta Cost         : ₹{opt['total_theta_cost']:>12,.2f}")
    print(f"  Vega Contribution  : ₹{opt['total_vega_contrib']:>12,.2f}")
    print(f"  Avg Break-even ΔS  : {opt['avg_breakeven_dS']} pts")
    print(SEP + "\n")

    adv = build_advanced_option_metrics(trades)
    if adv: print(adv)

    return df_t, df_day, df_ts

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  SUPERTREND + OPTIONS BACKTEST v5.0  (clean rebuild)")
    print("─"*60)
    print("  ENTRY CONDITIONS:")
    print(f"    1. LTF ST ({ST_PERIOD},{ST_MULTIPLIER}) bullish/bearish")
    print(f"    2. HTF ST ({HTF_ST_PERIOD},{HTF_ST_MULTIPLIER}) same direction  [{'ON' if HTF_FILTER_ENABLED else 'OFF'}]")
    print(f"    3. open > EMA{EMA9_PERIOD} + {EMA9_ENTRY_GAP}pts  (long)  |  open < EMA{EMA9_PERIOD} - {EMA9_ENTRY_GAP}pts  (short)")
    print(f"    4. EMA9 slope ≥ {EMA9_SLOPE_MIN_DEG}° and in direction  ({EMA9_SLOPE_CANDLES}-candle regression)")
    print(f"    5. |open - close| > {CANDLE_BODY_MIN} pts")
    print("  EXIT CONDITIONS (priority):")
    print(f"    1. Hard SL {HARD_SL_PCT}%")
    print(f"    2. Close < peak_EMA9 - {PEAK_EMA9_DROP}pts  (long) / > peak_EMA9 + {PEAK_EMA9_DROP}pts  (short)")
    print(f"    3. EMA9 slope reversal (turns negative for long / positive for short)")
    print(f"    4. EOD")
    print("═"*60)

    try:
        df, htf_enabled, vix_series = fetch_data()
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(e); return

    trades = run_backtest(df, htf_enabled, vix_series)
    df_t, df_day, df_ts = print_results(trades, htf_enabled)
    print("  Building HTML report …")
    fname = export_html(df, trades, df_day, df_ts, htf_enabled, vix_series)
    print(f"\n  ✅  Done.  Report → {fname or 'N/A'}\n")

if __name__ == "__main__":
    main()