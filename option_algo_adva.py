"""
Supertrend Intraday Backtesting Strategy — v4.4 (Greeks / Options Edition)
─────────────────────────────────────────────────────────────────────────────
CHANGES vs v4.3
  NEW — LTF SUPERTREND EXIT FILTER:
    An additional exit condition based on a Lower Time Frame (LTF) Supertrend
    computed on the same base data (1-min) but with a tighter multiplier (2.5).
    Because the multiplier is smaller, the LTF ST line hugs price more closely
    and triggers an exit earlier than the main ST(14, 4.0) would.

    Config keys (in CONFIG section):
      LTF_ST_EXIT_ENABLED = True / False   ← master on/off switch
      LTF_ST_PERIOD       = 14             ← ATR period
      LTF_ST_MULTIPLIER   = 2.5            ← tighter = closer stop

    Exit behaviour:
      Long  : exits when close < LTF_ST_val  (or low touches it if CLOSE_BASED_EXIT=False)
      Short : exits when close > LTF_ST_val  (or high touches it)
    Exit reason label: "LTF ST Exit (x2.5)"

    Priority order (first triggered wins):
      1. Hard SL   2. LTF ST Exit   3. Main ST Exit / Flip

RETAINED from v4.3:
  NEW — 5-POINT CONFIRMATION ENTRY FILTER:
    When all existing entry conditions are satisfied on a candle (the
    "trigger candle"), its close is stored as `trigger_close` instead of
    entering immediately.  An actual trade is opened only when a subsequent
    candle closes:
      • Long  : close > trigger_close + ENTRY_CONFIRM_POINTS  (default 5)
      • Short : close < trigger_close - ENTRY_CONFIRM_POINTS  (default 5)

    The pending trigger is CANCELLED if any of the following occur before
    confirmation fires:
      - Supertrend flips direction
      - HTF filter turns against the trade direction
      - Market close is reached (EOD)
      - A new conflicting trigger overrides it
      - MAX_TRADES_PER_DAY already reached on that day

FIXES from v4.2 (all retained):
  FIX-A  : Gamma attribution uses gamma_effect_static × lot_size always.
  FIX-B  : Break-even average filters to positive-only values.
  FIX-C  : VIX CSV auto-detects tab vs comma separator.

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

DATA_SOURCE      = "yfinance"       # "csv" or "yfinance"
BASE_CSV         = "Book1min.csv"
HTF_CSV          = "Book1.csv"
VIX_CSV          = None          # ← your VIX CSV file

YF_TICKER        = "^NSEI"
YF_BASE_INTERVAL = "5m"
YF_HTF_INTERVAL  = "15m"
YF_PERIOD        = "7d"
YF_START         = None
YF_END           = None
YF_VIX_TICKER    = "^INDIAVIX"

SYMBOL           = "NIFTY"

# ── Supertrend ─────────────────────────────────────────────────────────────────
ST_PERIOD         = 10
ST_MULTIPLIER     = 3
HTF_ST_PERIOD     = 10
HTF_ST_MULTIPLIER = 3.0

# ── Risk ───────────────────────────────────────────────────────────────────────
HARD_SL_ENABLED  = True
HARD_SL_PCT      = 0.25
EXIT_USE_HTF_ST  = False
CLOSE_BASED_EXIT = True

# ── LTF Supertrend Exit (same base timeframe, tighter multiplier) ───────────────
# When True, an open trade also exits if price crosses the LTF ST line
# (computed on the same 1-min data but with a smaller multiplier = tighter stop).
# This acts as an earlier exit signal before the main ST(14, 4.0) is breached.
#
#   LTF_ST_EXIT_ENABLED = True   → LTF ST exit is active
#   LTF_ST_EXIT_ENABLED = False  → LTF ST exit is ignored (v4.3 behaviour)
#
# The exit price used is the LTF ST line value at that candle (same as main ST exit).
# CLOSE_BASED_EXIT applies here too:
#   True  → exit when close crosses the LTF ST line
#   False → exit when the candle's high/low touches the LTF ST line
LTF_ST_EXIT_ENABLED  = True           # ← set False to disable
LTF_ST_PERIOD        = 14             # ATR period for LTF ST (same as main by default)
LTF_ST_MULTIPLIER    = 5.0            # tighter multiplier → closer stop line

# ── Filters ────────────────────────────────────────────────────────────────────
TRADE_WINDOW_ENABLED = True
TRADE_WINDOWS = [
    (dtime(9, 30), dtime(11, 30)),
    (dtime(14,  0), dtime(14, 45)),
]
CONFIRM_CANDLES            = 5
MIN_GAP_ENABLED            = True
MIN_GAP_PCT                = 0.20
VOLUME_FILTER_ENABLED      = False
VOLUME_MULTIPLIER          = 1.5
VOLUME_LOOKBACK            = 25
MAX_TRADES_PER_DAY_ENABLED = True
MAX_TRADES_PER_DAY         = 5

# ── NEW: 5-Point Confirmation Entry ────────────────────────────────────────────
# When all conditions are met on a "trigger candle", the actual entry is
# deferred until a subsequent candle closes beyond this many points away
# from the trigger candle's close.
#   Long  entry fires when: close > trigger_close + ENTRY_CONFIRM_POINTS
#   Short entry fires when: close < trigger_close - ENTRY_CONFIRM_POINTS
ENTRY_CONFIRM_POINTS = 5.0

# ── Market ─────────────────────────────────────────────────────────────────────
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 15)
TRADING_HOURS_PER_DAY = 6.25
IST          = pytz.timezone("Asia/Kolkata")
CHART_DAYS   = 5

# ══════════════════════════════════════════════════════════════════════════════
#  GREEKS PARAMETERS  ← USER INPUTS
# ══════════════════════════════════════════════════════════════════════════════

OPTION_LOT_SIZE   = 65
OPTION_DELTA      = 0.20
OPTION_GAMMA      = 0.0006
OPTION_THETA      = -10.53            # daily theta (negative for long options)
OPTION_VEGA       = 5.0               # per 1-vol-pt change
OPTION_DIRECTION  = "long"            # "long" = buying options; "short" = selling

USE_DYNAMIC_DELTA = True

LOW_POINTS_THRESHOLD = 20

# ── Transaction cost / Slippage ────────────────────────────────────────────────
SLIPPAGE_PER_CONTRACT = 50.0
BROKERAGE_PER_TRADE   = 0.0

# ── Risk normalisation ─────────────────────────────────────────────────────────
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
                       trade_direction="long",
                       vix_at_entry=None, vix_at_exit=None,
                       delta=None, gamma=None, theta_daily=None, vega=None,
                       lot_size=None, opt_direction=None):
    abs_delta   = abs(delta  if delta  is not None else OPTION_DELTA)
    gamma       = gamma      if gamma      is not None else OPTION_GAMMA
    theta_daily = theta_daily if theta_daily is not None else OPTION_THETA
    vega        = vega        if vega        is not None else OPTION_VEGA
    lot_size    = lot_size    if lot_size    is not None else OPTION_LOT_SIZE
    opt_direction = opt_direction or OPTION_DIRECTION

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
    step_size = dS_total / steps
    cur_delta = signed_delta
    delta_effect_dyn = 0.0

    for _ in range(steps):
        delta_effect_dyn += cur_delta * step_size
        if USE_DYNAMIC_DELTA:
            cur_delta = cur_delta + gamma * step_size

    gamma_effect        = 0.0 if USE_DYNAMIC_DELTA else 0.5 * gamma * (dS_total ** 2)
    gamma_effect_static = 0.5 * gamma * (dS_total ** 2)

    option_change = delta_effect_dyn + gamma_effect + theta_effect + vega_effect
    final_delta   = cur_delta

    sign        = 1 if opt_direction == "long" else -1
    pnl_per_lot = option_change * sign
    gross_pnl   = pnl_per_lot * lot_size
    slippage    = SLIPPAGE_PER_CONTRACT + BROKERAGE_PER_TRADE
    total_pnl   = gross_pnl - slippage

    gamma_pnl_total = gamma_effect_static * lot_size

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
        "gamma_pnl_total"     : round(gamma_pnl_total, 4),
        "theta_effect"        : round(theta_effect, 4),
        "vega_effect"         : round(vega_effect, 4),
        "option_change"       : round(option_change, 4),
        "pnl_per_lot"         : round(pnl_per_lot, 4),
        "gross_pnl"           : round(gross_pnl, 2),
        "slippage"            : round(slippage, 2),
        "total_pnl"           : round(total_pnl, 2),
        "lot_size"            : lot_size,
        "opt_direction"       : opt_direction,
        "breakeven_dS"        : breakeven_dS,
        "trade_direction"     : trade_direction,
    }


def _solve_breakeven(delta, gamma, const_term):
    if gamma == 0:
        if delta == 0:
            return None
        return round(-const_term / delta, 4)
    try:
        a = 0.5 * gamma
        b = delta
        c = const_term
        discriminant = b**2 - 4 * a * c
        if discriminant >= 0:
            r1 = (-b + math.sqrt(discriminant)) / (2 * a)
            r2 = (-b - math.sqrt(discriminant)) / (2 * a)
            candidates = [r for r in [r1, r2] if abs(r) < 10000]
            if candidates:
                return round(min(candidates, key=abs), 4)
        def f(x): return a * x * x + b * x + c
        root = brentq(f, -5000, 5000)
        return round(root, 4)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#  CSV / YFINANCE LOADERS
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
    if VIX_CSV and os.path.exists(VIX_CSV):
        print(f"\n  Loading VIX from CSV: {VIX_CSV}")
        for sep in ["\t", ","]:
            try:
                dv = pd.read_csv(VIX_CSV, sep=sep)
                if len(dv.columns) >= 2:
                    break
            except Exception:
                continue
        dt_col = _find_col(dv.columns, _DT_CANDS)
        c_col  = _find_col(dv.columns, _C_CANDS)
        if dt_col and c_col:
            dv[dt_col] = pd.to_datetime(dv[dt_col], errors="coerce")
            dv = dv.dropna(subset=[dt_col]).set_index(dt_col)
            dv.index = dv.index.normalize()
            vix_out = dv[c_col].rename("VIX")
            print(f"    VIX rows loaded: {len(vix_out)}  |  {vix_out.index[0].date()} → {vix_out.index[-1].date()}")
            return vix_out
        else:
            print(f"  ⚠  VIX CSV loaded but could not find Date/Close columns. Found: {list(dv.columns)}")
            return None
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

    # ── LTF Supertrend — computed on same base data, tighter multiplier ────────
    if LTF_ST_EXIT_ENABLED:
        df_ltf = compute_supertrend(df[["Open","High","Low","Close","Volume"]].copy(),
                                    LTF_ST_PERIOD, LTF_ST_MULTIPLIER)
        df["LTF_ST_val"]  = df_ltf["ST_val"].values
        df["LTF_ST_bull"] = df_ltf["ST_bull"].values
        df.dropna(subset=["LTF_ST_val"], inplace=True)
        print(f"  ✅  LTF ST ENABLED — ST({LTF_ST_PERIOD},{LTF_ST_MULTIPLIER})")
    else:
        # Fill with sentinel values so the backtest loop can always index them
        df["LTF_ST_val"]  = np.nan
        df["LTF_ST_bull"] = True
        print(f"  ⚪  LTF ST EXIT DISABLED")

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
#  BACKTEST ENGINE  — v4.3: 5-POINT CONFIRMATION ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(df, htf_enabled, vix_series):
    trades = []

    # ── Active trade state ────────────────────────────────────────────────────
    in_trade      = False
    direction     = None
    entry_price   = None
    entry_time    = None
    peak          = None
    hard_sl_price = None

    # ── NEW: Pending confirmation trigger state ───────────────────────────────
    # When all entry conditions are met on candle i, we record:
    #   pending_direction  : "long" or "short"
    #   pending_trigger_close : close price of the trigger candle
    #   pending_date_key   : date of the trigger (for daily trade count check)
    # Entry fires on the NEXT candle(s) once the confirmation threshold is hit.
    # The trigger is cleared on any invalidating event (flip, HTF change, EOD).
    pending_direction     = None
    pending_trigger_close = None
    pending_date_key      = None

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

    # LTF ST arrays — populated only when LTF_ST_EXIT_ENABLED=True;
    # otherwise ltf_st_vals is all-nan and the exit check is always False.
    ltf_st_vals  = df["LTF_ST_val"].values.astype(float)
    ltf_st_bulls = df["LTF_ST_bull"].values.astype(bool)

    exit_st  = htf_vals if (EXIT_USE_HTF_ST and htf_enabled) else st_vals
    vol_avg  = pd.Series(volumes).rolling(VOLUME_LOOKBACK).mean().values
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

        vix_e = get_vix(e_time)
        vix_x = get_vix(x_time)
        opt = compute_option_pnl(
            entry_price_S  = e_price,
            exit_price_S   = x_price,
            holding_hours  = holding_hours,
            trade_direction= dir_,
            vix_at_entry   = vix_e,
            vix_at_exit    = vix_x,
        )

        trades.append({
            "Date"              : e_time.strftime("%Y-%m-%d"),
            "Direction"         : "Long  ↑" if dir_ == "long" else "Short ↓",
            "Entry Time"        : fmt(e_time),
            "Entry Price"       : round(e_price, 2),
            "Hard SL Price"     : round(hard_sl, 2) if hard_sl else "OFF",
            "Exit ST Line"      : round(sl_at_exit, 2),
            "Exit Time"         : fmt(x_time),
            "Exit Price"        : round(x_price, 2),
            "Peak"              : round(pk, 2),
            "Holding Mins"      : hold_mins,
            "Holding Time"      : hold_str,
            "Points Captured"   : round(abs(x_price - e_price), 4),
            "P&L %"             : pnl,
            "Exit Reason"       : reason,
            "Result"            : "WIN" if pnl > 0 else "LOSS",
            "Opt_dS"            : opt["dS"],
            "Opt_DeltaEntry"    : opt["delta_entry"],
            "Opt_DeltaExit"     : opt["delta_exit"],
            "Opt_Gamma"         : opt["gamma"],
            "Opt_Theta_daily"   : opt["theta_daily"],
            "Opt_Vega"          : opt["vega"],
            "Opt_VIX_Entry"     : opt["vix_at_entry"],
            "Opt_VIX_Exit"      : opt["vix_at_exit"],
            "Opt_VIX_Chg"       : opt["vix_change"],
            "Opt_DeltaEffect"   : opt["delta_effect"],
            "Opt_GammaEffect"   : opt["gamma_effect"],
            "Opt_GammaPnL"      : opt["gamma_pnl_total"],
            "Opt_ThetaEffect"   : opt["theta_effect"],
            "Opt_VegaEffect"    : opt["vega_effect"],
            "Opt_Change"        : opt["option_change"],
            "Opt_PnL_PerLot"    : opt["pnl_per_lot"],
            "Opt_GrossPnL"      : opt["gross_pnl"],
            "Opt_Slippage"      : opt["slippage"],
            "Opt_PnL_Total"     : opt["total_pnl"],
            "Opt_BreakevenDS"   : opt["breakeven_dS"],
            "Opt_LotSize"       : opt["lot_size"],
        })

    def open_trade(dir_, price, time_, high_, low_, date_key):
        nonlocal in_trade, direction, entry_price, entry_time, peak, hard_sl_price
        in_trade      = True
        direction     = dir_
        entry_price   = price
        entry_time    = time_
        peak          = high_ if dir_ == "long" else low_
        hard_sl_price = (round(price * (1 - HARD_SL_PCT/100), 4) if dir_ == "long"
                         else round(price * (1 + HARD_SL_PCT/100), 4)) if HARD_SL_ENABLED else None
        daily_trades[date_key] = daily_trades.get(date_key, 0) + 1

    def clear_pending():
        """Cancel any pending confirmation trigger."""
        nonlocal pending_direction, pending_trigger_close, pending_date_key
        pending_direction     = None
        pending_trigger_close = None
        pending_date_key      = None

    def set_pending(dir_, trigger_close, date_key):
        """
        Record a pending entry trigger.
        dir_          : "long" or "short"
        trigger_close : close price of the candle that satisfied all conditions
        date_key      : date object for daily trade count
        """
        nonlocal pending_direction, pending_trigger_close, pending_date_key
        pending_direction     = dir_
        pending_trigger_close = trigger_close
        pending_date_key      = date_key

    # ─────────────────────────────────────────────────────────────────────────
    for i in range(1, len(df)):
        c_close    = closes[i]
        c_high     = highs[i]
        c_low      = lows[i]
        c_time     = times[i]
        c_tod      = c_time.time()
        c_st       = st_vals[i]
        c_exit_st  = exit_st[i]
        c_bull     = st_bulls[i]
        prev_bull  = st_bulls[i-1]
        c_htf_bull = htf_bulls[i]
        date_key   = c_time.date()
        prev_close = closes[i-1]

        # LTF ST values for this candle
        c_ltf_st_val  = ltf_st_vals[i]
        c_ltf_st_bull = ltf_st_bulls[i]

        flipped_bull = (not prev_bull) and c_bull
        flipped_bear = prev_bull and (not c_bull)
        if flipped_bull or flipped_bear:
            flip_candle = i
            # Any ST flip cancels a pending trigger — the market structure changed
            clear_pending()

        # ── EOD forced exit ───────────────────────────────────────────────────
        if c_tod >= MARKET_CLOSE:
            if in_trade:
                pk_f = max(peak, c_high) if direction == "long" else min(peak, c_low)
                record(direction, entry_time, entry_price, c_time, c_close,
                       pk_f, c_exit_st, "EOD Exit", hard_sl_price)
                in_trade = False
            clear_pending()   # cancel any trigger at EOD
            continue

        if c_tod < MARKET_OPEN:
            continue

        # ── Manage open trade ─────────────────────────────────────────────────
        if in_trade:
            if direction == "long":
                if c_high > peak: peak = c_high
                hard_hit = HARD_SL_ENABLED and hard_sl_price and c_low <= hard_sl_price

                # ── LTF ST exit check (priority 2 — between Hard SL and main ST) ──
                # Fires when LTF ST(14, 2.5) line is breached before main ST(14, 4.0).
                # Only active when LTF_ST_EXIT_ENABLED=True AND LTF ST value is valid.
                ltf_hit = (LTF_ST_EXIT_ENABLED
                           and not np.isnan(c_ltf_st_val)
                           and (
                               (CLOSE_BASED_EXIT and c_close < c_ltf_st_val) or
                               (not CLOSE_BASED_EXIT and c_low <= c_ltf_st_val)
                           ))

                st_hit   = ((c_close < c_exit_st) or flipped_bear) if CLOSE_BASED_EXIT \
                           else (c_low <= c_exit_st or flipped_bear)

                if hard_hit or ltf_hit or st_hit:
                    if hard_hit:
                        reason  = f"Hard SL ({HARD_SL_PCT}%)"
                        exit_px = hard_sl_price
                    elif ltf_hit:
                        reason  = f"LTF ST Exit ({LTF_ST_MULTIPLIER}x)"
                        exit_px = c_ltf_st_val
                    elif (CLOSE_BASED_EXIT and c_close < c_exit_st) or \
                         (not CLOSE_BASED_EXIT and c_low <= c_exit_st):
                        reason  = "HTF ST Exit" if (EXIT_USE_HTF_ST and htf_enabled) else "ST Stop Loss"
                        exit_px = c_exit_st
                    else:
                        reason  = "ST Flip Bear"
                        exit_px = c_exit_st

                    record("long", entry_time, entry_price, c_time, round(exit_px, 2),
                           peak, c_exit_st, reason, hard_sl_price)
                    in_trade = False

                    # Attempt immediate reverse-to-short trigger (set pending, not open)
                    if (not hard_hit and not ltf_hit and not c_bull and
                            (not htf_enabled or not c_htf_bull) and
                            in_window(c_tod) and
                            daily_trades.get(date_key, 0) < MAX_TRADES_PER_DAY and
                            c_close < prev_close):
                        set_pending("short", c_close, date_key)

            elif direction == "short":
                if c_low < peak: peak = c_low
                hard_hit = HARD_SL_ENABLED and hard_sl_price and c_high >= hard_sl_price

                # ── LTF ST exit check (priority 2 — between Hard SL and main ST) ──
                ltf_hit = (LTF_ST_EXIT_ENABLED
                           and not np.isnan(c_ltf_st_val)
                           and (
                               (CLOSE_BASED_EXIT and c_close > c_ltf_st_val) or
                               (not CLOSE_BASED_EXIT and c_high >= c_ltf_st_val)
                           ))

                st_hit   = ((c_close > c_exit_st) or flipped_bull) if CLOSE_BASED_EXIT \
                           else (c_high >= c_exit_st or flipped_bull)

                if hard_hit or ltf_hit or st_hit:
                    if hard_hit:
                        reason  = f"Hard SL ({HARD_SL_PCT}%)"
                        exit_px = hard_sl_price
                    elif ltf_hit:
                        reason  = f"LTF ST Exit ({LTF_ST_MULTIPLIER}x)"
                        exit_px = c_ltf_st_val
                    elif (CLOSE_BASED_EXIT and c_close > c_exit_st) or \
                         (not CLOSE_BASED_EXIT and c_high >= c_exit_st):
                        reason  = "HTF ST Exit" if (EXIT_USE_HTF_ST and htf_enabled) else "ST Stop Loss"
                        exit_px = c_exit_st
                    else:
                        reason  = "ST Flip Bull"
                        exit_px = c_exit_st

                    record("short", entry_time, entry_price, c_time, round(exit_px, 2),
                           peak, c_exit_st, reason, hard_sl_price)
                    in_trade = False

                    # Attempt immediate reverse-to-long trigger (set pending, not open)
                    if (not hard_hit and not ltf_hit and c_bull and
                            (not htf_enabled or c_htf_bull) and
                            in_window(c_tod) and
                            daily_trades.get(date_key, 0) < MAX_TRADES_PER_DAY and
                            c_close > prev_close):
                        set_pending("long", c_close, date_key)

            continue  # don't look for new entries while in a trade

        # ── Check pending confirmation trigger ────────────────────────────────
        # A trigger is pending from a previous candle. Check whether this candle
        # confirms the entry by closing beyond the threshold.
        if pending_direction is not None:

            # ── Invalidation checks: cancel trigger if market moved against us ─
            # 1. ST direction flips against the pending trade (already cleared above on flip)
            # 2. HTF turns against us mid-wait
            htf_invalid = (
                (pending_direction == "long"  and htf_enabled and not c_htf_bull) or
                (pending_direction == "short" and htf_enabled and c_htf_bull)
            )
            # 3. ST direction is now against pending trade
            st_invalid = (
                (pending_direction == "long"  and not c_bull) or
                (pending_direction == "short" and c_bull)
            )
            # 4. Trade window closed
            window_invalid = not in_window(c_tod)
            # 5. Daily cap reached (use the date when trigger was set)
            cap_invalid = (MAX_TRADES_PER_DAY_ENABLED and
                           daily_trades.get(date_key, 0) >= MAX_TRADES_PER_DAY)

            if htf_invalid or st_invalid or window_invalid or cap_invalid:
                clear_pending()
                # Fall through to fresh entry-scan below

            else:
                # ── Confirmation check ────────────────────────────────────────
                # Long  : this candle closes > trigger_close + ENTRY_CONFIRM_POINTS
                # Short : this candle closes < trigger_close - ENTRY_CONFIRM_POINTS
                long_confirmed  = (pending_direction == "long"  and
                                   c_close > pending_trigger_close + ENTRY_CONFIRM_POINTS)
                short_confirmed = (pending_direction == "short" and
                                   c_close < pending_trigger_close - ENTRY_CONFIRM_POINTS)

                if long_confirmed or short_confirmed:
                    # Enter at this candle's close — confirmation achieved
                    open_trade(pending_direction, c_close, c_time, c_high, c_low, date_key)
                    clear_pending()

                # Not yet confirmed and not invalidated — keep waiting
                continue

        # ── Fresh entry scan (no trade open, no pending trigger) ──────────────
        if not in_window(c_tod):                                                    continue
        if (i - flip_candle) < CONFIRM_CANDLES:                                    continue
        if MIN_GAP_ENABLED and abs(c_close - c_st) / c_close * 100 < MIN_GAP_PCT:  continue
        if (VOLUME_FILTER_ENABLED and not np.isnan(vol_avg[i])
                and vol_avg[i] > 0
                and volumes[i] < VOLUME_MULTIPLIER * vol_avg[i]):                   continue
        if (MAX_TRADES_PER_DAY_ENABLED and
                daily_trades.get(date_key, 0) >= MAX_TRADES_PER_DAY):              continue

        if c_bull and (not htf_enabled or c_htf_bull):
            # All conditions met for a potential long.
            # Instead of entering immediately, record the trigger candle's close.
            # The actual entry fires when a subsequent candle closes
            # more than ENTRY_CONFIRM_POINTS above this close.
            if c_close > prev_close:
                set_pending("long", c_close, date_key)

        elif not c_bull and (not htf_enabled or not c_htf_bull):
            # All conditions met for a potential short.
            # Entry fires when a subsequent candle closes
            # more than ENTRY_CONFIRM_POINTS below this close.
            if c_close < prev_close:
                set_pending("short", c_close, date_key)

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
    if not trades: return {}
    df_t = pd.DataFrame(trades)

    total_opt_pnl  = df_t["Opt_PnL_Total"].sum()
    win_opt_pnl    = df_t.loc[df_t["Opt_PnL_Total"] > 0, "Opt_PnL_Total"].sum()
    loss_opt_pnl   = df_t.loc[df_t["Opt_PnL_Total"] <= 0, "Opt_PnL_Total"].sum()
    opt_wins       = (df_t["Opt_PnL_Total"] > 0).sum()
    opt_losses     = (df_t["Opt_PnL_Total"] <= 0).sum()
    best_opt_trade = df_t.loc[df_t["Opt_PnL_Total"].idxmax()]
    worst_opt_trade= df_t.loc[df_t["Opt_PnL_Total"].idxmin()]

    total_delta_c  = df_t["Opt_DeltaEffect"].sum() * OPTION_LOT_SIZE
    total_gamma_c  = df_t["Opt_GammaPnL"].sum()
    total_theta    = df_t["Opt_ThetaEffect"].sum() * OPTION_LOT_SIZE
    total_vega_c   = df_t["Opt_VegaEffect"].sum() * OPTION_LOT_SIZE
    total_slippage = df_t["Opt_Slippage"].sum()

    be_series  = df_t["Opt_BreakevenDS"].dropna() if "Opt_BreakevenDS" in df_t.columns else pd.Series(dtype=float)
    be_positive = be_series[be_series > 0]
    avg_be      = be_positive.mean() if not be_positive.empty else None

    return {
        "df_t"               : df_t,
        "total_opt_pnl"      : round(total_opt_pnl, 2),
        "win_opt_pnl"        : round(win_opt_pnl, 2),
        "loss_opt_pnl"       : round(loss_opt_pnl, 2),
        "opt_wins"           : int(opt_wins),
        "opt_losses"         : int(opt_losses),
        "opt_win_rate"       : round(opt_wins / len(df_t) * 100, 1),
        "best_opt_trade"     : best_opt_trade,
        "worst_opt_trade"    : worst_opt_trade,
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
    lines = ["", SEP, "  OPTION P/L ANALYTICS METRICS FRAMEWORK  (v4.3 — 5pt Confirm Entry)", SEP]

    def safe_div(a, b): return a / b if b != 0 else 0.0
    def format_money(v): return f"₹{v:,.2f}" if not pd.isna(v) else "N/A"

    returns = df["Opt_PnL_Total"]
    wins    = df[returns > 0]
    losses  = df[returns < 0]

    net_pnl      = returns.sum()
    avg_trade    = returns.mean()
    med_trade    = returns.median()
    win_rate     = safe_div(len(wins), len(returns))
    avg_win      = wins["Opt_PnL_Total"].mean()  if len(wins)   > 0 else 0
    avg_loss     = losses["Opt_PnL_Total"].mean() if len(losses) > 0 else 0
    rr_ratio     = abs(safe_div(avg_win, avg_loss))
    expectancy   = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    gross_profit = wins["Opt_PnL_Total"].sum()
    gross_loss   = abs(losses["Opt_PnL_Total"].sum())
    pf           = safe_div(gross_profit, gross_loss)
    total_slip   = df["Opt_Slippage"].sum() if "Opt_Slippage" in df.columns else 0

    lines += [
        "  SECTION 3 — BASIC PERFORMANCE METRICS", DASH,
        f"  Total Net P/L        : {format_money(net_pnl)}",
        f"  Total Slippage Cost  : {format_money(total_slip)}  (₹{SLIPPAGE_PER_CONTRACT}/trade)",
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

    if "Entry Price" in df.columns:
        capital_used = df["Entry Price"] * OPTION_LOT_SIZE
    else:
        capital_used = pd.Series([OPTION_LOT_SIZE * 22000] * len(df))

    trade_returns  = returns / capital_used
    sharpe_raw     = safe_div(trade_returns.mean(), trade_returns.std())

    trading_days   = df["Date"].nunique() if "Date" in df.columns else 1
    total_trades   = len(df)
    trades_per_day = safe_div(total_trades, max(trading_days, 1))
    trades_per_yr  = trades_per_day * 250
    sharpe_annual  = sharpe_raw * math.sqrt(max(trades_per_yr, 1))

    sortino_raw    = safe_div(trade_returns.mean(),
                              (trade_returns[trade_returns < 0].std()
                               if (trade_returns < 0).any() else 1))
    calmar         = safe_div(returns.mean() * 252, abs(max_drawdown))

    lines += [
        "  SECTION 4 — RISK METRICS", DASH,
        f"  Maximum Drawdown (MDD) : {format_money(max_drawdown)}",
        f"  Max Drawdown %         : {mdd_pct:.2f}%",
        f"  Ulcer Index            : {ulcer:.4f}",
        f"  Std Dev of Returns     : {std_dev:.2f}",
        f"  Downside Deviation     : {downside:.2f}",
        f"  Sharpe (raw, per trade): {sharpe_raw:.4f}",
        f"  Sharpe (annualised)    : {sharpe_annual:.4f}",
        f"  Sortino Ratio          : {sortino_raw:.4f}",
        f"  Calmar Ratio (Proxy)   : {calmar:.4f}",
        SEP
    ]

    lot = df["Opt_LotSize"].iloc[0] if "Opt_LotSize" in df.columns else OPTION_LOT_SIZE
    avg_delta_exp = (df["Opt_DeltaEntry"].abs() * lot).mean()
    gex           = (df["Opt_Gamma"] * lot).sum()
    theta_per_hr  = (df["Opt_Theta_daily"].iloc[0] / TRADING_HOURS_PER_DAY
                     if "Opt_Theta_daily" in df.columns and len(df) > 0
                     else OPTION_THETA / TRADING_HOURS_PER_DAY)
    total_gamma_contrib_abs = df["Opt_GammaPnL"].abs().sum()
    total_opt_change_abs    = (df["Opt_Change"].abs() * lot).sum()
    gamma_utilisation       = safe_div(total_gamma_contrib_abs, total_opt_change_abs)
    theta_eff_abs           = (df["Opt_ThetaEffect"].abs() * lot).sum()
    theta_eff_pct           = safe_div(theta_eff_abs, total_opt_change_abs) * 100
    delta_drift             = (df["Opt_DeltaExit"].abs() - df["Opt_DeltaEntry"].abs()).mean()

    lines += [
        "  SECTION 5 — GREEKS EXPOSURE METRICS", DASH,
        f"  Avg Delta Exposure   : {avg_delta_exp:.4f}",
        f"  Gamma Exposure (GEX) : {gex:.6f}",
        f"  Theta Exposure / Hr  : {theta_per_hr:.4f}",
        f"  Avg Delta Drift      : {delta_drift:.4f}",
        f"  Gamma Utilisation    : {gamma_utilisation:.4f}x",
        f"  Time Decay Contrib % : {theta_eff_pct:.2f}%",
        SEP
    ]

    be_all     = df["Opt_BreakevenDS"].dropna() if "Opt_BreakevenDS" in df.columns else pd.Series(dtype=float)
    be_pos     = be_all[be_all > 0]
    be_neg     = be_all[be_all < 0]
    be_avg_pos = be_pos.mean()   if not be_pos.empty else float("nan")
    be_avg_all = be_all.mean()   if not be_all.empty else float("nan")
    be_min     = be_all.min()    if not be_all.empty else float("nan")
    be_max     = be_all.max()    if not be_all.empty else float("nan")
    pct_needs_move = len(be_pos) / len(be_all) * 100 if len(be_all) > 0 else 0
    lines += [
        "  SECTION 6 — BREAK-EVEN ANALYSIS", DASH,
        f"  Avg Move Req (positive only) : {be_avg_pos:.2f} pts" if not np.isnan(be_avg_pos) else "  Avg Move Req (positive only) : N/A",
        f"  Avg Move Req (all trades)    : {be_avg_all:.2f} pts" if not np.isnan(be_avg_all) else "  Avg Move Req (all trades)    : N/A",
        f"  Min Break-even               : {be_min:.2f} pts"     if not np.isnan(be_min)     else "  Min Break-even               : N/A",
        f"  Max Break-even               : {be_max:.2f} pts"     if not np.isnan(be_max)     else "  Max Break-even               : N/A",
        f"  Trades needing move          : {pct_needs_move:.1f}%  ({len(be_pos)} of {len(be_all)})",
        f"  Trades already profitable    : {100-pct_needs_move:.1f}%  ({len(be_neg)} of {len(be_all)})",
        SEP
    ]

    skew       = returns.skew()
    kurt       = returns.kurtosis()
    left_tail  = returns.quantile(0.05)
    right_tail = returns.quantile(0.95)
    lines += [
        "  SECTION 7 — DISTRIBUTION METRICS", DASH,
        f"  Skewness             : {skew:.2f}",
        f"  Kurtosis             : {kurt:.2f}",
        f"  Left Tail Risk (5%)  : {format_money(left_tail)}",
        f"  Right Tail Gain (95%): {format_money(right_tail)}",
        SEP
    ]

    hold_hrs  = df["Holding Mins"].sum() / 60.0 if "Holding Mins" in df.columns else 0
    pnl_per_hr = safe_div(net_pnl, hold_hrs)
    theta_cost_total = (df["Opt_ThetaEffect"].abs().sum() * lot)
    theta_eff  = safe_div(net_pnl, theta_cost_total)
    move_eff   = safe_div(net_pnl, df["Opt_dS"].abs().sum())

    lines += [
        "  SECTION 8 — TIME EFFICIENCY METRICS", DASH,
        f"  Profit per Hour      : {format_money(pnl_per_hr)}",
        f"  Theta Efficiency     : {theta_eff:.2f}",
        f"  Move Efficiency      : {move_eff:.2f}",
        f"  Gamma Utilisation    : {gamma_utilisation:.4f}x",
        SEP
    ]

    worst_loss = abs(losses["Opt_PnL_Total"].min()) if len(losses) > 0 else 1
    max_loss   = MAX_LOSS_PER_TRADE if MAX_LOSS_PER_TRADE else worst_loss
    ror_series = returns / max_loss
    avg_ror    = ror_series.mean()

    if "Entry Price" in df.columns:
        margin_proxy = (df["Entry Price"] * OPTION_LOT_SIZE).max()
    else:
        margin_proxy = OPTION_LOT_SIZE * 22000
    max_margin = MAX_MARGIN_USED if MAX_MARGIN_USED else margin_proxy
    cap_eff    = safe_div(net_pnl, max_margin)

    lines += [
        "  SECTION 9 — CAPITAL EFFICIENCY", DASH,
        f"  Avg Return on Risk   : {avg_ror:.4f}",
        f"  Capital Efficiency   : {cap_eff:.4f}",
        f"  Max Margin Proxy     : {format_money(max_margin)}",
        SEP
    ]

    roll_sharpe = returns.rolling(20).apply(
        lambda x: safe_div(x.mean(), x.std()), raw=True).dropna()
    avg_roll_sharpe = roll_sharpe.mean() if not roll_sharpe.empty else 0
    roll_win  = (returns > 0).rolling(20).mean().dropna()
    avg_roll_win = roll_win.mean() * 100 if not roll_win.empty else 0

    x_eq = np.arange(len(equity_curve))
    if len(x_eq) > 1:
        slope, _ = np.polyfit(x_eq, equity_curve, 1)
        r2       = np.corrcoef(x_eq, equity_curve)[0, 1] ** 2
    else:
        slope, r2 = 0, 0

    is_win  = returns > 0
    streaks = is_win.ne(is_win.shift()).cumsum()
    cons_wins   = is_win.groupby(streaks).sum().max()
    cons_losses = (~is_win).groupby(streaks).sum().max()

    lines += [
        "  SECTION 10 — STRATEGY STABILITY METRICS", DASH,
        f"  Rolling 20-Trade Sharpe: {avg_roll_sharpe:.2f}",
        f"  Rolling Win Rate       : {avg_roll_win:.2f}%",
        f"  Equity Curve Slope     : {slope:.2f}",
        f"  Equity Curve R²        : {r2:.4f}",
        f"  Consecutive Wins       : {cons_wins}",
        f"  Consecutive Losses     : {cons_losses}",
        SEP
    ]

    def sim_pnl(ds_mult=1.0, iv_add_pts=0.0, hr_add=0.0):
        total = 0.0
        for _, row in df.iterrows():
            orig_dS  = row["Opt_dS"]
            new_dS   = orig_dS * ds_mult
            d_entry  = row["Opt_DeltaEntry"]
            g        = row["Opt_Gamma"]
            theta_d  = row["Opt_Theta_daily"] if "Opt_Theta_daily" in row.index else OPTION_THETA
            vega_v   = row["Opt_Vega"] if "Opt_Vega" in row.index else OPTION_VEGA
            dt_hrs   = max(row["Holding Mins"] / 60.0 + hr_add, 0)
            theta_e  = (theta_d / TRADING_HOURS_PER_DAY) * dt_hrs
            vix_chg  = row["Opt_VIX_Chg"] if "Opt_VIX_Chg" in row.index else 0.0
            iv_dec   = (vix_chg + iv_add_pts) / 100.0
            iv_dec   = max(-0.03, min(0.03, iv_dec))
            vega_e   = vega_v * iv_dec
            if USE_DYNAMIC_DELTA:
                steps     = 10
                step_size = new_dS / steps
                cur_d     = d_entry
                d_eff     = 0.0
                for _ in range(steps):
                    d_eff += cur_d * step_size
                    cur_d  = cur_d + g * step_size
                g_eff = 0.0
            else:
                d_eff = d_entry * new_dS
                g_eff = 0.5 * g * new_dS ** 2
            opt_ch = d_eff + g_eff + theta_e + vega_e
            sign   = 1 if OPTION_DIRECTION == "long" else -1
            pnl_t  = opt_ch * sign * OPTION_LOT_SIZE - SLIPPAGE_PER_CONTRACT
            total += pnl_t
        return total

    lines += [
        "  SECTION 11 — SENSITIVITY ANALYSIS", DASH,
        f"  Base Strategy Net P/L : {format_money(net_pnl)}"
    ]
    try:
        scenarios = [
            ("ΔS +10%",              dict(ds_mult=1.10)),
            ("ΔS -10%",              dict(ds_mult=0.90)),
            ("ΔS +20%",              dict(ds_mult=1.20)),
            ("ΔS -20%",              dict(ds_mult=0.80)),
            ("IV +5% (→+3% cap)",    dict(iv_add_pts=5.0)),
            ("IV -5% (→-3% cap)",    dict(iv_add_pts=-5.0)),
            ("Hold +1h",             dict(hr_add=1.0)),
            ("Hold -1h",             dict(hr_add=-1.0)),
        ]
        for label, kwargs in scenarios:
            sim = sim_pnl(**kwargs)
            lines.append(f"  P/L if {label:<28}: {format_money(sim)}  (Δ: {format_money(sim - net_pnl)})")
    except Exception as e:
        lines.append(f"  Sensitivity Simulation Error: {e}")

    lines += [SEP, "  SECTION 12 — OUTLIER / TAIL DEPENDENCY", DASH]
    try:
        sorted_returns = returns.sort_values(ascending=False)
        n_remove       = min(5, len(sorted_returns) - 1)
        trimmed        = sorted_returns.iloc[n_remove:]
        trimmed_total  = trimmed.sum()
        drop_pct       = (1 - safe_div(trimmed_total, net_pnl)) * 100 if net_pnl != 0 else 0
        tail_dependent = drop_pct > 30
        lines += [
            f"  Full P/L              : {format_money(net_pnl)}",
            f"  P/L (top {n_remove} removed)  : {format_money(trimmed_total)}",
            f"  Drop %                : {drop_pct:.1f}%",
            f"  Tail Dependent?       : {'⚠ YES — strategy relies on outliers' if tail_dependent else '✅ NO — robust distribution'}",
        ]
    except Exception as e:
        lines.append(f"  Outlier Check Error: {e}")

    lines.append(SEP + "\n")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDERS
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
        name="ST Bullish", mode="lines", line=dict(color="#00d4aa", width=2.5)))
    fig.add_trace(go.Scatter(
        x=df_c.index, y=df_c["ST_val"].where(~df_c["ST_bull"]),
        name="ST Bearish", mode="lines", line=dict(color="#ff4757", width=2.5)))
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
                        subplot_titles=("Option P&L per Trade — Net after Slippage (₹)",
                                        "Cumulative Option P&L (₹)"),
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
        x=["Delta", "Gamma", "Theta", "Vega"],
        y=[delta_tot, gamma_tot, theta_tot, vega_tot],
        marker_color=["#00d4aa","#3d9bff","#ff4757","#ffd700"],
        text=[f"₹{v:,.0f}" for v in [delta_tot, gamma_tot, theta_tot, vega_tot]],
        textposition="outside"))
    fig.update_layout(template="plotly_dark", height=380,
                      title="Greek P&L Attribution (₹, all trades)",
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
                      title="Theta Cost vs Delta Gain per Trade (₹)",
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
                      title="Break-even ΔS Distribution",
                      xaxis_title="ΔS (underlying points)", yaxis_title="Count",
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
                      title="Daily Option P&L — Net after Slippage (₹)",
                      margin=dict(l=50,r=20,t=60,b=80),
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="'Courier New', monospace"),
                      xaxis_tickangle=-30)
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

def build_vix_chart(df_t):
    has_vix = df_t["Opt_VIX_Entry"].notna().any()
    if not has_vix:
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("India VIX at Entry/Exit",
                                        "Vega P&L Effect (₹)"),
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

def df_to_html_table(df, id_="", row_color_col=None, pnl_col=None):
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
                     "Best Trade %", "Worst Trade %", "Opt P&L (₹)", "Net Points",
                     "PnL_Total", "GrossPnL"):
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

    candle_json  = build_candle_chart(df)
    pnl_json     = build_pnl_chart(df_t)
    attr_json    = build_greek_attribution_chart(df_t)
    delta_json   = build_delta_drift_chart(df_t)
    td_json      = build_theta_vs_delta_chart(df_t)
    be_json      = build_breakeven_chart(df_t)
    eq_dd_json   = build_equity_drawdown_chart(df_t)
    daily_json   = build_daily_opt_pnl_chart(df_day) if df_day is not None and not df_day.empty else None
    vix_json     = build_vix_chart(df_t)

    tl_cols = ["Date","Direction","Entry Time","Entry Price","Exit Time","Exit Price",
               "Holding Time","Points Captured","P&L %","Exit Reason","Result",
               "Opt_DeltaEffect","Opt_GammaEffect","Opt_GammaPnL","Opt_ThetaEffect",
               "Opt_VegaEffect","Opt_Change","Opt_GrossPnL","Opt_Slippage",
               "Opt_PnL_Total","Opt_BreakevenDS"]
    tl_display = df_t[[c for c in tl_cols if c in df_t.columns]].copy()
    tl_display.columns = [c.replace("Opt_","") for c in tl_display.columns]
    tl_html  = df_to_html_table(tl_display, id_="trade-log-tbl", row_color_col="Result")
    day_html = df_to_html_table(df_day, row_color_col="Day Result") if df_day is not None else ""
    ts_html  = df_to_html_table(df_ts, row_color_col="Verdict") if df_ts is not None else ""

    opt_cols = ["Date","Direction","Entry Time","Entry Price","Exit Time","Exit Price",
                "Opt_DeltaEntry","Opt_DeltaExit","Opt_Gamma","Opt_Theta_daily","Opt_Vega",
                "Opt_VIX_Entry","Opt_VIX_Exit","Opt_VIX_Chg",
                "Opt_DeltaEffect","Opt_GammaEffect","Opt_GammaPnL",
                "Opt_ThetaEffect","Opt_VegaEffect",
                "Opt_Change","Opt_PnL_PerLot","Opt_GrossPnL","Opt_Slippage",
                "Opt_PnL_Total","Opt_BreakevenDS"]
    opt_tbl = df_t[[c for c in opt_cols if c in df_t.columns]].copy()
    opt_tbl.columns = [c.replace("Opt_","") for c in opt_tbl.columns]
    opt_html = df_to_html_table(opt_tbl, pnl_col="PnL_Total")

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
    {card("Confirm Filter", f"+/- {ENTRY_CONFIRM_POINTS} pts", "5-pt close confirmation")}
    """

    opt_total_sign = "pos-card" if opt_sum["total_opt_pnl"] >= 0 else "neg-card"
    opt_cards = f"""
    {card("Total Option P&L", f"₹{opt_sum['total_opt_pnl']:,.2f}", f"{opt_sum['opt_wins']}W / {opt_sum['opt_losses']}L", opt_total_sign)}
    {card("Option Win Rate", f"{opt_sum['opt_win_rate']}%")}
    {card("Total Profit", f"₹{opt_sum['win_opt_pnl']:,.2f}", "from winning option trades", "pos-card")}
    {card("Total Loss", f"₹{opt_sum['loss_opt_pnl']:,.2f}", "from losing option trades", "neg-card")}
    {card("Delta Contribution", f"₹{opt_sum['total_delta_contrib']:,.2f}")}
    {card("Gamma Contribution", f"₹{opt_sum['total_gamma_contrib']:,.2f}")}
    {card("Theta Cost", f"₹{opt_sum['total_theta_cost']:,.2f}", "", "neg-card")}
    {card("Vega Contribution", f"₹{opt_sum['total_vega_contrib']:,.2f}")}
    {card("Slippage Cost", f"₹{opt_sum['total_slippage']:,.2f}", f"₹{SLIPPAGE_PER_CONTRACT}/trade", "neg-card")}
    {card("Avg Break-even ΔS", f"{opt_sum['avg_breakeven_dS']} pts" if opt_sum['avg_breakeven_dS'] else "—")}
    """

    confirm_info = f"""
    <div class="formula-box">
      <p class="formula-title">v4.3 — 5-Point Confirmation Entry Logic</p>
      <code>TRIGGER CANDLE  : All existing conditions satisfied (ST bull/bear, HTF, gap, window, etc.)</code>
      <code>                  → trigger_close = candle.close   (NO trade entered yet)</code>
      <br>
      <code>LONG  CONFIRM   : next candle(s) close  &gt;  trigger_close + {ENTRY_CONFIRM_POINTS} pts  → Enter Long  at that close</code>
      <code>SHORT CONFIRM   : next candle(s) close  &lt;  trigger_close - {ENTRY_CONFIRM_POINTS} pts  → Enter Short at that close</code>
      <br>
      <code>CANCEL TRIGGER  : ST flips  |  HTF turns against  |  trade window closes  |  daily cap hit  |  EOD</code>
      <br>
      <code>CONFIG          : ENTRY_CONFIRM_POINTS = {ENTRY_CONFIRM_POINTS}   (change this value in CONFIG section)</code>
    </div>"""

    def plot_div(fig_json, div_id):
        if fig_json is None:
            return f'<div class="chart-placeholder">No data available</div>'
        return f'<div id="{div_id}" class="chart-container"></div>'

    def plot_script(fig_json, div_id):
        if fig_json is None: return ""
        return f"Plotly.react('{div_id}', {json.dumps(fig_json['data'])}, {json.dumps(fig_json['layout'])}, {{responsive:true}});\n"

    all_scripts = (
        plot_script(candle_json,  "chart-candle") +
        plot_script(pnl_json,     "chart-pnl") +
        plot_script(attr_json,    "chart-attr") +
        plot_script(delta_json,   "chart-delta") +
        plot_script(td_json,      "chart-td") +
        plot_script(be_json,      "chart-be") +
        plot_script(eq_dd_json,   "chart-eqdd") +
        (plot_script(daily_json,  "chart-daily") if daily_json else "") +
        (plot_script(vix_json,    "chart-vix")   if vix_json   else "")
    )

    vix_section = f"""
    <section id="sec-vix">
      <h2>📉 India VIX &amp; Vega Analysis</h2>
      {plot_div(vix_json, "chart-vix")}
    </section>""" if vix_json else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{SYMBOL} — Supertrend + Options Report v4.4</title>
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
  body {{ background: var(--bg); color: var(--text); font-family: 'Rajdhani', sans-serif; font-size: 15px; line-height: 1.5; }}
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg2); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}
  header {{ background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); border-bottom: 1px solid var(--border2); padding: 28px 36px 20px; }}
  .header-top {{ display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px; }}
  .logo {{ font-family: 'Share Tech Mono', monospace; font-size: 11px; color: var(--text3); letter-spacing: 2px; text-transform: uppercase; }}
  h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: 1px; background: linear-gradient(90deg, var(--green), var(--gold)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .meta {{ font-family: 'Share Tech Mono', monospace; font-size: 11px; color: var(--text3); margin-top: 6px; }}
  nav {{ background: var(--bg3); border-bottom: 1px solid var(--border); padding: 0 36px; display: flex; gap: 0; overflow-x: auto; }}
  nav a {{ display: inline-block; padding: 12px 18px; color: var(--text2); text-decoration: none; font-weight: 600; font-size: 13px; letter-spacing: .5px; border-bottom: 2px solid transparent; white-space: nowrap; transition: all .2s; }}
  nav a:hover {{ color: var(--green); border-bottom-color: var(--green); }}
  main {{ padding: 28px 36px; max-width: 1600px; }}
  section {{ margin-bottom: 52px; scroll-margin-top: 60px; }}
  h2 {{ font-size: 1.25rem; font-weight: 700; color: var(--gold); margin-bottom: 18px; letter-spacing: .5px; padding-bottom: 8px; border-bottom: 1px solid var(--border2); }}
  h3 {{ font-size: 1rem; font-weight: 600; color: var(--text2); margin: 20px 0 10px; }}
  .card-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(185px, 1fr)); gap: 14px; }}
  .stat-card {{ background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; position: relative; overflow: hidden; }}
  .stat-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--green), transparent); }}
  .pos-card::before {{ background: linear-gradient(90deg, var(--green), transparent); }}
  .neg-card::before {{ background: linear-gradient(90deg, var(--red), transparent); }}
  .stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text3); margin-bottom: 6px; }}
  .stat-value {{ font-size: 1.5rem; font-weight: 700; font-family: 'Share Tech Mono', monospace; color: var(--text); }}
  .pos-card .stat-value {{ color: var(--green); }}
  .neg-card .stat-value {{ color: var(--red); }}
  .stat-sub {{ font-size: 11px; color: var(--text3); margin-top: 4px; }}
  .table-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 18px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  thead tr {{ background: var(--bg4); }}
  th {{ padding: 10px 12px; text-align: left; font-weight: 700; font-size: 11px; text-transform: uppercase; letter-spacing: .6px; color: var(--text2); border-bottom: 1px solid var(--border2); white-space: nowrap; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; font-family: 'Share Tech Mono', monospace; font-size: 12px; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,.025); }}
  .win-row td {{ background: var(--win-bg); }}
  .loss-row td {{ background: var(--loss-bg); }}
  .pos-val {{ color: var(--green) !important; font-weight: 700; }}
  .neg-val {{ color: var(--red) !important; font-weight: 700; }}
  .chart-container {{ border-radius: 8px; border: 1px solid var(--border); overflow: hidden; background: var(--bg2); margin-bottom: 18px; }}
  .chart-placeholder {{ background: var(--bg3); border: 1px dashed var(--border2); border-radius: 8px; padding: 40px; text-align: center; color: var(--text3); font-style: italic; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .formula-box {{ background: var(--bg4); border: 1px solid var(--border2); border-left: 3px solid var(--gold); border-radius: 6px; padding: 16px 20px; margin-top: 16px; }}
  .formula-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--gold); margin-bottom: 10px; }}
  code {{ font-family: 'Share Tech Mono', monospace; font-size: 13px; color: var(--green); display: block; margin: 4px 0; }}
  .disclaimer {{ background: var(--bg3); border: 1px solid var(--border2); border-left: 3px solid var(--red); border-radius: 6px; padding: 14px 18px; margin-bottom: 28px; font-size: 12px; color: var(--text3); }}
  footer {{ padding: 24px 36px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text3); text-align: center; font-family: 'Share Tech Mono', monospace; letter-spacing: 1px; }}
  @media (max-width: 900px) {{ main {{ padding: 16px 12px; }} .chart-row {{ grid-template-columns: 1fr; }} h1 {{ font-size: 1.4rem; }} }}
</style>
</head>
<body>
<header>
  <div class="logo">{SYMBOL} · Supertrend v4.4 · LTF ST Exit + 5-Point Confirmation Entry</div>
  <div class="header-top">
    <div>
      <h1>⚡ {SYMBOL} — Options Backtest Report v4.4</h1>
      <div class="meta">
        ST({ST_PERIOD},{ST_MULTIPLIER}) &nbsp;|&nbsp; HTF: {"ON" if htf_enabled else "OFF"} &nbsp;|&nbsp;
        LTF ST Exit: {"ON — ST("+str(LTF_ST_PERIOD)+","+str(LTF_ST_MULTIPLIER)+")" if LTF_ST_EXIT_ENABLED else "OFF"} &nbsp;|&nbsp;
        Confirm: ±{ENTRY_CONFIRM_POINTS} pts &nbsp;|&nbsp;
        Lot: {OPTION_LOT_SIZE} &nbsp;|&nbsp; Δ={OPTION_DELTA} Γ={OPTION_GAMMA} Θ={OPTION_THETA}/d ν={OPTION_VEGA} &nbsp;|&nbsp;
        Mode: {OPTION_DIRECTION.upper()} &nbsp;|&nbsp; Dyn-Δ: {"ON" if USE_DYNAMIC_DELTA else "OFF"} &nbsp;|&nbsp;
        Slippage: ₹{SLIPPAGE_PER_CONTRACT}/trade
      </div>
    </div>
  </div>
</header>
<nav>
  <a href="#sec-overview">📊 Overview</a>
  <a href="#sec-chart">📈 Chart</a>
  <a href="#sec-options">🎯 Option P&amp;L</a>
  <a href="#sec-greeks">⚗️ Greeks</a>
  <a href="#sec-eqdd">📉 Equity/DD</a>
  <a href="#sec-trades">📋 Trade Log</a>
  <a href="#sec-daily">📅 Daily</a>
  <a href="#sec-timeslot">🕐 Time Slots</a>
  {"<a href='#sec-vix'>📈 VIX/Vega</a>" if vix_json else ""}
  <a href="#sec-model">📐 Model</a>
</nav>
<main>
  <div class="disclaimer">
    ⚠ <strong>DISCLAIMER:</strong> Greek-approximation (Taylor expansion). Greeks assumed locally constant per trade. Not financial advice.
  </div>
  <section id="sec-overview">
    <h2>📊 Strategy Overview</h2>
    <div class="card-grid">{strat_cards}</div>
    <h3>Entry Confirmation Logic</h3>
    {confirm_info}
    <h3>Option P&amp;L Summary</h3>
    <div class="card-grid">{opt_cards}</div>
  </section>
  <section id="sec-chart">
    <h2>📈 Price &amp; Supertrend Chart (Last {CHART_DAYS} Days)</h2>
    {plot_div(candle_json, "chart-candle")}
  </section>
  <section id="sec-options">
    <h2>🎯 Option P&amp;L Analysis</h2>
    {plot_div(pnl_json, "chart-pnl")}
    {"" if daily_json is None else plot_div(daily_json, "chart-daily")}
    <h3>Per-Trade Option Detail</h3>
    {opt_html}
  </section>
  <section id="sec-greeks">
    <h2>⚗️ Greeks Attribution</h2>
    <div class="chart-row">
      {plot_div(attr_json, "chart-attr")}
      {plot_div(delta_json, "chart-delta")}
    </div>
    <div class="chart-row">
      {plot_div(td_json, "chart-td")}
      {plot_div(be_json, "chart-be")}
    </div>
  </section>
  <section id="sec-eqdd">
    <h2>📉 Equity Curve &amp; Drawdown</h2>
    {plot_div(eq_dd_json, "chart-eqdd")}
  </section>
  {vix_section}
  <section id="sec-trades">
    <h2>📋 Complete Trade Log</h2>
    {tl_html}
  </section>
  <section id="sec-daily">
    <h2>📅 Daily P&amp;L Breakdown</h2>
    {day_html}
  </section>
  <section id="sec-timeslot">
    <h2>🕐 Time Slot P&amp;L Analysis</h2>
    {ts_html}
  </section>
  <section id="sec-model">
    <h2>📐 Model Parameters</h2>
    {confirm_info}
  </section>
</main>
<footer>
  {SYMBOL} Supertrend v4.4 — LTF ST Exit ST({LTF_ST_PERIOD},{LTF_ST_MULTIPLIER}) {"ON" if LTF_ST_EXIT_ENABLED else "OFF"} &nbsp;·&nbsp;
  ±{ENTRY_CONFIRM_POINTS} pts confirm &nbsp;|&nbsp;
  Δ={OPTION_DELTA} Γ={OPTION_GAMMA} Θ={OPTION_THETA}/d ν={OPTION_VEGA} Lot={OPTION_LOT_SIZE} &nbsp;|&nbsp;
  {OPTION_DIRECTION.upper()} · Slip:₹{SLIPPAGE_PER_CONTRACT}
</footer>
<script>
{all_scripts}
</script>
</body>
</html>"""

    fname = f"{SYMBOL}_options_report_v4_4.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅  HTML Report saved → {fname}")
    return fname

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSOLE PRINT
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(trades, htf_enabled):
    SEP  = "═" * 110
    DASH = "─" * 110
    if not trades:
        print("  ⚠  No trades."); return None, None, None

    df_t   = pd.DataFrame(trades)
    df_day = build_daily_summary(trades)
    df_ts  = build_time_slot_summary(trades)
    opt    = build_option_summary(trades)

    total = len(df_t); wins = (df_t["P&L %"] > 0).sum()
    print("\n" + SEP)
    print(f"  {SYMBOL}  ST({ST_PERIOD},{ST_MULTIPLIER})  |  Trades: {total}  |  "
          f"Win Rate: {wins/total*100:.1f}%  |  HTF: {'ON' if htf_enabled else 'OFF'}  |  "
          f"LTF ST: {'ON ('+str(LTF_ST_MULTIPLIER)+'x)' if LTF_ST_EXIT_ENABLED else 'OFF'}  |  "
          f"Confirm: ±{ENTRY_CONFIRM_POINTS} pts  |  Slippage: ₹{SLIPPAGE_PER_CONTRACT}/trade")
    print(SEP)

    cols = ["Date","Direction","Entry Time","Entry Price","Exit Time","Exit Price",
            "Holding Time","Points Captured","P&L %","Opt_GrossPnL","Opt_Slippage",
            "Opt_PnL_Total","Exit Reason","Result"]
    print(df_t[[c for c in cols if c in df_t.columns]].to_string(index=False))

    print("\n" + SEP); print("  PER-DAY P&L"); print(DASH)
    print(df_day.to_string(index=False))

    print("\n" + SEP); print("  TIME SLOT ANALYSIS"); print(DASH)
    print(df_ts.to_string(index=False))

    print("\n" + SEP); print("  OPTION P&L SUMMARY  (v4.4)"); print(DASH)
    print(f"  Total Option P&L (net)   : ₹{opt['total_opt_pnl']:>12,.2f}")
    print(f"  Total Slippage Cost      : ₹{opt['total_slippage']:>12,.2f}")
    print(f"  Delta Contribution       : ₹{opt['total_delta_contrib']:>12,.2f}")
    print(f"  Gamma Contribution       : ₹{opt['total_gamma_contrib']:>12,.2f}")
    print(f"  Theta Cost               : ₹{opt['total_theta_cost']:>12,.2f}")
    print(f"  Vega Contribution        : ₹{opt['total_vega_contrib']:>12,.2f}")
    print(DASH)
    print(f"  Avg Break-even ΔS        : {opt['avg_breakeven_dS']} pts")
    print(SEP + "\n")

    adv = build_advanced_option_metrics(trades)
    if adv:
        print(adv)

    return df_t, df_day, df_ts

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  SUPERTREND + OPTIONS BACKTEST v4.4")
    print(f"  NEW: LTF ST Exit — ST({LTF_ST_PERIOD},{LTF_ST_MULTIPLIER})  {'ENABLED' if LTF_ST_EXIT_ENABLED else 'DISABLED'}")
    print(f"  5-Point Confirmation Entry (±{ENTRY_CONFIRM_POINTS} pts)")
    print("  Greeks: Δ={} Γ={} Θ={}/d ν={} | Lot={} | Slip=₹{}/trade".format(
          OPTION_DELTA, OPTION_GAMMA, OPTION_THETA, OPTION_VEGA,
          OPTION_LOT_SIZE, SLIPPAGE_PER_CONTRACT))
    print("═"*60)

    try:
        df, htf_enabled, vix_series = fetch_data()
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(e); return

    trades = run_backtest(df, htf_enabled, vix_series)
    df_t, df_day, df_ts = print_results(trades, htf_enabled)

    print("  Building HTML Report …")
    fname = export_html(df, trades, df_day, df_ts, htf_enabled, vix_series)

    print("\n  ✅  Done. Report → " + (fname or "N/A"))
    print(f"\n  ENTRY CONFIRMATION LOGIC:")
    print(f"  • Trigger candle  : all conditions met → trigger_close = candle.close")
    print(f"  • Long  entry     : next close  > trigger_close + {ENTRY_CONFIRM_POINTS} pts")
    print(f"  • Short entry     : next close  < trigger_close - {ENTRY_CONFIRM_POINTS} pts")
    print(f"  • Cancel trigger  : ST flip | HTF invalid | window closed | EOD | daily cap\n")

if __name__ == "__main__":
    main()