"""
╔══════════════════════════════════════════════════════════════════╗
║   SPEED DEMON SCALPER  v4  — "VWAP INSTITUTIONAL RETEST"       ║
║   + OPTIONS P&L ENGINE  (ported from Supertrend v4.5)           ║
╠══════════════════════════════════════════════════════════════════╣
║  WHAT'S NEW vs v3                                                ║
║  ✦ VWAP Indicator (anchored per-day, session-reset)             ║
║  ✦ VWAP Institutional Bias Filter (long > VWAP, short < VWAP)  ║
║  ✦ VWAP Distance Gate: abs(close-vwap)/vwap < 0.8%             ║
║    → Prevents chasing expanded gamma on options entries          ║
║                                                                  ║
║  CARRY-OVER FROM v3                                              ║
║  ✦ ADX Regime Filter                                            ║
║  ✦ Soft EOD — Trail stays alive until 15:25                    ║
║  ✦ Tighter Short Entry: 3-bar confirmation required             ║
║  ✦ Session Gate: Midday & Euro disabled by default              ║
║  ✦ Lower BE threshold for longs (0.3% instead of 0.5%)         ║
║  ✦ Consecutive-loss daily circuit breaker                       ║
║  ✦ Full per-trade MAE/MFE tracking in CSV output                ║
║                                                                  ║
║  ENTRY MODEL (v4)                                                ║
║  HTF Trend → ADX Regime → EMA Structure →                       ║
║  VWAP Institutional Bias → EMA Retest Entry                     ║
║                                                                  ║
║  OPTIONS ENGINE (from Supertrend v4.5)                          ║
║  ✦ Delta / Gamma / Theta / Vega Greek attribution               ║
║  ✦ Dynamic delta (Gamma-adjusted path simulation)               ║
║  ✦ Break-even ΔS computation per trade                          ║
║  ✦ Full Option P&L summary & per-trade detail                   ║
║  ✦ Time-slot P&L breakdown                                      ║
║  ✦ Advanced metrics: Sharpe, Sortino, Calmar, R², etc.          ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python speed_demon_v4_vwap.py              ← yfinance (default)
    python speed_demon_v4_vwap.py 5m.csv       ← CSV file
"""

import pandas as pd
import numpy as np
import sys, os, math
from datetime import time
from scipy.optimize import brentq

# ══════════════════════════════════════════════════════════════════
#   CONFIGURATION — STRATEGY
# ══════════════════════════════════════════════════════════════════

DATA_MODE        = 'yfinance'
YFINANCE_SYMBOL  = '^NSEI'
YFINANCE_DAYS    = 60
YFINANCE_TZ      = 'Asia/Kolkata'
CSV_5MIN_PATH    = 'nifty_5m.csv'

# ── EMA Periods (5m) ────────────────────────────────────────────
EMA_FAST         = 9
EMA_SLOW         = 21
EMA_MACRO        = 50

# ── HTF (15m) EMA Periods — Long bias filter ────────────────────
HTF_EMA_FAST     = 9
HTF_EMA_SLOW     = 21
HTF_EMA_TREND    = 50

# ── ATR ─────────────────────────────────────────────────────────
ATR_PERIOD       = 14

# ── ADX Regime Filter ───────────────────────────────────────────
ADX_PERIOD       = 14
ADX_MIN          = 18

# ── VWAP Distance Gate ──────────────────────────────────────────
# Prevents chasing entries when price has already moved too far
# from VWAP (gamma already expanded = bad options entry)
VWAP_MAX_DIST_PCT = 0.0    # 0.8% max distance from VWAP

# ── Asymmetric SL ───────────────────────────────────────────────
LONG_ATR_SL_MULT    = 1.2
SHORT_ATR_SL_MULT   = 0.75

# ── Break-Even Trigger ──────────────────────────────────────────
LONG_BE_PCT         = 0.2
SHORT_BE_PCT        = 0.3

# ── Trailing Stop ───────────────────────────────────────────────
TRAIL_ATR_MULT      = 0.6

# ── EMA Cross Exit ──────────────────────────────────────────────
ENABLE_EMA_CROSS_EXIT = True

# ── Momentum (Slope Reversal) Exit ──────────────────────────────
ENABLE_SLOPE_EXIT     = True
SLOPE_EXIT_CANDLES    = 4

# ── Short Entry Confirmation ─────────────────────────────────────
SHORT_CONFIRM_BARS  = 3

# ── Anti-Chop Filters ───────────────────────────────────────────
SPREAD_PCT_MIN      = 0.04
SLOPE_CANDLES       = 6
SLOPE_MIN           = 0.0
PRICE_GAP_MIN       = 5.0
MIDDAY_SPREAD_MULT  = 2.0

# ── Retest tolerance ────────────────────────────────────────────
RETEST_ATR_MULT     = 0.25

# ── Max trades per day ───────────────────────────────────────────
MAX_TRADES_PER_DAY  = 3

# ── Consecutive-loss circuit breaker ────────────────────────────
MAX_CONSEC_LOSSES   = 2

# ── Directions ──────────────────────────────────────────────────
ENABLE_LONG         = True
ENABLE_SHORT        = True

# ── Session Gates ────────────────────────────────────────────────
ENABLE_MIDDAY       = False
ENABLE_EURO         = False

# ── Sessions ────────────────────────────────────────────────────
OBSERVE_START       = time(9,  15)
OBSERVE_END         = time(9,  30)
PRIME_START         = time(9,  30)
PRIME_END           = time(10, 30)
MIDDAY_START        = time(11, 30)
MIDDAY_END          = time(13, 30)
EURO_START          = time(14, 15)
EURO_END            = time(15, 0)
SQUAREOFF_START     = time(15, 0)

# ── Soft EOD settings ───────────────────────────────────────────
EOD_HARD_EXIT       = time(15, 25)
EOD_SOFT_EXIT       = time(15, 0)


# ══════════════════════════════════════════════════════════════════
#   CONFIGURATION — OPTIONS ENGINE
# ══════════════════════════════════════════════════════════════════

OPTION_LOT_SIZE          = 65
OPTION_DELTA             = 0.45
OPTION_GAMMA             = 0.0007
OPTION_THETA             = -10.53       # daily theta (negative for long options)
OPTION_VEGA              = 5.0          # per 1-vol-pt change
OPTION_DIRECTION         = "long"       # "long" = buying options; "short" = selling

USE_DYNAMIC_DELTA        = True         # use Gamma-adjusted delta path simulation
TRADING_HOURS_PER_DAY    = 6.25

SLIPPAGE_PER_CONTRACT    = 50.0
BROKERAGE_PER_TRADE      = 0.0

MAX_LOSS_PER_TRADE       = None
MAX_MARGIN_USED          = None

LOW_POINTS_THRESHOLD     = 20


# ══════════════════════════════════════════════════════════════════
#   INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high       = df['high'].astype(float)
    low        = df['low'].astype(float)
    close      = df['close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high  = df['high'].astype(float)
    low   = df['low'].astype(float)

    up_move   = high.diff()
    down_move = -(low.diff())

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_vals = compute_atr(df, period)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()

    plus_di  = 100 * plus_dm_s  / atr_vals.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr_vals.replace(0, np.nan)

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / dx_denom
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


# ── STEP 1: VWAP Indicator ──────────────────────────────────────
# Anchored per trading day (resets at 09:15 each day).
# This mirrors how prop desks and institutions use intraday VWAP.

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session-anchored VWAP — resets at 09:15 each trading day.
    Uses Typical Price (HLC/3) × Volume, cumulated per day.

    Why session-anchored:
      A rolling VWAP bleeds in prior-day data and becomes meaningless
      for intraday institutional reference levels. Resetting at open
      gives the true institutional cost basis for that session.
    """
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    date_grp = df['timestamp'].dt.date

    cumvol = df.groupby(date_grp)['volume'].cumsum()
    cumpv  = (tp * df['volume']).groupby(date_grp).cumsum()

    vwap = cumpv / cumvol.replace(0, np.nan)
    return vwap


def compute_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_fast']            = compute_ema(df['close'], EMA_FAST)
    df['ema_slow']            = compute_ema(df['close'], EMA_SLOW)
    df['ema_macro']           = compute_ema(df['close'], EMA_MACRO)
    df['atr']                 = compute_atr(df, ATR_PERIOD)
    df['adx']                 = compute_adx(df, ADX_PERIOD)
    df['ema_slow_slope']      = df['ema_slow'].diff(SLOPE_CANDLES)
    df['ema_slow_slope_exit'] = df['ema_slow'].diff(SLOPE_EXIT_CANDLES)
    # ── STEP 2: Add VWAP to indicator pipeline ──────────────────
    df['vwap']                = compute_vwap(df)
    bearish_cross = (df['ema_fast'] < df['ema_slow']).astype(int)
    df['consec_bearish_bars'] = bearish_cross.groupby(
        (bearish_cross != bearish_cross.shift()).cumsum()
    ).cumcount() + 1
    df['consec_bearish_bars'] = df['consec_bearish_bars'] * bearish_cross
    return df


def compute_htf_bias(df_5m: pd.DataFrame) -> pd.DataFrame:
    df_5m = df_5m.copy()
    df_5m = df_5m.set_index('timestamp')

    df_15m = df_5m['close'].resample('15min').ohlc().dropna()
    df_15m.columns = ['open', 'high', 'low', 'close']

    df_15m['htf_ema_fast']  = compute_ema(df_15m['close'], HTF_EMA_FAST)
    df_15m['htf_ema_slow']  = compute_ema(df_15m['close'], HTF_EMA_SLOW)
    df_15m['htf_ema_trend'] = compute_ema(df_15m['close'], HTF_EMA_TREND)

    df_15m['htf_long_bias'] = (
        (df_15m['htf_ema_fast'] > df_15m['htf_ema_slow']) &
        (df_15m['htf_ema_slow'] > df_15m['htf_ema_trend'])
    )
    df_15m['htf_short_bias'] = (
        (df_15m['htf_ema_fast'] < df_15m['htf_ema_slow']) &
        (df_15m['htf_ema_slow'] < df_15m['htf_ema_trend'])
    )

    htf_cols = df_15m[['htf_ema_fast', 'htf_ema_slow', 'htf_ema_trend',
                        'htf_long_bias', 'htf_short_bias']]
    htf_reindexed = htf_cols.reindex(df_5m.index, method='ffill')
    return df_5m.join(htf_reindexed).reset_index()


# ══════════════════════════════════════════════════════════════════
#   ANTI-CHOP + REGIME FILTERS
# ══════════════════════════════════════════════════════════════════

def get_session(t: time) -> str:
    if OBSERVE_START <= t < OBSERVE_END:
        return 'observe'
    if PRIME_START <= t < PRIME_END:
        return 'prime'
    if MIDDAY_START <= t < MIDDAY_END:
        return 'midday'
    if EURO_START <= t < EURO_END:
        return 'euro'
    if SQUAREOFF_START <= t:
        return 'squareoff'
    return 'outside'


def chop_filters_pass(close: float, ema_fast: float, ema_slow: float,
                      slope: float, adx_val: float, session: str) -> bool:
    if adx_val < ADX_MIN:
        return False
    spread_pct    = abs(ema_fast - ema_slow) / close * 100
    spread_thresh = SPREAD_PCT_MIN * (MIDDAY_SPREAD_MULT if session == 'midday' else 1.0)
    if spread_pct < spread_thresh:
        return False
    if abs(slope) <= SLOPE_MIN:
        return False
    if abs(close - ema_slow) < PRICE_GAP_MIN:
        return False
    return True


# ══════════════════════════════════════════════════════════════════
#   OPTIONS ENGINE  (ported from Supertrend v4.5)
# ══════════════════════════════════════════════════════════════════

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


def compute_option_pnl(entry_price_S, exit_price_S, holding_hours,
                       trade_direction="long"):
    abs_delta    = abs(OPTION_DELTA)
    gamma        = OPTION_GAMMA
    theta_daily  = OPTION_THETA
    vega         = OPTION_VEGA
    lot_size     = OPTION_LOT_SIZE
    opt_direction = OPTION_DIRECTION

    signed_delta = abs_delta if trade_direction == "long" else -abs_delta

    dS_total     = exit_price_S - entry_price_S
    theta_per_hr = theta_daily / TRADING_HOURS_PER_DAY
    theta_effect = theta_per_hr * holding_hours

    vega_effect  = 0.0   # no VIX data in Speed Demon flow

    steps     = 10
    step_size = dS_total / steps
    cur_delta = signed_delta
    delta_effect_dyn = 0.0

    for _ in range(steps):
        delta_effect_dyn += cur_delta * step_size
        if USE_DYNAMIC_DELTA:
            cur_delta = cur_delta + gamma * step_size

    gamma_effect_static = 0.5 * gamma * (dS_total ** 2)

    option_change = delta_effect_dyn + theta_effect + vega_effect
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
        "delta_effect"        : round(delta_effect_dyn, 4),
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
        "breakeven_dS"        : breakeven_dS,
        "trade_direction"     : trade_direction,
    }


# ══════════════════════════════════════════════════════════════════
#   DATA LOADERS
# ══════════════════════════════════════════════════════════════════

def _standardise(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(tz)
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert(tz)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[(df['timestamp'].dt.time >= time(9, 15)) &
            (df['timestamp'].dt.time <= time(15, 30))].reset_index(drop=True)
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
    # Ensure volume column exists (needed for VWAP)
    if 'volume' not in df.columns:
        df['volume'] = 0
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    return df


def fetch_yfinance(interval: str, label: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance"); sys.exit(1)
    import warnings
    print(f"  Fetching {YFINANCE_SYMBOL} {interval} (last {YFINANCE_DAYS} days) ...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = yf.download(tickers=YFINANCE_SYMBOL, period=f'{YFINANCE_DAYS}d',
                          interval=interval, progress=False, auto_adjust=True)
    if raw.empty:
        print(f"ERROR: No {interval} data for '{YFINANCE_SYMBOL}'."); sys.exit(1)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={'adj close': 'close', 'adj_close': 'close'})
    if 'volume' not in raw.columns:
        raw['volume'] = 0
    raw = raw[['open', 'high', 'low', 'close', 'volume']].dropna()
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize('UTC')
    raw.index = raw.index.tz_convert(YFINANCE_TZ)
    df = raw.reset_index()
    ts_col = next(c for c in df.columns if c.lower() in ('datetime', 'date', 'timestamp'))
    df = df.rename(columns={ts_col: 'timestamp'})
    df = _standardise(df, YFINANCE_TZ)
    print(f"    [{label}] Rows: {len(df)}  |  {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def load_csv(filepath: str, label: str = '') -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    ts_col = next((c for c in ['timestamp', 'datetime', 'date_time', 'time', 'date']
                   if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"No timestamp column. Got: {list(df.columns)}")
    df['timestamp'] = pd.to_datetime(df[ts_col], dayfirst=True)
    if ts_col != 'timestamp':
        df = df.drop(columns=[ts_col])
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = _standardise(df, 'Asia/Kolkata')
    print(f"  [{label}] Rows: {len(df)}  |  {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ══════════════════════════════════════════════════════════════════
#   BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame) -> tuple:
    closes           = df['close'].astype(float).values
    highs            = df['high'].astype(float).values
    lows             = df['low'].astype(float).values
    ema_fast         = df['ema_fast'].values
    ema_slow         = df['ema_slow'].values
    ema_macro        = df['ema_macro'].values
    atrs             = df['atr'].values
    adx_vals         = df['adx'].values
    slopes           = df['ema_slow_slope'].values
    slopes_exit      = df['ema_slow_slope_exit'].values
    consec_bearish   = df['consec_bearish_bars'].values
    htf_long_bias    = df['htf_long_bias'].values
    # ── STEP 3: Load VWAP array ─────────────────────────────────
    vwap_vals        = df['vwap'].values
    ts_list          = df['timestamp'].tolist()
    n                = len(df)

    trades           = []
    equity           = 0.0
    eq_curve         = []

    in_trade         = False
    direction        = None
    entry_price      = 0.0
    entry_time       = None
    stop_loss        = 0.0
    be_triggered     = False
    be_level         = 0.0
    trail_active     = False
    sl_dist_initial  = 0.0
    trade_max_favor  = 0.0
    trade_max_adverse= 0.0

    prev_date        = None
    daily_trades     = {}
    daily_consec_loss= {}

    def do_enter(dir_str, close_price, ts_now, atr_val):
        nonlocal in_trade, direction, entry_price, entry_time
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, trade_max_favor, trade_max_adverse

        sl_mult  = LONG_ATR_SL_MULT if dir_str == 'long' else SHORT_ATR_SL_MULT
        sl_dist  = atr_val * sl_mult

        in_trade          = True
        direction         = dir_str
        entry_price       = close_price
        entry_time        = ts_now
        be_triggered      = False
        trail_active      = False
        sl_dist_initial   = sl_dist
        trade_max_favor   = 0.0
        trade_max_adverse = 0.0

        if direction == 'long':
            stop_loss = entry_price - sl_dist
            be_level  = entry_price * (1 + LONG_BE_PCT / 100)
        else:
            stop_loss = entry_price + sl_dist
            be_level  = entry_price * (1 - SHORT_BE_PCT / 100)

    def do_exit(exit_price, ts_now, reason):
        nonlocal in_trade, direction, entry_price, entry_time
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, equity
        nonlocal trade_max_favor, trade_max_adverse

        pnl = round(
            (exit_price - entry_price) if direction == 'long'
            else (entry_price - exit_price), 2
        )
        equity += pnl

        hold_secs    = (ts_now - entry_time).total_seconds()
        holding_hours = hold_secs / 3600.0

        opt = compute_option_pnl(
            entry_price_S  = entry_price,
            exit_price_S   = exit_price,
            holding_hours  = holding_hours,
            trade_direction= direction,
        )

        date_str = str(ts_now.date())
        if pnl <= 0:
            daily_consec_loss[date_str] = daily_consec_loss.get(date_str, 0) + 1
        else:
            daily_consec_loss[date_str] = 0

        hold_mins = int(hold_secs // 60)

        trades.append({
            # ── Core trade info ──────────────────────────────────
            'direction':         direction,
            'entry_time':        entry_time,
            'exit_time':         ts_now,
            'entry_price':       entry_price,
            'exit_price':        exit_price,
            'sl_at_entry':       (entry_price - sl_dist_initial
                                  if direction == 'long'
                                  else entry_price + sl_dist_initial),
            'sl_at_exit':        stop_loss,
            'be_triggered':      be_triggered,
            'pnl':               pnl,
            'mfe_pts':           round(trade_max_favor,   2),
            'mae_pts':           round(trade_max_adverse, 2),
            'exit_reason':       reason,
            'holding_mins':      hold_mins,
            # ── Options Greeks data ──────────────────────────────
            'opt_dS':            opt['dS'],
            'opt_delta_entry':   opt['delta_entry'],
            'opt_delta_exit':    opt['delta_exit'],
            'opt_gamma':         opt['gamma'],
            'opt_theta_daily':   opt['theta_daily'],
            'opt_vega':          opt['vega'],
            'opt_delta_effect':  opt['delta_effect'],
            'opt_gamma_pnl':     opt['gamma_pnl_total'],
            'opt_theta_effect':  opt['theta_effect'],
            'opt_vega_effect':   opt['vega_effect'],
            'opt_change':        opt['option_change'],
            'opt_pnl_per_lot':   opt['pnl_per_lot'],
            'opt_gross_pnl':     opt['gross_pnl'],
            'opt_slippage':      opt['slippage'],
            'opt_total_pnl':     opt['total_pnl'],
            'opt_breakeven_dS':  opt['breakeven_dS'],
            'opt_lot_size':      opt['lot_size'],
        })

        in_trade          = False
        direction         = None
        entry_price       = 0.0
        entry_time        = None
        stop_loss         = 0.0
        be_triggered      = False
        be_level          = 0.0
        trail_active      = False
        sl_dist_initial   = 0.0
        trade_max_favor   = 0.0
        trade_max_adverse = 0.0

    for idx in range(n):
        close      = closes[idx]
        high_c     = highs[idx]
        low_c      = lows[idx]
        ef         = ema_fast[idx]
        es         = ema_slow[idx]
        em         = ema_macro[idx]
        atr        = float(atrs[idx])
        adx_v      = float(adx_vals[idx]) if not np.isnan(adx_vals[idx]) else 0.0
        slope      = float(slopes[idx])      if not np.isnan(slopes[idx])      else 0.0
        slope_exit = float(slopes_exit[idx]) if not np.isnan(slopes_exit[idx]) else 0.0
        cb         = int(consec_bearish[idx])
        # ── STEP 3 cont: Unpack VWAP for this bar ───────────────
        vwap       = float(vwap_vals[idx]) if not np.isnan(vwap_vals[idx]) else np.nan
        ts         = ts_list[idx]
        c_time     = ts.time()
        c_date     = ts.date()
        date_str   = str(c_date)
        session    = get_session(c_time)

        if c_date != prev_date:
            prev_date = c_date

        if np.isnan(ef) or np.isnan(es) or np.isnan(em) or np.isnan(atr):
            eq_curve.append({'timestamp': ts, 'equity': equity})
            continue

        htf_long = bool(htf_long_bias[idx]) if not pd.isna(htf_long_bias[idx]) else False

        # ── Manage open trade ────────────────────────────────────
        if in_trade:
            if direction == 'long':
                favor   = high_c - entry_price
                adverse = entry_price - low_c
            else:
                favor   = entry_price - low_c
                adverse = high_c - entry_price
            trade_max_favor   = max(trade_max_favor,   favor)
            trade_max_adverse = max(trade_max_adverse, adverse)

            if not be_triggered:
                if direction == 'long'  and close >= be_level:
                    stop_loss    = entry_price + 1.0
                    be_triggered = True
                    trail_active = True
                elif direction == 'short' and close <= be_level:
                    stop_loss    = entry_price - 1.0
                    be_triggered = True
                    trail_active = True

            if trail_active:
                trail_dist = atr * TRAIL_ATR_MULT
                if direction == 'long':
                    stop_loss = max(stop_loss, close - trail_dist)
                else:
                    stop_loss = min(stop_loss, close + trail_dist)

            exit_p = None
            exit_r = None

            if direction == 'long' and close <= stop_loss:
                exit_p = stop_loss
                exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'
            elif direction == 'short' and close >= stop_loss:
                exit_p = stop_loss
                exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'

            elif ENABLE_EMA_CROSS_EXIT and be_triggered:
                if direction == 'long'  and ef < es:
                    exit_p = close
                    exit_r = 'EMA_CROSS_EXIT'
                elif direction == 'short' and ef > es:
                    exit_p = close
                    exit_r = 'EMA_CROSS_EXIT'

            elif ENABLE_SLOPE_EXIT and be_triggered:
                if direction == 'long'  and slope_exit < -SLOPE_MIN:
                    exit_p = close
                    exit_r = 'SLOPE_REVERSAL_EXIT'
                elif direction == 'short' and slope_exit > SLOPE_MIN:
                    exit_p = close
                    exit_r = 'SLOPE_REVERSAL_EXIT'

            elif c_time >= EOD_SOFT_EXIT and not be_triggered:
                exit_p = close
                exit_r = 'SOFT_EOD_EXIT'

            elif c_time >= EOD_HARD_EXIT:
                exit_p = close
                exit_r = 'HARD_EOD_EXIT'

            if exit_p is not None:
                do_exit(exit_p, ts, exit_r)

        # ── Entry Logic ──────────────────────────────────────────
        if not in_trade:
            allowed_sessions = ['prime']
            if ENABLE_MIDDAY: allowed_sessions.append('midday')
            if ENABLE_EURO:   allowed_sessions.append('euro')
            if session not in allowed_sessions:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            if daily_trades.get(date_str, 0) >= MAX_TRADES_PER_DAY:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            if daily_consec_loss.get(date_str, 0) >= MAX_CONSEC_LOSSES:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            if not chop_filters_pass(close, ef, es, slope, adx_v, session):
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            # ── STEP 4: VWAP Distance Gate ───────────────────────
            # Skip entry if VWAP is invalid or price has already
            # moved >0.8% from VWAP (gamma expanded = bad entry)
            if np.isnan(vwap) or vwap == 0:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            dist_vwap = abs(close - vwap) / vwap
            if dist_vwap > VWAP_MAX_DIST_PCT / 100.0:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            retest_tol = atr * RETEST_ATR_MULT

            # ── STEP 5: Long Entry — requires price > VWAP ───────
            if ENABLE_LONG and htf_long:
                if (ef > es                         and
                    close > em                      and
                    close > vwap                    and   # ← VWAP bullish side
                    slope > 0                       and
                    abs(close - ef) <= retest_tol   and
                    close > es):
                    do_enter('long', close, ts, atr)
                    daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

            # ── STEP 6: Short Entry — requires price < VWAP ──────
            if not in_trade and ENABLE_SHORT:
                if (ef < es                         and
                    close < em                      and
                    close < vwap                    and   # ← VWAP bearish side
                    slope < 0                       and
                    abs(close - ef) <= retest_tol   and
                    close < es                      and
                    cb >= SHORT_CONFIRM_BARS):
                    do_enter('short', close, ts, atr)
                    daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

        eq_curve.append({'timestamp': ts, 'equity': equity})

    if in_trade:
        do_exit(closes[-1], ts_list[-1], 'END_OF_DATA')

    return pd.DataFrame(trades), pd.DataFrame(eq_curve)


# ══════════════════════════════════════════════════════════════════
#   METRICS & REPORTING
# ══════════════════════════════════════════════════════════════════

def compute_metrics(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return {'message': 'No trades found.'}
    pnl    = tdf['pnl']
    wins   = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    total  = len(tdf)
    gp, gl = wins.sum(), abs(losses.sum())
    cum    = pnl.cumsum()
    max_dd = (cum.cummax() - cum).max()
    avg_mfe = tdf['mfe_pts'].mean()
    avg_mae = tdf['mae_pts'].mean()
    return {
        'Total Trades':        total,
        'Winning Trades':      len(wins),
        'Losing Trades':       len(losses),
        'Win Rate (%)':        round(len(wins) / total * 100, 2),
        'Avg Profit (pts)':    round(wins.mean(),   2) if len(wins)   else 0,
        'Avg Loss (pts)':      round(losses.mean(), 2) if len(losses) else 0,
        'Largest Win (pts)':   round(pnl.max(), 2),
        'Largest Loss (pts)':  round(pnl.min(), 2),
        'Profit Factor':       round(gp / gl, 2) if gl > 0 else float('inf'),
        'Total P&L (pts)':     round(pnl.sum(), 2),
        'Max Drawdown (pts)':  round(max_dd, 2),
        'Avg MFE (pts)':       round(avg_mfe, 2),
        'Avg MAE (pts)':       round(avg_mae, 2),
    }


def fmt_ts(ts):
    return ts.strftime('%d-%m-%Y %H:%M') if hasattr(ts, 'strftime') else str(ts)


def safe_div(a, b):
    return a / b if b != 0 else 0.0


# ══════════════════════════════════════════════════════════════════
#   OPTION SUMMARY BUILDERS
# ══════════════════════════════════════════════════════════════════

def build_option_summary(tdf: pd.DataFrame) -> dict:
    total_opt_pnl   = tdf['opt_total_pnl'].sum()
    win_opt_pnl     = tdf.loc[tdf['opt_total_pnl'] > 0, 'opt_total_pnl'].sum()
    loss_opt_pnl    = tdf.loc[tdf['opt_total_pnl'] <= 0, 'opt_total_pnl'].sum()
    opt_wins        = (tdf['opt_total_pnl'] > 0).sum()
    opt_losses      = (tdf['opt_total_pnl'] <= 0).sum()

    total_delta_c   = tdf['opt_delta_effect'].sum() * OPTION_LOT_SIZE
    total_gamma_c   = tdf['opt_gamma_pnl'].sum()
    total_theta     = tdf['opt_theta_effect'].sum() * OPTION_LOT_SIZE
    total_vega_c    = tdf['opt_vega_effect'].sum() * OPTION_LOT_SIZE
    total_slippage  = tdf['opt_slippage'].sum()

    be_series  = tdf['opt_breakeven_dS'].dropna()
    be_positive = be_series[be_series > 0]
    avg_be      = be_positive.mean() if not be_positive.empty else None

    return {
        'total_opt_pnl'      : round(total_opt_pnl,  2),
        'win_opt_pnl'        : round(win_opt_pnl,    2),
        'loss_opt_pnl'       : round(loss_opt_pnl,   2),
        'opt_wins'           : int(opt_wins),
        'opt_losses'         : int(opt_losses),
        'opt_win_rate'       : round(opt_wins / len(tdf) * 100, 1),
        'total_theta_cost'   : round(total_theta,    2),
        'total_delta_contrib': round(total_delta_c,  2),
        'total_gamma_contrib': round(total_gamma_c,  2),
        'total_vega_contrib' : round(total_vega_c,   2),
        'total_slippage'     : round(total_slippage, 2),
        'avg_breakeven_dS'   : round(avg_be, 2) if avg_be else None,
        'total_trades'       : len(tdf),
    }


def build_time_slot_summary(tdf: pd.DataFrame) -> pd.DataFrame:
    SLOTS = [
        ("09:15", "09:30"), ("09:30", "10:00"), ("10:00", "10:30"), ("10:30", "11:00"),
        ("11:00", "11:30"), ("11:30", "12:00"), ("12:00", "12:30"), ("12:30", "13:00"),
        ("13:00", "13:30"), ("13:30", "14:00"), ("14:00", "14:30"), ("14:30", "15:00"),
        ("15:00", "15:15"),
    ]

    tdf2 = tdf.copy()
    tdf2['entry_tod'] = tdf2['entry_time'].apply(
        lambda t: t.strftime('%H:%M') if hasattr(t, 'strftime') else str(t)[11:16]
    )

    rows = []
    for s, e in SLOTS:
        grp = tdf2[(tdf2['entry_tod'] >= s) & (tdf2['entry_tod'] < e)]
        if grp.empty:
            rows.append({
                'Time Slot': f"{s}–{e}", 'Trades': 0, 'Winners': 0, 'Losers': 0,
                'Win Rate %': '—', 'Total P&L %': 0.0, 'Avg P&L %': 0.0,
                'Best %': 0.0, 'Worst %': 0.0, 'Opt P&L (₹)': 0.0, 'Verdict': '—'
            })
            continue
        total = len(grp)
        wins  = (grp['pnl'] > 0).sum()
        pnl_pct = grp['pnl'].sum() / grp['entry_price'].mean() * 100
        wr    = wins / total * 100
        rows.append({
            'Time Slot'   : f"{s}–{e}",
            'Trades'      : total,
            'Winners'     : wins,
            'Losers'      : total - wins,
            'Win Rate %'  : round(wr, 1),
            'Total P&L %' : round(pnl_pct, 4),
            'Avg P&L %'   : round(pnl_pct / total, 4),
            'Best %'      : round((grp['pnl'] / grp['entry_price'] * 100).max(), 4),
            'Worst %'     : round((grp['pnl'] / grp['entry_price'] * 100).min(), 4),
            'Opt P&L (₹)' : round(grp['opt_total_pnl'].sum(), 2),
            'Verdict'     : '🟢 Trade' if pnl_pct > 0 and wr >= 50 else '🔴 Avoid',
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
#   ADVANCED METRICS  (Section 3–12 from Supertrend v4.5)
# ══════════════════════════════════════════════════════════════════

def build_advanced_option_metrics(tdf: pd.DataFrame) -> str:
    SEP  = "=" * 69
    DASH = "-" * 69

    def fmt_money(v):
        return f"₹{v:,.2f}" if not pd.isna(v) else "N/A"

    returns = tdf['opt_total_pnl']
    wins    = tdf[returns > 0]
    losses  = tdf[returns < 0]

    net_pnl      = returns.sum()
    avg_trade    = returns.mean()
    med_trade    = returns.median()
    win_rate     = safe_div(len(wins), len(returns))
    avg_win      = wins['opt_total_pnl'].mean()  if len(wins)   > 0 else 0
    avg_loss     = losses['opt_total_pnl'].mean() if len(losses) > 0 else 0
    rr_ratio     = abs(safe_div(avg_win, avg_loss))
    expectancy   = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    gross_profit = wins['opt_total_pnl'].sum()
    gross_loss   = abs(losses['opt_total_pnl'].sum())
    pf           = safe_div(gross_profit, gross_loss)
    total_slip   = tdf['opt_slippage'].sum()

    lines = ["", SEP,
             "  OPTION P/L ANALYTICS METRICS FRAMEWORK  (v4 — VWAP Institutional Retest)",
             SEP,
             "  SECTION 3 — BASIC PERFORMANCE METRICS", DASH,
             f"  Total Net P/L        : {fmt_money(net_pnl)}",
             f"  Total Slippage Cost  : {fmt_money(total_slip)}  (₹{SLIPPAGE_PER_CONTRACT}/trade)",
             f"  Average Trade P/L    : {fmt_money(avg_trade)}",
             f"  Median Trade P/L     : {fmt_money(med_trade)}",
             f"  Win Rate             : {win_rate*100:.2f}%",
             f"  Average Winner       : {fmt_money(avg_win)}",
             f"  Average Loser        : {fmt_money(avg_loss)}",
             f"  Risk Reward Ratio    : {rr_ratio:.2f}",
             f"  Expectancy           : {fmt_money(expectancy)}",
             f"  Profit Factor        : {pf:.2f}",
             SEP]

    # Section 4 — Risk
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
    downside = losses['opt_total_pnl'].std() if len(losses) > 1 else 0

    capital_used = tdf['entry_price'] * OPTION_LOT_SIZE
    trade_returns  = returns / capital_used
    sharpe_raw     = safe_div(trade_returns.mean(), trade_returns.std())

    trading_days   = tdf['entry_time'].apply(lambda t: t.date()).nunique()
    total_trades   = len(tdf)
    trades_per_day = safe_div(total_trades, max(trading_days, 1))
    trades_per_yr  = trades_per_day * 250
    sharpe_annual  = sharpe_raw * math.sqrt(max(trades_per_yr, 1))

    sortino_raw    = safe_div(trade_returns.mean(),
                              trade_returns[trade_returns < 0].std()
                              if (trade_returns < 0).any() else 1)
    calmar         = safe_div(returns.mean() * 252, abs(max_drawdown))

    lines += [
        "  SECTION 4 — RISK METRICS", DASH,
        f"  Maximum Drawdown (MDD) : {fmt_money(max_drawdown)}",
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

    # Section 5 — Greeks
    lot = OPTION_LOT_SIZE
    avg_delta_exp           = (tdf['opt_delta_entry'].abs() * lot).mean()
    gex                     = (tdf['opt_gamma'] * lot).sum()
    theta_per_hr            = OPTION_THETA / TRADING_HOURS_PER_DAY
    total_gamma_contrib_abs = tdf['opt_gamma_pnl'].abs().sum()
    total_opt_change_abs    = (tdf['opt_change'].abs() * lot).sum()
    gamma_utilisation       = safe_div(total_gamma_contrib_abs, total_opt_change_abs)
    theta_eff_abs           = (tdf['opt_theta_effect'].abs() * lot).sum()
    theta_eff_pct           = safe_div(theta_eff_abs, total_opt_change_abs) * 100
    delta_drift             = (tdf['opt_delta_exit'].abs() - tdf['opt_delta_entry'].abs()).mean()

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

    # Section 6 — Break-even
    be_all  = tdf['opt_breakeven_dS'].dropna()
    be_pos  = be_all[be_all > 0]
    be_avg_pos = be_pos.mean()   if not be_pos.empty else float('nan')
    be_avg_all = be_all.mean()   if not be_all.empty else float('nan')
    be_min     = be_all.min()    if not be_all.empty else float('nan')
    be_max     = be_all.max()    if not be_all.empty else float('nan')
    pct_needs  = len(be_pos) / len(be_all) * 100 if len(be_all) > 0 else 0

    lines += [
        "  SECTION 6 — BREAK-EVEN ANALYSIS", DASH,
        f"  Avg Move Req (positive only) : {be_avg_pos:.2f} pts" if not np.isnan(be_avg_pos) else "  Avg Move Req (positive only) : N/A",
        f"  Avg Move Req (all trades)    : {be_avg_all:.2f} pts" if not np.isnan(be_avg_all) else "  Avg Move Req (all trades)    : N/A",
        f"  Min Break-even               : {be_min:.2f} pts"     if not np.isnan(be_min)     else "  Min Break-even               : N/A",
        f"  Max Break-even               : {be_max:.2f} pts"     if not np.isnan(be_max)     else "  Max Break-even               : N/A",
        f"  Trades needing move          : {pct_needs:.1f}%  ({len(be_pos)} of {len(be_all)})",
        f"  Trades already profitable    : {100-pct_needs:.1f}%  ({len(be_all)-len(be_pos)} of {len(be_all)})",
        SEP
    ]

    # Section 7 — Distribution
    skew       = returns.skew()
    kurt       = returns.kurtosis()
    left_tail  = returns.quantile(0.05)
    right_tail = returns.quantile(0.95)
    lines += [
        "  SECTION 7 — DISTRIBUTION METRICS", DASH,
        f"  Skewness             : {skew:.2f}",
        f"  Kurtosis             : {kurt:.2f}",
        f"  Left Tail Risk (5%)  : {fmt_money(left_tail)}",
        f"  Right Tail Gain (95%): {fmt_money(right_tail)}",
        SEP
    ]

    # Section 8 — Time efficiency
    hold_hrs   = tdf['holding_mins'].sum() / 60.0
    pnl_per_hr = safe_div(net_pnl, hold_hrs)
    theta_cost_total = tdf['opt_theta_effect'].abs().sum() * lot
    theta_eff  = safe_div(net_pnl, theta_cost_total)
    move_eff   = safe_div(net_pnl, tdf['opt_dS'].abs().sum())

    lines += [
        "  SECTION 8 — TIME EFFICIENCY METRICS", DASH,
        f"  Profit per Hour      : {fmt_money(pnl_per_hr)}",
        f"  Theta Efficiency     : {theta_eff:.2f}",
        f"  Move Efficiency      : {move_eff:.2f}",
        f"  Gamma Utilisation    : {gamma_utilisation:.4f}x",
        SEP
    ]

    # Section 9 — Capital efficiency
    worst_loss = abs(losses['opt_total_pnl'].min()) if len(losses) > 0 else 1
    max_loss   = MAX_LOSS_PER_TRADE if MAX_LOSS_PER_TRADE else worst_loss
    ror_series = returns / max_loss
    avg_ror    = ror_series.mean()

    margin_proxy = (tdf['entry_price'] * OPTION_LOT_SIZE).max()
    max_margin   = MAX_MARGIN_USED if MAX_MARGIN_USED else margin_proxy
    cap_eff      = safe_div(net_pnl, max_margin)

    lines += [
        "  SECTION 9 — CAPITAL EFFICIENCY", DASH,
        f"  Avg Return on Risk   : {avg_ror:.4f}",
        f"  Capital Efficiency   : {cap_eff:.4f}",
        f"  Max Margin Proxy     : {fmt_money(max_margin)}",
        SEP
    ]

    # Section 10 — Stability
    roll_sharpe = returns.rolling(20).apply(
        lambda x: safe_div(x.mean(), x.std()), raw=True).dropna()
    avg_roll_sharpe = roll_sharpe.mean() if not roll_sharpe.empty else 0
    roll_win        = (returns > 0).rolling(20).mean().dropna()
    avg_roll_win    = roll_win.mean() * 100 if not roll_win.empty else 0

    x_eq = np.arange(len(equity_curve))
    if len(x_eq) > 1:
        slope_eq, _ = np.polyfit(x_eq, equity_curve, 1)
        r2          = np.corrcoef(x_eq, equity_curve)[0, 1] ** 2
    else:
        slope_eq, r2 = 0, 0

    is_win  = returns > 0
    streaks = is_win.ne(is_win.shift()).cumsum()
    cons_wins   = is_win.groupby(streaks).sum().max()
    cons_losses = (~is_win).groupby(streaks).sum().max()

    lines += [
        "  SECTION 10 — STRATEGY STABILITY METRICS", DASH,
        f"  Rolling 20-Trade Sharpe: {avg_roll_sharpe:.2f}",
        f"  Rolling Win Rate       : {avg_roll_win:.2f}%",
        f"  Equity Curve Slope     : {slope_eq:.2f}",
        f"  Equity Curve R²        : {r2:.4f}",
        f"  Consecutive Wins       : {cons_wins}",
        f"  Consecutive Losses     : {cons_losses}",
        SEP
    ]

    # Section 11 — Sensitivity
    lines += ["  SECTION 11 — SENSITIVITY ANALYSIS", DASH,
              f"  Base Strategy Net P/L : {fmt_money(net_pnl)}"]

    def sim_pnl(ds_mult=1.0, hr_add=0.0):
        total = 0.0
        for _, row in tdf.iterrows():
            orig_dS  = row['opt_dS']
            new_dS   = orig_dS * ds_mult
            d_entry  = row['opt_delta_entry']
            g        = row['opt_gamma']
            dt_hrs   = max(row['holding_mins'] / 60.0 + hr_add, 0)
            theta_e  = (OPTION_THETA / TRADING_HOURS_PER_DAY) * dt_hrs
            if USE_DYNAMIC_DELTA:
                steps = 10; step = new_dS / steps; cur_d = d_entry; d_eff = 0.0
                for _ in range(steps):
                    d_eff += cur_d * step
                    cur_d  = cur_d + g * step
            else:
                d_eff = d_entry * new_dS
            opt_ch = d_eff + theta_e
            sign   = 1 if OPTION_DIRECTION == "long" else -1
            total += opt_ch * sign * OPTION_LOT_SIZE - SLIPPAGE_PER_CONTRACT
        return total

    try:
        for label, kwargs in [
            ("ΔS +10%",  dict(ds_mult=1.10)),
            ("ΔS -10%",  dict(ds_mult=0.90)),
            ("ΔS +20%",  dict(ds_mult=1.20)),
            ("ΔS -20%",  dict(ds_mult=0.80)),
            ("Hold +1h", dict(hr_add=1.0)),
            ("Hold -1h", dict(hr_add=-1.0)),
        ]:
            sim = sim_pnl(**kwargs)
            lines.append(f"  P/L if {label:<28}: {fmt_money(sim)}  (Δ: {fmt_money(sim - net_pnl)})")
    except Exception as e:
        lines.append(f"  Sensitivity Error: {e}")

    # Section 12 — Outlier
    lines += [SEP, "  SECTION 12 — OUTLIER / TAIL DEPENDENCY", DASH]
    try:
        sorted_r  = returns.sort_values(ascending=False)
        n_remove  = min(5, len(sorted_r) - 1)
        trimmed   = sorted_r.iloc[n_remove:].sum()
        drop_pct  = (1 - safe_div(trimmed, net_pnl)) * 100 if net_pnl != 0 else 0
        tail_dep  = drop_pct > 30
        lines += [
            f"  Full P/L              : {fmt_money(net_pnl)}",
            f"  P/L (top {n_remove} removed)  : {fmt_money(trimmed)}",
            f"  Drop %                : {drop_pct:.1f}%",
            f"  Tail Dependent?       : {'⚠ YES — strategy relies on outliers' if tail_dep else '✅ NO — robust distribution'}",
        ]
    except Exception as e:
        lines.append(f"  Outlier Check Error: {e}")

    lines.append(SEP + "\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#   PRINT RESULTS
# ══════════════════════════════════════════════════════════════════

def print_results(tdf: pd.DataFrame):
    SEP  = '─' * 170
    SEP2 = '═' * 110

    print(f"\n{SEP}")
    print(f"  TRADE LOG  ({len(tdf)} trades)")
    print(SEP)
    print(f"{'#':<5} {'Dir':<6} {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>9} {'Exit':>9} {'SL@Entry':>10} {'SL@Exit':>9}"
          f" {'MFE':>7} {'MAE':>7} {'P&L':>9}"
          f" {'Opt Gross':>11} {'Slip':>7} {'Opt Net':>10}  Reason")
    print(SEP)
    for i, r in tdf.iterrows():
        print(f"{i+1:<5} {r['direction'].upper():<6} {fmt_ts(r['entry_time']):<22}"
              f" {fmt_ts(r['exit_time']):<22}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f}"
              f" {r['sl_at_entry']:>10.2f} {r['sl_at_exit']:>9.2f}"
              f" {r['mfe_pts']:>7.1f} {r['mae_pts']:>7.1f}"
              f" {r['pnl']:>+9.2f}"
              f" {r['opt_gross_pnl']:>11.2f} {r['opt_slippage']:>7.2f}"
              f" {r['opt_total_pnl']:>10.2f}  {r['exit_reason']}")

    metrics = compute_metrics(tdf)
    print(f"\n{'─'*62}")
    print("  UNDERLYING PERFORMANCE METRICS")
    print(f"{'─'*62}")
    for k, v in metrics.items():
        print(f"  {k:<30}: {v}")

    print(f"\n{'─'*62}")
    print("  DIRECTION BREAKDOWN")
    print(f"{'─'*62}")
    for d in ['long', 'short']:
        sub = tdf[tdf['direction'] == d]
        if sub.empty:
            continue
        pnl_s = sub['pnl']
        w = (pnl_s > 0).sum()
        print(f"  {d.upper():<6}  trades={len(sub)}  wins={w}  "
              f"wr={w/len(sub)*100:.1f}%  total={pnl_s.sum():.1f}  "
              f"avg={pnl_s.mean():.1f}  "
              f"avg_mfe={sub['mfe_pts'].mean():.1f}  "
              f"avg_mae={sub['mae_pts'].mean():.1f}")

    print(f"\n{'─'*62}")
    print("  EXIT REASON BREAKDOWN")
    print(f"{'─'*62}")
    bd = (tdf.groupby('exit_reason')['pnl']
          .agg(count='count', total_pnl='sum', avg_pnl='mean').round(2))
    print(bd.to_string())

    print(f"\n{'─'*62}")
    print("  MONTHLY P&L BREAKDOWN")
    print(f"{'─'*62}")
    tdf2 = tdf.copy()
    tdf2['month'] = tdf2['entry_time'].dt.tz_localize(None).dt.to_period('M').astype(str)
    monthly = (tdf2.groupby('month')['pnl']
               .agg(trades='count', total_pnl='sum',
                    win_trades=lambda x: (x > 0).sum())
               .assign(win_rate=lambda x: (x['win_trades'] / x['trades'] * 100).round(1))
               .round(2))
    print(monthly.to_string())

    print(f"\n{'─'*110}")
    print("  TIME SLOT P&L BREAKDOWN  (incl. Option P&L)")
    print(f"{'─'*110}")
    df_ts = build_time_slot_summary(tdf)
    print(df_ts.to_string(index=False))

    opt = build_option_summary(tdf)
    print(f"\n{SEP2}")
    print("  OPTION P&L SUMMARY")
    print(f"{'─'*62}")
    print(f"  Total Option P&L (net)   : ₹{opt['total_opt_pnl']:>12,.2f}")
    print(f"  Total Slippage Cost      : ₹{opt['total_slippage']:>12,.2f}")
    print(f"  Delta Contribution       : ₹{opt['total_delta_contrib']:>12,.2f}")
    print(f"  Gamma Contribution       : ₹{opt['total_gamma_contrib']:>12,.2f}")
    print(f"  Theta Cost               : ₹{opt['total_theta_cost']:>12,.2f}")
    print(f"  Vega Contribution        : ₹{opt['total_vega_contrib']:>12,.2f}")
    print(f"{'─'*62}")
    print(f"  Avg Break-even ΔS        : {opt['avg_breakeven_dS']} pts")
    print(SEP2)

    print(build_advanced_option_metrics(tdf))

    print(f"\n{'─'*62}")
    print("  MFE vs MAE ANALYSIS (SL Quality Check)")
    print(f"{'─'*62}")
    sl_hits = tdf[tdf['exit_reason'] == 'STOP_LOSS']
    if not sl_hits.empty:
        print(f"  SL trades where MAE < 50th pctile: "
              f"{(sl_hits['mae_pts'] < sl_hits['mae_pts'].quantile(0.5)).sum()} / {len(sl_hits)}")
        print(f"  Avg MAE on SL trades : {sl_hits['mae_pts'].mean():.1f} pts")
        print(f"  Avg MFE on SL trades : {sl_hits['mfe_pts'].mean():.1f} pts")
        print(f"  Trades that went in-favor before stopping: "
              f"{(sl_hits['mfe_pts'] > 5).sum()} / {len(sl_hits)}")


# ══════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    BAR = '═' * 68
    print(f"\n{BAR}")
    print("  SPEED DEMON SCALPER  v4  +  VWAP INSTITUTIONAL RETEST  +  OPTIONS ENGINE")
    print(f"{BAR}")

    cli_args = sys.argv[1:]
    mode     = 'csv' if cli_args else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\nData source : yfinance  |  Symbol : {YFINANCE_SYMBOL}")
        df        = fetch_yfinance('5m', '5m')
        save_base = os.path.join(os.getcwd(),
                                 YFINANCE_SYMBOL.replace('^', '').replace('.', '_') + '_speed_demon_v4_vwap')
    elif mode == 'csv':
        path5 = cli_args[0] if cli_args else CSV_5MIN_PATH
        if not os.path.exists(path5):
            print(f"ERROR: File not found: {path5}"); sys.exit(1)
        print(f"\nLoading 5m CSV : {path5}")
        df        = load_csv(path5, '5m')
        save_base = os.path.splitext(path5)[0] + '_speed_demon_v4_vwap'
    else:
        print(f"ERROR: Unknown DATA_MODE '{DATA_MODE}'"); sys.exit(1)

    print(f"\n{'─'*62}")
    print("  STRATEGY CONFIGURATION  (v4 — VWAP Institutional Retest)")
    print(f"{'─'*62}")
    print(f"  EMA Fast / Slow / Macro (5m)    : {EMA_FAST} / {EMA_SLOW} / {EMA_MACRO}")
    print(f"  HTF EMAs (15m)                  : {HTF_EMA_FAST} / {HTF_EMA_SLOW} / {HTF_EMA_TREND}")
    print(f"  ATR Period                       : {ATR_PERIOD}")
    print(f"  ADX Filter                       : ADX({ADX_PERIOD}) >= {ADX_MIN}")
    print(f"  VWAP Distance Gate               : ≤ {VWAP_MAX_DIST_PCT}% from VWAP")
    print(f"  LONG  SL Multiplier             : {LONG_ATR_SL_MULT}× ATR")
    print(f"  SHORT SL Multiplier             : {SHORT_ATR_SL_MULT}× ATR")
    print(f"  LONG  Break-Even Trigger        : +{LONG_BE_PCT}%")
    print(f"  SHORT Break-Even Trigger        : -{SHORT_BE_PCT}%")
    print(f"  Trailing Stop (post-BE)         : ATR × {TRAIL_ATR_MULT}")
    print(f"  Short Confirmation Bars         : {SHORT_CONFIRM_BARS} consecutive")
    print(f"  Soft EOD                        : {EOD_SOFT_EXIT}")
    print(f"  Hard EOD                        : {EOD_HARD_EXIT}")
    print(f"  Max Trades / Day                : {MAX_TRADES_PER_DAY}")
    print(f"  Consec-Loss Circuit Breaker     : halt after {MAX_CONSEC_LOSSES} losses")
    print(f"  Session: Prime                  : ON  (09:30–10:30)")
    print(f"  Session: Midday                 : {'ON' if ENABLE_MIDDAY else 'OFF'}  (11:30–13:30)")
    print(f"  Session: Euro                   : {'ON' if ENABLE_EURO else 'OFF'}  (14:15–15:00)")
    print(f"{'─'*62}")
    print(f"  OPTIONS ENGINE")
    print(f"{'─'*62}")
    print(f"  Lot Size                        : {OPTION_LOT_SIZE}")
    print(f"  Delta (Δ)                       : {OPTION_DELTA}")
    print(f"  Gamma (Γ)                       : {OPTION_GAMMA}")
    print(f"  Theta (Θ) per day               : {OPTION_THETA}")
    print(f"  Vega  (ν)                       : {OPTION_VEGA}")
    print(f"  Option Direction                : {OPTION_DIRECTION.upper()}")
    print(f"  Dynamic Delta (Γ-adjusted)      : {'ON' if USE_DYNAMIC_DELTA else 'OFF'}")
    print(f"  Slippage per contract           : ₹{SLIPPAGE_PER_CONTRACT}")
    print(f"{'─'*62}")

    print(f"\nComputing 5m indicators + ADX + VWAP ...")
    df = compute_indicators_5m(df)

    print(f"Computing 15m HTF bias ...")
    df = compute_htf_bias(df)

    print("Running backtest ...\n")
    trades_df, equity_df = run_backtest(df)

    print(f"\n{BAR}")
    print("  BACKTEST RESULTS")
    print(f"{BAR}")

    if trades_df.empty:
        print("\n  No trades generated.")
        print("  Try: reduce ADX_MIN, SPREAD_PCT_MIN, PRICE_GAP_MIN, or VWAP_MAX_DIST_PCT.")
        return

    print_results(trades_df)

    trades_out = save_base + '_trades.csv'
    equity_out = save_base + '_equity.csv'
    trades_df.to_csv(trades_out, index=False)
    equity_df.to_csv(equity_out, index=False)
    print(f"\n{'─'*62}")
    print(f"  Trades saved : {trades_out}")
    print(f"  Equity curve : {equity_out}")
    print(f"{BAR}\n")


if __name__ == '__main__':
    main()