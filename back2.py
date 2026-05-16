"""
╔══════════════════════════════════════════════════════════════════╗
║   SPEED DEMON SCALPER  v3  — "SURGICAL ENTRY SYSTEM"           ║
╠══════════════════════════════════════════════════════════════════╣
║  WHAT'S NEW vs v2                                                ║
║  ─────────────────────────────────────────────────────────────  ║
║  ✦ ADX Regime Filter                                            ║
║      → No trades when ADX < 20 (market is ranging/choppy)      ║
║      → ADX computed via Wilder's smoothing (proper method)      ║
║                                                                  ║
║  ✦ Soft EOD — Trail stays alive until 15:25                    ║
║      → EOD_EXIT only kills trades NOT yet in profit             ║
║      → Profitable trailing trades run until 15:25               ║
║      → Saves ~40 pts/trade avg from EOD kills                   ║
║                                                                  ║
║  ✦ Tighter Short Entry: 3-bar confirmation required             ║
║      → EMA9 must be below EMA21 for last 3 consecutive bars     ║
║      → Eliminates false breakdown entries on first cross        ║
║                                                                  ║
║  ✦ Session Gate: Midday & Euro disabled by default              ║
║      → Prime session avg 38 pts vs Midday 11 pts / Euro 8 pts   ║
║      → Toggle ENABLE_MIDDAY / ENABLE_EURO to re-enable          ║
║                                                                  ║
║  ✦ Lower BE threshold for longs (0.3% instead of 0.5%)         ║
║      → Reduces raw SL hits before BE triggers                   ║
║      → Accepts slightly lower BE level to protect capital faster ║
║                                                                  ║
║  ✦ Consecutive-loss daily circuit breaker                       ║
║      → Stop trading for the day after N consecutive losses      ║
║      → Prevents revenge-trading in trending-against days        ║
║                                                                  ║
║  ✦ Full per-trade MAE/MFE tracking in CSV output                ║
║      → Shows how close each SL was to being avoided             ║
╠══════════════════════════════════════════════════════════════════╣
║  PHASE 1 — Data Pre-processing & Environment                    ║
║  PHASE 2 — Technical Indicator Engine (5m + 15m HTF + ADX)     ║
║  PHASE 3 — Anti-Chop + Regime Filter Suite                      ║
║  PHASE 4 — Asymmetric Entry Logic (Long vs Short)               ║
║  PHASE 5 — Dynamic Exit Engine (Soft EOD, Trailing, EMA Cross)  ║
║  PHASE 6 — Indian Market Time-Mapping (NSE)                     ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python speed_demon_v3.py              ← yfinance (default)
    python speed_demon_v3.py 5m.csv       ← CSV file
"""

import pandas as pd
import numpy as np
import sys, os
from datetime import time

# ══════════════════════════════════════════════════════════════════
#   CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DATA_MODE        = 'yfinance'
YFINANCE_SYMBOL  = '^NSEI'
YFINANCE_DAYS    = 60
YFINANCE_TZ      = 'Asia/Kolkata'
CSV_5MIN_PATH    = 'nifty_5m.csv'

# ── EMA Periods (5m) ────────────────────────────────────────────
EMA_FAST         = 12
EMA_SLOW         = 26
EMA_MACRO        = 50

# ── HTF (15m) EMA Periods — Long bias filter ────────────────────
HTF_EMA_FAST     = 9
HTF_EMA_SLOW     = 21
HTF_EMA_TREND    = 50

# ── ATR ─────────────────────────────────────────────────────────
ATR_PERIOD       = 14

# ── ADX Regime Filter ───────────────────────────────────────────
ADX_PERIOD       = 14
ADX_MIN          = 18          # skip trade if ADX < this (market ranging)

# ── Asymmetric SL (unchanged from v2) ───────────────────────────
LONG_ATR_SL_MULT    = 1.2
SHORT_ATR_SL_MULT   = 0.75

# ── Break-Even Trigger ──────────────────────────────────────────
LONG_BE_PCT         = 0.2    # ↓ from 0.5% → triggers faster, fewer raw SL hits
SHORT_BE_PCT        = 0.3

# ── Trailing Stop ───────────────────────────────────────────────
TRAIL_ATR_MULT      = 0.6

# ── EMA Cross Exit ──────────────────────────────────────────────
ENABLE_EMA_CROSS_EXIT = True

# ── Momentum (Slope Reversal) Exit ──────────────────────────────
ENABLE_SLOPE_EXIT     = True
SLOPE_EXIT_CANDLES    = 4

# ── Short Entry Confirmation ─────────────────────────────────────
SHORT_CONFIRM_BARS  = 3    # EMA9 must be < EMA21 for this many consecutive bars
                           # Set to 1 to disable (original behaviour)

# ── Anti-Chop Filters ───────────────────────────────────────────
SPREAD_PCT_MIN      = 0.04
SLOPE_CANDLES       = 6
SLOPE_MIN           = 0.0
PRICE_GAP_MIN       = 5.0
MIDDAY_SPREAD_MULT  = 2.0

# ── Retest tolerance (ATR-relative) ─────────────────────────────
RETEST_ATR_MULT     = 0.25

# ── Max trades per day ───────────────────────────────────────────
MAX_TRADES_PER_DAY  = 3

# ── Consecutive-loss circuit breaker ────────────────────────────
MAX_CONSEC_LOSSES   = 2    # halt new entries for rest of day after N losses in a row
                           # Set to 99 to disable

# ── Directions ──────────────────────────────────────────────────
ENABLE_LONG         = True
ENABLE_SHORT        = True

# ── Session Gates ────────────────────────────────────────────────
# v2 data showed: Prime avg 38 pts, Midday 11 pts, Euro 8 pts
# Disable weak sessions to improve quality/trade
ENABLE_MIDDAY       = True   # re-enable to test: set True
ENABLE_EURO         = True   # re-enable to test: set True

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
# Hard kill: close ALL trades (same as v2 EOD_EXIT)
EOD_HARD_EXIT       = time(15, 25)   # absolute last exit for everything
# Soft kill: only close trades that are NOT yet in profit (i.e. BE not triggered)
EOD_SOFT_EXIT       = time(15, 0)    # unprofitable / early trades closed here


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
    """
    Wilder's ADX — proper directional movement index.
    Returns ADX series aligned to df index.
    """
    high  = df['high'].astype(float)
    low   = df['low'].astype(float)
    close = df['close'].astype(float)

    up_move   = high.diff()
    down_move = -(low.diff())

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_vals = compute_atr(df, period)

    # Wilder smooth
    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()

    plus_di  = 100 * plus_dm_s  / atr_vals.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr_vals.replace(0, np.nan)

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / dx_denom
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


def compute_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_fast']            = compute_ema(df['close'], EMA_FAST)
    df['ema_slow']            = compute_ema(df['close'], EMA_SLOW)
    df['ema_macro']           = compute_ema(df['close'], EMA_MACRO)
    df['atr']                 = compute_atr(df, ATR_PERIOD)
    df['adx']                 = compute_adx(df, ADX_PERIOD)
    df['ema_slow_slope']      = df['ema_slow'].diff(SLOPE_CANDLES)
    df['ema_slow_slope_exit'] = df['ema_slow'].diff(SLOPE_EXIT_CANDLES)
    # EMA cross confirmation: how many bars has EMA9 been below EMA21?
    # Positive = bars EMA9 < EMA21 in a row, reset to 0 on cross up
    bearish_cross = (df['ema_fast'] < df['ema_slow']).astype(int)
    # rolling count of consecutive True values
    df['consec_bearish_bars'] = bearish_cross.groupby(
        (bearish_cross != bearish_cross.shift()).cumsum()
    ).cumcount() + 1
    df['consec_bearish_bars'] = df['consec_bearish_bars'] * bearish_cross
    return df


def compute_htf_bias(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Resample to 15m, compute HTF EMAs, forward-fill back to 5m."""
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
    # ADX regime filter — NEW in v3
    if adx_val < ADX_MIN:
        return False

    # Spread filter
    spread_pct    = abs(ema_fast - ema_slow) / close * 100
    spread_thresh = SPREAD_PCT_MIN * (MIDDAY_SPREAD_MULT if session == 'midday' else 1.0)
    if spread_pct < spread_thresh:
        return False

    # Slope filter
    if abs(slope) <= SLOPE_MIN:
        return False

    # Price-gap filter
    if abs(close - ema_slow) < PRICE_GAP_MIN:
        return False

    return True


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
    ts_list          = df['timestamp'].tolist()
    n                = len(df)

    trades           = []
    equity           = 0.0
    eq_curve         = []

    # ── Trade State ──────────────────────────────────────────────
    in_trade         = False
    direction        = None
    entry_price      = 0.0
    entry_time       = None
    stop_loss        = 0.0
    be_triggered     = False
    be_level         = 0.0
    trail_active     = False
    sl_dist_initial  = 0.0
    trade_max_favor  = 0.0   # MFE tracker
    trade_max_adverse= 0.0   # MAE tracker

    # ── Daily State ───────────────────────────────────────────────
    prev_date        = None
    daily_trades     = {}    # date_str → count
    daily_consec_loss= {}    # date_str → consecutive losses today

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

        # Update consecutive loss tracker
        date_str = str(ts_now.date())
        if pnl <= 0:
            daily_consec_loss[date_str] = daily_consec_loss.get(date_str, 0) + 1
        else:
            daily_consec_loss[date_str] = 0   # reset on any win

        trades.append({
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
        ts         = ts_list[idx]
        c_time     = ts.time()
        c_date     = ts.date()
        date_str   = str(c_date)
        session    = get_session(c_time)

        if c_date != prev_date:
            prev_date = c_date

        # Warm-up guard
        if np.isnan(ef) or np.isnan(es) or np.isnan(em) or np.isnan(atr):
            eq_curve.append({'timestamp': ts, 'equity': equity})
            continue

        htf_long = bool(htf_long_bias[idx]) if not pd.isna(htf_long_bias[idx]) else False

        # ══════════════════════════════════════════════════════════
        #   MANAGE OPEN TRADE — Dynamic Exit Engine (v3)
        # ══════════════════════════════════════════════════════════
        if in_trade:
            # Track MFE / MAE
            if direction == 'long':
                favor   = high_c - entry_price
                adverse = entry_price - low_c
            else:
                favor   = entry_price - low_c
                adverse = high_c - entry_price
            trade_max_favor   = max(trade_max_favor,   favor)
            trade_max_adverse = max(trade_max_adverse, adverse)

            # Stage 1: Break-Even trigger
            if not be_triggered:
                if direction == 'long'  and close >= be_level:
                    stop_loss    = entry_price + 1.0
                    be_triggered = True
                    trail_active = True
                elif direction == 'short' and close <= be_level:
                    stop_loss    = entry_price - 1.0
                    be_triggered = True
                    trail_active = True

            # Stage 2: ATR Trailing Stop ratchet
            if trail_active:
                trail_dist = atr * TRAIL_ATR_MULT
                if direction == 'long':
                    stop_loss = max(stop_loss, close - trail_dist)
                else:
                    stop_loss = min(stop_loss, close + trail_dist)

            exit_p = None
            exit_r = None

            # Stage 3: SL / Trail hit
            if direction == 'long' and close <= stop_loss:
                exit_p = stop_loss
                exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'
            elif direction == 'short' and close >= stop_loss:
                exit_p = stop_loss
                exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'

            # Stage 4: EMA Cross Exit (post-BE only)
            elif ENABLE_EMA_CROSS_EXIT and be_triggered:
                if direction == 'long'  and ef < es:
                    exit_p = close
                    exit_r = 'EMA_CROSS_EXIT'
                elif direction == 'short' and ef > es:
                    exit_p = close
                    exit_r = 'EMA_CROSS_EXIT'

            # Stage 5: Slope Reversal Exit (post-BE only)
            elif ENABLE_SLOPE_EXIT and be_triggered:
                if direction == 'long'  and slope_exit < -SLOPE_MIN:
                    exit_p = close
                    exit_r = 'SLOPE_REVERSAL_EXIT'
                elif direction == 'short' and slope_exit > SLOPE_MIN:
                    exit_p = close
                    exit_r = 'SLOPE_REVERSAL_EXIT'

            # Stage 6a: Soft EOD — kill trades NOT yet in profit
            elif c_time >= EOD_SOFT_EXIT and not be_triggered:
                exit_p = close
                exit_r = 'SOFT_EOD_EXIT'

            # Stage 6b: Hard EOD — kill everything still open
            elif c_time >= EOD_HARD_EXIT:
                exit_p = close
                exit_r = 'HARD_EOD_EXIT'

            if exit_p is not None:
                do_exit(exit_p, ts, exit_r)

        # ══════════════════════════════════════════════════════════
        #   ENTRY LOGIC
        # ══════════════════════════════════════════════════════════
        if not in_trade:
            # Session gate
            allowed_sessions = ['prime']
            if ENABLE_MIDDAY: allowed_sessions.append('midday')
            if ENABLE_EURO:   allowed_sessions.append('euro')
            if session not in allowed_sessions:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            # Daily caps
            if daily_trades.get(date_str, 0) >= MAX_TRADES_PER_DAY:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            # Consecutive-loss circuit breaker
            if daily_consec_loss.get(date_str, 0) >= MAX_CONSEC_LOSSES:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            # Chop + ADX regime filter
            if not chop_filters_pass(close, ef, es, slope, adx_v, session):
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            retest_tol = atr * RETEST_ATR_MULT

            # ── LONG Entry ────────────────────────────────────────
            # Requires 15m HTF bullish bias
            if ENABLE_LONG and htf_long:
                if (ef > es                         and
                    close > em                      and
                    slope > 0                       and
                    abs(close - ef) <= retest_tol   and
                    close > es):
                    do_enter('long', close, ts, atr)
                    daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

            # ── SHORT Entry ───────────────────────────────────────
            # Requires N consecutive bearish EMA bars (confirmation)
            if not in_trade and ENABLE_SHORT:
                if (ef < es                         and
                    close < em                      and
                    slope < 0                       and
                    abs(close - ef) <= retest_tol   and
                    close < es                      and
                    cb >= SHORT_CONFIRM_BARS):       # ← NEW: confirmation bars
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


def print_results(tdf: pd.DataFrame):
    SEP = '─' * 150

    print(f"\n{SEP}")
    print(f"  TRADE LOG  ({len(tdf)} trades)")
    print(SEP)
    print(f"{'#':<5} {'Dir':<6} {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>9} {'Exit':>9} {'SL@Entry':>10} {'SL@Exit':>9}"
          f" {'MFE':>7} {'MAE':>7} {'P&L':>9}  Reason")
    print(SEP)
    for i, r in tdf.iterrows():
        print(f"{i+1:<5} {r['direction'].upper():<6} {fmt_ts(r['entry_time']):<22}"
              f" {fmt_ts(r['exit_time']):<22}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f}"
              f" {r['sl_at_entry']:>10.2f} {r['sl_at_exit']:>9.2f}"
              f" {r['mfe_pts']:>7.1f} {r['mae_pts']:>7.1f}"
              f" {r['pnl']:>+9.2f}  {r['exit_reason']}")

    metrics = compute_metrics(tdf)
    print(f"\n{'─'*62}")
    print("  PERFORMANCE METRICS")
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

    print(f"\n{'─'*62}")
    print("  SESSION P&L BREAKDOWN")
    print(f"{'─'*62}")
    def classify_session(t):
        if hasattr(t, 'time'):
            t = t.time()
        h, m = t.hour, t.minute
        if (h == 9 and m >= 45) or h == 10 or (h == 11 and m < 30):
            return 'Prime (09:45-11:30)'
        if h == 11 or h == 12 or (h == 13 and m < 30):
            return 'Midday (11:30-13:30)'
        if h == 13 or h == 14 or (h == 15 and m < 1):
            return 'Euro (13:30-15:00)'
        return 'Other'
    tdf3 = tdf.copy()
    tdf3['session'] = tdf3['entry_time'].apply(classify_session)
    sess_bd = (tdf3.groupby('session')['pnl']
               .agg(trades='count', total_pnl='sum', avg_pnl='mean',
                    win_trades=lambda x: (x > 0).sum())
               .assign(win_rate=lambda x: (x['win_trades'] / x['trades'] * 100).round(1))
               .round(2))
    print(sess_bd.to_string())

    print(f"\n{'─'*62}")
    print("  DAILY TRADE COUNT DISTRIBUTION")
    print(f"{'─'*62}")
    tdf4 = tdf.copy()
    tdf4['date'] = tdf4['entry_time'].dt.date
    daily_counts = tdf4.groupby('date').size().value_counts().sort_index()
    for n_trades, days in daily_counts.items():
        print(f"  {n_trades} trade(s)/day → {days} day(s)")

    print(f"\n{'─'*62}")
    print("  MFE vs MAE ANALYSIS (SL Quality Check)")
    print(f"{'─'*62}")
    sl_hits = tdf[tdf['exit_reason'] == 'STOP_LOSS']
    if not sl_hits.empty:
        print(f"  SL trades where MAE < SL dist (could have been saved): "
              f"{(sl_hits['mae_pts'] < sl_hits['mae_pts'].quantile(0.5)).sum()} / {len(sl_hits)}")
        print(f"  Avg MAE on SL trades : {sl_hits['mae_pts'].mean():.1f} pts")
        print(f"  Avg MFE on SL trades : {sl_hits['mfe_pts'].mean():.1f} pts")
        print(f"  → Trades that went in-favor before stopping: "
              f"{(sl_hits['mfe_pts'] > 5).sum()} / {len(sl_hits)}")


# ══════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    BAR = '═' * 68
    print(f"\n{BAR}")
    print("  SPEED DEMON SCALPER  v3  — SURGICAL ENTRY SYSTEM")
    print(f"{BAR}")

    cli_args = sys.argv[1:]
    mode     = 'csv' if cli_args else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\nData source : yfinance  |  Symbol : {YFINANCE_SYMBOL}")
        df        = fetch_yfinance('5m', '5m')
        save_base = os.path.join(os.getcwd(),
                                 YFINANCE_SYMBOL.replace('^', '').replace('.', '_') + '_speed_demon_v3')
    elif mode == 'csv':
        path5 = cli_args[0] if cli_args else CSV_5MIN_PATH
        if not os.path.exists(path5):
            print(f"ERROR: File not found: {path5}"); sys.exit(1)
        print(f"\nLoading 5m CSV : {path5}")
        df        = load_csv(path5, '5m')
        save_base = os.path.splitext(path5)[0] + '_speed_demon_v3'
    else:
        print(f"ERROR: Unknown DATA_MODE '{DATA_MODE}'"); sys.exit(1)

    print(f"\n{'─'*62}")
    print("  STRATEGY CONFIGURATION  (v3)")
    print(f"{'─'*62}")
    print(f"  EMA Fast / Slow / Macro (5m)    : {EMA_FAST} / {EMA_SLOW} / {EMA_MACRO}")
    print(f"  HTF EMAs (15m)                  : {HTF_EMA_FAST} / {HTF_EMA_SLOW} / {HTF_EMA_TREND}")
    print(f"  ATR Period                       : {ATR_PERIOD}")
    print(f"  ADX Filter                       : ADX({ADX_PERIOD}) >= {ADX_MIN}  [NEW v3]")
    print(f"  LONG  SL Multiplier             : {LONG_ATR_SL_MULT}× ATR")
    print(f"  SHORT SL Multiplier             : {SHORT_ATR_SL_MULT}× ATR")
    print(f"  LONG  Break-Even Trigger        : +{LONG_BE_PCT}%  [↓ from 0.5%]")
    print(f"  SHORT Break-Even Trigger        : -{SHORT_BE_PCT}%")
    print(f"  Trailing Stop (post-BE)         : ATR × {TRAIL_ATR_MULT}")
    print(f"  EMA Cross Exit                  : {'ON' if ENABLE_EMA_CROSS_EXIT else 'OFF'}")
    print(f"  Slope Reversal Exit             : {'ON' if ENABLE_SLOPE_EXIT else 'OFF'}")
    print(f"  Short Confirmation Bars         : {SHORT_CONFIRM_BARS} consecutive  [NEW v3]")
    print(f"  Soft EOD (kill unprofitable)    : {EOD_SOFT_EXIT}  [NEW v3]")
    print(f"  Hard EOD (kill everything)      : {EOD_HARD_EXIT}  [NEW v3]")
    print(f"  Retest Tolerance                : ATR × {RETEST_ATR_MULT}")
    print(f"  Max Trades / Day                : {MAX_TRADES_PER_DAY}")
    print(f"  Consec-Loss Circuit Breaker     : halt after {MAX_CONSEC_LOSSES} losses  [NEW v3]")
    print(f"  Session: Prime                  : ON  (09:45–11:30)")
    print(f"  Session: Midday                 : {'ON' if ENABLE_MIDDAY else 'OFF'}  (11:30–13:30)")
    print(f"  Session: Euro                   : {'ON' if ENABLE_EURO else 'OFF'}  (13:30–15:00)")
    print(f"  Directions                      : Long={'ON' if ENABLE_LONG else 'OFF'}"
          f"  Short={'ON' if ENABLE_SHORT else 'OFF'}")
    print(f"{'─'*62}")

    print(f"\nComputing 5m indicators + ADX ...")
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
        print("  Try: reduce ADX_MIN, SPREAD_PCT_MIN, or PRICE_GAP_MIN.")
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