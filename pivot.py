"""
╔══════════════════════════════════════════════════════════════════╗
║   BREAKOUT + EMA CONTRACTION SCALPER  v1                        ║
╠══════════════════════════════════════════════════════════════════╣
║  CORE LOGIC                                                      ║
║  ─────────────────────────────────────────────────────────────  ║
║  LONG  : Price breaks ABOVE the 17-candle HIGH                  ║
║          AND EMA9 & EMA21 are contracting (gap shrinking)        ║
║          AND EMA9 crosses above EMA21 within the last N bars     ║
║                                                                  ║
║  SHORT : Price breaks BELOW the 17-candle LOW                   ║
║          AND EMA9 & EMA21 are contracting (gap shrinking)        ║
║          AND EMA9 crosses below EMA21 within the last N bars     ║
║                                                                  ║
║  CONTRACTION definition                                          ║
║    → |EMA9 − EMA21| on bar-1 < |EMA9 − EMA21| on bar-N         ║
║      (gap is narrowing over CONTRACTION_LOOKBACK bars)           ║
║    → AND EMA9/EMA21 have not yet crossed  (pre-cross zone)       ║
║    → OR a fresh cross happened ≤ CROSS_CONFIRM_BARS ago          ║
║                                                                  ║
║  EXIT STACK (same as Speed Demon v3)                             ║
║    1. Initial ATR stop-loss                                      ║
║    2. Break-even → trail ratchet                                 ║
║    3. EMA cross exit (post-BE)                                   ║
║    4. Slope reversal exit (post-BE)                              ║
║    5. Soft EOD at 15:00 (unprofitable trades only)               ║
║    6. Hard EOD at 15:25                                          ║
║                                                                  ║
║  OPTIONAL FILTERS (toggle in CONFIG)                             ║
║    ADX_FILTER   – skip ranging markets                           ║
║    HTF_FILTER   – 15m EMA trend alignment                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Usage                                                           ║
║    python breakout_ema_cross_v1.py              ← yfinance       ║
║    python breakout_ema_cross_v1.py data.csv     ← CSV file       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import sys, os
from datetime import time

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# ── Data ─────────────────────────────────────────────────────────
DATA_MODE        = 'yfinance'       # 'yfinance' | 'csv'
YFINANCE_SYMBOL  = '^NSEI'
YFINANCE_DAYS    = 60
YFINANCE_TZ      = 'Asia/Kolkata'
CSV_PATH         = 'nifty_5m.csv'

# ── Core Strategy Parameters ─────────────────────────────────────
BREAKOUT_CANDLES      = 25     # look-back window for high/low breakout level
EMA_FAST              = 9
EMA_SLOW              = 21

# ── EMA Contraction Logic ─────────────────────────────────────────
# How many bars back to measure the EMA gap (to detect contraction)
CONTRACTION_LOOKBACK  = 5      # gap must be shrinking over this many bars
# Fresh cross is valid if it happened within this many bars
CROSS_CONFIRM_BARS    = 3      # 0 = must be contracting but NOT yet crossed
                               # set to 0 to require pre-cross zone only

# ── Breakout Confirmation ─────────────────────────────────────────
# Price must close above/below the 17-candle level (not just wick)
REQUIRE_CLOSE_BREAK   = True   # False = candle high/low touch is enough

# ── Optional Filters ──────────────────────────────────────────────
ENABLE_ADX_FILTER     = True   # skip trades when market is ranging
ADX_PERIOD            = 14
ADX_MIN               = 18     # minimum ADX to allow entries

ENABLE_HTF_FILTER     = False  # require 15m trend alignment
HTF_EMA_FAST          = 9
HTF_EMA_SLOW          = 21
HTF_EMA_TREND         = 50

# ── ATR ───────────────────────────────────────────────────────────
ATR_PERIOD            = 14

# ── Stop-Loss (ATR multiples) ─────────────────────────────────────
LONG_ATR_SL_MULT      = 1.2
SHORT_ATR_SL_MULT     = 0.75

# ── Break-Even Trigger ────────────────────────────────────────────
LONG_BE_PCT           = 0.2    # % move in favour before BE kicks in
SHORT_BE_PCT          = 0.3

# ── Trailing Stop (post-BE) ───────────────────────────────────────
TRAIL_ATR_MULT        = 0.6

# ── Post-BE Exit Logic ────────────────────────────────────────────
ENABLE_EMA_CROSS_EXIT = True
ENABLE_SLOPE_EXIT     = True
SLOPE_EXIT_CANDLES    = 4

# ── Session Gates ─────────────────────────────────────────────────
ENABLE_LONG           = True
ENABLE_SHORT          = True
ENABLE_MIDDAY         = True
ENABLE_EURO           = True

OBSERVE_START         = time(9,  15)
OBSERVE_END           = time(9,  30)
PRIME_START           = time(9,  30)
PRIME_END             = time(10, 30)
MIDDAY_START          = time(11, 30)
MIDDAY_END            = time(13, 30)
EURO_START            = time(14, 15)
EURO_END              = time(15, 0)
SQUAREOFF_START       = time(15, 0)

EOD_SOFT_EXIT         = time(15, 0)
EOD_HARD_EXIT         = time(15, 25)

# ── Daily Limits ──────────────────────────────────────────────────
MAX_TRADES_PER_DAY    = 3
MAX_CONSEC_LOSSES     = 2


# ══════════════════════════════════════════════════════════════════
#  INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low = df['high'].astype(float), df['low'].astype(float)
    up_move   = high.diff()
    down_move = -(low.diff())
    plus_dm   = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_vals  = compute_atr(df, period)
    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    plus_di    = 100 * plus_dm_s  / atr_vals.replace(0, np.nan)
    minus_di   = 100 * minus_dm_s / atr_vals.replace(0, np.nan)
    dx_denom   = (plus_di + minus_di).replace(0, np.nan)
    dx         = 100 * (plus_di - minus_di).abs() / dx_denom
    return dx.ewm(alpha=1/period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_fast']  = compute_ema(df['close'], EMA_FAST)
    df['ema_slow']  = compute_ema(df['close'], EMA_SLOW)
    df['atr']       = compute_atr(df, ATR_PERIOD)
    df['adx']       = compute_adx(df, ADX_PERIOD) if ENABLE_ADX_FILTER else 99.0

    # ── 17-candle rolling high/low (exclude current candle → shift 1) ──
    # We use the HIGH of the previous 17 candles as the breakout level
    # so the current candle's break is the signal
    df['breakout_high'] = df['high'].shift(1).rolling(BREAKOUT_CANDLES).max()
    df['breakout_low']  = df['low'].shift(1).rolling(BREAKOUT_CANDLES).min()

    # ── EMA gap (absolute distance between EMA9 and EMA21) ────────────
    df['ema_gap'] = (df['ema_fast'] - df['ema_slow']).abs()

    # ── Contraction flag: is the gap shrinking over CONTRACTION_LOOKBACK bars? ──
    # gap_now < gap N bars ago → contracting
    df['ema_gap_prev']   = df['ema_gap'].shift(CONTRACTION_LOOKBACK)
    df['contracting']    = df['ema_gap'] < df['ema_gap_prev']

    # ── Cross detection ───────────────────────────────────────────────
    # bullish cross: EMA9 crosses above EMA21 (was below, now above)
    df['bullish_cross'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
    df['bearish_cross'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))

    # ── Bars since last cross ─────────────────────────────────────────
    # 0 means current bar is a cross, 1 = crossed 1 bar ago, etc.
    def bars_since(cross_series: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=cross_series.index)
        count = np.inf
        for i in range(len(cross_series)):
            if cross_series.iloc[i]:
                count = 0
            else:
                count += 1
            out.iloc[i] = count
        return out

    df['bars_since_bullish_cross'] = bars_since(df['bullish_cross'])
    df['bars_since_bearish_cross'] = bars_since(df['bearish_cross'])

    # ── Slope for exit ────────────────────────────────────────────────
    df['ema_slow_slope_exit'] = df['ema_slow'].diff(SLOPE_EXIT_CANDLES)

    return df


def compute_htf_bias(df_5m: pd.DataFrame) -> pd.DataFrame:
    df_5m = df_5m.copy().set_index('timestamp')
    df_15m = df_5m['close'].resample('15min').ohlc().dropna()
    df_15m.columns = ['open', 'high', 'low', 'close']
    df_15m['htf_fast']  = compute_ema(df_15m['close'], HTF_EMA_FAST)
    df_15m['htf_slow']  = compute_ema(df_15m['close'], HTF_EMA_SLOW)
    df_15m['htf_trend'] = compute_ema(df_15m['close'], HTF_EMA_TREND)
    df_15m['htf_long_bias']  = (df_15m['htf_fast'] > df_15m['htf_slow']) & (df_15m['htf_slow'] > df_15m['htf_trend'])
    df_15m['htf_short_bias'] = (df_15m['htf_fast'] < df_15m['htf_slow']) & (df_15m['htf_slow'] < df_15m['htf_trend'])
    htf_reindexed = df_15m[['htf_long_bias', 'htf_short_bias']].reindex(df_5m.index, method='ffill')
    return df_5m.join(htf_reindexed).reset_index()


# ══════════════════════════════════════════════════════════════════
#  SESSION HELPER
# ══════════════════════════════════════════════════════════════════

def get_session(t: time) -> str:
    if OBSERVE_START <= t < OBSERVE_END:  return 'observe'
    if PRIME_START   <= t < PRIME_END:    return 'prime'
    if MIDDAY_START  <= t < MIDDAY_END:   return 'midday'
    if EURO_START    <= t < EURO_END:     return 'euro'
    if t >= SQUAREOFF_START:              return 'squareoff'
    return 'outside'


# ══════════════════════════════════════════════════════════════════
#  DATA LOADERS  (same as base script)
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


def fetch_yfinance() -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance"); sys.exit(1)
    import warnings
    print(f"  Fetching {YFINANCE_SYMBOL} 5m (last {YFINANCE_DAYS} days) ...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = yf.download(tickers=YFINANCE_SYMBOL, period=f'{YFINANCE_DAYS}d',
                          interval='5m', progress=False, auto_adjust=True)
    if raw.empty:
        print(f"ERROR: No data for '{YFINANCE_SYMBOL}'."); sys.exit(1)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={'adj close': 'close', 'adj_close': 'close'})
    if 'volume' not in raw.columns: raw['volume'] = 0
    raw = raw[['open', 'high', 'low', 'close', 'volume']].dropna()
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize('UTC')
    raw.index = raw.index.tz_convert(YFINANCE_TZ)
    df = raw.reset_index()
    ts_col = next(c for c in df.columns if c.lower() in ('datetime', 'date', 'timestamp'))
    df = df.rename(columns={ts_col: 'timestamp'})
    df = _standardise(df, YFINANCE_TZ)
    print(f"    Rows: {len(df)}  |  {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    ts_col = next((c for c in ['timestamp', 'datetime', 'date_time', 'time', 'date']
                   if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"No timestamp column found. Got: {list(df.columns)}")
    df['timestamp'] = pd.to_datetime(df[ts_col], dayfirst=True)
    if ts_col != 'timestamp':
        df = df.drop(columns=[ts_col])
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = _standardise(df, 'Asia/Kolkata')
    print(f"    Rows: {len(df)}  |  {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ══════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame) -> tuple:
    closes      = df['close'].astype(float).values
    highs       = df['high'].astype(float).values
    lows        = df['low'].astype(float).values
    ema_fast    = df['ema_fast'].values
    ema_slow    = df['ema_slow'].values
    atrs        = df['atr'].values
    adx_vals    = df['adx'].values
    br_high     = df['breakout_high'].values
    br_low      = df['breakout_low'].values
    contracting = df['contracting'].values
    bsc_bull    = df['bars_since_bullish_cross'].values   # bars since bullish cross
    bsc_bear    = df['bars_since_bearish_cross'].values   # bars since bearish cross
    slopes_exit = df['ema_slow_slope_exit'].values
    ts_list     = df['timestamp'].tolist()

    htf_long_bias  = df['htf_long_bias'].values  if 'htf_long_bias'  in df.columns else np.ones(len(df), dtype=bool)
    htf_short_bias = df['htf_short_bias'].values if 'htf_short_bias' in df.columns else np.ones(len(df), dtype=bool)

    n       = len(df)
    trades  = []
    equity  = 0.0
    eq_curve= []

    # ── Trade state ───────────────────────────────────────────────
    in_trade        = False
    direction       = None
    entry_price     = 0.0
    entry_time      = None
    stop_loss       = 0.0
    be_triggered    = False
    be_level        = 0.0
    trail_active    = False
    sl_dist_initial = 0.0
    mfe             = 0.0
    mae             = 0.0

    # ── Daily state ───────────────────────────────────────────────
    daily_trades     = {}
    daily_consec_loss= {}

    def do_enter(dir_str, close_p, ts_now, atr_val):
        nonlocal in_trade, direction, entry_price, entry_time
        nonlocal stop_loss, be_triggered, be_level, trail_active, sl_dist_initial, mfe, mae
        sl_mult        = LONG_ATR_SL_MULT if dir_str == 'long' else SHORT_ATR_SL_MULT
        sl_dist        = atr_val * sl_mult
        in_trade       = True
        direction      = dir_str
        entry_price    = close_p
        entry_time     = ts_now
        be_triggered   = False
        trail_active   = False
        sl_dist_initial= sl_dist
        mfe = mae      = 0.0
        if direction == 'long':
            stop_loss = entry_price - sl_dist
            be_level  = entry_price * (1 + LONG_BE_PCT / 100)
        else:
            stop_loss = entry_price + sl_dist
            be_level  = entry_price * (1 - SHORT_BE_PCT / 100)

    def do_exit(exit_p, ts_now, reason):
        nonlocal in_trade, direction, entry_price, entry_time
        nonlocal stop_loss, be_triggered, be_level, trail_active, sl_dist_initial, equity, mfe, mae

        pnl = round(
            (exit_p - entry_price) if direction == 'long'
            else (entry_price - exit_p), 2
        )
        equity += pnl
        date_str = str(ts_now.date())
        daily_consec_loss[date_str] = 0 if pnl > 0 else daily_consec_loss.get(date_str, 0) + 1

        trades.append({
            'direction':   direction,
            'entry_time':  entry_time,
            'exit_time':   ts_now,
            'entry_price': entry_price,
            'exit_price':  exit_p,
            'sl_initial':  (entry_price - sl_dist_initial if direction == 'long'
                            else entry_price + sl_dist_initial),
            'sl_final':    stop_loss,
            'be_triggered':be_triggered,
            'pnl':         pnl,
            'mfe_pts':     round(mfe, 2),
            'mae_pts':     round(mae, 2),
            'exit_reason': reason,
        })

        in_trade = False; direction = None; entry_price = 0.0; entry_time = None
        stop_loss = 0.0; be_triggered = False; be_level = 0.0
        trail_active = False; sl_dist_initial = 0.0; mfe = mae = 0.0

    for idx in range(n):
        close   = closes[idx]
        high_c  = highs[idx]
        low_c   = lows[idx]
        ef      = ema_fast[idx]
        es      = ema_slow[idx]
        atr     = float(atrs[idx])
        adx_v   = float(adx_vals[idx]) if not np.isnan(adx_vals[idx]) else 0.0
        b_high  = float(br_high[idx])  if not np.isnan(br_high[idx])  else np.inf
        b_low   = float(br_low[idx])   if not np.isnan(br_low[idx])   else -np.inf
        is_cont = bool(contracting[idx]) if not np.isnan(contracting[idx]) else False
        bull_cs = float(bsc_bull[idx])  if not np.isnan(bsc_bull[idx])  else np.inf
        bear_cs = float(bsc_bear[idx])  if not np.isnan(bsc_bear[idx])  else np.inf
        sl_exit = float(slopes_exit[idx]) if not np.isnan(slopes_exit[idx]) else 0.0
        ts      = ts_list[idx]
        c_time  = ts.time()
        date_str= str(ts.date())
        session = get_session(c_time)

        # Warm-up guard
        if any(np.isnan(x) for x in [ef, es, atr, b_high, b_low]):
            eq_curve.append({'timestamp': ts, 'equity': equity})
            continue

        htf_long  = bool(htf_long_bias[idx])  if not np.isnan(float(htf_long_bias[idx]))  else True
        htf_short = bool(htf_short_bias[idx]) if not np.isnan(float(htf_short_bias[idx])) else True

        # ══════════════════════════════════════════════════════════
        #  MANAGE OPEN TRADE
        # ══════════════════════════════════════════════════════════
        if in_trade:
            # MFE / MAE tracking
            if direction == 'long':
                mfe = max(mfe, high_c - entry_price)
                mae = max(mae, entry_price - low_c)
            else:
                mfe = max(mfe, entry_price - low_c)
                mae = max(mae, high_c - entry_price)

            # Break-even trigger
            if not be_triggered:
                if direction == 'long'  and close >= be_level:
                    stop_loss = entry_price + 1.0; be_triggered = True; trail_active = True
                elif direction == 'short' and close <= be_level:
                    stop_loss = entry_price - 1.0; be_triggered = True; trail_active = True

            # Trail ratchet
            if trail_active:
                td = atr * TRAIL_ATR_MULT
                if direction == 'long':  stop_loss = max(stop_loss, close - td)
                else:                    stop_loss = min(stop_loss, close + td)

            exit_p = None; exit_r = None

            # SL / Trail hit
            if direction == 'long'  and close <= stop_loss:
                exit_p, exit_r = stop_loss, ('TRAIL_SL' if trail_active else 'STOP_LOSS')
            elif direction == 'short' and close >= stop_loss:
                exit_p, exit_r = stop_loss, ('TRAIL_SL' if trail_active else 'STOP_LOSS')

            # EMA cross exit (post-BE)
            elif ENABLE_EMA_CROSS_EXIT and be_triggered:
                if   direction == 'long'  and ef < es: exit_p, exit_r = close, 'EMA_CROSS_EXIT'
                elif direction == 'short' and ef > es: exit_p, exit_r = close, 'EMA_CROSS_EXIT'

            # Slope reversal exit (post-BE)
            elif ENABLE_SLOPE_EXIT and be_triggered:
                if   direction == 'long'  and sl_exit < 0: exit_p, exit_r = close, 'SLOPE_REV_EXIT'
                elif direction == 'short' and sl_exit > 0: exit_p, exit_r = close, 'SLOPE_REV_EXIT'

            # Soft EOD — kill only unprofitable trades
            elif c_time >= EOD_SOFT_EXIT and not be_triggered:
                exit_p, exit_r = close, 'SOFT_EOD_EXIT'

            # Hard EOD — kill everything
            elif c_time >= EOD_HARD_EXIT:
                exit_p, exit_r = close, 'HARD_EOD_EXIT'

            if exit_p is not None:
                do_exit(exit_p, ts, exit_r)

        # ══════════════════════════════════════════════════════════
        #  ENTRY LOGIC
        # ══════════════════════════════════════════════════════════
        if not in_trade:
            # Session gate
            allowed = ['prime']
            if ENABLE_MIDDAY: allowed.append('midday')
            if ENABLE_EURO:   allowed.append('euro')
            if session not in allowed:
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

            # ADX filter
            if ENABLE_ADX_FILTER and adx_v < ADX_MIN:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            # ── ENTRY CONDITIONS ──────────────────────────────────
            #
            #  EMA Contraction + Cross logic:
            #   - "contracting"  = gap is shrinking over last N bars
            #   - "fresh cross"  = cross happened within CROSS_CONFIRM_BARS
            #
            #  LONG trigger:
            #    1. Price breaks above 17-candle high  (close break or wick)
            #    2. EMA gap is contracting              (momentum building)
            #    3. EMA9 has crossed above EMA21 within CROSS_CONFIRM_BARS bars
            #       OR still pre-cross but converging
            #
            #  SHORT trigger: mirror image

            # ── LONG ─────────────────────────────────────────────
            if ENABLE_LONG:
                # Breakout condition
                if REQUIRE_CLOSE_BREAK:
                    long_break = close > b_high
                else:
                    long_break = high_c > b_high

                # EMA contraction + cross condition
                # Either: contracting and not yet crossed (ef < es, gap shrinking → imminent bullish cross)
                # Or: fresh bullish cross happened within CROSS_CONFIRM_BARS
                pre_cross_long   = is_cont and (ef < es)            # converging, not yet crossed
                fresh_cross_long = bull_cs <= CROSS_CONFIRM_BARS     # just crossed bullish

                ema_condition_long = pre_cross_long or fresh_cross_long

                # HTF alignment
                htf_ok_long = htf_long if ENABLE_HTF_FILTER else True

                if long_break and ema_condition_long and htf_ok_long:
                    do_enter('long', close, ts, atr)
                    daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

            # ── SHORT ────────────────────────────────────────────
            if not in_trade and ENABLE_SHORT:
                if REQUIRE_CLOSE_BREAK:
                    short_break = close < b_low
                else:
                    short_break = low_c < b_low

                pre_cross_short   = is_cont and (ef > es)           # converging, not yet crossed bearish
                fresh_cross_short = bear_cs <= CROSS_CONFIRM_BARS    # just crossed bearish

                ema_condition_short = pre_cross_short or fresh_cross_short

                htf_ok_short = htf_short if ENABLE_HTF_FILTER else True

                if short_break and ema_condition_short and htf_ok_short:
                    do_enter('short', close, ts, atr)
                    daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

        eq_curve.append({'timestamp': ts, 'equity': equity})

    if in_trade:
        do_exit(closes[-1], ts_list[-1], 'END_OF_DATA')

    return pd.DataFrame(trades), pd.DataFrame(eq_curve)


# ══════════════════════════════════════════════════════════════════
#  METRICS & REPORTING
# ══════════════════════════════════════════════════════════════════

def compute_metrics(tdf: pd.DataFrame) -> dict:
    if tdf.empty: return {'message': 'No trades.'}
    pnl    = tdf['pnl']
    wins   = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    total  = len(tdf)
    gp, gl = wins.sum(), abs(losses.sum())
    cum    = pnl.cumsum()
    max_dd = (cum.cummax() - cum).max()
    return {
        'Total Trades':       total,
        'Winning Trades':     len(wins),
        'Losing Trades':      len(losses),
        'Win Rate (%)':       round(len(wins) / total * 100, 2),
        'Avg Win (pts)':      round(wins.mean(),   2) if len(wins)   else 0,
        'Avg Loss (pts)':     round(losses.mean(), 2) if len(losses) else 0,
        'Largest Win (pts)':  round(pnl.max(), 2),
        'Largest Loss (pts)': round(pnl.min(), 2),
        'Profit Factor':      round(gp / gl, 2) if gl > 0 else float('inf'),
        'Total P&L (pts)':    round(pnl.sum(), 2),
        'Max Drawdown (pts)': round(max_dd, 2),
        'Avg MFE (pts)':      round(tdf['mfe_pts'].mean(), 2),
        'Avg MAE (pts)':      round(tdf['mae_pts'].mean(), 2),
    }


def fmt(ts):
    return ts.strftime('%d-%m-%Y %H:%M') if hasattr(ts, 'strftime') else str(ts)


def print_results(tdf: pd.DataFrame):
    SEP = '─' * 155
    print(f"\n{SEP}")
    print(f"  TRADE LOG  ({len(tdf)} trades)")
    print(SEP)
    print(f"{'#':<5} {'Dir':<6} {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>9} {'Exit':>9} {'SL Init':>9} {'SL Final':>9}"
          f" {'MFE':>7} {'MAE':>7} {'P&L':>9}  {'Reason'}")
    print(SEP)
    for i, r in tdf.iterrows():
        print(f"{i+1:<5} {r['direction'].upper():<6} {fmt(r['entry_time']):<22}"
              f" {fmt(r['exit_time']):<22}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f}"
              f" {r['sl_initial']:>9.2f} {r['sl_final']:>9.2f}"
              f" {r['mfe_pts']:>7.1f} {r['mae_pts']:>7.1f}"
              f" {r['pnl']:>+9.2f}  {r['exit_reason']}")

    m = compute_metrics(tdf)
    print(f"\n{'─'*60}\n  PERFORMANCE METRICS\n{'─'*60}")
    for k, v in m.items():
        print(f"  {k:<30}: {v}")

    print(f"\n{'─'*60}\n  DIRECTION BREAKDOWN\n{'─'*60}")
    for d in ['long', 'short']:
        sub = tdf[tdf['direction'] == d]
        if sub.empty: continue
        p = sub['pnl']; w = (p > 0).sum()
        print(f"  {d.upper():<6}  trades={len(sub)}  wins={w}  wr={w/len(sub)*100:.1f}%"
              f"  total={p.sum():.1f}  avg={p.mean():.1f}"
              f"  avg_mfe={sub['mfe_pts'].mean():.1f}  avg_mae={sub['mae_pts'].mean():.1f}")

    print(f"\n{'─'*60}\n  EXIT REASON BREAKDOWN\n{'─'*60}")
    bd = (tdf.groupby('exit_reason')['pnl']
          .agg(count='count', total_pnl='sum', avg_pnl='mean').round(2))
    print(bd.to_string())

    print(f"\n{'─'*60}\n  MONTHLY P&L\n{'─'*60}")
    tdf2 = tdf.copy()
    tdf2['month'] = tdf2['entry_time'].dt.tz_localize(None).dt.to_period('M').astype(str)
    monthly = (tdf2.groupby('month')['pnl']
               .agg(trades='count', total_pnl='sum',
                    win_trades=lambda x: (x > 0).sum())
               .assign(win_rate=lambda x: (x['win_trades'] / x['trades'] * 100).round(1))
               .round(2))
    print(monthly.to_string())

    print(f"\n{'─'*60}\n  SESSION P&L\n{'─'*60}")
    def classify_session(t):
        if hasattr(t, 'time'): t = t.time()
        h, m = t.hour, t.minute
        if (h == 9 and m >= 30) or h == 10 or (h == 11 and m < 30): return 'Prime'
        if h == 11 or h == 12 or (h == 13 and m < 30):              return 'Midday'
        return 'Euro/Other'
    tdf3 = tdf.copy()
    tdf3['session'] = tdf3['entry_time'].apply(classify_session)
    sess = (tdf3.groupby('session')['pnl']
            .agg(trades='count', total_pnl='sum', avg_pnl='mean',
                 win_trades=lambda x: (x > 0).sum())
            .assign(win_rate=lambda x: (x['win_trades'] / x['trades'] * 100).round(1))
            .round(2))
    print(sess.to_string())

    print(f"\n{'─'*60}\n  BREAKOUT SIGNAL QUALITY (MFE vs MAE)\n{'─'*60}")
    sl_hits = tdf[tdf['exit_reason'] == 'STOP_LOSS']
    if not sl_hits.empty:
        print(f"  SL trades total            : {len(sl_hits)}")
        print(f"  Avg MAE on SL trades       : {sl_hits['mae_pts'].mean():.1f} pts")
        print(f"  Avg MFE on SL trades       : {sl_hits['mfe_pts'].mean():.1f} pts")
        print(f"  SL trades with >5 MFE first: {(sl_hits['mfe_pts'] > 5).sum()}")
    print()


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    BAR = '═' * 68
    print(f"\n{BAR}")
    print("  BREAKOUT + EMA CONTRACTION SCALPER  v1")
    print(f"{BAR}")
    print(f"""
  CORE SIGNAL:
    LONG  → price closes above 17-candle HIGH
            + EMA{EMA_FAST}/EMA{EMA_SLOW} gap contracting (or fresh bullish cross ≤{CROSS_CONFIRM_BARS} bars)

    SHORT → price closes below 17-candle LOW
            + EMA{EMA_FAST}/EMA{EMA_SLOW} gap contracting (or fresh bearish cross ≤{CROSS_CONFIRM_BARS} bars)
""")

    cli_args = sys.argv[1:]
    if cli_args:
        path = cli_args[0]
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}"); sys.exit(1)
        print(f"  Loading CSV: {path}")
        df       = load_csv(path)
        basename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              os.path.splitext(os.path.basename(path))[0] + '_breakout_ema')
    else:
        df       = fetch_yfinance()
        basename = YFINANCE_SYMBOL.replace('^', '').replace('.', '_') + '_breakout_ema'

    print(f"\n  Computing indicators ...")
    df = compute_indicators(df)

    if ENABLE_HTF_FILTER:
        print(f"  Computing 15m HTF bias ...")
        df = compute_htf_bias(df)
    else:
        df['htf_long_bias']  = True
        df['htf_short_bias'] = True

    print(f"\n{'─'*60}")
    print(f"  CONFIG SUMMARY")
    print(f"{'─'*60}")
    print(f"  Breakout candles              : {BREAKOUT_CANDLES}")
    print(f"  EMA Fast / Slow               : EMA{EMA_FAST} / EMA{EMA_SLOW}")
    print(f"  Contraction lookback          : {CONTRACTION_LOOKBACK} bars")
    print(f"  Cross confirm window          : ≤ {CROSS_CONFIRM_BARS} bars after cross")
    print(f"  Close-bar breakout required   : {REQUIRE_CLOSE_BREAK}")
    print(f"  ADX filter                    : {'ON (min ' + str(ADX_MIN) + ')' if ENABLE_ADX_FILTER else 'OFF'}")
    print(f"  HTF 15m filter                : {'ON' if ENABLE_HTF_FILTER else 'OFF'}")
    print(f"  Long SL mult                  : {LONG_ATR_SL_MULT}× ATR")
    print(f"  Short SL mult                 : {SHORT_ATR_SL_MULT}× ATR")
    print(f"  Long BE trigger               : +{LONG_BE_PCT}%")
    print(f"  Short BE trigger              : -{SHORT_BE_PCT}%")
    print(f"  Trail mult (post-BE)          : {TRAIL_ATR_MULT}× ATR")
    print(f"  Max trades/day                : {MAX_TRADES_PER_DAY}")
    print(f"  Consec-loss breaker           : {MAX_CONSEC_LOSSES}")
    print(f"{'─'*60}\n")

    print("  Running backtest ...")
    trades_df, equity_df = run_backtest(df)

    print(f"\n{BAR}")
    print("  BACKTEST RESULTS")
    print(f"{BAR}")

    if trades_df.empty:
        print("\n  No trades generated.")
        print("  Tips: loosen CONTRACTION_LOOKBACK, reduce ADX_MIN, or widen CROSS_CONFIRM_BARS.")
        return

    print_results(trades_df)

    trades_out = basename + '_trades.csv'
    equity_out = basename + '_equity.csv'
    trades_df.to_csv(trades_out, index=False)
    equity_df.to_csv(equity_out, index=False)
    print(f"  Trades saved : {trades_out}")
    print(f"  Equity curve : {equity_out}")
    print(f"{BAR}\n")


if __name__ == '__main__':
    main()