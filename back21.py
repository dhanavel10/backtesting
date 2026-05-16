
import pandas as pd
import numpy as np
import sys, os, json, webbrowser
from datetime import time

# ══════════════════════════════════════════════════════════════════
#   CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DATA_MODE        = 'yfinance'
YFINANCE_SYMBOL  = '^NSEI'
YFINANCE_DAYS    = 60
YFINANCE_TZ      = 'Asia/Kolkata'
CSV_5MIN_PATH    = 'nifty_5m.csv'

EMA_FAST         = 9
EMA_SLOW         = 21
EMA_MACRO        = 50

HTF_EMA_FAST     = 12
HTF_EMA_SLOW     = 26
HTF_EMA_TREND    = 50

ATR_PERIOD       = 14

ADX_PERIOD       = 14
ADX_MIN          = 18

LONG_ATR_SL_MULT    = 1.2
SHORT_ATR_SL_MULT   = 0.75

LONG_BE_PCT         = 0.2
SHORT_BE_PCT        = 0.3

# ── Swing Structure Trail (v4) ────────────────────────────────────
# How many bars to look back when finding the most recent swing
# high (for shorts) or swing low (for longs).
# A "confirmed" swing low  = the lowest low over SWING_LOOKBACK bars.
# A "confirmed" swing high = the highest high over SWING_LOOKBACK bars.
# Buffer is added below (longs) / above (shorts) so a 1-tick poke
# doesn't trigger an exit.
SWING_LOOKBACK      = 5     # bars to define a swing point
SWING_BUFFER_PTS    = 5     # extra pts of buffer beyond the swing level

# NOTE: TRAIL_ATR_MULT is kept in config but is no longer used for
# the trailing stop. It is retained so v3 HTML/CSV outputs remain
# comparable and the param does not disappear silently.
TRAIL_ATR_MULT      = 0.6

ENABLE_EMA_CROSS_EXIT = True

ENABLE_SLOPE_EXIT     = True
SLOPE_EXIT_CANDLES    = 4

SHORT_CONFIRM_BARS  = 3

SPREAD_PCT_MIN      = 0.04
SLOPE_CANDLES       = 6
SLOPE_MIN           = 0.0
PRICE_GAP_MIN       = 5.0
MIDDAY_SPREAD_MULT  = 2.0

RETEST_ATR_MULT     = 0.25

EMA9_PULLAWAY_PTS   = 50

MAX_TRADES_PER_DAY  = 3

MAX_CONSEC_LOSSES   = 2

ENABLE_LONG         = True
ENABLE_SHORT        = True

# ── Path Enable/Disable Flags (NEW in v4) ─────────────────────────
ENABLE_PATH_A       = False   # Path A: EMA9 Retest
ENABLE_PATH_B       = True   # Path B: EMA9 Pullaway 50pts

ENABLE_MIDDAY       = True
ENABLE_EURO         = False

OBSERVE_START       = time(9,  15)
OBSERVE_END         = time(9,  30)
PRIME_START         = time(9,  30)
PRIME_END           = time(10, 30)
MIDDAY_START        = time(11, 30)
MIDDAY_END          = time(13, 30)
EURO_START          = time(14, 15)
EURO_END            = time(15,  0)
SQUAREOFF_START     = time(15,  0)

EOD_HARD_EXIT       = time(15,  0)   # all trades squared off at 15:00, no exceptions


# ══════════════════════════════════════════════════════════════════
#   INDICATOR ENGINE  (unchanged from v3)
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
    return dx.ewm(alpha=1/period, adjust=False).mean()


def compute_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_fast']            = compute_ema(df['close'], EMA_FAST)
    df['ema_slow']            = compute_ema(df['close'], EMA_SLOW)
    df['ema_macro']           = compute_ema(df['close'], EMA_MACRO)
    df['atr']                 = compute_atr(df, ATR_PERIOD)
    df['adx']                 = compute_adx(df, ADX_PERIOD)
    df['ema_slow_slope']      = df['ema_slow'].diff(SLOPE_CANDLES)
    df['ema_slow_slope_exit'] = df['ema_slow'].diff(SLOPE_EXIT_CANDLES)
    bearish_cross = (df['ema_fast'] < df['ema_slow']).astype(int)
    df['consec_bearish_bars'] = bearish_cross.groupby(
        (bearish_cross != bearish_cross.shift()).cumsum()
    ).cumcount() + 1
    df['consec_bearish_bars'] = df['consec_bearish_bars'] * bearish_cross
    return df


def compute_htf_bias(df_5m: pd.DataFrame) -> pd.DataFrame:
    df_5m = df_5m.copy().set_index('timestamp')
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
#   SWING STRUCTURE HELPERS  (new in v4)
# ══════════════════════════════════════════════════════════════════

def get_swing_low(lows: np.ndarray, idx: int, lookback: int) -> float:
    """
    Return the lowest low over the last `lookback` bars ending at idx-1
    (we exclude the current bar so the swing is 'confirmed' by prior price).
    Falls back to lows[idx] if there isn't enough history yet.
    """
    start = max(0, idx - lookback)
    window = lows[start:idx]          # excludes current bar
    if len(window) == 0:
        return float(lows[idx])
    return float(np.min(window))


def get_swing_high(highs: np.ndarray, idx: int, lookback: int) -> float:
    """
    Return the highest high over the last `lookback` bars ending at idx-1.
    Falls back to highs[idx] if there isn't enough history yet.
    """
    start = max(0, idx - lookback)
    window = highs[start:idx]
    if len(window) == 0:
        return float(highs[idx])
    return float(np.max(window))


# ══════════════════════════════════════════════════════════════════
#   ANTI-CHOP + REGIME FILTERS  (unchanged from v3)
# ══════════════════════════════════════════════════════════════════

def get_session(t: time) -> str:
    if OBSERVE_START <= t < OBSERVE_END:  return 'observe'
    if PRIME_START   <= t < PRIME_END:    return 'prime'
    if MIDDAY_START  <= t < MIDDAY_END:   return 'midday'
    if EURO_START    <= t < EURO_END:     return 'euro'
    if SQUAREOFF_START <= t:              return 'squareoff'
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
#   DATA LOADERS  (unchanged from v3)
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

    opening_high     = {}
    opening_close    = {}

    in_trade         = False
    direction        = None
    entry_price      = 0.0
    entry_time       = None
    entry_idx        = -1
    entry_path       = ''
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

    def do_enter(dir_str, close_price, ts_now, atr_val, idx, path):
        nonlocal in_trade, direction, entry_price, entry_time, entry_idx, entry_path
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, trade_max_favor, trade_max_adverse

        sl_mult  = LONG_ATR_SL_MULT if dir_str == 'long' else SHORT_ATR_SL_MULT
        sl_dist  = atr_val * sl_mult

        in_trade          = True
        direction         = dir_str
        entry_price       = close_price
        entry_time        = ts_now
        entry_idx         = idx
        entry_path        = path
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

    def do_exit(exit_price, ts_now, reason, exit_idx):
        nonlocal in_trade, direction, entry_price, entry_time, entry_idx, entry_path
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, equity
        nonlocal trade_max_favor, trade_max_adverse

        pnl = round(
            (exit_price - entry_price) if direction == 'long'
            else (entry_price - exit_price), 2
        )
        equity += pnl

        date_str = str(ts_now.date())
        if pnl <= 0:
            daily_consec_loss[date_str] = daily_consec_loss.get(date_str, 0) + 1
        else:
            daily_consec_loss[date_str] = daily_consec_loss.get(date_str, 0)
            # Do NOT reset — any loss in the day keeps the block active

        trades.append({
            'direction':         direction,
            'entry_path':        entry_path,
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
            'entry_idx':         entry_idx,
            'exit_idx':          exit_idx,
        })

        in_trade          = False
        direction         = None
        entry_price       = 0.0
        entry_time        = None
        entry_idx         = -1
        entry_path        = ''
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

        if c_time == time(9, 30):
            opening_high[date_str]  = high_c
            opening_close[date_str] = close

        if np.isnan(ef) or np.isnan(es) or np.isnan(em) or np.isnan(atr):
            eq_curve.append({'timestamp': ts, 'equity': equity})
            continue

        htf_long = bool(htf_long_bias[idx]) if not pd.isna(htf_long_bias[idx]) else False

        # ══════════════════════════════════════════════════════════
        #   MANAGE OPEN TRADE — Dynamic Exit Engine (v4)
        # ══════════════════════════════════════════════════════════
        if in_trade:
            if direction == 'long':
                favor   = high_c - entry_price
                adverse = entry_price - low_c
            else:
                favor   = entry_price - low_c
                adverse = high_c - entry_price
            trade_max_favor   = max(trade_max_favor,   favor)
            trade_max_adverse = max(trade_max_adverse, adverse)

            # ── Break-even trigger (same as v3) ──────────────────
            if not be_triggered:
                if direction == 'long'  and close >= be_level:
                    stop_loss    = entry_price + 1.0
                    be_triggered = True
                    trail_active = True
                elif direction == 'short' and close <= be_level:
                    stop_loss    = entry_price - 1.0
                    be_triggered = True
                    trail_active = True

            # ── Trailing Stop: Swing Structure (v4 change) ────────
            # Once trail is active, anchor the SL to the most recent
            # confirmed swing low (longs) or swing high (shorts).
            # This replaces the fixed ATR-multiplier trail from v3.
            if trail_active:
                if direction == 'long':
                    swing_sl = get_swing_low(lows, idx, SWING_LOOKBACK) - SWING_BUFFER_PTS
                    # SL can only move UP (ratchet), never backwards
                    stop_loss = max(stop_loss, swing_sl)
                else:
                    swing_sl = get_swing_high(highs, idx, SWING_LOOKBACK) + SWING_BUFFER_PTS
                    # SL can only move DOWN (ratchet), never backwards
                    stop_loss = min(stop_loss, swing_sl)

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
            elif c_time >= EOD_HARD_EXIT:
                # Intraday strategy — close ALL open trades at EOD, no exceptions.
                # EOD_SOFT_EXIT is intentionally removed; profitable trades must
                # also be squared off before market close.
                exit_p = close
                exit_r = 'EOD_EXIT'

            if exit_p is not None:
                do_exit(exit_p, ts, exit_r, idx)

        # ══════════════════════════════════════════════════════════
        #   ENTRY LOGIC  (with Path A/B enable flags)
        # ══════════════════════════════════════════════════════════
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

            if daily_consec_loss.get(date_str, 0) >= 1:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            if not chop_filters_pass(close, ef, es, slope, adx_v, session):
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            retest_tol = atr * RETEST_ATR_MULT

            # ── LONG Entry ────────────────────────────────────────
            if ENABLE_LONG and htf_long:
                # Path A: EMA9 Retest
                if ENABLE_PATH_A:
                    path_a_long = (
                        ef > es                         and
                        close > em                      and
                        slope > 0                       and
                        abs(close - ef) <= retest_tol   and
                        close > es
                    )
                    if path_a_long:
                        do_enter('long', close, ts, atr, idx, 'A')
                        daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

                # Path B: EMA9 Pullaway
                if not in_trade and ENABLE_PATH_B:
                    path_b_long = (
                        ef > es                           and
                        close > em                        and
                        slope > 0                         and
                        (close - ef) >= EMA9_PULLAWAY_PTS and
                        date_str in opening_high          and
                        close > opening_high[date_str]
                    )
                    if path_b_long:
                        do_enter('long', close, ts, atr, idx, 'B')
                        daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

            # ── SHORT Entry ───────────────────────────────────────
            if not in_trade and ENABLE_SHORT:
                # Path A: EMA9 Retest
                if ENABLE_PATH_A:
                    path_a_short = (
                        ef < es                         and
                        close < em                      and
                        slope < 0                       and
                        abs(close - ef) <= retest_tol   and
                        close < es                      and
                        cb >= SHORT_CONFIRM_BARS
                    )
                    if path_a_short:
                        do_enter('short', close, ts, atr, idx, 'A')
                        daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

                # Path B: EMA9 Pullaway
                if not in_trade and ENABLE_PATH_B:
                    path_b_short = (
                        ef < es                           and
                        close < em                        and
                        slope < 0                         and
                        (ef - close) >= EMA9_PULLAWAY_PTS and
                        cb >= SHORT_CONFIRM_BARS           and
                        date_str in opening_close         and
                        close < opening_close[date_str]
                    )
                    if path_b_short:
                        do_enter('short', close, ts, atr, idx, 'B')
                        daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

        eq_curve.append({'timestamp': ts, 'equity': equity})

    if in_trade:
        do_exit(closes[-1], ts_list[-1], 'END_OF_DATA', n - 1)

    return pd.DataFrame(trades), pd.DataFrame(eq_curve)


# ══════════════════════════════════════════════════════════════════
#   METRICS & REPORTING  (unchanged from v3)
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
        'Avg MFE (pts)':       round(tdf['mfe_pts'].mean(), 2),
        'Avg MAE (pts)':       round(tdf['mae_pts'].mean(), 2),
    }


def fmt_ts(ts):
    return ts.strftime('%d-%m-%Y %H:%M') if hasattr(ts, 'strftime') else str(ts)


def print_results(tdf: pd.DataFrame):
    SEP = '─' * 162

    print(f"\n{SEP}")
    print(f"  TRADE LOG  ({len(tdf)} trades)")
    print(SEP)
    print(f"{'#':<5} {'Dir':<6} {'Path':<6} {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>9} {'Exit':>9} {'SL@Entry':>10} {'SL@Exit':>9}"
          f" {'MFE':>7} {'MAE':>7} {'P&L':>9}  Reason")
    print(SEP)
    for i, r in tdf.iterrows():
        print(f"{i+1:<5} {r['direction'].upper():<6} {'Path-'+r['entry_path']:<6}"
              f" {fmt_ts(r['entry_time']):<22} {fmt_ts(r['exit_time']):<22}"
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

    print(f"\n{'─'*62}")
    print("  ENTRY PATH BREAKDOWN  (A = EMA9 Retest  |  B = EMA9 Pullaway 50pts)")
    print(f"{'─'*62}")
    for path in ['A', 'B']:
        sub = tdf[tdf['entry_path'] == path]
        if sub.empty:
            status = 'DISABLED' if (path == 'A' and not ENABLE_PATH_A) or (path == 'B' and not ENABLE_PATH_B) else 'no trades'
            print(f"  Path {path} : {status}")
            continue
        p        = sub['pnl']
        wins     = p[p > 0]
        losses   = p[p <= 0]
        cum_win  = wins.sum()
        cum_loss = losses.sum()
        print(f"  Path {path} : trades={len(sub)}  wins={len(wins)}  losses={len(losses)}  "
              f"wr={len(wins)/len(sub)*100:.1f}%  "
              f"cum-profit={cum_win:+.1f} pts  cum-loss={cum_loss:+.1f} pts  "
              f"net={p.sum():+.1f} pts")
        for d in ['long', 'short']:
            dsub = sub[sub['direction'] == d]
            if dsub.empty:
                continue
            dp = dsub['pnl']
            dw = dp[dp > 0]
            dl = dp[dp <= 0]
            print(f"          {d.upper():<6}  trades={len(dsub)}  wins={len(dw)}  "
                  f"cum-profit={dw.sum():+.1f}  cum-loss={dl.sum():+.1f}  "
                  f"net={dp.sum():+.1f}  avg={dp.mean():.1f}")
    print()

    # ── v4 settings summary ───────────────────────────────────────
    print(f"{'─'*62}")
    print("  v4 SWING TRAIL SETTINGS")
    print(f"{'─'*62}")
    print(f"  Swing Lookback  : {SWING_LOOKBACK} bars")
    print(f"  Swing Buffer    : {SWING_BUFFER_PTS} pts")
    print(f"  Path A (Retest) : {'ENABLED' if ENABLE_PATH_A else 'DISABLED'}")
    print(f"  Path B (Pullaway): {'ENABLED' if ENABLE_PATH_B else 'DISABLED'}")
    print(f"  EOD Square-Off  : {EOD_HARD_EXIT}  (all trades, no exceptions)")
    print(f"  (ATR trail from v3 is disabled — swing structure is used instead)")
    print()


# ══════════════════════════════════════════════════════════════════
#   HTML REPORT WITH INTERACTIVE TRADE CHART  (unchanged from v3)
# ══════════════════════════════════════════════════════════════════

def build_html_report(trades_df: pd.DataFrame, raw_df: pd.DataFrame) -> str:
    CONTEXT = 10

    def row_to_dict(r):
        ei    = int(r['entry_idx'])
        xi    = int(r['exit_idx'])
        start = max(0, ei - CONTEXT)
        end   = min(len(raw_df), xi + CONTEXT + 1)
        chunk = raw_df.iloc[start:end][
            ['timestamp', 'open', 'high', 'low', 'close', 'ema_fast', 'ema_slow', 'ema_macro']
        ].copy()
        chunk['timestamp'] = chunk['timestamp'].dt.strftime('%H:%M %d%b')
        return {
            'direction':   r['direction'],
            'entry_path':  r['entry_path'],
            'entry_time':  fmt_ts(r['entry_time']),
            'exit_time':   fmt_ts(r['exit_time']),
            'entry_price': round(r['entry_price'], 2),
            'exit_price':  round(r['exit_price'],  2),
            'sl_at_entry': round(r['sl_at_entry'], 2),
            'sl_at_exit':  round(r['sl_at_exit'],  2),
            'be_triggered':bool(r['be_triggered']),
            'pnl':         round(r['pnl'],      2),
            'mfe':         round(r['mfe_pts'],  2),
            'mae':         round(r['mae_pts'],  2),
            'exit_reason': r['exit_reason'],
            'candles':     chunk.to_dict(orient='records'),
            'rel_entry':   ei - start,
            'rel_exit':    xi - start,
        }

    trade_data   = [row_to_dict(r) for _, r in trades_df.iterrows()]
    trade_json   = json.dumps(trade_data)
    metrics      = compute_metrics(trades_df)
    metrics_rows = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in metrics.items())

    path_summary = {}
    for path in ['A', 'B']:
        sub = trades_df[trades_df['entry_path'] == path]
        if sub.empty:
            path_summary[path] = {'trades': 0, 'net': 0, 'cum_profit': 0, 'cum_loss': 0}
        else:
            p = sub['pnl']
            path_summary[path] = {
                'trades':     len(sub),
                'net':        round(p.sum(), 1),
                'cum_profit': round(p[p > 0].sum(), 1),
                'cum_loss':   round(p[p <= 0].sum(), 1),
            }

    table_rows = ''
    for i, t in enumerate(trade_data):
        pnl_cls  = 'win' if t['pnl'] > 0 else 'loss'
        dir_cls  = 'long' if t['direction'] == 'long' else 'short'
        path_cls = 'path-a' if t['entry_path'] == 'A' else 'path-b'
        sign     = '+' if t['pnl'] > 0 else ''
        table_rows += f"""
        <tr onclick="showChart({i})" class="trade-row" id="row-{i}">
          <td>{i+1}</td>
          <td class="{dir_cls}">{'▲ LONG' if t['direction']=='long' else '▼ SHORT'}</td>
          <td><span class="badge {path_cls}">Path {t['entry_path']}</span></td>
          <td>{t['entry_time']}</td>
          <td>{t['exit_time']}</td>
          <td>{t['entry_price']:.2f}</td>
          <td>{t['exit_price']:.2f}</td>
          <td class="mfe">+{t['mfe']:.1f}</td>
          <td class="mae">-{t['mae']:.1f}</td>
          <td class="{pnl_cls}">{sign}{t['pnl']:.2f}</td>
          <td><span class="badge">{t['exit_reason']}</span></td>
        </tr>"""

    total_pnl = metrics.get('Total P&L (pts)', 0)
    pnl_color = 'green' if total_pnl >= 0 else 'red'
    ps = path_summary

    path_a_status = 'ENABLED' if ENABLE_PATH_A else 'DISABLED'
    path_b_status = 'ENABLED' if ENABLE_PATH_B else 'DISABLED'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Speed Demon v4 – Trade Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#080b12;color:#e2e8f0;font-family:'Courier New',monospace;font-size:12px}}
  header{{background:#0d1117;border-bottom:1px solid #1e2336;padding:10px 20px;
          display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
  header h1{{font-size:13px;color:#f1f5f9;letter-spacing:1px}}
  header .sub{{font-size:9px;color:#475569;margin-top:2px}}
  .kpis{{display:flex;gap:20px;margin-left:auto;flex-wrap:wrap}}
  .kpi{{text-align:right}}
  .kpi .label{{font-size:9px;color:#475569;text-transform:uppercase}}
  .kpi .val{{font-size:14px;font-weight:700}}
  .green{{color:#4ade80}}.red{{color:#f87171}}
  .path-bar{{background:#0d1117;border-bottom:1px solid #1e2336;
             padding:8px 20px;display:flex;gap:32px}}
  .path-card{{display:flex;flex-direction:column;gap:2px}}
  .path-card .ph{{font-size:10px;font-weight:700;letter-spacing:1px}}
  .path-card.pa .ph{{color:#f59e0b}}.path-card.pb .ph{{color:#60a5fa}}
  .path-card .prow{{font-size:10px;color:#94a3b8}}
  .path-card .prow span{{font-weight:700}}
  .profit-val{{color:#4ade80}}.loss-val{{color:#f87171}}.net-val{{color:#e2e8f0}}
  .status-badge{{padding:2px 6px;border-radius:3px;font-size:8px;margin-left:6px}}
  .status-enabled{{background:#4ade8022;color:#4ade80;border:1px solid #4ade8044}}
  .status-disabled{{background:#f8717122;color:#f87171;border:1px solid #f8717144}}
  .main{{display:flex;height:calc(100vh - 98px)}}
  .left{{width:56%;overflow-y:auto;border-right:1px solid #1e2336}}
  .right{{flex:1;overflow-y:auto;padding:14px;display:none;flex-direction:column;gap:10px}}
  .right.visible{{display:flex}}
  table{{width:100%;border-collapse:collapse}}
  thead th{{background:#0d1117;color:#475569;font-size:9px;text-transform:uppercase;
            padding:5px 7px;text-align:left;position:sticky;top:0;z-index:1;
            border-bottom:1px solid #1e2336}}
  .trade-row{{cursor:pointer;border-bottom:1px solid #0f1520}}
  .trade-row:nth-child(even){{background:#0a0d14}}
  .trade-row:hover,.trade-row.active{{background:#131929!important;
    border-left:3px solid #3b82f6}}
  td{{padding:5px 7px}}
  .long{{color:#4ade80;font-weight:700}}.short{{color:#f87171;font-weight:700}}
  .win{{color:#4ade80;font-weight:700}}.loss{{color:#f87171;font-weight:700}}
  .mfe{{color:#a3e635}}.mae{{color:#fb923c}}
  .badge{{background:#1e2336;color:#94a3b8;padding:2px 5px;border-radius:3px;font-size:9px}}
  .badge.path-a{{background:#f59e0b22;color:#f59e0b;border:1px solid #f59e0b44}}
  .badge.path-b{{background:#60a5fa22;color:#60a5fa;border:1px solid #60a5fa44}}
  canvas{{display:block;width:100%;background:#0f1117;border-radius:8px}}
  .chart-title{{font-size:12px;font-weight:700;color:#f1f5f9;
               display:flex;align-items:center;gap:8px}}
  .close-btn{{margin-left:auto;background:#1e2336;border:none;color:#94a3b8;
              padding:4px 10px;border-radius:4px;cursor:pointer;font-family:inherit}}
  .box{{background:#0d1117;border:1px solid #1e2336;border-radius:8px;padding:10px}}
  .box h3{{font-size:9px;color:#475569;text-transform:uppercase;
           letter-spacing:1px;margin-bottom:8px}}
  .dg{{display:grid;grid-template-columns:1fr 1fr;gap:4px 16px}}
  .dg .row{{display:flex;justify-content:space-between;
            border-bottom:1px solid #1e2336;padding-bottom:3px}}
  .dg .k{{color:#475569}}.dg .v{{color:#cbd5e1;font-weight:600}}
  .box table td{{padding:3px 5px}}
  .box table tr:nth-child(even){{background:#131929}}
  /* v4 badge */
  .v4tag{{background:#7c3aed22;color:#a78bfa;border:1px solid #7c3aed55;
          padding:2px 7px;border-radius:3px;font-size:9px;margin-left:8px}}
</style>
</head>
<body>
<header>
  <div>
    <h1>SPEED DEMON SCALPER v4 — Trade Report <span class="v4tag">SWING TRAIL</span></h1>
    <div class="sub">Swing structure trailing stop · lookback={SWING_LOOKBACK} bars · buffer={SWING_BUFFER_PTS} pts · EOD square-off {EOD_HARD_EXIT} · Click any row for chart</div>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="label">Trades</div>
      <div class="val">{metrics.get('Total Trades',0)}</div></div>
    <div class="kpi"><div class="label">Win Rate</div>
      <div class="val green">{metrics.get('Win Rate (%)',0)}%</div></div>
    <div class="kpi"><div class="label">Profit Factor</div>
      <div class="val">{metrics.get('Profit Factor',0)}</div></div>
    <div class="kpi"><div class="label">Total P&L</div>
      <div class="val {pnl_color}">{total_pnl:+.1f} pts</div></div>
  </div>
</header>

<div class="path-bar">
  <div class="path-card pa">
    <div class="ph">PATH A — EMA9 Retest 
      <span class="status-badge status-{'enabled' if ENABLE_PATH_A else 'disabled'}">{path_a_status}</span>
    </div>
    <div class="prow">Trades: <span>{ps['A']['trades']}</span> &nbsp;|&nbsp;
      Cum Profit: <span class="profit-val">{ps['A']['cum_profit']:+.1f}</span> &nbsp;|&nbsp;
      Cum Loss: <span class="loss-val">{ps['A']['cum_loss']:+.1f}</span> &nbsp;|&nbsp;
      Net: <span class="net-val">{ps['A']['net']:+.1f} pts</span>
    </div>
  </div>
  <div class="path-card pb">
    <div class="ph">PATH B — EMA9 Pullaway 50pts
      <span class="status-badge status-{'enabled' if ENABLE_PATH_B else 'disabled'}">{path_b_status}</span>
    </div>
    <div class="prow">Trades: <span>{ps['B']['trades']}</span> &nbsp;|&nbsp;
      Cum Profit: <span class="profit-val">{ps['B']['cum_profit']:+.1f}</span> &nbsp;|&nbsp;
      Cum Loss: <span class="loss-val">{ps['B']['cum_loss']:+.1f}</span> &nbsp;|&nbsp;
      Net: <span class="net-val">{ps['B']['net']:+.1f} pts</span>
    </div>
  </div>
</div>

<div class="main">
  <div class="left">
    <table>
      <thead><tr>
        <th>#</th><th>Dir</th><th>Path</th><th>Entry Time</th><th>Exit Time</th>
        <th>Entry</th><th>Exit</th><th>MFE</th><th>MAE</th><th>P&L</th><th>Reason</th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>
  <div class="right" id="right-panel">
    <div class="chart-title">
      <span id="chart-label">Trade Chart</span>
      <button class="close-btn" onclick="closeChart()">✕ Close</button>
    </div>
    <canvas id="tradeCanvas" height="260"></canvas>
    <div class="box">
      <h3>Trade Details</h3>
      <div class="dg" id="details-grid"></div>
    </div>
    <div class="box">
      <h3>Performance Metrics</h3>
      <table>{metrics_rows}</table>
    </div>
  </div>
</div>
<script>
const TRADES={trade_json};
const EMA_FAST={EMA_FAST},EMA_SLOW={EMA_SLOW},EMA_MACRO={EMA_MACRO};
let activeIdx=null;

function showChart(idx){{
  if(activeIdx===idx){{closeChart();return;}}
  activeIdx=idx;
  document.querySelectorAll('.trade-row').forEach((r,i)=>r.classList.toggle('active',i===idx));
  document.getElementById('right-panel').classList.add('visible');
  const t=TRADES[idx];
  document.getElementById('chart-label').textContent=
    `Trade #${{idx+1}}  ·  ${{t.direction.toUpperCase()}}  ·  Path ${{t.entry_path}}  ·  ${{t.entry_time}} → ${{t.exit_time}}`;
  drawChart(t);
  const dg=document.getElementById('details-grid');
  const rows=[
    ['Entry Price',  t.entry_price.toFixed(2)],
    ['Exit Price',   t.exit_price.toFixed(2)],
    ['Entry Path',   'Path ' + t.entry_path + (t.entry_path==='A'?' (Retest)':' (Pullaway)')],
    ['P&L',          (t.pnl>0?'+':'')+t.pnl.toFixed(2)+' pts'],
    ['MFE',          '+'+t.mfe.toFixed(2)+' pts'],
    ['MAE',          '-'+t.mae.toFixed(2)+' pts'],
    ['SL at Entry',  t.sl_at_entry.toFixed(2)],
    ['SL at Exit',   t.sl_at_exit.toFixed(2)],
    ['Break-Even',   t.be_triggered?'YES ✓':'NO'],
    ['Exit Reason',  t.exit_reason],
    ['Entry Time',   t.entry_time],
    ['Exit Time',    t.exit_time],
  ];
  dg.innerHTML=rows.map(([k,v])=>
    `<div class="row"><span class="k">${{k}}</span><span class="v">${{v}}</span></div>`
  ).join('');
}}

function closeChart(){{
  activeIdx=null;
  document.getElementById('right-panel').classList.remove('visible');
  document.querySelectorAll('.trade-row').forEach(r=>r.classList.remove('active'));
}}

function drawChart(t){{
  const canvas=document.getElementById('tradeCanvas');
  const ctx=canvas.getContext('2d');
  const DPR=window.devicePixelRatio||1;
  const W=canvas.offsetWidth||620,CH=260;
  canvas.width=W*DPR; canvas.height=CH*DPR;
  canvas.style.height=CH+'px';
  ctx.scale(DPR,DPR);
  ctx.fillStyle='#0f1117'; ctx.fillRect(0,0,W,CH);

  const cs=t.candles,n=cs.length;
  if(!n)return;

  const PL=58,PR=12,PT=14,PB=28;
  const cW=W-PL-PR,cH=CH-PT-PB;

  const allP=cs.flatMap(c=>[c.high,c.low,
    isNaN(c.ema_fast)?c.close:c.ema_fast,
    isNaN(c.ema_slow)?c.close:c.ema_slow]);
  allP.push(t.sl_at_entry,t.sl_at_exit,t.entry_price,t.exit_price);
  const pMin=Math.min(...allP)-2,pMax=Math.max(...allP)+2;

  const xS=i=>PL+(i+0.5)*(cW/n);
  const yS=p=>PT+cH-((p-pMin)/(pMax-pMin))*cH;
  const bw=Math.max(3,(cW/n)*0.55);

  ctx.font='9px monospace'; ctx.textAlign='right';
  for(let g=0;g<=5;g++){{
    const f=g/5,y=PT+cH*(1-f),p=pMin+f*(pMax-pMin);
    ctx.strokeStyle='#1e2336'; ctx.lineWidth=1; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(PL,y); ctx.lineTo(W-PR,y); ctx.stroke();
    ctx.fillStyle='#4a5568'; ctx.fillText(p.toFixed(0),PL-3,y+4);
  }}

  [t.rel_entry,t.rel_exit].forEach((ri,k)=>{{
    ctx.strokeStyle=k===0?'#4ade8055':'#f8717155';
    ctx.lineWidth=1.5; ctx.setLineDash([4,2]);
    ctx.beginPath(); ctx.moveTo(xS(ri),PT); ctx.lineTo(xS(ri),PT+cH); ctx.stroke();
    ctx.setLineDash([]);
  }});

  const hLine=(p,col,lbl)=>{{
    const y=yS(p);
    ctx.strokeStyle=col; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(PL,y); ctx.lineTo(W-PR,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=col; ctx.font='8px monospace'; ctx.textAlign='left';
    ctx.fillText(lbl,W-PR+2,y+3); ctx.textAlign='right';
  }};
  hLine(t.sl_at_entry,'#ef444488','SL');
  hLine(t.entry_price,'#4ade8033','EP');

  const drawEMA=(key,col)=>{{
    ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.setLineDash([]);
    ctx.beginPath(); let s=false;
    cs.forEach((c,i)=>{{
      const v=c[key]; if(v==null||isNaN(v))return;
      const x=xS(i),y=yS(v);
      if(!s){{ctx.moveTo(x,y);s=true;}}else ctx.lineTo(x,y);
    }});
    ctx.stroke();
  }};
  drawEMA('ema_fast','#f59e0b');
  drawEMA('ema_slow','#60a5fa');
  drawEMA('ema_macro','#a78bfa');

  cs.forEach((c,i)=>{{
    const x=xS(i),bull=c.close>=c.open,col=bull?'#22c55e':'#ef4444';
    ctx.strokeStyle=col; ctx.lineWidth=1; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(x,yS(c.high)); ctx.lineTo(x,yS(c.low)); ctx.stroke();
    const by=yS(Math.max(c.open,c.close)),bh=Math.max(1,Math.abs(yS(c.open)-yS(c.close)));
    ctx.fillStyle=col; ctx.fillRect(x-bw/2,by,bw,bh);
  }});

  [[t.rel_entry,t.entry_price,'#4ade80','E'],[t.rel_exit,t.exit_price,'#f87171','X']]
    .forEach(([ri,price,col,lbl])=>{{
      const x=xS(ri),y=yS(price);
      ctx.fillStyle=col;
      ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2); ctx.fill();
      ctx.fillStyle=col; ctx.font='bold 9px monospace'; ctx.textAlign='center';
      ctx.fillText(`${{lbl}} ${{price.toFixed(0)}}`,x,y+16);
    }});

  const step=Math.max(1,Math.floor(n/6));
  ctx.fillStyle='#4a5568'; ctx.font='8px monospace'; ctx.textAlign='center';
  cs.forEach((c,i)=>{{ if(i%step!==0)return; ctx.fillText(c.timestamp,xS(i),CH-4); }});

  const leg=[['EMA'+EMA_FAST,'#f59e0b'],['EMA'+EMA_SLOW,'#60a5fa'],
             ['EMA'+EMA_MACRO,'#a78bfa'],['Entry','#4ade80'],['Exit','#f87171']];
  let lx=PL+4;
  leg.forEach(([lbl,col])=>{{
    ctx.fillStyle=col; ctx.fillRect(lx,PT+4,14,2);
    ctx.fillStyle='#94a3b8'; ctx.font='8px monospace'; ctx.textAlign='left';
    ctx.fillText(lbl,lx+17,PT+10); lx+=58;
  }});
}}

window.addEventListener('resize',()=>{{ if(activeIdx!==null)drawChart(TRADES[activeIdx]); }});
</script>
</body>
</html>"""


def save_and_open_report(trades_df: pd.DataFrame, raw_df: pd.DataFrame, basename: str):
    html        = build_html_report(trades_df, raw_df)
    report_path = basename + '_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML report  : {report_path}")
    try:
        webbrowser.open('file://' + os.path.abspath(report_path))
        print(f"  (Opened in your default browser)")
    except Exception:
        print(f"  (Open manually: {os.path.abspath(report_path)})")


# ══════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    BAR = '═' * 68
    print(f"\n{BAR}")
    print("  SPEED DEMON SCALPER  v4  — SWING STRUCTURE TRAIL")
    print(f"{BAR}")

    cli_args = sys.argv[1:]
    mode     = 'csv' if cli_args else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\nData source : yfinance  |  Symbol : {YFINANCE_SYMBOL}")
        df        = fetch_yfinance('5m', '5m')
        save_base = os.path.join(os.getcwd(),
                                 YFINANCE_SYMBOL.replace('^', '').replace('.', '_') + '_speed_demon_v4')
    elif mode == 'csv':
        path5 = cli_args[0] if cli_args else CSV_5MIN_PATH
        if not os.path.exists(path5):
            print(f"ERROR: File not found: {path5}"); sys.exit(1)
        print(f"\nLoading 5m CSV : {path5}")
        df        = load_csv(path5, '5m')
        save_base = os.path.splitext(path5)[0] + '_speed_demon_v4'
    else:
        print(f"ERROR: Unknown DATA_MODE '{DATA_MODE}'"); sys.exit(1)

    print(f"\n{'─'*62}")
    print("  STRATEGY CONFIGURATION  (v4)")
    print(f"{'─'*62}")
    print(f"  EMA Fast / Slow / Macro (5m)    : {EMA_FAST} / {EMA_SLOW} / {EMA_MACRO}")
    print(f"  HTF EMAs (15m)                  : {HTF_EMA_FAST} / {HTF_EMA_SLOW} / {HTF_EMA_TREND}")
    print(f"  ATR Period                       : {ATR_PERIOD}")
    print(f"  ADX Filter                       : ADX({ADX_PERIOD}) >= {ADX_MIN}")
    print(f"  LONG  SL Multiplier             : {LONG_ATR_SL_MULT}× ATR")
    print(f"  SHORT SL Multiplier             : {SHORT_ATR_SL_MULT}× ATR")
    print(f"  LONG  Break-Even Trigger        : +{LONG_BE_PCT}%")
    print(f"  SHORT Break-Even Trigger        : -{SHORT_BE_PCT}%")
    print(f"  Trailing Stop (post-BE)         : SWING STRUCTURE  ← v4 change")
    print(f"    Swing Lookback                : {SWING_LOOKBACK} bars")
    print(f"    Swing Buffer                  : {SWING_BUFFER_PTS} pts beyond swing level")
    print(f"  EMA Cross Exit                  : {'ON' if ENABLE_EMA_CROSS_EXIT else 'OFF'}")
    print(f"  Slope Reversal Exit             : {'ON' if ENABLE_SLOPE_EXIT else 'OFF'}")
    print(f"  Short Confirmation Bars         : {SHORT_CONFIRM_BARS} consecutive")
    print(f"  EOD Square-Off (all trades)     : {EOD_HARD_EXIT}  [intraday — no overnight]")
    print(f"  Retest Tolerance (Path A)       : ATR × {RETEST_ATR_MULT}")
    print(f"  EMA9 Pullaway Pts  (Path B)     : {EMA9_PULLAWAY_PTS} pts")
    print(f"  Path A (EMA9 Retest)            : {'ENABLED' if ENABLE_PATH_A else 'DISABLED'}")
    print(f"  Path B (EMA9 Pullaway)          : {'ENABLED' if ENABLE_PATH_B else 'DISABLED'}")
    print(f"  Max Trades / Day                : {MAX_TRADES_PER_DAY}")
    print(f"  Consec-Loss Circuit Breaker     : halt after {MAX_CONSEC_LOSSES} losses")
    print(f"  Session: Prime                  : ON  (09:30–10:30)")
    print(f"  Session: Midday                 : {'ON' if ENABLE_MIDDAY else 'OFF'}  (11:30–13:30)")
    print(f"  Session: Euro                   : {'ON' if ENABLE_EURO else 'OFF'}  (14:15–15:00)")
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

    save_and_open_report(trades_df, df, save_base)
    print(f"{BAR}\n")


if __name__ == '__main__':
    main()