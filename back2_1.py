import pandas as pd
import numpy as np
import sys, os, json, webbrowser
from datetime import time, datetime, timedelta

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

# ── Swing Structure Trail ─────────────────────────────────────────
SWING_LOOKBACK      = 5
SWING_BUFFER_PTS    = 5
TRAIL_ATR_MULT      = 0.6   # kept for reference, not used

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
ENABLE_PATH_A       = False
ENABLE_PATH_B       = True
ENABLE_MIDDAY       = True
ENABLE_EURO         = False

OBSERVE_START   = time(9,  15)
OBSERVE_END     = time(9,  30)
PRIME_START     = time(9,  30)
PRIME_END       = time(10, 30)
MIDDAY_START    = time(11, 30)
MIDDAY_END      = time(13, 30)
EURO_START      = time(14, 15)
EURO_END        = time(15,  0)
SQUAREOFF_START = time(15,  0)
EOD_HARD_EXIT   = time(15,  0)


# ══════════════════════════════════════════════════════════════════
#   S/R ZONE INTEGRATION CONFIG  (new)
# ══════════════════════════════════════════════════════════════════

# Master switch — set False to reproduce original v4 behaviour exactly
ENABLE_SR_EXIT      = True

# How many points from zone centre counts as "approaching" the zone.
# When price enters this proximity to an opposing S/R zone,
# the tighter trail and optional auto-exit kick in.
SR_APPROACH_PTS     = 20.0     # e.g. zone at 24000 → tighten when price > 23980 (long)

# When True: close the trade as soon as price reaches the zone band edge.
# When False: only tighten the trailing stop near the zone (softer).
SR_EXIT_AT_ZONE     = True

# When SR_EXIT_AT_ZONE is False, how many bars to look back for the tighter trail
SR_TIGHT_SWING_LOOKBACK = 2    # 2 bars = 10-min window (vs default 5)
SR_TIGHT_SWING_BUFFER   = 2    # tighter buffer too

# S/R zone builder params (mirrors the detector script)
SR_LEFT_BARS        = 10
SR_RIGHT_BARS       = 10
SR_CLUSTER_TOLERANCE = 15.0    # absolute pts — pivots within this join one cluster
SR_ZONE_HALF_BAND   = 15.0     # zone = level ± this pts  →  30-pt total band
SR_MIN_WICK_TOUCHES = 3
SR_MIN_SESSIONS     = 2
SR_MIN_REJECTIONS   = 1
SR_TOP_N            = 20

# Minimum strength score for a zone to be used in exit decisions.
# Zones below this score are ignored even if nearby.
SR_MIN_STRENGTH     = 30.0


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
#   SWING STRUCTURE HELPERS
# ══════════════════════════════════════════════════════════════════

def get_swing_low(lows: np.ndarray, idx: int, lookback: int) -> float:
    start = max(0, idx - lookback)
    window = lows[start:idx]
    if len(window) == 0:
        return float(lows[idx])
    return float(np.min(window))


def get_swing_high(highs: np.ndarray, idx: int, lookback: int) -> float:
    start = max(0, idx - lookback)
    window = highs[start:idx]
    if len(window) == 0:
        return float(highs[idx])
    return float(np.max(window))


# ══════════════════════════════════════════════════════════════════
#   S/R ZONE BUILDER  (ported from detector script)
# ══════════════════════════════════════════════════════════════════

def _detect_pivots_sr(df: pd.DataFrame, left_bars: int, right_bars: int):
    """Detect confirmed pivot highs and lows for S/R computation."""
    from scipy.signal import argrelextrema

    highs = df['high'].values
    lows  = df['low'].values

    raw_phi = argrelextrema(highs, np.greater_equal, order=left_bars)[0]
    raw_plo = argrelextrema(lows,  np.less_equal,    order=left_bars)[0]

    def confirm(idx_arr, values, is_high):
        confirmed = []
        for i in idx_arr:
            lw = values[max(0, i - left_bars):i]
            rw = values[i+1:min(i + right_bars + 1, len(values))]
            if len(lw) == 0 or len(rw) == 0:
                continue
            if is_high and values[i] > np.max(lw) and values[i] > np.max(rw):
                confirmed.append(i)
            elif not is_high and values[i] < np.min(lw) and values[i] < np.min(rw):
                confirmed.append(i)
        return np.array(confirmed)

    phi = confirm(raw_phi, highs, True)
    plo = confirm(raw_plo, lows,  False)

    rows = []
    for i in phi:
        rows.append({'price': highs[i], 'date': df['timestamp'].iloc[i],
                     'type': 'high', 'session': df['timestamp'].iloc[i].date()})
    for i in plo:
        rows.append({'price': lows[i],  'date': df['timestamp'].iloc[i],
                     'type': 'low',  'session': df['timestamp'].iloc[i].date()})
    return pd.DataFrame(rows)


def _cluster_pivots(pivots: pd.DataFrame, tolerance: float) -> list:
    if len(pivots) == 0:
        return []
    sorted_p    = pivots.sort_values('price').reset_index(drop=True)
    prices      = sorted_p['price'].values
    clusters    = []
    group_start = 0
    for i in range(1, len(prices)):
        if prices[i] - prices[group_start] > tolerance:
            grp = sorted_p.iloc[group_start:i]
            clusters.append({
                'price':     grp['price'].mean(),
                'price_min': grp['price'].min(),
                'price_max': grp['price'].max(),
                'spread':    grp['price'].max() - grp['price'].min(),
                'n_pivots':  len(grp),
                'types':     list(grp['type']),
                'sessions':  list(grp['session'].unique()),
                'dates':     list(grp['date']),
            })
            group_start = i
    grp = sorted_p.iloc[group_start:]
    clusters.append({
        'price':     grp['price'].mean(),
        'price_min': grp['price'].min(),
        'price_max': grp['price'].max(),
        'spread':    grp['price'].max() - grp['price'].min(),
        'n_pivots':  len(grp),
        'types':     list(grp['type']),
        'sessions':  list(grp['session'].unique()),
        'dates':     list(grp['date']),
    })
    return clusters


def _analyse_level(df: pd.DataFrame, level: float, half_band: float,
                   min_rejection_pts: float = 8.0) -> dict:
    upper = level + half_band
    lower = level - half_band
    wick_touches = body_rejections = inside_bars = 0
    sessions_touched = set()
    last_touch_dt = None
    from collections import defaultdict
    session_held = defaultdict(bool)

    for dt, row in df.iterrows():
        hi, lo, cl = row['high'], row['low'], row['close']
        session = dt.date() if hasattr(dt, 'date') else dt

        wick_in = (hi >= lower) and (lo <= upper)
        if wick_in:
            wick_touches += 1
            sessions_touched.add(session)
            last_touch_dt = dt
            if cl > upper + min_rejection_pts or cl < lower - min_rejection_pts:
                body_rejections += 1
                session_held[session] = True
            elif (lo < lower - 5 and cl < lower - 5) or (hi > upper + 5 and cl > upper + 5):
                session_held[session] = False
        if hi <= upper and lo >= lower:
            inside_bars += 1

    return {
        'wick_touches':      wick_touches,
        'body_rejections':   body_rejections,
        'inside_bars':       inside_bars,
        'sessions_touched':  len(sessions_touched),
        'consecutive_holds': sum(1 for v in session_held.values() if v),
        'last_touch':        last_touch_dt,
    }


def build_sr_zones(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Build S/R zones from the same 5m dataset used for backtesting.
    Returns a DataFrame sorted by strength (descending).
    Zones include: price, upper, lower, type (Support/Resistance), strength.
    """
    if not ENABLE_SR_EXIT:
        return pd.DataFrame()

    try:
        from scipy.signal import argrelextrema
    except ImportError:
        print("  WARNING: scipy not installed — S/R zones disabled. pip install scipy")
        return pd.DataFrame()

    print("  Building S/R zones from 5m data...")

    # Need a timestamp-indexed version for _analyse_level
    df_idx = df_5m.copy()
    if 'timestamp' in df_idx.columns:
        df_idx = df_idx.set_index('timestamp')

    pivots = _detect_pivots_sr(df_5m, SR_LEFT_BARS, SR_RIGHT_BARS)
    if len(pivots) == 0:
        print("  WARNING: No pivots found for S/R zones.")
        return pd.DataFrame()

    clusters = _cluster_pivots(pivots, SR_CLUSTER_TOLERANCE)
    current_price = float(df_5m['close'].iloc[-1])
    latest_ts     = df_5m['timestamp'].iloc[-1]

    candidates = []
    for cl in clusters:
        level = cl['price']
        info  = _analyse_level(df_idx, level, SR_ZONE_HALF_BAND)

        if info['wick_touches']     < SR_MIN_WICK_TOUCHES: continue
        if info['sessions_touched'] < SR_MIN_SESSIONS:     continue
        if info['body_rejections']  < SR_MIN_REJECTIONS:   continue

        days_ago = (latest_ts - info['last_touch']).days if info['last_touch'] else 999
        recency  = np.exp(-days_ago / 15)

        touch_score     = min(info['wick_touches']    / 15, 1.0)
        rejection_score = min(info['body_rejections'] / max(info['wick_touches'], 1), 1.0)
        session_score   = min(info['sessions_touched'] / 10, 1.0)
        hold_score      = min(info['consecutive_holds'] / 5,  1.0)
        n_h = cl['types'].count('high')
        n_l = cl['types'].count('low')
        convergence  = 1.0 if (n_h > 0 and n_l > 0) else 0.5
        spread_score = max(0.0, 1.0 - cl['spread'] / SR_CLUSTER_TOLERANCE)

        strength = (
            0.28 * touch_score     +
            0.22 * rejection_score +
            0.18 * recency         +
            0.12 * session_score   +
            0.08 * hold_score      +
            0.07 * convergence     +
            0.05 * spread_score
        ) * 100

        if strength < SR_MIN_STRENGTH:
            continue

        zone_type = 'Support' if level < current_price else 'Resistance'
        candidates.append({
            'price':    round(level, 2),
            'upper':    round(level + SR_ZONE_HALF_BAND, 2),
            'lower':    round(level - SR_ZONE_HALF_BAND, 2),
            'type':     zone_type,
            'strength': round(strength, 1),
            'touches':  info['wick_touches'],
            'sessions': info['sessions_touched'],
        })

    if not candidates:
        print("  WARNING: No zones passed quality filters.")
        return pd.DataFrame()

    zones_df = pd.DataFrame(candidates).sort_values('strength', ascending=False)

    # Remove overlapping zones (keep strongest)
    df_z = zones_df.reset_index(drop=True)
    keep = [True] * len(df_z)
    for i in range(len(df_z)):
        if not keep[i]: continue
        for j in range(i + 1, len(df_z)):
            if not keep[j]: continue
            if abs(df_z.loc[i, 'price'] - df_z.loc[j, 'price']) < SR_ZONE_HALF_BAND * 2:
                keep[j] = False
    zones_df = df_z[keep].head(SR_TOP_N).reset_index(drop=True)

    print(f"  ✓ {len(zones_df)} S/R zones built  "
          f"({(zones_df['type']=='Resistance').sum()} resistance, "
          f"{(zones_df['type']=='Support').sum()} support)\n")
    return zones_df


def get_nearest_opposing_zone(price: float, direction: str,
                               zones_df: pd.DataFrame) -> dict | None:
    """
    For a LONG trade, return the nearest Resistance zone above price.
    For a SHORT trade, return the nearest Support zone below price.
    Returns None if no qualifying zone exists.
    """
    if zones_df is None or zones_df.empty:
        return None

    if direction == 'long':
        candidates = zones_df[
            (zones_df['type'] == 'Resistance') &
            (zones_df['lower'] > price)
        ]
        if candidates.empty:
            return None
        return candidates.loc[candidates['lower'].idxmin()].to_dict()
    else:
        candidates = zones_df[
            (zones_df['type'] == 'Support') &
            (zones_df['upper'] < price)
        ]
        if candidates.empty:
            return None
        return candidates.loc[candidates['upper'].idxmax()].to_dict()


# ══════════════════════════════════════════════════════════════════
#   ANTI-CHOP + REGIME FILTERS
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
#   BACKTEST ENGINE  (S/R-integrated exit engine)
# ══════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, zones_df: pd.DataFrame) -> tuple:
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
    # S/R state per trade
    active_sr_zone   = None   # nearest opposing zone at entry
    sr_tightened     = False  # have we already tightened near the zone?

    prev_date        = None
    daily_trades     = {}
    daily_consec_loss= {}

    def do_enter(dir_str, close_price, ts_now, atr_val, idx, path):
        nonlocal in_trade, direction, entry_price, entry_time, entry_idx, entry_path
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, trade_max_favor, trade_max_adverse
        nonlocal active_sr_zone, sr_tightened

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
        sr_tightened      = False

        if direction == 'long':
            stop_loss = entry_price - sl_dist
            be_level  = entry_price * (1 + LONG_BE_PCT / 100)
        else:
            stop_loss = entry_price + sl_dist
            be_level  = entry_price * (1 - SHORT_BE_PCT / 100)

        # Locate the nearest opposing S/R zone at entry
        active_sr_zone = get_nearest_opposing_zone(close_price, dir_str, zones_df)

    def do_exit(exit_price, ts_now, reason, exit_idx):
        nonlocal in_trade, direction, entry_price, entry_time, entry_idx, entry_path
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, equity
        nonlocal trade_max_favor, trade_max_adverse
        nonlocal active_sr_zone, sr_tightened

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
            'sr_zone_price':     active_sr_zone['price'] if active_sr_zone else None,
            'sr_tightened':      sr_tightened,
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
        active_sr_zone    = None
        sr_tightened      = False

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
        #   MANAGE OPEN TRADE — S/R-Integrated Exit Engine
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

            # ── Break-even trigger ────────────────────────────────
            if not be_triggered:
                if direction == 'long'  and close >= be_level:
                    stop_loss    = entry_price + 1.0
                    be_triggered = True
                    trail_active = True
                elif direction == 'short' and close <= be_level:
                    stop_loss    = entry_price - 1.0
                    be_triggered = True
                    trail_active = True

            # ── S/R Zone proximity check  ─────────────────────────
            # Determine if we are approaching an opposing S/R zone.
            # This happens independently of whether trail is active yet —
            # we check every bar so we catch fast moves.
            sr_approach = False
            sr_zone_exit_now = False

            if ENABLE_SR_EXIT and active_sr_zone is not None:
                z = active_sr_zone
                if direction == 'long':
                    # approaching resistance from below
                    dist_to_zone = z['lower'] - close   # positive = haven't reached yet
                    if dist_to_zone <= SR_APPROACH_PTS:
                        sr_approach = True
                    # Price has entered the zone band
                    if close >= z['lower']:
                        sr_zone_exit_now = SR_EXIT_AT_ZONE
                else:
                    # approaching support from above
                    dist_to_zone = close - z['upper']   # positive = haven't reached yet
                    if dist_to_zone <= SR_APPROACH_PTS:
                        sr_approach = True
                    # Price has entered the zone band
                    if close <= z['upper']:
                        sr_zone_exit_now = SR_EXIT_AT_ZONE

            # ── Trailing Stop — with optional S/R tightening ─────
            if trail_active:
                # Near an S/R zone → tighten the swing trail
                if sr_approach and not sr_tightened:
                    sr_tightened = True

                if sr_tightened and ENABLE_SR_EXIT:
                    lookback = SR_TIGHT_SWING_LOOKBACK
                    buffer   = SR_TIGHT_SWING_BUFFER
                else:
                    lookback = SWING_LOOKBACK
                    buffer   = SWING_BUFFER_PTS

                if direction == 'long':
                    swing_sl  = get_swing_low(lows, idx, lookback) - buffer
                    stop_loss = max(stop_loss, swing_sl)
                else:
                    swing_sl  = get_swing_high(highs, idx, lookback) + buffer
                    stop_loss = min(stop_loss, swing_sl)

            # ── Determine exit ────────────────────────────────────
            exit_p = None
            exit_r = None

            # 1. S/R zone exit (highest priority after hard SL)
            if ENABLE_SR_EXIT and sr_zone_exit_now and be_triggered:
                # Only take the SR exit if we're in profit (be_triggered guarantees that)
                exit_p = close
                exit_r = 'SR_ZONE_EXIT'

            # 2. Hard stop / trail stop
            elif direction == 'long' and close <= stop_loss:
                exit_p = stop_loss
                exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'
            elif direction == 'short' and close >= stop_loss:
                exit_p = stop_loss
                exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'

            # 3. EMA cross exit (post-BE only)
            elif ENABLE_EMA_CROSS_EXIT and be_triggered:
                if direction == 'long'  and ef < es:
                    exit_p = close
                    exit_r = 'EMA_CROSS_EXIT'
                elif direction == 'short' and ef > es:
                    exit_p = close
                    exit_r = 'EMA_CROSS_EXIT'

            # 4. Slope reversal exit (post-BE only)
            elif ENABLE_SLOPE_EXIT and be_triggered:
                if direction == 'long'  and slope_exit < -SLOPE_MIN:
                    exit_p = close
                    exit_r = 'SLOPE_REVERSAL_EXIT'
                elif direction == 'short' and slope_exit > SLOPE_MIN:
                    exit_p = close
                    exit_r = 'SLOPE_REVERSAL_EXIT'

            # 5. EOD hard exit
            elif c_time >= EOD_HARD_EXIT:
                exit_p = close
                exit_r = 'EOD_EXIT'

            if exit_p is not None:
                do_exit(exit_p, ts, exit_r, idx)

        # ══════════════════════════════════════════════════════════
        #   ENTRY LOGIC
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
          f" {'MFE':>7} {'MAE':>7} {'P&L':>9}  {'SRZone':>9}  Reason")
    print(SEP)
    for i, r in tdf.iterrows():
        sr_lbl = f"{r['sr_zone_price']:.0f}" if r.get('sr_zone_price') and not pd.isna(r.get('sr_zone_price', float('nan'))) else '—'
        print(f"{i+1:<5} {r['direction'].upper():<6} {'Path-'+r['entry_path']:<6}"
              f" {fmt_ts(r['entry_time']):<22} {fmt_ts(r['exit_time']):<22}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f}"
              f" {r['sl_at_entry']:>10.2f} {r['sl_at_exit']:>9.2f}"
              f" {r['mfe_pts']:>7.1f} {r['mae_pts']:>7.1f}"
              f" {r['pnl']:>+9.2f}  {sr_lbl:>9}  {r['exit_reason']}")

    metrics = compute_metrics(tdf)
    print(f"\n{'─'*62}")
    print("  PERFORMANCE METRICS")
    print(f"{'─'*62}")
    for k, v in metrics.items():
        print(f"  {k:<30}: {v}")

    print(f"\n{'─'*62}")
    print("  S/R EXIT ANALYSIS")
    print(f"{'─'*62}")
    sr_exits = tdf[tdf['exit_reason'] == 'SR_ZONE_EXIT']
    tightened = tdf[tdf.get('sr_tightened', pd.Series([False]*len(tdf))).astype(bool)] if 'sr_tightened' in tdf.columns else pd.DataFrame()
    print(f"  SR_ZONE_EXIT trades   : {len(sr_exits)}")
    if not sr_exits.empty:
        print(f"  Avg P&L on SR exits   : {sr_exits['pnl'].mean():.1f} pts")
        print(f"  Win rate on SR exits  : {(sr_exits['pnl']>0).mean()*100:.1f}%")
    if not tightened.empty:
        print(f"  Trades w/ SR tighten  : {len(tightened)}  (trail tightened near zone)")

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
        print(f"  Path {path} : trades={len(sub)}  wins={len(wins)}  losses={len(losses)}  "
              f"wr={len(wins)/len(sub)*100:.1f}%  "
              f"cum-profit={wins.sum():+.1f} pts  cum-loss={losses.sum():+.1f} pts  "
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

    print(f"{'─'*62}")
    print("  v4+SR SETTINGS SUMMARY")
    print(f"{'─'*62}")
    print(f"  Swing Lookback          : {SWING_LOOKBACK} bars  (normal)")
    print(f"  Swing Buffer            : {SWING_BUFFER_PTS} pts  (normal)")
    print(f"  SR Exit Enabled         : {'YES' if ENABLE_SR_EXIT else 'NO'}")
    print(f"  SR Exit At Zone         : {'YES — close trade on zone entry' if SR_EXIT_AT_ZONE else 'NO — tighten trail only'}")
    print(f"  SR Approach Buffer      : {SR_APPROACH_PTS} pts from zone lower/upper edge")
    print(f"  SR Tight Lookback       : {SR_TIGHT_SWING_LOOKBACK} bars  (when near zone)")
    print(f"  SR Tight Buffer         : {SR_TIGHT_SWING_BUFFER} pts  (when near zone)")
    print(f"  SR Min Strength         : {SR_MIN_STRENGTH}")
    print(f"  SR Zone Band            : ±{SR_ZONE_HALF_BAND} pts  ({int(SR_ZONE_HALF_BAND*2)} pts total)")
    print(f"  Path A (EMA9 Retest)    : {'ENABLED' if ENABLE_PATH_A else 'DISABLED'}")
    print(f"  Path B (EMA9 Pullaway)  : {'ENABLED' if ENABLE_PATH_B else 'DISABLED'}")
    print(f"  EOD Square-Off          : {EOD_HARD_EXIT}  (all trades)")
    print()


# ══════════════════════════════════════════════════════════════════
#   HTML REPORT
# ══════════════════════════════════════════════════════════════════

def build_html_report(trades_df: pd.DataFrame, raw_df: pd.DataFrame,
                      zones_df: pd.DataFrame) -> str:
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
        sr_price = r.get('sr_zone_price', None)
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
            'sr_zone_price': round(float(sr_price), 2) if sr_price and not pd.isna(float(sr_price)) else None,
            'sr_tightened':  bool(r.get('sr_tightened', False)),
            'candles':     chunk.to_dict(orient='records'),
            'rel_entry':   ei - start,
            'rel_exit':    xi - start,
        }

    trade_data   = [row_to_dict(r) for _, r in trades_df.iterrows()]
    trade_json   = json.dumps(trade_data)
    metrics      = compute_metrics(trades_df)
    metrics_rows = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in metrics.items())

    # Zone summary for header
    zones_json = '[]'
    if not zones_df.empty:
        zones_json = zones_df[['price','upper','lower','type','strength','touches','sessions']].to_json(orient='records')

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
        sr_badge = ''
        if t['exit_reason'] == 'SR_ZONE_EXIT':
            sr_badge = ' <span class="badge sr-exit">SR</span>'
        elif t.get('sr_tightened'):
            sr_badge = ' <span class="badge sr-tight">SR↑</span>'
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
          <td><span class="badge">{t['exit_reason']}</span>{sr_badge}</td>
        </tr>"""

    total_pnl  = metrics.get('Total P&L (pts)', 0)
    pnl_color  = 'green' if total_pnl >= 0 else 'red'
    ps         = path_summary
    sr_exit_count = len(trades_df[trades_df['exit_reason'] == 'SR_ZONE_EXIT']) if not trades_df.empty else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Speed Demon v4+SR – Trade Report</title>
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
             padding:8px 20px;display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start}}
  .path-card{{display:flex;flex-direction:column;gap:2px}}
  .path-card .ph{{font-size:10px;font-weight:700;letter-spacing:1px}}
  .path-card.pa .ph{{color:#f59e0b}}.path-card.pb .ph{{color:#60a5fa}}
  .path-card.sr-card .ph{{color:#a78bfa}}
  .path-card .prow{{font-size:10px;color:#94a3b8}}
  .path-card .prow span{{font-weight:700}}
  .profit-val{{color:#4ade80}}.loss-val{{color:#f87171}}.net-val{{color:#e2e8f0}}
  .status-badge{{padding:2px 6px;border-radius:3px;font-size:8px;margin-left:6px}}
  .status-enabled{{background:#4ade8022;color:#4ade80;border:1px solid #4ade8044}}
  .status-disabled{{background:#f8717122;color:#f87171;border:1px solid #f8717144}}
  .main{{display:flex;height:calc(100vh - 108px)}}
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
  .badge.sr-exit{{background:#a78bfa22;color:#a78bfa;border:1px solid #a78bfa44}}
  .badge.sr-tight{{background:#818cf822;color:#818cf8;border:1px solid #818cf844}}
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
  .v4tag{{background:#7c3aed22;color:#a78bfa;border:1px solid #7c3aed55;
          padding:2px 7px;border-radius:3px;font-size:9px;margin-left:8px}}
  .srtag{{background:#059669aa;color:#6ee7b7;border:1px solid #059669;
          padding:2px 7px;border-radius:3px;font-size:9px;margin-left:4px}}
  .sr-zone-line{{stroke:#a78bfa;stroke-width:1;stroke-dasharray:4 3;opacity:0.7}}
</style>
</head>
<body>
<header>
  <div>
    <h1>SPEED DEMON SCALPER v4+SR — Trade Report
      <span class="v4tag">SWING TRAIL</span>
      <span class="srtag">S/R EXIT</span>
    </h1>
    <div class="sub">S/R zone exits enabled · approach={SR_APPROACH_PTS}pts · zone band=±{SR_ZONE_HALF_BAND}pts · EOD {EOD_HARD_EXIT} · Click any row for chart</div>
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
    <div class="kpi"><div class="label">SR Exits</div>
      <div class="val" style="color:#a78bfa">{sr_exit_count}</div></div>
  </div>
</header>

<div class="path-bar">
  <div class="path-card pa">
    <div class="ph">PATH A — EMA9 Retest
      <span class="status-badge status-{'enabled' if ENABLE_PATH_A else 'disabled'}">{'ENABLED' if ENABLE_PATH_A else 'DISABLED'}</span>
    </div>
    <div class="prow">Trades: <span>{ps['A']['trades']}</span> &nbsp;|&nbsp;
      Net: <span class="net-val">{ps['A']['net']:+.1f} pts</span>
    </div>
  </div>
  <div class="path-card pb">
    <div class="ph">PATH B — EMA9 Pullaway 50pts
      <span class="status-badge status-{'enabled' if ENABLE_PATH_B else 'disabled'}">{'ENABLED' if ENABLE_PATH_B else 'DISABLED'}</span>
    </div>
    <div class="prow">Trades: <span>{ps['B']['trades']}</span> &nbsp;|&nbsp;
      Cum Profit: <span class="profit-val">{ps['B']['cum_profit']:+.1f}</span> &nbsp;|&nbsp;
      Cum Loss: <span class="loss-val">{ps['B']['cum_loss']:+.1f}</span> &nbsp;|&nbsp;
      Net: <span class="net-val">{ps['B']['net']:+.1f} pts</span>
    </div>
  </div>
  <div class="path-card sr-card">
    <div class="ph">S/R ZONE EXIT ENGINE
      <span class="status-badge status-{'enabled' if ENABLE_SR_EXIT else 'disabled'}">{'ENABLED' if ENABLE_SR_EXIT else 'DISABLED'}</span>
    </div>
    <div class="prow">
      Mode: <span>{'Exit at zone' if SR_EXIT_AT_ZONE else 'Tighten trail'}</span> &nbsp;|&nbsp;
      Approach: <span>{SR_APPROACH_PTS}pts</span> &nbsp;|&nbsp;
      Zone band: <span>±{SR_ZONE_HALF_BAND}pts</span> &nbsp;|&nbsp;
      SR exits: <span style="color:#a78bfa">{sr_exit_count}</span>
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
const ZONES={zones_json};
const EMA_FAST={EMA_FAST},EMA_SLOW={EMA_SLOW},EMA_MACRO={EMA_MACRO};
const SR_HALF_BAND={SR_ZONE_HALF_BAND};
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
    ['SR Zone',      t.sr_zone_price ? t.sr_zone_price.toFixed(0)+' ±'+SR_HALF_BAND+'pts' : '—'],
    ['SR Tightened', t.sr_tightened?'YES (trail tightened)':'NO'],
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
  // Include SR zone bounds if present
  if(t.sr_zone_price){{
    allP.push(t.sr_zone_price + SR_HALF_BAND);
    allP.push(t.sr_zone_price - SR_HALF_BAND);
  }}
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

  // SR zone band overlay
  if(t.sr_zone_price){{
    const zu=yS(t.sr_zone_price+SR_HALF_BAND);
    const zl=yS(t.sr_zone_price-SR_HALF_BAND);
    ctx.fillStyle='rgba(167,139,250,0.07)';
    ctx.fillRect(PL,zu,cW,zl-zu);
    ctx.strokeStyle='rgba(167,139,250,0.5)';
    ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(PL,zu); ctx.lineTo(W-PR,zu); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PL,zl); ctx.lineTo(W-PR,zl); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='rgba(167,139,250,0.8)'; ctx.font='8px monospace'; ctx.textAlign='left';
    ctx.fillText('SR '+t.sr_zone_price.toFixed(0),W-PR+2,yS(t.sr_zone_price)+3);
    ctx.textAlign='right';
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
             ['EMA'+EMA_MACRO,'#a78bfa'],['Entry','#4ade80'],['Exit','#f87171'],
             ['SR Zone','rgba(167,139,250,0.7)']];
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


def save_and_open_report(trades_df: pd.DataFrame, raw_df: pd.DataFrame,
                         zones_df: pd.DataFrame, basename: str):
    html        = build_html_report(trades_df, raw_df, zones_df)
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
    print("  SPEED DEMON SCALPER  v4 + S/R EXIT ENGINE")
    print(f"{BAR}")

    cli_args = sys.argv[1:]
    mode     = 'csv' if cli_args else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\nData source : yfinance  |  Symbol : {YFINANCE_SYMBOL}")
        df        = fetch_yfinance('5m', '5m')
        save_base = os.path.join(os.getcwd(),
                                 YFINANCE_SYMBOL.replace('^', '').replace('.', '_') + '_speed_demon_v4sr')
    elif mode == 'csv':
        path5 = cli_args[0] if cli_args else CSV_5MIN_PATH
        if not os.path.exists(path5):
            print(f"ERROR: File not found: {path5}"); sys.exit(1)
        print(f"\nLoading 5m CSV : {path5}")
        df        = load_csv(path5, '5m')
        save_base = os.path.splitext(path5)[0] + '_speed_demon_v4sr'
    else:
        print(f"ERROR: Unknown DATA_MODE '{DATA_MODE}'"); sys.exit(1)

    print(f"\n{'─'*62}")
    print("  STRATEGY CONFIGURATION  (v4+SR)")
    print(f"{'─'*62}")
    print(f"  EMA Fast / Slow / Macro (5m)    : {EMA_FAST} / {EMA_SLOW} / {EMA_MACRO}")
    print(f"  HTF EMAs (15m)                  : {HTF_EMA_FAST} / {HTF_EMA_SLOW} / {HTF_EMA_TREND}")
    print(f"  ATR Period                       : {ATR_PERIOD}")
    print(f"  ADX Filter                       : ADX({ADX_PERIOD}) >= {ADX_MIN}")
    print(f"  LONG  SL Multiplier             : {LONG_ATR_SL_MULT}× ATR")
    print(f"  SHORT SL Multiplier             : {SHORT_ATR_SL_MULT}× ATR")
    print(f"  LONG  Break-Even Trigger        : +{LONG_BE_PCT}%")
    print(f"  SHORT Break-Even Trigger        : -{SHORT_BE_PCT}%")
    print(f"  Trailing Stop (base)            : SWING STRUCTURE")
    print(f"    Swing Lookback (normal)        : {SWING_LOOKBACK} bars")
    print(f"    Swing Buffer   (normal)        : {SWING_BUFFER_PTS} pts")
    print(f"  ── S/R EXIT ENGINE ──────────────────────────────────")
    print(f"  SR Exit Enabled                 : {'YES' if ENABLE_SR_EXIT else 'NO'}")
    print(f"  SR Exit Mode                    : {'Exit trade when price enters zone' if SR_EXIT_AT_ZONE else 'Tighten trail near zone only'}")
    print(f"  SR Approach Buffer              : {SR_APPROACH_PTS} pts  (tighten trail this many pts before zone)")
    print(f"  SR Tight Swing Lookback         : {SR_TIGHT_SWING_LOOKBACK} bars  (when near zone)")
    print(f"  SR Tight Swing Buffer           : {SR_TIGHT_SWING_BUFFER} pts  (when near zone)")
    print(f"  SR Zone Band                    : ±{SR_ZONE_HALF_BAND} pts  ({int(SR_ZONE_HALF_BAND*2)} pts total)")
    print(f"  SR Cluster Tolerance            : {SR_CLUSTER_TOLERANCE} pts")
    print(f"  SR Min Strength Score           : {SR_MIN_STRENGTH}")
    print(f"  SR Min Wick Touches             : {SR_MIN_WICK_TOUCHES}")
    print(f"  SR Min Sessions                 : {SR_MIN_SESSIONS}")
    print(f"  SR Min Rejections               : {SR_MIN_REJECTIONS}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  EMA Cross Exit                  : {'ON' if ENABLE_EMA_CROSS_EXIT else 'OFF'}")
    print(f"  Slope Reversal Exit             : {'ON' if ENABLE_SLOPE_EXIT else 'OFF'}")
    print(f"  Path A (EMA9 Retest)            : {'ENABLED' if ENABLE_PATH_A else 'DISABLED'}")
    print(f"  Path B (EMA9 Pullaway)          : {'ENABLED' if ENABLE_PATH_B else 'DISABLED'}")
    print(f"  EOD Square-Off                  : {EOD_HARD_EXIT}  [intraday — no overnight]")
    print(f"  Max Trades / Day                : {MAX_TRADES_PER_DAY}")
    print(f"{'─'*62}")

    print(f"\nComputing 5m indicators + ADX ...")
    df = compute_indicators_5m(df)

    print(f"Computing 15m HTF bias ...")
    df = compute_htf_bias(df)

    # Build S/R zones from the same dataset
    print(f"\nBuilding S/R zones ...")
    zones_df = build_sr_zones(df)

    print("Running backtest ...\n")
    trades_df, equity_df = run_backtest(df, zones_df)

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

    if not zones_df.empty:
        zones_out = save_base + '_sr_zones.csv'
        zones_df.to_csv(zones_out, index=False)
        print(f"  S/R zones    : {zones_out}")

    save_and_open_report(trades_df, df, zones_df, save_base)
    print(f"{BAR}\n")


if __name__ == '__main__':
    main()