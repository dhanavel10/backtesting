import pandas as pd
import numpy as np
import sys, os, json, webbrowser
from datetime import time, timedelta
from scipy.signal import argrelextrema
from collections import defaultdict

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

SWING_LOOKBACK      = 5
SWING_BUFFER_PTS    = 5
TRAIL_ATR_MULT      = 0.6       # retained for reference only

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

OBSERVE_START       = time(9,  15)
OBSERVE_END         = time(9,  30)
PRIME_START         = time(9,  30)
PRIME_END           = time(10, 30)
MIDDAY_START        = time(11, 30)
MIDDAY_END          = time(13, 30)
EURO_START          = time(14, 15)
EURO_END            = time(15,  0)
SQUAREOFF_START     = time(15,  0)

EOD_HARD_EXIT       = time(15,  0)

# ══════════════════════════════════════════════════════════════════
#   S/R INTEGRATION CONFIG  (v5 additions)
# ══════════════════════════════════════════════════════════════════
#
# ENTRY FILTER (light — does not over-restrict):
#   Skip entry only when price is within SR_ENTRY_WALL_PTS of an
#   opposing S/R zone (long → resistance above, short → support below).
#   This avoids entering directly into a known wall.
#   Set to 0 to disable entry filtering entirely.
SR_ENTRY_WALL_PTS       = 20    # pts — if next opposing zone < this away, skip

# EXIT ENHANCEMENT — Zone Target Exit:
#   When an open trade approaches a known S/R zone in its favour
#   (long approaching resistance / short approaching support),
#   exit at the zone's NEAR edge instead of waiting for the trail to fire.
#   This locks in profit BEFORE the zone can absorb momentum.
ENABLE_SR_TARGET_EXIT   = False
SR_TARGET_APPROACH_PTS  = 10   # exit when price is within this many pts of zone near-edge

# TIGHTEN TRAIL NEAR S/R:
#   Once within SR_TIGHTEN_WITHIN_PTS of an opposing zone (but not
#   close enough to trigger SR_TARGET_EXIT), switch to a tighter
#   swing lookback so the trail hugs price more aggressively.
ENABLE_SR_TIGHT_TRAIL   = True
SR_TIGHTEN_WITHIN_PTS   = 30   # pts — zone distance threshold to activate tight trail
SR_TIGHT_LOOKBACK       = 3    # bars — replaces SWING_LOOKBACK when near zone

# S/R ZONE COMPUTATION SETTINGS (mirrors Script 2):
SR_LEFT_BARS        = 10       # pivot window left
SR_RIGHT_BARS       = 10       # pivot window right
SR_CLUSTER_TOL      = 15.0     # absolute pts — pivot clustering radius
SR_ZONE_HALF_BAND   = 15.0     # zone = level ± this pts  (30pt total band)
SR_MIN_TOUCHES      = 3        # min wick touches
SR_MIN_SESSIONS     = 2        # min distinct days
SR_MIN_REJECTIONS   = 1        # min body rejections
SR_TOP_N            = 30       # max zones to keep

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
    start  = max(0, idx - lookback)
    window = lows[start:idx]
    return float(np.min(window)) if len(window) > 0 else float(lows[idx])


def get_swing_high(highs: np.ndarray, idx: int, lookback: int) -> float:
    start  = max(0, idx - lookback)
    window = highs[start:idx]
    return float(np.max(window)) if len(window) > 0 else float(highs[idx])


# ══════════════════════════════════════════════════════════════════
#   S/R ZONE ENGINE  (adapted from Script 2, look-ahead-free)
# ══════════════════════════════════════════════════════════════════

def _detect_pivots_array(highs: np.ndarray, lows: np.ndarray,
                          left: int = 10, right: int = 10) -> tuple:
    """
    Detect confirmed pivot highs and lows on raw arrays.
    Returns (pivot_high_prices, pivot_high_idxs,
             pivot_low_prices,  pivot_low_idxs)
    """
    raw_phi = argrelextrema(highs, np.greater_equal, order=left)[0]
    raw_plo = argrelextrema(lows,  np.less_equal,    order=left)[0]

    def confirm(idx_arr, values, is_high):
        out = []
        for i in idx_arr:
            lw = values[max(0, i - left):i]
            rw = values[i+1:min(i + right + 1, len(values))]
            if len(lw) == 0 or len(rw) == 0:
                continue
            if is_high and values[i] > np.max(lw) and values[i] > np.max(rw):
                out.append(i)
            elif not is_high and values[i] < np.min(lw) and values[i] < np.min(rw):
                out.append(i)
        return np.array(out, dtype=int)

    phi = confirm(raw_phi, highs, True)
    plo = confirm(raw_plo, lows,  False)
    return (highs[phi] if len(phi) else np.array([]),  phi,
            lows[plo]  if len(plo) else np.array([]),  plo)


def _cluster_prices(prices: np.ndarray, tolerance: float) -> list:
    """Single-linkage absolute-point clustering. Returns list of cluster dicts."""
    if len(prices) == 0:
        return []
    sp = np.sort(prices)
    clusters, g_start = [], 0
    for i in range(1, len(sp)):
        if sp[i] - sp[g_start] > tolerance:
            clusters.append({'price': sp[g_start:i].mean(),
                             'min':   sp[g_start:i].min(),
                             'max':   sp[g_start:i].max(),
                             'n':     i - g_start})
            g_start = i
    grp = sp[g_start:]
    clusters.append({'price': grp.mean(), 'min': grp.min(),
                     'max': grp.max(), 'n': len(grp)})
    return clusters


def _score_level(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 dates: np.ndarray, level: float, half_band: float,
                 min_rejection_pts: float = 8.0) -> dict:
    """Count wick touches, body rejections, and distinct sessions."""
    upper = level + half_band
    lower = level - half_band
    wick_touches = body_rejections = inside_bars = 0
    sessions = set()
    last_idx  = -1
    for i in range(len(closes)):
        hi, lo, cl = highs[i], lows[i], closes[i]
        if (hi >= lower) and (lo <= upper):
            wick_touches += 1
            sessions.add(dates[i])
            last_idx = i
            if cl > upper + min_rejection_pts or cl < lower - min_rejection_pts:
                body_rejections += 1
        if hi <= upper and lo >= lower:
            inside_bars += 1
    return {
        'wick_touches':    wick_touches,
        'body_rejections': body_rejections,
        'inside_bars':     inside_bars,
        'sessions':        len(sessions),
        'last_idx':        last_idx,
    }


def _remove_overlaps_list(zones: list, half_band: float) -> list:
    zones = sorted(zones, key=lambda z: -z['strength'])
    keep  = [True] * len(zones)
    for i in range(len(zones)):
        if not keep[i]: continue
        for j in range(i+1, len(zones)):
            if not keep[j]: continue
            if abs(zones[i]['price'] - zones[j]['price']) < half_band * 2:
                keep[j] = False
    return [z for z, k in zip(zones, keep) if k]


def build_sr_zones_from_arrays(
    highs:     np.ndarray,
    lows:      np.ndarray,
    closes:    np.ndarray,
    dates:     np.ndarray,           # array of date objects (one per bar)
    n_bars:    int,                  # use only the first n_bars rows
    left:      int   = SR_LEFT_BARS,
    right:     int   = SR_RIGHT_BARS,
    clus_tol:  float = SR_CLUSTER_TOL,
    half_band: float = SR_ZONE_HALF_BAND,
    min_touch: int   = SR_MIN_TOUCHES,
    min_sess:  int   = SR_MIN_SESSIONS,
    min_rej:   int   = SR_MIN_REJECTIONS,
    top_n:     int   = SR_TOP_N,
) -> list:
    """
    Compute S/R zones using ONLY data[0:n_bars].
    Returns a list of zone dicts sorted by strength (descending).
    No look-ahead: caller must ensure n_bars excludes today's data.
    """
    if n_bars < left + right + 2:
        return []

    H = highs[:n_bars]
    L = lows[:n_bars]
    C = closes[:n_bars]
    D = dates[:n_bars]

    ph_prices, _, pl_prices, _ = _detect_pivots_array(H, L, left, right)
    all_prices = np.concatenate([ph_prices, pl_prices])
    if len(all_prices) == 0:
        return []

    clusters = _cluster_prices(all_prices, clus_tol)
    total_bars = n_bars
    zones = []

    for cl in clusters:
        level = cl['price']
        info  = _score_level(H, L, C, D, level, half_band)

        if info['wick_touches']    < min_touch: continue
        if info['sessions']        < min_sess:  continue
        if info['body_rejections'] < min_rej:   continue

        recency = np.exp(-max(0, total_bars - info['last_idx'] - 1) / 60.0)

        touch_score     = min(info['wick_touches']    / 15, 1.0)
        rejection_score = min(info['body_rejections'] / max(info['wick_touches'], 1), 1.0)
        session_score   = min(info['sessions']         / 10, 1.0)

        strength = (
            0.35 * touch_score +
            0.25 * rejection_score +
            0.25 * recency +
            0.15 * session_score
        ) * 100

        zones.append({
            'price':    round(level, 2),
            'upper':    round(level + half_band, 2),
            'lower':    round(level - half_band, 2),
            'strength': round(strength, 1),
            'touches':  info['wick_touches'],
            'sessions': info['sessions'],
        })

    zones = _remove_overlaps_list(zones, half_band)
    zones = sorted(zones, key=lambda z: -z['strength'])
    return zones[:top_n]


# ══════════════════════════════════════════════════════════════════
#   DAILY S/R SNAPSHOT ENGINE  (look-ahead-free)
#
#  Strategy:
#    At the START of each trading day (first bar), compute S/R zones
#    using ALL data UP TO (but NOT including) today's bars.
#    These zones are frozen for the entire session — exactly what a
#    live trader would have available at market open.
#
#  This is the only correct approach to avoid look-ahead bias.
# ══════════════════════════════════════════════════════════════════

def precompute_daily_sr(df: pd.DataFrame) -> dict:
    """
    Returns a dict: {date: [zone_dict, ...]}
    For each trading day, zones are built from all prior-day data only.
    """
    highs  = df['high'].astype(float).values
    lows   = df['low'].astype(float).values
    closes = df['close'].astype(float).values
    dates  = np.array([ts.date() for ts in df['timestamp']])

    unique_dates = sorted(set(dates))
    daily_sr     = {}

    print(f"  Pre-computing S/R snapshots for {len(unique_dates)} trading days ...")
    for i, today in enumerate(unique_dates):
        if i == 0:
            daily_sr[today] = []      # no prior data on day 0
            continue

        # Use all bars BEFORE today (strictly prior-day data only)
        cutoff_idx = int(np.searchsorted(dates, today, side='left'))
        if cutoff_idx < SR_LEFT_BARS + SR_RIGHT_BARS + 2:
            daily_sr[today] = []
            continue

        zones = build_sr_zones_from_arrays(
            highs, lows, closes, dates, n_bars=cutoff_idx
        )
        daily_sr[today] = zones

    # Summary
    days_with_zones = sum(1 for z in daily_sr.values() if z)
    print(f"  ✓ {days_with_zones}/{len(unique_dates)} days have S/R zones")
    return daily_sr


def get_nearest_resistance(zones: list, price: float) -> float | None:
    """Return the lower edge of the nearest resistance zone above price."""
    candidates = [z['lower'] for z in zones if z['price'] > price]
    return min(candidates) if candidates else None


def get_nearest_support(zones: list, price: float) -> float | None:
    """Return the upper edge of the nearest support zone below price."""
    candidates = [z['upper'] for z in zones if z['price'] < price]
    return max(candidates) if candidates else None


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
#   BACKTEST ENGINE  (v5 — S/R integrated)
# ══════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, daily_sr: dict) -> tuple:
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

    trades     = []
    equity     = 0.0
    eq_curve   = []

    opening_high  = {}
    opening_close = {}

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
    sr_target         = None   # v5: zone-based profit target for open trade

    prev_date         = None
    daily_trades      = {}
    daily_consec_loss = {}

    # ── v5 tracking counters ──────────────────────────────────
    sr_entry_skips   = 0
    sr_target_exits  = 0
    sr_tighten_count = 0

    def do_enter(dir_str, close_price, ts_now, atr_val, idx, path,
                 zones_today):
        nonlocal in_trade, direction, entry_price, entry_time, entry_idx, entry_path
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, trade_max_favor, trade_max_adverse, sr_target

        sl_mult = LONG_ATR_SL_MULT if dir_str == 'long' else SHORT_ATR_SL_MULT
        sl_dist = atr_val * sl_mult

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
            # v5: set a zone target at the nearest resistance above entry
            if ENABLE_SR_TARGET_EXIT and zones_today:
                r_edge = get_nearest_resistance(zones_today, close_price)
                # Only use as target if it's meaningfully above entry
                # (at least 20 pts so we have room to profit)
                sr_target = r_edge if (r_edge is not None and
                                       r_edge - close_price >= 20) else None
            else:
                sr_target = None
        else:
            stop_loss = entry_price + sl_dist
            be_level  = entry_price * (1 - SHORT_BE_PCT / 100)
            # v5: set a zone target at the nearest support below entry
            if ENABLE_SR_TARGET_EXIT and zones_today:
                s_edge = get_nearest_support(zones_today, close_price)
                sr_target = s_edge if (s_edge is not None and
                                       close_price - s_edge >= 20) else None
            else:
                sr_target = None

    def do_exit(exit_price, ts_now, reason, exit_idx):
        nonlocal in_trade, direction, entry_price, entry_time, entry_idx, entry_path
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, equity, sr_target
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
            'sr_target_was_set': sr_target is not None,
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
        sr_target         = None

    for idx in range(n):
        close    = closes[idx]
        high_c   = highs[idx]
        low_c    = lows[idx]
        ef       = ema_fast[idx]
        es       = ema_slow[idx]
        em       = ema_macro[idx]
        atr      = float(atrs[idx])
        adx_v    = float(adx_vals[idx]) if not np.isnan(adx_vals[idx]) else 0.0
        slope    = float(slopes[idx])      if not np.isnan(slopes[idx])      else 0.0
        slope_exit = float(slopes_exit[idx]) if not np.isnan(slopes_exit[idx]) else 0.0
        cb       = int(consec_bearish[idx])
        ts       = ts_list[idx]
        c_time   = ts.time()
        c_date   = ts.date()
        date_str = str(c_date)
        session  = get_session(c_time)

        if c_date != prev_date:
            prev_date = c_date

        if c_time == time(9, 30):
            opening_high[date_str]  = high_c
            opening_close[date_str] = close

        if np.isnan(ef) or np.isnan(es) or np.isnan(em) or np.isnan(atr):
            eq_curve.append({'timestamp': ts, 'equity': equity})
            continue

        htf_long = bool(htf_long_bias[idx]) if not pd.isna(htf_long_bias[idx]) else False

        # Today's pre-computed S/R zones (look-ahead-free)
        zones_today = daily_sr.get(c_date, [])

        # ══════════════════════════════════════════════════════
        #   MANAGE OPEN TRADE
        # ══════════════════════════════════════════════════════
        if in_trade:
            if direction == 'long':
                favor   = high_c - entry_price
                adverse = entry_price - low_c
            else:
                favor   = entry_price - low_c
                adverse = high_c - entry_price
            trade_max_favor   = max(trade_max_favor,   favor)
            trade_max_adverse = max(trade_max_adverse, adverse)

            # ── Break-even trigger ─────────────────────────────
            if not be_triggered:
                if direction == 'long'  and close >= be_level:
                    stop_loss    = entry_price + 1.0
                    be_triggered = True
                    trail_active = True
                elif direction == 'short' and close <= be_level:
                    stop_loss    = entry_price - 1.0
                    be_triggered = True
                    trail_active = True

            # ── v5: Determine effective trail lookback ─────────
            # Near a zone? Use tighter lookback so trail hugs price.
            effective_lookback = SWING_LOOKBACK
            if ENABLE_SR_TIGHT_TRAIL and trail_active and zones_today:
                if direction == 'long':
                    r_edge = get_nearest_resistance(zones_today, close)
                    if r_edge is not None and (r_edge - close) <= SR_TIGHTEN_WITHIN_PTS:
                        effective_lookback = SR_TIGHT_LOOKBACK
                        sr_tighten_count += 1
                else:
                    s_edge = get_nearest_support(zones_today, close)
                    if s_edge is not None and (close - s_edge) <= SR_TIGHTEN_WITHIN_PTS:
                        effective_lookback = SR_TIGHT_LOOKBACK
                        sr_tighten_count += 1

            # ── Trailing Stop: Swing Structure ─────────────────
            if trail_active:
                if direction == 'long':
                    swing_sl = (get_swing_low(lows, idx, effective_lookback)
                                - SWING_BUFFER_PTS)
                    stop_loss = max(stop_loss, swing_sl)
                else:
                    swing_sl = (get_swing_high(highs, idx, effective_lookback)
                                + SWING_BUFFER_PTS)
                    stop_loss = min(stop_loss, swing_sl)

            exit_p = None
            exit_r = None

            # ── v5: Zone Target Exit (highest priority after SL) ─
            # Exit when price approaches the near edge of the
            # opposing S/R zone, capturing profit BEFORE the zone
            # absorbs momentum. Only fires after BE to avoid early exits.
            if (ENABLE_SR_TARGET_EXIT and be_triggered and
                    sr_target is not None and exit_p is None):
                if direction == 'long' and close >= sr_target - SR_TARGET_APPROACH_PTS:
                    exit_p = close
                    exit_r = 'SR_TARGET_EXIT'
                    sr_target_exits += 1
                elif direction == 'short' and close <= sr_target + SR_TARGET_APPROACH_PTS:
                    exit_p = close
                    exit_r = 'SR_TARGET_EXIT'
                    sr_target_exits += 1

            # ── Standard exits ─────────────────────────────────
            if exit_p is None:
                if direction == 'long' and close <= stop_loss:
                    exit_p = stop_loss
                    exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'
                elif direction == 'short' and close >= stop_loss:
                    exit_p = stop_loss
                    exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'

            if exit_p is None and ENABLE_EMA_CROSS_EXIT and be_triggered:
                if direction == 'long'  and ef < es:
                    exit_p = close; exit_r = 'EMA_CROSS_EXIT'
                elif direction == 'short' and ef > es:
                    exit_p = close; exit_r = 'EMA_CROSS_EXIT'

            if exit_p is None and ENABLE_SLOPE_EXIT and be_triggered:
                if direction == 'long'  and slope_exit < -SLOPE_MIN:
                    exit_p = close; exit_r = 'SLOPE_REVERSAL_EXIT'
                elif direction == 'short' and slope_exit > SLOPE_MIN:
                    exit_p = close; exit_r = 'SLOPE_REVERSAL_EXIT'

            if exit_p is None and c_time >= EOD_HARD_EXIT:
                exit_p = close; exit_r = 'EOD_EXIT'

            if exit_p is not None:
                do_exit(exit_p, ts, exit_r, idx)

        # ══════════════════════════════════════════════════════
        #   ENTRY LOGIC
        # ══════════════════════════════════════════════════════
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

            # ── LONG Entry ─────────────────────────────────────
            if ENABLE_LONG and htf_long:
                if ENABLE_PATH_A:
                    path_a_long = (
                        ef > es and close > em and slope > 0 and
                        abs(close - ef) <= retest_tol and close > es
                    )
                    if path_a_long:
                        # v5: skip if entering directly into a resistance wall
                        wall_blocked = False
                        if SR_ENTRY_WALL_PTS > 0 and zones_today:
                            r_edge = get_nearest_resistance(zones_today, close)
                            if r_edge is not None and (r_edge - close) < SR_ENTRY_WALL_PTS:
                                wall_blocked = True
                                sr_entry_skips += 1
                        if not wall_blocked:
                            do_enter('long', close, ts, atr, idx, 'A', zones_today)
                            daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

                if not in_trade and ENABLE_PATH_B:
                    path_b_long = (
                        ef > es and close > em and slope > 0 and
                        (close - ef) >= EMA9_PULLAWAY_PTS and
                        date_str in opening_high and
                        close > opening_high[date_str]
                    )
                    if path_b_long:
                        # v5: skip if resistance zone is too close above entry
                        # (already ran 50pts from EMA9 — don't enter into a wall)
                        wall_blocked = False
                        if SR_ENTRY_WALL_PTS > 0 and zones_today:
                            r_edge = get_nearest_resistance(zones_today, close)
                            if r_edge is not None and (r_edge - close) < SR_ENTRY_WALL_PTS:
                                wall_blocked = True
                                sr_entry_skips += 1
                        if not wall_blocked:
                            do_enter('long', close, ts, atr, idx, 'B', zones_today)
                            daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

            # ── SHORT Entry ────────────────────────────────────
            if not in_trade and ENABLE_SHORT:
                if ENABLE_PATH_A:
                    path_a_short = (
                        ef < es and close < em and slope < 0 and
                        abs(close - ef) <= retest_tol and close < es and
                        cb >= SHORT_CONFIRM_BARS
                    )
                    if path_a_short:
                        wall_blocked = False
                        if SR_ENTRY_WALL_PTS > 0 and zones_today:
                            s_edge = get_nearest_support(zones_today, close)
                            if s_edge is not None and (close - s_edge) < SR_ENTRY_WALL_PTS:
                                wall_blocked = True
                                sr_entry_skips += 1
                        if not wall_blocked:
                            do_enter('short', close, ts, atr, idx, 'A', zones_today)
                            daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

                if not in_trade and ENABLE_PATH_B:
                    path_b_short = (
                        ef < es and close < em and slope < 0 and
                        (ef - close) >= EMA9_PULLAWAY_PTS and
                        cb >= SHORT_CONFIRM_BARS and
                        date_str in opening_close and
                        close < opening_close[date_str]
                    )
                    if path_b_short:
                        wall_blocked = False
                        if SR_ENTRY_WALL_PTS > 0 and zones_today:
                            s_edge = get_nearest_support(zones_today, close)
                            if s_edge is not None and (close - s_edge) < SR_ENTRY_WALL_PTS:
                                wall_blocked = True
                                sr_entry_skips += 1
                        if not wall_blocked:
                            do_enter('short', close, ts, atr, idx, 'B', zones_today)
                            daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

        eq_curve.append({'timestamp': ts, 'equity': equity})

    if in_trade:
        do_exit(closes[-1], ts_list[-1], 'END_OF_DATA', n - 1)

    sr_stats = {
        'sr_entry_skips':   sr_entry_skips,
        'sr_target_exits':  sr_target_exits,
        'sr_tighten_bars':  sr_tighten_count,
    }
    return pd.DataFrame(trades), pd.DataFrame(eq_curve), sr_stats


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


def print_results(tdf: pd.DataFrame, sr_stats: dict):
    SEP = '─' * 170

    print(f"\n{SEP}")
    print(f"  TRADE LOG  ({len(tdf)} trades)")
    print(SEP)
    print(f"{'#':<5} {'Dir':<6} {'Path':<6} {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>9} {'Exit':>9} {'SL@Entry':>10} {'SL@Exit':>9}"
          f" {'MFE':>7} {'MAE':>7} {'P&L':>9}  Reason")
    print(SEP)
    for i, r in tdf.iterrows():
        sr_flag = ' [SR]' if r.get('sr_target_was_set') else ''
        print(f"{i+1:<5} {r['direction'].upper():<6} {'Path-'+r['entry_path']:<6}"
              f" {fmt_ts(r['entry_time']):<22} {fmt_ts(r['exit_time']):<22}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f}"
              f" {r['sl_at_entry']:>10.2f} {r['sl_at_exit']:>9.2f}"
              f" {r['mfe_pts']:>7.1f} {r['mae_pts']:>7.1f}"
              f" {r['pnl']:>+9.2f}  {r['exit_reason']}{sr_flag}")

    metrics = compute_metrics(tdf)
    print(f"\n{'─'*62}")
    print("  PERFORMANCE METRICS")
    print(f"{'─'*62}")
    for k, v in metrics.items():
        print(f"  {k:<30}: {v}")

    print(f"\n{'─'*62}")
    print("  v5 S/R INTEGRATION STATS")
    print(f"{'─'*62}")
    print(f"  Entry skips (wall filter)   : {sr_stats['sr_entry_skips']}")
    print(f"  SR Target exits fired       : {sr_stats['sr_target_exits']}")
    print(f"  Bars with tight trail active: {sr_stats['sr_tighten_bars']}")
    if not tdf.empty:
        sr_exits = tdf[tdf['exit_reason'] == 'SR_TARGET_EXIT']
        if len(sr_exits):
            print(f"  Avg P&L on SR exits         : {sr_exits['pnl'].mean():.1f} pts")
            print(f"  Win rate on SR exits         : "
                  f"{(sr_exits['pnl'] > 0).mean()*100:.1f}%")

    print(f"\n{'─'*62}")
    print("  DIRECTION BREAKDOWN")
    print(f"{'─'*62}")
    for d in ['long', 'short']:
        sub = tdf[tdf['direction'] == d]
        if sub.empty: continue
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
    print("  ENTRY PATH BREAKDOWN")
    print(f"{'─'*62}")
    for path in ['A', 'B']:
        sub = tdf[tdf['entry_path'] == path]
        if sub.empty:
            status = 'DISABLED' if (path=='A' and not ENABLE_PATH_A) or (path=='B' and not ENABLE_PATH_B) else 'no trades'
            print(f"  Path {path} : {status}"); continue
        p = sub['pnl']
        wins = p[p > 0]; losses = p[p <= 0]
        print(f"  Path {path} : trades={len(sub)}  wins={len(wins)}  losses={len(losses)}  "
              f"wr={len(wins)/len(sub)*100:.1f}%  "
              f"cum-profit={wins.sum():+.1f} pts  cum-loss={losses.sum():+.1f} pts  "
              f"net={p.sum():+.1f} pts")
        for d in ['long', 'short']:
            dsub = sub[sub['direction'] == d]
            if dsub.empty: continue
            dp = dsub['pnl']; dw = dp[dp > 0]; dl = dp[dp <= 0]
            print(f"          {d.upper():<6}  trades={len(dsub)}  wins={len(dw)}  "
                  f"cum-profit={dw.sum():+.1f}  cum-loss={dl.sum():+.1f}  "
                  f"net={dp.sum():+.1f}  avg={dp.mean():.1f}")

    print(f"\n{'─'*62}")
    print("  v5 S/R CONFIG SUMMARY")
    print(f"{'─'*62}")
    print(f"  Entry wall filter           : {SR_ENTRY_WALL_PTS} pts  "
          f"({'ON' if SR_ENTRY_WALL_PTS > 0 else 'OFF'})")
    print(f"  SR Target exit              : {'ON' if ENABLE_SR_TARGET_EXIT else 'OFF'}  "
          f"(approach within {SR_TARGET_APPROACH_PTS} pts of zone edge)")
    print(f"  Tight trail near zone       : {'ON' if ENABLE_SR_TIGHT_TRAIL else 'OFF'}  "
          f"(within {SR_TIGHTEN_WITHIN_PTS} pts → lookback {SR_TIGHT_LOOKBACK} bars)")
    print(f"  Zone band                   : ±{SR_ZONE_HALF_BAND:.0f} pts ({SR_ZONE_HALF_BAND*2:.0f} pt total)")
    print(f"  Pivot window                : {SR_LEFT_BARS}L / {SR_RIGHT_BARS}R bars")
    print(f"  Cluster tolerance           : {SR_CLUSTER_TOL} pts")
    print(f"  Min wick touches / sessions : {SR_MIN_TOUCHES} / {SR_MIN_SESSIONS}")
    print()


# ══════════════════════════════════════════════════════════════════
#   HTML REPORT  (v5 — adds SR zone overlay to trade chart)
# ══════════════════════════════════════════════════════════════════

def build_html_report(trades_df: pd.DataFrame, raw_df: pd.DataFrame,
                      daily_sr: dict) -> str:
    CONTEXT = 10

    def row_to_dict(r):
        ei    = int(r['entry_idx'])
        xi    = int(r['exit_idx'])
        start = max(0, ei - CONTEXT)
        end   = min(len(raw_df), xi + CONTEXT + 1)
        chunk = raw_df.iloc[start:end][
            ['timestamp', 'open', 'high', 'low', 'close',
             'ema_fast', 'ema_slow', 'ema_macro']
        ].copy()
        chunk['timestamp'] = chunk['timestamp'].dt.strftime('%H:%M %d%b')

        # Attach S/R zones valid for this trade's date
        trade_date = r['entry_time'].date()
        zones      = daily_sr.get(trade_date, [])
        zone_lines = [{'price': z['price'], 'upper': z['upper'],
                       'lower': z['lower'], 'type': 'zone'} for z in zones]

        return {
            'direction':         r['direction'],
            'entry_path':        r['entry_path'],
            'entry_time':        fmt_ts(r['entry_time']),
            'exit_time':         fmt_ts(r['exit_time']),
            'entry_price':       round(r['entry_price'], 2),
            'exit_price':        round(r['exit_price'],  2),
            'sl_at_entry':       round(r['sl_at_entry'], 2),
            'sl_at_exit':        round(r['sl_at_exit'],  2),
            'be_triggered':      bool(r['be_triggered']),
            'pnl':               round(r['pnl'],     2),
            'mfe':               round(r['mfe_pts'], 2),
            'mae':               round(r['mae_pts'], 2),
            'exit_reason':       r['exit_reason'],
            'candles':           chunk.to_dict(orient='records'),
            'rel_entry':         ei - start,
            'rel_exit':          xi - start,
            'sr_zones':          zone_lines,
            'sr_target_was_set': bool(r.get('sr_target_was_set', False)),
        }

    trade_data   = [row_to_dict(r) for _, r in trades_df.iterrows()]
    trade_json   = json.dumps(trade_data)
    metrics      = compute_metrics(trades_df)
    metrics_rows = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>'
                           for k, v in metrics.items())

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
        pnl_cls = 'win' if t['pnl'] > 0 else 'loss'
        dir_cls = 'long' if t['direction'] == 'long' else 'short'
        path_cls = 'path-a' if t['entry_path'] == 'A' else 'path-b'
        sign     = '+' if t['pnl'] > 0 else ''
        sr_badge = '<span class="badge sr-exit">SR✓</span> ' if t['exit_reason'] == 'SR_TARGET_EXIT' else ''
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
          <td>{sr_badge}<span class="badge">{t['exit_reason']}</span></td>
        </tr>"""

    total_pnl  = metrics.get('Total P&L (pts)', 0)
    pnl_color  = 'green' if total_pnl >= 0 else 'red'
    ps         = path_summary
    pa_status  = 'ENABLED' if ENABLE_PATH_A else 'DISABLED'
    pb_status  = 'ENABLED' if ENABLE_PATH_B else 'DISABLED'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Speed Demon v5 – Trade Report</title>
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
  .v5tag{{background:#0ea5e922;color:#38bdf8;border:1px solid #0ea5e955;
          padding:2px 7px;border-radius:3px;font-size:9px;margin-left:8px}}
  .sr-bar{{background:#0d1117;border-bottom:1px solid #1e2336;
           padding:6px 20px;font-size:9px;color:#64748b;display:flex;gap:24px}}
  .sr-bar span{{color:#94a3b8}}
</style>
</head>
<body>
<header>
  <div>
    <h1>SPEED DEMON SCALPER v5 — Trade Report <span class="v5tag">S/R INTEGRATED</span></h1>
    <div class="sub">Swing trail + S/R zone exits · Entry wall filter {SR_ENTRY_WALL_PTS}pts · 
      Zone target exit within {SR_TARGET_APPROACH_PTS}pts · 
      Tight trail within {SR_TIGHTEN_WITHIN_PTS}pts of zone · 
      Zones built from prior-day data only (no look-ahead)</div>
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
      <span class="status-badge status-{'enabled' if ENABLE_PATH_A else 'disabled'}">{pa_status}</span>
    </div>
    <div class="prow">Trades: <span>{ps['A']['trades']}</span> &nbsp;|&nbsp;
      Profit: <span class="profit-val">{ps['A']['cum_profit']:+.1f}</span> &nbsp;|&nbsp;
      Loss: <span class="loss-val">{ps['A']['cum_loss']:+.1f}</span> &nbsp;|&nbsp;
      Net: <span class="net-val">{ps['A']['net']:+.1f} pts</span></div>
  </div>
  <div class="path-card pb">
    <div class="ph">PATH B — EMA9 Pullaway 50pts
      <span class="status-badge status-{'enabled' if ENABLE_PATH_B else 'disabled'}">{pb_status}</span>
    </div>
    <div class="prow">Trades: <span>{ps['B']['trades']}</span> &nbsp;|&nbsp;
      Profit: <span class="profit-val">{ps['B']['cum_profit']:+.1f}</span> &nbsp;|&nbsp;
      Loss: <span class="loss-val">{ps['B']['cum_loss']:+.1f}</span> &nbsp;|&nbsp;
      Net: <span class="net-val">{ps['B']['net']:+.1f} pts</span></div>
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
    <canvas id="tradeCanvas" height="270"></canvas>
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
    `Trade #${{idx+1}} · ${{t.direction.toUpperCase()}} · Path ${{t.entry_path}} · ${{t.entry_time}} → ${{t.exit_time}}`;
  drawChart(t);
  const dg=document.getElementById('details-grid');
  const rows=[
    ['Entry Price',  t.entry_price.toFixed(2)],
    ['Exit Price',   t.exit_price.toFixed(2)],
    ['Entry Path',   'Path '+t.entry_path+(t.entry_path==='A'?' (Retest)':' (Pullaway)')],
    ['P&L',          (t.pnl>0?'+':'')+t.pnl.toFixed(2)+' pts'],
    ['MFE',          '+'+t.mfe.toFixed(2)+' pts'],
    ['MAE',          '-'+t.mae.toFixed(2)+' pts'],
    ['SL at Entry',  t.sl_at_entry.toFixed(2)],
    ['SL at Exit',   t.sl_at_exit.toFixed(2)],
    ['Break-Even',   t.be_triggered?'YES ✓':'NO'],
    ['SR Target Set',t.sr_target_was_set?'YES ✓':'NO'],
    ['Exit Reason',  t.exit_reason],
    ['Entry Time',   t.entry_time],
    ['Exit Time',    t.exit_time],
    ['SR Zones Today', t.sr_zones.length+' zones'],
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
  const W=canvas.offsetWidth||620,CH=270;
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
  // Include SR zone prices in scale
  t.sr_zones.forEach(z=>allP.push(z.upper,z.lower));
  const pMin=Math.min(...allP)-2,pMax=Math.max(...allP)+2;

  const xS=i=>PL+(i+0.5)*(cW/n);
  const yS=p=>PT+cH-((p-pMin)/(pMax-pMin))*cH;
  const bw=Math.max(3,(cW/n)*0.55);

  // Grid
  ctx.font='9px monospace'; ctx.textAlign='right';
  for(let g=0;g<=5;g++){{
    const f=g/5,y=PT+cH*(1-f),p=pMin+f*(pMax-pMin);
    ctx.strokeStyle='#1e2336'; ctx.lineWidth=1; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(PL,y); ctx.lineTo(W-PR,y); ctx.stroke();
    ctx.fillStyle='#4a5568'; ctx.fillText(p.toFixed(0),PL-3,y+4);
  }}

  // S/R zone bands — draw BEHIND candles
  t.sr_zones.forEach(z=>{{
    const isVis = (z.upper >= pMin && z.lower <= pMax);
    if(!isVis)return;
    const y1=yS(z.upper),y2=yS(z.lower);
    // Colour: above entry = resistance (red tint), below = support (green tint)
    const isRes = z.price > t.entry_price;
    ctx.fillStyle=isRes?'rgba(239,83,80,0.08)':'rgba(38,166,154,0.08)';
    ctx.fillRect(PL,Math.min(y1,y2),cW,Math.abs(y2-y1));
    ctx.strokeStyle=isRes?'rgba(239,83,80,0.35)':'rgba(38,166,154,0.35)';
    ctx.lineWidth=1; ctx.setLineDash([3,3]);
    const ymid=yS(z.price);
    ctx.beginPath(); ctx.moveTo(PL,ymid); ctx.lineTo(W-PR,ymid); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=isRes?'rgba(239,83,80,0.7)':'rgba(38,166,154,0.7)';
    ctx.font='7px monospace'; ctx.textAlign='left';
    ctx.fillText((isRes?'R':'S')+z.price.toFixed(0),W-PR+2,ymid+3);
    ctx.textAlign='right';
  }});

  // Entry / exit vertical lines
  [t.rel_entry,t.rel_exit].forEach((ri,k)=>{{
    ctx.strokeStyle=k===0?'#4ade8055':'#f8717155';
    ctx.lineWidth=1.5; ctx.setLineDash([4,2]);
    ctx.beginPath(); ctx.moveTo(xS(ri),PT); ctx.lineTo(xS(ri),PT+cH); ctx.stroke();
    ctx.setLineDash([]);
  }});

  // Static lines
  const hLine=(p,col,lbl)=>{{
    if(p<pMin||p>pMax)return;
    const y=yS(p);
    ctx.strokeStyle=col; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(PL,y); ctx.lineTo(W-PR,y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=col; ctx.font='8px monospace'; ctx.textAlign='left';
    ctx.fillText(lbl,W-PR+2,y+3); ctx.textAlign='right';
  }};
  hLine(t.sl_at_entry,'#ef444488','SL');
  hLine(t.entry_price,'#4ade8033','EP');

  // EMAs
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

  // Candles
  cs.forEach((c,i)=>{{
    const x=xS(i),bull=c.close>=c.open,col=bull?'#22c55e':'#ef4444';
    ctx.strokeStyle=col; ctx.lineWidth=1; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(x,yS(c.high)); ctx.lineTo(x,yS(c.low)); ctx.stroke();
    const by=yS(Math.max(c.open,c.close)),bh=Math.max(1,Math.abs(yS(c.open)-yS(c.close)));
    ctx.fillStyle=col; ctx.fillRect(x-bw/2,by,bw,bh);
  }});

  // Entry / exit dots
  [[t.rel_entry,t.entry_price,'#4ade80','E'],[t.rel_exit,t.exit_price,'#f87171','X']]
    .forEach(([ri,price,col,lbl])=>{{
      const x=xS(ri),y=yS(price);
      ctx.fillStyle=col;
      ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2); ctx.fill();
      ctx.fillStyle=col; ctx.font='bold 9px monospace'; ctx.textAlign='center';
      ctx.fillText(`${{lbl}} ${{price.toFixed(0)}}`,x,y+16);
    }});

  // X-axis labels
  const step=Math.max(1,Math.floor(n/6));
  ctx.fillStyle='#4a5568'; ctx.font='8px monospace'; ctx.textAlign='center';
  cs.forEach((c,i)=>{{ if(i%step!==0)return; ctx.fillText(c.timestamp,xS(i),CH-4); }});

  // Legend
  const leg=[['EMA'+EMA_FAST,'#f59e0b'],['EMA'+EMA_SLOW,'#60a5fa'],
             ['EMA'+EMA_MACRO,'#a78bfa'],['Entry','#4ade80'],['Exit','#f87171'],
             ['S/R','rgba(148,163,184,0.6)']];
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


def save_and_open_report(trades_df, raw_df, daily_sr, basename):
    html = build_html_report(trades_df, raw_df, daily_sr)
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
    print("  SPEED DEMON SCALPER  v5  — S/R ZONE INTEGRATED")
    print(f"{BAR}")

    cli_args = sys.argv[1:]
    mode     = 'csv' if cli_args else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\nData source : yfinance  |  Symbol : {YFINANCE_SYMBOL}")
        df        = fetch_yfinance('5m', '5m')
        save_base = os.path.join(os.getcwd(),
                                 YFINANCE_SYMBOL.replace('^','').replace('.','_') + '_speed_demon_v5')
    elif mode == 'csv':
        path5 = cli_args[0] if cli_args else CSV_5MIN_PATH
        if not os.path.exists(path5):
            print(f"ERROR: File not found: {path5}"); sys.exit(1)
        print(f"\nLoading 5m CSV : {path5}")
        df        = load_csv(path5, '5m')
        save_base = os.path.splitext(path5)[0] + '_speed_demon_v5'
    else:
        print(f"ERROR: Unknown DATA_MODE '{DATA_MODE}'"); sys.exit(1)

    print(f"\n{'─'*62}")
    print("  v5 S/R CONFIG")
    print(f"{'─'*62}")
    print(f"  Entry wall filter (skip if zone <N pts away) : {SR_ENTRY_WALL_PTS} pts")
    print(f"  SR Target exit (exit within N pts of zone)   : "
          f"{'ON' if ENABLE_SR_TARGET_EXIT else 'OFF'} / {SR_TARGET_APPROACH_PTS} pts")
    print(f"  Tight trail near zone (within N pts)         : "
          f"{'ON' if ENABLE_SR_TIGHT_TRAIL else 'OFF'} / {SR_TIGHTEN_WITHIN_PTS} pts "
          f"→ lookback {SR_TIGHT_LOOKBACK} bars")
    print(f"  Zone band                                    : "
          f"±{SR_ZONE_HALF_BAND:.0f} pts ({SR_ZONE_HALF_BAND*2:.0f} pt total)")
    print(f"  Pivot window (bars each side)                : {SR_LEFT_BARS}")
    print(f"  Min touches / sessions / rejections          : "
          f"{SR_MIN_TOUCHES} / {SR_MIN_SESSIONS} / {SR_MIN_REJECTIONS}")
    print(f"{'─'*62}")

    print(f"\nComputing 5m indicators ...")
    df = compute_indicators_5m(df)
    print(f"Computing 15m HTF bias ...")
    df = compute_htf_bias(df)

    print(f"\nBuilding daily S/R snapshots (prior-day only — no look-ahead) ...")
    daily_sr = precompute_daily_sr(df)

    print("Running backtest ...\n")
    trades_df, equity_df, sr_stats = run_backtest(df, daily_sr)

    print(f"\n{BAR}")
    print("  BACKTEST RESULTS")
    print(f"{BAR}")

    if trades_df.empty:
        print("\n  No trades generated.")
        print("  Try: reduce ADX_MIN, SPREAD_PCT_MIN, PRICE_GAP_MIN, or SR_ENTRY_WALL_PTS.")
        return

    print_results(trades_df, sr_stats)

    trades_out = save_base + '_trades.csv'
    equity_out = save_base + '_equity.csv'
    trades_df.to_csv(trades_out, index=False)
    equity_df.to_csv(equity_out, index=False)
    print(f"\n{'─'*62}")
    print(f"  Trades saved : {trades_out}")
    print(f"  Equity curve : {equity_out}")

    save_and_open_report(trades_df, df, daily_sr, save_base)
    print(f"{BAR}\n")


if __name__ == '__main__':
    main()