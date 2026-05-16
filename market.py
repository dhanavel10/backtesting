"""
╔══════════════════════════════════════════════════════════════════╗
║   SPEED DEMON SCALPER  v3 + MARKET STRUCTURE  (v2)              ║
╠══════════════════════════════════════════════════════════════════╣
║  NEW vs v3:                                                       ║
║  ✦ Market Structure Trend Gate: longs only in HH/HL uptrend,    ║
║    shorts only in LH/LL downtrend                                ║
║  ✦ S/R Zone Proximity Filter: entry must be near a zone edge    ║
║  ✦ Zone-Anchored Stop Loss: SL placed at zone boundary          ║
║  ✦ Report shows: structure_trend, zone_sl_used columns          ║
╠══════════════════════════════════════════════════════════════════╣
║  Usage                                                            ║
║    python ema_backtest_v2.py              <- yfinance             ║
║    python ema_backtest_v2.py 5m.csv       <- CSV file            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import sys, os, json, webbrowser
from datetime import time
from pathlib import Path

# ── Import market structure helpers ─────────────────────────────
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
from market_analysis import (
    find_pivots, enforce_alternation, classify_market_structure,
    find_sr_zones, compute_atr as _compute_atr_ma,
)

# ══════════════════════════════════════════════════════════════════
#   CONFIGURATION  (identical to v3)
# ══════════════════════════════════════════════════════════════════
DATA_MODE        = 'yfinance'
YFINANCE_SYMBOL  = '^NSEI'
YFINANCE_DAYS    = 60
YFINANCE_TZ      = 'Asia/Kolkata'
CSV_5MIN_PATH    = 'nifty_5m.csv'

EMA_FAST         = 9
EMA_SLOW         = 21
EMA_MACRO        = 50
HTF_EMA_FAST     = 9
HTF_EMA_SLOW     = 21
HTF_EMA_TREND    = 50
ATR_PERIOD       = 14
ADX_PERIOD       = 14
ADX_MIN          = 18

LONG_ATR_SL_MULT    = 1.2
SHORT_ATR_SL_MULT   = 0.75
LONG_BE_PCT         = 0.2
SHORT_BE_PCT        = 0.3
TRAIL_ATR_MULT      = 0.6
ENABLE_EMA_CROSS_EXIT = True
ENABLE_SLOPE_EXIT     = True
SLOPE_EXIT_CANDLES    = 4
SHORT_CONFIRM_BARS    = 3
SPREAD_PCT_MIN        = 0.04
SLOPE_CANDLES         = 6
SLOPE_MIN             = 0.0
PRICE_GAP_MIN         = 5.0
MIDDAY_SPREAD_MULT    = 2.0
RETEST_ATR_MULT       = 0.25
EMA9_PULLAWAY_PTS     = 50
MAX_TRADES_PER_DAY    = 3
MAX_CONSEC_LOSSES     = 2
ENABLE_LONG           = True
ENABLE_SHORT          = True
ENABLE_MIDDAY         = True
ENABLE_EURO           = False
OBSERVE_START  = time(9,  15);  OBSERVE_END  = time(9,  30)
PRIME_START    = time(9,  30);  PRIME_END    = time(10, 30)
MIDDAY_START   = time(11, 30);  MIDDAY_END   = time(13, 30)
EURO_START     = time(14, 15);  EURO_END     = time(15,  0)
SQUAREOFF_START= time(15,  0)
EOD_HARD_EXIT  = time(15, 25);  EOD_SOFT_EXIT = time(15,  0)

# ── NEW: Market Structure Configuration ─────────────────────────
ENABLE_STRUCTURE_FILTER  = True   # master toggle
STRUCTURE_BARS_LEFT      = 5     # pivot window (smaller = faster, more pivots)
STRUCTURE_BARS_RIGHT     = 5
ZONE_ATR_MULT            = 1.5   # zone clustering threshold (× ATR) - slightly wider to catch more touches
ZONE_MIN_TOUCHES         = 2     # min pivot touches to form a zone
ZONE_PROXIMITY_ATR       = 3.0   # price within N×ATR of a zone edge to qualify (loosened from 2.0 to 3.0)
USE_ZONE_SL              = True  # anchor SL to zone boundary (tighter)
ZONE_SL_BUFFER_ATR       = 0.8   # extra buffer beyond zone edge for SL
STRICT_TREND_FILTER      = False # If True, requires HH/HL for longs. If False, just prevents buying into Resistance.


# ══════════════════════════════════════════════════════════════════
#   INDICATOR ENGINE  (same as v3)
# ══════════════════════════════════════════════════════════════════
def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_atr(df, period):
    h = df['high'].astype(float); lo = df['low'].astype(float)
    c = df['close'].astype(float); pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_adx(df, period):
    h = df['high'].astype(float); lo = df['low'].astype(float)
    up = h.diff(); dn = -(lo.diff())
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_v = compute_atr(df, period)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_v.replace(0, np.nan)
    ndi = 100 * pd.Series(ndm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_v.replace(0, np.nan)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def compute_indicators_5m(df):
    df = df.copy()
    df['ema_fast']  = compute_ema(df['close'], EMA_FAST)
    df['ema_slow']  = compute_ema(df['close'], EMA_SLOW)
    df['ema_macro'] = compute_ema(df['close'], EMA_MACRO)
    df['atr']       = compute_atr(df, ATR_PERIOD)
    df['adx']       = compute_adx(df, ADX_PERIOD)
    df['ema_slow_slope']      = df['ema_slow'].diff(SLOPE_CANDLES)
    df['ema_slow_slope_exit'] = df['ema_slow'].diff(SLOPE_EXIT_CANDLES)
    bc = (df['ema_fast'] < df['ema_slow']).astype(int)
    df['consec_bearish_bars'] = bc.groupby((bc != bc.shift()).cumsum()).cumcount() + 1
    df['consec_bearish_bars'] = df['consec_bearish_bars'] * bc
    return df

def compute_htf_bias(df_5m):
    df_5m = df_5m.copy().set_index('timestamp')
    df_15m = df_5m['close'].resample('15min').ohlc().dropna()
    df_15m.columns = ['open', 'high', 'low', 'close']
    df_15m['htf_ef']  = compute_ema(df_15m['close'], HTF_EMA_FAST)
    df_15m['htf_es']  = compute_ema(df_15m['close'], HTF_EMA_SLOW)
    df_15m['htf_et']  = compute_ema(df_15m['close'], HTF_EMA_TREND)
    df_15m['htf_long_bias']  = (df_15m['htf_ef'] > df_15m['htf_es']) & (df_15m['htf_es'] > df_15m['htf_et'])
    df_15m['htf_short_bias'] = (df_15m['htf_ef'] < df_15m['htf_es']) & (df_15m['htf_es'] < df_15m['htf_et'])
    htf = df_15m[['htf_ef','htf_es','htf_et','htf_long_bias','htf_short_bias']].reindex(df_5m.index, method='ffill')
    return df_5m.join(htf).reset_index()


# ══════════════════════════════════════════════════════════════════
#   MARKET STRUCTURE CONTEXT  (new in v2)
# ══════════════════════════════════════════════════════════════════

def build_market_context(df: pd.DataFrame, atr_val: float) -> dict:
    """
    Pre-compute, for each bar index:
        structure_trend  : 'up' | 'down' | None
        last_label       : 'HH' | 'HL' | 'LH' | 'LL' | ''

    Returns:
        context_by_pos   : list[dict]  indexed by bar position
        sr_zones         : list[dict]  (for SL anchoring + zone proximity)
    """
    # Compute raw pivots + market structure on whole df
    raw_pivots    = find_pivots(df, STRUCTURE_BARS_LEFT, STRUCTURE_BARS_RIGHT)
    alt_pivots    = enforce_alternation(raw_pivots)
    labeled, _    = classify_market_structure(alt_pivots)

    # For zone computation we need the full raw pivots
    sr_zones      = find_sr_zones(raw_pivots, atr_val, ZONE_MIN_TOUCHES)

    n = len(df)

    # For each confirmed pivot position, store its label
    # A pivot at position `pos` is confirmed only after `pos + STRUCTURE_BARS_RIGHT` candles
    confirmed_events = []   # (confirmed_at_bar, trend, label)
    for _, row in labeled.iterrows():
        pos           = int(row['pos'])
        confirmed_at  = pos + STRUCTURE_BARS_RIGHT
        confirmed_events.append((confirmed_at, row.get('trend'), row.get('label', '')))

    confirmed_events.sort(key=lambda x: x[0])

    # Build per-bar lookup
    context_by_pos = [{'trend': None, 'last_label': ''} for _ in range(n)]
    cur_trend = None
    cur_label = ''
    ev_idx    = 0
    for i in range(n):
        while ev_idx < len(confirmed_events) and confirmed_events[ev_idx][0] <= i:
            cur_trend = confirmed_events[ev_idx][1]
            cur_label = confirmed_events[ev_idx][2]
            ev_idx   += 1
        context_by_pos[i] = {'trend': cur_trend, 'last_label': cur_label}

    return context_by_pos, sr_zones


def nearest_zone(close: float, zones: list, atr: float, side: str):
    """
    Find the nearest zone on the given side ('support' or 'resistance').
    support   = zones whose center is BELOW close
    resistance= zones whose center is ABOVE close

    Returns the zone dict or None.
    """
    proximity = atr * ZONE_PROXIMITY_ATR
    candidates = []
    for z in zones:
        dist = abs(z['center'] - close)
        if side == 'support'    and z['center'] < close  and dist <= proximity:
            candidates.append((dist, z))
        elif side == 'resistance' and z['center'] > close and dist <= proximity:
            candidates.append((dist, z))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[0][1]


def structure_allows_long(idx: int, close: float, atr: float,
                           ctx: list, zones: list) -> tuple:
    """
    Returns (allowed: bool, reason: str, zone: dict|None)

    Conditions:
      1. OPTIONAL: Structure trend = 'up'  (last confirmed pivot was HH or HL)
      2. REQUIRED: Price is NOT pressing into a strong resistance zone from below
      3. PREFERRED: Price is near a support zone
    """
    if not ENABLE_STRUCTURE_FILTER:
        return True, 'FILTER_OFF', None

    c = ctx[idx]
    if STRICT_TREND_FILTER and c['trend'] != 'up':
        return False, f"STRUCT_BEARISH({c['last_label']})", None

    # Check resistance: if price is pressing into resistance → skip
    r_zone = nearest_zone(close, zones, atr, 'resistance')
    if r_zone and close >= r_zone['bottom'] - atr * 0.5: # increased buffer to 0.5 ATR
        return False, 'AT_RESISTANCE', None

    # Prefer entries near a support zone (bounce)
    s_zone = nearest_zone(close, zones, atr, 'support')
    if s_zone:
        return True, f"NEAR_SUPPORT({s_zone['center']:.0f})", s_zone
        
    if STRICT_TREND_FILTER:
        return True, 'STRUCTURE_UP', None
    else:
        return True, 'NO_RESISTANCE', None


def structure_allows_short(idx: int, close: float, atr: float,
                            ctx: list, zones: list) -> tuple:
    """
    Returns (allowed: bool, reason: str, zone: dict|None)

    Conditions:
      1. OPTIONAL: Structure trend = 'down'  (last confirmed pivot was LH or LL)
      2. REQUIRED: Price is NOT pressing into a strong support zone from above
      3. PREFERRED: Price is near a resistance zone
    """
    if not ENABLE_STRUCTURE_FILTER:
        return True, 'FILTER_OFF', None

    c = ctx[idx]
    if STRICT_TREND_FILTER and c['trend'] != 'down':
        return False, f"STRUCT_BULLISH({c['last_label']})", None

    # Check support: if price is pressing into support → skip
    s_zone = nearest_zone(close, zones, atr, 'support')
    if s_zone and close <= s_zone['top'] + atr * 0.5: # increased buffer to 0.5 ATR
        return False, 'AT_SUPPORT', None

    # Prefer entries near a resistance zone (rejection)
    r_zone = nearest_zone(close, zones, atr, 'resistance')
    if r_zone:
        return True, f"NEAR_RESISTANCE({r_zone['center']:.0f})", r_zone
        
    if STRICT_TREND_FILTER:
        return True, 'STRUCTURE_DOWN', None
    else:
        return True, 'NO_SUPPORT', None


def zone_anchored_sl(close: float, direction: str, atr: float,
                     zone, atr_sl: float) -> float:
    """
    If USE_ZONE_SL and a reference zone is available, place SL at zone edge
    with a buffer.  Always keeps the SL safer (tighter without being too tight).
    """
    if not USE_ZONE_SL or zone is None:
        return atr_sl

    buf = atr * ZONE_SL_BUFFER_ATR
    if direction == 'long':
        # SL below the zone bottom
        z_sl = zone['bottom'] - buf
        # Use whichever is tighter (closer to entry) but dont go below ATR floor
        return max(atr_sl, z_sl)
    else:
        # SL above the zone top
        z_sl = zone['top'] + buf
        return min(atr_sl, z_sl)


# ══════════════════════════════════════════════════════════════════
#   ANTI-CHOP FILTERS  (same as v3)
# ══════════════════════════════════════════════════════════════════
def get_session(t):
    if OBSERVE_START <= t < OBSERVE_END:  return 'observe'
    if PRIME_START   <= t < PRIME_END:    return 'prime'
    if MIDDAY_START  <= t < MIDDAY_END:   return 'midday'
    if EURO_START    <= t < EURO_END:     return 'euro'
    if SQUAREOFF_START <= t:              return 'squareoff'
    return 'outside'

def chop_filters_pass(close, ef, es, slope, adx_v, session):
    if adx_v < ADX_MIN:                return False
    spread = abs(ef - es) / close * 100
    thresh = SPREAD_PCT_MIN * (MIDDAY_SPREAD_MULT if session == 'midday' else 1.0)
    if spread < thresh:                return False
    if abs(slope) <= SLOPE_MIN:        return False
    if abs(close - es) < PRICE_GAP_MIN: return False
    return True


# ══════════════════════════════════════════════════════════════════
#   DATA LOADERS  (same as v3)
# ══════════════════════════════════════════════════════════════════
def _standardise(df, tz):
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(tz)
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert(tz)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[(df['timestamp'].dt.time >= time(9, 15)) &
            (df['timestamp'].dt.time <= time(15, 30))].reset_index(drop=True)
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
    return df

def fetch_yfinance(interval, label):
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance"); sys.exit(1)
    import warnings
    print(f"  Fetching {YFINANCE_SYMBOL} {interval} (last {YFINANCE_DAYS}d) ...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = yf.download(tickers=YFINANCE_SYMBOL, period=f'{YFINANCE_DAYS}d',
                          interval=interval, progress=False, auto_adjust=True)
    if raw.empty:
        print(f"ERROR: No data for '{YFINANCE_SYMBOL}'."); sys.exit(1)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={'adj close': 'close', 'adj_close': 'close'})
    if 'volume' not in raw.columns: raw['volume'] = 0
    raw = raw[['open', 'high', 'low', 'close', 'volume']].dropna()
    if raw.index.tz is None: raw.index = raw.index.tz_localize('UTC')
    raw.index = raw.index.tz_convert(YFINANCE_TZ)
    df = raw.reset_index()
    ts_col = next(c for c in df.columns if c.lower() in ('datetime', 'date', 'timestamp'))
    df = df.rename(columns={ts_col: 'timestamp'})
    df = _standardise(df, YFINANCE_TZ)
    print(f"    [{label}] Rows: {len(df)}")
    return df

def load_csv(filepath, label=''):
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    ts_col = next((c for c in ['timestamp','datetime','date_time','time','date']
                   if c in df.columns), None)
    if not ts_col: raise ValueError(f"No timestamp col. Got: {list(df.columns)}")
    df['timestamp'] = pd.to_datetime(df[ts_col], dayfirst=True)
    if ts_col != 'timestamp': df = df.drop(columns=[ts_col])
    for col in ['open','high','low','close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open','high','low','close'])
    df = _standardise(df, 'Asia/Kolkata')
    print(f"  [{label}] Rows: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════
#   BACKTEST ENGINE  (v2 — with structure gates)
# ══════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, ctx: list, zones: list) -> tuple:
    closes         = df['close'].astype(float).values
    highs          = df['high'].astype(float).values
    lows           = df['low'].astype(float).values
    ema_fast       = df['ema_fast'].values
    ema_slow       = df['ema_slow'].values
    ema_macro      = df['ema_macro'].values
    atrs           = df['atr'].values
    adx_vals       = df['adx'].values
    slopes         = df['ema_slow_slope'].values
    slopes_exit    = df['ema_slow_slope_exit'].values
    consec_bearish = df['consec_bearish_bars'].values
    htf_long_bias  = df['htf_long_bias'].values
    ts_list        = df['timestamp'].tolist()
    n              = len(df)

    trades = []; equity = 0.0; eq_curve = []
    opening_high = {}; opening_close = {}
    in_trade = False; direction = None
    entry_price = 0.0; entry_time = None; entry_idx = -1; entry_path = ''
    struct_trend_at_entry = ''; struct_reason_at_entry = ''
    stop_loss = 0.0; be_triggered = False; be_level = 0.0
    trail_active = False; sl_dist_initial = 0.0
    trade_max_favor = 0.0; trade_max_adverse = 0.0
    zone_sl_used = False
    prev_date = None; daily_trades = {}; daily_consec_loss = {}

    def do_enter(dir_str, close_price, ts_now, atr_val, idx, path, s_trend, s_reason, zone):
        nonlocal in_trade, direction, entry_price, entry_time, entry_idx, entry_path
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, trade_max_favor, trade_max_adverse
        nonlocal struct_trend_at_entry, struct_reason_at_entry, zone_sl_used

        sl_mult   = LONG_ATR_SL_MULT if dir_str == 'long' else SHORT_ATR_SL_MULT
        atr_sl_dist = atr_val * sl_mult

        in_trade    = True; direction = dir_str
        entry_price = close_price; entry_time = ts_now; entry_idx = idx
        entry_path  = path; be_triggered = False; trail_active = False
        sl_dist_initial = atr_sl_dist
        trade_max_favor = trade_max_adverse = 0.0
        struct_trend_at_entry  = s_trend
        struct_reason_at_entry = s_reason

        if direction == 'long':
            atr_sl   = entry_price - atr_sl_dist
            final_sl = zone_anchored_sl(entry_price, 'long', atr_val, zone, atr_sl)
            be_level = entry_price * (1 + LONG_BE_PCT / 100)
        else:
            atr_sl   = entry_price + atr_sl_dist
            final_sl = zone_anchored_sl(entry_price, 'short', atr_val, zone, atr_sl)
            be_level = entry_price * (1 - SHORT_BE_PCT / 100)

        stop_loss    = final_sl
        zone_sl_used = USE_ZONE_SL and zone is not None
        # Recalc sl_dist_initial from actual stop (for reporting)
        sl_dist_initial = abs(entry_price - stop_loss)

    def do_exit(exit_price, ts_now, reason, exit_idx):
        nonlocal in_trade, direction, entry_price, entry_time, entry_idx, entry_path
        nonlocal stop_loss, be_triggered, be_level, trail_active, sl_dist_initial, equity
        nonlocal trade_max_favor, trade_max_adverse, zone_sl_used
        nonlocal struct_trend_at_entry, struct_reason_at_entry

        pnl = round((exit_price - entry_price) if direction == 'long'
                    else (entry_price - exit_price), 2)
        equity += pnl
        date_str = str(ts_now.date())
        if pnl <= 0:
            daily_consec_loss[date_str] = daily_consec_loss.get(date_str, 0) + 1
        else:
            daily_consec_loss[date_str] = 0

        trades.append({
            'direction':         direction,
            'entry_path':        entry_path,
            'struct_trend':      struct_trend_at_entry,
            'struct_reason':     struct_reason_at_entry,
            'zone_sl_used':      zone_sl_used,
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

        in_trade = False; direction = None
        entry_price = 0.0; entry_time = None; entry_idx = -1; entry_path = ''
        stop_loss = 0.0; be_triggered = False; be_level = 0.0
        trail_active = False; sl_dist_initial = 0.0
        trade_max_favor = trade_max_adverse = 0.0
        zone_sl_used = False
        struct_trend_at_entry = struct_reason_at_entry = ''

    for idx in range(n):
        close   = closes[idx]; high_c = highs[idx]; low_c = lows[idx]
        ef      = ema_fast[idx]; es = ema_slow[idx]; em = ema_macro[idx]
        atr     = float(atrs[idx])
        adx_v   = float(adx_vals[idx])   if not np.isnan(adx_vals[idx])   else 0.0
        slope   = float(slopes[idx])      if not np.isnan(slopes[idx])      else 0.0
        sl_exit = float(slopes_exit[idx]) if not np.isnan(slopes_exit[idx]) else 0.0
        cb      = int(consec_bearish[idx])
        ts      = ts_list[idx]
        c_time  = ts.time(); c_date = ts.date(); date_str = str(c_date)
        session = get_session(c_time)
        if c_date != prev_date: prev_date = c_date
        if c_time == time(9, 30):
            opening_high[date_str]  = high_c
            opening_close[date_str] = close
        if np.isnan(ef) or np.isnan(es) or np.isnan(em) or np.isnan(atr):
            eq_curve.append({'timestamp': ts, 'equity': equity}); continue

        htf_long = bool(htf_long_bias[idx]) if not pd.isna(htf_long_bias[idx]) else False

        # ── Manage open trade ────────────────────────────────────
        if in_trade:
            favor   = (high_c - entry_price) if direction == 'long' else (entry_price - low_c)
            adverse = (entry_price - low_c)   if direction == 'long' else (high_c - entry_price)
            trade_max_favor   = max(trade_max_favor,   favor)
            trade_max_adverse = max(trade_max_adverse, adverse)

            if not be_triggered:
                if direction == 'long'  and close >= be_level:
                    stop_loss = entry_price + 1.0; be_triggered = True; trail_active = True
                elif direction == 'short' and close <= be_level:
                    stop_loss = entry_price - 1.0; be_triggered = True; trail_active = True

            if trail_active:
                tr_dist = atr * TRAIL_ATR_MULT
                if direction == 'long':  stop_loss = max(stop_loss, close - tr_dist)
                else:                    stop_loss = min(stop_loss, close + tr_dist)

            exit_p = exit_r = None
            if   direction == 'long'  and close <= stop_loss: exit_p = stop_loss; exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'
            elif direction == 'short' and close >= stop_loss: exit_p = stop_loss; exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'
            elif ENABLE_EMA_CROSS_EXIT and be_triggered:
                if   direction == 'long'  and ef < es: exit_p = close; exit_r = 'EMA_CROSS_EXIT'
                elif direction == 'short' and ef > es: exit_p = close; exit_r = 'EMA_CROSS_EXIT'
            elif ENABLE_SLOPE_EXIT and be_triggered:
                if   direction == 'long'  and sl_exit < -SLOPE_MIN: exit_p = close; exit_r = 'SLOPE_REVERSAL'
                elif direction == 'short' and sl_exit > SLOPE_MIN:  exit_p = close; exit_r = 'SLOPE_REVERSAL'
            elif c_time >= EOD_SOFT_EXIT and not be_triggered: exit_p = close; exit_r = 'SOFT_EOD'
            elif c_time >= EOD_HARD_EXIT:                      exit_p = close; exit_r = 'HARD_EOD'
            if exit_p is not None: do_exit(exit_p, ts, exit_r, idx)

        # ── Entry logic ──────────────────────────────────────────
        if not in_trade:
            allowed = ['prime']
            if ENABLE_MIDDAY: allowed.append('midday')
            if ENABLE_EURO:   allowed.append('euro')
            if session not in allowed:
                eq_curve.append({'timestamp': ts, 'equity': equity}); continue
            if daily_trades.get(date_str, 0) >= MAX_TRADES_PER_DAY:
                eq_curve.append({'timestamp': ts, 'equity': equity}); continue
            if daily_consec_loss.get(date_str, 0) >= MAX_CONSEC_LOSSES:
                eq_curve.append({'timestamp': ts, 'equity': equity}); continue
            if not chop_filters_pass(close, ef, es, slope, adx_v, session):
                eq_curve.append({'timestamp': ts, 'equity': equity}); continue

            retest_tol = atr * RETEST_ATR_MULT

            # ── LONG Entry ───────────────────────────────────────
            if ENABLE_LONG and htf_long:
                ok_l, reason_l, zone_l = structure_allows_long(idx, close, atr, ctx, zones)
                if ok_l:
                    pa = (ef > es and close > em and slope > 0 and
                          abs(close - ef) <= retest_tol and close > es)
                    pb = (ef > es and close > em and slope > 0 and
                          (close - ef) >= EMA9_PULLAWAY_PTS and
                          date_str in opening_high and close > opening_high[date_str])
                    if pa or pb:
                        do_enter('long', close, ts, atr, idx,
                                 'A' if pa else 'B',
                                 ctx[idx]['trend'] or '', reason_l, zone_l)
                        daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

            # ── SHORT Entry ──────────────────────────────────────
            if not in_trade and ENABLE_SHORT:
                ok_s, reason_s, zone_s = structure_allows_short(idx, close, atr, ctx, zones)
                if ok_s:
                    pa = (ef < es and close < em and slope < 0 and
                          abs(close - ef) <= retest_tol and close < es and
                          cb >= SHORT_CONFIRM_BARS)
                    pb = (ef < es and close < em and slope < 0 and
                          (ef - close) >= EMA9_PULLAWAY_PTS and
                          cb >= SHORT_CONFIRM_BARS and
                          date_str in opening_close and close < opening_close[date_str])
                    if pa or pb:
                        do_enter('short', close, ts, atr, idx,
                                 'A' if pa else 'B',
                                 ctx[idx]['trend'] or '', reason_s, zone_s)
                        daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

        eq_curve.append({'timestamp': ts, 'equity': equity})

    if in_trade:
        do_exit(closes[-1], ts_list[-1], 'END_OF_DATA', n - 1)

    return pd.DataFrame(trades), pd.DataFrame(eq_curve)


# ══════════════════════════════════════════════════════════════════
#   METRICS & REPORTING
# ══════════════════════════════════════════════════════════════════

def compute_metrics(tdf):
    if tdf.empty: return {'message': 'No trades found.'}
    pnl = tdf['pnl']; wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    total = len(tdf); gp = wins.sum(); gl = abs(losses.sum())
    cum = pnl.cumsum(); max_dd = (cum.cummax() - cum).max()
    return {
        'Total Trades':       total,
        'Winning Trades':     len(wins),
        'Losing Trades':      len(losses),
        'Win Rate (%)':       round(len(wins) / total * 100, 2),
        'Avg Profit (pts)':   round(wins.mean(),   2) if len(wins)   else 0,
        'Avg Loss (pts)':     round(losses.mean(), 2) if len(losses) else 0,
        'Largest Win (pts)':  round(pnl.max(), 2),
        'Largest Loss (pts)': round(pnl.min(), 2),
        'Profit Factor':      round(gp / gl, 2) if gl > 0 else float('inf'),
        'Total P&L (pts)':    round(pnl.sum(), 2),
        'Max Drawdown (pts)': round(max_dd, 2),
        'Avg MFE (pts)':      round(tdf['mfe_pts'].mean(), 2),
        'Avg MAE (pts)':      round(tdf['mae_pts'].mean(), 2),
    }

def fmt_ts(ts):
    return ts.strftime('%d-%m-%Y %H:%M') if hasattr(ts, 'strftime') else str(ts)

def print_results(tdf):
    SEP = '-' * 180
    print(f'\n{SEP}')
    print(f'  TRADE LOG  ({len(tdf)} trades)')
    print(SEP)
    print(f"{'#':<4} {'Dir':<6} {'Path':<5} {'Struct':<7} {'Reason':<26} {'ZoneSL':<7}"
          f" {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>8} {'Exit':>8} {'MFE':>6} {'MAE':>6} {'P&L':>8}  ExitReason")
    print(SEP)
    for i, r in tdf.iterrows():
        zsl = 'YES' if r['zone_sl_used'] else 'no'
        print(f"{i+1:<4} {r['direction'].upper():<6} {('P-'+r['entry_path']):<5}"
              f" {str(r['struct_trend']):<7} {str(r['struct_reason']):<26} {zsl:<7}"
              f" {fmt_ts(r['entry_time']):<22} {fmt_ts(r['exit_time']):<22}"
              f" {r['entry_price']:>8.1f} {r['exit_price']:>8.1f}"
              f" {r['mfe_pts']:>6.1f} {r['mae_pts']:>6.1f}"
              f" {r['pnl']:>+8.2f}  {r['exit_reason']}")

    m = compute_metrics(tdf)
    print(f"\n{'─'*60}\n  PERFORMANCE METRICS\n{'─'*60}")
    for k, v in m.items(): print(f"  {k:<30}: {v}")

    # Structure filter effectiveness
    print(f"\n{'─'*60}\n  STRUCTURE TREND BREAKDOWN\n{'─'*60}")
    for tr in ['up', 'down']:
        sub = tdf[tdf['struct_trend'] == tr]
        if sub.empty: continue
        p = sub['pnl']
        w = (p > 0).sum()
        print(f"  Trend={tr:<5}  trades={len(sub):>3}  wins={w:>3}  "
              f"wr={w/len(sub)*100:.1f}%  net={p.sum():+.1f}  avg={p.mean():+.1f}")

    # Zone SL effectiveness
    print(f"\n{'─'*60}\n  ZONE SL vs ATR SL BREAKDOWN\n{'─'*60}")
    for zv in [True, False]:
        sub = tdf[tdf['zone_sl_used'] == zv]
        if sub.empty: continue
        p = sub['pnl']
        w = (p > 0).sum()
        label = 'Zone-SL' if zv else 'ATR-SL '
        print(f"  {label}  trades={len(sub):>3}  wins={w:>3}  "
              f"wr={w/len(sub)*100:.1f}%  net={p.sum():+.1f}  "
              f"avg_mae={sub['mae_pts'].mean():.1f}")

    print(f"\n{'─'*60}\n  EXIT REASON BREAKDOWN\n{'─'*60}")
    bd = tdf.groupby('exit_reason')['pnl'].agg(count='count',
         total_pnl='sum', avg_pnl='mean').round(2)
    print(bd.to_string())
    print()


# ══════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    BAR  = '=' * 72
    args = sys.argv[1:]
    print(f'\n{BAR}')
    print('  SPEED DEMON  v3 + MARKET STRUCTURE  (v2)')
    print(BAR)

    cli_args = sys.argv[1:]
    mode     = 'csv' if cli_args else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\n  Source: yfinance  |  {YFINANCE_SYMBOL}")
        df        = fetch_yfinance('5m', '5m')
        save_base = YFINANCE_SYMBOL.replace('^','').replace('.','_') + '_v2'
    elif mode == 'csv':
        path5 = cli_args[0] if cli_args else CSV_5MIN_PATH
        if not os.path.exists(path5):
            print(f"ERROR: {path5}"); sys.exit(1)
        print(f"\n  Loading CSV: {path5}")
        df        = load_csv(path5, '5m')
        save_base = os.path.splitext(path5)[0] + '_v2'
    else:
        print(f"ERROR: Unknown DATA_MODE '{DATA_MODE}'"); sys.exit(1)

    # ── Indicators ───────────────────────────────────────────────
    print("  Computing indicators ...")
    df = compute_indicators_5m(df)
    print("  Computing 15m HTF bias ...")
    df = compute_htf_bias(df)

    # ── Market structure context ─────────────────────────────────
    print(f"  Building market structure context"
          f" (barsL={STRUCTURE_BARS_LEFT}, barsR={STRUCTURE_BARS_RIGHT}) ...")
    atr_val = float(df['atr'].dropna().iloc[-1])
    ctx, zones = build_market_context(df, atr_val)
    print(f"    S/R zones found: {len(zones)}")
    print(f"    Structure filter: {'ON' if ENABLE_STRUCTURE_FILTER else 'OFF'}")
    print(f"    Zone SL: {'ON' if USE_ZONE_SL else 'OFF'}")

    # ── Config summary ───────────────────────────────────────────
    print(f"\n{'─'*60}\n  CONFIGURATION  (v2 additions)\n{'─'*60}")
    print(f"  Structure filter          : {'ON' if ENABLE_STRUCTURE_FILTER else 'OFF'}")
    print(f"  Structure pivot window    : L={STRUCTURE_BARS_LEFT}  R={STRUCTURE_BARS_RIGHT}")
    print(f"  Zone ATR mult             : {ZONE_ATR_MULT}")
    print(f"  Zone proximity gate       : {ZONE_PROXIMITY_ATR} × ATR")
    print(f"  Zone-anchored SL          : {'ON' if USE_ZONE_SL else 'OFF'}")
    print(f"  Zone SL buffer            : {ZONE_SL_BUFFER_ATR} × ATR\n")

    # ── Backtest ─────────────────────────────────────────────────
    print("  Running backtest ...")
    trades_df, equity_df = run_backtest(df, ctx, zones)

    print(f'\n{BAR}\n  BACKTEST RESULTS\n{BAR}')
    if trades_df.empty:
        print("\n  No trades generated.")
        print("  Try: reduce ADX_MIN / STRUCTURE_BARS_LEFT / ZONE_MIN_TOUCHES")
        return

    print_results(trades_df)

    trades_csv = save_base + '_trades.csv'
    equity_csv = save_base + '_equity.csv'
    trades_df.to_csv(trades_csv, index=False)
    equity_df.to_csv(equity_csv, index=False)
    print(f"  Trades: {trades_csv}")
    print(f"  Equity: {equity_csv}")
    print(f'\n{BAR}\n')


if __name__ == '__main__':
    main()
