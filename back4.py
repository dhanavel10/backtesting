"""
╔══════════════════════════════════════════════════════════════════╗
║   SPEED DEMON SCALPER  v8  — "BACK TO BASICS"                  ║
╠══════════════════════════════════════════════════════════════════╣
║  THE REAL PROBLEM (7 versions of data now confirms this)        ║
║                                                                  ║
║  Every version: WR ↑, avg_profit ↓, total P&L ↓                ║
║  v3: 37 trades, avg_win=102pts, P&L=1221  ← best               ║
║  v7: 33 trades, avg_win= 43pts, P&L= 705  ← worst              ║
║                                                                  ║
║  Why? Three compounding problems:                                ║
║  1. The same filters that remove losses also remove the          ║
║     setup conditions for 150-250pt winners.                     ║
║  2. The EMA cross exit fires on profitable Tier2 trades         ║
║     mid-trend, killing avg_win. 13 Tier2 TRAIL_SL at 40pts      ║
║     with avg_mfe=61pts = 21pt gap per trade = 273pts lost.      ║
║  3. The running trail still updates every single bar on Tier2.  ║
║     On a 5m chart, close - 0.6×ATR fires during normal          ║
║     intrabar pullbacks even in strong Tier2 trends.             ║
║                                                                  ║
║  v8 PHILOSOPHY: MAX CAPTURE, MINIMAL FILTERS                    ║
║  ─────────────────────────────────────────────────────────────  ║
║                                                                  ║
║  CHANGE 1 — EMA Cross Exit REMOVED entirely                     ║
║    v6 data: Tier2 TRAIL_SL avg=40pts, avg_mfe=61pts             ║
║    The cross exit is responsible for that 21pt gap.             ║
║    Let the running-high trail handle all exits instead.         ║
║    This is the single biggest avg_win improvement possible.     ║
║                                                                  ║
║  CHANGE 2 — Running-high trail for BOTH tiers                   ║
║    Tier2 also switches to running-high trail (same as Tier3)    ║
║    Trail only moves when price makes new extreme since BE.      ║
║    Tier2 trail dist = 0.6×ATR (unchanged, appropriate width)   ║
║    Tier3 trail dist = 1.2×ATR (unchanged, wider)                ║
║    This stops the trail from eating 21pts per winner on Tier2.  ║
║                                                                  ║
║  CHANGE 3 — Midday + Euro sessions RE-ENABLED                  ║
║    Were disabled since v3. With better exits (running-high      ║
║    trail + no cross exit), more trades = more P&L.             ║
║    These sessions add volume without changing the strategy.     ║
║    If results are negative, re-disable per-session.            ║
║                                                                  ║
║  KEPT from v7 (proven correct)                                  ║
║  ✦ Running-high trail with profit floor BE (BUG FIXED)          ║
║  ✦ Tier3 micro-trail disabled (pullbacks too deep)              ║
║  ✦ Tier2 micro-trail (saves ~8pts per rescued trade)            ║
║  ✦ ADX floor=25, Tier3 requires ADX rising                      ║
║  ✦ HTF 15m bias for longs                                       ║
║  ✦ Volume spike + 3-bar confirm for shorts                      ║
║  ✦ 3-stage EOD (unchanged)                                      ║
║  ✦ Consec-loss circuit breaker                                  ║
║                                                                  ║
║  v1→v2→v3→v4→v5→v6→v7→v8                                        ║
║  PF : 2.1→2.76→3.36→3.57→3.50→3.96→?→ ?                       ║
║  P&L: 739→1454→1221→983→833→705→?→  ?                          ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python speed_demon_v8.py              ← yfinance (default)
    python speed_demon_v8.py 5m.csv       ← CSV file
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
EMA_FAST         = 9
EMA_SLOW         = 21
EMA_MACRO        = 100

# ── HTF (15m) EMA Periods ───────────────────────────────────────
HTF_EMA_FAST     = 9
HTF_EMA_SLOW     = 21
HTF_EMA_TREND    = 50

# ── ATR ─────────────────────────────────────────────────────────
ATR_PERIOD       = 14

# ── ADX ─────────────────────────────────────────────────────────
ADX_PERIOD           = 14
ADX_MIN              = 25      # Tier 1 (20-25) banned
ADX_TIER2            = 25      # 25-35 = Tier 2 normal
ADX_TIER3            = 35      # 35+   = Tier 3 strong
ADX_RISING_BARS      = 3
TIER3_REQUIRE_ADX_RISING = True

# ── Asymmetric SL ────────────────────────────────────────────────
LONG_ATR_SL_MULT     = 1.2
SHORT_ATR_SL_MULT    = 0.75

# ── Micro-Trail (Tier 2 only) ─────────────────────────────────────
ENABLE_MICRO_TRAIL      = True
TIER3_MICRO_TRAIL       = False   # Tier3 pullbacks are too deep
MICRO_TRAIL_TRIGGER     = 15.0
MICRO_TRAIL_SL_FACTOR   = 0.5

# ── Profit Floor Break-Even ───────────────────────────────────────
# BE triggers on: +30pts in-favor OR +0.3%
# SL moves to entry ± (initial_sl_distance × lock_ratio)
LONG_BE_PCT          = 0.3
SHORT_BE_PCT         = 0.3
BE_POINTS_TRIGGER    = 30.0
TIER2_BE_LOCK_RATIO  = 0.3    # ~12pt profit floor on typical Nifty trade
TIER3_BE_LOCK_RATIO  = 0.5    # ~20pt profit floor

# ── Running-High Trail (BOTH TIERS now use this) ─────────────────
# Trail only advances when price makes a new extreme since BE.
# SL = running_peak - ATR × trail_mult  (long)
# SL = running_trough + ATR × trail_mult (short)
TIER2_TRAIL_ATR_MULT  = 0.6   # Tier2: standard width
TIER3_TRAIL_ATR_MULT  = 1.2   # Tier3: wider for deeper pullbacks

# ── EMA Cross Exit — REMOVED in v8 ───────────────────────────────
# Responsible for cutting Tier2 TRAIL_SL 21pts short of MFE.
# Running-high trail handles all exits instead.
ENABLE_EMA_CROSS_EXIT = False  # ← OFF

# ── Short Entry Filters ──────────────────────────────────────────
SHORT_CONFIRM_BARS   = 3
VOL_PERIOD           = 20
VOL_SPIKE_MULT       = 1.2
ENABLE_VOL_FILTER    = True

# ── Anti-Chop Filters ────────────────────────────────────────────
SPREAD_PCT_MIN       = 0.04
SLOPE_CANDLES        = 6
SLOPE_MIN            = 0.0
PRICE_GAP_MIN        = 5.0
MIDDAY_SPREAD_MULT   = 2.0

# ── Retest tolerance ─────────────────────────────────────────────
RETEST_ATR_MULT      = 0.15

# ── Daily caps ───────────────────────────────────────────────────
MAX_TRADES_PER_DAY   = 3
MAX_CONSEC_LOSSES    = 2

# ── Directions ───────────────────────────────────────────────────
ENABLE_LONG          = True
ENABLE_SHORT         = True

# ── Session Gates — RE-ENABLED ────────────────────────────────────
# Midday and Euro re-enabled. Better exits mean more trades = more P&L.
# Set to False individually if a session shows negative contribution.
ENABLE_MIDDAY        = True   # ← re-enabled from v3
ENABLE_EURO          = True   # ← re-enabled from v3

# ── Sessions ─────────────────────────────────────────────────────
OBSERVE_START        = time(9,  15)
OBSERVE_END          = time(9,  45)
PRIME_START          = time(9,  45)
PRIME_END            = time(11, 30)
MIDDAY_START         = time(11, 30)
MIDDAY_END           = time(13, 30)
EURO_START           = time(13, 30)
EURO_END             = time(15,  0)
SQUAREOFF_START      = time(15,  0)

# ── 3-Stage EOD ──────────────────────────────────────────────────
EOD_KILL_LOSERS      = time(15,  0)   # kill currently-losing trades
EOD_KILL_NON_TRAIL   = time(15, 10)   # kill trades not yet at BE
EOD_HARD_EXIT        = time(15, 25)   # kill everything


# ══════════════════════════════════════════════════════════════════
#   INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df, period):
    h, l, c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_adx(df, period):
    h, l = df['high'].astype(float), df['low'].astype(float)
    up, dn = h.diff(), -(l.diff())
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = compute_atr(df, period)
    pdm_s = pd.Series(pdm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    mdm_s = pd.Series(mdm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * pdm_s / atr.replace(0, np.nan)
    mdi = 100 * mdm_s / atr.replace(0, np.nan)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def compute_indicators_5m(df):
    df = df.copy()
    df['ema_fast']       = compute_ema(df['close'], EMA_FAST)
    df['ema_slow']       = compute_ema(df['close'], EMA_SLOW)
    df['ema_macro']      = compute_ema(df['close'], EMA_MACRO)
    df['atr']            = compute_atr(df, ATR_PERIOD)
    df['adx']            = compute_adx(df, ADX_PERIOD)
    df['ema_slow_slope'] = df['ema_slow'].diff(SLOPE_CANDLES)
    df['vol_ma']         = df['volume'].rolling(VOL_PERIOD).mean()
    df['adx_prev']       = df['adx'].shift(ADX_RISING_BARS)

    bearish = (df['ema_fast'] < df['ema_slow']).astype(int)
    df['consec_bearish_bars'] = (
        bearish.groupby((bearish != bearish.shift()).cumsum()).cumcount() + 1
    ) * bearish
    return df


def compute_htf_bias(df_5m):
    df = df_5m.copy().set_index('timestamp')
    d15 = df['close'].resample('15min').ohlc().dropna()
    d15.columns = ['open', 'high', 'low', 'close']
    d15['htf_ema_fast']  = compute_ema(d15['close'], HTF_EMA_FAST)
    d15['htf_ema_slow']  = compute_ema(d15['close'], HTF_EMA_SLOW)
    d15['htf_ema_trend'] = compute_ema(d15['close'], HTF_EMA_TREND)
    d15['htf_long_bias'] = (
        (d15['htf_ema_fast'] > d15['htf_ema_slow']) &
        (d15['htf_ema_slow'] > d15['htf_ema_trend'])
    )
    cols = d15[['htf_long_bias']].reindex(df.index, method='ffill')
    return df.join(cols).reset_index()


# ══════════════════════════════════════════════════════════════════
#   FILTERS
# ══════════════════════════════════════════════════════════════════

def get_session(t):
    if OBSERVE_START   <= t < OBSERVE_END:  return 'observe'
    if PRIME_START     <= t < PRIME_END:    return 'prime'
    if MIDDAY_START    <= t < MIDDAY_END:   return 'midday'
    if EURO_START      <= t < EURO_END:     return 'euro'
    if SQUAREOFF_START <= t:                return 'squareoff'
    return 'outside'


def get_adx_tier(v):
    if v >= ADX_TIER3: return 3
    if v >= ADX_TIER2: return 2
    return 1


def chop_ok(close, ef, es, slope, adx_v, session):
    if adx_v < ADX_MIN: return False
    spread = abs(ef - es) / close * 100
    if spread < SPREAD_PCT_MIN * (MIDDAY_SPREAD_MULT if session == 'midday' else 1.0): return False
    if abs(slope) <= SLOPE_MIN: return False
    if abs(close - es) < PRICE_GAP_MIN: return False
    return True


# ══════════════════════════════════════════════════════════════════
#   DATA LOADERS
# ══════════════════════════════════════════════════════════════════

def _standardise(df, tz):
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(tz)
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert(tz)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[(df['timestamp'].dt.time >= time(9,15)) &
            (df['timestamp'].dt.time <= time(15,30))].reset_index(drop=True)
    df[['open','high','low','close']] = df[['open','high','low','close']].ffill()
    if 'volume' not in df.columns: df['volume'] = 0
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    return df


def fetch_yfinance(interval, label):
    try: import yfinance as yf
    except ImportError: print("pip install yfinance"); sys.exit(1)
    import warnings
    print(f"  Fetching {YFINANCE_SYMBOL} {interval} ({YFINANCE_DAYS}d)...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = yf.download(YFINANCE_SYMBOL, period=f'{YFINANCE_DAYS}d',
                          interval=interval, progress=False, auto_adjust=True)
    if raw.empty: print("No data"); sys.exit(1)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = [c[0].lower() for c in raw.columns]
    else: raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={'adj close':'close','adj_close':'close'})
    if 'volume' not in raw.columns: raw['volume'] = 0
    raw = raw[['open','high','low','close','volume']].dropna()
    if raw.index.tz is None: raw.index = raw.index.tz_localize('UTC')
    raw.index = raw.index.tz_convert(YFINANCE_TZ)
    df = raw.reset_index()
    tc = next(c for c in df.columns if c.lower() in ('datetime','date','timestamp'))
    df = df.rename(columns={tc:'timestamp'})
    df = _standardise(df, YFINANCE_TZ)
    print(f"  [{label}] {len(df)} rows | {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def load_csv(path, label=''):
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
    tc = next((c for c in ['timestamp','datetime','date_time','time','date'] if c in df.columns), None)
    if not tc: raise ValueError(f"No timestamp. Got: {list(df.columns)}")
    df['timestamp'] = pd.to_datetime(df[tc], dayfirst=True)
    if tc != 'timestamp': df = df.drop(columns=[tc])
    for col in ['open','high','low','close']:
        if col not in df.columns: raise ValueError(f"Missing: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open','high','low','close'])
    df = _standardise(df, 'Asia/Kolkata')
    print(f"  [{label}] {len(df)} rows | {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ══════════════════════════════════════════════════════════════════
#   BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════

def run_backtest(df):
    closes        = df['close'].astype(float).values
    highs         = df['high'].astype(float).values
    lows          = df['low'].astype(float).values
    ema_fast      = df['ema_fast'].values
    ema_slow      = df['ema_slow'].values
    ema_macro     = df['ema_macro'].values
    atrs          = df['atr'].values
    adx_vals      = df['adx'].values
    adx_prev_vals = df['adx_prev'].values
    slopes        = df['ema_slow_slope'].values
    consec_bear   = df['consec_bearish_bars'].values
    vol_vals      = df['volume'].values
    vol_ma_vals   = df['vol_ma'].values
    htf_long_bias = df['htf_long_bias'].values
    ts_list       = df['timestamp'].tolist()
    n             = len(df)

    trades = []; equity = 0.0; eq_curve = []

    # Trade state
    in_trade = False; direction = None
    entry_price = 0.0; entry_time = None
    stop_loss = 0.0; sl_at_entry = 0.0; sl_dist_at_entry = 0.0
    adx_tier_at_entry = 2; atr_at_entry = 0.0
    micro_triggered = False; be_triggered = False; trail_active = False
    running_extreme = 0.0
    trade_mfe = 0.0; trade_mae = 0.0

    # Daily state
    prev_date = None; daily_trades = {}; daily_consec_loss = {}

    def favor(p):
        return (p - entry_price) if direction == 'long' else (entry_price - p)

    def do_enter(dir_str, close_p, ts_now, atr_val, adx_val):
        nonlocal in_trade, direction, entry_price, entry_time
        nonlocal stop_loss, sl_at_entry, sl_dist_at_entry
        nonlocal micro_triggered, be_triggered, trail_active
        nonlocal trade_mfe, trade_mae, adx_tier_at_entry, atr_at_entry, running_extreme

        tier    = get_adx_tier(adx_val)
        sl_mult = LONG_ATR_SL_MULT if dir_str == 'long' else SHORT_ATR_SL_MULT
        sl_dist = atr_val * sl_mult

        in_trade = True; direction = dir_str
        entry_price = close_p; entry_time = ts_now
        atr_at_entry = atr_val; adx_tier_at_entry = tier
        sl_dist_at_entry = sl_dist
        micro_triggered = False; be_triggered = False; trail_active = False
        trade_mfe = 0.0; trade_mae = 0.0; running_extreme = close_p

        stop_loss = (entry_price - sl_dist) if dir_str == 'long' else (entry_price + sl_dist)
        sl_at_entry = stop_loss

    def do_exit(exit_p, ts_now, reason):
        nonlocal in_trade, direction, entry_price, entry_time
        nonlocal stop_loss, sl_at_entry, sl_dist_at_entry
        nonlocal micro_triggered, be_triggered, trail_active
        nonlocal trade_mfe, trade_mae, adx_tier_at_entry, atr_at_entry
        nonlocal running_extreme, equity

        pnl = round((exit_p - entry_price) if direction == 'long'
                    else (entry_price - exit_p), 2)
        equity += pnl
        ds = str(ts_now.date())
        daily_consec_loss[ds] = 0 if pnl > 0 else daily_consec_loss.get(ds, 0) + 1

        trades.append({'direction': direction, 'entry_time': entry_time,
                       'exit_time': ts_now, 'entry_price': entry_price,
                       'exit_price': exit_p, 'sl_at_entry': sl_at_entry,
                       'sl_at_exit': stop_loss, 'adx_tier': adx_tier_at_entry,
                       'be_triggered': be_triggered, 'micro_triggered': micro_triggered,
                       'pnl': pnl, 'mfe_pts': round(trade_mfe, 2),
                       'mae_pts': round(trade_mae, 2), 'exit_reason': reason})

        in_trade = False; direction = None; entry_price = 0.0; entry_time = None
        stop_loss = 0.0; sl_at_entry = 0.0; sl_dist_at_entry = 0.0
        micro_triggered = False; be_triggered = False; trail_active = False
        trade_mfe = 0.0; trade_mae = 0.0; adx_tier_at_entry = 2
        atr_at_entry = 0.0; running_extreme = 0.0

    for idx in range(n):
        close   = closes[idx]; high_c = highs[idx]; low_c = lows[idx]
        ef      = ema_fast[idx]; es = ema_slow[idx]; em = ema_macro[idx]
        atr     = float(atrs[idx])
        adx_v   = float(adx_vals[idx])      if not np.isnan(adx_vals[idx])      else 0.0
        adx_pr  = float(adx_prev_vals[idx]) if not np.isnan(adx_prev_vals[idx]) else 0.0
        slope   = float(slopes[idx])         if not np.isnan(slopes[idx])         else 0.0
        cb      = int(consec_bear[idx])
        vol     = float(vol_vals[idx])
        vol_ma  = float(vol_ma_vals[idx])   if not np.isnan(vol_ma_vals[idx])   else 0.0
        ts      = ts_list[idx]
        c_time  = ts.time(); c_date = ts.date(); ds = str(c_date)
        session = get_session(c_time)

        if c_date != prev_date: prev_date = c_date
        if np.isnan(ef) or np.isnan(es) or np.isnan(em) or np.isnan(atr):
            eq_curve.append({'timestamp': ts, 'equity': equity}); continue

        htf_long   = bool(htf_long_bias[idx]) if not pd.isna(htf_long_bias[idx]) else False
        adx_rising = adx_v > adx_pr

        # ══════════════════════════════════════════════════════════
        #   MANAGE OPEN TRADE
        # ══════════════════════════════════════════════════════════
        if in_trade:
            fav_h = (high_c - entry_price) if direction == 'long' else (entry_price - low_c)
            adv   = (entry_price - low_c)  if direction == 'long' else (high_c - entry_price)
            trade_mfe = max(trade_mfe, fav_h)
            trade_mae = max(trade_mae, adv)
            cf = favor(close)
            is_t3 = (adx_tier_at_entry == 3)

            # Stage 1: Micro-Trail (Tier2 only)
            if ENABLE_MICRO_TRAIL and not be_triggered and not micro_triggered:
                if not is_t3 or TIER3_MICRO_TRAIL:
                    if cf >= MICRO_TRAIL_TRIGGER:
                        md = atr_at_entry * MICRO_TRAIL_SL_FACTOR
                        if direction == 'long': stop_loss = max(stop_loss, entry_price - md)
                        else:                   stop_loss = min(stop_loss, entry_price + md)
                        micro_triggered = True

            # Stage 2: Profit Floor Break-Even
            if not be_triggered:
                be_ok = cf >= BE_POINTS_TRIGGER or (
                    (direction == 'long'  and close >= entry_price * (1 + LONG_BE_PCT  / 100)) or
                    (direction == 'short' and close <= entry_price * (1 - SHORT_BE_PCT / 100))
                )
                if be_ok:
                    ratio  = TIER3_BE_LOCK_RATIO if is_t3 else TIER2_BE_LOCK_RATIO
                    floor  = sl_dist_at_entry * ratio
                    if direction == 'long':  stop_loss = max(stop_loss, entry_price + floor)
                    else:                    stop_loss = min(stop_loss, entry_price - floor)
                    be_triggered    = True
                    trail_active    = True
                    running_extreme = high_c if direction == 'long' else low_c

            # Stage 3: Running-High Trail (BOTH tiers)
            # Only advances SL when price makes a new extreme since BE triggered.
            # For Tier2 this replaces the every-bar standard trail that was cutting
            # winners 21pts short. Trail dist is the same (0.6×ATR), just
            # updates only on new highs/lows instead of every single close.
            if trail_active:
                tm = TIER3_TRAIL_ATR_MULT if is_t3 else TIER2_TRAIL_ATR_MULT
                td = atr * tm
                if direction == 'long':
                    if high_c > running_extreme: running_extreme = high_c
                    stop_loss = max(stop_loss, running_extreme - td)
                else:
                    if low_c < running_extreme: running_extreme = low_c
                    stop_loss = min(stop_loss, running_extreme + td)

            exit_p = None; exit_r = None

            # Stage 4: SL hit
            if direction == 'long' and close <= stop_loss:
                exit_p = stop_loss
                exit_r = 'TRAIL_SL' if trail_active else ('MICRO_SL' if micro_triggered else 'STOP_LOSS')
            elif direction == 'short' and close >= stop_loss:
                exit_p = stop_loss
                exit_r = 'TRAIL_SL' if trail_active else ('MICRO_SL' if micro_triggered else 'STOP_LOSS')

            # Stage 5: EOD Kill Losers (15:00)
            elif c_time >= EOD_KILL_LOSERS and not be_triggered and cf < -5.0:
                exit_p = close; exit_r = 'EOD_KILL_LOSER'

            # Stage 6: EOD Kill Non-BE (15:10)
            elif c_time >= EOD_KILL_NON_TRAIL and not be_triggered:
                exit_p = close; exit_r = 'EOD_KILL_NON_TRAIL'

            # Stage 7: Hard EOD (15:25)
            elif c_time >= EOD_HARD_EXIT:
                exit_p = close; exit_r = 'HARD_EOD_EXIT'

            if exit_p is not None: do_exit(exit_p, ts, exit_r)

        # ══════════════════════════════════════════════════════════
        #   ENTRY LOGIC
        # ══════════════════════════════════════════════════════════
        if not in_trade:
            allowed = ['prime']
            if ENABLE_MIDDAY: allowed.append('midday')
            if ENABLE_EURO:   allowed.append('euro')
            if session not in allowed:
                eq_curve.append({'timestamp': ts, 'equity': equity}); continue

            if daily_trades.get(ds, 0) >= MAX_TRADES_PER_DAY:
                eq_curve.append({'timestamp': ts, 'equity': equity}); continue

            if daily_consec_loss.get(ds, 0) >= MAX_CONSEC_LOSSES:
                eq_curve.append({'timestamp': ts, 'equity': equity}); continue

            if not chop_ok(close, ef, es, slope, adx_v, session):
                eq_curve.append({'timestamp': ts, 'equity': equity}); continue

            tier_e = get_adx_tier(adx_v)
            if TIER3_REQUIRE_ADX_RISING and tier_e == 3 and not adx_rising:
                eq_curve.append({'timestamp': ts, 'equity': equity}); continue

            retest = atr * RETEST_ATR_MULT

            # LONG
            if ENABLE_LONG and htf_long:
                if ef > es and close > em and slope > 0 and abs(close - ef) <= retest and close > es:
                    do_enter('long', close, ts, atr, adx_v)
                    daily_trades[ds] = daily_trades.get(ds, 0) + 1

            # SHORT
            if not in_trade and ENABLE_SHORT:
                vol_ok = not ENABLE_VOL_FILTER or vol_ma == 0 or vol >= vol_ma * VOL_SPIKE_MULT
                if ef < es and close < em and slope < 0 and abs(close-ef) <= retest and close < es \
                        and cb >= SHORT_CONFIRM_BARS and vol_ok:
                    do_enter('short', close, ts, atr, adx_v)
                    daily_trades[ds] = daily_trades.get(ds, 0) + 1

        eq_curve.append({'timestamp': ts, 'equity': equity})

    if in_trade: do_exit(closes[-1], ts_list[-1], 'END_OF_DATA')
    return pd.DataFrame(trades), pd.DataFrame(eq_curve)


# ══════════════════════════════════════════════════════════════════
#   METRICS & REPORTING
# ══════════════════════════════════════════════════════════════════

def compute_metrics(tdf):
    if tdf.empty: return {'message': 'No trades.'}
    pnl = tdf['pnl']
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]; total = len(tdf)
    gp, gl = wins.sum(), abs(losses.sum())
    cum = pnl.cumsum(); dd = (cum.cummax() - cum).max()
    return {
        'Total Trades':       total,
        'Winning Trades':     len(wins),
        'Losing Trades':      len(losses),
        'Win Rate (%)':       round(len(wins)/total*100, 2),
        'Avg Profit (pts)':   round(wins.mean(), 2) if len(wins) else 0,
        'Avg Loss (pts)':     round(losses.mean(), 2) if len(losses) else 0,
        'Largest Win (pts)':  round(pnl.max(), 2),
        'Largest Loss (pts)': round(pnl.min(), 2),
        'Profit Factor':      round(gp/gl, 2) if gl > 0 else float('inf'),
        'Total P&L (pts)':    round(pnl.sum(), 2),
        'Max Drawdown (pts)': round(dd, 2),
        'Avg MFE (pts)':      round(tdf['mfe_pts'].mean(), 2),
        'Avg MAE (pts)':      round(tdf['mae_pts'].mean(), 2),
    }


def fmt(ts):
    return ts.strftime('%d-%m-%Y %H:%M') if hasattr(ts, 'strftime') else str(ts)


def print_results(tdf):
    SEP = '─' * 165
    print(f"\n{SEP}\n  TRADE LOG  ({len(tdf)} trades)\n{SEP}")
    print(f"{'#':<5} {'Dir':<6} {'T':<5} {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>9} {'Exit':>9} {'SL@Entry':>9} {'SL@Exit':>9}"
          f" {'MFE':>7} {'MAE':>7} {'P&L':>9}  Reason")
    print(SEP)
    for i, r in tdf.iterrows():
        fl  = f"T{r.get('adx_tier','?')}"
        fl += 'M' if r.get('micro_triggered') else ' '
        fl += 'B' if r.get('be_triggered')    else ' '
        print(f"{i+1:<5} {r['direction'].upper():<6} {fl:<5}"
              f" {fmt(r['entry_time']):<22} {fmt(r['exit_time']):<22}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f}"
              f" {r['sl_at_entry']:>9.2f} {r['sl_at_exit']:>9.2f}"
              f" {r['mfe_pts']:>7.1f} {r['mae_pts']:>7.1f}"
              f" {r['pnl']:>+9.2f}  {r['exit_reason']}")
    print("\n  Legend: T2/T3=Tier | M=Micro | B=BE triggered")

    m = compute_metrics(tdf)
    print(f"\n{'─'*62}\n  PERFORMANCE METRICS\n{'─'*62}")
    for k, v in m.items(): print(f"  {k:<30}: {v}")

    print(f"\n{'─'*62}\n  DIRECTION BREAKDOWN\n{'─'*62}")
    for d in ['long', 'short']:
        s = tdf[tdf['direction']==d]
        if s.empty: continue
        p = s['pnl']; w = (p>0).sum()
        print(f"  {d.upper():<6}  n={len(s)}  w={w}  wr={w/len(s)*100:.1f}%"
              f"  total={p.sum():.1f}  avg={p.mean():.1f}"
              f"  avg_mfe={s['mfe_pts'].mean():.1f}  avg_mae={s['mae_pts'].mean():.1f}")

    print(f"\n{'─'*62}\n  ADX TIER BREAKDOWN\n{'─'*62}")
    for tier, lbl in [(2,'Normal(25-35)'), (3,'Strong(35+  )')]:
        s = tdf[tdf['adx_tier']==tier]
        if s.empty: continue
        p = s['pnl']; w = (p>0).sum()
        print(f"  Tier{tier}[{lbl}]  n={len(s)}  w={w}  wr={w/len(s)*100:.1f}%"
              f"  total={p.sum():.1f}  avg={p.mean():.1f}  avg_mfe={s['mfe_pts'].mean():.1f}")

    print(f"\n{'─'*62}\n  EXIT REASON BREAKDOWN\n{'─'*62}")
    bd = tdf.groupby('exit_reason')['pnl'].agg(count='count',total='sum',avg='mean').round(2)
    print(bd.to_string())

    print(f"\n{'─'*62}\n  SESSION BREAKDOWN\n{'─'*62}")
    # Infer session from entry time
    def sess(t):
        tt = t.time()
        if PRIME_START  <= tt < PRIME_END:  return 'prime'
        if MIDDAY_START <= tt < MIDDAY_END: return 'midday'
        if EURO_START   <= tt < EURO_END:   return 'euro'
        return 'other'
    tdf2 = tdf.copy()
    tdf2['session'] = tdf2['entry_time'].apply(sess)
    for s, grp in tdf2.groupby('session'):
        p = grp['pnl']; w = (p>0).sum()
        print(f"  {s:<8}  n={len(grp):>3}  wr={w/len(grp)*100:.1f}%"
              f"  total={p.sum():>8.1f}  avg={p.mean():>7.1f}"
              f"  avg_mfe={grp['mfe_pts'].mean():.1f}")

    print(f"\n{'─'*62}\n  MONTHLY P&L BREAKDOWN\n{'─'*62}")
    tdf3 = tdf.copy()
    tdf3['month'] = tdf3['entry_time'].dt.tz_localize(None).dt.to_period('M').astype(str)
    mo = (tdf3.groupby('month')['pnl']
          .agg(n='count', total='sum', wins=lambda x:(x>0).sum())
          .assign(wr=lambda x:(x['wins']/x['n']*100).round(1)).round(2))
    print(mo.to_string())

    print(f"\n{'─'*62}\n  DAILY TRADE COUNT\n{'─'*62}")
    tdf4 = tdf.copy(); tdf4['date'] = tdf4['entry_time'].dt.date
    for n_t, days in tdf4.groupby('date').size().value_counts().sort_index().items():
        print(f"  {n_t} trade(s)/day → {days} day(s)")

    print(f"\n{'─'*62}\n  STOP PROTECTION & TRAIL ANALYSIS\n{'─'*62}")
    mt  = tdf[tdf['micro_triggered']==True]
    bt  = tdf[tdf['be_triggered']==True]
    tr  = tdf[tdf['exit_reason']=='TRAIL_SL']
    ms  = tdf[tdf['exit_reason']=='MICRO_SL']
    rs  = tdf[tdf['exit_reason']=='STOP_LOSS']
    eod = tdf[tdf['exit_reason']=='HARD_EOD_EXIT']

    print(f"  Micro-Trail reached  : {len(mt):>3} / {len(tdf)}")
    print(f"  Break-Even reached   : {len(bt):>3} / {len(tdf)}")
    if not tr.empty:
        gap = tr['mfe_pts'].mean() - tr['pnl'].mean()
        print(f"  TRAIL_SL  : n={len(tr):>3}  avg={tr['pnl'].mean():.1f}  "
              f"avg_mfe={tr['mfe_pts'].mean():.1f}  MFE-exit gap={gap:.1f}"
              f"  {'✅ capturing well' if gap < 20 else '⚠ trail still leaving profit'}")
    if not ms.empty:
        print(f"  MICRO_SL  : n={len(ms):>3}  avg={ms['pnl'].mean():.1f}")
    if not rs.empty:
        went = (rs['mfe_pts'] > 5).sum()
        print(f"  STOP_LOSS : n={len(rs):>3}  avg={rs['pnl'].mean():.1f}  "
              f"MAE={rs['mae_pts'].mean():.1f}  MFE={rs['mfe_pts'].mean():.1f}  "
              f"in-favor>5pts: {went}/{len(rs)}")
    if not eod.empty:
        print(f"  HARD_EOD  : n={len(eod):>3}  avg={eod['pnl'].mean():.1f}  "
              f"avg_mfe={eod['mfe_pts'].mean():.1f}  "
              f"{'⚠ trail not capturing these — widen trail?' if eod['mfe_pts'].mean() > 80 else '~OK'}")

    print(f"\n{'─'*62}\n  TIER 3 EXIT ANALYSIS\n{'─'*62}")
    t3 = tdf[tdf['adx_tier']==3]
    if not t3.empty:
        for reason, g in t3.groupby('exit_reason'):
            extra = ""
            if reason == 'TRAIL_SL':
                gap = g['mfe_pts'].mean() - g['pnl'].mean()
                extra = f"  gap={gap:.1f}pts {'✅' if gap < 25 else '⚠ widen TIER3_TRAIL_ATR_MULT'}"
            print(f"  {reason:<22} n={len(g):>3}  avg={g['pnl'].mean():>7.1f}"
                  f"  avg_mfe={g['mfe_pts'].mean():>7.1f}{extra}")
    print()


def main():
    BAR = '═' * 68
    print(f"\n{BAR}\n  SPEED DEMON SCALPER  v8  — BACK TO BASICS\n{BAR}")

    cli_args = sys.argv[1:]
    mode = 'csv' if cli_args else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\nSource: yfinance | {YFINANCE_SYMBOL}")
        df = fetch_yfinance('5m', '5m')
        save_base = os.path.join(os.getcwd(),
                                 YFINANCE_SYMBOL.replace('^','').replace('.','_') + '_speed_demon_v8')
    elif mode == 'csv':
        path = cli_args[0] if cli_args else CSV_5MIN_PATH
        if not os.path.exists(path): print(f"Not found: {path}"); sys.exit(1)
        print(f"\nLoading: {path}")
        df = load_csv(path, '5m')
        save_base = os.path.splitext(path)[0] + '_speed_demon_v8'
    else:
        print(f"Unknown DATA_MODE: {DATA_MODE}"); sys.exit(1)

    print(f"\n{'─'*62}\n  CONFIG  (v8)\n{'─'*62}")
    print(f"  ADX floor/T2/T3           : {ADX_MIN}/{ADX_TIER2}/{ADX_TIER3}  T3-rising={'ON' if TIER3_REQUIRE_ADX_RISING else 'OFF'}")
    print(f"  SL: Long={LONG_ATR_SL_MULT}×  Short={SHORT_ATR_SL_MULT}×ATR")
    print(f"  Micro-Trail (T2)          : +{MICRO_TRAIL_TRIGGER}pts → SL±{MICRO_TRAIL_SL_FACTOR}×ATR  T3={'ON' if TIER3_MICRO_TRAIL else 'OFF'}")
    print(f"  BE trigger                : +{BE_POINTS_TRIGGER}pts OR +{LONG_BE_PCT}%")
    print(f"  BE profit floor T2/T3     : sl_dist × {TIER2_BE_LOCK_RATIO} / {TIER3_BE_LOCK_RATIO}")
    print(f"  Running-High Trail T2/T3  : {TIER2_TRAIL_ATR_MULT}×ATR / {TIER3_TRAIL_ATR_MULT}×ATR  (new extremes only)")
    print(f"  EMA Cross Exit            : {'ON' if ENABLE_EMA_CROSS_EXIT else 'REMOVED (was cutting wins short)'}")
    print(f"  Sessions                  : Prime=ON  Midday={'ON' if ENABLE_MIDDAY else 'OFF'}  Euro={'ON' if ENABLE_EURO else 'OFF'}")
    print(f"  Short: {SHORT_CONFIRM_BARS}bars + vol≥{VOL_SPIKE_MULT}×  EOD: {EOD_KILL_LOSERS}/{EOD_KILL_NON_TRAIL}/{EOD_HARD_EXIT}")
    print(f"{'─'*62}")

    print("\nComputing indicators ..."); df = compute_indicators_5m(df)
    print("Computing HTF bias ...");   df = compute_htf_bias(df)
    print("Running backtest ...\n")

    trades_df, equity_df = run_backtest(df)

    print(f"\n{BAR}\n  RESULTS\n{BAR}")
    if trades_df.empty:
        print("\n  No trades. Lower ADX_MIN or SPREAD_PCT_MIN."); return

    print_results(trades_df)

    to = save_base + '_trades.csv'
    eo = save_base + '_equity.csv'
    trades_df.to_csv(to, index=False)
    equity_df.to_csv(eo, index=False)
    print(f"{'─'*62}\n  Trades: {to}\n  Equity: {eo}\n{BAR}\n")


if __name__ == '__main__':
    main()