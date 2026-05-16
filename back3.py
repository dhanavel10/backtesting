"""
╔══════════════════════════════════════════════════════════════════╗
║   SPEED DEMON SCALPER  v4  — "IRON STOP SYSTEM"                ║
╠══════════════════════════════════════════════════════════════════╣
║  WHAT'S NEW vs v3                                                ║
║  ─────────────────────────────────────────────────────────────  ║
║                                                                  ║
║  ✦ Micro-Trail  (the #1 fix from v3 MFE/MAE data)              ║
║      v3 showed 18/20 SL trades moved 34 pts in-favor first      ║
║      Solution: once trade moves MICRO_TRAIL_TRIGGER pts          ║
║      in-favor, immediately lock SL to entry - 0.5×ATR           ║
║      (half-loss protection) BEFORE full break-even kicks in      ║
║      Stages: Entry → Micro-Trail → Break-Even → Full ATR Trail   ║
║                                                                  ║
║  ✦ Points-Based Break-Even (dual BE trigger)                    ║
║      BE triggers on EITHER condition, whichever comes first:     ║
║        (a) Price moves BE_POINTS_TRIGGER pts in favor            ║
║        (b) Price moves LONG/SHORT_BE_PCT% in favor               ║
║      Fixes the SOFT_EOD_EXIT problem (7 trades avg +46 pts       ║
║      that were profitable but BE hadn't triggered yet)           ║
║                                                                  ║
║  ✦ Volume Spike Filter for SHORT entries                        ║
║      Short entry requires volume >= VOL_SPIKE_MULT × 20-bar avg ║
║      Captures high-conviction breakdown bars, not drift          ║
║                                                                  ║
║  ✦ Scaled Position Sizing (Tier system)                         ║
║      ADX 20-25 : Tier 1 (half size / tight SL)                  ║
║      ADX 25-35 : Tier 2 (normal size)                           ║
║      ADX 35+   : Tier 3 (full size / wider trail)               ║
║      P&L reported in pts — multiply by lot size for ₹           ║
║                                                                  ║
║  ✦ Refined Soft EOD                                             ║
║      15:00 → close trades with pnl < -5 pts (losing)            ║
║      15:10 → close ALL trades not yet trailing                   ║
║      15:25 → hard close everything                               ║
╠══════════════════════════════════════════════════════════════════╣
║  v1→v2→v3→v4 Journey                                            ║
║    PF  : 2.1 → 2.76 → 3.36 → ?                                 ║
║    P&L : 739 → 1454 → 1221 → ?  (v3 had fewer trades)          ║
║    DD  : 265 → 196  → 137  → ?                                  ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python speed_demon_v4.py              ← yfinance (default)
    python speed_demon_v4.py 5m.csv       ← CSV file
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

# ── ADX Regime Filter + Tier System ─────────────────────────────
ADX_PERIOD       = 14
ADX_MIN          = 20       # below this: no trades
ADX_TIER2        = 25       # 20-25 = Tier1, 25-35 = Tier2, 35+ = Tier3
ADX_TIER3        = 35

# ── Asymmetric SL Multipliers ────────────────────────────────────
LONG_ATR_SL_MULT    = 1.2
SHORT_ATR_SL_MULT   = 0.75

# ── ADX Tier SL adjustments ──────────────────────────────────────
# Tier1 (low trend): tighten SL to reduce loss per trade
TIER1_SL_FACTOR     = 0.8   # SL = base_sl × 0.8  (tighter in weak trend)
# Tier3 (strong trend): loosen trail to let it run
TIER3_TRAIL_FACTOR  = 1.3   # trail_dist = base × 1.3 (more room in strong trend)

# ── Micro-Trail (NEW v4) ─────────────────────────────────────────
# Once price moves MICRO_TRAIL_TRIGGER pts in-favor,
# SL locks to entry - (ATR × MICRO_TRAIL_SL_FACTOR)
# This is BEFORE break-even triggers.
MICRO_TRAIL_TRIGGER     = 15.0   # pts in-favor to activate micro-trail
MICRO_TRAIL_SL_FACTOR   = 0.5    # SL = entry ± (ATR × 0.5) — half-loss protection
ENABLE_MICRO_TRAIL      = True

# ── Break-Even Trigger (dual: points OR percent, whichever first) ─
LONG_BE_PCT          = 0.3      # % in-favor
SHORT_BE_PCT         = 0.3      # % in-favor (tightened from 0.4)
BE_POINTS_TRIGGER    = 30.0     # pts in-favor — NEW: catches trades BE% misses
                                 # set to 9999 to use % only

# ── Trailing Stop ────────────────────────────────────────────────
TRAIL_ATR_MULT       = 0.6

# ── EMA Cross Exit ───────────────────────────────────────────────
ENABLE_EMA_CROSS_EXIT = True

# ── Slope Reversal Exit ──────────────────────────────────────────
ENABLE_SLOPE_EXIT    = True
SLOPE_EXIT_CANDLES   = 4

# ── Short Entry Filters ──────────────────────────────────────────
SHORT_CONFIRM_BARS   = 3       # consecutive bars EMA9 < EMA21
VOL_PERIOD           = 20      # rolling avg volume window
VOL_SPIKE_MULT       = 1.2     # short entry needs vol >= 1.2× avg vol
                                # set to 0.0 to disable
ENABLE_VOL_FILTER    = False

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

# ── Session Gates ─────────────────────────────────────────────────
ENABLE_MIDDAY        = False
ENABLE_EURO          = True

# ── Sessions ─────────────────────────────────────────────────────
OBSERVE_START       = time(9,  15)
OBSERVE_END         = time(9,  30)
PRIME_START         = time(9,  30)
PRIME_END           = time(10, 30)
MIDDAY_START        = time(11, 30)
MIDDAY_END          = time(13, 30)
EURO_START          = time(14, 15)
EURO_END            = time(15, 0)
SQUAREOFF_START     = time(15, 0)

# ── Refined EOD (3-stage) ─────────────────────────────────────────
# Stage 1 (15:00): close trades where pnl currently NEGATIVE
EOD_KILL_LOSERS      = time(15,  0)
# Stage 2 (15:10): close trades not yet trailing (BE not triggered)
EOD_KILL_NON_TRAIL   = time(15, 10)
# Stage 3 (15:25): hard close everything
EOD_HARD_EXIT        = time(15, 25)


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
    """Wilder's ADX."""
    high  = df['high'].astype(float)
    low   = df['low'].astype(float)
    up_m  = high.diff()
    dn_m  = -(low.diff())
    plus_dm  = np.where((up_m > dn_m) & (up_m > 0),   up_m,   0.0)
    minus_dm = np.where((dn_m > up_m) & (dn_m > 0),   dn_m,   0.0)
    atr_vals = compute_atr(df, period)
    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * plus_dm_s  / atr_vals.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr_vals.replace(0, np.nan)
    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx  = 100 * (plus_di - minus_di).abs() / dx_denom
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
    df['vol_ma']              = df['volume'].rolling(VOL_PERIOD).mean()

    # Consecutive bearish EMA bars (for short confirmation)
    bearish = (df['ema_fast'] < df['ema_slow']).astype(int)
    df['consec_bearish_bars'] = (
        bearish.groupby((bearish != bearish.shift()).cumsum()).cumcount() + 1
    ) * bearish
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
    htf_ri = htf_cols.reindex(df_5m.index, method='ffill')
    return df_5m.join(htf_ri).reset_index()


# ══════════════════════════════════════════════════════════════════
#   FILTERS
# ══════════════════════════════════════════════════════════════════

def get_session(t: time) -> str:
    if OBSERVE_START <= t < OBSERVE_END:   return 'observe'
    if PRIME_START   <= t < PRIME_END:     return 'prime'
    if MIDDAY_START  <= t < MIDDAY_END:    return 'midday'
    if EURO_START    <= t < EURO_END:      return 'euro'
    if SQUAREOFF_START <= t:               return 'squareoff'
    return 'outside'


def get_adx_tier(adx_val: float) -> int:
    """1 = weak trend, 2 = normal, 3 = strong trend."""
    if adx_val >= ADX_TIER3: return 3
    if adx_val >= ADX_TIER2: return 2
    return 1


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
        print(f"ERROR: No data for '{YFINANCE_SYMBOL}'."); sys.exit(1)
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
    consec_bear    = df['consec_bearish_bars'].values
    vol_vals       = df['volume'].values
    vol_ma_vals    = df['vol_ma'].values
    htf_long_bias  = df['htf_long_bias'].values
    ts_list        = df['timestamp'].tolist()
    n              = len(df)

    trades         = []
    equity         = 0.0
    eq_curve       = []

    # ── Trade State ───────────────────────────────────────────────
    in_trade            = False
    direction           = None
    entry_price         = 0.0
    entry_time          = None
    stop_loss           = 0.0
    sl_at_entry         = 0.0
    adx_tier_at_entry   = 1
    atr_at_entry        = 0.0

    micro_triggered     = False   # micro-trail stage
    be_triggered        = False   # break-even stage
    trail_active        = False   # full ATR trail stage

    trade_mfe           = 0.0
    trade_mae           = 0.0

    # ── Daily State ────────────────────────────────────────────────
    prev_date           = None
    daily_trades        = {}
    daily_consec_loss   = {}

    def _current_favor(close_p):
        return (close_p - entry_price) if direction == 'long' else (entry_price - close_p)

    def do_enter(dir_str, close_p, ts_now, atr_val, adx_val):
        nonlocal in_trade, direction, entry_price, entry_time, stop_loss, sl_at_entry
        nonlocal micro_triggered, be_triggered, trail_active
        nonlocal trade_mfe, trade_mae, adx_tier_at_entry, atr_at_entry

        tier    = get_adx_tier(adx_val)
        sl_mult = LONG_ATR_SL_MULT if dir_str == 'long' else SHORT_ATR_SL_MULT
        # Tier1: tighten SL a bit (weaker trend = less conviction)
        if tier == 1:
            sl_mult = sl_mult * TIER1_SL_FACTOR

        sl_dist = atr_val * sl_mult

        in_trade            = True
        direction           = dir_str
        entry_price         = close_p
        entry_time          = ts_now
        atr_at_entry        = atr_val
        adx_tier_at_entry   = tier
        micro_triggered     = False
        be_triggered        = False
        trail_active        = False
        trade_mfe           = 0.0
        trade_mae           = 0.0

        if direction == 'long':
            stop_loss  = entry_price - sl_dist
        else:
            stop_loss  = entry_price + sl_dist
        sl_at_entry = stop_loss

    def do_exit(exit_p, ts_now, reason):
        nonlocal in_trade, direction, entry_price, entry_time, stop_loss, sl_at_entry
        nonlocal micro_triggered, be_triggered, trail_active
        nonlocal trade_mfe, trade_mae, adx_tier_at_entry, atr_at_entry, equity

        pnl = round(
            (exit_p - entry_price) if direction == 'long'
            else (entry_price - exit_p), 2
        )
        equity += pnl

        date_str = str(ts_now.date())
        if pnl <= 0:
            daily_consec_loss[date_str] = daily_consec_loss.get(date_str, 0) + 1
        else:
            daily_consec_loss[date_str] = 0

        trades.append({
            'direction':        direction,
            'entry_time':       entry_time,
            'exit_time':        ts_now,
            'entry_price':      entry_price,
            'exit_price':       exit_p,
            'sl_at_entry':      sl_at_entry,
            'sl_at_exit':       stop_loss,
            'adx_tier':         adx_tier_at_entry,
            'be_triggered':     be_triggered,
            'micro_triggered':  micro_triggered,
            'pnl':              pnl,
            'mfe_pts':          round(trade_mfe, 2),
            'mae_pts':          round(trade_mae, 2),
            'exit_reason':      reason,
        })

        in_trade            = False
        direction           = None
        entry_price         = 0.0
        entry_time          = None
        stop_loss           = 0.0
        sl_at_entry         = 0.0
        micro_triggered     = False
        be_triggered        = False
        trail_active        = False
        trade_mfe           = 0.0
        trade_mae           = 0.0
        adx_tier_at_entry   = 1
        atr_at_entry        = 0.0

    for idx in range(n):
        close      = closes[idx]
        high_c     = highs[idx]
        low_c      = lows[idx]
        ef         = ema_fast[idx]
        es         = ema_slow[idx]
        em         = ema_macro[idx]
        atr        = float(atrs[idx])
        adx_v      = float(adx_vals[idx])  if not np.isnan(adx_vals[idx])  else 0.0
        slope      = float(slopes[idx])     if not np.isnan(slopes[idx])     else 0.0
        slope_exit = float(slopes_exit[idx])if not np.isnan(slopes_exit[idx])else 0.0
        cb         = int(consec_bear[idx])
        vol        = float(vol_vals[idx])
        vol_ma     = float(vol_ma_vals[idx]) if not np.isnan(vol_ma_vals[idx]) else 0.0
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

        # ══════════════════════════════════════════════════════════
        #   MANAGE OPEN TRADE — 4-Stage Exit Engine
        # ══════════════════════════════════════════════════════════
        if in_trade:
            # Track MFE / MAE using bar highs/lows
            favor   = (high_c - entry_price) if direction == 'long' else (entry_price - low_c)
            adverse = (entry_price - low_c)  if direction == 'long' else (high_c - entry_price)
            trade_mfe = max(trade_mfe, favor)
            trade_mae = max(trade_mae, adverse)

            current_favor = _current_favor(close)

            # ── Stage 0: Micro-Trail (NEW v4) ─────────────────────
            # Activates BEFORE break-even. Locks SL to half-loss once
            # trade moves MICRO_TRAIL_TRIGGER pts in-favor.
            if ENABLE_MICRO_TRAIL and not micro_triggered and not be_triggered:
                if current_favor >= MICRO_TRAIL_TRIGGER:
                    micro_sl_dist = atr_at_entry * MICRO_TRAIL_SL_FACTOR
                    if direction == 'long':
                        new_sl = entry_price - micro_sl_dist
                        stop_loss = max(stop_loss, new_sl)  # only improve
                    else:
                        new_sl = entry_price + micro_sl_dist
                        stop_loss = min(stop_loss, new_sl)
                    micro_triggered = True

            # ── Stage 1: Break-Even (dual trigger: pts OR %) ──────
            if not be_triggered:
                be_pts_hit = current_favor >= BE_POINTS_TRIGGER
                be_pct_hit = (
                    (direction == 'long'  and close >= entry_price * (1 + LONG_BE_PCT  / 100)) or
                    (direction == 'short' and close <= entry_price * (1 - SHORT_BE_PCT / 100))
                )
                if be_pts_hit or be_pct_hit:
                    if direction == 'long':
                        stop_loss = max(stop_loss, entry_price + 1.0)
                    else:
                        stop_loss = min(stop_loss, entry_price - 1.0)
                    be_triggered = True
                    trail_active = True

            # ── Stage 2: ATR Trailing Stop (Tier-aware) ───────────
            if trail_active:
                tier_now = get_adx_tier(adx_v)
                # Strong trend (Tier3): give the trail more room
                t_mult = TRAIL_ATR_MULT * (TIER3_TRAIL_FACTOR if tier_now == 3 else 1.0)
                trail_dist = atr * t_mult
                if direction == 'long':
                    stop_loss = max(stop_loss, close - trail_dist)
                else:
                    stop_loss = min(stop_loss, close + trail_dist)

            exit_p = None
            exit_r = None

            # ── Stage 3: Hard SL / Trail hit ─────────────────────
            if direction == 'long' and close <= stop_loss:
                exit_p = stop_loss
                if trail_active:     exit_r = 'TRAIL_SL'
                elif micro_triggered:exit_r = 'MICRO_SL'
                else:                exit_r = 'STOP_LOSS'

            elif direction == 'short' and close >= stop_loss:
                exit_p = stop_loss
                if trail_active:     exit_r = 'TRAIL_SL'
                elif micro_triggered:exit_r = 'MICRO_SL'
                else:                exit_r = 'STOP_LOSS'

            # ── Stage 4: EMA Cross Exit (post-BE) ─────────────────
            elif ENABLE_EMA_CROSS_EXIT and be_triggered:
                if direction == 'long'  and ef < es:
                    exit_p = close; exit_r = 'EMA_CROSS_EXIT'
                elif direction == 'short' and ef > es:
                    exit_p = close; exit_r = 'EMA_CROSS_EXIT'

            # ── Stage 5: Slope Reversal Exit (post-BE) ────────────
            elif ENABLE_SLOPE_EXIT and be_triggered:
                if direction == 'long'  and slope_exit < -SLOPE_MIN:
                    exit_p = close; exit_r = 'SLOPE_REVERSAL_EXIT'
                elif direction == 'short' and slope_exit > SLOPE_MIN:
                    exit_p = close; exit_r = 'SLOPE_REVERSAL_EXIT'

            # ── Stage 6a: EOD — kill losing / flat trades at 15:00 ─
            elif c_time >= EOD_KILL_LOSERS and not be_triggered:
                cur_pnl = current_favor  # proxy for pnl direction
                if cur_pnl < -5.0:       # trade is losing
                    exit_p = close; exit_r = 'EOD_KILL_LOSER'

            # ── Stage 6b: EOD — kill non-trailing at 15:10 ────────
            elif c_time >= EOD_KILL_NON_TRAIL and not trail_active:
                exit_p = close; exit_r = 'EOD_KILL_NON_TRAIL'

            # ── Stage 6c: Hard EOD ─────────────────────────────────
            elif c_time >= EOD_HARD_EXIT:
                exit_p = close; exit_r = 'HARD_EOD_EXIT'

            if exit_p is not None:
                do_exit(exit_p, ts, exit_r)

        # ══════════════════════════════════════════════════════════
        #   ENTRY LOGIC
        # ══════════════════════════════════════════════════════════
        if not in_trade:
            allowed = ['prime']
            if ENABLE_MIDDAY: allowed.append('midday')
            if ENABLE_EURO:   allowed.append('euro')
            if session not in allowed:
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

            retest_tol = atr * RETEST_ATR_MULT

            # ── LONG Entry ────────────────────────────────────────
            if ENABLE_LONG and htf_long:
                if (ef > es and close > em and slope > 0 and
                        abs(close - ef) <= retest_tol and close > es):
                    do_enter('long', close, ts, atr, adx_v)
                    daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

            # ── SHORT Entry (+ volume spike filter) ───────────────
            if not in_trade and ENABLE_SHORT:
                vol_ok = (not ENABLE_VOL_FILTER or
                          vol_ma == 0 or
                          vol >= vol_ma * VOL_SPIKE_MULT)
                if (ef < es and close < em and slope < 0 and
                        abs(close - ef) <= retest_tol and close < es and
                        cb >= SHORT_CONFIRM_BARS and vol_ok):
                    do_enter('short', close, ts, atr, adx_v)
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
    SEP = '─' * 155

    print(f"\n{SEP}")
    print(f"  TRADE LOG  ({len(tdf)} trades)")
    print(SEP)
    print(f"{'#':<5} {'Dir':<6} {'T':<3} {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>9} {'Exit':>9} {'SL@Entry':>9} {'SL@Exit':>9}"
          f" {'MFE':>7} {'MAE':>7} {'P&L':>9}  Reason")
    print(SEP)
    for i, r in tdf.iterrows():
        micro_flag = 'M' if r.get('micro_triggered', False) else ' '
        be_flag    = 'B' if r.get('be_triggered',    False) else ' '
        tier_str   = f"T{r.get('adx_tier', '?')}"
        flags      = f"{tier_str}{micro_flag}{be_flag}"
        print(f"{i+1:<5} {r['direction'].upper():<6} {flags:<5} "
              f"{fmt_ts(r['entry_time']):<22} {fmt_ts(r['exit_time']):<22}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f}"
              f" {r['sl_at_entry']:>9.2f} {r['sl_at_exit']:>9.2f}"
              f" {r['mfe_pts']:>7.1f} {r['mae_pts']:>7.1f}"
              f" {r['pnl']:>+9.2f}  {r['exit_reason']}")
    print(f"\n  Legend: T1/T2/T3 = ADX Tier | M = Micro-Trail triggered | B = Break-Even triggered")

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
        if sub.empty: continue
        pnl_s = sub['pnl']
        w = (pnl_s > 0).sum()
        print(f"  {d.upper():<6}  trades={len(sub)}  wins={w}  "
              f"wr={w/len(sub)*100:.1f}%  total={pnl_s.sum():.1f}  "
              f"avg={pnl_s.mean():.1f}  "
              f"avg_mfe={sub['mfe_pts'].mean():.1f}  avg_mae={sub['mae_pts'].mean():.1f}")

    print(f"\n{'─'*62}")
    print("  ADX TIER BREAKDOWN")
    print(f"{'─'*62}")
    for tier in [1, 2, 3]:
        sub = tdf[tdf['adx_tier'] == tier]
        if sub.empty: continue
        pnl_s = sub['pnl']
        w = (pnl_s > 0).sum()
        adx_label = {1: 'Weak  (20-25)', 2: 'Normal(25-35)', 3: 'Strong(35+  )'}[tier]
        print(f"  Tier {tier} [{adx_label}]  trades={len(sub)}  wins={w}  "
              f"wr={w/len(sub)*100:.1f}%  total={pnl_s.sum():.1f}  avg={pnl_s.mean():.1f}")

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
    def cs(t):
        if hasattr(t, 'time'): t = t.time()
        h, m = t.hour, t.minute
        if (h == 9 and m >= 45) or h == 10 or (h == 11 and m < 30): return 'Prime'
        if h == 11 or h == 12 or (h == 13 and m < 30):               return 'Midday'
        if h == 13 or h == 14 or (h == 15 and m < 1):                return 'Euro'
        return 'Other'
    tdf3 = tdf.copy()
    tdf3['session'] = tdf3['entry_time'].apply(cs)
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
    for n_t, days in daily_counts.items():
        print(f"  {n_t} trade(s)/day → {days} day(s)")

    print(f"\n{'─'*62}")
    print("  MICRO-TRAIL & BE EFFECTIVENESS")
    print(f"{'─'*62}")
    micro_trades = tdf[tdf['micro_triggered'] == True]
    be_trades    = tdf[tdf['be_triggered']    == True]
    sl_hits      = tdf[tdf['exit_reason']     == 'STOP_LOSS']
    micro_sl     = tdf[tdf['exit_reason']     == 'MICRO_SL']
    print(f"  Trades reaching Micro-Trail stage  : {len(micro_trades)} / {len(tdf)}")
    print(f"  Trades reaching Break-Even stage   : {len(be_trades)} / {len(tdf)}")
    print(f"  Exits via MICRO_SL (saved from full SL) : {len(micro_sl)}")
    if not micro_sl.empty:
        print(f"    → Avg MICRO_SL loss : {micro_sl['pnl'].mean():.1f} pts "
              f"(vs avg full SL: {sl_hits['pnl'].mean():.1f} pts)" if not sl_hits.empty else "")
    print(f"\n  RAW STOP_LOSS analysis:")
    if not sl_hits.empty:
        print(f"    Count              : {len(sl_hits)}")
        print(f"    Avg MAE            : {sl_hits['mae_pts'].mean():.1f} pts")
        print(f"    Avg MFE            : {sl_hits['mfe_pts'].mean():.1f} pts")
        print(f"    Went in-favor >5pt : {(sl_hits['mfe_pts'] > 5).sum()} / {len(sl_hits)}")
    else:
        print(f"    No raw STOP_LOSS exits — micro-trail is working!")


# ══════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    BAR = '═' * 68
    print(f"\n{BAR}")
    print("  SPEED DEMON SCALPER  v4  — IRON STOP SYSTEM")
    print(f"{BAR}")

    cli_args = sys.argv[1:]
    mode     = 'csv' if cli_args else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\nData source : yfinance  |  Symbol : {YFINANCE_SYMBOL}")
        df        = fetch_yfinance('5m', '5m')
        save_base = os.path.join(os.getcwd(),
                                 YFINANCE_SYMBOL.replace('^','').replace('.','_') + '_speed_demon_v4')
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
    print(f"  ADX Filter                       : >= {ADX_MIN}  |  Tiers: T1<{ADX_TIER2}, T2<{ADX_TIER3}, T3+")
    print(f"  LONG  SL                         : {LONG_ATR_SL_MULT}× ATR  (T1: ×{TIER1_SL_FACTOR} tighter)")
    print(f"  SHORT SL                         : {SHORT_ATR_SL_MULT}× ATR (T1: ×{TIER1_SL_FACTOR} tighter)")
    print(f"  Micro-Trail                      : {'ON' if ENABLE_MICRO_TRAIL else 'OFF'}"
          f"  trigger={MICRO_TRAIL_TRIGGER}pts  SL=entry±{MICRO_TRAIL_SL_FACTOR}×ATR")
    print(f"  Break-Even (dual)                : +{BE_POINTS_TRIGGER}pts OR +{LONG_BE_PCT}% (long)")
    print(f"                                   : -{BE_POINTS_TRIGGER}pts OR -{SHORT_BE_PCT}% (short)")
    print(f"  ATR Trail (post-BE)              : ATR × {TRAIL_ATR_MULT}  (T3: ×{TIER3_TRAIL_FACTOR})")
    print(f"  Short Confirm Bars               : {SHORT_CONFIRM_BARS}")
    print(f"  Volume Filter (shorts)           : {'ON' if ENABLE_VOL_FILTER else 'OFF'}"
          f"  >= {VOL_SPIKE_MULT}× {VOL_PERIOD}-bar avg")
    print(f"  EOD Kill Losers                  : {EOD_KILL_LOSERS}")
    print(f"  EOD Kill Non-Trailing            : {EOD_KILL_NON_TRAIL}")
    print(f"  EOD Hard Exit                    : {EOD_HARD_EXIT}")
    print(f"  Max Trades / Day                 : {MAX_TRADES_PER_DAY}")
    print(f"  Consec-Loss Breaker              : {MAX_CONSEC_LOSSES}")
    print(f"  Sessions                         : Prime=ON  Midday={'ON' if ENABLE_MIDDAY else 'OFF'}"
          f"  Euro={'ON' if ENABLE_EURO else 'OFF'}")
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
        print("  Try: lower ADX_MIN, SPREAD_PCT_MIN, PRICE_GAP_MIN, or VOL_SPIKE_MULT.")
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