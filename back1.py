"""
╔══════════════════════════════════════════════════════════════════╗
║   SPEED DEMON SCALPER  v2  — "PRECISION EXIT SYSTEM"           ║
╠══════════════════════════════════════════════════════════════════╣
║  WHAT'S NEW vs v1                                                ║
║  ─────────────────────────────────────────────────────────────  ║
║  ✦ 15-minute HTF Bias Filter for LONG entries                   ║
║      → Long only when 15m EMA9 > 15m EMA21 > 15m EMA50         ║
║      → Avoids counter-trend longs in bearish macro structure    ║
║                                                                  ║
║  ✦ Asymmetric SL/RR by direction                                ║
║      LONG  : SL = 1.2× ATR  |  RR = 2.5  (room to breathe)    ║
║      SHORT : SL = 0.75× ATR |  RR = 3.5  (catch waterfalls)   ║
║                                                                  ║
║  ✦ No Hard Take-Profit — Dynamic Exit Logic Instead             ║
║      Stage 1 : Break-Even SL after BREAKEVEN_PCT move           ║
║      Stage 2 : ATR Trailing Stop kicks in after BE              ║
║      Stage 3 : EMA Cross Exit — exit when EMA9 crosses EMA21   ║
║      Stage 4 : Momentum Exit — exit on EMA9 slope reversal      ║
║      Stage 5 : Session Exit — force-close at EOD                ║
║                                                                  ║
║  ✦ Max 3 Trades Per Day Cap                                     ║
║      → Prevents overtrading in choppy conditions                ║
║                                                                  ║
║  ✦ 15-Minute HTF data auto-resampled from 5m bars               ║
╠══════════════════════════════════════════════════════════════════╣
║  PHASE 1 — Data Pre-processing & Environment                    ║
║  PHASE 2 — Technical Indicator Engine (5m + 15m HTF)           ║
║  PHASE 3 — Anti-Chop Filter Suite                               ║
║  PHASE 4 — Asymmetric Entry Logic (Long vs Short)               ║
║  PHASE 5 — Dynamic Exit Engine (No Hard TP)                     ║
║  PHASE 6 — Indian Market Time-Mapping (NSE)                     ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python speed_demon_v2.py              ← yfinance (default)
    python speed_demon_v2.py 5m.csv       ← CSV file
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

# ── HTF (15m) EMA Periods — for Long bias filter ────────────────
HTF_EMA_FAST     = 9
HTF_EMA_SLOW     = 21
HTF_EMA_TREND    = 50     # 15m trend spine

# ── ATR ─────────────────────────────────────────────────────────
ATR_PERIOD       = 14

# ── Asymmetric SL/RR ────────────────────────────────────────────
#    LONG: wider SL to let grind trades breathe
LONG_ATR_SL_MULT    = 1.2    # SL = entry - (ATR × 1.2)
LONG_INITIAL_RR     = 2.5    # used only to compute initial min-target reference
#    SHORT: tight SL to catch high-velocity drops
SHORT_ATR_SL_MULT   = 0.75   # SL = entry + (ATR × 0.75)
SHORT_INITIAL_RR    = 3.5    # reference only

# ── Trailing Stop (activates after Break-Even) ──────────────────
TRAIL_ATR_MULT      = 0.6    # trail distance = ATR × this
                              # higher = looser trail (more room), lower = tighter

# ── Break-Even Trigger ──────────────────────────────────────────
#    LONG:  BE triggers after price moves +LONG_BE_PCT%
#    SHORT: BE triggers after price moves -SHORT_BE_PCT%
LONG_BE_PCT         = 0.5    # % in-favor move for longs
SHORT_BE_PCT        = 0.4    # % in-favor move for shorts (quicker BE, protect gains)

# ── EMA Cross Exit ──────────────────────────────────────────────
#    Exit long if EMA9 crosses below EMA21 (trend reversal signal)
#    Exit short if EMA9 crosses above EMA21
ENABLE_EMA_CROSS_EXIT = True

# ── Momentum (Slope Reversal) Exit ──────────────────────────────
#    Exit if EMA21 slope flips against the trade direction after BE
ENABLE_SLOPE_EXIT     = True
SLOPE_EXIT_CANDLES    = 4    # candles to measure slope for exit check

# ── Anti-Chop Filters ───────────────────────────────────────────
SPREAD_PCT_MIN      = 0.04
SLOPE_CANDLES       = 6
SLOPE_MIN           = 0.0
PRICE_GAP_MIN       = 5.0
MIDDAY_SPREAD_MULT  = 2.0

# ── Retest tolerance (ATR-relative) ─────────────────────────────
RETEST_ATR_MULT     = 0.15   # touch = within ATR × 0.15 of EMA9

# ── Max trades per day ───────────────────────────────────────────
MAX_TRADES_PER_DAY  = 3

# ── Directions ──────────────────────────────────────────────────
ENABLE_LONG         = True
ENABLE_SHORT        = True

# ── Sessions ────────────────────────────────────────────────────
OBSERVE_START       = time(9,  15)
OBSERVE_END         = time(9,  45)   # extended: first 30 min is noise
PRIME_START         = time(9,  45)
PRIME_END           = time(11, 30)
MIDDAY_START        = time(11, 30)
MIDDAY_END          = time(13, 30)
EURO_START          = time(13, 30)
EURO_END            = time(15, 0)
SQUAREOFF_START     = time(15, 0)
EOD_EXIT_TIME       = time(15, 15)


# ══════════════════════════════════════════════════════════════════
#   INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high      = df['high'].astype(float)
    low       = df['low'].astype(float)
    close     = df['close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 5-minute indicators."""
    df = df.copy()
    df['ema_fast']       = compute_ema(df['close'], EMA_FAST)
    df['ema_slow']       = compute_ema(df['close'], EMA_SLOW)
    df['ema_macro']      = compute_ema(df['close'], EMA_MACRO)
    df['atr']            = compute_atr(df, ATR_PERIOD)
    # Entry slope: over SLOPE_CANDLES bars
    df['ema_slow_slope'] = df['ema_slow'].diff(SLOPE_CANDLES)
    # Exit slope: shorter window for faster reaction
    df['ema_slow_slope_exit'] = df['ema_slow'].diff(SLOPE_EXIT_CANDLES)
    return df


def compute_htf_bias(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 5m bars to 15m, compute HTF EMAs, then forward-fill
    back to 5m index so each 5m bar knows the current HTF bias.

    Returns a DataFrame indexed like df_5m with columns:
        htf_ema_fast, htf_ema_slow, htf_ema_trend,
        htf_long_bias (bool), htf_short_bias (bool)
    """
    df_5m = df_5m.copy()
    df_5m = df_5m.set_index('timestamp')

    # Resample to 15m OHLC
    df_15m = df_5m['close'].resample('15min').ohlc().dropna()
    df_15m.columns = ['open', 'high', 'low', 'close']

    df_15m['htf_ema_fast']  = compute_ema(df_15m['close'], HTF_EMA_FAST)
    df_15m['htf_ema_slow']  = compute_ema(df_15m['close'], HTF_EMA_SLOW)
    df_15m['htf_ema_trend'] = compute_ema(df_15m['close'], HTF_EMA_TREND)

    # Bullish HTF: fast > slow > trend (all stacked up)
    df_15m['htf_long_bias'] = (
        (df_15m['htf_ema_fast'] > df_15m['htf_ema_slow']) &
        (df_15m['htf_ema_slow'] > df_15m['htf_ema_trend'])
    )
    # Bearish HTF: fast < slow < trend (all stacked down)
    df_15m['htf_short_bias'] = (
        (df_15m['htf_ema_fast'] < df_15m['htf_ema_slow']) &
        (df_15m['htf_ema_slow'] < df_15m['htf_ema_trend'])
    )

    htf_cols = df_15m[['htf_ema_fast', 'htf_ema_slow', 'htf_ema_trend',
                        'htf_long_bias', 'htf_short_bias']]

    # Reindex to 5m timestamps, forward-fill (each 5m bar uses latest 15m value)
    htf_reindexed = htf_cols.reindex(df_5m.index, method='ffill')

    # Merge back
    result = df_5m.join(htf_reindexed).reset_index()
    return result


# ══════════════════════════════════════════════════════════════════
#   ANTI-CHOP FILTERS
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
                      ema_slow_slope: float, session: str) -> bool:
    spread_pct    = abs(ema_fast - ema_slow) / close * 100
    spread_thresh = SPREAD_PCT_MIN * (MIDDAY_SPREAD_MULT if session == 'midday' else 1.0)
    if spread_pct < spread_thresh:
        return False
    if abs(ema_slow_slope) <= SLOPE_MIN:
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
#   BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame) -> tuple:
    closes          = df['close'].astype(float).values
    ema_fast        = df['ema_fast'].values
    ema_slow        = df['ema_slow'].values
    ema_macro       = df['ema_macro'].values
    atrs            = df['atr'].values
    slopes          = df['ema_slow_slope'].values
    slopes_exit     = df['ema_slow_slope_exit'].values
    htf_long_bias   = df['htf_long_bias'].values   # bool
    htf_short_bias  = df['htf_short_bias'].values  # bool
    ts_list         = df['timestamp'].tolist()
    n               = len(df)

    trades          = []
    equity          = 0.0
    eq_curve        = []

    # ── Trade State ───────────────────────────────────────────────
    in_trade        = False
    direction       = None
    entry_price     = 0.0
    entry_time      = None
    stop_loss       = 0.0
    be_triggered    = False
    be_level        = 0.0
    trail_active    = False
    sl_dist_initial = 0.0   # stored for reference in reporting

    # ── Daily tracking ────────────────────────────────────────────
    prev_date       = None
    daily_trades    = {}    # date_str -> count

    def do_enter(dir_str, close_price, ts_now, atr_val):
        nonlocal in_trade, direction, entry_price, entry_time
        nonlocal stop_loss, be_triggered, be_level, trail_active, sl_dist_initial

        sl_mult   = LONG_ATR_SL_MULT if dir_str == 'long' else SHORT_ATR_SL_MULT
        sl_dist   = atr_val * sl_mult

        in_trade        = True
        direction       = dir_str
        entry_price     = close_price
        entry_time      = ts_now
        be_triggered    = False
        trail_active    = False
        sl_dist_initial = sl_dist

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

        pnl = round(
            (exit_price - entry_price) if direction == 'long'
            else (entry_price - exit_price), 2
        )
        equity += pnl
        trades.append({
            'direction':    direction,
            'entry_time':   entry_time,
            'exit_time':    ts_now,
            'entry_price':  entry_price,
            'exit_price':   exit_price,
            'stop_loss_at_exit': stop_loss,
            'pnl':          pnl,
            'exit_reason':  reason,
        })
        in_trade        = False
        direction       = None
        entry_price     = 0.0
        entry_time      = None
        stop_loss       = 0.0
        be_triggered    = False
        be_level        = 0.0
        trail_active    = False
        sl_dist_initial = 0.0

    prev_date = None

    for idx in range(n):
        close      = closes[idx]
        ef         = ema_fast[idx]
        es         = ema_slow[idx]
        em         = ema_macro[idx]
        atr        = float(atrs[idx])
        slope      = float(slopes[idx])      if not np.isnan(slopes[idx])      else 0.0
        slope_exit = float(slopes_exit[idx]) if not np.isnan(slopes_exit[idx]) else 0.0
        ts         = ts_list[idx]
        c_time     = ts.time()
        c_date     = ts.date()
        session    = get_session(c_time)
        date_str   = str(c_date)

        # Day reset
        if c_date != prev_date:
            prev_date = c_date

        # Warm-up guard
        if np.isnan(ef) or np.isnan(es) or np.isnan(em) or np.isnan(atr):
            eq_curve.append({'timestamp': ts, 'equity': equity})
            continue

        # HTF bias values (may be NaN early)
        htf_long  = bool(htf_long_bias[idx])  if not pd.isna(htf_long_bias[idx])  else False
        htf_short = bool(htf_short_bias[idx]) if not pd.isna(htf_short_bias[idx]) else False

        # ══════════════════════════════════════════════════════════
        #   MANAGE OPEN TRADE  —  Dynamic Exit Engine
        # ══════════════════════════════════════════════════════════
        if in_trade:

            # ── Stage 1: Break-Even trigger ───────────────────────
            if not be_triggered:
                if direction == 'long'  and close >= be_level:
                    stop_loss    = entry_price + 1.0   # 1 pt above entry (avoid slippage BE)
                    be_triggered = True
                    trail_active = True
                elif direction == 'short' and close <= be_level:
                    stop_loss    = entry_price - 1.0   # 1 pt below entry
                    be_triggered = True
                    trail_active = True

            # ── Stage 2: ATR Trailing Stop (after BE) ─────────────
            if trail_active:
                trail_dist = atr * TRAIL_ATR_MULT
                if direction == 'long':
                    new_trail_sl = close - trail_dist
                    stop_loss = max(stop_loss, new_trail_sl)   # ratchet up only
                else:
                    new_trail_sl = close + trail_dist
                    stop_loss = min(stop_loss, new_trail_sl)   # ratchet down only

            exit_p = None
            exit_r = None

            # ── Stage 3: Hard Stop Loss / Trailing Stop Hit ───────
            if direction == 'long' and close <= stop_loss:
                exit_p = stop_loss
                exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'
            elif direction == 'short' and close >= stop_loss:
                exit_p = stop_loss
                exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'

            # ── Stage 4: EMA Cross Exit ───────────────────────────
            # EMA9 has crossed EMA21 against the trade — trend reversal
            elif ENABLE_EMA_CROSS_EXIT and be_triggered:
                if direction == 'long'  and ef < es:
                    exit_p = close
                    exit_r = 'EMA_CROSS_EXIT'
                elif direction == 'short' and ef > es:
                    exit_p = close
                    exit_r = 'EMA_CROSS_EXIT'

            # ── Stage 5: Momentum (Slope Reversal) Exit ───────────
            # EMA21 slope has flipped against trade direction post-BE
            elif ENABLE_SLOPE_EXIT and be_triggered:
                if direction == 'long'  and slope_exit < -SLOPE_MIN:
                    exit_p = close
                    exit_r = 'SLOPE_REVERSAL_EXIT'
                elif direction == 'short' and slope_exit > SLOPE_MIN:
                    exit_p = close
                    exit_r = 'SLOPE_REVERSAL_EXIT'

            # ── Stage 6: EOD Force-Close ──────────────────────────
            elif c_time >= EOD_EXIT_TIME:
                exit_p = close
                exit_r = 'EOD_EXIT'

            if exit_p is not None:
                do_exit(exit_p, ts, exit_r)

        # ══════════════════════════════════════════════════════════
        #   ENTRY LOGIC
        # ══════════════════════════════════════════════════════════
        if not in_trade:
            if session not in ('prime', 'euro', 'midday'):
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            # Max trades per day gate
            if daily_trades.get(date_str, 0) >= MAX_TRADES_PER_DAY:
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            # Anti-chop filters
            if not chop_filters_pass(close, ef, es, slope, session):
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            # ATR-relative retest tolerance
            retest_tol = atr * RETEST_ATR_MULT

            # ── LONG Entry ────────────────────────────────────────
            # Requires: 5m trend aligned + 15m HTF bullish bias
            if ENABLE_LONG and htf_long:
                if (ef > es            and   # 5m fast above slow
                    close > em         and   # above 5m macro
                    slope > 0          and   # 5m upward slope
                    abs(close - ef) <= retest_tol and  # retesting EMA9
                    close > es):             # still above EMA21
                    do_enter('long', close, ts, atr)
                    daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

            # ── SHORT Entry ───────────────────────────────────────
            # No HTF restriction on shorts — shorts work in any macro regime
            if not in_trade and ENABLE_SHORT:
                if (ef < es            and   # 5m fast below slow
                    close < em         and   # below 5m macro
                    slope < 0          and   # 5m downward slope
                    abs(close - ef) <= retest_tol and  # retesting EMA9
                    close < es):             # still below EMA21
                    do_enter('short', close, ts, atr)
                    daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

        eq_curve.append({'timestamp': ts, 'equity': equity})

    # Force-close remaining open trade
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
    }


def fmt_ts(ts):
    return ts.strftime('%d-%m-%Y %H:%M') if hasattr(ts, 'strftime') else str(ts)


def print_results(tdf: pd.DataFrame):
    SEP = '─' * 140

    print(f"\n{SEP}")
    print(f"  TRADE LOG  ({len(tdf)} trades)")
    print(SEP)
    print(f"{'#':<5} {'Dir':<6} {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>9} {'Exit':>9} {'SL@Exit':>9} {'P&L':>9}  Reason")
    print(SEP)
    for i, r in tdf.iterrows():
        print(f"{i+1:<5} {r['direction'].upper():<6} {fmt_ts(r['entry_time']):<22}"
              f" {fmt_ts(r['exit_time']):<22}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f}"
              f" {r['stop_loss_at_exit']:>9.2f}"
              f" {r['pnl']:>+9.2f}  {r['exit_reason']}")

    metrics = compute_metrics(tdf)
    print(f"\n{'─'*60}")
    print("  PERFORMANCE METRICS")
    print(f"{'─'*60}")
    for k, v in metrics.items():
        print(f"  {k:<30}: {v}")

    print(f"\n{'─'*60}")
    print("  DIRECTION BREAKDOWN")
    print(f"{'─'*60}")
    for d in ['long', 'short']:
        sub = tdf[tdf['direction'] == d]['pnl']
        if sub.empty:
            continue
        w = (sub > 0).sum()
        print(f"  {d.upper():<6}  trades={len(sub)}  wins={w}  "
              f"wr={w/len(sub)*100:.1f}%  total={sub.sum():.1f}  avg={sub.mean():.1f}")

    print(f"\n{'─'*60}")
    print("  EXIT REASON BREAKDOWN")
    print(f"{'─'*60}")
    bd = (tdf.groupby('exit_reason')['pnl']
          .agg(count='count', total_pnl='sum', avg_pnl='mean').round(2))
    print(bd.to_string())

    print(f"\n{'─'*60}")
    print("  MONTHLY P&L BREAKDOWN")
    print(f"{'─'*60}")
    tdf2 = tdf.copy()
    tdf2['month'] = tdf2['entry_time'].dt.tz_localize(None).dt.to_period('M').astype(str)
    monthly = (tdf2.groupby('month')['pnl']
               .agg(trades='count', total_pnl='sum',
                    win_trades=lambda x: (x > 0).sum())
               .assign(win_rate=lambda x: (x['win_trades'] / x['trades'] * 100).round(1))
               .round(2))
    print(monthly.to_string())

    print(f"\n{'─'*60}")
    print("  SESSION P&L BREAKDOWN")
    print(f"{'─'*60}")
    def classify_session(t):
        h = t.hour
        m = t.minute
        if (h == 9  and m >= 45) or (h == 10) or (h == 11 and m == 0):
            return 'Prime (09:45-11:30)'
        if h == 11 or (h == 12) or (h == 13 and m == 0):
            return 'Midday (11:30-13:30)'
        if h == 13 or (h == 14) or (h == 15 and m == 0):
            return 'Euro (13:30-15:00)'
        return 'Other'
    tdf3 = tdf.copy()
    tdf3['session'] = tdf3['entry_time'].apply(
        lambda ts: classify_session(ts.time() if hasattr(ts, 'time') else ts)
    )
    sess_bd = (tdf3.groupby('session')['pnl']
               .agg(trades='count', total_pnl='sum', avg_pnl='mean',
                    win_trades=lambda x: (x > 0).sum())
               .assign(win_rate=lambda x: (x['win_trades'] / x['trades'] * 100).round(1))
               .round(2))
    print(sess_bd.to_string())

    print(f"\n{'─'*60}")
    print("  DAILY TRADE COUNT DISTRIBUTION")
    print(f"{'─'*60}")
    tdf4 = tdf.copy()
    tdf4['date'] = tdf4['entry_time'].dt.date
    daily_counts = tdf4.groupby('date').size().value_counts().sort_index()
    for n_trades, days in daily_counts.items():
        print(f"  {n_trades} trade(s)/day → {days} day(s)")


# ══════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    BAR = '═' * 68
    print(f"\n{BAR}")
    print("  SPEED DEMON SCALPER  v2  — PRECISION EXIT SYSTEM")
    print(f"{BAR}")

    cli_args = sys.argv[1:]
    mode     = 'csv' if cli_args else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\nData source : yfinance  |  Symbol : {YFINANCE_SYMBOL}")
        df        = fetch_yfinance('5m', '5m')
        save_base = os.path.join(os.getcwd(),
                                 YFINANCE_SYMBOL.replace('^', '').replace('.', '_') + '_speed_demon_v2')
    elif mode == 'csv':
        path5 = cli_args[0] if cli_args else CSV_5MIN_PATH
        if not os.path.exists(path5):
            print(f"ERROR: File not found: {path5}"); sys.exit(1)
        print(f"\nLoading 5m CSV : {path5}")
        df        = load_csv(path5, '5m')
        save_base = os.path.splitext(path5)[0] + '_speed_demon_v2'
    else:
        print(f"ERROR: Unknown DATA_MODE '{DATA_MODE}'"); sys.exit(1)

    # Config summary
    print(f"\n{'─'*60}")
    print("  STRATEGY CONFIGURATION  (v2)")
    print(f"{'─'*60}")
    print(f"  EMA Fast / Slow / Macro (5m)    : {EMA_FAST} / {EMA_SLOW} / {EMA_MACRO}")
    print(f"  HTF EMAs (15m)                  : {HTF_EMA_FAST} / {HTF_EMA_SLOW} / {HTF_EMA_TREND}")
    print(f"  ATR Period                       : {ATR_PERIOD}")
    print(f"  LONG  SL Multiplier             : {LONG_ATR_SL_MULT}× ATR")
    print(f"  SHORT SL Multiplier             : {SHORT_ATR_SL_MULT}× ATR")
    print(f"  LONG  Break-Even Trigger        : +{LONG_BE_PCT}% in-favor")
    print(f"  SHORT Break-Even Trigger        : -{SHORT_BE_PCT}% in-favor (faster)")
    print(f"  Trailing Stop (post-BE)         : ATR × {TRAIL_ATR_MULT}")
    print(f"  EMA Cross Exit                  : {'ON' if ENABLE_EMA_CROSS_EXIT else 'OFF'}")
    print(f"  Slope Reversal Exit             : {'ON' if ENABLE_SLOPE_EXIT else 'OFF'}")
    print(f"  Retest Tolerance                : ATR × {RETEST_ATR_MULT}")
    print(f"  Max Trades / Day                : {MAX_TRADES_PER_DAY}")
    print(f"  Anti-Chop Spread Min            : {SPREAD_PCT_MIN}% (×{MIDDAY_SPREAD_MULT} mid-day)")
    print(f"  Price-Gap Min                   : {PRICE_GAP_MIN} pts from EMA21")
    print(f"  Directions                      : Long={'ON' if ENABLE_LONG else 'OFF'}"
          f"  Short={'ON' if ENABLE_SHORT else 'OFF'}")
    print(f"  Observation Zone                : {OBSERVE_START}–{OBSERVE_END} (no trades)")
    print(f"  Prime Window                    : {PRIME_START}–{PRIME_END}")
    print(f"  EOD Force-Close                 : {EOD_EXIT_TIME}")
    print(f"  HTF Long Bias                   : 15m EMA{HTF_EMA_FAST} > EMA{HTF_EMA_SLOW} > EMA{HTF_EMA_TREND}")
    print(f"  HTF Short Bias                  : No HTF restriction (shorts work in any regime)")
    print(f"{'─'*60}")

    print(f"\nComputing 5m indicators ...")
    df = compute_indicators_5m(df)

    print(f"Computing 15m HTF bias (resampled) ...")
    df = compute_htf_bias(df)

    print("Running backtest ...\n")
    trades_df, equity_df = run_backtest(df)

    print(f"\n{BAR}")
    print("  BACKTEST RESULTS")
    print(f"{BAR}")

    if trades_df.empty:
        print("\n  No trades generated.")
        print("  Try: reduce SPREAD_PCT_MIN, PRICE_GAP_MIN, or RETEST_ATR_MULT.")
        return

    print_results(trades_df)

    trades_out = save_base + '_trades.csv'
    equity_out = save_base + '_equity.csv'
    trades_df.to_csv(trades_out, index=False)
    equity_df.to_csv(equity_out, index=False)
    print(f"\n{'─'*60}")
    print(f"  Trades saved : {trades_out}")
    print(f"  Equity curve : {equity_out}")
    print(f"{BAR}\n")


if __name__ == '__main__':
    main()