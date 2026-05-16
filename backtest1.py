"""
╔══════════════════════════════════════════════════════════════════╗
║   SPEED DEMON SCALPER  v1  — "ANTI-CHOP EMA SYSTEM"            ║
╠══════════════════════════════════════════════════════════════════╣
║  PHASE 1 — Data Pre-processing & Environment                    ║
║    DD-MM-YYYY format, IST timezone, forward-fill NaNs           ║
║                                                                  ║
║  PHASE 2 — Technical Indicator Engine                           ║
║    EMA 9  : Fast Signal                                          ║
║    EMA 21 : Trend Confirmation                                   ║
║    EMA 200: Macro Filter ("Grandmaster" line)                    ║
║    ATR 14 : Dynamic Stop Loss sizing                             ║
║                                                                  ║
║  PHASE 3 — Anti-Chop Filter Suite                               ║
║    Spread Filter  : |EMA9 - EMA21| must be >= 0.04% of price    ║
║    Slope Filter   : EMA21 slope over 3 candles must be non-flat ║
║    Price-Gap Filter: Price must not "hug" EMA21                 ║
║                                                                  ║
║  PHASE 4 — Entry & Exit Logic                                   ║
║    Long  : EMA9>EMA21, Close>EMA200, +slope, retest EMA9        ║
║    Short : EMA9<EMA21, Close<EMA200, -slope, retest EMA9        ║
║    SL    : ATR-based (14-period ATR)                             ║
║    TP    : 1:1.5 or 1:2 Risk-to-Reward                          ║
║    Trail : Break Even after 1% in-favor move                    ║
║                                                                  ║
║  PHASE 5 — Indian Market Time-Mapping (NSE)                     ║
║    09:15–09:45 : Observation Zone (no trades)                   ║
║    09:45–11:30 : Prime Scalping Window                          ║
║    11:30–13:30 : Mid-Day Lull (tighter chop filters)            ║
║    13:30–15:00 : European Open (secondary momentum)             ║
║    15:00–15:30 : Square-off Zone (no new trades)                ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python supertrend_simple.py              ← yfinance (default)
    python supertrend_simple.py 5m.csv       ← CSV file
"""

import pandas as pd
import numpy as np
import sys, os
from datetime import time

# ══════════════════════════════════════════════════════════════════
#   CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DATA_MODE        = 'yfinance'      # 'yfinance' or 'csv'
YFINANCE_SYMBOL  = '^NSEI'
YFINANCE_DAYS    = 60
YFINANCE_TZ      = 'Asia/Kolkata'
CSV_5MIN_PATH    = 'nifty_5m.csv'

# ── EMA Periods ─────────────────────────────────────────────────
EMA_FAST         = 9
EMA_SLOW         = 21
EMA_MACRO        = 100

# ── ATR ─────────────────────────────────────────────────────────
ATR_PERIOD       = 14
ATR_SL_MULT      = 0.75    # SL = entry ± (ATR × multiplier)

# ── Anti-Chop Filters ───────────────────────────────────────────
SPREAD_PCT_MIN   = 0.04   # % minimum |EMA9 - EMA21| / price
SLOPE_CANDLES    = 6      # candles for EMA21 slope calculation
SLOPE_MIN        = 0.0    # EMA21 must move at least this many pts over SLOPE_CANDLES
PRICE_GAP_MIN    = 5.0    # pts: price must be at least this far from EMA21

# ── Mid-Day Lull: tighter filters ───────────────────────────────
MIDDAY_SPREAD_MULT = 2.0  # multiply SPREAD_PCT_MIN during mid-day lull

# ── Risk / Reward ───────────────────────────────────────────────
RISK_REWARD      = 3.5    # TP = entry ± (SL_dist × RISK_REWARD)
BREAKEVEN_PCT    = 1.0    # % in-favor move to activate break-even SL

# ── Retest tolerance ────────────────────────────────────────────
RETEST_TOLERANCE = 2.0    # pts: price is "touching" EMA9 within this range

# ── Sessions ────────────────────────────────────────────────────
OBSERVE_START    = time(9,  15)
OBSERVE_END      = time(9,  30)   # no trades before this
PRIME_START      = time(9,  30)
PRIME_END        = time(11, 30)
MIDDAY_START     = time(11, 30)
MIDDAY_END       = time(13, 30)
EURO_START       = time(13, 30)
EURO_END         = time(15, 0)
SQUAREOFF_START  = time(15, 0)
EOD_EXIT_TIME    = time(15, 15)   # force-close all open trades

# ── Directions ──────────────────────────────────────────────────
ENABLE_LONG      = True
ENABLE_SHORT     = True


# ══════════════════════════════════════════════════════════════════
#   INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high  = df['high'].astype(float)
    low   = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_fast']  = compute_ema(df['close'], EMA_FAST)
    df['ema_slow']  = compute_ema(df['close'], EMA_SLOW)
    df['ema_macro'] = compute_ema(df['close'], EMA_MACRO)
    df['atr']       = compute_atr(df, ATR_PERIOD)
    # EMA21 slope: change over last SLOPE_CANDLES candles
    df['ema_slow_slope'] = df['ema_slow'].diff(SLOPE_CANDLES)
    return df


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
    """Returns True if market is trending (not choppy)."""
    # Spread filter
    spread_pct     = abs(ema_fast - ema_slow) / close * 100
    spread_thresh  = SPREAD_PCT_MIN * (MIDDAY_SPREAD_MULT if session == 'midday' else 1.0)
    if spread_pct < spread_thresh:
        return False

    # Slope filter
    if abs(ema_slow_slope) <= SLOPE_MIN:
        return False

    # Price-gap filter: price must not hug EMA21
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
    # Forward-fill NaNs (Phase 1)
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
    # Phase 1: dayfirst=True for DD-MM-YYYY Indian format
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
    closes     = df['close'].astype(float).values
    ema_fast   = df['ema_fast'].values
    ema_slow   = df['ema_slow'].values
    ema_macro  = df['ema_macro'].values
    atrs       = df['atr'].values
    slopes     = df['ema_slow_slope'].values
    ts_list    = df['timestamp'].tolist()
    n          = len(df)

    trades     = []
    equity     = 0.0
    eq_curve   = []

    # ── Trade state ───────────────────────────────────────────
    in_trade     = False
    direction    = None
    entry_price  = 0.0
    entry_time   = None
    stop_loss    = 0.0
    take_profit  = 0.0
    be_triggered = False
    be_level     = 0.0

    def do_enter(direction_str, close_price, ts_now, atr_val):
        nonlocal in_trade, direction, entry_price, entry_time
        nonlocal stop_loss, take_profit, be_triggered, be_level
        in_trade    = True
        direction   = direction_str
        entry_price = close_price
        entry_time  = ts_now
        be_triggered= False
        sl_dist     = atr_val * ATR_SL_MULT
        if direction == 'long':
            stop_loss   = entry_price - sl_dist
            take_profit = entry_price + sl_dist * RISK_REWARD
            be_level    = entry_price * (1 + BREAKEVEN_PCT / 100)
        else:
            stop_loss   = entry_price + sl_dist
            take_profit = entry_price - sl_dist * RISK_REWARD
            be_level    = entry_price * (1 - BREAKEVEN_PCT / 100)

    def do_exit(exit_price, ts_now, reason):
        nonlocal in_trade, direction, entry_price, entry_time
        nonlocal stop_loss, take_profit, be_triggered, be_level, equity
        pnl = round((exit_price - entry_price) if direction == 'long'
                    else (entry_price - exit_price), 2)
        equity += pnl
        trades.append({
            'direction':   direction,
            'entry_time':  entry_time,
            'exit_time':   ts_now,
            'entry_price': entry_price,
            'exit_price':  exit_price,
            'stop_loss':   stop_loss,
            'take_profit': take_profit,
            'pnl':         pnl,
            'exit_reason': reason,
        })
        in_trade     = False
        direction    = None
        entry_price  = 0.0
        entry_time   = None
        stop_loss    = 0.0
        take_profit  = 0.0
        be_triggered = False
        be_level     = 0.0

    prev_date = None

    for idx in range(n):
        close    = closes[idx]
        ef       = ema_fast[idx]
        es       = ema_slow[idx]
        em       = ema_macro[idx]
        atr      = float(atrs[idx])
        slope    = float(slopes[idx]) if not np.isnan(slopes[idx]) else 0.0
        ts       = ts_list[idx]
        c_time   = ts.time()
        c_date   = ts.date()
        session  = get_session(c_time)

        # Day reset
        if c_date != prev_date:
            prev_date = c_date

        # Skip if indicators not warmed up
        if np.isnan(ef) or np.isnan(es) or np.isnan(em) or np.isnan(atr):
            eq_curve.append({'timestamp': ts, 'equity': equity})
            continue

        # ══════════════════════════════════════════════════════
        # MANAGE OPEN TRADE
        # ══════════════════════════════════════════════════════
        if in_trade:
            # Activate break-even SL
            if not be_triggered:
                if direction == 'long'  and close >= be_level:
                    stop_loss    = entry_price
                    be_triggered = True
                elif direction == 'short' and close <= be_level:
                    stop_loss    = entry_price
                    be_triggered = True

            exit_p = None
            exit_r = None

            # Stop Loss
            if direction == 'long' and close <= stop_loss:
                exit_p = stop_loss
                exit_r = 'BREAKEVEN_SL' if be_triggered else 'STOP_LOSS'
            elif direction == 'short' and close >= stop_loss:
                exit_p = stop_loss
                exit_r = 'BREAKEVEN_SL' if be_triggered else 'STOP_LOSS'

            # Take Profit
            elif direction == 'long'  and close >= take_profit:
                exit_p = take_profit
                exit_r = 'TAKE_PROFIT'
            elif direction == 'short' and close <= take_profit:
                exit_p = take_profit
                exit_r = 'TAKE_PROFIT'

            # EOD force-close
            elif c_time >= EOD_EXIT_TIME:
                exit_p = close
                exit_r = 'EOD_EXIT'

            if exit_p is not None:
                do_exit(exit_p, ts, exit_r)

        # ══════════════════════════════════════════════════════
        # ENTRY LOGIC
        # ══════════════════════════════════════════════════════
        if not in_trade:
            # Only trade in prime or euro sessions; no new entries in squareoff/observe
            if session not in ('prime', 'euro', 'midday'):
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            # Anti-chop filters
            if not chop_filters_pass(close, ef, es, slope, session):
                eq_curve.append({'timestamp': ts, 'equity': equity})
                continue

            # ── LONG Entry ────────────────────────────────────
            # EMA9 > EMA21, Close > EMA200, slope positive, price retests EMA9
            if ENABLE_LONG:
                if (ef > es and                             # fast above slow
                    close > em and                          # above macro
                    slope > 0 and                           # upward slope
                    abs(close - ef) <= RETEST_TOLERANCE and # retesting EMA9
                    close > es):                            # still above EMA21
                    do_enter('long', close, ts, atr)

            # ── SHORT Entry ───────────────────────────────────
            # EMA9 < EMA21, Close < EMA200, slope negative, price retests EMA9
            if not in_trade and ENABLE_SHORT:
                if (ef < es and                             # fast below slow
                    close < em and                          # below macro
                    slope < 0 and                           # downward slope
                    abs(close - ef) <= RETEST_TOLERANCE and # retesting EMA9
                    close < es):                            # still below EMA21
                    do_enter('short', close, ts, atr)

        eq_curve.append({'timestamp': ts, 'equity': equity})

    # Force-close at end of data
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
    SEP = '─' * 130
    print(f"\n{SEP}")
    print(f"  TRADE LOG  ({len(tdf)} trades)")
    print(SEP)
    print(f"{'#':<5} {'Dir':<6} {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>9} {'Exit':>9} {'SL':>9} {'TP':>9} {'P&L':>9}  Reason")
    print(SEP)
    for i, r in tdf.iterrows():
        print(f"{i+1:<5} {r['direction'].upper():<6} {fmt_ts(r['entry_time']):<22}"
              f" {fmt_ts(r['exit_time']):<22}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f}"
              f" {r['stop_loss']:>9.2f} {r['take_profit']:>9.2f}"
              f" {r['pnl']:>+9.2f}  {r['exit_reason']}")

    metrics = compute_metrics(tdf)
    print(f"\n{'─'*55}")
    print("  PERFORMANCE METRICS")
    print(f"{'─'*55}")
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")

    print(f"\n{'─'*55}")
    print("  DIRECTION BREAKDOWN")
    print(f"{'─'*55}")
    for d in ['long', 'short']:
        sub = tdf[tdf['direction'] == d]['pnl']
        if sub.empty:
            continue
        w = (sub > 0).sum()
        print(f"  {d.upper():<6}  trades={len(sub)}  wins={w}  "
              f"wr={w/len(sub)*100:.1f}%  total={sub.sum():.1f}  avg={sub.mean():.1f}")

    print(f"\n{'─'*55}")
    print("  EXIT REASON BREAKDOWN")
    print(f"{'─'*55}")
    bd = (tdf.groupby('exit_reason')['pnl']
          .agg(count='count', total_pnl='sum', avg_pnl='mean').round(2))
    print(bd.to_string())

    print(f"\n{'─'*55}")
    print("  MONTHLY P&L BREAKDOWN")
    print(f"{'─'*55}")
    tdf2 = tdf.copy()
    tdf2['month'] = tdf2['entry_time'].dt.tz_localize(None).dt.to_period('M').astype(str)
    monthly = (tdf2.groupby('month')['pnl']
               .agg(trades='count', total_pnl='sum',
                    win_trades=lambda x: (x > 0).sum())
               .assign(win_rate=lambda x: (x['win_trades'] / x['trades'] * 100).round(1))
               .round(2))
    print(monthly.to_string())


# ══════════════════════════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    BAR = '═' * 64
    print(f"\n{BAR}")
    print("  SPEED DEMON SCALPER  v1  — ANTI-CHOP EMA SYSTEM")
    print(f"{BAR}")

    cli_args = sys.argv[1:]
    mode     = 'csv' if cli_args else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\nData source : yfinance  |  Symbol : {YFINANCE_SYMBOL}")
        df        = fetch_yfinance('5m', '5m')
        save_base = os.path.join(os.getcwd(),
                                 YFINANCE_SYMBOL.replace('^', '').replace('.', '_') + '_speed_demon')
    elif mode == 'csv':
        path5 = cli_args[0] if cli_args else CSV_5MIN_PATH
        if not os.path.exists(path5):
            print(f"ERROR: File not found: {path5}"); sys.exit(1)
        print(f"\nLoading 5m CSV : {path5}")
        df        = load_csv(path5, '5m')
        save_base = os.path.splitext(path5)[0] + '_speed_demon'
    else:
        print(f"ERROR: Unknown DATA_MODE '{DATA_MODE}'"); sys.exit(1)

    # Config summary
    print(f"\n{'─'*55}")
    print("  STRATEGY CONFIGURATION")
    print(f"{'─'*55}")
    print(f"  EMA Fast / Slow / Macro : {EMA_FAST} / {EMA_SLOW} / {EMA_MACRO}")
    print(f"  ATR Period              : {ATR_PERIOD}")
    print(f"  ATR SL Multiplier       : {ATR_SL_MULT}x  →  TP ratio: 1:{RISK_REWARD}")
    print(f"  Break-Even Trigger      : +{BREAKEVEN_PCT}% in-favor move")
    print(f"  Anti-Chop Spread Min    : {SPREAD_PCT_MIN}% (×{MIDDAY_SPREAD_MULT} in mid-day lull)")
    print(f"  Slope Filter            : EMA21 slope over {SLOPE_CANDLES} candles > {SLOPE_MIN}")
    print(f"  Price-Gap Min           : {PRICE_GAP_MIN} pts from EMA21")
    print(f"  Retest Tolerance        : ±{RETEST_TOLERANCE} pts from EMA9")
    print(f"  Directions              : Long={'ON' if ENABLE_LONG else 'OFF'}"
          f"  Short={'ON' if ENABLE_SHORT else 'OFF'}")
    print(f"  Observation Zone        : {OBSERVE_START}–{OBSERVE_END} (no trades)")
    print(f"  Prime Window            : {PRIME_START}–{PRIME_END}")
    print(f"  Mid-Day Lull            : {MIDDAY_START}–{MIDDAY_END} (tighter filters)")
    print(f"  European Open           : {EURO_START}–{EURO_END}")
    print(f"  Square-Off Zone         : {SQUAREOFF_START}+ (no new trades)")
    print(f"  EOD Force-Close         : {EOD_EXIT_TIME}")
    print(f"{'─'*55}")

    print(f"\nComputing indicators (EMA {EMA_FAST}/{EMA_SLOW}/{EMA_MACRO}, ATR {ATR_PERIOD}) ...")
    df = compute_indicators(df)

    print("Running backtest ...\n")
    trades_df, equity_df = run_backtest(df)

    print(f"\n{BAR}")
    print("  BACKTEST RESULTS")
    print(f"{BAR}")

    if trades_df.empty:
        print("\n  No trades generated.")
        print("  Try: reduce SPREAD_PCT_MIN, reduce PRICE_GAP_MIN, or widen RETEST_TOLERANCE.")
        return

    print_results(trades_df)

    trades_out = save_base + '_trades.csv'
    equity_out = save_base + '_equity.csv'
    trades_df.to_csv(trades_out, index=False)
    equity_df.to_csv(equity_out, index=False)
    print(f"\n{'─'*55}")
    print(f"  Trades saved : {trades_out}")
    print(f"  Equity curve : {equity_out}")
    print(f"{BAR}\n")


if __name__ == '__main__':
    main()