"""
╔══════════════════════════════════════════════════════════════════╗
║   SUPERTREND SEQUENTIAL CONFIRMATION BACKTESTER  v3.0           ║
║   Primary TF  : 5-min   (your entry chart)                      ║
║   HTF Filter  : 15-min  (trend confirmation)                    ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    python supertrend_backtest.py <5min_csv> [15min_csv]

    If you supply only one CSV, HTF filter is disabled automatically.
    If you supply two CSVs, HTF 15-min filter is applied.

CSV FORMAT  (both files):
    date/datetime/timestamp | open | high | low | close | volume
    Date format: DD-MM-YYYY HH:MM  OR  YYYY-MM-DD HH:MM
"""

import pandas as pd
import numpy as np
import sys, os
from datetime import time

# ══════════════════════════════════════════════════════════════════
#   ★  STRATEGY CONFIGURATION — CHANGE ANYTHING HERE  ★
# ══════════════════════════════════════════════════════════════════

# ── Supertrend parameters (applied to BOTH timeframes) ──
ST_ATR_PERIOD   = 7       # ATR period for Supertrend
ST_MULTIPLIER   = 4.0      # ATR multiplier for Supertrend

# ── HTF (15-min) Supertrend filter ──
HTF_FILTER_ON   = True     # True = require HTF trend agreement before entry
                           # False = ignore HTF, trade on 5-min ST only

# ── Stop Loss ──
# HYBRID SL SYSTEM (always active):
#   At entry   → fixed % SL (SL_PCT) always used — guarantees SL is on correct side
#   While open → if USE_FIXED_SL=False, SL trails with the ST line (only improves, never worsens)
#                The ST line only updates the SL when it moves beyond the fixed % SL
#   USE_FIXED_SL=True  → keeps SL locked at the initial % level forever (no trailing)
#   USE_FIXED_SL=False → SL starts at % level, then trails ST line as it moves in your favour
USE_FIXED_SL    = False    # False = trail with ST line after initial % SL (recommended)
                           # True  = fixed % SL only, no trailing
SL_PCT          = 0.0025   # Initial SL distance from entry (0.0025 = 0.25% ≈ 60pts on Nifty 24k)
                           # Tune this: 0.0020=~48pts | 0.0025=~60pts | 0.0030=~72pts

# ── ST Touch exit ──
ST_TOUCH_EXIT   = True     # True = exit when close crosses ST line
                           # False = disable ST touch exit (only SL + trailing)
ST_TOUCH_MIN_PROFIT = 0    # Minimum profit pts before ST touch can exit
                           # Set to 0 to exit on any ST touch
                           # Set to e.g. 20 to only exit on ST touch if profit >= 20 pts

# ── Trailing profit exit ──
TRAIL_ACTIVATE_PTS  = 60  # Minimum peak profit (pts) to activate trailing exit
TRAIL_DRAWDOWN_PCT  = 0.50 # Exit if profit drops this fraction from peak
                           # 0.50 = exit if profit falls 50% from peak (was 0.66)
                           # 0.66 = original setting

# ── Time filter ──
NO_ENTRY_AFTER  = time(13, 30)   # Block new entries after this time (IST)
                                  # Set to time(15, 15) to disable filter
EOD_EXIT_TIME   = time(15, 20)   # Force-exit any open trade at/after this time
                                  # Script auto-detects last candle if no candle at this exact time

# ── 70-point drop / recovery rule ──
DROP_RULE_PTS   = 70       # Points drop threshold to activate reversal rule
DROP_RECOVERY   = 0.50     # Fraction of drop that must recover before entry (0.5 = 50%)

# ── Candle Body Strength Filter ──
# The confirmation candle (the one that triggers entry) must have a minimum body size.
# Body = abs(close - open). This filters out doji/spinning-top candles that are weak.
# Set to 0 to disable.
CANDLE_BODY_FILTER  = True   # True = apply body filter on confirmation candle
MIN_CANDLE_BODY_PCT = 0.30   # Confirmation candle body must be >= this % of its range (high-low)
                             # 0.30 = body must cover 30% of the candle's range
                             # 0.00 = disabled | 0.50 = strong filter

# ── Consecutive SL Cooldown ──
# After hitting N consecutive stop losses, stop trading for the rest of that day.
# This prevents over-trading in choppy/trending-against markets.
# Set MAX_CONSEC_SL = 999 to disable.
MAX_CONSEC_SL   = 2          # Max consecutive SL hits before pausing for the day
                             # 1 = stop after first SL | 2 = stop after two in a row | 999 = off

# ── Minimum Candles Between Trades ──
# After any exit, wait at least this many candles before next entry.
# Prevents immediately re-entering a choppy market after a loss.
# Set to 0 to disable (original behaviour: only wait for next ST flip).
MIN_CANDLES_BETWEEN_TRADES = 3   # Candles to wait after exit before re-entry allowed
                                  # 0 = disabled | 3 = ~15min on 5-min chart

# ══════════════════════════════════════════════════════════════════
#   END OF CONFIGURATION
# ══════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
# 1. SUPERTREND INDICATOR
# ─────────────────────────────────────────────

def compute_supertrend(df: pd.DataFrame,
                       atr_period: int   = ST_ATR_PERIOD,
                       multiplier: float = ST_MULTIPLIER,
                       col_suffix: str   = '') -> pd.DataFrame:
    """
    Compute Supertrend. Adds columns:
        st_line{suffix}, st_direction{suffix}   (1=bullish, -1=bearish)
    """
    df   = df.copy()
    high  = df['high'].astype(float).values
    low   = df['low'].astype(float).values
    close = df['close'].astype(float).values
    n     = len(df)

    # True Range
    tr    = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i]  - close[i-1]))

    # Wilder's smoothed ATR (RMA)
    atr    = np.empty(n)
    atr[0] = tr[0]
    alpha  = 1.0 / atr_period
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i-1]

    hl2       = (high + low) / 2.0
    raw_upper = hl2 + multiplier * atr
    raw_lower = hl2 - multiplier * atr

    final_upper = raw_upper.copy()
    final_lower = raw_lower.copy()

    for i in range(1, n):
        final_upper[i] = raw_upper[i] \
            if (raw_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) \
            else final_upper[i-1]
        final_lower[i] = raw_lower[i] \
            if (raw_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) \
            else final_lower[i-1]

    st_line      = np.empty(n)
    st_direction = np.ones(n, dtype=int)
    st_line[0]   = final_lower[0]

    for i in range(1, n):
        if st_direction[i-1] == 1:
            if close[i] < final_lower[i]:
                st_direction[i] = -1
                st_line[i]      = final_upper[i]
            else:
                st_direction[i] = 1
                st_line[i]      = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                st_direction[i] = 1
                st_line[i]      = final_lower[i]
            else:
                st_direction[i] = -1
                st_line[i]      = final_upper[i]

    df[f'st_line{col_suffix}']      = st_line
    df[f'st_direction{col_suffix}'] = st_direction
    return df


# ─────────────────────────────────────────────
# 2. BUILD HTF (15-MIN) SUPERTREND LOOKUP
# ─────────────────────────────────────────────

def build_htf_lookup(htf_df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series indexed by timestamp of HTF candles,
    values = HTF ST direction (1 or -1).
    Used to look up the *last known* HTF direction at any 5-min candle time.
    """
    htf_df = compute_supertrend(htf_df, col_suffix='_htf')
    # Index by timestamp for merge
    return htf_df.set_index('timestamp')['st_direction_htf']


def get_htf_direction_at(htf_series: pd.Series, ts) -> int:
    """
    Return the HTF ST direction that was valid AT time ts.
    Uses the last HTF candle whose timestamp <= ts (forward-fill logic).
    Returns 0 if no HTF data available yet.
    """
    candidates = htf_series[htf_series.index <= ts]
    if candidates.empty:
        return 0
    return int(candidates.iloc[-1])


# ─────────────────────────────────────────────
# 3. EOD FLAG PRE-COMPUTATION
# ─────────────────────────────────────────────

def compute_eod_flags(df: pd.DataFrame) -> np.ndarray:
    """
    Returns boolean array: True on the candle where EOD exit should trigger.
    Logic:
      - First candle of each day with time >= EOD_EXIT_TIME, OR
      - Last candle of day (if no candle reaches EOD_EXIT_TIME) with time >= 15:00
    """
    df    = df.copy()
    dates = df['timestamp'].dt.date
    times = df['timestamp'].dt.time
    is_eod = np.zeros(len(df), dtype=bool)

    for date_val, grp in df.groupby(dates):
        grp_times = grp['timestamp'].dt.time
        # First candle >= EOD_EXIT_TIME
        eod_mask  = grp_times >= EOD_EXIT_TIME
        if eod_mask.any():
            first_eod_idx = grp.index[eod_mask][0]
            is_eod[first_eod_idx] = True
        else:
            # No candle at/after EOD_EXIT_TIME → use last candle if >= 15:00
            last_idx = grp.index[-1]
            if grp_times.iloc[-1] >= time(15, 0):
                is_eod[last_idx] = True

    return is_eod


# ─────────────────────────────────────────────
# 4. BACKTEST ENGINE
# ─────────────────────────────────────────────

def run_backtest(df: pd.DataFrame,
                 session_open_time: time,
                 htf_series: pd.Series = None) -> tuple:
    """
    Main backtest loop.
    htf_series: optional HTF ST direction series (indexed by timestamp).
    Returns (trades_df, equity_curve_df).
    """
    use_htf = (htf_series is not None) and HTF_FILTER_ON

    # ── Helpers ──────────────────────────────
    def make_entry(close_price, direction, ts, idx, st_line_val):
        # Initial SL is always a fixed % from entry price.
        # This guarantees SL is on the CORRECT side of price at entry,
        # even if the ST line value is on the wrong side right after a flip.
        sl = (close_price - close_price * SL_PCT) if direction == 'long' \
             else (close_price + close_price * SL_PCT)
        return {'position':    direction,
                'entry_price': close_price,
                'entry_time':  ts,
                'entry_index': idx,
                'stop_loss':   sl,
                'peak_profit': 0.0}

    def update_dynamic_sl(trd, st_line_val):
        """
        After entry, trail SL with the ST line — but ONLY if:
          - USE_FIXED_SL is False (dynamic mode)
          - The ST line is on the CORRECT side of price (below price for long, above for short)
          - Moving SL to ST line would IMPROVE it (ratchet only, never worsen)
        This avoids the bug where ST line sits above entry price right after a bullish flip.
        """
        if USE_FIXED_SL:
            return
        d = trd['position']
        ep = trd['entry_price']
        if d == 'long':
            # Only use ST line as SL if it's below current SL AND below entry price
            # (ensures it's actually a valid stop, not above price)
            if st_line_val < ep and st_line_val > trd['stop_loss']:
                trd['stop_loss'] = st_line_val
        else:
            # Only use ST line as SL if it's above current SL AND above entry price
            if st_line_val > ep and st_line_val < trd['stop_loss']:
                trd['stop_loss'] = st_line_val

    def record_exit(trd, exit_price, exit_time, reason, tlist):
        d   = trd['position']
        pnl = (exit_price - trd['entry_price']) if d == 'long' \
              else (trd['entry_price'] - exit_price)
        tlist.append({
            'entry_time':  trd['entry_time'],
            'exit_time':   exit_time,
            'direction':   d,
            'entry_price': trd['entry_price'],
            'exit_price':  exit_price,
            'stop_loss':   trd['stop_loss'],
            'peak_profit': trd['peak_profit'],
            'pnl':         round(pnl, 2),
            'exit_reason': reason,
        })
        return round(pnl, 2)

    # ── Pre-compute EOD flags ─────────────────
    is_eod = compute_eod_flags(df)

    # ── State ─────────────────────────────────
    state          = "WAITING_FOR_FLIP"
    trade          = None
    flip_close     = None
    flip_dir       = None
    drop_ref       = None
    lowest_since   = None
    highest_since  = None
    open_ref_close = None
    open_dir       = None
    open_pending   = False
    current_date   = None
    trades_list    = []
    equity         = 0.0
    equity_curve   = []

    # ── Cooldown & gap tracking ───────────────
    consec_sl_count  = 0      # consecutive SL hits this day
    day_sl_paused    = False  # True = no more entries today (too many consec SLs)
    last_exit_idx    = -999   # candle index of last exit (for gap filter)

    n = len(df)

    for idx in range(n):
        row     = df.iloc[idx]
        ts      = row['timestamp']
        close   = float(row['close'])
        st_dir  = int(row['st_direction'])
        st_line = float(row['st_line'])
        c_time  = ts.time()
        c_date  = ts.date()

        # ── HTF direction at this candle ──────
        htf_dir = get_htf_direction_at(htf_series, ts) if use_htf else 0

        # ── Day boundary reset ────────────────
        if c_date != current_date:
            current_date   = c_date
            open_ref_close = None
            open_dir       = None
            open_pending   = False
            consec_sl_count = 0       # reset consecutive SL counter each new day
            day_sl_paused   = False   # reset pause each new day
            if state != "IN_TRADE":
                drop_ref      = None
                lowest_since  = None
                highest_since = None

        # ── Update extremes (not in trade) ────
        if state != "IN_TRADE" and drop_ref is not None:
            lowest_since  = min(lowest_since,  close) if lowest_since  is not None else close
            highest_since = max(highest_since, close) if highest_since is not None else close

        # ── EOD / time blocks ─────────────────
        is_eod_candle = bool(is_eod[idx])
        eod_block     = is_eod_candle or (c_time >= EOD_EXIT_TIME)
        candle_gap_ok  = (idx - last_exit_idx) >= MIN_CANDLES_BETWEEN_TRADES
        entry_blocked  = (eod_block
                          or (c_time > NO_ENTRY_AFTER)
                          or day_sl_paused
                          or not candle_gap_ok)

        # ── Market open candle: record ref ────
        if c_time == session_open_time:
            open_ref_close = close
            open_dir       = 'long' if st_dir == 1 else 'short'
            open_pending   = (state == "WAITING_FOR_FLIP")
            drop_ref       = close
            lowest_since   = close
            highest_since  = close
            equity_curve.append({'timestamp': ts, 'equity': equity})
            continue

        # ════════════════════════════════════════
        # IN_TRADE — manage open position
        # ════════════════════════════════════════
        if state == "IN_TRADE":
            d           = trade['position']
            entry_price = trade['entry_price']
            profit_pts  = (close - entry_price) if d == 'long' else (entry_price - close)

            # Update peak profit
            if profit_pts > trade['peak_profit']:
                trade['peak_profit'] = profit_pts

            # Update dynamic SL (ratchets with ST line)
            update_dynamic_sl(trade, st_line)

            exit_triggered = False
            exit_reason    = ""

            # 0. EOD Exit (highest priority)
            if is_eod_candle:
                exit_triggered, exit_reason = True, "EOD_EXIT"

            # 1. Stop Loss
            if not exit_triggered:
                if d == 'long'  and close <= trade['stop_loss']:
                    exit_triggered, exit_reason = True, "STOP_LOSS"
                elif d == 'short' and close >= trade['stop_loss']:
                    exit_triggered, exit_reason = True, "STOP_LOSS"

            # 2. ST Touch Exit (optional, with minimum profit guard)
            if not exit_triggered and ST_TOUCH_EXIT:
                if d == 'long'  and close <= st_line and profit_pts >= ST_TOUCH_MIN_PROFIT:
                    exit_triggered, exit_reason = True, "ST_TOUCH"
                elif d == 'short' and close >= st_line and profit_pts >= ST_TOUCH_MIN_PROFIT:
                    exit_triggered, exit_reason = True, "ST_TOUCH"

            # 3. Trailing Profit Exit
            if not exit_triggered and trade['peak_profit'] >= TRAIL_ACTIVATE_PTS:
                protection = trade['peak_profit'] * (1.0 - TRAIL_DRAWDOWN_PCT)
                if profit_pts < protection:
                    exit_triggered, exit_reason = True, "TRAILING_PROFIT"

            if exit_triggered:
                pnl    = record_exit(trade, close, ts, exit_reason, trades_list)
                equity += pnl
                trade  = None
                state  = "EXITED_WAITING_FOR_NEXT_FLIP"
                flip_close = flip_dir = drop_ref = lowest_since = highest_since = None
                open_pending  = False
                last_exit_idx = idx   # track candle index of exit for gap filter

                # Update consecutive SL counter
                if exit_reason == "STOP_LOSS":
                    consec_sl_count += 1
                    if consec_sl_count >= MAX_CONSEC_SL:
                        day_sl_paused = True   # no more entries today
                else:
                    consec_sl_count = 0        # reset on any non-SL exit

            equity_curve.append({'timestamp': ts, 'equity': equity})
            continue

        # ── HTF agreement check ───────────────
        def htf_agrees(direction):
            """Return True if HTF trend agrees with intended trade direction."""
            if not use_htf:
                return True
            if htf_dir == 0:
                return False   # No HTF data yet — skip
            return (direction == 'long' and htf_dir == 1) or \
                   (direction == 'short' and htf_dir == -1)

        # ── Entry helper ──────────────────────
        def try_enter(direction):
            """Attempt entry if all conditions met. Returns True if entered."""
            nonlocal state, trade, drop_ref, lowest_since, highest_since
            if entry_blocked:
                return False
            if not htf_agrees(direction):
                return False

            # ── Candle body strength filter ──
            if CANDLE_BODY_FILTER:
                candle_range = float(row['high']) - float(row['low'])
                candle_body  = abs(close - float(row['open']))
                if candle_range > 0:
                    body_pct = candle_body / candle_range
                    if body_pct < MIN_CANDLE_BODY_PCT:
                        return False   # Weak candle (doji/spinning top) — skip
                # Also check direction alignment: for long, close must be > open; short: close < open
                if direction == 'long'  and close <= float(row['open']):
                    return False   # Bearish candle — skip long entry
                if direction == 'short' and close >= float(row['open']):
                    return False   # Bullish candle — skip short entry

            if direction == 'long':
                drop_pts = (drop_ref - lowest_since) if (drop_ref and lowest_since is not None) else 0
                if drop_pts >= DROP_RULE_PTS:
                    if (close - lowest_since) >= drop_pts * DROP_RECOVERY:
                        trade = make_entry(close, 'long', ts, idx, st_line)
                        state = "IN_TRADE"
                        drop_ref = lowest_since = highest_since = None
                        return True
                    return False   # Waiting for recovery
                if close > flip_close:
                    trade = make_entry(close, 'long', ts, idx, st_line)
                    state = "IN_TRADE"
                    drop_ref = lowest_since = highest_since = None
                    return True

            elif direction == 'short':
                rise_pts = (highest_since - drop_ref) if (drop_ref and highest_since is not None) else 0
                if rise_pts >= DROP_RULE_PTS:
                    if (highest_since - close) >= rise_pts * DROP_RECOVERY:
                        trade = make_entry(close, 'short', ts, idx, st_line)
                        state = "IN_TRADE"
                        drop_ref = lowest_since = highest_since = None
                        return True
                    return False   # Waiting for pullback
                if close < flip_close:
                    trade = make_entry(close, 'short', ts, idx, st_line)
                    state = "IN_TRADE"
                    drop_ref = lowest_since = highest_since = None
                    return True

            return False

        # ════════════════════════════════════════
        # EXITED_WAITING_FOR_NEXT_FLIP
        # ════════════════════════════════════════
        if state == "EXITED_WAITING_FOR_NEXT_FLIP":
            if idx > 0 and int(df.iloc[idx-1]['st_direction']) != st_dir:
                flip_close    = close
                flip_dir      = 'long' if st_dir == 1 else 'short'
                drop_ref      = close
                lowest_since  = close
                highest_since = close
                state         = "WAITING_FOR_CONFIRMATION"
            equity_curve.append({'timestamp': ts, 'equity': equity})
            continue

        # ════════════════════════════════════════
        # WAITING_FOR_FLIP
        # ════════════════════════════════════════
        if state == "WAITING_FOR_FLIP":

            # Market open sequential entry
            if open_pending and open_ref_close is not None and c_time > session_open_time:
                entered = False
                if open_dir == 'long' and not entry_blocked:
                    drop_pts = (drop_ref - lowest_since) if (drop_ref and lowest_since is not None) else 0
                    if drop_pts >= DROP_RULE_PTS:
                        if (close - lowest_since) >= drop_pts * DROP_RECOVERY and htf_agrees('long'):
                            trade = make_entry(close, 'long', ts, idx, st_line)
                            state, open_pending = "IN_TRADE", False
                            drop_ref = lowest_since = highest_since = None
                            entered = True
                    elif st_dir == 1 and close > open_ref_close and htf_agrees('long'):
                        trade = make_entry(close, 'long', ts, idx, st_line)
                        state, open_pending = "IN_TRADE", False
                        drop_ref = lowest_since = highest_since = None
                        entered = True

                elif open_dir == 'short' and not entry_blocked:
                    rise_pts = (highest_since - drop_ref) if (drop_ref and highest_since is not None) else 0
                    if rise_pts >= DROP_RULE_PTS:
                        if (highest_since - close) >= rise_pts * DROP_RECOVERY and htf_agrees('short'):
                            trade = make_entry(close, 'short', ts, idx, st_line)
                            state, open_pending = "IN_TRADE", False
                            drop_ref = lowest_since = highest_since = None
                            entered = True
                    elif st_dir == -1 and close < open_ref_close and htf_agrees('short'):
                        trade = make_entry(close, 'short', ts, idx, st_line)
                        state, open_pending = "IN_TRADE", False
                        drop_ref = lowest_since = highest_since = None
                        entered = True

                if state == "IN_TRADE":
                    equity_curve.append({'timestamp': ts, 'equity': equity})
                    continue

            # ST Flip detection
            if not entry_blocked and idx > 0 and int(df.iloc[idx-1]['st_direction']) != st_dir:
                flip_close    = close
                flip_dir      = 'long' if st_dir == 1 else 'short'
                drop_ref      = close
                lowest_since  = close
                highest_since = close
                open_pending  = False
                state         = "WAITING_FOR_CONFIRMATION"

            equity_curve.append({'timestamp': ts, 'equity': equity})
            continue

        # ════════════════════════════════════════
        # WAITING_FOR_CONFIRMATION
        # ════════════════════════════════════════
        if state == "WAITING_FOR_CONFIRMATION":

            # Another ST flip while waiting? Reset to new flip
            if idx > 0 and int(df.iloc[idx-1]['st_direction']) != st_dir:
                flip_close    = close
                flip_dir      = 'long' if st_dir == 1 else 'short'
                drop_ref      = close
                lowest_since  = close
                highest_since = close
                equity_curve.append({'timestamp': ts, 'equity': equity})
                continue

            if flip_dir:
                try_enter(flip_dir)

            equity_curve.append({'timestamp': ts, 'equity': equity})
            continue

        equity_curve.append({'timestamp': ts, 'equity': equity})

    # ── Force-close open trade at end of data ──
    if state == "IN_TRADE" and trade is not None:
        last = df.iloc[-1]
        pnl  = record_exit(trade, float(last['close']), last['timestamp'],
                           'END_OF_DATA', trades_list)
        equity += pnl

    return pd.DataFrame(trades_list), pd.DataFrame(equity_curve)


# ─────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────

def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"message": "No trades found."}
    pnl    = trades_df['pnl']
    wins   = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    total  = len(trades_df)
    gp     = wins.sum()
    gl     = abs(losses.sum())
    cum    = pnl.cumsum()
    max_dd = (cum.cummax() - cum).max()
    return {
        'Total Trades':              total,
        'Winning Trades':            len(wins),
        'Losing Trades':             len(losses),
        'Win Rate (%)':              round(len(wins) / total * 100, 2),
        'Average Profit (pts)':      round(wins.mean(),   2) if len(wins)   else 0,
        'Average Loss (pts)':        round(losses.mean(), 2) if len(losses) else 0,
        'Largest Win (pts)':         round(pnl.max(), 2),
        'Largest Loss (pts)':        round(pnl.min(), 2),
        'Profit Factor':             round(gp / gl, 2) if gl > 0 else float('inf'),
        'Total P&L (pts)':           round(pnl.sum(), 2),
        'Max Drawdown (pts)':        round(max_dd, 2),
        'Trades with Profit > 100':  int((pnl > 100).sum()),
        'Trades with Profit > 200':  int((pnl > 200).sum()),
        'Trades with Loss > 100':    int((pnl < -100).sum()),
    }


# ─────────────────────────────────────────────
# 6. CSV LOADER
# ─────────────────────────────────────────────

def load_csv(filepath: str, label: str = ''):
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Auto-detect timestamp column
    ts_col = None
    for candidate in ['timestamp', 'datetime', 'date_time', 'time', 'date']:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError(f"[{label}] No timestamp column found. Got: {list(df.columns)}")

    df['timestamp'] = pd.to_datetime(df[ts_col], dayfirst=True)
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Kolkata')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')
    if ts_col != 'timestamp':
        df = df.drop(columns=[ts_col])

    # Rename OHLC columns
    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ['open', 'high', 'low', 'close', 'volume', 'timestamp']:
            continue
        if   'open'  in cl: rename_map[col] = 'open'
        elif 'high'  in cl: rename_map[col] = 'high'
        elif 'low'   in cl: rename_map[col] = 'low'
        elif 'close' in cl: rename_map[col] = 'close'
        elif 'vol'   in cl: rename_map[col] = 'volume'
    df = df.rename(columns=rename_map)

    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"[{label}] Missing column '{col}'. Got: {list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Detect candle interval
    delta_mins = int((df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds() // 60) \
                 if len(df) >= 2 else 5
    session_open = time(9, 15) if delta_mins >= 15 else time(9, 30)

    print(f"  [{label}] Interval: {delta_mins} min | Session open: {session_open} | Rows: {len(df)}")

    # Filter to trading hours
    df = df[(df['timestamp'].dt.time >= session_open) &
            (df['timestamp'].dt.time <= time(15, 30))].reset_index(drop=True)

    return df, session_open, delta_mins


# ─────────────────────────────────────────────
# 7. PRINT CONFIG SUMMARY
# ─────────────────────────────────────────────

def print_config(htf_available: bool):
    print(f"\n  {'─'*50}")
    print(f"  ACTIVE CONFIGURATION")
    print(f"  {'─'*50}")
    print(f"  ST ATR Period          : {ST_ATR_PERIOD}")
    print(f"  ST Multiplier          : {ST_MULTIPLIER}")
    print(f"  HTF Filter (15-min)    : {'ON' if HTF_FILTER_ON and htf_available else 'OFF'}")
    sl_desc = 'Fixed {:.2%} only'.format(SL_PCT) if USE_FIXED_SL else 'Hybrid: {:.2%} initial → ST line trail'.format(SL_PCT)
    print(f"  Stop Loss Type         : {sl_desc}")
    print(f"  ST Touch Exit          : {'ON (min profit: ' + str(ST_TOUCH_MIN_PROFIT) + ' pts)' if ST_TOUCH_EXIT else 'OFF'}")
    print(f"  Trailing Activate      : {TRAIL_ACTIVATE_PTS} pts peak")
    print(f"  Trailing Drawdown      : {TRAIL_DRAWDOWN_PCT:.0%} from peak")
    print(f"  No Entry After         : {NO_ENTRY_AFTER} IST")
    print(f"  EOD Exit Time          : {EOD_EXIT_TIME} IST (first candle at/after)")
    print(f"  70-pt Drop Rule        : {DROP_RULE_PTS} pts | Recovery: {DROP_RECOVERY:.0%}")
    print(f"  Candle Body Filter     : {'ON — min body ' + str(int(MIN_CANDLE_BODY_PCT*100)) + '% of range + direction aligned' if CANDLE_BODY_FILTER else 'OFF'}")
    print(f"  Consec SL Cooldown     : pause after {MAX_CONSEC_SL} consecutive SL hits/day")
    print(f"  Min Candles Between    : {MIN_CANDLES_BETWEEN_TRADES} candles gap after any exit")
    print(f"  {'─'*50}")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def fmt_ts(ts):
    return ts.strftime('%d-%m-%Y %H:%M IST') if hasattr(ts, 'strftime') else str(ts)


def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python supertrend_backtest.py <5min_csv>")
        print("  python supertrend_backtest.py <5min_csv> <15min_csv>")
        print("\nCSV columns : date/datetime | open | high | low | close | volume")
        print("Date format : DD-MM-YYYY HH:MM  or  YYYY-MM-DD HH:MM")
        sys.exit(1)

    primary_csv = sys.argv[1]
    htf_csv     = sys.argv[2] if len(sys.argv) >= 3 else None

    if not os.path.exists(primary_csv):
        print(f"Error: File not found: {primary_csv}")
        sys.exit(1)
    if htf_csv and not os.path.exists(htf_csv):
        print(f"Error: HTF file not found: {htf_csv}")
        sys.exit(1)

    SEP60 = '═' * 60

    print(f"\n{SEP60}")
    print("  SUPERTREND SEQUENTIAL CONFIRMATION BACKTESTER  v3.0")
    print(f"{SEP60}")

    # ── Load primary (5-min) data ──
    print(f"\nLoading PRIMARY data: {primary_csv}")
    df, session_open, primary_interval = load_csv(primary_csv, label='5-min')
    print(f"  Date range : {fmt_ts(df['timestamp'].min())}  →  {fmt_ts(df['timestamp'].max())}")

    # ── Load HTF (15-min) data ──
    htf_series = None
    htf_available = False
    if htf_csv:
        print(f"\nLoading HTF data     : {htf_csv}")
        htf_df, _, htf_interval = load_csv(htf_csv, label='15-min')
        print(f"  Date range : {fmt_ts(htf_df['timestamp'].min())}  →  {fmt_ts(htf_df['timestamp'].max())}")
        htf_series    = build_htf_lookup(htf_df)
        htf_available = True
    else:
        print("\n  No HTF CSV provided — HTF filter disabled.")

    print_config(htf_available)

    # ── Compute 5-min Supertrend ──
    print(f"\nComputing 5-min Supertrend (ATR={ST_ATR_PERIOD}, Mult={ST_MULTIPLIER})...")
    df = compute_supertrend(df, atr_period=ST_ATR_PERIOD, multiplier=ST_MULTIPLIER)

    # ── Run backtest ──
    print("Running backtest...\n")
    trades_df, equity_df = run_backtest(df, session_open, htf_series)

    print(f"\n{SEP60}")
    print("  BACKTEST RESULTS")
    print(f"{SEP60}")

    if trades_df.empty:
        print("No trades were generated.")
        return

    # ── Trade log ──
    SEP = '─' * 145
    print(f"\n{SEP}")
    print(f"  TRADE LOG  ({len(trades_df)} trades)")
    print(SEP)
    htf_col = '  HTF' if htf_available and HTF_FILTER_ON else ''
    print(f"{'#':<5} {'Entry Time':<24} {'Exit Time':<24} {'Dir':<7}"
          f" {'Entry':>9} {'Exit':>9} {'SL':>9} {'PeakPft':>9} {'P&L':>9}  {'Reason'}")
    print(SEP)
    for i, r in trades_df.iterrows():
        print(f"{i+1:<5} {fmt_ts(r['entry_time']):<24} {fmt_ts(r['exit_time']):<24}"
              f" {r['direction'].upper():<7}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f}"
              f" {r['stop_loss']:>9.2f} {r['peak_profit']:>9.2f}"
              f" {r['pnl']:>+9.2f}  {r['exit_reason']}")

    # ── Metrics ──
    metrics = compute_metrics(trades_df)
    print(f"\n{'─'*60}")
    print("  PERFORMANCE METRICS")
    print(f"{'─'*60}")
    for k, v in metrics.items():
        print(f"  {k:<35}: {v}")

    # ── Exit reason breakdown ──
    print(f"\n{'─'*60}")
    print("  EXIT REASON BREAKDOWN")
    print(f"{'─'*60}")
    bd = (trades_df.groupby('exit_reason')['pnl']
          .agg(count='count', total_pnl='sum', avg_pnl='mean')
          .round(2))
    print(bd.to_string())

    # ── Monthly breakdown ──
    print(f"\n{'─'*60}")
    print("  MONTHLY P&L BREAKDOWN")
    print(f"{'─'*60}")
    trades_df['month'] = trades_df['entry_time'].dt.tz_localize(None).dt.to_period('M').astype(str)
    monthly = (trades_df.groupby('month')['pnl']
               .agg(trades='count', total_pnl='sum', win_trades=lambda x: (x > 0).sum())
               .assign(win_rate=lambda x: (x['win_trades'] / x['trades'] * 100).round(1))
               .round(2))
    print(monthly.to_string())

    # ── Save outputs ──
    base = os.path.splitext(primary_csv)[0]
    try:
        open(base + '_trades.csv', 'a').close()
        trades_out = base + '_trades.csv'
        equity_out = base + '_equity.csv'
    except OSError:
        base       = os.path.join(os.getcwd(), os.path.splitext(os.path.basename(primary_csv))[0])
        trades_out = base + '_trades.csv'
        equity_out = base + '_equity.csv'

    trades_df.to_csv(trades_out, index=False)
    equity_df.to_csv(equity_out,  index=False)

    print(f"\n{'─'*60}")
    print(f"  Trades saved  : {trades_out}")
    print(f"  Equity curve  : {equity_out}")
    print(f"{SEP60}\n")


if __name__ == '__main__':
    main()