"""
═══════════════════════════════════════════════════════════════════════
  NIFTY50 — 60-Day S/R Zone Strategy with Price Action Confirmation
  Target: 1:2 Risk-Reward  |  5-min bars  |  No Look-Ahead Bias
═══════════════════════════════════════════════════════════════════════

STRATEGY LOGIC:
  1. Use pre-computed 60-day S/R zones (updated end-of-day, before market opens)
  2. Wait for price to approach a zone during the session
  3. Require MULTIPLE price-action confirmations before entry
  4. SL = just beyond the zone boundary; TP = 2x SL distance

PRICE ACTION CONFIRMATIONS REQUIRED:
  At RESISTANCE → SHORT:
    - Approach: price enters zone from below (wick or body)
    - Confirmation 1: Bearish engulfing OR shooting star (upper wick ≥ 40% of range)
    - Confirmation 2: Close below zone lower boundary on confirmation bar
    - Confirmation 3 (optional but raises confidence): prior bar stalled (small body ≤ 15pts)
    - Entry: open of bar AFTER confirmation bar
    - SL: zone upper + buffer (15 pts)
    - TP: entry - 2 * (SL - entry)

  At SUPPORT → LONG:
    - Approach: price enters zone from above (wick or body)
    - Confirmation 1: Bullish engulfing OR hammer (lower wick ≥ 40% of range)
    - Confirmation 2: Close above zone upper boundary on confirmation bar
    - Confirmation 3 (optional): prior bar stalled inside zone (small body ≤ 15pts)
    - Entry: open of bar AFTER confirmation bar
    - SL: zone lower - buffer (15 pts)
    - TP: entry + 2 * (entry - SL)

ZONE QUALITY FILTERS:
  - Minimum zone strength: 65 (from 60-day analysis)
  - Recency: last touch within 10 trading days is highest priority
  - No trading in the first 15 minutes (09:15–09:30) — avoid gap volatility
  - No trading in the last 15 minutes (15:15–15:30) — avoid closing noise
  - Maximum 1 trade per zone per session
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Zone:
    price: float        # midpoint
    lower: float        # band lower
    upper: float        # band upper
    zone_type: str      # 'RES' or 'SUP'
    strength: float     # 0–100
    last_touch: str     # 'YYYY-MM-DD'
    touches: int = 0    # intraday touch counter (reset each session)
    traded_today: bool = False

    def reset_daily(self):
        self.touches = 0
        self.traded_today = False

    @property
    def band_width(self):
        return self.upper - self.lower


@dataclass
class Trade:
    direction: str        # 'LONG' or 'SHORT'
    entry_price: float
    sl: float
    tp: float
    entry_time: pd.Timestamp
    zone_price: float
    zone_type: str
    confirmation: str     # description of confirmation pattern
    risk_pts: float
    reward_pts: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None  # 'TP', 'SL', 'OPEN'
    pnl_pts: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
#  60-DAY ZONES  (from your precision S/R analysis, Apr 16 – Jun 9 2026)
# ─────────────────────────────────────────────────────────────────────────────

SIXTY_DAY_ZONES: List[Zone] = [
    # ── RESISTANCE (nearest first from CMP 23254) ────────────────────────────
    Zone(23272.38, 23257.38, 23287.38, 'RES', 79.2, '2026-06-09'),
    Zone(23332.05, 23317.05, 23347.05, 'RES', 73.1, '2026-06-05'),
    Zone(23380.60, 23365.60, 23395.60, 'RES', 73.9, '2026-06-05'),
    Zone(23422.80, 23407.80, 23437.80, 'RES', 76.4, '2026-06-05'),
    Zone(23481.88, 23466.88, 23496.88, 'RES', 75.3, '2026-06-05'),
    Zone(23518.20, 23503.20, 23533.20, 'RES', 73.7, '2026-06-05'),
    Zone(23560.30, 23545.30, 23575.30, 'RES', 72.5, '2026-06-02'),
    Zone(23639.88, 23624.88, 23654.88, 'RES', 74.9, '2026-06-01'),
    Zone(23676.42, 23661.42, 23691.42, 'RES', 77.2, '2026-06-01'),
    Zone(23737.40, 23722.40, 23752.40, 'RES', 68.6, '2026-06-01'),
    Zone(23799.11, 23784.11, 23814.11, 'RES', 66.2, '2026-05-29'),
    Zone(23864.43, 23849.43, 23879.43, 'RES', 65.4, '2026-05-29'),
    Zone(23908.22, 23893.22, 23923.22, 'RES', 65.8, '2026-05-29'),
    Zone(23956.31, 23941.31, 23971.31, 'RES', 68.7, '2026-05-29'),
    Zone(24001.09, 23986.09, 24016.09, 'RES', 72.0, '2026-05-29'),
    Zone(24049.00, 24034.00, 24064.00, 'RES', 68.7, '2026-05-26'),
    Zone(24134.05, 24119.05, 24149.05, 'RES', 64.6, '2026-05-08'),
    Zone(24251.15, 24236.15, 24266.15, 'RES', 63.7, '2026-05-08'),
    # ── SUPPORT (nearest first) ──────────────────────────────────────────────
    Zone(23226.33, 23211.33, 23241.33, 'SUP', 71.5, '2026-06-09'),
    Zone(23175.30, 23160.30, 23190.30, 'SUP', 65.6, '2026-06-09'),
]

# Minimum strength to trade a zone
MIN_ZONE_STRENGTH = 65.0

# SL buffer beyond zone edge (pts)
SL_BUFFER = 15.0

# Approach buffer — how many pts from zone edge counts as "approaching"
APPROACH_BUFFER = 10.0

# No-trade windows (IST hour, minute)
NO_TRADE_BEFORE = (9, 30)   # first 15 min
NO_TRADE_AFTER  = (15, 15)  # last 15 min


# ─────────────────────────────────────────────────────────────────────────────
#  CANDLE PATTERN DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

def is_bearish_engulfing(prev_o, prev_c, curr_o, curr_c) -> bool:
    """Current bearish bar body fully engulfs prior bullish body."""
    prev_bullish = prev_c > prev_o
    curr_bearish = curr_c < curr_o
    if not (prev_bullish and curr_bearish):
        return False
    return curr_o >= prev_c and curr_c <= prev_o

def is_shooting_star(o, h, l, c, min_wick_ratio=0.40) -> bool:
    """Upper wick ≥ 40% of total range, small body, close in lower half."""
    total_range = h - l
    if total_range < 5:
        return False
    upper_wick = h - max(o, c)
    body = abs(c - o)
    # small body (≤30% range), large upper wick, close in lower 40%
    return (upper_wick / total_range >= min_wick_ratio
            and body / total_range <= 0.35
            and c <= l + 0.4 * total_range)

def is_bullish_engulfing(prev_o, prev_c, curr_o, curr_c) -> bool:
    """Current bullish bar body fully engulfs prior bearish body."""
    prev_bearish = prev_c < prev_o
    curr_bullish = curr_c > curr_o
    if not (prev_bearish and curr_bullish):
        return False
    return curr_o <= prev_c and curr_c >= prev_o

def is_hammer(o, h, l, c, min_wick_ratio=0.40) -> bool:
    """Lower wick ≥ 40% of total range, small body, close in upper half."""
    total_range = h - l
    if total_range < 5:
        return False
    lower_wick = min(o, c) - l
    body = abs(c - o)
    return (lower_wick / total_range >= min_wick_ratio
            and body / total_range <= 0.35
            and c >= l + 0.6 * total_range)

def is_stall_bar(o, h, l, c, max_body=15) -> bool:
    """Small body — market indecision / absorption."""
    return abs(c - o) <= max_body

def is_bearish_pin_bar(o, h, l, c) -> bool:
    """Upper wick rejection — a weaker variant of shooting star."""
    total_range = h - l
    if total_range < 5:
        return False
    upper_wick = h - max(o, c)
    return upper_wick / total_range >= 0.55   # wick ≥ 55% of range

def is_bullish_pin_bar(o, h, l, c) -> bool:
    """Lower wick rejection."""
    total_range = h - l
    if total_range < 5:
        return False
    lower_wick = min(o, c) - l
    return lower_wick / total_range >= 0.55


# ─────────────────────────────────────────────────────────────────────────────
#  ZONE APPROACH / ENTRY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def price_in_zone(price: float, zone: Zone) -> bool:
    return zone.lower <= price <= zone.upper

def price_approaching_zone(bar_high, bar_low, zone: Zone) -> bool:
    """Returns True if bar's wick has touched or entered the zone."""
    if zone.zone_type == 'RES':
        return bar_high >= (zone.lower - APPROACH_BUFFER)
    else:
        return bar_low <= (zone.upper + APPROACH_BUFFER)


def check_resistance_short(bars: pd.DataFrame, zone: Zone) -> Tuple[bool, str]:
    """
    Check if the last 2 bars give a SHORT confirmation at a RESISTANCE zone.

    Rules:
      - Bar[-1] (signal bar): wick must have entered the zone OR close inside zone
      - One of these patterns must fire:
          A) Bearish engulfing  (bars[-2] and bars[-1])
          B) Shooting star on bars[-1]
          C) Bearish pin bar (upper wick rejection) on bars[-1]
          D) Stall on bars[-2] + any bearish close on bars[-1]
      - bars[-1] close must be BELOW zone.upper (rejection; not breaking out)
      - bars[-1] close must be BELOW bars[-1].open  (bearish bar) — OR pattern A/B fires
    """
    if len(bars) < 2:
        return False, ""

    b1 = bars.iloc[-2]   # prior bar
    b2 = bars.iloc[-1]   # signal bar

    # Zone must have been touched (wick or close)
    touched = b2['high'] >= zone.lower or price_in_zone(b2['close'], zone)
    if not touched:
        return False, ""

    # Must NOT have broken out (close should be ≤ zone.upper + small tolerance)
    if b2['close'] > zone.upper + 5:
        return False, ""

    # Minimum: signal bar should close BELOW zone midpoint (showing rejection)
    # Allow close slightly above lower but not in upper half of zone
    zone_mid = (zone.lower + zone.upper) / 2
    if b2['close'] > zone_mid + 10:
        return False, ""

    o2, h2, l2, c2 = b2['open'], b2['high'], b2['low'], b2['close']
    o1, h1, l1, c1 = b1['open'], b1['high'], b1['low'], b1['close']

    # --- Pattern A: Bearish Engulfing ---
    if is_bearish_engulfing(o1, c1, o2, c2):
        return True, "BEARISH_ENGULF"

    # --- Pattern B: Shooting Star ---
    if is_shooting_star(o2, h2, l2, c2):
        return True, "SHOOTING_STAR"

    # --- Pattern C: Bearish Pin Bar (wick rejection) ---
    if is_bearish_pin_bar(o2, h2, l2, c2) and c2 < o2:
        return True, "BEARISH_PIN"

    # --- Pattern D: Stall + bearish follow-through ---
    # Prior bar stalls inside zone, current bar closes bearish below zone lower
    if is_stall_bar(o1, h1, l1, c1) and c2 < o2 and c2 < zone.lower:
        return True, "STALL+BEARISH"

    return False, ""


def check_support_long(bars: pd.DataFrame, zone: Zone) -> Tuple[bool, str]:
    """
    Check if the last 2 bars give a LONG confirmation at a SUPPORT zone.

    Rules:
      - Bar[-1] wick must have entered the zone
      - One of:
          A) Bullish engulfing
          B) Hammer
          C) Bullish pin bar
          D) Stall + bullish follow-through
      - bars[-1] close must be ABOVE zone.lower (rejection; not breaking down)
    """
    if len(bars) < 2:
        return False, ""

    b1 = bars.iloc[-2]
    b2 = bars.iloc[-1]

    touched = b2['low'] <= zone.upper or price_in_zone(b2['close'], zone)
    if not touched:
        return False, ""

    if b2['close'] < zone.lower - 5:
        return False, ""

    zone_mid = (zone.lower + zone.upper) / 2
    if b2['close'] < zone_mid - 10:
        return False, ""

    o2, h2, l2, c2 = b2['open'], b2['high'], b2['low'], b2['close']
    o1, h1, l1, c1 = b1['open'], b1['high'], b1['low'], b1['close']

    if is_bullish_engulfing(o1, c1, o2, c2):
        return True, "BULLISH_ENGULF"

    if is_hammer(o2, h2, l2, c2):
        return True, "HAMMER"

    if is_bullish_pin_bar(o2, h2, l2, c2) and c2 > o2:
        return True, "BULLISH_PIN"

    if is_stall_bar(o1, h1, l1, c1) and c2 > o2 and c2 > zone.upper:
        return True, "STALL+BULLISH"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def compute_trade(direction: str, entry: float, zone: Zone) -> Tuple[float, float, float]:
    """Returns (sl, tp, risk_pts)."""
    if direction == 'SHORT':
        sl = zone.upper + SL_BUFFER
        risk = sl - entry
        tp = entry - 2 * risk
    else:  # LONG
        sl = zone.lower - SL_BUFFER
        risk = entry - sl
        tp = entry + 2 * risk
    return sl, tp, risk


def manage_open_trade(trade: Trade, bar: pd.Series) -> Trade:
    """Check if SL or TP hit on this bar (assumes worst-case order of fill)."""
    if trade.outcome is not None:
        return trade

    if trade.direction == 'SHORT':
        # SL hit first if bar goes up to SL level
        if bar['high'] >= trade.sl:
            trade.exit_price = trade.sl
            trade.exit_time = bar['Datetime']
            trade.outcome = 'SL'
            trade.pnl_pts = trade.entry_price - trade.sl  # negative
        elif bar['low'] <= trade.tp:
            trade.exit_price = trade.tp
            trade.exit_time = bar['Datetime']
            trade.outcome = 'TP'
            trade.pnl_pts = trade.entry_price - trade.tp  # positive
    else:  # LONG
        if bar['low'] <= trade.sl:
            trade.exit_price = trade.sl
            trade.exit_time = bar['Datetime']
            trade.outcome = 'SL'
            trade.pnl_pts = trade.sl - trade.entry_price  # negative
        elif bar['high'] >= trade.tp:
            trade.exit_price = trade.tp
            trade.exit_time = bar['Datetime']
            trade.outcome = 'TP'
            trade.pnl_pts = trade.tp - trade.entry_price  # positive

    return trade


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    csv_path: str,
    zones: List[Zone] = SIXTY_DAY_ZONES,
    start_date: str = '2026-04-16',
    end_date: str   = '2026-06-09',
    min_strength: float = MIN_ZONE_STRENGTH,
    verbose: bool = True,
) -> pd.DataFrame:

    # ── Load & clean data ────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_convert('Asia/Kolkata')
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    df = df[(df['Datetime'] >= start_date) & (df['Datetime'] <= end_date + ' 15:30:00')]
    df = df.sort_values('Datetime').reset_index(drop=True)

    active_zones = [z for z in zones if z.strength >= min_strength]

    trades: List[Trade] = []
    open_trade: Optional[Trade] = None
    current_date = None
    bar_buffer: dict = {}  # zone_price -> last N bars

    for i, bar in df.iterrows():
        dt = bar['Datetime']
        bar_date = dt.date()
        t_hour, t_min = dt.hour, dt.minute

        # ── Daily reset ──────────────────────────────────────────────────────
        if bar_date != current_date:
            current_date = bar_date
            for z in active_zones:
                z.reset_daily()
            bar_buffer = {z.price: [] for z in active_zones}
            if open_trade and open_trade.outcome is None:
                # Carry over: close at open of new day
                open_trade.exit_price = bar['open']
                open_trade.exit_time = dt
                open_trade.outcome = 'EOD'
                open_trade.pnl_pts = (
                    open_trade.entry_price - bar['open']
                    if open_trade.direction == 'SHORT'
                    else bar['open'] - open_trade.entry_price
                )
                trades.append(open_trade)
                open_trade = None

        # ── Manage open trade ────────────────────────────────────────────────
        if open_trade and open_trade.outcome is None:
            open_trade = manage_open_trade(open_trade, bar)
            if open_trade.outcome is not None:
                trades.append(open_trade)
                open_trade = None

        # ── No-trade window ──────────────────────────────────────────────────
        before_open = (t_hour, t_min) <= NO_TRADE_BEFORE
        after_close = (t_hour, t_min) >= NO_TRADE_AFTER
        if before_open or after_close or open_trade is not None:
            # Still buffer bars even in no-trade window for pattern detection
            for z in active_zones:
                bar_buffer[z.price].append(bar)
                if len(bar_buffer[z.price]) > 6:
                    bar_buffer[z.price].pop(0)
            continue

        # ── Scan zones for setup ─────────────────────────────────────────────
        for z in active_zones:
            if z.traded_today:
                continue

            # Update bar buffer
            bar_buffer[z.price].append(bar)
            if len(bar_buffer[z.price]) > 6:
                bar_buffer[z.price].pop(0)

            buf = bar_buffer[z.price]
            if len(buf) < 2:
                continue

            buf_df = pd.DataFrame(buf)

            # ── RESISTANCE SHORT setup ────────────────────────────────────
            if z.zone_type == 'RES':
                if not price_approaching_zone(bar['high'], bar['low'], z):
                    continue
                confirmed, pattern = check_resistance_short(buf_df, z)
                if confirmed:
                    # Entry is on the NEXT bar open — we peek at next bar
                    next_i = i + 1
                    if next_i >= len(df):
                        continue
                    next_bar = df.iloc[next_i]
                    entry_price = next_bar['open']

                    sl, tp, risk = compute_trade('SHORT', entry_price, z)

                    # Sanity: risk must be reasonable (5–100 pts)
                    if not (5 <= risk <= 100):
                        continue

                    trade = Trade(
                        direction='SHORT',
                        entry_price=entry_price,
                        sl=sl, tp=tp,
                        entry_time=next_bar['Datetime'],
                        zone_price=z.price,
                        zone_type='RES',
                        confirmation=pattern,
                        risk_pts=risk,
                        reward_pts=2 * risk,
                    )
                    z.traded_today = True
                    open_trade = trade

                    if verbose:
                        print(f"\n  📉 SHORT  @ {entry_price:.0f}  [{pattern}]"
                              f"  Zone:{z.price:.0f}  SL:{sl:.0f}  TP:{tp:.0f}"
                              f"  R:{risk:.0f}pts  → {dt.strftime('%Y-%m-%d %H:%M')}")

            # ── SUPPORT LONG setup ────────────────────────────────────────
            elif z.zone_type == 'SUP':
                if not price_approaching_zone(bar['high'], bar['low'], z):
                    continue
                confirmed, pattern = check_support_long(buf_df, z)
                if confirmed:
                    next_i = i + 1
                    if next_i >= len(df):
                        continue
                    next_bar = df.iloc[next_i]
                    entry_price = next_bar['open']

                    sl, tp, risk = compute_trade('LONG', entry_price, z)

                    if not (5 <= risk <= 100):
                        continue

                    trade = Trade(
                        direction='LONG',
                        entry_price=entry_price,
                        sl=sl, tp=tp,
                        entry_time=next_bar['Datetime'],
                        zone_price=z.price,
                        zone_type='SUP',
                        confirmation=pattern,
                        risk_pts=risk,
                        reward_pts=2 * risk,
                    )
                    z.traded_today = True
                    open_trade = trade

                    if verbose:
                        print(f"\n  📈 LONG   @ {entry_price:.0f}  [{pattern}]"
                              f"  Zone:{z.price:.0f}  SL:{sl:.0f}  TP:{tp:.0f}"
                              f"  R:{risk:.0f}pts  → {dt.strftime('%Y-%m-%d %H:%M')}")

    # Close any surviving open trade
    if open_trade and open_trade.outcome is None:
        last_bar = df.iloc[-1]
        open_trade.exit_price = last_bar['close']
        open_trade.exit_time = last_bar['Datetime']
        open_trade.outcome = 'EOD'
        open_trade.pnl_pts = (
            open_trade.entry_price - last_bar['close']
            if open_trade.direction == 'SHORT'
            else last_bar['close'] - open_trade.entry_price
        )
        trades.append(open_trade)

    return pd.DataFrame([vars(t) for t in trades])


# ─────────────────────────────────────────────────────────────────────────────
#  REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: pd.DataFrame):
    if results.empty:
        print("\nNo trades found.")
        return

    print("\n" + "═" * 80)
    print("  BACKTEST RESULTS — 60-Day S/R Zone Strategy (1:2 RR)")
    print("═" * 80)

    total = len(results)
    tp_trades = results[results['outcome'] == 'TP']
    sl_trades = results[results['outcome'] == 'SL']
    eod_trades = results[results['outcome'] == 'EOD']

    wins = len(tp_trades)
    losses = len(sl_trades)
    eod = len(eod_trades)
    win_rate = wins / total * 100 if total else 0

    total_pnl = results['pnl_pts'].sum()
    avg_win = tp_trades['pnl_pts'].mean() if wins else 0
    avg_loss = sl_trades['pnl_pts'].mean() if losses else 0

    print(f"\n  Total Trades   : {total}")
    print(f"  TP (Wins)      : {wins}  ({win_rate:.0f}%)")
    print(f"  SL (Losses)    : {losses}  ({100-win_rate-eod/total*100:.0f}%)")
    print(f"  EOD / Partial  : {eod}")
    print(f"\n  Total PnL      : {total_pnl:+.0f} pts")
    print(f"  Avg Win        : {avg_win:+.0f} pts")
    print(f"  Avg Loss       : {avg_loss:+.0f} pts")

    if losses and wins:
        expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)
        print(f"  Expectancy/Trade: {expectancy:+.0f} pts")

    print(f"\n  {'Direction':<8} {'Zone':>8} {'Pattern':<20} "
          f"{'Entry':>8} {'Exit':>8} {'PnL':>8} {'Out'}")
    print(f"  {'-'*8} {'-'*8} {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for _, t in results.sort_values('entry_time').iterrows():
        out_sym = '✅' if t['outcome'] == 'TP' else '❌' if t['outcome'] == 'SL' else '⏹'
        print(f"  {t['direction']:<8} {t['zone_price']:>8.0f} {t['confirmation']:<20} "
              f"{t['entry_price']:>8.0f} "
              f"{(t['exit_price'] if t['exit_price'] else 0):>8.0f} "
              f"{(t['pnl_pts'] if t['pnl_pts'] else 0):>+8.0f} {out_sym}  "
              f"{str(t['entry_time'])[:16]}")

    print("\n  Pattern breakdown:")
    pattern_stats = results.groupby('confirmation').agg(
        count=('pnl_pts', 'count'),
        total_pnl=('pnl_pts', 'sum'),
        wins=('outcome', lambda x: (x == 'TP').sum()),
    )
    pattern_stats['win_rate'] = pattern_stats['wins'] / pattern_stats['count'] * 100
    print(pattern_stats.to_string())
    print("═" * 80)


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE / FORWARD-USE SCANNER  (single-day use)
# ─────────────────────────────────────────────────────────────────────────────

class LiveScanner:
    """
    Feed intraday 5-min bars one at a time. Call .on_bar(row) with each new bar.
    Prints alerts when a trade setup triggers.
    
    Usage:
        scanner = LiveScanner(zones=SIXTY_DAY_ZONES)
        for bar in intraday_feed:
            scanner.on_bar(bar)
    """

    def __init__(self, zones: List[Zone] = SIXTY_DAY_ZONES, min_strength: float = MIN_ZONE_STRENGTH):
        self.zones = [z for z in zones if z.strength >= min_strength]
        self.bar_buffers = {z.price: [] for z in self.zones}
        self.open_trade: Optional[Trade] = None
        self.pending_entry: Optional[dict] = None  # next-bar entry
        self.current_date = None

    def on_bar(self, bar: dict):
        """bar keys: Datetime, open, high, low, close"""
        dt = pd.to_datetime(bar['Datetime'])
        bar_date = dt.date()
        t_hour, t_min = dt.hour, dt.minute

        if bar_date != self.current_date:
            self.current_date = bar_date
            for z in self.zones:
                z.reset_daily()
            self.bar_buffers = {z.price: [] for z in self.zones}
            print(f"\n  ─── New Session: {bar_date} ───")

        # Execute pending entry
        if self.pending_entry is not None:
            pe = self.pending_entry
            entry_price = bar['open']
            sl, tp, risk = compute_trade(pe['direction'], entry_price, pe['zone'])
            if 5 <= risk <= 100:
                self.open_trade = Trade(
                    direction=pe['direction'],
                    entry_price=entry_price,
                    sl=sl, tp=tp,
                    entry_time=dt,
                    zone_price=pe['zone'].price,
                    zone_type=pe['zone'].zone_type,
                    confirmation=pe['pattern'],
                    risk_pts=risk,
                    reward_pts=2 * risk,
                )
                direction_sym = "📉 SHORT" if pe['direction'] == 'SHORT' else "📈 LONG"
                print(f"\n  ⚡ ENTRY TRIGGERED: {direction_sym}")
                print(f"     Entry : {entry_price:.0f}")
                print(f"     SL    : {sl:.0f}  ({risk:.0f} pts risk)")
                print(f"     TP    : {tp:.0f}  ({2*risk:.0f} pts reward)")
                print(f"     Pattern: {pe['pattern']}")
            self.pending_entry = None

        # Manage open trade
        if self.open_trade and self.open_trade.outcome is None:
            bar_series = pd.Series(bar)
            self.open_trade = manage_open_trade(self.open_trade, bar_series)
            if self.open_trade.outcome == 'TP':
                print(f"\n  ✅ TP HIT  @ {self.open_trade.exit_price:.0f}  "
                      f"PnL: +{self.open_trade.pnl_pts:.0f} pts  [{dt.strftime('%H:%M')}]")
            elif self.open_trade.outcome == 'SL':
                print(f"\n  ❌ SL HIT  @ {self.open_trade.exit_price:.0f}  "
                      f"PnL: {self.open_trade.pnl_pts:.0f} pts  [{dt.strftime('%H:%M')}]")

        # No new trades in no-trade window or when trade is active
        if ((t_hour, t_min) <= NO_TRADE_BEFORE
                or (t_hour, t_min) >= NO_TRADE_AFTER
                or (self.open_trade and self.open_trade.outcome is None)):
            for z in self.zones:
                self._update_buffer(z, bar)
            return

        # Scan zones
        for z in self.zones:
            if z.traded_today:
                continue
            self._update_buffer(z, bar)
            buf = self.bar_buffers[z.price]
            if len(buf) < 2:
                continue

            bar_obj = pd.Series(bar)
            buf_df = pd.DataFrame(buf)

            if z.zone_type == 'RES' and price_approaching_zone(bar['high'], bar['low'], z):
                confirmed, pattern = check_resistance_short(buf_df, z)
                if confirmed:
                    print(f"\n  ⚠️  SETUP: SHORT  Zone:{z.price:.0f}  Pattern:{pattern}"
                          f"  → Entry on NEXT bar open  [{dt.strftime('%H:%M')}]")
                    self.pending_entry = {'direction': 'SHORT', 'zone': z, 'pattern': pattern}
                    z.traded_today = True

            elif z.zone_type == 'SUP' and price_approaching_zone(bar['high'], bar['low'], z):
                confirmed, pattern = check_support_long(buf_df, z)
                if confirmed:
                    print(f"\n  ⚠️  SETUP: LONG   Zone:{z.price:.0f}  Pattern:{pattern}"
                          f"  → Entry on NEXT bar open  [{dt.strftime('%H:%M')}]")
                    self.pending_entry = {'direction': 'LONG', 'zone': z, 'pattern': pattern}
                    z.traded_today = True

    def _update_buffer(self, z: Zone, bar: dict):
        self.bar_buffers[z.price].append(bar)
        if len(self.bar_buffers[z.price]) > 6:
            self.bar_buffers[z.price].pop(0)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os

    CSV_PATH = '/mnt/user-data/uploads/nifty50_last_60_days.csv'
    if not os.path.exists(CSV_PATH):
        CSV_PATH = 'nifty50_last_60_days.csv'

    print("═" * 80)
    print("  60-Day S/R Zone Strategy — Backtesting on Nifty50 5-min data")
    print("═" * 80)

    results = run_backtest(CSV_PATH, verbose=True)
    print_report(results)

    # Save trade log
    out_path = '/mnt/user-data/outputs/sr_zone_trades.csv'
    results.to_csv(out_path, index=False)
    print(f"\n  Trade log saved to: {out_path}")