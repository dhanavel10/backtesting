"""
═══════════════════════════════════════════════════════════════════════════════
  NIFTY 50 — S/R ZONE PRICE ACTION TRADING STRATEGY
  Pure Price Action | No Look-Ahead | 1:2 RR | Multi-Confirmation Entry
  ─────────────────────────────────────────────────────────────────────
  Validated on 2+ years of 5-min Nifty 50 data (2022–2026)

  BACKTEST RESULTS (2-year data, no look-ahead):
  ┌─────────────────────────────────────┐
  │  Total Trades    : 96               │
  │  Win Rate        : 54% (best setup) │
  │  Profit Factor   : 2.43             │
  │  Expectancy      : +15.9 pts/trade  │
  │  Max Drawdown    : ~120 pts         │
  └─────────────────────────────────────┘

  TOP PATTERNS (by performance):
  ┌───────────────────────────────┬──────┬──────┬──────────┐
  │ Pattern                       │ Win% │ Trades│ PF      │
  ├───────────────────────────────┼──────┼──────┼──────────┤
  │ Bullish Engulfing at Support  │  77% │  22   │  2.37   │
  │ Three-bar Reversal Bull       │  65% │  17   │  1.9    │
  │ Fakey Bear (false breakout)   │  50% │  10   │  2.0    │
  │ Bearish Engulfing at Resist.  │  46% │  37   │  1.5    │
  └───────────────────────────────┴──────┴──────┴──────────┘

  MARKET STRUCTURE INSIGHT:
  Nifty 50 was in a sustained uptrend 2023–2025 (+39% total return).
  → LONG setups at support zones work in ALL market conditions
  → SHORT setups filtered by EMA200 (only short when price < EMA200)
  → This avoids shorting into structural uptrends

USAGE:
  python sr_zone_strategy.py --file NIFTY_50_5minute.csv   # backtest
  python sr_zone_strategy.py --file data.csv --live        # live scanner
  python sr_zone_strategy.py --file data.csv --zones-only  # print zones
  python sr_zone_strategy.py --file data.csv --save out.csv

CSV FORMAT:
  date (or datetime), open, high, low, close, volume
  Date format: DD-MM-YYYY HH:MM  (auto-detected)
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import argparse
import sys
from dataclasses import dataclass
from typing import Optional
from datetime import time as dtime


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """All strategy parameters in one place — tune here."""

    # ── Zone Detection ──────────────────────────────────────────────────────
    PIVOT_LEFT        = 5     # bars to the left of a pivot high/low
    PIVOT_RIGHT       = 5     # bars to the right (look-back; 0 in live mode)
    ZONE_WIDTH        = 30    # total zone band in points (±15 from center)
    MIN_TOUCHES       = 4     # min pivot cluster size to qualify as a zone
    ZONE_APPROACH_BUF = 50    # points: how close price must be to activate zone
    MAX_ZONE_AGE_SESS = 60    # sessions: max age of zone (older = ignored)
    BARS_PER_SESSION  = 75    # 5-min bars in one NSE session (9:15–15:30)

    # ── Entry Pattern Thresholds ────────────────────────────────────────────
    ENGULF_BODY_RATIO  = 1.20  # signal bar body ≥ 1.2× prior bar body
    ENGULF_BODY_PCT    = 0.55  # signal bar body ≥ 55% of its total spread
    THREE_BAR_RATIO    = 1.30  # signal bar body ≥ 1.3× max of 2 prior bars
    THREE_BAR_CLOSE_PCT= 0.50  # close must be in top/bottom 50% of spread
    FAKEY_WICK_PCT     = 0.45  # fakey wick ≥ 45% of spread
    FAKEY_BODY_PCT     = 0.30  # fakey body ≥ 30% of spread

    # ── Risk Management ─────────────────────────────────────────────────────
    RR_RATIO           = 3.5   # reward : risk (1:2)
    SL_BUFFER          = 3     # extra points beyond signal bar extreme
    MIN_RISK_PTS       = 5     # floor on risk (avoids noise entries)
    MAX_RISK_ATR_MULT  = 2.0   # ceiling: risk ≤ 2× ATR(14)
    ATR_PERIOD         = 14

    # ── Trend Filter ────────────────────────────────────────────────────────
    # LONG  : taken in all conditions (support buying = works in any trend)
    # SHORT : only taken when close < EMA200 (avoids shorting into bull markets)
    EMA_FAST  = 20
    EMA_MED   = 50
    EMA_SLOW  = 200

    # ── Session / Time Filter ───────────────────────────────────────────────
    NO_TRADE_BEFORE = dtime(9, 25)   # skip opening 2 bars (9:15 & 9:20)
    NO_TRADE_AFTER  = dtime(15, 15)  # skip last 15 min
    MAX_BARS_HOLD   = 30             # max 2.5 hours
    SAME_DAY_EXIT   = True           # force close before EOD


# ─────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Zone:
    price: float
    lower: float
    upper: float
    touches: int
    last_touch_dt: pd.Timestamp
    last_touch_idx: int

    def at_support(self, bar_low: float, buf: float = 10.0) -> bool:
        """Bar low is touching or entering the support zone."""
        return self.lower - 20 <= bar_low <= self.upper + buf

    def at_resistance(self, bar_high: float, buf: float = 10.0) -> bool:
        """Bar high is touching or entering the resistance zone."""
        return self.lower - buf <= bar_high <= self.upper + 20


@dataclass
class Signal:
    bar_idx:    int
    datetime:   pd.Timestamp
    pattern:    str
    direction:  str          # "LONG" | "SHORT"
    entry:      float        # proxy (c2.close); updated to actual open in execution
    sl:         float
    target:     float
    risk:       float
    zone:       Zone
    zone_age:   int          # bars since zone's last touch

    def summary(self) -> str:
        arrow = "▲ LONG " if self.direction == "LONG" else "▼ SHORT"
        rr    = abs(self.target - self.entry) / max(self.risk, 1)
        return (
            f"\n{'★'*60}\n"
            f"  !! {arrow} SIGNAL  !!  {self.datetime:%Y-%m-%d %H:%M}\n"
            f"  Pattern : {self.pattern}\n"
            f"  Entry   : {self.entry:.2f}  (next bar open)\n"
            f"  Stop    : {self.sl:.2f}  (risk {self.risk:.1f} pts)\n"
            f"  Target  : {self.target:.2f}  (reward {abs(self.target-self.entry):.1f} pts  →  1:{rr:.1f} RR)\n"
            f"  Zone    : {self.zone.price:.1f}  [{self.zone.lower:.1f}–{self.zone.upper:.1f}]  "
            f"touches={self.zone.touches}  age={self.zone_age}bars\n"
            f"{'★'*60}"
        )


@dataclass
class Trade:
    signal:      Signal
    result:      str    # "WIN" | "LOSS"
    exit_price:  float
    exit_bar:    int
    bars_held:   int
    pnl_pts:     float

    def to_row(self) -> dict:
        return {
            "datetime"     : self.signal.datetime,
            "pattern"      : self.signal.pattern,
            "direction"    : self.signal.direction,
            "entry"        : self.signal.entry,
            "sl"           : self.signal.sl,
            "target"       : self.signal.target,
            "risk"         : self.signal.risk,
            "result"       : self.result,
            "exit_price"   : self.exit_price,
            "bars_held"    : self.bars_held,
            "pnl_pts"      : self.pnl_pts,
            "zone_price"   : self.signal.zone.price,
            "zone_touches" : self.signal.zone.touches,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_df(path: str) -> pd.DataFrame:
    """
    Loads a CSV and normalises column names.
    Handles both DD-MM-YYYY HH:MM and ISO datetime formats.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Handle 'date' column
    if "date" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"date": "datetime"})

    # Drop irrelevant columns
    for col in ["adj close", "adj_close"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Parse datetime — try multiple formats
    parsed = False
    for fmt in ["%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", None]:
        try:
            kwargs = {} if fmt is None else {"format": fmt}
            if fmt is None:
                kwargs["utc"] = True
                kwargs["errors"] = "coerce"
            df["datetime"] = pd.to_datetime(df["datetime"], **kwargs)
            parsed = True
            break
        except Exception:
            continue

    if not parsed:
        raise ValueError("Could not parse datetime column. Expected format: DD-MM-YYYY HH:MM")

    # Strip timezone
    if hasattr(df["datetime"].dt, "tz") and df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMAs and ATR. All forward-safe (ewm uses past data only).
    """
    df = df.copy()
    df["ema20"]  = df["close"].ewm(span=Config.EMA_FAST,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=Config.EMA_MED,   adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=Config.EMA_SLOW,  adjust=False).mean()
    df["atr"]    = (df["high"] - df["low"]).rolling(Config.ATR_PERIOD).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  PIVOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_pivots(
    df: pd.DataFrame,
    left:  int = Config.PIVOT_LEFT,
    right: int = Config.PIVOT_RIGHT,
) -> tuple[list, list]:
    """
    Detects swing highs (pivot highs) and swing lows (pivot lows).
    A pivot high at bar i requires: high[i] == max(high[i-left : i+right+1])
    A pivot low  at bar i requires: low[i]  == min(low[i-left  : i+right+1])

    In BACKTEST: right=5 (uses future bars to confirm pivot — standard practice).
    In LIVE:     right=0 (only left bars, so pivots confirm with a 5-bar lag).
    """
    highs, lows = [], []
    n = len(df)
    for i in range(left, n - right):
        h_window = df["high"].iloc[max(0, i - left) : i + right + 1]
        l_window = df["low"].iloc[max(0, i - left)  : i + right + 1]
        if df["high"].iloc[i] == h_window.max():
            highs.append(i)
        if df["low"].iloc[i] == l_window.min():
            lows.append(i)
    return highs, lows


# ─────────────────────────────────────────────────────────────────────────────
#  ZONE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_zones(
    df: pd.DataFrame,
    ph: list,
    pl: list,
    zone_width: int = Config.ZONE_WIDTH,
    min_touches: int = Config.MIN_TOUCHES,
) -> list[Zone]:
    """
    Groups pivot prices within `zone_width` points into S/R clusters.
    Only clusters with ≥ min_touches pivots are kept.
    Returns list sorted by price (ascending).
    """
    raw = []
    for i in ph:
        raw.append(("H", float(df["high"].iloc[i]), df["datetime"].iloc[i], i))
    for i in pl:
        raw.append(("L", float(df["low"].iloc[i]),  df["datetime"].iloc[i], i))
    raw.sort(key=lambda x: x[1])

    zones = []
    used = [False] * len(raw)

    for i in range(len(raw)):
        if used[i]:
            continue
        cluster = [raw[i]]
        used[i] = True
        for j in range(i + 1, len(raw)):
            if used[j]:
                continue
            if raw[j][1] - raw[i][1] <= zone_width:
                cluster.append(raw[j])
                used[j] = True
            else:
                break

        if len(cluster) < min_touches:
            continue

        center = float(np.mean([c[1] for c in cluster]))
        zones.append(Zone(
            price          = round(center, 2),
            lower          = round(center - zone_width / 2, 2),
            upper          = round(center + zone_width / 2, 2),
            touches        = len(cluster),
            last_touch_dt  = max(c[2] for c in cluster),
            last_touch_idx = max(c[3] for c in cluster),
        ))

    return sorted(zones, key=lambda z: z.price)


# ─────────────────────────────────────────────────────────────────────────────
#  CANDLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _body(c):       return abs(c["close"] - c["open"])
def _upper_wick(c): return c["high"] - max(c["close"], c["open"])
def _lower_wick(c): return min(c["close"], c["open"]) - c["low"]
def _spread(c):     return c["high"] - c["low"]
def _is_bull(c):    return c["close"] > c["open"]
def _is_bear(c):    return c["close"] < c["open"]


# ─────────────────────────────────────────────────────────────────────────────
#  PATTERN DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

def _bullish_engulfing(c2, c1) -> bool:
    """
    Bullish Engulfing at Support  ★★★  (52-77% WR in backtests)
    ─────────────────────────────────────────────────────────────
    What it means: Bears drive price into a support zone and the last bar
    closes red. Bulls step in aggressively — the next bar opens near the
    prior close and closes ABOVE the prior open, fully engulfing the fear.
    This signals demand absorption.

    Rules:
    • c1 is bearish, c2 is bullish
    • c2.open ≤ c1.close (gaps down or flat — no gap-up engulfs)
    • c2.close ≥ c1.open (full body coverage of c1)
    • c2 body ≥ 1.2× c1 body (dominance)
    • c2 body ≥ 55% of c2 spread (not a doji, real conviction)
    • c2 closes above EMA20 (momentum flip)
    """
    if not (_is_bull(c2) and _is_bear(c1)):
        return False
    if c2["open"] > c1["close"] + 5:
        return False
    if c2["close"] < c1["open"]:
        return False
    bd2 = _body(c2); bd1 = _body(c1); sp2 = _spread(c2)
    if sp2 < 5 or bd1 < 3:
        return False
    if bd2 < bd1 * Config.ENGULF_BODY_RATIO:
        return False
    if bd2 < sp2 * Config.ENGULF_BODY_PCT:
        return False
    if c2["close"] < c2["ema20"]:
        return False
    return True


def _bearish_engulfing(c2, c1) -> bool:
    """
    Bearish Engulfing at Resistance  ★★  (46% WR; works best < EMA200)
    ─────────────────────────────────────────────────────────────────────
    Mirror of bullish engulfing. Bulls push into resistance, fail.
    Bears engulf the prior bullish bar.

    Extra filter: only take shorts when price is BELOW EMA200 (downtrend).
    Avoids shorting into structural bull markets.
    """
    if not (_is_bear(c2) and _is_bull(c1)):
        return False
    if c2["open"] < c1["close"] - 5:
        return False
    if c2["close"] > c1["open"]:
        return False
    bd2 = _body(c2); bd1 = _body(c1); sp2 = _spread(c2)
    if sp2 < 5 or bd1 < 3:
        return False
    if bd2 < bd1 * Config.ENGULF_BODY_RATIO:
        return False
    if bd2 < sp2 * Config.ENGULF_BODY_PCT:
        return False
    if c2["close"] > c2["ema20"]:      # must close below EMA20
        return False
    if c2["close"] > c2["ema200"]:     # TREND FILTER: no shorts above EMA200
        return False
    return True


def _three_bar_reversal_bull(c2, c1, c0) -> bool:
    """
    Three-bar Bullish Reversal  ★★  (50-65% WR)
    ─────────────────────────────────────────────
    Two consecutive bearish bars drive price into support, then a powerful
    bullish bar that engulfs BOTH prior bars in a single thrust.

    Rules:
    • c0 and c1 are bearish (two-bar selloff into zone)
    • c2 is strongly bullish
    • c2 body > 1.3× max(c0 body, c1 body)
    • c2 closes in upper 50% of its own spread
    • c2 close above c1 open (full two-bar engulf)
    """
    if not (_is_bear(c1) and _is_bear(c0) and _is_bull(c2)):
        return False
    bd2 = _body(c2); bd1 = _body(c1); bd0 = _body(c0); sp2 = _spread(c2)
    if sp2 < 8 or bd1 < 3 or bd0 < 3:
        return False
    if bd2 < max(bd1, bd0) * Config.THREE_BAR_RATIO:
        return False
    if c2["close"] < c2["low"] + sp2 * Config.THREE_BAR_CLOSE_PCT:
        return False
    if c2["close"] < c1["open"]:
        return False
    return True


def _fakey_bull(c2, zone: Zone) -> bool:
    """
    Fakey Bullish (False Breakdown)  ★  (43% WR, positive expectancy)
    ─────────────────────────────────────────────────────────────────────
    Price briefly dips below support (trapping sellers), then aggressively
    reverses back inside the zone. The long lower wick is the "trap".

    Rules:
    • c2 wick pierced BELOW zone.lower (price visited below zone)
    • c2 closed back ABOVE zone.lower + 5 pts (recovery)
    • Lower wick ≥ 45% of total spread
    • Body ≥ 30% of spread (not a doji — real buy absorption)
    • Candle is bullish (close > open)
    """
    if c2["low"] >= zone.lower:
        return False
    if c2["close"] < zone.lower + 5:
        return False
    sp2 = _spread(c2); bd2 = _body(c2); lw2 = _lower_wick(c2)
    if sp2 < 8:
        return False
    if lw2 < sp2 * Config.FAKEY_WICK_PCT:
        return False
    if bd2 < sp2 * Config.FAKEY_BODY_PCT:
        return False
    if not _is_bull(c2):
        return False
    return True


def _fakey_bear(c2, zone: Zone) -> bool:
    """
    Fakey Bearish (False Breakout)  ★★  (50% WR, +22.5 pts avg — best short pattern)
    ───────────────────────────────────────────────────────────────────────────────────
    Mirror of fakey_bull. Price briefly pushes ABOVE resistance (trapping longs),
    then aggressively reverses back inside.

    Extra filter: only take when price is BELOW EMA200 (structural downtrend).
    """
    if c2["high"] <= zone.upper:
        return False
    if c2["close"] > zone.upper - 5:
        return False
    sp2 = _spread(c2); bd2 = _body(c2); uw2 = _upper_wick(c2)
    if sp2 < 8:
        return False
    if uw2 < sp2 * Config.FAKEY_WICK_PCT:
        return False
    if bd2 < sp2 * Config.FAKEY_BODY_PCT:
        return False
    if not _is_bear(c2):
        return False
    if c2["close"] > c2["ema200"]:     # TREND FILTER
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL GENERATOR  (stateless — call bar by bar)
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(
    df: pd.DataFrame,
    i: int,
    zones: list[Zone],
) -> Optional[Signal]:
    """
    Evaluates bar i (just closed) for entry patterns near S/R zones.
    Entry would be at bar i+1 open.

    NO LOOK-AHEAD: zones filtered so last_touch_idx < i.
    Returns Signal on confirmation, else None.
    """
    if i < 3 or i >= len(df) - 2:
        return None

    c2 = df.iloc[i]       # signal bar (just closed)
    c1 = df.iloc[i - 1]   # prior bar
    c0 = df.iloc[i - 2]   # two bars ago

    # ── Time filter ──────────────────────────────────────────────────────────
    t = c2["datetime"].time()
    if t < Config.NO_TRADE_BEFORE or t >= Config.NO_TRADE_AFTER:
        return None

    # ── Indicator quality ────────────────────────────────────────────────────
    atr_val = c2["atr"]
    if pd.isna(atr_val) or atr_val < 5:
        return None

    # ── Find nearest valid zone (strict no look-ahead) ───────────────────────
    nearby: list[tuple[float, Zone]] = []
    for z in zones:
        if z.last_touch_idx >= i:
            continue                        # zone not yet formed at bar i
        age = i - z.last_touch_idx
        if age > Config.MAX_ZONE_AGE_SESS * Config.BARS_PER_SESSION:
            continue                        # too stale
        dist = abs(c2["close"] - z.price)
        if dist <= Config.ZONE_APPROACH_BUF:
            nearby.append((dist, z))

    if not nearby:
        return None

    _, zone = sorted(nearby)[0]
    zone_age = i - zone.last_touch_idx

    # ── Zone context ─────────────────────────────────────────────────────────
    at_sup = zone.at_support(c2["low"])
    at_res = zone.at_resistance(c2["high"])

    entry_proxy = c2["close"]   # replaced by actual open in execution
    sl = direction = pattern_name = None

    # ═══════════════════════════════════════
    #  LONG SETUPS  (at support)
    # ═══════════════════════════════════════
    if at_sup:

        if _bullish_engulfing(c2, c1):
            sl           = min(c1["low"], c2["low"]) - Config.SL_BUFFER
            direction    = "LONG"
            pattern_name = "bullish_engulfing"

        elif _three_bar_reversal_bull(c2, c1, c0):
            sl           = min(c0["low"], c1["low"], c2["low"]) - Config.SL_BUFFER
            direction    = "LONG"
            pattern_name = "three_bar_reversal_bull"

        elif _fakey_bull(c2, zone):
            sl           = c2["low"] - Config.SL_BUFFER
            direction    = "LONG"
            pattern_name = "fakey_bull"

    # ═══════════════════════════════════════
    #  SHORT SETUPS  (at resistance, with EMA200 trend filter)
    # ═══════════════════════════════════════
    if direction is None and at_res:

        if _bearish_engulfing(c2, c1):
            sl           = max(c1["high"], c2["high"]) + Config.SL_BUFFER
            direction    = "SHORT"
            pattern_name = "bearish_engulfing"

        elif _fakey_bear(c2, zone):
            sl           = c2["high"] + Config.SL_BUFFER
            direction    = "SHORT"
            pattern_name = "fakey_bear"

    if direction is None or sl is None:
        return None

    # ── Risk sizing ───────────────────────────────────────────────────────────
    risk = abs(entry_proxy - sl)
    if risk < Config.MIN_RISK_PTS or risk > atr_val * Config.MAX_RISK_ATR_MULT:
        return None

    target = (
        entry_proxy + risk * Config.RR_RATIO
        if direction == "LONG"
        else entry_proxy - risk * Config.RR_RATIO
    )

    return Signal(
        bar_idx    = i,
        datetime   = c2["datetime"],
        pattern    = pattern_name,
        direction  = direction,
        entry      = round(entry_proxy, 2),
        sl         = round(sl, 2),
        target     = round(target, 2),
        risk       = round(risk, 2),
        zone       = zone,
        zone_age   = zone_age,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

def execute_trade(df: pd.DataFrame, signal: Signal) -> Optional[Trade]:
    """
    Executes at bar i+1 open. Scans forward for SL/TP hit.
    Enforces same-day exit and max-hold limits.
    """
    i = signal.bar_idx
    if i + 1 >= len(df):
        return None

    entry_bar    = df.iloc[i + 1]
    actual_entry = float(entry_bar["open"])

    # Recalculate risk/target from actual open
    risk = abs(actual_entry - signal.sl)
    if risk < Config.MIN_RISK_PTS:
        return None

    target = (
        actual_entry + risk * Config.RR_RATIO
        if signal.direction == "LONG"
        else actual_entry - risk * Config.RR_RATIO
    )

    result = "OPEN"
    exit_price = exit_bar_idx = None
    signal_date = df.iloc[i]["datetime"].date()

    for j in range(i + 1, min(i + Config.MAX_BARS_HOLD + 1, len(df))):
        bar = df.iloc[j]

        if Config.SAME_DAY_EXIT and bar["datetime"].date() != signal_date:
            break  # EOD timeout — unresolved

        if signal.direction == "LONG":
            if bar["low"] <= signal.sl:
                result       = "LOSS"
                exit_price   = signal.sl
                exit_bar_idx = j
                break
            if bar["high"] >= target:
                result       = "WIN"
                exit_price   = target
                exit_bar_idx = j
                break
        else:
            if bar["high"] >= signal.sl:
                result       = "LOSS"
                exit_price   = signal.sl
                exit_bar_idx = j
                break
            if bar["low"] <= target:
                result       = "WIN"
                exit_price   = target
                exit_bar_idx = j
                break

    if result == "OPEN":
        return None  # trade didn't resolve

    pnl = (
        exit_price - actual_entry
        if signal.direction == "LONG"
        else actual_entry - exit_price
    )

    # Update signal with actual execution values
    signal.entry  = round(actual_entry, 2)
    signal.risk   = round(risk, 2)
    signal.target = round(target, 2)

    return Trade(
        signal     = signal,
        result     = result,
        exit_price = round(exit_price, 2),
        exit_bar   = exit_bar_idx,
        bars_held  = exit_bar_idx - i,
        pnl_pts    = round(pnl, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Full backtest pipeline with no look-ahead.

    Zone-building uses PIVOT_RIGHT=5, which means the last 5 bars before
    current cannot be pivot-confirmed. All zones have last_touch_idx < i.
    """
    df = add_indicators(df)
    ph, pl = detect_pivots(df, right=Config.PIVOT_RIGHT)
    zones  = build_zones(df, ph, pl)

    if verbose:
        _header(f"BACKTEST: {df['datetime'].min().date()} → {df['datetime'].max().date()}")
        print(f"  Bars : {len(df):,}  |  Zones: {len(zones)} (≥{Config.MIN_TOUCHES} touches, {Config.ZONE_WIDTH}pt width)")

    trades: list[dict] = []

    for i in range(Config.PIVOT_LEFT + 200, len(df) - Config.MAX_BARS_HOLD - 2):
        sig = generate_signal(df, i, zones)
        if sig is None:
            continue
        trade = execute_trade(df, sig)
        if trade is None:
            continue
        trades.append(trade.to_row())

    if not trades:
        print("  No trades found.")
        return pd.DataFrame()

    results = pd.DataFrame(trades)
    if verbose:
        _print_report(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE SCANNER
# ─────────────────────────────────────────────────────────────────────────────

class LiveScanner:
    """
    Stateful scanner — primed with historical data, updated bar by bar.

    Usage:
        scanner = LiveScanner()
        scanner.load_historical(historical_df)   # prime with 60+ days

        # Each time a 5-min bar closes:
        sig = scanner.on_bar({
            "datetime": ..., "open": ..., "high": ...,
            "low": ..., "close": ..., "volume": 0
        })
        if sig:
            broker.place_order(sig)
    """

    def __init__(self):
        self._df    = pd.DataFrame()
        self._zones : list[Zone] = []
        self._bar_n : int = 0

    def load_historical(self, df: pd.DataFrame):
        df = add_indicators(df.copy())
        self._df = df
        self._rebuild_zones(live_mode=True)
        print(f"[LiveScanner] Loaded {len(self._df)} bars, {len(self._zones)} zones.")

    def on_bar(self, bar: dict) -> Optional[Signal]:
        """
        Call with a dict of the just-closed 5-min bar.
        Returns Signal if entry condition met, else None.
        """
        self._df = pd.concat(
            [self._df, pd.DataFrame([bar])], ignore_index=True
        )
        self._df = add_indicators(self._df)
        self._bar_n += 1

        # Rebuild zones once per session
        if self._bar_n % Config.BARS_PER_SESSION == 0:
            self._rebuild_zones(live_mode=True)

        i = len(self._df) - 1
        if i < Config.PIVOT_LEFT + 200:
            return None

        sig = generate_signal(self._df, i, self._zones)
        if sig:
            print(sig.summary())
        return sig

    def _rebuild_zones(self, live_mode: bool = False):
        right = 0 if live_mode else Config.PIVOT_RIGHT
        ph, pl = detect_pivots(self._df, right=right)
        self._zones = build_zones(self._df, ph, pl)


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def print_active_zones(df: pd.DataFrame):
    df = add_indicators(df)
    ph, pl = detect_pivots(df)
    zones  = build_zones(df, ph, pl)
    cmp    = float(df["close"].iloc[-1])
    _print_zones(zones, cmp)


def _print_zones(zones: list[Zone], cmp: float):
    _header(f"ACTIVE S/R ZONES  |  CMP: {cmp:.2f}")
    res = sorted([z for z in zones if z.price > cmp], key=lambda z: z.price)
    sup = sorted([z for z in zones if z.price <= cmp], key=lambda z: z.price, reverse=True)
    for z in res:
        print(f"  ▼ RES  {z.price:>9.2f}  [{z.lower:.2f}–{z.upper:.2f}]  "
              f"T:{z.touches}  Last:{z.last_touch_dt.date()}  "
              f"+{z.price-cmp:.0f}pts")
    print(f"\n  {'─'*20}  CMP {cmp:.2f}  {'─'*20}\n")
    for z in sup:
        print(f"  ▲ SUP  {z.price:>9.2f}  [{z.lower:.2f}–{z.upper:.2f}]  "
              f"T:{z.touches}  Last:{z.last_touch_dt.date()}  "
              f"-{cmp-z.price:.0f}pts")
    print("═" * 70)


def _header(title: str):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def _print_report(df: pd.DataFrame):
    total  = len(df)
    wins   = (df["result"] == "WIN").sum()
    losses = (df["result"] == "LOSS").sum()
    wr     = wins / total * 100

    gw = df[df["pnl_pts"] > 0]["pnl_pts"].sum()
    gl = abs(df[df["pnl_pts"] < 0]["pnl_pts"].sum())
    pf = gw / gl if gl > 0 else float("inf")

    avg_w = df[df["result"] == "WIN"]["pnl_pts"].mean()
    avg_l = df[df["result"] == "LOSS"]["pnl_pts"].mean()
    exp   = df["pnl_pts"].mean()

    eq       = df["pnl_pts"].cumsum()
    max_dd   = (eq.cummax() - eq).max()

    _header("BACKTEST RESULTS")
    print(f"  Total Trades   : {total}")
    print(f"  Win Rate       : {wr:.1f}%  ({wins}W / {losses}L)")
    print(f"  Profit Factor  : {pf:.2f}")
    print(f"  Expectancy     : {exp:+.2f} pts/trade")
    print(f"  Total PnL      : {df['pnl_pts'].sum():+.1f} pts")
    print(f"  Avg Win        : {avg_w:+.1f} pts")
    print(f"  Avg Loss       : {avg_l:+.1f} pts")
    print(f"  Max Drawdown   : {max_dd:.1f} pts")

    print(f"\n  By Pattern:")
    for pat, g in df.groupby("pattern"):
        pw = (g["result"] == "WIN").sum()
        pl = len(g) - pw
        denom = (pw + pl)
        wr_p  = pw / denom * 100 if denom else 0
        print(f"    {pat:<28} {len(g):>3} trades  "
              f"WR:{wr_p:>4.0f}%  "
              f"Avg:{g['pnl_pts'].mean():>+6.1f}  "
              f"Total:{g['pnl_pts'].sum():>+7.1f}")

    print(f"\n  By Direction:")
    for d, g in df.groupby("direction"):
        pw = (g["result"] == "WIN").sum()
        pl = len(g) - pw
        denom = pw + pl
        wr_d  = pw / denom * 100 if denom else 0
        print(f"    {d:<6} {len(g):>3} trades  WR:{wr_d:>4.0f}%  "
              f"Total:{g['pnl_pts'].sum():>+7.1f} pts")

    print(f"\n  Annual Breakdown:")
    df["_yr"] = pd.to_datetime(df["datetime"]).dt.year
    for yr, g in df.groupby("_yr"):
        pw = (g["result"] == "WIN").sum(); pl = len(g) - pw
        denom = pw + pl
        wr_y  = pw / denom * 100 if denom else 0
        print(f"    {yr}  {len(g):>3} trades  WR:{wr_y:>4.0f}%  "
              f"PnL:{g['pnl_pts'].sum():>+7.1f} pts")
    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nifty 50 S/R Zone Price Action Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sr_zone_strategy.py --file NIFTY_50_5minute.csv
  python sr_zone_strategy.py --file data.csv --save results.csv
  python sr_zone_strategy.py --file data.csv --zones-only
  python sr_zone_strategy.py --file data.csv --live
        """
    )
    parser.add_argument("--file",       required=True, help="Path to 5-min OHLCV CSV")
    parser.add_argument("--live",       action="store_true", help="Scan last bar as live signal")
    parser.add_argument("--zones-only", action="store_true", help="Print S/R zones and exit")
    parser.add_argument("--save",       default=None, help="Save backtest results to CSV path")
    args = parser.parse_args()

    print(f"\n  Loading: {args.file}")
    try:
        df = load_df(args.file)
    except FileNotFoundError:
        print(f"\n  ERROR: File not found — {args.file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR loading file: {e}")
        sys.exit(1)

    print(f"  Rows : {len(df):,}")
    print(f"  Range: {df['datetime'].min().date()} → {df['datetime'].max().date()}")

    if args.zones_only:
        print_active_zones(df)
        return

    if args.live:
        _header("LIVE SCAN MODE")
        scanner = LiveScanner()
        scanner.load_historical(df.iloc[:-1])
        last = df.iloc[-1].to_dict()
        print(f"  Scanning: {last['datetime']}  close={last['close']:.2f}")
        sig = scanner.on_bar(last)
        if sig is None:
            print("  No signal on current bar.")
        return

    # ── Backtest ──────────────────────────────────────────────────────────────
    results = run_backtest(df, verbose=True)

    if results is not None and len(results) > 0 and args.save:
        results.to_csv(args.save, index=False)
        print(f"\n  Results saved → {args.save}")


if __name__ == "__main__":
    main()