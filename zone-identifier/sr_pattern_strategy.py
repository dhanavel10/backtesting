"""
sr_pattern_strategy.py
======================
60-day backtest: NIFTY 5-min S/R + rejection-candle entries.

Pipeline
--------
1. Fetch 60d/5m NIFTY via yfinance.
2. Build TWO tiers of S/R zones:
     a. HTF anchors  (python_visual.build_precision_zones)
        - multi-session validated, point-clustered, scored.
     b. Intraday zones (realtime_sr.RealtimeSREngine)
        - rebuilt bar-by-bar from causal pivots + KDE.
   A zone is HIGH-CONVICTION when both tiers agree within MATCH_DIST_PTS.
3. On every closed bar, run a candlestick-pattern detector:
        bullish_pin / hammer  -> at Support
        bearish_pin / shooting_star -> at Resistance
   Pattern + zone proximity + strength filter = entry signal.
4. Trade management (point-based, 1-unit, intraday):
     entry  = next bar open
     SL     = far side of the signal candle (or zone far edge, whichever tighter)
     TRAIL  = once price moves TRAIL_START_PTS in favor, trail SL by TRAIL_STEP_PTS
              from the running high/low — no fixed target, let winners run.
     exits  = trailing SL hit / 15:15 EOD / 20-bar time-stop
5. Outputs:
     nifty_sr_pattern_trades.csv     per-trade ledger
     nifty_sr_pattern_equity.csv     equity curve
     nifty_sr_pattern_report.html    plotly chart + summary
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# 0.  IMPORTS FROM YOUR EXISTING MODULES
# ──────────────────────────────────────────────────────────────────────
from realtime_sr import RealtimeSREngine, Zone
from python_visual import (
    fetch_intraday_chunked,
    build_precision_zones,
)


# ══════════════════════════════════════════════════════════════════════
# 1. CONFIG  (tune from .env or here)
# ══════════════════════════════════════════════════════════════════════

TICKER              = os.getenv("TICKER",          "^NSEI")
DAYS                = int(os.getenv("BT_DAYS",     "55"))     # yfinance 5m cap = 60d
INTERVAL            = "5m"

# HTF precision-zone params (python_visual)
HTF_LEFT_BARS       = 10
HTF_RIGHT_BARS      = 10
HTF_CLUSTER_TOL     = 15.0
HTF_ZONE_HALF_BAND  = 15.0
HTF_MIN_TOUCHES     = 3
HTF_MIN_SESSIONS    = 2
HTF_MIN_REJECTIONS  = 1
HTF_TOP_N           = 30

# Intraday engine params (realtime_sr)
RT_REVERSAL_THR     = float(os.getenv("REVERSAL_THR",  "30.0"))
RT_BANDWIDTH        = float(os.getenv("BANDWIDTH",     "7.0"))
RT_ZONE_HW          = float(os.getenv("ZONE_HW",       "17.5"))
RT_HALF_LIFE_MIN    = float(os.getenv("HALF_LIFE_MIN", "120.0"))
RT_TOP_N            = 10

# Zone-match: HTF and RT consider the same level if within this many points
MATCH_DIST_PTS      = 12.0

# Entry filters
PROXIMITY_PTS       = 5.0     # candle extreme must be inside zone or within this many pts of edge
MIN_ZONE_STRENGTH   = 50.0    # RT strength % (only applied when no HTF confluence)
REQUIRE_HTF         = False   # if True, demand HTF confluence (high precision, fewer trades)
REQUIRE_ANCHORED    = False   # if True, demand RT zone is anchored

# Pin-bar pattern thresholds
PIN_WICK_BODY_RATIO = 2.0     # wick >= N * body
PIN_WICK_TOTAL_PCT  = 0.55    # rejection wick >= N% of total range
PIN_OPP_WICK_PCT    = 0.25    # opposite wick <= N% of total range
PIN_CLOSE_THIRD     = 0.66    # close in upper/lower third confirmation

# Supertrend filter
ST_PERIOD           = int(os.getenv("ST_PERIOD",     "7"))     # ATR period
ST_MULTIPLIER       = float(os.getenv("ST_MULT",     "3.0"))   # band multiplier

# Risk / trade management
# Trailing stop — replaces fixed target
TRAIL_START_PTS     = float(os.getenv("TRAIL_START_PTS", "10.0"))  # min favorable move before trailing activates
TRAIL_STEP_PTS      = float(os.getenv("TRAIL_STEP_PTS",  "20.0"))  # trail distance from running extreme

TIME_STOP_BARS      = 20      # exit if not done within N bars
EOD_HHMM            = (15, 15) # exit any open trade at this clock time
ONE_TRADE_AT_A_TIME = True
COST_PER_TRADE_PTS  = 0.5     # slippage+brokerage per round trip in points
MAX_TRADES_PER_DAY  = int(os.getenv("MAX_TRADES_PER_DAY", "3"))

START_CAPITAL       = 100000.0  # for equity curve (point-based, scaled by 1 unit)


# ══════════════════════════════════════════════════════════════════════
# 2. DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CompositeZone:
    """A zone usable by the strategy. May come from HTF, RT, or both."""
    price:      float
    lower:      float
    upper:      float
    type:       str             # "Support" | "Resistance"
    strength:   float           # RT strength% if available, else mapped from HTF strength
    source:     str             # "HTF" | "RT" | "BOTH"
    htf_score:  Optional[float] = None
    rt_anchored: bool = False
    n_pivots:   int = 0


@dataclass
class Signal:
    ts:         datetime
    bar_idx:    int
    direction:  str             # "LONG" | "SHORT"
    pattern:    str             # "hammer" | "shooting_star" | ...
    zone:       CompositeZone
    sig_high:   float
    sig_low:    float
    sig_open:   float
    sig_close:  float
    confluences: List[str] = field(default_factory=list)


@dataclass
class Trade:
    entry_ts:   datetime
    exit_ts:    Optional[datetime] = None
    direction:  str = ""
    entry:      float = 0.0
    sl:         float = 0.0         # initial hard SL from signal candle
    trail_sl:   float = 0.0         # current trailing stop (moves in favor)
    trail_extreme: float = 0.0      # running high (LONG) or low (SHORT)
    trail_active: bool = False       # True once TRAIL_START_PTS achieved
    exit_px:    Optional[float] = None
    exit_reason: str = ""
    pnl_pts:    float = 0.0
    bars_held:  int = 0
    pattern:    str = ""
    zone_price: float = 0.0
    zone_source: str = ""
    zone_strength: float = 0.0
    confluences: str = ""
    mfe:        float = 0.0     # max favourable excursion
    mae:        float = 0.0     # max adverse excursion


# ══════════════════════════════════════════════════════════════════════
# 3A. SUPERTREND INDICATOR
# ══════════════════════════════════════════════════════════════════════

def compute_supertrend(df: pd.DataFrame,
                       period: int = ST_PERIOD,
                       multiplier: float = ST_MULTIPLIER):
    """
    Returns (direction_series, supertrend_line_series).
    direction:  +1 = price above supertrend (bullish trend)
               -1 = price below supertrend (bearish trend)

    Only take LONG entries when direction == +1 (trend is up).
    Only take SHORT entries when direction == -1 (trend is down).
    This avoids trading against the prevailing intraday trend.
    """
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n = len(df)

    # True range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))

    # Wilder smoothed ATR
    atr = np.empty(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    hl2 = (high + low) / 2.0
    bu  = hl2 + multiplier * atr    # basic upper band
    bl  = hl2 - multiplier * atr    # basic lower band

    # Final (ratcheted) bands
    fu = bu.copy()
    fl = bl.copy()
    for i in range(1, n):
        fu[i] = bu[i] if (bu[i] < fu[i-1] or close[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = bl[i] if (bl[i] > fl[i-1] or close[i-1] < fl[i-1]) else fl[i-1]

    # Supertrend line + direction
    direction = np.ones(n, dtype=int)
    st = np.empty(n)
    st[0] = fl[0]

    for i in range(1, n):
        if st[i - 1] == fu[i - 1]:          # was bearish
            if close[i] > fu[i]:
                direction[i] = 1;  st[i] = fl[i]
            else:
                direction[i] = -1; st[i] = fu[i]
        else:                                # was bullish
            if close[i] < fl[i]:
                direction[i] = -1; st[i] = fu[i]
            else:
                direction[i] = 1;  st[i] = fl[i]

    return (pd.Series(direction, index=df.index, name="st_dir"),
            pd.Series(st,        index=df.index, name="st_line"))


# ══════════════════════════════════════════════════════════════════════
# 3B. CANDLESTICK PATTERN DETECTOR
# ══════════════════════════════════════════════════════════════════════

def detect_pattern(o, h, l, c) -> Optional[str]:
    """
    Return pattern name or None.
    Focus: pin bar / hammer / shooting star (rejection candles).
    """
    total = h - l
    if total <= 0:
        return None
    body       = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    body_ratio = body / total

    # Hammer / bullish pin — long lower wick, small body, close in upper third
    if (lower_wick >= PIN_WICK_BODY_RATIO * max(body, 1e-6)
            and lower_wick / total >= PIN_WICK_TOTAL_PCT
            and upper_wick / total <= PIN_OPP_WICK_PCT
            and c >= l + PIN_CLOSE_THIRD * total
            and body_ratio <= 0.4):
        return "hammer"

    # Shooting star / bearish pin — long upper wick, small body, close in lower third
    if (upper_wick >= PIN_WICK_BODY_RATIO * max(body, 1e-6)
            and upper_wick / total >= PIN_WICK_TOTAL_PCT
            and lower_wick / total <= PIN_OPP_WICK_PCT
            and c <= l + (1 - PIN_CLOSE_THIRD) * total
            and body_ratio <= 0.4):
        return "shooting_star"

    return None


# ══════════════════════════════════════════════════════════════════════
# 4. ZONE COMPOSER  (merge HTF + RT)
# ══════════════════════════════════════════════════════════════════════

def htf_zones_to_composite(htf_df: pd.DataFrame, current_price: float) -> List[CompositeZone]:
    """Convert the precision-zone DataFrame into CompositeZone list."""
    out: List[CompositeZone] = []
    if htf_df is None or len(htf_df) == 0:
        return out
    for _, z in htf_df.iterrows():
        out.append(CompositeZone(
            price=float(z["price"]),
            lower=float(z["lower"]),
            upper=float(z["upper"]),
            type="Support" if float(z["price"]) < current_price else "Resistance",
            strength=float(z["strength"]),
            source="HTF",
            htf_score=float(z["strength"]),
            n_pivots=int(z["n_pivots"]),
        ))
    return out


def compose_zones(htf_zones: List[CompositeZone], rt_zones: List[Zone],
                  current_price: float) -> List[CompositeZone]:
    """
    Merge HTF (static for the run) + RT (refreshed per bar).
    A zone present in both is marked source=BOTH and strength boosted.
    """
    composed: List[CompositeZone] = []
    used_htf = set()

    for rt in rt_zones:
        match = None
        for i, htf in enumerate(htf_zones):
            if i in used_htf:
                continue
            if abs(rt.price - htf.price) <= MATCH_DIST_PTS:
                match = (i, htf)
                break
        if match:
            i, htf = match
            used_htf.add(i)
            ztype = "Support" if rt.price < current_price else "Resistance"
            composed.append(CompositeZone(
                price=rt.price,
                lower=min(rt.lower, htf.lower),
                upper=max(rt.upper, htf.upper),
                type=ztype,
                strength=min(100.0, rt.strength + 25.0),  # confluence bonus
                source="BOTH",
                htf_score=htf.htf_score,
                rt_anchored=rt.anchored,
                n_pivots=rt.n_pivots,
            ))
        else:
            ztype = "Support" if rt.price < current_price else "Resistance"
            composed.append(CompositeZone(
                price=rt.price, lower=rt.lower, upper=rt.upper, type=ztype,
                strength=rt.strength, source="RT",
                rt_anchored=rt.anchored, n_pivots=rt.n_pivots,
            ))

    # HTF zones with no RT counterpart still count — they're the long-term anchors
    for i, htf in enumerate(htf_zones):
        if i in used_htf:
            continue
        ztype = "Support" if htf.price < current_price else "Resistance"
        composed.append(CompositeZone(
            price=htf.price, lower=htf.lower, upper=htf.upper, type=ztype,
            strength=htf.htf_score or 50.0, source="HTF",
            htf_score=htf.htf_score, n_pivots=htf.n_pivots,
        ))

    return composed


# ══════════════════════════════════════════════════════════════════════
# 5. ENTRY LOGIC
# ══════════════════════════════════════════════════════════════════════

def candle_near_zone(side: str, h: float, l: float, z: CompositeZone) -> bool:
    """
    Bullish setup (side='LONG') wants the candle LOW inside, or within
    PROXIMITY_PTS below, a Support zone.
    Bearish setup (side='SHORT') wants the HIGH inside or within
    PROXIMITY_PTS above, a Resistance zone.
    """
    if side == "LONG" and z.type == "Support":
        return (z.lower - PROXIMITY_PTS) <= l <= z.upper
    if side == "SHORT" and z.type == "Resistance":
        return z.lower <= h <= (z.upper + PROXIMITY_PTS)
    return False


def pick_signal(o, h, l, c, zones: List[CompositeZone],
                ts, bar_idx, st_dir: int = 0) -> Optional[Signal]:
    pattern = detect_pattern(o, h, l, c)
    if pattern is None:
        return None

    side = "LONG" if pattern == "hammer" else "SHORT"

    # Supertrend alignment: only trade WITH the trend
    if st_dir == 1 and side == "SHORT":
        return None   # supertrend bullish → skip shorts
    if st_dir == -1 and side == "LONG":
        return None   # supertrend bearish → skip longs

    candidates = [z for z in zones if candle_near_zone(side, h, l, z)]
    if not candidates:
        return None

    # rank: BOTH > HTF > RT, then by strength
    src_rank = {"BOTH": 0, "HTF": 1, "RT": 2}
    candidates.sort(key=lambda z: (src_rank.get(z.source, 9), -z.strength))
    zone = candidates[0]

    # Filters
    if REQUIRE_HTF and zone.source == "RT":
        return None
    if REQUIRE_ANCHORED and zone.source == "RT" and not zone.rt_anchored:
        return None
    if zone.source == "RT" and zone.strength < MIN_ZONE_STRENGTH:
        return None

    conf = []
    if zone.source == "BOTH":  conf.append("HTF+RT confluence")
    if zone.rt_anchored:        conf.append("anchored")
    if zone.htf_score and zone.htf_score >= 70: conf.append(f"HTF score {zone.htf_score:.0f}")
    if zone.strength >= 80:     conf.append("very strong zone")
    if st_dir != 0:             conf.append(f"ST {'bullish' if st_dir == 1 else 'bearish'}")

    return Signal(
        ts=ts, bar_idx=bar_idx, direction=side, pattern=pattern, zone=zone,
        sig_high=h, sig_low=l, sig_open=o, sig_close=c, confluences=conf,
    )


# ══════════════════════════════════════════════════════════════════════
# 6. RISK PARAMETERS PER SIGNAL
# ══════════════════════════════════════════════════════════════════════

def compute_trade_levels(sig: Signal) -> Optional[tuple]:
    """
    Return (sl,) — initial hard stop from the signal candle / zone far edge.
    No fixed target; exit is managed by the trailing stop.
    """
    if sig.direction == "LONG":
        sl_candle = sig.sig_low  - 0.5
        sl_zone   = sig.zone.lower - 0.5
        return (min(sl_candle, sl_zone),)
    else:
        sl_candle = sig.sig_high + 0.5
        sl_zone   = sig.zone.upper + 0.5
        return (max(sl_candle, sl_zone),)


# ══════════════════════════════════════════════════════════════════════
# 7. BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, htf_df: pd.DataFrame) -> tuple:
    """Returns (trades, equity_df, signals, zone_snapshot_at_end)."""
    htf_compose_full = htf_zones_to_composite(htf_df, float(df["Close"].iloc[-1]))

    engine = RealtimeSREngine(
        reversal_threshold=RT_REVERSAL_THR,
        bandwidth=RT_BANDWIDTH,
        zone_half_width=RT_ZONE_HW,
        half_life_min=RT_HALF_LIFE_MIN,
    )

    # Precompute supertrend for all bars
    st_dir_series, _ = compute_supertrend(df)
    st_dirs = st_dir_series.values   # numpy array, index-aligned

    trades:  List[Trade]  = []
    signals: List[Signal] = []
    open_trade: Optional[Trade] = None

    bar_times = df.index.to_pydatetime() if hasattr(df.index, "to_pydatetime") \
                else [pd.Timestamp(t).to_pydatetime() for t in df.index]

    rows = df[["Open", "High", "Low", "Close"]].values
    n    = len(rows)

    cum_pnl = 0.0
    equity_rows = []

    # Daily trade cap tracking
    day_trade_count = 0
    current_day     = None

    for i in range(n):
        o, h, l, c = rows[i]
        ts = bar_times[i]

        # Reset daily counter at session start
        bar_date = ts.date()
        if bar_date != current_day:
            current_day     = bar_date
            day_trade_count = 0

        # --- 1. Feed bar to RT engine (HTF zones are static)
        engine.on_candle(float(h), float(l), ts)

        # --- 2. Manage open trade FIRST (stop check → trail update)
        if open_trade is not None:
            ot = open_trade
            ot.bars_held += 1
            hit_stop = False

            if ot.direction == "LONG":
                ot.mfe = max(ot.mfe, float(h) - ot.entry)
                ot.mae = max(ot.mae, ot.entry - float(l))
                # check trailing SL hit this bar
                if float(l) <= ot.trail_sl:
                    ot.exit_px = ot.trail_sl
                    hit_stop   = True
                else:
                    # advance the running high, then ratchet trail up
                    ot.trail_extreme = max(ot.trail_extreme, float(h))
                    favor = ot.trail_extreme - ot.entry
                    if favor >= TRAIL_START_PTS:
                        ot.trail_active = True
                    if ot.trail_active:
                        new_trail = ot.trail_extreme - TRAIL_STEP_PTS
                        ot.trail_sl = max(ot.trail_sl, new_trail)
            else:
                ot.mfe = max(ot.mfe, ot.entry - float(l))
                ot.mae = max(ot.mae, float(h) - ot.entry)
                if float(h) >= ot.trail_sl:
                    ot.exit_px = ot.trail_sl
                    hit_stop   = True
                else:
                    ot.trail_extreme = min(ot.trail_extreme, float(l))
                    favor = ot.entry - ot.trail_extreme
                    if favor >= TRAIL_START_PTS:
                        ot.trail_active = True
                    if ot.trail_active:
                        new_trail = ot.trail_extreme + TRAIL_STEP_PTS
                        ot.trail_sl = min(ot.trail_sl, new_trail)

            eod_now = ts.time() >= dtime(*EOD_HHMM)
            time_up = ot.bars_held >= TIME_STOP_BARS

            if hit_stop or eod_now or time_up:
                if not hit_stop:
                    ot.exit_px     = float(c)
                    ot.exit_reason = "EOD" if eod_now else "TIME_STOP"
                else:
                    ot.exit_reason = "TRAIL_SL"
                ot.exit_ts = ts
                ot.pnl_pts = (ot.exit_px - ot.entry) if ot.direction == "LONG" \
                             else (ot.entry - ot.exit_px)
                ot.pnl_pts -= COST_PER_TRADE_PTS
                cum_pnl   += ot.pnl_pts
                trades.append(ot)
                open_trade = None

        # --- 3. Compose zones & look for signal (only when flat)
        rt_zones = engine.zones(current_price=float(c), now=ts, top_n=RT_TOP_N)
        # HTF type re-evaluated against current price
        htf_now = [
            CompositeZone(
                price=z.price, lower=z.lower, upper=z.upper,
                type="Support" if z.price < c else "Resistance",
                strength=z.strength, source="HTF",
                htf_score=z.htf_score, n_pivots=z.n_pivots,
            )
            for z in htf_compose_full
        ]
        composed = compose_zones(htf_now, rt_zones, float(c))

        if (open_trade is None
                and ts.time() < dtime(*EOD_HHMM)
                and day_trade_count < MAX_TRADES_PER_DAY):
            sig = pick_signal(float(o), float(h), float(l), float(c),
                              composed, ts, i, st_dir=int(st_dirs[i]))
            if sig is not None and i + 1 < n:
                levels = compute_trade_levels(sig)
                if levels is not None:
                    (sl,) = levels
                    entry = float(rows[i + 1][0])   # next bar open
                    ok = (sig.direction == "LONG"  and entry > sl) or \
                         (sig.direction == "SHORT" and entry < sl)
                    if ok:
                        open_trade = Trade(
                            entry_ts=bar_times[i + 1],
                            direction=sig.direction,
                            entry=entry, sl=sl,
                            trail_sl=sl,
                            trail_extreme=entry,
                            pattern=sig.pattern,
                            zone_price=sig.zone.price,
                            zone_source=sig.zone.source,
                            zone_strength=sig.zone.strength,
                            confluences=", ".join(sig.confluences),
                        )
                        signals.append(sig)
                        day_trade_count += 1

        equity_rows.append({"ts": ts, "cum_pnl": cum_pnl,
                            "equity": START_CAPITAL + cum_pnl})

    if open_trade is not None:
        open_trade.exit_ts     = bar_times[-1]
        open_trade.exit_px     = float(rows[-1][3])
        open_trade.exit_reason = "EOD_FINAL"
        open_trade.pnl_pts     = (open_trade.exit_px - open_trade.entry) \
                                  if open_trade.direction == "LONG" \
                                  else (open_trade.entry - open_trade.exit_px)
        open_trade.pnl_pts -= COST_PER_TRADE_PTS
        trades.append(open_trade)

    eq_df = pd.DataFrame(equity_rows).set_index("ts")
    return trades, eq_df, signals, composed


# ══════════════════════════════════════════════════════════════════════
# 8. STATS & REPORT
# ══════════════════════════════════════════════════════════════════════

def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        "entry_ts":    t.entry_ts,
        "exit_ts":     t.exit_ts,
        "direction":   t.direction,
        "pattern":     t.pattern,
        "entry":       round(t.entry, 2),
        "init_sl":     round(t.sl, 2),
        "trail_sl_final": round(t.trail_sl, 2),
        "trail_active": t.trail_active,
        "exit_px":     round(t.exit_px or 0, 2),
        "exit_reason": t.exit_reason,
        "pnl_pts":     round(t.pnl_pts, 2),
        "bars_held":   t.bars_held,
        "zone_price":  round(t.zone_price, 2),
        "zone_src":    t.zone_source,
        "zone_str":    round(t.zone_strength, 1),
        "mfe":         round(t.mfe, 2),
        "mae":         round(t.mae, 2),
        "confluences": t.confluences,
    } for t in trades])


def compute_stats(tr_df: pd.DataFrame) -> dict:
    if tr_df.empty:
        return {"trades": 0}
    wins   = tr_df[tr_df["pnl_pts"] > 0]
    losses = tr_df[tr_df["pnl_pts"] <= 0]
    total  = tr_df["pnl_pts"].sum()
    gross_w = wins["pnl_pts"].sum()
    gross_l = abs(losses["pnl_pts"].sum())
    return {
        "trades":     len(tr_df),
        "wins":       len(wins),
        "losses":     len(losses),
        "win_rate":   100 * len(wins) / len(tr_df),
        "avg_win":    wins["pnl_pts"].mean() if len(wins) else 0,
        "avg_loss":   losses["pnl_pts"].mean() if len(losses) else 0,
        "expectancy": tr_df["pnl_pts"].mean(),
        "total_pts":  total,
        "profit_factor": (gross_w / gross_l) if gross_l > 0 else float("inf"),
        "best":       tr_df["pnl_pts"].max(),
        "worst":      tr_df["pnl_pts"].min(),
        "longs":      int((tr_df["direction"] == "LONG").sum()),
        "shorts":     int((tr_df["direction"] == "SHORT").sum()),
    }


def print_stats(stats: dict):
    if stats.get("trades", 0) == 0:
        print("\n  No trades generated.\n")
        return
    print("\n" + "═" * 70)
    print("  BACKTEST RESULTS")
    print("═" * 70)
    print(f"  Trades            : {stats['trades']}  "
          f"({stats['longs']}L / {stats['shorts']}S)")
    print(f"  Win rate          : {stats['win_rate']:.1f}%   "
          f"({stats['wins']}W / {stats['losses']}L)")
    print(f"  Total points      : {stats['total_pts']:+.1f} pts")
    print(f"  Expectancy/trade  : {stats['expectancy']:+.2f} pts")
    print(f"  Avg win / avg loss: {stats['avg_win']:+.2f}  /  {stats['avg_loss']:+.2f}")
    print(f"  Profit factor     : {stats['profit_factor']:.2f}")
    print(f"  Best / Worst      : {stats['best']:+.1f}  /  {stats['worst']:+.1f}")
    print("═" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────
# 9. PLOTLY REPORT
# ──────────────────────────────────────────────────────────────────────

def build_html_report(df: pd.DataFrame, htf_df: pd.DataFrame,
                      trades: List[Trade], eq_df: pd.DataFrame,
                      signals: List[Signal], stats: dict,
                      out_path: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # last 10 sessions for chart density
    sessions = sorted(set(df.index.date))
    last_sess = sessions[-10:] if len(sessions) > 10 else sessions
    df_plot = df[df.index.date >= last_sess[0]]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=("Price + S/R + Trades", "Equity Curve (points)"),
    )

    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"], close=df_plot["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="NIFTY",
    ), row=1, col=1)

    # HTF zones as horizontal bands
    if htf_df is not None and len(htf_df) > 0:
        for _, z in htf_df.iterrows():
            color = "rgba(38,166,154,0.08)" if z["type"] == "Support" else "rgba(239,83,80,0.08)"
            edge  = "rgba(38,166,154,0.5)"  if z["type"] == "Support" else "rgba(239,83,80,0.5)"
            fig.add_hrect(y0=z["lower"], y1=z["upper"], fillcolor=color,
                          line_width=0, row=1, col=1)
            fig.add_hline(y=z["price"], line_color=edge, line_width=1,
                          line_dash="dot", row=1, col=1)

    # Trade markers
    for t in trades:
        if t.entry_ts.date() < last_sess[0]:
            continue
        emoji_e = "▲" if t.direction == "LONG" else "▼"
        emoji_x = "✓" if t.pnl_pts > 0 else "✗"
        col_e = "#42a5f5" if t.direction == "LONG" else "#ab47bc"
        col_x = "#26a69a" if t.pnl_pts > 0 else "#ef5350"
        fig.add_trace(go.Scatter(
            x=[t.entry_ts], y=[t.entry], mode="markers+text",
            marker=dict(symbol="triangle-up" if t.direction == "LONG" else "triangle-down",
                        size=14, color=col_e, line=dict(width=1, color="white")),
            text=[emoji_e], showlegend=False, hovertext=[
                f"{t.direction} entry<br>pattern={t.pattern}<br>zone={t.zone_price}<br>"
                f"src={t.zone_source} str={t.zone_strength:.0f}<br>"
                f"init_SL={t.sl:.1f}  trail_active={t.trail_active}"
            ],
        ), row=1, col=1)
        if t.exit_ts is not None:
            fig.add_trace(go.Scatter(
                x=[t.exit_ts], y=[t.exit_px], mode="markers",
                marker=dict(symbol="x", size=11, color=col_x,
                            line=dict(width=1, color="white")),
                showlegend=False, hovertext=[
                    f"exit={t.exit_reason}<br>pnl={t.pnl_pts:+.1f}<br>"
                    f"bars={t.bars_held}<br>conf={t.confluences}"
                ],
            ), row=1, col=1)
            fig.add_shape(type="line", x0=t.entry_ts, x1=t.exit_ts,
                          y0=t.entry, y1=t.exit_px, line=dict(color=col_x, width=1.5, dash="dot"),
                          row=1, col=1)

    # equity curve
    fig.add_trace(go.Scatter(
        x=eq_df.index, y=eq_df["cum_pnl"],
        mode="lines", line=dict(color="#ffd54f", width=1.5),
        name="Cum PnL (pts)",
    ), row=2, col=1)

    title_bits = [
        f"<b>{TICKER}</b> · {DAYS}d · 5m · S/R + Pin-Bar",
        f"Trades: {stats.get('trades',0)}  "
        f"WR: {stats.get('win_rate',0):.1f}%  "
        f"Total: {stats.get('total_pts',0):+.1f} pts  "
        f"PF: {stats.get('profit_factor',0):.2f}",
    ]
    fig.update_layout(
        title="<br>".join(title_bits),
        xaxis_rangeslider_visible=False,
        template="plotly_dark", height=900, showlegend=False,
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        margin=dict(l=60, r=30, t=80, b=40),
    )
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[15.5, 9.25], pattern="hour"),
    ])
    fig.write_html(out_path)
    print(f"  Report  : {out_path}")


# ══════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 70)
    print(f"  S/R + PIN-BAR STRATEGY  ·  {TICKER}  ·  {DAYS}d / {INTERVAL}")
    print("═" * 70 + "\n")

    df = fetch_intraday_chunked(ticker=TICKER, interval=INTERVAL,
                                days=DAYS, chunk_days=7)

    print("Building HTF precision zones (60d multi-session anchors)...")
    htf_df = build_precision_zones(
        df,
        left_bars=HTF_LEFT_BARS, right_bars=HTF_RIGHT_BARS,
        cluster_tolerance=HTF_CLUSTER_TOL,
        zone_half_band=HTF_ZONE_HALF_BAND,
        min_wick_touches=HTF_MIN_TOUCHES,
        min_sessions=HTF_MIN_SESSIONS,
        min_rejections=HTF_MIN_REJECTIONS,
        top_n=HTF_TOP_N,
    )
    print(f"  -> {len(htf_df)} HTF zones\n")

    print("Running 60-day bar-by-bar backtest...")
    trades, eq_df, signals, _ = run_backtest(df, htf_df)
    print(f"  -> {len(signals)} signals  /  {len(trades)} executed trades\n")

    tr_df = trades_to_df(trades)
    stats = compute_stats(tr_df)
    print_stats(stats)

    # outputs
    tr_path  = "nifty_sr_trail_trades.csv"
    eq_path  = "nifty_sr_trail_equity.csv"
    rep_path = "nifty_sr_trail_report.html"
    if not tr_df.empty:
        tr_df.to_csv(tr_path, index=False)
        print(f"  Trades  : {tr_path}")
    eq_df.to_csv(eq_path)
    print(f"  Equity  : {eq_path}")

    build_html_report(df, htf_df, trades, eq_df, signals, stats, rep_path)
    print()


if __name__ == "__main__":
    main()
