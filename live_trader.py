"""
live_trader.py — Speed Demon v6 Real-Time Trading Engine
=========================================================
Mirrors back22.py (Speed Demon v6) strategy logic on live NIFTY tick data.

Architecture:
  nifty_websocker.py (port 8086)
        ↓ tick feed (ws)
  live_trader.py
        ├── FiveMinCandleBuilder   — ticks → 5m OHLC bars
        ├── IndicatorEngine        — EMA9/21/50, HTF 15m bias, ATR, ADX
        ├── build_sr_zones (back22) — precision-pivot S/R zones, rebuilt daily
        │                             from a yfinance seed + live bars (zero-lookahead)
        ├── SRContext + gates      — 5 gates from back22.py
        ├── TradeManager           — entry, SL trail, BE, giveback lock, exits
        └── console + live_trades.csv

Start the tick feed first:
    cd zone-identifier && uvicorn nifty_websocker:app --port 8086

Then run the live trader:
    python live_trader.py

Signal mode only — no broker API wired. Add your broker's place_order() call
in the do_enter() / do_exit() hooks at the bottom of TradeManager.
"""

import asyncio
import csv
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import websockets
import numpy as np
import pandas as pd

# ── zone-identifier path ──────────────────────────────────────────
_ZI = Path(__file__).parent / "zone-identifier"
if str(_ZI) not in sys.path:
    sys.path.insert(0, str(_ZI))

# Precision S/R zones use the SAME engine as the backtest (back22.py), which is
# itself the zero-lookahead port of zone-identifier/python_visual.py. Importing
# build_sr_zones from back22 keeps live + backtest zones bit-for-bit identical —
# there is no second implementation to drift. fetch_yfinance seeds the prior-day
# history the pivot detector needs.  (back22's main() is __main__-guarded, so the
# import has no side effects beyond defining functions/constants.)
from back22 import build_sr_zones, fetch_yfinance  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(_ZI / ".env")
except ImportError:
    pass


# ════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION  (mirrors back22.py — edit here to tune)
# ════════════════════════════════════════════════════════════════

# ── WebSocket (tick feed) ────────────────────────────────────────
WS_HOST         = os.getenv("WS_HOST", "localhost")
WS_PORT         = int(os.getenv("WS_PORT", "8086"))
WS_PATH         = os.getenv("WS_PATH", "/ws")
WS_URL          = f"ws://{WS_HOST}:{WS_PORT}{WS_PATH}"
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "3"))

# ── EMA / trend ──────────────────────────────────────────────────
EMA_FAST        = 9
EMA_SLOW        = 21
EMA_MACRO       = 50
HTF_EMA_FAST    = 12
HTF_EMA_SLOW    = 26
HTF_EMA_TREND   = 50
ATR_PERIOD      = 14
ADX_PERIOD      = 14
ADX_MIN         = 18

LONG_ATR_SL_MULT    = 1.2
SHORT_ATR_SL_MULT   = 0.75
LONG_BE_PCT         = 0.2     # % above entry that triggers break-even for longs
SHORT_BE_PCT        = 0.3     # % below entry that triggers break-even for shorts

SWING_LOOKBACK      = 5
SWING_BUFFER_PTS    = 5

ENABLE_EMA_CROSS_EXIT = True
ENABLE_SLOPE_EXIT     = True
SLOPE_EXIT_CANDLES    = 4

# No-progress give-up (dead trades that never move) ───────────────
ENABLE_NOPROGRESS_EXIT = True
NOPROGRESS_BARS        = 12    # bars elapsed with no real progress
NOPROGRESS_MIN_FAVOR   = 10.0  # if max favourable < this → exit

# Peak give-back lock ─────────────────────────────────────────────
ENABLE_GIVEBACK_LOCK   = True
GIVEBACK_ARM_PTS       = 40.0  # arm once peak favourable ≥ this
GIVEBACK_KEEP_FRAC     = 0.5   # keep ≥ 50% of peak

SHORT_CONFIRM_BARS  = 3
SPREAD_PCT_MIN      = 0.04
SLOPE_CANDLES       = 6
SLOPE_MIN           = 0.0
PRICE_GAP_MIN       = 5.0
MIDDAY_SPREAD_MULT  = 2.0
RETEST_ATR_MULT     = 0.25
EMA9_PULLAWAY_PTS   = 30

MAX_TRADES_PER_DAY  = 3
SKIP_WEEKDAYS       = {1}   # Tuesday (weekly expiry) — {} to trade every day
DAILY_LOSS_LIMIT    = 1     # stop new entries after this many losses in a day

ENABLE_LONG   = True
ENABLE_SHORT  = True
ENABLE_PATH_A = True
ENABLE_PATH_B = True
ENABLE_MIDDAY = True
ENABLE_EURO   = False

OBSERVE_START    = dtime(9,  15)
OBSERVE_END      = dtime(9,  30)
PRIME_START      = dtime(9,  30)
PRIME_END        = dtime(10, 30)
MIDDAY_START     = dtime(11, 30)
MIDDAY_END       = dtime(13, 30)
EURO_START       = dtime(14, 15)
EURO_END         = dtime(15,  0)
SQUAREOFF_START  = dtime(15,  0)
EOD_HARD_EXIT    = dtime(15,  0)

# ── S/R Gates ────────────────────────────────────────────────────
SR_ENTRY_BUFFER_PTS  = 7.0
SR_MIN_SPACE_PTS     = 40.0
SR_ALIGN_MAX_DIST    = 40.0
SR_REQUIRE_ALIGNMENT = False
SR_RR_MIN            = 1.3
SR_RR_MAX            = 0       # 0 = disabled
SR_FALLBACK_RR       = 3.0     # ATR-multiple TP when no opposing zone exists
SR_EXIT_BUFFER_PTS   = 12.0
SR_MIN_ZONE_STRENGTH = 0.0

# ── Precision S/R zones (python_visual.py / back22.py method) ─────
# Pivot detection → absolute-point clustering → touch/rejection scoring.
# These MUST match back22.py so live zones == backtest zones.
SR_LOOKBACK_DAYS     = int(os.getenv("SR_LOOKBACK_DAYS",    "60"))   # prior days per build
SR_MIN_HISTORY_DAYS  = int(os.getenv("SR_MIN_HISTORY_DAYS", "10"))   # min history before zones exist
SR_LEFT_BARS         = 10
SR_RIGHT_BARS        = 10
SR_CLUSTER_TOLERANCE = 15.0    # absolute points
SR_ZONE_HALF_BAND    = 15.0    # zone = level ± 15 pts → 30-pt band
SR_MIN_WICK_TOUCHES  = 3
SR_MIN_SESSIONS      = 2
SR_MIN_REJECTIONS    = 1
SR_TOP_N             = int(os.getenv("TOP_N", "20"))

# Seed history (prior-day 5m bars the pivot detector needs on startup)
SR_HIST_SYMBOL       = os.getenv("SR_HIST_SYMBOL", "^NSEI")
SR_HIST_DAYS         = int(os.getenv("SR_HIST_DAYS", "60"))

# ── Output ───────────────────────────────────────────────────────
TRADE_LOG_CSV    = "live_trades.csv"
DASH_PORT_TRADER = int(os.getenv("DASH_PORT_TRADER", "8089"))
CHART_CANDLES    = 120   # candles streamed per dashboard payload

# ── Indicator warm-up ────────────────────────────────────────────
MIN_BARS = max(EMA_MACRO, ADX_PERIOD * 2, HTF_EMA_TREND * 3) + 10  # ~160 bars
MAX_BARS = 400   # rolling buffer cap


# ════════════════════════════════════════════════════════════════
# 2.  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class Zone:
    """A precision S/R zone (mirrors a build_sr_zones() dict)."""
    price:    float
    lower:    float
    upper:    float
    type:     str           # "Support" / "Resistance"
    strength: float
    n_pivots: int = 0       # wick-touch count (display only)


@dataclass
class Bar:
    ts:    datetime
    open:  float
    high:  float
    low:   float
    close: float
    ticks: int = 0


@dataclass
class Indicators:
    ema_fast:       float
    ema_slow:       float
    ema_macro:      float
    atr:            float
    adx:            float
    slope:          float   # EMA21.diff(SLOPE_CANDLES)
    slope_exit:     float   # EMA21.diff(SLOPE_EXIT_CANDLES)
    consec_bearish: int     # consecutive bars EMA9 < EMA21
    htf_long_bias:  bool
    htf_short_bias: bool


@dataclass
class TradeRecord:
    direction:    str
    entry_path:   str
    entry_time:   datetime
    entry_price:  float
    sl_at_entry:  float
    exit_time:    Optional[datetime] = None
    exit_price:   Optional[float]    = None
    sr_tp:        Optional[float]    = None
    pnl:          Optional[float]    = None
    exit_reason:  str   = ''
    mfe_pts:      float = 0.0
    mae_pts:      float = 0.0
    be_triggered: bool  = False
    sr_notes:     str   = ''


# ════════════════════════════════════════════════════════════════
# 3.  5-MIN CANDLE BUILDER
# ════════════════════════════════════════════════════════════════

class FiveMinCandleBuilder:
    def __init__(self):
        self._slot  = None
        self._open = self._high = self._low = self._close = None
        self._ticks = 0

    @staticmethod
    def _floor5(ts: datetime) -> datetime:
        return ts.replace(second=0, microsecond=0, minute=(ts.minute // 5) * 5)

    def on_tick(self, price: float, ts: datetime) -> Optional[Bar]:
        """Feed a tick. Returns a closed Bar when the 5m slot changes, else None."""
        slot = self._floor5(ts)
        if self._slot is None:
            self._slot = slot
            self._open = self._high = self._low = self._close = price
            self._ticks = 1
            return None
        if slot == self._slot:
            if price > self._high: self._high = price
            if price < self._low:  self._low  = price
            self._close = price
            self._ticks += 1
            return None
        closed = Bar(self._slot, self._open, self._high, self._low, self._close, self._ticks)
        self._slot  = slot
        self._open = self._high = self._low = self._close = price
        self._ticks = 1
        return closed

    @property
    def current(self) -> Optional[Bar]:
        if self._slot is None:
            return None
        return Bar(self._slot, self._open, self._high, self._low, self._close, self._ticks)


# ════════════════════════════════════════════════════════════════
# 4.  INDICATOR ENGINE  (pandas on rolling candle buffer)
# ════════════════════════════════════════════════════════════════

def compute_indicators(bars: List[Bar]) -> Optional[Indicators]:
    """Compute all strategy indicators from a list of closed 5m bars.
    Returns None if the buffer is too short for reliable values.
    """
    if len(bars) < MIN_BARS:
        return None

    df = pd.DataFrame([
        {"ts": b.ts, "open": b.open, "high": b.high, "low": b.low, "close": b.close}
        for b in bars
    ]).set_index("ts")

    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    # 5m EMAs
    ef  = c.ewm(span=EMA_FAST,  adjust=False).mean()
    es  = c.ewm(span=EMA_SLOW,  adjust=False).mean()
    em  = c.ewm(span=EMA_MACRO, adjust=False).mean()

    # ATR (Wilder's EMA of true range)
    pc  = c.shift(1)
    tr  = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # ADX
    up  = h.diff(); dn = -(l.diff())
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pds = pd.Series(pdm, index=df.index).ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
    mds = pd.Series(mdm, index=df.index).ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
    pdi = 100 * pds / atr.replace(0, np.nan)
    mdi = 100 * mds / atr.replace(0, np.nan)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()

    # EMA slopes
    slope      = es.diff(SLOPE_CANDLES)
    slope_exit = es.diff(SLOPE_EXIT_CANDLES)

    # Consecutive bearish bars (EMA9 < EMA21)
    bc   = (ef < es).astype(int)
    grp  = (bc != bc.shift()).cumsum()
    cbars = (bc.groupby(grp).cumcount() + 1) * bc

    # HTF 15m bias: resample closed 5m bars → 15m, compute EMAs
    htf_long = htf_short = False
    df15 = df[["close"]].resample("15min").last().dropna()
    if len(df15) >= HTF_EMA_TREND:
        hf = df15["close"].ewm(span=HTF_EMA_FAST,  adjust=False).mean()
        hs = df15["close"].ewm(span=HTF_EMA_SLOW,  adjust=False).mean()
        ht = df15["close"].ewm(span=HTF_EMA_TREND, adjust=False).mean()
        htf_long  = bool(hf.iloc[-1] > hs.iloc[-1] and hs.iloc[-1] > ht.iloc[-1])
        htf_short = bool(hf.iloc[-1] < hs.iloc[-1] and hs.iloc[-1] < ht.iloc[-1])

    i = -1  # last bar
    return Indicators(
        ema_fast       = float(ef.iloc[i]),
        ema_slow       = float(es.iloc[i]),
        ema_macro      = float(em.iloc[i]),
        atr            = float(atr.iloc[i]),
        adx            = float(adx.iloc[i]) if not np.isnan(adx.iloc[i]) else 0.0,
        slope          = float(slope.iloc[i]) if not np.isnan(slope.iloc[i]) else 0.0,
        slope_exit     = float(slope_exit.iloc[i]) if not np.isnan(slope_exit.iloc[i]) else 0.0,
        consec_bearish = int(cbars.iloc[i]),
        htf_long_bias  = htf_long,
        htf_short_bias = htf_short,
    )


# ════════════════════════════════════════════════════════════════
# 4b. PRECISION S/R ZONE BUILDER  (shared with back22.py / python_visual)
# ════════════════════════════════════════════════════════════════
#
# back22.build_sr_zones() does pivot-detection → point-clustering →
# touch/rejection scoring on a window of 5m bars. It needs weeks of prior
# history, so on startup we seed a yfinance pull, then fold each closed live
# bar in. Zones for day T are built from bars STRICTLY BEFORE T (zero
# lookahead) and frozen for the day — exactly back22's per-day behaviour.

_ZONE_KWARGS = dict(
    half_band   = SR_ZONE_HALF_BAND,
    cluster_tol = SR_CLUSTER_TOLERANCE,
    min_wt      = SR_MIN_WICK_TOUCHES,
    min_sess    = SR_MIN_SESSIONS,
    min_br      = SR_MIN_REJECTIONS,
    top_n       = SR_TOP_N,
    left_bars   = SR_LEFT_BARS,
    right_bars  = SR_RIGHT_BARS,
)


def seed_history() -> pd.DataFrame:
    """Load prior-day 5m bars for pivot detection. Returns a tz-naive frame with
    columns timestamp/open/high/low/close (back22.fetch_yfinance output, with the
    tz stripped so it compares cleanly with naive live-tick timestamps)."""
    import back22
    back22.YFINANCE_SYMBOL = SR_HIST_SYMBOL
    back22.YFINANCE_DAYS   = SR_HIST_DAYS
    df = fetch_yfinance('5m', 'SR-seed')[['timestamp', 'open', 'high', 'low', 'close']].copy()
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df


def combined_history(hist_df: pd.DataFrame, live_rows: List[dict]) -> pd.DataFrame:
    """Merge seed history with live closed bars; live bars win on a shared 5m slot."""
    if not live_rows:
        return hist_df
    merged = pd.concat([hist_df, pd.DataFrame(live_rows)], ignore_index=True)
    return (merged.drop_duplicates(subset='timestamp', keep='last')
                  .sort_values('timestamp')
                  .reset_index(drop=True))


def build_zones_asof(hist_df: pd.DataFrame, live_rows: List[dict],
                     today, current_price: float) -> List[Zone]:
    """Build today's precision zones from bars strictly before `today` (a date).
    Returns [] until SR_MIN_HISTORY_DAYS of history exist."""
    df = combined_history(hist_df, live_rows)
    dates = df['timestamp'].dt.date
    prior = df[dates < today]
    if prior.empty:
        return []
    prior_days = sorted(prior['timestamp'].dt.date.unique())
    if len(prior_days) < SR_MIN_HISTORY_DAYS:
        return []
    window = set(prior_days[-SR_LOOKBACK_DAYS:])
    sl = prior[prior['timestamp'].dt.date.isin(window)]
    hist_idx = sl.set_index('timestamp')[['open', 'high', 'low', 'close']]
    zdicts = build_sr_zones(hist_idx, **_ZONE_KWARGS)
    return [_dict_to_zone(z, current_price) for z in zdicts]


def _dict_to_zone(z: dict, current_price: float) -> Zone:
    """build_sr_zones() dict → Zone. Re-label Support/Resistance against the live
    price (gates use numeric prices, not the label, so this is display-only)."""
    return Zone(
        price    = z["price"],
        lower    = z["lower"],
        upper    = z["upper"],
        type     = "Support" if z["price"] < current_price else "Resistance",
        strength = z["strength"],
        n_pivots = z.get("wt", 0),
    )


def seed_indicator_bars(hist_df: pd.DataFrame, bars: deque,
                        bar_indicators: deque) -> int:
    """Pre-fill the rolling bar buffer (+ chart EMA snapshots) from the same
    yfinance seed used for zones, so EMA/ATR/ADX/HTF are valid from the FIRST
    live bar — no ~160-bar live warm-up. Bars in the current still-forming 5m
    slot are dropped so they don't collide with the first live close. Returns
    the number of bars seeded (capped at MAX_BARS, newest kept)."""
    if hist_df is None or hist_df.empty:
        return 0
    cur_slot = FiveMinCandleBuilder._floor5(datetime.now())
    df = (hist_df[hist_df['timestamp'] < cur_slot]
          .tail(MAX_BARS).reset_index(drop=True))
    if df.empty:
        return 0
    c  = df['close'].astype(float)
    ef = c.ewm(span=EMA_FAST,  adjust=False).mean()
    es = c.ewm(span=EMA_SLOW,  adjust=False).mean()
    em = c.ewm(span=EMA_MACRO, adjust=False).mean()
    for i, row in df.iterrows():
        bars.append(Bar(
            row['timestamp'].to_pydatetime(),
            float(row['open']), float(row['high']),
            float(row['low']),  float(row['close']),
        ))
        bar_indicators.append({
            "ef": float(ef.iloc[i]), "es": float(es.iloc[i]), "em": float(em.iloc[i]),
        })
    return len(df)


# ════════════════════════════════════════════════════════════════
# 5.  S/R GATE ADAPTER  (adapts Zone → back22 5-gate logic)
# ════════════════════════════════════════════════════════════════

def _z(z: Zone) -> dict:
    return {
        "price":    z.price,
        "upper":    z.upper,
        "lower":    z.lower,
        "type":     z.type,
        "strength": z.strength,
        "n_pivots": z.n_pivots,
    }


class SRContext:
    def __init__(self, zones: List[Zone], min_strength: float = 0.0):
        self.zones = [_z(z) for z in zones if z.strength >= min_strength]

    def price_in_zone(self, price: float, buffer: float = 0.0):
        for z in self.zones:
            if z["lower"] - buffer <= price <= z["upper"] + buffer:
                return True, z
        return False, None

    def nearest_opposing(self, price: float, direction: str):
        candidates = [
            (z, z["price"] - price) for z in self.zones if direction == "up"   and z["price"] > price
        ] + [
            (z, price - z["price"]) for z in self.zones if direction == "down" and z["price"] < price
        ]
        if not candidates:
            return None, float("inf")
        return min(candidates, key=lambda x: x[1])

    def nearest_supportive(self, price: float, direction: str):
        candidates = [
            (z, price - z["price"]) for z in self.zones if direction == "up"   and z["price"] < price
        ] + [
            (z, z["price"] - price) for z in self.zones if direction == "down" and z["price"] > price
        ]
        if not candidates:
            return None, float("inf")
        return min(candidates, key=lambda x: x[1])

    def sr_tp(self, price: float, direction: str) -> Optional[float]:
        opp, _ = self.nearest_opposing(price, direction)
        if opp is None:
            return None
        return opp["lower"] if direction == "up" else opp["upper"]

    def approaching_zone(self, price: float, direction: str, buffer: float) -> bool:
        tp = self.sr_tp(price, direction)
        return tp is not None and abs(price - tp) <= buffer


def sr_gate_check(
    price: float, direction: str, sr_ctx: SRContext, atr_sl_dist: float
) -> Tuple[bool, Optional[float], float, Optional[str], str]:
    """Run Gates 1–4. Returns (allowed, tp_price, rr, gate_failed, notes)."""
    # Gate 1: not inside any zone
    in_zone, wz = sr_ctx.price_in_zone(price, SR_ENTRY_BUFFER_PTS)
    if in_zone:
        return False, None, 0.0, "G1_IN_ZONE", f"In/near zone@{wz['price']:.0f}"

    # Gate 2: minimum free space to opposing zone
    opp, opp_dist = sr_ctx.nearest_opposing(price, direction)
    if opp_dist < SR_MIN_SPACE_PTS:
        return False, None, 0.0, "G2_NO_SPACE", f"Opp zone {opp_dist:.0f}pts away"

    # Gate 3: alignment (optional)
    sup, sup_dist = sr_ctx.nearest_supportive(price, direction)
    aligned = sup is not None and sup_dist <= SR_ALIGN_MAX_DIST
    if SR_REQUIRE_ALIGNMENT and not aligned:
        return False, None, 0.0, "G3_NO_ALIGN", "No supportive zone"

    # Gate 4: R:R check
    tp = sr_ctx.sr_tp(price, direction)
    fallback = tp is None
    if fallback:
        tp = (price + atr_sl_dist * SR_FALLBACK_RR if direction == "up"
              else price - atr_sl_dist * SR_FALLBACK_RR)
    reward = abs(tp - price)
    rr = reward / atr_sl_dist if atr_sl_dist > 0 else 0.0
    if rr < SR_RR_MIN:
        return False, tp, rr, "G4_LOW_RR", f"R:R={rr:.2f} < {SR_RR_MIN}"
    if (not fallback) and SR_RR_MAX and rr > SR_RR_MAX:
        return False, tp, rr, "G4_HIGH_RR", f"R:R={rr:.2f} > {SR_RR_MAX}"

    space_tag = "ATR_RUNWAY" if fallback else f"{opp_dist:.0f}pts"
    sup_tag   = f"SUP@{sup['price']:.0f}" if (aligned and sup) else "no_align"
    return True, tp, rr, None, f"TP={tp:.0f} RR={rr:.2f} space={space_tag} {sup_tag}"


# ════════════════════════════════════════════════════════════════
# 6.  SESSION & FILTER HELPERS
# ════════════════════════════════════════════════════════════════

def get_session(t: dtime) -> str:
    if OBSERVE_START <= t < OBSERVE_END:   return "observe"
    if PRIME_START   <= t < PRIME_END:     return "prime"
    if MIDDAY_START  <= t < MIDDAY_END:    return "midday"
    if EURO_START    <= t < EURO_END:      return "euro"
    if SQUAREOFF_START <= t:               return "squareoff"
    return "outside"


def chop_filters_pass(close, ef, es, slope, adx_v, session) -> bool:
    if adx_v < ADX_MIN:
        return False
    spread = abs(ef - es) / close * 100
    thresh = SPREAD_PCT_MIN * (MIDDAY_SPREAD_MULT if session == "midday" else 1.0)
    if spread < thresh:
        return False
    if abs(slope) <= SLOPE_MIN:
        return False
    if abs(close - es) < PRICE_GAP_MIN:
        return False
    return True


def swing_low(bars: List[Bar], lookback: int) -> float:
    w = bars[-lookback:] if len(bars) >= lookback else bars
    return float(min(b.low for b in w)) if w else float("inf")


def swing_high(bars: List[Bar], lookback: int) -> float:
    w = bars[-lookback:] if len(bars) >= lookback else bars
    return float(max(b.high for b in w)) if w else 0.0


# ════════════════════════════════════════════════════════════════
# 7.  TRADE MANAGER
# ════════════════════════════════════════════════════════════════

class TradeManager:
    """Manages a single open position plus the closed-trade log."""

    def __init__(self):
        self.in_trade:       bool            = False
        self.direction:      Optional[str]   = None
        self.entry_price:    float           = 0.0
        self.entry_time:     Optional[datetime] = None
        self.entry_path:     str             = ""
        self.sr_tp_price:    Optional[float] = None
        self.sr_notes:       str             = ""
        self.stop_loss:      float           = 0.0
        self.sl_dist_initial:float           = 0.0
        self.be_triggered:   bool            = False
        self.be_level:       float           = 0.0
        self.trail_active:   bool            = False
        self.trade_max_favor:float           = 0.0
        self.trade_max_adv:  float           = 0.0
        self.entry_bar_idx:  int             = 0
        self.bar_count:      int             = 0  # incremented by the main loop

        self.closed_trades: List[TradeRecord] = []

    # ── broker hook ───────────────────────────────────────────────
    def _place_order(self, direction: str, price: float, sl: float, tp: Optional[float]):
        """Override this to send orders to a live broker (e.g. Zerodha/Fyers API)."""
        pass   # signal mode — no live orders

    def _close_order(self, reason: str, exit_price: float):
        """Override this to send exit orders to a live broker."""
        pass   # signal mode

    # ── enter ─────────────────────────────────────────────────────
    def enter(self, direction: str, price: float, ts: datetime,
              atr: float, path: str, notes: str, tp: Optional[float],
              bar_idx: int):
        sl_mult          = LONG_ATR_SL_MULT if direction == "long" else SHORT_ATR_SL_MULT
        sl_dist          = atr * sl_mult
        self.in_trade    = True
        self.direction   = direction
        self.entry_price = price
        self.entry_time  = ts
        self.entry_path  = path
        self.sr_tp_price = tp
        self.sr_notes    = notes
        self.sl_dist_initial = sl_dist
        self.be_triggered    = False
        self.trail_active    = False
        self.trade_max_favor = 0.0
        self.trade_max_adv   = 0.0
        self.entry_bar_idx   = bar_idx

        if direction == "long":
            self.stop_loss = price - sl_dist
            self.be_level  = price * (1 + LONG_BE_PCT / 100)
        else:
            self.stop_loss = price + sl_dist
            self.be_level  = price * (1 - SHORT_BE_PCT / 100)

        self._place_order(direction, price, self.stop_loss, tp)

    # ── exit ──────────────────────────────────────────────────────
    def exit(self, exit_price: float, ts: datetime, reason: str,
             daily_loss_counts: Dict[str, int]) -> TradeRecord:
        pnl = round(
            (exit_price - self.entry_price) if self.direction == "long"
            else (self.entry_price - exit_price), 2
        )
        date_str = str(ts.date())
        if pnl <= 0:
            daily_loss_counts[date_str] = daily_loss_counts.get(date_str, 0) + 1

        rec = TradeRecord(
            direction    = self.direction,
            entry_path   = self.entry_path,
            entry_time   = self.entry_time,
            entry_price  = self.entry_price,
            sl_at_entry  = (self.entry_price - self.sl_dist_initial
                            if self.direction == "long"
                            else self.entry_price + self.sl_dist_initial),
            exit_time    = ts,
            exit_price   = exit_price,
            sr_tp        = self.sr_tp_price,
            pnl          = pnl,
            exit_reason  = reason,
            mfe_pts      = round(self.trade_max_favor, 2),
            mae_pts      = round(self.trade_max_adv,   2),
            be_triggered = self.be_triggered,
            sr_notes     = self.sr_notes,
        )
        self.closed_trades.append(rec)
        self._close_order(reason, exit_price)
        self._reset()
        return rec

    def _reset(self):
        self.in_trade = False; self.direction = None
        self.entry_price = self.stop_loss = self.be_level = 0.0
        self.sl_dist_initial = self.trade_max_favor = self.trade_max_adv = 0.0
        self.be_triggered = self.trail_active = False
        self.entry_time = self.sr_tp_price = None
        self.entry_path = self.sr_notes = ""

    # ── tick-level SL monitoring ──────────────────────────────────
    def check_tick_sl(self, price: float, ts: datetime) -> Optional[Tuple[float, str]]:
        """Fast path: only checks stop-loss hit. Call on every tick."""
        if not self.in_trade:
            return None
        if ts.time() >= EOD_HARD_EXIT:
            return price, "EOD_EXIT"
        if self.direction == "long"  and price <= self.stop_loss:
            return self.stop_loss, "TRAIL_SL" if self.trail_active else "STOP_LOSS"
        if self.direction == "short" and price >= self.stop_loss:
            return self.stop_loss, "TRAIL_SL" if self.trail_active else "STOP_LOSS"
        return None

    # ── bar-close management ──────────────────────────────────────
    def update_on_bar(self, bar: Bar, ind: Indicators,
                      bars: List[Bar], sr_ctx: SRContext) -> Optional[Tuple[float, str]]:
        """
        Called on every closed bar while a trade is open.
        Updates MFE/MAE, break-even, swing trail, giveback lock,
        then checks all exits in priority order.
        Returns (exit_price, reason) if an exit is triggered, else None.
        """
        if not self.in_trade:
            return None

        # MFE / MAE
        prev_peak = self.trade_max_favor
        fav = (bar.high - self.entry_price) if self.direction == "long" else (self.entry_price - bar.low)
        adv = (self.entry_price - bar.low)  if self.direction == "long" else (bar.high - self.entry_price)
        self.trade_max_favor = max(self.trade_max_favor, fav)
        self.trade_max_adv   = max(self.trade_max_adv,   adv)

        # Break-even
        if not self.be_triggered:
            if   self.direction == "long"  and bar.close >= self.be_level:
                self.stop_loss    = self.entry_price + 1.0
                self.be_triggered = True
                self.trail_active = True
            elif self.direction == "short" and bar.close <= self.be_level:
                self.stop_loss    = self.entry_price - 1.0
                self.be_triggered = True
                self.trail_active = True

        # Swing structure trail (only after BE)
        if self.trail_active and len(bars) >= SWING_LOOKBACK:
            if self.direction == "long":
                self.stop_loss = max(self.stop_loss,
                                     swing_low(bars, SWING_LOOKBACK) - SWING_BUFFER_PTS)
            else:
                self.stop_loss = min(self.stop_loss,
                                     swing_high(bars, SWING_LOOKBACK) + SWING_BUFFER_PTS)

        # Giveback lock — ratchets stop to keep ≥ KEEP_FRAC of prior-bar peak
        if ENABLE_GIVEBACK_LOCK and prev_peak >= GIVEBACK_ARM_PTS:
            if self.direction == "long":
                self.stop_loss = max(self.stop_loss,
                                     self.entry_price + GIVEBACK_KEEP_FRAC * prev_peak)
            else:
                self.stop_loss = min(self.stop_loss,
                                     self.entry_price - GIVEBACK_KEEP_FRAC * prev_peak)

        # ── Exit checks (same priority as back22.py) ───────────────

        # Gate 5: S/R early exit (only after BE)
        if self.be_triggered:
            dir_ud = "up" if self.direction == "long" else "down"
            if sr_ctx.approaching_zone(bar.close, dir_ud, SR_EXIT_BUFFER_PTS):
                return bar.close, "SR_ZONE_EXIT"

        # No-progress give-up
        bars_elapsed = self.bar_count - self.entry_bar_idx
        if (ENABLE_NOPROGRESS_EXIT and not self.be_triggered and
                bars_elapsed >= NOPROGRESS_BARS and
                self.trade_max_favor < NOPROGRESS_MIN_FAVOR):
            return bar.close, "NO_PROGRESS"

        # Stop loss (bar close crosses SL)
        if self.direction == "long"  and bar.close <= self.stop_loss:
            return self.stop_loss, "TRAIL_SL" if self.trail_active else "STOP_LOSS"
        if self.direction == "short" and bar.close >= self.stop_loss:
            return self.stop_loss, "TRAIL_SL" if self.trail_active else "STOP_LOSS"

        # EMA cross exit (after BE)
        if ENABLE_EMA_CROSS_EXIT and self.be_triggered:
            if self.direction == "long"  and ind.ema_fast < ind.ema_slow:
                return bar.close, "EMA_CROSS_EXIT"
            if self.direction == "short" and ind.ema_fast > ind.ema_slow:
                return bar.close, "EMA_CROSS_EXIT"

        # Slope reversal exit (after BE)
        if ENABLE_SLOPE_EXIT and self.be_triggered:
            if self.direction == "long"  and ind.slope_exit < -SLOPE_MIN:
                return bar.close, "SLOPE_REV_EXIT"
            if self.direction == "short" and ind.slope_exit >  SLOPE_MIN:
                return bar.close, "SLOPE_REV_EXIT"

        # EOD hard exit
        if bar.ts.time() >= EOD_HARD_EXIT:
            return bar.close, "EOD_EXIT"

        return None


# ════════════════════════════════════════════════════════════════
# 8.  TRADE CSV LOG
# ════════════════════════════════════════════════════════════════

_CSV_FIELDS = [
    "direction", "entry_path", "entry_time", "exit_time",
    "entry_price", "exit_price", "sl_at_entry", "sr_tp",
    "pnl", "mfe_pts", "mae_pts", "be_triggered", "exit_reason", "sr_notes",
]


def write_trade_csv(rec: TradeRecord):
    path = Path(TRADE_LOG_CSV)
    new  = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "direction":   rec.direction,
            "entry_path":  rec.entry_path,
            "entry_time":  rec.entry_time.strftime("%Y-%m-%d %H:%M") if rec.entry_time else "",
            "exit_time":   rec.exit_time.strftime("%Y-%m-%d %H:%M")  if rec.exit_time  else "",
            "entry_price": f"{rec.entry_price:.2f}",
            "exit_price":  f"{rec.exit_price:.2f}"  if rec.exit_price  is not None else "",
            "sl_at_entry": f"{rec.sl_at_entry:.2f}",
            "sr_tp":       f"{rec.sr_tp:.0f}"        if rec.sr_tp       is not None else "",
            "pnl":         f"{rec.pnl:.2f}"           if rec.pnl         is not None else "",
            "mfe_pts":     f"{rec.mfe_pts:.2f}",
            "mae_pts":     f"{rec.mae_pts:.2f}",
            "be_triggered":rec.be_triggered,
            "exit_reason": rec.exit_reason,
            "sr_notes":    rec.sr_notes,
        })


# ════════════════════════════════════════════════════════════════
# 9.  DASHBOARD BROADCASTER
# ════════════════════════════════════════════════════════════════

class _DashboardBroadcaster:
    def __init__(self):
        self._clients: set = set()
        self._last_msg: Optional[str] = None

    def register(self, ws):
        self._clients.add(ws)
        if self._last_msg:
            asyncio.ensure_future(self._send_one(ws, self._last_msg))

    def unregister(self, ws):
        self._clients.discard(ws)

    async def _send_one(self, ws, msg: str):
        try:
            await ws.send(msg)
        except Exception:
            pass

    async def broadcast(self, payload: dict):
        msg = json.dumps(payload, default=str)
        self._last_msg = msg
        if self._clients:
            await asyncio.gather(
                *[self._send_one(ws, msg) for ws in list(self._clients)]
            )


_broadcaster = _DashboardBroadcaster()


async def _dash_handler(websocket):
    _broadcaster.register(websocket)
    try:
        await websocket.wait_closed()
    finally:
        _broadcaster.unregister(websocket)


def _build_payload(trigger: str, bars, bar_indicators, current_zones, ind,
                   trade_mgr, price: float, ts: datetime, equity: float,
                   daily_trades: dict, daily_loss_counts: dict,
                   gate_counts: dict, session: str, bar_idx: int) -> dict:
    date_str = str(ts.date())

    # Candle data for chart (last CHART_CANDLES bars with their EMA snapshots)
    bar_list = list(bars)[-CHART_CANDLES:]
    ind_list = list(bar_indicators)[-CHART_CANDLES:]
    candles  = []
    for i, b in enumerate(bar_list):
        bi = ind_list[i] if i < len(ind_list) else {}
        candles.append({
            "t":  int(b.ts.timestamp()),
            "tl": b.ts.strftime("%H:%M"),
            "o":  round(b.open, 2),  "h": round(b.high, 2),
            "l":  round(b.low, 2),   "c": round(b.close, 2),
            "ef": round(bi["ef"], 2) if bi.get("ef") is not None else None,
            "es": round(bi["es"], 2) if bi.get("es") is not None else None,
            "em": round(bi["em"], 2) if bi.get("em") is not None else None,
        })

    # Open trade state (includes live P&L vs current tick price)
    open_trade = None
    if trade_mgr.in_trade:
        cur_pnl    = round(
            (price - trade_mgr.entry_price) if trade_mgr.direction == "long"
            else (trade_mgr.entry_price - price), 2
        )
        sl_initial = round(
            trade_mgr.entry_price - trade_mgr.sl_dist_initial
            if trade_mgr.direction == "long"
            else trade_mgr.entry_price + trade_mgr.sl_dist_initial, 2
        )
        open_trade = {
            "direction":    trade_mgr.direction,
            "path":         trade_mgr.entry_path,
            "entry_price":  round(trade_mgr.entry_price, 2),
            "entry_time":   trade_mgr.entry_time.strftime("%H:%M"),
            "entry_unix":   int(trade_mgr.entry_time.timestamp()),
            "stop_loss":    round(trade_mgr.stop_loss, 2),
            "sl_initial":   sl_initial,
            "tp":           round(trade_mgr.sr_tp_price, 2) if trade_mgr.sr_tp_price else None,
            "be_triggered": trade_mgr.be_triggered,
            "mfe":          round(trade_mgr.trade_max_favor, 2),
            "mae":          round(trade_mgr.trade_max_adv, 2),
            "pnl_current":  cur_pnl,
            "sr_notes":     trade_mgr.sr_notes,
            "risk_pts":     round(trade_mgr.sl_dist_initial, 2),
        }

    # Closed trades (newest first, last 60)
    closed = []
    for t in reversed(list(trade_mgr.closed_trades)[-60:]):
        closed.append({
            "direction":   t.direction,
            "path":        t.entry_path,
            "entry_time":   t.entry_time.strftime("%H:%M %d/%m") if t.entry_time else "",
            "exit_time":    t.exit_time.strftime("%H:%M")         if t.exit_time  else "",
            "entry_unix":   int(t.entry_time.timestamp())          if t.entry_time else None,
            "exit_unix":    int(t.exit_time.timestamp())           if t.exit_time  else None,
            "entry_price":  round(t.entry_price, 2),
            "exit_price":   round(t.exit_price, 2)  if t.exit_price  is not None else None,
            "pnl":          round(t.pnl, 2)          if t.pnl         is not None else None,
            "mfe":          round(t.mfe_pts, 2),
            "mae":          round(t.mae_pts, 2),
            "exit_reason":  t.exit_reason,
            "be_triggered": t.be_triggered,
            "sr_tp":        round(t.sr_tp, 2)        if t.sr_tp       is not None else None,
        })

    # S/R zones
    zones = [{
        "price":    round(z.price, 2),
        "upper":    round(z.upper, 2),
        "lower":    round(z.lower, 2),
        "type":     z.type,
        "strength": round(z.strength, 1),
        "n_pivots": z.n_pivots,
    } for z in current_zones]

    all_tr = trade_mgr.closed_trades
    wins   = sum(1 for t in all_tr if t.pnl and t.pnl > 0)

    return {
        "trigger":       trigger,
        "ts":            ts.strftime("%H:%M:%S"),
        "ts_unix":       int(ts.timestamp()),
        "price":         round(price, 2),
        "session":       session,
        "bar_count":     bar_idx,
        "equity":        round(equity, 2),
        "wins":          wins,
        "losses":        len(all_tr) - wins,
        "daily_trades":  daily_trades.get(date_str, 0),
        "daily_losses":  daily_loss_counts.get(date_str, 0),
        "warming_up":    bar_idx < MIN_BARS,
        "bars_to_live":  max(0, MIN_BARS - bar_idx),
        "candles":       candles,
        "bar_indicators": ind_list,
        "zones":         zones,
        "open_trade":    open_trade,
        "closed_trades": closed,
        "indicators":    {
            "ema_fast":       round(ind.ema_fast, 2),
            "ema_slow":       round(ind.ema_slow, 2),
            "ema_macro":      round(ind.ema_macro, 2),
            "atr":            round(ind.atr, 2),
            "adx":            round(ind.adx, 2),
            "slope":          round(ind.slope, 4),
            "htf_long":       ind.htf_long_bias,
            "htf_short":      ind.htf_short_bias,
            "consec_bearish": ind.consec_bearish,
        } if ind else None,
        "gate_counts":   gate_counts,
    }


# ════════════════════════════════════════════════════════════════
# 10. TICK PARSER  (same multi-format parser as live_sr.py)
# ════════════════════════════════════════════════════════════════

def parse_tick(raw: str) -> Tuple[Optional[float], datetime]:
    now = datetime.now()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            return float(raw.strip()), now
        except ValueError:
            return None, now

    if isinstance(data, list):
        price  = float(data[0])
        ts_raw = data[1] if len(data) > 1 else None
    elif isinstance(data, dict):
        price = float(data.get("price") or data.get("ltp") or
                      data.get("last_price") or data.get("close") or 0)
        if price == 0:
            return None, now
        ts_raw = data.get("timestamp") or data.get("ts") or data.get("time")
    elif isinstance(data, (int, float)):
        return float(data), now
    else:
        return None, now

    if ts_raw is None:
        return price, now
    if isinstance(ts_raw, (int, float)):
        return price, datetime.fromtimestamp(ts_raw / 1000.0 if ts_raw > 1e12 else ts_raw)
    if isinstance(ts_raw, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return price, datetime.strptime(ts_raw, fmt)
            except ValueError:
                continue
    return price, now


# ════════════════════════════════════════════════════════════════
# 10. CONSOLE OUTPUT
# ════════════════════════════════════════════════════════════════

try:                                            # Windows ANSI support
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

R   = "\033[0m"
GRN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"
CYN = "\033[96m"; MAG = "\033[95m"; BLD = "\033[1m"; DIM = "\033[2m"
SEP = "═" * 80


def _htf_tag(ind: Indicators) -> str:
    if ind.htf_long_bias:  return f"{GRN}HTF_LONG{R}"
    if ind.htf_short_bias: return f"{RED}HTF_SHORT{R}"
    return f"{DIM}HTF_NEUTRAL{R}"


def log_bar(bar: Bar, ind: Optional[Indicators], n_zones: int, bar_idx: int):
    d   = "▲" if bar.close >= bar.open else "▼"
    col = GRN if bar.close >= bar.open else RED
    htf = _htf_tag(ind) if ind else f"{DIM}warming up ({bar_idx}/{MIN_BARS}){R}"
    print(f"  {col}{d} {bar.ts.strftime('%H:%M')}{R}  "
          f"O:{bar.open:.1f} H:{bar.high:.1f} L:{bar.low:.1f} C:{bar.close:.1f}  "
          f"zones:{n_zones}  {htf}")


def log_zones(zones: List[Zone], cmp: float):
    supp = sorted([z for z in zones if z.type == "Support"],    key=lambda z: -z.price)
    res  = sorted([z for z in zones if z.type == "Resistance"], key=lambda z:  z.price)
    print(f"\n  {'─'*62}")
    print(f"  S/R ZONES ({len(zones)})   CMP: {BLD}{cmp:.2f}{R}")
    print(f"  {'─'*62}")
    for z in res:
        print(f"  {RED}RES{R}  {z.price:>8.2f}  [{z.lower:.0f}–{z.upper:.0f}]"
              f"  str={z.strength:.1f}  pivots={z.n_pivots}")
    print(f"  {'─'*20} CMP {cmp:.2f} {'─'*20}")
    for z in supp:
        print(f"  {GRN}SUP{R}  {z.price:>8.2f}  [{z.lower:.0f}–{z.upper:.0f}]"
              f"  str={z.strength:.1f}  pivots={z.n_pivots}")
    print(f"  {'─'*62}\n")


def log_entry(direction: str, path: str, price: float, sl: float,
              tp: Optional[float], ts: datetime, notes: str):
    col   = GRN if direction == "long" else RED
    arrow = "▲ LONG " if direction == "long" else "▼ SHORT"
    tp_s  = f"{tp:.0f}" if tp else "N/A"
    print(f"\n  {BLD}{col}{'━'*62}{R}")
    print(f"  {BLD}{col}SIGNAL  {arrow}  Path-{path}{R}  "
          f"@ {price:.2f}   SL={sl:.2f}   TP={tp_s}   "
          f"{ts.strftime('%H:%M:%S')}")
    print(f"  {DIM}{notes}{R}")
    print(f"  {BLD}{col}{'━'*62}{R}\n")


def log_exit(rec: TradeRecord, equity: float):
    col  = GRN if rec.pnl and rec.pnl > 0 else RED
    sign = "+" if rec.pnl and rec.pnl > 0 else ""
    print(f"  {BLD}EXIT {rec.direction.upper()}{R}  "
          f"{rec.entry_time.strftime('%H:%M')}→{rec.exit_time.strftime('%H:%M')}  "
          f"entry={rec.entry_price:.2f} exit={rec.exit_price:.2f}  "
          f"{BLD}{col}P&L: {sign}{rec.pnl:.2f} pts{R}  "
          f"reason={rec.exit_reason}  "
          f"cumPnL={equity:+.2f}\n")


def log_block(gate: str, note: str, direction: str, price: float):
    print(f"  {DIM}✗ {direction.upper()} [{gate}]: {note}  @{price:.2f}{R}")


def print_daily_summary(trades: List[TradeRecord], equity: float,
                        gate_counts: Dict[str, int], date_str: str):
    day_trades = [t for t in trades if str(t.entry_time.date()) == date_str]
    print(f"\n  {'─'*62}")
    print(f"  EOD SUMMARY  {date_str}  trades={len(day_trades)}  cumPnL={equity:+.2f}")
    for t in day_trades:
        col  = GRN if t.pnl and t.pnl > 0 else RED
        sign = "+" if t.pnl and t.pnl > 0 else ""
        print(f"    {col}{t.direction.upper()} {t.entry_path}  "
              f"{t.entry_time.strftime('%H:%M')}→{t.exit_time.strftime('%H:%M')}  "
              f"{sign}{t.pnl:.2f}  {t.exit_reason}{R}")
    if gate_counts:
        blocked = ", ".join(f"{k}:{v}" for k, v in sorted(gate_counts.items()))
        print(f"  Gates blocked: {blocked}")
    print(f"  {'─'*62}\n")


# ════════════════════════════════════════════════════════════════
# 11. MAIN TRADING LOOP
# ════════════════════════════════════════════════════════════════

async def trading_loop():
    candle_builder  = FiveMinCandleBuilder()
    trade_mgr       = TradeManager()
    bars:           deque = deque(maxlen=MAX_BARS)  # closed 5m bars
    bar_indicators: deque = deque(maxlen=MAX_BARS)  # EMA snapshots per bar (for chart)
    current_zones:  List[Zone]          = []
    sr_ctx:         Optional[SRContext] = None
    tick_count:     int = 0

    # Precision S/R history: yfinance seed + live closed bars folded in.
    # Zones are rebuilt once per calendar day (zone_date) from bars strictly
    # before that day — frozen intraday, identical to back22's per-day zones.
    try:
        hist_df = seed_history()
        n_hist_days = hist_df['timestamp'].dt.date.nunique()
        print(f"  {GRN}Seeded S/R history{R}: {len(hist_df)} bars / "
              f"{n_hist_days} sessions from {SR_HIST_SYMBOL}")
    except Exception as e:
        print(f"  {RED}S/R seed failed ({e}) — zones disabled until history loads{R}")
        hist_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
        n_hist_days = 0
    live_rows:  List[dict]          = []   # closed live bars (folded into history)
    zone_date:  Optional[object]    = None  # date the current zones were built for

    # Daily state (reset each calendar day)
    daily_trades:      Dict[str, int]   = {}
    daily_loss_counts: Dict[str, int]   = {}
    opening_high:      Dict[str, float] = {}
    opening_close:     Dict[str, float] = {}
    gate_counts:       Dict[str, int]   = {}
    equity:            float = 0.0
    bar_idx:           int   = 0
    last_eod_date:     Optional[str] = None

    # ── Skip live warm-up: seed the indicator buffer from the yfinance history
    #    so EMA/ATR/ADX/HTF are valid from the first live bar (signals live now). ──
    n_seed = seed_indicator_bars(hist_df, bars, bar_indicators)
    if n_seed:
        bar_idx = len(bars)
        trade_mgr.bar_count = bar_idx
        print(f"  {GRN}Seeded indicator buffer{R}: {n_seed} bars "
              f"→ warm-up bypassed (signals active immediately)")
    else:
        print(f"  {YEL}No seed bars — falling back to {MIN_BARS}-bar live warm-up{R}")

    print(f"\n{SEP}")
    print(f"  {BLD}SPEED DEMON v6 — LIVE TRADING ENGINE{R}")
    print(f"  Tick feed  : {WS_URL}")
    print(f"  Trade log  : {TRADE_LOG_CSV}")
    print(f"  Warm-up    : " + (
        f"bypassed — {bar_idx} bars seeded from yfinance"
        if bar_idx >= MIN_BARS else
        f"{MIN_BARS} closed bars before signals activate"))
    print(f"  Sessions   : PRIME 09:30–10:30"
          f"{'  MIDDAY 11:30–13:30' if ENABLE_MIDDAY else ''}"
          f"{'  EURO 14:15–15:00' if ENABLE_EURO else ''}")
    print(f"  Zones      : precision-pivot (back22/python_visual)  "
          f"lookback={SR_LOOKBACK_DAYS}d  band=±{SR_ZONE_HALF_BAND:.0f}  top_n={SR_TOP_N}")
    print(SEP + "\n")

    import websockets

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print(f"  {GRN}Connected{R} → {WS_URL}\n")

                async for raw_msg in ws:
                    price, ts = parse_tick(str(raw_msg))
                    if price is None:
                        continue

                    c_time   = ts.time()
                    c_date   = ts.date()
                    date_str = str(c_date)
                    session  = get_session(c_time)

                    # ── CANDLE CLOSE ──────────────────────────────
                    closed = candle_builder.on_tick(price, ts)
                    if closed is not None:
                        bar_idx += 1
                        trade_mgr.bar_count = bar_idx
                        bars.append(closed)

                        # Fold this closed bar into the rolling history.
                        live_rows.append({
                            "timestamp": closed.ts, "open": closed.open,
                            "high": closed.high, "low": closed.low,
                            "close": closed.close,
                        })

                        # Rebuild precision zones once per calendar day, using only
                        # bars strictly before today (zero lookahead, == back22).
                        if c_date != zone_date:
                            zone_date     = c_date
                            current_zones = build_zones_asof(
                                hist_df, live_rows, c_date, price)
                            sr_ctx = SRContext(current_zones, SR_MIN_ZONE_STRENGTH)
                            log_zones(current_zones, price)

                        # Record opening bar values
                        if closed.ts.time() == dtime(9, 30):
                            opening_high[date_str]  = closed.high
                            opening_close[date_str] = closed.close

                        # Compute indicators
                        ind = compute_indicators(list(bars))
                        bar_indicators.append({
                            "ef": ind.ema_fast  if ind else None,
                            "es": ind.ema_slow  if ind else None,
                            "em": ind.ema_macro if ind else None,
                        })
                        log_bar(closed, ind, len(current_zones), bar_idx)

                        # EOD daily summary (trigger once per day)
                        if (closed.ts.time() >= EOD_HARD_EXIT and
                                date_str != last_eod_date):
                            last_eod_date = date_str
                            print_daily_summary(trade_mgr.closed_trades,
                                                equity, gate_counts, date_str)

                        # ── OPEN TRADE: bar-close management ────────
                        _bar_exited = False
                        if trade_mgr.in_trade and ind and sr_ctx:
                            result = trade_mgr.update_on_bar(
                                closed, ind, list(bars), sr_ctx)
                            if result:
                                exit_p, exit_r = result
                                rec = trade_mgr.exit(exit_p, ts, exit_r,
                                                     daily_loss_counts)
                                equity += rec.pnl
                                log_exit(rec, equity)
                                write_trade_csv(rec)
                                _bar_exited = True

                        # ── ENTRY LOGIC (only on bar close, no open trade) ─
                        _allowed_sess = ["prime"]
                        if ENABLE_MIDDAY: _allowed_sess.append("midday")
                        if ENABLE_EURO:   _allowed_sess.append("euro")
                        _can_enter = (
                            not trade_mgr.in_trade and ind and sr_ctx and
                            c_date.weekday() not in SKIP_WEEKDAYS and
                            bool(current_zones) and
                            session in _allowed_sess and
                            daily_trades.get(date_str, 0) < MAX_TRADES_PER_DAY and
                            daily_loss_counts.get(date_str, 0) < DAILY_LOSS_LIMIT and
                            chop_filters_pass(closed.close, ind.ema_fast, ind.ema_slow,
                                              ind.slope, ind.adx, session)
                        )
                        if _can_enter:
                            ef    = ind.ema_fast;  es = ind.ema_slow
                            em    = ind.ema_macro
                            close = closed.close;  atr = ind.atr
                            slope = ind.slope;     cb  = ind.consec_bearish
                            rtol  = atr * RETEST_ATR_MULT

                            # ── LONG entries ──────────────────────────
                            if ENABLE_LONG and ind.htf_long_bias:

                                # Path A — EMA9 retest
                                if (ENABLE_PATH_A and
                                        ef > es and close > em and slope > 0 and
                                        abs(close - ef) <= rtol and close > es):
                                    sl_d = atr * LONG_ATR_SL_MULT
                                    ok, tp, rr, gf, notes = sr_gate_check(
                                        close, "up", sr_ctx, sl_d)
                                    if ok:
                                        trade_mgr.enter("long", close, ts, atr,
                                                        "A", notes, tp, bar_idx)
                                        daily_trades[date_str] = \
                                            daily_trades.get(date_str, 0) + 1
                                        log_entry("long", "A", close,
                                                  close - sl_d, tp, ts, notes)
                                    else:
                                        gate_counts[gf] = gate_counts.get(gf, 0) + 1
                                        log_block(gf, notes, "long-A", close)

                                # Path B — EMA9 pull-away breakout
                                if (not trade_mgr.in_trade and ENABLE_PATH_B and
                                        ef > es and close > em and slope > 0 and
                                        (close - ef) >= EMA9_PULLAWAY_PTS and
                                        date_str in opening_high and
                                        close > opening_high[date_str]):
                                    sl_d = atr * LONG_ATR_SL_MULT
                                    ok, tp, rr, gf, notes = sr_gate_check(
                                        close, "up", sr_ctx, sl_d)
                                    if ok:
                                        trade_mgr.enter("long", close, ts, atr,
                                                        "B", notes, tp, bar_idx)
                                        daily_trades[date_str] = \
                                            daily_trades.get(date_str, 0) + 1
                                        log_entry("long", "B", close,
                                                  close - sl_d, tp, ts, notes)
                                    else:
                                        gate_counts[gf] = gate_counts.get(gf, 0) + 1
                                        log_block(gf, notes, "long-B", close)

                            # ── SHORT entries ─────────────────────────
                            if not trade_mgr.in_trade and ENABLE_SHORT:

                                # Path A — EMA9 retest short
                                if (ENABLE_PATH_A and
                                        ef < es and close < em and slope < 0 and
                                        abs(close - ef) <= rtol and close < es and
                                        cb >= SHORT_CONFIRM_BARS):
                                    sl_d = atr * SHORT_ATR_SL_MULT
                                    ok, tp, rr, gf, notes = sr_gate_check(
                                        close, "down", sr_ctx, sl_d)
                                    if ok:
                                        trade_mgr.enter("short", close, ts, atr,
                                                        "A", notes, tp, bar_idx)
                                        daily_trades[date_str] = \
                                            daily_trades.get(date_str, 0) + 1
                                        log_entry("short", "A", close,
                                                  close + sl_d, tp, ts, notes)
                                    else:
                                        gate_counts[gf] = gate_counts.get(gf, 0) + 1
                                        log_block(gf, notes, "short-A", close)

                                # Path B — EMA9 pull-away breakdown
                                if (not trade_mgr.in_trade and ENABLE_PATH_B and
                                        ef < es and close < em and slope < 0 and
                                        (ef - close) >= EMA9_PULLAWAY_PTS and
                                        cb >= SHORT_CONFIRM_BARS and
                                        date_str in opening_close and
                                        close < opening_close[date_str]):
                                    sl_d = atr * SHORT_ATR_SL_MULT
                                    ok, tp, rr, gf, notes = sr_gate_check(
                                        close, "down", sr_ctx, sl_d)
                                    if ok:
                                        trade_mgr.enter("short", close, ts, atr,
                                                        "B", notes, tp, bar_idx)
                                        daily_trades[date_str] = \
                                            daily_trades.get(date_str, 0) + 1
                                        log_entry("short", "B", close,
                                                  close + sl_d, tp, ts, notes)
                                    else:
                                        gate_counts[gf] = gate_counts.get(gf, 0) + 1
                                        log_block(gf, notes, "short-B", close)

                        # ── Broadcast full state after bar close ──────
                        _bar_trigger = (
                            "TRADE_EXIT"  if _bar_exited and not trade_mgr.in_trade else
                            "TRADE_ENTRY" if trade_mgr.in_trade else
                            "BAR_CLOSE"
                        )
                        await _broadcaster.broadcast(_build_payload(
                            _bar_trigger, bars, bar_indicators, current_zones, ind,
                            trade_mgr, price, ts, equity, daily_trades,
                            daily_loss_counts, gate_counts, session, bar_idx
                        ))

                    # ── INTRABAR: tick-level SL monitoring ────────
                    elif trade_mgr.in_trade:
                        tick_count += 1
                        result = trade_mgr.check_tick_sl(price, ts)
                        if result:
                            exit_p, exit_r = result
                            rec = trade_mgr.exit(exit_p, ts, exit_r,
                                                 daily_loss_counts)
                            equity += rec.pnl
                            log_exit(rec, equity)
                            write_trade_csv(rec)
                            await _broadcaster.broadcast(_build_payload(
                                "TRADE_EXIT", bars, bar_indicators, current_zones, ind,
                                trade_mgr, price, ts, equity, daily_trades,
                                daily_loss_counts, gate_counts, session, bar_idx
                            ))
                        elif tick_count % 5 == 0:
                            # Light tick update for live P&L display
                            await _broadcaster.broadcast({
                                "trigger":    "TICK",
                                "ts":         ts.strftime("%H:%M:%S"),
                                "price":      round(price, 2),
                                "equity":     round(equity, 2),
                                "open_pnl":   round(
                                    (price - trade_mgr.entry_price)
                                    if trade_mgr.direction == "long"
                                    else (trade_mgr.entry_price - price), 2
                                ) if trade_mgr.in_trade else None,
                                "stop_loss":  round(trade_mgr.stop_loss, 2)
                                              if trade_mgr.in_trade else None,
                            })

        except (ConnectionRefusedError, OSError) as e:
            print(f"  {RED}Feed unavailable: {e} — retry in {RECONNECT_DELAY}s{R}")
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as e:
            print(f"  {RED}Error: {e} — retry in {RECONNECT_DELAY}s{R}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(RECONNECT_DELAY)


# ════════════════════════════════════════════════════════════════
# 12. ENTRY POINT
# ════════════════════════════════════════════════════════════════

async def _run_all():
    """Run dashboard WS server + trading loop concurrently."""
    import websockets as _ws
    dash_server = await _ws.serve(
        _dash_handler, "0.0.0.0", DASH_PORT_TRADER,
        ping_interval=20, ping_timeout=30
    )
    print(f"  Dashboard WS  → ws://localhost:{DASH_PORT_TRADER}")
    print(f"  Open          → live_dashboard.html in your browser\n")
    try:
        await trading_loop()
    finally:
        dash_server.close()
        await dash_server.wait_closed()


def main():
    print("\nPress Ctrl-C to stop.\n")
    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
