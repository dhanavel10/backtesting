"""
Real-Time S/R Zone Detector — NIFTY Intraday
=============================================
Candle-based pipeline: analysis runs only on fully-formed OHLC bars.
Backtest and live feed the exact same data format → identical results.

Pipeline:
    closed 5-min OHLC candle
        ↓  ZigZagPivotDetector.on_candle()   (H and L both known, no path ambiguity)
    pivots
        ↓  DensityZoneMap                    (KDE with time decay)
    live S/R zones
        ↓  ZoneReactionTracker               (NEW — bar-by-bar reaction report)
    printed reaction report per bar

Reaction events reported per bar:
    APPROACHING  — bar's H or L came within the approach buffer of a zone edge
    INSIDE_ZONE  — bar traded inside the zone (H > lower AND L < upper)
    REVERSAL     — bar entered zone and closed with conviction away from it
    BREAKOUT     — bar closed decisively beyond the zone boundary
    RETEST       — bar approached a previously broken zone from the flip side
    STALLING     — bar opened & closed inside zone with no breakout or reversal

Live usage:
    engine.on_candle(high, low, bar_time)   ← call once per closed 5-min bar
    zones = engine.zones(live_price, now)   ← call any time for current zones
"""

import os
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# ════════════════════════════════════════════════════════════════
# 0.  ANSI COLOURS  (degrade gracefully on Windows)
# ════════════════════════════════════════════════════════════════

try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

C_RESET   = "\033[0m"
C_RED     = "\033[91m"
C_GREEN   = "\033[92m"
C_YELLOW  = "\033[93m"
C_CYAN    = "\033[96m"
C_BLUE    = "\033[94m"
C_MAGENTA = "\033[95m"
C_BOLD    = "\033[1m"
C_DIM     = "\033[2m"

EVENT_COLORS = {
    "INSIDE_ZONE": C_YELLOW,
    "APPROACHING": C_CYAN,
    "REVERSAL":    C_GREEN + C_BOLD,
    "BREAKOUT":    C_RED   + C_BOLD,
    "RETEST":      C_MAGENTA,
    "STALLING":    C_DIM,
}
EVENT_ICONS = {
    "INSIDE_ZONE": "●",
    "APPROACHING": "→",
    "REVERSAL":    "↩",
    "BREAKOUT":    "⚡",
    "RETEST":      "↺",
    "STALLING":    "⏸",
}


# ════════════════════════════════════════════════════════════════
# 1. RANGE BAR BUILDER
# ════════════════════════════════════════════════════════════════

@dataclass
class RangeBar:
    open: float
    high: float
    low: float
    close: float
    start_time: datetime
    end_time: datetime
    tick_count: int


class RangeBarBuilder:
    """
    Aggregates ticks into bars of a fixed PRICE RANGE (not time).

    Tuning for Nifty:
        range_size = 3 pts  → very fast, ~200–400 bars/day, scalping
        range_size = 5 pts  → balanced, ~80–150 bars/day  (recommended)
        range_size = 10 pts → slow, ~30–60 bars/day, swing
    """

    def __init__(self, range_size: float = 8.0):
        self.range_size = range_size
        self._reset()

    def _reset(self):
        self.o = self.h = self.l = None
        self.start_ts = None
        self.tick_count = 0

    def on_tick(self, price: float, ts: datetime) -> Optional[RangeBar]:
        if self.o is None:
            self.o = self.h = self.l = price
            self.start_ts = ts
            self.tick_count = 1
            return None

        self.h = max(self.h, price)
        self.l = min(self.l, price)
        self.tick_count += 1

        if (self.h - self.l) >= self.range_size:
            bar = RangeBar(
                open=self.o, high=self.h, low=self.l, close=price,
                start_time=self.start_ts, end_time=ts,
                tick_count=self.tick_count,
            )
            self.o = self.h = self.l = price
            self.start_ts = ts
            self.tick_count = 1
            return bar
        return None


# ════════════════════════════════════════════════════════════════
# 2. ZIGZAG REAL-TIME PIVOT DETECTOR
# ════════════════════════════════════════════════════════════════

@dataclass
class Pivot:
    price: float
    time: datetime
    type: str               # "high" or "low"
    swing_size: float
    confirmed: bool = True


class ZigZagPivotDetector:
    """
    Streaming swing-pivot detector.
    Pivot CONFIRMED when price reverses by ≥ reversal_threshold from
    the running extremum.
    """

    def __init__(self, reversal_threshold: float = 30.0):
        self.reversal = reversal_threshold
        self.direction = 0
        self.ext_price: Optional[float] = None
        self.ext_time:  Optional[datetime] = None
        self.last_pivot_price: Optional[float] = None
        self.pivots: list[Pivot] = []

    def on_bar(self, bar: RangeBar) -> Optional[Pivot]:
        if self.direction == 0:
            self.ext_price = bar.high
            self.ext_time = bar.end_time
            self.direction = 1
            self.last_pivot_price = bar.low
            return None

        if self.direction == 1:
            if bar.high > self.ext_price:
                self.ext_price = bar.high
                self.ext_time = bar.end_time
            elif (self.ext_price - bar.low) >= self.reversal:
                pivot = Pivot(
                    price=self.ext_price, time=self.ext_time, type="high",
                    swing_size=self.ext_price - (self.last_pivot_price or self.ext_price),
                )
                self.pivots.append(pivot)
                self.last_pivot_price = self.ext_price
                self.direction = -1
                self.ext_price = bar.low
                self.ext_time = bar.end_time
                return pivot
        else:
            if bar.low < self.ext_price:
                self.ext_price = bar.low
                self.ext_time = bar.end_time
            elif (bar.high - self.ext_price) >= self.reversal:
                pivot = Pivot(
                    price=self.ext_price, time=self.ext_time, type="low",
                    swing_size=(self.last_pivot_price or self.ext_price) - self.ext_price,
                )
                self.pivots.append(pivot)
                self.last_pivot_price = self.ext_price
                self.direction = 1
                self.ext_price = bar.high
                self.ext_time = bar.end_time
                return pivot
        return None

    def on_candle(self, high: float, low: float, end_time: datetime) -> list:
        confirmed = []

        if self.direction == 0:
            self.ext_price = high
            self.ext_time  = end_time
            self.direction = 1
            self.last_pivot_price = low
            return confirmed

        if self.direction == 1:
            if high > self.ext_price:
                self.ext_price = high
                self.ext_time  = end_time
            if (self.ext_price - low) >= self.reversal:
                pivot = Pivot(
                    price=self.ext_price, time=self.ext_time, type="high",
                    swing_size=self.ext_price - (self.last_pivot_price or self.ext_price),
                )
                self.pivots.append(pivot)
                confirmed.append(pivot)
                self.last_pivot_price = self.ext_price
                self.direction = -1
                self.ext_price = low
                self.ext_time  = end_time
        else:
            if low < self.ext_price:
                self.ext_price = low
                self.ext_time  = end_time
            if (high - self.ext_price) >= self.reversal:
                pivot = Pivot(
                    price=self.ext_price, time=self.ext_time, type="low",
                    swing_size=(self.last_pivot_price or self.ext_price) - self.ext_price,
                )
                self.pivots.append(pivot)
                confirmed.append(pivot)
                self.last_pivot_price = self.ext_price
                self.direction = 1
                self.ext_price = high
                self.ext_time  = end_time

        return confirmed

    def provisional_pivot(self) -> Optional[Pivot]:
        if self.ext_price is None:
            return None
        return Pivot(
            price=self.ext_price, time=self.ext_time,
            type="high" if self.direction == 1 else "low",
            swing_size=abs(self.ext_price - (self.last_pivot_price or self.ext_price)),
            confirmed=False,
        )


# ════════════════════════════════════════════════════════════════
# 3. KDE DENSITY ZONE MAP
# ════════════════════════════════════════════════════════════════

@dataclass
class Zone:
    price: float
    lower: float
    upper: float
    strength: float
    type: str          # "Support" / "Resistance"
    n_pivots: int
    anchored: bool = False


_ANCHOR_REL_THRESHOLD = 0.45


class DensityZoneMap:
    """
    Weighted KDE → local-maxima → S/R zones with anchoring.
    See module docstring for full explanation.
    """

    def __init__(
        self,
        bandwidth:          float = 8.0,
        zone_half_width:    float = 17.5,
        merge_distance:     float = 12.0,
        half_life_min:      float = 120.0,
        min_rel_density:    float = 0.15,
        provisional_weight: float = 0.5,
        anchor_threshold:   float = _ANCHOR_REL_THRESHOLD,
    ):
        self.bw = bandwidth
        self.zone_hw = zone_half_width
        self.merge_dist = merge_distance
        self.decay_lambda = np.log(2) / (half_life_min * 60.0)
        self.min_rel = min_rel_density
        self.provisional_weight = provisional_weight
        self.anchor_threshold = anchor_threshold
        self.pivots: list[Pivot] = []
        self._anchored_prices: list[float] = []

    def add_pivot(self, pivot: Pivot):
        self.pivots.append(pivot)

    def _build_density(self, prices, weights, grid_step=0.5):
        p_lo = float(prices.min()) - 4 * self.bw
        p_hi = float(prices.max()) + 4 * self.bw
        grid = np.arange(p_lo, p_hi, grid_step)
        diffs   = (grid[:, None] - prices[None, :]) / self.bw
        density = (weights[None, :] * np.exp(-0.5 * diffs ** 2)).sum(axis=1)
        return grid, density

    def _find_raw_peaks(self, grid, density):
        thresh = density.max() * self.min_rel
        peaks = []
        for i in range(1, len(density) - 1):
            if (density[i] >= density[i - 1] and density[i] >= density[i + 1]
                    and density[i] >= thresh):
                peaks.append(i)
        return peaks

    def _update_anchors(self, raw_peak_prices, peak_densities, d_max):
        strong_peaks = [
            p for p, d in zip(raw_peak_prices, peak_densities)
            if d >= d_max * self.anchor_threshold
        ]
        for sp in strong_peaks:
            if not any(abs(sp - a) < self.merge_dist for a in self._anchored_prices):
                self._anchored_prices.append(sp)
        all_peak_prices = raw_peak_prices
        self._anchored_prices = [
            a for a in self._anchored_prices
            if any(abs(a - p) < self.merge_dist for p in all_peak_prices)
        ]

    def compute(self, current_price, now, provisional=None, top_n=10, grid_step=0.5):
        if not self.pivots:
            return []

        prices, weights = [], []
        for p in self.pivots:
            age = max(0.0, (now - p.time).total_seconds())
            w = max(1.0, p.swing_size / 10.0) * np.exp(-self.decay_lambda * age)
            if w < 0.05:
                continue
            prices.append(p.price)
            weights.append(w)

        if provisional is not None:
            prices.append(provisional.price)
            weights.append(
                max(1.0, provisional.swing_size / 10.0) * self.provisional_weight)

        if not prices:
            self._anchored_prices.clear()
            return []

        prices  = np.asarray(prices)
        weights = np.asarray(weights)

        grid, density = self._build_density(prices, weights, grid_step)
        peaks_idx = self._find_raw_peaks(grid, density)
        if not peaks_idx:
            return []

        d_max = float(density.max())
        raw_peak_prices  = [float(grid[i]) for i in peaks_idx]
        raw_peak_density = [float(density[i]) for i in peaks_idx]

        self._update_anchors(raw_peak_prices, raw_peak_density, d_max)

        zones: list[Zone] = []
        used_anchors: set[float] = set()

        for i, idx in enumerate(peaks_idx):
            raw_price   = raw_peak_prices[i]
            raw_density = raw_peak_density[i]
            is_strong   = raw_density >= d_max * self.anchor_threshold

            near_anchors = [a for a in self._anchored_prices
                            if abs(a - raw_price) < self.merge_dist]

            if near_anchors and is_strong:
                anchor  = min(near_anchors, key=lambda a: abs(a - raw_price))
                centre  = anchor
                anchored = True
                used_anchors.add(anchor)
            else:
                centre  = raw_price
                anchored = False

            d_at_centre = float(
                np.interp(centre, grid, density)
                if grid[0] <= centre <= grid[-1]
                else raw_density)
            n_contrib = int(np.sum(np.abs(prices - centre) <= 2 * self.bw))

            zones.append(Zone(
                price=round(centre, 2),
                lower=round(centre - self.zone_hw, 2),
                upper=round(centre + self.zone_hw, 2),
                strength=d_at_centre,
                type="Support" if centre < current_price else "Resistance",
                n_pivots=n_contrib,
                anchored=anchored,
            ))

        for a in self._anchored_prices:
            if a in used_anchors:
                continue
            if grid[0] <= a <= grid[-1]:
                d_at_a = float(np.interp(a, grid, density))
            else:
                d_at_a = 0.0
            if d_at_a < d_max * self.min_rel:
                continue
            n_contrib = int(np.sum(np.abs(prices - a) <= 2 * self.bw))
            zones.append(Zone(
                price=round(a, 2),
                lower=round(a - self.zone_hw, 2),
                upper=round(a + self.zone_hw, 2),
                strength=d_at_a,
                type="Support" if a < current_price else "Resistance",
                n_pivots=n_contrib,
                anchored=True,
            ))

        zones = self._merge(zones)

        if zones:
            s_max = max(z.strength for z in zones)
            if s_max > 0:
                for z in zones:
                    z.strength = round(100.0 * z.strength / s_max, 1)

        zones.sort(key=lambda z: z.strength, reverse=True)
        return zones[:top_n]

    def _merge(self, zones):
        if not zones:
            return zones
        zones.sort(key=lambda z: z.price)
        out: list[Zone] = []
        for z in zones:
            if out and (z.price - out[-1].price) < self.merge_dist:
                prev = out[-1]
                if prev.anchored and not z.anchored:
                    centre = prev.price
                elif z.anchored and not prev.anchored:
                    centre = z.price
                else:
                    centre = round((prev.price + z.price) / 2, 2)
                stronger = prev if prev.strength >= z.strength else z
                out[-1] = Zone(
                    price=centre,
                    lower=round(centre - self.zone_hw, 2),
                    upper=round(centre + self.zone_hw, 2),
                    strength=stronger.strength + 0.3 * min(prev.strength, z.strength),
                    type=stronger.type,
                    n_pivots=prev.n_pivots + z.n_pivots,
                    anchored=prev.anchored or z.anchored,
                )
            else:
                out.append(z)
        return out


# ════════════════════════════════════════════════════════════════
# 4. END-TO-END ENGINE
# ════════════════════════════════════════════════════════════════

class RealtimeSREngine:
    def __init__(
        self,
        reversal_threshold: float = 30.0,
        bandwidth:          float = 8.0,
        zone_half_width:    float = 17.5,
        half_life_min:      float = 120.0,
    ):
        self.zz   = ZigZagPivotDetector(reversal_threshold=reversal_threshold)
        self.zmap = DensityZoneMap(
            bandwidth=bandwidth,
            zone_half_width=zone_half_width,
            half_life_min=half_life_min,
        )
        self.last_pivot: Optional[Pivot] = None

    def on_candle(self, high: float, low: float, bar_time: datetime):
        for pivot in self.zz.on_candle(high, low, bar_time):
            self.zmap.add_pivot(pivot)
            self.last_pivot = pivot

    def zones(self, current_price: float, now: datetime, top_n: int = 10) -> list[Zone]:
        return self.zmap.compute(
            current_price=current_price,
            now=now,
            provisional=self.zz.provisional_pivot(),
            top_n=top_n,
        )


# ════════════════════════════════════════════════════════════════
# 5. ZONE REACTION TRACKER  (bar-by-bar edition)
# ════════════════════════════════════════════════════════════════
#
# In the live script (live_sr.py) this runs tick-by-tick.
# Here it runs once per closed OHLC bar — every field maps 1-for-1
# but "ticks_inside" is replaced by "bars_inside".
#
# Reaction classification per bar:
#   APPROACHING  — bar's near edge came within APPROACH_BUFFER of the zone
#                  boundary (H approached resistance, or L approached support)
#   INSIDE_ZONE  — bar's range overlaps the zone (H > lower AND L < upper)
#   REVERSAL     — bar was inside zone AND close is convincingly away from
#                  entry side (close < lower for resistance, close > upper for
#                  support), AND the candle body is directional
#   BREAKOUT     — bar closed beyond the zone boundary by ≥ BREAKOUT_MIN_GAP;
#                  body must close outside (not just a wick)
#   RETEST       — bar approached a previously BROKEN zone from the flip side
#   STALLING     — bar's range overlapped zone but close stayed inside;
#                  no reversal and no breakout
# ════════════════════════════════════════════════════════════════

# ---------- tunables (can also be set via .env) ------------------
APPROACH_BUFFER_PTS  = float(os.getenv("APPROACH_BUFFER_PTS",  "5.0"))
# pts below/above zone edge that count as "approaching"
BREAKOUT_MIN_GAP_PCT = float(os.getenv("BREAKOUT_MIN_GAP_PCT", "0.2"))
# breakout close must be ≥ this fraction × zone_width beyond the zone edge
REVERSAL_BODY_PCT    = float(os.getenv("REVERSAL_BODY_PCT",     "0.4"))
# candle body must span ≥ this fraction of zone_width to qualify as reversal
STALL_BARS_LIMIT     = int(os.getenv("STALL_BARS_LIMIT",        "3"))
# bars inside without breakout or reversal → stalling label
# -----------------------------------------------------------------


class ReactionState(str, Enum):
    IDLE        = "IDLE"
    APPROACHING = "APPROACHING"
    INSIDE      = "INSIDE"
    REVERSAL    = "REVERSAL"
    BREAKOUT    = "BREAKOUT"
    RETEST      = "RETEST"
    STALLING    = "STALLING"


@dataclass
class ZoneReaction:
    zone_price:   float
    zone_lower:   float
    zone_upper:   float
    zone_type:    str

    state:        ReactionState = ReactionState.IDLE
    entry_bar_close: Optional[float] = None   # close of bar that first entered zone
    bars_inside:  int = 0
    bars_beyond:  int = 0
    broken:       bool = False
    broken_side:  Optional[str] = None        # "above" | "below"
    events:       List[dict] = field(default_factory=list)


class ZoneReactionTracker:
    """
    Bar-by-bar zone reaction tracker.
    Call  update_bar(open, high, low, close, bar_time, zones)
    after every closed candle to get the list of new reaction events.
    """

    def __init__(self):
        self._reactions: Dict[float, ZoneReaction] = {}

    # ── internal helpers ───────────────────────────────────────
    @staticmethod
    def _key(z: Zone) -> float:
        return round(z.price, 2)

    def _get_or_create(self, z: Zone) -> ZoneReaction:
        k = self._key(z)
        if k not in self._reactions:
            self._reactions[k] = ZoneReaction(
                zone_price=z.price, zone_lower=z.lower,
                zone_upper=z.upper, zone_type=z.type,
            )
        else:
            r = self._reactions[k]
            r.zone_lower = z.lower
            r.zone_upper = z.upper
            r.zone_type  = z.type
        return self._reactions[k]

    def _zone_width(self, r: ZoneReaction) -> float:
        return max(r.zone_upper - r.zone_lower, 1.0)

    def _overlaps_zone(self, high: float, low: float, r: ZoneReaction) -> bool:
        """Bar's range penetrates the zone (at least partially)."""
        return high >= r.zone_lower and low <= r.zone_upper

    def _bar_close_inside(self, close: float, r: ZoneReaction) -> bool:
        return r.zone_lower <= close <= r.zone_upper

    def _bar_close_above(self, close: float, r: ZoneReaction) -> bool:
        return close > r.zone_upper

    def _bar_close_below(self, close: float, r: ZoneReaction) -> bool:
        return close < r.zone_lower

    def _approaching_resistance(self, high: float, r: ZoneReaction) -> bool:
        return r.zone_lower - APPROACH_BUFFER_PTS <= high < r.zone_lower

    def _approaching_support(self, low: float, r: ZoneReaction) -> bool:
        return r.zone_upper < low <= r.zone_upper + APPROACH_BUFFER_PTS

    def _breakout_strength(self, gap: float, r: ZoneReaction) -> str:
        ratio = gap / self._zone_width(r)
        if ratio >= 0.75:   return "STRONG"
        if ratio >= 0.35:   return "MODERATE"
        return "WEAK"

    def _reversal_strength(self, body: float, r: ZoneReaction) -> str:
        ratio = body / self._zone_width(r)
        if ratio >= 1.5:   return "STRONG"
        if ratio >= 0.75:  return "MODERATE"
        return "WEAK"

    @staticmethod
    def _make_event(etype: str, r: ZoneReaction, bar_close: float,
                    bar_time: datetime, description: str,
                    extra: dict = None) -> dict:
        ev = {
            "event":       etype,
            "description": description,
            "bar_close":   round(bar_close, 2),
            "zone_price":  round(r.zone_price, 2),
            "zone_lower":  round(r.zone_lower, 2),
            "zone_upper":  round(r.zone_upper, 2),
            "zone_type":   r.zone_type,
            "ts":          bar_time.strftime("%H:%M"),
        }
        if extra:
            ev.update(extra)
        return ev

    # ── main bar-level update ──────────────────────────────────
    def update_bar(
        self,
        o: float, h: float, l: float, c: float,
        bar_time: datetime,
        zones: List[Zone],
    ) -> List[dict]:
        """
        Call once per closed bar with its OHLC and the current zone list.
        Returns list of new reaction-event dicts (may be empty).
        """
        new_events: List[dict] = []

        for z in zones:
            r   = self._get_or_create(z)
            w   = self._zone_width(r)
            evt = None

            # ── 1. RETEST  (broken zone, price returning) ─────────────────
            if r.broken:
                if r.broken_side == "above":
                    # zone was broken upward — retest = price falls back near zone upper
                    if l <= r.zone_upper + APPROACH_BUFFER_PTS:
                        if r.state != ReactionState.RETEST:
                            r.state = ReactionState.RETEST
                            evt = self._make_event(
                                "RETEST", r, c, bar_time,
                                f"Retesting previously broken RESISTANCE from above  "
                                f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                                f"bar_low={l:.2f}  close={c:.2f}  "
                                f"{'Holding above? Bullish continuation setup.' if c > r.zone_upper else 'Failed retest — watch for reversal down.'}",
                                extra={"direction": "DOWN_RETEST"})
                    else:
                        r.state = ReactionState.IDLE

                elif r.broken_side == "below":
                    # zone was broken downward — retest = price bounces back near zone lower
                    if h >= r.zone_lower - APPROACH_BUFFER_PTS:
                        if r.state != ReactionState.RETEST:
                            r.state = ReactionState.RETEST
                            evt = self._make_event(
                                "RETEST", r, c, bar_time,
                                f"Retesting previously broken SUPPORT from below  "
                                f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                                f"bar_high={h:.2f}  close={c:.2f}  "
                                f"{'Holding below? Bearish continuation setup.' if c < r.zone_lower else 'Failed retest — watch for reversal up.'}",
                                extra={"direction": "UP_RETEST"})
                    else:
                        r.state = ReactionState.IDLE

                if evt:
                    r.events.append(evt)
                    new_events.append(evt)
                continue   # don't also classify as inside/approach

            # ── 2. BREAKOUT  (bar closes decisively outside zone) ─────────
            if self._overlaps_zone(h, l, r) or r.state == ReactionState.INSIDE:
                breakout_min_gap = w * BREAKOUT_MIN_GAP_PCT

                if self._bar_close_above(c, r) and (c - r.zone_upper) >= breakout_min_gap:
                    # Only fire breakout for Resistance zones (price closing above)
                    gap      = c - r.zone_upper
                    strength = self._breakout_strength(gap, r)
                    r.broken      = True
                    r.broken_side = "above"
                    r.state       = ReactionState.BREAKOUT
                    r.bars_inside = 0
                    r.entry_bar_close = None
                    evt = self._make_event(
                        "BREAKOUT", r, c, bar_time,
                        f"⚡ {strength} BREAKOUT above {r.zone_type.upper()}  "
                        f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                        f"close={c:.2f}  gap={gap:.2f} pts  "
                        f"body={abs(c-o):.2f} pts  "
                        f"{'★ Strong momentum — expect follow-through.' if strength == 'STRONG' else 'Moderate — watch for retest before continuation.'}",
                        extra={"gap": round(gap, 2), "strength": strength,
                               "direction": "UP", "bar_body": round(abs(c-o), 2)})

                elif self._bar_close_below(c, r) and (r.zone_lower - c) >= breakout_min_gap:
                    # Only fire breakout for Support zones (price closing below)
                    gap      = r.zone_lower - c
                    strength = self._breakout_strength(gap, r)
                    r.broken      = True
                    r.broken_side = "below"
                    r.state       = ReactionState.BREAKOUT
                    r.bars_inside = 0
                    r.entry_bar_close = None
                    evt = self._make_event(
                        "BREAKOUT", r, c, bar_time,
                        f"⚡ {strength} BREAKOUT below {r.zone_type.upper()}  "
                        f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                        f"close={c:.2f}  gap={gap:.2f} pts  "
                        f"body={abs(c-o):.2f} pts  "
                        f"{'★ Strong momentum — expect follow-through.' if strength == 'STRONG' else 'Moderate — watch for retest before continuation.'}",
                        extra={"gap": round(gap, 2), "strength": strength,
                               "direction": "DOWN", "bar_body": round(abs(c-o), 2)})

                # ── 3. REVERSAL  (entered zone, close convincingly outside entry side) ──
                elif self._overlaps_zone(h, l, r):
                    r.bars_inside += 1
                    if r.entry_bar_close is None:
                        r.entry_bar_close = c

                    body = abs(c - o)

                    # Reversal at Resistance: bar entered zone (L < upper) but closed below lower
                    # OR strong bearish close well inside zone after touching upper
                    resistance_reversal = (
                        r.zone_type == "Resistance"
                        and h >= r.zone_lower           # bar touched zone
                        and c < r.zone_lower            # closed back below zone
                        and body >= w * REVERSAL_BODY_PCT
                    )
                    # Reversal at Support: bar entered zone (H > lower) but closed above upper
                    # OR strong bullish close well inside zone after touching lower
                    support_reversal = (
                        r.zone_type == "Support"
                        and l <= r.zone_upper           # bar touched zone
                        and c > r.zone_upper            # closed back above zone
                        and body >= w * REVERSAL_BODY_PCT
                    )

                    if resistance_reversal or support_reversal:
                        direction = "DOWN" if resistance_reversal else "UP"
                        strength  = self._reversal_strength(body, r)
                        r.state       = ReactionState.REVERSAL
                        r.bars_inside = 0
                        r.entry_bar_close = None
                        evt = self._make_event(
                            "REVERSAL", r, c, bar_time,
                            f"↩ {strength} REVERSAL at {r.zone_type.upper()}  "
                            f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                            f"O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}  "
                            f"body={body:.2f} pts  direction={direction}  "
                            f"{'★ Strong rejection — high probability entry.' if strength == 'STRONG' else 'Moderate — wait for next bar confirmation.'}",
                            extra={"direction": direction, "strength": strength,
                                   "bar_body": round(body, 2)})

                    # ── 4. STALLING ──────────────────────────────────────────
                    elif r.bars_inside >= STALL_BARS_LIMIT and self._bar_close_inside(c, r):
                        if r.state != ReactionState.STALLING:
                            r.state = ReactionState.STALLING
                            evt = self._make_event(
                                "STALLING", r, c, bar_time,
                                f"⏸ STALLING inside {r.zone_type.upper()}  "
                                f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                                f"{r.bars_inside} bars inside with no breakout or reversal  "
                                f"close={c:.2f}  — Avoid trading until direction clears.",
                                extra={"bars_inside": r.bars_inside})

                    # ── 5. INSIDE_ZONE (first entry) ─────────────────────────
                    elif r.state not in (ReactionState.INSIDE, ReactionState.STALLING,
                                         ReactionState.REVERSAL):
                        r.state = ReactionState.INSIDE
                        evt = self._make_event(
                            "INSIDE_ZONE", r, c, bar_time,
                            f"● Price entered {r.zone_type.upper()} zone  "
                            f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                            f"O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}  "
                            f"Watch for reversal or breakout on next bar.",
                            extra={"bars_inside": r.bars_inside})

            # ── 6. APPROACHING ────────────────────────────────────────────
            elif (r.zone_type == "Resistance" and self._approaching_resistance(h, r)) or \
                 (r.zone_type == "Support"    and self._approaching_support(l, r)):
                if r.state not in (ReactionState.APPROACHING, ReactionState.INSIDE,
                                   ReactionState.REVERSAL):
                    r.state = ReactionState.APPROACHING
                    dist = (r.zone_lower - h) if r.zone_type == "Resistance" else (l - r.zone_upper)
                    evt = self._make_event(
                        "APPROACHING", r, c, bar_time,
                        f"→ Approaching {r.zone_type.upper()} zone  "
                        f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                        f"{'H' if r.zone_type=='Resistance' else 'L'}="
                        f"{h if r.zone_type=='Resistance' else l:.2f}  "
                        f"dist={dist:.2f} pts  — Zone test imminent.",
                        extra={"distance_pts": round(dist, 2)})

            # ── 7. Price moved away — reset transient states ───────────────
            else:
                if r.state in (ReactionState.APPROACHING, ReactionState.REVERSAL):
                    r.state = ReactionState.IDLE
                elif r.state in (ReactionState.INSIDE, ReactionState.STALLING):
                    # price exited zone without breakout confirmation → treat as reversal attempt
                    r.state = ReactionState.IDLE
                    r.bars_inside = 0
                    r.entry_bar_close = None

            if evt:
                r.events.append(evt)
                if len(r.events) > 200:
                    r.events = r.events[-200:]
                new_events.append(evt)

        return new_events


# ════════════════════════════════════════════════════════════════
# 6. CONSOLE PRINTERS
# ════════════════════════════════════════════════════════════════

SEP  = "=" * 90
SEP2 = "-" * 90


def print_reaction_report(events: List[dict], cmp: float, bar_time: datetime):
    """Print zone reaction events for a single bar."""
    if not events:
        return
    print(f"\n  {'─'*86}")
    print(f"  {C_BOLD}ZONE REACTION REPORT{C_RESET}  bar={bar_time.strftime('%H:%M')}  "
          f"close={C_BOLD}{cmp:.2f}{C_RESET}")
    print(f"  {'─'*86}")
    for ev in events:
        etype = ev["event"]
        color = EVENT_COLORS.get(etype, "")
        icon  = EVENT_ICONS.get(etype, "•")
        zone_label = f"[{ev['zone_lower']:.2f}–{ev['zone_upper']:.2f}]"
        print(f"  {color}{C_BOLD}{icon}  {etype:<12}{C_RESET}  "
              f"{ev['ts']}  {ev['zone_type']:<12} {zone_label}  "
              f"close={ev['bar_close']:.2f}")
        print(f"  {color}     └─ {ev['description']}{C_RESET}")
    print(f"  {'─'*86}")


def print_zones(zones, cmp, bar_time):
    supp = sorted([z for z in zones if z.type == "Support"],    key=lambda z: z.price, reverse=True)
    res  = sorted([z for z in zones if z.type == "Resistance"], key=lambda z: z.price)
    print(f"\n{SEP}")
    print(f"  S/R ZONES  @  {bar_time.strftime('%H:%M')}   CMP: {cmp:.2f}")
    print(SEP)
    print(f"  {'Type':<12} {'Price':>8}  {'Range':>22}  {'Str':>5}  {'Pivots':>6}  Anchored")
    print(f"  {SEP2}")
    for z in res:
        flag = "★ FIXED" if z.anchored else "  float"
        print(f"  {C_RED}{'RES':<12} {z.price:>8.2f}  [{z.lower:.1f}–{z.upper:.1f}]"
              f"  {z.strength:>5.1f}  {z.n_pivots:>6}  {flag}{C_RESET}")
    print(f"  {'─'*35}  CMP {cmp:.2f}  {'─'*40}")
    for z in supp:
        flag = "★ FIXED" if z.anchored else "  float"
        print(f"  {C_GREEN}{'SUP':<12} {z.price:>8.2f}  [{z.lower:.1f}–{z.upper:.1f}]"
              f"  {z.strength:>5.1f}  {z.n_pivots:>6}  {flag}{C_RESET}")
    print(SEP)


def print_bar_header(bar_num, bar_time, o, h, l, c, n_pivots):
    d     = "^" if c >= o else "v"
    color = C_GREEN if c >= o else C_RED
    spread = h - l
    print(f"\n  {color}{C_BOLD}BAR {bar_num:>3} {d}{C_RESET}  "
          f"{bar_time.strftime('%H:%M')}  "
          f"O:{o:.1f} H:{h:.1f} L:{l:.1f} C:{c:.1f}  "
          f"spread:{spread:.1f}  pivots:{n_pivots}")


# ════════════════════════════════════════════════════════════════
# 7. DEMO — simulate from 5m OHLC
# ════════════════════════════════════════════════════════════════

def simulate_ticks_from_ohlc(df, ticks_per_bar: int = 20):
    rng = np.random.default_rng(42)
    for ts, row in df.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        if rng.random() < 0.5:
            path = [o, h, l, c]
        else:
            path = [o, l, h, c]
        seg_len = ticks_per_bar // 3
        full = []
        for a, b in zip(path[:-1], path[1:]):
            full.extend(np.linspace(a, b, seg_len, endpoint=False))
        full.append(c)
        step = timedelta(minutes=5) / len(full)
        for i, px in enumerate(full):
            yield ts + i * step, float(px)


# ════════════════════════════════════════════════════════════════
# 8. MAIN — backtest / demo mode
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yfinance as yf
    if load_dotenv:
        load_dotenv()

    print("Fetching 5m data for ^NSEI...")
    df = yf.download("^NSEI", period="1d", interval="5m",
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    try:
        df.index = df.index.tz_convert("Asia/Kolkata")
    except TypeError:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    df = df.between_time("09:15", "15:30").dropna()
    print(f"  {len(df)} bars loaded\n")

    REVERSAL_THRESHOLD = float(os.getenv("REVERSAL_THR",  "30.0"))
    BANDWIDTH          = float(os.getenv("BANDWIDTH",     "7.0"))
    ZONE_HALF_WIDTH    = float(os.getenv("ZONE_HW",       "17.5"))
    HALF_LIFE_MIN      = float(os.getenv("HALF_LIFE_MIN", "120.0"))
    TOP_N              = int(os.getenv("TOP_N",           "10"))

    # Print active config
    print(f"  Config: reversal={REVERSAL_THRESHOLD}  bw={BANDWIDTH}  "
          f"zone_hw={ZONE_HALF_WIDTH}  half_life={HALF_LIFE_MIN}min")
    print(f"  Reaction: approach_buf={APPROACH_BUFFER_PTS}pts  "
          f"breakout_gap={BREAKOUT_MIN_GAP_PCT*100:.0f}% zone_width  "
          f"reversal_body={REVERSAL_BODY_PCT*100:.0f}% zone_width  "
          f"stall_bars={STALL_BARS_LIMIT}\n")

    engine   = RealtimeSREngine(
        reversal_threshold=REVERSAL_THRESHOLD,
        bandwidth=BANDWIDTH, zone_half_width=ZONE_HALF_WIDTH, half_life_min=HALF_LIFE_MIN,
    )
    tracker  = ZoneReactionTracker()
    bar_num  = 0

    # ── event counters for end-of-day summary ──────────────────
    event_counts: Dict[str, int] = {}

    for ts, row in df.iterrows():
        bar_time = ts.to_pydatetime()
        o = float(row["Open"]); h = float(row["High"])
        l = float(row["Low"]);  c = float(row["Close"])

        # 1. Feed candle to SR engine (updates pivots + zones)
        engine.on_candle(h, l, bar_time)
        bar_num += 1

        # 2. Get current zones
        zones = engine.zones(current_price=c, now=bar_time, top_n=TOP_N)

        # 3. Run reaction tracker — classify price vs zones
        events = tracker.update_bar(o, h, l, c, bar_time, zones)

        # 4. Print
        print_bar_header(bar_num, bar_time, o, h, l, c, len(engine.zz.pivots))

        if events:
            print_reaction_report(events, c, bar_time)
            for ev in events:
                event_counts[ev["event"]] = event_counts.get(ev["event"], 0) + 1

        # Print full zone table every 12 bars (~1 hour) or when a significant
        # event occurred (breakout or reversal)
        sig = any(e["event"] in ("BREAKOUT", "REVERSAL") for e in events)
        if bar_num % 12 == 0 or sig:
            print_zones(zones, c, bar_time)

    # ── End-of-day summary ──────────────────────────────────────
    if bar_num > 0:
        last_row = df.iloc[-1]
        last_c   = float(last_row["Close"])
        last_ts  = df.index[-1].to_pydatetime()
        zones    = engine.zones(current_price=last_c, now=last_ts, top_n=TOP_N)

        print(f"\n{SEP}")
        print(f"  END-OF-DAY SUMMARY  CMP:{last_c:.2f}  bars:{bar_num}  "
              f"pivots:{len(engine.zz.pivots)}")
        print(SEP)

        # Zone table
        if zones:
            supp = sorted([z for z in zones if z.type == "Support"],
                          key=lambda z: z.price, reverse=True)
            res  = sorted([z for z in zones if z.type == "Resistance"],
                          key=lambda z: z.price)
            print(f"  {'Type':<12} {'Price':>8}  {'Range':>22}  {'Str':>5}  {'Pivots':>6}  Anchored")
            print(f"  {'─'*70}")
            for z in res:
                print(f"  {C_RED}{'RES':<12} {z.price:>8.2f}  [{z.lower:.1f}–{z.upper:.1f}]"
                      f"  {z.strength:>5.1f}  {z.n_pivots:>6}  "
                      f"{'★ FIXED' if z.anchored else '  float'}{C_RESET}")
            print(f"  {'─'*30}  CMP {last_c:.2f}")
            for z in supp:
                print(f"  {C_GREEN}{'SUP':<12} {z.price:>8.2f}  [{z.lower:.1f}–{z.upper:.1f}]"
                      f"  {z.strength:>5.1f}  {z.n_pivots:>6}  "
                      f"{'★ FIXED' if z.anchored else '  float'}{C_RESET}")
        else:
            print("  (no zones)")

        # Event summary
        if event_counts:
            print(f"\n  {'─'*50}")
            print(f"  REACTION EVENT SUMMARY  (whole day)")
            print(f"  {'─'*50}")
            for etype, count in sorted(event_counts.items(), key=lambda x: -x[1]):
                icon  = EVENT_ICONS.get(etype, "•")
                color = EVENT_COLORS.get(etype, "")
                print(f"  {color}{icon}  {etype:<14}  {count:>4} event(s){C_RESET}")

        print(SEP)