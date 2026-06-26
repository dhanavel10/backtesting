"""
Real-Time S/R Zone Detector from Tick Data — NIFTY Intraday
============================================================
Pure price-action pipeline using only (timestamp, price) ticks.

Pipeline:
    tick stream
        ↓  RangeBarBuilder       (filters tick noise)
    range bars
        ↓  ZigZagPivotDetector   (real-time swing detection)
    pivots
        ↓  DensityZoneMap        (KDE with time decay)
    live S/R zones

Why this combination:
    • Range bars are activity-aware — no fake pivots in dead zones
    • ZigZag detects pivots the moment price reverses by N points
      (no waiting for right_bars of confirmation)
    • KDE density updates incrementally — O(1) per new pivot
    • Time decay makes recent pivots dominate (intraday relevance)
"""

import numpy as np
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional


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
    A new bar closes the moment (high − low) ≥ range_size.

    Tuning for Nifty:
        range_size = 3 pts  → very fast, ~200–400 bars/day, scalping
        range_size = 5 pts  → balanced, ~80–150 bars/day  (recommended)
        range_size = 10 pts → slow, ~30–60 bars/day, swing
    """

    def __init__(self, range_size: float = 5.0):
        self.range_size = range_size
        self._reset()

    def _reset(self):
        self.o = self.h = self.l = None
        self.start_ts = None
        self.tick_count = 0

    def on_tick(self, price: float, ts: datetime) -> Optional[RangeBar]:
        """Returns a completed RangeBar when the range is filled, else None."""
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
            # Next bar opens at the closing price
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
    swing_size: float       # points moved in the swing that produced this pivot
    confirmed: bool = True  # True once reversal threshold exceeded


class ZigZagPivotDetector:
    """
    Streaming swing-pivot detector.

    State machine:
        direction = +1  → in an up-swing, tracking the highest high seen
        direction = −1  → in a down-swing, tracking the lowest low seen

    Pivot CONFIRMED when price reverses by ≥ reversal_threshold from
    the running extremum. Lag = exactly that threshold's worth of points.

    Also exposes the PROVISIONAL pivot (the current running extremum) —
    this is the "live" not-yet-confirmed swing point, useful for
    early-warning S/R zone formation.
    """

    def __init__(self, reversal_threshold: float = 15.0):
        self.reversal = reversal_threshold
        self.direction = 0                  # 0 = uninitialized
        self.ext_price: Optional[float] = None
        self.ext_time:  Optional[datetime] = None
        self.last_pivot_price: Optional[float] = None
        self.pivots: list[Pivot] = []

    def on_bar(self, bar: RangeBar) -> Optional[Pivot]:
        """
        Feed a closed range bar. Returns a newly CONFIRMED pivot if one
        was just produced, else None.
        """
        # Initialize on first bar
        if self.direction == 0:
            self.ext_price = bar.high
            self.ext_time = bar.end_time
            self.direction = 1  # arbitrary start
            self.last_pivot_price = bar.low
            return None

        if self.direction == 1:
            # In up-swing — extend if higher high
            if bar.high > self.ext_price:
                self.ext_price = bar.high
                self.ext_time = bar.end_time
            # Check for reversal down
            elif (self.ext_price - bar.low) >= self.reversal:
                pivot = Pivot(
                    price=self.ext_price,
                    time=self.ext_time,
                    type="high",
                    swing_size=self.ext_price - (self.last_pivot_price or self.ext_price),
                )
                self.pivots.append(pivot)
                self.last_pivot_price = self.ext_price
                # Flip swing
                self.direction = -1
                self.ext_price = bar.low
                self.ext_time = bar.end_time
                return pivot
        else:
            # In down-swing — extend if lower low
            if bar.low < self.ext_price:
                self.ext_price = bar.low
                self.ext_time = bar.end_time
            # Check for reversal up
            elif (bar.high - self.ext_price) >= self.reversal:
                pivot = Pivot(
                    price=self.ext_price,
                    time=self.ext_time,
                    type="low",
                    swing_size=(self.last_pivot_price or self.ext_price) - self.ext_price,
                )
                self.pivots.append(pivot)
                self.last_pivot_price = self.ext_price
                self.direction = 1
                self.ext_price = bar.high
                self.ext_time = bar.end_time
                return pivot
        return None

    def provisional_pivot(self) -> Optional[Pivot]:
        """Current running extremum — not yet confirmed, but live."""
        if self.ext_price is None:
            return None
        return Pivot(
            price=self.ext_price,
            time=self.ext_time,
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
    strength: float    # 0–100, relative
    type: str          # "Support" / "Resistance"
    n_pivots: int      # contributing pivots (after decay pruning)


class DensityZoneMap:
    """
    Each pivot drops a Gaussian vote at its price, weighted by:
        • swing_size       (bigger swings ⇒ stronger pivot)
        • exp(−age / τ)    (time decay)

    S/R zones = local maxima of the summed density along the price axis.

    Tuning for Nifty:
        bandwidth        = 6–10 pts   (KDE smoothing — tighter = sharper zones)
        zone_half_width  = 10–15 pts  (visual band around each zone center)
        merge_distance   = 12 pts     (zones closer than this get merged)
        half_life_min    = 90–180     (how long a pivot stays "fresh")
        min_rel_density  = 0.15       (peak must be ≥ 15% of max to qualify)
    """

    def __init__(
        self,
        bandwidth:       float = 8.0,
        zone_half_width: float = 12.0,
        merge_distance:  float = 12.0,
        half_life_min:   float = 120.0,
        min_rel_density: float = 0.15,
        provisional_weight: float = 0.5,
    ):
        self.bw = bandwidth
        self.zone_hw = zone_half_width
        self.merge_dist = merge_distance
        self.decay_lambda = np.log(2) / (half_life_min * 60.0)   # per second
        self.min_rel = min_rel_density
        self.provisional_weight = provisional_weight
        self.pivots: list[Pivot] = []

    def add_pivot(self, pivot: Pivot):
        self.pivots.append(pivot)

    def compute(
        self,
        current_price: float,
        now: datetime,
        provisional: Optional[Pivot] = None,
        top_n: int = 10,
        grid_step: float = 0.5,
    ) -> list[Zone]:
        if not self.pivots:
            return []

        # Decay weights and prune dead pivots
        prices, weights = [], []
        for p in self.pivots:
            age = max(0.0, (now - p.time).total_seconds())
            w = max(1.0, p.swing_size / 10.0) * np.exp(-self.decay_lambda * age)
            if w < 0.05:
                continue
            prices.append(p.price)
            weights.append(w)

        # Add the provisional (live) extremum as a soft vote
        if provisional is not None:
            prices.append(provisional.price)
            weights.append(max(1.0, provisional.swing_size / 10.0) * self.provisional_weight)

        if not prices:
            return []

        prices  = np.asarray(prices)
        weights = np.asarray(weights)

        # Evaluate KDE on a fine price grid
        p_lo = float(prices.min()) - 4 * self.bw
        p_hi = float(prices.max()) + 4 * self.bw
        grid = np.arange(p_lo, p_hi, grid_step)

        # vectorized sum of Gaussians
        diffs   = (grid[:, None] - prices[None, :]) / self.bw
        density = (weights[None, :] * np.exp(-0.5 * diffs ** 2)).sum(axis=1)

        # Find local maxima above threshold
        thresh = density.max() * self.min_rel
        peaks_idx = []
        for i in range(1, len(density) - 1):
            if density[i] >= density[i - 1] and density[i] >= density[i + 1] \
               and density[i] >= thresh:
                peaks_idx.append(i)

        if not peaks_idx:
            return []

        # Build zones
        zones: list[Zone] = []
        for i in peaks_idx:
            price = float(grid[i])
            # Count contributing pivots (within 2σ)
            n_contrib = int(np.sum(np.abs(prices - price) <= 2 * self.bw))
            zones.append(Zone(
                price=round(price, 2),
                lower=round(price - self.zone_hw, 2),
                upper=round(price + self.zone_hw, 2),
                strength=float(density[i]),
                type="Support" if price < current_price else "Resistance",
                n_pivots=n_contrib,
            ))

        # Merge zones that overlap
        zones = self._merge(zones)

        # Normalize strength to 0–100
        if zones:
            s_max = max(z.strength for z in zones)
            for z in zones:
                z.strength = round(100.0 * z.strength / s_max, 1)

        zones.sort(key=lambda z: z.strength, reverse=True)
        return zones[:top_n]

    def _merge(self, zones: list[Zone]) -> list[Zone]:
        if not zones:
            return zones
        zones.sort(key=lambda z: z.price)
        out: list[Zone] = []
        for z in zones:
            if out and (z.price - out[-1].price) < self.merge_dist:
                # Merge into stronger
                stronger = out[-1] if out[-1].strength >= z.strength else z
                weaker   = z       if stronger is out[-1] else out[-1]
                merged = Zone(
                    price=round((out[-1].price + z.price) / 2, 2),
                    lower=round(min(out[-1].lower, z.lower), 2),
                    upper=round(max(out[-1].upper, z.upper), 2),
                    strength=stronger.strength + 0.3 * weaker.strength,
                    type=stronger.type,
                    n_pivots=out[-1].n_pivots + z.n_pivots,
                )
                out[-1] = merged
            else:
                out.append(z)
        return out


# ════════════════════════════════════════════════════════════════
# 4. END-TO-END ENGINE
# ════════════════════════════════════════════════════════════════

class RealtimeSREngine:
    """
    Single entry-point: feed it ticks, query zones any time.

        engine = RealtimeSREngine()
        for ts, px in tick_stream:
            engine.on_tick(px, ts)
        zones = engine.zones(current_price=px, now=ts)
    """

    def __init__(
        self,
        range_size:         float = 5.0,
        reversal_threshold: float = 15.0,
        bandwidth:          float = 8.0,
        zone_half_width:    float = 12.0,
        half_life_min:      float = 120.0,
    ):
        self.bars  = RangeBarBuilder(range_size=range_size)
        self.zz    = ZigZagPivotDetector(reversal_threshold=reversal_threshold)
        self.zmap  = DensityZoneMap(
            bandwidth=bandwidth,
            zone_half_width=zone_half_width,
            half_life_min=half_life_min,
        )
        self.last_bar:   Optional[RangeBar] = None
        self.last_pivot: Optional[Pivot]    = None

    def on_tick(self, price: float, ts: datetime):
        bar = self.bars.on_tick(price, ts)
        if bar is not None:
            self.last_bar = bar
            pivot = self.zz.on_bar(bar)
            if pivot is not None:
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
# 5. DEMO — simulate ticks from 5m OHLC if no live feed available
# ════════════════════════════════════════════════════════════════

def simulate_ticks_from_ohlc(df, ticks_per_bar: int = 20):
    """
    Crude tick simulator from 5m OHLC.
    For each bar, walks: open → high → low → close (or O→L→H→C).
    Good enough to validate the pipeline; replace with a real feed in prod.
    """
    rng = np.random.default_rng(42)
    for ts, row in df.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        # Pick path
        if rng.random() < 0.5:
            path = [o, h, l, c]
        else:
            path = [o, l, h, c]
        # Interpolate ticks along the path
        seg_len = ticks_per_bar // 3
        full = []
        for a, b in zip(path[:-1], path[1:]):
            full.extend(np.linspace(a, b, seg_len, endpoint=False))
        full.append(c)
        # Spread across bar duration
        step = timedelta(minutes=5) / len(full)
        for i, px in enumerate(full):
            yield ts + i * step, float(px)


if __name__ == "__main__":
    # Example: hook into your existing data loader
    import yfinance as yf

    print("Fetching recent 5m data for simulation...")
    df = yf.download("^NSEI", period="1d", interval="5m",
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    try:
        df.index = df.index.tz_convert("Asia/Kolkata")
    except TypeError:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    df = df.between_time("09:15", "15:30").dropna()
    print(f"  {len(df)} bars\n")

    # Run the engine on simulated ticks
    engine = RealtimeSREngine(
        range_size         = 4.0,   # 4-pt range bars
        reversal_threshold = 12.0,  # 12-pt swing reversal
        bandwidth          = 7.0,   # KDE smoothing
        zone_half_width    = 12.0,  # 24-pt visual zone band
        half_life_min      = 120.0, # 2-hour relevance half-life
    )

    last_ts, last_px = None, None
    n_ticks = 0
    for ts, px in simulate_ticks_from_ohlc(df, ticks_per_bar=15):
        engine.on_tick(px, ts)
        last_ts, last_px = ts, px
        n_ticks += 1

    print(f"Processed {n_ticks} simulated ticks")
    print(f"Built  {len(engine.bars.__dict__) and len(engine.zz.pivots)} confirmed pivots")
    print(f"Current price: {last_px:.2f} @ {last_ts}\n")

    zones = engine.zones(current_price=last_px, now=last_ts, top_n=10)

    print("═" * 80)
    print(f"  LIVE S/R ZONES  —  CMP: {last_px:.2f}")
    print("═" * 80)
    print(f"  {'Type':<12} {'Price':>8}  {'Range':>18}  {'Str':>5}  {'Pivots':>6}")
    print("  " + "─" * 70)
    for z in zones:
        sym = "▲ SUPPORT" if z.type == "Support" else "▼ RESISTANCE"
        rng_str = f"{z.lower:.1f}–{z.upper:.1f}"
        print(f"  {sym:<12} {z.price:>8.2f}  {rng_str:>18}  "
              f"{z.strength:>5.1f}  {z.n_pivots:>6}")
    print("═" * 80)