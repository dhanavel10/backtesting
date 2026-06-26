"""
swing_engine.py
===============
Causal (zero-lookahead) pivot detection for realtime S/R systems.

Design constraint: at bar N, we may only use bars 0..N to make decisions.
No future bar data is ever accessed. This is critical for live trading.

Two pivot styles are supported:

  1. FIXED WINDOW (left_bars only)
     A pivot high is confirmed as soon as we see `left_bars` consecutive
     bars where no high exceeded the candidate. Faster confirmation but
     slightly more noise.

  2. PERCENTAGE REVERSAL (recommended for S/R zone accuracy)
     A pivot high is confirmed when price *reverses* at least `rev_pct`%
     from the candidate peak. This directly encodes the "if price falls
     N% from a recent high, that high becomes resistance" rule requested
     in the spec. Much more meaningful for options buying entries.

Both methods emit PivotEvent objects via registered callbacks.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, List, Optional

from tick_processor import CandleBar

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────

class PivotType(Enum):
    HIGH = "high"
    LOW  = "low"


@dataclass
class PivotEvent:
    symbol:     str
    pivot_type: PivotType
    price:      float          # exact wick price of the pivot
    bar_index:  int            # bar index when the pivot *occurred*
    confirm_bar_index: int     # bar index when we *confirmed* it
    ts:         float          # epoch of the pivot bar open
    session:    str            # YYYY-MM-DD trading session
    rev_pct:    float = 0.0   # actual reversal % that confirmed it
    method:     str = ""

    @property
    def label(self) -> str:
        return "PH" if self.pivot_type == PivotType.HIGH else "PL"


# ─────────────────────────────────────────────────────────────────
# Fixed-window pivot detector (original approach, causal version)
# ─────────────────────────────────────────────────────────────────

class FixedWindowPivotDetector:
    """
    Confirms a pivot high/low once `left_bars` bars have passed
    without a higher high (for pivot high) or lower low (for pivot low).

    This is the causal equivalent of the historical argrelextrema approach.
    Confirmation lag = left_bars candles.
    """

    def __init__(
        self,
        symbol:    str,
        left_bars: int = 10,
    ):
        self.symbol    = symbol
        self.left_bars = left_bars
        self._bars:    Deque[CandleBar] = deque(maxlen=left_bars + 1)
        self._callbacks: List[Callable[[PivotEvent], None]] = []

    def on_pivot(self, fn: Callable[[PivotEvent], None]):
        self._callbacks.append(fn)

    def _emit(self, event: PivotEvent):
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Pivot callback error: {e}")

    def process_bar(self, bar: CandleBar):
        self._bars.append(bar)
        if len(self._bars) < self.left_bars + 1:
            return

        bars   = list(self._bars)
        center = bars[0]        # oldest bar in window = candidate
        window = bars[1:]       # the `left_bars` confirmers after candidate

        session = str(bar.dt_open.date())

        # Pivot HIGH: candidate high strictly > all highs in the window after it
        if center.high > max(b.high for b in window):
            self._emit(PivotEvent(
                symbol=self.symbol,
                pivot_type=PivotType.HIGH,
                price=center.high,
                bar_index=center.bar_index,
                confirm_bar_index=bar.bar_index,
                ts=center.ts_open,
                session=session,
                method="fixed_window",
            ))

        # Pivot LOW: candidate low strictly < all lows in the window after it
        if center.low < min(b.low for b in window):
            self._emit(PivotEvent(
                symbol=self.symbol,
                pivot_type=PivotType.LOW,
                price=center.low,
                bar_index=center.bar_index,
                confirm_bar_index=bar.bar_index,
                ts=center.ts_open,
                session=session,
                method="fixed_window",
            ))


# ─────────────────────────────────────────────────────────────────
# Percentage-Reversal Pivot Detector  ← RECOMMENDED
# ─────────────────────────────────────────────────────────────────

class ReversalPivotDetector:
    """
    A pivot HIGH is confirmed when price *falls* at least `rev_pct`%
    from the running peak. The peak level itself becomes the S/R price.

    A pivot LOW is confirmed when price *rises* at least `rev_pct`%
    from the running trough.

    Why this is better for options buying:
      - Directly answers "where did price reverse significantly?"
      - The reversal magnitude is tunable (e.g. 0.3% on Nifty ≈ 72 pts)
      - No look-ahead lag — confirmation is immediate once reversal occurs
      - Maps naturally to "recent high/low becomes resistance/support"

    Internal state machine (per direction):
      TRENDING_UP   → tracking peak_high; emit pivot_high when drop >= rev_pct
      TRENDING_DOWN → tracking trough_low; emit pivot_low when rise >= rev_pct
    """

    def __init__(
        self,
        symbol:         str,
        rev_pct:        float = 0.30,    # 0.30% reversal to confirm pivot
        min_swing_pts:  float = 20.0,    # minimum absolute point swing
    ):
        self.symbol        = symbol
        self.rev_pct       = rev_pct
        self.min_swing_pts = min_swing_pts

        self._peak_high:   Optional[float] = None
        self._peak_bar:    Optional[CandleBar] = None
        self._trough_low:  Optional[float] = None
        self._trough_bar:  Optional[CandleBar] = None

        self._last_pivot_type: Optional[PivotType] = None
        self._bar_count:  int = 0
        self._callbacks: List[Callable[[PivotEvent], None]] = []

        # Track recent pivots for zone engine
        self.recent_highs: Deque[PivotEvent] = deque(maxlen=50)
        self.recent_lows:  Deque[PivotEvent] = deque(maxlen=50)

    def on_pivot(self, fn: Callable[[PivotEvent], None]):
        self._callbacks.append(fn)

    def _emit(self, event: PivotEvent):
        if event.pivot_type == PivotType.HIGH:
            self.recent_highs.append(event)
        else:
            self.recent_lows.append(event)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Pivot callback error: {e}")

    def process_bar(self, bar: CandleBar):
        self._bar_count += 1
        session = str(bar.dt_open.date())

        # ── Initialize ──────────────────────────────────────
        if self._peak_high is None:
            self._peak_high   = bar.high
            self._peak_bar    = bar
            self._trough_low  = bar.low
            self._trough_bar  = bar
            return

        # ── Update running peak and trough ──────────────────
        if bar.high >= self._peak_high:
            self._peak_high = bar.high
            self._peak_bar  = bar

        if bar.low <= self._trough_low:
            self._trough_low = bar.low
            self._trough_bar = bar

        current_close = bar.close

        # ── Check for PIVOT HIGH confirmation ───────────────
        # Price must have fallen >= rev_pct% AND >= min_swing_pts from peak
        if self._peak_bar is not None and self._peak_bar.bar_index < bar.bar_index:
            drop_pct = (self._peak_high - current_close) / self._peak_high * 100
            drop_pts = self._peak_high - current_close

            if drop_pct >= self.rev_pct and drop_pts >= self.min_swing_pts:
                # Only emit if this is a new pivot (not same bar as last high pivot)
                if (self._last_pivot_type != PivotType.HIGH or
                        self._peak_bar.bar_index > self.recent_highs[-1].bar_index
                        if self.recent_highs else True):

                    event = PivotEvent(
                        symbol=self.symbol,
                        pivot_type=PivotType.HIGH,
                        price=self._peak_high,
                        bar_index=self._peak_bar.bar_index,
                        confirm_bar_index=bar.bar_index,
                        ts=self._peak_bar.ts_open,
                        session=str(self._peak_bar.dt_open.date()),
                        rev_pct=round(drop_pct, 3),
                        method="reversal",
                    )
                    self._emit(event)
                    self._last_pivot_type = PivotType.HIGH
                    logger.debug(
                        f"[PIVOT HIGH] {self.symbol} @ {self._peak_high:.2f}  "
                        f"drop={drop_pct:.2f}% ({drop_pts:.0f}pts)  "
                        f"confirmed at bar#{bar.bar_index}"
                    )
                    # Reset trough tracking from current level
                    self._trough_low = bar.low
                    self._trough_bar = bar
                    # Reset peak to current close so next peak starts fresh
                    self._peak_high = bar.close
                    self._peak_bar  = bar

        # ── Check for PIVOT LOW confirmation ────────────────
        if self._trough_bar is not None and self._trough_bar.bar_index < bar.bar_index:
            rise_pct = (current_close - self._trough_low) / self._trough_low * 100
            rise_pts = current_close - self._trough_low

            if rise_pct >= self.rev_pct and rise_pts >= self.min_swing_pts:
                if (self._last_pivot_type != PivotType.LOW or
                        self._trough_bar.bar_index > self.recent_lows[-1].bar_index
                        if self.recent_lows else True):

                    event = PivotEvent(
                        symbol=self.symbol,
                        pivot_type=PivotType.LOW,
                        price=self._trough_low,
                        bar_index=self._trough_bar.bar_index,
                        confirm_bar_index=bar.bar_index,
                        ts=self._trough_bar.ts_open,
                        session=str(self._trough_bar.dt_open.date()),
                        rev_pct=round(rise_pct, 3),
                        method="reversal",
                    )
                    self._emit(event)
                    self._last_pivot_type = PivotType.LOW
                    logger.debug(
                        f"[PIVOT LOW] {self.symbol} @ {self._trough_low:.2f}  "
                        f"rise={rise_pct:.2f}% ({rise_pts:.0f}pts)  "
                        f"confirmed at bar#{bar.bar_index}"
                    )
                    # Reset
                    self._peak_high = bar.high
                    self._peak_bar  = bar
                    self._trough_low = bar.close
                    self._trough_bar = bar


# ─────────────────────────────────────────────────────────────────
# Composite detector — runs BOTH methods and deduplicates
# ─────────────────────────────────────────────────────────────────

class CompositePivotDetector:
    """
    Runs FixedWindow + Reversal detectors in parallel.
    Deduplicates pivots within `dedup_pts` points of each other.
    Provides a single stream of high-confidence pivots.
    """

    def __init__(
        self,
        symbol:        str,
        left_bars:     int   = 10,
        rev_pct:       float = 0.30,
        min_swing_pts: float = 20.0,
        dedup_pts:     float = 15.0,
    ):
        self.symbol    = symbol
        self.dedup_pts = dedup_pts

        self.fixed    = FixedWindowPivotDetector(symbol, left_bars)
        self.reversal = ReversalPivotDetector(symbol, rev_pct, min_swing_pts)

        self._callbacks: List[Callable[[PivotEvent], None]] = []
        self._seen_highs: Deque[float] = deque(maxlen=30)
        self._seen_lows:  Deque[float] = deque(maxlen=30)

        self.fixed.on_pivot(self._on_raw_pivot)
        self.reversal.on_pivot(self._on_raw_pivot)

        # Expose recent pivots from reversal detector (primary)
        self.recent_highs = self.reversal.recent_highs
        self.recent_lows  = self.reversal.recent_lows

    def on_pivot(self, fn: Callable[[PivotEvent], None]):
        self._callbacks.append(fn)

    def _on_raw_pivot(self, event: PivotEvent):
        """Dedup and forward."""
        bucket = self._seen_highs if event.pivot_type == PivotType.HIGH else self._seen_lows
        for prev in bucket:
            if abs(event.price - prev) <= self.dedup_pts:
                return   # already emitted a nearby pivot
        bucket.append(event.price)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Composite pivot callback error: {e}")

    def process_bar(self, bar: CandleBar):
        self.fixed.process_bar(bar)
        self.reversal.process_bar(bar)
