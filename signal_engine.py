"""
signal_engine.py
================
Generates actionable signals from zone interactions.

Signal types for intraday options buying:
  - REJECTION_LONG   : bullish rejection from support → BUY CE / exit PE
  - REJECTION_SHORT  : bearish rejection from resistance → BUY PE / exit CE
  - BREAKOUT_LONG    : confirmed breakout above resistance → BUY CE
  - BREAKOUT_SHORT   : confirmed breakdown below support → BUY PE
  - ZONE_APPROACH    : price approaching key zone (early warning)
  - ZONE_TEST        : first wick into zone this session

Each signal carries:
  - strength (zone strength at time of signal)
  - confirmation level (how many criteria met)
  - suggested entry range
  - suggested stop (zone midpoint or far edge)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict
from collections import deque
import time

from tick_processor import CandleBar, Tick
from zone_engine import SRZone, ZoneEngine

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Signal Data Structure
# ─────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    signal_id:    int
    signal_type:  str       # REJECTION_LONG | REJECTION_SHORT | BREAKOUT_LONG | BREAKOUT_SHORT | ZONE_APPROACH | ZONE_TEST
    symbol:       str
    price:        float     # trigger price
    zone_price:   float     # associated zone center
    zone_id:      int
    zone_type:    str       # Support | Resistance
    zone_strength: float
    ts:           float
    bar_index:    int
    session:      str

    # Options guidance
    action:       str = ""  # "BUY CE" | "BUY PE" | "WATCH"
    entry_range:  tuple = field(default_factory=tuple)  # (low, high)
    stop_level:   float = 0.0
    target_level: float = 0.0

    # Confirmation quality
    confirmations: int   = 0   # 0-5: how many criteria confirmed
    confidence:    str   = ""  # LOW | MEDIUM | HIGH

    def to_dict(self) -> dict:
        return {
            "signal_id":    self.signal_id,
            "signal_type":  self.signal_type,
            "symbol":       self.symbol,
            "price":        round(self.price, 2),
            "zone_price":   round(self.zone_price, 2),
            "zone_id":      self.zone_id,
            "zone_type":    self.zone_type,
            "zone_strength": round(self.zone_strength, 1),
            "ts":           self.ts,
            "bar_index":    self.bar_index,
            "action":       self.action,
            "entry_low":    round(self.entry_range[0], 2) if self.entry_range else 0,
            "entry_high":   round(self.entry_range[1], 2) if self.entry_range else 0,
            "stop_level":   round(self.stop_level, 2),
            "target_level": round(self.target_level, 2),
            "confirmations": self.confirmations,
            "confidence":   self.confidence,
            "session":      self.session,
        }


# ─────────────────────────────────────────────────────────────────
# Rejection Criteria Checker
# ─────────────────────────────────────────────────────────────────

class RejectionChecker:
    """
    Checks a candle for bullish/bearish rejection patterns at a zone.

    Criteria for STRONG rejection (each adds 1 to confirmation count):
      1. Wick extended into zone (mandatory)
      2. Close outside zone on correct side
      3. Wick-to-body ratio >= min_wick_body_ratio
      4. Candle body size >= min_body_pts
      5. Previous candle also showed rejection
      6. Volume spike (if volume data meaningful)
    """

    def __init__(
        self,
        min_wick_body_ratio: float = 1.5,   # wick >= 1.5x body size
        min_body_pts:        float = 5.0,
        approach_pts:        float = 30.0,  # "approaching" threshold
    ):
        self.min_wick_body_ratio = min_wick_body_ratio
        self.min_body_pts        = min_body_pts
        self.approach_pts        = approach_pts

    def check_bullish_rejection(self, bar: CandleBar, zone: SRZone) -> dict:
        """Check for bullish rejection (bounce from support)."""
        result = {"confirmed": False, "confirmations": 0, "details": []}

        # 1. Wick must have entered support zone
        if not (bar.low <= zone.upper and bar.low >= zone.lower - zone.half_band):
            return result
        result["confirmations"] += 1
        result["details"].append("wick_entered_support")

        # 2. Close above zone upper
        if bar.close > zone.upper:
            result["confirmations"] += 1
            result["details"].append("close_above_zone")

        # 3. Lower wick dominates (bullish rejection candle)
        body      = abs(bar.close - bar.open)
        lower_wick = min(bar.open, bar.close) - bar.low
        if body > 0 and lower_wick / body >= self.min_wick_body_ratio:
            result["confirmations"] += 1
            result["details"].append("long_lower_wick")

        # 4. Body size meaningful
        if body >= self.min_body_pts:
            result["confirmations"] += 1
            result["details"].append("meaningful_body")

        # 5. Closed bullish
        if bar.close > bar.open:
            result["confirmations"] += 1
            result["details"].append("bullish_close")

        result["confirmed"] = result["confirmations"] >= 2
        return result

    def check_bearish_rejection(self, bar: CandleBar, zone: SRZone) -> dict:
        """Check for bearish rejection (reversal from resistance)."""
        result = {"confirmed": False, "confirmations": 0, "details": []}

        # 1. Wick must have entered resistance zone
        if not (bar.high >= zone.lower and bar.high <= zone.upper + zone.half_band):
            return result
        result["confirmations"] += 1
        result["details"].append("wick_entered_resistance")

        # 2. Close below zone lower
        if bar.close < zone.lower:
            result["confirmations"] += 1
            result["details"].append("close_below_zone")

        # 3. Upper wick dominates (bearish rejection candle)
        body       = abs(bar.close - bar.open)
        upper_wick = bar.high - max(bar.open, bar.close)
        if body > 0 and upper_wick / body >= self.min_wick_body_ratio:
            result["confirmations"] += 1
            result["details"].append("long_upper_wick")

        # 4. Body size meaningful
        if body >= self.min_body_pts:
            result["confirmations"] += 1
            result["details"].append("meaningful_body")

        # 5. Closed bearish
        if bar.close < bar.open:
            result["confirmations"] += 1
            result["details"].append("bearish_close")

        result["confirmed"] = result["confirmations"] >= 2
        return result

    def is_approaching(self, price: float, zone: SRZone) -> bool:
        return abs(price - zone.price) <= self.approach_pts


# ─────────────────────────────────────────────────────────────────
# Breakout Checker
# ─────────────────────────────────────────────────────────────────

class BreakoutChecker:
    """
    Confirms breakout only when BOTH conditions are met:
      1. Close >= N points beyond zone edge
      2. Sustained: next candle does NOT return into zone
    """

    def __init__(
        self,
        breakout_pts:   float = 15.0,   # points beyond zone edge
        min_zone_tests: int   = 2,      # zone must have been tested before breakout
    ):
        self.breakout_pts   = breakout_pts
        self.min_zone_tests = min_zone_tests
        self._pending:      Dict[int, dict] = {}   # zone_id → pending breakout

    def check_breakout(self, bar: CandleBar, zone: SRZone) -> dict:
        result = {"type": None, "confirmed": False, "confirmations": 0}

        if zone.wick_touches < self.min_zone_tests:
            return result

        # Resistance breakout
        if zone.zone_type == "Resistance":
            if bar.close > zone.upper + self.breakout_pts:
                zid = zone.zone_id
                if zid not in self._pending:
                    self._pending[zid] = {"bar_index": bar.bar_index, "type": "BREAKOUT_LONG"}
                    result["type"] = "BREAKOUT_LONG"
                    result["confirmations"] = 1
                else:
                    # Second bar confirms
                    result["type"] = "BREAKOUT_LONG"
                    result["confirmed"] = True
                    result["confirmations"] = 2
                    del self._pending[zid]
            else:
                # Reset if price retreated
                if zone.zone_id in self._pending:
                    del self._pending[zone.zone_id]

        # Support breakdown
        elif zone.zone_type == "Support":
            if bar.close < zone.lower - self.breakout_pts:
                zid = zone.zone_id
                if zid not in self._pending:
                    self._pending[zid] = {"bar_index": bar.bar_index, "type": "BREAKOUT_SHORT"}
                    result["type"] = "BREAKOUT_SHORT"
                    result["confirmations"] = 1
                else:
                    result["type"] = "BREAKOUT_SHORT"
                    result["confirmed"] = True
                    result["confirmations"] = 2
                    del self._pending[zid]
            else:
                if zone.zone_id in self._pending:
                    del self._pending[zone.zone_id]

        return result


# ─────────────────────────────────────────────────────────────────
# Signal Engine
# ─────────────────────────────────────────────────────────────────

class SignalEngine:

    def __init__(
        self,
        symbol:              str,
        zone_engine:         ZoneEngine,
        min_zone_strength:   float = 20.0,
        approach_pts:        float = 30.0,
        max_signals_history: int   = 200,
    ):
        self.symbol            = symbol
        self.zone_engine       = zone_engine
        self.min_zone_strength = min_zone_strength
        self.approach_pts      = approach_pts

        self.rejection_checker = RejectionChecker(approach_pts=approach_pts)
        self.breakout_checker  = BreakoutChecker()

        self._signal_id:   int = 0
        self._callbacks:   List[Callable[[Signal], None]] = []
        self.signals:      deque = deque(maxlen=max_signals_history)

        # Track approach alerts to avoid repeating
        self._alerted_approaches: Dict[int, int] = {}   # zone_id → bar_index

    def on_signal(self, fn: Callable[[Signal], None]):
        self._callbacks.append(fn)

    def _emit(self, signal: Signal):
        self.signals.append(signal)
        for cb in self._callbacks:
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    def _new_signal(self, **kwargs) -> Signal:
        sig = Signal(signal_id=self._signal_id, **kwargs)
        self._signal_id += 1
        return sig

    def _confidence(self, confirmations: int) -> str:
        if confirmations >= 4: return "HIGH"
        if confirmations >= 2: return "MEDIUM"
        return "LOW"

    def process_candle(self, bar: CandleBar):
        """
        Run all signal checks against every active zone.
        Called once per closed candle.
        """
        current_price = bar.close
        session       = str(bar.dt_open.date())

        zones = self.zone_engine.get_active_zones(
            current_price, min_strength=self.min_zone_strength
        )

        for zone in zones:
            # ── Zone Approach Warning ────────────────────────
            dist = abs(current_price - zone.price)
            if dist <= self.approach_pts:
                last_alert = self._alerted_approaches.get(zone.zone_id, -999)
                if bar.bar_index - last_alert > 10:   # don't repeat for 10 bars
                    self._alerted_approaches[zone.zone_id] = bar.bar_index
                    action = "WATCH CE" if zone.zone_type == "Support" else "WATCH PE"
                    sig = self._new_signal(
                        signal_type="ZONE_APPROACH",
                        symbol=self.symbol,
                        price=current_price,
                        zone_price=zone.price,
                        zone_id=zone.zone_id,
                        zone_type=zone.zone_type,
                        zone_strength=zone.strength,
                        ts=bar.ts_close,
                        bar_index=bar.bar_index,
                        session=session,
                        action=action,
                        entry_range=(current_price - 20, current_price + 20),
                        stop_level=zone.price - zone.half_band * 2 if zone.zone_type == "Support"
                                   else zone.price + zone.half_band * 2,
                        confirmations=1,
                        confidence="LOW",
                    )
                    self._emit(sig)
                    logger.info(f"[SIGNAL] ZONE_APPROACH  {self.symbol} → {zone.zone_type} "
                                f"@ {zone.price:.0f}  dist={dist:.0f}pts")

            # ── Rejection Signals ────────────────────────────
            if zone.zone_type == "Support":
                rej = self.rejection_checker.check_bullish_rejection(bar, zone)
                if rej["confirmed"] and zone.strength >= self.min_zone_strength:
                    # Next resistance target
                    above, _ = self.zone_engine.get_nearest_zones(current_price, n_above=1, n_below=0)
                    target = above[0].price if above else current_price + 100

                    sig = self._new_signal(
                        signal_type="REJECTION_LONG",
                        symbol=self.symbol,
                        price=current_price,
                        zone_price=zone.price,
                        zone_id=zone.zone_id,
                        zone_type=zone.zone_type,
                        zone_strength=zone.strength,
                        ts=bar.ts_close,
                        bar_index=bar.bar_index,
                        session=session,
                        action="BUY CE",
                        entry_range=(bar.close, bar.close + 15),
                        stop_level=zone.lower - 10,
                        target_level=target,
                        confirmations=rej["confirmations"],
                        confidence=self._confidence(rej["confirmations"]),
                    )
                    self._emit(sig)
                    logger.info(f"[SIGNAL] REJECTION_LONG  {self.symbol} @ {current_price:.0f}  "
                                f"support={zone.price:.0f}  conf={rej['confirmations']}")

            elif zone.zone_type == "Resistance":
                rej = self.rejection_checker.check_bearish_rejection(bar, zone)
                if rej["confirmed"] and zone.strength >= self.min_zone_strength:
                    _, below = self.zone_engine.get_nearest_zones(current_price, n_above=0, n_below=1)
                    target = below[0].price if below else current_price - 100

                    sig = self._new_signal(
                        signal_type="REJECTION_SHORT",
                        symbol=self.symbol,
                        price=current_price,
                        zone_price=zone.price,
                        zone_id=zone.zone_id,
                        zone_type=zone.zone_type,
                        zone_strength=zone.strength,
                        ts=bar.ts_close,
                        bar_index=bar.bar_index,
                        session=session,
                        action="BUY PE",
                        entry_range=(bar.close - 15, bar.close),
                        stop_level=zone.upper + 10,
                        target_level=target,
                        confirmations=rej["confirmations"],
                        confidence=self._confidence(rej["confirmations"]),
                    )
                    self._emit(sig)
                    logger.info(f"[SIGNAL] REJECTION_SHORT  {self.symbol} @ {current_price:.0f}  "
                                f"resistance={zone.price:.0f}  conf={rej['confirmations']}")

            # ── Breakout Signals ─────────────────────────────
            bo = self.breakout_checker.check_breakout(bar, zone)
            if bo["type"] and bo["confirmed"] and zone.strength >= self.min_zone_strength:
                action = "BUY CE" if bo["type"] == "BREAKOUT_LONG" else "BUY PE"
                sig = self._new_signal(
                    signal_type=bo["type"],
                    symbol=self.symbol,
                    price=current_price,
                    zone_price=zone.price,
                    zone_id=zone.zone_id,
                    zone_type=zone.zone_type,
                    zone_strength=zone.strength,
                    ts=bar.ts_close,
                    bar_index=bar.bar_index,
                    session=session,
                    action=action,
                    entry_range=(bar.close - 10, bar.close + 10),
                    stop_level=zone.price,
                    confirmations=bo["confirmations"],
                    confidence=self._confidence(bo["confirmations"]),
                )
                self._emit(sig)
                logger.info(f"[SIGNAL] {bo['type']}  {self.symbol} @ {current_price:.0f}  "
                            f"zone={zone.price:.0f}")

    def get_recent_signals(self, n: int = 20, signal_type: str = None) -> List[Signal]:
        sigs = list(self.signals)
        if signal_type:
            sigs = [s for s in sigs if s.signal_type == signal_type]
        return sigs[-n:]
