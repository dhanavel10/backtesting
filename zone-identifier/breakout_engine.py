"""
breakout_engine.py
==================
Intraday 5m Candle Breakout Strategy Engine.

Solves Blockers 4 & 5 (entry lag, no trade management):

Entry Logic
-----------
  Breakout is triggered when a 5m candle CLOSES beyond a zone boundary:
    • Long  : close > zone.upper  AND  (close - zone.upper) >= min_breakout_pts
    • Short : close < zone.lower  AND  (zone.lower - close) >= min_breakout_pts

  To filter fakeouts, a CONFIRMATION candle is required (default: 1):
    • The candle after the breakout candle must also close in breakout direction.
    • This adds only 5 minutes of lag but cuts false signals significantly.

  The "10-minute lag" from delayed pivots is addressed by using the PRE-BUILT
  historical zone map (built at market open) for entry signals. Delayed pivot
  confirmation is used only for ZONE MAP REFRESH — not for the entry trigger
  itself. This gives the best of both worlds:
    • No entry lag (historical zones available from bar 1)
    • Gradually improving zone quality as new pivots are confirmed live

Stop-Loss
---------
  Long  : sl = zone.upper - sl_pts   (below upper band — zone becomes new support)
  Short : sl = zone.lower + sl_pts   (above lower band — zone becomes new resistance)
  Default sl_pts = zone_half_band (i.e., if price returns to zone midpoint, exit)

Target
------
  Target = next zone level in breakout direction.
  If no next zone exists within max_target_pts, use fixed R:R multiple.
  Default R:R = 2.0 (target = 2× risk from entry)

Trade State Machine
-------------------
  IDLE → WAITING_CONFIRM → IN_TRADE → IDLE
  Parallel long + short signals are not allowed; first signal wins.

Options Context (advisory only — execution not in scope)
---------
  After a signal fires, the engine suggests:
    • Instrument: CE for long / PE for short
    • Strike: nearest ATM or one OTM step based on dist to target
    • Max premium: entry_price × 0.03 (3% of Nifty as premium cap)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Optional, Callable
from enum import Enum


# ════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════

class TradeDirection(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


class TradeState(Enum):
    IDLE             = "IDLE"
    WAITING_CONFIRM  = "WAITING_CONFIRM"
    IN_TRADE         = "IN_TRADE"


@dataclass
class Signal:
    direction:      TradeDirection
    zone_price:     float
    zone_upper:     float
    zone_lower:     float
    zone_strength:  float
    entry_price:    float         # expected fill ~ candle close
    sl:             float
    target:         float
    risk_pts:       float
    reward_pts:     float
    rr_ratio:       float
    timestamp:      datetime
    confirm_needed: bool = True
    confirmed:      bool = False
    # Options advisory
    option_type:    str  = ""
    option_strike:  int  = 0

    def __str__(self):
        arrow = "▲ LONG" if self.direction == TradeDirection.LONG else "▼ SHORT"
        conf  = "⚡ CONFIRMED" if self.confirmed else "⏳ WAITING CONFIRM"
        return (
            f"[{conf}] {arrow}  Entry≈{self.entry_price:.0f}  "
            f"SL={self.sl:.0f}  TGT={self.target:.0f}  "
            f"R:R={self.rr_ratio:.1f}  Zone={self.zone_price:.0f}  "
            f"Str={self.zone_strength:.0f}"
        )


@dataclass
class TradeResult:
    signal:     Signal
    exit_price: float
    exit_time:  datetime
    exit_reason: str   # "TARGET" | "SL" | "EOD" | "MANUAL"
    pnl_pts:    float
    pnl_pct:    float

    def __str__(self):
        emoji = "✅" if self.pnl_pts > 0 else "❌"
        return (
            f"{emoji} {self.exit_reason}  "
            f"Entry={self.signal.entry_price:.0f}  "
            f"Exit={self.exit_price:.0f}  "
            f"P&L={self.pnl_pts:+.0f} pts  "
            f"({self.pnl_pct:+.2f}%)"
        )


# ════════════════════════════════════════════════════════════════
# BREAKOUT ENGINE
# ════════════════════════════════════════════════════════════════

class BreakoutEngine:
    """
    Processes confirmed 5m candles, checks against active S/R zones,
    and emits buy/sell signals.

    Parameters
    ----------
    zone_engine         : ZoneEngine instance (pre-loaded with history)
    min_breakout_pts    : minimum close beyond zone boundary to trigger signal
    confirm_candles     : number of extra candles needed to confirm breakout
    sl_pts              : stop-loss distance from zone boundary
    rr_ratio            : used when no next zone exists for target
    max_target_pts      : cap on target distance (avoid chasing distant zones)
    eod_exit_time       : force-exit all trades at this time (default 15:15)
    on_signal           : callback when a new signal (pending) is generated
    on_signal_confirmed : callback when signal is confirmed
    on_trade_closed     : callback when a trade exits
    """

    EOD_EXIT = dtime(15, 15)

    def __init__(
        self,
        zone_engine,
        min_breakout_pts:    float = 20.0,
        confirm_candles:     int   = 1,
        sl_pts:              float = None,
        rr_ratio:            float = 2.0,
        max_target_pts:      float = 200.0,
        invalidation_margin: float = 25.0,   # pts beyond zone before it's wiped
        on_signal:           Optional[Callable] = None,
        on_signal_confirmed: Optional[Callable] = None,
        on_trade_closed:     Optional[Callable] = None,
    ):
        self.ze                  = zone_engine
        self.min_breakout_pts    = min_breakout_pts
        self.confirm_candles     = confirm_candles
        self._sl_pts             = sl_pts
        self.rr_ratio            = rr_ratio
        self.max_target_pts      = max_target_pts
        self.invalidation_margin = invalidation_margin
        self.on_signal           = on_signal
        self.on_signal_confirmed = on_signal_confirmed
        self.on_trade_closed     = on_trade_closed

        self.state:              TradeState        = TradeState.IDLE
        self.pending_signal:     Optional[Signal]  = None
        self.active_signal:      Optional[Signal]  = None
        self._confirm_countdown: int               = 0
        self._prev_candle:       Optional[pd.Series] = None

        self.trade_log:          list[TradeResult] = []
        self.signal_log:         list[Signal]      = []

    # ── Main entry point ─────────────────────────────────────

    def on_candle(self, candle: pd.Series):
        """
        Called by CandleBuffer.on_candle_closed for every confirmed 5m bar.
        Drives the state machine.
        """
        ltp = float(candle["Close"])
        ts  = candle.get("timestamp", datetime.now())

        # EOD forced exit
        if isinstance(ts, datetime) and ts.time() >= self.EOD_EXIT:
            if self.state == TradeState.IN_TRADE:
                self._exit_trade(ltp, ts, "EOD")
            self.state = TradeState.IDLE
            self._prev_candle = candle
            return

        # ── Critical order: check breakout FIRST, then invalidate ──
        # If we invalidate before checking, the zone that triggered
        # the breakout gets wiped and the signal never fires.
        if self.state == TradeState.IDLE:
            self._check_for_breakout(candle, ltp, ts)
        elif self.state == TradeState.WAITING_CONFIRM:
            self._check_confirmation(candle, ltp, ts)
        elif self.state == TradeState.IN_TRADE:
            self._manage_open_trade(candle, ltp, ts)

        # Invalidate zones AFTER signal check, with wider margin
        # so zones aren't wiped on the first candle that crosses them
        self.ze.invalidate_broken_zones(ltp, margin=self.invalidation_margin)

        self._prev_candle = candle

    # ── State: IDLE — scan for breakouts ─────────────────────

    def _check_for_breakout(self, candle, ltp, ts):
        # Use ALL non-invalidated zones (active) for breakout detection
        sup, res = self.ze.get_nearest_zones(ltp, n=6)

        # Check resistance breakout (long) — price closed above upper band
        for _, zone in res.iterrows():
            clearance = ltp - zone["upper"]
            if clearance >= self.min_breakout_pts:
                sig = self._build_signal(candle, zone, TradeDirection.LONG, ltp, ts)
                if sig:
                    self._emit_signal(sig)
                    return   # one signal at a time

        # Check support breakdown (short) — price closed below lower band
        for _, zone in sup.iterrows():
            clearance = zone["lower"] - ltp
            if clearance >= self.min_breakout_pts:
                sig = self._build_signal(candle, zone, TradeDirection.SHORT, ltp, ts)
                if sig:
                    self._emit_signal(sig)
                    return

    def _build_signal(self, candle, zone, direction, ltp, ts) -> Optional[Signal]:
        sl_pts = self._sl_pts if self._sl_pts is not None else self.ze.zone_half_band

        if direction == TradeDirection.LONG:
            entry  = ltp
            sl     = zone["upper"] - sl_pts
            risk   = entry - sl
            # Target = next resistance zone, else fixed R:R
            next_target = self._next_zone_price(ltp, direction)
            if next_target and (next_target - entry) <= self.max_target_pts:
                target    = next_target
                reward    = target - entry
            else:
                reward = risk * self.rr_ratio
                target = entry + reward
            opt_type   = "CE"
            opt_strike = int(round(entry / 50) * 50)   # ATM rounded to 50

        else:  # SHORT
            entry  = ltp
            sl     = zone["lower"] + sl_pts
            risk   = sl - entry
            next_target = self._next_zone_price(ltp, direction)
            if next_target and (entry - next_target) <= self.max_target_pts:
                target = next_target
                reward = entry - target
            else:
                reward = risk * self.rr_ratio
                target = entry - reward
            opt_type   = "PE"
            opt_strike = int(round(entry / 50) * 50)

        if risk <= 0:
            return None

        rr = round(reward / risk, 2)

        # Quality gate: only take trades with R:R ≥ 1.5
        if rr < 1.5:
            return None

        return Signal(
            direction     = direction,
            zone_price    = zone["price"],
            zone_upper    = zone["upper"],
            zone_lower    = zone["lower"],
            zone_strength = zone["strength"],
            entry_price   = round(entry, 2),
            sl            = round(sl, 2),
            target        = round(target, 2),
            risk_pts      = round(risk, 2),
            reward_pts    = round(reward, 2),
            rr_ratio      = rr,
            timestamp     = ts,
            confirm_needed= self.confirm_candles > 0,
            option_type   = opt_type,
            option_strike = opt_strike,
        )

    def _next_zone_price(self, ltp, direction) -> Optional[float]:
        """Find the closest zone in the breakout direction."""
        active = self.ze.get_active_zones()
        if len(active) == 0:
            return None
        if direction == TradeDirection.LONG:
            candidates = active[active["price"] > ltp + 30]
        else:
            candidates = active[active["price"] < ltp - 30]

        if len(candidates) == 0:
            return None

        candidates = candidates.copy()
        candidates["dist"] = (candidates["price"] - ltp).abs()
        return float(candidates.sort_values("dist").iloc[0]["price"])

    # ── State: WAITING_CONFIRM ────────────────────────────────

    def _emit_signal(self, sig: Signal):
        self.pending_signal    = sig
        self.signal_log.append(sig)
        self._confirm_countdown = self.confirm_candles

        if self.confirm_candles == 0:
            sig.confirmed = True
            self.active_signal = sig
            self.state = TradeState.IN_TRADE
            if self.on_signal_confirmed:
                self.on_signal_confirmed(sig)
        else:
            self.state = TradeState.WAITING_CONFIRM
            if self.on_signal:
                self.on_signal(sig)
        print(f"[BreakoutEngine] 🔔 Signal: {sig}")

    def _check_confirmation(self, candle, ltp, ts):
        sig = self.pending_signal
        if sig is None:
            self.state = TradeState.IDLE
            return

        self._confirm_countdown -= 1
        in_direction = (
            (sig.direction == TradeDirection.LONG  and ltp > sig.zone_upper) or
            (sig.direction == TradeDirection.SHORT and ltp < sig.zone_lower)
        )

        if not in_direction:
            # Price has pulled back — cancel signal
            print(f"[BreakoutEngine] ⚠ Signal cancelled — price pulled back  "
                  f"(LTP={ltp:.0f}  zone_upper={sig.zone_upper:.0f}  "
                  f"zone_lower={sig.zone_lower:.0f})")
            self.pending_signal = None
            self.state = TradeState.IDLE
            return

        if self._confirm_countdown <= 0:
            sig.confirmed = True
            self.active_signal = sig
            self.state = TradeState.IN_TRADE
            print(f"[BreakoutEngine] ✅ Confirmed: {sig}")
            if self.on_signal_confirmed:
                self.on_signal_confirmed(sig)

    # ── State: IN_TRADE — manage open position ─────────────────

    def _manage_open_trade(self, candle, ltp, ts):
        sig = self.active_signal
        if sig is None:
            self.state = TradeState.IDLE
            return

        # Check SL hit (use candle Low/High for realistic fills)
        if sig.direction == TradeDirection.LONG:
            sl_hit  = float(candle["Low"])  <= sig.sl
            tgt_hit = float(candle["High"]) >= sig.target
        else:
            sl_hit  = float(candle["High"]) >= sig.sl
            tgt_hit = float(candle["Low"])  <= sig.target

        if tgt_hit:
            self._exit_trade(sig.target, ts, "TARGET")
        elif sl_hit:
            self._exit_trade(sig.sl, ts, "SL")

    def _exit_trade(self, exit_price: float, ts, reason: str):
        sig = self.active_signal
        if sig is None:
            return

        if sig.direction == TradeDirection.LONG:
            pnl_pts = exit_price - sig.entry_price
        else:
            pnl_pts = sig.entry_price - exit_price

        result = TradeResult(
            signal      = sig,
            exit_price  = round(exit_price, 2),
            exit_time   = ts,
            exit_reason = reason,
            pnl_pts     = round(pnl_pts, 2),
            pnl_pct     = round(pnl_pts / sig.entry_price * 100, 3),
        )
        self.trade_log.append(result)
        print(f"[BreakoutEngine] {result}")

        if self.on_trade_closed:
            self.on_trade_closed(result)

        self.active_signal  = None
        self.pending_signal = None
        self.state          = TradeState.IDLE

    # ── Reporting ────────────────────────────────────────────

    def get_trade_summary(self) -> dict:
        if not self.trade_log:
            return {"trades": 0}

        pnls = [t.pnl_pts for t in self.trade_log]
        wins = [p for p in pnls if p > 0]
        loss = [p for p in pnls if p <= 0]

        return {
            "trades":       len(pnls),
            "wins":         len(wins),
            "losses":       len(loss),
            "win_rate":     round(len(wins) / len(pnls) * 100, 1),
            "total_pts":    round(sum(pnls), 2),
            "avg_win":      round(np.mean(wins), 2) if wins else 0,
            "avg_loss":     round(np.mean(loss), 2) if loss else 0,
            "best_trade":   round(max(pnls), 2),
            "worst_trade":  round(min(pnls), 2),
            "profit_factor": round(
                sum(wins) / abs(sum(loss)), 2
            ) if loss and sum(loss) != 0 else float("inf"),
        }