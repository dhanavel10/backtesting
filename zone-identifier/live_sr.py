"""
live_sr.py — WebSocket live feed → 5-min candles → S/R zones + Dashboard + Reaction Reports
=============================================================================================
Connects to ws://localhost:8086 for tick data.
Serves dashboard state on ws://localhost:8087 (dashboard connects here).

NEW: Real-time price reaction reports at every S/R zone:
    • APPROACHING   — price within proximity buffer, heading toward zone
    • INSIDE ZONE   — price has entered the zone range
    • REVERSAL      — price entered zone and is now moving away (rejection)
    • BREAKOUT      — price has closed a candle decisively beyond the zone
    • RETEST        — price returning to a recently broken zone from the other side
    • STALLING      — price inside zone with no conviction either way

Tick message formats accepted (auto-detected):
    {"price": 24123.45, "timestamp": "2024-01-15T09:17:32.123"}
    {"ltp":   24123.45, "timestamp": 1705298252123}   <- epoch ms
    {"price": 24123.45}                                <- local clock
    [24123.45, "2024-01-15T09:17:32"]                 <- array
    24123.45                                           <- bare float
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from dotenv import load_dotenv

from realtime_sr import RealtimeSREngine, Zone

load_dotenv()

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

WS_HOST   = os.getenv("WS_HOST", "localhost")
WS_PORT   = int(os.getenv("WS_PORT",   "8086"))
WS_PATH   = os.getenv("WS_PATH",       "/ws")
DASH_PORT = int(os.getenv("DASH_PORT", "8087"))
WS_URL    = f"ws://{WS_HOST}:{WS_PORT}{WS_PATH}"

REVERSAL_THRESHOLD       = float(os.getenv("REVERSAL_THR",  "30.0"))
BANDWIDTH                = float(os.getenv("BANDWIDTH",     "7.0"))
ZONE_HALF_WIDTH          = float(os.getenv("ZONE_HW",       "17.5"))
HALF_LIFE_MIN            = float(os.getenv("HALF_LIFE_MIN", "120.0"))
TOP_N_ZONES              = int(os.getenv("TOP_N",           "10"))
ZONE_PRINT_INTERVAL_SECS = int(os.getenv("ZONE_INTERVAL",   "30"))
RECONNECT_DELAY          = int(os.getenv("RECONNECT_DELAY", "3"))
MAX_CANDLES              = 120
RANGE_SIZE               = ZONE_HALF_WIDTH * 2

# Reaction detection tunables
APPROACH_BUFFER_PCT   = float(os.getenv("APPROACH_BUFFER_PCT", "0.3"))   # % of zone width beyond upper/lower
BREAKOUT_CONFIRM_BARS = int(os.getenv("BREAKOUT_CONFIRM_BARS", "1"))      # closed bars beyond zone to confirm breakout
REVERSAL_MOVE_PCT     = float(os.getenv("REVERSAL_MOVE_PCT",   "0.4"))    # % of zone width moved away = reversal
STALL_TICKS_LIMIT     = int(os.getenv("STALL_TICKS_LIMIT",     "50"))     # ticks inside zone with <REVERSAL_MOVE_PCT = stall


# ════════════════════════════════════════════════════════════════
# ZONE REACTION STATE MACHINE
# ════════════════════════════════════════════════════════════════

class ReactionState(str, Enum):
    IDLE        = "IDLE"         # price far from zone
    APPROACHING = "APPROACHING"  # price heading toward zone, within buffer
    INSIDE      = "INSIDE"       # price inside zone boundaries
    REVERSAL    = "REVERSAL"     # price entered zone and bounced away
    BREAKOUT    = "BREAKOUT"     # price closed bar beyond zone decisively
    RETEST      = "RETEST"       # price returning to broken zone from flip side
    STALLING    = "STALLING"     # price stuck inside zone, no conviction


@dataclass
class ZoneReaction:
    """Tracks how price is interacting with one specific zone."""
    zone_price:     float
    zone_lower:     float
    zone_upper:     float
    zone_type:      str                         # Support / Resistance

    state:          ReactionState = ReactionState.IDLE
    prev_state:     ReactionState = ReactionState.IDLE
    entry_price:    Optional[float] = None      # price when entered zone
    entry_time:     Optional[datetime] = None
    peak_excursion: float = 0.0                 # max move beyond zone mid while inside
    ticks_inside:   int = 0
    bars_beyond:    int = 0                     # closed bars outside zone (breakout count)
    broken:         bool = False                # True once a confirmed breakout occurred
    broken_side:    Optional[str] = None        # "above" | "below"
    last_report:    Optional[str] = None        # description of last event
    last_report_ts: Optional[datetime] = None
    events:         List[dict] = field(default_factory=list)  # full event log


class ZoneReactionTracker:
    """
    Maintains one ZoneReaction per zone (keyed by zone price).
    On every tick + every bar-close, call update_tick() / update_bar().
    """

    def __init__(self):
        self._reactions: Dict[float, ZoneReaction] = {}

    def _key(self, z: Zone) -> float:
        return round(z.price, 2)

    def _get_or_create(self, z: Zone) -> ZoneReaction:
        k = self._key(z)
        if k not in self._reactions:
            self._reactions[k] = ZoneReaction(
                zone_price=z.price, zone_lower=z.lower,
                zone_upper=z.upper, zone_type=z.type,
            )
        else:
            # refresh boundaries in case zone floated
            r = self._reactions[k]
            r.zone_lower = z.lower
            r.zone_upper = z.upper
            r.zone_type  = z.type
        return self._reactions[k]

    # ── helpers ────────────────────────────────────────────────
    def _approach_buffer(self, r: ZoneReaction) -> float:
        width = r.zone_upper - r.zone_lower
        return width * (APPROACH_BUFFER_PCT / 100.0) if width else 5.0

    def _is_inside(self, price: float, r: ZoneReaction) -> bool:
        return r.zone_lower <= price <= r.zone_upper

    def _is_approaching(self, price: float, r: ZoneReaction) -> bool:
        buf = self._approach_buffer(r)
        if r.zone_type == "Resistance":
            return r.zone_lower - buf <= price < r.zone_lower
        else:
            return r.zone_upper < price <= r.zone_upper + buf

    def _excursion_pct(self, price: float, r: ZoneReaction) -> float:
        """How far price has moved from entry price relative to zone width."""
        if r.entry_price is None:
            return 0.0
        width = r.zone_upper - r.zone_lower or 1.0
        return abs(price - r.entry_price) / width * 100.0

    def _log_event(self, r: ZoneReaction, event: str, price: float, ts: datetime, extra: dict = None):
        rec = {"ts": ts.strftime("%H:%M:%S"), "event": event, "price": round(price, 2),
               "zone": round(r.zone_price, 2), "zone_type": r.zone_type}
        if extra:
            rec.update(extra)
        r.events.append(rec)
        r.last_report    = event
        r.last_report_ts = ts
        if len(r.events) > 200:
            r.events = r.events[-200:]

    # ── main update (called every tick) ────────────────────────
    def update_tick(self, price: float, ts: datetime, zones: List[Zone]) -> List[dict]:
        """
        Returns list of reaction-event dicts (only NEW events this tick).
        """
        new_events = []
        active_keys = set()

        for z in zones:
            k = self._key(z)
            active_keys.add(k)
            r = self._get_or_create(z)
            prev = r.state
            event = None

            inside   = self._is_inside(price, r)
            approach = self._is_approaching(price, r)

            # ── state transitions ──────────────────────────────
            if r.broken:
                # zone was broken — watch for retest
                if r.broken_side == "above" and price <= r.zone_upper + self._approach_buffer(r):
                    if r.state != ReactionState.RETEST:
                        r.state = ReactionState.RETEST
                        event = self._make_event("RETEST", r, price, ts,
                            f"Price retesting broken zone from above  [zone {r.zone_lower:.2f}–{r.zone_upper:.2f}]")
                elif r.broken_side == "below" and price >= r.zone_lower - self._approach_buffer(r):
                    if r.state != ReactionState.RETEST:
                        r.state = ReactionState.RETEST
                        event = self._make_event("RETEST", r, price, ts,
                            f"Price retesting broken zone from below  [zone {r.zone_lower:.2f}–{r.zone_upper:.2f}]")
                else:
                    r.state = ReactionState.IDLE

            elif inside:
                r.ticks_inside += 1
                if r.entry_price is None:
                    r.entry_price = price
                    r.entry_time  = ts

                # update peak excursion
                r.peak_excursion = max(r.peak_excursion, self._excursion_pct(price, r))

                if r.state not in (ReactionState.INSIDE, ReactionState.STALLING,
                                   ReactionState.REVERSAL):
                    r.state = ReactionState.INSIDE
                    event = self._make_event("INSIDE_ZONE", r, price, ts,
                        f"Price entered {'RESISTANCE' if r.zone_type=='Resistance' else 'SUPPORT'} zone "
                        f"[{r.zone_lower:.2f} – {r.zone_upper:.2f}]  entry@{price:.2f}")

                # check reversal: moved > REVERSAL_MOVE_PCT of zone width away from entry
                elif r.state == ReactionState.INSIDE:
                    excursion = self._excursion_pct(price, r)
                    if excursion >= REVERSAL_MOVE_PCT * 100:
                        direction = "UP" if price > r.entry_price else "DOWN"
                        r.state = ReactionState.REVERSAL
                        strength_label = self._reversal_strength(excursion)
                        event = self._make_event("REVERSAL", r, price, ts,
                            f"{strength_label} REVERSAL ↕ at {r.zone_type.upper()} zone  "
                            f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                            f"entry@{r.entry_price:.2f} → now@{price:.2f}  ({direction})  "
                            f"move={excursion:.1f}% of zone width",
                            extra={"direction": direction, "excursion_pct": round(excursion, 1)})

                # stalling: lots of ticks inside, no significant move
                elif r.state == ReactionState.INSIDE and r.ticks_inside >= STALL_TICKS_LIMIT:
                    excursion = self._excursion_pct(price, r)
                    if excursion < REVERSAL_MOVE_PCT * 100:
                        r.state = ReactionState.STALLING
                        event = self._make_event("STALLING", r, price, ts,
                            f"Price STALLING inside {r.zone_type.upper()} zone  "
                            f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                            f"{r.ticks_inside} ticks  no conviction — wait for breakout or rejection")

            elif approach:
                if r.state not in (ReactionState.APPROACHING, ReactionState.INSIDE,
                                   ReactionState.REVERSAL, ReactionState.STALLING):
                    r.state = ReactionState.APPROACHING
                    dist    = abs(price - (r.zone_upper if r.zone_type == "Support" else r.zone_lower))
                    event = self._make_event("APPROACHING", r, price, ts,
                        f"Price APPROACHING {r.zone_type.upper()} zone  "
                        f"[{r.zone_lower:.2f}–{r.zone_upper:.2f}]  "
                        f"CMP={price:.2f}  dist={dist:.2f} pts")

            else:
                # price moved away from zone cleanly
                if r.state in (ReactionState.INSIDE, ReactionState.STALLING):
                    # exited zone — was it a breakout or clean exit after reversal?
                    if r.zone_type == "Resistance" and price > r.zone_upper:
                        pass  # bar_close will confirm breakout
                    elif r.zone_type == "Support" and price < r.zone_lower:
                        pass  # bar_close will confirm breakout
                    # reversal already set — just move to IDLE
                    if r.state != ReactionState.REVERSAL:
                        r.state = ReactionState.IDLE
                        r.entry_price    = None
                        r.ticks_inside   = 0
                        r.peak_excursion = 0.0
                elif r.state == ReactionState.APPROACHING:
                    r.state = ReactionState.IDLE
                elif r.state == ReactionState.REVERSAL:
                    # price moved far from zone after reversal — confirm and reset
                    r.state       = ReactionState.IDLE
                    r.entry_price = None
                    r.ticks_inside = 0
                    r.peak_excursion = 0.0

            if event:
                self._log_event(r, event["event"], price, ts, event.get("extra"))
                new_events.append(event)

        return new_events

    # ── bar close update ────────────────────────────────────────
    def update_bar(self, candle_close: float, candle_high: float, candle_low: float,
                   ts: datetime, zones: List[Zone]) -> List[dict]:
        """
        Called on every 5-min bar close. Checks for confirmed breakouts.
        A breakout is confirmed when price CLOSES decisively outside the zone.
        """
        new_events = []
        for z in zones:
            r = self._get_or_create(z)
            if r.broken:
                continue

            # Bar closed above resistance zone
            if z.type == "Resistance" and candle_close > z.upper:
                gap = candle_close - z.upper
                r.bars_beyond += 1
                if r.bars_beyond >= BREAKOUT_CONFIRM_BARS:
                    r.broken      = True
                    r.broken_side = "above"
                    r.state       = ReactionState.BREAKOUT
                    strength      = self._breakout_strength(gap, z)
                    event = self._make_event("BREAKOUT", r, candle_close, ts,
                        f"⚡ {strength} BREAKOUT above RESISTANCE zone  "
                        f"[{z.lower:.2f}–{z.upper:.2f}]  "
                        f"bar closed @ {candle_close:.2f}  gap={gap:.2f} pts above zone  "
                        f"{'★ STRONG — expect momentum continuation' if strength=='STRONG' else 'Watch for retest before continuation'}",
                        extra={"gap": round(gap, 2), "strength": strength, "direction": "UP"})
                    self._log_event(r, event["event"], candle_close, ts, event.get("extra"))
                    new_events.append(event)
                    r.entry_price    = None
                    r.ticks_inside   = 0
                    r.peak_excursion = 0.0

            # Bar closed below support zone
            elif z.type == "Support" and candle_close < z.lower:
                gap = z.lower - candle_close
                r.bars_beyond += 1
                if r.bars_beyond >= BREAKOUT_CONFIRM_BARS:
                    r.broken      = True
                    r.broken_side = "below"
                    r.state       = ReactionState.BREAKOUT
                    strength      = self._breakout_strength(gap, z)
                    event = self._make_event("BREAKOUT", r, candle_close, ts,
                        f"⚡ {strength} BREAKOUT below SUPPORT zone  "
                        f"[{z.lower:.2f}–{z.upper:.2f}]  "
                        f"bar closed @ {candle_close:.2f}  gap={gap:.2f} pts below zone  "
                        f"{'★ STRONG — expect momentum continuation' if strength=='STRONG' else 'Watch for retest before breakdown continues'}",
                        extra={"gap": round(gap, 2), "strength": strength, "direction": "DOWN"})
                    self._log_event(r, event["event"], candle_close, ts, event.get("extra"))
                    new_events.append(event)
                    r.entry_price    = None
                    r.ticks_inside   = 0
                    r.peak_excursion = 0.0

            else:
                r.bars_beyond = 0   # reset counter if price returns inside

        return new_events

    # ── event builders ─────────────────────────────────────────
    @staticmethod
    def _make_event(etype: str, r: ZoneReaction, price: float, ts: datetime,
                    description: str, extra: dict = None) -> dict:
        ev = {
            "event":       etype,
            "description": description,
            "price":       round(price, 2),
            "zone_price":  round(r.zone_price, 2),
            "zone_lower":  round(r.zone_lower, 2),
            "zone_upper":  round(r.zone_upper, 2),
            "zone_type":   r.zone_type,
            "ts":          ts.strftime("%H:%M:%S"),
        }
        if extra:
            ev.update(extra)
        return ev

    @staticmethod
    def _reversal_strength(excursion_pct: float) -> str:
        if excursion_pct >= 200:
            return "STRONG"
        elif excursion_pct >= 100:
            return "MODERATE"
        return "WEAK"

    @staticmethod
    def _breakout_strength(gap: float, z: Zone) -> str:
        zone_width = z.upper - z.lower or 1.0
        ratio = gap / zone_width
        if ratio >= 0.75:
            return "STRONG"
        elif ratio >= 0.35:
            return "MODERATE"
        return "WEAK"

    def snapshot(self) -> List[dict]:
        """Current state of all tracked reactions — for dashboard broadcast."""
        out = []
        for r in self._reactions.values():
            out.append({
                "zone_price":  round(r.zone_price, 2),
                "zone_lower":  round(r.zone_lower, 2),
                "zone_upper":  round(r.zone_upper, 2),
                "zone_type":   r.zone_type,
                "state":       r.state.value,
                "ticks_inside": r.ticks_inside,
                "broken":      r.broken,
                "broken_side": r.broken_side,
                "last_report": r.last_report,
                "last_report_ts": r.last_report_ts.strftime("%H:%M:%S") if r.last_report_ts else None,
                "events":      r.events[-5:],   # last 5 events per zone
            })
        return out


# ════════════════════════════════════════════════════════════════
# 5-MIN CANDLE BUILDER
# ════════════════════════════════════════════════════════════════

@dataclass
class Candle:
    open:       float
    high:       float
    low:        float
    close:      float
    bar_time:   datetime
    tick_count: int


class FiveMinCandleBuilder:
    def __init__(self):
        self._slot  = None
        self._open  = self._high = self._low = self._close = None
        self._ticks = 0

    @staticmethod
    def _floor5(ts: datetime) -> datetime:
        return ts.replace(second=0, microsecond=0, minute=(ts.minute // 5) * 5)

    def on_tick(self, price: float, ts: datetime) -> Optional[Candle]:
        slot = self._floor5(ts)
        if self._slot is None:
            self._slot = slot
            self._open = self._high = self._low = self._close = price
            self._ticks = 1
            return None
        if slot == self._slot:
            self._high  = max(self._high, price)
            self._low   = min(self._low,  price)
            self._close = price
            self._ticks += 1
            return None
        closed = Candle(self._open, self._high, self._low, self._close, self._slot, self._ticks)
        self._slot  = slot
        self._open  = self._high = self._low = self._close = price
        self._ticks = 1
        return closed

    @property
    def current_bar(self) -> Optional[Candle]:
        if self._slot is None:
            return None
        return Candle(self._open, self._high, self._low, self._close, self._slot, self._ticks)


# ════════════════════════════════════════════════════════════════
# TICK PARSER
# ════════════════════════════════════════════════════════════════

def parse_tick(raw: str) -> tuple:
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
        ts = datetime.fromtimestamp(ts_raw / 1000.0 if ts_raw > 1e12 else ts_raw)
        return price, ts
    if isinstance(ts_raw, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
            try:
                return price, datetime.strptime(ts_raw, fmt)
            except ValueError:
                continue
    return price, now


# ════════════════════════════════════════════════════════════════
# DASHBOARD STATE BROADCASTER
# ════════════════════════════════════════════════════════════════

class DashboardBroadcaster:
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
        msg = json.dumps(payload)
        self._last_msg = msg
        if self._clients:
            await asyncio.gather(*[self._send_one(ws, msg) for ws in list(self._clients)])


broadcaster = DashboardBroadcaster()


async def _dashboard_handler(websocket):
    broadcaster.register(websocket)
    try:
        await websocket.wait_closed()
    finally:
        broadcaster.unregister(websocket)


# ════════════════════════════════════════════════════════════════
# PAYLOAD SERIALISATION
# ════════════════════════════════════════════════════════════════

def _c(c: Candle) -> dict:
    return {"t": c.bar_time.strftime("%H:%M"), "o": round(c.open, 2),
            "h": round(c.high, 2), "l": round(c.low, 2),
            "c": round(c.close, 2), "n": c.tick_count}

def _z(z: Zone) -> dict:
    return {"price": round(z.price, 2), "lower": round(z.lower, 2),
            "upper": round(z.upper, 2), "strength": round(z.strength, 1),
            "type": z.type, "n_pivots": z.n_pivots, "anchored": z.anchored}

def _p(p) -> dict:
    return {"price": round(p.price, 2), "time": p.time.strftime("%H:%M:%S"),
            "type": p.type, "swing": round(p.swing_size, 1), "confirmed": p.confirmed}

def build_payload(engine, candle_builder, closed_candles, zones,
                  cmp, ts, tick_count, bar_count, trigger,
                  reaction_tracker: ZoneReactionTracker,
                  new_reaction_events: List[dict]) -> dict:
    prov   = engine.zz.provisional_pivot()
    pivots = [_p(p) for p in engine.zz.pivots[-60:]]
    if prov:
        pd = _p(prov); pd["confirmed"] = False; pivots.append(pd)
    live = candle_builder.current_bar
    return {
        "trigger":          trigger,
        "ts":               ts.strftime("%H:%M:%S"),
        "cmp":              round(cmp, 2),
        "tick_count":       tick_count,
        "bar_count":        bar_count,
        "pivot_count":      len(engine.zz.pivots),
        "candles":          [_c(c) for c in closed_candles],
        "live_bar":         _c(live) if live else None,
        "zones":            [_z(z) for z in zones],
        "pivots":           pivots,
        "zone_reactions":   reaction_tracker.snapshot(),
        "new_events":       new_reaction_events,
        "config":           {
            "range_size":   RANGE_SIZE,
            "reversal":     REVERSAL_THRESHOLD,
            "zone_hw":      ZONE_HALF_WIDTH,
        },
    }


# ════════════════════════════════════════════════════════════════
# CONSOLE DISPLAY
# ════════════════════════════════════════════════════════════════

SEP  = "=" * 90
SEP2 = "-" * 90

# ANSI colour codes (gracefully degrade on Windows)
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

C_RESET  = "\033[0m"
C_RED    = "\033[91m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN   = "\033[96m"
C_BLUE   = "\033[94m"
C_MAGENTA= "\033[95m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"

EVENT_COLORS = {
    "INSIDE_ZONE": C_YELLOW,
    "APPROACHING": C_CYAN,
    "REVERSAL":    C_GREEN,
    "BREAKOUT":    C_RED + C_BOLD,
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


def print_reaction_events(events: List[dict], cmp: float, ts: datetime):
    """Pretty-print new zone reaction events to console."""
    if not events:
        return
    print(f"\n{'─'*90}")
    print(f"  {C_BOLD}PRICE REACTION REPORT{C_RESET}  CMP: {C_BOLD}{cmp:.2f}{C_RESET}  @  {ts.strftime('%H:%M:%S')}")
    print(f"{'─'*90}")
    for ev in events:
        etype  = ev["event"]
        color  = EVENT_COLORS.get(etype, "")
        icon   = EVENT_ICONS.get(etype, "•")
        zone_label = f"[{ev['zone_lower']:.2f} – {ev['zone_upper']:.2f}]"
        print(f"  {color}{C_BOLD}{icon}  {etype:<12}{C_RESET}  {ev['ts']}  "
              f"{ev['zone_type']:<12} {zone_label}  px={ev['price']:.2f}")
        print(f"  {color}   └─ {ev['description']}{C_RESET}")
    print(f"{'─'*90}")


def print_zones(zones, cmp, trigger, ts, candle=None):
    print(f"\n{SEP}")
    print(f"  S/R ZONES  [{trigger}]  CMP: {cmp:.2f}  @  {ts.strftime('%H:%M:%S')}")
    if candle:
        print(f"  Bar {candle.bar_time.strftime('%H:%M')}  "
              f"O:{candle.open:.1f} H:{candle.high:.1f} "
              f"L:{candle.low:.1f} C:{candle.close:.1f}  ticks:{candle.tick_count}")
    print(SEP)
    if not zones:
        print("  (no zones yet)")
        print(SEP); return
    supp = sorted([z for z in zones if z.type == "Support"],    key=lambda z: z.price, reverse=True)
    res  = sorted([z for z in zones if z.type == "Resistance"], key=lambda z: z.price)
    print(f"  {'Type':<12} {'Price':>8}  {'Lower':>8}  {'Upper':>8}  {'Str':>5}  {'Pivots':>6}  {'Anchored':>8}")
    print(f"  {SEP2}")
    for z in res:
        anchor_flag = "★ FIXED" if z.anchored else "  float"
        print(f"  {C_RED}{'RES':<12} {z.price:>8.2f}  {z.lower:>8.2f}  {z.upper:>8.2f}  "
              f"{z.strength:>5.1f}  {z.n_pivots:>6}  {anchor_flag}{C_RESET}")
    print(f"  {'─'*35}  CMP {cmp:.2f}  {'─'*40}")
    for z in supp:
        anchor_flag = "★ FIXED" if z.anchored else "  float"
        print(f"  {C_GREEN}{'SUP':<12} {z.price:>8.2f}  {z.lower:>8.2f}  {z.upper:>8.2f}  "
              f"{z.strength:>5.1f}  {z.n_pivots:>6}  {anchor_flag}{C_RESET}")
    print(SEP)


def print_candle(c, n_pivots):
    d = "^" if c.close >= c.open else "v"
    color = C_GREEN if c.close >= c.open else C_RED
    print(f"  {color}CANDLE {d} {c.bar_time.strftime('%H:%M')}  "
          f"O:{c.open:.1f} H:{c.high:.1f} L:{c.low:.1f} C:{c.close:.1f}  "
          f"spread:{c.high-c.low:.1f}  ticks:{c.tick_count}  pivots:{n_pivots}{C_RESET}")


def print_realtime_status(reaction_tracker: ZoneReactionTracker, cmp: float, ts: datetime):
    """
    Periodic summary: show status of every zone being tracked.
    Only prints zones that are not IDLE.
    """
    active = [r for r in reaction_tracker._reactions.values()
              if r.state != ReactionState.IDLE]
    if not active:
        return
    print(f"\n  {C_CYAN}── ACTIVE ZONE STATUS @ {ts.strftime('%H:%M:%S')}  CMP={cmp:.2f} ──{C_RESET}")
    for r in active:
        color = EVENT_COLORS.get(r.state.value, "")
        icon  = EVENT_ICONS.get(r.state.value, "•")
        bkn   = f"  {C_RED}[BROKEN {r.broken_side}]{C_RESET}" if r.broken else ""
        print(f"  {color}{icon} {r.state.value:<12}{C_RESET}  "
              f"{r.zone_type:<12} [{r.zone_lower:.2f}–{r.zone_upper:.2f}]"
              f"  ticks_inside={r.ticks_inside}{bkn}")


# ════════════════════════════════════════════════════════════════
# MAIN TICK LOOP
# ════════════════════════════════════════════════════════════════

async def tick_loop():
    engine          = RealtimeSREngine(
        reversal_threshold=REVERSAL_THRESHOLD,
        bandwidth=BANDWIDTH, zone_half_width=ZONE_HALF_WIDTH, half_life_min=HALF_LIFE_MIN,
    )
    candle_builder  = FiveMinCandleBuilder()
    reaction_tracker = ZoneReactionTracker()
    closed_candles  = []
    last_zone_ts    = datetime.now()
    last_status_ts  = datetime.now()
    tick_count = bar_count = 0
    current_zones: List[Zone] = []

    import websockets

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print(f"  Connected to tick feed {WS_URL}\n")
                async for raw_msg in ws:
                    price, ts = parse_tick(str(raw_msg))
                    if price is None:
                        continue

                    tick_count += 1
                    closed = candle_builder.on_tick(price, ts)

                    trigger           = None
                    new_reaction_evts = []

                    # ── BAR CLOSE ─────────────────────────────
                    if closed is not None:
                        bar_count += 1
                        closed_candles.append(closed)
                        if len(closed_candles) > MAX_CANDLES:
                            closed_candles.pop(0)

                        engine.on_candle(closed.high, closed.low, closed.bar_time)
                        print_candle(closed, len(engine.zz.pivots))

                        current_zones = engine.zones(current_price=price, now=ts, top_n=TOP_N_ZONES)

                        # bar-close reaction check (breakout confirmation)
                        bar_evts = reaction_tracker.update_bar(
                            closed.close, closed.high, closed.low, ts, current_zones)
                        new_reaction_evts.extend(bar_evts)

                        # tick-level reaction on bar-close price
                        tick_evts = reaction_tracker.update_tick(price, ts, current_zones)
                        new_reaction_evts.extend(tick_evts)

                        trigger = "BAR_CLOSE"
                        print_zones(current_zones, price, "5-MIN BAR CLOSE", ts, closed)
                        if new_reaction_evts:
                            print_reaction_events(new_reaction_evts, price, ts)
                        last_zone_ts = datetime.now()

                    # ── LIVE UPDATE (periodic) ─────────────────
                    elif (ZONE_PRINT_INTERVAL_SECS > 0 and
                          (datetime.now() - last_zone_ts).total_seconds() >= ZONE_PRINT_INTERVAL_SECS):
                        current_zones = engine.zones(current_price=price, now=ts, top_n=TOP_N_ZONES)

                        tick_evts = reaction_tracker.update_tick(price, ts, current_zones)
                        new_reaction_evts.extend(tick_evts)

                        trigger = "LIVE_UPDATE"
                        print_zones(current_zones, price, "LIVE UPDATE", ts, candle_builder.current_bar)
                        if new_reaction_evts:
                            print_reaction_events(new_reaction_evts, price, ts)
                        else:
                            print_realtime_status(reaction_tracker, price, ts)
                        last_zone_ts = datetime.now()

                    # ── EVERY TICK (reaction check, no zone recompute) ──
                    else:
                        if current_zones:
                            tick_evts = reaction_tracker.update_tick(price, ts, current_zones)
                            new_reaction_evts.extend(tick_evts)
                            if new_reaction_evts:
                                print_reaction_events(new_reaction_evts, price, ts)

                        if tick_count % 100 == 0:
                            trigger = "TICK"
                            current_zones = engine.zones(current_price=price, now=ts, top_n=TOP_N_ZONES)
                            b = candle_builder.current_bar
                            bs = f"bar {b.bar_time.strftime('%H:%M')} H:{b.high:.1f} L:{b.low:.1f}" if b else "no bar"
                            print(f"  tick #{tick_count:>6}  px={price:.2f}  pivots={len(engine.zz.pivots)}  {bs}")

                    # ── broadcast to dashboard ─────────────────
                    if trigger or new_reaction_evts:
                        payload = build_payload(
                            engine, candle_builder, closed_candles,
                            current_zones, price, ts, tick_count, bar_count,
                            trigger or "REACTION",
                            reaction_tracker, new_reaction_evts,
                        )
                        await broadcaster.broadcast(payload)

        except (ConnectionRefusedError, OSError) as e:
            print(f"  Tick feed unavailable: {e}  — retry in {RECONNECT_DELAY}s")
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as e:
            print(f"  Error: {e}  — retry in {RECONNECT_DELAY}s")
            await asyncio.sleep(RECONNECT_DELAY)


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

async def main():
    import websockets
    dash_server = await websockets.serve(_dashboard_handler, "0.0.0.0", DASH_PORT)
    print(f"\n{'='*90}")
    print(f"  Tick feed   : {WS_URL}")
    print(f"  Dashboard   : open dashboard.html in your browser")
    print(f"  reversal={REVERSAL_THRESHOLD}  bw={BANDWIDTH}  zone_hw={ZONE_HALF_WIDTH}")
    print(f"  Reaction detection:")
    print(f"    approach_buffer={APPROACH_BUFFER_PCT}%  breakout_confirm={BREAKOUT_CONFIRM_BARS} bar(s)")
    print(f"    reversal_move={REVERSAL_MOVE_PCT*100:.0f}% of zone width  stall_ticks={STALL_TICKS_LIMIT}")
    print(f"{'='*90}\n")
    await asyncio.gather(tick_loop(), dash_server.wait_closed())

if __name__ == "__main__":
    print("\nPress Ctrl-C to stop.\n")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Stopped.")