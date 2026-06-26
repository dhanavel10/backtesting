"""
live_sr.py — WebSocket live feed → 5-min candles → S/R zones + Dashboard
=========================================================================
Connects to ws://localhost:8086 for tick data.
Serves dashboard state on ws://localhost:8087 (dashboard connects here).
Open dashboard.html in a browser to see the live chart.

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
from typing import Optional
from dataclasses import dataclass

import numpy as np

from realtime_sr import RealtimeSREngine, Zone

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

WS_HOST   = os.getenv("WS_HOST", "localhost")
WS_PORT   = int(os.getenv("WS_PORT",   "8086"))   # tick feed IN
WS_PATH   = os.getenv("WS_PATH",       "/ws")     # endpoint path
DASH_PORT = int(os.getenv("DASH_PORT", "8087"))   # dashboard OUT
WS_URL    = f"ws://{WS_HOST}:{WS_PORT}{WS_PATH}"

RANGE_SIZE               = float(os.getenv("RANGE_SIZE",    "4.0"))
REVERSAL_THRESHOLD       = float(os.getenv("REVERSAL_THR",  "12.0"))
BANDWIDTH                = float(os.getenv("BANDWIDTH",     "7.0"))
ZONE_HALF_WIDTH          = float(os.getenv("ZONE_HW",       "12.0"))
HALF_LIFE_MIN            = float(os.getenv("HALF_LIFE_MIN", "120.0"))
TOP_N_ZONES              = int(os.getenv("TOP_N",           "10"))
ZONE_PRINT_INTERVAL_SECS = int(os.getenv("ZONE_INTERVAL",   "30"))
RECONNECT_DELAY          = int(os.getenv("RECONNECT_DELAY", "3"))
MAX_CANDLES              = 120   # closed bars kept for chart


# ════════════════════════════════════════════════════════════════
# 5-MIN CANDLE BUILDER  (unchanged logic)
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
# TICK PARSER  (unchanged logic)
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
            "type": z.type, "n_pivots": z.n_pivots}

def _p(p) -> dict:
    return {"price": round(p.price, 2), "time": p.time.strftime("%H:%M:%S"),
            "type": p.type, "swing": round(p.swing_size, 1), "confirmed": p.confirmed}

def build_payload(engine, candle_builder, closed_candles, zones,
                  cmp, ts, tick_count, bar_count, trigger) -> dict:
    prov    = engine.zz.provisional_pivot()
    pivots  = [_p(p) for p in engine.zz.pivots[-60:]]
    if prov:
        pd = _p(prov); pd["confirmed"] = False; pivots.append(pd)
    live = candle_builder.current_bar
    return {
        "trigger":     trigger,
        "ts":          ts.strftime("%H:%M:%S"),
        "cmp":         round(cmp, 2),
        "tick_count":  tick_count,
        "bar_count":   bar_count,
        "pivot_count": len(engine.zz.pivots),
        "candles":     [_c(c) for c in closed_candles],
        "live_bar":    _c(live) if live else None,
        "zones":       [_z(z) for z in zones],
        "pivots":      pivots,
        "config":      {"range_size": RANGE_SIZE, "reversal": REVERSAL_THRESHOLD,
                        "zone_hw": ZONE_HALF_WIDTH},
    }


# ════════════════════════════════════════════════════════════════
# CONSOLE DISPLAY  (unchanged)
# ════════════════════════════════════════════════════════════════

SEP = "=" * 80

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
    print(f"  {'Type':<12} {'Price':>8}  {'Lower':>8}  {'Upper':>8}  {'Str':>5}  {'Pivots':>6}")
    print(f"  {'-'*68}")
    for z in res:
        print(f"  {'RES':<12} {z.price:>8.2f}  {z.lower:>8.2f}  {z.upper:>8.2f}  {z.strength:>5.1f}  {z.n_pivots:>6}")
    print(f"  {'-'*30}  CMP {cmp:.2f}  {'-'*28}")
    for z in supp:
        print(f"  {'SUP':<12} {z.price:>8.2f}  {z.lower:>8.2f}  {z.upper:>8.2f}  {z.strength:>5.1f}  {z.n_pivots:>6}")
    print(SEP)

def print_candle(c, n_pivots):
    d = "^" if c.close >= c.open else "v"
    print(f"  CANDLE {d} {c.bar_time.strftime('%H:%M')}  "
          f"O:{c.open:.1f} H:{c.high:.1f} L:{c.low:.1f} C:{c.close:.1f}  "
          f"spread:{c.high-c.low:.1f}  ticks:{c.tick_count}  pivots:{n_pivots}")


# ════════════════════════════════════════════════════════════════
# MAIN TICK LOOP
# ════════════════════════════════════════════════════════════════

async def tick_loop():
    engine         = RealtimeSREngine(
        range_size=RANGE_SIZE, reversal_threshold=REVERSAL_THRESHOLD,
        bandwidth=BANDWIDTH, zone_half_width=ZONE_HALF_WIDTH, half_life_min=HALF_LIFE_MIN,
    )
    candle_builder = FiveMinCandleBuilder()
    closed_candles = []
    last_zone_ts   = datetime.now()
    tick_count = bar_count = 0

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
                    engine.on_tick(price, ts)
                    closed = candle_builder.on_tick(price, ts)

                    trigger = None
                    zones   = []

                    if closed is not None:
                        bar_count += 1
                        closed_candles.append(closed)
                        if len(closed_candles) > MAX_CANDLES:
                            closed_candles.pop(0)
                        print_candle(closed, len(engine.zz.pivots))
                        zones   = engine.zones(current_price=price, now=ts, top_n=TOP_N_ZONES)
                        trigger = "BAR_CLOSE"
                        print_zones(zones, price, "5-MIN BAR CLOSE", ts, closed)
                        last_zone_ts = datetime.now()

                    elif (ZONE_PRINT_INTERVAL_SECS > 0 and
                          (datetime.now() - last_zone_ts).total_seconds() >= ZONE_PRINT_INTERVAL_SECS):
                        zones   = engine.zones(current_price=price, now=ts, top_n=TOP_N_ZONES)
                        trigger = "LIVE_UPDATE"
                        print_zones(zones, price, "LIVE UPDATE", ts, candle_builder.current_bar)
                        last_zone_ts = datetime.now()

                    elif tick_count % 100 == 0:
                        trigger = "TICK"
                        zones   = engine.zones(current_price=price, now=ts, top_n=TOP_N_ZONES)
                        b = candle_builder.current_bar
                        bs = f"bar {b.bar_time.strftime('%H:%M')} H:{b.high:.1f} L:{b.low:.1f}" if b else "no bar"
                        print(f"  tick #{tick_count:>6}  px={price:.2f}  pivots={len(engine.zz.pivots)}  {bs}")

                    if trigger:
                        payload = build_payload(engine, candle_builder, closed_candles,
                                                zones, price, ts, tick_count, bar_count, trigger)
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
    dash_server = await websockets.serve(_dashboard_handler, "localhost", DASH_PORT)
    print(f"\n{'='*80}")
    print(f"  Tick feed   : {WS_URL}")
    print(f"  Dashboard   : open dashboard.html in your browser")
    print(f"  range={RANGE_SIZE}  reversal={REVERSAL_THRESHOLD}  bw={BANDWIDTH}  zone_hw={ZONE_HALF_WIDTH}")
    print(f"{'='*80}\n")
    await asyncio.gather(tick_loop(), dash_server.wait_closed())

if __name__ == "__main__":
    print("\nPress Ctrl-C to stop.\n")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Stopped.")