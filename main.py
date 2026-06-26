"""
main.py
=======
Orchestrator for the realtime S/R detection system.

Starts:
  1. TickFeedClient       — connects to your localhost tick feed
  2. Full SR pipeline     — aggregator → swing → zone → signal
  3. WebSocket server     — broadcasts state to the dashboard (port 8766)
  4. HTTP server          — serves dashboard.html (port 8080)

Dashboard connect: http://localhost:8080/dashboard.html

Tick feed expected at: ws://localhost:8765
  (adapt TICK_FEED_URI to your broker's WebSocket URL)

Run:
    python main.py

    # Or with custom settings:
    python main.py --symbol BANKNIFTY --half-band 20 --rev-pct 0.25
"""

import asyncio
import json
import logging
import argparse
import http.server
import threading
import os
import time
from datetime import datetime, timezone
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol

from tick_processor  import CandleAggregator, TickFeedClient, CandleBar, Tick
from swing_engine    import ReversalPivotDetector, PivotEvent
from zone_engine     import ZoneEngine, ZoneEvent
from signal_engine   import SignalEngine, Signal

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Global state (broadcast to all connected dashboard clients)
# ─────────────────────────────────────────────────────────────────

class LiveState:
    def __init__(self):
        self.symbol:        str   = "NIFTY"
        self.current_price: float = 0.0
        self.current_ts:    float = 0.0
        self.zones:         list  = []
        self.recent_signals:list  = []
        self.recent_pivots: list  = []
        self.tick_count:    int   = 0
        self.bar_count:     int   = 0
        self.last_bar:      dict  = {}

    def to_dict(self) -> dict:
        return {
            "type":          "state_update",
            "symbol":        self.symbol,
            "current_price": round(self.current_price, 2),
            "current_ts":    self.current_ts,
            "zones":         self.zones,
            "signals":       self.recent_signals[-30:],
            "pivots":        self.recent_pivots[-20:],
            "tick_count":    self.tick_count,
            "bar_count":     self.bar_count,
            "last_bar":      self.last_bar,
            "server_ts":     time.time(),
        }


LIVE = LiveState()
CLIENTS: Set[WebSocketServerProtocol] = set()


async def broadcast(message: dict):
    """Send a JSON message to all connected dashboard clients."""
    if not CLIENTS:
        return
    data = json.dumps(message, default=str)
    await asyncio.gather(
        *[ws.send(data) for ws in list(CLIENTS)],
        return_exceptions=True,
    )


# ─────────────────────────────────────────────────────────────────
# Pipeline callbacks
# ─────────────────────────────────────────────────────────────────

loop: asyncio.AbstractEventLoop = None   # set in main


def on_candle(bar: CandleBar, zone_engine: ZoneEngine, signal_engine: SignalEngine):
    LIVE.bar_count    += 1
    LIVE.current_price = bar.close
    LIVE.current_ts    = bar.ts_close
    LIVE.last_bar      = bar.to_dict()

    # Refresh active zones for broadcast
    zones = zone_engine.get_active_zones(bar.close, max_zones=25)
    LIVE.zones = [z.to_dict() for z in zones]

    logger.debug(f"[BAR #{bar.bar_index}] close={bar.close:.2f}  "
                 f"zones={len(zones)}  sigs={len(LIVE.recent_signals)}")

    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(
            broadcast({"type": "bar", "bar": bar.to_dict(), "zones": LIVE.zones}),
            loop,
        )


def on_pivot(pivot: PivotEvent):
    pd = {
        "symbol":      pivot.symbol,
        "type":        pivot.pivot_type.value,
        "price":       round(pivot.price, 2),
        "bar_index":   pivot.bar_index,
        "ts":          pivot.ts,
        "session":     pivot.session,
        "rev_pct":     pivot.rev_pct,
        "method":      pivot.method,
    }
    LIVE.recent_pivots.append(pd)
    LIVE.recent_pivots = LIVE.recent_pivots[-50:]
    logger.info(f"[PIVOT] {pivot.label}  @ {pivot.price:.2f}  "
                f"rev={pivot.rev_pct:.2f}%  session={pivot.session}")

    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(
            broadcast({"type": "pivot", "pivot": pd}), loop)


def on_zone_event(event: ZoneEvent):
    logger.info(f"[ZONE {event.event_type.upper()}]  "
                f"id={event.zone_id}  "
                f"@ {event.zone.price:.2f}  "
                f"str={event.zone.strength:.1f}")

    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(
            broadcast({
                "type":       "zone_event",
                "event_type": event.event_type,
                "zone":       event.zone.to_dict(),
            }), loop)


def on_signal(signal: Signal):
    sd = signal.to_dict()
    LIVE.recent_signals.append(sd)
    logger.info(
        f"[SIGNAL ★]  {signal.signal_type:<20}  {signal.action:<10}  "
        f"price={signal.price:.2f}  zone={signal.zone_price:.2f}  "
        f"str={signal.zone_strength:.0f}  conf={signal.confidence}"
    )

    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(
            broadcast({"type": "signal", "signal": sd}), loop)


def on_tick(tick: Tick):
    LIVE.tick_count   += 1
    LIVE.current_price = tick.ltp
    LIVE.current_ts    = tick.ts


# ─────────────────────────────────────────────────────────────────
# WebSocket server (dashboard ↔ backend)
# ─────────────────────────────────────────────────────────────────

async def ws_handler(ws: WebSocketServerProtocol):
    CLIENTS.add(ws)
    logger.info(f"Dashboard connected  ({len(CLIENTS)} total)")
    try:
        # Send full state on connect
        await ws.send(json.dumps(LIVE.to_dict(), default=str))
        async for _ in ws:
            pass   # dashboard doesn't send commands (yet)
    except websockets.exceptions.ConnectionClosedOK:
        pass
    except Exception as e:
        logger.warning(f"Dashboard WS error: {e}")
    finally:
        CLIENTS.discard(ws)
        logger.info(f"Dashboard disconnected  ({len(CLIENTS)} remaining)")


async def periodic_broadcast():
    """Send full state snapshot every 5 seconds (keeps dashboard in sync)."""
    while True:
        await asyncio.sleep(5)
        if CLIENTS and LIVE.current_price > 0:
            await broadcast(LIVE.to_dict())


# ─────────────────────────────────────────────────────────────────
# HTTP server (serves dashboard.html)
# ─────────────────────────────────────────────────────────────────

def start_http_server(port: int = 8080, directory: str = "."):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        def log_message(self, *args):
            pass   # suppress HTTP access logs

    server = http.server.HTTPServer(("", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info(f"HTTP server → http://localhost:{port}/dashboard.html")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def build_pipeline(args) -> tuple:
    """Build and wire the full realtime pipeline."""
    interval_map = {"1m": 60, "3m": 180, "5m": 300, "15m": 900}
    interval_sec = interval_map.get(args.interval, 300)

    aggregator = CandleAggregator(interval_seconds=interval_sec)

    pivot_detector = ReversalPivotDetector(
        symbol        = args.symbol,
        rev_pct       = args.rev_pct,
        min_swing_pts = args.min_swing,
    )

    zone_engine = ZoneEngine(
        symbol            = args.symbol,
        half_band         = args.half_band,
        cluster_tolerance = args.cluster_tol,
        min_wick_touches  = args.min_touches,
        min_sessions      = args.min_sessions,
        min_rejections    = args.min_rejections,
    )

    signal_engine = SignalEngine(
        symbol            = args.symbol,
        zone_engine       = zone_engine,
        min_zone_strength = args.min_strength,
    )

    # Wire: aggregator → pivot → zone → signal
    aggregator.on_candle_closed(
        lambda bar: on_candle(bar, zone_engine, signal_engine)
    )
    aggregator.on_candle_closed(pivot_detector.process_bar)
    aggregator.on_candle_closed(zone_engine.on_candle)
    aggregator.on_candle_closed(signal_engine.process_candle)

    pivot_detector.on_pivot(on_pivot)
    pivot_detector.on_pivot(zone_engine.on_pivot)
    zone_engine.on_zone_event(on_zone_event)
    signal_engine.on_signal(on_signal)

    tick_client = TickFeedClient(
        uri        = args.tick_uri,
        aggregator = aggregator,
        symbols    = [args.symbol] if args.symbol else [],
    )
    tick_client.on_tick(on_tick)

    LIVE.symbol = args.symbol
    return tick_client, zone_engine, signal_engine


async def async_main(args):
    global loop
    loop = asyncio.get_running_loop()

    tick_client, zone_engine, signal_engine = build_pipeline(args)

    # Start HTTP server for dashboard
    start_http_server(port=args.http_port, directory=os.path.dirname(__file__) or ".")

    # Start WebSocket server for dashboard
    logger.info(f"WebSocket server → ws://localhost:{args.ws_port}")
    ws_server = await websockets.serve(ws_handler, "localhost", args.ws_port)

    # Start periodic broadcast
    bcast_task = asyncio.create_task(periodic_broadcast())

    logger.info(f"Connecting to tick feed: {args.tick_uri}")
    logger.info("System is LIVE. Open http://localhost:8080/dashboard.html")
    logger.info("─" * 60)

    try:
        await tick_client.run()
    except asyncio.CancelledError:
        pass
    finally:
        ws_server.close()
        bcast_task.cancel()
        logger.info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Realtime S/R Zone System")
    parser.add_argument("--symbol",       default="NIFTY")
    parser.add_argument("--tick-uri",     default="ws://localhost:8765",
                        help="WebSocket URI for tick feed")
    parser.add_argument("--interval",     default="5m")
    parser.add_argument("--rev-pct",      default=0.30, type=float)
    parser.add_argument("--min-swing",    default=20.0, type=float)
    parser.add_argument("--half-band",    default=15.0, type=float)
    parser.add_argument("--cluster-tol",  default=15.0, type=float)
    parser.add_argument("--min-touches",  default=2,    type=int)
    parser.add_argument("--min-sessions", default=1,    type=int)
    parser.add_argument("--min-rejections", default=1,  type=int)
    parser.add_argument("--min-strength", default=15.0, type=float)
    parser.add_argument("--ws-port",      default=8766, type=int)
    parser.add_argument("--http-port",    default=8080, type=int)
    args = parser.parse_args()

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
