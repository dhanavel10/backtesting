"""
tick_processor.py
=================
Ingests raw ticks from a localhost WebSocket feed and aggregates
them into completed candles of any interval (1m, 3m, 5m, etc.).

Expected incoming tick JSON (adapt schema to your feed):
{
    "symbol": "NIFTY",
    "ltp":    24512.50,
    "volume": 123456,
    "ts":     1716000000.123   # epoch seconds (float)
}

Emits CandleBar named-tuples to registered callbacks when a candle closes.
"""

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
import websockets
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class Tick:
    symbol:    str
    ltp:       float
    volume:    int
    ts:        float   # epoch seconds

    @classmethod
    def from_dict(cls, d: dict) -> "Tick":
        return cls(
            symbol = str(d.get("symbol", "UNKNOWN")),
            ltp    = float(d["ltp"]),
            volume = int(d.get("volume", 0)),
            ts     = float(d.get("ts", time.time())),
        )


@dataclass
class CandleBar:
    symbol:    str
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    int
    ts_open:   float   # epoch of candle open
    ts_close:  float   # epoch of candle close
    interval:  int     # seconds
    bar_index: int     # monotonically increasing bar counter

    @property
    def dt_open(self) -> datetime:
        return datetime.fromtimestamp(self.ts_open, tz=timezone.utc)

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    def to_dict(self) -> dict:
        return {
            "symbol":    self.symbol,
            "open":      self.open,
            "high":      self.high,
            "low":       self.low,
            "close":     self.close,
            "volume":    self.volume,
            "ts_open":   self.ts_open,
            "ts_close":  self.ts_close,
            "interval":  self.interval,
            "bar_index": self.bar_index,
        }


# ─────────────────────────────────────────────────────────────────
# In-progress candle accumulator (per symbol × interval)
# ─────────────────────────────────────────────────────────────────

@dataclass
class _OpenBar:
    symbol:    str
    interval:  int
    bucket:    int       # floor(epoch / interval) * interval  →  slot start
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    int
    tick_count: int = 0
    bar_index:  int = 0


class CandleAggregator:
    """
    Aggregates raw ticks into fixed-interval OHLCV candles.

    Time-bucketing: every tick is assigned to bucket = floor(ts/interval)*interval
    A new candle closes when a tick arrives in the NEXT bucket.

    No look-ahead: a candle is only emitted when it's *confirmed closed*
    (i.e. when the first tick of the next bucket arrives).
    """

    def __init__(self, interval_seconds: int = 300):   # 300s = 5m
        self.interval  = interval_seconds
        self._open_bars: Dict[str, _OpenBar] = {}
        self._bar_counters: Dict[str, int]   = defaultdict(int)
        self._callbacks: List[Callable[[CandleBar], None]] = []
        self.closed_candles: Dict[str, List[CandleBar]] = defaultdict(list)

    def on_candle_closed(self, fn: Callable[[CandleBar], None]):
        """Register a callback to receive completed CandleBar objects."""
        self._callbacks.append(fn)

    def process_tick(self, tick: Tick):
        """Feed a single tick; may emit 0 or 1 closed CandleBar."""
        bucket = int(tick.ts // self.interval) * self.interval
        sym    = tick.symbol

        if sym not in self._open_bars:
            # First tick ever for this symbol
            self._open_bars[sym] = _OpenBar(
                symbol=sym, interval=self.interval, bucket=bucket,
                open=tick.ltp, high=tick.ltp, low=tick.ltp, close=tick.ltp,
                volume=tick.volume,
            )
            return

        bar = self._open_bars[sym]

        if bucket == bar.bucket:
            # Same candle — update OHLCV
            bar.high    = max(bar.high, tick.ltp)
            bar.low     = min(bar.low,  tick.ltp)
            bar.close   = tick.ltp
            bar.volume += tick.volume
            bar.tick_count += 1

        else:
            # New bucket → close the old candle, open a new one
            idx = self._bar_counters[sym]
            closed = CandleBar(
                symbol=sym,
                open=bar.open, high=bar.high, low=bar.low, close=bar.close,
                volume=bar.volume,
                ts_open=bar.bucket,
                ts_close=bar.bucket + self.interval - 1,
                interval=self.interval,
                bar_index=idx,
            )
            self._bar_counters[sym] += 1
            self.closed_candles[sym].append(closed)
            logger.debug(f"[CANDLE CLOSED] {sym} bar#{idx}  "
                         f"O={closed.open} H={closed.high} "
                         f"L={closed.low} C={closed.close}")

            for cb in self._callbacks:
                try:
                    cb(closed)
                except Exception as e:
                    logger.error(f"Candle callback error: {e}")

            # Open fresh bar
            self._open_bars[sym] = _OpenBar(
                symbol=sym, interval=self.interval, bucket=bucket,
                open=tick.ltp, high=tick.ltp, low=tick.ltp, close=tick.ltp,
                volume=tick.volume, bar_index=idx + 1,
            )

    def get_live_bar(self, symbol: str) -> Optional[_OpenBar]:
        """Return the currently building (unclosed) bar."""
        return self._open_bars.get(symbol)

    def get_history(self, symbol: str, n: int = 200) -> List[CandleBar]:
        """Return last N closed candles for a symbol."""
        return self.closed_candles[symbol][-n:]


# ─────────────────────────────────────────────────────────────────
# WebSocket client — connects to localhost tick feed
# ─────────────────────────────────────────────────────────────────

class TickFeedClient:
    """
    Connects to a localhost WebSocket tick feed and pushes ticks
    through the CandleAggregator.

    Feed must send JSON objects conforming to the Tick schema above.
    Reconnects automatically on disconnect.

    uri example: "ws://localhost:8765"
    """

    def __init__(
        self,
        uri:         str = "ws://localhost:8765",
        aggregator:  Optional[CandleAggregator] = None,
        symbols:     Optional[List[str]] = None,
        tick_callbacks: Optional[List[Callable[[Tick], None]]] = None,
    ):
        self.uri            = uri
        self.aggregator     = aggregator or CandleAggregator()
        self.symbols        = set(symbols or [])        # empty = accept all
        self._tick_cbs      = tick_callbacks or []
        self._running       = False
        self.stats          = {"ticks": 0, "errors": 0, "reconnects": 0}

    def on_tick(self, fn: Callable[[Tick], None]):
        self._tick_cbs.append(fn)

    async def _handle_message(self, raw: str):
        try:
            data = json.loads(raw)
            tick = Tick.from_dict(data)

            # Symbol filter
            if self.symbols and tick.symbol not in self.symbols:
                return

            self.stats["ticks"] += 1

            for cb in self._tick_cbs:
                cb(tick)

            self.aggregator.process_tick(tick)

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.stats["errors"] += 1
            logger.warning(f"Bad tick message: {e}  raw={raw[:120]}")

    async def run(self, retry_interval: float = 3.0):
        """Main async loop — reconnects forever until cancelled."""
        self._running = True
        while self._running:
            try:
                logger.info(f"Connecting to {self.uri}...")
                async with websockets.connect(
                    self.uri,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    logger.info(f"Connected. Streaming ticks...")
                    self.stats["reconnects"] += 1
                    async for msg in ws:
                        await self._handle_message(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"WebSocket error: {e}. Retrying in {retry_interval}s...")
                await asyncio.sleep(retry_interval)

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────
# Simulated feed for testing (replays a list of ticks)
# ─────────────────────────────────────────────────────────────────

class SimulatedTickFeed:
    """
    Replays historical price data as synthetic ticks.
    Used for backtesting and local validation.

    Converts OHLCV bars into a sequence of 4 ticks: O→H→L→C
    (standard OHLC simulation; preserves swing extremes).
    """

    def __init__(
        self,
        aggregator: CandleAggregator,
        speed:      float = 0.0,    # 0 = instant replay
    ):
        self.aggregator = aggregator
        self.speed      = speed

    def replay_bars(self, bars: list, symbol: str = "NIFTY"):
        """
        bars: list of dicts with keys: open, high, low, close, volume, ts_open, interval
        Replays each bar as O→H→L→C ticks within its time bucket.
        """
        for bar in bars:
            ts0      = bar["ts_open"]
            interval = bar["interval"]
            quarter  = interval / 4

            for i, price in enumerate([
                bar["open"],
                bar["high"],
                bar["low"],
                bar["close"],
            ]):
                tick = Tick(
                    symbol = symbol,
                    ltp    = price,
                    volume = bar["volume"] // 4,
                    ts     = ts0 + i * quarter,
                )
                self.aggregator.process_tick(tick)

            if self.speed > 0:
                time.sleep(self.speed)

        # Force-close last bar with a synthetic next-bucket tick
        if bars:
            last = bars[-1]
            flush_tick = Tick(
                symbol = symbol,
                ltp    = last["close"],
                volume = 0,
                ts     = last["ts_open"] + last["interval"],
            )
            self.aggregator.process_tick(flush_tick)
