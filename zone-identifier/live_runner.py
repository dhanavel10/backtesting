"""
live_runner.py
==============
Live intraday runner — connects to a LOCAL WebSocket server for testing.

No Kite Connect required. Drop-in replacement for the Kite version.

Your local WebSocket server should send JSON ticks on each message:
    {"last_price": 24150.5, "timestamp": "2024-01-15 09:20:00", "volume": 1200}

Timestamp formats accepted (all auto-detected):
    "2024-01-15 09:20:00"          ← plain datetime string
    "2024-01-15T09:20:00"          ← ISO 8601
    "2024-01-15 09:20:00.123456"   ← with microseconds
    1705296000                     ← unix epoch (int or float)
    (omitted)                      ← falls back to system clock

Volume field accepted:
    "volume" | "volume_traded" | "last_traded_quantity"  ← any of these

Price field accepted:
    "last_price" | "last_traded_price" | "ltp" | "price"  ← any of these

How to start
------------
1. Start your local WebSocket server (whatever sends the ticks).
2. Set WS_URL below to match your server's address.
3. For historical zone seeding, either:
      a) Point HISTORY_CSV to a CSV of 5m OHLCV data  (recommended for testing)
      b) Set USE_YFINANCE = True to pull from Yahoo Finance
4. Run:  python live_runner.py

Outputs
-------
  Console  — real-time log of zones, signals, confirmations, exits
  trade_log.csv  — appended after every closed trade
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

# ── Flat-folder import fix ──────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from zone_engine    import ZoneEngine, fetch_intraday_chunked
from candle_buffer  import CandleBuffer
from breakout_engine import BreakoutEngine, TradeDirection

# ════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit these
# ════════════════════════════════════════════════════════════════

# ── Local WebSocket ──────────────────────────────────────────────
WS_URL = "ws://localhost:8765"      # change to your server's address/port

# ── Historical data source for pre-market zone build ─────────────
#    Option A: point to a CSV file with columns:
#              datetime, Open, High, Low, Close, Volume
#              (datetime must be IST, 5-minute bars)
HISTORY_CSV  = ""                   # e.g. "data/nifty_5m.csv"  — leave "" to skip

#    Option B: fetch from Yahoo Finance (requires internet + yfinance installed)
USE_YFINANCE = True                 # set False if no internet
YFINANCE_TICKER = "^NSEI"
YFINANCE_DAYS   = 60

# ── Zone parameters (same as original script) ───────────────────
ZONE_CONFIG = dict(
    cluster_tolerance = 15.0,
    zone_half_band    = 15.0,
    min_wick_touches  = 3,
    min_sessions      = 2,
    min_rejections    = 1,
    top_n             = 20,
    rebuild_every     = 12,         # rebuild zones every 12 candles (~1 hr)
)

# ── Strategy parameters ─────────────────────────────────────────
STRATEGY_CONFIG = dict(
    min_breakout_pts = 20.0,        # min close beyond zone boundary
    confirm_candles  = 1,           # extra candles needed to confirm
    rr_ratio         = 2.0,         # fallback R:R when no next zone exists
    max_target_pts   = 200.0,       # max target distance
)

# ── Reconnect settings ──────────────────────────────────────────
RECONNECT_DELAY_SEC = 5
MAX_RECONNECT_TRIES = 20

# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LocalRunner")


# ════════════════════════════════════════════════════════════════
# HISTORY LOADER
# ════════════════════════════════════════════════════════════════

def load_history() -> pd.DataFrame:
    """
    Load 5m historical data for zone map seeding.
    Tries CSV first, then yfinance.
    """
    # ── Option A: CSV ────────────────────────────────────────
    if HISTORY_CSV and os.path.exists(HISTORY_CSV):
        log.info(f"Loading history from CSV: {HISTORY_CSV}")
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)

        # Normalise column names (flexible)
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower()
            if cl in ("open", "o"):        col_map[c] = "Open"
            elif cl in ("high", "h"):      col_map[c] = "High"
            elif cl in ("low", "l"):       col_map[c] = "Low"
            elif cl in ("close", "c"):     col_map[c] = "Close"
            elif cl in ("volume", "vol"):  col_map[c] = "Volume"
        df.rename(columns=col_map, inplace=True)

        # Ensure IST timezone
        if df.index.tzinfo is None:
            try:
                import pytz
                df.index = df.index.tz_localize("Asia/Kolkata")
            except Exception:
                pass
        df = df.between_time("09:15", "15:30")
        df.dropna(inplace=True)
        log.info(f"✓ CSV loaded: {len(df)} bars  "
                 f"{df.index[0].date()} → {df.index[-1].date()}")
        return df

    # ── Option B: yfinance ────────────────────────────────────
    if USE_YFINANCE:
        log.info("No CSV found — fetching history from Yahoo Finance...")
        try:
            return fetch_intraday_chunked(
                ticker=YFINANCE_TICKER,
                days=YFINANCE_DAYS,
            )
        except Exception as e:
            log.error(f"yfinance fetch failed: {e}")

    raise RuntimeError(
        "No historical data available.\n"
        "Either set HISTORY_CSV to a valid path or set USE_YFINANCE=True."
    )


# ════════════════════════════════════════════════════════════════
# SIGNAL CALLBACKS
# ════════════════════════════════════════════════════════════════

def on_signal(sig):
    arrow = "▲ LONG" if sig.direction == TradeDirection.LONG else "▼ SHORT"
    log.info("─" * 60)
    log.info(f"🔔 PENDING SIGNAL  {arrow}")
    log.info(f"   Zone  : {sig.zone_price:.0f}  "
             f"[{sig.zone_lower:.0f} – {sig.zone_upper:.0f}]  "
             f"strength={sig.zone_strength:.0f}")
    log.info(f"   Entry : ≈ {sig.entry_price:.0f}   "
             f"SL={sig.sl:.0f}   TGT={sig.target:.0f}   "
             f"R:R={sig.rr_ratio}")
    log.info(f"   Option: Buy {sig.option_strike}{sig.option_type}  "
             f"(max premium ≈ {sig.entry_price * 0.03:.0f})")
    log.info(f"   ⏳ Waiting 1 confirmation candle...")
    log.info("─" * 60)


def on_signal_confirmed(sig):
    arrow = "▲ LONG" if sig.direction == TradeDirection.LONG else "▼ SHORT"
    log.info("═" * 60)
    log.info(f"✅ CONFIRMED ENTRY  {arrow}")
    log.info(f"   Entry : {sig.entry_price:.0f}   "
             f"SL={sig.sl:.0f}   TGT={sig.target:.0f}   "
             f"R:R={sig.rr_ratio}")
    log.info(f"   Option: {sig.option_strike}{sig.option_type}")
    log.info("═" * 60)
    # ─── Add your order placement here when ready ────────────
    # e.g. your_broker.place_order(...)
    # ─────────────────────────────────────────────────────────


def on_trade_closed(result):
    emoji   = "✅" if result.pnl_pts > 0 else "❌"
    dir_sym = "▲" if result.signal.direction == TradeDirection.LONG else "▼"
    log.info("─" * 60)
    log.info(f"{emoji} TRADE CLOSED  [{result.exit_reason}]  {dir_sym}")
    log.info(f"   Entry={result.signal.entry_price:.0f}  "
             f"Exit={result.exit_price:.0f}  "
             f"P&L={result.pnl_pts:+.0f} pts")
    log.info("─" * 60)

    # ─── Append to CSV log ────────────────────────────────────
    row = {
        "date":          result.exit_time,
        "direction":     result.signal.direction.value,
        "entry":         result.signal.entry_price,
        "sl":            result.signal.sl,
        "target":        result.signal.target,
        "exit":          result.exit_price,
        "exit_reason":   result.exit_reason,
        "pnl_pts":       result.pnl_pts,
        "zone":          result.signal.zone_price,
        "zone_strength": result.signal.zone_strength,
    }
    path = "trade_log.csv"
    pd.DataFrame([row]).to_csv(
        path, mode="a",
        header=not os.path.exists(path),
        index=False,
    )
    log.info(f"   → Logged to {path}")


# ════════════════════════════════════════════════════════════════
# WEBSOCKET CLIENT
# ════════════════════════════════════════════════════════════════

class LocalWebSocketRunner:
    """
    Connects to a local WebSocket server and pipes ticks into the
    CandleBuffer → ZoneEngine → BreakoutEngine pipeline.

    Handles:
      - Auto-reconnect on disconnect
      - Single-tick JSON:  {"last_price": 24150, "timestamp": "...", "volume": 100}
      - Batch-tick JSON:   [{"last_price": ...}, {"last_price": ...}]
      - Noisy/non-JSON messages (silently skipped)
    """

    def __init__(self, ze: ZoneEngine, be: BreakoutEngine, buf: CandleBuffer):
        self.ze  = ze
        self.be  = be
        self.buf = buf
        self._tick_count = 0

    async def run(self):
        """Main async loop with auto-reconnect."""
        tries = 0
        while tries < MAX_RECONNECT_TRIES:
            try:
                await self._connect()
                tries = 0
            except OSError as e:
                tries += 1
                log.warning(
                    f"Cannot reach {WS_URL} — is your server running?  "
                    f"({e})  Retry {tries}/{MAX_RECONNECT_TRIES} "
                    f"in {RECONNECT_DELAY_SEC}s..."
                )
                await asyncio.sleep(RECONNECT_DELAY_SEC)
            except Exception as e:
                tries += 1
                log.warning(
                    f"WebSocket error: {e}  "
                    f"Retry {tries}/{MAX_RECONNECT_TRIES} "
                    f"in {RECONNECT_DELAY_SEC}s..."
                )
                await asyncio.sleep(RECONNECT_DELAY_SEC)

        log.error("Max reconnect attempts reached. Stopping.")

    async def _connect(self):
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets not installed — run:  pip install websockets"
            )

        log.info(f"Connecting to {WS_URL} ...")
        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            log.info(f"✓ Connected to {WS_URL}  — listening for ticks")
            async for raw_msg in ws:
                self._handle_message(raw_msg)

    def _handle_message(self, raw: str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.debug(f"Skipped non-JSON: {str(raw)[:60]}")
            return

        # Support both single tick and list of ticks
        ticks = data if isinstance(data, list) else [data]
        for tick in ticks:
            if isinstance(tick, dict):
                self._feed_tick(tick)

    def _feed_tick(self, tick: dict):
        self._tick_count += 1
        self.buf.on_tick(tick)

        # Heartbeat log every 100 ticks (debug level — won't clutter console)
        if self._tick_count % 100 == 0:
            ltp = (tick.get("last_price") or tick.get("ltp")
                   or tick.get("price") or "?")
            log.debug(
                f"  ♥  tick #{self._tick_count}  "
                f"LTP={ltp}  candles={self.buf.bar_count}"
            )


# ════════════════════════════════════════════════════════════════
# PIPELINE BUILDER
# ════════════════════════════════════════════════════════════════

def build_pipeline():
    """
    Build the full pipeline:
      history → ZoneEngine → BreakoutEngine → CandleBuffer
    Returns (ze, be, buf) — all wired together, ready to receive ticks.
    """
    log.info("=" * 60)
    log.info("  Nifty Breakout — Local WebSocket Mode")
    log.info("=" * 60)

    # ── History + zones ──────────────────────────────────────
    df_history = load_history()

    ze = ZoneEngine(**ZONE_CONFIG)
    ze.build_from_history(df=df_history)

    zones = ze.get_active_zones()
    cp    = float(df_history["Close"].iloc[-1])
    log.info(f"✓ Zone map: {len(zones)} active zones  (last close ≈ {cp:.0f})")
    log.info("")

    res = zones[zones["type"] == "Resistance"].sort_values("price")
    sup = zones[zones["type"] == "Support"].sort_values("price", ascending=False)

    log.info("  ┌─ RESISTANCE (nearest first) ─────────────────")
    for _, z in res.iterrows():
        log.info(f"  │  {z['price']:.0f}  [{z['lower']:.0f}–{z['upper']:.0f}]  "
                 f"str={z['strength']:.0f}  dist={z['dist_pts']:+.0f}")

    log.info(f"  ├─ CMP ~ {cp:.0f}")

    for _, z in sup.iterrows():
        log.info(f"  │  {z['price']:.0f}  [{z['lower']:.0f}–{z['upper']:.0f}]  "
                 f"str={z['strength']:.0f}  dist={z['dist_pts']:+.0f}")
    log.info("  └──────────────────────────────────────────────")
    log.info("")

    # ── Breakout engine ──────────────────────────────────────
    be = BreakoutEngine(
        zone_engine         = ze,
        on_signal           = on_signal,
        on_signal_confirmed = on_signal_confirmed,
        on_trade_closed     = on_trade_closed,
        **STRATEGY_CONFIG,
    )

    # ── Candle buffer ─────────────────────────────────────────
    # Every time a 5m candle closes:
    #   1. Feed into ZoneEngine rolling buffer (zone quality improves over session)
    #   2. Feed into BreakoutEngine (signal check)
    def on_candle_closed(candle):
        ze.add_candle(candle)
        be.on_candle(candle)

    buf = CandleBuffer(
        interval_minutes = 5,
        pivot_right_bars = ze.right_bars,
        on_candle_closed = on_candle_closed,
    )

    return ze, be, buf


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    ze, be, buf = build_pipeline()

    runner = LocalWebSocketRunner(ze, be, buf)

    log.info(f"WebSocket target : {WS_URL}")
    log.info("Press Ctrl+C to stop.\n")

    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        log.info("\nStopped by user.")
    finally:
        summary = be.get_trade_summary()
        log.info("")
        log.info("─" * 60)
        log.info("SESSION SUMMARY")
        if summary.get("trades", 0) > 0:
            log.info(f"  Trades      : {summary['trades']}")
            log.info(f"  Win / Loss  : {summary['wins']} / {summary['losses']}")
            log.info(f"  Win rate    : {summary['win_rate']}%")
            log.info(f"  Total P&L   : {summary['total_pts']:+.0f} pts")
            log.info(f"  Prof factor : {summary['profit_factor']}")
        else:
            log.info("  No trades taken this session.")
        log.info("─" * 60)


if __name__ == "__main__":
    main()
