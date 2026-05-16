"""
candle_buffer.py
================
Tick → 5-minute OHLCV candle builder with delayed pivot confirmation.

Solves Blocker 1 & 3:
  - Accumulates ticks into 5m OHLCV in real time
  - Implements the "10-bar right confirmation" window so a pivot is only
    confirmed once 10 more candles have formed to its right — matching the
    original script's right_bars=10 logic exactly.
  - Emits a 'pivot_confirmed' event when a delayed pivot is locked in.
  - Emits a 'candle_closed' event the moment each 5m bar closes, so the
    breakout engine can react immediately (without waiting for pivot confirm).

Kite Connect integration:
  - on_tick(tick_data) feeds raw tick data in
  - Compatible with Kite WebSocket tick format:
      {"instrument_token": ..., "last_price": ..., "timestamp": ...}
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from collections import deque
from typing import Callable, Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

# NSE session boundaries
SESSION_START = dtime(9, 15)
SESSION_END   = dtime(15, 30)


class CandleBuffer:
    """
    Converts real-time ticks into confirmed 5-minute OHLCV candles.

    Parameters
    ----------
    interval_minutes : int
        Candle size in minutes (default 5).
    pivot_right_bars : int
        Number of candles needed to the RIGHT before a pivot is confirmed.
        Match this with ZoneEngine.right_bars (default 10).
    on_candle_closed : Callable[[pd.Series], None]
        Called immediately when a candle closes.
    on_pivot_confirmed : Callable[[dict], None]
        Called when a delayed pivot is confirmed (10 candles later).
    """

    def __init__(
        self,
        interval_minutes:   int      = 5,
        pivot_right_bars:   int      = 10,
        on_candle_closed:   Optional[Callable] = None,
        on_pivot_confirmed: Optional[Callable] = None,
    ):
        self.interval_minutes   = interval_minutes
        self.pivot_right_bars   = pivot_right_bars
        self.on_candle_closed   = on_candle_closed
        self.on_pivot_confirmed = on_pivot_confirmed

        # Current candle being formed
        self._current_open   = None
        self._current_high   = None
        self._current_low    = None
        self._current_close  = None
        self._current_volume = 0
        self._current_ts     = None   # candle open timestamp

        # Confirmed closed candles (rolling deque)
        self._candles: deque = deque(maxlen=500)

        # Pending pivot candidates (waiting for right_bars confirmation)
        # Each entry: {"bar_idx": int, "price": float, "type": "high"/"low",
        #              "timestamp": datetime, "candles_right": int}
        self._pending_pivots: list = []

        self._bar_idx = 0   # increments each time a candle closes

    # ── Public: feed ticks in ────────────────────────────────

    def on_tick(self, tick: dict):
        """
        Feed a single tick dict.

        Accepts both local WebSocket and Kite Connect field names:
          price   : 'last_price'  | 'last_traded_price' | 'ltp' | 'price'
          time    : 'timestamp'   | 'exchange_timestamp' | 'time'
          volume  : 'volume'      | 'volume_traded'      | 'last_traded_quantity'

        Your local WebSocket format (auto-detected):
          {"last_price": 24150.5, "timestamp": "2024-01-15 09:20:00", "volume": 1200}
        """
        # ── Price ────────────────────────────────────────────
        ltp = (tick.get("last_price")
               or tick.get("last_traded_price")
               or tick.get("ltp")
               or tick.get("price"))
        if ltp is None:
            return   # malformed tick — skip silently

        # ── Timestamp ────────────────────────────────────────
        ts = (tick.get("timestamp")
              or tick.get("exchange_timestamp")
              or tick.get("time"))
        if ts is None:
            ts = datetime.now()   # fallback to system clock

        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        if hasattr(ts, "tzinfo"):
            if ts.tzinfo is None:
                try:
                    ts = IST.localize(ts)
                except Exception:
                    pass
            else:
                try:
                    ts = ts.astimezone(IST)
                except Exception:
                    pass

        # ── Volume ───────────────────────────────────────────
        vol = (tick.get("volume")                  # local WS key
               or tick.get("volume_traded")        # Kite full mode
               or tick.get("last_traded_quantity") # Kite LTP mode
               or 0)

        # ── Session filter ───────────────────────────────────
        t = ts.time() if hasattr(ts, "time") else ts
        if t < SESSION_START or t > SESSION_END:
            return

        self._process_tick(float(ltp), int(vol), ts)

    # ── Public: inject a pre-formed candle (for backtesting / replay) ──

    def inject_candle(self, candle: pd.Series):
        """
        Directly inject a closed candle (for backtesting or Kite historical replay).
        Candle must have: Open, High, Low, Close, Volume and a DatetimeIndex name.
        """
        self._close_candle_with(candle)

    # ── Public: accessors ────────────────────────────────────

    def get_candles_df(self) -> pd.DataFrame:
        """Return all confirmed candles as a DataFrame."""
        if not self._candles:
            return pd.DataFrame()
        return pd.DataFrame(list(self._candles)).set_index("timestamp")

    @property
    def latest_candle(self) -> Optional[dict]:
        return self._candles[-1] if self._candles else None

    @property
    def bar_count(self) -> int:
        return self._bar_idx

    # ── Internal: tick processing ─────────────────────────────

    def _candle_open_time(self, ts: datetime) -> datetime:
        """Floor timestamp to nearest interval boundary."""
        total_minutes = ts.hour * 60 + ts.minute
        floored       = (total_minutes // self.interval_minutes) * self.interval_minutes
        return ts.replace(hour=floored // 60, minute=floored % 60,
                          second=0, microsecond=0)

    def _process_tick(self, ltp: float, vol: int, ts: datetime):
        candle_ts = self._candle_open_time(ts)

        if self._current_ts is None:
            # Very first tick
            self._start_new_candle(ltp, vol, candle_ts)
            return

        if candle_ts > self._current_ts:
            # New candle period — close the previous one
            closed = pd.Series({
                "Open":      self._current_open,
                "High":      self._current_high,
                "Low":       self._current_low,
                "Close":     self._current_close,
                "Volume":    self._current_volume,
                "timestamp": self._current_ts,
            })
            self._close_candle_with(closed)
            self._start_new_candle(ltp, vol, candle_ts)
        else:
            # Same candle — update OHLCV
            self._current_high   = max(self._current_high, ltp)
            self._current_low    = min(self._current_low, ltp)
            self._current_close  = ltp
            self._current_volume += vol

    def _start_new_candle(self, ltp, vol, ts):
        self._current_open   = ltp
        self._current_high   = ltp
        self._current_low    = ltp
        self._current_close  = ltp
        self._current_volume = vol
        self._current_ts     = ts

    def _close_candle_with(self, candle: pd.Series):
        bar = {
            "timestamp": candle.name if hasattr(candle, "name") and candle.name is not None
                         else candle.get("timestamp"),
            "Open":   float(candle["Open"]),
            "High":   float(candle["High"]),
            "Low":    float(candle["Low"]),
            "Close":  float(candle["Close"]),
            "Volume": int(candle.get("Volume", 0)),
            "bar_idx": self._bar_idx,
        }
        self._candles.append(bar)
        self._bar_idx += 1

        # Fire immediate callback
        if self.on_candle_closed:
            self.on_candle_closed(pd.Series(bar))

        # Age all pending pivots by 1 candle
        self._age_pending_pivots(bar)

        # Check if the candle that just closed could be a new pivot candidate
        self._check_new_pivot_candidate()

    # ── Delayed pivot confirmation ─────────────────────────────

    def _check_new_pivot_candidate(self):
        """
        Look at the bar `pivot_right_bars` positions ago — we now have
        enough bars to its right to check if it's a true pivot.
        We don't confirm here; we add it to pending and let
        _age_pending_pivots fire the callback once enough right bars accrue.

        To match original logic, we actually start checking from the moment
        a potential candidate emerges and count right bars from there.
        """
        candles = list(self._candles)
        n = len(candles)
        if n < 3:
            return

        # The candidate is the bar right in the middle of the confirmed window
        # Since we need right_bars to the right, the candidate is bar at
        # position [n - 1 - right_bars_so_far]. We add fresh candidates
        # for the bar that just got its first right-confirmation (bar n-2).
        candidate_idx_in_deque = n - 2  # second-to-last bar
        if candidate_idx_in_deque < 1:
            return

        c      = candles[candidate_idx_in_deque]
        c_prev = candles[candidate_idx_in_deque - 1]
        c_next = candles[-1]   # the newly closed bar = first right bar

        # Naive pivot candidate check (full confirmation comes after right_bars)
        is_high_candidate = c["High"] > c_prev["High"] and c["High"] > c_next["High"]
        is_low_candidate  = c["Low"]  < c_prev["Low"]  and c["Low"]  < c_next["Low"]

        if is_high_candidate:
            self._pending_pivots.append({
                "bar_idx":       c["bar_idx"],
                "price":         c["High"],
                "type":          "high",
                "timestamp":     c["timestamp"],
                "candles_right": 1,
                "confirmed":     False,
            })
        if is_low_candidate:
            self._pending_pivots.append({
                "bar_idx":       c["bar_idx"],
                "price":         c["Low"],
                "type":          "low",
                "timestamp":     c["timestamp"],
                "candles_right": 1,
                "confirmed":     False,
            })

    def _age_pending_pivots(self, new_bar: dict):
        """
        Increment right-bar count for all pending pivots.
        Confirm a pivot once it has pivot_right_bars bars to its right
        AND it is still the extreme in that window.
        """
        candles = list(self._candles)
        confirmed_indices = []

        for i, piv in enumerate(self._pending_pivots):
            if piv["confirmed"]:
                continue
            piv["candles_right"] += 1

            if piv["candles_right"] >= self.pivot_right_bars:
                # Verify it is still the true extreme over the full window
                bar_pos = next(
                    (j for j, c in enumerate(candles) if c["bar_idx"] == piv["bar_idx"]),
                    None
                )
                if bar_pos is None:
                    confirmed_indices.append(i)   # bar scrolled out — drop
                    continue

                right_slice = candles[bar_pos + 1:]
                if piv["type"] == "high":
                    right_vals = [c["High"] for c in right_slice]
                    is_still_pivot = all(piv["price"] > v for v in right_vals)
                else:
                    right_vals = [c["Low"] for c in right_slice]
                    is_still_pivot = all(piv["price"] < v for v in right_vals)

                piv["confirmed"] = True
                confirmed_indices.append(i)

                if is_still_pivot and self.on_pivot_confirmed:
                    self.on_pivot_confirmed(piv)

        # Remove processed pivots (reverse order to keep indices stable)
        for i in sorted(set(confirmed_indices), reverse=True):
            self._pending_pivots.pop(i)
