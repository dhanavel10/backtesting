"""
slope_zones.py — Intraday Triangle / Wedge / Channel Pattern Detector
======================================================================
Detects converging-trendline patterns ("sloppy zones") formed during
intraday volatility compression. Reuses the same ZigZag pivot stream
that drives realtime_sr.py / live_sr.py.

Pattern types detected:
    • Symmetrical Triangle      (highs fall, lows rise)
    • Ascending Triangle        (flat resistance, rising support)
    • Descending Triangle       (falling resistance, flat support)
    • Rising  Wedge             (both rising, upper less steep)
    • Falling Wedge             (both falling, upper less steep down)
    • Rising / Falling Channel  (parallel trendlines)
    • Broadening / Expanding    (diverging trendlines)
    • Sideways Rectangle        (both flat)

Each detection reports:
    • start_time / end_time     — pattern formation window
    • upper / lower anchors     — pivot timestamps + prices
    • slope (pts/hr) and R²     — line quality
    • apex_time / apex_price    — projected convergence
    • width_start / width_end   — compression measurement
    • strength                  — composite score 0..100

Modes:
    python slope_zones.py backtest      # yfinance simulated ticks
    python slope_zones.py live          # WebSocket tick feed (ws://localhost:8086)
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from realtime_sr import (
    RealtimeSREngine, Pivot, simulate_ticks_from_ohlc,
)


# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

WS_HOST   = os.getenv("WS_HOST", "localhost")
WS_PORT   = int(os.getenv("WS_PORT", "8086"))
WS_PATH   = os.getenv("WS_PATH",    "/ws")
WS_URL    = f"ws://{WS_HOST}:{WS_PORT}{WS_PATH}"

# Range-bar / pivot engine — keep aligned with live_sr.py defaults
RANGE_SIZE          = float(os.getenv("RANGE_SIZE",   "4.0"))
REVERSAL_THRESHOLD  = float(os.getenv("REVERSAL_THR", "12.0"))

# Pattern detector
MIN_TOUCHES         = int(os.getenv("MIN_TOUCHES",    "2"))      # per trendline
MIN_R2              = float(os.getenv("MIN_R2",       "0.55"))   # average fit quality
FLAT_SLOPE_PTS_HR   = float(os.getenv("FLAT_SLOPE_HR","5.0"))    # |slope| < this → flat
MAX_COMPRESSION     = float(os.getenv("MAX_COMPRESS", "0.75"))   # width_end / width_start

# Sliding lookback windows scanned every bar (minutes)
WINDOWS_MIN = [60, 120, 180]

PRINT_INTERVAL_SECS = int(os.getenv("PATTERN_INTERVAL", "30"))
RECONNECT_DELAY     = int(os.getenv("RECONNECT_DELAY",  "3"))


# ════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class TrendLine:
    slope:     float                    # price units per SECOND
    intercept: float                    # price at t=0 of fit window
    r2:        float                    # 0..1 fit quality
    anchors:   list                     # [(datetime, price), ...]

    def price_at(self, t_secs: float) -> float:
        return self.slope * t_secs + self.intercept

    def slope_per_hour(self) -> float:
        return self.slope * 3600.0


@dataclass
class TrianglePattern:
    kind:        str
    window_min:  int
    start_time:  datetime
    end_time:    datetime
    upper:       TrendLine
    lower:       TrendLine
    apex_time:   Optional[datetime]
    apex_price:  Optional[float]
    width_start: float
    width_end:   float
    compression: float                  # width_end / width_start  (1 = none, <1 = compressing)
    strength:    float                  # 0..100 composite
    confirmed:   bool                   # passed all quality gates

    @property
    def n_upper_touches(self) -> int: return len(self.upper.anchors)

    @property
    def n_lower_touches(self) -> int: return len(self.lower.anchors)

    @property
    def duration_min(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60.0


# ════════════════════════════════════════════════════════════════
# LINEAR FIT  (least-squares)
# ════════════════════════════════════════════════════════════════

def _fit_line(anchors: list, t0: datetime) -> Optional[TrendLine]:
    if len(anchors) < 2:
        return None
    xs = np.array([(t - t0).total_seconds() for t, _ in anchors], dtype=float)
    ys = np.array([p for _, p in anchors], dtype=float)

    x_mean = xs.mean()
    y_mean = ys.mean()
    sx     = xs - x_mean
    sy     = ys - y_mean
    denom  = float((sx ** 2).sum())

    if denom == 0:
        slope, intercept = 0.0, float(y_mean)
        r2 = 1.0 if np.allclose(ys, y_mean) else 0.0
    else:
        slope     = float((sx * sy).sum() / denom)
        intercept = float(y_mean - slope * x_mean)
        y_hat     = slope * xs + intercept
        ss_res    = float(((ys - y_hat) ** 2).sum())
        ss_tot    = float(((ys - y_mean) ** 2).sum())
        r2        = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return TrendLine(
        slope=slope,
        intercept=intercept,
        r2=max(0.0, min(1.0, r2)),
        anchors=list(anchors),
    )


# ════════════════════════════════════════════════════════════════
# CLASSIFIER
# ════════════════════════════════════════════════════════════════

def _classify(upper: TrendLine, lower: TrendLine,
              flat_pts_hr: float = FLAT_SLOPE_PTS_HR) -> str:
    us = upper.slope_per_hour()
    ls = lower.slope_per_hour()
    flat_u = abs(us) < flat_pts_hr
    flat_l = abs(ls) < flat_pts_hr

    if flat_u and flat_l:
        return "Sideways Rectangle"
    if flat_u and ls > 0:
        return "Ascending Triangle"
    if flat_l and us < 0:
        return "Descending Triangle"
    if us < 0 and ls > 0:
        return "Symmetrical Triangle"
    if us > 0 and ls < 0:
        return "Broadening / Expanding"
    if us > 0 and ls > 0:
        return "Rising Wedge" if us < ls else "Rising Channel"
    if us < 0 and ls < 0:
        return "Falling Wedge" if us > ls else "Falling Channel"
    return "Unclassified"


# ════════════════════════════════════════════════════════════════
# DETECTOR
# ════════════════════════════════════════════════════════════════

class TrianglePatternDetector:
    """
    Scans the ZigZag pivot list over multiple lookback windows
    (default 60 / 120 / 180 minutes) for converging-trendline patterns.
    """

    def __init__(
        self,
        windows_min:        list = None,
        min_touches:        int   = MIN_TOUCHES,
        min_r2:             float = MIN_R2,
        flat_slope_pts_hr:  float = FLAT_SLOPE_PTS_HR,
        max_compression:    float = MAX_COMPRESSION,
    ):
        self.windows_min     = windows_min or list(WINDOWS_MIN)
        self.min_touches     = min_touches
        self.min_r2          = min_r2
        self.flat_pts_hr     = flat_slope_pts_hr
        self.max_compression = max_compression

    def detect(self, pivots: list, now: datetime) -> list:
        out = []
        for w in self.windows_min:
            patt = self._detect_window(pivots, now, w)
            if patt is not None:
                out.append(patt)
        return out

    def _detect_window(self, pivots, now: datetime, window_min: int) -> Optional[TrianglePattern]:
        cutoff = now - timedelta(minutes=window_min)
        wp = [p for p in pivots if p.time >= cutoff]
        if len(wp) < 4:
            return None

        highs = [(p.time, p.price) for p in wp if p.type == "high"]
        lows  = [(p.time, p.price) for p in wp if p.type == "low"]

        if len(highs) < self.min_touches or len(lows) < self.min_touches:
            return None

        t0    = wp[0].time
        upper = _fit_line(highs, t0)
        lower = _fit_line(lows,  t0)
        if upper is None or lower is None:
            return None

        start_time = min(p.time for p in wp)
        end_time   = max(p.time for p in wp)
        x_start    = (start_time - t0).total_seconds()
        x_end      = (end_time   - t0).total_seconds()

        width_start = upper.price_at(x_start) - lower.price_at(x_start)
        width_end   = upper.price_at(x_end)   - lower.price_at(x_end)
        if width_start <= 0 or width_end <= 0:
            return None
        compression = width_end / width_start

        # Apex (lines intersect) — only if convergence is in the near future
        apex_time = apex_price = None
        if upper.slope != lower.slope:
            apex_x = (lower.intercept - upper.intercept) / (upper.slope - lower.slope)
            if x_end <= apex_x <= x_end + window_min * 60 * 3:
                apex_time  = t0 + timedelta(seconds=apex_x)
                apex_price = upper.price_at(apex_x)

        kind   = _classify(upper, lower, self.flat_pts_hr)
        avg_r2 = (upper.r2 + lower.r2) / 2.0

        is_convergent = ("Triangle" in kind) or ("Wedge" in kind)
        confirmed = (avg_r2 >= self.min_r2
                     and compression <= self.max_compression
                     and is_convergent)

        # Strength: blend of fit quality + tightness + touch count
        r2_score          = avg_r2 * 100.0
        compression_score = max(0.0, 1.0 - compression) * 100.0
        touch_bonus       = min(30.0, (len(highs) + len(lows) - 4) * 5.0)
        strength = round(0.4 * r2_score + 0.4 * compression_score + 0.2 * touch_bonus, 1)

        return TrianglePattern(
            kind=kind,
            window_min=window_min,
            start_time=start_time,
            end_time=end_time,
            upper=upper,
            lower=lower,
            apex_time=apex_time,
            apex_price=apex_price,
            width_start=round(width_start, 2),
            width_end=round(width_end, 2),
            compression=round(compression, 3),
            strength=strength,
            confirmed=confirmed,
        )


# ════════════════════════════════════════════════════════════════
# CONSOLE OUTPUT
# ════════════════════════════════════════════════════════════════

SEP = "=" * 88

def print_patterns(patterns: list, cmp: float, trigger: str, ts: datetime):
    print(f"\n{SEP}")
    print(f"  SLOPE / TRIANGLE ZONES  [{trigger}]  CMP: {cmp:.2f}  @  {ts.strftime('%H:%M:%S')}")
    print(SEP)
    if not patterns:
        print("  (no converging patterns detected in any window)")
        print(SEP); return

    for p in patterns:
        flag = "★ CONFIRMED" if p.confirmed else "  forming  "
        print(f"\n  [{p.window_min:>3}-min]  {p.kind:<24}  {flag}   "
              f"strength:{p.strength:>5.1f}")
        print(f"     formed   : {p.start_time.strftime('%H:%M:%S')}  →  "
              f"{p.end_time.strftime('%H:%M:%S')}   ({p.duration_min:.0f} min)")

        print(f"     upper    : slope={p.upper.slope_per_hour():>+7.2f} pts/hr   "
              f"R²={p.upper.r2:.2f}   touches={p.n_upper_touches}")
        for t, px in p.upper.anchors:
            print(f"          ↳ high @ {t.strftime('%H:%M:%S')}   {px:>8.2f}")

        print(f"     lower    : slope={p.lower.slope_per_hour():>+7.2f} pts/hr   "
              f"R²={p.lower.r2:.2f}   touches={p.n_lower_touches}")
        for t, px in p.lower.anchors:
            print(f"          ↳ low  @ {t.strftime('%H:%M:%S')}   {px:>8.2f}")

        print(f"     width    : start={p.width_start:>6.1f}  →  end={p.width_end:>6.1f}   "
              f"compression={p.compression:.2f}")
        if p.apex_time:
            print(f"     apex     : {p.apex_time.strftime('%H:%M:%S')}  @  {p.apex_price:.2f}")
        else:
            print(f"     apex     : (lines do not converge inside projection window)")
    print(f"\n{SEP}")


# ════════════════════════════════════════════════════════════════
# BACKTEST  (yfinance simulated ticks)
# ════════════════════════════════════════════════════════════════

def run_backtest():
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

    engine   = RealtimeSREngine(range_size=RANGE_SIZE,
                                reversal_threshold=REVERSAL_THRESHOLD)
    detector = TrianglePatternDetector()

    def _floor5(ts: datetime) -> datetime:
        return ts.replace(second=0, microsecond=0, minute=(ts.minute // 5) * 5)

    cur_slot = cur_o = cur_h = cur_l = cur_c = None
    cur_ticks = bar_num = n_ticks = 0
    last_ts = last_px = None

    print(f"{SEP}")
    print(f"  BACKTEST  —  range={RANGE_SIZE}  reversal={REVERSAL_THRESHOLD}  "
          f"windows={WINDOWS_MIN} min")
    print(f"{SEP}")

    for ts, px in simulate_ticks_from_ohlc(df, ticks_per_bar=15):
        engine.on_tick(px, ts)
        n_ticks += 1
        last_ts, last_px = ts, px
        slot = _floor5(ts)

        if cur_slot is None:
            cur_slot, cur_o, cur_h, cur_l, cur_c, cur_ticks = slot, px, px, px, px, 1
        elif slot == cur_slot:
            cur_h = max(cur_h, px); cur_l = min(cur_l, px)
            cur_c = px;             cur_ticks += 1
        else:
            bar_num += 1
            d = "^" if cur_c >= cur_o else "v"
            print(f"\n  BAR {bar_num:>3} {d}  {cur_slot.strftime('%H:%M')}  "
                  f"O:{cur_o:.1f} H:{cur_h:.1f} L:{cur_l:.1f} C:{cur_c:.1f}  "
                  f"ticks:{cur_ticks}  pivots:{len(engine.zz.pivots)}")
            patterns = detector.detect(engine.zz.pivots, now=ts)
            if patterns:
                print_patterns(patterns, px, "BAR_CLOSE", ts)
            cur_slot, cur_o, cur_h, cur_l, cur_c, cur_ticks = slot, px, px, px, px, 1

    if last_ts is not None:
        patterns = detector.detect(engine.zz.pivots, now=last_ts)
        print(f"\n{SEP}")
        print(f"  END-OF-DAY SUMMARY   bars={bar_num}   pivots={len(engine.zz.pivots)}   "
              f"ticks={n_ticks}")
        print(SEP)
        print_patterns(patterns, last_px, "END_OF_DAY", last_ts)


# ════════════════════════════════════════════════════════════════
# LIVE  (WebSocket tick feed)
# ════════════════════════════════════════════════════════════════

def parse_tick(raw: str):
    now = datetime.now()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            return float(raw.strip()), now
        except ValueError:
            return None, now

    if isinstance(data, list):
        price  = float(data[0]) if data else 0
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
                    "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                    "%d-%m-%Y %H:%M:%S"):
            try:
                return price, datetime.strptime(ts_raw, fmt)
            except ValueError:
                continue
    return price, now


async def run_live():
    import websockets

    engine   = RealtimeSREngine(range_size=RANGE_SIZE,
                                reversal_threshold=REVERSAL_THRESHOLD)
    detector = TrianglePatternDetector()

    def _floor5(ts): return ts.replace(second=0, microsecond=0,
                                       minute=(ts.minute // 5) * 5)
    cur_slot = cur_o = cur_h = cur_l = cur_c = None
    cur_ticks = bar_count = tick_count = 0
    last_print = datetime.now()

    print(f"\n{SEP}")
    print(f"  LIVE  tick feed : {WS_URL}")
    print(f"  range={RANGE_SIZE}  reversal={REVERSAL_THRESHOLD}  "
          f"windows={WINDOWS_MIN} min")
    print(f"{SEP}\n")

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print(f"  Connected to {WS_URL}\n")
                async for raw_msg in ws:
                    price, ts = parse_tick(str(raw_msg))
                    if price is None:
                        continue

                    tick_count += 1
                    engine.on_tick(price, ts)
                    slot = _floor5(ts)

                    if cur_slot is None:
                        cur_slot, cur_o, cur_h, cur_l, cur_c, cur_ticks = slot, price, price, price, price, 1
                        continue

                    if slot == cur_slot:
                        cur_h = max(cur_h, price); cur_l = min(cur_l, price)
                        cur_c = price;             cur_ticks += 1
                    else:
                        # ── 5-min bar closed → always run pattern detection
                        bar_count += 1
                        d = "^" if cur_c >= cur_o else "v"
                        print(f"\n  CANDLE {d} {cur_slot.strftime('%H:%M')}  "
                              f"O:{cur_o:.1f} H:{cur_h:.1f} L:{cur_l:.1f} C:{cur_c:.1f}  "
                              f"ticks:{cur_ticks}  pivots:{len(engine.zz.pivots)}")
                        patterns = detector.detect(engine.zz.pivots, now=ts)
                        print_patterns(patterns, price, "BAR_CLOSE", ts)
                        last_print = datetime.now()
                        cur_slot, cur_o, cur_h, cur_l, cur_c, cur_ticks = slot, price, price, price, price, 1
                        continue

                    # ── Intra-bar refresh
                    if (PRINT_INTERVAL_SECS > 0 and
                        (datetime.now() - last_print).total_seconds() >= PRINT_INTERVAL_SECS):
                        patterns = detector.detect(engine.zz.pivots, now=ts)
                        print_patterns(patterns, price, "LIVE_UPDATE", ts)
                        last_print = datetime.now()

        except (ConnectionRefusedError, OSError) as e:
            print(f"  Tick feed unavailable: {e}  — retry in {RECONNECT_DELAY}s")
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as e:
            print(f"  Error: {e}  — retry in {RECONNECT_DELAY}s")
            await asyncio.sleep(RECONNECT_DELAY)


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "backtest"

    if mode == "live":
        print("\nPress Ctrl-C to stop.\n")
        try:
            asyncio.run(run_live())
        except KeyboardInterrupt:
            print("\n  Stopped.")
    elif mode == "backtest":
        run_backtest()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage:  python slope_zones.py [backtest|live]")
        sys.exit(1)
