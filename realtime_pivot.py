"""
Time-Bounded Intraday S/R Zone Detector — NIFTY 50
====================================================

What's different from the original:
  • Zones are detected within SLIDING TIME WINDOWS inside each session
    (e.g. 09:15–10:30, 10:00–11:30, 11:00–13:00 …) so short-lived
    intra-session consolidations are caught, not just day-level pivots.
  • Every zone carries exact zone_start and zone_end timestamps.
  • Intraday Score (0–100) is computed purely from same-session data.
  • Previous-day S/R is NEVER mixed in — each session is independent.
  • Support and Resistance are reported separately with entry/exit context.

Install
-------
  pip install pandas numpy scipy plotly yfinance colorama
"""

# ════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BLUE   = Fore.BLUE
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BLUE = RESET = ""


# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════
class Config:
    TICKER            = "^NSEI"
    INTERVAL          = "5m"
    BACKTEST_DAYS     = 30          # how many calendar days to look back

    # ATR
    ATR_PERIOD        = 14
    ATR_MULT          = 0.7         # pivot confirmed when reversal ≥ ATR_MULT × ATR

    # Clustering
    CLUSTER_ATR_MULT  = 0.8         # pivots within N×ATR merge into one zone
    MIN_PIVOTS_ZONE   = 2           # at least 2 pivots to form a zone
    MIN_TOUCHES       = 2           # minimum price touches to qualify

    # Time-window scanning (all times IST HH:MM)
    # Each tuple is (window_start, window_end) in minutes from 09:15
    # We slide a window of WINDOW_BARS bars with SLIDE_BARS step
    WINDOW_BARS       = 18          # bars in one scanning window (~90 min on 5m)
    SLIDE_BARS        = 6           # slide step (~30 min)
    MIN_WINDOW_BARS   = 10          # ignore windows shorter than this

    # Zone validity
    MIN_ZONE_BARS     = 6           # zone must span at least N bars (~30 min)
    SLOPE_THRESHOLD   = 0.05        # pts/bar; below → flat zone

    # Output
    TOP_N_PER_SESSION = 20
    SHOW_CHART        = True
    CHART_DAYS        = 3           # how many recent sessions to chart


# ════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════
@dataclass
class Candle:
    open:    float
    high:    float
    low:     float
    close:   float
    volume:  float
    time:    datetime
    bar_idx: int = 0


@dataclass
class Pivot:
    price:   float
    time:    datetime
    bar_idx: int
    kind:    str        # "high" | "low"
    atr_at:  float


@dataclass
class TimeBoundedZone:
    """
    An S/R zone that was ACTIVE during [zone_start, zone_end].
    Contains exactly the pivots that formed inside that window.
    """
    pivots:      List[Pivot]
    kind:        str            # "support" | "resistance" | "both"
    session:     date

    zone_start:  datetime       # time of first confirming pivot
    zone_end:    datetime       # time of last confirming pivot

    mid:         float          # zone centre price
    upper:       float
    lower:       float
    band:        float          # upper - lower

    slope:       float = 0.0
    intercept:   float = 0.0

    # Scoring components
    touch_count:    int   = 0
    duration_bars:  int   = 0   # bars the zone was active
    atr_tightness:  float = 0.0 # band / atr  (lower = tighter = better)
    reversal_count: int   = 0   # how many times price reversed at zone
    intraday_score: float = 0.0 # 0–100

    def price_at(self, bar_idx: int) -> float:
        if abs(self.slope) < Config.SLOPE_THRESHOLD:
            return self.mid
        return self.slope * bar_idx + self.intercept


# ════════════════════════════════════════════════════════════════
# ATR CALCULATOR
# ════════════════════════════════════════════════════════════════
class ATRCalculator:
    def __init__(self, period: int = Config.ATR_PERIOD):
        self.period      = period
        self.atr: Optional[float] = None
        self._prev_close: Optional[float] = None

    def update(self, candle: Candle) -> Optional[float]:
        if self._prev_close is None:
            self._prev_close = candle.close
            return None
        tr = max(
            candle.high - candle.low,
            abs(candle.high - self._prev_close),
            abs(candle.low  - self._prev_close),
        )
        self.atr = tr if self.atr is None else (
            (self.atr * (self.period - 1) + tr) / self.period
        )
        self._prev_close = candle.close
        return self.atr

    @staticmethod
    def batch(candles: List[Candle], period: int = Config.ATR_PERIOD) -> List[Optional[float]]:
        calc = ATRCalculator(period)
        return [calc.update(c) for c in candles]


# ════════════════════════════════════════════════════════════════
# PIVOT DETECTOR  (no look-ahead)
# ════════════════════════════════════════════════════════════════
class ATRPivotDetector:
    def __init__(self, atr_mult: float = Config.ATR_MULT):
        self.atr_mult  = atr_mult
        self.atr_calc  = ATRCalculator()
        self.pivots:   List[Pivot] = []
        self._swing_high: Optional[Candle] = None
        self._swing_low:  Optional[Candle] = None
        self._bar_idx: int = 0

    def update(self, candle: Candle) -> List[Pivot]:
        candle.bar_idx = self._bar_idx
        atr = self.atr_calc.update(candle)
        new = []

        if atr is None:
            self._bar_idx += 1
            return new

        thr = self.atr_mult * atr

        if self._swing_high is None or candle.high > self._swing_high.high:
            self._swing_high = candle
        if self._swing_low  is None or candle.low  < self._swing_low.low:
            self._swing_low  = candle

        if (self._swing_high is not None and
                self._swing_high.bar_idx < self._bar_idx and
                (self._swing_high.high - candle.close) >= thr):
            p = Pivot(price=self._swing_high.high, time=self._swing_high.time,
                      bar_idx=self._swing_high.bar_idx, kind="high", atr_at=atr)
            self.pivots.append(p); new.append(p)
            self._swing_high = candle

        if (self._swing_low is not None and
                self._swing_low.bar_idx < self._bar_idx and
                (candle.close - self._swing_low.low) >= thr):
            p = Pivot(price=self._swing_low.low, time=self._swing_low.time,
                      bar_idx=self._swing_low_idx if hasattr(self, '_swing_low_idx') else self._bar_idx,
                      kind="low", atr_at=atr)
            self.pivots.append(p); new.append(p)
            self._swing_low = candle

        self._bar_idx += 1
        return new

    def reset(self):
        self.pivots = []
        self.atr_calc = ATRCalculator()
        self._swing_high = self._swing_low = None
        self._bar_idx = 0


# ════════════════════════════════════════════════════════════════
# TIME-WINDOW ZONE SCANNER
# ════════════════════════════════════════════════════════════════
class TimeWindowZoneScanner:
    """
    Scans each intraday session with a sliding window.
    For each window, builds S/R zones from pivots in that window only.
    Deduplicates overlapping zones at the end.
    """

    # ── internal helpers ────────────────────────────────────────
    @staticmethod
    def _cluster_pivots(pivots: List[Pivot], tolerance: float) -> List[List[Pivot]]:
        if not pivots:
            return []
        sorted_p = sorted(pivots, key=lambda p: p.price)
        clusters = [[sorted_p[0]]]
        for piv in sorted_p[1:]:
            if piv.price - clusters[-1][-1].price <= tolerance:
                clusters[-1].append(piv)
            else:
                clusters.append([piv])
        return clusters

    @staticmethod
    def _count_touches(candles: List[Candle], lo: float, hi: float) -> int:
        return sum(1 for c in candles if c.high >= lo and c.low <= hi)

    @staticmethod
    def _count_reversals(candles: List[Candle], lo: float, hi: float) -> int:
        """Count direction changes that happened within the zone band."""
        rev = 0
        inside_prev = False
        direction_prev = 0
        for i, c in enumerate(candles):
            inside = c.high >= lo and c.low <= hi
            if inside and i > 0:
                direction = 1 if c.close > candles[i-1].close else -1
                if inside_prev and direction != direction_prev and direction_prev != 0:
                    rev += 1
                direction_prev = direction
            inside_prev = inside
        return rev

    @staticmethod
    def _fit_zone(cluster: List[Pivot], candles_in_window: List[Candle],
                  atr: float, session: date) -> Optional[TimeBoundedZone]:
        if len(cluster) < Config.MIN_PIVOTS_ZONE:
            return None

        n_high = sum(1 for p in cluster if p.kind == "high")
        n_low  = sum(1 for p in cluster if p.kind == "low")
        kind   = "resistance" if n_high > n_low else (
                 "support"    if n_low  > n_high else "both")

        xs = np.array([p.bar_idx for p in cluster], dtype=float)
        ys = np.array([p.price   for p in cluster], dtype=float)

        if len(xs) >= 2:
            slope, intercept, *_ = stats.linregress(xs, ys)
        else:
            slope, intercept = 0.0, float(ys[0])

        mid   = float(np.mean(ys))
        half  = 0.5 * atr
        lo, hi = mid - half, mid + half
        band  = hi - lo

        touches   = TimeWindowZoneScanner._count_touches(candles_in_window, lo, hi)
        reversals = TimeWindowZoneScanner._count_reversals(candles_in_window, lo, hi)

        if touches < Config.MIN_TOUCHES:
            return None

        zone_start = min(p.time for p in cluster)
        zone_end   = max(p.time for p in cluster)
        dur_bars   = max(p.bar_idx for p in cluster) - min(p.bar_idx for p in cluster)

        if dur_bars < Config.MIN_ZONE_BARS:
            return None

        # ── Intraday Score ─────────────────────────────────────────
        # Component 1: touch density (normalised to 10 touches = 1.0)
        touch_score = min(touches / 10, 1.0)

        # Component 2: duration (normalised to 24 bars = 1.0)
        dur_score = min(dur_bars / 24, 1.0)

        # Component 3: tightness — smaller band relative to ATR is better
        tightness = band / atr if atr > 0 else 1.0
        tight_score = max(0, 1 - tightness)

        # Component 4: reversals (normalised to 4 = 1.0)
        rev_score = min(reversals / 4, 1.0)

        # Component 5: pivot count
        piv_score = min(len(cluster) / 6, 1.0)

        intraday_score = round(
            (touch_score * 30 +
             dur_score   * 20 +
             tight_score * 20 +
             rev_score   * 20 +
             piv_score   * 10), 1
        )

        return TimeBoundedZone(
            pivots         = cluster,
            kind           = kind,
            session        = session,
            zone_start     = zone_start,
            zone_end       = zone_end,
            mid            = round(mid, 2),
            upper          = round(hi, 2),
            lower          = round(lo, 2),
            band           = round(band, 2),
            slope          = round(slope, 4),
            intercept      = round(intercept, 4),
            touch_count    = touches,
            duration_bars  = dur_bars,
            atr_tightness  = round(tightness, 3),
            reversal_count = reversals,
            intraday_score = intraday_score,
        )

    # ── public API ───────────────────────────────────────────────
    def scan_session(self, candles: List[Candle], sess_date: date,
                     atr_values: List[Optional[float]]) -> List[TimeBoundedZone]:
        """
        Slide a window of WINDOW_BARS bars across the session.
        Detect pivots and build zones per window, then deduplicate.
        """
        n = len(candles)
        all_zones: List[TimeBoundedZone] = []

        step    = Config.SLIDE_BARS
        win     = Config.WINDOW_BARS

        for start in range(0, n - Config.MIN_WINDOW_BARS, step):
            end = min(start + win, n)
            if end - start < Config.MIN_WINDOW_BARS:
                continue

            window_candles = candles[start:end]

            # Fresh pivot detector for this window (no cross-window bleed)
            det = ATRPivotDetector(atr_mult=Config.ATR_MULT)
            for c in window_candles:
                det.update(c)

            if len(det.pivots) < Config.MIN_PIVOTS_ZONE:
                continue

            # Use the ATR at the end of the window
            w_atr = None
            for i in range(end - 1, start - 1, -1):
                if atr_values[i] is not None:
                    w_atr = atr_values[i]
                    break
            if w_atr is None:
                continue

            tolerance = Config.CLUSTER_ATR_MULT * w_atr
            clusters  = self._cluster_pivots(det.pivots, tolerance)

            for cluster in clusters:
                zone = self._fit_zone(cluster, window_candles, w_atr, sess_date)
                if zone is not None:
                    all_zones.append(zone)

        return self._deduplicate(all_zones)

    @staticmethod
    def _deduplicate(zones: List[TimeBoundedZone]) -> List[TimeBoundedZone]:
        """
        Remove near-duplicate zones (same kind, overlapping price band,
        overlapping time). Keep the one with higher intraday_score.
        """
        zones = sorted(zones, key=lambda z: z.intraday_score, reverse=True)
        kept: List[TimeBoundedZone] = []

        for z in zones:
            duplicate = False
            for k in kept:
                if z.kind != k.kind:
                    continue
                # Price overlap
                price_overlap = (z.lower <= k.upper and z.upper >= k.lower)
                # Time overlap
                time_overlap  = (z.zone_start <= k.zone_end and
                                 z.zone_end   >= k.zone_start)
                if price_overlap and time_overlap:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(z)

        return kept


# ════════════════════════════════════════════════════════════════
# DATA FETCHING
# ════════════════════════════════════════════════════════════════
def fetch_data(ticker: str = Config.TICKER,
               days:   int  = Config.BACKTEST_DAYS,
               interval: str = Config.INTERVAL) -> pd.DataFrame:
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    chunks, cursor = [], start_dt

    print(f"  Fetching {ticker} ({interval}) for last {days} days …")
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=7), end_dt)
        try:
            chunk = yf.download(
                ticker,
                start    = cursor.strftime("%Y-%m-%d"),
                end      = chunk_end.strftime("%Y-%m-%d"),
                interval = interval,
                auto_adjust = True,
                progress = False,
            )
            if len(chunk):
                chunk.columns = [c[0] if isinstance(c, tuple) else c
                                  for c in chunk.columns]
                chunks.append(chunk)
        except Exception as e:
            print(f"  Chunk error: {e}")
        cursor = chunk_end

    if not chunks:
        raise ValueError(f"No data for {ticker}")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    try:
        df.index = df.index.tz_convert("Asia/Kolkata")
    except TypeError:
        try:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        except Exception:
            pass

    df = df.between_time("09:15", "15:30")
    return df


def df_to_candles(df: pd.DataFrame) -> List[Candle]:
    candles = []
    for i, (dt, row) in enumerate(df.iterrows()):
        c = Candle(
            open   = float(row["Open"]),
            high   = float(row["High"]),
            low    = float(row["Low"]),
            close  = float(row["Close"]),
            volume = float(row.get("Volume", 0)),
            time   = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt,
            bar_idx= i,
        )
        candles.append(c)
    return candles


# ════════════════════════════════════════════════════════════════
# CONSOLE PRINTING
# ════════════════════════════════════════════════════════════════
def print_session_zones(sess_date: date, zones: List[TimeBoundedZone],
                         close: float, atr: float):
    supports    = sorted([z for z in zones if z.kind == "support"],
                          key=lambda z: z.intraday_score, reverse=True)
    resistances = sorted([z for z in zones if z.kind == "resistance"],
                          key=lambda z: z.intraday_score, reverse=True)
    both        = sorted([z for z in zones if z.kind == "both"],
                          key=lambda z: z.intraday_score, reverse=True)

    n_zones = len(zones)
    print(f"\n{'═'*110}")
    print(f"  {CYAN}{sess_date}  ({n_zones} zones){RESET}  "
          f"│  Close: {YELLOW}{close:.2f}{RESET}  │  ATR: {atr:.1f}")
    print(f"{'═'*110}")

    header = (f"  {'TYPE':<11} {'MID':>8} {'LOWER':>8} {'UPPER':>8} "
              f"{'BAND':>6} {'TCH':>4} {'REV':>4} {'DUR':>4} "
              f"{'SCORE':>6}  {'ZONE START':<18} {'ZONE END':<18}  {'DIST':>7}")
    print(header)
    print("  " + "─" * 106)

    def _row(zone: TimeBoundedZone):
        dist = zone.mid - close
        slp  = (f"slope={zone.slope:+.3f}"
                if abs(zone.slope) >= Config.SLOPE_THRESHOLD else "flat")
        if zone.kind == "resistance":
            c, sym = RED,    "▼ RES"
        elif zone.kind == "support":
            c, sym = GREEN,  "▲ SUP"
        else:
            c, sym = YELLOW, "● BOTH"

        score_str = f"{zone.intraday_score:>5.1f}"
        start_str = zone.zone_start.strftime("%H:%M")
        end_str   = zone.zone_end.strftime("%H:%M")

        print(f"  {c}{sym:<11}{RESET} "
              f"{zone.mid:>8.1f} {zone.lower:>8.1f} {zone.upper:>8.1f} "
              f"{zone.band:>6.1f} {zone.touch_count:>4} {zone.reversal_count:>4} "
              f"{zone.duration_bars:>4} "
              f"{c}{score_str}{RESET}  "
              f"{c}{start_str:<18}{RESET} {end_str:<18}  {dist:>+7.0f}  {slp}")

    if resistances:
        print(f"\n  {RED}── RESISTANCE ──{RESET}")
        for z in resistances:
            _row(z)

    if both:
        print(f"\n  {YELLOW}── BOTH (S&R) ──{RESET}")
        for z in both:
            _row(z)

    if supports:
        print(f"\n  {GREEN}── SUPPORT ──{RESET}")
        for z in supports:
            _row(z)

    print()


# ════════════════════════════════════════════════════════════════
# PLOTLY CHART
# ════════════════════════════════════════════════════════════════
def plot_session(sess_df: pd.DataFrame, zones: List[TimeBoundedZone],
                 ticker: str, sess_date: date):
    """Interactive Plotly chart with time-bounded zones."""
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x    = sess_df.index,
        open = sess_df["Open"],  high  = sess_df["High"],
        low  = sess_df["Low"],   close = sess_df["Close"],
        increasing_line_color  = "#26a69a",
        decreasing_line_color  = "#ef5350",
        increasing_fillcolor   = "#26a69a",
        decreasing_fillcolor   = "#ef5350",
        name = ticker, showlegend = False,
    ))

    for zone in sorted(zones, key=lambda z: z.intraday_score):
        is_sup = zone.kind == "support"
        is_res = zone.kind == "resistance"
        alpha  = 0.07 + (zone.intraday_score / 100) * 0.22

        if is_res:
            fc, lc = f"rgba(239,83,80,{alpha:.2f})",   "rgba(239,83,80,0.9)"
            sym    = "R"
        elif is_sup:
            fc, lc = f"rgba(38,166,154,{alpha:.2f})",  "rgba(38,166,154,0.9)"
            sym    = "S"
        else:
            fc, lc = f"rgba(255,235,59,{alpha:.2f})",  "rgba(255,235,59,0.9)"
            sym    = "B"

        # Draw zone only between its start and end time + a small tail
        tail = timedelta(minutes=15)
        x0   = zone.zone_start
        x1   = zone.zone_end + tail

        # Filled rectangle using scatter fill
        fig.add_shape(
            type    = "rect",
            x0=x0, x1=x1,
            y0=zone.lower, y1=zone.upper,
            fillcolor = fc,
            line      = dict(color=lc, width=1.2),
            layer     = "below",
        )

        # Mid-line
        fig.add_shape(
            type      = "line",
            x0=x0, x1=x1,
            y0=zone.mid, y1=zone.mid,
            line      = dict(color=lc, width=1.5, dash="dot"),
        )

        # Label at zone end
        label = (f"{sym} {zone.mid:.0f}  "
                 f"[{zone.zone_start.strftime('%H:%M')}–"
                 f"{zone.zone_end.strftime('%H:%M')}]  "
                 f"score={zone.intraday_score}  T={zone.touch_count}")
        fig.add_annotation(
            x         = x1,
            y         = zone.mid,
            text      = label,
            showarrow = False,
            xanchor   = "left",
            yanchor   = "middle",
            font      = dict(size=9, color=lc),
            bgcolor   = "rgba(0,0,0,0.65)",
            borderpad = 2,
        )

    fig.update_layout(
        title = (f"{ticker} — Time-Bounded Intraday S/R Zones  [{sess_date}]  "
                 f"({len(zones)} zones)"),
        xaxis_rangeslider_visible = False,
        template     = "plotly_dark",
        height       = 740,
        showlegend   = False,
        margin       = dict(l=60, r=320, t=55, b=40),
        plot_bgcolor = "#0d1117",
        paper_bgcolor= "#0d1117",
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[15.5, 9.25], pattern="hour"),
    ])
    fig.show()


# ════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ════════════════════════════════════════════════════════════════
def run(ticker:    str  = Config.TICKER,
        days:      int  = Config.BACKTEST_DAYS,
        show_chart: bool = Config.SHOW_CHART,
        chart_days: int  = Config.CHART_DAYS) -> pd.DataFrame:

    print(f"\n{'═'*110}")
    print(f"  TIME-BOUNDED INTRADAY S/R ZONE DETECTOR")
    print(f"  Ticker: {ticker}  │  Look-back: {days} days  │  Timeframe: {Config.INTERVAL}")
    print(f"  Window: {Config.WINDOW_BARS} bars (~{Config.WINDOW_BARS*5} min)  "
          f"│  Slide: {Config.SLIDE_BARS} bars (~{Config.SLIDE_BARS*5} min)")
    print(f"{'═'*110}\n")

    df = fetch_data(ticker, days)
    sessions = sorted(set(df.index.date))
    print(f"  {len(sessions)} trading sessions found\n")

    scanner    = TimeWindowZoneScanner()
    all_records: List[dict] = []

    for sess_date in sessions:
        sess_df = df[df.index.date == sess_date]
        if len(sess_df) < 20:
            continue

        candles   = df_to_candles(sess_df)
        atr_vals  = ATRCalculator.batch(candles)

        zones = scanner.scan_session(candles, sess_date, atr_vals)
        zones = sorted(zones, key=lambda z: z.intraday_score, reverse=True)
        zones = zones[:Config.TOP_N_PER_SESSION]

        close = candles[-1].close
        atr   = next((v for v in reversed(atr_vals) if v is not None), 0.0)

        print_session_zones(sess_date, zones, close, atr)

        # Chart for the last N sessions
        if show_chart and sess_date in sessions[-chart_days:]:
            plot_session(sess_df, zones, ticker, sess_date)

        for z in zones:
            all_records.append({
                "session":       z.session,
                "kind":          z.kind,
                "zone_start":    z.zone_start.strftime("%H:%M"),
                "zone_end":      z.zone_end.strftime("%H:%M"),
                "duration_bars": z.duration_bars,
                "mid":           z.mid,
                "lower":         z.lower,
                "upper":         z.upper,
                "band":          z.band,
                "touches":       z.touch_count,
                "reversals":     z.reversal_count,
                "pivots":        len(z.pivots),
                "slope":         z.slope,
                "intraday_score":z.intraday_score,
                "atr":           round(atr, 2),
            })

    results_df = pd.DataFrame(all_records)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'═'*110}")
    print(f"  SUMMARY")
    print(f"{'═'*110}")
    if len(results_df):
        print(f"  Total zones detected : {len(results_df)}")
        print(f"  Sessions covered     : {results_df['session'].nunique()}")
        print(f"  Avg zones/session    : {len(results_df)/results_df['session'].nunique():.1f}")
        print(f"  Avg intraday score   : {results_df['intraday_score'].mean():.1f}")
        print(f"\n  Score distribution:")
        bins = [(80,100,"High quality"), (60,80,"Medium"), (0,60,"Low")]
        for lo, hi, lbl in bins:
            cnt = ((results_df["intraday_score"] >= lo) &
                   (results_df["intraday_score"] < hi)).sum()
            print(f"    {lbl:<15} ({lo:>3}–{hi}): {cnt:>4} zones")

        print(f"\n  Top zones by score across all sessions:")
        top = results_df.nlargest(10, "intraday_score")[
            ["session","kind","zone_start","zone_end","mid","touches",
             "reversals","intraday_score"]
        ].to_string(index=False)
        print("  " + top.replace("\n", "\n  "))

    print(f"{'═'*110}\n")
    return results_df


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    ticker     = sys.argv[1] if len(sys.argv) > 1 else Config.TICKER
    days       = int(sys.argv[2]) if len(sys.argv) > 2 else Config.BACKTEST_DAYS
    chart_days = int(sys.argv[3]) if len(sys.argv) > 3 else Config.CHART_DAYS

    results = run(
        ticker     = ticker,
        days       = days,
        show_chart = True,
        chart_days = chart_days,
    )

    # Optional: save to CSV
    if len(results):
        out = f"intraday_sr_zones_{ticker.replace('^','')}.csv"
        results.to_csv(out, index=False)
        print(f"  Saved → {out}")