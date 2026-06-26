"""
backtest_validator.py
=====================
Replays historical data through the EXACT same realtime pipeline
(tick_processor → swing_engine → zone_engine → signal_engine)
and produces a detailed report of:

  - Every pivot detected (and when it was confirmed)
  - Every zone created / strengthened
  - Every signal generated
  - Final zone map with strength scores
  - Signal quality breakdown (by type and confidence)

This lets you verify the system's S/R accuracy against price history
BEFORE going live.

Usage:
    python backtest_validator.py --ticker ^NSEI --days 30 --interval 5m

Requirements:
    pip install yfinance pandas plotly
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import numpy as np

# ── Ensure local modules are importable ──────────────────────────
sys.path.insert(0, ".")

from tick_processor import CandleAggregator, CandleBar, SimulatedTickFeed
from swing_engine   import ReversalPivotDetector, PivotEvent
from zone_engine    import ZoneEngine, SRZone, ZoneEvent
from signal_engine  import SignalEngine, Signal

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Data Fetching (same as original script, no changes)
# ─────────────────────────────────────────────────────────────────

def fetch_historical(
    ticker:     str = "^NSEI",
    interval:   str = "5m",
    days:       int = 30,
    chunk_days: int = 7,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    chunks, cursor = [], start_dt

    logger.info(f"Fetching {interval} data for {ticker} ({days} days)...")

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        try:
            chunk = yf.download(
                ticker,
                start    = cursor.strftime("%Y-%m-%d"),
                end      = chunk_end.strftime("%Y-%m-%d"),
                interval = interval,
                auto_adjust = True,
                progress    = False,
            )
            if len(chunk) > 0:
                chunk.columns = [c[0] if isinstance(c, tuple) else c for c in chunk.columns]
                chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Fetch error {cursor.date()}→{chunk_end.date()}: {e}")
        cursor = chunk_end

    if not chunks:
        raise ValueError(f"No data for {ticker}.")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.dropna(inplace=True)

    try:
        df.index = df.index.tz_convert("Asia/Kolkata")
    except TypeError:
        try:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        except Exception:
            pass

    df = df.between_time("09:15", "15:30")
    logger.info(f"✓ {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def df_to_bar_dicts(df: pd.DataFrame, symbol: str, interval_seconds: int = 300) -> list:
    """Convert a yfinance DataFrame to the list-of-dicts format SimulatedTickFeed expects."""
    bars = []
    for ts, row in df.iterrows():
        epoch = ts.timestamp()
        bars.append({
            "open":     float(row["Open"]),
            "high":     float(row["High"]),
            "low":      float(row["Low"]),
            "close":    float(row["Close"]),
            "volume":   int(row.get("Volume", 0)),
            "ts_open":  epoch,
            "interval": interval_seconds,
            "symbol":   symbol,
        })
    return bars


# ─────────────────────────────────────────────────────────────────
# Backtest Runner
# ─────────────────────────────────────────────────────────────────

class BacktestRunner:

    def __init__(
        self,
        symbol:            str   = "NIFTY",
        interval_seconds:  int   = 300,
        rev_pct:           float = 0.30,
        min_swing_pts:     float = 20.0,
        half_band:         float = 15.0,
        cluster_tolerance: float = 15.0,
        min_wick_touches:  int   = 2,
        min_sessions:      int   = 1,
        min_rejections:    int   = 1,
        min_zone_strength: float = 15.0,
    ):
        self.symbol           = symbol
        self.interval_seconds = interval_seconds

        # ── Pipeline components ───────────────────────────────
        self.aggregator = CandleAggregator(interval_seconds=interval_seconds)

        self.pivot_detector = ReversalPivotDetector(
            symbol        = symbol,
            rev_pct       = rev_pct,
            min_swing_pts = min_swing_pts,
        )

        self.zone_engine = ZoneEngine(
            symbol            = symbol,
            half_band         = half_band,
            cluster_tolerance = cluster_tolerance,
            min_wick_touches  = min_wick_touches,
            min_sessions      = min_sessions,
            min_rejections    = min_rejections,
        )

        self.signal_engine = SignalEngine(
            symbol            = symbol,
            zone_engine       = self.zone_engine,
            min_zone_strength = min_zone_strength,
        )

        # ── Wire up the pipeline ──────────────────────────────
        self.aggregator.on_candle_closed(self._on_candle)
        self.pivot_detector.on_pivot(self.zone_engine.on_pivot)
        self.zone_engine.on_zone_event(self._on_zone_event)
        self.signal_engine.on_signal(self._on_signal)

        # ── Logging ───────────────────────────────────────────
        self.all_pivots:  List[PivotEvent] = []
        self.all_zones:   List[ZoneEvent]  = []
        self.all_signals: List[Signal]     = []
        self.candle_log:  List[CandleBar]  = []

    def _on_candle(self, bar: CandleBar):
        self.candle_log.append(bar)
        self.pivot_detector.process_bar(bar)
        self.zone_engine.on_candle(bar)
        self.signal_engine.process_candle(bar)

    def _on_zone_event(self, event: ZoneEvent):
        self.all_zones.append(event)

    def _on_signal(self, signal: Signal):
        self.all_signals.append(signal)

    def run(self, bars: list):
        """Feed all historical bars through the pipeline."""
        sim = SimulatedTickFeed(self.aggregator, speed=0.0)
        sim.replay_bars(bars, symbol=self.symbol)

        # Collect pivots from detector
        self.all_pivots = (
            list(self.pivot_detector.recent_highs) +
            list(self.pivot_detector.recent_lows)
        )

        logger.info(
            f"Backtest complete: "
            f"{len(self.candle_log)} bars  |  "
            f"{len(self.all_pivots)} pivots  |  "
            f"{self.zone_engine.zone_count()['active']} active zones  |  "
            f"{len(self.all_signals)} signals"
        )


# ─────────────────────────────────────────────────────────────────
# Report Printer
# ─────────────────────────────────────────────────────────────────

def print_backtest_report(runner: BacktestRunner, current_price: float):
    SEP  = "═" * 100
    sep2 = "─" * 100

    print(f"\n{SEP}")
    print(f"  BACKTEST VALIDATION REPORT   |   {runner.symbol}")
    print(SEP)

    # ── Pivot summary ─────────────────────────────────────────
    highs = [p for p in runner.all_pivots if p.pivot_type.value == "high"]
    lows  = [p for p in runner.all_pivots if p.pivot_type.value == "low"]
    print(f"\n  PIVOTS DETECTED: {len(runner.all_pivots)} total  "
          f"({len(highs)} highs + {len(lows)} lows)")
    print(f"  {'Bar':>6}  {'Session':<12}  {'Type':<6}  {'Price':>8}  "
          f"{'Rev%':>6}  {'Confirm lag':>12}")
    print("  " + sep2[:85])
    for p in sorted(runner.all_pivots, key=lambda x: x.bar_index)[-30:]:
        lag = p.confirm_bar_index - p.bar_index
        print(f"  {p.bar_index:>6}  {p.session:<12}  {p.label:<6}  "
              f"{p.price:>8.2f}  {p.rev_pct:>6.3f}  {lag:>8} bars")

    # ── Zone summary ─────────────────────────────────────────
    active_zones = runner.zone_engine.get_active_zones(current_price, max_zones=30)
    supports    = sorted([z for z in active_zones if z.price < current_price],
                         key=lambda z: z.price, reverse=True)
    resistances = sorted([z for z in active_zones if z.price >= current_price],
                         key=lambda z: z.price)

    zc = runner.zone_engine.zone_count()
    print(f"\n{SEP}")
    print(f"  S/R ZONES  |  active={zc['active']}  broken={zc['broken']}  expired={zc['expired']}")
    print(f"  CMP: {current_price:.2f}  |  Showing {len(active_zones)} qualified zones")
    print(SEP)
    hdr = (f"  {'Type':<10} {'Price':>8} {'Lower':>8} {'Upper':>8} "
           f"{'Str':>6} {'WkTch':>6} {'Rej':>5} {'InBar':>6} "
           f"{'Sess':>5} {'Conv':>5} {'Dist':>7}")
    print(hdr)
    print("  " + sep2[:95])

    def zrow(z):
        sym = "▲ SUP" if z.price < current_price else "▼ RES"
        dist = z.price - current_price
        return (f"  {sym:<10} {z.price:>8.2f} {z.lower:>8.2f} {z.upper:>8.2f} "
                f"{z.strength:>6.1f} {z.wick_touches:>6} {z.body_rejections:>5} "
                f"{z.inside_bars:>6} {len(z.sessions):>5} {z.convergence:>5} {dist:>+7.0f}")

    print(f"\n  ── RESISTANCE (nearest first) ─────────")
    for z in resistances[:10]: print(zrow(z))
    print(f"\n  {'─'*40}  CMP {current_price:.2f}  {'─'*40}")
    print(f"\n  ── SUPPORT (nearest first) ─────────")
    for z in supports[:10]: print(zrow(z))

    # ── Signal summary ─────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SIGNALS GENERATED: {len(runner.all_signals)}")
    print(SEP)

    from collections import Counter
    type_counts = Counter(s.signal_type for s in runner.all_signals)
    conf_counts = Counter(s.confidence  for s in runner.all_signals)
    print(f"  By type:       " + "  |  ".join(f"{k}: {v}" for k, v in type_counts.items()))
    print(f"  By confidence: " + "  |  ".join(f"{k}: {v}" for k, v in conf_counts.items()))

    print(f"\n  Last 20 signals:")
    print(f"  {'Bar':>6}  {'Session':<12}  {'Type':<18}  {'Action':<10}  "
          f"{'Price':>8}  {'Zone':>8}  {'Str':>6}  {'Conf':<8}")
    print("  " + sep2[:95])
    for s in runner.all_signals[-20:]:
        print(f"  {s.bar_index:>6}  {s.session:<12}  {s.signal_type:<18}  "
              f"{s.action:<10}  {s.price:>8.2f}  {s.zone_price:>8.2f}  "
              f"{s.zone_strength:>6.1f}  {s.confidence:<8}")

    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────
# Plotly Validation Chart
# ─────────────────────────────────────────────────────────────────

def plot_backtest(runner: BacktestRunner, df: pd.DataFrame, current_price: float):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("plotly not installed — skipping chart")
        return

    # Use last 10 sessions of candles
    candles = runner.candle_log
    if not candles:
        return

    last_sessions = sorted(set(datetime.fromtimestamp(c.ts_open, tz=timezone.utc).date()
                               for c in candles))[-10:]
    session_set   = set(last_sessions)
    plot_candles  = [c for c in candles
                     if datetime.fromtimestamp(c.ts_open, tz=timezone.utc).date() in session_set]

    xs      = [datetime.fromtimestamp(c.ts_open, tz=timezone.utc) for c in plot_candles]
    opens   = [c.open   for c in plot_candles]
    highs   = [c.high   for c in plot_candles]
    lows    = [c.low    for c in plot_candles]
    closes  = [c.close  for c in plot_candles]
    volumes = [c.volume for c in plot_candles]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.80, 0.20], vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=xs, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
        name=runner.symbol,
    ), row=1, col=1)

    vcols = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(closes, opens)]
    fig.add_trace(go.Bar(x=xs, y=volumes, marker_color=vcols, opacity=0.4,
                         name="Volume"), row=2, col=1)

    # Draw active zones
    active_zones = runner.zone_engine.get_active_zones(current_price, max_zones=25)
    for zone in active_zones:
        is_sup = zone.price < current_price
        alpha  = 0.06 + (zone.strength / 100) * 0.18
        la     = 0.4  + (zone.strength / 100) * 0.6
        fc = f"rgba(38,166,154,{alpha:.2f})" if is_sup else f"rgba(239,83,80,{alpha:.2f})"
        lc = f"rgba(38,166,154,{la:.2f})"   if is_sup else f"rgba(239,83,80,{la:.2f})"

        fig.add_hrect(y0=zone.lower, y1=zone.upper,
                      fillcolor=fc, line_width=0, row=1, col=1)
        fig.add_hline(y=zone.price, line_color=lc,
                      line_width=1.5, line_dash="solid", row=1, col=1)

        sym   = "S" if is_sup else "R"
        label = (f"{sym} {zone.price:.0f} [{zone.lower:.0f}–{zone.upper:.0f}]  "
                 f"str={zone.strength:.0f}  T={zone.wick_touches} Rej={zone.body_rejections}")
        fig.add_annotation(
            x=xs[-1], y=zone.price, text=label,
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=9, color=lc),
            bgcolor="rgba(0,0,0,0.55)", borderpad=2, row=1, col=1,
        )

    # Mark pivot points on chart
    plot_bar_set = set(id(c) for c in plot_candles)
    for pivot in runner.all_pivots:
        pivot_dt = datetime.fromtimestamp(pivot.ts, tz=timezone.utc)
        if pivot_dt.date() not in session_set:
            continue
        color = "#26a69a" if pivot.pivot_type.value == "low" else "#ef5350"
        sym   = "▲" if pivot.pivot_type.value == "low" else "▼"
        fig.add_annotation(
            x=pivot_dt, y=pivot.price,
            text=f"{sym}{pivot.price:.0f}",
            showarrow=False, yanchor="middle",
            font=dict(size=8, color=color),
            row=1, col=1,
        )

    # Mark signals
    signal_markers = {
        "REJECTION_LONG":  ("▲", "#00e5ff"),
        "REJECTION_SHORT": ("▼", "#ff6b6b"),
        "BREAKOUT_LONG":   ("⬆", "#69ff47"),
        "BREAKOUT_SHORT":  ("⬇", "#ff4747"),
        "ZONE_APPROACH":   ("◆", "#ffd700"),
    }
    for sig in runner.all_signals[-100:]:
        sig_dt = datetime.fromtimestamp(sig.ts, tz=timezone.utc)
        if sig_dt.date() not in session_set:
            continue
        marker, color = signal_markers.get(sig.signal_type, ("●", "#ffffff"))
        fig.add_annotation(
            x=sig_dt, y=sig.price,
            text=f"{marker}",
            showarrow=False,
            font=dict(size=14, color=color),
            row=1, col=1,
        )

    # CMP line
    fig.add_hline(y=current_price,
                  line_color="rgba(255,235,59,0.85)",
                  line_width=1.5, line_dash="dash", row=1, col=1)
    fig.add_annotation(
        x=xs[-1], y=current_price, text=f"CMP {current_price:.2f}",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=10, color="rgba(255,235,59,0.9)"), row=1, col=1,
    )

    fig.update_layout(
        title=(f"{runner.symbol} — Backtest Validation  "
               f"(5m · causal S/R · {len(active_zones)} zones · "
               f"{len(runner.all_signals)} signals)"),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=860,
        showlegend=False,
        margin=dict(l=60, r=280, t=55, b=40),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
    )
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[15.5, 9.25], pattern="hour"),
    ])
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", row=1, col=1)

    fig.show()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="S/R Backtest Validator")
    parser.add_argument("--ticker",      default="^NSEI",  help="Yahoo Finance ticker")
    parser.add_argument("--symbol",      default="NIFTY",  help="Display symbol name")
    parser.add_argument("--days",        default=30,   type=int)
    parser.add_argument("--interval",    default="5m")
    parser.add_argument("--rev-pct",     default=0.30, type=float, help="Reversal %% to confirm pivot")
    parser.add_argument("--min-swing",   default=20.0, type=float, help="Min absolute swing (pts)")
    parser.add_argument("--half-band",   default=15.0, type=float, help="Zone half-band (pts)")
    parser.add_argument("--cluster-tol", default=15.0, type=float, help="Cluster tolerance (pts)")
    parser.add_argument("--min-touches", default=2,    type=int)
    parser.add_argument("--min-sessions",default=1,    type=int)
    parser.add_argument("--min-rejections", default=1, type=int)
    parser.add_argument("--no-chart",    action="store_true")
    args = parser.parse_args()

    interval_map = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800}
    interval_sec = interval_map.get(args.interval, 300)

    # ── Fetch data ────────────────────────────────────────────
    df = fetch_historical(
        ticker   = args.ticker,
        interval = args.interval,
        days     = args.days,
    )

    bars = df_to_bar_dicts(df, symbol=args.symbol, interval_seconds=interval_sec)
    current_price = float(df["Close"].iloc[-1])

    # ── Run backtest ──────────────────────────────────────────
    runner = BacktestRunner(
        symbol            = args.symbol,
        interval_seconds  = interval_sec,
        rev_pct           = args.rev_pct,
        min_swing_pts     = args.min_swing,
        half_band         = args.half_band,
        cluster_tolerance = args.cluster_tol,
        min_wick_touches  = args.min_touches,
        min_sessions      = args.min_sessions,
        min_rejections    = args.min_rejections,
    )
    runner.run(bars)

    # ── Report ────────────────────────────────────────────────
    print_backtest_report(runner, current_price)

    if not args.no_chart:
        plot_backtest(runner, df, current_price)


if __name__ == "__main__":
    main()
