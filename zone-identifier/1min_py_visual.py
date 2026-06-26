"""
pivot_plot.py — Candle-based pivot detection on 1-day NIFTY data
=================================================================
1-MINUTE CHART VERSION

Processes fully-formed 1-min OHLC candles — identical logic to live_sr.py.
Backtest and live feed the same data format, so results are identical.

No look-ahead: a pivot is confirmed only when the candle that crosses the
reversal threshold closes.  Both H and L of each candle are known at that
point — no intra-bar path ambiguity.

WHY THE PARAMETERS CHANGE FROM THE 5m VERSION
-----------------------------------------------
1. REVERSAL_THRESHOLD (the ZigZag's swing-size filter) must shrink.
   30 pts was sized for 5-min candles, where each bar already contains a
   meaningful chunk of intraday range. On 1-min bars, typical swings
   between turning points are much smaller — a 30-pt threshold would
   barely fire and you'd get almost no pivots. 12 pts is a reasonable
   1m starting point (roughly the same proportion of "noise filtered
   out" as 30 pts was on 5m); push it up/down depending on how chatty
   you want the ZigZag to be.

   As of this version the threshold can also be set ADAPTIVELY from
   ATR instead of a hardcoded point value — see THRESHOLD_MODE below.
   A fixed point threshold is fine for a single week of fairly stable
   volatility, but it silently drifts out of proportion if you run
   this over months, across instruments, or across a volatility
   regime change. ATR mode keeps "what counts as a real swing" stable
   in relative terms instead of absolute points.

2. Candle width on the chart (`bar_w`) must shrink to match — 1-min
   candles are ~5x narrower in time than 5-min candles, so drawing them
   at the old 2.2-minute width would make adjacent candles overlap.

3. The x-axis tick locator is set denser (every 5 min instead of every
   15) since a 1-day 1m chart has ~5x more candles to spread the same
   horizontal space across — without this, the time labels become too
   sparse to read the chart precisely.

Everything else (download logic, pivot-confirmation logic, plotting
style, legend) is unchanged — only the interval and the
interval-dependent point/width/tick parameters move.

LAG IS NOW MEASURED IN TRADING BARS, NOT WALL-CLOCK MINUTES
-------------------------------------------------------------
Previously, lag was `(confirmed_at - extremum_time).total_seconds() / 60`.
That's correct within a session, but a pivot formed near the close and
confirmed on the next session's open candle would show a lag of ~1000+
minutes (the overnight/weekend gap), even though only 1-2 *trading* bars
actually elapsed. That's a metric bug, not a detection bug — the pivot
itself was still found and confirmed correctly.

Lag is now computed as the number of candles between the extremum bar
and the confirming bar (using each bar's position in the loaded data),
so it always reflects actual trading time elapsed. Pivots whose
confirmation crosses a session boundary are additionally tagged
"[session-edge]" in the printed output so you can see at a glance which
pivots needed the next day's open to confirm.

CSV INPUT
---------
You can now pass your own 1-min OHLC CSV file on the command line instead
of pulling live data from Yahoo Finance:

    python pivot_plot_1m.py path/to/your_data.csv

If no file is given, it falls back to the original yfinance download.

Expected CSV columns (case-insensitive):
    - A datetime column named one of: Datetime, Date, Timestamp, Time
    - Open, High, Low, Close   (Volume is optional)

Timestamp handling:
    - Timezone-naive timestamps are assumed to already be IST (the normal
      case for an NSE broker/data-feed export).
    - Timezone-aware timestamps (e.g. UTC) are converted to IST.

THRESHOLD_MODE
---------------
Controlled via env vars (see Config section below):
    THRESHOLD_MODE=fixed   -> use REVERSAL_THRESHOLD as a flat point value
                               (original behaviour, default)
    THRESHOLD_MODE=atr     -> derive the threshold from the loaded data's
                               own ATR, so it scales with the instrument's
                               actual recent volatility instead of a
                               hand-picked point value. Useful when this
                               script is pointed at different symbols,
                               different weeks, or different volatility
                               regimes without retuning REVERSAL_THR by hand.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

from realtime_sr import ZigZagPivotDetector

load_dotenv()   # load .env before any os.getenv() call below

# ── Config ──────────────────────────────────────────────────────────────────
SYMBOL             = "^NSEI"
PERIOD             = "7d"
INTERVAL           = "1m"

# "fixed" = use REVERSAL_THRESHOLD as a flat point value (old behaviour).
# "atr"   = derive the threshold from the loaded data's own ATR so it
#           scales with actual recent volatility instead of a fixed
#           point count. See THRESHOLD_MODE docstring section above.
THRESHOLD_MODE     = os.getenv("THRESHOLD_MODE", "fixed").strip().lower()

REVERSAL_THRESHOLD = float(os.getenv("REVERSAL_THR", "6.0"))
# Lowered from 12.0 → 6.0 so smaller/minor pullbacks on the 1m chart now
# qualify as confirmed pivots instead of being filtered out as noise.
# Push this even lower (e.g. 4.0) to catch nearly every wiggle, or raise
# it back up if 6.0 still produces too many minor/insignificant pivots.
# Only used when THRESHOLD_MODE="fixed".

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULT   = float(os.getenv("ATR_MULT", "0.5"))
# Only used when THRESHOLD_MODE="atr". The resolved threshold is
# median(ATR(ATR_PERIOD)) * ATR_MULT over the loaded data, so a 0.5x
# multiplier means "a swing must be at least half of a typical recent
# 1-min true range to count as a pivot." Raise ATR_MULT for fewer/larger
# pivots, lower it for more/smaller ones.

# Tag pivots whose confirming bar falls on a different calendar day than
# their extremum bar (i.e. confirmation needed the next session to fire).
SESSION_EDGE_TAG = "[session-edge]"

# Candidate column names we'll recognize when reading a CSV
_DT_COL_CANDIDATES = ["Datetime", "datetime", "DateTime", "Date", "date",
                       "Timestamp", "timestamp", "Time", "time"]


# ── Data ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Candle-based ZigZag pivot detector (1-min NIFTY)."
    )
    parser.add_argument(
        "csv_file", nargs="?", default=None,
        help="Path to a 1-min OHLC CSV file. If omitted, live data is "
             "downloaded from Yahoo Finance instead.",
    )
    return parser.parse_args()


def load_csv_data(path: str) -> pd.DataFrame:
    """
    Load 1-min OHLC candles from a CSV file and normalize it into the
    same shape download_data() produces: a DatetimeIndex in IST, columns
    Open/High/Low/Close(/Volume), restricted to the 09:15-15:30 session.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # Locate the datetime column
    dt_col = next((c for c in _DT_COL_CANDIDATES if c in df.columns), None)
    if dt_col is None:
        raise ValueError(
            f"Could not find a datetime column in '{path}'.\n"
            f"Expected one of: {_DT_COL_CANDIDATES}\n"
            f"Found columns: {list(df.columns)}"
        )

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col)
    df.index.name = "Datetime"

    # Normalize OHLCV column names regardless of case in the source file
    rename_map = {}
    for col in df.columns:
        norm = col.strip().lower()
        if norm == "open":
            rename_map[col] = "Open"
        elif norm == "high":
            rename_map[col] = "High"
        elif norm == "low":
            rename_map[col] = "Low"
        elif norm == "close":
            rename_map[col] = "Close"
        elif norm == "volume":
            rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV '{path}' is missing required column(s): {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Timezone: assume naive timestamps are already IST; convert if tz-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")

    df = df.sort_index()
    df = df.between_time("09:15", "15:30").dropna(subset=required)
    return df


def download_data():
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL,
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    try:
        df.index = df.index.tz_convert("Asia/Kolkata")
    except TypeError:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    df = df.between_time("09:15", "15:30").dropna()
    return df


# ── Threshold resolution ─────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Classic Wilder true-range / ATR, computed on the loaded bars."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(1, period // 2)).mean()


def resolve_threshold(df: pd.DataFrame) -> float:
    """
    Returns the reversal-threshold point value to actually feed the
    ZigZag detector, based on THRESHOLD_MODE.

    "fixed" -> REVERSAL_THRESHOLD as-is.
    "atr"   -> median ATR(ATR_PERIOD) over the loaded data * ATR_MULT.
               Falls back to REVERSAL_THRESHOLD if ATR can't be computed
               (e.g. too few bars).
    """
    if THRESHOLD_MODE != "atr":
        return REVERSAL_THRESHOLD

    atr = compute_atr(df, ATR_PERIOD).dropna()
    if atr.empty:
        print(f"  [threshold] ATR mode requested but not enough bars to "
              f"compute ATR({ATR_PERIOD}) — falling back to fixed "
              f"REVERSAL_THRESHOLD={REVERSAL_THRESHOLD}")
        return REVERSAL_THRESHOLD

    threshold = float(atr.median() * ATR_MULT)
    print(f"  [threshold] ATR mode: median ATR({ATR_PERIOD}) = "
          f"{atr.median():.2f} pt  ×  ATR_MULT={ATR_MULT}  "
          f"->  resolved threshold = {threshold:.2f} pt")
    return threshold


# ── Pivot detection ─────────────────────────────────────────────────────────

def detect_pivots(df: pd.DataFrame, threshold: float):
    """
    Iterate closed 1-min candles → ZigZagPivotDetector.on_candle().
    Identical logic to live_sr.py — backtest results match live exactly.

    Lag is recorded both ways:
      - lag_minutes : wall-clock minutes between extremum and confirmation
                      (kept for reference; can be huge across a session gap)
      - lag_bars    : number of *trading* candles between extremum and
                      confirmation, using each bar's position in `df`.
                      This is the one that's actually meaningful for
                      latency analysis, since it can't be inflated by
                      overnight/weekend gaps.
    """
    zz     = ZigZagPivotDetector(reversal_threshold=threshold)
    pivots = []

    # Position of every bar in the loaded data, used to convert
    # (extremum_time, confirmed_time) into a trading-bar lag instead of
    # a wall-clock one.
    bar_pos = {ts: i for i, ts in enumerate(df.index)}

    for ts, row in df.iterrows():
        bar_time   = ts.to_pydatetime()
        new_pivots = zz.on_candle(
            high     = float(row["High"]),
            low      = float(row["Low"]),
            end_time = bar_time,
        )
        for pivot in new_pivots:
            extremum_ts  = pd.Timestamp(pivot.time)
            confirmed_ts = pd.Timestamp(bar_time)

            lag_bars = None
            if extremum_ts in bar_pos and confirmed_ts in bar_pos:
                lag_bars = bar_pos[confirmed_ts] - bar_pos[extremum_ts]

            session_edge = extremum_ts.date() != confirmed_ts.date()

            pivots.append({
                "price":        pivot.price,
                "time":         pivot.time,      # candle where extremum last peaked
                "confirmed_at": bar_time,        # candle that confirmed the reversal
                "type":         pivot.type,
                "swing":        pivot.swing_size,
                "lag_minutes":  (confirmed_ts - extremum_ts).total_seconds() / 60,
                "lag_bars":     lag_bars,
                "session_edge": session_edge,
            })

    return pivots


def summarize_pivots(pivots, df):
    """Print aggregate swing/lag stats — overall and per trading day."""
    if not pivots:
        print("  No pivots to summarize.")
        return

    pdf = pd.DataFrame(pivots)
    pdf["day"] = pdf["time"].apply(lambda t: pd.Timestamp(t).date())

    print("\n  ── Summary ───────────────────────────────────────────────")
    print(f"  Total pivots        : {len(pdf)}  "
          f"({(pdf['type'] == 'high').sum()} highs, "
          f"{(pdf['type'] == 'low').sum()} lows)")
    print(f"  Swing  (pt)          mean={pdf['swing'].mean():.1f}  "
          f"median={pdf['swing'].median():.1f}  max={pdf['swing'].max():.1f}")

    bar_lags = pdf["lag_bars"].dropna()
    if not bar_lags.empty:
        print(f"  Lag (trading bars)   mean={bar_lags.mean():.1f}  "
              f"median={bar_lags.median():.1f}  max={bar_lags.max():.0f}")

    edge_count = int(pdf["session_edge"].sum())
    if edge_count:
        print(f"  Session-edge pivots  : {edge_count} "
              f"(confirmed on the next session's bars — see "
              f"'{SESSION_EDGE_TAG}' tags above)")

    print("\n  Per-day breakdown:")
    by_day = pdf.groupby("day").agg(
        n=("swing", "count"),
        avg_swing=("swing", "mean"),
        avg_lag_bars=("lag_bars", "mean"),
    )
    for day, row in by_day.iterrows():
        avg_lag = f"{row['avg_lag_bars']:.1f}" if pd.notna(row["avg_lag_bars"]) else "n/a"
        print(f"    {day}   n={int(row['n']):>3}   "
              f"avg_swing={row['avg_swing']:.1f} pt   avg_lag={avg_lag} bars")
    print("  ──────────────────────────────────────────────────────────")


# ── Plot ────────────────────────────────────────────────────────────────────

def plot(df, pivots, threshold):
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Convert index to plain datetime for matplotlib
    times = [t.to_pydatetime() for t in df.index]
    t_num = mdates.date2num(times)   # float day values — reliable for bar width

    opens  = df["Open"].values.flatten().astype(float)
    highs  = df["High"].values.flatten().astype(float)
    lows   = df["Low"].values.flatten().astype(float)
    closes = df["Close"].values.flatten().astype(float)

    # ── Candlestick bars ────────────────────────────────────────────────
    bar_w = 0.5 / (24 * 60)   # 0.5 minutes in matplotlib day-fraction units
    # (narrower than the 5m version's 2.2 min, since 1m bars sit ~5x closer
    # together in time and would otherwise overlap)
    for i in range(len(t_num)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        color = "#26a69a" if c >= o else "#ef5350"
        # wick
        ax.plot([t_num[i], t_num[i]], [l, h],
                color=color, linewidth=0.9, zorder=1)
        # body
        body_bot = min(o, c)
        body_h   = max(abs(c - o), 0.5)
        ax.bar(t_num[i], body_h, bottom=body_bot,
               width=bar_w, color=color, alpha=0.85, zorder=2, align="center")

    ax.xaxis_date()

    # ── ZigZag connector line ───────────────────────────────────────────
    if len(pivots) >= 2:
        zz_t = mdates.date2num([p["time"] for p in pivots])
        zz_p = [p["price"] for p in pivots]
        ax.plot(zz_t, zz_p,
                color="#ffd700", linewidth=1.5, linestyle="--",
                alpha=0.65, zorder=3)

    # ── Pivot markers + labels ──────────────────────────────────────────
    for p in pivots:
        t_n = mdates.date2num(p["time"])
        if p["type"] == "high":
            ax.scatter(t_n, p["price"],
                       marker="v", color="#ff4757", s=100, zorder=6, linewidths=0)
            ax.annotate(
                f'{p["price"]:.0f}',
                xy=(t_n, p["price"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, color="#ff6b81", fontweight="bold",
            )
        else:
            ax.scatter(t_n, p["price"],
                       marker="^", color="#2ed573", s=100, zorder=6, linewidths=0)
            ax.annotate(
                f'{p["price"]:.0f}',
                xy=(t_n, p["price"]),
                xytext=(0, -14), textcoords="offset points",
                ha="center", va="top",
                fontsize=7.5, color="#7bed9f", fontweight="bold",
            )

    # ── Axes formatting ─────────────────────────────────────────────────
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    # (denser ticks than the 5m version's 15-min spacing — a 1-day 1m chart
    # has ~5x more candles to spread the same horizontal space across, so
    # the labels need to be closer together to stay readable/precise)
    plt.xticks(rotation=45, color="#8b949e", fontsize=9)
    plt.yticks(color="#8b949e", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.grid(True, alpha=0.12, color="#30363d", linewidth=0.7)
    ax.tick_params(colors="#8b949e")

    date_str = times[0].strftime("%Y-%m-%d") if times else ""
    mode_str = "ATR-adaptive" if THRESHOLD_MODE == "atr" else "fixed"
    ax.set_title(
        f"NIFTY 50  ·  {date_str}    "
        f"Reversal = {threshold:.1f} pt ({mode_str})   ·   {len(pivots)} pivots confirmed"
        f"   ·   Candle-based  ·   ✓ Backtest = Live",
        color="#e6edf3", fontsize=12, pad=14, loc="left",
    )
    ax.set_xlabel("Time (IST)", color="#8b949e", labelpad=10)
    ax.set_ylabel("Price  (₹)", color="#8b949e", labelpad=10)

    legend_els = [
        Patch(facecolor="#26a69a", alpha=0.85, label="Bullish candle"),
        Patch(facecolor="#ef5350", alpha=0.85, label="Bearish candle"),
        Line2D([0], [0], color="#ffd700", linewidth=1.5,
               linestyle="--", label="ZigZag"),
        Line2D([0], [0], marker="v", color="w",
               markerfacecolor="#ff4757", markersize=10,
               linestyle="None", label="Pivot High"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="#2ed573", markersize=10,
               linestyle="None", label="Pivot Low"),
    ]
    ax.legend(handles=legend_els, loc="upper left",
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9", fontsize=9, framealpha=0.9)

    plt.tight_layout(pad=1.5)
    plt.show()


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.csv_file:
        print(f"\n  Loading 1-min data from '{args.csv_file}'...")
        df = load_csv_data(args.csv_file)
    else:
        print(f"\n  Downloading {SYMBOL}  ({PERIOD}, {INTERVAL})...")
        df = download_data()

    if df.empty:
        print("  No data returned.")
        return
    print(f"  {len(df)} bars  "
          f"({df.index[0].strftime('%H:%M')} → {df.index[-1].strftime('%H:%M')})")

    threshold = resolve_threshold(df)

    print(f"\n  Detecting pivots  (reversal={threshold:.2f} pt, candle-based)...")
    pivots = detect_pivots(df, threshold)
    print(f"  {len(pivots)} confirmed pivots:\n")

    for p in pivots:
        arrow = "▼ HIGH" if p["type"] == "high" else "▲ LOW "
        lag_bars_str = f"{p['lag_bars']}b" if p["lag_bars"] is not None else "n/a"
        edge_tag = f"  {SESSION_EDGE_TAG}" if p["session_edge"] else ""
        print(f"    {arrow}  {p['price']:>8.2f}"
              f"  extremum @ {p['time'].strftime('%Y-%m-%d %H:%M:%S')}"
              f"  confirmed @ {p['confirmed_at'].strftime('%Y-%m-%d %H:%M:%S')}"
              f"  (lag {lag_bars_str}, {p['lag_minutes']:.1f}m wall-clock)"
              f"  swing={p['swing']:.1f} pt{edge_tag}")

    summarize_pivots(pivots, df)

    plot(df, pivots, threshold)


if __name__ == "__main__":
    main()