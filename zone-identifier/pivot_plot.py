"""
pivot_plot.py — Candle-based pivot detection on 1-day NIFTY data
=================================================================
Processes fully-formed 5-min OHLC candles — identical logic to live_sr.py.
Backtest and live feed the same data format, so results are identical.

No look-ahead: a pivot is confirmed only when the candle that crosses the
reversal threshold closes.  Both H and L of each candle are known at that
point — no intra-bar path ambiguity.
"""

import os
import numpy as np
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
PERIOD             = "1d"
INTERVAL           = "5m"
REVERSAL_THRESHOLD = float(os.getenv("REVERSAL_THR", "30.0"))


# ── Data ────────────────────────────────────────────────────────────────────

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


# ── Pivot detection ─────────────────────────────────────────────────────────

def detect_pivots(df):
    """
    Iterate closed 5-min candles → ZigZagPivotDetector.on_candle().
    Identical logic to live_sr.py — backtest results match live exactly.
    """
    zz     = ZigZagPivotDetector(reversal_threshold=REVERSAL_THRESHOLD)
    pivots = []

    for ts, row in df.iterrows():
        bar_time   = ts.to_pydatetime()
        new_pivots = zz.on_candle(
            high     = float(row["High"]),
            low      = float(row["Low"]),
            end_time = bar_time,
        )
        for pivot in new_pivots:
            pivots.append({
                "price":        pivot.price,
                "time":         pivot.time,      # candle where extremum last peaked
                "confirmed_at": bar_time,        # candle that confirmed the reversal
                "type":         pivot.type,
                "swing":        pivot.swing_size,
            })

    return pivots


# ── Plot ────────────────────────────────────────────────────────────────────

def plot(df, pivots):
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
    bar_w = 2.2 / (24 * 60)   # 2.2 minutes in matplotlib day-fraction units
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
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    plt.xticks(rotation=45, color="#8b949e", fontsize=9)
    plt.yticks(color="#8b949e", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.grid(True, alpha=0.12, color="#30363d", linewidth=0.7)
    ax.tick_params(colors="#8b949e")

    date_str = times[0].strftime("%Y-%m-%d") if times else ""
    ax.set_title(
        f"NIFTY 50  ·  {date_str}    "
        f"Reversal = {REVERSAL_THRESHOLD} pt   ·   {len(pivots)} pivots confirmed"
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
    print(f"\n  Downloading {SYMBOL}  ({PERIOD}, {INTERVAL})...")
    df = download_data()
    if df.empty:
        print("  No data returned.")
        return
    print(f"  {len(df)} bars  "
          f"({df.index[0].strftime('%H:%M')} → {df.index[-1].strftime('%H:%M')})")

    print(f"\n  Detecting pivots  (reversal={REVERSAL_THRESHOLD} pt, candle-based)...")
    pivots = detect_pivots(df)
    print(f"  {len(pivots)} confirmed pivots:\n")

    for p in pivots:
        arrow = "▼ HIGH" if p["type"] == "high" else "▲ LOW "
        lag = (p["confirmed_at"] - p["time"]).total_seconds() / 60
        print(f"    {arrow}  {p['price']:>8.2f}"
              f"  extremum @ {p['time'].strftime('%H:%M:%S')}"
              f"  confirmed @ {p['confirmed_at'].strftime('%H:%M:%S')}"
              f"  (lag {lag:.1f}m)  swing={p['swing']:.1f} pt")

    plot(df, pivots)


if __name__ == "__main__":
    main()
