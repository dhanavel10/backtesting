"""
Nifty 5-min Pivot Identifier (Price Action Trading)
====================================================
Rules:
  1. Always ignore the 1st 5-min candle of every trading day
  2. Price zone must move >= ZONE_PTS points between consecutive pivots;
     CONF_CANDLES consecutive candles must not break the previous extreme
  3. Ignore small red candles in a bullish move
     (candle range < SMALL_CANDLE_PCT% of close)
  4. Swing High -> use candle HIGH price
     Swing Low  -> use candle LOW  price

Usage:
  python nifty_pivot.py                      # live Nifty data (yfinance)
  python nifty_pivot.py --demo               # realistic synthetic demo
  python nifty_pivot.py --zone 75 --conf 3  # custom params
"""

import argparse
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# CONFIG
SYMBOL           = "^NSEI"
PERIOD           = "60d"
INTERVAL         = "5m"
ZONE_PTS         = 50
CONF_CANDLES     = 5
SMALL_CANDLE_PCT = 0.10
OUTPUT_PNG       = "/mnt/user-data/outputs/nifty_pivot_chart.png"
OUTPUT_CSV       = "/mnt/user-data/outputs/nifty_pivots.csv"


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_yfinance(symbol, period, interval):
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_via_requests(symbol):
    import requests
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?interval=5m&range=60d&includePrePost=false")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    data = r.json()["chart"]["result"][0]
    ts = data["timestamp"]
    q  = data["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Open": q["open"], "High": q["high"],
         "Low": q["low"], "Close": q["close"], "Volume": q["volume"]},
        index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Kolkata")
    )
    df.index.name = "Datetime"
    return df


def generate_synthetic_nifty(n_days=55, candles_per_day=75,
                               base=22000.0, seed=42):
    """Realistic GBM-based Nifty 5-min OHLCV data for demo/testing."""
    rng  = np.random.default_rng(seed)
    rows = []
    import datetime as dt
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
    start = dt.date(2025, 1, 6)
    price = base
    day_count = 0

    d = start
    while day_count < n_days:
        if d.weekday() < 5:
            open_time = IST.localize(
                dt.datetime.combine(d, dt.time(9, 15))
            )
            for c in range(candles_per_day):
                t   = open_time + dt.timedelta(minutes=5 * c)
                pos = c / candles_per_day
                iv  = 0.0012 + 0.0008 * np.exp(-8 * (pos - 0.5) ** 2)
                ret = rng.normal(0.00004, 0.0001) + iv * rng.standard_normal()
                close = price * (1 + ret)
                hi  = close * (1 + abs(rng.normal(0, 0.0006)))
                lo  = close * (1 - abs(rng.normal(0, 0.0006)))
                op  = price
                hi  = max(hi, op, close)
                lo  = min(lo, op, close)
                rows.append({"Datetime": t, "Open": op, "High": hi,
                             "Low": lo, "Close": close,
                             "Volume": int(rng.integers(50_000, 500_000))})
                price = close
            day_count += 1
        d += dt.timedelta(days=1)

    df = pd.DataFrame(rows).set_index("Datetime")
    # Index is already tz-aware (IST via pytz.localize)
    return df


def load_data(symbol, period, interval, demo=False):
    if demo:
        print("  [DEMO MODE] Generating synthetic Nifty 5-min data ...")
        df = generate_synthetic_nifty()
    else:
        print(f"  Fetching {symbol} | {interval} | {period} via yfinance ...")
        try:
            df = fetch_yfinance(symbol, period, interval)
            if df.empty:
                raise ValueError("Empty result")
            print("  yfinance: OK")
        except Exception as e1:
            print(f"  yfinance failed ({e1}). Trying direct HTTP ...")
            try:
                df = fetch_via_requests(symbol)
                print("  Direct HTTP: OK")
            except Exception as e2:
                sys.exit(
                    f"\nERROR: Could not fetch live data ({e2}).\n"
                    "Run with --demo flag:\n  python nifty_pivot.py --demo\n"
                )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    IST = "Asia/Kolkata"
    if df.index.tz is None:
        # Assume UTC (yfinance default for Indian indices) then convert to IST
        df.index = df.index.tz_localize("UTC").tz_convert(IST)
    else:
        df.index = df.index.tz_convert(IST)
    # Keep tz-aware index so IST is preserved throughout
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["High", "Low", "Close"], inplace=True)
    df.sort_index(inplace=True)

    print(f"  Candles: {len(df):,}  |  "
          f"{df.index[0].strftime('%d %b %Y %H:%M IST')} -> "
          f"{df.index[-1].strftime('%d %b %Y %H:%M IST')}")
    return df


# ── Pivot detection ───────────────────────────────────────────────────────────

def mark_first_candles(df):
    # Normalize to date in IST (index is already tz-aware Asia/Kolkata)
    dates = df.index.normalize()
    return (~pd.Series(dates, index=df.index).duplicated(keep="first")).values


def is_small_red(opens, highs, lows, closes, i, pct):
    if closes[i] >= opens[i]:
        return False
    return ((highs[i] - lows[i]) / closes[i] * 100) < pct


def confirm_swing_high(i, highs, first_of_day, small_red_mask, conf):
    peak = highs[i]
    confirmed, j, n = 0, i + 1, len(highs)
    while j < n and confirmed < conf:
        if first_of_day[j]:
            j += 1; continue
        if small_red_mask[j]:
            j += 1; continue
        if highs[j] <= peak:
            confirmed += 1
        else:
            return False
        j += 1
    return confirmed >= conf


def confirm_swing_low(i, lows, first_of_day, conf):
    trough = lows[i]
    confirmed, j, n = 0, i + 1, len(lows)
    while j < n and confirmed < conf:
        if first_of_day[j]:
            j += 1; continue
        if lows[j] >= trough:
            confirmed += 1
        else:
            return False
        j += 1
    return confirmed >= conf


def identify_pivots(df, zone_pts, conf_candles, small_pct):
    n = len(df)
    opens  = df["Open"].values.astype(float)
    highs  = df["High"].values.astype(float)
    lows   = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    idx    = df.index

    first_of_day  = mark_first_candles(df)
    small_red_mask = np.array([
        is_small_red(opens, highs, lows, closes, i, small_pct)
        for i in range(n)
    ])

    pivots = []

    def last_type(ptype):
        for p in reversed(pivots):
            if p["type"] == ptype:
                return p
        return None

    for i in range(1, n - conf_candles - 1):
        if first_of_day[i]:   # Rule 1
            continue

        # --- Swing High ---
        if confirm_swing_high(i, highs, first_of_day, small_red_mask, conf_candles):
            peak      = highs[i]         # Rule 4: use HIGH
            last_high = last_type("high")
            last_low  = last_type("low")

            if last_low is not None and (peak - last_low["price"]) < zone_pts:
                pass  # zone too small
            elif last_high is not None and abs(peak - last_high["price"]) < zone_pts:
                if peak > last_high["price"]:
                    for k in range(len(pivots) - 1, -1, -1):
                        if pivots[k]["type"] == "high":
                            pivots[k] = {"datetime": idx[i], "type": "high",
                                         "price": peak, "candle_idx": i}
                            break
            else:
                pivots.append({"datetime": idx[i], "type": "high",
                                "price": peak, "candle_idx": i})

        # --- Swing Low ---
        if confirm_swing_low(i, lows, first_of_day, conf_candles):
            trough    = lows[i]          # Rule 4: use LOW
            last_low  = last_type("low")
            last_high = last_type("high")

            if last_high is not None and (last_high["price"] - trough) < zone_pts:
                pass
            elif last_low is not None and abs(trough - last_low["price"]) < zone_pts:
                if trough < last_low["price"]:
                    for k in range(len(pivots) - 1, -1, -1):
                        if pivots[k]["type"] == "low":
                            pivots[k] = {"datetime": idx[i], "type": "low",
                                         "price": trough, "candle_idx": i}
                            break
            else:
                pivots.append({"datetime": idx[i], "type": "low",
                                "price": trough, "candle_idx": i})

    if not pivots:
        return pd.DataFrame(columns=["datetime", "type", "price", "candle_idx"])

    res = pd.DataFrame(pivots)
    res.set_index("datetime", inplace=True)
    return res


# ── Charting ─────────────────────────────────────────────────────────────────

def plot_chart(df, pivots, zone_pts, conf_candles, out_path, demo=False):
    BG     = "#0d1117"
    GRID   = "#1c2030"
    BULL   = "#26a69a"
    BEAR   = "#ef5350"
    MUTED  = "#8b949e"
    TEXT   = "#c9d1d9"
    HIGH_C = "#1D9E75"
    LOW_C  = "#E24B4A"
    ZZ_C   = "#555e70"

    n      = len(df)
    x      = np.arange(n)
    opens  = df["Open"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values

    fig, (ax, ax_vol) = plt.subplots(
        2, 1, figsize=(24, 13),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True, facecolor=BG)

    # OHLC bars
    for i in range(n):
        col = BULL if closes[i] >= opens[i] else BEAR
        ax.plot([i, i], [lows[i], highs[i]], color=col, lw=0.45, alpha=0.7)
        body_lo = min(opens[i], closes[i])
        body_hi = max(opens[i], closes[i])
        ax.bar(i, body_hi - body_lo, bottom=body_lo,
               width=0.6, color=col, alpha=0.85, linewidth=0)

    # Swing pivot markers
    swing_highs = pivots[pivots["type"] == "high"] if not pivots.empty else pd.DataFrame()
    swing_lows  = pivots[pivots["type"] == "low"]  if not pivots.empty else pd.DataFrame()

    if not swing_highs.empty:
        ax.scatter(swing_highs["candle_idx"], swing_highs["price"],
                   marker="v", color=HIGH_C, s=180, zorder=6,
                   edgecolors="white", linewidths=0.8)
        for _, row in swing_highs.iterrows():
            ax.annotate(f"{row['price']:,.0f}",
                        xy=(row["candle_idx"], row["price"]),
                        xytext=(0, 11), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.8,
                        color=HIGH_C, fontweight="bold")

    if not swing_lows.empty:
        ax.scatter(swing_lows["candle_idx"], swing_lows["price"],
                   marker="^", color=LOW_C, s=180, zorder=6,
                   edgecolors="white", linewidths=0.8)
        for _, row in swing_lows.iterrows():
            ax.annotate(f"{row['price']:,.0f}",
                        xy=(row["candle_idx"], row["price"]),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", va="top", fontsize=6.8,
                        color=LOW_C, fontweight="bold")

    # Pivot zigzag
    if not pivots.empty:
        ax.plot(pivots["candle_idx"].values, pivots["price"].values,
                color=ZZ_C, lw=0.9, linestyle="--", alpha=0.55, zorder=3)

    # Volume
    vol_colors = [BULL if closes[i] >= opens[i] else BEAR for i in range(n)]
    ax_vol.bar(x, df["Volume"].values, color=vol_colors, alpha=0.5, width=0.6)
    ax_vol.set_facecolor(BG)
    ax_vol.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _:
            f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))
    ax_vol.tick_params(colors=MUTED, labelsize=7)
    ax_vol.set_ylabel("Volume", fontsize=9, color=MUTED)

    # X-axis
    step   = max(1, n // 28)
    xticks = x[::step]
    xlbls  = [df.index[i].strftime("%d %b\n%H:%M") for i in xticks]
    ax_vol.set_xticks(xticks)
    ax_vol.set_xticklabels(xlbls, fontsize=7, color=MUTED)
    ax_vol.set_xlim(-2, n + 1)

    # Styling
    for a in (ax, ax_vol):
        a.set_facecolor(BG)
        for sp in a.spines.values():
            sp.set_color(GRID)
        a.tick_params(colors=MUTED)
        a.grid(True, color=GRID, lw=0.4, linestyle="--")

    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.set_ylabel("Price (INR)", fontsize=10, color=MUTED)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    days  = df.index.normalize().nunique()
    label = "DEMO -- Synthetic Nifty" if demo else "Nifty 50 (^NSEI)"
    ax.set_title(
        f"{label}  |  5-min candles  |  {days} trading days  "
        f"|  Zone >= {zone_pts} pts  |  {conf_candles}-candle confirm  "
        f"|  Swing Highs: {len(swing_highs)}   Swing Lows: {len(swing_lows)}",
        fontsize=11, color=TEXT, pad=14, loc="left")

    handles = [
        Line2D([0],[0], marker="v", color="w", markerfacecolor=HIGH_C,
               markersize=9, label=f"Swing High  ({len(swing_highs)})"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor=LOW_C,
               markersize=9, label=f"Swing Low   ({len(swing_lows)})"),
        mpatches.Patch(facecolor=BULL, alpha=0.85, label="Bullish candle"),
        mpatches.Patch(facecolor=BEAR, alpha=0.85, label="Bearish candle"),
        Line2D([0],[0], color=ZZ_C, lw=1, linestyle="--", label="Pivot zigzag"),
    ]
    ax.legend(handles=handles, loc="upper left",
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor=TEXT, fontsize=9, framealpha=0.85)

    plt.tight_layout(h_pad=0.0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart saved -> {out_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Nifty 5-min Pivot Identifier")
    p.add_argument("--demo",      action="store_true",
                   help="Use synthetic data (no internet required)")
    p.add_argument("--symbol",    default=SYMBOL)
    p.add_argument("--zone",      type=float, default=ZONE_PTS,
                   help="Min zone move in points (default 50)")
    p.add_argument("--conf",      type=int,   default=CONF_CANDLES,
                   help="Confirmation candles (default 5)")
    p.add_argument("--small-pct", type=float, default=SMALL_CANDLE_PCT,
                   help="Small red candle threshold %% (default 0.10)")
    return p.parse_args()


def main():
    args = parse_args()

    sep = "=" * 55
    print(f"\n{sep}")
    print("  NIFTY 5-MIN PIVOT IDENTIFIER  (Price Action)")
    print(sep)
    print(f"  Symbol           : {args.symbol}")
    print(f"  Zone pts (min)   : {args.zone}")
    print(f"  Conf candles     : {args.conf}")
    print(f"  Small red candle : < {args.small_pct}% of close")
    print(f"  Mode             : {'DEMO (synthetic)' if args.demo else 'LIVE (yfinance)'}")
    print("-" * 55)

    print("\n[1] Loading data ...")
    df = load_data(args.symbol, PERIOD, INTERVAL, demo=args.demo)

    print("\n[2] Identifying pivots ...")
    pivots = identify_pivots(df, args.zone, args.conf, args.small_pct)

    sh = pivots[pivots["type"] == "high"] if not pivots.empty else pd.DataFrame()
    sl = pivots[pivots["type"] == "low"]  if not pivots.empty else pd.DataFrame()
    print(f"  Swing Highs : {len(sh)}")
    print(f"  Swing Lows  : {len(sl)}")
    print(f"  Total pivots: {len(pivots)}")

    print("\n[3] Pivot table (all times IST):")
    print("-" * 50)
    if not pivots.empty:
        disp = pivots.reset_index()[["datetime", "type", "price"]].copy()
        disp["datetime"] = disp["datetime"].dt.strftime("%d %b %Y  %H:%M IST")
        disp["price"]    = disp["price"].map(lambda v: f"{v:,.2f}")
        disp.columns     = ["Datetime (IST)", "Type", "Price (INR)"]
        print(disp.to_string(index=False))
    else:
        print("  No pivots found. Try reducing --zone or --conf.")
    print("-" * 45)

    if not pivots.empty:
        prices    = pivots["price"]
        hi_prices = sh["price"] if not sh.empty else pd.Series(dtype=float)
        lo_prices = sl["price"] if not sl.empty else pd.Series(dtype=float)
        zones     = [abs(pivots["price"].iloc[k] - pivots["price"].iloc[k-1])
                     for k in range(1, len(pivots))]
        print("\n[4] Statistics:")
        if not hi_prices.empty:
            print(f"  Highest swing high : {hi_prices.max():,.2f}")
        if not lo_prices.empty:
            print(f"  Lowest  swing low  : {lo_prices.min():,.2f}")
        if not hi_prices.empty and not lo_prices.empty:
            print(f"  Total pivot range  : {hi_prices.max() - lo_prices.min():,.2f} pts")
        if zones:
            print(f"  Avg zone move      : {np.mean(zones):,.2f} pts")
            print(f"  Max zone move      : {max(zones):,.2f} pts")

    print("\n[5] Generating chart ...")
    plot_chart(df, pivots, args.zone, args.conf, OUTPUT_PNG, demo=args.demo)

    print("\n[6] Saving CSV (times in IST) ...")
    if not pivots.empty:
        csv_df = pivots.reset_index()[["datetime", "type", "price"]].copy()
        csv_df["datetime"] = csv_df["datetime"].dt.strftime("%Y-%m-%d %H:%M IST")
        csv_df.to_csv(OUTPUT_CSV, index=False)
        print(f"  CSV saved -> {OUTPUT_CSV}")

    print(f"\n{sep}\n  Done.\n{sep}\n")
    return df, pivots


if __name__ == "__main__":
    df, pivots = main()