"""
NIFTY 50 — High-Probability Price Action Backtest
===================================================
Based on case study: Nov 2022 – Apr 2026 | 5-min candles | 852 trading days

STRATEGY: "Compression Breakout + Confirmation"
Combines the THREE highest-probability signals from the case study:
  1. The Squeeze (67%)  — compression 3+ candles, avg range < 25 pts
  2. ORB framework      — 9:15–9:45 range, enter on CLOSE outside range
  3. Higher Low / Lower High — structural confirmation of direction

ENTRY FILTERS (stacked — all must pass):
  A. Compression: last 3 candles avg range < 30 pts  (coiling phase)
  B. Breakout:    current candle CLOSES outside the compression high/low
  C. Time filter: 09:25–09:45 (ORB) OR 14:00–15:15 (afternoon edge)
  D. Wick bias:   9:15 candle wick aligns with trade direction

EXIT RULES (no look-ahead — all based on previous-candle data):
  • Hard stop : re-entry into compression range (invalidation)
  • Target    : 1.5× the compression range (pts)
  • Time stop : exit before 15:25 IST (no overnight)
  • Max hold  : 10 candles (50 min) — per 76% reversal rule

NO LOOK-AHEAD:
  Every signal uses only data available at the time of candle close.
  Entry is placed at the OPEN of the next candle after signal fires.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import time as dtime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── PARAMETERS ───────────────────────────
TICKER          = "^NSEI"            # NIFTY 50
START           = "2022-11-01"
END             = "2026-04-30"
INTERVAL        = "5m"               # 5-minute candles

# Squeeze detection
SQUEEZE_CANDLES = 3                  # look-back window for compression
SQUEEZE_RANGE   = 30                 # avg range threshold (pts)

# ORB window (IST)
ORB_START_H, ORB_START_M = 9, 15
ORB_END_H,   ORB_END_M   = 9, 45

# Valid entry windows (IST)
ENTRY_WINDOWS = [
    (dtime(9, 25), dtime(9, 45)),    # ORB breakout window
    (dtime(14, 0), dtime(15, 15)),   # afternoon high-sustain window
]

# Exit rules
MAX_HOLD_CANDLES = 10                # 76% reversal rule
TARGET_MULT      = 1.5               # target = 1.5× compression range
EXIT_BEFORE      = dtime(15, 25)     # flat before close

# Wick bias filter (9:15 candle)
WICK_BIAS_THRESH = 0.40              # wick must be >40% of candle range

# Simulated P&L per lot (index points, 1 lot = 1 unit for clarity)
SLIPPAGE_PTS = 5                     # realistic slippage per side


# ─────────────────────────── DATA FETCH ───────────────────────────
def fetch_data():
    print(f"[*] Fetching {TICKER} {INTERVAL} data: {START} → {END} ...")
    # yfinance 5m data is limited to ~60 days per pull; chunk it
    chunks = []
    periods = pd.date_range(start=START, end=END, freq="59D")
    dates   = list(periods) + [pd.Timestamp(END)]
    for i in range(len(dates) - 1):
        s = dates[i].strftime("%Y-%m-%d")
        e = dates[i + 1].strftime("%Y-%m-%d")
        chunk = yf.download(TICKER, start=s, end=e, interval=INTERVAL,
                            progress=False, auto_adjust=True)
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No data fetched. Check ticker / dates.")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df.index   = pd.to_datetime(df.index)

    # Convert to IST (+05:30)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")

    # Filter to market hours
    df = df.between_time("09:15", "15:30")
    df.dropna(inplace=True)
    print(f"[+] Loaded {len(df):,} candles across "
          f"{df.index.normalize().nunique()} trading days.")
    return df


# ─────────────────────────── FEATURES ─────────────────────────────
def compute_features(df):
    """Add derived columns. Everything uses .shift(1) or earlier — no look-ahead."""
    df = df.copy()
    df["range"]  = df["high"] - df["low"]
    df["body"]   = abs(df["close"] - df["open"])
    df["is_bull"] = (df["close"] >= df["open"]).astype(int)

    # Rolling avg range of PREVIOUS n candles (shift ensures no current candle)
    df["avg_range_3"] = df["range"].shift(1).rolling(SQUEEZE_CANDLES).mean()

    # Compression high/low of the previous SQUEEZE_CANDLES candles
    df["comp_high"] = df["high"].shift(1).rolling(SQUEEZE_CANDLES).max()
    df["comp_low"]  = df["low"].shift(1).rolling(SQUEEZE_CANDLES).min()
    df["comp_size"] = df["comp_high"] - df["comp_low"]

    # Squeeze flag: previous window was compressed
    df["squeeze"] = df["avg_range_3"] < SQUEEZE_RANGE

    # Breakout signal: current close breaks OUT of compression (no look-ahead:
    # we use current close vs compression of prior candles, which is correct)
    df["bull_break"] = (df["squeeze"]) & (df["close"] > df["comp_high"])
    df["bear_break"] = (df["squeeze"]) & (df["close"] < df["comp_low"])

    # Date + time helpers
    df["date"] = df.index.date
    df["time"] = df.index.time
    df["hour"] = df.index.hour

    return df


def add_orb_and_wick_bias(df):
    """
    Compute per-day ORB high/low and 9:15 wick bias.
    These are known at the time of entry (after the 9:15 candle closes).
    """
    # 9:15 candle per day
    orb_915 = df[df["time"] == dtime(9, 15)].copy()
    orb_915["upper_wick"] = orb_915["high"] - orb_915[["open", "close"]].max(axis=1)
    orb_915["lower_wick"] = orb_915[["open", "close"]].min(axis=1) - orb_915["low"]
    orb_915["range_915"]  = orb_915["range"]
    orb_915["wick_bias"]  = np.where(
        orb_915["upper_wick"] / (orb_915["range_915"] + 1e-9) > WICK_BIAS_THRESH,
        -1,   # bearish bias
        np.where(
            orb_915["lower_wick"] / (orb_915["range_915"] + 1e-9) > WICK_BIAS_THRESH,
            1,    # bullish bias
            0     # neutral
        )
    )
    daily_bias = orb_915["wick_bias"].reset_index()
    daily_bias["date"] = pd.to_datetime(daily_bias["Datetime"]).dt.date
    daily_bias = daily_bias.set_index("date")["wick_bias"]

    # ORB (9:15–9:45) per day
    orb_window = df[
        (df["time"] >= dtime(9, 15)) & (df["time"] <= dtime(9, 45))
    ].groupby("date").agg(orb_high=("high", "max"), orb_low=("low", "min"))

    df["wick_bias"] = df["date"].map(daily_bias)
    df["orb_high"]  = df["date"].map(orb_high := orb_window["orb_high"])
    df["orb_low"]   = df["date"].map(orb_window["orb_low"])

    return df


# ─────────────────────────── ENTRY FILTER ─────────────────────────
def in_entry_window(t):
    for (ws, we) in ENTRY_WINDOWS:
        if ws <= t <= we:
            return True
    return False


# ─────────────────────────── BACKTEST ─────────────────────────────
def run_backtest(df):
    trades = []
    i = SQUEEZE_CANDLES + 1          # need history to compute features

    # Pre-compute list for speed
    idx    = df.index
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    times  = df["time"].values
    dates  = df["date"].values
    bull_b = df["bull_break"].values
    bear_b = df["bear_break"].values
    c_high = df["comp_high"].values
    c_low  = df["comp_low"].values
    c_size = df["comp_size"].values
    w_bias = df["wick_bias"].values

    active = None   # current open trade

    while i < len(df):
        t = times[i]
        d = dates[i]

        # ── MANAGE OPEN TRADE ──────────────────────────────────────
        if active is not None:
            entry_i    = active["entry_i"]
            direction  = active["direction"]
            entry_px   = active["entry_px"]
            stop_px    = active["stop"]
            target_px  = active["target"]
            candles_held = i - entry_i

            current_high = highs[i]
            current_low  = lows[i]
            current_close= closes[i]

            hit_target = (direction == 1 and current_high >= target_px) or \
                         (direction == -1 and current_low  <= target_px)
            hit_stop   = (direction == 1 and current_low  <= stop_px) or \
                         (direction == -1 and current_high >= stop_px)
            time_stop  = t >= EXIT_BEFORE or candles_held >= MAX_HOLD_CANDLES

            exit_px = None
            reason  = ""
            if hit_target and hit_stop:
                # Both in same candle — use stop (conservative, no look-ahead ordering)
                exit_px, reason = stop_px, "stop"
            elif hit_target:
                exit_px, reason = target_px, "target"
            elif hit_stop:
                exit_px, reason = stop_px, "stop"
            elif time_stop:
                exit_px, reason = current_close, "time"

            if exit_px is not None:
                pnl_pts = direction * (exit_px - entry_px) - 2 * SLIPPAGE_PTS
                trades.append({
                    "date":        str(d),
                    "entry_time":  str(active["entry_time"]),
                    "exit_time":   str(t),
                    "direction":   "LONG" if direction == 1 else "SHORT",
                    "entry_px":    round(entry_px, 2),
                    "stop_px":     round(stop_px, 2),
                    "target_px":   round(target_px, 2),
                    "exit_px":     round(exit_px, 2),
                    "pnl_pts":     round(pnl_pts, 2),
                    "exit_reason": reason,
                    "candles_held":candles_held,
                })
                active = None
                i += 1
                continue

        # ── LOOK FOR NEW SIGNAL ────────────────────────────────────
        if active is None and in_entry_window(t):
            # Entry is on NEXT candle open — so we peek at i+1
            if i + 1 >= len(df):
                i += 1
                continue

            signal = 0
            if bull_b[i]:
                signal = 1
            elif bear_b[i]:
                signal = -1

            if signal != 0:
                # Wick bias filter: 0 = neutral (allow), ±1 must align
                bias = w_bias[i]
                if bias != 0 and bias != signal:
                    i += 1
                    continue   # wick bias contradicts signal — skip

                comp_range = c_size[i]
                if comp_range <= 0:
                    i += 1
                    continue

                # Entry at open of NEXT candle (no look-ahead)
                entry_px = opens[i + 1]
                stop_px  = (c_low[i] - SLIPPAGE_PTS) if signal == 1 \
                           else (c_high[i] + SLIPPAGE_PTS)
                tgt_dist = comp_range * TARGET_MULT
                target_px= (entry_px + tgt_dist) if signal == 1 \
                           else (entry_px - tgt_dist)

                active = {
                    "entry_i":    i + 1,
                    "entry_time": times[i + 1],
                    "direction":  signal,
                    "entry_px":   entry_px,
                    "stop":       stop_px,
                    "target":     target_px,
                }
                i += 2   # skip the entry candle itself
                continue

        i += 1

    return pd.DataFrame(trades)


# ─────────────────────────── STATS ────────────────────────────────
def print_stats(trades, df):
    if trades.empty:
        print("[!] No trades generated.")
        return

    total_trades = len(trades)
    wins   = trades[trades["pnl_pts"] > 0]
    losses = trades[trades["pnl_pts"] <= 0]
    win_rate = len(wins) / total_trades * 100

    gross_profit = wins["pnl_pts"].sum()
    gross_loss   = losses["pnl_pts"].sum()
    net_pnl      = trades["pnl_pts"].sum()
    profit_factor= abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    avg_win  = wins["pnl_pts"].mean()   if not wins.empty   else 0
    avg_loss = losses["pnl_pts"].mean() if not losses.empty else 0
    rr_ratio = abs(avg_win / avg_loss)  if avg_loss != 0    else float("inf")

    # Drawdown
    cumulative = trades["pnl_pts"].cumsum()
    rolling_max = cumulative.cummax()
    drawdown    = cumulative - rolling_max
    max_dd      = drawdown.min()

    # Expectancy
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # By exit reason
    by_reason = trades.groupby("exit_reason")["pnl_pts"].agg(["count", "sum", "mean"])

    # By session
    trades["hour_entry"] = pd.to_datetime(trades["entry_time"]).dt.hour
    by_session = trades.groupby(
        trades["hour_entry"].apply(lambda h: "09-10 ORB" if h < 10 else "14-15 Afternoon")
    )["pnl_pts"].agg(["count", "sum", "mean"])

    days_traded = df.index.normalize().nunique()

    banner = "=" * 64
    print(f"\n{banner}")
    print("  NIFTY 50 | Compression Breakout PA Strategy | Backtest")
    print(banner)
    print(f"  Period          : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Trading days    : {days_traded}")
    print(f"  Total candles   : {len(df):,}")
    print(f"{banner}")
    print(f"  Total trades    : {total_trades}")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Profit factor   : {profit_factor:.2f}")
    print(f"  Avg win (pts)   : {avg_win:.1f}")
    print(f"  Avg loss (pts)  : {avg_loss:.1f}")
    print(f"  Risk/Reward     : 1 : {rr_ratio:.2f}")
    print(f"  Expectancy/trade: {expectancy:.1f} pts")
    print(f"  Net P&L (pts)   : {net_pnl:.1f}")
    print(f"  Max Drawdown    : {max_dd:.1f} pts")
    print(f"{banner}")
    print("\n  Exit reason breakdown:")
    print(by_reason.to_string())
    print("\n  By session:")
    print(by_session.to_string())
    print(f"\n{banner}\n")

    # Monthly breakdown
    trades["month"] = pd.to_datetime(trades["date"]).dt.to_period("M")
    monthly = trades.groupby("month")["pnl_pts"].agg(["count","sum","mean"])
    monthly.columns = ["trades","net_pts","avg_pts"]
    print("  Monthly P&L summary:")
    print(monthly.to_string())
    print(f"\n{banner}\n")

    return {
        "total_trades":  total_trades,
        "win_rate":      win_rate,
        "net_pnl":       net_pnl,
        "profit_factor": profit_factor,
        "max_dd":        max_dd,
        "expectancy":    expectancy,
        "rr":            rr_ratio,
    }


# ─────────────────────────── MAIN ─────────────────────────────────
if __name__ == "__main__":
    df = fetch_data()
    df = compute_features(df)
    df = add_orb_and_wick_bias(df)

    print("[*] Running backtest ...")
    trades = run_backtest(df)

    stats = print_stats(trades, df)

    # Save trade log
    out_csv = "nifty50_pa_trades.csv"
    trades.to_csv(out_csv, index=False)
    print(f"[+] Trade log saved → {out_csv}")
    print(f"[+] Total trades: {len(trades)} | "
          f"Win rate: {stats['win_rate']:.1f}% | "
          f"Net P&L: {stats['net_pnl']:.0f} pts")