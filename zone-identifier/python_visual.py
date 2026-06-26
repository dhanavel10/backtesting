"""
Precision S/R Zone Detector — NIFTY 50 (Intraday Options Buying)
=================================================================
Built for intraday option buyers who need tight 20–30 point zones.

Core philosophy:
  A zone is only valid if price has REPEATEDLY reacted from the EXACT
  same level. Fuzzy, wide zones are useless for options entries.

Five-layer validation (all must pass):
  1. Pivot detection        — confirmed swing highs/lows (50-min window)
  2. Exact price clustering — absolute POINT-based grouping (not %)
  3. Touch counting         — minimum N wicks that entered the band
  4. Rejection confirmation — price must have closed AWAY from the level
  5. Multi-session check    — level respected across different trading days

Output: zones with a 20–30 point band, ranked by composite strength score.

Requirements:
    pip install yfinance pandas numpy scipy scikit-learn plotly
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# 1. DATA FETCHING
# ════════════════════════════════════════════════════════════════

def fetch_intraday_chunked(
    ticker:     str = "^NSEI",
    interval:   str = "5m",
    days:       int = 55,
    chunk_days: int = 7,
) -> pd.DataFrame:
    """
    Fetch 5-min intraday data in weekly chunks (Yahoo Finance hard limit).
    Filters to NSE session: 09:15 – 15:30 IST.
    """
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    chunks, cursor = [], start_dt

    print(f"Fetching {interval} data for {ticker} ({days} days in {chunk_days}-day chunks)...")

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        try:
            chunk = yf.download(
                ticker,
                start=cursor.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if len(chunk) > 0:
                chunk.columns = [c[0] if isinstance(c, tuple) else c for c in chunk.columns]
                chunks.append(chunk)
                print(f"  {cursor.date()} → {chunk_end.date()} : {len(chunk)} bars")
            else:
                print(f"  {cursor.date()} → {chunk_end.date()} : no data (holiday/weekend)")
        except Exception as e:
            print(f"  {cursor.date()} → {chunk_end.date()} : ERROR — {e}")
        cursor = chunk_end

    if not chunks:
        raise ValueError(f"No data returned for {ticker}. Check ticker and connectivity.")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    # Convert to IST and filter to NSE session
    try:
        df.index = df.index.tz_convert("Asia/Kolkata")
    except TypeError:
        try:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        except Exception:
            pass

    df = df.between_time("09:15", "15:30")

    print(f"\n✓ {len(df)} candles | "
          f"{df.index[0].date()} → {df.index[-1].date()} "
          f"({(df.index[-1].date() - df.index[0].date()).days} calendar days)\n")
    return df


# ════════════════════════════════════════════════════════════════
# 2. PIVOT DETECTION
#    Strict left+right confirmation — only real swing points survive
# ════════════════════════════════════════════════════════════════

def detect_pivots(
    df:         pd.DataFrame,
    left_bars:  int = 10,
    right_bars: int = 10,
) -> tuple:
    """
    Pivot high : bar whose High is strictly highest vs left_bars to its
                 left AND right_bars to its right.
    Pivot low  : same logic but for Lows.

    left_bars=10, right_bars=10 on 5m = 50-min window each side.
    This removes noise and keeps only meaningful intraday swing points.
    """
    highs = df["High"].values
    lows  = df["Low"].values

    raw_phi = argrelextrema(highs, np.greater_equal, order=left_bars)[0]
    raw_plo = argrelextrema(lows,  np.less_equal,    order=left_bars)[0]

    def confirm_both_sides(idx_arr, values, is_high):
        confirmed = []
        for i in idx_arr:
            lw = values[max(0, i - left_bars):i]
            rw = values[i+1:min(i + right_bars + 1, len(values))]
            if len(lw) == 0 or len(rw) == 0:
                continue
            if is_high and values[i] > np.max(lw) and values[i] > np.max(rw):
                confirmed.append(i)
            elif not is_high and values[i] < np.min(lw) and values[i] < np.min(rw):
                confirmed.append(i)
        return np.array(confirmed)

    phi = confirm_both_sides(raw_phi, highs, True)
    plo = confirm_both_sides(raw_plo, lows,  False)

    def make_pivot_df(idx_arr, price_col, ptype):
        if len(idx_arr) == 0:
            return pd.DataFrame(columns=["price","date","bar_idx","type","session"])
        return pd.DataFrame({
            "price":   df[price_col].iloc[idx_arr].values,
            "date":    df.index[idx_arr],
            "bar_idx": idx_arr,
            "type":    ptype,
            "session": [d.date() for d in df.index[idx_arr]],
        })

    pivot_highs = make_pivot_df(phi, "High", "high")
    pivot_lows  = make_pivot_df(plo, "Low",  "low")
    return pivot_highs, pivot_lows


def print_pivot_summary(pivot_highs: pd.DataFrame, pivot_lows: pd.DataFrame):
    """Print detected pivots to console in chronological tables."""

    def _print_block(pivots, title, price_label, symbol):
        print("\n" + "═" * 65)
        print(f"  {title}  ({len(pivots)} detected)")
        print("═" * 65)
        if len(pivots) == 0:
            print("  (none — try reducing left_bars / right_bars)")
            return
        print(f"  {'#':>4}  {'Date & Time':<22}  {price_label:>10}  Session")
        print("  " + "─" * 55)
        for i, (_, row) in enumerate(pivots.sort_values("date").iterrows(), 1):
            dt_str = str(row["date"])[:19]
            print(f"  {i:>4}  {dt_str:<22}  {row['price']:>10.2f}  {symbol}  {row['session']}")

    _print_block(pivot_highs, "PIVOT HIGHS", "High Price", "▲")
    _print_block(pivot_lows,  "PIVOT LOWS",  "Low Price",  "▼")

    total = len(pivot_highs) + len(pivot_lows)
    print("═" * 65)
    print(f"  Total: {total} pivots  "
          f"({len(pivot_highs)} highs + {len(pivot_lows)} lows)\n")


# ════════════════════════════════════════════════════════════════
# 3. ABSOLUTE POINT-BASED CLUSTERING
#    THE key fix: group pivots by exact point distance, not %
# ════════════════════════════════════════════════════════════════

def cluster_pivots_by_points(
    pivots:    pd.DataFrame,
    tolerance: float = 15.0,   # absolute points (NOT percentage)
) -> list:
    """
    Group pivots whose prices are within `tolerance` points of each other.

    Why absolute points instead of percentage:
      On Nifty @ 24,000 — 1% = 240 points → huge zones
      15 absolute points = 0.06% → precision zones for options buying

    Algorithm: single-linkage clustering on sorted price array.
    A new cluster starts whenever the gap to previous price > tolerance.
    """
    if len(pivots) == 0:
        return []

    sorted_p = pivots.sort_values("price").reset_index(drop=True)
    prices   = sorted_p["price"].values

    clusters = []
    group_start = 0

    for i in range(1, len(prices)):
        # Gap from first member of current group to current price
        if prices[i] - prices[group_start] > tolerance:
            # Close current cluster
            grp = sorted_p.iloc[group_start:i]
            clusters.append(_make_cluster(grp))
            group_start = i

    # Final cluster
    grp = sorted_p.iloc[group_start:]
    clusters.append(_make_cluster(grp))

    return clusters


def _make_cluster(grp: pd.DataFrame) -> dict:
    return {
        "price":     grp["price"].mean(),
        "price_min": grp["price"].min(),
        "price_max": grp["price"].max(),
        "spread":    grp["price"].max() - grp["price"].min(),
        "n_pivots":  len(grp),
        "types":     list(grp["type"]),
        "sessions":  list(grp["session"].unique()) if "session" in grp.columns else [],
        "dates":     list(grp["date"]),
    }


# ════════════════════════════════════════════════════════════════
# 4. TOUCH & REJECTION ANALYSIS
#    Precise wick-level scanning across the full 60-day dataset
# ════════════════════════════════════════════════════════════════

def analyse_level(
    df:                pd.DataFrame,
    level:             float,
    half_band:         float = 15.0,
    min_rejection_pts: float =  8.0,
) -> dict:
    """
    Scan every 5m candle and measure how price behaved near `level`.

    wick_touch      : wick entered the band (High >= lower AND Low <= upper)
    body_rejection  : close is > min_rejection_pts outside the band
                      → strong sign the level held
    inside_bar      : entire candle body within band (accumulation/indecision)
    sessions_touched: number of unique trading days that visited the level
    consecutive_holds: sessions where level held (closed outside after touching)
    """
    upper = level + half_band
    lower = level - half_band

    wick_touches     = 0
    body_rejections  = 0
    inside_bars      = 0
    sessions_touched = set()
    last_touch_dt    = None
    session_held     = defaultdict(bool)

    for dt, row in df.iterrows():
        hi, lo, cl = row["High"], row["Low"], row["Close"]
        session = dt.date()

        wick_in = (hi >= lower) and (lo <= upper)

        if wick_in:
            wick_touches += 1
            sessions_touched.add(session)
            last_touch_dt = dt

            if cl > upper + min_rejection_pts or cl < lower - min_rejection_pts:
                body_rejections += 1
                session_held[session] = True
            elif (lo < lower - 5 and cl < lower - 5) or (hi > upper + 5 and cl > upper + 5):
                session_held[session] = False  # broke through

        if hi <= upper and lo >= lower:
            inside_bars += 1

    return {
        "wick_touches":      wick_touches,
        "body_rejections":   body_rejections,
        "inside_bars":       inside_bars,
        "sessions_touched":  len(sessions_touched),
        "consecutive_holds": sum(1 for v in session_held.values() if v),
        "last_touch":        last_touch_dt,
    }


# ════════════════════════════════════════════════════════════════
# 5. PRECISION ZONE BUILDER
# ════════════════════════════════════════════════════════════════

def build_precision_zones(
    df:                pd.DataFrame,
    left_bars:         int   = 10,
    right_bars:        int   = 10,
    cluster_tolerance: float = 15.0,  # absolute points
    zone_half_band:    float = 15.0,  # zone = level ± this (absolute points)
    min_wick_touches:  int   =  3,
    min_sessions:      int   =  2,
    min_rejections:    int   =  1,
    top_n:             int   = 20,
) -> pd.DataFrame:
    """
    Full precision pipeline:
      pivot detection → point clustering → touch/rejection analysis → scoring
    Zone total width = zone_half_band × 2 points (default = 30 pts)
    """

    # ── Step 1: Pivots ─────────────────────────────────────
    print("Step 1 — Detecting pivot highs and lows...")
    pivot_highs, pivot_lows = detect_pivots(df, left_bars, right_bars)
    print_pivot_summary(pivot_highs, pivot_lows)

    all_pivots = pd.concat([pivot_highs, pivot_lows], ignore_index=True)
    if len(all_pivots) == 0:
        raise ValueError("No pivots found. Reduce left_bars / right_bars.")

    # ── Step 2: Absolute point clustering ──────────────────
    print("Step 2 — Clustering by absolute point proximity...")
    clusters = cluster_pivots_by_points(all_pivots, tolerance=cluster_tolerance)
    print(f"  {len(clusters)} raw clusters from {len(all_pivots)} pivots\n")

    # ── Step 3: Touch & rejection scoring ──────────────────
    print("Step 3 — Scoring each level (touches, rejections, sessions)...")
    current_price = float(df["Close"].iloc[-1])
    latest_date   = df.index[-1]
    candidates    = []

    for cl in clusters:
        level = cl["price"]
        info  = analyse_level(df, level, half_band=zone_half_band)

        # Hard filters — ALL must pass
        if info["wick_touches"]    < min_wick_touches: continue
        if info["sessions_touched"] < min_sessions:    continue
        if info["body_rejections"] < min_rejections:   continue

        # Recency (exponential decay, half-life ~15 trading days)
        days_ago = (latest_date - info["last_touch"]).days if info["last_touch"] else 999
        recency  = np.exp(-days_ago / 15)

        # Scoring components
        touch_score     = min(info["wick_touches"]    / 15, 1.0)
        rejection_score = min(info["body_rejections"] / max(info["wick_touches"], 1), 1.0)
        session_score   = min(info["sessions_touched"] / 10, 1.0)
        hold_score      = min(info["consecutive_holds"] / 5,  1.0)
        inside_score    = min(info["inside_bars"] / 10, 1.0)

        # Pivot convergence bonus (both high and low pivots near same level = stronger)
        n_h = cl["types"].count("high")
        n_l = cl["types"].count("low")
        convergence     = 1.0 if (n_h > 0 and n_l > 0) else 0.5
        conv_label      = "Both" if convergence == 1.0 else ("High" if n_h >= n_l else "Low")

        # Tightness bonus (tighter cluster spread = more precise level)
        spread_score    = max(0.0, 1.0 - cl["spread"] / cluster_tolerance)

        strength = (
            0.28 * touch_score     +
            0.22 * rejection_score +
            0.18 * recency         +
            0.12 * session_score   +
            0.08 * hold_score      +
            0.07 * convergence     +
            0.05 * spread_score
        ) * 100

        zone_type = "Support" if level < current_price else "Resistance"

        candidates.append({
            "price":          round(level, 2),
            "upper":          round(level + zone_half_band, 2),
            "lower":          round(level - zone_half_band, 2),
            "band_pts":       int(zone_half_band * 2),
            "type":           zone_type,
            "strength":       round(strength, 1),
            "wick_touches":   info["wick_touches"],
            "body_rejections":info["body_rejections"],
            "inside_bars":    info["inside_bars"],
            "sessions":       info["sessions_touched"],
            "holds":          info["consecutive_holds"],
            "recency_score":  round(recency * 100, 1),
            "last_touch":     info["last_touch"],
            "n_pivots":       cl["n_pivots"],
            "spread_pts":     round(cl["spread"], 1),
            "convergence":    conv_label,
            "dist_pts":       round(level - current_price, 1),
        })

    if not candidates:
        raise ValueError(
            "No zones passed all filters.\n"
            "Try lowering: min_wick_touches, min_sessions, or min_rejections."
        )

    zones_df = pd.DataFrame(candidates).sort_values("strength", ascending=False)

    # ── Step 4: Remove overlapping zones ───────────────────
    print("Step 4 — Removing overlapping zones (keep strongest)...")
    zones_df = _remove_overlaps(zones_df, zone_half_band)
    zones_df = zones_df.head(top_n).reset_index(drop=True)

    print(f"  ✓ {len(zones_df)} precision zones  "
          f"(band = {int(zone_half_band*2)} pts per zone)\n")
    return zones_df


def _remove_overlaps(zones_df: pd.DataFrame, half_band: float) -> pd.DataFrame:
    """Keep only the stronger zone when two bands overlap."""
    df = zones_df.sort_values("strength", ascending=False).reset_index(drop=True)
    keep = [True] * len(df)
    for i in range(len(df)):
        if not keep[i]: continue
        for j in range(i + 1, len(df)):
            if not keep[j]: continue
            if abs(df.loc[i, "price"] - df.loc[j, "price"]) < half_band * 2:
                keep[j] = False
    return df[keep].reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# 6. CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════

def print_zone_summary(zones_df: pd.DataFrame, current_price: float):
    """Print precision zones split into resistance above and support below."""
    supports    = zones_df[zones_df["type"] == "Support"].sort_values("price", ascending=False)
    resistances = zones_df[zones_df["type"] == "Resistance"].sort_values("price")
    band        = zones_df["band_pts"].iloc[0]

    sep = "═" * 115
    hdr = (f"  {'Type':<10} {'Price':>8} {'Lower':>8} {'Upper':>8} {'Band':>5} "
           f"{'Str':>5} {'WkTch':>6} {'BdyRej':>7} {'InBar':>6} "
           f"{'Sess':>5} {'Holds':>6} {'Recv%':>6} {'Conv':>5} {'Dist':>7} {'LastTouch':<12}")
    div = "  " + "─" * 109

    def row_str(z):
        sym = "▲ SUP" if z["type"] == "Support" else "▼ RES"
        last = str(z["last_touch"])[:10] if z["last_touch"] is not None else "N/A"
        return (f"  {sym:<10} {z['price']:>8.2f} {z['lower']:>8.2f} {z['upper']:>8.2f} "
                f"{z['band_pts']:>5} {z['strength']:>5.1f} {z['wick_touches']:>6} "
                f"{z['body_rejections']:>7} {z['inside_bars']:>6} {z['sessions']:>5} "
                f"{z['holds']:>6} {z['recency_score']:>6.1f} {z['convergence']:>5} "
                f"{z['dist_pts']:>+7.0f} {last:<12}")

    print("\n" + sep)
    print(f"  PRECISION S/R ZONES  —  CMP: {current_price:.2f}  |  "
          f"Band width: {band} pts per zone  |  "
          f"{len(resistances)} resistance  +  {len(supports)} support")
    print(sep)
    print(hdr)

    print("\n  ── RESISTANCE (nearest first) " + "─" * 65)
    if len(resistances) == 0:
        print("  (none above current price)")
    for _, z in resistances.iterrows():
        print(row_str(z))

    print(f"\n  {'─'*42}  CMP {current_price:.2f}  {'─'*42}")

    print("\n  ── SUPPORT (nearest first) " + "─" * 68)
    if len(supports) == 0:
        print("  (none below current price)")
    for _, z in supports.iterrows():
        print(row_str(z))

    print("\n" + sep)
    print(f"  Columns: WkTch=wick touches | BdyRej=body rejections | "
          f"InBar=inside bars | Sess=sessions | Recv%=recency | "
          f"Conv=pivot type | Dist=pts from CMP | LastTouch=date of last price visit\n")


# ════════════════════════════════════════════════════════════════
# 7. INTERACTIVE CHART
# ════════════════════════════════════════════════════════════════

def plot_zones(df: pd.DataFrame, zones_df: pd.DataFrame, ticker: str = ""):
    """
    Plotly 5m candlestick chart — last 10 sessions — with precision S/R overlays.
    Zone opacity scales with strength. Labels show exact point range.
    """
    last_sessions = sorted(set(df.index.date))[-10:]
    df_plot = df[df.index.date >= last_sessions[0]]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.80, 0.20], vertical_spacing=0.02,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"],   close=df_plot["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
        name=ticker,
    ), row=1, col=1)

    # Volume
    vcol = ["#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df_plot["Close"], df_plot["Open"])]
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot["Volume"],
        marker_color=vcol, opacity=0.45, name="Volume",
    ), row=2, col=1)

    # Zone bands (strength drives opacity)
    for _, z in zones_df.iterrows():
        is_sup = z["type"] == "Support"
        alpha  = 0.07 + (z["strength"] / 100) * 0.20
        la     = 0.5  + (z["strength"] / 100) * 0.5

        fc = f"rgba(38,166,154,{alpha:.2f})"  if is_sup else f"rgba(239,83,80,{alpha:.2f})"
        lc = f"rgba(38,166,154,{la:.2f})"    if is_sup else f"rgba(239,83,80,{la:.2f})"

        fig.add_hrect(y0=z["lower"], y1=z["upper"],
                      fillcolor=fc, line_width=0, row=1, col=1)
        fig.add_hline(y=z["price"], line_color=lc,
                      line_width=1.5, line_dash="solid", row=1, col=1)

        sym   = "S" if is_sup else "R"
        label = (f"{sym} {z['price']:.0f}  [{z['lower']:.0f}–{z['upper']:.0f}]  "
                 f"str={z['strength']:.0f}  T={z['wick_touches']} Rej={z['body_rejections']}")
        fig.add_annotation(
            x=df_plot.index[-1], y=z["price"], text=label,
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=9, color=lc),
            bgcolor="rgba(0,0,0,0.55)", borderpad=2,
            row=1, col=1,
        )

    # Current price line
    cp = float(df["Close"].iloc[-1])
    fig.add_hline(y=cp, line_color="rgba(255,235,59,0.85)",
                  line_width=1.5, line_dash="dash", row=1, col=1)
    fig.add_annotation(
        x=df_plot.index[-1], y=cp, text=f"CMP {cp:.2f}",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=10, color="rgba(255,235,59,0.9)"),
        row=1, col=1,
    )

    # Session separators
    for sd in last_sessions[1:]:
        bars = df_plot[df_plot.index.date == sd]
        if len(bars):
            fig.add_vline(x=bars.index[0],
                          line_dash="dot",
                          line_color="rgba(180,180,180,0.2)",
                          line_width=1, row=1, col=1)

    band = zones_df["band_pts"].iloc[0]
    fig.update_layout(
        title=f"{ticker} — Precision S/R Zones  (5m · ±{band//2} pt bands · last 10 sessions)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=820,
        showlegend=False,
        margin=dict(l=60, r=270, t=55, b=40),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
    )
    fig.update_yaxes(title_text="Nifty 50", row=1, col=1,
                     gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[15.5, 9.25], pattern="hour"),
    ])

    fig.show()
    return fig


# ════════════════════════════════════════════════════════════════
# 8. MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Ticker & data ────────────────────────────────────────
    TICKER     = "^NSEI"    # Nifty 50  |  ^NSEBANK = Bank Nifty
    INTERVAL   = "5m"       # 5-minute candles
    DAYS       = 55         # max 60 days for 5m data
    CHUNK_DAYS = 7          # chunk size for API calls

    # ── Pivot window ─────────────────────────────────────────
    # 10 bars each side = 50-min swing window on 5m chart
    # Increase to 12-15 for fewer but stronger pivots
    LEFT_BARS  = 10
    RIGHT_BARS = 10

    # ── Zone precision — ALL IN ABSOLUTE POINTS ──────────────
    #
    #  CLUSTER_TOLERANCE : pivots within this many points join one cluster
    #    15 pts → cluster radius ≈ ±7.5 pts
    #    Tighter (10 pts) = more precise but fewer zones
    #
    #  ZONE_HALF_BAND : zone extends ± this many points from the level
    #    15 pts → 30-point total band  (recommended start)
    #    10 pts → 20-point total band  (tighter, requires more data)
    #
    CLUSTER_TOLERANCE = 15.0   # points
    ZONE_HALF_BAND    = 15.0   # points  →  30-pt wide zones

    # ── Quality filters ──────────────────────────────────────
    MIN_WICK_TOUCHES = 3   # wicks that entered the band
    MIN_SESSIONS     = 2   # must appear on at least N different days
    MIN_REJECTIONS   = 1   # candles that closed outside the zone

    TOP_N = 20             # maximum zones to display

    # ────────────────────────────────────────────────────────
    df = fetch_intraday_chunked(
        ticker=TICKER, interval=INTERVAL,
        days=DAYS, chunk_days=CHUNK_DAYS,
    )

    zones = build_precision_zones(
        df,
        left_bars         = LEFT_BARS,
        right_bars        = RIGHT_BARS,
        cluster_tolerance = CLUSTER_TOLERANCE,
        zone_half_band    = ZONE_HALF_BAND,
        min_wick_touches  = MIN_WICK_TOUCHES,
        min_sessions      = MIN_SESSIONS,
        min_rejections    = MIN_REJECTIONS,
        top_n             = TOP_N,
    )

    current_price = float(df["Close"].iloc[-1])
    print_zone_summary(zones, current_price)

    fig = plot_zones(df, zones, ticker=TICKER)

    # ── Exports (uncomment to use) ────────────────────────────
    # fig.write_html("nifty_sr_precision.html")
    # zones.to_csv("nifty_sr_precision.csv", index=False)