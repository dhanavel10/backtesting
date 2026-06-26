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


def fetch_intraday_chunked(ticker="^NSEI", interval="5m", days=55, chunk_days=7):
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
                print(f"  {cursor.date()} -> {chunk_end.date()} : {len(chunk)} bars")
            else:
                print(f"  {cursor.date()} -> {chunk_end.date()} : no data (holiday/weekend)")
        except Exception as e:
            print(f"  {cursor.date()} -> {chunk_end.date()} : ERROR - {e}")
        cursor = chunk_end

    if not chunks:
        raise ValueError(f"No data returned for {ticker}. Check ticker and connectivity.")

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

    print(f"\n{len(df)} candles | "
          f"{df.index[0].date()} -> {df.index[-1].date()} "
          f"({(df.index[-1].date() - df.index[0].date()).days} calendar days)\n")
    return df


def detect_pivots(df, left_bars=10, right_bars=10):
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


def print_pivot_summary(pivot_highs, pivot_lows):
    def _print_block(pivots, title, price_label, symbol):
        print("\n" + "=" * 65)
        print(f"  {title}  ({len(pivots)} detected)")
        print("=" * 65)
        if len(pivots) == 0:
            print("  (none - try reducing left_bars / right_bars)")
            return
        print(f"  {'#':>4}  {'Date & Time':<22}  {price_label:>10}  Session")
        print("  " + "-" * 55)
        for i, (_, row) in enumerate(pivots.sort_values("date").iterrows(), 1):
            dt_str = str(row["date"])[:19]
            print(f"  {i:>4}  {dt_str:<22}  {row['price']:>10.2f}  {symbol}  {row['session']}")

    _print_block(pivot_highs, "PIVOT HIGHS", "High Price", "^")
    _print_block(pivot_lows,  "PIVOT LOWS",  "Low Price",  "v")

    total = len(pivot_highs) + len(pivot_lows)
    print("=" * 65)
    print(f"  Total: {total} pivots  "
          f"({len(pivot_highs)} highs + {len(pivot_lows)} lows)\n")


def cluster_pivots_by_points(pivots, tolerance=15.0):
    if len(pivots) == 0:
        return []

    sorted_p = pivots.sort_values("price").reset_index(drop=True)
    prices   = sorted_p["price"].values

    clusters = []
    group_start = 0

    for i in range(1, len(prices)):
        if prices[i] - prices[group_start] > tolerance:
            grp = sorted_p.iloc[group_start:i]
            clusters.append(_make_cluster(grp))
            group_start = i

    grp = sorted_p.iloc[group_start:]
    clusters.append(_make_cluster(grp))

    return clusters


def _make_cluster(grp):
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


def analyse_level(df, level, half_band=15.0, min_rejection_pts=8.0):
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
                session_held[session] = False

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


def build_precision_zones(df, left_bars=10, right_bars=10, cluster_tolerance=15.0,
                           zone_half_band=15.0, min_wick_touches=3, min_sessions=2,
                           min_rejections=1, top_n=20):
    print("Step 1 - Detecting pivot highs and lows...")
    pivot_highs, pivot_lows = detect_pivots(df, left_bars, right_bars)
    print_pivot_summary(pivot_highs, pivot_lows)

    all_pivots = pd.concat([pivot_highs, pivot_lows], ignore_index=True)
    if len(all_pivots) == 0:
        raise ValueError("No pivots found. Reduce left_bars / right_bars.")

    print("Step 2 - Clustering by absolute point proximity...")
    clusters = cluster_pivots_by_points(all_pivots, tolerance=cluster_tolerance)
    print(f"  {len(clusters)} raw clusters from {len(all_pivots)} pivots\n")

    print("Step 3 - Scoring each level (touches, rejections, sessions)...")
    current_price = float(df["Close"].iloc[-1])
    latest_date   = df.index[-1]
    candidates    = []

    for cl in clusters:
        level = cl["price"]
        info  = analyse_level(df, level, half_band=zone_half_band)

        if info["wick_touches"]    < min_wick_touches: continue
        if info["sessions_touched"] < min_sessions:    continue
        if info["body_rejections"] < min_rejections:   continue

        days_ago = (latest_date - info["last_touch"]).days if info["last_touch"] else 999
        recency  = np.exp(-days_ago / 15)

        touch_score     = min(info["wick_touches"]    / 15, 1.0)
        rejection_score = min(info["body_rejections"] / max(info["wick_touches"], 1), 1.0)
        session_score   = min(info["sessions_touched"] / 10, 1.0)
        hold_score      = min(info["consecutive_holds"] / 5,  1.0)
        inside_score    = min(info["inside_bars"] / 10, 1.0)

        n_h = cl["types"].count("high")
        n_l = cl["types"].count("low")
        convergence     = 1.0 if (n_h > 0 and n_l > 0) else 0.5
        conv_label      = "Both" if convergence == 1.0 else ("High" if n_h >= n_l else "Low")

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

    print("Step 4 - Removing overlapping zones (keep strongest)...")
    zones_df = _remove_overlaps(zones_df, zone_half_band)
    zones_df = zones_df.head(top_n).reset_index(drop=True)

    print(f"  {len(zones_df)} precision zones  "
          f"(band = {int(zone_half_band*2)} pts per zone)\n")
    return zones_df


def _remove_overlaps(zones_df, half_band):
    df = zones_df.sort_values("strength", ascending=False).reset_index(drop=True)
    keep = [True] * len(df)
    for i in range(len(df)):
        if not keep[i]: continue
        for j in range(i + 1, len(df)):
            if not keep[j]: continue
            if abs(df.loc[i, "price"] - df.loc[j, "price"]) < half_band * 2:
                keep[j] = False
    return df[keep].reset_index(drop=True)


def print_zone_summary(zones_df, current_price):
    supports    = zones_df[zones_df["type"] == "Support"].sort_values("price", ascending=False)
    resistances = zones_df[zones_df["type"] == "Resistance"].sort_values("price")
    band        = zones_df["band_pts"].iloc[0]

    sep = "=" * 115
    hdr = (f"  {'Type':<10} {'Price':>8} {'Lower':>8} {'Upper':>8} {'Band':>5} "
           f"{'Str':>5} {'WkTch':>6} {'BdyRej':>7} {'InBar':>6} "
           f"{'Sess':>5} {'Holds':>6} {'Recv%':>6} {'Conv':>5} {'Dist':>7} {'LastTouch':<12}")

    def row_str(z):
        sym = "SUP" if z["type"] == "Support" else "RES"
        last = str(z["last_touch"])[:10] if z["last_touch"] is not None else "N/A"
        return (f"  {sym:<10} {z['price']:>8.2f} {z['lower']:>8.2f} {z['upper']:>8.2f} "
                f"{z['band_pts']:>5} {z['strength']:>5.1f} {z['wick_touches']:>6} "
                f"{z['body_rejections']:>7} {z['inside_bars']:>6} {z['sessions']:>5} "
                f"{z['holds']:>6} {z['recency_score']:>6.1f} {z['convergence']:>5} "
                f"{z['dist_pts']:>+7.0f} {last:<12}")

    print("\n" + sep)
    print(f"  PRECISION S/R ZONES  -  CMP: {current_price:.2f}  |  "
          f"Band width: {band} pts per zone  |  "
          f"{len(resistances)} resistance  +  {len(supports)} support")
    print(sep)
    print(hdr)

    print("\n  -- RESISTANCE (nearest first) " + "-" * 65)
    if len(resistances) == 0:
        print("  (none above current price)")
    for _, z in resistances.iterrows():
        print(row_str(z))

    print(f"\n  {'-'*42}  CMP {current_price:.2f}  {'-'*42}")

    print("\n  -- SUPPORT (nearest first) " + "-" * 68)
    if len(supports) == 0:
        print("  (none below current price)")
    for _, z in supports.iterrows():
        print(row_str(z))

    print("\n" + sep + "\n")