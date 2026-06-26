"""
Precision S/R Zone Detector — NIFTY 50 (Intraday Options Buying)
HTML Interactive Output Version
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from collections import defaultdict
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")


# ════════════════════════════════════════════════════════════════
# 1. DATA FETCHING
# ════════════════════════════════════════════════════════════════

def fetch_intraday_chunked(ticker="^NSEI", interval="5m", days=55, chunk_days=7):
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    chunks, cursor = [], start_dt

    print(f"Fetching {interval} data for {ticker}...")

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
        except Exception as e:
            print(f"  ERROR: {e}")
        cursor = chunk_end

    if not chunks:
        raise ValueError(f"No data returned for {ticker}.")

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
    print(f"✓ {len(df)} candles | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ════════════════════════════════════════════════════════════════
# 2. PIVOT DETECTION
# ════════════════════════════════════════════════════════════════

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

    return make_pivot_df(phi, "High", "high"), make_pivot_df(plo, "Low", "low")


# ════════════════════════════════════════════════════════════════
# 3. CLUSTERING
# ════════════════════════════════════════════════════════════════

def cluster_pivots_by_points(pivots, tolerance=15.0):
    if len(pivots) == 0:
        return []
    sorted_p = pivots.sort_values("price").reset_index(drop=True)
    prices   = sorted_p["price"].values
    clusters = []
    group_start = 0
    for i in range(1, len(prices)):
        if prices[i] - prices[group_start] > tolerance:
            clusters.append(_make_cluster(sorted_p.iloc[group_start:i]))
            group_start = i
    clusters.append(_make_cluster(sorted_p.iloc[group_start:]))
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


# ════════════════════════════════════════════════════════════════
# 4. TOUCH & REJECTION ANALYSIS — with full candle detail
# ════════════════════════════════════════════════════════════════

def classify_candle(open_, high, low, close):
    body = close - open_
    body_size = abs(body)
    candle_range = high - low
    if candle_range == 0:
        return "Doji", "neutral"
    body_ratio = body_size / candle_range
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low

    if body_ratio < 0.1:
        return "Doji", "neutral"
    if body_ratio < 0.3:
        if upper_wick > body_size * 2:
            return "Shooting Star", "bearish"
        if lower_wick > body_size * 2:
            return "Hammer", "bullish"
        return "Spinning Top", "neutral"
    if body > 0:
        if lower_wick > body_size * 1.5 and upper_wick < body_size * 0.3:
            return "Bullish Hammer", "bullish"
        if body_ratio > 0.7:
            return "Bullish Marubozu", "bullish"
        return "Bullish Candle", "bullish"
    else:
        if upper_wick > body_size * 1.5 and lower_wick < body_size * 0.3:
            return "Bearish Shooting Star", "bearish"
        if body_ratio > 0.7:
            return "Bearish Marubozu", "bearish"
        return "Bearish Candle", "bearish"


def analyse_level_detailed(df, level, half_band=15.0, min_rejection_pts=8.0, zone_type="Support"):
    upper = level + half_band
    lower = level - half_band

    touches = []
    sessions_touched = set()
    last_touch_dt = None
    session_held = defaultdict(bool)

    wick_touches    = 0
    body_rejections = 0
    inside_bars     = 0

    df_list = list(df.iterrows())   # indexed access for pre-candle capture

    for bar_idx, (dt, row) in enumerate(df_list):
        hi, lo, op, cl = row["High"], row["Low"], row["Open"], row["Close"]
        session = dt.date()

        wick_in = (hi >= lower) and (lo <= upper)

        if wick_in:
            wick_touches += 1
            sessions_touched.add(session)
            last_touch_dt = dt

            broke_up   = cl > upper + min_rejection_pts
            broke_down = cl < lower - min_rejection_pts
            is_inside  = hi <= upper and lo >= lower

            # "rejection" = zone held (bounced in the expected direction)
            # "breakout"  = zone was broken (closed through it in the wrong direction)
            is_held     = (zone_type == "Support" and broke_up) or \
                          (zone_type == "Resistance" and broke_down)
            is_breakout = (zone_type == "Support" and broke_down) or \
                          (zone_type == "Resistance" and broke_up)

            if is_held:
                body_rejections += 1
                session_held[session] = True

            if is_breakout:
                reaction = "breakout"
            elif is_held:
                reaction = "rejection"
            elif is_inside:
                reaction = "inside"
            else:
                reaction = "touch"

            candle_name, candle_sentiment = classify_candle(op, hi, lo, cl)

            # Capture 3 candles immediately before a breakout for context
            prev_candles = []
            if is_breakout:
                start = max(0, bar_idx - 3)
                for p_dt, p_row in df_list[start:bar_idx]:
                    pn, ps = classify_candle(
                        float(p_row["Open"]), float(p_row["High"]),
                        float(p_row["Low"]),  float(p_row["Close"])
                    )
                    prev_candles.append({
                        "datetime":  str(p_dt)[:16],
                        "open":      round(float(p_row["Open"]),  2),
                        "high":      round(float(p_row["High"]),  2),
                        "low":       round(float(p_row["Low"]),   2),
                        "close":     round(float(p_row["Close"]), 2),
                        "candle":    pn,
                        "sentiment": ps,
                        "move_pts":  round(float(p_row["Close"] - p_row["Open"]), 2),
                    })

            touches.append({
                "datetime":     str(dt)[:19],
                "date":         str(session),
                "open":         round(float(op), 2),
                "high":         round(float(hi), 2),
                "low":          round(float(lo), 2),
                "close":        round(float(cl), 2),
                "reaction":     reaction,
                "candle":       candle_name,
                "sentiment":    candle_sentiment,
                "move_pts":     round(float(cl - op), 2),
                "prev_candles": prev_candles,
            })

        if hi <= upper and lo >= lower:
            inside_bars += 1

    return {
        "wick_touches":      wick_touches,
        "body_rejections":   body_rejections,
        "inside_bars":       inside_bars,
        "sessions_touched":  len(sessions_touched),
        "consecutive_holds": sum(1 for v in session_held.values() if v),
        "last_touch":        last_touch_dt,
        "touch_details":     touches,
    }


# ════════════════════════════════════════════════════════════════
# 5. PRECISION ZONE BUILDER
# ════════════════════════════════════════════════════════════════

def build_precision_zones(
    df, left_bars=10, right_bars=10,
    cluster_tolerance=15.0, zone_half_band=15.0,
    min_wick_touches=3, min_sessions=2, min_rejections=1, top_n=20,
):
    print("Step 1 — Detecting pivots...")
    pivot_highs, pivot_lows = detect_pivots(df, left_bars, right_bars)
    print(f"  {len(pivot_highs)} highs + {len(pivot_lows)} lows")

    all_pivots = pd.concat([pivot_highs, pivot_lows], ignore_index=True)

    print("Step 2 — Clustering...")
    clusters = cluster_pivots_by_points(all_pivots, tolerance=cluster_tolerance)
    print(f"  {len(clusters)} raw clusters")

    print("Step 3 — Scoring levels...")
    current_price = float(df["Close"].iloc[-1])
    latest_date   = df.index[-1]
    candidates    = []

    for cl in clusters:
        level     = cl["price"]
        zone_type = "Support" if level < current_price else "Resistance"
        info      = analyse_level_detailed(df, level, half_band=zone_half_band, zone_type=zone_type)

        if info["wick_touches"]    < min_wick_touches: continue
        if info["sessions_touched"] < min_sessions:    continue
        if info["body_rejections"] < min_rejections:   continue

        days_ago = (latest_date - info["last_touch"]).days if info["last_touch"] else 999
        recency  = np.exp(-days_ago / 15)

        touch_score     = min(info["wick_touches"]    / 15, 1.0)
        rejection_score = min(info["body_rejections"] / max(info["wick_touches"], 1), 1.0)
        session_score   = min(info["sessions_touched"] / 10, 1.0)
        hold_score      = min(info["consecutive_holds"] / 5,  1.0)

        n_h = cl["types"].count("high")
        n_l = cl["types"].count("low")
        convergence  = 1.0 if (n_h > 0 and n_l > 0) else 0.5
        conv_label   = "Both" if convergence == 1.0 else ("High" if n_h >= n_l else "Low")
        spread_score = max(0.0, 1.0 - cl["spread"] / cluster_tolerance)

        strength = (
            0.28 * touch_score     +
            0.22 * rejection_score +
            0.18 * recency         +
            0.12 * session_score   +
            0.08 * hold_score      +
            0.07 * convergence     +
            0.05 * spread_score
        ) * 100

        candidates.append({
            "price":           round(level, 2),
            "upper":           round(level + zone_half_band, 2),
            "lower":           round(level - zone_half_band, 2),
            "band_pts":        int(zone_half_band * 2),
            "type":            zone_type,
            "strength":        round(strength, 1),
            "wick_touches":    info["wick_touches"],
            "body_rejections": info["body_rejections"],
            "inside_bars":     info["inside_bars"],
            "sessions":        info["sessions_touched"],
            "holds":           info["consecutive_holds"],
            "recency_score":   round(recency * 100, 1),
            "last_touch":      str(info["last_touch"])[:19] if info["last_touch"] else "N/A",
            "n_pivots":        cl["n_pivots"],
            "spread_pts":      round(cl["spread"], 1),
            "convergence":     conv_label,
            "dist_pts":        round(level - current_price, 1),
            "touch_details":   info["touch_details"],
        })

    if not candidates:
        raise ValueError("No zones passed all filters. Try lowering thresholds.")

    zones_df = pd.DataFrame(candidates).sort_values("strength", ascending=False)

    print("Step 4 — Removing overlaps...")
    zones_df = _remove_overlaps(zones_df, zone_half_band)
    zones_df = zones_df.head(top_n).reset_index(drop=True)
    print(f"✓ {len(zones_df)} final zones\n")
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


# ════════════════════════════════════════════════════════════════
# 6. HTML GENERATOR
# ════════════════════════════════════════════════════════════════

def generate_html(zones_df, current_price, ticker, df_ohlc):
    """Generate complete interactive HTML dashboard."""

    ohlc_data = []
    for dt, row in df_ohlc.iterrows():
        ohlc_data.append({
            "t": str(dt)[:19],
            "o": round(float(row["Open"]), 2),
            "h": round(float(row["High"]), 2),
            "l": round(float(row["Low"]), 2),
            "c": round(float(row["Close"]), 2),
            "v": int(row["Volume"]) if "Volume" in row else 0,
        })

    zones_json = zones_df.to_dict(orient="records")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Precision S/R Zones — {ticker}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

  :root {{
    --bg:      #0a0e14;
    --bg2:     #0f1520;
    --bg3:     #141c2a;
    --bg4:     #1a2235;
    --border:  #1e2d42;
    --border2: #253247;
    --text:    #c8d4e8;
    --text2:   #7a8fa8;
    --text3:   #4a5e75;
    --sup:     #00e5a0;
    --sup-dim: rgba(0,229,160,0.12);
    --sup-glow:rgba(0,229,160,0.25);
    --res:     #ff4d6d;
    --res-dim: rgba(255,77,109,0.12);
    --res-glow:rgba(255,77,109,0.25);
    --bull:    #26a69a;
    --bear:    #ef5350;
    --neutral: #7a8fa8;
    --accent:  #3d8fff;
    --gold:    #ffd24d;
    --purple:  #9b7fff;
    --radius:  8px;
    --radius2: 12px;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* ── HEADER ── */
  .header {{
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
  }}
  .header-left {{ display: flex; align-items: center; gap: 16px; }}
  .ticker-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px; font-weight: 700;
    color: #fff; letter-spacing: 1px;
  }}
  .header-meta {{ font-size: 11px; color: var(--text2); line-height: 1.8; }}
  .cmp-display {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px; font-weight: 600; color: var(--gold);
    display: flex; align-items: baseline; gap: 8px;
  }}
  .cmp-label {{ font-size: 10px; color: var(--text3); font-weight: 400; letter-spacing: 1px; }}

  /* ── LAYOUT ── */
  .layout {{
    display: grid;
    grid-template-columns: 360px 1fr;
    height: calc(100vh - 57px);
  }}

  /* ── ZONE LIST ── */
  .zone-list-panel {{
    background: var(--bg2);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    display: flex; flex-direction: column;
  }}
  .panel-header {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    background: var(--bg3);
  }}
  .panel-title {{
    font-size: 10px; font-weight: 600; letter-spacing: 2px;
    color: var(--text2); text-transform: uppercase;
  }}
  .count-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; padding: 2px 7px;
    background: var(--bg4); border: 1px solid var(--border2);
    border-radius: 20px; color: var(--text2);
  }}
  .section-label {{
    padding: 8px 16px 6px;
    font-size: 9px; font-weight: 700; letter-spacing: 2.5px;
    color: var(--text3); text-transform: uppercase;
    background: var(--bg);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }}
  .section-label.res {{ color: rgba(255,77,109,0.6); }}
  .section-label.sup {{ color: rgba(0,229,160,0.6); }}
  .cmp-divider {{
    padding: 9px 16px;
    background: rgba(255,210,77,0.06);
    border-top: 1px solid rgba(255,210,77,0.2);
    border-bottom: 1px solid rgba(255,210,77,0.2);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--gold);
    display: flex; align-items: center; gap: 8px;
  }}
  .cmp-line {{ flex: 1; height: 1px; background: rgba(255,210,77,0.2); }}

  .zone-card {{
    padding: 11px 16px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background 0.15s;
    display: flex; align-items: center; gap: 12px;
    position: relative;
  }}
  .zone-card:hover {{ background: var(--bg3); }}
  .zone-card.active {{ background: var(--bg4); }}
  .zone-card.active::before {{
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  }}
  .zone-card.active.sup::before {{ background: var(--sup); }}
  .zone-card.active.res::before {{ background: var(--res); }}

  .zone-type-dot {{
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  }}
  .sup .zone-type-dot {{ background: var(--sup); box-shadow: 0 0 6px var(--sup); }}
  .res .zone-type-dot {{ background: var(--res); box-shadow: 0 0 6px var(--res); }}

  .zone-card-body {{ flex: 1; min-width: 0; }}
  .zone-price-row {{
    display: flex; align-items: baseline;
    justify-content: space-between; margin-bottom: 3px;
  }}
  .zone-price {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px; font-weight: 600; color: #fff;
  }}
  .zone-dist {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; }}
  .zone-dist.sup {{ color: var(--sup); }}
  .zone-dist.res {{ color: var(--res); }}
  .zone-range {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: var(--text3); margin-bottom: 5px;
  }}
  .zone-pills {{ display: flex; gap: 4px; flex-wrap: wrap; }}
  .pill {{
    font-size: 9px; padding: 1px 5px; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    background: var(--bg4); border: 1px solid var(--border2); color: var(--text2);
  }}
  .pill.sup {{ background: var(--sup-dim); border-color: rgba(0,229,160,0.25); color: var(--sup); }}
  .pill.res {{ background: var(--res-dim); border-color: rgba(255,77,109,0.25); color: var(--res); }}

  .str-bar-wrap {{ width: 38px; flex-shrink: 0; }}
  .str-bar-bg {{
    height: 38px; width: 6px; background: var(--bg4); border-radius: 3px;
    margin: 0 auto; position: relative; overflow: hidden;
  }}
  .str-bar-fill {{
    position: absolute; bottom: 0; left: 0; right: 0;
    border-radius: 3px; transition: height 0.3s;
  }}
  .sup .str-bar-fill {{ background: linear-gradient(to top, var(--sup), rgba(0,229,160,0.3)); }}
  .res .str-bar-fill {{ background: linear-gradient(to top, var(--res), rgba(255,77,109,0.3)); }}
  .str-val {{
    text-align: center; margin-top: 3px;
    font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--text3);
  }}

  /* ── DETAIL PANEL ── */
  .detail-panel {{
    background: var(--bg);
    overflow-y: auto;
    display: flex; flex-direction: column;
  }}
  .detail-empty {{
    flex: 1; display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 16px; color: var(--text3);
    font-size: 13px; text-align: center;
  }}
  .detail-empty-icon {{ font-size: 40px; opacity: 0.3; }}
  .detail-content {{ padding: 20px 24px; }}

  .detail-zone-header {{
    display: flex; align-items: flex-start;
    justify-content: space-between; margin-bottom: 20px;
    padding-bottom: 16px; border-bottom: 1px solid var(--border);
  }}
  .detail-zone-price {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 32px; font-weight: 700; color: #fff; line-height: 1;
  }}
  .detail-zone-type {{
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; margin-top: 4px;
  }}
  .detail-zone-type.sup {{ color: var(--sup); }}
  .detail-zone-type.res {{ color: var(--res); }}
  .detail-zone-range {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: var(--text2); margin-top: 4px;
  }}
  .strength-circle {{
    width: 64px; height: 64px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; border: 2px solid; margin-bottom: 4px;
  }}
  .strength-circle.sup {{ border-color: var(--sup); box-shadow: 0 0 12px var(--sup-glow); }}
  .strength-circle.res {{ border-color: var(--res); box-shadow: 0 0 12px var(--res-glow); }}
  .strength-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px; font-weight: 700; color: #fff;
  }}
  .strength-label {{ font-size: 8px; color: var(--text3); letter-spacing: 1px; }}

  /* ── STATS GRID ── */
  .stats-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
    margin-bottom: 20px;
  }}
  .stat-card {{
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 10px 12px;
  }}
  .stat-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px; font-weight: 600; color: #fff; line-height: 1;
    margin-bottom: 4px;
  }}
  .stat-lbl {{ font-size: 9px; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; }}

  /* ── DISTRIBUTION BARS ── */
  .distr-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
    margin-bottom: 20px;
  }}
  .distr-card {{
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius2); padding: 14px;
  }}
  .distr-title {{
    font-size: 9px; color: var(--text3); text-transform: uppercase;
    letter-spacing: 1.5px; margin-bottom: 10px; font-weight: 600;
  }}
  .distr-row {{ margin-bottom: 8px; }}
  .distr-row:last-child {{ margin-bottom: 0; }}
  .distr-label-row {{
    display: flex; justify-content: space-between;
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    margin-bottom: 3px;
  }}
  .distr-pct {{ color: #fff; font-weight: 600; }}
  .distr-bar-bg {{
    height: 5px; background: var(--bg4); border-radius: 3px; overflow: hidden;
  }}
  .distr-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.4s; }}

  /* ── CHART ── */
  .chart-section {{ margin-bottom: 20px; }}
  .section-hdr {{
    font-size: 10px; font-weight: 600; letter-spacing: 2px;
    color: var(--text3); text-transform: uppercase;
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
  }}
  .section-hdr::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}
  .chart-wrap {{
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius2); padding: 12px;
    position: relative;
    overflow-x: auto;
    overflow-y: hidden;
    cursor: default;
  }}
  #miniChart {{ display: block; }}
  .chart-legend {{
    display: flex; gap: 14px; padding: 7px 10px;
    background: var(--bg4); border-radius: var(--radius);
    margin-top: 8px; font-size: 9px; color: var(--text3);
    flex-wrap: wrap;
  }}
  .chart-legend span {{ display: flex; align-items: center; gap: 4px; }}
  .leg-dot {{
    width: 8px; height: 8px; border-radius: 50%; display: inline-block;
  }}
  .leg-tri {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 7px solid var(--gold);
    display: inline-block;
  }}

  /* ── TOUCH TABLE ── */
  .touch-table-wrap {{
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius2); overflow: hidden; margin-bottom: 20px;
    overflow-x: auto;
  }}
  .touch-table {{
    width: 100%; border-collapse: collapse; font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
  }}
  .touch-table th {{
    background: var(--bg4); padding: 8px 10px;
    text-align: left; font-size: 9px; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--text3); border-bottom: 1px solid var(--border2);
    white-space: nowrap;
  }}
  .touch-table td {{
    padding: 6px 10px; border-bottom: 1px solid var(--border);
    color: var(--text); white-space: nowrap;
  }}
  .touch-table tr:last-child td {{ border-bottom: none; }}
  .touch-table tr:hover td {{ background: var(--bg4); }}

  .candle-bull {{ color: var(--bull); }}
  .candle-bear {{ color: var(--bear); }}
  .candle-neutral {{ color: var(--neutral); }}
  .reaction-rejection {{ color: var(--gold); font-weight: 700; }}
  .reaction-touch {{ color: var(--text2); }}
  .reaction-inside {{ color: var(--purple); font-weight: 600; }}
  .reaction-breakout {{
    color: var(--res); font-weight: 700; letter-spacing: 0.5px;
    text-shadow: 0 0 8px rgba(255,77,109,0.5);
  }}
  .breakout-row {{ background: rgba(255,77,109,0.05) !important; border-left: 3px solid var(--res); }}
  .prev-candles-row td {{ background: rgba(255,77,109,0.03); padding: 4px 10px; }}
  .prev-candle-chip {{
    display: inline-flex; align-items: center; gap: 3px;
    background: var(--bg4); border: 1px solid var(--border2);
    border-radius: 4px; padding: 2px 6px; margin-right: 4px;
    font-size: 8px; color: var(--text2);
  }}
  .leg-diamond {{
    width: 8px; height: 8px; background: rgba(255,60,80,0.9);
    transform: rotate(45deg); display: inline-block; flex-shrink: 0;
  }}

  /* ── SESSION CARDS ── */
  .session-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 6px; margin-bottom: 20px;
  }}
  .session-card {{
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 8px 10px; font-size: 10px;
  }}
  .session-date {{
    font-family: 'JetBrains Mono', monospace;
    color: var(--text2); margin-bottom: 4px; font-size: 9px;
  }}
  .session-result {{ font-weight: 600; }}
  .session-result.held {{ color: var(--sup); }}
  .session-result.broken {{ color: var(--res); }}
  .session-touches {{ font-size: 9px; color: var(--text3); margin-top: 2px; }}

  /* ── FILTER BAR ── */
  .filter-bar {{
    display: flex; gap: 6px; padding: 8px 16px;
    background: var(--bg3); border-bottom: 1px solid var(--border);
  }}
  .filter-btn {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; padding: 3px 8px; border-radius: 4px;
    border: 1px solid var(--border2); background: var(--bg4);
    color: var(--text2); cursor: pointer; transition: all 0.15s;
    letter-spacing: 1px;
  }}
  .filter-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .filter-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}

  .no-select {{ user-select: none; }}

  /* scrollbar */
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--text3); }}
</style>
</head>
<body>

<!-- HEADER -->
<header class="header no-select">
  <div class="header-left">
    <div class="ticker-badge">{ticker.replace('^','')}</div>
    <div class="header-meta">
      Precision S/R Zones &nbsp;·&nbsp; 5m Intraday<br>
      Band: ±{zones_df['band_pts'].iloc[0]//2} pts &nbsp;·&nbsp; Updated: {datetime.now().strftime('%d %b %Y %H:%M')}
    </div>
  </div>
  <div class="cmp-display">
    <span class="cmp-label">CMP</span>
    <span>{current_price:,.2f}</span>
  </div>
</header>

<!-- MAIN LAYOUT -->
<div class="layout">

  <!-- LEFT: ZONE LIST -->
  <div class="zone-list-panel">
    <div class="panel-header">
      <span class="panel-title">S/R Zones</span>
      <span class="count-badge" id="zoneCount">Loading…</span>
    </div>
    <div class="filter-bar">
      <button class="filter-btn active" onclick="filterZones('all', this)">ALL</button>
      <button class="filter-btn" onclick="filterZones('res', this)">RES</button>
      <button class="filter-btn" onclick="filterZones('sup', this)">SUP</button>
      <button class="filter-btn" onclick="filterZones('strong', this)">STR &gt;60</button>
    </div>
    <div id="zoneListBody"></div>
  </div>

  <!-- RIGHT: DETAIL PANEL -->
  <div class="detail-panel" id="detailPanel">
    <div class="detail-empty" id="detailEmpty">
      <div class="detail-empty-icon">⬡</div>
      <div>Select a zone to explore its full history</div>
      <div style="font-size:11px;color:var(--text3)">Touch events · Candle formations · Session reactions</div>
    </div>
    <div class="detail-content" id="detailContent" style="display:none"></div>
  </div>

</div>

<script>
const ZONES = {json.dumps(zones_json, default=str)};
const OHLC  = {json.dumps(ohlc_data)};
const CMP   = {current_price};
let activeFilter = 'all';

// ── SVG MINI-CANDLE renderer ─────────────────────────────────
function svgMiniCandle(o, h, l, c) {{
  const W = 14, H = 28;
  const bull = c >= o;
  const col  = bull ? '#26a69a' : '#ef5350';
  const pRange = (h - l) || 1;
  const py   = p => (((h - p) / pRange) * H).toFixed(1);
  const bTop = Math.min(+py(o), +py(c));
  const bBot = Math.max(+py(o), +py(c));
  const bH   = Math.max(bBot - bTop, 1.5);
  return '<svg width="14" height="28" style="vertical-align:middle;display:inline-block">'
    + '<line x1="7" y1="' + py(h) + '" x2="7" y2="' + bTop + '" stroke="' + col + '" stroke-width="1"/>'
    + '<line x1="7" y1="' + bBot  + '" x2="7" y2="' + py(l) + '" stroke="' + col + '" stroke-width="1"/>'
    + '<rect x="2" y="' + bTop + '" width="10" height="' + bH + '" fill="' + col + '" rx="1"/>'
    + '</svg>';
}}

// ── BUILD ZONE LIST ──────────────────────────────────────────
function buildZoneList(filter) {{
  const body = document.getElementById('zoneListBody');
  body.innerHTML = '';

  const res = ZONES.filter(z => z.type === 'Resistance').sort((a,b) => a.price - b.price);
  const sup = ZONES.filter(z => z.type === 'Support').sort((a,b) => b.price - a.price);

  let allZones = [...res.slice().reverse(), ...sup];
  if (filter === 'res') allZones = res.slice().reverse();
  else if (filter === 'sup') allZones = sup;
  else if (filter === 'strong') allZones = [...res.slice().reverse(), ...sup].filter(z => z.strength > 60);

  document.getElementById('zoneCount').textContent = allZones.length + ' zones';

  const resZones = allZones.filter(z => z.type === 'Resistance');
  if (resZones.length) {{
    body.insertAdjacentHTML('beforeend', '<div class="section-label res">▼ RESISTANCE</div>');
    resZones.forEach(z => body.appendChild(makeZoneCard(z)));
  }}

  if (filter === 'all' || filter === 'strong') {{
    body.insertAdjacentHTML('beforeend', '<div class="cmp-divider"><div class="cmp-line"></div><span>CMP ' + CMP.toLocaleString('en-IN', {{minimumFractionDigits:2}}) + '</span><div class="cmp-line"></div></div>');
  }}

  const supZones = allZones.filter(z => z.type === 'Support');
  if (supZones.length) {{
    body.insertAdjacentHTML('beforeend', '<div class="section-label sup">▲ SUPPORT</div>');
    supZones.forEach(z => body.appendChild(makeZoneCard(z)));
  }}
}}

function makeZoneCard(z) {{
  const isSup  = z.type === 'Support';
  const cls    = isSup ? 'sup' : 'res';
  const distSign = z.dist_pts > 0 ? '+' : '';
  const div = document.createElement('div');
  div.className = 'zone-card ' + cls;
  div.dataset.price = z.price;
  div.onclick = () => showDetail(z);
  div.innerHTML =
    '<div class="zone-type-dot"></div>' +
    '<div class="zone-card-body">' +
      '<div class="zone-price-row">' +
        '<span class="zone-price">' + z.price.toFixed(2) + '</span>' +
        '<span class="zone-dist ' + cls + '">' + distSign + z.dist_pts.toFixed(0) + ' pts</span>' +
      '</div>' +
      '<div class="zone-range">' + z.lower.toFixed(0) + ' – ' + z.upper.toFixed(0) + ' · ' + z.band_pts + 'pt band</div>' +
      '<div class="zone-pills">' +
        '<span class="pill ' + cls + '">' + z.wick_touches + 'T</span>' +
        '<span class="pill ' + cls + '">' + z.body_rejections + 'R</span>' +
        '<span class="pill">' + z.sessions + 'd</span>' +
        '<span class="pill">' + z.convergence + '</span>' +
        '<span class="pill">Rcn ' + z.recency_score + '%</span>' +
      '</div>' +
    '</div>' +
    '<div class="str-bar-wrap">' +
      '<div class="str-bar-bg"><div class="str-bar-fill" style="height:' + z.strength + '%"></div></div>' +
      '<div class="str-val">' + z.strength.toFixed(0) + '</div>' +
    '</div>';
  return div;
}}

// ── SHOW DETAIL ──────────────────────────────────────────────
function showDetail(z) {{
  document.querySelectorAll('.zone-card').forEach(c => {{
    c.classList.toggle('active', parseFloat(c.dataset.price) === z.price);
  }});

  document.getElementById('detailEmpty').style.display = 'none';
  const cont = document.getElementById('detailContent');
  cont.style.display = 'block';

  const isSup = z.type === 'Support';
  const cls   = isSup ? 'sup' : 'res';
  const label = isSup ? 'SUPPORT' : 'RESISTANCE';

  const allTouches = z.touch_details || [];
  const totalT = allTouches.length || 1;

  // Session breakdown
  const sessionMap = {{}};
  allTouches.forEach(t => {{
    if (!sessionMap[t.date]) sessionMap[t.date] = {{touches:[], rejections:0}};
    sessionMap[t.date].touches.push(t);
    if (t.reaction === 'rejection') sessionMap[t.date].rejections++;
  }});
  const sessions = Object.entries(sessionMap).sort((a,b) => b[0].localeCompare(a[0]));

  // Candle distribution
  const candleCounts = {{}};
  allTouches.forEach(t => {{ candleCounts[t.candle] = (candleCounts[t.candle] || 0) + 1; }});
  const topCandles = Object.entries(candleCounts).sort((a,b) => b[1]-a[1]).slice(0,5);

  // Reaction distribution
  const reactionCounts = {{rejection:0, breakout:0, touch:0, inside:0}};
  allTouches.forEach(t => {{ reactionCounts[t.reaction] = (reactionCounts[t.reaction]||0)+1; }});
  const rejPct = Math.round(reactionCounts.rejection / totalT * 100);
  const brkPct = Math.round(reactionCounts.breakout  / totalT * 100);
  const tchPct = Math.round(reactionCounts.touch      / totalT * 100);
  const insPct = Math.round(reactionCounts.inside     / totalT * 100);

  // Sentiment distribution
  const sentCounts = {{bullish:0, bearish:0, neutral:0}};
  allTouches.forEach(t => {{ sentCounts[t.sentiment] = (sentCounts[t.sentiment]||0)+1; }});
  const bullPct = Math.round(sentCounts.bullish / totalT * 100);
  const bearPct = Math.round(sentCounts.bearish / totalT * 100);
  const neutPct = Math.max(0, 100 - bullPct - bearPct);

  // Strength label
  const strLabel = z.strength >= 70 ? 'Very Strong' :
                   z.strength >= 55 ? 'Strong' :
                   z.strength >= 40 ? 'Moderate' : 'Weak';

  // Build reaction rows HTML (string concat to avoid template-literal nesting issues)
  const reactionRows = [
    ['Held (Rejection)', rejPct, 'var(--gold)'],
    ['Breakout',         brkPct, 'var(--res)'],
    ['Touch (Neutral)',  tchPct, 'var(--accent)'],
    ['Inside Bar',       insPct, 'var(--purple)'],
  ].map(function(r) {{
    return '<div class="distr-row">'
      + '<div class="distr-label-row"><span style="color:' + r[2] + '">' + r[0] + '</span><span class="distr-pct">' + r[1] + '%</span></div>'
      + '<div class="distr-bar-bg"><div class="distr-bar-fill" style="width:' + r[1] + '%;background:' + r[2] + '"></div></div>'
      + '</div>';
  }}).join('');

  const sentRows = [
    ['Bullish', bullPct, 'var(--bull)'],
    ['Bearish', bearPct, 'var(--bear)'],
    ['Neutral', neutPct, 'var(--neutral)'],
  ].map(function(r) {{
    return '<div class="distr-row">'
      + '<div class="distr-label-row"><span style="color:' + r[2] + '">' + r[0] + '</span><span class="distr-pct">' + r[1] + '%</span></div>'
      + '<div class="distr-bar-bg"><div class="distr-bar-fill" style="width:' + r[1] + '%;background:' + r[2] + '"></div></div>'
      + '</div>';
  }}).join('');

  const candleRows = topCandles.map(function(entry) {{
    const name = entry[0], count = entry[1];
    const pct = Math.round(count / totalT * 100);
    return '<div class="distr-row">'
      + '<div class="distr-label-row"><span style="color:var(--text2)">' + name + '</span><span class="distr-pct">' + count + ' (' + pct + '%)</span></div>'
      + '<div class="distr-bar-bg"><div class="distr-bar-fill" style="width:' + pct + '%;background:var(--accent)"></div></div>'
      + '</div>';
  }}).join('');

  // Session cards HTML
  const sessionCards = sessions.map(function(entry) {{
    const date = entry[0], data = entry[1];
    const held = data.rejections > 0;
    return '<div class="session-card">'
      + '<div class="session-date">' + date + '</div>'
      + '<div class="session-result ' + (held ? 'held' : 'broken') + '">' + (held ? 'Held/Rejected' : 'Tested') + '</div>'
      + '<div class="session-touches">' + data.touches.length + ' touch' + (data.touches.length !== 1 ? 'es' : '') + ' · ' + data.rejections + ' rej</div>'
      + '</div>';
  }}).join('');

  // Touch table rows
  const touchRows = allTouches.slice().reverse().map(function(t) {{
    const isBreakout = t.reaction === 'breakout';
    const sentClass  = t.sentiment === 'bullish' ? 'candle-bull' : t.sentiment === 'bearish' ? 'candle-bear' : 'candle-neutral';
    const reacClass  = isBreakout               ? 'reaction-breakout'  :
                       t.reaction === 'rejection' ? 'reaction-rejection' :
                       t.reaction === 'inside'    ? 'reaction-inside'    : 'reaction-touch';
    const reacLabel  = isBreakout               ? '⚡ BREAKOUT'  :
                       t.reaction === 'rejection' ? '✓ HELD'       :
                       t.reaction === 'inside'    ? 'INSIDE'        : 'TOUCH';
    const moveSign   = t.move_pts > 0 ? '+' : '';
    const clsColor   = {{'bullish':'candle-bull','bearish':'candle-bear'}}[t.sentiment] || 'candle-neutral';
    const rowClass   = isBreakout ? 'breakout-row' : '';

    // Pre-candle context sub-row (only for breakouts)
    let prevRow = '';
    if (isBreakout && t.prev_candles && t.prev_candles.length > 0) {{
      const chips = t.prev_candles.map(function(p) {{
        const ps = p.sentiment === 'bullish' ? 'candle-bull' : p.sentiment === 'bearish' ? 'candle-bear' : 'candle-neutral';
        const pm = p.move_pts > 0 ? '+' : '';
        return '<span class="prev-candle-chip">'
          + svgMiniCandle(p.open, p.high, p.low, p.close)
          + '<span class="' + ps + '" style="font-size:8px">' + p.candle + ' ' + pm + p.move_pts.toFixed(1) + '</span>'
          + '</span>';
      }}).join('');
      prevRow = '<tr class="prev-candles-row">'
        + '<td colspan="2" style="color:var(--text3);font-size:9px;letter-spacing:1px;padding-left:20px">↳ PRE-CANDLES</td>'
        + '<td colspan="7">' + chips + '</td>'
        + '</tr>';
    }}

    return '<tr class="' + rowClass + '">'
      + '<td style="color:var(--text2)">' + t.datetime.slice(0,16) + '</td>'
      + '<td style="text-align:center">' + svgMiniCandle(t.open, t.high, t.low, t.close) + '</td>'
      + '<td>' + t.open.toFixed(1) + '</td>'
      + '<td style="color:var(--bull)">' + t.high.toFixed(1) + '</td>'
      + '<td style="color:var(--bear)">' + t.low.toFixed(1) + '</td>'
      + '<td class="' + clsColor + '">' + t.close.toFixed(1) + '</td>'
      + '<td class="' + clsColor + '">' + moveSign + t.move_pts.toFixed(1) + '</td>'
      + '<td class="' + sentClass + '">' + t.candle + '</td>'
      + '<td class="' + reacClass + '">' + reacLabel + '</td>'
      + '</tr>'
      + prevRow;
  }}).join('');

  cont.innerHTML =
    // ── ZONE HEADER
    '<div class="detail-zone-header">'
    + '<div>'
    + '<div class="detail-zone-price">' + z.price.toLocaleString('en-IN', {{minimumFractionDigits:2}}) + '</div>'
    + '<div class="detail-zone-type ' + cls + '">' + label + '</div>'
    + '<div class="detail-zone-range">' + z.lower.toFixed(0) + ' – ' + z.upper.toFixed(0) + ' (' + z.band_pts + 'pt band) · ' + Math.abs(z.dist_pts).toFixed(0) + ' pts from CMP</div>'
    + '<div style="margin-top:6px;font-size:10px;color:var(--text2)">Last tested: ' + z.last_touch.split('T')[0] + ' · Convergence: ' + z.convergence + ' pivots</div>'
    + '</div>'
    + '<div style="text-align:right">'
    + '<div class="strength-circle ' + cls + '"><div class="strength-num">' + z.strength.toFixed(0) + '</div><div class="strength-label">STR</div></div>'
    + '<div style="font-size:10px;color:var(--text2);text-align:center">' + strLabel + '</div>'
    + '</div>'
    + '</div>'

    // ── STATS GRID
    + '<div class="stats-grid">'
    + '<div class="stat-card"><div class="stat-val">' + z.wick_touches + '</div><div class="stat-lbl">Wick Touches</div></div>'
    + '<div class="stat-card"><div class="stat-val">' + z.body_rejections + '</div><div class="stat-lbl">Rejections</div></div>'
    + '<div class="stat-card"><div class="stat-val">' + z.sessions + '</div><div class="stat-lbl">Sessions Hit</div></div>'
    + '<div class="stat-card"><div class="stat-val">' + z.holds + '</div><div class="stat-lbl">Times Held</div></div>'
    + '<div class="stat-card"><div class="stat-val">' + z.inside_bars + '</div><div class="stat-lbl">Inside Bars</div></div>'
    + '<div class="stat-card"><div class="stat-val">' + z.recency_score + '%</div><div class="stat-lbl">Recency</div></div>'
    + '<div class="stat-card"><div class="stat-val">' + z.n_pivots + '</div><div class="stat-lbl">Pivot Hits</div></div>'
    + '<div class="stat-card"><div class="stat-val">' + z.spread_pts + '</div><div class="stat-lbl">Spread pts</div></div>'
    + '</div>'

    // ── REACTION + SENTIMENT DISTRIBUTION
    + '<div class="section-hdr">Reaction &amp; Candle Analysis</div>'
    + '<div class="distr-grid">'
    + '<div class="distr-card"><div class="distr-title">Reaction Breakdown (' + totalT + ' touches)</div>' + reactionRows + '</div>'
    + '<div class="distr-card"><div class="distr-title">Candle Sentiment</div>' + sentRows + '</div>'
    + '</div>'

    // ── TOP CANDLE FORMATIONS
    + '<div class="section-hdr">Candle Formation Frequency</div>'
    + '<div class="distr-card" style="margin-bottom:20px">' + candleRows + '</div>'

    // ── MINI CHART
    + '<div class="section-hdr">Price Action Near Zone</div>'
    + '<div class="chart-wrap">'
    + '<canvas id="miniChart" height="190"></canvas>'
    + '<div class="chart-legend">'
    + '<span><span class="leg-tri"></span>Held (Rejection)</span>'
    + '<span><span class="leg-diamond"></span>Breakout</span>'
    + '<span><span class="leg-dot" style="background:var(--accent)"></span>Touch</span>'
    + '<span><span class="leg-dot" style="background:var(--purple)"></span>Inside</span>'
    + '<span><span class="leg-dot" style="background:rgba(0,229,160,0.5);width:20px;border-radius:0;height:3px"></span>Zone band</span>'
    + '<span><span class="leg-dot" style="background:var(--gold);width:20px;border-radius:0;height:1px"></span>CMP</span>'
    + '</div>'
    + '</div>'

    // ── SESSION HISTORY
    + '<div class="section-hdr" style="margin-top:20px">Session History (' + sessions.length + ' days)</div>'
    + '<div class="session-grid">' + sessionCards + '</div>'

    // ── TOUCH TABLE
    + '<div class="section-hdr">All Touch Events (' + allTouches.length + ' candles)</div>'
    + '<div class="touch-table-wrap"><table class="touch-table">'
    + '<thead><tr>'
    + '<th>Date/Time</th><th>Viz</th><th>O</th><th>H</th><th>L</th><th>C</th><th>Move</th><th>Candle</th><th>Reaction</th>'
    + '</tr></thead>'
    + '<tbody>' + touchRows + '</tbody>'
    + '</table></div>';

  drawMiniChart(z);
  cont.scrollTop = 0;
}}

// ── MINI CANDLESTICK CHART ────────────────────────────────────
function drawMiniChart(z) {{
  const canvas = document.getElementById('miniChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const data = OHLC;
  if (data.length === 0) return;

  // Fixed candle width so all bars are readable regardless of dataset size
  const BAR_W   = 3;
  const BAR_GAP = 1;
  const H   = 220;
  const pad = {{l:48, r:60, t:18, b:28}};
  const W   = pad.l + data.length * (BAR_W + BAR_GAP) + pad.r;

  canvas.width        = W;
  canvas.height       = H;
  canvas.style.height = H + 'px';

  const isSup = z.type === 'Support';

  let pMin = Math.min(...data.map(d => d.l));
  let pMax = Math.max(...data.map(d => d.h));
  const pRange = pMax - pMin;
  pMin -= pRange * 0.03;
  pMax += pRange * 0.03;

  const toY  = p => pad.t + (pMax - p) / (pMax - pMin) * (H - pad.t - pad.b);
  const barX = i => pad.l + i * (BAR_W + BAR_GAP);

  // Background
  ctx.fillStyle = '#0a0e14';
  ctx.fillRect(0, 0, W, H);

  // Horizontal price grid + sticky Y-axis labels on left
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {{
    const y     = pad.t + i * (H - pad.t - pad.b) / 5;
    const price = pMax  - i * (pMax - pMin) / 5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = 'rgba(120,140,170,0.7)';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(price.toFixed(0), pad.l - 4, y + 3);
    // Right-side label too
    ctx.textAlign = 'left';
    ctx.fillText(price.toFixed(0), W - pad.r + 4, y + 3);
  }}
  ctx.textAlign = 'left';

  // Day separator lines + date labels at bottom
  let prevDate = null;
  data.forEach((d, i) => {{
    const dayStr = d.t.slice(0, 10);
    if (dayStr !== prevDate) {{
      prevDate = dayStr;
      const x = barX(i);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
      // Date label — show Mon DD
      const dt = new Date(d.t);
      const label = dt.toLocaleDateString('en-IN', {{month:'short', day:'numeric'}});
      ctx.fillStyle = 'rgba(100,130,160,0.7)';
      ctx.font = '8px monospace';
      ctx.fillText(label, x + 1, H - 6);
    }}
  }});

  // Zone band across full chart width
  const zy1 = toY(z.upper);
  const zy2 = toY(z.lower);
  ctx.fillStyle = isSup ? 'rgba(0,229,160,0.07)' : 'rgba(255,77,109,0.07)';
  ctx.fillRect(pad.l, zy1, W - pad.l - pad.r, zy2 - zy1);

  // Zone upper + lower boundary lines
  ctx.strokeStyle = isSup ? 'rgba(0,229,160,0.45)' : 'rgba(255,77,109,0.45)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, zy1); ctx.lineTo(W - pad.r, zy1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l, zy2); ctx.lineTo(W - pad.r, zy2); ctx.stroke();

  // Zone mid-price dashed
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = isSup ? 'rgba(0,229,160,0.35)' : 'rgba(255,77,109,0.35)';
  ctx.beginPath(); ctx.moveTo(pad.l, toY(z.price)); ctx.lineTo(W - pad.r, toY(z.price)); ctx.stroke();
  ctx.setLineDash([]);

  // Zone price label pinned to right edge
  const zoneY = toY(z.price);
  ctx.fillStyle = isSup ? 'rgba(0,229,160,0.9)' : 'rgba(255,77,109,0.9)';
  ctx.font = 'bold 9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText((isSup ? 'S ' : 'R ') + z.price.toFixed(0), W - pad.r + 4, zoneY - 1);
  // Upper / lower labels
  ctx.font = '8px monospace';
  ctx.fillStyle = isSup ? 'rgba(0,229,160,0.6)' : 'rgba(255,77,109,0.6)';
  ctx.fillText(z.upper.toFixed(0), W - pad.r + 4, zy1 + 3);
  ctx.fillText(z.lower.toFixed(0), W - pad.r + 4, zy2 + 3);

  // Build touch map for marker overlay
  const touchMap = {{}};
  (z.touch_details || []).forEach(t => {{
    touchMap[t.datetime.slice(0, 16)] = t;
  }});

  // Draw candles
  data.forEach((d, i) => {{
    const x    = barX(i);
    const yO   = toY(d.o);
    const yC   = toY(d.c);
    const yH   = toY(d.h);
    const yL   = toY(d.l);
    const bull = d.c >= d.o;
    const col  = bull ? '#26a69a' : '#ef5350';
    const cx   = x + BAR_W / 2;

    // Wick
    ctx.strokeStyle = col; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, yH); ctx.lineTo(cx, yL); ctx.stroke();

    // Body
    ctx.fillStyle = bull ? 'rgba(38,166,154,0.85)' : 'rgba(239,83,80,0.85)';
    const top = Math.min(yO, yC);
    const ht  = Math.max(Math.abs(yC - yO), 1);
    ctx.fillRect(x, top, BAR_W, ht);
  }});

  // Touch markers on top of candles
  data.forEach((d, i) => {{
    const t = touchMap[d.t.slice(0, 16)];
    if (!t) return;
    const cx   = barX(i) + BAR_W / 2;
    const bull = d.c >= d.o;

    if (t.reaction === 'breakout') {{
      // Red diamond — zone was broken
      const by = bull ? toY(d.h) - 9 : toY(d.l) + 9;
      const s  = 5;
      ctx.beginPath();
      ctx.moveTo(cx,     by - s);
      ctx.lineTo(cx + s, by    );
      ctx.lineTo(cx,     by + s);
      ctx.lineTo(cx - s, by    );
      ctx.closePath();
      ctx.fillStyle = 'rgba(255,60,80,0.95)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }} else if (t.reaction === 'rejection') {{
      // Gold triangle — zone held
      const ty = bull ? toY(d.h) - 7 : toY(d.l) + 7;
      const s  = 4;
      ctx.beginPath();
      if (bull) {{
        ctx.moveTo(cx, ty - s); ctx.lineTo(cx - s, ty + s); ctx.lineTo(cx + s, ty + s);
      }} else {{
        ctx.moveTo(cx, ty + s); ctx.lineTo(cx - s, ty - s); ctx.lineTo(cx + s, ty - s);
      }}
      ctx.closePath();
      ctx.fillStyle = 'rgba(255,210,77,0.95)';
      ctx.fill();
    }} else {{
      // Circle — touch or inside
      const cy = bull ? toY(d.h) - 5 : toY(d.l) + 5;
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = t.reaction === 'inside' ? 'rgba(155,127,255,0.9)' : 'rgba(61,143,255,0.85)';
      ctx.fill();
    }}
  }});

  // CMP dashed line
  const cmpY = toY(CMP);
  ctx.strokeStyle = 'rgba(255,210,77,0.65)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(pad.l, cmpY); ctx.lineTo(W - pad.r, cmpY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(255,210,77,0.85)'; ctx.font = 'bold 9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('CMP ' + CMP.toFixed(0), W - pad.r + 4, cmpY + 3);

  // Auto-scroll the container to the rightmost (latest) candle
  const wrap = canvas.parentElement;
  wrap.scrollLeft = wrap.scrollWidth;
}}

// ── FILTER ───────────────────────────────────────────────────
function filterZones(f, btn) {{
  activeFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  buildZoneList(f);
}}

// ── INIT ─────────────────────────────────────────────────────
buildZoneList('all');
window.addEventListener('load', () => {{
  if (ZONES.length > 0) showDetail(ZONES[0]);
}});
window.addEventListener('resize', () => {{
  const active = document.querySelector('.zone-card.active');
  if (active) {{
    const z = ZONES.find(z => z.price === parseFloat(active.dataset.price));
    if (z) drawMiniChart(z);
  }}
}});
</script>
</body>
</html>"""
    return html


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TICKER            = "^NSEI"
    INTERVAL          = "5m"
    DAYS              = 55
    CHUNK_DAYS        = 7
    LEFT_BARS         = 10
    RIGHT_BARS        = 10
    CLUSTER_TOLERANCE = 15.0
    ZONE_HALF_BAND    = 15.0
    MIN_WICK_TOUCHES  = 3
    MIN_SESSIONS      = 2
    MIN_REJECTIONS    = 1
    TOP_N             = 20

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

    print("\nGenerating interactive HTML dashboard...")
    html = generate_html(zones, current_price, TICKER, df)

    out_path = "nifty_sr_zones.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✓ Saved: {out_path}")
    print(f"  Open in browser → {len(zones)} zones, click any to see full touch history")
