"""
NIFTY 50 — Wick Zone Detector  (Horizontal + Slanting)
=======================================================
Detects:
  1. HORIZONTAL S/R zones from clusters of long wicks  (original, unchanged)
  2. SLANTING trendline zones from the same wick-tip clusters
        → fits trend lines through upper / lower wick tips,
          draws them as slanted parallelogram bands (like Kite trendline tools)

Usage
-----
# Static chart from CSV:
    python wick_zone_detector.py --file NIFTY_50_5minute.csv --lookback 300 --display 75

# Real-time mode (fetches live NIFTY data using yfinance):
    python wick_zone_detector.py --realtime --interval 5m --lookback 300 --display 75

# Save chart image instead of showing:
    python wick_zone_detector.py --file NIFTY_50_5minute.csv --save chart.png

# Disable slanting zones:
    python wick_zone_detector.py --file NIFTY_50_5minute.csv --no_slant

Requirements
------------
    pip install pandas numpy matplotlib mplfinance yfinance scipy
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import mplfinance as mpf
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION  (tweak these to match your preference)
# ─────────────────────────────────────────────────────────────
class Config:
    # ── Zone detection ──────────────────────────────────────
    WICK_MIN_PCT    = 0.30    # wick must be ≥ 30 % of the candle range
    BODY_MAX_PCT    = 0.70    # body must be ≤ 70 % of range (ensures visible wick)
    MIN_WICK_POINTS = 4       # absolute minimum wick size in index points
    ZONE_WIDTH_PCT  = 0.0035  # price tolerance to cluster tips (0.35 %)
    MIN_TOUCHES     = 3       # minimum wicks in a cluster to form a zone
    ZONE_BUFFER     = 2       # extra points padding on zone edges
    PROXIMITY_PCT   = 0.06    # only zones within 6 % of current price
    LOOKBACK        = 300     # candles used to detect zones

    # ── Display ─────────────────────────────────────────────
    DISPLAY_CANDLES = 75      # candles shown on the chart
    MAX_ZONES_UPPER = 3       # max resistance zones drawn
    MAX_ZONES_LOWER = 3       # max support zones drawn
    ZONE_EXTEND_RIGHT = True  # extend zone box to the right edge of chart

    # ── Zone box appearance (like Kite purple boxes) ─────────
    UPPER_COLOR     = "#B44FCC"   # resistance = purple
    LOWER_COLOR     = "#5B4FCC"   # support = purple-blue
    BOX_ALPHA       = 0.18        # fill transparency
    BOX_EDGE_ALPHA  = 0.70        # border transparency
    BOX_LINEWIDTH   = 1.2

    # ── Slanting trendline zone config ───────────────────────
    SLANT_ENABLED         = True
    SLANT_MIN_TOUCHES     = 3       # minimum wick tips to form a trendline
    SLANT_MIN_SPAN        = 10      # minimum candle span between first & last touch
    SLANT_MAX_RSQUARED_THRESH = 0.55  # R² must be ≥ this (linearity check)
    SLANT_PROXIMITY_PCT   = 0.08    # up to 8 % away from current price
    SLANT_BAND_SIGMA      = 1.0     # band half-width = sigma * std(residuals)
    SLANT_EXTEND_RIGHT    = 15      # extend trendline N candles past last touch
    SLANT_EXTEND_LEFT     = 0       # extend trendline N candles before first touch
    MAX_SLANT_UPPER       = 2       # max resistance trendlines drawn
    MAX_SLANT_LOWER       = 2       # max support trendlines drawn

    # ── Slanting zone appearance ─────────────────────────────
    SLANT_UPPER_COLOR     = "#FF8C42"   # orange for descending resistance
    SLANT_LOWER_COLOR     = "#42C5FF"   # cyan for ascending support
    SLANT_ALPHA           = 0.15
    SLANT_EDGE_ALPHA      = 0.75
    SLANT_LINE_WIDTH      = 1.5
    SLANT_LINE_STYLE      = "--"        # dashed trendline spine

    # ── Chart style ──────────────────────────────────────────
    BULL_COLOR      = "#26a69a"   # Kite-style green
    BEAR_COLOR      = "#ef5350"   # Kite-style red
    BG_COLOR        = "#131722"   # dark background
    GRID_COLOR      = "#1e222d"
    TEXT_COLOR      = "#d1d4dc"
    EMA_COLORS      = ["#2196F3", "#FF5722"]  # EMA 9, EMA 21


# ─────────────────────────────────────────────────────────────
# ZONE DETECTION ENGINE  (original – unchanged)
# ─────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add body, wick, and EMA columns."""
    df = df.copy()
    df["body"]        = abs(df["close"] - df["open"])
    df["candle_range"]= df["high"] - df["low"]
    df["upper_wick"]  = df["high"]  - df[["open","close"]].max(axis=1)
    df["lower_wick"]  = df[["open","close"]].min(axis=1) - df["low"]
    df["ema9"]        = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"]       = df["close"].ewm(span=21, adjust=False).mean()
    return df


def _collect_wick_tips(data: pd.DataFrame, cfg: Config):
    """
    Return two lists: upper_tips, lower_tips.
    Each entry: dict with idx, date, tip, junction, wick_size, wick_pct, recency.
    """
    n = len(data)
    upper_tips, lower_tips = [], []

    for i, row in data.iterrows():
        cr = row["candle_range"]
        if cr < 2:
            continue

        uw_pct   = row["upper_wick"] / cr
        lw_pct   = row["lower_wick"] / cr
        body_pct = row["body"] / (cr + 0.01)
        recency  = (i + 1) / n   # 0 → 1, higher = more recent

        if (uw_pct   >= cfg.WICK_MIN_PCT
                and body_pct <= cfg.BODY_MAX_PCT
                and row["upper_wick"] >= cfg.MIN_WICK_POINTS):
            upper_tips.append({
                "idx": i, "date": row["date"],
                "tip": row["high"],
                "junction": max(row["open"], row["close"]),
                "wick_size": row["upper_wick"],
                "wick_pct": uw_pct, "recency": recency
            })

        if (lw_pct   >= cfg.WICK_MIN_PCT
                and body_pct <= cfg.BODY_MAX_PCT
                and row["lower_wick"] >= cfg.MIN_WICK_POINTS):
            lower_tips.append({
                "idx": i, "date": row["date"],
                "tip": row["low"],
                "junction": min(row["open"], row["close"]),
                "wick_size": row["lower_wick"],
                "wick_pct": lw_pct, "recency": recency
            })

    return upper_tips, lower_tips


def _cluster_tips(tips: list, side: str, data: pd.DataFrame, cfg: Config):
    """
    Cluster nearby wick tips into zones, score them, filter, and return
    a list of zone dicts sorted by relevance score.
    """
    if not tips:
        return []

    current_px = data["close"].iloc[-1]
    n          = len(data)

    # ── 1. Build clusters (greedy, price-sorted) ────────────
    tips_sorted = sorted(tips, key=lambda x: x["tip"])
    used        = [False] * len(tips_sorted)
    clusters    = []

    for i, t1 in enumerate(tips_sorted):
        if used[i]:
            continue
        cluster  = [t1]
        used[i]  = True
        ref      = t1["tip"]

        for j, t2 in enumerate(tips_sorted):
            if used[j] or j == i:
                continue
            if abs(t2["tip"] - ref) / (ref + 0.01) <= cfg.ZONE_WIDTH_PCT:
                cluster.append(t2)
                used[j] = True

        clusters.append(cluster)

    # ── 2. Convert clusters → zone dicts ────────────────────
    zones = []
    for cluster in clusters:
        if len(cluster) < cfg.MIN_TOUCHES:
            continue

        tip_vals  = [c["tip"]      for c in cluster]
        junctions = [c["junction"] for c in cluster]
        dates     = [c["date"]     for c in cluster]
        idxs      = [c["idx"]      for c in cluster]

        if side == "upper":
            zone_low  = min(junctions) - cfg.ZONE_BUFFER
            zone_high = max(tip_vals)  + cfg.ZONE_BUFFER
        else:
            zone_low  = min(tip_vals)  - cfg.ZONE_BUFFER
            zone_high = max(junctions) + cfg.ZONE_BUFFER

        zone_center = (zone_low + zone_high) / 2

        # ── Proximity filter ────────────────────────────────
        if abs(zone_center - current_px) / current_px > cfg.PROXIMITY_PCT:
            continue

        # ── Broken zone check ────────────────────────────────
        last_touch_date = max(dates)
        post = data[data["date"] > last_touch_date]

        if side == "upper":
            broken = (len(post) > 0 and
                      post["close"].max() > zone_high + cfg.ZONE_BUFFER * 2)
        else:
            broken = (len(post) > 0 and
                      post["close"].min() < zone_low - cfg.ZONE_BUFFER * 2)

        # ── Relevance score ──────────────────────────────────
        avg_recency  = np.mean([c["recency"]   for c in cluster])
        avg_wick     = np.mean([c["wick_size"]  for c in cluster])
        recency_bonus = 1.5 if avg_recency > 0.7 else 1.0
        broken_penalty = 0.5 if broken else 1.0

        score = (len(cluster)
                 * recency_bonus
                 * broken_penalty
                 * (1 + avg_wick / 50))

        zones.append({
            "side":             side,
            "zone_low":         round(zone_low,  2),
            "zone_high":        round(zone_high, 2),
            "center":           zone_center,
            "touches":          len(cluster),
            "score":            score,
            "avg_recency":      avg_recency,
            "max_wick":         max(c["wick_size"] for c in cluster),
            "first_touch_date": min(dates),
            "last_touch_date":  max(dates),
            "first_touch_idx":  min(idxs),
            "last_touch_idx":   max(idxs),
            "broken":           broken,
        })

    # ── 3. Merge overlapping zones ──────────────────────────
    zones.sort(key=lambda x: x["score"], reverse=True)
    merged = []
    used_z = [False] * len(zones)

    for i, z1 in enumerate(zones):
        if used_z[i]:
            continue
        group    = [z1]
        used_z[i] = True

        for j, z2 in enumerate(zones):
            if used_z[j] or j == i:
                continue
            if z1["zone_low"] <= z2["zone_high"] and z2["zone_low"] <= z1["zone_high"]:
                group.append(z2)
                used_z[j] = True

        if len(group) == 1:
            merged.append(z1)
        else:
            merged.append({
                "side":             side,
                "zone_low":         min(g["zone_low"]         for g in group),
                "zone_high":        max(g["zone_high"]        for g in group),
                "center":           np.mean([g["center"]       for g in group]),
                "touches":          sum(g["touches"]          for g in group),
                "score":            max(g["score"]            for g in group),
                "avg_recency":      max(g["avg_recency"]      for g in group),
                "max_wick":         max(g["max_wick"]         for g in group),
                "first_touch_date": min(g["first_touch_date"] for g in group),
                "last_touch_date":  max(g["last_touch_date"]  for g in group),
                "first_touch_idx":  min(g["first_touch_idx"]  for g in group),
                "last_touch_idx":   max(g["last_touch_idx"]   for g in group),
                "broken":           any(g["broken"]           for g in group),
            })

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


def detect_wick_zones(df: pd.DataFrame, cfg: Config = None):
    """
    Main entry point for horizontal zones.
    Returns (upper_zones, lower_zones).
    Also returns all raw wick tips for the slanting engine.
    """
    if cfg is None:
        cfg = Config()

    data = df.tail(cfg.LOOKBACK).copy().reset_index(drop=True)
    data = compute_indicators(data)

    upper_tips, lower_tips = _collect_wick_tips(data, cfg)
    upper_zones = _cluster_tips(upper_tips, "upper", data, cfg)
    lower_zones = _cluster_tips(lower_tips, "lower", data, cfg)

    return upper_zones[:cfg.MAX_ZONES_UPPER], lower_zones[:cfg.MAX_ZONES_LOWER]


def _collect_all_wick_tips(df: pd.DataFrame, cfg: Config):
    """Collect all wick tips from the lookback window (no clustering)."""
    data = df.tail(cfg.LOOKBACK).copy().reset_index(drop=True)
    data = compute_indicators(data)
    upper_tips, lower_tips = _collect_wick_tips(data, cfg)
    return upper_tips, lower_tips, data


# ─────────────────────────────────────────────────────────────
# SLANTING TRENDLINE ZONE ENGINE  (new logic)
# ─────────────────────────────────────────────────────────────

def _fit_trendline(tips: list, side: str):
    """
    Fit a linear regression through wick tip positions.
    Returns (slope, intercept, r_squared, residual_std).
    """
    xs = np.array([t["idx"] for t in tips], dtype=float)
    ys = np.array([t["tip"] for t in tips], dtype=float)

    if len(xs) < 2:
        return None

    # Linear regression
    A = np.vstack([xs, np.ones(len(xs))]).T
    result = np.linalg.lstsq(A, ys, rcond=None)
    slope, intercept = result[0]

    # R² calculation
    y_pred = slope * xs + intercept
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r_sq   = 1 - ss_res / (ss_tot + 1e-10)

    residuals = ys - y_pred
    res_std   = np.std(residuals)

    return {
        "slope":     slope,
        "intercept": intercept,
        "r_squared": r_sq,
        "res_std":   res_std,
        "xs":        xs,
        "ys":        ys,
    }


def _score_trendline(fit: dict, tips: list, current_px: float,
                     side: str, n_data: int, cfg: Config):
    """
    Compute a relevance score for a candidate trendline.
    """
    n_touches  = len(tips)
    r_sq       = max(0, fit["r_squared"])
    span       = fit["xs"].max() - fit["xs"].min()

    # Recency: position of last touch in lookback window
    last_idx   = fit["xs"].max()
    recency    = last_idx / (n_data + 1)

    # Midpoint price of line at the last touch
    mid_price  = fit["slope"] * last_idx + fit["intercept"]
    proximity  = abs(mid_price - current_px) / (current_px + 1e-6)

    if proximity > cfg.SLANT_PROXIMITY_PCT:
        return -1  # out of proximity → discard

    recency_bonus = 1.5 if recency > 0.7 else 1.0
    span_bonus    = min(span / 50, 2.0)  # longer spans get slightly more credit

    score = (n_touches * r_sq * recency_bonus * span_bonus
             * (1 - proximity / cfg.SLANT_PROXIMITY_PCT))
    return score


def _generate_trendline_candidates(tips: list, side: str,
                                   current_px: float, n_data: int,
                                   cfg: Config):
    """
    Generate trendline candidates from all subsets of wick tips.
    Strategy:
      - Sort tips by candle index
      - Use a sliding anchor approach: for each tip as anchor, fit through
        subsequent tips that stay on the correct side of the line
      - Score and filter candidates
    """
    if len(tips) < cfg.SLANT_MIN_TOUCHES:
        return []

    # Sort by candle index
    tips_sorted = sorted(tips, key=lambda t: t["idx"])
    n           = len(tips_sorted)
    candidates  = []

    for anchor_i in range(n - cfg.SLANT_MIN_TOUCHES + 1):
        for end_i in range(anchor_i + cfg.SLANT_MIN_TOUCHES - 1, n):
            subset = tips_sorted[anchor_i: end_i + 1]

            # Span check
            span = subset[-1]["idx"] - subset[0]["idx"]
            if span < cfg.SLANT_MIN_SPAN:
                continue

            fit = _fit_trendline(subset, side)
            if fit is None or fit["r_squared"] < cfg.SLANT_MAX_RSQUARED_THRESH:
                continue

            # Direction sanity check:
            # upper (resistance) trendlines are typically flat or descending
            # lower (support) trendlines are typically flat or ascending
            # We allow both directions but give a slight preference check later
            score = _score_trendline(fit, subset, current_px, side, n_data, cfg)
            if score < 0:
                continue

            candidates.append({
                "side":             side,
                "fit":              fit,
                "tips":             subset,
                "n_touches":        len(subset),
                "score":            score,
                "r_squared":        fit["r_squared"],
                "slope":            fit["slope"],
                "intercept":        fit["intercept"],
                "res_std":          fit["res_std"],
                "first_idx":        subset[0]["idx"],
                "last_idx":         subset[-1]["idx"],
                "first_date":       subset[0]["date"],
                "last_date":        subset[-1]["date"],
            })

    return candidates


def _deduplicate_trendlines(candidates: list, cfg: Config):
    """
    Remove near-duplicate trendlines:
    two trendlines are duplicates if they share > 60 % of the same tip indices.
    Keep the higher-scoring one.
    """
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept       = []
    used       = [False] * len(candidates)

    for i, c1 in enumerate(candidates):
        if used[i]:
            continue
        kept.append(c1)
        used[i] = True
        idxs1   = set(int(x) for x in c1["fit"]["xs"])

        for j, c2 in enumerate(candidates):
            if used[j] or j == i:
                continue
            idxs2    = set(int(x) for x in c2["fit"]["xs"])
            overlap  = len(idxs1 & idxs2) / max(len(idxs1 | idxs2), 1)
            if overlap > 0.60:
                used[j] = True

    return kept


def detect_slant_zones(df: pd.DataFrame, cfg: Config = None):
    """
    Detect slanting trendline zones.
    Returns (upper_slant_zones, lower_slant_zones).

    Each zone dict contains:
        side, slope, intercept, res_std, first_idx, last_idx,
        n_touches, score, r_squared, broken, direction
    """
    if cfg is None:
        cfg = Config()

    upper_tips, lower_tips, data = _collect_all_wick_tips(df, cfg)
    current_px = data["close"].iloc[-1]
    n_data     = len(data)

    # ── Upper (resistance) trendlines ───────────────────────
    upper_cands = _generate_trendline_candidates(
        upper_tips, "upper", current_px, n_data, cfg)
    upper_cands = _deduplicate_trendlines(upper_cands, cfg)

    # ── Lower (support) trendlines ──────────────────────────
    lower_cands = _generate_trendline_candidates(
        lower_tips, "lower", current_px, n_data, cfg)
    lower_cands = _deduplicate_trendlines(lower_cands, cfg)

    # ── Broken check for slanting zones ─────────────────────
    def check_broken_slant(zone, data, side):
        post = data[data.index > zone["last_idx"]]
        if len(post) == 0:
            return False
        for i_post, row in post.iterrows():
            line_val = zone["slope"] * i_post + zone["intercept"]
            band     = cfg.SLANT_BAND_SIGMA * zone["res_std"] + cfg.ZONE_BUFFER
            if side == "upper" and row["close"] > line_val + band * 2:
                return True
            if side == "lower" and row["close"] < line_val - band * 2:
                return True
        return False

    # ── Annotate and sort ────────────────────────────────────
    def annotate(zones, side):
        result = []
        for z in zones:
            z["broken"]    = check_broken_slant(z, data, side)
            z["direction"] = ("descending" if z["slope"] < -0.5
                              else "ascending" if z["slope"] > 0.5
                              else "horizontal")
            if z["broken"]:
                z["score"] *= 0.5
            result.append(z)
        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    upper_zones = annotate(upper_cands, "upper")[:cfg.MAX_SLANT_UPPER]
    lower_zones = annotate(lower_cands, "lower")[:cfg.MAX_SLANT_LOWER]

    return upper_zones, lower_zones, data


# ─────────────────────────────────────────────────────────────
# CHART RENDERING
# ─────────────────────────────────────────────────────────────

def plot_wick_zones(df: pd.DataFrame, upper_zones: list, lower_zones: list,
                    upper_slant: list = None, lower_slant: list = None,
                    cfg: Config = None, title: str = "NIFTY 50 – Wick S/R Zones",
                    save_path: str = None):
    """
    Draw a dark-themed candlestick chart with:
      - EMA 9 and EMA 21
      - Upper (resistance) zones in purple      [horizontal]
      - Lower (support) zones in blue-purple    [horizontal]
      - Upper slanting trendline bands in orange [slanting]
      - Lower slanting trendline bands in cyan   [slanting]
    """
    if cfg is None:
        cfg = Config()
    if upper_slant is None:
        upper_slant = []
    if lower_slant is None:
        lower_slant = []

    # ── Prepare display window ───────────────────────────────
    disp = df.tail(cfg.DISPLAY_CANDLES).copy()
    disp = compute_indicators(disp)

    disp.index = pd.to_datetime(disp["date"])
    disp.index.name = "Date"
    disp.rename(columns={"open": "Open", "high": "High",
                         "low": "Low", "close": "Close",
                         "volume": "Volume"}, inplace=True)

    # ── mplfinance style ─────────────────────────────────────
    mc = mpf.make_marketcolors(
        up=cfg.BULL_COLOR, down=cfg.BEAR_COLOR,
        edge={"up": cfg.BULL_COLOR, "down": cfg.BEAR_COLOR},
        wick={"up": cfg.BULL_COLOR, "down": cfg.BEAR_COLOR},
        volume={"up": cfg.BULL_COLOR, "down": cfg.BEAR_COLOR},
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor=cfg.BG_COLOR,
        edgecolor=cfg.GRID_COLOR,
        figcolor=cfg.BG_COLOR,
        gridcolor=cfg.GRID_COLOR,
        gridstyle="--",
        gridaxis="both",
        y_on_right=True,
    )

    ap = [
        mpf.make_addplot(disp["ema9"],  color=cfg.EMA_COLORS[0], width=1.2, label="EMA 9"),
        mpf.make_addplot(disp["ema21"], color=cfg.EMA_COLORS[1], width=1.2, label="EMA 21"),
    ]

    fig, axes = mpf.plot(
        disp, type="candle", style=style,
        addplot=ap,
        returnfig=True,
        figsize=(16, 8),
        title=dict(title=f"\n{title}", color=cfg.TEXT_COLOR, fontsize=13),
        tight_layout=True,
    )
    ax = axes[0]

    n_display  = len(disp)
    right_edge = n_display + 1

    # ── Index mapping helpers ─────────────────────────────────
    # Detection window is the last LOOKBACK rows of df
    # Display window is the last DISPLAY_CANDLES rows of df
    # We need to map "detection window index" → "display x position"

    lookback_start_in_df = len(df) - cfg.LOOKBACK   # abs index of first detection row
    display_start_in_df  = len(df) - cfg.DISPLAY_CANDLES  # abs index of first display row

    def det_idx_to_disp_x(det_idx: int) -> float:
        """
        Convert a 0-based index within the detection window
        to an x position on the display chart.
        """
        abs_idx  = lookback_start_in_df + det_idx
        disp_x   = abs_idx - display_start_in_df
        return float(disp_x)

    # ── Draw HORIZONTAL zone boxes ───────────────────────────
    def draw_horizontal_zone(zone, color):
        y_lo  = zone["zone_low"]
        y_hi  = zone["zone_high"]
        x_lo  = max(0, det_idx_to_disp_x(zone["first_touch_idx"]))
        x_hi  = (right_edge if cfg.ZONE_EXTEND_RIGHT
                 else min(n_display - 1,
                          det_idx_to_disp_x(zone["last_touch_idx"])))
        width = x_hi - x_lo
        if width <= 0:
            return

        rect = mpatches.FancyBboxPatch(
            (x_lo, y_lo), width, y_hi - y_lo,
            boxstyle="square,pad=0",
            linewidth=cfg.BOX_LINEWIDTH,
            edgecolor=color + hex(int(cfg.BOX_EDGE_ALPHA * 255))[2:].zfill(2),
            facecolor=color + hex(int(cfg.BOX_ALPHA * 255))[2:].zfill(2),
            zorder=2,
        )
        ax.add_patch(rect)

        status = "✗" if zone["broken"] else "✓"
        label  = (f"{zone['zone_low']:.0f}–{zone['zone_high']:.0f}  "
                  f"({zone['touches']}T {status})")
        ax.text(
            x_hi + 0.3, (y_lo + y_hi) / 2,
            label,
            color=color, fontsize=7.5, va="center",
            fontweight="bold", zorder=5,
            bbox=dict(facecolor=cfg.BG_COLOR, alpha=0.7,
                      edgecolor="none", pad=1),
        )

    for z in upper_zones:
        draw_horizontal_zone(z, cfg.UPPER_COLOR)
    for z in lower_zones:
        draw_horizontal_zone(z, cfg.LOWER_COLOR)

    # ── Draw SLANTING trendline zones ────────────────────────
    def draw_slant_zone(zone, color):
        slope     = zone["slope"]
        intercept = zone["intercept"]
        band      = cfg.SLANT_BAND_SIGMA * zone["res_std"] + cfg.ZONE_BUFFER

        # x range in detection-window coords
        x_start_det = max(0, zone["first_idx"] - cfg.SLANT_EXTEND_LEFT)
        x_end_det   = zone["last_idx"]  + cfg.SLANT_EXTEND_RIGHT

        # Convert to display coords
        x_start = det_idx_to_disp_x(x_start_det)
        x_end   = det_idx_to_disp_x(x_end_det)

        # Clip to visible chart area
        x_start = max(-1, x_start)
        x_end   = min(right_edge + cfg.SLANT_EXTEND_RIGHT, x_end)

        if x_end <= x_start:
            return

        # Trendline values at start and end (in detection-window index space)
        y_mid_start = slope * x_start_det + intercept
        y_mid_end   = slope * x_end_det   + intercept

        # Parallelogram corners: upper edge and lower edge
        # Upper edge = line + band, Lower edge = line - band
        corners = np.array([
            [x_start, y_mid_start + band],   # top-left
            [x_end,   y_mid_end   + band],   # top-right
            [x_end,   y_mid_end   - band],   # bottom-right
            [x_start, y_mid_start - band],   # bottom-left
        ])

        # Filled parallelogram
        poly = Polygon(
            corners, closed=True,
            facecolor=color + hex(int(cfg.SLANT_ALPHA * 255))[2:].zfill(2),
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(poly)

        # Draw upper & lower boundary lines
        for sign in [+1, -1]:
            ax.plot(
                [x_start, x_end],
                [y_mid_start + sign * band, y_mid_end + sign * band],
                color=color,
                linewidth=cfg.SLANT_LINE_WIDTH,
                linestyle=cfg.SLANT_LINE_STYLE,
                alpha=cfg.SLANT_EDGE_ALPHA,
                zorder=3,
            )

        # Draw the spine (centre trendline)
        ax.plot(
            [x_start, x_end],
            [y_mid_start, y_mid_end],
            color=color,
            linewidth=0.7,
            linestyle=":",
            alpha=0.5,
            zorder=3,
        )

        # Label at the right end
        y_label   = y_mid_end
        status    = "✗" if zone["broken"] else "✓"
        direction = zone.get("direction", "")
        dir_arrow = "↘" if slope < -0.5 else "↗" if slope > 0.5 else "→"
        label = (f"{dir_arrow} {zone['n_touches']}T  "
                 f"R²={zone['r_squared']:.2f} {status}")
        ax.text(
            x_end + 0.3, y_label,
            label,
            color=color, fontsize=7.0, va="center",
            fontweight="bold", zorder=5,
            bbox=dict(facecolor=cfg.BG_COLOR, alpha=0.7,
                      edgecolor="none", pad=1),
        )

        # Mark the actual wick tips that form this trendline
        for tip in zone["tips"]:
            x_tip = det_idx_to_disp_x(tip["idx"])
            if 0 <= x_tip <= n_display:
                ax.scatter(x_tip, tip["tip"], marker="o",
                           s=20, color=color, alpha=0.8,
                           zorder=6, linewidths=0)

    for z in upper_slant:
        draw_slant_zone(z, cfg.SLANT_UPPER_COLOR)
    for z in lower_slant:
        draw_slant_zone(z, cfg.SLANT_LOWER_COLOR)

    # ── Legend ───────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=cfg.UPPER_COLOR + "50",
                       edgecolor=cfg.UPPER_COLOR, label="Resistance zone (H)"),
        mpatches.Patch(facecolor=cfg.LOWER_COLOR + "50",
                       edgecolor=cfg.LOWER_COLOR, label="Support zone (H)"),
        mpatches.Patch(facecolor=cfg.SLANT_UPPER_COLOR + "40",
                       edgecolor=cfg.SLANT_UPPER_COLOR, label="Resistance trendline (S)"),
        mpatches.Patch(facecolor=cfg.SLANT_LOWER_COLOR + "40",
                       edgecolor=cfg.SLANT_LOWER_COLOR, label="Support trendline (S)"),
        plt.Line2D([0], [0], color=cfg.EMA_COLORS[0], linewidth=1.5, label="EMA 9"),
        plt.Line2D([0], [0], color=cfg.EMA_COLORS[1], linewidth=1.5, label="EMA 21"),
    ]
    ax.legend(handles=handles, loc="upper left",
              facecolor=cfg.BG_COLOR, edgecolor="#444",
              fontsize=7.5, labelcolor=cfg.TEXT_COLOR)

    # ── Print zone summary ───────────────────────────────────
    current_price = df["close"].iloc[-1]
    print(f"\n{'='*62}")
    print(f"  Current price : {current_price:.2f}")
    print(f"  Detection window: last {cfg.LOOKBACK} candles")
    print(f"{'='*62}")

    print(f"\n  ── HORIZONTAL ZONES ──────────────────────────────────────")
    print(f"  RESISTANCE ZONES (upper wicks):")
    for i, z in enumerate(upper_zones, 1):
        st   = "BROKEN" if z["broken"] else "ACTIVE"
        dist = (z["center"] - current_price) / current_price * 100
        print(f"    [{i}] {z['zone_low']:.0f}–{z['zone_high']:.0f}  |  "
              f"{z['touches']} touches  |  {dist:+.2f}%  |  {st}")

    print(f"  SUPPORT ZONES (lower wicks):")
    for i, z in enumerate(lower_zones, 1):
        st   = "BROKEN" if z["broken"] else "ACTIVE"
        dist = (z["center"] - current_price) / current_price * 100
        print(f"    [{i}] {z['zone_low']:.0f}–{z['zone_high']:.0f}  |  "
              f"{z['touches']} touches  |  {dist:+.2f}%  |  {st}")

    print(f"\n  ── SLANTING TRENDLINE ZONES ──────────────────────────────")
    print(f"  RESISTANCE TRENDLINES (upper wick clusters):")
    for i, z in enumerate(upper_slant, 1):
        st    = "BROKEN" if z["broken"] else "ACTIVE"
        slope = z["slope"]
        dir_s = "DESC" if slope < -0.5 else "ASC" if slope > 0.5 else "FLAT"
        # price at last touch
        px_at_end = slope * z["last_idx"] + z["intercept"]
        dist      = (px_at_end - current_price) / current_price * 100
        print(f"    [{i}] slope={slope:+.3f} ({dir_s})  |  "
              f"{z['n_touches']} touches  |  R²={z['r_squared']:.2f}  |  "
              f"{dist:+.2f}%  |  {st}")

    print(f"  SUPPORT TRENDLINES (lower wick clusters):")
    for i, z in enumerate(lower_slant, 1):
        st    = "BROKEN" if z["broken"] else "ACTIVE"
        slope = z["slope"]
        dir_s = "DESC" if slope < -0.5 else "ASC" if slope > 0.5 else "FLAT"
        px_at_end = slope * z["last_idx"] + z["intercept"]
        dist      = (px_at_end - current_price) / current_price * 100
        print(f"    [{i}] slope={slope:+.3f} ({dir_s})  |  "
              f"{z['n_touches']} touches  |  R²={z['r_squared']:.2f}  |  "
              f"{dist:+.2f}%  |  {st}")
    print(f"{'='*62}\n")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=cfg.BG_COLOR)
        print(f"  Chart saved → {save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """Load OHLCV from your CSV (supports dd-mm-yyyy HH:MM format)."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    if date_col:
        df.rename(columns={date_col: "date"}, inplace=True)
        for fmt in ["%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M"]:
            try:
                df["date"] = pd.to_datetime(df["date"], format=fmt)
                break
            except Exception:
                continue
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
    else:
        raise ValueError("No date/time column found in CSV.")

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_realtime(symbol: str = "^NSEI", interval: str = "5m",
                  period: str = "5d") -> pd.DataFrame:
    """Fetch live OHLCV data using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    print(f"  Fetching {symbol} ({interval}, {period}) from Yahoo Finance …")
    ticker = yf.Ticker(symbol)
    raw    = ticker.history(period=period, interval=interval, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No data returned for {symbol}.")

    df = raw.reset_index()
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = next((c for c in df.columns if "datetime" in c or "date" in c), None)
    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Loaded {len(df)} candles  |  "
          f"{df['date'].iloc[0].strftime('%d-%b %H:%M')} → "
          f"{df['date'].iloc[-1].strftime('%d-%b %H:%M')}")
    return df


# ─────────────────────────────────────────────────────────────
# REAL-TIME LOOP
# ─────────────────────────────────────────────────────────────

def run_realtime_loop(cfg: Config, interval_min: int = 5,
                      symbol: str = "^NSEI", refresh_sec: int = 60):
    """Continuously fetch data, detect zones, and update the chart."""
    import time
    matplotlib.use("TkAgg")
    plt.ion()

    period_map = {1: "1d", 5: "5d", 15: "5d", 30: "1mo", 60: "1mo"}
    period     = period_map.get(interval_min, "5d")
    interval   = f"{interval_min}m"

    print(f"\n  Real-time mode: {symbol} | {interval} | refreshing every {refresh_sec}s")
    print("  Press Ctrl+C to stop.\n")

    fig = None
    while True:
        try:
            df = load_realtime(symbol=symbol, interval=interval, period=period)
            upper_h, lower_h = detect_wick_zones(df, cfg)

            upper_s, lower_s = ([], [])
            if cfg.SLANT_ENABLED:
                upper_s, lower_s, _ = detect_slant_zones(df, cfg)

            if fig is not None:
                plt.close(fig)

            fig = plot_wick_zones(
                df, upper_h, lower_h, upper_s, lower_s, cfg,
                title=f"{symbol} ({interval}) — Live Wick Zones",
            )
            plt.pause(0.1)

        except KeyboardInterrupt:
            print("\n  Stopped.")
            break
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(refresh_sec)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NIFTY Wick Zone Detector — Horizontal + Slanting")
    parser.add_argument("--file",        type=str,  help="Path to CSV file")
    parser.add_argument("--realtime",    action="store_true",
                        help="Enable live data mode (uses yfinance)")
    parser.add_argument("--symbol",      type=str,  default="^NSEI")
    parser.add_argument("--interval",    type=str,  default="5m")
    parser.add_argument("--lookback",    type=int,  default=300)
    parser.add_argument("--display",     type=int,  default=75)
    parser.add_argument("--min_touches", type=int,  default=3)
    parser.add_argument("--proximity",   type=float,default=0.06)
    parser.add_argument("--max_zones",   type=int,  default=3)
    parser.add_argument("--save",        type=str,  default=None)
    parser.add_argument("--refresh",     type=int,  default=60)
    # Slanting-specific
    parser.add_argument("--no_slant",    action="store_true",
                        help="Disable slanting trendline zone detection")
    parser.add_argument("--slant_min_touches", type=int,   default=3,
                        help="Min wick touches for slanting zone (default: 3)")
    parser.add_argument("--slant_r2",    type=float, default=0.55,
                        help="Min R² for trendline fit (default: 0.55)")
    parser.add_argument("--slant_proximity", type=float, default=0.08,
                        help="Proximity filter for slant zones (default: 0.08)")
    parser.add_argument("--max_slant",   type=int,  default=2,
                        help="Max slanting zones per side (default: 2)")
    args = parser.parse_args()

    cfg = Config()
    cfg.LOOKBACK         = args.lookback
    cfg.DISPLAY_CANDLES  = args.display
    cfg.MIN_TOUCHES      = args.min_touches
    cfg.PROXIMITY_PCT    = args.proximity
    cfg.MAX_ZONES_UPPER  = args.max_zones
    cfg.MAX_ZONES_LOWER  = args.max_zones

    cfg.SLANT_ENABLED              = not args.no_slant
    cfg.SLANT_MIN_TOUCHES          = args.slant_min_touches
    cfg.SLANT_MAX_RSQUARED_THRESH  = args.slant_r2
    cfg.SLANT_PROXIMITY_PCT        = args.slant_proximity
    cfg.MAX_SLANT_UPPER            = args.max_slant
    cfg.MAX_SLANT_LOWER            = args.max_slant

    if args.realtime:
        interval_min = int(args.interval.replace("m","").replace("h","60"))
        run_realtime_loop(cfg, interval_min=interval_min,
                          symbol=args.symbol, refresh_sec=args.refresh)

    elif args.file:
        print(f"\n  Loading: {args.file}")
        df = load_csv(args.file)
        print(f"  Rows: {len(df)}  |  "
              f"{df['date'].iloc[0].strftime('%d-%b-%Y')} → "
              f"{df['date'].iloc[-1].strftime('%d-%b-%Y')}")

        upper_h, lower_h = detect_wick_zones(df, cfg)

        upper_s, lower_s = [], []
        if cfg.SLANT_ENABLED:
            print("\n  Detecting slanting zones …")
            upper_s, lower_s, _ = detect_slant_zones(df, cfg)

        plot_wick_zones(df, upper_h, lower_h, upper_s, lower_s, cfg,
                        title="NIFTY 50  5 min  —  Wick S/R Zones (H + S)",
                        save_path=args.save)
    else:
        parser.print_help()


# ─────────────────────────────────────────────────────────────
# PROGRAMMATIC API
# ─────────────────────────────────────────────────────────────

def get_zones_from_df(df: pd.DataFrame,
                      lookback: int = 300,
                      min_touches: int = 3,
                      proximity_pct: float = 0.06,
                      max_zones: int = 3,
                      detect_slant: bool = True) -> dict:
    """
    Convenience function for programmatic use.
    Returns:
        {
          'upper':       [ horizontal resistance zones ],
          'lower':       [ horizontal support zones ],
          'upper_slant': [ slanting resistance trendline zones ],
          'lower_slant': [ slanting support trendline zones ],
          'current_price': float
        }
    """
    cfg = Config()
    cfg.LOOKBACK        = lookback
    cfg.MIN_TOUCHES     = min_touches
    cfg.PROXIMITY_PCT   = proximity_pct
    cfg.MAX_ZONES_UPPER = max_zones
    cfg.MAX_ZONES_LOWER = max_zones
    cfg.SLANT_ENABLED   = detect_slant

    upper_h, lower_h = detect_wick_zones(df, cfg)

    upper_s, lower_s = [], []
    if detect_slant:
        upper_s, lower_s, _ = detect_slant_zones(df, cfg)

    return {
        "upper":         upper_h,
        "lower":         lower_h,
        "upper_slant":   upper_s,
        "lower_slant":   lower_s,
        "current_price": df["close"].iloc[-1],
    }


if __name__ == "__main__":
    main()