"""
pivot_detector.py — Swing high/low pivot detection.
Based on: The Art and Science of Technical Analysis (Adam Grimes)

Pivots are the fundamental building block of all market structure analysis.
A pivot high: bar whose high exceeds n bars on each side.
A pivot low: bar whose low is less than n bars on each side.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Pivot:
    index: int
    price: float
    pivot_type: str          # 'high' or 'low'
    strength: int            # look-back window used (n)
    score: float = 0.0       # composite significance score
    atr_normalized: float = 0.0  # distance to nearest neighbor / ATR


@dataclass
class ZigzagPoint:
    index: int
    price: float
    pivot_type: str          # 'high' or 'low'
    swing_from_prev: float = 0.0   # magnitude of swing from previous zigzag point


# ─── CORE PIVOT DETECTION ─────────────────────────────────────────────────────

def find_pivot_highs(high: np.ndarray, n: int = 3,
                     atr_vals: np.ndarray = None,
                     min_atr_dist: float = 0.3) -> List[Pivot]:
    """
    Detect pivot highs: bar i is a PH if high[i] > high[i-k] and high[i] > high[i+k]
    for all k in 1..n.

    Args:
        high:          High prices array
        n:             Look-left and look-right bars (default 3)
        atr_vals:      ATR array for significance filtering
        min_atr_dist:  Minimum ATR-normalized distance to count as pivot
    """
    pivots: List[Pivot] = []
    for i in range(n, len(high) - n):
        window = high[max(0, i - n):i + n + 1]
        if high[i] == np.max(window) and np.sum(window == high[i]) == 1:
            piv = Pivot(index=i, price=high[i], pivot_type='high', strength=n)
            if atr_vals is not None and not np.isnan(atr_vals[i]) and atr_vals[i] > 0:
                # Check ATR-normalized height vs surrounding bars
                left_max = np.max(high[max(0, i - n):i]) if i > 0 else high[i]
                right_max = np.max(high[i + 1:i + n + 1]) if i < len(high) - 1 else high[i]
                neighbor_max = max(left_max, right_max)
                piv.atr_normalized = (high[i] - neighbor_max) / atr_vals[i]
                if piv.atr_normalized < min_atr_dist:
                    continue
            pivots.append(piv)
    return pivots


def find_pivot_lows(low: np.ndarray, n: int = 3,
                    atr_vals: np.ndarray = None,
                    min_atr_dist: float = 0.3) -> List[Pivot]:
    """Detect pivot lows symmetrically to find_pivot_highs."""
    pivots: List[Pivot] = []
    for i in range(n, len(low) - n):
        window = low[max(0, i - n):i + n + 1]
        if low[i] == np.min(window) and np.sum(window == low[i]) == 1:
            piv = Pivot(index=i, price=low[i], pivot_type='low', strength=n)
            if atr_vals is not None and not np.isnan(atr_vals[i]) and atr_vals[i] > 0:
                left_min = np.min(low[max(0, i - n):i]) if i > 0 else low[i]
                right_min = np.min(low[i + 1:i + n + 1]) if i < len(low) - 1 else low[i]
                neighbor_min = min(left_min, right_min)
                piv.atr_normalized = (neighbor_min - low[i]) / atr_vals[i]
                if piv.atr_normalized < min_atr_dist:
                    continue
            pivots.append(piv)
    return pivots


def grimes_pivots(high: np.ndarray, low: np.ndarray,
                  order: int = 2) -> Tuple[List[Pivot], List[Pivot]]:
    """
    Grimes' exact 3-order pivot system from Chapter 1.

    ORDER 1 (book's base definition):
        Pivot High: high[i] > high[i-1] AND high[i] > high[i+1]
        Pivot Low:  low[i]  < low[i-1]  AND low[i]  < low[i+1]

    ORDER 2 (book's intermediate pivots):
        A 1st-order pivot HIGH that has lower 1st-order pivot highs
        on BOTH sides of it.
        These mark meaningful turning points — the bread and butter of
        swing analysis.

    ORDER 3 (book's major inflections):
        A 2nd-order pivot HIGH that has lower 2nd-order pivot highs
        on BOTH sides of it.
        Grimes: "Almost without exception these mark the best entries
        on both sides of the market."

    Returns: (pivot_highs, pivot_lows) — all up to the requested order.
    Note: higher-order pivots are a SUBSET of lower-order pivots.
    All returned pivots have .strength set to their order (1, 2, or 3).
    """
    # ── Order 1 ──────────────────────────────────────────────────────────────
    o1_highs: List[Pivot] = []
    o1_lows:  List[Pivot] = []

    for i in range(1, len(high) - 1):
        if high[i] > high[i - 1] and high[i] > high[i + 1]:
            o1_highs.append(Pivot(i, float(high[i]), 'high', strength=1))
    for i in range(1, len(low) - 1):
        if low[i] < low[i - 1] and low[i] < low[i + 1]:
            o1_lows.append(Pivot(i, float(low[i]), 'low', strength=1))

    if order == 1:
        return o1_highs, o1_lows

    # ── Order 2 ──────────────────────────────────────────────────────────────
    # A 1st-order PH is 2nd-order if the nearest 1st-order PH on each side
    # is LOWER than it.
    o2_highs: List[Pivot] = []
    for k, ph in enumerate(o1_highs):
        left_lower  = all(p.price < ph.price for p in o1_highs[:k])   if k > 0 else True
        right_lower = all(p.price < ph.price for p in o1_highs[k+1:]) if k < len(o1_highs) - 1 else True
        # Simplified: just check the immediate neighbours in the 1st-order list
        left_ok  = (o1_highs[k-1].price < ph.price) if k > 0 else True
        right_ok = (o1_highs[k+1].price < ph.price) if k < len(o1_highs) - 1 else True
        if left_ok and right_ok:
            o2_highs.append(Pivot(ph.index, ph.price, 'high', strength=2))

    o2_lows: List[Pivot] = []
    for k, pl in enumerate(o1_lows):
        left_ok  = (o1_lows[k-1].price > pl.price) if k > 0 else True
        right_ok = (o1_lows[k+1].price > pl.price) if k < len(o1_lows) - 1 else True
        if left_ok and right_ok:
            o2_lows.append(Pivot(pl.index, pl.price, 'low', strength=2))

    if order == 2:
        all_highs = o1_highs + o2_highs
        all_lows  = o1_lows  + o2_lows
        all_highs.sort(key=lambda p: p.index)
        all_lows.sort(key=lambda p: p.index)
        return all_highs, all_lows

    # ── Order 3 ──────────────────────────────────────────────────────────────
    o3_highs: List[Pivot] = []
    for k, ph in enumerate(o2_highs):
        left_ok  = (o2_highs[k-1].price < ph.price) if k > 0 else True
        right_ok = (o2_highs[k+1].price < ph.price) if k < len(o2_highs) - 1 else True
        if left_ok and right_ok:
            o3_highs.append(Pivot(ph.index, ph.price, 'high', strength=3))

    o3_lows: List[Pivot] = []
    for k, pl in enumerate(o2_lows):
        left_ok  = (o2_lows[k-1].price > pl.price) if k > 0 else True
        right_ok = (o2_lows[k+1].price > pl.price) if k < len(o2_lows) - 1 else True
        if left_ok and right_ok:
            o3_lows.append(Pivot(pl.index, pl.price, 'low', strength=3))

    all_highs = o1_highs + o2_highs + o3_highs
    all_lows  = o1_lows  + o2_lows  + o3_lows
    all_highs.sort(key=lambda p: p.index)
    all_lows.sort(key=lambda p: p.index)
    return all_highs, all_lows


def find_all_pivots(high: np.ndarray, low: np.ndarray,
                    n: int = 3, atr_vals: np.ndarray = None,
                    min_atr_dist: float = 0.3) -> List[Pivot]:
    """Find all pivot highs and lows, sorted by bar index."""
    ph = find_pivot_highs(high, n, atr_vals, min_atr_dist)
    pl = find_pivot_lows(low, n, atr_vals, min_atr_dist)
    all_pivots = ph + pl
    all_pivots.sort(key=lambda p: p.index)
    return all_pivots


# ─── PIVOT SCORING ────────────────────────────────────────────────────────────

def score_pivots(pivots: List[Pivot], high: np.ndarray, low: np.ndarray,
                 atr_vals: np.ndarray, total_bars: int) -> List[Pivot]:
    """
    Score each pivot by composite significance.
    Score factors:
      1. ATR-normalized height (already in piv.atr_normalized)
      2. Dominance span: how many bars the pivot was the extreme
      3. Recency: more recent pivots weighted higher
    """
    for piv in pivots:
        if np.isnan(atr_vals[piv.index]):
            piv.score = piv.atr_normalized
            continue

        # Dominance span: scan outward until a higher/lower bar is found
        span = 0
        if piv.pivot_type == 'high':
            arr = high
            for d in range(1, piv.index + 1):
                if arr[piv.index - d] > piv.price:
                    break
                span += 1
            for d in range(1, total_bars - piv.index):
                if arr[piv.index + d] > piv.price:
                    break
                span += 1
        else:
            arr = low
            for d in range(1, piv.index + 1):
                if arr[piv.index - d] < piv.price:
                    break
                span += 1
            for d in range(1, total_bars - piv.index):
                if arr[piv.index + d] < piv.price:
                    break
                span += 1

        dominance_score = min(span / total_bars * 10, 1.0)
        recency_score = piv.index / total_bars  # more recent = higher
        atr_score = min(piv.atr_normalized / 2.0, 1.0)  # cap at 2×ATR

        piv.score = 0.4 * atr_score + 0.4 * dominance_score + 0.2 * recency_score

    return pivots


# ─── ZIGZAG CONSTRUCTION ─────────────────────────────────────────────────────

def build_zigzag(pivots: List[Pivot]) -> List[ZigzagPoint]:
    """
    Construct the market structure skeleton (zigzag) from raw pivots.
    Rules:
      - Alternate between highs and lows
      - Between two lows (with no intervening high), keep only the lowest
      - Between two highs (with no intervening low), keep only the highest
    """
    if not pivots:
        return []

    # Group consecutive same-type pivots; keep the most extreme
    cleaned: List[Pivot] = []
    i = 0
    while i < len(pivots):
        group_type = pivots[i].pivot_type
        group = [pivots[i]]
        j = i + 1
        while j < len(pivots) and pivots[j].pivot_type == group_type:
            group.append(pivots[j])
            j += 1
        if group_type == 'high':
            best = max(group, key=lambda p: p.price)
        else:
            best = min(group, key=lambda p: p.price)
        cleaned.append(best)
        i = j

    # Convert to ZigzagPoints
    zz: List[ZigzagPoint] = []
    for k, piv in enumerate(cleaned):
        zpt = ZigzagPoint(index=piv.index, price=piv.price, pivot_type=piv.pivot_type)
        if k > 0:
            zpt.swing_from_prev = abs(piv.price - cleaned[k - 1].price)
        zz.append(zpt)

    return zz


# ─── MULTI-STRENGTH PIVOTS ────────────────────────────────────────────────────

def multi_strength_pivots(high: np.ndarray, low: np.ndarray,
                          atr_vals: np.ndarray,
                          strengths: Tuple[int, ...] = (2, 3, 5)
                          ) -> dict:
    """
    Compute pivots at multiple strengths for multi-timeframe-like analysis
    within a single timeframe.
    Returns dict: {strength: [Pivot, ...]}
    """
    result = {}
    for n in strengths:
        pivots = find_all_pivots(high, low, n, atr_vals)
        pivots = score_pivots(pivots, high, low, atr_vals, len(high))
        result[n] = pivots
    return result


# ─── RECENT PIVOTS ────────────────────────────────────────────────────────────

def recent_pivots(pivots: List[Pivot], n_last: int = 10) -> List[Pivot]:
    """Return the last n_last confirmed pivots by bar index."""
    return sorted(pivots, key=lambda p: p.index)[-n_last:]


def last_pivot_high(pivots: List[Pivot]) -> Optional[Pivot]:
    highs = [p for p in pivots if p.pivot_type == 'high']
    return max(highs, key=lambda p: p.index) if highs else None


def last_pivot_low(pivots: List[Pivot]) -> Optional[Pivot]:
    lows = [p for p in pivots if p.pivot_type == 'low']
    return max(lows, key=lambda p: p.index) if lows else None


# ─── SWING METRICS ────────────────────────────────────────────────────────────

def swing_metrics(zigzag: List[ZigzagPoint]) -> pd.DataFrame:
    """
    Compute per-swing metrics for trend health analysis.
    Returns DataFrame with columns:
        swing_index, type, magnitude, duration, velocity, start_price, end_price
    """
    if len(zigzag) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(1, len(zigzag)):
        prev = zigzag[i - 1]
        curr = zigzag[i]
        magnitude = abs(curr.price - prev.price)
        duration = curr.index - prev.index
        velocity = magnitude / duration if duration > 0 else 0.0
        direction = 'up' if curr.price > prev.price else 'down'
        rows.append({
            'swing_index': i,
            'direction': direction,
            'magnitude': magnitude,
            'duration': duration,
            'velocity': velocity,
            'start_bar': prev.index,
            'end_bar': curr.index,
            'start_price': prev.price,
            'end_price': curr.price,
        })
    return pd.DataFrame(rows)


# ─── DATAFRAME INTEGRATION ────────────────────────────────────────────────────

def detect_pivots_from_df(df: pd.DataFrame, n: int = 3,
                          min_atr_dist: float = 0.3) -> Tuple[List[Pivot], List[ZigzagPoint]]:
    """
    Convenience wrapper: detect pivots and zigzag from a DataFrame.
    DataFrame must have 'high', 'low', 'atr' columns.
    """
    high = df['high'].values
    low = df['low'].values
    atr_vals = df['atr'].values if 'atr' in df.columns else None

    pivots = find_all_pivots(high, low, n, atr_vals, min_atr_dist)
    if atr_vals is not None:
        pivots = score_pivots(pivots, high, low, atr_vals, len(high))
    zigzag = build_zigzag(pivots)
    return pivots, zigzag
