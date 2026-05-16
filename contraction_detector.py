"""
Price Contraction Detector (Pure Price Action — No Indicators)
==============================================================

Detects converging-range contractions (descending triangles, ascending triangles,
symmetric triangles, wedges, pennants) on OHLC data using only swing pivots,
trendline geometry, and bar-range compression.

Pipeline:
    1. Swing pivot detection (fractal, +/- N bars)
    2. Prior-trend filter (a contraction must follow a directional move)
    3. Sliding window over OHLC -> collect pivots inside window
    4. Fit upper line through pivot-highs, lower line through pivot-lows (least-squares)
    5. Geometric validation:
        - upper slope <= 0
        - lower slope >= 0   (loose: either flat, or converging from below)
        - upper slope < lower slope  (lines actually converge going right)
        - range compression: avg bar-range late < avg bar-range early
        - R^2 quality of fits (pivots actually lie on the lines)
        - containment: most bars stay between the lines
        - duration in [MIN_LEN, MAX_LEN]
        - apex (intersection) is ahead and within reasonable distance
    6. Greedy de-duplication of overlapping windows -> keep best per region
    7. Breakout state machine after detection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


# ============================================================================
# CONFIG  (tuned for 5-min NIFTY; adjust for other timeframes)
# ============================================================================

@dataclass
class Config:
    # --- swing pivot detection ---
    pivot_left: int = 3          # bars to the left a high/low must beat
    pivot_right: int = 3         # bars to the right a high/low must beat

    # --- contraction window scan ---
    min_len: int = 15            # minimum bars in contraction
    max_len: int = 40            # maximum bars in contraction
    step: int = 1                # slide step (1 = every bar)

    # --- pivot-count requirement inside window ---
    min_pivots_each_side: int = 2  # need >=2 highs and >=2 lows to fit a line

    # --- geometry thresholds ---
    max_upper_slope: float = 0.0      # upper line must not rise (<=0)
    min_lower_slope: float = -1e9     # loose: lower can fall slightly (set 0 for strict)
    convergence_eps: float = 1e-9     # upper_slope must be < lower_slope by this much
    min_r2: float = 0.40              # min R^2 for both line fits
    min_containment: float = 0.85     # >=85% of bars within [lower_line, upper_line]

    # --- range compression ---
    compression_ratio: float = 0.80   # avg(range, last third) / avg(range, first third) < this

    # --- prior-trend filter ---
    trend_lookback: int = 25
    trend_strength: float = 1.5       # |close_now - close_then| / avg_range >= this

    # --- apex sanity ---
    max_apex_distance_mult: float = 2.0  # apex within max_len * this bars to the right

    # --- breakout ---
    breakout_buffer_mult: float = 0.10   # buffer = this * line_height_at_breakout (in price units)
                                         # 0.10 means breakout candle close must clear line by 10% of
                                         # current wedge height; protects against ticks brushing line

    # --- de-duplication ---
    dedup_overlap_frac: float = 0.50  # if two patterns overlap >50%, keep the higher-quality one


CFG = Config()


# ============================================================================
# 1.  SWING PIVOT DETECTION
# ============================================================================

def find_pivots(highs: np.ndarray, lows: np.ndarray, left: int, right: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fractal pivots. A bar i is a swing high if highs[i] > all highs in [i-left, i-1] and [i+1, i+right].
    Same for lows.

    Returns:
        pivot_high_idx : np.ndarray of indices that are swing highs
        pivot_low_idx  : np.ndarray of indices that are swing lows
    """
    n = len(highs)
    ph, pl = [], []
    for i in range(left, n - right):
        win_h = highs[i - left : i + right + 1]
        win_l = lows[i - left : i + right + 1]
        if highs[i] == win_h.max() and (win_h == highs[i]).sum() == 1:
            ph.append(i)
        if lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            pl.append(i)
    return np.array(ph, dtype=int), np.array(pl, dtype=int)


# ============================================================================
# 2.  LINE FITTING + GEOMETRY HELPERS
# ============================================================================

def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Least-squares line y = slope*x + intercept.
    Returns (slope, intercept, r_squared).
    """
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0, 0.0
    x = x.astype(float)
    y = y.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return float(slope), float(intercept), float(r2)


def line_at(slope: float, intercept: float, x: float) -> float:
    return slope * x + intercept


def apex_x(slope_u: float, intercept_u: float, slope_l: float, intercept_l: float) -> Optional[float]:
    """X-coordinate where upper and lower lines intersect."""
    denom = slope_u - slope_l
    if abs(denom) < 1e-12:
        return None
    return (intercept_l - intercept_u) / denom


# ============================================================================
# 3.  WINDOW VALIDATION  (the heart)
# ============================================================================

@dataclass
class Contraction:
    start_idx: int
    end_idx: int                 # inclusive
    upper_slope: float
    upper_intercept: float
    lower_slope: float
    lower_intercept: float
    upper_r2: float
    lower_r2: float
    n_pivot_highs: int
    n_pivot_lows: int
    compression: float           # ratio late/early avg range
    containment: float           # frac of bars inside lines
    apex_idx: float              # x of apex
    quality: float               # composite score [0..1]

    def to_dict(self):
        return asdict(self)


def validate_window(
    df: pd.DataFrame,
    start: int,
    end: int,
    pivot_high_idx: np.ndarray,
    pivot_low_idx: np.ndarray,
    cfg: Config,
) -> Optional[Contraction]:
    """
    Test bars [start..end] (inclusive) for contraction. Returns Contraction or None.
    """
    length = end - start + 1
    if length < cfg.min_len or length > cfg.max_len:
        return None

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    # --- pivots inside window ---
    ph_in = pivot_high_idx[(pivot_high_idx >= start) & (pivot_high_idx <= end)]
    pl_in = pivot_low_idx[(pivot_low_idx >= start) & (pivot_low_idx <= end)]
    if len(ph_in) < cfg.min_pivots_each_side or len(pl_in) < cfg.min_pivots_each_side:
        return None

    # --- prior-trend filter ---
    look = cfg.trend_lookback
    if start - look < 0:
        return None
    avg_range = np.mean(highs[start - look : start] - lows[start - look : start])
    if avg_range <= 0:
        return None
    trend_move = abs(closes[start] - closes[start - look]) / avg_range
    if trend_move < cfg.trend_strength:
        return None

    # --- fit upper line through pivot-highs, lower through pivot-lows ---
    x_u = ph_in.astype(float)
    y_u = highs[ph_in]
    x_l = pl_in.astype(float)
    y_l = lows[pl_in]

    slope_u, intercept_u, r2_u = fit_line(x_u, y_u)
    slope_l, intercept_l, r2_l = fit_line(x_l, y_l)

    # --- geometry checks ---
    if slope_u > cfg.max_upper_slope:
        return None
    if slope_l < cfg.min_lower_slope:
        return None
    if slope_u >= slope_l - cfg.convergence_eps:   # must converge
        return None
    if r2_u < cfg.min_r2 or r2_l < cfg.min_r2:
        return None

    # --- containment: most bars between the two lines ---
    xs = np.arange(start, end + 1, dtype=float)
    upper_vals = slope_u * xs + intercept_u
    lower_vals = slope_l * xs + intercept_l
    bars_in = np.sum((highs[start:end + 1] <= upper_vals + 1e-6) & (lows[start:end + 1] >= lower_vals - 1e-6))
    containment = bars_in / length
    if containment < cfg.min_containment:
        return None

    # --- range compression: third-by-third ---
    third = max(length // 3, 3)
    rng = highs[start:end + 1] - lows[start:end + 1]
    early = rng[:third].mean()
    late = rng[-third:].mean()
    if early <= 0:
        return None
    compression = late / early
    if compression > cfg.compression_ratio:
        return None

    # --- apex must be ahead and within reach ---
    ax = apex_x(slope_u, intercept_u, slope_l, intercept_l)
    if ax is None or ax <= end:
        return None
    if ax - end > cfg.max_len * cfg.max_apex_distance_mult:
        return None

    # --- composite quality score (used for de-dup ranking) ---
    quality = float(
        0.30 * min(r2_u, r2_l)
        + 0.20 * containment
        + 0.20 * max(0.0, 1.0 - compression)            # smaller compression -> higher score
        + 0.15 * min(1.0, (len(ph_in) + len(pl_in)) / 8)
        + 0.15 * min(1.0, trend_move / 3.0)
    )

    return Contraction(
        start_idx=start,
        end_idx=end,
        upper_slope=slope_u,
        upper_intercept=intercept_u,
        lower_slope=slope_l,
        lower_intercept=intercept_l,
        upper_r2=r2_u,
        lower_r2=r2_l,
        n_pivot_highs=len(ph_in),
        n_pivot_lows=len(pl_in),
        compression=compression,
        containment=containment,
        apex_idx=ax,
        quality=quality,
    )


# ============================================================================
# 4.  HISTORICAL SCAN
# ============================================================================

def scan_history(df: pd.DataFrame, cfg: Config = CFG) -> List[Contraction]:
    """
    Scan entire dataframe for contractions. Returns de-duplicated list ordered by end_idx.

    Optimization:
      - Only anchor on end-bars that ARE swing pivots (high or low). The contraction
        boundary is always a pivot, so non-pivot end-bars cannot start a new pattern
        that wasn't already covered by an adjacent pivot end-bar.
      - For each anchor end-bar, only test a coarse grid of lengths
        (min_len, min_len+5, ..., max_len), then refine around hits. This is ~6x fewer
        validations with negligible quality loss because patterns are locally stable
        in length.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    pivot_high_idx, pivot_low_idx = find_pivots(highs, lows, cfg.pivot_left, cfg.pivot_right)
    pivot_set = set(pivot_high_idx.tolist()) | set(pivot_low_idx.tolist())
    pivot_anchors = sorted(pivot_set)

    # coarse + refine length grid
    coarse = list(range(cfg.min_len, cfg.max_len + 1, 4))
    if cfg.max_len not in coarse:
        coarse.append(cfg.max_len)

    candidates: List[Contraction] = []

    for end in pivot_anchors:
        if end < cfg.trend_lookback + cfg.min_len:
            continue
        if end >= n:
            continue
        # coarse pass
        coarse_hits = []
        for length in coarse:
            start = end - length + 1
            if start < cfg.trend_lookback:
                continue
            c = validate_window(df, start, end, pivot_high_idx, pivot_low_idx, cfg)
            if c is not None:
                coarse_hits.append(length)
                candidates.append(c)
        # refine around each coarse hit (+/- 3)
        seen_lengths = set(coarse_hits)
        for hit_len in coarse_hits:
            for delta in (-3, -2, -1, 1, 2, 3):
                length = hit_len + delta
                if length in seen_lengths:
                    continue
                if length < cfg.min_len or length > cfg.max_len:
                    continue
                seen_lengths.add(length)
                start = end - length + 1
                if start < cfg.trend_lookback:
                    continue
                c = validate_window(df, start, end, pivot_high_idx, pivot_low_idx, cfg)
                if c is not None:
                    candidates.append(c)

    return _dedup(candidates, cfg)


def _dedup(cands: List[Contraction], cfg: Config) -> List[Contraction]:
    """
    Greedy: sort by quality desc, keep a candidate if it does not overlap an already-kept
    one by more than dedup_overlap_frac.
    """
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c.quality, reverse=True)
    kept: List[Contraction] = []
    for c in cands:
        overlap = False
        for k in kept:
            inter = max(0, min(c.end_idx, k.end_idx) - max(c.start_idx, k.start_idx) + 1)
            min_len = min(c.end_idx - c.start_idx + 1, k.end_idx - k.start_idx + 1)
            if inter / min_len > cfg.dedup_overlap_frac:
                overlap = True
                break
        if not overlap:
            kept.append(c)
    return sorted(kept, key=lambda c: c.end_idx)


# ============================================================================
# 5.  LIVE DETECTION
# ============================================================================

def detect_live(df: pd.DataFrame, cfg: Config = CFG) -> Optional[Contraction]:
    """
    Treats the LAST bar of df as 'now'. Looks for an active contraction ending
    at the latest bar (within the last few bars to allow some lag).

    Returns the highest-quality active contraction, or None.
    """
    n = len(df)
    if n < cfg.trend_lookback + cfg.max_len:
        return None

    highs = df["high"].values
    lows = df["low"].values
    pivot_high_idx, pivot_low_idx = find_pivots(highs, lows, cfg.pivot_left, cfg.pivot_right)

    best: Optional[Contraction] = None
    # the contraction must END at or very near the last bar
    for end in range(n - 1, max(n - 6, cfg.min_len - 2), -1):
        for length in range(cfg.min_len, cfg.max_len + 1):
            start = end - length + 1
            if start < cfg.trend_lookback:
                continue
            c = validate_window(df, start, end, pivot_high_idx, pivot_low_idx, cfg)
            if c is None:
                continue
            if best is None or c.quality > best.quality:
                best = c
    return best


# ============================================================================
# 6.  BREAKOUT SIGNAL (state machine over each detected contraction)
# ============================================================================

@dataclass
class BreakoutSignal:
    contraction_end_idx: int
    breakout_idx: int
    breakout_direction: str         # 'up' or 'down'
    breakout_price: float
    line_price_at_break: float
    bars_after_pattern: int


def detect_breakout(df: pd.DataFrame, c: Contraction, cfg: Config = CFG,
                    look_forward: int = 30) -> Optional[BreakoutSignal]:
    """
    After a contraction is identified, watch the next `look_forward` bars.
    Signal fires on the first bar whose CLOSE clears the projected upper or lower
    trendline by a buffer.
    """
    closes = df["close"].values
    n = len(df)
    end = c.end_idx
    stop = min(n, end + 1 + look_forward)

    for i in range(end + 1, stop):
        upper_now = c.upper_slope * i + c.upper_intercept
        lower_now = c.lower_slope * i + c.lower_intercept
        height = max(upper_now - lower_now, 1e-9)
        buffer = cfg.breakout_buffer_mult * height

        if closes[i] > upper_now + buffer:
            return BreakoutSignal(
                contraction_end_idx=end,
                breakout_idx=i,
                breakout_direction="up",
                breakout_price=float(closes[i]),
                line_price_at_break=float(upper_now),
                bars_after_pattern=i - end,
            )
        if closes[i] < lower_now - buffer:
            return BreakoutSignal(
                contraction_end_idx=end,
                breakout_idx=i,
                breakout_direction="down",
                breakout_price=float(closes[i]),
                line_price_at_break=float(lower_now),
                bars_after_pattern=i - end,
            )
    return None


# ============================================================================
# CSV LOADER
# ============================================================================

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df