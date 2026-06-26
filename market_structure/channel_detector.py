"""
channel_detector.py — Trend channel detection and overextension analysis.
Based on: The Art and Science of Technical Analysis (Adam Grimes)

Channels define the rate of trend and mark emotional extremes.
Keltner channels show overextension; parabolic moves precede climaxes.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy import stats as scipy_stats

from pivot_detector import ZigzagPoint, Pivot


@dataclass
class TrendLine:
    slope: float               # price change per bar
    intercept: float           # price at bar 0
    r_squared: float           # fit quality 0-1
    anchor_bars: List[int]     # bars used to fit the line
    direction: str             # 'support' or 'resistance'
    start_bar: int
    end_bar: int

    def value_at(self, bar: int) -> float:
        return self.slope * bar + self.intercept

    def distance_from(self, price: float, bar: int) -> float:
        return price - self.value_at(bar)


@dataclass
class Channel:
    base_line: TrendLine       # lower line (for uptrend) or upper line (for downtrend)
    parallel_line: TrendLine   # upper line (for uptrend) or lower line (for downtrend)
    channel_width: float       # in price units
    channel_width_atr: float   # in ATR units
    direction: str             # 'up' or 'down'
    r_squared: float           # average fit quality

    def price_position(self, price: float, bar: int) -> float:
        """Return 0.0=at base line, 1.0=at parallel line."""
        base_val = self.base_line.value_at(bar)
        par_val = self.parallel_line.value_at(bar)
        width = par_val - base_val
        if abs(width) < 1e-9:
            return 0.5
        return (price - base_val) / width

    def at_upper_extreme(self, price: float, bar: int, threshold: float = 0.85) -> bool:
        return self.price_position(price, bar) > threshold

    def at_lower_extreme(self, price: float, bar: int, threshold: float = 0.15) -> bool:
        return self.price_position(price, bar) < threshold

    def outside_channel(self, price: float, bar: int) -> int:
        pos = self.price_position(price, bar)
        if pos > 1.0:
            return 1   # above channel
        elif pos < 0.0:
            return -1  # below channel
        return 0


@dataclass
class ClimaxSignal:
    bar_index: int
    climax_type: str           # 'bullish_exhaustion' or 'bearish_exhaustion'
    strength: float            # 0-1
    bar_range_atr_ratio: float
    close_position: float      # 0=at low, 1=at high
    outside_keltner: bool
    macd_divergence: bool
    notes: List[str]


# ─── TREND LINE FITTING ───────────────────────────────────────────────────────

def fit_trend_line_to_pivots(pivots: List[Pivot], direction: str,
                              min_pivots: int = 2) -> Optional[TrendLine]:
    """
    Fit a trend line to pivot prices using linear regression.
    direction: 'support' (pivot lows) or 'resistance' (pivot highs)
    """
    if direction == 'support':
        pts = [(p.index, p.price) for p in pivots if p.pivot_type == 'low']
    else:
        pts = [(p.index, p.price) for p in pivots if p.pivot_type == 'high']

    if len(pts) < min_pivots:
        return None

    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    slope, intercept, r_value, p_value, se = scipy_stats.linregress(xs, ys)

    return TrendLine(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_value ** 2),
        anchor_bars=[int(x) for x in xs],
        direction=direction,
        start_bar=int(xs[0]),
        end_bar=int(xs[-1])
    )


def fit_channel(pivots: List[Pivot], trend_direction: str,
                current_atr: float) -> Optional[Channel]:
    """
    Build a parallel channel:
    - For uptrend: base = regression through pivot lows, parallel = through pivot highs
    - For downtrend: base = regression through pivot highs, parallel = through pivot lows

    Args:
        pivots: List of Pivot objects
        trend_direction: 'up' or 'down'
        current_atr: current ATR for width normalization
    """
    if trend_direction == 'up':
        base = fit_trend_line_to_pivots(pivots, 'support')
        parallel = fit_trend_line_to_pivots(pivots, 'resistance')
    else:
        base = fit_trend_line_to_pivots(pivots, 'resistance')
        parallel = fit_trend_line_to_pivots(pivots, 'support')

    if base is None or parallel is None:
        return None

    # Force parallel line to be parallel to base (same slope)
    # Adjust intercept of parallel to best fit the opposing pivots
    if trend_direction == 'up':
        opp_pts = [(p.index, p.price) for p in pivots if p.pivot_type == 'high']
    else:
        opp_pts = [(p.index, p.price) for p in pivots if p.pivot_type == 'low']

    if opp_pts:
        # Shift parallel line to pass through the most extreme opposing pivot
        if trend_direction == 'up':
            best_opp = max(opp_pts, key=lambda x: x[1])
        else:
            best_opp = min(opp_pts, key=lambda x: x[1])

        new_intercept = best_opp[1] - base.slope * best_opp[0]
        parallel = TrendLine(
            slope=base.slope,
            intercept=new_intercept,
            r_squared=parallel.r_squared,
            anchor_bars=parallel.anchor_bars,
            direction=parallel.direction,
            start_bar=parallel.start_bar,
            end_bar=parallel.end_bar
        )

    # Compute width at the midpoint
    mid_bar = (base.start_bar + base.end_bar) // 2
    width = abs(parallel.value_at(mid_bar) - base.value_at(mid_bar))
    width_atr = width / current_atr if current_atr > 0 else 0.0

    return Channel(
        base_line=base,
        parallel_line=parallel,
        channel_width=width,
        channel_width_atr=width_atr,
        direction=trend_direction,
        r_squared=(base.r_squared + parallel.r_squared) / 2
    )


# ─── RATE OF TREND ANALYSIS ───────────────────────────────────────────────────

def analyze_rate_of_trend(zigzag: List[ZigzagPoint],
                           direction: int) -> dict:
    """
    Measure acceleration/deceleration of trend by comparing successive
    impulse leg slopes.

    Returns dict with:
      - slopes: list of slopes for each impulse leg
      - is_accelerating: each impulse steeper than the last
      - is_decelerating: each impulse shallower than the last
      - is_parabolic: acceleration is extreme (slope doubled)
      - deceleration_degree: 0-1 (0=none, 1=severe)
    """
    if len(zigzag) < 4:
        return {'slopes': [], 'is_accelerating': False,
                'is_decelerating': False, 'is_parabolic': False,
                'deceleration_degree': 0.0}

    # Extract impulse segments
    impulse_segments = []
    for i in range(1, len(zigzag)):
        prev = zigzag[i - 1]
        curr = zigzag[i]
        is_impulse = (direction == 1 and curr.price > prev.price) or \
                     (direction == -1 and curr.price < prev.price)
        if is_impulse:
            duration = curr.index - prev.index
            if duration > 0:
                slope = (curr.price - prev.price) / duration
                impulse_segments.append((slope, prev.index, curr.index))

    if len(impulse_segments) < 2:
        return {'slopes': [], 'is_accelerating': False,
                'is_decelerating': False, 'is_parabolic': False,
                'deceleration_degree': 0.0}

    slopes = [abs(s[0]) for s in impulse_segments]

    # Acceleration: slopes increasing
    is_accelerating = all(slopes[i] > slopes[i - 1] for i in range(1, len(slopes)))
    is_decelerating = all(slopes[i] < slopes[i - 1] for i in range(1, len(slopes)))

    # Parabolic: last slope > 2× the first slope
    is_parabolic = len(slopes) >= 2 and slopes[-1] > 2 * slopes[0]

    # Deceleration degree
    if len(slopes) >= 2:
        decel = max(0.0, (slopes[0] - slopes[-1]) / slopes[0]) if slopes[0] > 0 else 0.0
    else:
        decel = 0.0

    return {
        'slopes': slopes,
        'is_accelerating': is_accelerating,
        'is_decelerating': is_decelerating,
        'is_parabolic': is_parabolic,
        'deceleration_degree': float(decel)
    }


# ─── CLIMAX DETECTION ─────────────────────────────────────────────────────────

def detect_parabolic_climax(high: np.ndarray, low: np.ndarray,
                             close: np.ndarray, atr_vals: np.ndarray,
                             kc_upper: np.ndarray, kc_lower: np.ndarray,
                             macd_divergence: np.ndarray,
                             lookback: int = 5,
                             range_atr_threshold: float = 2.0) -> List[ClimaxSignal]:
    """
    Detect parabolic climax / exhaustion bars.

    Bullish exhaustion (top): large up bar with close near the LOW (sellers step in)
    Bearish exhaustion (bottom): large down bar with close near the HIGH (buyers step in)

    Both require:
    - Bar range > range_atr_threshold × ATR
    - Bar close outside Keltner channel
    - Optional: MACD divergence
    """
    signals = []
    n = len(close)

    for i in range(max(lookback, 20), n):
        current_atr = float(atr_vals[i]) if not np.isnan(atr_vals[i]) else float(np.nanmean(atr_vals[max(0, i-20):i]))
        if current_atr == 0:
            continue

        bar_range = high[i] - low[i]
        bar_ratio = bar_range / current_atr
        close_pos = (close[i] - low[i]) / bar_range if bar_range > 0 else 0.5

        # Is it the largest range bar in recent lookback?
        recent_ranges = high[max(0, i - lookback):i] - low[max(0, i - lookback):i]
        is_largest = bar_range >= np.max(recent_ranges) * 0.85

        above_kc = not np.isnan(kc_upper[i]) and close[i] > kc_upper[i]
        below_kc = not np.isnan(kc_lower[i]) and close[i] < kc_lower[i]
        has_div = macd_divergence[i] != 0 if i < len(macd_divergence) else False

        if bar_ratio >= range_atr_threshold and is_largest:
            notes = [f"Bar range = {bar_ratio:.1f}× ATR"]

            # Bullish exhaustion: big up bar but closes near its LOW
            if close[i] > close[i - 1] and close_pos < 0.35 and above_kc:
                strength = _climax_strength(bar_ratio, close_pos, above_kc, has_div, 'bull')
                signals.append(ClimaxSignal(
                    bar_index=i,
                    climax_type='bullish_exhaustion',
                    strength=float(strength),
                    bar_range_atr_ratio=float(bar_ratio),
                    close_position=float(close_pos),
                    outside_keltner=above_kc,
                    macd_divergence=has_div,
                    notes=notes + (["Outside Keltner"] if above_kc else [])
                         + (["MACD divergence"] if has_div else [])
                ))

            # Bearish exhaustion: big down bar but closes near its HIGH
            elif close[i] < close[i - 1] and close_pos > 0.65 and below_kc:
                strength = _climax_strength(bar_ratio, 1 - close_pos, below_kc, has_div, 'bear')
                signals.append(ClimaxSignal(
                    bar_index=i,
                    climax_type='bearish_exhaustion',
                    strength=float(strength),
                    bar_range_atr_ratio=float(bar_ratio),
                    close_position=float(close_pos),
                    outside_keltner=below_kc,
                    macd_divergence=has_div,
                    notes=notes + (["Outside Keltner"] if below_kc else [])
                         + (["MACD divergence"] if has_div else [])
                ))

    return signals


def _climax_strength(bar_ratio: float, close_extreme: float,
                     outside_kc: bool, has_div: bool, ctype: str) -> float:
    """Compute 0-1 climax signal strength."""
    s = 0.0
    s += min((bar_ratio - 1.5) / 2.0, 0.4)   # bar size contribution
    s += (1.0 - close_extreme) * 0.3           # how extreme the close is
    if outside_kc:
        s += 0.2
    if has_div:
        s += 0.1
    return float(np.clip(s, 0.0, 1.0))


# ─── KELTNER POSITION ANALYSIS ────────────────────────────────────────────────

def keltner_analysis(close: np.ndarray, kc_upper: np.ndarray,
                     kc_lower: np.ndarray, kc_mid: np.ndarray,
                     lookback: int = 5) -> dict:
    """
    Summarize recent Keltner channel position for trading decisions.
    """
    if len(close) < lookback:
        return {}

    recent_close = close[-lookback:]
    recent_upper = kc_upper[-lookback:]
    recent_lower = kc_lower[-lookback:]

    width = recent_upper - recent_lower
    pos = np.where(width > 0, (recent_close - recent_lower) / width, 0.5)

    bars_above = int(np.sum(recent_close > recent_upper))
    bars_below = int(np.sum(recent_close < recent_lower))
    current_pos = float(pos[-1]) if len(pos) > 0 else 0.5

    # Consecutive bars outside channel (exhaustion signal)
    consec_above = 0
    consec_below = 0
    for k in range(len(recent_close) - 1, -1, -1):
        if recent_close[k] > recent_upper[k]:
            consec_above += 1
        else:
            break
    for k in range(len(recent_close) - 1, -1, -1):
        if recent_close[k] < recent_lower[k]:
            consec_below += 1
        else:
            break

    # Trend hugging (consistently near one band)
    hugging_upper = np.mean(pos > 0.7) > 0.6
    hugging_lower = np.mean(pos < 0.3) > 0.6

    return {
        'current_position': current_pos,
        'is_overextended_up': current_pos > 1.0,
        'is_overextended_down': current_pos < 0.0,
        'consecutive_bars_above': consec_above,
        'consecutive_bars_below': consec_below,
        'exhaustion_up': consec_above >= 2,    # 2+ bars above = exhaustion
        'exhaustion_down': consec_below >= 2,
        'trend_hugging_upper': hugging_upper,
        'trend_hugging_lower': hugging_lower,
        'avg_position': float(np.mean(pos)),
    }


# ─── MICRO TREND LINES (Al Brooks style) ─────────────────────────────────────

def fit_micro_trend_line(close: np.ndarray, n_bars: int = 5) -> Optional[TrendLine]:
    """
    Fit a very short-term trend line to the last n_bars of close prices.
    Used for intraday momentum entry timing.
    Returns None if fit is too poor (r² < 0.7).
    """
    if len(close) < n_bars:
        return None
    recent = close[-n_bars:]
    xs = np.arange(len(recent), dtype=float)
    slope, intercept, r_value, _, _ = scipy_stats.linregress(xs, recent)
    if r_value ** 2 < 0.7:
        return None
    global_start = len(close) - n_bars
    return TrendLine(
        slope=float(slope),
        intercept=float(intercept + slope * global_start),  # shift intercept to global bar scale
        r_squared=float(r_value ** 2),
        anchor_bars=list(range(global_start, len(close))),
        direction='support' if slope > 0 else 'resistance',
        start_bar=global_start,
        end_bar=len(close) - 1
    )
