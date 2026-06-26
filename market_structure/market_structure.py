"""
market_structure.py — Market regime detection based on pivot structure.
Based on: The Art and Science of Technical Analysis (Adam Grimes)

Answers: "What Wyckoff phase is this market in, and what trades are valid?"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from pivot_detector import Pivot, ZigzagPoint, swing_metrics


class MarketState(Enum):
    STRONG_UPTREND   = "strong_uptrend"
    WEAK_UPTREND     = "weak_uptrend"
    ACCUMULATION     = "accumulation"
    RANGE            = "range"
    DISTRIBUTION     = "distribution"
    WEAK_DOWNTREND   = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    BREAKOUT_LONG    = "breakout_long"
    BREAKOUT_SHORT   = "breakout_short"
    TRANSITION       = "transition"


class PullbackType(Enum):
    NONE    = "none"
    SIMPLE  = "simple"
    COMPLEX = "complex"
    FAILED  = "failed"


@dataclass
class MarketRegime:
    state: MarketState
    confidence: float          # 0-1
    trend_direction: int       # +1 up, -1 down, 0 flat
    strength_score: float      # 0-10 trend strength
    is_strengthening: bool
    is_weakening: bool
    in_pullback: bool
    pullback_type: PullbackType
    pullback_depth_pct: float  # as fraction of prior impulse
    ab_cd_target: Optional[float]  # measured move objective
    key_level_high: Optional[float]
    key_level_low: Optional[float]
    last_impulse_size: float
    warning_signs: List[str]


# ─── DOW THEORY ───────────────────────────────────────────────────────────────

def dow_theory_trend(zigzag: List[ZigzagPoint], min_swings: int = 4
                     ) -> Tuple[int, float]:
    """
    Classify trend using Dow Theory: series of HH+HL (up) or LL+LH (down).
    Returns: (direction: +1/-1/0, confidence: 0-1)
    """
    if len(zigzag) < min_swings:
        return 0, 0.0

    last_n = zigzag[-min_swings:]
    highs = [p for p in last_n if p.pivot_type == 'high']
    lows = [p for p in last_n if p.pivot_type == 'low']

    if len(highs) < 2 or len(lows) < 2:
        return 0, 0.0

    # Check last 2 highs and last 2 lows
    hh = highs[-1].price > highs[-2].price   # Higher High
    hl = lows[-1].price > lows[-2].price      # Higher Low
    ll = lows[-1].price < lows[-2].price      # Lower Low
    lh = highs[-1].price < highs[-2].price    # Lower High

    if hh and hl:
        direction = 1
        confidence = 0.7 + (0.3 if len(zigzag) >= 6 and _extended_hh_hl(zigzag) else 0)
    elif ll and lh:
        direction = -1
        confidence = 0.7 + (0.3 if len(zigzag) >= 6 and _extended_ll_lh(zigzag) else 0)
    elif hh and not hl:
        direction = 1
        confidence = 0.4   # HH but HL failed — weakening uptrend
    elif ll and not lh:
        direction = -1
        confidence = 0.4
    elif hl and not hh:
        direction = 0
        confidence = 0.3   # HL but no HH — possible base building
    else:
        direction = 0
        confidence = 0.2

    return direction, min(confidence, 1.0)


def _extended_hh_hl(zigzag: List[ZigzagPoint]) -> bool:
    """Check last 3 consecutive HH+HL."""
    highs = [p for p in zigzag[-8:] if p.pivot_type == 'high']
    lows = [p for p in zigzag[-8:] if p.pivot_type == 'low']
    if len(highs) < 3 or len(lows) < 3:
        return False
    return (highs[-1].price > highs[-2].price > highs[-3].price and
            lows[-1].price > lows[-2].price > lows[-3].price)


def _extended_ll_lh(zigzag: List[ZigzagPoint]) -> bool:
    """Check last 3 consecutive LL+LH."""
    highs = [p for p in zigzag[-8:] if p.pivot_type == 'high']
    lows = [p for p in zigzag[-8:] if p.pivot_type == 'low']
    if len(highs) < 3 or len(lows) < 3:
        return False
    return (lows[-1].price < lows[-2].price < lows[-3].price and
            highs[-1].price < highs[-2].price < highs[-3].price)


# ─── TREND STRENGTH ───────────────────────────────────────────────────────────

def trend_strength_score(zigzag: List[ZigzagPoint], direction: int) -> Tuple[float, bool, bool, List[str]]:
    """
    Score trend strength 0-10 and detect strengthening/weakening.
    Returns: (score, is_strengthening, is_weakening, warning_signs)
    """
    if len(zigzag) < 4 or direction == 0:
        return 5.0, False, False, []

    swings = swing_metrics(zigzag)
    if swings.empty:
        return 5.0, False, False, []

    warnings = []
    score = 5.0

    # Split into impulse and pullback swings
    if direction == 1:
        impulse_swings = swings[swings['direction'] == 'up']
        pullback_swings = swings[swings['direction'] == 'down']
    else:
        impulse_swings = swings[swings['direction'] == 'down']
        pullback_swings = swings[swings['direction'] == 'up']

    if len(impulse_swings) < 2:
        return score, False, False, warnings

    # 1. Impulse magnitude trend
    imp_mags = impulse_swings['magnitude'].values
    if len(imp_mags) >= 2:
        if imp_mags[-1] > imp_mags[-2]:
            score += 1.5  # strengthening impulses
        elif imp_mags[-1] < imp_mags[-2] * 0.75:
            score -= 2.0
            warnings.append("Impulse legs shrinking — trend weakening")

    # 2. Impulse velocity trend
    imp_vels = impulse_swings['velocity'].values
    if len(imp_vels) >= 2:
        if imp_vels[-1] < imp_vels[-2] * 0.8:
            score -= 1.5
            warnings.append("Impulse velocity declining")

    # 3. Pullback depth ratio (pullback / preceding impulse)
    pb_mags = pullback_swings['magnitude'].values
    imp_for_pb = impulse_swings['magnitude'].values[:len(pb_mags)]
    if len(pb_mags) > 0 and len(imp_for_pb) > 0:
        depths = pb_mags / np.maximum(imp_for_pb, 1e-9)
        latest_depth = depths[-1]
        if latest_depth > 0.618:
            score -= 2.0
            warnings.append(f"Pullback depth {latest_depth:.1%} > 61.8% — possible reversal")
        elif latest_depth < 0.382:
            score += 1.0  # shallow pullback = strong trend

        if len(depths) >= 2 and depths[-1] > depths[-2] * 1.3:
            score -= 1.0
            warnings.append("Pullback depth increasing — trend losing momentum")

    # 4. Duration comparison (impulse should take longer than pullback)
    if not pullback_swings.empty and not impulse_swings.empty:
        last_pb_dur = pullback_swings['duration'].values[-1]
        last_imp_dur = impulse_swings['duration'].values[-1]
        if last_pb_dur > last_imp_dur:
            score -= 1.0
            warnings.append("Pullback lasting longer than preceding impulse")

    score = np.clip(score, 0.0, 10.0)
    is_strengthening = score > 7.0 and len(warnings) == 0
    is_weakening = score < 4.0 or len(warnings) >= 2
    return score, is_strengthening, is_weakening, warnings


# ─── RANGE DETECTION ─────────────────────────────────────────────────────────

def is_in_range(zigzag: List[ZigzagPoint], close: np.ndarray,
                atr: float, range_lookback: int = 20) -> Tuple[bool, float, float]:
    """
    Detect if market is in a trading range.
    Returns: (is_range, range_high, range_low)
    """
    if len(zigzag) < 4:
        return False, np.nan, np.nan

    recent = zigzag[-8:]
    highs = [p.price for p in recent if p.pivot_type == 'high']
    lows = [p.price for p in recent if p.pivot_type == 'low']

    if not highs or not lows:
        return False, np.nan, np.nan

    range_high = max(highs)
    range_low = min(lows)
    range_height = range_high - range_low

    # Range qualification: height < 3×ATR and no clear HH/HL sequence
    is_range_height = range_height < 3 * atr if atr > 0 else True

    # Check last close bars are within range bounds
    recent_close = close[-range_lookback:]
    pct_inside = np.mean((recent_close >= range_low) & (recent_close <= range_high))
    price_contained = pct_inside > 0.75

    return is_range_height and price_contained, range_high, range_low


# ─── PULLBACK ANALYSIS ────────────────────────────────────────────────────────

def analyze_pullback(zigzag: List[ZigzagPoint], direction: int,
                     current_price: float, atr: float
                     ) -> Tuple[bool, PullbackType, float, Optional[float]]:
    """
    Determine if we are currently in a pullback and classify it.
    Returns: (in_pullback, pullback_type, depth_pct, ab_cd_target)
    """
    if len(zigzag) < 4 or direction == 0:
        return False, PullbackType.NONE, 0.0, None

    recent = zigzag[-6:]
    if direction == 1:
        # Uptrend: pullback = downswing after an upswing
        last_high = max((p for p in recent if p.pivot_type == 'high'), key=lambda x: x.index, default=None)
        last_low = max((p for p in recent if p.pivot_type == 'low'), key=lambda x: x.index, default=None)
        if last_high is None or last_low is None:
            return False, PullbackType.NONE, 0.0, None

        if last_low.index > last_high.index:
            # We are in a pullback (last movement is downward)
            in_pullback = current_price < last_high.price

            # Find the impulse before this pullback
            highs_sorted = sorted([p for p in recent if p.pivot_type == 'high'], key=lambda x: x.index)
            if len(highs_sorted) >= 2:
                prior_impulse = highs_sorted[-1].price - min(
                    [p.price for p in recent if p.pivot_type == 'low' and p.index < highs_sorted[-1].index],
                    default=highs_sorted[-1].price
                )
            else:
                prior_impulse = last_high.price - last_low.price

            current_pb_depth = last_high.price - current_price
            depth_pct = current_pb_depth / prior_impulse if prior_impulse > 0 else 0.0

            # Detect complex pullback: look for second leg
            lows_in_pb = [p for p in recent if p.pivot_type == 'low' and p.index > last_high.index]
            if len(lows_in_pb) >= 2:
                pb_type = PullbackType.COMPLEX
                # AB=CD: second leg ≈ first leg
                leg_a = lows_in_pb[0].price
                leg_b = max([p.price for p in recent
                             if p.pivot_type == 'high' and lows_in_pb[0].index < p.index < lows_in_pb[-1].index],
                            default=leg_a)
                leg_c = lows_in_pb[-1].price
                ab_length = leg_b - leg_a
                ab_cd_target = leg_c + ab_length  # projected bounce from second leg
            else:
                pb_type = PullbackType.SIMPLE if depth_pct < 0.65 else PullbackType.FAILED
                ab_cd_target = None

            return in_pullback, pb_type, depth_pct, ab_cd_target
    else:
        # Downtrend: pullback = upswing after a downswing
        last_low = max((p for p in recent if p.pivot_type == 'low'), key=lambda x: x.index, default=None)
        last_high = max((p for p in recent if p.pivot_type == 'high'), key=lambda x: x.index, default=None)
        if last_high is None or last_low is None:
            return False, PullbackType.NONE, 0.0, None

        if last_high.index > last_low.index:
            in_pullback = current_price > last_low.price
            prior_impulse = last_low.price - max(
                [p.price for p in recent if p.pivot_type == 'high' and p.index < last_low.index],
                default=last_low.price
            )
            prior_impulse = abs(prior_impulse)
            current_pb_depth = current_price - last_low.price
            depth_pct = current_pb_depth / prior_impulse if prior_impulse > 0 else 0.0

            highs_in_pb = [p for p in recent if p.pivot_type == 'high' and p.index > last_low.index]
            if len(highs_in_pb) >= 2:
                pb_type = PullbackType.COMPLEX
                ab_cd_target = None  # simplified; compute similarly to long case
            else:
                pb_type = PullbackType.SIMPLE if depth_pct < 0.65 else PullbackType.FAILED
                ab_cd_target = None

            return in_pullback, pb_type, depth_pct, ab_cd_target

    return False, PullbackType.NONE, 0.0, None


# ─── WYCKOFF PHASE DETECTION ──────────────────────────────────────────────────

def detect_wyckoff_phase(zigzag: List[ZigzagPoint], direction: int,
                         is_range: bool, range_high: float, range_low: float,
                         close: np.ndarray, atr: float) -> MarketState:
    """
    Refine market state within ranges using Wyckoff logic.
    Accumulation: range following downtrend with springs at support.
    Distribution: range following uptrend with upthrusts at resistance.
    """
    if not is_range:
        return MarketState.RANGE

    if len(close) < 5:
        return MarketState.RANGE

    recent_close = close[-5:]
    # Check if prior trend was up (distribution) or down (accumulation)
    if len(zigzag) >= 6:
        early_prices = [z.price for z in zigzag[-6:-3]]
        late_prices = [z.price for z in zigzag[-3:]]
        prior_was_up = np.mean(late_prices) < np.mean(early_prices)  # now at top of down move
        prior_was_down = np.mean(late_prices) > np.mean(early_prices)  # now at top of up move

        # Spring detection (recent close near or below support)
        near_support = np.any(recent_close < range_low + 0.5 * atr)
        # Upthrust detection (recent close near or above resistance)
        near_resistance = np.any(recent_close > range_high - 0.5 * atr)

        if prior_was_down and near_support:
            return MarketState.ACCUMULATION
        if prior_was_up and near_resistance:
            return MarketState.DISTRIBUTION

    return MarketState.RANGE


# ─── MASTER REGIME DETECTOR ───────────────────────────────────────────────────

def detect_market_regime(zigzag: List[ZigzagPoint], close: np.ndarray,
                         high: np.ndarray, low: np.ndarray,
                         atr: np.ndarray, kc_upper: np.ndarray,
                         kc_lower: np.ndarray, macd_div: np.ndarray
                         ) -> MarketRegime:
    """
    Main entry point: synthesize all structural analysis into a MarketRegime.
    """
    if len(close) < 20 or len(zigzag) < 4:
        return MarketRegime(
            state=MarketState.TRANSITION, confidence=0.0, trend_direction=0,
            strength_score=5.0, is_strengthening=False, is_weakening=False,
            in_pullback=False, pullback_type=PullbackType.NONE,
            pullback_depth_pct=0.0, ab_cd_target=None,
            key_level_high=None, key_level_low=None,
            last_impulse_size=0.0, warning_signs=["Insufficient data"]
        )

    current_atr = atr[-1] if not np.isnan(atr[-1]) else np.nanmean(atr[-20:])
    current_price = close[-1]

    # 1. Dow Theory direction + confidence
    direction, dow_confidence = dow_theory_trend(zigzag)

    # 2. Range check
    in_range, range_high, range_low = is_in_range(zigzag, close, current_atr)

    # 3. Trend strength
    strength, is_strengthening, is_weakening, warnings = trend_strength_score(zigzag, direction)

    # 4. Pullback analysis
    in_pullback, pb_type, pb_depth, ab_cd = analyze_pullback(
        zigzag, direction, current_price, current_atr)

    # 5. Keltner position (overextension check)
    kc_pos = (current_price - kc_lower[-1]) / (kc_upper[-1] - kc_lower[-1]) \
        if not (np.isnan(kc_upper[-1]) or np.isnan(kc_lower[-1])) else 0.5

    # 6. Determine market state
    if in_range:
        state = detect_wyckoff_phase(
            zigzag, direction, True, range_high, range_low, close, current_atr)
    elif direction == 1:
        if strength >= 7 and not is_weakening:
            state = MarketState.STRONG_UPTREND
        elif strength >= 4:
            state = MarketState.WEAK_UPTREND
        else:
            state = MarketState.TRANSITION
    elif direction == -1:
        if strength >= 7 and not is_weakening:
            state = MarketState.STRONG_DOWNTREND
        elif strength >= 4:
            state = MarketState.WEAK_DOWNTREND
        else:
            state = MarketState.TRANSITION
    else:
        state = MarketState.RANGE

    # 7. Detect breakout (recent range break)
    if len(zigzag) >= 4:
        recent_highs = [p.price for p in zigzag[-8:] if p.pivot_type == 'high']
        recent_lows = [p.price for p in zigzag[-8:] if p.pivot_type == 'low']
        if recent_highs and recent_lows:
            prev_range_high = sorted(recent_highs)[-1]
            prev_range_low = sorted(recent_lows)[0]
            bars_ago_high = next(
                (len(close) - 1 - p.index for p in reversed(zigzag) if p.pivot_type == 'high'), 999)
            bars_ago_low = next(
                (len(close) - 1 - p.index for p in reversed(zigzag) if p.pivot_type == 'low'), 999)
            if current_price > prev_range_high and bars_ago_high <= 5:
                state = MarketState.BREAKOUT_LONG
            elif current_price < prev_range_low and bars_ago_low <= 5:
                state = MarketState.BREAKOUT_SHORT

    # 8. Key levels
    ph_list = [p for p in zigzag if p.pivot_type == 'high']
    pl_list = [p for p in zigzag if p.pivot_type == 'low']
    key_high = ph_list[-1].price if ph_list else None
    key_low = pl_list[-1].price if pl_list else None

    # 9. Last impulse size
    swings = swing_metrics(zigzag)
    last_imp = 0.0
    if not swings.empty:
        if direction == 1:
            up_swings = swings[swings['direction'] == 'up']
            last_imp = up_swings['magnitude'].values[-1] if not up_swings.empty else 0.0
        else:
            down_swings = swings[swings['direction'] == 'down']
            last_imp = down_swings['magnitude'].values[-1] if not down_swings.empty else 0.0

    # 10. Confidence adjustment
    confidence = dow_confidence
    if is_weakening:
        confidence *= 0.7
    if in_range:
        confidence = min(confidence, 0.6)

    return MarketRegime(
        state=state,
        confidence=min(confidence, 1.0),
        trend_direction=direction,
        strength_score=strength,
        is_strengthening=is_strengthening,
        is_weakening=is_weakening,
        in_pullback=in_pullback,
        pullback_type=pb_type,
        pullback_depth_pct=pb_depth,
        ab_cd_target=ab_cd,
        key_level_high=key_high,
        key_level_low=key_low,
        last_impulse_size=last_imp,
        warning_signs=warnings
    )
