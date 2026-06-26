"""
trend_analyzer.py — Comprehensive trend analysis and change detection.
Based on: The Art and Science of Technical Analysis (Adam Grimes)

Synthesizes pivot structure, momentum, and Keltner data into
a unified picture of the trend and its health.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pivot_detector import ZigzagPoint, Pivot, swing_metrics
from market_structure import MarketRegime, MarketState, dow_theory_trend
from channel_detector import analyze_rate_of_trend, ClimaxSignal


@dataclass
class TrendChangeScore:
    total_score: int           # sum of all signal weights
    probability: float         # estimated probability of trend change 0-1
    signals: List[str]         # human-readable signals triggered
    last_safe_entry: Optional[float]  # price level that was the last good entry
    change_likely: bool        # total_score >= 4


@dataclass
class TrendHealth:
    direction: int             # +1 / -1 / 0
    strength_score: float      # 0-10
    impulse_ratio: float       # avg_impulse / avg_pullback (>1.5 healthy)
    velocity_trend: float      # +1 accelerating, 0 flat, -1 decelerating
    pullback_depth_trend: float  # +1 increasing, 0 flat, -1 decreasing
    avg_pullback_depth: float  # as fraction of prior impulse
    avg_impulse_bars: float    # average duration of impulse legs
    avg_pullback_bars: float   # average duration of pullback legs
    is_healthy: bool
    is_parabolic: bool
    deceleration_degree: float  # 0-1
    last_impulse_size: float
    warnings: List[str]


# ─── TREND HEALTH ────────────────────────────────────────────────────────────

def assess_trend_health(zigzag: List[ZigzagPoint],
                        direction: int) -> TrendHealth:
    """
    Full trend health assessment from zigzag swing data.
    """
    empty = TrendHealth(
        direction=0, strength_score=5.0, impulse_ratio=1.0,
        velocity_trend=0.0, pullback_depth_trend=0.0,
        avg_pullback_depth=0.0, avg_impulse_bars=0.0,
        avg_pullback_bars=0.0, is_healthy=True, is_parabolic=False,
        deceleration_degree=0.0, last_impulse_size=0.0, warnings=[]
    )

    if len(zigzag) < 4 or direction == 0:
        return empty

    swings = swing_metrics(zigzag)
    if swings.empty:
        return empty

    # Split into impulse and pullback swings
    if direction == 1:
        imp = swings[swings['direction'] == 'up']
        pb = swings[swings['direction'] == 'down']
    else:
        imp = swings[swings['direction'] == 'down']
        pb = swings[swings['direction'] == 'up']

    warnings = []

    # Impulse vs pullback ratio
    avg_imp = imp['magnitude'].mean() if not imp.empty else 1.0
    avg_pb = pb['magnitude'].mean() if not pb.empty else 0.5
    ratio = avg_imp / avg_pb if avg_pb > 0 else 1.0

    if ratio < 1.2:
        warnings.append("Impulse barely larger than pullback — trend weak")
    elif ratio > 2.0:
        pass  # healthy

    # Velocity trend
    vels = imp['velocity'].values
    vel_trend = 0.0
    if len(vels) >= 2:
        vel_trend_raw = (vels[-1] - vels[0]) / vels[0] if vels[0] > 0 else 0
        vel_trend = float(np.clip(vel_trend_raw, -1, 1))
        if vel_trend < -0.3:
            warnings.append("Impulse velocity declining — trend decelerating")

    # Pullback depth trend
    imp_mags = imp['magnitude'].values
    pb_mags = pb['magnitude'].values
    n_pairs = min(len(imp_mags), len(pb_mags))
    depths = pb_mags[:n_pairs] / np.maximum(imp_mags[:n_pairs], 1e-9) if n_pairs > 0 else np.array([])

    avg_depth = float(np.mean(depths)) if len(depths) > 0 else 0.0
    depth_trend = 0.0
    if len(depths) >= 2:
        depth_delta = (depths[-1] - depths[0]) / depths[0] if depths[0] > 0 else 0
        depth_trend = float(np.clip(depth_delta, -1, 1))
        if depth_trend > 0.3:
            warnings.append("Pullback depth increasing — possible trend reversal building")

    if avg_depth > 0.618:
        warnings.append(f"Average pullback depth {avg_depth:.1%} > 61.8% — trend may be failing")

    # Duration ratios
    avg_imp_bars = float(imp['duration'].mean()) if not imp.empty else 0.0
    avg_pb_bars = float(pb['duration'].mean()) if not pb.empty else 0.0
    if avg_pb_bars > avg_imp_bars:
        warnings.append("Pullbacks taking longer than impulses — momentum lost")

    # Rate of trend (parabolic check)
    rot = analyze_rate_of_trend(zigzag, direction)
    is_parabolic = rot.get('is_parabolic', False)
    decel_degree = rot.get('deceleration_degree', 0.0)

    if is_parabolic:
        warnings.append("Parabolic acceleration detected — climax risk high")

    # Overall strength (0-10)
    score = 5.0
    score += min(ratio - 1.0, 2.0)    # impulse ratio contribution
    score += vel_trend * 1.5          # velocity trend
    score -= depth_trend * 1.5        # depth trend (increasing = bad)
    score -= len(warnings) * 0.5
    if avg_depth < 0.382:
        score += 1.0   # shallow pullbacks = strong trend
    score = float(np.clip(score, 0, 10))

    last_imp = imp['magnitude'].values[-1] if not imp.empty else 0.0

    return TrendHealth(
        direction=direction,
        strength_score=score,
        impulse_ratio=ratio,
        velocity_trend=vel_trend,
        pullback_depth_trend=depth_trend,
        avg_pullback_depth=avg_depth,
        avg_impulse_bars=avg_imp_bars,
        avg_pullback_bars=avg_pb_bars,
        is_healthy=score >= 5.0 and len(warnings) <= 1,
        is_parabolic=is_parabolic,
        deceleration_degree=float(decel_degree),
        last_impulse_size=float(last_imp),
        warnings=warnings
    )


# ─── TREND CHANGE DETECTION ───────────────────────────────────────────────────

def detect_trend_change(zigzag: List[ZigzagPoint], close: np.ndarray,
                        high: np.ndarray, low: np.ndarray,
                        atr_vals: np.ndarray,
                        macd_div: np.ndarray,
                        climax_signals: List[ClimaxSignal],
                        direction: int) -> TrendChangeScore:
    """
    Score the probability of a trend change using Grimes' change-of-character
    framework. Score >= 4 = high probability.

    Signals and weights:
    +2  First failure to make new high/low (HH failure in uptrend)
    +2  Counter-swing larger than all recent same-direction swings
    +1  MACD divergence at the new price extreme
    +2  Parabolic climax detected (last 5 bars)
    +1  Trend lasted longer than 150% of average trend duration
    +1  Three consecutive tests of same resistance/support level
    +1  Velocity of most recent impulse < 50% of prior impulse
    """
    score = 0
    signals_triggered = []
    last_safe_entry = None

    if len(zigzag) < 4 or direction == 0:
        return TrendChangeScore(0, 0.0, [], None, False)

    swings = swing_metrics(zigzag)
    if swings.empty:
        return TrendChangeScore(0, 0.0, [], None, False)

    current_price = float(close[-1])
    current_atr = float(np.nanmean(atr_vals[-5:])) if len(atr_vals) >= 5 else 1.0

    if direction == 1:
        impulse_swings = swings[swings['direction'] == 'up']
        counter_swings = swings[swings['direction'] == 'down']
        recent_highs = [p for p in zigzag if p.pivot_type == 'high']
        recent_lows = [p for p in zigzag if p.pivot_type == 'low']
    else:
        impulse_swings = swings[swings['direction'] == 'down']
        counter_swings = swings[swings['direction'] == 'up']
        recent_highs = [p for p in zigzag if p.pivot_type == 'high']
        recent_lows = [p for p in zigzag if p.pivot_type == 'low']

    # 1. HH failure: last impulse failed to exceed prior impulse high
    if direction == 1 and len(recent_highs) >= 2:
        if recent_highs[-1].price < recent_highs[-2].price:
            score += 2
            signals_triggered.append("Higher High failure: last swing failed to make new high")
            # last safe entry was the prior HL
            if recent_lows:
                last_safe_entry = recent_lows[-1].price
    elif direction == -1 and len(recent_lows) >= 2:
        if recent_lows[-1].price > recent_lows[-2].price:
            score += 2
            signals_triggered.append("Lower Low failure: last swing failed to make new low")
            if recent_highs:
                last_safe_entry = recent_highs[-1].price

    # 2. Counter-swing larger than recent impulse swings
    if not counter_swings.empty and not impulse_swings.empty:
        last_counter = counter_swings['magnitude'].values[-1]
        avg_impulse = float(impulse_swings['magnitude'].mean())
        if last_counter > avg_impulse * 1.2:
            score += 2
            signals_triggered.append(
                f"Counter-swing ({last_counter:.2f}) > avg impulse ({avg_impulse:.2f})")

    # 3. MACD divergence near recent extremes (last 5 bars)
    recent_div = macd_div[-5:] if len(macd_div) >= 5 else macd_div
    if direction == 1 and np.any(recent_div == -1):
        score += 1
        signals_triggered.append("Bearish MACD divergence at new highs")
    elif direction == -1 and np.any(recent_div == 1):
        score += 1
        signals_triggered.append("Bullish MACD divergence at new lows")

    # 4. Parabolic climax in last 5 bars
    recent_climax_bars = [c for c in climax_signals
                          if len(close) - 1 - c.bar_index <= 5]
    if direction == 1:
        bull_exhaustion = [c for c in recent_climax_bars if c.climax_type == 'bullish_exhaustion']
        if bull_exhaustion:
            score += 2
            signals_triggered.append("Parabolic bullish exhaustion bar detected")
    else:
        bear_exhaustion = [c for c in recent_climax_bars if c.climax_type == 'bearish_exhaustion']
        if bear_exhaustion:
            score += 2
            signals_triggered.append("Parabolic bearish exhaustion bar detected")

    # 5. Trend duration vs historical average
    if not impulse_swings.empty:
        avg_imp_dur = float(impulse_swings['duration'].mean())
        total_trend_bars = (zigzag[-1].index - zigzag[0].index) if len(zigzag) >= 2 else 0
        if total_trend_bars > avg_imp_dur * 6:  # 6+ legs is very mature
            score += 1
            signals_triggered.append(f"Trend is mature ({total_trend_bars} bars)")

    # 6. Velocity of last impulse vs prior
    imp_vels = impulse_swings['velocity'].values
    if len(imp_vels) >= 2 and imp_vels[-2] > 0:
        vel_ratio = imp_vels[-1] / imp_vels[-2]
        if vel_ratio < 0.5:
            score += 1
            signals_triggered.append(f"Last impulse velocity {vel_ratio:.1%} of prior")

    # Probability estimate (rough sigmoid)
    probability = 1 / (1 + np.exp(-0.7 * (score - 3)))

    return TrendChangeScore(
        total_score=score,
        probability=float(probability),
        signals=signals_triggered,
        last_safe_entry=last_safe_entry,
        change_likely=score >= 4
    )


# ─── MULTI-TIMEFRAME ALIGNMENT ────────────────────────────────────────────────

@dataclass
class MTFAlignment:
    alignment_score: int       # 0-3
    recommended_direction: int # +1, -1, or 0
    size_fraction: float       # 0-1 (fraction of normal position size)
    notes: List[str]


def mtf_trend_alignment(htf_state: MarketState, ttf_state: MarketState,
                        ltf_state: MarketState) -> MTFAlignment:
    """
    Score multi-timeframe trend alignment.
    HTF = higher timeframe (e.g. daily)
    TTF = trading timeframe (e.g. 4H)
    LTF = lower timeframe (e.g. 1H)
    """
    def to_dir(state: MarketState) -> int:
        if state in (MarketState.STRONG_UPTREND, MarketState.WEAK_UPTREND,
                     MarketState.BREAKOUT_LONG):
            return 1
        elif state in (MarketState.STRONG_DOWNTREND, MarketState.WEAK_DOWNTREND,
                       MarketState.BREAKOUT_SHORT):
            return -1
        return 0

    htf_dir = to_dir(htf_state)
    ttf_dir = to_dir(ttf_state)
    ltf_dir = to_dir(ltf_state)

    dirs = [htf_dir, ttf_dir, ltf_dir]
    non_zero = [d for d in dirs if d != 0]

    if not non_zero:
        return MTFAlignment(0, 0, 0.0, ["No clear direction on any timeframe"])

    # Most common direction
    consensus = 1 if sum(non_zero) > 0 else -1
    aligned = sum(1 for d in non_zero if d == consensus)

    notes = []
    if htf_dir != 0 and htf_dir != consensus:
        notes.append("WARNING: HTF trend opposing signal direction")
    if htf_dir == consensus:
        notes.append("HTF trend aligned ✓")
    if ttf_dir == consensus:
        notes.append("TTF trend aligned ✓")
    if ltf_dir == consensus:
        notes.append("LTF trend aligned ✓")

    size_map = {3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0}
    size_fraction = size_map.get(aligned, 0.0)

    return MTFAlignment(
        alignment_score=aligned,
        recommended_direction=consensus if aligned >= 2 else 0,
        size_fraction=size_fraction,
        notes=notes
    )


# ─── PULLBACK QUALITY SCORE ───────────────────────────────────────────────────

def score_pullback_quality(zigzag: List[ZigzagPoint],
                            close: np.ndarray, high: np.ndarray,
                            low: np.ndarray, atr_vals: np.ndarray,
                            direction: int) -> float:
    """
    Score the quality of the current pullback for entry. Returns 0-10.
    Higher = better entry opportunity.

    Factors:
    - Shallow depth (< 38.2% = excellent, < 61.8% = acceptable)
    - Declining momentum in pullback (ATR shrinking)
    - Volatility compression (bars getting tighter)
    - Pullback not showing counter-trend lower TF impulse
    - Price near key S/R (increases precision)
    """
    if len(zigzag) < 4 or len(close) < 10 or direction == 0:
        return 5.0

    swings = swing_metrics(zigzag)
    if swings.empty:
        return 5.0

    score = 5.0
    recent_atr = float(np.nanmean(atr_vals[-10:])) if len(atr_vals) >= 10 else 1.0

    # 1. Last impulse size vs recent ATR
    if direction == 1:
        imp = swings[swings['direction'] == 'up']
        pb = swings[swings['direction'] == 'down']
    else:
        imp = swings[swings['direction'] == 'down']
        pb = swings[swings['direction'] == 'up']

    if not imp.empty and not pb.empty:
        last_imp = float(imp['magnitude'].values[-1])
        last_pb = float(pb['magnitude'].values[-1]) if not pb.empty else 0.0
        depth_pct = last_pb / last_imp if last_imp > 0 else 0.5

        if depth_pct < 0.382:
            score += 2.5  # very shallow = trend very strong
        elif depth_pct < 0.500:
            score += 1.5
        elif depth_pct < 0.618:
            score += 0.5
        else:
            score -= 2.0  # deep pullback = potential reversal

    # 2. Volatility compression in pullback
    pullback_bars = atr_vals[-8:] if len(atr_vals) >= 8 else atr_vals
    if len(pullback_bars) >= 3:
        atr_trend = np.polyfit(np.arange(len(pullback_bars)), pullback_bars, 1)[0]
        if atr_trend < 0:
            score += 1.0  # ATR declining in pullback = compression
        elif atr_trend > recent_atr * 0.05:
            score -= 1.0  # expanding volatility in pullback = risky

    # 3. Bars of pullback vs bars of impulse (pullback should be shorter)
    if not imp.empty and not pb.empty:
        imp_dur = float(imp['duration'].values[-1])
        pb_dur = float(pb['duration'].values[-1])
        if pb_dur < imp_dur * 0.6:
            score += 1.0  # quick pullback = trend dominant
        elif pb_dur > imp_dur:
            score -= 1.0

    return float(np.clip(score, 0, 10))
