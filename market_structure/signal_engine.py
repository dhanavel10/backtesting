"""
signal_engine.py — Trade signal generation for all six Grimes trade types.
Based on: The Art and Science of Technical Analysis (Adam Grimes)

Six trade types:
1. Pullback (simple & complex)
2. Failure Test (spring/upthrust)
3. Breakout
4. The Anti (trend termination)
5. Failed Breakout
6. Breakout on first pullback
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from market_structure import MarketRegime, MarketState, PullbackType
from sr_zones import SRZone, ZoneState, ZoneType, detect_spring, detect_upthrust
from trend_analyzer import TrendHealth, TrendChangeScore, score_pullback_quality
from channel_detector import ClimaxSignal, keltner_analysis


@dataclass
class TradeSignal:
    signal_type: str           # pullback, failure_test, breakout, anti, failed_breakout, pb_after_bo
    direction: str             # long or short
    entry_price: float
    stop_price: float
    target_1: float            # 1× risk
    target_2: float            # pattern-based (swing high/low)
    target_3: float            # measured move
    risk_per_unit: float       # abs(entry - stop)
    rr_1: float                # reward/risk to target_1
    rr_2: float
    rr_3: float
    confidence: float          # 0-1
    market_state: MarketState
    pullback_quality: float    # 0-10 (for pullback signals)
    notes: List[str]
    bar_index: int
    # Risk management
    position_size: float = 0.0
    risk_fraction: float = 0.02


def _make_signal(signal_type: str, direction: str,
                 entry: float, stop: float,
                 t1: float, t2: float, t3: float,
                 confidence: float, state: MarketState,
                 bar_index: int, notes: List[str],
                 pb_quality: float = 5.0) -> TradeSignal:
    risk = abs(entry - stop)
    if risk == 0:
        return None

    def rr(target): return abs(target - entry) / risk

    return TradeSignal(
        signal_type=signal_type,
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_1=t1,
        target_2=t2,
        target_3=t3,
        risk_per_unit=risk,
        rr_1=rr(t1),
        rr_2=rr(t2),
        rr_3=rr(t3),
        confidence=float(np.clip(confidence, 0, 1)),
        market_state=state,
        pullback_quality=pb_quality,
        notes=notes,
        bar_index=bar_index
    )


# ─── CONFLUENCE SCORING ───────────────────────────────────────────────────────

def score_confluence(signal: TradeSignal, htf_state: MarketState,
                     ltf_momentum: int, sr_proximity: bool,
                     macd_div: int, vol_contracting: bool,
                     mtf_score: int) -> float:
    """
    Adjust signal confidence based on confirming/opposing factors.
    Returns updated confidence 0-1.
    """
    conf = signal.confidence
    direction_int = 1 if signal.direction == 'long' else -1

    # HTF alignment
    htf_up = htf_state in (MarketState.STRONG_UPTREND, MarketState.WEAK_UPTREND,
                           MarketState.BREAKOUT_LONG)
    htf_down = htf_state in (MarketState.STRONG_DOWNTREND, MarketState.WEAK_DOWNTREND,
                             MarketState.BREAKOUT_SHORT)

    if direction_int == 1 and htf_up:
        conf += 0.15
    elif direction_int == -1 and htf_down:
        conf += 0.15
    elif direction_int == 1 and htf_down:
        conf -= 0.15   # going against HTF
    elif direction_int == -1 and htf_up:
        conf -= 0.15

    # LTF momentum alignment
    if ltf_momentum == direction_int:
        conf += 0.10
    elif ltf_momentum == -direction_int:
        conf -= 0.05

    # At significant S/R zone
    if sr_proximity:
        conf += 0.10

    # MACD divergence supporting
    if direction_int == 1 and macd_div == 1:
        conf += 0.10
    elif direction_int == -1 and macd_div == -1:
        conf += 0.10

    # Volatility contraction before entry
    if vol_contracting:
        conf += 0.05

    # MTF alignment bonus
    conf += (mtf_score - 2) * 0.05  # +0.05 for 3-aligned, 0 for 2-aligned, -0.05 for 1

    return float(np.clip(conf, 0, 1))


# ─── SIGNAL 1: PULLBACK ───────────────────────────────────────────────────────

def scan_pullback(regime: MarketRegime, health: TrendHealth,
                  high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  atr_vals: np.ndarray, zones: List[SRZone],
                  bar_index: int, atr_entry_buffer: float = 0.1) -> Optional[TradeSignal]:
    """
    Generate pullback signal for long (uptrend) or short (downtrend).
    Setup: confirmed trend + current pullback that is shallow and compressing.
    """
    if not regime.in_pullback:
        return None
    if regime.pullback_type == PullbackType.FAILED:
        return None

    direction = regime.trend_direction
    if direction == 0:
        return None

    # Must be in a trend state
    valid_states_long = {MarketState.STRONG_UPTREND, MarketState.WEAK_UPTREND}
    valid_states_short = {MarketState.STRONG_DOWNTREND, MarketState.WEAK_DOWNTREND}
    if direction == 1 and regime.state not in valid_states_long:
        return None
    if direction == -1 and regime.state not in valid_states_short:
        return None

    # Depth check
    if regime.pullback_depth_pct > 0.618:
        return None

    current_price = float(close[-1])
    current_atr = float(atr_vals[-1]) if not np.isnan(atr_vals[-1]) else float(np.nanmean(atr_vals[-10:]))

    pb_quality = score_pullback_quality([], close, high, low, atr_vals, direction)
    # (zigzag not passed here; caller should pass it for full score)

    notes = [f"Trend: {'UP' if direction == 1 else 'DOWN'}, "
             f"Pullback: {regime.pullback_type.value}, "
             f"Depth: {regime.pullback_depth_pct:.1%}"]

    if direction == 1:
        # Buy the pullback
        if regime.key_level_low is None:
            return None
        entry = current_price + atr_entry_buffer * current_atr
        stop = regime.key_level_low - 0.25 * current_atr
        t1 = entry + abs(entry - stop)
        t2 = (regime.key_level_high or entry + 2 * abs(entry - stop))
        t3 = (regime.ab_cd_target or t2 * 1.2)
        base_conf = 0.45 + min(health.strength_score / 10 * 0.25, 0.25)
        if regime.state == MarketState.STRONG_UPTREND:
            base_conf += 0.1
    else:
        # Sell the pullback
        if regime.key_level_high is None:
            return None
        entry = current_price - atr_entry_buffer * current_atr
        stop = regime.key_level_high + 0.25 * current_atr
        t1 = entry - abs(stop - entry)
        t2 = (regime.key_level_low or entry - 2 * abs(stop - entry))
        t3 = (regime.ab_cd_target or t2 * 0.8)
        base_conf = 0.45 + min(health.strength_score / 10 * 0.25, 0.25)
        if regime.state == MarketState.STRONG_DOWNTREND:
            base_conf += 0.1

    signal_type = ("complex_pullback" if regime.pullback_type == PullbackType.COMPLEX
                   else "pullback")
    direction_str = 'long' if direction == 1 else 'short'

    return _make_signal(
        signal_type, direction_str, entry, stop, t1, t2, t3,
        base_conf, regime.state, bar_index, notes, pb_quality
    )


# ─── SIGNAL 2: FAILURE TEST (SPRING / UPTHRUST) ──────────────────────────────

def scan_failure_test(regime: MarketRegime, springs: List[dict],
                      upthrusts: List[dict], high: np.ndarray,
                      low: np.ndarray, close: np.ndarray,
                      atr_vals: np.ndarray, bar_index: int) -> List[TradeSignal]:
    """Generate failure test signals from detected springs/upthrusts."""
    signals = []
    current_atr = float(np.nanmean(atr_vals[-5:])) if len(atr_vals) >= 5 else 1.0

    for s in springs:
        if s['bar_index'] < bar_index - 2:
            continue  # too old
        entry = s['entry_trigger'] + 0.1 * current_atr
        stop = s['stop']
        risk = abs(entry - stop)
        if risk < 0.1 * current_atr:
            continue
        t1 = entry + risk
        t2 = entry + risk * 2
        t3 = entry + risk * 3

        notes = [f"Spring at {s['zone_bottom']:.2f}, probe to {s['probe_low']:.2f}",
                 f"Spring strength: {s['strength']:.2f}"]

        conf = 0.50 + s['strength'] * 0.2
        if regime.state == MarketState.ACCUMULATION:
            conf += 0.10
        if regime.state == MarketState.STRONG_DOWNTREND:
            conf -= 0.15  # going against the trend

        sig = _make_signal('failure_test', 'long', entry, stop, t1, t2, t3,
                           conf, regime.state, bar_index, notes)
        if sig:
            signals.append(sig)

    for u in upthrusts:
        if u['bar_index'] < bar_index - 2:
            continue
        entry = u['entry_trigger'] - 0.1 * current_atr
        stop = u['stop']
        risk = abs(stop - entry)
        if risk < 0.1 * current_atr:
            continue
        t1 = entry - risk
        t2 = entry - risk * 2
        t3 = entry - risk * 3

        notes = [f"Upthrust at {u['zone_top']:.2f}, probe to {u['probe_high']:.2f}",
                 f"Upthrust strength: {u['strength']:.2f}"]

        conf = 0.50 + u['strength'] * 0.2
        if regime.state == MarketState.DISTRIBUTION:
            conf += 0.10
        if regime.state == MarketState.STRONG_UPTREND:
            conf -= 0.15

        sig = _make_signal('failure_test', 'short', entry, stop, t1, t2, t3,
                           conf, regime.state, bar_index, notes)
        if sig:
            signals.append(sig)

    return signals


# ─── SIGNAL 3: BREAKOUT ───────────────────────────────────────────────────────

def scan_breakout(regime: MarketRegime, zones: List[SRZone],
                  high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  atr_vals: np.ndarray, bar_index: int) -> Optional[TradeSignal]:
    """
    Generate breakout signal from a trading range.
    Pre-breakout: higher lows into resistance (ascending) or
    volatility compression before level.
    """
    if regime.state not in (MarketState.RANGE, MarketState.ACCUMULATION,
                             MarketState.DISTRIBUTION, MarketState.BREAKOUT_LONG,
                             MarketState.BREAKOUT_SHORT):
        return None

    current_price = float(close[-1])
    current_atr = float(np.nanmean(atr_vals[-5:])) if len(atr_vals) >= 5 else 1.0

    # Find most relevant resistance or support for breakout
    active_zones = [z for z in zones if z.state not in (ZoneState.BROKEN,)
                    and z.is_significant]

    if not active_zones:
        return None

    # Upside breakout: price above resistance zone
    if regime.state in (MarketState.RANGE, MarketState.ACCUMULATION,
                        MarketState.BREAKOUT_LONG):
        res_zones = [z for z in active_zones
                     if z.zone_type in (ZoneType.RESISTANCE, ZoneType.BOTH)
                     and z.zone_top > current_price - 0.5 * current_atr]
        if not res_zones:
            return None
        nearest_res = min(res_zones, key=lambda z: abs(z.zone_center - current_price))

        # Is price breaking out?
        if current_price > nearest_res.zone_top + 0.1 * current_atr:
            entry = current_price
            stop = nearest_res.zone_bottom - 0.25 * current_atr
            range_height = nearest_res.zone_top - (regime.key_level_low or
                                                    nearest_res.zone_bottom - 2 * current_atr)
            t1 = entry + abs(entry - stop)
            t2 = nearest_res.zone_top + range_height
            t3 = t2 + range_height * 0.5

            # Breakout quality checks
            last_bar_range = float(high[-1] - low[-1])
            strong_bar = last_bar_range > 1.0 * current_atr
            bar_close_pos = (close[-1] - low[-1]) / last_bar_range if last_bar_range > 0 else 0.5
            strong_close = bar_close_pos > 0.6

            conf = 0.40
            if strong_bar:
                conf += 0.15
            if strong_close:
                conf += 0.10
            if nearest_res.touch_count >= 2:
                conf += 0.05
            if regime.state == MarketState.ACCUMULATION:
                conf += 0.10

            notes = [f"Breaking above resistance at {nearest_res.zone_top:.2f}",
                     f"Zone touches: {nearest_res.touch_count}",
                     f"Bar quality: range={last_bar_range/current_atr:.1f}×ATR"]

            return _make_signal('breakout', 'long', entry, stop, t1, t2, t3,
                                conf, regime.state, bar_index, notes)

    # Downside breakout
    if regime.state in (MarketState.RANGE, MarketState.DISTRIBUTION,
                        MarketState.BREAKOUT_SHORT):
        sup_zones = [z for z in active_zones
                     if z.zone_type in (ZoneType.SUPPORT, ZoneType.BOTH)
                     and z.zone_bottom < current_price + 0.5 * current_atr]
        if not sup_zones:
            return None
        nearest_sup = min(sup_zones, key=lambda z: abs(z.zone_center - current_price))

        if current_price < nearest_sup.zone_bottom - 0.1 * current_atr:
            entry = current_price
            stop = nearest_sup.zone_top + 0.25 * current_atr
            range_height = (regime.key_level_high or
                            nearest_sup.zone_top + 2 * current_atr) - nearest_sup.zone_bottom
            t1 = entry - abs(stop - entry)
            t2 = nearest_sup.zone_bottom - range_height
            t3 = t2 - range_height * 0.5

            last_bar_range = float(high[-1] - low[-1])
            strong_bar = last_bar_range > 1.0 * current_atr
            bar_close_pos = (close[-1] - low[-1]) / last_bar_range if last_bar_range > 0 else 0.5
            strong_close = bar_close_pos < 0.4

            conf = 0.40
            if strong_bar:
                conf += 0.15
            if strong_close:
                conf += 0.10
            if regime.state == MarketState.DISTRIBUTION:
                conf += 0.10

            notes = [f"Breaking below support at {nearest_sup.zone_bottom:.2f}"]
            return _make_signal('breakout', 'short', entry, stop, t1, t2, t3,
                                conf, regime.state, bar_index, notes)

    return None


# ─── SIGNAL 4: THE ANTI ───────────────────────────────────────────────────────

def scan_anti(regime: MarketRegime, change_score: TrendChangeScore,
              health: TrendHealth, climax_signals: List[ClimaxSignal],
              high: np.ndarray, low: np.ndarray, close: np.ndarray,
              atr_vals: np.ndarray, bar_index: int) -> Optional[TradeSignal]:
    """
    The Anti: first pullback after a potential trend change.
    Requires: change_score >= 3 AND a sharp counter-trend move just occurred.
    """
    if change_score.total_score < 3:
        return None

    direction = regime.trend_direction  # existing trend direction
    if direction == 0:
        return None

    current_price = float(close[-1])
    current_atr = float(np.nanmean(atr_vals[-5:])) if len(atr_vals) >= 5 else 1.0

    # Anti is the FIRST pullback after a counter-trend thrust
    # Detect: we are currently in a pullback after a counter-trend impulse
    if not regime.in_pullback:
        return None

    notes = [f"Anti setup: change score {change_score.total_score}"]
    notes.extend(change_score.signals[:3])

    # Anti goes COUNTER to the existing trend
    anti_direction = -direction  # opposite of trend

    if anti_direction == 1:
        # Going long: counter to existing downtrend
        entry = current_price + 0.1 * current_atr
        if change_score.last_safe_entry:
            stop = change_score.last_safe_entry + 0.25 * current_atr
        else:
            stop = low[-1] - 0.5 * current_atr

        # MMO: first counter-thrust magnitude from the trend extreme
        counter_thrust = health.last_impulse_size
        t1 = entry + counter_thrust * 0.5
        t2 = entry + counter_thrust
        t3 = entry + counter_thrust * 1.5
    else:
        # Going short: counter to existing uptrend
        entry = current_price - 0.1 * current_atr
        if change_score.last_safe_entry:
            stop = change_score.last_safe_entry - 0.25 * current_atr
        else:
            stop = high[-1] + 0.5 * current_atr

        counter_thrust = health.last_impulse_size
        t1 = entry - counter_thrust * 0.5
        t2 = entry - counter_thrust
        t3 = entry - counter_thrust * 1.5

    base_conf = 0.35 + change_score.probability * 0.3
    if health.is_parabolic:
        base_conf += 0.10

    direction_str = 'long' if anti_direction == 1 else 'short'
    notes.append(f"Anti {direction_str} — countertrend; reduce size by 50%")

    sig = _make_signal('anti', direction_str, entry, stop, t1, t2, t3,
                       base_conf, regime.state, bar_index, notes)
    if sig:
        sig.risk_fraction = 0.01  # Half-size for countertrend trades
    return sig


# ─── SIGNAL 5: FAILED BREAKOUT ────────────────────────────────────────────────

def scan_failed_breakout(regime: MarketRegime, zones: List[SRZone],
                          high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          atr_vals: np.ndarray, bar_index: int,
                          breakout_lookback: int = 8) -> Optional[TradeSignal]:
    """
    Failed breakout: a recent breakout is now reversing.
    Signal: price re-enters the range after appearing to break out.
    """
    if len(close) < breakout_lookback:
        return None

    current_price = float(close[-1])
    current_atr = float(np.nanmean(atr_vals[-5:])) if len(atr_vals) >= 5 else 1.0

    # Look for recent extreme high or low that was a "breakout"
    recent_high = float(np.max(high[-breakout_lookback:]))
    recent_low = float(np.min(low[-breakout_lookback:]))
    prior_high = float(np.max(high[-breakout_lookback * 2:-breakout_lookback])) if len(high) > breakout_lookback * 2 else recent_high
    prior_low = float(np.min(low[-breakout_lookback * 2:-breakout_lookback])) if len(low) > breakout_lookback * 2 else recent_low

    signals = []

    # Failed upside breakout: recent new high, now price falling back
    upside_breakout = recent_high > prior_high + 0.3 * current_atr
    now_retreating_down = current_price < prior_high - 0.5 * current_atr
    if upside_breakout and now_retreating_down:
        entry = current_price
        stop = recent_high + 0.5 * current_atr
        t1 = entry - abs(stop - entry)
        t2 = prior_low + (prior_high - prior_low) * 0.3   # 30% of range from bottom
        t3 = prior_low

        conf = 0.40
        # Strong reversal bar? (large down bar)
        last_range = float(high[-1] - low[-1])
        if last_range > 1.5 * current_atr and close[-1] < close[-2]:
            conf += 0.15

        notes = [f"Failed upside breakout: high={recent_high:.2f} now retreating to {current_price:.2f}",
                 "DANGER: Reduce size; use strict stop; do not add to position"]

        sig = _make_signal('failed_breakout', 'short', entry, stop, t1, t2, t3,
                           conf, regime.state, bar_index, notes)
        if sig:
            sig.risk_fraction = 0.01  # Half size — most dangerous trade
            return sig

    # Failed downside breakout: recent new low, now price rallying back
    downside_breakout = recent_low < prior_low - 0.3 * current_atr
    now_retreating_up = current_price > prior_low + 0.5 * current_atr
    if downside_breakout and now_retreating_up:
        entry = current_price
        stop = recent_low - 0.5 * current_atr
        t1 = entry + abs(entry - stop)
        t2 = prior_high - (prior_high - prior_low) * 0.3
        t3 = prior_high

        conf = 0.40
        last_range = float(high[-1] - low[-1])
        if last_range > 1.5 * current_atr and close[-1] > close[-2]:
            conf += 0.15

        notes = [f"Failed downside breakout: low={recent_low:.2f} now rallying to {current_price:.2f}",
                 "DANGER: Reduce size; use strict stop"]

        sig = _make_signal('failed_breakout', 'long', entry, stop, t1, t2, t3,
                           conf, regime.state, bar_index, notes)
        if sig:
            sig.risk_fraction = 0.01
            return sig

    return None


# ─── SIGNAL 6: FIRST PULLBACK AFTER BREAKOUT ─────────────────────────────────

def scan_pb_after_breakout(regime: MarketRegime, zones: List[SRZone],
                            high: np.ndarray, low: np.ndarray, close: np.ndarray,
                            atr_vals: np.ndarray, bar_index: int) -> Optional[TradeSignal]:
    """
    Enter the first pullback after a confirmed breakout.
    Grimes: this converts a breakout trade into a high-probability pullback trade.
    """
    if regime.state not in (MarketState.BREAKOUT_LONG, MarketState.BREAKOUT_SHORT,
                            MarketState.STRONG_UPTREND, MarketState.STRONG_DOWNTREND):
        return None

    if not regime.in_pullback:
        return None

    current_price = float(close[-1])
    current_atr = float(np.nanmean(atr_vals[-5:])) if len(atr_vals) >= 5 else 1.0

    direction = regime.trend_direction

    if direction == 1:
        # Find the breakout level (approximate from last major resistance zone)
        res_zones = [z for z in zones if z.zone_type in (ZoneType.RESISTANCE, ZoneType.BOTH)
                     and z.state == ZoneState.BROKEN and z.zone_center < current_price]
        if not res_zones:
            return None
        bo_level = max(res_zones, key=lambda z: z.zone_center)

        entry = current_price + 0.1 * current_atr
        stop = bo_level.zone_bottom - 0.25 * current_atr   # below the former breakout level
        t1 = entry + abs(entry - stop)
        t2 = regime.key_level_high or entry + 2 * abs(entry - stop)
        t3 = t2 + abs(t2 - entry) * 0.5

        conf = 0.55  # This is one of the best setups
        if regime.state == MarketState.STRONG_UPTREND:
            conf += 0.10
        if regime.pullback_depth_pct < 0.382:
            conf += 0.10  # very shallow = strong continuation

        notes = [f"First pullback after breakout above {bo_level.zone_center:.2f}",
                 f"Pullback depth: {regime.pullback_depth_pct:.1%}"]

        return _make_signal('pb_after_breakout', 'long', entry, stop, t1, t2, t3,
                            conf, regime.state, bar_index, notes)

    else:  # direction == -1
        sup_zones = [z for z in zones if z.zone_type in (ZoneType.SUPPORT, ZoneType.BOTH)
                     and z.state == ZoneState.BROKEN and z.zone_center > current_price]
        if not sup_zones:
            return None
        bo_level = min(sup_zones, key=lambda z: z.zone_center)

        entry = current_price - 0.1 * current_atr
        stop = bo_level.zone_top + 0.25 * current_atr
        t1 = entry - abs(stop - entry)
        t2 = regime.key_level_low or entry - 2 * abs(stop - entry)
        t3 = t2 - abs(t2 - entry) * 0.5

        conf = 0.55
        if regime.state == MarketState.STRONG_DOWNTREND:
            conf += 0.10

        notes = [f"First pullback after breakdown below {bo_level.zone_center:.2f}"]
        return _make_signal('pb_after_breakout', 'short', entry, stop, t1, t2, t3,
                            conf, regime.state, bar_index, notes)


# ─── MASTER SIGNAL SCANNER ────────────────────────────────────────────────────

def scan_all_signals(regime: MarketRegime, health: TrendHealth,
                     change_score: TrendChangeScore,
                     high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     atr_vals: np.ndarray, zones: List[SRZone],
                     springs: List[dict], upthrusts: List[dict],
                     climax_signals: List[ClimaxSignal],
                     htf_state: MarketState, ltf_momentum: int,
                     macd_div: np.ndarray, vol_state: np.ndarray,
                     mtf_alignment_score: int,
                     bar_index: int,
                     min_confidence: float = 0.45) -> List[TradeSignal]:
    """
    Run all six scanners and apply confluence scoring.
    Returns list of signals sorted by confidence (descending).
    """
    signals: List[TradeSignal] = []
    current_vol_state = int(vol_state[-1]) if len(vol_state) > 0 else 1
    vol_contracting = (current_vol_state == 0)
    current_macd_div = int(macd_div[-1]) if len(macd_div) > 0 else 0
    current_price = float(close[-1])
    current_atr = float(np.nanmean(atr_vals[-5:])) if len(atr_vals) >= 5 else 1.0

    # S/R proximity
    near_zone = any(abs(z.zone_center - current_price) < 0.5 * current_atr
                    for z in zones if z.is_significant)

    # 1. Pullback
    sig = scan_pullback(regime, health, high, low, close, atr_vals, zones, bar_index)
    if sig:
        sig.confidence = score_confluence(sig, htf_state, ltf_momentum,
                                           near_zone, current_macd_div,
                                           vol_contracting, mtf_alignment_score)
        signals.append(sig)

    # 2. Failure tests
    for sig in scan_failure_test(regime, springs, upthrusts, high, low, close, atr_vals, bar_index):
        sig.confidence = score_confluence(sig, htf_state, ltf_momentum,
                                           True, current_macd_div,
                                           vol_contracting, mtf_alignment_score)
        signals.append(sig)

    # 3. Breakout
    sig = scan_breakout(regime, zones, high, low, close, atr_vals, bar_index)
    if sig:
        sig.confidence = score_confluence(sig, htf_state, ltf_momentum,
                                           near_zone, current_macd_div,
                                           vol_contracting, mtf_alignment_score)
        signals.append(sig)

    # 4. Anti
    sig = scan_anti(regime, change_score, health, climax_signals, high, low, close, atr_vals, bar_index)
    if sig:
        sig.confidence = score_confluence(sig, htf_state, ltf_momentum,
                                           near_zone, current_macd_div,
                                           vol_contracting, mtf_alignment_score)
        signals.append(sig)

    # 5. Failed breakout
    sig = scan_failed_breakout(regime, zones, high, low, close, atr_vals, bar_index)
    if sig:
        sig.confidence = score_confluence(sig, htf_state, ltf_momentum,
                                           near_zone, current_macd_div,
                                           vol_contracting, mtf_alignment_score)
        signals.append(sig)

    # 6. First pullback after breakout
    sig = scan_pb_after_breakout(regime, zones, high, low, close, atr_vals, bar_index)
    if sig:
        sig.confidence = score_confluence(sig, htf_state, ltf_momentum,
                                           True, current_macd_div,
                                           vol_contracting, mtf_alignment_score)
        signals.append(sig)

    # Filter by minimum confidence and minimum RR
    valid = [s for s in signals
             if s.confidence >= min_confidence
             and s.rr_1 >= 1.0
             and s.risk_per_unit > 0]

    valid.sort(key=lambda s: s.confidence, reverse=True)
    return valid
