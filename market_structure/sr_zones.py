"""
sr_zones.py — Support and resistance zone identification and scoring.
Based on: The Art and Science of Technical Analysis (Adam Grimes)

Key Grimes principle: MOST S/R levels are no better than random.
This module applies statistical tests to identify genuinely significant zones.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from pivot_detector import Pivot, ZigzagPoint


class ZoneState(Enum):
    INTACT   = "intact"
    TESTED   = "tested"
    BROKEN   = "broken"
    FLIPPED  = "flipped"
    WEAKENED = "weakened"


class ZoneType(Enum):
    SUPPORT    = "support"
    RESISTANCE = "resistance"
    BOTH       = "both"       # zone has acted as both S and R (flip zone)


@dataclass
class SRZone:
    zone_center: float
    zone_top: float
    zone_bottom: float
    zone_type: ZoneType
    state: ZoneState = ZoneState.INTACT
    touch_count: int = 0
    bounce_count: int = 0
    break_count: int = 0
    bounce_rate: float = 0.0       # bounces / total approaches
    strength_score: float = 0.0    # composite 0-1
    is_significant: bool = False   # passed statistical test
    last_touch_bar: int = 0
    first_touch_bar: int = 0
    atr_width: float = 0.0         # zone width in ATR units
    notes: List[str] = field(default_factory=list)


# ─── ZONE CREATION FROM PIVOTS ────────────────────────────────────────────────

def build_sr_zones(pivots: List[Pivot], atr_vals: np.ndarray,
                   high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   merge_distance_atr: float = 0.5,
                   min_touches: int = 2) -> List[SRZone]:
    """
    Cluster pivot highs and lows into S/R zones.

    1. Collect all pivot prices.
    2. Merge pivots within merge_distance × ATR.
    3. Validate each zone.
    4. Return sorted by strength (descending).
    """
    if not pivots:
        return []

    current_atr = float(np.nanmean(atr_vals[-20:])) if len(atr_vals) >= 20 else float(np.nanmean(atr_vals))
    merge_dist = merge_distance_atr * current_atr

    # Separate pivot prices by type
    ph_prices = [(p.price, p.index, 'high') for p in pivots if p.pivot_type == 'high']
    pl_prices = [(p.price, p.index, 'low') for p in pivots if p.pivot_type == 'low']
    all_prices = ph_prices + pl_prices
    all_prices.sort(key=lambda x: x[0])

    if not all_prices:
        return []

    # Cluster
    clusters: List[List[tuple]] = []
    current_cluster = [all_prices[0]]
    for item in all_prices[1:]:
        if abs(item[0] - current_cluster[-1][0]) <= merge_dist:
            current_cluster.append(item)
        else:
            clusters.append(current_cluster)
            current_cluster = [item]
    clusters.append(current_cluster)

    # Build zone objects
    zones: List[SRZone] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        prices = [c[0] for c in cluster]
        bars = [c[1] for c in cluster]
        types = [c[2] for c in cluster]

        zone_center = float(np.mean(prices))
        zone_top = float(max(prices)) + merge_dist * 0.25
        zone_bottom = float(min(prices)) - merge_dist * 0.25

        has_high = 'high' in types
        has_low = 'low' in types
        if has_high and has_low:
            zone_type = ZoneType.BOTH
        elif has_high:
            zone_type = ZoneType.RESISTANCE
        else:
            zone_type = ZoneType.SUPPORT

        zone = SRZone(
            zone_center=zone_center,
            zone_top=zone_top,
            zone_bottom=zone_bottom,
            zone_type=zone_type,
            touch_count=len(cluster),
            first_touch_bar=min(bars),
            last_touch_bar=max(bars),
            atr_width=(zone_top - zone_bottom) / current_atr if current_atr > 0 else 0.0
        )
        zones.append(zone)

    # Score and validate each zone
    zones = [_score_zone(z, high, low, close, atr_vals) for z in zones]
    zones = [_update_zone_state(z, close, atr_vals) for z in zones]
    zones.sort(key=lambda z: z.strength_score, reverse=True)
    return zones


# ─── ZONE SCORING ─────────────────────────────────────────────────────────────

def _score_zone(zone: SRZone, high: np.ndarray, low: np.ndarray,
                close: np.ndarray, atr_vals: np.ndarray) -> SRZone:
    """
    Score zone significance.
    Factors:
      1. Touch count (more = stronger, up to 3; 3+ starts weakening)
      2. Bounce rate (fraction of approaches that resulted in bounce)
      3. Recency (recent touches weighted more)
      4. Zone width (tighter zones = more precise = stronger signal)
    """
    current_atr = float(np.nanmean(atr_vals[-20:])) if len(atr_vals) >= 20 else float(np.nanmean(atr_vals))
    approaches, bounces = _count_approaches_and_bounces(
        zone, high, low, close, atr_vals, approach_dist_atr=0.5)

    zone.touch_count = approaches
    zone.bounce_count = bounces
    zone.bounce_rate = bounces / approaches if approaches > 0 else 0.0

    # Touch count score (optimal at 2-3 touches)
    touch_score = min(zone.touch_count / 3.0, 1.0) if zone.touch_count <= 3 else max(1.0 - (zone.touch_count - 3) * 0.1, 0.3)

    # Bounce rate score
    bounce_score = max(0.0, (zone.bounce_rate - 0.5) / 0.5)  # 50% = 0, 100% = 1

    # Recency score
    total_bars = len(close)
    recency_score = zone.last_touch_bar / total_bars if total_bars > 0 else 0.5

    # Width score (narrower = more precise)
    width_score = max(0.0, 1.0 - zone.atr_width)

    # Statistical significance test
    # Grimes warns: bounce_rate > 65% with n >= 5 meaningful
    zone.is_significant = (zone.bounce_rate >= 0.60 and approaches >= 3) or \
                          (zone.bounce_rate >= 0.50 and approaches >= 5)

    zone.strength_score = (0.3 * touch_score + 0.35 * bounce_score +
                           0.2 * recency_score + 0.15 * width_score)

    return zone


def _count_approaches_and_bounces(zone: SRZone, high: np.ndarray,
                                   low: np.ndarray, close: np.ndarray,
                                   atr_vals: np.ndarray,
                                   approach_dist_atr: float = 0.5
                                   ) -> Tuple[int, int]:
    """Count how many times price approached and bounced from the zone."""
    approaches = 0
    bounces = 0
    in_approach = False

    for i in range(1, len(close) - 2):
        current_atr = float(atr_vals[i]) if not np.isnan(atr_vals[i]) else 1.0
        approach_dist = approach_dist_atr * current_atr

        near_zone = (low[i] <= zone.zone_top + approach_dist and
                     high[i] >= zone.zone_bottom - approach_dist)

        if near_zone and not in_approach:
            in_approach = True
            approaches += 1

        if in_approach and not near_zone:
            in_approach = False
            # Check if price bounced (moved at least 0.5×ATR away from zone)
            post_move = abs(close[min(i + 1, len(close) - 1)] - zone.zone_center)
            if post_move > 0.5 * current_atr:
                bounces += 1

    return approaches, bounces


# ─── ZONE STATE MANAGEMENT ────────────────────────────────────────────────────

def _update_zone_state(zone: SRZone, close: np.ndarray,
                       atr_vals: np.ndarray) -> SRZone:
    """Update zone state based on most recent price action."""
    if len(close) == 0:
        return zone

    current_price = close[-1]
    current_atr = float(np.nanmean(atr_vals[-5:])) if len(atr_vals) >= 5 else 1.0

    if zone.touch_count >= 4:
        zone.state = ZoneState.WEAKENED

    # Check if broken
    if zone.zone_type in (ZoneType.RESISTANCE, ZoneType.BOTH):
        if current_price > zone.zone_top + 0.3 * current_atr:
            # Check if it held previously as support (flip)
            recent_closes = close[-10:]
            was_below = np.any(recent_closes < zone.zone_bottom)
            if was_below:
                zone.state = ZoneState.FLIPPED
                zone.zone_type = ZoneType.SUPPORT
                zone.notes.append("Former resistance; flipped to support")
            else:
                zone.state = ZoneState.BROKEN

    if zone.zone_type in (ZoneType.SUPPORT, ZoneType.BOTH):
        if current_price < zone.zone_bottom - 0.3 * current_atr:
            recent_closes = close[-10:]
            was_above = np.any(recent_closes > zone.zone_top)
            if was_above:
                zone.state = ZoneState.FLIPPED
                zone.zone_type = ZoneType.RESISTANCE
                zone.notes.append("Former support; flipped to resistance")
            else:
                zone.state = ZoneState.BROKEN

    return zone


# ─── SPRING / UPTHRUST DETECTION ──────────────────────────────────────────────

def detect_spring(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  zones: List[SRZone], atr_vals: np.ndarray,
                  lookback: int = 3) -> List[dict]:
    """
    Detect Wyckoff Springs at support zones within the last `lookback` bars.
    A spring: low < zone_bottom AND close > zone_center AND close_position > 0.5
    """
    springs = []
    current_atr = float(np.nanmean(atr_vals[-5:])) if len(atr_vals) >= 5 else 1.0

    for i in range(max(0, len(close) - lookback), len(close)):
        bar_range = high[i] - low[i]
        if bar_range == 0:
            continue
        close_pos = (close[i] - low[i]) / bar_range

        for zone in zones:
            if zone.state == ZoneState.BROKEN:
                continue
            if zone.zone_type not in (ZoneType.SUPPORT, ZoneType.BOTH):
                continue

            # Spring conditions
            probe_below = low[i] < zone.zone_bottom
            closes_above_mid = close[i] > zone.zone_center
            strong_close = close_pos > 0.5

            if probe_below and closes_above_mid and strong_close:
                probe_depth = (zone.zone_bottom - low[i]) / current_atr
                strength = close_pos * (1.0 - min(probe_depth, 1.0) * 0.3)
                springs.append({
                    'bar_index': i,
                    'type': 'spring',
                    'zone_center': zone.zone_center,
                    'zone_bottom': zone.zone_bottom,
                    'probe_low': float(low[i]),
                    'close': float(close[i]),
                    'strength': float(np.clip(strength, 0, 1)),
                    'entry_trigger': float(high[i]),    # buy above this bar
                    'stop': float(low[i]) - 0.25 * current_atr,
                })

    return springs


def detect_upthrust(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    zones: List[SRZone], atr_vals: np.ndarray,
                    lookback: int = 3) -> List[dict]:
    """
    Detect Wyckoff Upthrusts at resistance zones.
    An upthrust: high > zone_top AND close < zone_center AND close_position < 0.5
    """
    upthrusts = []
    current_atr = float(np.nanmean(atr_vals[-5:])) if len(atr_vals) >= 5 else 1.0

    for i in range(max(0, len(close) - lookback), len(close)):
        bar_range = high[i] - low[i]
        if bar_range == 0:
            continue
        close_pos = (close[i] - low[i]) / bar_range

        for zone in zones:
            if zone.state == ZoneState.BROKEN:
                continue
            if zone.zone_type not in (ZoneType.RESISTANCE, ZoneType.BOTH):
                continue

            probe_above = high[i] > zone.zone_top
            closes_below_mid = close[i] < zone.zone_center
            weak_close = close_pos < 0.5

            if probe_above and closes_below_mid and weak_close:
                probe_height = (high[i] - zone.zone_top) / current_atr
                strength = (1.0 - close_pos) * (1.0 - min(probe_height, 1.0) * 0.3)
                upthrusts.append({
                    'bar_index': i,
                    'type': 'upthrust',
                    'zone_center': zone.zone_center,
                    'zone_top': zone.zone_top,
                    'probe_high': float(high[i]),
                    'close': float(close[i]),
                    'strength': float(np.clip(strength, 0, 1)),
                    'entry_trigger': float(low[i]),      # sell below this bar
                    'stop': float(high[i]) + 0.25 * current_atr,
                })

    return upthrusts


# ─── ZONE PROXIMITY ───────────────────────────────────────────────────────────

def nearest_zones(price: float, zones: List[SRZone], atr: float,
                  n: int = 3) -> List[SRZone]:
    """Return n nearest active zones sorted by proximity to price."""
    active = [z for z in zones if z.state not in (ZoneState.BROKEN,)]
    active.sort(key=lambda z: abs(z.zone_center - price))
    return active[:n]


def price_at_zone(price: float, zone: SRZone, atr: float,
                  tolerance_atr: float = 0.5) -> bool:
    """True if price is within tolerance×ATR of the zone center."""
    return abs(price - zone.zone_center) <= tolerance_atr * atr


def find_nearest_resistance(price: float, zones: List[SRZone]) -> Optional[SRZone]:
    """Find the nearest resistance zone above current price."""
    candidates = [z for z in zones
                  if z.zone_center > price
                  and z.zone_type in (ZoneType.RESISTANCE, ZoneType.BOTH)
                  and z.state not in (ZoneState.BROKEN,)]
    if not candidates:
        return None
    return min(candidates, key=lambda z: z.zone_center - price)


def find_nearest_support(price: float, zones: List[SRZone]) -> Optional[SRZone]:
    """Find the nearest support zone below current price."""
    candidates = [z for z in zones
                  if z.zone_center < price
                  and z.zone_type in (ZoneType.SUPPORT, ZoneType.BOTH)
                  and z.state not in (ZoneState.BROKEN,)]
    if not candidates:
        return None
    return max(candidates, key=lambda z: z.zone_center)


# ─── PREVIOUS DAY / WEEK LEVELS ───────────────────────────────────────────────

def pdhl_zones(daily_high: float, daily_low: float, daily_close: float,
               current_atr: float) -> List[SRZone]:
    """
    Previous day/session high, low, and close as S/R zones.
    These are watched by many participants — high significance.
    """
    zones = []
    for price, label in [(daily_high, 'PDH'), (daily_low, 'PDL'), (daily_close, 'PDC')]:
        zone_type = ZoneType.RESISTANCE if price >= daily_close else ZoneType.SUPPORT
        z = SRZone(
            zone_center=price,
            zone_top=price + 0.2 * current_atr,
            zone_bottom=price - 0.2 * current_atr,
            zone_type=zone_type,
            touch_count=1,
            strength_score=0.75,  # high base score for PDHL
            is_significant=True
        )
        z.notes.append(label)
        zones.append(z)
    return zones
