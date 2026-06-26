"""
indicators.py — Foundation layer for all technical indicators.
Based on: The Art and Science of Technical Analysis (Adam Grimes)

All indicator computations live here. Other modules import from this one.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ─── ATR ─────────────────────────────────────────────────────────────────────

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder smoothed ATR (same as Wilder RSI smoothing)."""
    tr = true_range(high, low, close)
    result = np.full(len(tr), np.nan)
    if len(tr) < period:
        return result
    result[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        result[i] = result[i - 1] * (1 - alpha) + tr[i] * alpha
    return result


# ─── EMA / SMA ───────────────────────────────────────────────────────────────

def ema(series: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA with α = 2/(period+1)."""
    alpha = 2.0 / (period + 1)
    result = np.full(len(series), np.nan)
    # Seed with SMA of first `period` bars
    valid_start = np.where(~np.isnan(series))[0]
    if len(valid_start) < period:
        return result
    start = valid_start[0]
    result[start + period - 1] = np.nanmean(series[start:start + period])
    for i in range(start + period, len(series)):
        if np.isnan(series[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def sma(series: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(series), np.nan)
    for i in range(period - 1, len(series)):
        result[i] = np.mean(series[i - period + 1:i + 1])
    return result


# ─── MACD ────────────────────────────────────────────────────────────────────

def macd(close: np.ndarray, fast: int = 3, slow: int = 10, signal: int = 16
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grimes' modified MACD (Appendix B of the book):
      Fast line  = SMA(3) - SMA(10)        [NOT EMA]
      Signal line = SMA(16) of fast line   [NOT EMA]
      No histogram used.
    Default parameters match the book exactly.
    """
    fast_sma = sma(close, fast)
    slow_sma = sma(close, slow)
    macd_line = fast_sma - slow_sma
    signal_line = sma(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def macd_divergence(high: np.ndarray, low: np.ndarray, macd_line: np.ndarray,
                    lookback: int = 20, atr_vals: np.ndarray = None
                    ) -> np.ndarray:
    """
    Detect MACD divergences on a rolling basis.
    Returns array: +1 = bullish divergence, -1 = bearish divergence, 0 = none.
    Bearish: price HH + MACD lower high.
    Bullish: price LL + MACD higher low.
    """
    result = np.zeros(len(high))
    for i in range(lookback, len(high)):
        window_h = high[i - lookback:i + 1]
        window_l = low[i - lookback:i + 1]
        window_m = macd_line[i - lookback:i + 1]
        if np.isnan(window_m[-1]):
            continue

        # Bearish divergence: price at/near new high, MACD lower high
        price_max_idx = np.argmax(window_h)
        if price_max_idx == lookback:  # current bar is the high
            prev_max_idx = np.argmax(window_h[:lookback])
            if (window_h[lookback] >= window_h[prev_max_idx] and
                    window_m[lookback] < window_m[prev_max_idx]):
                result[i] = -1

        # Bullish divergence: price at/near new low, MACD higher low
        price_min_idx = np.argmin(window_l)
        if price_min_idx == lookback:
            prev_min_idx = np.argmin(window_l[:lookback])
            if (window_l[lookback] <= window_l[prev_min_idx] and
                    window_m[lookback] > window_m[prev_min_idx]):
                result[i] = 1
    return result


# ─── KELTNER CHANNEL ─────────────────────────────────────────────────────────

def keltner_channel(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    period: int = 20, multiplier: float = 2.25
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grimes' modified Keltner Channel (Table 7.1, line 1491 of the book):
      Middle = EMA(20) of close
      Upper  = Middle + 2.25 × ATR(20)
      Lower  = Middle - 2.25 × ATR(20)
    The 2.25 multiplier is calibrated so that 85-90% of all bar ranges
    fall inside the bands across a large test universe (2.4M bars).
    """
    middle = ema(close, period)
    atr_vals = atr(high, low, close, period)
    upper = middle + multiplier * atr_vals
    lower = middle - multiplier * atr_vals
    return upper, middle, lower


def keltner_position(close: np.ndarray, upper: np.ndarray,
                     lower: np.ndarray) -> np.ndarray:
    """Returns normalized position 0.0 (at lower) to 1.0 (at upper)."""
    band_width = upper - lower
    pos = np.where(band_width > 0, (close - lower) / band_width, 0.5)
    return np.clip(pos, -0.5, 1.5)


# ─── BAR CHARACTER ────────────────────────────────────────────────────────────

def bar_close_position(high: np.ndarray, low: np.ndarray,
                       close: np.ndarray) -> np.ndarray:
    """
    Returns close position within the bar range: 0=at low, 1=at high.
    Used for spring/upthrust detection and bar character analysis.
    """
    bar_range = high - low
    pos = np.where(bar_range > 0, (close - low) / bar_range, 0.5)
    return np.clip(pos, 0.0, 1.0)


def bar_range_ratio(high: np.ndarray, low: np.ndarray,
                    atr_vals: np.ndarray) -> np.ndarray:
    """Range of each bar relative to ATR. >2.5 is a potential climax bar."""
    bar_range = high - low
    return np.where(atr_vals > 0, bar_range / atr_vals, 0.0)


# ─── MOMENTUM ─────────────────────────────────────────────────────────────────

def roc(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Rate of Change: (close[t] - close[t-n]) / close[t-n] * 100."""
    result = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if close[i - period] != 0:
            result[i] = (close[i] - close[i - period]) / close[i - period] * 100
    return result


def rolling_std(series: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation."""
    result = np.full(len(series), np.nan)
    for i in range(period - 1, len(series)):
        result[i] = np.std(series[i - period + 1:i + 1], ddof=0)
    return result


# ─── VOLATILITY STATE ─────────────────────────────────────────────────────────

def volatility_state(atr_vals: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Classify volatility relative to recent history.
    Returns: 0=contracting, 1=normal, 2=expanding
    """
    atr_ma = sma(atr_vals, period)
    result = np.ones(len(atr_vals))  # default normal
    for i in range(period, len(atr_vals)):
        if np.isnan(atr_ma[i]):
            continue
        ratio = atr_vals[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
        if ratio < 0.75:
            result[i] = 0  # contracting
        elif ratio > 1.5:
            result[i] = 2  # expanding
    return result


# ─── DATAFRAME HELPER ────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame, atr_period: int = 14,
                       keltner_period: int = 20, keltner_mult: float = 2.25
                       ) -> pd.DataFrame:
    """
    Convenience: compute all indicators and attach to DataFrame.
    DataFrame must have columns: open, high, low, close, volume (optional).
    """
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values

    df['atr'] = atr(h, l, c, atr_period)
    df['ema_9'] = ema(c, 9)
    df['ema_20'] = ema(c, 20)
    df['ema_50'] = ema(c, 50)
    df['ema_200'] = ema(c, 200)

    df['kc_upper'], df['kc_mid'], df['kc_lower'] = keltner_channel(
        h, l, c, keltner_period, keltner_mult)
    df['kc_position'] = keltner_position(c, df['kc_upper'].values,
                                          df['kc_lower'].values)

    ml, sl, hist = macd(c)
    df['macd_line'] = ml
    df['macd_signal'] = sl
    df['macd_hist'] = hist
    df['macd_divergence'] = macd_divergence(h, l, ml, lookback=20)

    df['bar_close_pos'] = bar_close_position(h, l, c)
    df['bar_range_ratio'] = bar_range_ratio(h, l, df['atr'].values)
    df['roc_14'] = roc(c, 14)
    df['volatility_state'] = volatility_state(df['atr'].values)
    return df
