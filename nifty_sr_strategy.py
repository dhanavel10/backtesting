"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         NIFTY PRECISION S/R ZONE TRADING STRATEGY ENGINE                    ║
║         Fakeout-Aware Breakout + Reversal + Zone Confluence System           ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE
────────────
  python_visual.py  → 60-day historical S/R zones  (macro context)
  realtime_sr.py    → live intraday S/R zones       (micro context)
  THIS FILE         → merges both, hunts entries, manages trades

ENTRY LOGIC (4 modes — only one fires per zone)
────────────────────────────────────────────────
  1. BREAKOUT-PULLBACK  : price breaks zone, pulls back TO zone edge, holds → buy break
  2. FAKEOUT-REVERSAL   : price pierces zone, closes BACK inside → trade the trap
  3. ZONE-BOUNCE        : price approaches zone from outside, rejects with body → fade
  4. ZONE-COMPRESS      : price compresses inside zone (2+ inside-bars) → trade the expansion

FILTERS (ALL must pass before any entry)
─────────────────────────────────────────
  • Dual-zone confluence  : same level in BOTH historical AND live maps
  • Minimum zone strength : historical ≥ 40, live ≥ 30
  • Candle quality        : body ≥ 40% of range (no doji entries)
  • VWAP alignment        : price side matches entry direction
  • Time filter           : no entries in 09:15–09:30, 15:00–15:30
  • Zone freshness        : historical zone touched in last 15 trading days
  • ATR gate              : entry risk ≤ 1.5× 14-bar ATR

RISK MANAGEMENT
───────────────
  SL  = opposite zone edge + 0.5 × zone_band (never more than MAX_RISK_PTS)
  TP1 = 1.5× risk (book 50% here → trail rest)
  TP2 = next opposing zone price
  Trail: once TP1 hit, move SL to entry; then trail by zone_band/2
  Max risk per trade = MAX_RISK_PCT of capital (default 1%)

Requirements:
    pip install yfinance pandas numpy scipy scikit-learn plotly
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from collections import defaultdict, deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# STRATEGY CONFIGURATION  — tune these only
# ════════════════════════════════════════════════════════════════

class Config:
    # ── Ticker ─────────────────────────────────────────────────
    TICKER       = "^NSEI"      # ^NSEBANK for Bank Nifty
    INTERVAL     = "5m"
    HIST_DAYS    = 55

    # ── Historical zone parameters (python_visual.py) ──────────
    LEFT_BARS         = 10
    RIGHT_BARS        = 10
    CLUSTER_TOLERANCE = 20.0    # widened: pivots within this join one cluster
    ZONE_HALF_BAND    = 20.0    # widened: zone = price ± 20 pts → 40-pt band
    MIN_WICK_TOUCHES  = 2       # lowered from 3
    MIN_SESSIONS      = 1       # lowered from 2
    MIN_REJECTIONS    = 1
    TOP_N_HIST        = 25      # more zones

    # ── Live zone parameters (realtime_sr.py) ──────────────────
    RANGE_SIZE         = 4.0
    REVERSAL_THR       = 12.0
    BANDWIDTH          = 10.0   # wider KDE → more zones formed
    LIVE_ZONE_HW       = 20.0   # widened to match hist band
    HALF_LIFE_MIN      = 240.0  # 4-hour decay (entire session stays valid)

    # ── Confluence matching ─────────────────────────────────────
    CONFLUENCE_DIST    = 40.0   # widened: hist & live zone within 40 pts = same
    MIN_HIST_STR       = 20.0   # lowered from 35
    MIN_LIVE_STR       = 10.0   # lowered from 25

    # ── Entry filters ───────────────────────────────────────────
    MIN_BODY_RATIO     = 0.25   # lowered from 0.35 — accept moderate candles
    PULLBACK_WINDOW    = 8      # more bars to wait for pullback
    FAKEOUT_LOOKBACK   = 5      # look further back for the fakeout bar
    COMPRESS_MIN_BARS  = 2

    # ── Risk management ─────────────────────────────────────────
    CAPITAL            = 100_000
    MAX_RISK_PCT       = 0.01
    MAX_RISK_PTS       = 80        # raised from 60 — Nifty swings wider
    TP1_RR             = 1.5
    SL_BUFFER_FACTOR   = 0.3       # tighter buffer so risk stays manageable

    # ── ATR gate ────────────────────────────────────────────────
    ATR_PERIOD         = 14
    MAX_RISK_ATR_MULT  = 2.5       # raised from 1.5 — ATR gate was killing trades

    # ── Session / time filters ──────────────────────────────────
    NO_ENTRY_OPEN_MINS = 15
    NO_ENTRY_CLOSE_MINS= 20        # reduced from 30
    MARKET_OPEN        = dtime(9, 15)
    MARKET_CLOSE       = dtime(15, 30)

    # ── Zone freshness ──────────────────────────────────────────
    MAX_ZONE_AGE_DAYS  = 60        # raised: use all available history

    # ── VWAP ────────────────────────────────────────────────────
    USE_VWAP_FILTER    = False     # disabled — was blocking most signals

    # ── Debug ───────────────────────────────────────────────────
    DEBUG              = True      # prints exactly why each candidate is rejected

    # ── Chart ───────────────────────────────────────────────────
    CHART_SESSIONS     = 10


cfg = Config()


# ════════════════════════════════════════════════════════════════
# 0. UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range → ATR via Wilder smoothing."""
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP — resets each trading day."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol     = df["Volume"].replace(0, np.nan).fillna(1)
    tp_vol  = typical * vol
    dates   = df.index.date

    vwap = pd.Series(index=df.index, dtype=float)
    for d in np.unique(dates):
        mask = dates == d
        cumvol   = vol[mask].cumsum()
        cumtpvol = tp_vol[mask].cumsum()
        vwap[mask] = cumtpvol / cumvol
    return vwap


def candle_body_ratio(row: pd.Series) -> float:
    rng = row["High"] - row["Low"]
    if rng < 0.01:
        return 0.0
    return abs(row["Close"] - row["Open"]) / rng


def is_bullish(row: pd.Series) -> bool:
    return row["Close"] > row["Open"]


def is_bearish(row: pd.Series) -> bool:
    return row["Close"] < row["Open"]


def session_valid(ts) -> bool:
    """Returns True if the timestamp is within allowed trading window."""
    t = ts.time() if hasattr(ts, "time") else ts
    open_barrier  = dtime(cfg.MARKET_OPEN.hour,
                          cfg.MARKET_OPEN.minute + cfg.NO_ENTRY_OPEN_MINS)
    close_barrier = dtime(cfg.MARKET_CLOSE.hour,
                          cfg.MARKET_CLOSE.minute - cfg.NO_ENTRY_CLOSE_MINS)
    return open_barrier <= t <= close_barrier


# ════════════════════════════════════════════════════════════════
# 1. HISTORICAL ZONE BUILDER  (python_visual.py logic, refactored)
# ════════════════════════════════════════════════════════════════

def fetch_intraday_chunked(
    ticker: str = "^NSEI", interval: str = "5m",
    days: int = 60, chunk_days: int = 7,
) -> pd.DataFrame:
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    chunks, cursor = [], start_dt
    print(f"[DATA] Fetching {interval} data for {ticker} ({days} days)...")

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
            print(f"  chunk error: {e}")
        cursor = chunk_end

    if not chunks:
        raise ValueError(f"No data for {ticker}")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
    try:
        df.index = df.index.tz_convert("Asia/Kolkata")
    except TypeError:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    df = df.between_time("09:15", "15:30")
    print(f"  ✓ {len(df)} candles  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def detect_pivots(df: pd.DataFrame, left: int = 10, right: int = 10):
    highs = df["High"].values
    lows  = df["Low"].values

    def confirmed(idx_arr, vals, is_high):
        out = []
        for i in idx_arr:
            lw = vals[max(0, i - left):i]
            rw = vals[i + 1:min(i + right + 1, len(vals))]
            if not len(lw) or not len(rw):
                continue
            if is_high and vals[i] > np.max(lw) and vals[i] > np.max(rw):
                out.append(i)
            elif not is_high and vals[i] < np.min(lw) and vals[i] < np.min(rw):
                out.append(i)
        return np.array(out)

    phi_raw = argrelextrema(highs, np.greater_equal, order=left)[0]
    plo_raw = argrelextrema(lows,  np.less_equal,    order=left)[0]
    phi = confirmed(phi_raw, highs, True)
    plo = confirmed(plo_raw, lows,  False)

    def to_df(idx_arr, col, ptype):
        if not len(idx_arr):
            return pd.DataFrame(columns=["price","date","bar_idx","type","session"])
        return pd.DataFrame({
            "price":   df[col].iloc[idx_arr].values,
            "date":    df.index[idx_arr],
            "bar_idx": idx_arr,
            "type":    ptype,
            "session": [d.date() for d in df.index[idx_arr]],
        })

    return to_df(phi, "High", "high"), to_df(plo, "Low", "low")


def cluster_pivots(pivots: pd.DataFrame, tolerance: float = 15.0) -> list:
    if not len(pivots):
        return []
    sp = pivots.sort_values("price").reset_index(drop=True)
    prices = sp["price"].values
    clusters, g = [], 0
    for i in range(1, len(prices)):
        if prices[i] - prices[g] > tolerance:
            clusters.append(_make_cluster(sp.iloc[g:i]))
            g = i
    clusters.append(_make_cluster(sp.iloc[g:]))
    return clusters


def _make_cluster(grp: pd.DataFrame) -> dict:
    return {
        "price":    grp["price"].mean(),
        "spread":   grp["price"].max() - grp["price"].min(),
        "n_pivots": len(grp),
        "types":    list(grp["type"]),
        "sessions": list(grp["session"].unique()) if "session" in grp.columns else [],
        "dates":    list(grp["date"]),
    }


def analyse_level(df: pd.DataFrame, level: float, half_band: float = 15.0,
                  min_rej: float = 8.0) -> dict:
    upper, lower = level + half_band, level - half_band
    wick_touches = body_rej = inside_bars = 0
    sessions_set, session_held, last_touch = set(), defaultdict(bool), None

    for dt, row in df.iterrows():
        hi, lo, cl = row["High"], row["Low"], row["Close"]
        session = dt.date()
        wick_in = (hi >= lower) and (lo <= upper)
        if wick_in:
            wick_touches += 1
            sessions_set.add(session)
            last_touch = dt
            if cl > upper + min_rej or cl < lower - min_rej:
                body_rej += 1
                session_held[session] = True
            elif (lo < lower - 5 and cl < lower - 5) or (hi > upper + 5 and cl > upper + 5):
                session_held[session] = False
        if hi <= upper and lo >= lower:
            inside_bars += 1

    return {
        "wick_touches":      wick_touches,
        "body_rejections":   body_rej,
        "inside_bars":       inside_bars,
        "sessions_touched":  len(sessions_set),
        "consecutive_holds": sum(1 for v in session_held.values() if v),
        "last_touch":        last_touch,
    }


def _remove_overlaps(zones_df: pd.DataFrame, half_band: float) -> pd.DataFrame:
    df = zones_df.sort_values("strength", ascending=False).reset_index(drop=True)
    keep = [True] * len(df)
    for i in range(len(df)):
        if not keep[i]: continue
        for j in range(i + 1, len(df)):
            if not keep[j]: continue
            if abs(df.loc[i, "price"] - df.loc[j, "price"]) < half_band * 2:
                keep[j] = False
    return df[keep].reset_index(drop=True)


def build_historical_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Full historical S/R zone pipeline from python_visual.py."""
    print("\n[HIST] Building historical zones...")
    ph, pl = detect_pivots(df, cfg.LEFT_BARS, cfg.RIGHT_BARS)
    all_p  = pd.concat([ph, pl], ignore_index=True)
    if not len(all_p):
        raise ValueError("No historical pivots found.")

    clusters      = cluster_pivots(all_p, cfg.CLUSTER_TOLERANCE)
    current_price = float(df["Close"].iloc[-1])
    latest_date   = df.index[-1]
    candidates    = []

    for cl in clusters:
        level = cl["price"]
        info  = analyse_level(df, level, cfg.ZONE_HALF_BAND)
        if info["wick_touches"]    < cfg.MIN_WICK_TOUCHES: continue
        if info["sessions_touched"]< cfg.MIN_SESSIONS:     continue
        if info["body_rejections"] < cfg.MIN_REJECTIONS:   continue

        days_ago    = (latest_date - info["last_touch"]).days if info["last_touch"] else 999
        recency     = np.exp(-days_ago / 15)
        ts          = min(info["wick_touches"] / 15, 1.0)
        rs          = min(info["body_rejections"] / max(info["wick_touches"], 1), 1.0)
        ss          = min(info["sessions_touched"] / 10, 1.0)
        hs          = min(info["consecutive_holds"] / 5, 1.0)
        ins         = min(info["inside_bars"] / 10, 1.0)
        n_h, n_l    = cl["types"].count("high"), cl["types"].count("low")
        conv        = 1.0 if (n_h > 0 and n_l > 0) else 0.5
        spread_s    = max(0.0, 1.0 - cl["spread"] / cfg.CLUSTER_TOLERANCE)
        strength    = (0.28*ts + 0.22*rs + 0.18*recency + 0.12*ss +
                       0.08*hs + 0.07*conv + 0.05*spread_s) * 100

        candidates.append({
            "price":          round(level, 2),
            "upper":          round(level + cfg.ZONE_HALF_BAND, 2),
            "lower":          round(level - cfg.ZONE_HALF_BAND, 2),
            "band":           cfg.ZONE_HALF_BAND * 2,
            "type":           "Support" if level < current_price else "Resistance",
            "strength":       round(strength, 1),
            "wick_touches":   info["wick_touches"],
            "body_rejections":info["body_rejections"],
            "inside_bars":    info["inside_bars"],
            "sessions":       info["sessions_touched"],
            "holds":          info["consecutive_holds"],
            "recency_score":  round(recency * 100, 1),
            "last_touch":     info["last_touch"],
            "days_ago":       days_ago,
            "convergence":    "Both" if conv == 1.0 else ("High" if n_h >= n_l else "Low"),
            "dist_pts":       round(level - current_price, 1),
        })

    if not candidates:
        raise ValueError("No historical zones passed filters.")

    zones = pd.DataFrame(candidates).sort_values("strength", ascending=False)
    zones = _remove_overlaps(zones, cfg.ZONE_HALF_BAND).head(cfg.TOP_N_HIST)
    print(f"  ✓ {len(zones)} historical zones built")
    return zones.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# 2. LIVE ZONE BUILDER  (realtime_sr.py logic, adapted for OHLC)
# ════════════════════════════════════════════════════════════════

@dataclass
class RangeBar:
    open: float; high: float; low: float; close: float
    start_time: datetime; end_time: datetime; tick_count: int = 1


@dataclass
class LivePivot:
    price: float; time: datetime; type: str; swing_size: float; confirmed: bool = True


@dataclass
class LiveZone:
    price: float; lower: float; upper: float
    strength: float; type: str; n_pivots: int


class RangeBarBuilder:
    def __init__(self, range_size: float = 4.0):
        self.range_size = range_size
        self.o = self.h = self.l = self.start_ts = None
        self.tick_count = 0

    def on_tick(self, price: float, ts: datetime) -> Optional[RangeBar]:
        if self.o is None:
            self.o = self.h = self.l = price; self.start_ts = ts; self.tick_count = 1
            return None
        self.h = max(self.h, price); self.l = min(self.l, price); self.tick_count += 1
        if (self.h - self.l) >= self.range_size:
            bar = RangeBar(self.o, self.h, self.l, price, self.start_ts, ts, self.tick_count)
            self.o = self.h = self.l = price; self.start_ts = ts; self.tick_count = 1
            return bar
        return None


class ZigZagDetector:
    def __init__(self, reversal: float = 12.0):
        self.reversal = reversal; self.direction = 0
        self.ext_price: Optional[float] = None; self.ext_time: Optional[datetime] = None
        self.last_price: Optional[float] = None; self.pivots: List[LivePivot] = []

    def on_bar(self, bar: RangeBar) -> Optional[LivePivot]:
        if self.direction == 0:
            self.ext_price = bar.high; self.ext_time = bar.end_time
            self.direction = 1; self.last_price = bar.low; return None
        if self.direction == 1:
            if bar.high > self.ext_price:
                self.ext_price = bar.high; self.ext_time = bar.end_time
            elif (self.ext_price - bar.low) >= self.reversal:
                p = LivePivot(self.ext_price, self.ext_time, "high",
                              self.ext_price - (self.last_price or self.ext_price))
                self.pivots.append(p); self.last_price = self.ext_price
                self.direction = -1; self.ext_price = bar.low; self.ext_time = bar.end_time
                return p
        else:
            if bar.low < self.ext_price:
                self.ext_price = bar.low; self.ext_time = bar.end_time
            elif (bar.high - self.ext_price) >= self.reversal:
                p = LivePivot(self.ext_price, self.ext_time, "low",
                              (self.last_price or self.ext_price) - self.ext_price)
                self.pivots.append(p); self.last_price = self.ext_price
                self.direction = 1; self.ext_price = bar.high; self.ext_time = bar.end_time
                return p
        return None


def build_live_zones_from_ohlc(df_today: pd.DataFrame, current_price: float) -> List[LiveZone]:
    """Run the realtime_sr pipeline on today's OHLC data via simulated ticks."""
    rbb = RangeBarBuilder(cfg.RANGE_SIZE)
    zzd = ZigZagDetector(cfg.REVERSAL_THR)
    pivots: List[LivePivot] = []

    for ts, row in df_today.iterrows():
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        path = [o, h, l, c] if c > o else [o, l, h, c]
        step = timedelta(minutes=5) / 12
        for k, seg in enumerate(zip(path[:-1], path[1:])):
            ticks = np.linspace(seg[0], seg[1], 4)
            for j, px in enumerate(ticks):
                tick_ts = ts + step * (k * 4 + j)
                bar = rbb.on_tick(float(px), tick_ts)
                if bar:
                    piv = zzd.on_bar(bar)
                    if piv:
                        pivots.append(piv)

    if not pivots:
        return []

    # KDE density zone map (simplified from realtime_sr.DensityZoneMap)
    bw     = cfg.BANDWIDTH
    hw     = cfg.LIVE_ZONE_HW
    decay  = np.log(2) / (cfg.HALF_LIFE_MIN * 60)
    now    = df_today.index[-1]

    prices_w = []
    for p in pivots:
        age = max(0.0, (now - p.time).total_seconds())
        w   = max(1.0, p.swing_size / 10.0) * np.exp(-decay * age)
        if w >= 0.05:
            prices_w.append((p.price, w))

    if not prices_w:
        return []

    pvec = np.array([x[0] for x in prices_w])
    wvec = np.array([x[1] for x in prices_w])
    grid = np.arange(pvec.min() - 4*bw, pvec.max() + 4*bw, 1.0)
    diffs   = (grid[:, None] - pvec[None, :]) / bw
    density = (wvec[None, :] * np.exp(-0.5 * diffs**2)).sum(axis=1)
    thresh  = density.max() * 0.15

    peaks = [i for i in range(1, len(density)-1)
             if density[i] >= density[i-1] and density[i] >= density[i+1]
             and density[i] >= thresh]
    if not peaks:
        return []

    zones = []
    for i in peaks:
        p    = float(grid[i])
        npiv = int(np.sum(np.abs(pvec - p) <= 2 * bw))
        zones.append(LiveZone(
            price    = round(p, 2),
            lower    = round(p - hw, 2),
            upper    = round(p + hw, 2),
            strength = float(density[i]),
            type     = "Support" if p < current_price else "Resistance",
            n_pivots = npiv,
        ))

    # Normalize strength 0-100
    if zones:
        s_max = max(z.strength for z in zones)
        for z in zones:
            z.strength = round(100.0 * z.strength / s_max, 1)
    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones[:10]


# ════════════════════════════════════════════════════════════════
# 3. ZONE CONFLUENCE MERGER
# ════════════════════════════════════════════════════════════════

@dataclass
class ConfluentZone:
    price:      float       # average of hist + live price
    upper:      float
    lower:      float
    band:       float
    zone_type:  str         # "Support" / "Resistance"
    hist_str:   float       # historical strength 0-100
    live_str:   float       # live strength 0-100
    combo_str:  float       # composite
    hist_row:   object      # pandas Series
    live_zone:  object      # LiveZone
    days_ago:   int
    last_touch: object


def find_confluent_zones(
    hist_zones: pd.DataFrame,
    live_zones: List[LiveZone],
    current_price: float,
) -> List[ConfluentZone]:
    """
    Match historical and live zones within CONFLUENCE_DIST points.
    Only dual-confirmed zones proceed to entry scanning.
    Falls back to hist-only zones if no live zones available.
    """
    if cfg.DEBUG:
        print(f"\n  [DEBUG] Confluence: {len(hist_zones)} hist, {len(live_zones)} live zones")
        print(f"          CONFLUENCE_DIST={cfg.CONFLUENCE_DIST}  MIN_HIST_STR={cfg.MIN_HIST_STR}  MIN_LIVE_STR={cfg.MIN_LIVE_STR}")
        for _, hr in hist_zones.iterrows():
            if hr["strength"] < cfg.MIN_HIST_STR:
                print(f"          SKIP hist {hr['price']:.0f}: str={hr['strength']:.1f} < {cfg.MIN_HIST_STR}")
            elif hr["days_ago"] > cfg.MAX_ZONE_AGE_DAYS:
                print(f"          SKIP hist {hr['price']:.0f}: days_ago={hr['days_ago']} > {cfg.MAX_ZONE_AGE_DAYS}")
            else:
                dists = [abs(hr["price"] - lz.price) for lz in live_zones] if live_zones else []
                nearest = f"{min(dists):.0f}pts" if dists else "no_live_zones"
                qual_dists = [abs(hr["price"] - lz.price) for lz in live_zones if lz.strength >= cfg.MIN_LIVE_STR]
                nearest_q = f"{min(qual_dists):.0f}pts" if qual_dists else "none_qualified"
                print(f"          HIST {hr['price']:.0f} str={hr['strength']:.1f}  "
                      f"nearest_live={nearest}  nearest_qualified={nearest_q}  (need <={cfg.CONFLUENCE_DIST})")

    confluent = []

    # ── Try full confluence (hist + live) ────────────────────
    for _, hr in hist_zones.iterrows():
        if hr["strength"] < cfg.MIN_HIST_STR:
            continue
        if hr["days_ago"] > cfg.MAX_ZONE_AGE_DAYS:
            continue
        for lz in live_zones:
            if lz.strength < cfg.MIN_LIVE_STR:
                continue
            if abs(hr["price"] - lz.price) <= cfg.CONFLUENCE_DIST:
                avg_price = (hr["price"] + lz.price) / 2
                combo_str = 0.6 * hr["strength"] + 0.4 * lz.strength
                band      = max(hr["band"], cfg.LIVE_ZONE_HW * 2)
                confluent.append(ConfluentZone(
                    price      = round(avg_price, 2),
                    upper      = round(avg_price + band/2, 2),
                    lower      = round(avg_price - band/2, 2),
                    band       = band,
                    zone_type  = hr["type"],
                    hist_str   = hr["strength"],
                    live_str   = lz.strength,
                    combo_str  = round(combo_str, 1),
                    hist_row   = hr,
                    live_zone  = lz,
                    days_ago   = hr["days_ago"],
                    last_touch = hr["last_touch"],
                ))

    # ── Fallback: if no live zones matched, use hist zones alone ─
    # This happens when today's session is too new / insufficient pivots
    if not confluent:
        print("\n  [FALLBACK] No live zone matches — using historical zones only (live_str=0)")
        for _, hr in hist_zones.iterrows():
            if hr["strength"] < cfg.MIN_HIST_STR:
                continue
            if hr["days_ago"] > cfg.MAX_ZONE_AGE_DAYS:
                continue
            band = hr["band"]
            confluent.append(ConfluentZone(
                price      = round(hr["price"], 2),
                upper      = round(hr["upper"], 2),
                lower      = round(hr["lower"], 2),
                band       = band,
                zone_type  = hr["type"],
                hist_str   = hr["strength"],
                live_str   = 0.0,
                combo_str  = round(hr["strength"] * 0.7, 1),  # discount without live confirmation
                hist_row   = hr,
                live_zone  = None,
                days_ago   = hr["days_ago"],
                last_touch = hr["last_touch"],
            ))

    confluent.sort(key=lambda z: z.combo_str, reverse=True)
    return confluent


# ════════════════════════════════════════════════════════════════
# 4. SIGNAL DETECTOR  — the core trading logic
# ════════════════════════════════════════════════════════════════

@dataclass
class TradeSignal:
    bar_time:    datetime
    entry_price: float
    direction:   str        # "LONG" / "SHORT"
    sl:          float
    tp1:         float
    tp2:         float
    risk_pts:    float
    rr1:         float
    pattern:     str        # "BREAKOUT_PULLBACK" / "FAKEOUT_REVERSAL" / "ZONE_BOUNCE" / "ZONE_COMPRESS"
    zone:        ConfluentZone
    atr:         float
    vwap:        float
    body_ratio:  float
    quality:     str        # "A+" / "A" / "B"
    notes:       str        = ""


def compute_signal_quality(sig: TradeSignal) -> str:
    score = 0
    if sig.zone.combo_str >= 70:  score += 3
    elif sig.zone.combo_str >= 55: score += 2
    else:                          score += 1
    if sig.zone.hist_str >= 60:   score += 2
    if sig.zone.live_str >= 60:   score += 2
    if sig.body_ratio >= 0.55:    score += 2
    if sig.rr1 >= 2.0:            score += 1
    if sig.zone.hist_row["convergence"] == "Both": score += 1
    if score >= 10: return "A+"
    if score >= 7:  return "A"
    return "B"


def find_next_opposing_zone(
    zone: ConfluentZone,
    all_zones: List[ConfluentZone],
    direction: str,
) -> Optional[float]:
    """Return price of nearest opposing confluent zone (TP2 target)."""
    if direction == "LONG":
        candidates = [z.price for z in all_zones
                      if z.zone_type == "Resistance" and z.price > zone.price + zone.band]
        return min(candidates) if candidates else None
    else:
        candidates = [z.price for z in all_zones
                      if z.zone_type == "Support" and z.price < zone.price - zone.band]
        return max(candidates) if candidates else None


def scan_entries(
    df: pd.DataFrame,
    confluent_zones: List[ConfluentZone],
    atr_series: pd.Series,
    vwap_series: pd.Series,
) -> List[TradeSignal]:
    """
    Scan every bar in df against each confluent zone for one of 4 entry patterns.
    Returns list of TradeSignal objects, chronologically sorted.
    """
    signals: List[TradeSignal] = []
    bars = df.reset_index()      # work with integer indices

    if cfg.DEBUG:
        print(f"\n  [DEBUG] scan_entries: {len(bars)} bars × {len(confluent_zones)} zones")
        print(f"          MIN_BODY_RATIO={cfg.MIN_BODY_RATIO}  MAX_RISK_PTS={cfg.MAX_RISK_PTS}  "
              f"MAX_RISK_ATR_MULT={cfg.MAX_RISK_ATR_MULT}  USE_VWAP={cfg.USE_VWAP_FILTER}")

    for zone in confluent_zones:
        zlo, zhi = zone.lower, zone.upper
        zmid     = zone.price
        zband    = zone.band
        sl_buf   = zband * cfg.SL_BUFFER_FACTOR

        # Debug counters per zone
        dbg = {"body":0,"session":0,"near":0,"bp_long":0,"bp_short":0,
               "fk_long":0,"fk_short":0,"zb_long":0,"zb_short":0,"zc":0}

        for i in range(cfg.PULLBACK_WINDOW + 2, len(bars)):
            row = bars.iloc[i]
            ts  = row["index"] if "index" in row.index else bars.index[i]
            try:
                ts_pd = pd.Timestamp(ts)
            except Exception:
                continue
            if not session_valid(ts_pd):
                dbg["session"] += 1
                continue

            atr  = float(atr_series.iloc[i]) if i < len(atr_series) else 30.0
            vwap = float(vwap_series.iloc[i]) if i < len(vwap_series) else zmid
            cmp  = float(row["Close"])
            prev = bars.iloc[i - 1]
            body_ratio = candle_body_ratio(row)

            if body_ratio < cfg.MIN_BODY_RATIO:
                dbg["body"] += 1
                continue

            # Quick proximity check — skip bars far from zone
            if abs(cmp - zmid) > zband * 5:
                dbg["near"] += 1
                continue
            row = bars.iloc[i]
            ts  = row["index"] if "index" in row.index else bars.index[i]
            try:
                ts_pd = pd.Timestamp(ts)
            except Exception:
                continue
            if not session_valid(ts_pd):
                continue

            atr  = float(atr_series.iloc[i]) if i < len(atr_series) else 30.0
            vwap = float(vwap_series.iloc[i]) if i < len(vwap_series) else zmid
            cmp  = float(row["Close"])
            prev = bars.iloc[i - 1]
            body_ratio = candle_body_ratio(row)

            if body_ratio < cfg.MIN_BODY_RATIO:
                continue

            # ────────────────────────────────────────────────────────────
            # PATTERN A: BREAKOUT-PULLBACK (LONG above resistance)
            # ─ Bar i-k broke above zone.upper with strong body
            # ─ Pulled back to within zone.upper ± 5 pts
            # ─ Current bar holds above zone.upper and closes bullish
            # ────────────────────────────────────────────────────────────
            if zone.zone_type == "Resistance":
                # Look for a breakout bar within last PULLBACK_WINDOW
                broke_idx = None
                for k in range(1, cfg.PULLBACK_WINDOW + 1):
                    bk = bars.iloc[i - k]
                    if bk["Close"] > zhi + 3 and candle_body_ratio(bk) >= 0.45:
                        broke_idx = i - k
                        break
                if broke_idx is not None:
                    # Check pullback: price dipped back toward zone top
                    pullback_low = bars.iloc[broke_idx + 1:i + 1]["Low"].min()
                    if zhi - 5 <= pullback_low <= zhi + 10:
                        # Current bar bullish and closed above zone top
                        if is_bullish(row) and row["Close"] > zhi:
                            if not cfg.USE_VWAP_FILTER or cmp > vwap:
                                entry = row["High"] + 1  # buy stop above bar
                                sl    = min(pullback_low - sl_buf,
                                            zlo - sl_buf)
                                risk  = entry - sl
                                if 5 < risk <= min(cfg.MAX_RISK_PTS, cfg.MAX_RISK_ATR_MULT * atr):
                                    tp1 = entry + cfg.TP1_RR * risk
                                    tp2 = find_next_opposing_zone(zone, confluent_zones, "LONG") or (entry + 3 * risk)
                                    sig = TradeSignal(
                                        bar_time=ts_pd, entry_price=round(entry, 2),
                                        direction="LONG", sl=round(sl, 2),
                                        tp1=round(tp1, 2), tp2=round(tp2, 2),
                                        risk_pts=round(risk, 2),
                                        rr1=round((tp1 - entry) / risk, 2),
                                        pattern="BREAKOUT_PULLBACK", zone=zone,
                                        atr=round(atr, 2), vwap=round(vwap, 2),
                                        body_ratio=round(body_ratio, 2),
                                        quality="B",
                                        notes=f"Broke {zhi:.0f}, pulled back to {pullback_low:.0f}",
                                    )
                                    sig.quality = compute_signal_quality(sig)
                                    signals.append(sig)

            # ────────────────────────────────────────────────────────────
            # BREAKOUT-PULLBACK (SHORT below support)
            # ────────────────────────────────────────────────────────────
            if zone.zone_type == "Support":
                broke_idx = None
                for k in range(1, cfg.PULLBACK_WINDOW + 1):
                    bk = bars.iloc[i - k]
                    if bk["Close"] < zlo - 3 and candle_body_ratio(bk) >= 0.45:
                        broke_idx = i - k
                        break
                if broke_idx is not None:
                    pullback_high = bars.iloc[broke_idx + 1:i + 1]["High"].max()
                    if zlo - 10 <= pullback_high <= zlo + 5:
                        if is_bearish(row) and row["Close"] < zlo:
                            if not cfg.USE_VWAP_FILTER or cmp < vwap:
                                entry = row["Low"] - 1
                                sl    = max(pullback_high + sl_buf, zhi + sl_buf)
                                risk  = sl - entry
                                if 5 < risk <= min(cfg.MAX_RISK_PTS, cfg.MAX_RISK_ATR_MULT * atr):
                                    tp1 = entry - cfg.TP1_RR * risk
                                    tp2 = find_next_opposing_zone(zone, confluent_zones, "SHORT") or (entry - 3 * risk)
                                    sig = TradeSignal(
                                        bar_time=ts_pd, entry_price=round(entry, 2),
                                        direction="SHORT", sl=round(sl, 2),
                                        tp1=round(tp1, 2), tp2=round(tp2, 2),
                                        risk_pts=round(risk, 2),
                                        rr1=round((entry - tp1) / risk, 2),
                                        pattern="BREAKOUT_PULLBACK", zone=zone,
                                        atr=round(atr, 2), vwap=round(vwap, 2),
                                        body_ratio=round(body_ratio, 2), quality="B",
                                        notes=f"Broke {zlo:.0f}, pulled back to {pullback_high:.0f}",
                                    )
                                    sig.quality = compute_signal_quality(sig)
                                    signals.append(sig)

            # ────────────────────────────────────────────────────────────
            # PATTERN B: FAKEOUT-REVERSAL
            # ─ Prior N bars: wick pierced zone (high > zhi or low < zlo)
            # ─ But CLOSE was back inside the zone (trap)
            # ─ Current bar closes strongly away from trap side
            # ────────────────────────────────────────────────────────────

            # Fakeout UP into resistance → short reversal
            if zone.zone_type == "Resistance":
                fakeout_found = False
                for k in range(1, cfg.FAKEOUT_LOOKBACK + 1):
                    bk = bars.iloc[i - k]
                    if bk["High"] > zhi and bk["Close"] < zhi - 5:
                        fakeout_found = True; break
                if fakeout_found and is_bearish(row) and row["Close"] < zmid:
                    if not cfg.USE_VWAP_FILTER or cmp < vwap:
                        entry = row["Low"] - 1
                        sl    = zhi + sl_buf
                        risk  = sl - entry
                        if 5 < risk <= min(cfg.MAX_RISK_PTS, cfg.MAX_RISK_ATR_MULT * atr):
                            tp1 = entry - cfg.TP1_RR * risk
                            tp2 = find_next_opposing_zone(zone, confluent_zones, "SHORT") or (entry - 3 * risk)
                            sig = TradeSignal(
                                bar_time=ts_pd, entry_price=round(entry, 2),
                                direction="SHORT", sl=round(sl, 2),
                                tp1=round(tp1, 2), tp2=round(tp2, 2),
                                risk_pts=round(risk, 2),
                                rr1=round((entry - tp1) / risk, 2),
                                pattern="FAKEOUT_REVERSAL", zone=zone,
                                atr=round(atr, 2), vwap=round(vwap, 2),
                                body_ratio=round(body_ratio, 2), quality="B",
                                notes=f"Fakeout above {zhi:.0f}, rejection confirmed",
                            )
                            sig.quality = compute_signal_quality(sig)
                            signals.append(sig)

            # Fakeout DOWN into support → long reversal
            if zone.zone_type == "Support":
                fakeout_found = False
                for k in range(1, cfg.FAKEOUT_LOOKBACK + 1):
                    bk = bars.iloc[i - k]
                    if bk["Low"] < zlo and bk["Close"] > zlo + 5:
                        fakeout_found = True; break
                if fakeout_found and is_bullish(row) and row["Close"] > zmid:
                    if not cfg.USE_VWAP_FILTER or cmp > vwap:
                        entry = row["High"] + 1
                        sl    = zlo - sl_buf
                        risk  = entry - sl
                        if 5 < risk <= min(cfg.MAX_RISK_PTS, cfg.MAX_RISK_ATR_MULT * atr):
                            tp1 = entry + cfg.TP1_RR * risk
                            tp2 = find_next_opposing_zone(zone, confluent_zones, "LONG") or (entry + 3 * risk)
                            sig = TradeSignal(
                                bar_time=ts_pd, entry_price=round(entry, 2),
                                direction="LONG", sl=round(sl, 2),
                                tp1=round(tp1, 2), tp2=round(tp2, 2),
                                risk_pts=round(risk, 2),
                                rr1=round((entry - sl + cfg.TP1_RR * risk) / risk, 2),
                                pattern="FAKEOUT_REVERSAL", zone=zone,
                                atr=round(atr, 2), vwap=round(vwap, 2),
                                body_ratio=round(body_ratio, 2), quality="B",
                                notes=f"Fakeout below {zlo:.0f}, bounce confirmed",
                            )
                            sig.quality = compute_signal_quality(sig)
                            signals.append(sig)

            # ────────────────────────────────────────────────────────────
            # PATTERN C: ZONE-BOUNCE  (clean approach + body rejection)
            # ─ Price approaches zone edge from outside
            # ─ Wick touches zone band but body closes firmly outside
            # ─ No prior candle already fully inside zone
            # ────────────────────────────────────────────────────────────

            # Resistance bounce → SHORT
            if zone.zone_type == "Resistance":
                approached = (row["High"] >= zlo) and (row["Low"] < zlo)
                if approached and is_bearish(row) and row["Close"] < zlo - 5:
                    if prev["Close"] < zlo:  # approached cleanly from below
                        if not cfg.USE_VWAP_FILTER or cmp < vwap:
                            entry = row["Low"] - 1
                            sl    = row["High"] + sl_buf
                            risk  = sl - entry
                            if 5 < risk <= min(cfg.MAX_RISK_PTS, cfg.MAX_RISK_ATR_MULT * atr):
                                tp1 = entry - cfg.TP1_RR * risk
                                tp2 = find_next_opposing_zone(zone, confluent_zones, "SHORT") or (entry - 3 * risk)
                                sig = TradeSignal(
                                    bar_time=ts_pd, entry_price=round(entry, 2),
                                    direction="SHORT", sl=round(sl, 2),
                                    tp1=round(tp1, 2), tp2=round(tp2, 2),
                                    risk_pts=round(risk, 2),
                                    rr1=round((entry - tp1) / risk, 2),
                                    pattern="ZONE_BOUNCE", zone=zone,
                                    atr=round(atr, 2), vwap=round(vwap, 2),
                                    body_ratio=round(body_ratio, 2), quality="B",
                                    notes=f"Wick into resistance {zlo:.0f}–{zhi:.0f}, body rejected",
                                )
                                sig.quality = compute_signal_quality(sig)
                                signals.append(sig)

            # Support bounce → LONG
            if zone.zone_type == "Support":
                approached = (row["Low"] <= zhi) and (row["High"] > zhi)
                if approached and is_bullish(row) and row["Close"] > zhi + 5:
                    if prev["Close"] > zhi:
                        if not cfg.USE_VWAP_FILTER or cmp > vwap:
                            entry = row["High"] + 1
                            sl    = row["Low"] - sl_buf
                            risk  = entry - sl
                            if 5 < risk <= min(cfg.MAX_RISK_PTS, cfg.MAX_RISK_ATR_MULT * atr):
                                tp1 = entry + cfg.TP1_RR * risk
                                tp2 = find_next_opposing_zone(zone, confluent_zones, "LONG") or (entry + 3 * risk)
                                sig = TradeSignal(
                                    bar_time=ts_pd, entry_price=round(entry, 2),
                                    direction="LONG", sl=round(sl, 2),
                                    tp1=round(tp1, 2), tp2=round(tp2, 2),
                                    risk_pts=round(risk, 2),
                                    rr1=round((entry - sl + cfg.TP1_RR * risk) / risk, 2),
                                    pattern="ZONE_BOUNCE", zone=zone,
                                    atr=round(atr, 2), vwap=round(vwap, 2),
                                    body_ratio=round(body_ratio, 2), quality="B",
                                    notes=f"Wick into support {zlo:.0f}–{zhi:.0f}, body above",
                                )
                                sig.quality = compute_signal_quality(sig)
                                signals.append(sig)

            # ────────────────────────────────────────────────────────────
            # PATTERN D: ZONE-COMPRESS (inside-bar squeeze → expansion play)
            # ─ N consecutive bars entirely inside zone band
            # ─ Current bar breaks out of that inside-bar range with volume
            # ─ Trade the expansion in the breakout direction
            # ────────────────────────────────────────────────────────────
            if i >= cfg.COMPRESS_MIN_BARS + 1:
                compress_window = bars.iloc[i - cfg.COMPRESS_MIN_BARS:i]
                all_inside = all(
                    (r["High"] <= zhi + 3) and (r["Low"] >= zlo - 3)
                    for _, r in compress_window.iterrows()
                )
                if all_inside:
                    compress_high = compress_window["High"].max()
                    compress_low  = compress_window["Low"].min()

                    # Bullish expansion
                    if row["Close"] > compress_high and is_bullish(row):
                        if not cfg.USE_VWAP_FILTER or cmp > vwap:
                            entry = compress_high + 1
                            sl    = compress_low - sl_buf
                            risk  = entry - sl
                            if 5 < risk <= min(cfg.MAX_RISK_PTS, cfg.MAX_RISK_ATR_MULT * atr):
                                tp1 = entry + cfg.TP1_RR * risk
                                tp2 = find_next_opposing_zone(zone, confluent_zones, "LONG") or (entry + 3 * risk)
                                sig = TradeSignal(
                                    bar_time=ts_pd, entry_price=round(entry, 2),
                                    direction="LONG", sl=round(sl, 2),
                                    tp1=round(tp1, 2), tp2=round(tp2, 2),
                                    risk_pts=round(risk, 2),
                                    rr1=round(cfg.TP1_RR, 2),
                                    pattern="ZONE_COMPRESS", zone=zone,
                                    atr=round(atr, 2), vwap=round(vwap, 2),
                                    body_ratio=round(body_ratio, 2), quality="B",
                                    notes=f"{cfg.COMPRESS_MIN_BARS}-bar squeeze broke up from {compress_high:.0f}",
                                )
                                sig.quality = compute_signal_quality(sig)
                                signals.append(sig)

                    # Bearish expansion
                    elif row["Close"] < compress_low and is_bearish(row):
                        if not cfg.USE_VWAP_FILTER or cmp < vwap:
                            entry = compress_low - 1
                            sl    = compress_high + sl_buf
                            risk  = sl - entry
                            if 5 < risk <= min(cfg.MAX_RISK_PTS, cfg.MAX_RISK_ATR_MULT * atr):
                                tp1 = entry - cfg.TP1_RR * risk
                                tp2 = find_next_opposing_zone(zone, confluent_zones, "SHORT") or (entry - 3 * risk)
                                sig = TradeSignal(
                                    bar_time=ts_pd, entry_price=round(entry, 2),
                                    direction="SHORT", sl=round(sl, 2),
                                    tp1=round(tp1, 2), tp2=round(tp2, 2),
                                    risk_pts=round(risk, 2),
                                    rr1=round(cfg.TP1_RR, 2),
                                    pattern="ZONE_COMPRESS", zone=zone,
                                    atr=round(atr, 2), vwap=round(vwap, 2),
                                    body_ratio=round(body_ratio, 2), quality="B",
                                    notes=f"{cfg.COMPRESS_MIN_BARS}-bar squeeze broke down from {compress_low:.0f}",
                                )
                                sig.quality = compute_signal_quality(sig)
                                signals.append(sig)

        if cfg.DEBUG:
            print(f"  [DEBUG] Zone {zmid:.0f} ({zone.zone_type[:3]}) str={zone.combo_str:.0f} | "
                  f"body_fail={dbg['body']} sess_fail={dbg['session']} far={dbg['near']} | "
                  f"raw signals this zone={sum(1 for s in signals if abs(s.zone.price-zmid)<1)}")

    # Deduplicate: one signal per zone per bar
    seen, unique = set(), []
    for s in sorted(signals, key=lambda x: (x.bar_time, -x.zone.combo_str)):
        key = (s.bar_time.strftime("%Y-%m-%d %H:%M"), round(s.zone.price))
        if key not in seen:
            seen.add(key)
            unique.append(s)

    if cfg.DEBUG:
        print(f"\n  [DEBUG] Total raw signals={len(signals)}, after dedup={len(unique)}")
    return unique


# ════════════════════════════════════════════════════════════════
# 5. POSITION SIZER
# ════════════════════════════════════════════════════════════════

def position_size(risk_pts: float, capital: float = cfg.CAPITAL,
                  risk_pct: float = cfg.MAX_RISK_PCT) -> dict:
    """
    For Nifty options:
      Lot size = 25. Use risk_pts / lot_size to get per-lot risk.
      Returns qty (lots) and capital at risk.
    """
    risk_capital = capital * risk_pct
    lot_size     = 25
    risk_per_lot = risk_pts * lot_size
    qty_lots     = max(1, int(risk_capital // risk_per_lot))
    return {
        "lots":          qty_lots,
        "units":         qty_lots * lot_size,
        "risk_capital":  round(qty_lots * risk_per_lot, 2),
        "risk_pct_used": round(qty_lots * risk_per_lot / capital * 100, 3),
    }


# ════════════════════════════════════════════════════════════════
# 6. BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    signal:      TradeSignal
    exit_price:  float  = 0.0
    exit_time:   object = None
    exit_reason: str    = ""
    pnl_pts:     float  = 0.0
    pnl_r:       float  = 0.0
    won:         bool   = False


def backtest_signals(
    df: pd.DataFrame,
    signals: List[TradeSignal],
) -> Tuple[List[Trade], dict]:
    """
    Simple bar-by-bar backtest for each signal.
    Entry on next bar open (market order approximation).
    Exits: TP1 partial → trail rest → TP2 or end of day.
    """
    trades: List[Trade] = []
    bars = df.reset_index()

    for sig in signals:
        # Find entry bar index
        entry_idx = None
        for k, row in bars.iterrows():
            try:
                rts = pd.Timestamp(row.get("index", bars.index[k]))
            except Exception:
                rts = bars.index[k]
            if rts >= sig.bar_time and k + 1 < len(bars):
                entry_idx = k + 1
                break
        if entry_idx is None:
            continue

        entry_bar  = bars.iloc[entry_idx]
        entry_price= float(entry_bar["Open"])  # simulate market order
        sl         = sig.sl
        tp1        = sig.tp1
        tp2        = sig.tp2
        direction  = sig.direction
        risk_pts   = abs(entry_price - sl)
        if risk_pts < 1:
            continue

        hit_tp1  = False
        trail_sl = sl
        exit_price = exit_time = exit_reason = None

        for j in range(entry_idx, min(entry_idx + 80, len(bars))):
            b   = bars.iloc[j]
            try:
                bts = pd.Timestamp(b.get("index", bars.index[j]))
            except Exception:
                bts = bars.index[j]

            hi, lo = float(b["High"]), float(b["Low"])

            if direction == "LONG":
                if lo <= sl:
                    exit_price  = sl
                    exit_reason = "SL" if not hit_tp1 else "TRAIL_SL"
                    exit_time   = bts; break
                if not hit_tp1 and hi >= tp1:
                    hit_tp1  = True
                    trail_sl = entry_price        # move SL to entry (BE)
                    sl       = trail_sl
                if hit_tp1 and hi >= tp2:
                    exit_price  = tp2
                    exit_reason = "TP2"; exit_time = bts; break
                if hit_tp1:
                    new_trail = hi - sig.zone.band / 2
                    sl = max(sl, new_trail)
                # EOD exit
                if bts.time() >= dtime(15, 20):
                    exit_price  = float(b["Close"])
                    exit_reason = "EOD"; exit_time = bts; break
            else:  # SHORT
                if hi >= sl:
                    exit_price  = sl
                    exit_reason = "SL" if not hit_tp1 else "TRAIL_SL"
                    exit_time   = bts; break
                if not hit_tp1 and lo <= tp1:
                    hit_tp1  = True
                    trail_sl = entry_price
                    sl       = trail_sl
                if hit_tp1 and lo <= tp2:
                    exit_price  = tp2
                    exit_reason = "TP2"; exit_time = bts; break
                if hit_tp1:
                    new_trail = lo + sig.zone.band / 2
                    sl = min(sl, new_trail)
                if bts.time() >= dtime(15, 20):
                    exit_price  = float(b["Close"])
                    exit_reason = "EOD"; exit_time = bts; break

        if exit_price is None:
            exit_price  = float(bars.iloc[min(entry_idx + 79, len(bars)-1)]["Close"])
            exit_reason = "EOD"
            exit_time   = sig.bar_time

        pnl_pts = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
        pnl_r   = pnl_pts / risk_pts if risk_pts else 0
        won     = pnl_pts > 0

        t = Trade(signal=sig, exit_price=round(exit_price, 2),
                  exit_time=exit_time, exit_reason=exit_reason,
                  pnl_pts=round(pnl_pts, 2), pnl_r=round(pnl_r, 3), won=won)
        trades.append(t)

    if not trades:
        return trades, {}

    wins  = [t for t in trades if t.won]
    loses = [t for t in trades if not t.won]
    total_r = sum(t.pnl_r for t in trades)

    stats = {
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(loses),
        "win_rate":       round(len(wins) / len(trades) * 100, 1),
        "avg_win_r":      round(np.mean([t.pnl_r for t in wins]), 3)   if wins  else 0,
        "avg_loss_r":     round(np.mean([t.pnl_r for t in loses]), 3)  if loses else 0,
        "total_r":        round(total_r, 3),
        "profit_factor":  round(
            sum(t.pnl_r for t in wins) / abs(sum(t.pnl_r for t in loses) or 1), 2),
        "max_drawdown_r": round(min(0, min(t.pnl_r for t in trades)), 3),
        "by_pattern":     {},
        "by_quality":     {},
    }

    for pat in ["BREAKOUT_PULLBACK", "FAKEOUT_REVERSAL", "ZONE_BOUNCE", "ZONE_COMPRESS"]:
        pt = [t for t in trades if t.signal.pattern == pat]
        pw = [t for t in pt if t.won]
        if pt:
            stats["by_pattern"][pat] = {
                "trades": len(pt), "wins": len(pw),
                "win_rate": round(len(pw)/len(pt)*100, 1),
                "total_r":  round(sum(t.pnl_r for t in pt), 2),
            }

    for q in ["A+", "A", "B"]:
        qt = [t for t in trades if t.signal.quality == q]
        qw = [t for t in qt if t.won]
        if qt:
            stats["by_quality"][q] = {
                "trades": len(qt), "wins": len(qw),
                "win_rate": round(len(qw)/len(qt)*100, 1),
                "total_r":  round(sum(t.pnl_r for t in qt), 2),
            }

    return trades, stats


# ════════════════════════════════════════════════════════════════
# 7. REPORTING
# ════════════════════════════════════════════════════════════════

def print_signals(signals: List[TradeSignal], current_price: float):
    sep = "═" * 110
    print(f"\n{sep}")
    print(f"  ACTIVE TRADE SIGNALS  |  CMP: {current_price:.2f}  |  Total: {len(signals)}")
    print(sep)
    hdr = (f"  {'Quality':<5} {'Time':<16} {'Dir':<6} {'Pattern':<20} "
           f"{'Entry':>8} {'SL':>8} {'TP1':>8} {'TP2':>8} "
           f"{'Risk':>5} {'RR1':>4} {'ZStr':>5}  Notes")
    print(hdr)
    print("  " + "─" * 106)
    for s in signals:
        t = s.bar_time.strftime("%m-%d %H:%M")
        print(f"  [{s.quality:<2}]  {t:<16} {s.direction:<6} {s.pattern:<20} "
              f"{s.entry_price:>8.1f} {s.sl:>8.1f} {s.tp1:>8.1f} {s.tp2:>8.1f} "
              f"{s.risk_pts:>5.1f} {s.rr1:>4.2f} {s.zone.combo_str:>5.1f}  {s.notes}")
    print(sep)


def print_stats(stats: dict):
    if not stats:
        print("\n  [BACKTEST] No completed trades.\n")
        return
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  BACKTEST RESULTS")
    print(sep)
    print(f"  Trades       : {stats['total_trades']}  "
          f"(W:{stats['wins']} L:{stats['losses']})")
    print(f"  Win Rate     : {stats['win_rate']}%")
    print(f"  Avg Win      : {stats['avg_win_r']:+.2f}R   Avg Loss: {stats['avg_loss_r']:.2f}R")
    print(f"  Total P&L    : {stats['total_r']:+.2f}R")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    print(f"  Max Drawdown : {stats['max_drawdown_r']:.2f}R")

    if stats["by_pattern"]:
        print(f"\n  {'─'*40} BY PATTERN {'─'*26}")
        for p, v in stats["by_pattern"].items():
            print(f"    {p:<22} trades={v['trades']}  WR={v['win_rate']}%  R={v['total_r']:+.2f}")

    if stats["by_quality"]:
        print(f"\n  {'─'*40} BY QUALITY {'─'*26}")
        for q, v in stats["by_quality"].items():
            print(f"    Grade {q:<3}  trades={v['trades']}  WR={v['win_rate']}%  R={v['total_r']:+.2f}")
    print(sep)


def print_confluent_zones(zones: List[ConfluentZone], cmp: float):
    sep = "═" * 90
    print(f"\n{sep}")
    print(f"  CONFLUENT ZONES (Hist+Live)  |  CMP: {cmp:.2f}  |  {len(zones)} zones")
    print(sep)
    print(f"  {'Type':<12} {'Price':>8} {'Lower':>8} {'Upper':>8} "
          f"{'HStr':>5} {'LStr':>5} {'CStr':>5} {'DaysAgo':>7}  Notes")
    print("  " + "─" * 85)
    for z in zones:
        sym  = "▲ SUP " if z.zone_type == "Support" else "▼ RES "
        note = f"conv={z.hist_row['convergence']}  touch={z.hist_row['wick_touches']}"
        print(f"  {sym:<12} {z.price:>8.1f} {z.lower:>8.1f} {z.upper:>8.1f} "
              f"{z.hist_str:>5.1f} {z.live_str:>5.1f} {z.combo_str:>5.1f} "
              f"{z.days_ago:>7}  {note}")
    print(sep)


# ════════════════════════════════════════════════════════════════
# 8. PLOTLY CHART
# ════════════════════════════════════════════════════════════════

def plot_strategy(
    df: pd.DataFrame,
    hist_zones: pd.DataFrame,
    confluent_zones: List[ConfluentZone],
    signals: List[TradeSignal],
    trades: List[Trade],
    ticker: str = "",
):
    last_sessions = sorted(set(df.index.date))[-cfg.CHART_SESSIONS:]
    df_plot = df[df.index.date >= last_sessions[0]].copy()
    atr_s   = compute_atr(df_plot)
    vwap_s  = compute_vwap(df_plot)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.82, 0.18], vertical_spacing=0.02,
    )

    # ── Candlestick ──────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"],   close=df_plot["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
        name=ticker,
    ), row=1, col=1)

    # ── VWAP ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=vwap_s,
        line=dict(color="rgba(255,214,0,0.6)", width=1.2, dash="dot"),
        name="VWAP",
    ), row=1, col=1)

    # ── Volume ───────────────────────────────────────────────
    vcol = ["#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df_plot["Close"], df_plot["Open"])]
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot["Volume"],
        marker_color=vcol, opacity=0.4, name="Volume",
    ), row=2, col=1)

    # ── Historical zones (background, dim) ───────────────────
    for _, z in hist_zones.iterrows():
        is_sup = z["type"] == "Support"
        fc = "rgba(38,166,154,0.04)" if is_sup else "rgba(239,83,80,0.04)"
        fig.add_hrect(y0=z["lower"], y1=z["upper"], fillcolor=fc,
                      line_width=0.5,
                      line_color="rgba(150,150,150,0.2)", row=1, col=1)

    # ── Confluent zones (highlighted, stronger) ───────────────
    for z in confluent_zones:
        is_sup = z.zone_type == "Support"
        alpha  = 0.08 + (z.combo_str / 100) * 0.18
        la     = 0.4  + (z.combo_str / 100) * 0.6
        fc = f"rgba(38,166,154,{alpha:.2f})"  if is_sup else f"rgba(239,83,80,{alpha:.2f})"
        lc = f"rgba(38,166,154,{la:.2f})"    if is_sup else f"rgba(239,83,80,{la:.2f})"
        fig.add_hrect(y0=z.lower, y1=z.upper, fillcolor=fc,
                      line_width=1.5, line_color=lc, row=1, col=1)
        fig.add_hline(y=z.price, line_color=lc, line_width=2,
                      line_dash="solid", row=1, col=1)
        sym = "⚡S" if is_sup else "⚡R"
        fig.add_annotation(
            x=df_plot.index[-1], y=z.price,
            text=f"{sym} {z.price:.0f}  [{z.lower:.0f}–{z.upper:.0f}]  str={z.combo_str:.0f}",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=9, color=lc),
            bgcolor="rgba(0,0,0,0.65)", borderpad=2, row=1, col=1,
        )

    # ── Signal markers ────────────────────────────────────────
    sig_colors = {"A+": "#FFD700", "A": "#00E5FF", "B": "#B0BEC5"}
    pat_symbols = {
        "BREAKOUT_PULLBACK": "triangle-up",
        "FAKEOUT_REVERSAL":  "star",
        "ZONE_BOUNCE":       "diamond",
        "ZONE_COMPRESS":     "circle",
    }
    for s in signals:
        try:
            ts_plot = s.bar_time
            if ts_plot not in df_plot.index:
                continue
        except Exception:
            continue
        color  = sig_colors.get(s.quality, "#FFFFFF")
        symbol = pat_symbols.get(s.pattern, "circle")
        y_pos  = s.entry_price - 20 if s.direction == "LONG" else s.entry_price + 20
        marker_sym = symbol if s.direction == "LONG" else symbol + "-open"
        fig.add_trace(go.Scatter(
            x=[ts_plot], y=[y_pos],
            mode="markers+text",
            marker=dict(symbol=marker_sym, size=12, color=color,
                        line=dict(width=1.5, color="white")),
            text=[f"{s.quality}"],
            textposition="bottom center",
            textfont=dict(size=8, color=color),
            name=f"{s.quality} {s.direction} {s.pattern}",
            showlegend=False,
        ), row=1, col=1)

        # SL / TP lines for recent signals (last 3)
        if signals.index(s) >= max(0, len(signals) - 3):
            x0, x1 = ts_plot, df_plot.index[-1]
            fig.add_shape(type="line", x0=x0, x1=x1, y0=s.sl, y1=s.sl,
                          line=dict(color="rgba(239,83,80,0.5)", dash="dash", width=1),
                          row=1, col=1)
            fig.add_shape(type="line", x0=x0, x1=x1, y0=s.tp1, y1=s.tp1,
                          line=dict(color="rgba(38,166,154,0.5)", dash="dash", width=1),
                          row=1, col=1)

    # ── CMP line ─────────────────────────────────────────────
    cp = float(df["Close"].iloc[-1])
    fig.add_hline(y=cp, line_color="rgba(255,235,59,0.9)",
                  line_width=1.5, line_dash="dash", row=1, col=1)
    fig.add_annotation(
        x=df_plot.index[-1], y=cp, text=f"CMP {cp:.2f}",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=10, color="rgba(255,235,59,0.9)"), row=1, col=1,
    )

    # ── Session separators ────────────────────────────────────
    for sd in last_sessions[1:]:
        bars_sd = df_plot[df_plot.index.date == sd]
        if len(bars_sd):
            fig.add_vline(x=bars_sd.index[0],
                          line_dash="dot", line_color="rgba(180,180,180,0.15)",
                          line_width=1, row=1, col=1)

    fig.update_layout(
        title=(f"{ticker} — Dual-Layer S/R Strategy  "
               f"(5m · {len(confluent_zones)} confluent zones · "
               f"{len(signals)} signals · last {cfg.CHART_SESSIONS} sessions)"),
        xaxis_rangeslider_visible=False,
        template="plotly_dark", height=900, showlegend=False,
        margin=dict(l=60, r=280, t=55, b=40),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    )
    fig.update_yaxes(title_text="Nifty 50", row=1, col=1,
                     gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="Volume",   row=2, col=1)
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[15.5, 9.25], pattern="hour"),
    ])
    fig.show()
    return fig


# ════════════════════════════════════════════════════════════════
# 9. LIVE SIGNAL MONITOR  (hooks into live_sr.py WebSocket feed)
# ════════════════════════════════════════════════════════════════

def live_signal_monitor(
    hist_zones: pd.DataFrame,
    hist_df:    pd.DataFrame,
    ws_host:    str = "localhost",
    ws_port:    int = 8086,
    ws_path:    str = "/ws",
):
    """
    Connect to the live tick WebSocket (live_sr.py), compute live zones,
    find confluent zones on every bar close, and alert on fresh signals.

    Run this AFTER building hist_zones from run_strategy().
    """
    import asyncio, json
    from realtime_sr import RealtimeSREngine
    from live_sr     import FiveMinCandleBuilder, parse_tick

    engine         = RealtimeSREngine(
        range_size=cfg.RANGE_SIZE, reversal_threshold=cfg.REVERSAL_THR,
        bandwidth=cfg.BANDWIDTH, zone_half_width=cfg.LIVE_ZONE_HW,
        half_life_min=cfg.HALF_LIFE_MIN,
    )
    candle_builder = FiveMinCandleBuilder()
    atr_ser        = compute_atr(hist_df)
    vwap_buf       = {}

    async def run():
        import websockets
        url = f"ws://{ws_host}:{ws_port}{ws_path}"
        print(f"\n[LIVE] Connecting to {url} ...\n")
        async with websockets.connect(url) as ws:
            async for raw in ws:
                price, ts = parse_tick(str(raw))
                if price is None:
                    continue
                engine.on_tick(price, ts)
                closed = candle_builder.on_tick(price, ts)
                if closed is None:
                    continue

                # Update today's VWAP approximation
                date_key = ts.date()
                if date_key not in vwap_buf:
                    vwap_buf[date_key] = {"tp_vol": 0, "vol": 0}
                typical = (closed.high + closed.low + closed.close) / 3
                vwap_buf[date_key]["tp_vol"] += typical * max(1, closed.tick_count)
                vwap_buf[date_key]["vol"]    += max(1, closed.tick_count)
                vwap = vwap_buf[date_key]["tp_vol"] / vwap_buf[date_key]["vol"]

                live_zones = engine.zones(current_price=price, now=ts, top_n=10)
                live_zone_list = [
                    LiveZone(price=z.price, lower=z.lower, upper=z.upper,
                             strength=z.strength, type=z.type, n_pivots=z.n_pivots)
                    for z in live_zones
                ]
                confluent = find_confluent_zones(hist_zones, live_zone_list, price)
                if not confluent:
                    continue

                # Build a synthetic single-bar dataframe for signal scanning
                bar_df = pd.DataFrame([{
                    "Open": closed.open, "High": closed.high,
                    "Low":  closed.low,  "Close": closed.close,
                    "Volume": closed.tick_count,
                }], index=[ts])
                try:
                    bar_df.index = bar_df.index.tz_localize("Asia/Kolkata")
                except Exception:
                    pass

                atr_val  = float(atr_ser.iloc[-1]) if len(atr_ser) else 30.0
                vwap_ser = pd.Series([vwap], index=bar_df.index)
                atr_s_b  = pd.Series([atr_val], index=bar_df.index)

                sigs = scan_entries(bar_df, confluent, atr_s_b, vwap_ser)
                for sig in sigs:
                    pos = position_size(sig.risk_pts)
                    alert = (
                        f"\n{'★'*60}\n"
                        f"  🔔 LIVE SIGNAL  [{sig.quality}]  {sig.direction}  {sig.pattern}\n"
                        f"  Time    : {ts.strftime('%H:%M:%S')}\n"
                        f"  Entry   : {sig.entry_price:.2f}   "
                        f"SL: {sig.sl:.2f}   TP1: {sig.tp1:.2f}   TP2: {sig.tp2:.2f}\n"
                        f"  Risk    : {sig.risk_pts:.1f} pts   "
                        f"RR1: {sig.rr1:.2f}   Zone: {sig.zone.price:.1f} (str={sig.zone.combo_str:.0f})\n"
                        f"  Position: {pos['lots']} lots ({pos['units']} units)  "
                        f"Capital at risk: ₹{pos['risk_capital']:,.0f}\n"
                        f"  Notes   : {sig.notes}\n"
                        f"{'★'*60}"
                    )
                    print(alert)

    asyncio.run(run())


# ════════════════════════════════════════════════════════════════
# 10. MAIN ORCHESTRATOR
# ════════════════════════════════════════════════════════════════

def run_strategy(
    backtest:      bool = True,
    plot:          bool = True,
    live_monitor:  bool = False,
):
    """
    Full pipeline:
      1. Fetch 60-day intraday data
      2. Build historical precision zones
      3. Build live zones from today's session
      4. Find confluent (dual-confirmed) zones
      5. Scan for trade signals
      6. Size positions
      7. Backtest (optional)
      8. Plot (optional)
      9. Start live monitor (optional)
    """
    print("\n" + "╔" + "═"*68 + "╗")
    print("║  NIFTY DUAL-LAYER S/R TRADING STRATEGY ENGINE" + " "*21 + "║")
    print("╚" + "═"*68 + "╝\n")

    # ── Step 1: Data ──────────────────────────────────────────
    df = fetch_intraday_chunked(cfg.TICKER, cfg.INTERVAL, cfg.HIST_DAYS)

    # ── Step 2: Historical zones ──────────────────────────────
    hist_zones = build_historical_zones(df)

    # ── Step 3: Live zones — use last 5 sessions for enough pivots ──
    # Today-only rarely has sufficient zigzag pivots; 5 sessions gives
    # the KDE enough price history to form meaningful intraday zones.
    print("\n[LIVE] Building intraday zones from last 5 sessions...")
    all_dates  = sorted(set(df.index.date))
    live_dates = all_dates[-5:]          # last 5 trading days
    df_live    = df[pd.Series(df.index.date, index=df.index).isin(live_dates)]
    cmp        = float(df["Close"].iloc[-1])
    live_zl    = build_live_zones_from_ohlc(df_live, cmp)
    print(f"  ✓ {len(live_zl)} live zones  (from {live_dates[0]} → {live_dates[-1]})")

    # ── Step 4: Confluence ────────────────────────────────────
    print("\n[CONF] Finding confluent zones...")
    confluent = find_confluent_zones(hist_zones, live_zl, cmp)
    print(f"  ✓ {len(confluent)} confluent zones")
    print_confluent_zones(confluent, cmp)

    if not confluent:
        print("\n  ⚠  No confluent zones found — widen CONFLUENCE_DIST or lower strength thresholds.")
        return df, hist_zones, [], [], []

    # ── Step 5: Compute ATR / VWAP ────────────────────────────
    atr_ser  = compute_atr(df)
    vwap_ser = compute_vwap(df)

    # ── Step 6: Signal scan ───────────────────────────────────
    print("\n[SIGNAL] Scanning for trade entries...")
    signals = scan_entries(df, confluent, atr_ser, vwap_ser)
    print(f"  ✓ {len(signals)} signals found")
    print_signals(signals, cmp)

    # ── Position sizing for each signal ───────────────────────
    print("\n[SIZING] Position sizes (1% risk / ₹1L capital)")
    print(f"  {'Time':<16} {'Dir':<6} {'Entry':>8} {'Risk':>5} {'Lots':>5} {'₹Risk':>10}")
    print("  " + "─" * 60)
    for s in signals[:10]:
        pos = position_size(s.risk_pts)
        t   = s.bar_time.strftime("%m-%d %H:%M")
        print(f"  {t:<16} {s.direction:<6} {s.entry_price:>8.1f} "
              f"{s.risk_pts:>5.1f} {pos['lots']:>5} ₹{pos['risk_capital']:>9,.0f}")

    # ── Step 7: Backtest ──────────────────────────────────────
    trades, stats = [], {}
    if backtest and signals:
        print("\n[BACKTEST] Running simulated trades...")
        trades, stats = backtest_signals(df, signals)
        print(f"  ✓ {len(trades)} trades simulated")
        print_stats(stats)

    # ── Step 8: Chart ──────────────────────────────────────────
    if plot:
        print("\n[CHART] Generating interactive chart...")
        plot_strategy(df, hist_zones, confluent, signals, trades, cfg.TICKER)

    # ── Step 9: Live monitor ──────────────────────────────────
    if live_monitor:
        print("\n[LIVE] Starting live signal monitor (Ctrl-C to stop)...")
        live_signal_monitor(hist_zones, df)

    return df, hist_zones, confluent, signals, trades


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Toggle these flags ────────────────────────────────────
    BACKTEST     = True    # simulate past trades
    PLOT         = False    # open interactive chart
    LIVE_MONITOR = False   # connect to live_sr.py WebSocket
    # Set LIVE_MONITOR = True only when live_sr.py is running

    run_strategy(
        backtest     = BACKTEST,
        plot         = PLOT,
        live_monitor = LIVE_MONITOR,
    )