"""
zone_engine.py
==============
Precision S/R Zone Engine — adapted from the original batch script.

Changes from original:
  • ZoneEngine class wraps all logic so it can be called from both
    - pre-market batch mode (full 60-day history)
    - rolling intraday mode (sliding window of candles)
  • add_candle() / rebuild_zones() allow incremental updates without
    reloading data from disk every time.
  • Zones carry an `invalidated` flag: once price closes clearly through
    a zone (by > half_band pts), that zone is marked stale.
  • All parameters remain identical to original — no behaviour change.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# DATA FETCH  (unchanged from original)
# ════════════════════════════════════════════════════════════════

def fetch_intraday_chunked(
    ticker:     str = "^NSEI",
    interval:   str = "5m",
    days:       int = 60,
    chunk_days: int = 7,
) -> pd.DataFrame:
    """
    Fetch 5-min intraday data in weekly chunks.
    Filters to NSE session: 09:15 – 15:30 IST.
    Used only for pre-market zone map build.
    """
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    chunks, cursor = [], start_dt

    print(f"[ZoneEngine] Fetching {interval} data for {ticker} ({days} days)...")

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
                chunk.columns = [c[0] if isinstance(c, tuple) else c
                                  for c in chunk.columns]
                chunks.append(chunk)
        except Exception as e:
            print(f"  [warn] {cursor.date()} → {chunk_end.date()} : {e}")
        cursor = chunk_end

    if not chunks:
        raise ValueError(f"No data returned for {ticker}.")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    try:
        df.index = df.index.tz_convert("Asia/Kolkata")
    except TypeError:
        try:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        except Exception:
            pass

    df = df.between_time("09:15", "15:30")
    print(f"[ZoneEngine] ✓ {len(df)} candles  "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


# ════════════════════════════════════════════════════════════════
# PIVOT DETECTION  (unchanged)
# ════════════════════════════════════════════════════════════════

def detect_pivots(df: pd.DataFrame, left_bars=10, right_bars=10):
    highs = df["High"].values
    lows  = df["Low"].values

    raw_phi = argrelextrema(highs, np.greater_equal, order=left_bars)[0]
    raw_plo = argrelextrema(lows,  np.less_equal,    order=left_bars)[0]

    def confirm(idx_arr, values, is_high):
        out = []
        for i in idx_arr:
            lw = values[max(0, i - left_bars):i]
            rw = values[i+1:min(i + right_bars + 1, len(values))]
            if len(lw) == 0 or len(rw) == 0:
                continue
            if is_high and values[i] > np.max(lw) and values[i] > np.max(rw):
                out.append(i)
            elif not is_high and values[i] < np.min(lw) and values[i] < np.min(rw):
                out.append(i)
        return np.array(out)

    phi = confirm(raw_phi, highs, True)
    plo = confirm(raw_plo, lows,  False)

    def make_df(idx_arr, col, ptype):
        if len(idx_arr) == 0:
            return pd.DataFrame(columns=["price","date","bar_idx","type","session"])
        return pd.DataFrame({
            "price":   df[col].iloc[idx_arr].values,
            "date":    df.index[idx_arr],
            "bar_idx": idx_arr,
            "type":    ptype,
            "session": [d.date() for d in df.index[idx_arr]],
        })

    return make_df(phi, "High", "high"), make_df(plo, "Low", "low")


# ════════════════════════════════════════════════════════════════
# CLUSTERING  (unchanged)
# ════════════════════════════════════════════════════════════════

def cluster_pivots_by_points(pivots: pd.DataFrame, tolerance=15.0):
    if len(pivots) == 0:
        return []
    sorted_p = pivots.sort_values("price").reset_index(drop=True)
    prices   = sorted_p["price"].values
    clusters, group_start = [], 0
    for i in range(1, len(prices)):
        if prices[i] - prices[group_start] > tolerance:
            clusters.append(_make_cluster(sorted_p.iloc[group_start:i]))
            group_start = i
    clusters.append(_make_cluster(sorted_p.iloc[group_start:]))
    return clusters


def _make_cluster(grp):
    return {
        "price":     grp["price"].mean(),
        "price_min": grp["price"].min(),
        "price_max": grp["price"].max(),
        "spread":    grp["price"].max() - grp["price"].min(),
        "n_pivots":  len(grp),
        "types":     list(grp["type"]),
        "sessions":  list(grp["session"].unique()) if "session" in grp.columns else [],
        "dates":     list(grp["date"]),
    }


# ════════════════════════════════════════════════════════════════
# TOUCH & REJECTION  (unchanged)
# ════════════════════════════════════════════════════════════════

def analyse_level(df, level, half_band=15.0, min_rejection_pts=8.0):
    upper = level + half_band
    lower = level - half_band
    wick_touches = body_rejections = inside_bars = 0
    sessions_touched = set()
    last_touch_dt = None
    session_held = defaultdict(bool)

    for dt, row in df.iterrows():
        hi, lo, cl = row["High"], row["Low"], row["Close"]
        session = dt.date()
        wick_in = (hi >= lower) and (lo <= upper)
        if wick_in:
            wick_touches += 1
            sessions_touched.add(session)
            last_touch_dt = dt
            if cl > upper + min_rejection_pts or cl < lower - min_rejection_pts:
                body_rejections += 1
                session_held[session] = True
            elif (lo < lower - 5 and cl < lower - 5) or (hi > upper + 5 and cl > upper + 5):
                session_held[session] = False
        if hi <= upper and lo >= lower:
            inside_bars += 1

    return {
        "wick_touches":      wick_touches,
        "body_rejections":   body_rejections,
        "inside_bars":       inside_bars,
        "sessions_touched":  len(sessions_touched),
        "consecutive_holds": sum(1 for v in session_held.values() if v),
        "last_touch":        last_touch_dt,
    }


# ════════════════════════════════════════════════════════════════
# ZONE BUILDER + INVALIDATION  (new wrapper logic)
# ════════════════════════════════════════════════════════════════

def _remove_overlaps(zones_df, half_band):
    df = zones_df.sort_values("strength", ascending=False).reset_index(drop=True)
    keep = [True] * len(df)
    for i in range(len(df)):
        if not keep[i]: continue
        for j in range(i + 1, len(df)):
            if not keep[j]: continue
            if abs(df.loc[i, "price"] - df.loc[j, "price"]) < half_band * 2:
                keep[j] = False
    return df[keep].reset_index(drop=True)


def build_zones_from_df(
    df,
    left_bars=10, right_bars=10,
    cluster_tolerance=15.0, zone_half_band=15.0,
    min_wick_touches=3, min_sessions=2, min_rejections=1,
    top_n=20,
    current_price=None,
) -> pd.DataFrame:
    """
    Core zone-building function — works on any DataFrame snapshot.
    Called by ZoneEngine for both batch and rolling modes.
    """
    ph, pl = detect_pivots(df, left_bars, right_bars)
    all_pivots = pd.concat([ph, pl], ignore_index=True)
    if len(all_pivots) == 0:
        return pd.DataFrame()

    clusters  = cluster_pivots_by_points(all_pivots, tolerance=cluster_tolerance)
    if current_price is None:
        current_price = float(df["Close"].iloc[-1])
    latest_date = df.index[-1]
    candidates  = []

    for cl in clusters:
        level = cl["price"]
        info  = analyse_level(df, level, half_band=zone_half_band)

        if info["wick_touches"]    < min_wick_touches: continue
        if info["sessions_touched"] < min_sessions:    continue
        if info["body_rejections"] < min_rejections:   continue

        days_ago = (latest_date - info["last_touch"]).days if info["last_touch"] else 999
        recency  = np.exp(-days_ago / 15)

        touch_score     = min(info["wick_touches"]    / 15, 1.0)
        rejection_score = min(info["body_rejections"] / max(info["wick_touches"], 1), 1.0)
        session_score   = min(info["sessions_touched"] / 10, 1.0)
        hold_score      = min(info["consecutive_holds"] / 5,  1.0)
        n_h = cl["types"].count("high")
        n_l = cl["types"].count("low")
        convergence  = 1.0 if (n_h > 0 and n_l > 0) else 0.5
        conv_label   = "Both" if convergence == 1.0 else ("High" if n_h >= n_l else "Low")
        spread_score = max(0.0, 1.0 - cl["spread"] / cluster_tolerance)

        strength = (
            0.28 * touch_score     +
            0.22 * rejection_score +
            0.18 * recency         +
            0.12 * session_score   +
            0.08 * hold_score      +
            0.07 * convergence     +
            0.05 * spread_score
        ) * 100

        zone_type = "Support" if level < current_price else "Resistance"

        candidates.append({
            "price":           round(level, 2),
            "upper":           round(level + zone_half_band, 2),
            "lower":           round(level - zone_half_band, 2),
            "band_pts":        int(zone_half_band * 2),
            "type":            zone_type,
            "strength":        round(strength, 1),
            "wick_touches":    info["wick_touches"],
            "body_rejections": info["body_rejections"],
            "inside_bars":     info["inside_bars"],
            "sessions":        info["sessions_touched"],
            "holds":           info["consecutive_holds"],
            "recency_score":   round(recency * 100, 1),
            "last_touch":      info["last_touch"],
            "n_pivots":        cl["n_pivots"],
            "spread_pts":      round(cl["spread"], 1),
            "convergence":     conv_label,
            "dist_pts":        round(level - current_price, 1),
            "invalidated":     False,   # NEW — tracks if zone has been broken
        })

    if not candidates:
        return pd.DataFrame()

    zones_df = pd.DataFrame(candidates).sort_values("strength", ascending=False)
    zones_df = _remove_overlaps(zones_df, zone_half_band)
    return zones_df.head(top_n).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# ZONE ENGINE CLASS  — the main interface used by the strategy
# ════════════════════════════════════════════════════════════════

class ZoneEngine:
    """
    Wraps zone building for both pre-market batch and live rolling modes.

    Usage
    -----
    # Pre-market (once at 9:00 AM)
    engine = ZoneEngine(ticker="^NSEI")
    engine.build_from_history(days=60)

    # Live (called after every confirmed 5m candle)
    engine.add_candle(new_row)          # add new OHLCV row
    engine.invalidate_broken_zones(ltp) # demote zones price has crossed
    zones = engine.get_active_zones()   # DataFrame of valid zones
    """

    def __init__(
        self,
        ticker:            str   = "^NSEI",
        left_bars:         int   = 10,
        right_bars:        int   = 10,
        cluster_tolerance: float = 15.0,
        zone_half_band:    float = 15.0,
        min_wick_touches:  int   = 3,
        min_sessions:      int   = 2,
        min_rejections:    int   = 1,
        top_n:             int   = 20,
        rolling_window:    int   = 500,  # candles kept in rolling buffer
        rebuild_every:     int   = 12,   # rebuild zones every N new candles
    ):
        self.ticker            = ticker
        self.left_bars         = left_bars
        self.right_bars        = right_bars
        self.cluster_tolerance = cluster_tolerance
        self.zone_half_band    = zone_half_band
        self.min_wick_touches  = min_wick_touches
        self.min_sessions      = min_sessions
        self.min_rejections    = min_rejections
        self.top_n             = top_n
        self.rolling_window    = rolling_window
        self.rebuild_every     = rebuild_every

        self._df: pd.DataFrame        = pd.DataFrame()
        self._zones: pd.DataFrame     = pd.DataFrame()
        self._candles_since_rebuild   = 0

    # ── Public: batch build ─────────────────────────────────

    def build_from_history(self, days=60, df: pd.DataFrame = None):
        """Load full history and build initial zone map."""
        if df is not None:
            self._df = df.copy()
        else:
            self._df = fetch_intraday_chunked(
                self.ticker, interval="5m", days=days
            )
        self._rebuild_zones()
        print(f"[ZoneEngine] ✓ Initial zone map: {len(self._zones)} zones")
        return self._zones.copy()

    # ── Public: live update ─────────────────────────────────

    def add_candle(self, row: pd.Series):
        """
        Append one confirmed 5m candle to the buffer.
        Rebuilds zones every `rebuild_every` candles.
        """
        self._df = pd.concat([self._df, row.to_frame().T])
        # Keep rolling window
        if len(self._df) > self.rolling_window:
            self._df = self._df.iloc[-self.rolling_window:]

        self._candles_since_rebuild += 1
        if self._candles_since_rebuild >= self.rebuild_every:
            self._rebuild_zones()
            self._candles_since_rebuild = 0

    def invalidate_broken_zones(self, current_price: float, margin: float = None):
        """
        Mark zones as invalidated when price has clearly crossed them.

        A resistance zone is invalidated when price closes above upper + margin.
        A support zone is invalidated when price closes below lower - margin.

        margin defaults to zone_half_band (i.e., full band clearance required).
        """
        if len(self._zones) == 0:
            return
        if margin is None:
            margin = self.zone_half_band

        for idx in self._zones.index:
            z = self._zones.loc[idx]
            if z["invalidated"]:
                continue
            if z["type"] == "Resistance" and current_price > z["upper"] + margin:
                self._zones.loc[idx, "invalidated"] = True
                self._zones.loc[idx, "type"]        = "Broken_R"  # was resistance, now possible support
            elif z["type"] == "Support" and current_price < z["lower"] - margin:
                self._zones.loc[idx, "invalidated"] = True
                self._zones.loc[idx, "type"]        = "Broken_S"

    def get_active_zones(self) -> pd.DataFrame:
        """Return only non-invalidated zones."""
        if len(self._zones) == 0:
            return pd.DataFrame()
        return self._zones[~self._zones["invalidated"]].copy()

    def get_nearest_zones(self, current_price: float, n=3):
        """
        Return n nearest support zones (below price) and
        n nearest resistance zones (above price).

        Also includes zones within 50 pts of price on either side
        so breakout candidates aren't missed when price is inside a zone.
        """
        active = self.get_active_zones()
        if len(active) == 0:
            return pd.DataFrame(), pd.DataFrame()

        active = active.copy()
        active["dist_abs"] = (active["price"] - current_price).abs()

        # Support: price > zone (zone is below) OR price is inside zone
        sup = active[active["price"] <= current_price + 5].copy()
        # Resistance: price < zone (zone is above) OR price is inside zone
        res = active[active["price"] >= current_price - 5].copy()

        sup = sup.sort_values("dist_abs").head(n)
        res = res.sort_values("dist_abs").head(n)
        return sup, res

    @property
    def df(self):
        return self._df.copy()

    @property
    def zones(self):
        return self._zones.copy()

    # ── Private ─────────────────────────────────────────────

    def _rebuild_zones(self):
        cp = float(self._df["Close"].iloc[-1])
        new_zones = build_zones_from_df(
            self._df,
            left_bars         = self.left_bars,
            right_bars        = self.right_bars,
            cluster_tolerance = self.cluster_tolerance,
            zone_half_band    = self.zone_half_band,
            min_wick_touches  = self.min_wick_touches,
            min_sessions      = self.min_sessions,
            min_rejections    = self.min_rejections,
            top_n             = self.top_n,
            current_price     = cp,
        )
        if len(new_zones) > 0:
            # Preserve invalidation state for zones that already existed
            if len(self._zones) > 0:
                invalidated_prices = set(
                    self._zones.loc[self._zones["invalidated"], "price"].round(0)
                )
                for idx in new_zones.index:
                    if round(new_zones.loc[idx, "price"], 0) in invalidated_prices:
                        new_zones.loc[idx, "invalidated"] = True
            self._zones = new_zones