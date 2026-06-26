"""
SPEED DEMON SCALPER v5 — EMA Momentum × S/R Conviction
=======================================================
Base: Speed Demon v4 (EMA9/21/50 pullaway + swing structure trail)
Layer: Precision S/R Zone engine grafted on top as a 5-gate filter

HOW THE TWO STRATEGIES COMBINE
────────────────────────────────────────────────────────────────────
The EMA logic identifies WHEN to enter (momentum + pullaway).
The S/R layer decides WHETHER to enter (is the trade well-positioned?).

Five S/R Gates (all must pass for an entry to fire):

  Gate 1 — ZONE PROXIMITY BLOCK
    Entry price must NOT be inside or within `sr_entry_buffer_pts`
    of any S/R zone. Prevents entering inside a zone — the worst
    possible timing.

  Gate 2 — MINIMUM FREE SPACE (room-to-run filter)
    For LONG:  nearest RESISTANCE above entry must be ≥ sr_min_space_pts away.
    For SHORT: nearest SUPPORT   below entry must be ≥ sr_min_space_pts away.
    Kills trades with a ceiling/floor right in the way.

  Gate 3 — S/R ZONE ALIGNMENT (conviction boost)
    LONG  entries score +1 if entry is between or just above a SUPPORT zone.
    SHORT entries score +1 if entry is between or just below a RESISTANCE zone.
    Low-conviction entries (score = 0) are optionally skipped.

  Gate 4 — S/R-ALIGNED TAKE PROFIT
    TP is set at the near edge (lower for LONG, upper for SHORT) of the
    nearest OPPOSING zone — not a fixed ATR multiple.
    If the TP would give R:R < sr_rr_min, the trade is skipped.

  Gate 5 — S/R EARLY EXIT (profit protection)
    While in a trade, if the trailing price reaches within
    `sr_exit_buffer_pts` of the opposing S/R zone, the position
    is closed immediately rather than waiting for a swing SL hit.
    Saves trades like #10 (MFE=100 → PnL=3) where price stalled at a zone.

Run:
    python speed_demon_sr_v5.py              # yfinance live data
    python speed_demon_sr_v5.py data.csv     # CSV file
"""

import pandas as pd
import numpy as np
import sys, os, json, webbrowser, warnings
from datetime import time
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# 1.  ALL CONFIGURATION
# ════════════════════════════════════════════════════════════════

# ── Data ─────────────────────────────────────────────────────────
DATA_MODE        = 'yfinance'
YFINANCE_SYMBOL  = '^NSEI'
YFINANCE_DAYS    = 60
YFINANCE_TZ      = 'Asia/Kolkata'
CSV_5MIN_PATH    = 'nifty_5m.csv'

# ── EMA Strategy (v4 base — unchanged) ────────────────────────────
EMA_FAST         = 9
EMA_SLOW         = 21
EMA_MACRO        = 50
HTF_EMA_FAST     = 12
HTF_EMA_SLOW     = 26
HTF_EMA_TREND    = 50
ATR_PERIOD       = 14
ADX_PERIOD       = 14
ADX_MIN          = 18

LONG_ATR_SL_MULT    = 1.2
SHORT_ATR_SL_MULT   = 0.75
LONG_BE_PCT         = 0.2
SHORT_BE_PCT        = 0.3

SWING_LOOKBACK      = 5
SWING_BUFFER_PTS    = 5
TRAIL_ATR_MULT      = 0.6   # kept for reference, not used (swing trail active)

ENABLE_EMA_CROSS_EXIT = True
ENABLE_SLOPE_EXIT     = True
SLOPE_EXIT_CANDLES    = 4
SHORT_CONFIRM_BARS    = 3
SPREAD_PCT_MIN        = 0.04
SLOPE_CANDLES         = 6
SLOPE_MIN             = 0.0
PRICE_GAP_MIN         = 5.0
MIDDAY_SPREAD_MULT    = 2.0
RETEST_ATR_MULT       = 0.25
EMA9_PULLAWAY_PTS     = 30

MAX_TRADES_PER_DAY    = 3
MAX_CONSEC_LOSSES     = 2    # kept for reference; circuit breaker is 1 loss/day (v4 logic)

ENABLE_LONG           = True
ENABLE_SHORT          = True
ENABLE_PATH_A         = True
ENABLE_PATH_B         = True
ENABLE_MIDDAY         = True
ENABLE_EURO           = False

OBSERVE_START         = time(9,  15)
OBSERVE_END           = time(9,  30)
PRIME_START           = time(9,  30)
PRIME_END             = time(10, 30)
MIDDAY_START          = time(11, 30)
MIDDAY_END            = time(13, 30)
EURO_START            = time(14, 15)
EURO_END              = time(15,  0)
SQUAREOFF_START       = time(15,  0)
EOD_HARD_EXIT         = time(15,  0)

# ── S/R Layer (v5 additions) ──────────────────────────────────────

# Zone detection params (match your precision zone script)
SR_LEFT_BARS          = 10
SR_RIGHT_BARS         = 10
SR_CLUSTER_TOLERANCE  = 15.0   # absolute points
SR_ZONE_HALF_BAND     = 15.0   # zone = level ± 15 pts → 30-pt total band
SR_MIN_WICK_TOUCHES   = 3
SR_MIN_SESSIONS       = 2
SR_MIN_REJECTIONS     = 1
SR_TOP_N              = 20

# Gate 1 — entry must NOT be inside or this close to any zone
SR_ENTRY_BUFFER_PTS   = 7.0   # pts beyond zone edge before entry is allowed

# Gate 2 — minimum clear space to the opposing zone
SR_MIN_SPACE_PTS      = 40.0   # LONG: resistance must be ≥60 pts above entry
                                # SHORT: support must be ≥60 pts below entry

# Gate 3 — alignment: entry must be within this many pts of a supportive zone
SR_ALIGN_MAX_DIST     = 40.0   # if nearest same-side zone is farther, no boost
SR_REQUIRE_ALIGNMENT  = False  # True = only trade when entry is near a supportive zone

# Gate 4 — minimum R:R using S/R-derived TP (skip trade if too low)
SR_RR_MIN             = 1.3
# Gate 4 fallback — when price has broken BEYOND all opposing zones (clean
# runway, no ceiling/floor), don't block the trade. Use an ATR-multiple TP.
SR_FALLBACK_RR        = 3.0   # TP = entry ± (atr_sl_dist * this) when no opp zone

# Gate 5 — early exit: how close to opposing zone before we close the trade
SR_EXIT_BUFFER_PTS    = 12.0   # exit when price is within 12 pts of opposing zone edge

# Optional: only use zones with strength >= this (0 = use all zones)
SR_MIN_ZONE_STRENGTH  = 0.0


# ════════════════════════════════════════════════════════════════
# 2.  S/R ZONE ENGINE (inline — no external import needed)
# ════════════════════════════════════════════════════════════════

from scipy.signal import argrelextrema
from collections import defaultdict

def _compute_ema_sr(series, period):
    return series.ewm(span=period, adjust=False).mean()

def _detect_pivots(df, left_bars=10, right_bars=10):
    highs = df["High"].values
    lows  = df["Low"].values
    raw_phi = argrelextrema(highs, np.greater_equal, order=left_bars)[0]
    raw_plo = argrelextrema(lows,  np.less_equal,    order=left_bars)[0]

    def confirm(idx_arr, values, is_high):
        confirmed = []
        for i in idx_arr:
            lw = values[max(0, i-left_bars):i]
            rw = values[i+1:min(i+right_bars+1, len(values))]
            if not len(lw) or not len(rw): continue
            if is_high and values[i] > np.max(lw) and values[i] > np.max(rw):
                confirmed.append(i)
            elif not is_high and values[i] < np.min(lw) and values[i] < np.min(rw):
                confirmed.append(i)
        return np.array(confirmed)

    phi = confirm(raw_phi, highs, True)
    plo = confirm(raw_plo, lows,  False)

    def make_pdf(idx_arr, col, ptype):
        if not len(idx_arr):
            return pd.DataFrame(columns=["price","date","type","session"])
        return pd.DataFrame({
            "price":   df[col].iloc[idx_arr].values,
            "date":    df.index[idx_arr],
            "type":    ptype,
            "session": [d.date() for d in df.index[idx_arr]],
        })
    return make_pdf(phi, "High", "high"), make_pdf(plo, "Low", "low")

def _cluster_by_points(pivots, tolerance=15.0):
    if not len(pivots): return []
    sp = pivots.sort_values("price").reset_index(drop=True)
    px = sp["price"].values
    clusters, gs = [], 0
    for i in range(1, len(px)):
        if px[i] - px[gs] > tolerance:
            grp = sp.iloc[gs:i]
            clusters.append({"price": grp["price"].mean(),
                             "types": list(grp["type"]),
                             "sessions": list(grp["session"].unique()),
                             "spread": grp["price"].max()-grp["price"].min(),
                             "n": len(grp)})
            gs = i
    grp = sp.iloc[gs:]
    clusters.append({"price": grp["price"].mean(), "types": list(grp["type"]),
                     "sessions": list(grp["session"].unique()),
                     "spread": grp["price"].max()-grp["price"].min(), "n": len(grp)})
    return clusters

def _analyse_level(df, level, half_band=15.0, min_rej=8.0):
    upper, lower = level+half_band, level-half_band
    wt, br, ib = 0, 0, 0
    sessions, last_touch = set(), None
    session_held = defaultdict(bool)
    for dt, row in df.iterrows():
        hi, lo, cl = row["High"], row["Low"], row["Close"]
        sess = dt.date()
        wick_in = (hi >= lower) and (lo <= upper)
        if wick_in:
            wt += 1; sessions.add(sess); last_touch = dt
            if cl > upper+min_rej or cl < lower-min_rej:
                br += 1; session_held[sess] = True
            elif (lo<lower-5 and cl<lower-5) or (hi>upper+5 and cl>upper+5):
                session_held[sess] = False
        if hi <= upper and lo >= lower: ib += 1
    return {"wt": wt, "br": br, "ib": ib,
            "sess": len(sessions),
            "holds": sum(1 for v in session_held.values() if v),
            "last_touch": last_touch}

def build_sr_zones(df_5m, half_band=15.0, cluster_tol=15.0,
                   min_wt=3, min_sess=2, min_br=1, top_n=20,
                   left_bars=10, right_bars=10):
    """Build S/R zones from 5m data. Returns list of zone dicts."""
    # yfinance-style columns → capitalized for zone detector
    col_map = {"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}
    df = df_5m.rename(columns={v:v for v in df_5m.columns})
    # Handle both capitalised and lower-case columns
    if "close" in df.columns:
        df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    ph, pl = _detect_pivots(df, left_bars, right_bars)
    all_p = pd.concat([ph, pl], ignore_index=True)
    if not len(all_p): return []
    clusters = _cluster_by_points(all_p, cluster_tol)
    current_price = float(df["Close"].iloc[-1])
    latest_date   = df.index[-1]
    zones = []
    for cl in clusters:
        lv = cl["price"]
        info = _analyse_level(df, lv, half_band)
        if info["wt"] < min_wt or info["sess"] < min_sess or info["br"] < min_br: continue
        days_ago = (latest_date - info["last_touch"]).days if info["last_touch"] else 999
        recency  = np.exp(-days_ago / 15)
        touch_s  = min(info["wt"] / 15, 1.0)
        rej_s    = min(info["br"] / max(info["wt"], 1), 1.0)
        sess_s   = min(info["sess"] / 10, 1.0)
        hold_s   = min(info["holds"] / 5, 1.0)
        n_h = cl["types"].count("high"); n_l = cl["types"].count("low")
        conv = 1.0 if (n_h > 0 and n_l > 0) else 0.5
        spr_s = max(0.0, 1.0 - cl["spread"] / cluster_tol)
        strength = (0.28*touch_s + 0.22*rej_s + 0.18*recency +
                    0.12*sess_s + 0.08*hold_s + 0.07*conv + 0.05*spr_s) * 100
        zones.append({
            "price":    round(lv, 2),
            "upper":    round(lv + half_band, 2),
            "lower":    round(lv - half_band, 2),
            "type":     "Support" if lv < current_price else "Resistance",
            "strength": round(strength, 1),
            "wt":       info["wt"],
            "br":       info["br"],
            "sess":     info["sess"],
        })
    zones.sort(key=lambda x: x["strength"], reverse=True)
    # Remove overlaps
    keep = [True] * len(zones)
    for i in range(len(zones)):
        if not keep[i]: continue
        for j in range(i+1, len(zones)):
            if not keep[j]: continue
            if abs(zones[i]["price"] - zones[j]["price"]) < half_band*2:
                keep[j] = False
    zones = [z for k, z in zip(keep, zones) if k][:top_n]
    return zones


# ════════════════════════════════════════════════════════════════
# 3.  S/R GATE FUNCTIONS  (the 5 gates)
# ════════════════════════════════════════════════════════════════

class SRContext:
    """Runtime S/R lookup context. Built once per backtest bar."""
    def __init__(self, zones, min_strength=0.0):
        self.zones = [z for z in zones if z["strength"] >= min_strength]

    def price_in_zone(self, price, buffer=0.0):
        """Gate 1: Is `price` inside or within `buffer` pts of any zone?"""
        for z in self.zones:
            if z["lower"] - buffer <= price <= z["upper"] + buffer:
                return True, z
        return False, None

    def nearest_opposing(self, price, direction):
        """
        Gate 2 & 4: Nearest zone in the trade's path.
        direction='up'  → nearest zone ABOVE price (for LONG trades)
        direction='down'→ nearest zone BELOW price (for SHORT trades)
        Returns (zone, distance_pts) or (None, inf)
        """
        candidates = []
        for z in self.zones:
            if direction == "up"   and z["price"] > price:
                candidates.append((z, z["price"] - price))
            elif direction == "down" and z["price"] < price:
                candidates.append((z, price - z["price"]))
        if not candidates: return None, float("inf")
        return min(candidates, key=lambda x: x[1])

    def nearest_supportive(self, price, direction):
        """
        Gate 3: Nearest zone that SUPPORTS the trade direction.
        LONG  → nearest zone BELOW price (should be a support = floor under us)
        SHORT → nearest zone ABOVE price (should be a resistance = ceiling above us)
        """
        candidates = []
        for z in self.zones:
            if direction == "up"   and z["price"] < price:
                candidates.append((z, price - z["price"]))
            elif direction == "down" and z["price"] > price:
                candidates.append((z, z["price"] - price))
        if not candidates: return None, float("inf")
        return min(candidates, key=lambda x: x[1])

    def sr_tp(self, price, direction):
        """
        Gate 4: Compute TP from the near-edge of the first opposing zone.
        direction='up'  → TP = lower edge of nearest zone above
        direction='down'→ TP = upper edge of nearest zone below
        Returns TP price or None.
        """
        opp, dist = self.nearest_opposing(price, direction)
        if opp is None: return None
        return opp["lower"] if direction == "up" else opp["upper"]

    def approaching_zone(self, price, direction, buffer):
        """
        Gate 5: Is `price` within `buffer` pts of the nearest opposing zone?
        direction='up' = LONG trade, check if resistance is near
        """
        tp_price = self.sr_tp(price, direction)
        if tp_price is None: return False
        dist = abs(price - tp_price)
        return dist <= buffer


def sr_gate_check(price, direction, sr_ctx, atr_sl_dist):
    """
    Run Gates 1-4. Returns (allowed, tp_price, rr, gate_failed, notes)
    direction = 'up' for LONG, 'down' for SHORT
    """
    notes = []

    # Gate 1: Not inside or touching any zone
    in_zone, which_zone = sr_ctx.price_in_zone(price, SR_ENTRY_BUFFER_PTS)
    if in_zone:
        return False, None, 0.0, "G1_IN_ZONE", f"In/near zone @ {which_zone['price']:.0f}"

    # Gate 2: Minimum free space to opposing zone
    opp_zone, opp_dist = sr_ctx.nearest_opposing(price, direction)
    if opp_dist < SR_MIN_SPACE_PTS:
        return False, None, 0.0, "G2_NO_SPACE", f"Opposing zone @ {opp_zone['price']:.0f} only {opp_dist:.0f}pts away"

    # Gate 3: Alignment check (optional)
    sup_zone, sup_dist = sr_ctx.nearest_supportive(price, direction)
    aligned = sup_zone is not None and sup_dist <= SR_ALIGN_MAX_DIST
    if SR_REQUIRE_ALIGNMENT and not aligned:
        return False, None, 0.0, "G3_NO_ALIGN", "No supportive zone nearby"

    # Gate 4: R:R check with S/R-derived TP
    tp = sr_ctx.sr_tp(price, direction)
    tp_is_fallback = False
    if tp is None:
        # No opposing zone = clean breakout runway. Don't block — use ATR target.
        runway = atr_sl_dist * SR_FALLBACK_RR
        tp = price + runway if direction == "up" else price - runway
        tp_is_fallback = True
    reward = abs(tp - price)
    risk   = atr_sl_dist
    rr     = reward / risk if risk > 0 else 0.0
    if rr < SR_RR_MIN:
        return False, tp, rr, "G4_LOW_RR", f"R:R={rr:.2f} < {SR_RR_MIN}"

    opp_str  = f"{opp_zone['strength']:.0f}" if opp_zone else "?"
    sup_info = f"SUP@{sup_zone['price']:.0f}(str={sup_zone['strength']:.0f})" if (aligned and sup_zone) else "no_align"
    tp_tag   = "ATR_RUNWAY" if tp_is_fallback else f"{opp_dist:.0f}pts"
    notes.append(f"TP={tp:.0f} RR={rr:.2f} space={tp_tag} opp_str={opp_str} {sup_info}")
    return True, tp, rr, None, " | ".join(notes)


# ════════════════════════════════════════════════════════════════
# 4.  INDICATOR ENGINE  (v4 — unchanged)
# ════════════════════════════════════════════════════════════════

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_atr(df, period):
    h = df['high'].astype(float); l = df['low'].astype(float); c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_adx(df, period):
    h = df['high'].astype(float); l = df['low'].astype(float)
    up = h.diff(); dn = -(l.diff())
    pdm = np.where((up>dn)&(up>0), up, 0.0)
    mdm = np.where((dn>up)&(dn>0), dn, 0.0)
    atr = compute_atr(df, period)
    pds = pd.Series(pdm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    mds = pd.Series(mdm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    pdi = 100*pds/atr.replace(0, np.nan)
    mdi = 100*mds/atr.replace(0, np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def compute_indicators_5m(df):
    df = df.copy()
    df['ema_fast']            = compute_ema(df['close'], EMA_FAST)
    df['ema_slow']            = compute_ema(df['close'], EMA_SLOW)
    df['ema_macro']           = compute_ema(df['close'], EMA_MACRO)
    df['atr']                 = compute_atr(df, ATR_PERIOD)
    df['adx']                 = compute_adx(df, ADX_PERIOD)
    df['ema_slow_slope']      = df['ema_slow'].diff(SLOPE_CANDLES)
    df['ema_slow_slope_exit'] = df['ema_slow'].diff(SLOPE_EXIT_CANDLES)
    bc = (df['ema_fast'] < df['ema_slow']).astype(int)
    df['consec_bearish_bars'] = bc.groupby((bc!=bc.shift()).cumsum()).cumcount()+1
    df['consec_bearish_bars'] *= bc
    return df

def compute_htf_bias(df_5m):
    df5 = df_5m.copy().set_index('timestamp')
    df15 = df5['close'].resample('15min').ohlc().dropna()
    df15.columns = ['open','high','low','close']
    df15['hf'] = compute_ema(df15['close'], HTF_EMA_FAST)
    df15['hs'] = compute_ema(df15['close'], HTF_EMA_SLOW)
    df15['ht'] = compute_ema(df15['close'], HTF_EMA_TREND)
    df15['htf_long_bias']  = (df15['hf']>df15['hs']) & (df15['hs']>df15['ht'])
    df15['htf_short_bias'] = (df15['hf']<df15['hs']) & (df15['hs']<df15['ht'])
    htf = df15[['hf','hs','ht','htf_long_bias','htf_short_bias']]
    return df5.join(htf.reindex(df5.index, method='ffill')).reset_index()


# ════════════════════════════════════════════════════════════════
# 5.  SWING HELPERS  (v4 — unchanged)
# ════════════════════════════════════════════════════════════════

def get_swing_low(lows, idx, lookback):
    w = lows[max(0, idx-lookback):idx]
    return float(np.min(w)) if len(w) else float(lows[idx])

def get_swing_high(highs, idx, lookback):
    w = highs[max(0, idx-lookback):idx]
    return float(np.max(w)) if len(w) else float(highs[idx])


# ════════════════════════════════════════════════════════════════
# 6.  FILTERS  (v4 — unchanged)
# ════════════════════════════════════════════════════════════════

def get_session(t):
    if OBSERVE_START <= t < OBSERVE_END: return 'observe'
    if PRIME_START   <= t < PRIME_END:   return 'prime'
    if MIDDAY_START  <= t < MIDDAY_END:  return 'midday'
    if EURO_START    <= t < EURO_END:    return 'euro'
    if SQUAREOFF_START <= t:             return 'squareoff'
    return 'outside'

def chop_filters_pass(close, ef, es, slope, adx_v, session):
    if adx_v < ADX_MIN: return False
    spread = abs(ef-es)/close*100
    thresh = SPREAD_PCT_MIN * (MIDDAY_SPREAD_MULT if session=='midday' else 1.0)
    if spread < thresh: return False
    if abs(slope) <= SLOPE_MIN: return False
    if abs(close-es) < PRICE_GAP_MIN: return False
    return True


# ════════════════════════════════════════════════════════════════
# 7.  DATA LOADERS  (v4 — unchanged)
# ════════════════════════════════════════════════════════════════

def _standardise(df, tz):
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(tz)
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert(tz)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[(df['timestamp'].dt.time >= time(9,15)) &
            (df['timestamp'].dt.time <= time(15,30))].reset_index(drop=True)
    df[['open','high','low','close']] = df[['open','high','low','close']].ffill()
    return df

def fetch_yfinance(interval, label):
    import yfinance as yf
    print(f"  Fetching {YFINANCE_SYMBOL} {interval} (last {YFINANCE_DAYS} days) ...")
    raw = yf.download(tickers=YFINANCE_SYMBOL, period=f'{YFINANCE_DAYS}d',
                      interval=interval, progress=False, auto_adjust=True)
    if raw.empty: print("ERROR: No data."); sys.exit(1)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={'adj close':'close','adj_close':'close'})
    if 'volume' not in raw.columns: raw['volume'] = 0
    raw = raw[['open','high','low','close','volume']].dropna()
    if raw.index.tz is None: raw.index = raw.index.tz_localize('UTC')
    raw.index = raw.index.tz_convert(YFINANCE_TZ)
    df = raw.reset_index()
    ts_col = next(c for c in df.columns if c.lower() in ('datetime','date','timestamp'))
    df = df.rename(columns={ts_col:'timestamp'})
    df = _standardise(df, YFINANCE_TZ)
    print(f"  [{label}] Rows: {len(df)} | {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df

def load_csv(filepath, label=''):
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
    ts_col = next((c for c in ['timestamp','datetime','date_time','time','date'] if c in df.columns), None)
    if ts_col is None: raise ValueError(f"No timestamp column. Got: {list(df.columns)}")
    df['timestamp'] = pd.to_datetime(df[ts_col], dayfirst=True)
    if ts_col != 'timestamp': df = df.drop(columns=[ts_col])
    for col in ['open','high','low','close']:
        if col not in df.columns: raise ValueError(f"Missing column '{col}'")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open','high','low','close'])
    df = _standardise(df, 'Asia/Kolkata')
    print(f"  [{label}] Rows: {len(df)} | {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ════════════════════════════════════════════════════════════════
# 8.  BACKTEST ENGINE  (v5 — S/R layer injected)
# ════════════════════════════════════════════════════════════════

def run_backtest(df, sr_zones):
    closes          = df['close'].astype(float).values
    highs           = df['high'].astype(float).values
    lows            = df['low'].astype(float).values
    ema_fast        = df['ema_fast'].values
    ema_slow        = df['ema_slow'].values
    ema_macro       = df['ema_macro'].values
    atrs            = df['atr'].values
    adx_vals        = df['adx'].values
    slopes          = df['ema_slow_slope'].values
    slopes_exit     = df['ema_slow_slope_exit'].values
    consec_bearish  = df['consec_bearish_bars'].values
    htf_long_bias   = df['htf_long_bias'].values
    ts_list         = df['timestamp'].tolist()
    n               = len(df)

    sr_ctx          = SRContext(sr_zones, SR_MIN_ZONE_STRENGTH)

    trades          = []
    equity          = 0.0
    eq_curve        = []

    opening_high    = {}
    opening_close   = {}

    in_trade        = False
    direction       = None
    entry_price     = 0.0
    entry_time      = None
    entry_idx       = -1
    entry_path      = ''
    sr_gate_notes   = ''
    sr_tp_price     = None     # S/R-derived TP for this trade
    stop_loss       = 0.0
    be_triggered    = False
    be_level        = 0.0
    trail_active    = False
    sl_dist_initial = 0.0
    trade_max_favor = 0.0
    trade_max_adv   = 0.0
    gate_failed_counts = {}   # track how many entries each gate blocked

    prev_date       = None
    daily_trades    = {}
    daily_consec_loss = {}

    def do_enter(dir_str, close_price, ts_now, atr_val, idx, path, gate_notes, tp_price):
        nonlocal in_trade, direction, entry_price, entry_time, entry_idx, entry_path
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, trade_max_favor, trade_max_adv
        nonlocal sr_gate_notes, sr_tp_price

        sl_mult        = LONG_ATR_SL_MULT if dir_str=='long' else SHORT_ATR_SL_MULT
        sl_dist        = atr_val * sl_mult

        in_trade       = True
        direction      = dir_str
        entry_price    = close_price
        entry_time     = ts_now
        entry_idx      = idx
        entry_path     = path
        sr_gate_notes  = gate_notes
        sr_tp_price    = tp_price
        be_triggered   = False
        trail_active   = False
        sl_dist_initial= sl_dist
        trade_max_favor= 0.0
        trade_max_adv  = 0.0

        if direction == 'long':
            stop_loss = entry_price - sl_dist
            be_level  = entry_price * (1 + LONG_BE_PCT/100)
        else:
            stop_loss = entry_price + sl_dist
            be_level  = entry_price * (1 - SHORT_BE_PCT/100)

    def do_exit(exit_price, ts_now, reason, exit_idx):
        nonlocal in_trade, direction, entry_price, entry_time, entry_idx, entry_path
        nonlocal stop_loss, be_triggered, be_level, trail_active
        nonlocal sl_dist_initial, equity
        nonlocal trade_max_favor, trade_max_adv
        nonlocal sr_tp_price, sr_gate_notes 

        pnl = round(
            (exit_price-entry_price) if direction=='long'
            else (entry_price-exit_price), 2
        )
        equity += pnl

        date_str = str(ts_now.date())
        if pnl <= 0:
            daily_consec_loss[date_str] = daily_consec_loss.get(date_str, 0) + 1

        trades.append({
            'direction':    direction,
            'entry_path':   entry_path,
            'entry_time':   entry_time,
            'exit_time':    ts_now,
            'entry_price':  entry_price,
            'exit_price':   exit_price,
            'sl_at_entry':  (entry_price - sl_dist_initial if direction=='long'
                             else entry_price + sl_dist_initial),
            'sl_at_exit':   stop_loss,
            'sr_tp':        sr_tp_price,
            'be_triggered': be_triggered,
            'pnl':          pnl,
            'mfe_pts':      round(trade_max_favor, 2),
            'mae_pts':      round(trade_max_adv,   2),
            'exit_reason':  reason,
            'entry_idx':    entry_idx,
            'exit_idx':     exit_idx,
            'sr_notes':     sr_gate_notes,
        })

        in_trade = False; direction = None; entry_price = 0.0
        entry_time = None; entry_idx = -1; entry_path = ''
        stop_loss = 0.0; be_triggered = False; be_level = 0.0
        trail_active = False; sl_dist_initial = 0.0
        trade_max_favor = 0.0; trade_max_adv = 0.0
        sr_gate_notes = ''; sr_tp_price = None

    # ─── MAIN BAR LOOP ──────────────────────────────────────────
    for idx in range(n):
        close     = closes[idx]; hi = highs[idx]; lo = lows[idx]
        ef        = ema_fast[idx]; es = ema_slow[idx]; em = ema_macro[idx]
        atr       = float(atrs[idx])
        adx_v     = float(adx_vals[idx]) if not np.isnan(adx_vals[idx]) else 0.0
        slope     = float(slopes[idx])      if not np.isnan(slopes[idx])      else 0.0
        slope_exit= float(slopes_exit[idx]) if not np.isnan(slopes_exit[idx]) else 0.0
        cb        = int(consec_bearish[idx])
        ts        = ts_list[idx]; c_time = ts.time(); c_date = ts.date()
        date_str  = str(c_date); session = get_session(c_time)

        if c_date != prev_date: prev_date = c_date
        if c_time == time(9,30):
            opening_high[date_str]  = hi
            opening_close[date_str] = close
        if np.isnan(ef) or np.isnan(es) or np.isnan(em) or np.isnan(atr):
            eq_curve.append({'timestamp':ts,'equity':equity}); continue

        htf_long = bool(htf_long_bias[idx]) if not pd.isna(htf_long_bias[idx]) else False

        # ── MANAGE OPEN TRADE ────────────────────────────────────
        if in_trade:
            fav = (hi-entry_price) if direction=='long' else (entry_price-lo)
            adv = (entry_price-lo) if direction=='long' else (hi-entry_price)
            trade_max_favor = max(trade_max_favor, fav)
            trade_max_adv   = max(trade_max_adv,   adv)

            # Break-even
            if not be_triggered:
                if direction=='long'  and close >= be_level:
                    stop_loss = entry_price+1.0; be_triggered = True; trail_active = True
                elif direction=='short' and close <= be_level:
                    stop_loss = entry_price-1.0; be_triggered = True; trail_active = True

            # Swing structure trail (v4)
            if trail_active:
                if direction=='long':
                    stop_loss = max(stop_loss, get_swing_low(lows,idx,SWING_LOOKBACK)-SWING_BUFFER_PTS)
                else:
                    stop_loss = min(stop_loss, get_swing_high(highs,idx,SWING_LOOKBACK)+SWING_BUFFER_PTS)

            exit_p = exit_r = None

            # Gate 5: S/R Early Exit — if approaching opposing zone, close now
            trade_dir_updown = 'up' if direction=='long' else 'down'
            if be_triggered and sr_ctx.approaching_zone(close, trade_dir_updown, SR_EXIT_BUFFER_PTS):
                exit_p = close; exit_r = 'SR_ZONE_EXIT'

            if exit_p is None:
                if direction=='long'  and close <= stop_loss:
                    exit_p = stop_loss; exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'
                elif direction=='short' and close >= stop_loss:
                    exit_p = stop_loss; exit_r = 'TRAIL_SL' if trail_active else 'STOP_LOSS'
                elif ENABLE_EMA_CROSS_EXIT and be_triggered:
                    if   direction=='long'  and ef < es: exit_p = close; exit_r = 'EMA_CROSS_EXIT'
                    elif direction=='short' and ef > es: exit_p = close; exit_r = 'EMA_CROSS_EXIT'
                elif ENABLE_SLOPE_EXIT and be_triggered:
                    if   direction=='long'  and slope_exit < -SLOPE_MIN: exit_p = close; exit_r = 'SLOPE_REV_EXIT'
                    elif direction=='short' and slope_exit >  SLOPE_MIN: exit_p = close; exit_r = 'SLOPE_REV_EXIT'
                elif c_time >= EOD_HARD_EXIT:
                    exit_p = close; exit_r = 'EOD_EXIT'

            if exit_p is not None:
                do_exit(exit_p, ts, exit_r, idx)

        # ── ENTRY LOGIC ──────────────────────────────────────────
        if not in_trade:
            allowed = ['prime']
            if ENABLE_MIDDAY: allowed.append('midday')
            if ENABLE_EURO:   allowed.append('euro')
            if session not in allowed:
                eq_curve.append({'timestamp':ts,'equity':equity}); continue
            if daily_trades.get(date_str,0) >= MAX_TRADES_PER_DAY:
                eq_curve.append({'timestamp':ts,'equity':equity}); continue
            if daily_consec_loss.get(date_str,0) >= 2:
                eq_curve.append({'timestamp':ts,'equity':equity}); continue
            if not chop_filters_pass(close, ef, es, slope, adx_v, session):
                eq_curve.append({'timestamp':ts,'equity':equity}); continue

            retest_tol = atr * RETEST_ATR_MULT

            # ── LONG ENTRY ────────────────────────────────────────
            if ENABLE_LONG and htf_long:
                # Path A
                if ENABLE_PATH_A:
                    if (ef>es and close>em and slope>0 and
                            abs(close-ef)<=retest_tol and close>es):
                        sl_dist = atr * LONG_ATR_SL_MULT
                        ok, tp, rr, gf, notes = sr_gate_check(close, 'up', sr_ctx, sl_dist)
                        if ok:
                            do_enter('long', close, ts, atr, idx, 'A', notes, tp)
                            daily_trades[date_str] = daily_trades.get(date_str,0)+1
                        else:
                            gate_failed_counts[gf] = gate_failed_counts.get(gf,0)+1

                # Path B
                if not in_trade and ENABLE_PATH_B:
                    if (ef>es and close>em and slope>0 and
                            (close-ef)>=EMA9_PULLAWAY_PTS and
                            date_str in opening_high and close>opening_high[date_str]):
                        sl_dist = atr * LONG_ATR_SL_MULT
                        ok, tp, rr, gf, notes = sr_gate_check(close, 'up', sr_ctx, sl_dist)
                        if ok:
                            do_enter('long', close, ts, atr, idx, 'B', notes, tp)
                            daily_trades[date_str] = daily_trades.get(date_str,0)+1
                        else:
                            gate_failed_counts[gf] = gate_failed_counts.get(gf,0)+1

            # ── SHORT ENTRY ──────────────────────────────────────
            if not in_trade and ENABLE_SHORT:
                # Path A
                if ENABLE_PATH_A:
                    if (ef<es and close<em and slope<0 and
                            abs(close-ef)<=retest_tol and close<es and cb>=SHORT_CONFIRM_BARS):
                        sl_dist = atr * SHORT_ATR_SL_MULT
                        ok, tp, rr, gf, notes = sr_gate_check(close, 'down', sr_ctx, sl_dist)
                        if ok:
                            do_enter('short', close, ts, atr, idx, 'A', notes, tp)
                            daily_trades[date_str] = daily_trades.get(date_str,0)+1
                        else:
                            gate_failed_counts[gf] = gate_failed_counts.get(gf,0)+1

                # Path B
                if not in_trade and ENABLE_PATH_B:
                    if (ef<es and close<em and slope<0 and
                            (ef-close)>=EMA9_PULLAWAY_PTS and cb>=SHORT_CONFIRM_BARS and
                            date_str in opening_close and close<opening_close[date_str]):
                        sl_dist = atr * SHORT_ATR_SL_MULT
                        ok, tp, rr, gf, notes = sr_gate_check(close, 'down', sr_ctx, sl_dist)
                        if ok:
                            do_enter('short', close, ts, atr, idx, 'B', notes, tp)
                            daily_trades[date_str] = daily_trades.get(date_str,0)+1
                        else:
                            gate_failed_counts[gf] = gate_failed_counts.get(gf,0)+1

        eq_curve.append({'timestamp':ts,'equity':equity})

    if in_trade:
        do_exit(closes[-1], ts_list[-1], 'END_OF_DATA', n-1)

    return pd.DataFrame(trades), pd.DataFrame(eq_curve), gate_failed_counts


# ════════════════════════════════════════════════════════════════
# 9.  METRICS & REPORT  (extended v5)
# ════════════════════════════════════════════════════════════════

def compute_metrics(tdf):
    if tdf.empty: return {'message':'No trades found.'}
    pnl = tdf['pnl']
    wins = pnl[pnl>0]; losses = pnl[pnl<=0]; total = len(tdf)
    gp, gl = wins.sum(), abs(losses.sum())
    cum = pnl.cumsum(); max_dd = (cum.cummax()-cum).max()
    return {
        'Total Trades':       total,
        'Winning Trades':     len(wins),
        'Losing Trades':      len(losses),
        'Win Rate (%)':       round(len(wins)/total*100, 2),
        'Avg Profit (pts)':   round(wins.mean(),   2) if len(wins)   else 0,
        'Avg Loss (pts)':     round(losses.mean(), 2) if len(losses) else 0,
        'Largest Win (pts)':  round(pnl.max(), 2),
        'Largest Loss (pts)': round(pnl.min(), 2),
        'Profit Factor':      round(gp/gl, 2) if gl>0 else float('inf'),
        'Total P&L (pts)':    round(pnl.sum(), 2),
        'Max Drawdown (pts)': round(max_dd, 2),
        'Avg MFE (pts)':      round(tdf['mfe_pts'].mean(), 2),
        'Avg MAE (pts)':      round(tdf['mae_pts'].mean(), 2),
    }

def fmt_ts(ts):
    return ts.strftime('%d-%m-%Y %H:%M') if hasattr(ts,'strftime') else str(ts)

def print_results(tdf, gate_counts, sr_zones):
    SEP = '─'*170
    print(f"\n{SEP}\n  TRADE LOG  ({len(tdf)} trades)\n{SEP}")
    print(f"{'#':<5} {'Dir':<6} {'Path':<6} {'Entry Time':<22} {'Exit Time':<22}"
          f" {'Entry':>9} {'Exit':>9} {'SR_TP':>8} {'MFE':>7} {'MAE':>7}"
          f" {'P&L':>9}  {'Reason':<18}  SR Notes")
    print(SEP)
    for i, r in tdf.iterrows():
        tp_str = f"{r['sr_tp']:.0f}" if r['sr_tp'] else "N/A"
        print(f"{i+1:<5} {r['direction'].upper():<6} {'Path-'+r['entry_path']:<6}"
              f" {fmt_ts(r['entry_time']):<22} {fmt_ts(r['exit_time']):<22}"
              f" {r['entry_price']:>9.2f} {r['exit_price']:>9.2f} {tp_str:>8}"
              f" {r['mfe_pts']:>7.1f} {r['mae_pts']:>7.1f}"
              f" {r['pnl']:>+9.2f}  {r['exit_reason']:<18}  {r['sr_notes'][:80]}")

    metrics = compute_metrics(tdf)
    print(f"\n{'─'*62}\n  PERFORMANCE METRICS\n{'─'*62}")
    for k,v in metrics.items(): print(f"  {k:<30}: {v}")

    print(f"\n{'─'*62}\n  DIRECTION BREAKDOWN\n{'─'*62}")
    for d in ['long','short']:
        sub = tdf[tdf['direction']==d]
        if sub.empty: continue
        p = sub['pnl']; w = (p>0).sum()
        print(f"  {d.upper():<6}  trades={len(sub)}  wins={w}  "
              f"wr={w/len(sub)*100:.1f}%  total={p.sum():.1f}  "
              f"avg={p.mean():.1f}  avg_mfe={sub['mfe_pts'].mean():.1f}  "
              f"avg_mae={sub['mae_pts'].mean():.1f}")

    print(f"\n{'─'*62}\n  EXIT REASON BREAKDOWN\n{'─'*62}")
    bd = (tdf.groupby('exit_reason')['pnl']
          .agg(count='count', total_pnl='sum', avg_pnl='mean').round(2))
    print(bd.to_string())

    print(f"\n{'─'*62}\n  MONTHLY P&L\n{'─'*62}")
    tdf2 = tdf.copy()
    tdf2['month'] = tdf2['entry_time'].dt.tz_localize(None).dt.to_period('M').astype(str)
    monthly = (tdf2.groupby('month')['pnl']
               .agg(trades='count', total_pnl='sum',
                    win_trades=lambda x:(x>0).sum())
               .assign(win_rate=lambda x:(x['win_trades']/x['trades']*100).round(1))
               .round(2))
    print(monthly.to_string())

    print(f"\n{'─'*62}\n  S/R GATE FILTER SUMMARY (v5)\n{'─'*62}")
    total_blocked = sum(gate_counts.values())
    print(f"  Total EMA signals blocked by S/R gates: {total_blocked}")
    for gate, cnt in sorted(gate_counts.items(), key=lambda x:-x[1]):
        print(f"  {gate:<20}: {cnt:>4} signals blocked")

    print(f"\n{'─'*62}\n  S/R EARLY EXIT (Gate 5) TRADES\n{'─'*62}")
    sr_exits = tdf[tdf['exit_reason']=='SR_ZONE_EXIT']
    if sr_exits.empty:
        print("  No SR_ZONE_EXIT trades.")
    else:
        for _, r in sr_exits.iterrows():
            print(f"  {fmt_ts(r['entry_time'])} {r['direction'].upper()} "
                  f"entry={r['entry_price']:.0f} exit={r['exit_price']:.0f} "
                  f"pnl={r['pnl']:+.2f} mfe={r['mfe_pts']:.0f}")

    print(f"\n{'─'*62}\n  ACTIVE S/R ZONES ({len(sr_zones)} zones)\n{'─'*62}")
    supports    = sorted([z for z in sr_zones if z['type']=='Support'],    key=lambda x:-x['price'])
    resistances = sorted([z for z in sr_zones if z['type']=='Resistance'], key=lambda x: x['price'])
    print(f"  {'Type':<12} {'Price':>8} {'Lower':>8} {'Upper':>8} {'Str':>6} {'WT':>4} {'BR':>4}")
    print(f"  ─── RESISTANCE ───────────────────────────────────────────")
    for z in resistances:
        print(f"  {'RES':<12} {z['price']:>8.0f} {z['lower']:>8.0f} {z['upper']:>8.0f} "
              f"{z['strength']:>6.1f} {z['wt']:>4} {z['br']:>4}")
    print(f"  ─── SUPPORT ──────────────────────────────────────────────")
    for z in supports:
        print(f"  {'SUP':<12} {z['price']:>8.0f} {z['lower']:>8.0f} {z['upper']:>8.0f} "
              f"{z['strength']:>6.1f} {z['wt']:>4} {z['br']:>4}")

    print(f"\n{'─'*62}\n  v5 S/R LAYER SETTINGS\n{'─'*62}")
    print(f"  Gate 1 (entry buffer)    : ±{SR_ENTRY_BUFFER_PTS} pts from any zone edge")
    print(f"  Gate 2 (min free space)  : {SR_MIN_SPACE_PTS} pts to opposing zone")
    print(f"  Gate 3 (alignment req)   : {'YES' if SR_REQUIRE_ALIGNMENT else 'NO (advisory only)'}")
    print(f"  Gate 4 (min R:R)         : {SR_RR_MIN}x using S/R-derived TP")
    print(f"  Gate 5 (early exit buf)  : {SR_EXIT_BUFFER_PTS} pts from opposing zone")
    print(f"  Zone half-band           : ±{SR_ZONE_HALF_BAND} pts ({SR_ZONE_HALF_BAND*2:.0f}-pt zones)")
    print(f"  Min zone strength        : {SR_MIN_ZONE_STRENGTH}")
    print()


# ════════════════════════════════════════════════════════════════
# 10. HTML REPORT (v5 — extended from v4 with S/R annotations)
# ════════════════════════════════════════════════════════════════

def build_html_report(trades_df, raw_df, sr_zones):
    CONTEXT = 10

    def row_to_dict(r):
        ei, xi = int(r['entry_idx']), int(r['exit_idx'])
        start   = max(0, ei-CONTEXT)
        end     = min(len(raw_df), xi+CONTEXT+1)
        chunk   = raw_df.iloc[start:end][
            ['timestamp','open','high','low','close','ema_fast','ema_slow','ema_macro']
        ].copy()
        chunk['timestamp'] = chunk['timestamp'].dt.strftime('%H:%M %d%b')
        return {
            'direction':   r['direction'], 'entry_path': r['entry_path'],
            'entry_time':  fmt_ts(r['entry_time']), 'exit_time': fmt_ts(r['exit_time']),
            'entry_price': round(r['entry_price'],2), 'exit_price': round(r['exit_price'],2),
            'sl_at_entry': round(r['sl_at_entry'],2), 'sl_at_exit': round(r['sl_at_exit'],2),
            'sr_tp':       round(r['sr_tp'],2) if r['sr_tp'] else None,
            'be_triggered':bool(r['be_triggered']), 'pnl': round(r['pnl'],2),
            'mfe': round(r['mfe_pts'],2), 'mae': round(r['mae_pts'],2),
            'exit_reason': r['exit_reason'], 'sr_notes': r.get('sr_notes',''),
            'candles': chunk.to_dict(orient='records'),
            'rel_entry': ei-start, 'rel_exit': xi-start,
        }

    trade_data   = [row_to_dict(r) for _,r in trades_df.iterrows()]
    trade_json   = json.dumps(trade_data)
    sr_zones_json= json.dumps(sr_zones)
    metrics      = compute_metrics(trades_df)
    metrics_rows = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k,v in metrics.items())

    table_rows = ''
    for i, t in enumerate(trade_data):
        pnl_cls = 'win' if t['pnl']>0 else 'loss'
        dir_cls = 'long' if t['direction']=='long' else 'short'
        sign    = '+' if t['pnl']>0 else ''
        tp_str  = f"{t['sr_tp']:.0f}" if t['sr_tp'] else 'N/A'
        er      = t['exit_reason']
        er_cls  = 'er-sr' if er=='SR_ZONE_EXIT' else ('er-trail' if 'TRAIL' in er else '')
        table_rows += f"""
        <tr onclick="showChart({i})" class="trade-row" id="row-{i}">
          <td>{i+1}</td>
          <td class="{dir_cls}">{'▲ LONG' if t['direction']=='long' else '▼ SHORT'}</td>
          <td><span class="badge path-{'a' if t['entry_path']=='A' else 'b'}">Path {t['entry_path']}</span></td>
          <td>{t['entry_time']}</td><td>{t['exit_time']}</td>
          <td>{t['entry_price']:.2f}</td><td>{t['exit_price']:.2f}</td>
          <td class="tp">{tp_str}</td>
          <td class="mfe">+{t['mfe']:.1f}</td><td class="mae">-{t['mae']:.1f}</td>
          <td class="{pnl_cls}">{sign}{t['pnl']:.2f}</td>
          <td><span class="badge {er_cls}">{er}</span></td>
        </tr>"""

    total_pnl = metrics.get('Total P&L (pts)', 0)
    pnl_color = 'green' if total_pnl >= 0 else 'red'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Speed Demon v5 — S/R Conviction Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#080b12;color:#e2e8f0;font-family:'Courier New',monospace;font-size:12px}}
header{{background:#0d1117;border-bottom:1px solid #1e2336;padding:10px 20px;
        display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
header h1{{font-size:13px;color:#f1f5f9;letter-spacing:1px}}
header .sub{{font-size:9px;color:#475569;margin-top:2px}}
.kpis{{display:flex;gap:20px;margin-left:auto;flex-wrap:wrap}}
.kpi{{text-align:right}}.kpi .label{{font-size:9px;color:#475569;text-transform:uppercase}}
.kpi .val{{font-size:14px;font-weight:700}}
.green{{color:#4ade80}}.red{{color:#f87171}}
.gate-bar{{background:#0a0d14;border-bottom:1px solid #1e2336;padding:6px 20px;
           display:flex;gap:24px;flex-wrap:wrap;font-size:10px;color:#64748b}}
.gate-bar span{{color:#94a3b8;font-weight:700}}
.v5tag{{background:#059e6f22;color:#34d399;border:1px solid #059e6f55;
        padding:2px 7px;border-radius:3px;font-size:9px;margin-left:8px}}
.main{{display:flex;height:calc(100vh - 86px)}}
.left{{width:58%;overflow-y:auto;border-right:1px solid #1e2336}}
.right{{flex:1;overflow-y:auto;padding:14px;display:none;flex-direction:column;gap:10px}}
.right.visible{{display:flex}}
table{{width:100%;border-collapse:collapse}}
thead th{{background:#0d1117;color:#475569;font-size:9px;text-transform:uppercase;
          padding:5px 7px;text-align:left;position:sticky;top:0;z-index:1;
          border-bottom:1px solid #1e2336}}
.trade-row{{cursor:pointer;border-bottom:1px solid #0f1520}}
.trade-row:nth-child(even){{background:#0a0d14}}
.trade-row:hover,.trade-row.active{{background:#131929!important;border-left:3px solid #3b82f6}}
td{{padding:5px 7px}}
.long{{color:#4ade80;font-weight:700}}.short{{color:#f87171;font-weight:700}}
.win{{color:#4ade80;font-weight:700}}.loss{{color:#f87171;font-weight:700}}
.mfe{{color:#a3e635}}.mae{{color:#fb923c}}.tp{{color:#c084fc;font-weight:700}}
.badge{{background:#1e2336;color:#94a3b8;padding:2px 5px;border-radius:3px;font-size:9px}}
.badge.path-a{{background:#f59e0b22;color:#f59e0b;border:1px solid #f59e0b44}}
.badge.path-b{{background:#60a5fa22;color:#60a5fa;border:1px solid #60a5fa44}}
.er-sr{{background:#c084fc22;color:#c084fc;border:1px solid #c084fc44}}
.er-trail{{background:#4ade8022;color:#4ade80;border:1px solid #4ade8044}}
canvas{{display:block;width:100%;background:#0f1117;border-radius:8px}}
.chart-title{{font-size:12px;font-weight:700;color:#f1f5f9;display:flex;align-items:center;gap:8px}}
.close-btn{{margin-left:auto;background:#1e2336;border:none;color:#94a3b8;
            padding:4px 10px;border-radius:4px;cursor:pointer;font-family:inherit}}
.box{{background:#0d1117;border:1px solid #1e2336;border-radius:8px;padding:10px}}
.box h3{{font-size:9px;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}}
.dg{{display:grid;grid-template-columns:1fr 1fr;gap:4px 16px}}
.dg .row{{display:flex;justify-content:space-between;border-bottom:1px solid #1e2336;padding-bottom:3px}}
.dg .k{{color:#475569}}.dg .v{{color:#cbd5e1;font-weight:600}}
.box table td{{padding:3px 5px}}
.box table tr:nth-child(even){{background:#131929}}
.sr-notes{{font-size:9px;color:#94a3b8;margin-top:4px;padding:4px 8px;
           background:#0a0d14;border-radius:4px;border-left:3px solid #c084fc}}
</style></head><body>
<header>
  <div>
    <h1>SPEED DEMON v5 — EMA × S/R Conviction <span class="v5tag">SR LAYER</span></h1>
    <div class="sub">Gate1=zone-block · Gate2=min-space({SR_MIN_SPACE_PTS}pts) · Gate3=alignment · Gate4=R:R≥{SR_RR_MIN} · Gate5=early-exit({SR_EXIT_BUFFER_PTS}pts) · Click row for chart</div>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="label">Trades</div><div class="val">{metrics.get('Total Trades',0)}</div></div>
    <div class="kpi"><div class="label">Win Rate</div><div class="val green">{metrics.get('Win Rate (%)',0)}%</div></div>
    <div class="kpi"><div class="label">Profit Factor</div><div class="val">{metrics.get('Profit Factor',0)}</div></div>
    <div class="kpi"><div class="label">Total P&L</div><div class="val {pnl_color}">{total_pnl:+.1f} pts</div></div>
  </div>
</header>
<div class="main">
  <div class="left">
    <table><thead><tr>
      <th>#</th><th>Dir</th><th>Path</th><th>Entry Time</th><th>Exit Time</th>
      <th>Entry</th><th>Exit</th><th class="tp">SR TP</th>
      <th>MFE</th><th>MAE</th><th>P&L</th><th>Reason</th>
    </tr></thead>
    <tbody>{table_rows}</tbody></table>
  </div>
  <div class="right" id="right-panel">
    <div class="chart-title">
      <span id="chart-label">Trade Chart</span>
      <button class="close-btn" onclick="closeChart()">✕</button>
    </div>
    <canvas id="tradeCanvas" height="260"></canvas>
    <div class="box"><h3>Trade Details</h3>
      <div class="dg" id="details-grid"></div>
      <div class="sr-notes" id="sr-notes-box"></div>
    </div>
    <div class="box"><h3>Performance Metrics</h3><table>{metrics_rows}</table></div>
  </div>
</div>
<script>
const TRADES={trade_json};
const SR_ZONES={sr_zones_json};
const EMA_FAST={EMA_FAST},EMA_SLOW={EMA_SLOW},EMA_MACRO={EMA_MACRO};
let activeIdx=null;
function showChart(idx){{
  if(activeIdx===idx){{closeChart();return;}}
  activeIdx=idx;
  document.querySelectorAll('.trade-row').forEach((r,i)=>r.classList.toggle('active',i===idx));
  document.getElementById('right-panel').classList.add('visible');
  const t=TRADES[idx];
  document.getElementById('chart-label').textContent=
    `Trade #${{idx+1}} · ${{t.direction.toUpperCase()}} · ${{t.entry_time}} → ${{t.exit_time}}`;
  drawChart(t);
  const rows=[
    ['Entry Price',t.entry_price.toFixed(2)],['Exit Price',t.exit_price.toFixed(2)],
    ['SR TP',t.sr_tp?t.sr_tp.toFixed(0):'N/A'],['P&L',(t.pnl>0?'+':'')+t.pnl.toFixed(2)+' pts'],
    ['MFE','+'+t.mfe.toFixed(2)+' pts'],['MAE','-'+t.mae.toFixed(2)+' pts'],
    ['SL@Entry',t.sl_at_entry.toFixed(2)],['SL@Exit',t.sl_at_exit.toFixed(2)],
    ['Break-Even',t.be_triggered?'YES ✓':'NO'],['Exit Reason',t.exit_reason],
  ];
  document.getElementById('details-grid').innerHTML=rows.map(([k,v])=>
    `<div class="row"><span class="k">${{k}}</span><span class="v">${{v}}</span></div>`
  ).join('');
  document.getElementById('sr-notes-box').textContent='S/R: '+t.sr_notes;
}}
function closeChart(){{
  activeIdx=null;
  document.getElementById('right-panel').classList.remove('visible');
  document.querySelectorAll('.trade-row').forEach(r=>r.classList.remove('active'));
}}
function drawChart(t){{
  const canvas=document.getElementById('tradeCanvas');
  const ctx=canvas.getContext('2d');
  const DPR=window.devicePixelRatio||1,W=canvas.offsetWidth||640,CH=260;
  canvas.width=W*DPR;canvas.height=CH*DPR;canvas.style.height=CH+'px';
  ctx.scale(DPR,DPR);ctx.fillStyle='#0f1117';ctx.fillRect(0,0,W,CH);
  const cs=t.candles,n=cs.length;if(!n)return;
  const PL=58,PR=12,PT=14,PB=28,cW=W-PL-PR,cH=CH-PT-PB;
  const allP=cs.flatMap(c=>[c.high,c.low,
    isNaN(c.ema_fast)?c.close:c.ema_fast,isNaN(c.ema_slow)?c.close:c.ema_slow]);
  allP.push(t.sl_at_entry,t.entry_price,t.exit_price);
  if(t.sr_tp)allP.push(t.sr_tp);
  const pMin=Math.min(...allP)-2,pMax=Math.max(...allP)+2;
  const xS=i=>PL+(i+0.5)*(cW/n),yS=p=>PT+cH-((p-pMin)/(pMax-pMin))*cH;
  const bw=Math.max(3,(cW/n)*0.55);
  for(let g=0;g<=5;g++){{
    const f=g/5,y=PT+cH*(1-f),p=pMin+f*(pMax-pMin);
    ctx.strokeStyle='#1e2336';ctx.lineWidth=1;ctx.setLineDash([]);
    ctx.beginPath();ctx.moveTo(PL,y);ctx.lineTo(W-PR,y);ctx.stroke();
    ctx.fillStyle='#4a5568';ctx.textAlign='right';ctx.font='9px monospace';
    ctx.fillText(p.toFixed(0),PL-3,y+4);
  }}
  // Draw SR zones as bands
  SR_ZONES.forEach(z=>{{
    const y0=yS(z.upper),y1=yS(z.lower);
    ctx.fillStyle=z.type==='Support'?'rgba(38,166,154,0.12)':'rgba(239,83,80,0.12)';
    ctx.fillRect(PL,y0,cW,y1-y0);
    ctx.strokeStyle=z.type==='Support'?'rgba(38,166,154,0.4)':'rgba(239,83,80,0.4)';
    ctx.lineWidth=1;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(PL,yS(z.price));ctx.lineTo(W-PR,yS(z.price));ctx.stroke();
    ctx.setLineDash([]);
  }});
  [t.rel_entry,t.rel_exit].forEach((ri,k)=>{{
    ctx.strokeStyle=k===0?'#4ade8055':'#f8717155';
    ctx.lineWidth=1.5;ctx.setLineDash([4,2]);
    ctx.beginPath();ctx.moveTo(xS(ri),PT);ctx.lineTo(xS(ri),PT+cH);ctx.stroke();
    ctx.setLineDash([]);
  }});
  const hLine=(p,col,lbl)=>{{
    const y=yS(p);
    ctx.strokeStyle=col;ctx.lineWidth=1;ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(PL,y);ctx.lineTo(W-PR,y);ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=col;ctx.font='8px monospace';ctx.textAlign='left';
    ctx.fillText(lbl,W-PR+2,y+3);ctx.textAlign='right';
  }};
  hLine(t.sl_at_entry,'#ef444488','SL');
  if(t.sr_tp)hLine(t.sr_tp,'#c084fcaa','TP');
  const drawEMA=(key,col)=>{{
    ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.setLineDash([]);
    ctx.beginPath();let s=false;
    cs.forEach((c,i)=>{{const v=c[key];if(v==null||isNaN(v))return;
      const x=xS(i),y=yS(v);if(!s){{ctx.moveTo(x,y);s=true;}}else ctx.lineTo(x,y);}});
    ctx.stroke();
  }};
  drawEMA('ema_fast','#f59e0b');drawEMA('ema_slow','#60a5fa');drawEMA('ema_macro','#a78bfa');
  cs.forEach((c,i)=>{{
    const x=xS(i),bull=c.close>=c.open,col=bull?'#22c55e':'#ef4444';
    ctx.strokeStyle=col;ctx.lineWidth=1;ctx.setLineDash([]);
    ctx.beginPath();ctx.moveTo(x,yS(c.high));ctx.lineTo(x,yS(c.low));ctx.stroke();
    const by=yS(Math.max(c.open,c.close)),bh=Math.max(1,Math.abs(yS(c.open)-yS(c.close)));
    ctx.fillStyle=col;ctx.fillRect(x-bw/2,by,bw,bh);
  }});
  [[t.rel_entry,t.entry_price,'#4ade80','E'],[t.rel_exit,t.exit_price,'#f87171','X']]
    .forEach(([ri,price,col,lbl])=>{{
      const x=xS(ri),y=yS(price);
      ctx.fillStyle=col;ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();
      ctx.fillStyle=col;ctx.font='bold 9px monospace';ctx.textAlign='center';
      ctx.fillText(`${{lbl}} ${{price.toFixed(0)}}`,x,y+16);
    }});
  const step=Math.max(1,Math.floor(n/6));
  ctx.fillStyle='#4a5568';ctx.font='8px monospace';ctx.textAlign='center';
  cs.forEach((c,i)=>{{if(i%step!==0)return;ctx.fillText(c.timestamp,xS(i),CH-4);}});
  const leg=[['EMA9','#f59e0b'],['EMA21','#60a5fa'],['EMA50','#a78bfa'],
             ['Entry','#4ade80'],['Exit','#f87171'],['SR TP','#c084fc']];
  let lx=PL+4;
  leg.forEach(([lbl,col])=>{{
    ctx.fillStyle=col;ctx.fillRect(lx,PT+4,14,2);
    ctx.fillStyle='#94a3b8';ctx.font='8px monospace';ctx.textAlign='left';
    ctx.fillText(lbl,lx+17,PT+10);lx+=55;
  }});
}}
window.addEventListener('resize',()=>{{if(activeIdx!==null)drawChart(TRADES[activeIdx]);}});
</script></body></html>"""

def save_and_open(trades_df, raw_df, sr_zones, basename):
    html = build_html_report(trades_df, raw_df, sr_zones)
    path = basename+'_v5_report.html'
    with open(path,'w',encoding='utf-8') as f: f.write(html)
    print(f"  HTML report : {path}")
    try:
        webbrowser.open('file://'+os.path.abspath(path))
        print(f"  (Opened in browser)")
    except: pass


# ════════════════════════════════════════════════════════════════
# 11. MAIN
# ════════════════════════════════════════════════════════════════

def main():
    BAR = '═'*72
    print(f"\n{BAR}\n  SPEED DEMON SCALPER v5 — EMA MOMENTUM × S/R CONVICTION\n{BAR}")

    cli = sys.argv[1:]
    mode = 'csv' if cli else DATA_MODE.strip().lower()

    if mode == 'yfinance':
        print(f"\nData: yfinance  |  {YFINANCE_SYMBOL}")
        df = fetch_yfinance('5m','5m')
        base = os.path.join(os.getcwd(), YFINANCE_SYMBOL.replace('^','').replace('.','_')+'_sdv5')
    else:
        path = cli[0] if cli else CSV_5MIN_PATH
        if not os.path.exists(path): print(f"ERROR: {path} not found"); sys.exit(1)
        df = load_csv(path,'5m')
        base = os.path.splitext(path)[0]+'_sdv5'

    print(f"\nBuilding S/R zones from {len(df)} bars ...")
    sr_zones = build_sr_zones(
        df, half_band=SR_ZONE_HALF_BAND, cluster_tol=SR_CLUSTER_TOLERANCE,
        min_wt=SR_MIN_WICK_TOUCHES, min_sess=SR_MIN_SESSIONS, min_br=SR_MIN_REJECTIONS,
        top_n=SR_TOP_N, left_bars=SR_LEFT_BARS, right_bars=SR_RIGHT_BARS,
    )
    print(f"  → {len(sr_zones)} precision zones built\n")

    print("Computing indicators ...")
    df = compute_indicators_5m(df)
    print("Computing 15m HTF bias ...")
    df = compute_htf_bias(df)

    print("Running v5 backtest (EMA + S/R gates) ...\n")
    trades_df, equity_df, gate_counts = run_backtest(df, sr_zones)

    print(f"\n{BAR}\n  RESULTS\n{BAR}")
    if trades_df.empty:
        print("\n  No trades. Try loosening SR_MIN_SPACE_PTS or SR_RR_MIN.")
        return

    print_results(trades_df, gate_counts, sr_zones)

    trades_df.to_csv(base+'_trades.csv', index=False)
    equity_df.to_csv(base+'_equity.csv', index=False)
    print(f"\n  Trades: {base}_trades.csv")
    print(f"  Equity: {base}_equity.csv")
    save_and_open(trades_df, df, sr_zones, base)
    print(f"\n{BAR}\n")

if __name__ == '__main__':
    main()