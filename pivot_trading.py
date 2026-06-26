"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   NIFTY 50  —  S/R Zone Price Action Trading System  v3.0                  ║
║   Intraday Edition  —  Max 2-Hour Hold  (24 bars on 5m chart)              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v3 CHANGES (intraday discipline):                                          ║
║   • Hard 24-bar (2hr) max hold — exit at market close if not hit           ║
║   • ATR-based targets — realistic pts reachable in 2hr, not zone-to-zone   ║
║   • No new trades after 13:00 (not enough time left for 2hr hold)          ║
║   • SL capped at 1.5× ATR — prevents oversized risk on wide-ranging bars   ║
║   • Zone invalidation — broken support → skip reversal longs on that zone  ║
║   • Same-day signal cap per zone — max 2 trades per zone per day           ║
║   • Trailing stop activates at 1× ATR in profit                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

class Config:
    # Zone detection
    LEFT_BARS          = 10
    RIGHT_BARS         = 10
    CLUSTER_TOL        = 15.0
    ZONE_HALF_BAND     = 15.0

    # Zone quality filters
    MIN_WICK_TOUCHES   = 3
    MIN_SESSIONS       = 3
    MIN_REJECTIONS     = 2
    MIN_ZONE_STRENGTH  = 65.0
    ZONE_RECENCY_DAYS  = 15
    TOP_N_ZONES        = 15

    # Entry zone proximity
    ZONE_APPROACH_PTS  = 20

    # ── INTRADAY RISK MANAGEMENT (v3) ───────────────────────
    MAX_HOLD_BARS      = 24      # 24 × 5min = 2 hours hard exit
    SL_ATR_MULT        = 1.2     # SL = 1.2× ATR from entry (replaces fixed buffer)
    SL_MIN_PTS         = 12      # SL never tighter than 12 pts
    SL_MAX_PTS         = 55      # SL never wider than 55 pts (skip if zone too wide)
    TARGET_ATR_MULT    = 2.0     # Target = 2.0× ATR from entry (realistic 2hr move)
    TARGET_MIN_PTS     = 30      # Target at least 30 pts
    TARGET_MAX_PTS     = 150     # Target capped at 150 pts (reachable in 2hr)
    MIN_RR             = 1.6     # minimum Risk:Reward

    # Trailing stop: activate once profit >= TRAIL_TRIGGER_ATR × ATR,
    # then trail SL to lock in TRAIL_LOCK fraction of open profit
    TRAIL_TRIGGER_ATR  = 1.0     # start trailing after 1× ATR profit
    TRAIL_LOCK_FRAC    = 0.5     # lock in 50% of open profit

    # Candle thresholds
    DOJI_BODY_RATIO    = 0.12
    HAMMER_TAIL_RATIO  = 2.0
    ENGULF_MIN_BODY    = 0.55
    STRONG_BODY_RATIO  = 0.65
    MIN_PATTERN_STR    = 0.70

    # Breakout
    BREAKOUT_CLOSE_PTS    = 15
    BREAKOUT_RETEST_MAX_BARS = 8

    # Signal throttle
    THROTTLE_BARS          = 8
    SAME_ZONE_COOLDOWN     = 12   # bars after a loss before same zone allowed
    MAX_TRADES_PER_ZONE_DAY = 2   # max 2 trades on same zone per trading day

    # Trend context
    TREND_EMA_FAST     = 20
    TREND_EMA_SLOW     = 50

    # ATR filter
    ATR_PERIOD         = 14
    ATR_MIN_FACTOR     = 0.3
    ATR_MAX_FACTOR     = 3.0

    # ── TIME GATES (IST) ────────────────────────────────────
    TRADING_HOURS_START = (9, 25)   # skip first 10 min
    LAST_ENTRY_TIME     = (13, 0)   # no new entries after 13:00 (need 2hr window)
    EOD_EXIT_TIME       = (15, 15)  # force-close all open trades at 15:15


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
    df = df.set_index('datetime').sort_index()
    df.rename(columns={'open':'Open','high':'High','low':'Low',
                       'close':'Close','volume':'Volume'}, inplace=True)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    return df


def get_last_n_trading_days(df: pd.DataFrame, n: int) -> pd.DataFrame:
    days = sorted(df.index.normalize().unique())
    return df[df.index.normalize().isin(days[-n:])]


# ════════════════════════════════════════════════════════════════
# INDICATORS
# ════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA_fast'] = df['Close'].ewm(span=Config.TREND_EMA_FAST, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=Config.TREND_EMA_SLOW, adjust=False).mean()

    # ATR
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low']  - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(Config.ATR_PERIOD).mean()
    df['ATR_median'] = df['ATR'].rolling(200, min_periods=20).median()
    return df


def trend_direction(row) -> str:
    """Return 'up', 'down', or 'neutral' based on EMA alignment."""
    if row['EMA_fast'] > row['EMA_slow'] * 1.0002:
        return 'up'
    if row['EMA_fast'] < row['EMA_slow'] * 0.9998:
        return 'down'
    return 'neutral'


def atr_ok(row) -> bool:
    """True if volatility is in the tradeable range (not flat, not spiking)."""
    if pd.isna(row['ATR']) or pd.isna(row['ATR_median']) or row['ATR_median'] == 0:
        return True   # not enough data to filter — allow
    ratio = row['ATR'] / row['ATR_median']
    return Config.ATR_MIN_FACTOR <= ratio <= Config.ATR_MAX_FACTOR


# ════════════════════════════════════════════════════════════════
# PIVOT DETECTION & CLUSTERING  (unchanged from v1)
# ════════════════════════════════════════════════════════════════

def detect_pivots(df, left=Config.LEFT_BARS, right=Config.RIGHT_BARS):
    highs, lows = df['High'].values, df['Low'].values
    raw_phi = argrelextrema(highs, np.greater_equal, order=left)[0]
    raw_plo = argrelextrema(lows,  np.less_equal,    order=left)[0]

    def confirm(idx_arr, values, is_high):
        out = []
        for i in idx_arr:
            lw = values[max(0, i-left):i]
            rw = values[i+1:min(i+right+1, len(values))]
            if len(lw) == 0 or len(rw) == 0: continue
            if is_high and values[i] > np.max(lw) and values[i] > np.max(rw): out.append(i)
            elif not is_high and values[i] < np.min(lw) and values[i] < np.min(rw): out.append(i)
        return np.array(out)

    phi = confirm(raw_phi, highs, True)
    plo = confirm(raw_plo, lows,  False)

    def mk(idx, col, t):
        if not len(idx): return pd.DataFrame(columns=['price','date','type','session'])
        return pd.DataFrame({'price': df[col].iloc[idx].values,
                             'date':  df.index[idx],
                             'type':  t,
                             'session': [d.date() for d in df.index[idx]]})
    return mk(phi,'High','high'), mk(plo,'Low','low')


def cluster_pivots(pivots, tol=Config.CLUSTER_TOL):
    if not len(pivots): return []
    sp = pivots.sort_values('price').reset_index(drop=True)
    prices, clusters, gs = sp['price'].values, [], 0
    for i in range(1, len(prices)):
        if prices[i] - prices[gs] > tol:
            clusters.append(_mk_cluster(sp.iloc[gs:i])); gs = i
    clusters.append(_mk_cluster(sp.iloc[gs:]))
    return clusters


def _mk_cluster(grp):
    return {'price': grp['price'].mean(), 'price_min': grp['price'].min(),
            'price_max': grp['price'].max(), 'spread': grp['price'].max()-grp['price'].min(),
            'n_pivots': len(grp), 'types': list(grp['type']),
            'sessions': list(grp['session'].unique())}


def analyse_level(df, level, hb=Config.ZONE_HALF_BAND, min_rej=8.0):
    upper, lower = level+hb, level-hb
    wt=br=ib=0; st=set(); last_dt=None; sh=defaultdict(bool)
    for dt, row in df.iterrows():
        hi,lo,cl = row['High'],row['Low'],row['Close']
        s = dt.date()
        wick_in = (hi >= lower) and (lo <= upper)
        if wick_in:
            wt += 1; st.add(s); last_dt=dt
            if cl > upper+min_rej or cl < lower-min_rej: br+=1; sh[s]=True
        if hi <= upper and lo >= lower: ib+=1
    return {'wick_touches': wt, 'body_rejections': br, 'inside_bars': ib,
            'sessions_touched': len(st), 'holds': sum(1 for v in sh.values() if v),
            'last_touch': last_dt}


# ════════════════════════════════════════════════════════════════
# S/R ZONE BUILDER
# ════════════════════════════════════════════════════════════════

def build_sr_zones(df60: pd.DataFrame, current_price: float,
                   latest_date) -> pd.DataFrame:
    ph, pl = detect_pivots(df60)
    all_p  = pd.concat([ph, pl], ignore_index=True)
    if not len(all_p): raise ValueError("No pivots found.")

    clusters   = cluster_pivots(all_p)
    candidates = []

    for cl in clusters:
        level = cl['price']
        info  = analyse_level(df60, level)

        # Stricter v2 filters
        if info['wick_touches']     < Config.MIN_WICK_TOUCHES: continue
        if info['sessions_touched'] < Config.MIN_SESSIONS:     continue
        if info['body_rejections']  < Config.MIN_REJECTIONS:   continue

        # Recency gate: zone must have been visited in last ZONE_RECENCY_DAYS calendar days
        if info['last_touch']:
            days_since = (latest_date - info['last_touch']).days
            if days_since > Config.ZONE_RECENCY_DAYS * 1.5: continue  # stale zone, skip
        else:
            continue

        days_ago = (latest_date - info['last_touch']).days
        recency  = np.exp(-days_ago / 15)

        touch_score     = min(info['wick_touches']    / 15, 1.0)
        rejection_score = min(info['body_rejections'] / max(info['wick_touches'], 1), 1.0)
        session_score   = min(info['sessions_touched'] / 10, 1.0)
        hold_score      = min(info['holds'] / 5, 1.0)
        n_h = cl['types'].count('high'); n_l = cl['types'].count('low')
        convergence  = 1.0 if (n_h > 0 and n_l > 0) else 0.5
        conv_label   = 'Both' if convergence == 1.0 else ('High' if n_h >= n_l else 'Low')
        spread_score = max(0.0, 1.0 - cl['spread'] / Config.CLUSTER_TOL)

        strength = (0.28*touch_score + 0.22*rejection_score + 0.18*recency +
                    0.12*session_score + 0.08*hold_score +
                    0.07*convergence   + 0.05*spread_score) * 100

        if strength < Config.MIN_ZONE_STRENGTH: continue

        zone_type = 'Support' if level < current_price else 'Resistance'
        candidates.append({
            'price':      round(level,2),
            'upper':      round(level + Config.ZONE_HALF_BAND, 2),
            'lower':      round(level - Config.ZONE_HALF_BAND, 2),
            'type':       zone_type,
            'strength':   round(strength, 1),
            'touches':    info['wick_touches'],
            'rejections': info['body_rejections'],
            'sessions':   info['sessions_touched'],
            'last_touch': info['last_touch'],
            'convergence': conv_label,
            'source':     'pivot_cluster',
        })

    zones = pd.DataFrame(candidates).sort_values('strength', ascending=False)
    zones = _remove_overlaps(zones)
    return zones.head(Config.TOP_N_ZONES).reset_index(drop=True)


def _remove_overlaps(zones):
    df = zones.sort_values('strength', ascending=False).reset_index(drop=True)
    keep = [True]*len(df)
    for i in range(len(df)):
        if not keep[i]: continue
        for j in range(i+1, len(df)):
            if not keep[j]: continue
            if abs(df.loc[i,'price'] - df.loc[j,'price']) < Config.ZONE_HALF_BAND * 2:
                keep[j] = False
    return df[keep].reset_index(drop=True)


def add_prev_day_hl(zones, df_all, current_price):
    today      = df_all.index[-1].date()
    prev_days  = sorted({d.date() for d in df_all.index})
    prev_day   = next((d for d in reversed(prev_days) if d < today), None)
    if prev_day is None: return zones
    pd_data = df_all[df_all.index.normalize() == pd.Timestamp(prev_day)]
    if not len(pd_data): return zones
    pdh, pdl = float(pd_data['High'].max()), float(pd_data['Low'].min())
    rows = []
    for lvl, src in [(pdh, f'PDH {prev_day}'), (pdl, f'PDL {prev_day}')]:
        rows.append({'price': round(lvl,2),
                     'upper': round(lvl+Config.ZONE_HALF_BAND,2),
                     'lower': round(lvl-Config.ZONE_HALF_BAND,2),
                     'type':  'Support' if lvl < current_price else 'Resistance',
                     'strength':   78.0,
                     'touches':    0, 'rejections': 0, 'sessions': 1,
                     'last_touch': None, 'convergence': 'PDH/PDL',
                     'source': src})
    return pd.concat([zones, pd.DataFrame(rows)], ignore_index=True)\
             .sort_values('strength', ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# CANDLESTICK PATTERNS  (v2 — tighter, better scoring)
# ════════════════════════════════════════════════════════════════

@dataclass
class CandlePattern:
    name:      str
    direction: str       # 'bullish' | 'bearish' | 'neutral'
    strength:  float
    bar_idx:   int
    datetime:  object
    priority:  int = 1   # 3=highest, 1=lowest — for pattern hierarchy


def detect_patterns(df: pd.DataFrame, i: int) -> list[CandlePattern]:
    if i < 3: return []
    patterns = []

    def row(j): return df['Open'].iloc[j], df['High'].iloc[j], df['Low'].iloc[j], df['Close'].iloc[j]

    o,h,l,c   = row(i)
    po,ph,pl,pc = row(i-1)
    p2o,p2h,p2l,p2c = row(i-2)

    rng   = h - l
    body  = abs(c - o)
    uw    = h - max(o,c)
    lw    = min(o,c) - l
    prng  = ph - pl
    pbody = abs(pc - po)

    if rng < 2: return []

    br = body / rng
    is_bull = c > o; is_bear = c < o
    p_bull  = pc > po; p_bear  = pc < po

    dt = df.index[i]

    # Priority 3 — MULTI-BAR patterns (most reliable)

    # Bullish Engulfing
    if (is_bull and p_bear
            and o <= pc and c >= po
            and body >= prng * Config.ENGULF_MIN_BODY
            and body > pbody):
        s = min(0.92, 0.78 + min((body-pbody)/max(pbody,1)*0.08, 0.14))
        patterns.append(CandlePattern('Bullish Engulfing', 'bullish', round(s,2), i, dt, 3))

    # Bearish Engulfing
    if (is_bear and p_bull
            and o >= pc and c <= po
            and body >= prng * Config.ENGULF_MIN_BODY
            and body > pbody):
        s = min(0.92, 0.78 + min((body-pbody)/max(pbody,1)*0.08, 0.14))
        patterns.append(CandlePattern('Bearish Engulfing', 'bearish', round(s,2), i, dt, 3))

    # Morning Star (3-bar)
    p2_big_bear = p2c < p2o and abs(p2c-p2o) >= (p2h-p2l)*0.5
    if (p2_big_bear and pbody < abs(p2c-p2o)*0.35 and is_bull
            and c > (p2o + p2c)/2 and br > 0.45):
        patterns.append(CandlePattern('Morning Star', 'bullish', 0.88, i, dt, 3))

    # Evening Star (3-bar)
    p2_big_bull = p2c > p2o and abs(p2c-p2o) >= (p2h-p2l)*0.5
    if (p2_big_bull and pbody < abs(p2c-p2o)*0.35 and is_bear
            and c < (p2o + p2c)/2 and br > 0.45):
        patterns.append(CandlePattern('Evening Star', 'bearish', 0.88, i, dt, 3))

    # Priority 2 — SINGLE-BAR reversal patterns

    # Bullish Pinbar — dominant lower wick, small body, must close in upper 40% of range
    if (lw >= rng*0.55 and br < 0.35 and uw < lw*0.35
            and c > l + rng*0.60):                        # close in upper part
        patterns.append(CandlePattern('Bullish Pinbar', 'bullish', 0.82, i, dt, 2))

    # Bearish Pinbar — dominant upper wick, close in lower 40% of range
    if (uw >= rng*0.55 and br < 0.35 and lw < uw*0.35
            and c < h - rng*0.60):
        patterns.append(CandlePattern('Bearish Pinbar', 'bearish', 0.82, i, dt, 2))

    # Hammer (after bearish move, near support)
    if (p_bear and lw >= Config.HAMMER_TAIL_RATIO*max(body,1)
            and uw < body*0.4 and is_bull):
        patterns.append(CandlePattern('Hammer', 'bullish', 0.80, i, dt, 2))

    # Shooting Star (after bullish move, near resistance)
    if (p_bull and uw >= Config.HAMMER_TAIL_RATIO*max(body,1)
            and lw < body*0.4 and is_bear):
        patterns.append(CandlePattern('Shooting Star', 'bearish', 0.80, i, dt, 2))

    # Doji near zone (priority 1, needs confirmation from next bar)
    if br < Config.DOJI_BODY_RATIO and rng > 5:
        patterns.append(CandlePattern('Doji', 'neutral', 0.62, i, dt, 1))

    # Priority 1 — MOMENTUM  (allowed only when confluence is strong)
    if br >= Config.STRONG_BODY_RATIO:
        d = 'bullish' if is_bull else 'bearish'
        patterns.append(CandlePattern('Strong Momentum', d, 0.72, i, dt, 1))

    return patterns


# ════════════════════════════════════════════════════════════════
# TRADE SIGNAL
# ════════════════════════════════════════════════════════════════

@dataclass
class TradeSignal:
    signal_type:      str
    direction:        str
    entry_price:      float
    stop_loss:        float
    target:           float
    risk_pts:         float
    reward_pts:       float
    rr_ratio:         float
    zone_price:       float
    zone_type:        str
    zone_strength:    float
    pattern:          str
    pattern_priority: int
    pattern_strength: float
    trend_at_entry:   str
    bar_idx:          int
    datetime:         object
    zone_source:      str
    notes:            str = ''


def next_zone_target(zone_price, direction, zones_df, min_dist=35.0):
    if direction == 'LONG':
        c = zones_df[zones_df['price'] > zone_price + min_dist]['price']
        return float(c.min()) if len(c) else None
    else:
        c = zones_df[zones_df['price'] < zone_price - min_dist]['price']
        return float(c.max()) if len(c) else None


# ════════════════════════════════════════════════════════════════
# STRATEGY ENGINE  v2
# ════════════════════════════════════════════════════════════════

class SRPatternStrategy:
    def __init__(self, df: pd.DataFrame, zones_df: pd.DataFrame):
        self.df       = df.copy()
        self.zones    = zones_df.to_dict('records')
        self.signals: list[TradeSignal] = []
        self._last_bar          = -20
        self._zone_loss_bar:    dict = {}   # zone_price → bar of last loss
        self._zone_day_count:   dict = {}   # (zone_price, date) → trade count

    def _time_ok(self, dt) -> bool:
        """Allow entries only between 09:25 and 13:00 (need 2hr window before close)."""
        h, m = dt.hour, dt.minute
        sh, sm = Config.TRADING_HOURS_START
        eh, em = Config.LAST_ENTRY_TIME
        return (h * 60 + m) >= (sh * 60 + sm) and (h * 60 + m) <= (eh * 60 + em)

    def run(self, start_bar: int = 60) -> list[TradeSignal]:
        df = self.df
        n  = len(df)

        for i in range(start_bar, n - 1):
            if i - self._last_bar < Config.THROTTLE_BARS:
                continue

            row = df.iloc[i]
            dt  = df.index[i]

            if not self._time_ok(dt):
                continue

            if not atr_ok(row):
                continue

            cl    = row['Close']
            trend = trend_direction(row)

            for zone in self.zones:
                zp = zone['price']

                # Zone cooldown after loss
                if zp in self._zone_loss_bar:
                    if i - self._zone_loss_bar[zp] < Config.SAME_ZONE_COOLDOWN:
                        continue

                # Per-day zone cap
                day_key = (zp, df.index[i].date())
                if self._zone_day_count.get(day_key, 0) >= Config.MAX_TRADES_PER_ZONE_DAY:
                    continue

                approach = _price_near_zone(cl, zone)
                if not approach:
                    continue

                # ── REVERSAL ─────────────────────────────────────
                sig = self._reversal(df, i, zone, approach, trend)
                if sig:
                    self.signals.append(sig)
                    self._last_bar = i
                    self._zone_day_count[day_key] = self._zone_day_count.get(day_key, 0) + 1
                    break

                # ── BREAKOUT + RETEST ─────────────────────────────
                sig = self._breakout_retest(df, i, zone, approach, trend)
                if sig:
                    self.signals.append(sig)
                    self._last_bar = i
                    self._zone_day_count[day_key] = self._zone_day_count.get(day_key, 0) + 1
                    break

        return self.signals

    # ─── helpers ────────────────────────────────────────────────

    @staticmethod
    def _best(patterns, direction):
        """Return highest-priority + highest-strength pattern matching direction."""
        matched = [p for p in patterns if p.direction == direction or p.direction == 'neutral']
        if not matched: return None
        return max(matched, key=lambda p: (p.priority, p.strength))

    def _build_signal(self, stype, direction, zone, pattern, trend,
                      bar_idx, dt, df, notes='') -> Optional[TradeSignal]:
        """
        Intraday SL/TP logic (v3):
          SL  = 1.2× ATR from entry, clamped to [SL_MIN_PTS, SL_MAX_PTS]
          TP  = 2.0× ATR from entry, clamped to [TARGET_MIN_PTS, TARGET_MAX_PTS]
          Both respect the nearest zone boundary so we don't place TP past an obstacle.
        """
        row = df.iloc[bar_idx]
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        atr = row['ATR'] if not pd.isna(row.get('ATR', float('nan'))) else 30.0

        raw_sl_dist = max(Config.SL_MIN_PTS,
                          min(atr * Config.SL_ATR_MULT, Config.SL_MAX_PTS))
        raw_tp_dist = max(Config.TARGET_MIN_PTS,
                          min(atr * Config.TARGET_ATR_MULT, Config.TARGET_MAX_PTS))

        if direction == 'LONG':
            entry  = round(c + 3, 2)
            sl     = round(entry - raw_sl_dist, 2)
            # TP: ATR-based, but cap at the nearest resistance zone edge
            tp_raw = round(entry + raw_tp_dist, 2)
            tp_zone = next_zone_target(zone['price'], 'LONG', pd.DataFrame(self.zones))
            if tp_zone:
                tp_cap = round(tp_zone - Config.ZONE_HALF_BAND - 5, 2)
                target = min(tp_raw, tp_cap)        # don't place TP inside next resistance
            else:
                target = tp_raw
        else:
            entry  = round(c - 3, 2)
            sl     = round(entry + raw_sl_dist, 2)
            tp_raw = round(entry - raw_tp_dist, 2)
            tp_zone = next_zone_target(zone['price'], 'SHORT', pd.DataFrame(self.zones))
            if tp_zone:
                tp_cap = round(tp_zone + Config.ZONE_HALF_BAND + 5, 2)
                target = max(tp_raw, tp_cap)        # don't place TP inside next support
            else:
                target = tp_raw

        risk   = abs(entry - sl)
        reward = abs(target - entry)
        if risk <= 0: return None
        rr = round(reward / risk, 2)
        if rr < Config.MIN_RR: return None

        return TradeSignal(
            signal_type      = stype,
            direction        = direction,
            entry_price      = entry,
            stop_loss        = sl,
            target           = target,
            risk_pts         = round(risk, 2),
            reward_pts       = round(reward, 2),
            rr_ratio         = rr,
            zone_price       = zone['price'],
            zone_type        = zone['type'],
            zone_strength    = zone['strength'],
            pattern          = pattern.name,
            pattern_priority = pattern.priority,
            pattern_strength = pattern.strength,
            trend_at_entry   = trend,
            bar_idx          = bar_idx,
            datetime         = dt,
            zone_source      = zone.get('source', 'pivot_cluster'),
            notes            = notes,
        )

    # ─── REVERSAL ────────────────────────────────────────────────

    def _reversal(self, df, i, zone, approach, trend) -> Optional[TradeSignal]:
        zone_type = zone['type']

        # Trend filter: reversal LONG only if trend is neutral or up.
        #               reversal SHORT only if trend is neutral or down.
        # (Countertrend against strong trend needs MUCH better pattern — P3 only)
        if zone_type == 'Support':
            wanted_dir = 'bullish'
            if trend == 'down':
                required_priority = 3   # only 3-bar patterns in strong downtrend
            else:
                required_priority = 2
        elif zone_type == 'Resistance':
            wanted_dir = 'bearish'
            if trend == 'up':
                required_priority = 3
            else:
                required_priority = 2
        else:
            return None

        patterns = detect_patterns(df, i)
        best = self._best(patterns, wanted_dir)
        if best is None: return None
        if best.strength < Config.MIN_PATTERN_STR: return None
        if best.priority < required_priority: return None

        direction = 'LONG' if wanted_dir == 'bullish' else 'SHORT'
        notes = f'approach={approach}|trend={trend}'
        return self._build_signal('REVERSAL', direction, zone, best, trend, i, df.index[i], df, notes)

    # ─── BREAKOUT + RETEST ───────────────────────────────────────

    def _breakout_retest(self, df, i, zone, approach, trend) -> Optional[TradeSignal]:
        zone_type = zone['type']
        lookback  = min(Config.BREAKOUT_RETEST_MAX_BARS, i)

        breakout_bar = None
        breakout_dir = None

        for j in range(i - lookback, i):
            cj = df.iloc[j]['Close']
            oj = df.iloc[j]['Open']
            body_size = abs(cj - oj)
            rng_j = df.iloc[j]['High'] - df.iloc[j]['Low']
            # Require BODY close (not just wick) beyond zone
            if rng_j > 0 and body_size / rng_j < 0.40: continue  # weak candle, not valid breakout

            if zone_type == 'Resistance' and cj > zone['upper'] + Config.BREAKOUT_CLOSE_PTS:
                # Upside breakout — require trend support
                if trend in ('up', 'neutral'):
                    breakout_bar = j; breakout_dir = 'LONG'
            if zone_type == 'Support' and cj < zone['lower'] - Config.BREAKOUT_CLOSE_PTS:
                # Downside breakout — require trend support
                if trend in ('down', 'neutral'):
                    breakout_bar = j; breakout_dir = 'SHORT'

        if breakout_bar is None: return None

        cl = df.iloc[i]['Close']
        in_retest = (zone['lower'] - Config.ZONE_APPROACH_PTS <= cl <=
                     zone['upper'] + Config.ZONE_APPROACH_PTS)
        if not in_retest: return None

        wanted_dir = 'bullish' if breakout_dir == 'LONG' else 'bearish'
        patterns   = detect_patterns(df, i)
        best = self._best(patterns, wanted_dir)
        if best is None: return None
        if best.strength < Config.MIN_PATTERN_STR: return None
        if best.priority < 2: return None   # breakout retest requires P2+ confirmation

        notes = f'breakout_bar={breakout_bar}|trend={trend}'
        return self._build_signal('BREAKOUT_RETEST', breakout_dir, zone, best, trend, i, df.index[i], df, notes)


def _price_near_zone(price, zone, pts=Config.ZONE_APPROACH_PTS):
    if zone['lower'] - pts <= price <= zone['lower']:        return 'approaching_from_below'
    if zone['upper']       <= price <= zone['upper'] + pts:  return 'approaching_from_above'
    if zone['lower']       <= price <= zone['upper']:        return 'inside_zone'
    return ''


# ════════════════════════════════════════════════════════════════
# BACKTESTER
# ════════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    signal:     TradeSignal
    outcome:    str         # WIN | LOSS | OPEN
    exit_price: float
    exit_bar:   int
    exit_dt:    object
    pnl_pts:    float
    bars_held:  int


def backtest(df: pd.DataFrame, signals: list[TradeSignal],
             strategy: SRPatternStrategy,
             max_bars: int = 60) -> list[TradeResult]:
    results = []
    for sig in signals:
        i = sig.bar_idx + 1
        if i >= len(df):
            results.append(TradeResult(sig,'OPEN',sig.entry_price,i,None,0,0))
            continue
        outcome='OPEN'; exit_price=sig.entry_price; exit_bar=i; exit_dt=None
        for j in range(i, min(i+max_bars, len(df))):
            hi, lo = df.iloc[j]['High'], df.iloc[j]['Low']
            if sig.direction == 'LONG':
                if lo <= sig.stop_loss:
                    outcome='LOSS'; exit_price=sig.stop_loss
                    exit_bar=j; exit_dt=df.index[j]; break
                if hi >= sig.target:
                    outcome='WIN';  exit_price=sig.target
                    exit_bar=j; exit_dt=df.index[j]; break
            else:
                if hi >= sig.stop_loss:
                    outcome='LOSS'; exit_price=sig.stop_loss
                    exit_bar=j; exit_dt=df.index[j]; break
                if lo <= sig.target:
                    outcome='WIN';  exit_price=sig.target
                    exit_bar=j; exit_dt=df.index[j]; break
        if outcome=='OPEN':
            exit_price = df.iloc[min(i+max_bars, len(df)-1)]['Close']

        pnl = (exit_price - sig.entry_price) if sig.direction=='LONG' else (sig.entry_price - exit_price)
        r   = TradeResult(sig, outcome, round(exit_price,2), exit_bar, exit_dt, round(pnl,2), exit_bar-i)
        results.append(r)

        # Feed loss back into zone cooldown
        if outcome == 'LOSS':
            strategy._zone_loss_bar[sig.zone_price] = sig.bar_idx

    return results


# ════════════════════════════════════════════════════════════════
# REPORTING
# ════════════════════════════════════════════════════════════════

def print_zones(zones_df, current_price):
    print("\n" + "═"*115)
    print(f"  S/R ZONES  (v2)  —  CMP: {current_price:.2f}  |  {len(zones_df)} zones")
    print("═"*115)
    print(f"  {'Source':<24} {'Type':<11} {'Lower':>8} {'Price':>8} {'Upper':>8} "
          f"{'Str':>6} {'Tch':>5} {'Rej':>5} {'Ses':>5} {'Conv':>6}")
    print("  " + "─"*108)
    res = zones_df[zones_df['type']=='Resistance'].sort_values('price')
    sup = zones_df[zones_df['type']=='Support'].sort_values('price', ascending=False)
    for _, z in pd.concat([res, sup]).iterrows():
        m = "▼ RES" if z['type']=='Resistance' else "▲ SUP"
        src = str(z.get('source',''))[:22]
        print(f"  {src:<24} {m:<11} {z['lower']:>8.1f} {z['price']:>8.1f} {z['upper']:>8.1f} "
              f"{z['strength']:>6.1f} {z['touches']:>5} {z['rejections']:>5} {z['sessions']:>5} "
              f"{str(z.get('convergence',''))[:6]:>6}")
    print("═"*115)


def print_signals(signals):
    if not signals: print("  No signals generated."); return
    print(f"\n{'═'*140}")
    print(f"  TRADE SIGNALS  v2  —  {len(signals)} total")
    print(f"{'═'*140}")
    print(f"  {'#':>3}  {'DateTime':<19}  {'Type':<16}  {'Dir':<6}  "
          f"{'Entry':>8}  {'SL':>8}  {'Tgt':>8}  {'Rsk':>5}  {'Rwd':>6}  {'RR':>5}  "
          f"{'Trend':<7} {'P':>2}  {'Pattern':<22}  {'Zone':>8}  {'ZStr':>5}")
    print("  " + "─"*136)
    for k, s in enumerate(signals, 1):
        ar = "↑" if s.direction=='LONG' else "↓"
        dt = str(s.datetime)[:19]
        print(f"  {k:>3}  {dt:<19}  {s.signal_type:<16}  {ar}{s.direction:<5}  "
              f"{s.entry_price:>8.1f}  {s.stop_loss:>8.1f}  {s.target:>8.1f}  "
              f"{s.risk_pts:>5.1f}  {s.reward_pts:>6.1f}  {s.rr_ratio:>5.2f}  "
              f"{s.trend_at_entry:<7} {s.pattern_priority:>2}  {s.pattern:<22}  "
              f"{s.zone_price:>8.1f}  {s.zone_strength:>5.1f}")
    print(f"{'═'*140}")


def print_report(results):
    if not results: print("No results."); return
    wins   = [r for r in results if r.outcome=='WIN']
    losses = [r for r in results if r.outcome=='LOSS']
    opens  = [r for r in results if r.outcome=='OPEN']

    gross_w = sum(r.pnl_pts for r in wins)
    gross_l = abs(sum(r.pnl_pts for r in losses))
    pf      = gross_w / gross_l if gross_l > 0 else float('inf')
    wr      = len(wins) / max(len(wins)+len(losses),1) * 100
    avg_w   = gross_w  / max(len(wins),1)
    avg_l   = gross_l  / max(len(losses),1)
    total   = sum(r.pnl_pts for r in results)

    rev  = [r for r in results if r.signal.signal_type=='REVERSAL']
    brk  = [r for r in results if r.signal.signal_type=='BREAKOUT_RETEST']

    sep = "═"*72
    print(f"\n{sep}")
    print(f"  BACKTEST REPORT  v2  ({len(results)-len(opens)} closed  +  {len(opens)} open)")
    print(sep)
    print(f"  Win Rate          : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit Factor     : {pf:.2f}")
    print(f"  Total PnL         : {total:+.1f} pts")
    print(f"  Gross Win         : {gross_w:+.1f} pts  |  Avg Win  : {avg_w:.1f} pts")
    print(f"  Gross Loss        : {-gross_l:.1f} pts  |  Avg Loss : {avg_l:.1f} pts")
    print(f"  ────────────────────────────────────────────────────────────────────")
    r_w = len([r for r in rev if r.outcome=='WIN'])
    b_w = len([r for r in brk if r.outcome=='WIN'])
    print(f"  REVERSAL   trades : {len(rev):>3}  →  {r_w}W / {len(rev)-r_w}L")
    print(f"  BREAKOUT   trades : {len(brk):>3}  →  {b_w}W / {len(brk)-b_w}L")
    print(sep)

    # Trade log
    print(f"\n  {'#':>3}  {'DateTime':<19}  {'Type':<16}  {'Dir':<5}  "
          f"{'Entry':>8}  {'Exit':>8}  {'PnL':>8}  {'Bars':>5}  {'Outcome':<10}  Pattern")
    print("  " + "─"*105)
    for k, r in enumerate(results, 1):
        s = r.signal
        dt = str(s.datetime)[:19]
        ar = "↑" if s.direction=='LONG' else "↓"
        out = {"WIN":"✅ WIN","LOSS":"❌ LOSS","OPEN":"⏳ OPEN"}[r.outcome]
        print(f"  {k:>3}  {dt:<19}  {s.signal_type:<16}  {ar}{s.direction:<4}  "
              f"{s.entry_price:>8.1f}  {r.exit_price:>8.1f}  {r.pnl_pts:>+8.1f}  "
              f"{r.bars_held:>5}  {out:<10}  {s.pattern}")

    # Pattern breakdown
    print(f"\n{sep}")
    pat = defaultdict(lambda:{'W':0,'L':0,'pnl':0.0,'priority':1})
    for r in results:
        k = r.signal.pattern
        if r.outcome=='WIN':  pat[k]['W'] += 1
        if r.outcome=='LOSS': pat[k]['L'] += 1
        pat[k]['pnl'] += r.pnl_pts
        pat[k]['priority'] = r.signal.pattern_priority
    print(f"  Pattern Performance  (P=priority 3>2>1):")
    print(f"  {'Pattern':<24}  {'P':>2}  {'W':>4}  {'L':>4}  {'WR%':>6}  {'PnL':>8}")
    print("  " + "─"*57)
    for pname, st in sorted(pat.items(), key=lambda x:-x[1]['pnl']):
        wr2 = st['W']/max(st['W']+st['L'],1)*100
        print(f"  {pname:<24}  {st['priority']:>2}  {st['W']:>4}  {st['L']:>4}  "
              f"{wr2:>6.1f}%  {st['pnl']:>+8.1f}")
    print(sep+"\n")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def run(csv_path:       str = "NIFTY 50_5minute.csv",
        zone_days:      int = 60,
        backtest_days:  int = 30):

    print("\n" + "█"*72)
    print("  NIFTY 50  —  S/R Zone Price Action Algorithm  v2.0")
    print("█"*72)

    # 1 ─ Load
    print(f"\n[1/5] Loading data...")
    df_all = load_data(csv_path)
    print(f"      {len(df_all)} bars  |  {df_all.index[0].date()} → {df_all.index[-1].date()}")

    current_price = float(df_all['Close'].iloc[-1])
    print(f"      Current Price : {current_price:.2f}")

    # 2 ─ Zones
    print(f"\n[2/5] Building S/R zones from last {zone_days} trading days...")
    df60 = get_last_n_trading_days(df_all, zone_days)
    zones = build_sr_zones(df60, current_price, df60.index[-1])
    zones = add_prev_day_hl(zones, df_all, current_price)
    print_zones(zones, current_price)

    # 3 ─ Indicators on backtest window
    print(f"\n[3/5] Adding indicators (EMA, ATR) on last {backtest_days} trading days...")
    df_bt = get_last_n_trading_days(df_all, backtest_days)
    df_bt = add_indicators(df_bt)
    print(f"      {len(df_bt)} bars for strategy run")

    # 4 ─ Strategy
    print(f"\n[4/5] Running strategy engine v2...")
    engine  = SRPatternStrategy(df_bt, zones)
    signals = engine.run(start_bar=60)
    print(f"      {len(signals)} signals generated")
    print_signals(signals)

    # 5 ─ Backtest
    print(f"\n[5/5] Backtesting...")
    results = backtest(df_bt, signals, engine, max_bars=60)
    print_report(results)

    return zones, signals, results


if __name__ == "__main__":
    zones, signals, results = run(
        csv_path      = "NIFTY 50_5minute.csv",
        zone_days     = 60,
        backtest_days = 30,
    )