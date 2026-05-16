"""
Contraction Zone Backtester
============================
Usage:
    python contraction_backtest.py --file your_data.csv

Optional flags:
    --lookback      int     Candles to measure avg range (default: 10)
    --threshold     float   Compression ratio to qualify as contraction (default: 0.70)
    --squeeze_len   int     Min consecutive shrinking candles for 'progressive' label (default: 3)
    --atr_period    int     ATR period for volatility context (default: 14)
    --min_zone_gap  int     Min candles between zones to avoid duplicates (default: 3)
    --output        str     Output CSV filename (default: contraction_zones.csv)
    --summary       str     Output summary TXT filename (default: contraction_summary.txt)

CSV must have columns: time (or date/datetime), open, high, low, close
"""

import argparse
import sys
import os
import math
from datetime import datetime

# ── dependency check ─────────────────────────────────────────────────────────
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("[ERROR] Required packages missing. Run:  pip install pandas numpy")
    sys.exit(1)

# ── column name aliases ───────────────────────────────────────────────────────
TIME_ALIASES  = ['time', 'date', 'datetime', 'timestamp', 'index', 'candle_time', 'bar']
OPEN_ALIASES  = ['open', 'o']
HIGH_ALIASES  = ['high', 'h']
LOW_ALIASES   = ['low', 'l']
CLOSE_ALIASES = ['close', 'c', 'ltp', 'last']

def find_col(df, aliases):
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None

# ── CSV loader ────────────────────────────────────────────────────────────────
def load_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    time_col  = find_col(df, TIME_ALIASES)
    open_col  = find_col(df, OPEN_ALIASES)
    high_col  = find_col(df, HIGH_ALIASES)
    low_col   = find_col(df, LOW_ALIASES)
    close_col = find_col(df, CLOSE_ALIASES)

    missing = [name for name, col in [('high', high_col), ('low', low_col), ('close', close_col)] if col is None]
    if missing:
        print(f"[ERROR] Could not find columns: {missing}")
        print(f"        Found columns: {list(df.columns)}")
        sys.exit(1)

    rename = {}
    if time_col:  rename[time_col]  = 'time'
    if open_col:  rename[open_col]  = 'open'
    rename[high_col]  = 'high'
    rename[low_col]   = 'low'
    rename[close_col] = 'close'
    df = df.rename(columns=rename)

    if 'time' not in df.columns:
        df['time'] = range(len(df))
    if 'open' not in df.columns:
        df['open'] = df['close']

    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['high', 'low', 'close']).reset_index(drop=True)
    return df

# ── technical indicators ──────────────────────────────────────────────────────
def compute_atr(df, period=14):
    high = df['high']
    low  = df['low']
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_body_ratio(df):
    full_range = (df['high'] - df['low']).replace(0, np.nan)
    body = (df['close'] - df['open']).abs()
    return (body / full_range).fillna(0)

def compute_wick_ratio(df):
    full_range = (df['high'] - df['low']).replace(0, np.nan)
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    total_wick = upper_wick + lower_wick
    return (total_wick / full_range).fillna(0)

def candle_direction(df):
    return np.where(df['close'] >= df['open'], 'Bullish', 'Bearish')

def volume_trend(df, lookback):
    if 'volume' in df.columns:
        avg_vol = df['volume'].rolling(lookback).mean().shift(1)
        curr_vol = df['volume']
        ratio = curr_vol / avg_vol.replace(0, np.nan)
        return ratio.fillna(1.0)
    return pd.Series([np.nan] * len(df))

# ── core contraction detector ─────────────────────────────────────────────────
def detect_contraction_zones(df, lookback=10, threshold=0.70,
                              squeeze_len=3, atr_period=14, min_zone_gap=3):
    df = df.copy().reset_index(drop=True)
    df['range']       = df['high'] - df['low']
    df['atr']         = compute_atr(df, atr_period)
    df['body_ratio']  = compute_body_ratio(df)
    df['wick_ratio']  = compute_wick_ratio(df)
    df['direction']   = candle_direction(df)
    df['vol_ratio']   = volume_trend(df, lookback)

    # rolling stats
    df['avg_range']   = df['range'].rolling(lookback).mean().shift(1)
    df['std_range']   = df['range'].rolling(lookback).std().shift(1)
    df['min_range']   = df['range'].rolling(lookback).min().shift(1)
    df['max_range']   = df['range'].rolling(lookback).max().shift(1)

    # price trend context
    df['sma_20']      = df['close'].rolling(20).mean()
    df['price_trend'] = np.where(df['close'] > df['sma_20'], 'Above SMA20', 'Below SMA20')

    zones = []
    last_zone_idx = -999

    for i in range(lookback, len(df)):
        avg_r  = df.at[i, 'avg_range']
        curr_r = df.at[i, 'range']

        if pd.isna(avg_r) or avg_r == 0:
            continue

        compression_ratio = curr_r / avg_r
        is_contracting = compression_ratio < threshold

        if not is_contracting:
            continue

        # skip if too close to last zone
        if (i - last_zone_idx) < min_zone_gap:
            continue

        # --- progressive squeeze detection ---
        tail_ranges = df['range'].iloc[max(0, i - squeeze_len):i + 1].tolist()
        is_progressive = all(
            tail_ranges[j] <= tail_ranges[j - 1]
            for j in range(1, len(tail_ranges))
        ) if len(tail_ranges) >= squeeze_len else False

        # --- zone strength score (0-100) ---
        score = 0
        score += max(0, (1 - compression_ratio / threshold)) * 40   # compression depth: up to 40
        score += (20 if is_progressive else 0)                        # progressive: 20
        std_r = df.at[i, 'std_range']
        if not pd.isna(std_r) and std_r > 0:
            z_score = (curr_r - avg_r) / std_r
            score += max(0, min(20, -z_score * 5))                   # std-dev significance: up to 20
        min_r = df.at[i, 'min_range']
        if not pd.isna(min_r) and min_r > 0:
            near_min = 1 - ((curr_r - min_r) / (avg_r - min_r + 1e-9))
            score += max(0, min(20, near_min * 20))                  # near-range-low bonus: up to 20
        score = min(100, round(score, 1))

        # zone classification
        if score >= 70:
            zone_type = 'Extreme Squeeze'
        elif score >= 50:
            zone_type = 'Strong Contraction'
        elif score >= 30:
            zone_type = 'Moderate Contraction'
        else:
            zone_type = 'Mild Contraction'

        # candle context window (lookback candles before this)
        window = df.iloc[max(0, i - lookback):i + 1]
        trend_dir = 'Uptrend' if window['close'].iloc[-1] > window['close'].iloc[0] else 'Downtrend'

        # ATR context
        atr_val = df.at[i, 'atr']
        atr_pct = (atr_val / df.at[i, 'close'] * 100) if not pd.isna(atr_val) else np.nan

        # rolling breakout levels (for post-zone sweep detection)
        prior_window = df.iloc[max(0, i - lookback):i]
        resistance   = prior_window['high'].max()
        support      = prior_window['low'].min()

        # volume context
        vol_ratio = df.at[i, 'vol_ratio']

        row = {
            # identification
            'bar_index'          : i,
            'time'               : df.at[i, 'time'],

            # candle prices
            'open'               : round(df.at[i, 'open'], 4),
            'high'               : round(df.at[i, 'high'], 4),
            'low'                : round(df.at[i, 'low'], 4),
            'close'              : round(df.at[i, 'close'], 4),

            # range metrics
            'candle_range'       : round(curr_r, 4),
            'avg_range_lookback' : round(avg_r, 4),
            'range_std_dev'      : round(std_r, 4) if not pd.isna(std_r) else '',
            'min_range_window'   : round(min_r, 4) if not pd.isna(min_r) else '',
            'max_range_window'   : round(df.at[i, 'max_range'], 4) if not pd.isna(df.at[i, 'max_range']) else '',
            'compression_ratio'  : round(compression_ratio, 4),
            'compression_pct'    : f"{round(compression_ratio * 100, 1)}%",

            # ATR
            'atr'                : round(atr_val, 4) if not pd.isna(atr_val) else '',
            'atr_pct_of_close'   : f"{round(atr_pct, 2)}%" if not pd.isna(atr_pct) else '',

            # candle anatomy
            'body_ratio'         : round(df.at[i, 'body_ratio'], 3),
            'wick_ratio'         : round(df.at[i, 'wick_ratio'], 3),
            'candle_direction'   : df.at[i, 'direction'],

            # zone classification
            'zone_type'          : zone_type,
            'zone_score'         : score,
            'is_progressive_squeeze': is_progressive,

            # market context
            'prior_trend'        : trend_dir,
            'price_vs_sma20'     : df.at[i, 'price_trend'],

            # key levels
            'resistance_level'   : round(resistance, 4),
            'support_level'      : round(support, 4),
            'range_above_support': round(df.at[i, 'close'] - support, 4),
            'range_below_resist' : round(resistance - df.at[i, 'close'], 4),

            # volume (if available)
            'volume_ratio_vs_avg': round(vol_ratio, 3) if not pd.isna(vol_ratio) else 'N/A',

            # lookahead breakout (filled in second pass)
            'candles_to_breakout': '',
            'breakout_direction' : '',
            'breakout_magnitude' : '',
            'max_move_next_5'    : '',
            'max_move_next_10'   : '',
        }
        zones.append((i, row))
        last_zone_idx = i

    # ── second pass: lookahead breakout analysis ──────────────────────────────
    for idx, (bar_i, row) in enumerate(zones):
        resist = row['resistance_level']
        supprt = row['support_level']
        zone_close = row['close']

        # look ahead up to 20 bars
        future = df.iloc[bar_i + 1: bar_i + 21]
        if len(future) == 0:
            continue

        broke_out = None
        candles_to = ''
        direction  = ''
        magnitude  = ''

        for j, (fi, frow) in enumerate(future.iterrows()):
            if frow['high'] > resist:
                broke_out = j + 1
                direction = 'Upside Breakout'
                magnitude = round(frow['high'] - resist, 4)
                break
            if frow['low'] < supprt:
                broke_out = j + 1
                direction = 'Downside Breakdown'
                magnitude = round(supprt - frow['low'], 4)
                break

        if broke_out:
            row['candles_to_breakout'] = broke_out
            row['breakout_direction']  = direction
            row['breakout_magnitude']  = magnitude

        next5  = future.iloc[:5]  if len(future) >= 5  else future
        next10 = future.iloc[:10] if len(future) >= 10 else future

        if len(next5) > 0:
            max5 = max(abs(next5['high'].max() - zone_close), abs(next5['low'].min() - zone_close))
            row['max_move_next_5'] = round(max5, 4)
        if len(next10) > 0:
            max10 = max(abs(next10['high'].max() - zone_close), abs(next10['low'].min() - zone_close))
            row['max_move_next_10'] = round(max10, 4)

    return [r for _, r in zones], df

# ── summary stats ─────────────────────────────────────────────────────────────
def build_summary(zones, df, args):
    if not zones:
        return "No contraction zones detected with current parameters."

    zdf = pd.DataFrame(zones)
    total = len(zdf)

    # zone type breakdown
    type_counts = zdf['zone_type'].value_counts().to_dict()
    progressive = zdf['is_progressive_squeeze'].sum()

    # compression stats
    comp_vals = pd.to_numeric(zdf['compression_ratio'], errors='coerce').dropna()

    # breakout stats
    broke_out = zdf[zdf['breakout_direction'] != '']
    upside    = (zdf['breakout_direction'] == 'Upside Breakout').sum()
    downside  = (zdf['breakout_direction'] == 'Downside Breakdown').sum()

    # score stats
    scores = zdf['zone_score']

    lines = [
        "=" * 65,
        "  CONTRACTION ZONE BACKTEST — SUMMARY REPORT",
        "=" * 65,
        "",
        f"  File analyzed         : {args.file}",
        f"  Total candles         : {len(df)}",
        f"  Analysis date         : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "─" * 65,
        "  PARAMETERS USED",
        "─" * 65,
        f"  Lookback period       : {args.lookback} candles",
        f"  Compression threshold : {int(args.threshold * 100)}%",
        f"  Progressive length    : {args.squeeze_len} candles",
        f"  ATR period            : {args.atr_period}",
        f"  Min zone gap          : {args.min_zone_gap} candles",
        "",
        "─" * 65,
        "  ZONE COUNTS",
        "─" * 65,
        f"  Total zones detected  : {total}",
        f"  Progressive squeezes  : {progressive}  ({round(progressive/total*100,1)}%)",
        "",
    ]
    for zone_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {zone_type:<30}: {count}  ({round(count/total*100,1)}%)")

    lines += [
        "",
        "─" * 65,
        "  COMPRESSION STATS",
        "─" * 65,
        f"  Mean compression ratio : {comp_vals.mean():.3f}  ({comp_vals.mean()*100:.1f}% of avg range)",
        f"  Min compression ratio  : {comp_vals.min():.3f}  ({comp_vals.min()*100:.1f}%)",
        f"  Max compression ratio  : {comp_vals.max():.3f}  ({comp_vals.max()*100:.1f}%)",
        f"  Std dev                : {comp_vals.std():.4f}",
        "",
        "─" * 65,
        "  ZONE SCORE STATS",
        "─" * 65,
        f"  Mean score    : {scores.mean():.1f} / 100",
        f"  Highest score : {scores.max():.1f}",
        f"  Lowest score  : {scores.min():.1f}",
        f"  Zones ≥ 70    : {(scores >= 70).sum()}",
        f"  Zones ≥ 50    : {(scores >= 50).sum()}",
        "",
        "─" * 65,
        "  BREAKOUT ANALYSIS  (lookahead ≤ 20 bars)",
        "─" * 65,
        f"  Zones with breakout   : {len(broke_out)}  ({round(len(broke_out)/total*100,1)}%)",
        f"  Upside breakouts      : {upside}",
        f"  Downside breakdowns   : {downside}",
    ]

    if len(broke_out) > 0:
        avg_candles = pd.to_numeric(broke_out['candles_to_breakout'], errors='coerce').mean()
        lines.append(f"  Avg candles to break  : {avg_candles:.1f}")
        mag = pd.to_numeric(broke_out['breakout_magnitude'], errors='coerce')
        lines.append(f"  Avg breakout magnitude: {mag.mean():.4f}")

    # top 5 zones by score
    top5 = zdf.nlargest(5, 'zone_score')[['time', 'close', 'zone_score', 'zone_type', 'compression_pct']]
    lines += [
        "",
        "─" * 65,
        "  TOP 5 ZONES BY SCORE",
        "─" * 65,
    ]
    for _, r in top5.iterrows():
        lines.append(f"  [{r['zone_score']:5.1f}]  {str(r['time'])[:19]:<22}  Close={r['close']}  "
                     f"Comp={r['compression_pct']}  {r['zone_type']}")

    lines += ["", "=" * 65, ""]
    return "\n".join(lines)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Contraction Zone Backtester')
    parser.add_argument('--file',       required=True,       help='Path to OHLC CSV file')
    parser.add_argument('--lookback',   type=int,   default=10,   help='Lookback period (default: 10)')
    parser.add_argument('--threshold',  type=float, default=0.70, help='Compression threshold 0-1 (default: 0.70)')
    parser.add_argument('--squeeze_len',type=int,   default=3,    help='Progressive squeeze candles (default: 3)')
    parser.add_argument('--atr_period', type=int,   default=14,   help='ATR period (default: 14)')
    parser.add_argument('--min_zone_gap',type=int,  default=3,    help='Min bars between zones (default: 3)')
    parser.add_argument('--output',     default='contraction_zones.csv',     help='Output CSV')
    parser.add_argument('--summary',    default='contraction_summary.txt',   help='Summary TXT')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[ERROR] File not found: {args.file}")
        sys.exit(1)

    print(f"\n  Loading {args.file} ...")
    df = load_csv(args.file)
    print(f"  Loaded {len(df)} candles.")

    print(f"  Detecting contraction zones (lookback={args.lookback}, threshold={args.threshold}, squeeze_len={args.squeeze_len}) ...")
    zones, enriched_df = detect_contraction_zones(
        df,
        lookback     = args.lookback,
        threshold    = args.threshold,
        squeeze_len  = args.squeeze_len,
        atr_period   = args.atr_period,
        min_zone_gap = args.min_zone_gap,
    )

    if not zones:
        print("\n  No contraction zones found. Try increasing --threshold or reducing --lookback.\n")
        sys.exit(0)

    print(f"  Found {len(zones)} contraction zones.\n")

    # save detailed CSV
    zdf = pd.DataFrame(zones)
    zdf.to_csv(args.output, index=False)
    print(f"  Detailed zones saved → {args.output}")

    # save summary
    summary = build_summary(zones, df, args)
    with open(args.summary, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  Summary report saved → {args.summary}")

    # print summary to terminal
    print()
    print(summary)

if __name__ == '__main__':
    main()