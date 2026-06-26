"""
run_analysis.py — Run the Adam Grimes market structure system on your own OHLCV data.

USAGE:
    py run_analysis.py                          # uses nifty50.csv by default
    py run_analysis.py mydata.csv               # your own file
    py run_analysis.py mydata.csv --equity 500000 --risk 0.02

YOUR CSV MUST HAVE COLUMNS (rename to match):
    datetime, open, high, low, close, volume

COLUMN NAME ALIASES (auto-detected):
    datetime: Date, Datetime, date, timestamp, time, Date/Time
    open:     Open, OPEN, o
    high:     High, HIGH, h
    low:      Low, LOW, l
    close:    Close, CLOSE, c, Adj Close
    volume:   Volume, VOLUME, vol, Volume(Shares)   (optional — defaults to 0)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'market_structure'))

import pandas as pd
import numpy as np
import argparse


# ─── COLUMN AUTO-DETECTION ───────────────────────────────────────────────────

ALIASES = {
    'datetime': ['datetime', 'date', 'timestamp', 'time', 'date/time', 'dt',
                 'price'],       # yfinance sometimes labels the datetime column "Price"
    'open':     ['open', 'o', 'first'],
    'high':     ['high', 'h', 'max'],
    'low':      ['low', 'l', 'min'],
    'close':    ['close', 'c', 'last', 'adj close', 'adj_close'],
    'volume':   ['volume', 'vol', 'qty', 'volume(shares)', 'shares'],
}

def detect_columns(df: pd.DataFrame) -> dict:
    """Map standardized names to actual column names in the dataframe."""
    lower_cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for standard, aliases in ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                mapping[standard] = lower_cols[alias]
                break
    return mapping


# ─── DATA LOADER ─────────────────────────────────────────────────────────────

def load_ohlcv(path: str) -> pd.DataFrame:
    """
    Load any OHLCV CSV file. Handles:
    - Various column name formats (Open/OPEN/o, etc.)
    - yfinance-style multi-header CSVs (Price/Ticker/Datetime rows)
    - Mixed date formats
    - Files with or without volume
    """
    # Read with first row as header
    df = pd.read_csv(path, low_memory=False)

    # Drop rows where the first column looks like a metadata row
    # (yfinance adds "Ticker" and "Datetime" rows below the actual header)
    first_col = df.columns[0]
    junk_values = {'ticker', 'datetime', 'date', 'nan', 'price', ''}
    mask = df[first_col].astype(str).str.lower().str.strip().isin(junk_values)
    df = df[~mask].reset_index(drop=True)

    # Detect and rename columns
    mapping = detect_columns(df)
    missing = [k for k in ['open', 'high', 'low', 'close'] if k not in mapping]
    if missing:
        print(f"\n[ERROR] Could not find these columns: {missing}")
        print(f"        Your CSV has: {list(df.columns)}")
        print(f"        Rename your columns to: datetime, open, high, low, close, volume")
        sys.exit(1)

    rename_map = {v: k for k, v in mapping.items()}
    df = df.rename(columns=rename_map)

    # Add volume=0 if missing
    if 'volume' not in df.columns:
        df['volume'] = 0

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=False)
    df = df.dropna(subset=['datetime'])

    # Parse numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = df['volume'].fillna(0).astype(int)

    # Clean and sort
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset=['datetime']).reset_index(drop=True)

    return df


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample to a higher timeframe. freq: '1h', '4h', '1D', '1W'"""
    df2 = df.set_index('datetime')
    result = df2[['open', 'high', 'low', 'close', 'volume']].resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open'])
    return result.reset_index()


def detect_timeframe(df: pd.DataFrame) -> str:
    """Guess the timeframe of the data from median bar spacing."""
    if len(df) < 3:
        return 'unknown'
    deltas = df['datetime'].diff().dropna()
    median_minutes = deltas.median().total_seconds() / 60
    if median_minutes < 2:      return '1min'
    elif median_minutes < 8:    return '5min'
    elif median_minutes < 20:   return '15min'
    elif median_minutes < 50:   return '30min'
    elif median_minutes < 90:   return '1h'
    elif median_minutes < 300:  return '4h'
    else:                       return 'daily'


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run Grimes market structure analysis')
    parser.add_argument('csv', nargs='?', default='nifty50.csv', help='Path to your OHLCV CSV file')
    parser.add_argument('--equity', type=float, default=500_000, help='Account equity (default: 500000)')
    parser.add_argument('--risk',   type=float, default=0.02,    help='Risk per trade as fraction (default: 0.02 = 2%%)')
    parser.add_argument('--ttf',    type=int,   default=500,     help='Number of recent bars to analyze on TTF (default: 500)')
    args = parser.parse_args()

    print("=" * 70)
    print("Market Structure Analysis — Adam Grimes Framework")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        print(f"\n[ERROR] File not found: {args.csv}")
        sys.exit(1)

    print(f"\n[LOAD] Reading {args.csv}...")
    df_raw = load_ohlcv(args.csv)

    tf = detect_timeframe(df_raw)
    print(f"       Rows          : {len(df_raw):,}")
    print(f"       Date range    : {df_raw['datetime'].iloc[0].date()} to {df_raw['datetime'].iloc[-1].date()}")
    print(f"       Price range   : {df_raw['low'].min():.2f} to {df_raw['high'].max():.2f}")
    print(f"       Detected TF   : {tf}")

    # ── Build timeframes ──────────────────────────────────────────────────────
    # TTF = trading timeframe (the timeframe you trade on)
    # HTF = higher timeframe (for bias, 4-12x the TTF)
    # LTF = lower timeframe (for entry timing, 1/4 of TTF)

    tf_map = {
        '1min':  ('5min',  '15min'),
        '5min':  ('1h',    '1h'),
        '15min': ('4h',    '4h'),
        '30min': ('4h',    '4h'),
        '1h':    ('1D',    '1D'),
        '4h':    ('1W',    '1W'),
        'daily': ('1W',    '1W'),
    }
    htf_freq, ltf_freq = tf_map.get(tf, ('1D', '1D'))

    df_htf = resample(df_raw, htf_freq)
    df_ttf = df_raw.tail(args.ttf).reset_index(drop=True)
    df_ltf = df_raw.tail(100).reset_index(drop=True)

    print(f"       TTF bars      : {len(df_ttf)} (last {args.ttf} bars of {tf})")
    print(f"       HTF ({htf_freq:>3s}) bars : {len(df_htf)}")

    # Previous day OHLC for PDHL zones (use data from lookback window)
    prev_window = df_raw.tail(100)
    prev_day = {
        'open':  float(prev_window['open'].iloc[0]),
        'high':  float(prev_window['high'].max()),
        'low':   float(prev_window['low'].min()),
        'close': float(prev_window['close'].iloc[-1]),
    }

    # ── Run scanner ───────────────────────────────────────────────────────────
    from scanner import MarketScanner
    scanner = MarketScanner(account_equity=args.equity, risk_fraction=args.risk)

    print(f"\n[SCAN] Running analysis (equity={args.equity:,.0f}, risk={args.risk:.1%})...")
    result = scanner.analyze(
        df_ttf=df_ttf,
        df_htf=df_htf.tail(200).reset_index(drop=True),
        df_ltf=df_ltf,
        prev_day_ohlc=prev_day
    )

    # ── Print results ─────────────────────────────────────────────────────────
    sep = "-" * 70

    print(f"\n{sep}")
    print("MARKET REGIME")
    print(sep)
    reg = result.regime
    direction_str = '+1 UP' if reg.trend_direction == 1 else ('-1 DOWN' if reg.trend_direction == -1 else '0 FLAT')
    print(f"  State          : {reg.state.name}")
    print(f"  Direction      : {direction_str}  (confidence {reg.confidence:.0%})")
    print(f"  Strength       : {reg.strength_score:.1f}/10")
    print(f"  In Pullback    : {reg.in_pullback}  type={reg.pullback_type.name}  depth={reg.pullback_depth_pct:.1%}")
    if reg.ab_cd_target:
        print(f"  AB=CD Target   : {reg.ab_cd_target:.2f}")
    if reg.key_level_high:
        print(f"  Key High       : {reg.key_level_high:.2f}")
    if reg.key_level_low:
        print(f"  Key Low        : {reg.key_level_low:.2f}")
    for w in reg.warning_signs:
        print(f"  [WARN] {w}")

    print(f"\n{sep}")
    print("TREND HEALTH")
    print(sep)
    h = result.health
    print(f"  Impulse Ratio  : {h.impulse_ratio:.2f}  (>1.5 = healthy, <1.0 = weak)")
    print(f"  Velocity       : {h.velocity_trend:+.2f}  (+1=accelerating, -1=decelerating)")
    print(f"  Avg Pullback   : {h.avg_pullback_depth:.1%} of prior impulse")
    print(f"  Healthy        : {h.is_healthy}  |  Parabolic: {h.is_parabolic}")
    for w in h.warnings:
        print(f"  [WARN] {w}")

    print(f"\n{sep}")
    print("TREND CHANGE SCORE")
    print(sep)
    cs = result.change_score
    print(f"  Score          : {cs.total_score}/9  ({cs.probability:.0%} probability of change)")
    print(f"  Change Likely  : {cs.change_likely}  (triggers at score >= 4)")
    if cs.signals:
        for s in cs.signals:
            print(f"  [SIGNAL] {s}")

    print(f"\n{sep}")
    print(f"S/R ZONES  ({len(result.sr_zones)} total)")
    print(sep)
    sig = [z for z in result.sr_zones if z.is_significant]
    print(f"  Statistically valid (bounce>=60%, touches>=3): {len(sig)}")
    current_price = float(df_ttf['close'].iloc[-1])
    for z in sorted(sig, key=lambda x: abs(x.zone_center - current_price))[:10]:
        dist = ((z.zone_center - current_price) / current_price) * 100
        print(f"  [{z.zone_type.name:10s}] {z.zone_bottom:.2f}-{z.zone_top:.2f}  "
              f"dist={dist:+.2f}%  touches={z.touch_count}  "
              f"bounce={z.bounce_rate:.0%}  {z.state.name}")

    print(f"\n{sep}")
    print("WYCKOFF SPRINGS & UPTHRUSTS")
    print(sep)
    print(f"  Springs   : {len(result.springs)}")
    print(f"  Upthrusts : {len(result.upthrusts)}")
    for s in result.springs:
        print(f"    SPRING   bar={s['bar_index']}  probe_low={s['probe_low']:.2f}  "
              f"close={s['close']:.2f}  entry_above={s['entry_trigger']:.2f}  strength={s['strength']:.2f}")
    for u in result.upthrusts:
        print(f"    UPTHRUST bar={u['bar_index']}  probe_high={u['probe_high']:.2f}  "
              f"close={u['close']:.2f}  entry_below={u['entry_trigger']:.2f}  strength={u['strength']:.2f}")

    print(f"\n{sep}")
    print(f"CLIMAX / EXHAUSTION SIGNALS  ({len(result.climax_signals)})")
    print(sep)
    for c in result.climax_signals[-5:]:
        print(f"  Bar {c.bar_index:4d}  [{c.climax_type:25s}]  "
              f"strength={c.strength:.2f}  range/ATR={c.bar_range_atr_ratio:.2f}  "
              f"outside_keltner={c.outside_keltner}")

    print(f"\n{sep}")
    print("MTF ALIGNMENT")
    print(sep)
    mtf = result.mtf_alignment
    print(f"  Score          : {mtf.alignment_score}/3")
    print(f"  Direction Bias : {mtf.recommended_direction}")
    print(f"  Position Size  : {mtf.size_fraction:.0%} of normal")

    print(f"\n{sep}")
    print(f"TRADE SIGNALS  ({len(result.signals)} qualified)")
    print(sep)
    if result.signals:
        for sig_item in sorted(result.signals, key=lambda s: s.confidence, reverse=True):
            print(f"\n  [{sig_item.signal_type.upper():22s}] {sig_item.direction.upper():<5s}  "
                  f"confidence={sig_item.confidence:.0%}  bar={sig_item.bar_index}")
            print(f"    Entry  : {sig_item.entry_price:.2f}")
            print(f"    Stop   : {sig_item.stop_price:.2f}  "
                  f"(risk/unit={sig_item.risk_per_unit:.2f})")
            print(f"    Target1: {sig_item.target_1:.2f}  ({sig_item.rr_1:.1f}R)")
            print(f"    Target2: {sig_item.target_2:.2f}  ({sig_item.rr_2:.1f}R)")
            print(f"    Target3: {sig_item.target_3:.2f}  ({sig_item.rr_3:.1f}R)")
            print(f"    Size   : {sig_item.position_size:.0f} units  "
                  f"(risk={sig_item.risk_fraction:.1%} of equity)")
            if sig_item.notes:
                for note in sig_item.notes:
                    print(f"    Note   : {note}")
    else:
        print("  No signals at current bar (try adjusting --ttf or check market conditions)")

    print(f"\n{sep}")
    print("CURRENT INDICATORS (last bar)")
    print(sep)
    from indicators import add_all_indicators as _aii
    _df = _aii(df_ttf.copy())
    close_now = _df['close'].iloc[-1]
    print(f"  Close          : {close_now:.2f}")
    print(f"  ATR(14)        : {_df['atr'].iloc[-1]:.2f}  "
          f"({_df['atr'].iloc[-1]/close_now*100:.2f}% of price)")
    print(f"  EMA 20/50/200  : {_df['ema_20'].iloc[-1]:.2f} / "
          f"{_df['ema_50'].iloc[-1]:.2f} / {_df['ema_200'].iloc[-1]:.2f}")
    print(f"  Keltner        : {_df['kc_lower'].iloc[-1]:.2f} — "
          f"{_df['kc_upper'].iloc[-1]:.2f}  (2.25x ATR, book exact)")
    print(f"  MACD fast/sig  : {_df['macd_line'].iloc[-1]:.4f} / "
          f"{_df['macd_signal'].iloc[-1]:.4f}  (SMA 3-10-16, book exact)")

    dd = result.drawdown_state
    if dd:
        print(f"\n{sep}")
        print("RISK STATE")
        print(sep)
        print(f"  Drawdown       : {dd.drawdown_pct:.1%} from peak")
        print(f"  Action         : {dd.action}")
        print(f"  Size Mult      : {dd.size_multiplier:.0%}")

    print(f"\n{'=' * 70}")
    print(result.summary())
    print("=" * 70)


if __name__ == '__main__':
    main()
