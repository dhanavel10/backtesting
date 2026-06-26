"""
test_system.py — Validate the market_structure pipeline with real NIFTY50 data.
Run from: d:\backtest.ai\
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'market_structure'))

import pandas as pd
import numpy as np


# ─── 1. LOAD & CLEAN DATA ─────────────────────────────────────────────────────
def load_nifty(path='nifty50.csv'):
    df = pd.read_csv(path, skiprows=2)
    df.columns = ['datetime', 'close', 'high', 'low', 'open', 'volume']
    df = df[df['close'].notna() & (df['close'] != 'Close')]
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df = df.dropna(subset=['datetime'])
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df


def resample_to_htf(df_5m, freq='1h'):
    df = df_5m.set_index('datetime')
    ohlcv = df[['open', 'high', 'low', 'close', 'volume']].resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open'])
    return ohlcv.reset_index()


# ─── 2. RUN SCANNER ──────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("NIFTY50 Market Structure Analysis - Adam Grimes Framework")
    print("=" * 70)

    df_5m = load_nifty('nifty50.csv')
    print(f"\n[DATA] 5-min bars loaded: {len(df_5m)} rows")
    print(f"       Range: {df_5m['datetime'].iloc[0]} to {df_5m['datetime'].iloc[-1]}")
    print(f"       Price range: {df_5m['low'].min():.0f} to {df_5m['high'].max():.0f}")

    df_1h = resample_to_htf(df_5m, '1h')
    print(f"       1h bars: {len(df_1h)}")

    df_ttf = df_5m.tail(500).reset_index(drop=True)
    df_ltf = df_5m.tail(100).reset_index(drop=True)

    prev_day = {
        'open':  float(df_5m['open'].iloc[-100]),
        'high':  float(df_5m['high'].iloc[-100:].max()),
        'low':   float(df_5m['low'].iloc[-100:].min()),
        'close': float(df_5m['close'].iloc[-1]),
    }

    from scanner import MarketScanner
    scanner = MarketScanner(account_equity=500_000, risk_fraction=0.02)

    print("\n[SCAN] Running full 14-step pipeline...")
    result = scanner.analyze(
        df_ttf=df_ttf,
        df_htf=df_1h.tail(200).reset_index(drop=True),
        df_ltf=df_ltf,
        prev_day_ohlc=prev_day
    )

    # ─── 3. RESULTS ──────────────────────────────────────────────────────────
    sep = "-" * 70
    print(f"\n{sep}")
    print("MARKET REGIME")
    print(sep)
    reg = result.regime
    print(f"  State            : {reg.state.name}")
    print(f"  Trend Direction  : {'+1 UP' if reg.trend_direction == 1 else '-1 DOWN' if reg.trend_direction == -1 else '0 FLAT'}")
    print(f"  Strength         : {reg.strength_score:.2f}/10")
    print(f"  Confidence       : {reg.confidence:.0%}")
    print(f"  Strengthening    : {reg.is_strengthening}  |  Weakening: {reg.is_weakening}")
    print(f"  In Pullback      : {reg.in_pullback}  ({reg.pullback_type.name})  depth={reg.pullback_depth_pct:.1%}")
    if reg.ab_cd_target:
        print(f"  AB=CD Target     : {reg.ab_cd_target:.2f}")
    if reg.key_level_high:
        print(f"  Key Level High   : {reg.key_level_high:.2f}")
    if reg.key_level_low:
        print(f"  Key Level Low    : {reg.key_level_low:.2f}")
    if reg.warning_signs:
        for w in reg.warning_signs:
            print(f"  [WARN] {w}")

    print(f"\n{sep}")
    print("TREND HEALTH")
    print(sep)
    h = result.health
    if h:
        print(f"  Direction        : {'+1 UP' if h.direction == 1 else '-1 DOWN' if h.direction == -1 else '0 FLAT'}")
        print(f"  Strength Score   : {h.strength_score:.2f}/10")
        print(f"  Impulse Ratio    : {h.impulse_ratio:.2f}  (>1.5 = healthy)")
        print(f"  Velocity Trend   : {h.velocity_trend:+.2f}  (+1=accel, -1=decel)")
        print(f"  Avg Pullback     : {h.avg_pullback_depth:.1%} of impulse")
        print(f"  Avg Impulse Bars : {h.avg_impulse_bars:.1f}  |  Avg Pullback Bars: {h.avg_pullback_bars:.1f}")
        print(f"  Healthy          : {h.is_healthy}  |  Parabolic: {h.is_parabolic}")
        if h.warnings:
            for w in h.warnings:
                print(f"  [WARN] {w}")

    print(f"\n{sep}")
    print("TREND CHANGE DETECTION")
    print(sep)
    cs = result.change_score
    if cs:
        print(f"  Score            : {cs.total_score}/9")
        print(f"  Probability      : {cs.probability:.0%}")
        print(f"  Change Likely    : {cs.change_likely}")
        if cs.last_safe_entry:
            print(f"  Last Safe Entry  : {cs.last_safe_entry:.2f}")
        if cs.signals:
            print(f"  Signals fired    : {', '.join(cs.signals)}")

    print(f"\n{sep}")
    print(f"S/R ZONES  ({len(result.sr_zones)} total)")
    print(sep)
    significant = [z for z in result.sr_zones if z.is_significant]
    print(f"  Statistically significant (bounce>=60%, touches>=3): {len(significant)}")
    for z in sorted(significant, key=lambda x: x.strength_score, reverse=True)[:8]:
        print(f"  [{z.zone_type.name:10s}] {z.zone_bottom:.0f}-{z.zone_top:.0f}  "
              f"strength={z.strength_score:.2f}  touches={z.touch_count}  "
              f"bounce={z.bounce_rate:.0%}  state={z.state.name}")

    print(f"\n{sep}")
    print("SPRINGS & UPTHRUSTS  (Wyckoff)")
    print(sep)
    print(f"  Springs   : {len(result.springs)}")
    print(f"  Upthrusts : {len(result.upthrusts)}")
    for s in result.springs[:3]:
        print(f"    Spring   bar={s.get('bar_index','?')}  probe_low={s.get('probe_low',0):.2f}  close={s.get('close',0):.2f}  strength={s.get('strength',0):.2f}")
    for u in result.upthrusts[:3]:
        print(f"    Upthrust bar={u.get('bar_index','?')}  probe_high={u.get('probe_high',0):.2f}  close={u.get('close',0):.2f}  strength={u.get('strength',0):.2f}")

    print(f"\n{sep}")
    print(f"CLIMAX SIGNALS  ({len(result.climax_signals)})")
    print(sep)
    for c in result.climax_signals[-5:]:
        print(f"  Bar {c.bar_index:4d}  [{c.climax_type:25s}]  strength={c.strength:.2f}  "
              f"range/ATR={c.bar_range_atr_ratio:.2f}  outside_keltner={c.outside_keltner}")

    print(f"\n{sep}")
    print("KELTNER CHANNEL STATE")
    print(sep)
    kc = result.keltner_state
    if kc:
        print(f"  Position (0-1)   : {kc.get('current_position', 'N/A'):.3f}  (>1 = above upper, <0 = below lower)")
        print(f"  Overextended Up  : {kc.get('is_overextended_up', False)}  ({kc.get('consecutive_bars_above', 0)} consecutive bars)")
        print(f"  Overextended Dn  : {kc.get('is_overextended_down', False)}  ({kc.get('consecutive_bars_below', 0)} consecutive bars)")
        print(f"  Exhaustion Up/Dn : {kc.get('exhaustion_up', False)} / {kc.get('exhaustion_down', False)}")
        print(f"  Hugging Upper/Lwr: {kc.get('trend_hugging_upper', False)} / {kc.get('trend_hugging_lower', False)}")

    print(f"\n{sep}")
    print("MTF ALIGNMENT")
    print(sep)
    mtf = result.mtf_alignment
    if mtf:
        print(f"  Alignment Score  : {mtf.alignment_score}/3")
        print(f"  Recommended Dir  : {mtf.recommended_direction}")
        print(f"  Size Fraction    : {mtf.size_fraction:.0%}")

    print(f"\n{sep}")
    print(f"TRADE SIGNALS  ({len(result.signals)} qualified, min confidence 45%, min RR 1.0)")
    print(sep)
    if result.signals:
        for sig in sorted(result.signals, key=lambda s: s.confidence, reverse=True):
            print(f"\n  [{sig.signal_type.upper():22s}] {sig.direction.upper():<5s}  confidence={sig.confidence:.0%}")
            print(f"    Entry / Stop     : {sig.entry_price:.2f} / {sig.stop_price:.2f}  (risk/unit={sig.risk_per_unit:.2f})")
            print(f"    T1 / T2 / T3     : {sig.target_1:.2f} / {sig.target_2:.2f} / {sig.target_3:.2f}")
            print(f"    RR  1 / 2 / 3    : {sig.rr_1:.2f}R / {sig.rr_2:.2f}R / {sig.rr_3:.2f}R")
            print(f"    Position Size    : {sig.position_size:.0f} units  (risk={sig.risk_fraction:.1%})")
            print(f"    Market State     : {sig.market_state.name if sig.market_state else 'N/A'}")
            if sig.notes:
                print(f"    Notes            : {sig.notes}")
    else:
        print("  No signals above confidence threshold at current bar.")
        print("  (Increase bar count or check market conditions)")

    print(f"\n{sep}")
    print("DRAWDOWN STATE")
    print(sep)
    dd = result.drawdown_state
    if dd:
        print(f"  Action           : {dd.action}")
        print(f"  DD from Peak     : {dd.drawdown_pct:.1%}")
        print(f"  Size Multiplier  : {dd.size_multiplier:.0%}")
    else:
        print("  No drawdown history yet (fresh account)")

    print(f"\n{sep}")
    print("INDICATOR SPOT CHECK (last bar)")
    print(sep)
    # Re-run indicators on TTF to read columns (scanner works on a copy)
    from indicators import add_all_indicators as _aii
    _df = _aii(df_ttf.copy())
    print(f"  Close            : {_df['close'].iloc[-1]:.2f}")
    for col, label in [('atr','ATR(14)'), ('kc_lower','Keltner Low'),
                        ('kc_upper','Keltner High'), ('kc_mid','Keltner Mid'),
                        ('macd_line','MACD Line'), ('macd_signal','MACD Signal'),
                        ('ema_20','EMA(20)'), ('ema_50','EMA(50)'), ('ema_200','EMA(200)'),
                        ('volatility_state','Volatility State')]:
        if col in _df.columns:
            val = _df[col].iloc[-1]
            if isinstance(val, str):
                print(f"  {label:<18s}: {val}")
            else:
                print(f"  {label:<18s}: {val:.4f}")

    if result.notes:
        print(f"\n{sep}")
        print("SYSTEM NOTES")
        print(sep)
        for n in result.notes:
            print(f"  {n}")

    # ─── 4. SUMMARY ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(result.summary())
    print(f"{'=' * 70}")
    print("Analysis complete.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
