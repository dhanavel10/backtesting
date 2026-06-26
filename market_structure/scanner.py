"""
scanner.py — Main orchestration: runs the complete market structure analysis pipeline.
Based on: The Art and Science of Technical Analysis (Adam Grimes)

Usage:
    from scanner import MarketScanner
    scanner = MarketScanner(account_equity=500_000, risk_fraction=0.02)
    result = scanner.analyze(df_daily, df_4h, df_1h)
    for signal in result.signals:
        print(signal)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from indicators import add_all_indicators
from pivot_detector import detect_pivots_from_df, swing_metrics, Pivot, ZigzagPoint
from market_structure import detect_market_regime, MarketRegime, MarketState
from sr_zones import (build_sr_zones, detect_spring, detect_upthrust,
                       SRZone, pdhl_zones)
from channel_detector import (fit_channel, detect_parabolic_climax,
                               keltner_analysis, analyze_rate_of_trend,
                               ClimaxSignal)
from trend_analyzer import (assess_trend_health, detect_trend_change,
                             mtf_trend_alignment, score_pullback_quality,
                             MTFAlignment, TrendHealth, TrendChangeScore)
from signal_engine import scan_all_signals, TradeSignal
from risk_manager import (fixed_fractional_size, check_drawdown,
                           trade_statistics, kelly_criterion,
                           DrawdownState, PositionSize)


@dataclass
class AnalysisResult:
    # Core market state
    regime: MarketRegime
    health: TrendHealth
    change_score: TrendChangeScore
    mtf_alignment: MTFAlignment

    # Structure
    sr_zones: List[SRZone]
    springs: List[dict]
    upthrusts: List[dict]
    climax_signals: List[ClimaxSignal]
    keltner_state: dict

    # Signals
    signals: List[TradeSignal]

    # Risk context
    drawdown_state: Optional[DrawdownState]
    notes: List[str]

    def best_signal(self) -> Optional[TradeSignal]:
        return self.signals[0] if self.signals else None

    def summary(self) -> str:
        lines = [
            f"Market State: {self.regime.state.value} (confidence={self.regime.confidence:.0%})",
            f"Trend Direction: {'+1 UP' if self.regime.trend_direction == 1 else '-1 DOWN' if self.regime.trend_direction == -1 else '0 FLAT'}",
            f"Strength: {self.regime.strength_score:.1f}/10",
            f"MTF Alignment: {self.mtf_alignment.alignment_score}/3 → size factor {self.mtf_alignment.size_fraction:.0%}",
            f"Change Score: {self.change_score.total_score} ({self.change_score.probability:.0%} probability)",
            f"S/R Zones: {len(self.sr_zones)} ({len([z for z in self.sr_zones if z.is_significant])} significant)",
            f"Signals: {len(self.signals)} valid",
        ]
        if self.signals:
            best = self.signals[0]
            lines.append(f"\nBest Signal: {best.signal_type.upper()} {best.direction.upper()} "
                        f"@ {best.entry_price:.2f} | Stop: {best.stop_price:.2f} "
                        f"| T1: {best.target_1:.2f} | Conf: {best.confidence:.0%}")
        if self.regime.warning_signs:
            lines.append(f"\nWarnings: {'; '.join(self.regime.warning_signs)}")
        return "\n".join(lines)


class MarketScanner:
    """
    Main analysis engine. Feed it OHLCV DataFrames and get complete
    market structure analysis + trade signals.

    DataFrame format required:
        columns: open, high, low, close, volume (volume optional)
        index: datetime
    """

    def __init__(self,
                 account_equity: float = 100_000,
                 risk_fraction: float = 0.02,
                 pivot_n: int = 3,
                 keltner_period: int = 20,
                 keltner_mult: float = 2.25,
                 atr_period: int = 14,
                 min_signal_confidence: float = 0.45):
        self.account_equity = account_equity
        self.risk_fraction = risk_fraction
        self.pivot_n = pivot_n
        self.keltner_period = keltner_period
        self.keltner_mult = keltner_mult
        self.atr_period = atr_period
        self.min_signal_confidence = min_signal_confidence

        # State tracking
        self.peak_equity = account_equity
        self.trade_log: List[dict] = []
        self.r_multiples: List[float] = []

    # ─── MAIN ANALYSIS ───────────────────────────────────────────────────────

    def analyze(self, df_ttf: pd.DataFrame,
                df_htf: Optional[pd.DataFrame] = None,
                df_ltf: Optional[pd.DataFrame] = None,
                prev_day_ohlc: Optional[dict] = None) -> AnalysisResult:
        """
        Run the full analysis pipeline on one or multiple timeframes.

        Args:
            df_ttf:         Trading Timeframe OHLCV DataFrame (required)
            df_htf:         Higher Timeframe DataFrame (optional, for HTF bias)
            df_ltf:         Lower Timeframe DataFrame (optional, for entry timing)
            prev_day_ohlc:  dict with 'high', 'low', 'close' for previous session
        """
        notes = []

        # ── Step 1: Indicators ───────────────────────────────────────────────
        df = add_all_indicators(df_ttf.copy(), self.atr_period,
                                self.keltner_period, self.keltner_mult)

        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        atr_v = df['atr'].values
        kc_u = df['kc_upper'].values
        kc_l = df['kc_lower'].values
        kc_m = df['kc_mid'].values
        macd_div_v = df['macd_divergence'].values
        vol_state_v = df['volatility_state'].values
        bar_index = len(c) - 1

        # ── Step 2: Pivots ────────────────────────────────────────────────────
        pivots, zigzag = detect_pivots_from_df(df, n=self.pivot_n)

        # ── Step 3: Market Regime ─────────────────────────────────────────────
        regime = detect_market_regime(
            zigzag, c, h, l, atr_v, kc_u, kc_l, macd_div_v)

        # ── Step 4: Trend Health ──────────────────────────────────────────────
        health = assess_trend_health(zigzag, regime.trend_direction)

        # ── Step 5: Rate of Trend ─────────────────────────────────────────────
        rot = analyze_rate_of_trend(zigzag, regime.trend_direction)
        if rot.get('is_parabolic'):
            notes.append("PARABOLIC ACCELERATION: climax risk is high")

        # ── Step 6: Climax Detection ──────────────────────────────────────────
        climax_signals = detect_parabolic_climax(
            h, l, c, atr_v, kc_u, kc_l, macd_div_v)

        # ── Step 7: Trend Change Score ────────────────────────────────────────
        change_score = detect_trend_change(
            zigzag, c, h, l, atr_v, macd_div_v, climax_signals,
            regime.trend_direction)

        if change_score.change_likely:
            notes.append(f"TREND CHANGE LIKELY: score={change_score.total_score}")

        # ── Step 8: S/R Zones ─────────────────────────────────────────────────
        sr_zones = build_sr_zones(pivots, atr_v, h, l, c)

        # Add previous day levels if provided
        if prev_day_ohlc:
            pdhl = pdhl_zones(
                prev_day_ohlc['high'], prev_day_ohlc['low'],
                prev_day_ohlc['close'],
                float(np.nanmean(atr_v[-5:])) if len(atr_v) >= 5 else 1.0
            )
            sr_zones = pdhl + sr_zones

        # ── Step 9: Springs / Upthrusts ──────────────────────────────────────
        springs = detect_spring(h, l, c, sr_zones, atr_v, lookback=3)
        upthrusts = detect_upthrust(h, l, c, sr_zones, atr_v, lookback=3)

        # ── Step 10: Keltner State ────────────────────────────────────────────
        kc_state = keltner_analysis(c, kc_u, kc_l, kc_m)
        if kc_state.get('exhaustion_up'):
            notes.append("Keltner exhaustion: 2+ bars above upper band")
        if kc_state.get('exhaustion_down'):
            notes.append("Keltner exhaustion: 2+ bars below lower band")

        # ── Step 11: HTF / LTF States ─────────────────────────────────────────
        htf_state = self._get_htf_state(df_htf) if df_htf is not None else regime.state
        ltf_momentum = self._get_ltf_momentum(df_ltf) if df_ltf is not None else 0

        # ── Step 12: MTF Alignment ────────────────────────────────────────────
        if df_ltf is not None:
            ltf_pivots, ltf_zigzag = detect_pivots_from_df(
                add_all_indicators(df_ltf.copy(), self.atr_period), n=self.pivot_n)
            ltf_atr = add_all_indicators(df_ltf.copy(), self.atr_period)['atr'].values
            ltf_regime = detect_market_regime(
                ltf_zigzag, df_ltf['close'].values, df_ltf['high'].values,
                df_ltf['low'].values, ltf_atr,
                add_all_indicators(df_ltf.copy())['kc_upper'].values,
                add_all_indicators(df_ltf.copy())['kc_lower'].values,
                np.zeros(len(df_ltf)))
            ltf_ms = ltf_regime.state
        else:
            ltf_ms = regime.state

        mtf_align = mtf_trend_alignment(htf_state, regime.state, ltf_ms)

        # ── Step 13: Signals ──────────────────────────────────────────────────
        signals = scan_all_signals(
            regime=regime,
            health=health,
            change_score=change_score,
            high=h, low=l, close=c, atr_vals=atr_v,
            zones=sr_zones,
            springs=springs,
            upthrusts=upthrusts,
            climax_signals=climax_signals,
            htf_state=htf_state,
            ltf_momentum=ltf_momentum,
            macd_div=macd_div_v,
            vol_state=vol_state_v,
            mtf_alignment_score=mtf_align.alignment_score,
            bar_index=bar_index,
            min_confidence=self.min_signal_confidence
        )

        # ── Step 14: Size signals ─────────────────────────────────────────────
        dd_state = check_drawdown(self.account_equity, self.peak_equity, self.account_equity)
        for sig in signals:
            adjusted_rf = sig.risk_fraction * dd_state.size_multiplier * mtf_align.size_fraction
            pos = fixed_fractional_size(
                self.account_equity, adjusted_rf, sig.entry_price, sig.stop_price)
            sig.position_size = pos.shares
            sig.risk_fraction = adjusted_rf

        # Filter: regime TRANSITION = no trades
        if regime.state == MarketState.TRANSITION:
            notes.append("Market in TRANSITION — no trades recommended")
            signals = []

        return AnalysisResult(
            regime=regime,
            health=health,
            change_score=change_score,
            mtf_alignment=mtf_align,
            sr_zones=sr_zones,
            springs=springs,
            upthrusts=upthrusts,
            climax_signals=climax_signals,
            keltner_state=kc_state,
            signals=signals,
            drawdown_state=dd_state,
            notes=notes
        )

    # ─── HELPERS ─────────────────────────────────────────────────────────────

    def _get_htf_state(self, df_htf: pd.DataFrame) -> MarketState:
        """Compute market state for the higher timeframe."""
        try:
            df_h = add_all_indicators(df_htf.copy(), self.atr_period,
                                       self.keltner_period, self.keltner_mult)
            piv_h, zz_h = detect_pivots_from_df(df_h, n=self.pivot_n)
            atr_h = df_h['atr'].values
            regime_h = detect_market_regime(
                zz_h, df_h['close'].values, df_h['high'].values,
                df_h['low'].values, atr_h,
                df_h['kc_upper'].values, df_h['kc_lower'].values,
                df_h['macd_divergence'].values)
            return regime_h.state
        except Exception:
            return MarketState.RANGE

    def _get_ltf_momentum(self, df_ltf: pd.DataFrame) -> int:
        """Get simple momentum direction from lower timeframe (-1, 0, +1)."""
        try:
            if len(df_ltf) < 5:
                return 0
            closes = df_ltf['close'].values[-5:]
            slope = np.polyfit(np.arange(5), closes, 1)[0]
            atr_est = float(np.nanmean((df_ltf['high'].values - df_ltf['low'].values)[-10:]))
            if atr_est == 0:
                return 0
            if slope > 0.1 * atr_est:
                return 1
            elif slope < -0.1 * atr_est:
                return -1
            return 0
        except Exception:
            return 0

    # ─── TRADE LOGGING ───────────────────────────────────────────────────────

    def log_trade(self, signal: TradeSignal, exit_price: float,
                  exit_reason: str = 'manual') -> dict:
        """Record a completed trade and update statistics."""
        from risk_manager import r_multiple as calc_r
        r = calc_r(signal.entry_price, exit_price, signal.stop_price,
                   signal.direction)
        pnl = (exit_price - signal.entry_price if signal.direction == 'long'
               else signal.entry_price - exit_price) * signal.position_size

        self.account_equity += pnl
        self.peak_equity = max(self.peak_equity, self.account_equity)
        self.r_multiples.append(r)

        trade = {
            'signal_type': signal.signal_type,
            'direction': signal.direction,
            'entry': signal.entry_price,
            'stop': signal.stop_price,
            'exit': exit_price,
            'exit_reason': exit_reason,
            'r_multiple': r,
            'pnl': pnl,
            'confidence': signal.confidence,
            'market_state': signal.market_state.value,
        }
        self.trade_log.append(trade)
        return trade

    def performance_report(self) -> dict:
        """Generate performance report from trade log."""
        if not self.r_multiples:
            return {'message': 'No completed trades yet'}

        stats = trade_statistics(self.r_multiples)
        kelly = kelly_criterion(
            stats.get('win_rate', 0.5),
            stats.get('avg_win_r', 1.5),
            stats.get('avg_loss_r', 1.0)
        )
        dd = check_drawdown(self.account_equity, self.peak_equity, self.account_equity)

        return {
            **stats,
            'kelly_full': kelly.full_kelly,
            'kelly_half': kelly.half_kelly,
            'kelly_quarter': kelly.quarter_kelly,
            'current_equity': self.account_equity,
            'peak_equity': self.peak_equity,
            'drawdown_pct': dd.drawdown_pct,
            'drawdown_action': dd.action,
            'recommended_risk_fraction': min(kelly.half_kelly, 0.05),
        }


# ─── STANDALONE FUNCTIONS ─────────────────────────────────────────────────────

def quick_scan(df: pd.DataFrame, account_equity: float = 100_000,
               risk_fraction: float = 0.02) -> AnalysisResult:
    """
    One-shot analysis on a single timeframe DataFrame.
    For quick scanning without multi-timeframe context.
    """
    scanner = MarketScanner(account_equity=account_equity,
                            risk_fraction=risk_fraction)
    return scanner.analyze(df)


def batch_scan(instruments: Dict[str, pd.DataFrame],
               account_equity: float = 100_000,
               risk_fraction: float = 0.02,
               min_confidence: float = 0.50) -> pd.DataFrame:
    """
    Scan multiple instruments and return a ranked signal table.

    Args:
        instruments: dict of {symbol: DataFrame}
        account_equity: trading account size
        risk_fraction: per-trade risk
        min_confidence: minimum confidence to include in output

    Returns:
        DataFrame with best signal per instrument, sorted by confidence
    """
    scanner = MarketScanner(account_equity=account_equity,
                            risk_fraction=risk_fraction,
                            min_signal_confidence=min_confidence)
    rows = []
    for symbol, df in instruments.items():
        try:
            result = scanner.analyze(df)
            if result.signals:
                best = result.signals[0]
                rows.append({
                    'symbol': symbol,
                    'signal_type': best.signal_type,
                    'direction': best.direction,
                    'entry': best.entry_price,
                    'stop': best.stop_price,
                    'target_1': best.target_1,
                    'target_2': best.target_2,
                    'rr_1': round(best.rr_1, 2),
                    'rr_2': round(best.rr_2, 2),
                    'confidence': round(best.confidence, 3),
                    'position_size': best.position_size,
                    'market_state': result.regime.state.value,
                    'trend_strength': round(result.regime.strength_score, 1),
                    'mtf_score': result.mtf_alignment.alignment_score,
                    'notes': '; '.join(best.notes[:2]),
                })
        except Exception as e:
            rows.append({'symbol': symbol, 'error': str(e)})

    if not rows:
        return pd.DataFrame()

    df_signals = pd.DataFrame([r for r in rows if 'confidence' in r])
    if not df_signals.empty:
        df_signals.sort_values('confidence', ascending=False, inplace=True)
        df_signals.reset_index(drop=True, inplace=True)
    return df_signals


# ─── EXAMPLE USAGE ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Example: analyze a DataFrame loaded from CSV
    # df = pd.read_csv('nifty50.csv', parse_dates=['date'], index_col='date')
    # df.columns = ['open', 'high', 'low', 'close', 'volume']

    # Create synthetic test data for demonstration
    np.random.seed(42)
    n = 500
    price = 18000.0
    rows_demo = []
    for i in range(n):
        # Simple random walk with slight upward drift
        chg = np.random.normal(0.0003, 0.012) * price
        open_ = price
        close_ = price + chg
        high_ = max(open_, close_) + abs(np.random.normal(0, 0.004)) * price
        low_ = min(open_, close_) - abs(np.random.normal(0, 0.004)) * price
        rows_demo.append({'open': open_, 'high': high_, 'low': low_,
                          'close': close_, 'volume': int(np.random.uniform(1e6, 5e6))})
        price = close_

    df_demo = pd.DataFrame(rows_demo)
    df_demo.index = pd.date_range('2023-01-01', periods=n, freq='D')

    scanner = MarketScanner(account_equity=500_000, risk_fraction=0.02)
    result = scanner.analyze(df_demo)

    print("=" * 60)
    print("MARKET STRUCTURE ANALYSIS DEMO")
    print("=" * 60)
    print(result.summary())
    print()
    if result.signals:
        print(f"\nAll signals ({len(result.signals)}):")
        for s in result.signals:
            print(f"  [{s.confidence:.0%}] {s.signal_type.upper()} {s.direction.upper()} "
                  f"entry={s.entry_price:.2f} stop={s.stop_price:.2f} "
                  f"T1={s.target_1:.2f} RR={s.rr_1:.1f}x "
                  f"size={s.position_size:.0f}")
    print()
    print("S/R Zones:")
    for z in result.sr_zones[:5]:
        sig_str = "✓ SIGNIFICANT" if z.is_significant else ""
        print(f"  {z.zone_type.value:12s} {z.zone_center:.2f} "
              f"[{z.zone_bottom:.2f}-{z.zone_top:.2f}] "
              f"touches={z.touch_count} bounce_rate={z.bounce_rate:.0%} {sig_str}")
