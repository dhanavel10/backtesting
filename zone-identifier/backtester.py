"""
backtester.py
=============
Walk-forward backtest for the breakout strategy.

Design
------
  • Splits data into a "warmup" period (for zone building) and a "test" period.
  • Feeds the test candles one-by-one through CandleBuffer → BreakoutEngine.
  • After every `rebuild_every` candles in the test period, ZoneEngine re-builds
    zones using history up to that point (no look-ahead bias).
  • Produces a full trade log and summary statistics.

Usage
-----
    from backtester import Backtester
    from zone_engine import ZoneEngine, fetch_intraday_chunked

    df = fetch_intraday_chunked("^NSEI", days=60)
    bt = Backtester(df, warmup_days=30)
    results = bt.run()
    bt.print_summary()
    bt.plot_equity_curve()
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys, os

# ── Flat-folder import fix ──────────────────────────────────────
# All files live in the same directory — add that dir to sys.path
# so Python can find zone_engine, candle_buffer, breakout_engine
# regardless of where you run the script from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from zone_engine    import ZoneEngine, fetch_intraday_chunked
from candle_buffer  import CandleBuffer
from breakout_engine import BreakoutEngine, TradeResult


class Backtester:
    """
    Walk-forward backtest — no look-ahead bias.

    Parameters
    ----------
    df           : Full historical 5m DataFrame (output of fetch_intraday_chunked)
    warmup_days  : Calendar days of initial data used only for zone seeding.
                   No trades taken during warmup.
    zone_kwargs  : Passed to ZoneEngine constructor.
    strategy_kwargs : Passed to BreakoutEngine constructor.
    """

    def __init__(
        self,
        df:           pd.DataFrame,
        warmup_days:  int  = 20,
        zone_kwargs:  dict = None,
        strat_kwargs: dict = None,
    ):
        self.df          = df.copy()
        self.warmup_days = warmup_days
        self.zone_kwargs = zone_kwargs  or {}
        self.strat_kwargs= strat_kwargs or {}

        self.trade_log:   list[TradeResult] = []
        self._ze   = None
        self._be   = None

    # ── Run ──────────────────────────────────────────────────

    def run(self) -> list[TradeResult]:
        df   = self.df
        dates = sorted(set(df.index.date))

        # Split into warmup / test
        cutoff_date = dates[self.warmup_days] if len(dates) > self.warmup_days else dates[-1]
        warmup_df   = df[df.index.date < cutoff_date]
        test_df     = df[df.index.date >= cutoff_date]

        print(f"[Backtester] Warmup: {warmup_df.index[0].date()} → "
              f"{warmup_df.index[-1].date()}  ({len(warmup_df)} bars)")
        print(f"[Backtester] Test  : {test_df.index[0].date()}  → "
              f"{test_df.index[-1].date()}  ({len(test_df)} bars)")

        # Build initial zone map from warmup data
        ze = ZoneEngine(**self.zone_kwargs)
        ze.build_from_history(df=warmup_df)

        be = BreakoutEngine(
            zone_engine=ze,
            on_trade_closed=self._record_trade,
            **self.strat_kwargs,
        )

        # Candle buffer feeds into breakout engine
        buf = CandleBuffer(
            interval_minutes=5,
            pivot_right_bars=ze.right_bars,
            on_candle_closed=be.on_candle,
        )

        # Add zone engine update hook — after every confirmed candle
        # also feed the candle into ZoneEngine's rolling buffer
        _orig_on_candle = be.on_candle

        def _patched_on_candle(candle):
            ze.add_candle(candle)
            _orig_on_candle(candle)

        be.on_candle = _patched_on_candle

        print(f"[Backtester] Replaying {len(test_df)} test candles...\n")

        for ts, row in test_df.iterrows():
            candle = pd.Series({
                "Open": row["Open"], "High": row["High"],
                "Low": row["Low"], "Close": row["Close"],
                "Volume": row.get("Volume", 0),
            }, name=ts)
            buf.inject_candle(candle)

        self._ze = ze
        self._be = be
        print(f"\n[Backtester] ✓ Backtest complete — {len(self.trade_log)} trades")
        return self.trade_log

    def _record_trade(self, result: TradeResult):
        self.trade_log.append(result)

    # ── Summary ───────────────────────────────────────────────

    def print_summary(self):
        if not self.trade_log:
            print("[Backtester] No trades taken.")
            return

        summary = self._be.get_trade_summary()
        pnls    = [t.pnl_pts for t in self.trade_log]
        reasons = {}
        for t in self.trade_log:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        sep = "═" * 65
        print("\n" + sep)
        print("  BACKTEST SUMMARY — Precision Zone Breakout Strategy")
        print(sep)
        print(f"  Total trades    : {summary['trades']}")
        print(f"  Win / Loss      : {summary['wins']} / {summary['losses']}")
        print(f"  Win rate        : {summary['win_rate']}%")
        print(f"  Total P&L (pts) : {summary['total_pts']:+.2f}")
        print(f"  Avg win (pts)   : {summary['avg_win']:+.2f}")
        print(f"  Avg loss (pts)  : {summary['avg_loss']:+.2f}")
        print(f"  Best trade      : {summary['best_trade']:+.2f} pts")
        print(f"  Worst trade     : {summary['worst_trade']:+.2f} pts")
        print(f"  Profit factor   : {summary['profit_factor']}")
        print(f"  Exits → {dict(reasons)}")
        print(sep)

        print("\n  TRADE LOG")
        print("  " + "─" * 60)
        for t in self.trade_log:
            dir_sym = "▲" if t.signal.direction.value == "LONG" else "▼"
            print(
                f"  {dir_sym} {t.signal.timestamp.strftime('%Y-%m-%d %H:%M')} "
                f"  Entry={t.signal.entry_price:.0f}  "
                f"Exit={t.exit_price:.0f}  "
                f"{t.exit_reason:<8}  "
                f"P&L={t.pnl_pts:+.0f} pts"
            )
        print(sep + "\n")

    def get_equity_curve(self) -> pd.Series:
        """Cumulative P&L in points over time."""
        if not self.trade_log:
            return pd.Series(dtype=float)
        times = [t.exit_time for t in self.trade_log]
        pnls  = [t.pnl_pts  for t in self.trade_log]
        s = pd.Series(pnls, index=times).cumsum()
        return s

    def plot_equity_curve(self):
        """Plot equity curve using plotly."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("[Backtester] plotly not installed — run: pip install plotly")
            return

        eq = self.get_equity_curve()
        if eq.empty:
            print("[Backtester] No trades to plot.")
            return

        color = ["#26a69a" if v >= 0 else "#ef5350" for v in eq.values]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            mode="lines+markers",
            line=dict(color="#26a69a", width=2),
            marker=dict(color=color, size=8),
            name="Cumulative P&L (pts)",
        ))
        fig.add_hline(y=0, line_dash="dash",
                      line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            title="Breakout Strategy — Equity Curve (Points)",
            template="plotly_dark",
            height=450,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            yaxis_title="Cumulative Points",
            xaxis_title="Trade Exit Time",
        )
        fig.show()


# ════════════════════════════════════════════════════════════════
# CONVENIENCE: run from command line
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Precision Zone Breakout — Walk-Forward Backtest")
    print("=" * 65)

    df = fetch_intraday_chunked("^NSEI", days=60)

    bt = Backtester(
        df,
        warmup_days=20,
        zone_kwargs={
            "cluster_tolerance": 15.0,
            "zone_half_band":    15.0,
            # Relaxed: 2 touches instead of 3 — more zones survive
            "min_wick_touches":  2,
            "min_sessions":      2,
            "min_rejections":    1,
        },
        strat_kwargs={
            # Lowered 20 → 10 pts: 5m Nifty candles rarely close
            # 20 pts clear of a zone boundary. 10 is realistic.
            "min_breakout_pts":    10.0,
            # No confirm candle in backtest — reduces missed entries
            "confirm_candles":     0,
            "rr_ratio":            2.0,
            # Wider margin before a zone is invalidated (was default=15)
            # so zones aren't wiped the moment price passes through once
            "invalidation_margin": 25.0,
        },
    )
    bt.run()
    bt.print_summary()
    bt.plot_equity_curve()