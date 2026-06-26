"""
backtest.py — Walk-forward backtester using the Adam Grimes market structure system.

USAGE:
    py backtest.py your_data.csv
    py backtest.py your_data.csv --equity 500000 --risk 0.01 --scan-every 5

HOW IT WORKS:
    1. Loads your full 2-year CSV
    2. Walks forward bar by bar
    3. Every `scan_every` bars: runs the full market structure scanner
    4. New signals → open a trade (if no trade already open)
    5. Every bar: checks if stop / T1 / T2 / T3 is hit
    6. At T1: move stop to breakeven (protect capital)
    7. At T2: move stop to T1 (lock in profit)
    8. Prints full performance report at the end

DATE FORMAT SUPPORTED:
    DD-MM-YYYY HH:MM   (your format, e.g. 01-11-2022 09:15)
    YYYY-MM-DD HH:MM   (ISO format)
    MM/DD/YYYY HH:MM   (US format)
    Any format pandas can auto-detect
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'market_structure'))

import pandas as pd
import numpy as np
import argparse
from dataclasses import dataclass, field
from typing import List, Optional


# ─── DATA LOADER ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Standardise column names (lowercase, strip spaces)
    df.columns = [c.lower().strip() for c in df.columns]

    # Map common aliases
    rename = {}
    aliases = {
        'datetime': ['date', 'datetime', 'timestamp', 'time', 'date/time', 'dt',
                     'price'],   # yfinance labels the datetime col "Price"
        'open':     ['open', 'o'],
        'high':     ['high', 'h'],
        'low':      ['low', 'l'],
        'close':    ['close', 'c', 'adj close', 'adj_close'],
        'volume':   ['volume', 'vol'],
    }
    for standard, opts in aliases.items():
        for opt in opts:
            if opt in df.columns and standard not in rename.values():
                rename[opt] = standard
                break
    df = df.rename(columns=rename)

    # Drop junk metadata rows (yfinance Ticker/Datetime rows)
    if 'datetime' in df.columns:
        mask = df['datetime'].astype(str).str.lower().str.strip().isin(
            {'ticker', 'datetime', 'date', 'nan', 'price', ''})
        df = df[~mask].reset_index(drop=True)

    # Parse datetime — try dayfirst (DD-MM-YYYY) first, then ISO
    if 'datetime' in df.columns:
        col = df['datetime']
    else:
        print("[ERROR] No datetime/date column found. Your columns:", list(df.columns))
        sys.exit(1)

    parsed = pd.to_datetime(col, dayfirst=True, errors='coerce')
    if parsed.isna().mean() > 0.5:   # if most failed, try ISO
        parsed = pd.to_datetime(col, dayfirst=False, errors='coerce')
    df['datetime'] = parsed
    df = df.dropna(subset=['datetime'])

    # Numeric columns
    for c in ['open', 'high', 'low', 'close']:
        if c not in df.columns:
            print(f"[ERROR] Column '{c}' not found. Your columns: {list(df.columns)}")
            sys.exit(1)
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'volume' not in df.columns:
        df['volume'] = 0
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)

    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    return df


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    r = df.set_index('datetime')[['open','high','low','close','volume']].resample(freq).agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    ).dropna(subset=['open'])
    return r.reset_index()


def detect_tf(df: pd.DataFrame) -> str:
    deltas = df['datetime'].diff().dropna()
    m = deltas.median().total_seconds() / 60
    if m < 2:    return '1min'
    if m < 8:    return '5min'
    if m < 20:   return '15min'
    if m < 50:   return '30min'
    if m < 90:   return '1h'
    if m < 300:  return '4h'
    return 'daily'


# ─── TRADE STATE ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    bar_open:    int
    direction:   str          # 'long' or 'short'
    entry:       float
    stop:        float
    t1:          float
    t2:          float
    t3:          float
    risk:        float        # abs(entry - stop)
    size:        float        # units
    risk_frac:   float
    signal_type: str
    confidence:  float
    bar_close:   int   = -1
    exit_price:  float = 0.0
    exit_reason: str   = ''
    r_multiple:  float = 0.0
    # partial exit tracking
    t1_hit:  bool = False
    t2_hit:  bool = False
    stop_be: bool = False    # stop moved to breakeven
    stop_t1: bool = False    # stop moved to T1


# ─── BACKTESTER ──────────────────────────────────────────────────────────────

class Backtester:
    def __init__(self, equity: float = 500_000, risk_frac: float = 0.02,
                 window: int = 500, scan_every: int = 5,
                 max_bars_in_trade: int = 100):
        self.initial_equity  = equity
        self.equity          = equity
        self.peak_equity     = equity
        self.risk_frac       = risk_frac
        self.window          = window        # bars of history to analyse
        self.scan_every      = scan_every    # re-run scanner every N bars
        self.max_bars_in_trade = max_bars_in_trade

        self.trades: List[Trade]      = []
        self.open_trade: Optional[Trade] = None
        self.equity_curve: List[float] = [equity]

    # ── core loop ──────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame, df_htf: pd.DataFrame) -> None:
        from scanner import MarketScanner
        scanner = MarketScanner(account_equity=self.equity,
                                risk_fraction=self.risk_frac)

        n = len(df)
        # Start after we have enough history to fill the analysis window
        start = max(self.window, 50)

        print(f"\n[BACKTEST] {n:,} bars total — starting analysis from bar {start}")
        print(f"           Scanning every {self.scan_every} bars  |  "
              f"Max trade duration: {self.max_bars_in_trade} bars\n")

        last_scan = -999
        signals_found = 0

        for i in range(start, n - 1):   # -1 so we always have a next bar to enter on
            bar = df.iloc[i]
            next_bar = df.iloc[i + 1]

            # ── 1. Manage open trade ────────────────────────────────────────
            if self.open_trade is not None:
                t = self.open_trade
                closed = self._manage_trade(t, bar, i)
                if closed:
                    self.open_trade = None

            # ── 2. Re-scan for new signals ──────────────────────────────────
            if (self.open_trade is None and
                    (i - last_scan) >= self.scan_every):

                last_scan = i
                df_ttf = df.iloc[max(0, i - self.window + 1): i + 1].reset_index(drop=True)

                # Align HTF up to current bar's timestamp
                htf_mask = df_htf['datetime'] <= bar['datetime']
                df_htf_slice = df_htf[htf_mask].tail(200).reset_index(drop=True)

                try:
                    result = scanner.analyze(df_ttf=df_ttf,
                                             df_htf=df_htf_slice if len(df_htf_slice) > 10 else None,
                                             df_ltf=None,
                                             prev_day_ohlc=None)
                except Exception:
                    continue

                # Update scanner equity for proper position sizing
                scanner.current_equity = self.equity

                if result.signals:
                    sig = result.signals[0]   # highest confidence signal
                    signals_found += 1

                    # Enter on next bar open (realistic execution)
                    entry = float(next_bar['open'])
                    direction = sig.direction

                    # Recalculate stop relative to actual entry
                    orig_risk = abs(sig.entry_price - sig.stop_price)
                    if direction == 'long':
                        stop = entry - orig_risk
                        t1   = entry + orig_risk * abs(sig.rr_1)
                        t2   = entry + orig_risk * abs(sig.rr_2)
                        t3   = entry + orig_risk * abs(sig.rr_3)
                    else:
                        stop = entry + orig_risk
                        t1   = entry - orig_risk * abs(sig.rr_1)
                        t2   = entry - orig_risk * abs(sig.rr_2)
                        t3   = entry - orig_risk * abs(sig.rr_3)

                    # Fixed fractional sizing
                    risk_amt = self.equity * self.risk_frac
                    size = max(1.0, np.floor(risk_amt / orig_risk)) if orig_risk > 0 else 1.0

                    trade = Trade(
                        bar_open=i + 1,
                        direction=direction,
                        entry=entry,
                        stop=stop,
                        t1=t1, t2=t2, t3=t3,
                        risk=orig_risk,
                        size=size,
                        risk_frac=self.risk_frac,
                        signal_type=sig.signal_type,
                        confidence=sig.confidence,
                    )
                    self.open_trade = trade

            # ── 3. Progress ─────────────────────────────────────────────────
            if (i % 2000 == 0):
                pct = (i - start) / max(1, n - start) * 100
                closed = len(self.trades)
                print(f"   {pct:5.1f}%  bar={i:5d}  equity={self.equity:>12,.0f}  "
                      f"closed_trades={closed}  signals_found={signals_found}")

        # Close any remaining open trade at last bar's close
        if self.open_trade is not None:
            t = self.open_trade
            last_close = float(df.iloc[-1]['close'])
            self._close_trade(t, len(df) - 1, last_close, 'end_of_data')

    def _manage_trade(self, t: Trade, bar: pd.Series, bar_idx: int) -> bool:
        """
        Simulate intra-bar trade management.
        Returns True if trade was closed this bar.
        Assumes bar order: open → (high or low depending on direction) → other extreme
        This is the standard assumption in bar-by-bar backtesting.
        """
        hi = float(bar['high'])
        lo = float(bar['low'])

        # Max bars protection
        if (bar_idx - t.bar_open) >= self.max_bars_in_trade:
            close_px = float(bar['close'])
            self._close_trade(t, bar_idx, close_px, 'time_stop')
            return True

        if t.direction == 'long':
            # Check stop first (assume worst case: low comes before high)
            if lo <= t.stop:
                self._close_trade(t, bar_idx, t.stop, 'stop')
                return True
            # T1
            if not t.t1_hit and hi >= t.t1:
                t.t1_hit = True
                t.stop = t.entry          # move stop to breakeven
                t.stop_be = True
                # Book 1/3 profit — track via r_multiple accumulation at close
                # (simplified: we mark it hit and close full at T3 or stop)
            # T2
            if t.t1_hit and not t.t2_hit and hi >= t.t2:
                t.t2_hit = True
                t.stop = t.t1             # move stop to T1
                t.stop_t1 = True
            # T3 — close full position
            if t.t2_hit and hi >= t.t3:
                self._close_trade(t, bar_idx, t.t3, 'target_3')
                return True

        else:  # short
            # Check stop first
            if hi >= t.stop:
                self._close_trade(t, bar_idx, t.stop, 'stop')
                return True
            if not t.t1_hit and lo <= t.t1:
                t.t1_hit = True
                t.stop = t.entry
                t.stop_be = True
            if t.t1_hit and not t.t2_hit and lo <= t.t2:
                t.t2_hit = True
                t.stop = t.t1
                t.stop_t1 = True
            if t.t2_hit and lo <= t.t3:
                self._close_trade(t, bar_idx, t.t3, 'target_3')
                return True

        return False

    def _close_trade(self, t: Trade, bar_idx: int, exit_px: float, reason: str):
        t.bar_close  = bar_idx
        t.exit_price = exit_px
        t.exit_reason = reason

        if t.risk > 0:
            if t.direction == 'long':
                pnl = exit_px - t.entry
            else:
                pnl = t.entry - exit_px
            t.r_multiple = pnl / t.risk
        else:
            t.r_multiple = 0.0

        # Update equity
        pnl_dollars = t.r_multiple * t.risk * t.size
        self.equity += pnl_dollars
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)
        self.trades.append(t)


# ─── PERFORMANCE REPORT ──────────────────────────────────────────────────────

def performance_report(bt: Backtester, df: pd.DataFrame):
    trades = bt.trades
    if not trades:
        print("\n[REPORT] No trades were taken.")
        return

    r_vals = np.array([t.r_multiple for t in trades])
    wins   = r_vals[r_vals > 0]
    losses = r_vals[r_vals <= 0]

    win_rate    = len(wins) / len(r_vals)
    avg_win     = float(np.mean(wins))    if len(wins)   > 0 else 0
    avg_loss    = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0
    expectancy  = float(np.mean(r_vals))
    profit_fac  = (float(np.sum(wins)) / abs(float(np.sum(losses)))
                   if len(losses) > 0 and np.sum(losses) != 0 else float('inf'))

    # Equity curve stats
    eq = np.array(bt.equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / peak
    max_dd = float(np.max(dd))

    total_return = (bt.equity - bt.initial_equity) / bt.initial_equity
    n_days = max(1, (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days)
    cagr   = (bt.equity / bt.initial_equity) ** (365 / n_days) - 1
    calmar = cagr / max_dd if max_dd > 0 else float('inf')

    rets = np.diff(eq) / eq[:-1]
    sharpe = (float(np.mean(rets)) / float(np.std(rets, ddof=1)) * np.sqrt(252)
              if len(rets) > 1 and np.std(rets) > 0 else 0.0)

    # Max consecutive losses
    max_consec_loss = 0
    cur = 0
    for r in r_vals:
        if r < 0:
            cur += 1
            max_consec_loss = max(max_consec_loss, cur)
        else:
            cur = 0

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    # Signal type breakdown
    sig_types = {}
    for t in trades:
        k = t.signal_type
        if k not in sig_types:
            sig_types[k] = {'n': 0, 'wins': 0, 'r_sum': 0.0}
        sig_types[k]['n'] += 1
        sig_types[k]['wins'] += (1 if t.r_multiple > 0 else 0)
        sig_types[k]['r_sum'] += t.r_multiple

    sep = "=" * 70
    print(f"\n{sep}")
    print("BACKTEST PERFORMANCE REPORT")
    print(sep)
    print(f"  Period         : {df['datetime'].iloc[0].date()} to {df['datetime'].iloc[-1].date()}")
    print(f"  Initial Equity : {bt.initial_equity:>12,.0f}")
    print(f"  Final Equity   : {bt.equity:>12,.0f}  ({total_return:+.1%})")
    print(f"  CAGR           : {cagr:+.2%}")
    print(f"  Max Drawdown   : {max_dd:.2%}")
    print(f"  Calmar Ratio   : {calmar:.2f}  (CAGR / MaxDD)")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")

    print(f"\n  Total Trades   : {len(trades)}")
    print(f"  Win Rate       : {win_rate:.1%}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg Win        : +{avg_win:.2f}R")
    print(f"  Avg Loss       : -{avg_loss:.2f}R")
    print(f"  Expectancy     : {expectancy:+.3f}R per trade")
    print(f"  Profit Factor  : {profit_fac:.2f}  (>1.5 = good, >2.0 = excellent)")
    print(f"  Max Consec Loss: {max_consec_loss}")
    print(f"  Best Trade     : {float(np.max(r_vals)):+.2f}R")
    print(f"  Worst Trade    : {float(np.min(r_vals)):+.2f}R")

    print(f"\n  Exit Breakdown :")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / len(trades) * 100
        print(f"    {reason:<20s}: {count:4d}  ({pct:.1f}%)")

    print(f"\n  Signal Type Breakdown :")
    for stype, stats in sorted(sig_types.items(), key=lambda x: -x[1]['n']):
        wr = stats['wins'] / stats['n'] if stats['n'] > 0 else 0
        avg_r = stats['r_sum'] / stats['n'] if stats['n'] > 0 else 0
        print(f"    {stype:<24s}: {stats['n']:3d} trades  "
              f"WR={wr:.0%}  avgR={avg_r:+.2f}")

    print(sep)

    # Monthly breakdown
    print("\nMONTHLY BREAKDOWN")
    print("-" * 50)
    trade_df = pd.DataFrame([{
        'month':      pd.Timestamp(df['datetime'].iloc[t.bar_close]).to_period('M'),
        'r_multiple': t.r_multiple,
        'win':        t.r_multiple > 0,
    } for t in trades if t.bar_close < len(df)])

    if not trade_df.empty:
        monthly = trade_df.groupby('month').agg(
            trades=('r_multiple', 'count'),
            total_r=('r_multiple', 'sum'),
            win_rate=('win', 'mean')
        ).reset_index()
        for _, row in monthly.iterrows():
            bar_char = '+' if row['total_r'] >= 0 else '-'
            bar_len  = min(20, int(abs(row['total_r']) * 2))
            bar_str  = bar_char * bar_len
            print(f"  {str(row['month']):<10s}  {row['trades']:2.0f} trades  "
                  f"WR={row['win_rate']:.0%}  {row['total_r']:+6.2f}R  {bar_str}")

    print(sep)
    return trade_df


def save_trade_log(trades: List[Trade], path: str):
    rows = [{
        'bar_open':    t.bar_open,
        'bar_close':   t.bar_close,
        'direction':   t.direction,
        'signal_type': t.signal_type,
        'confidence':  round(t.confidence, 2),
        'entry':       round(t.entry, 4),
        'stop':        round(t.stop, 4),
        'target_1':    round(t.t1, 4),
        'target_2':    round(t.t2, 4),
        'target_3':    round(t.t3, 4),
        'exit_price':  round(t.exit_price, 4),
        'exit_reason': t.exit_reason,
        'r_multiple':  round(t.r_multiple, 3),
        'size':        t.size,
        't1_hit':      t.t1_hit,
        't2_hit':      t.t2_hit,
    } for t in trades]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n[SAVED] Trade log: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Walk-forward backtest using Grimes framework')
    parser.add_argument('csv',          nargs='?', default='nifty50.csv')
    parser.add_argument('--equity',     type=float, default=500_000)
    parser.add_argument('--risk',       type=float, default=0.02,
                        help='Risk per trade (default 0.02 = 2%%)')
    parser.add_argument('--window',     type=int,   default=500,
                        help='Bars of history for each analysis (default 500)')
    parser.add_argument('--scan-every', type=int,   default=5,
                        help='Scan for new signals every N bars (default 5)')
    parser.add_argument('--max-bars',   type=int,   default=100,
                        help='Max bars to hold a trade (default 100)')
    parser.add_argument('--output',     type=str,   default='trade_log.csv')
    args = parser.parse_args()

    print("=" * 70)
    print("Walk-Forward Backtest — Adam Grimes Market Structure Framework")
    print("=" * 70)

    # Load
    print(f"\n[LOAD] {args.csv}...")
    df = load_csv(args.csv)
    tf = detect_tf(df)
    print(f"       {len(df):,} bars  |  {df['datetime'].iloc[0].date()} to "
          f"{df['datetime'].iloc[-1].date()}  |  TF: {tf}")
    print(f"       Price: {df['low'].min():.2f} to {df['high'].max():.2f}")

    # HTF
    htf_map = {'1min':'15min','5min':'1h','15min':'4h',
               '30min':'4h','1h':'1D','4h':'1W','daily':'1W'}
    htf_freq = htf_map.get(tf, '1D')
    df_htf   = resample(df, htf_freq)
    print(f"       HTF ({htf_freq}): {len(df_htf)} bars")

    # Run
    bt = Backtester(
        equity     = args.equity,
        risk_frac  = args.risk,
        window     = args.window,
        scan_every = args.scan_every,
        max_bars_in_trade = args.max_bars,
    )
    bt.run(df, df_htf)

    # Report
    performance_report(bt, df)

    # Save trade log
    if bt.trades:
        save_trade_log(bt.trades, args.output)


if __name__ == '__main__':
    main()
