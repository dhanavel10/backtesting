"""
risk_manager.py — Position sizing, Kelly criterion, Monte Carlo, drawdown management.
Based on: The Art and Science of Technical Analysis (Adam Grimes)

Core principles from the book:
- Never risk more than 2% per trade (fixed fractional default)
- Kelly criterion shows the mathematically optimal fraction
- Monte Carlo reveals true probability of ruin
- Drawdown rules are non-negotiable
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─── POSITION SIZING ─────────────────────────────────────────────────────────

@dataclass
class PositionSize:
    shares: float              # units to trade
    risk_amount: float         # dollar risk on this trade
    risk_pct: float            # fraction of equity risked
    position_value: float      # total position value
    position_pct: float        # fraction of equity in position
    notes: List[str] = field(default_factory=list)


def fixed_fractional_size(account_equity: float, risk_fraction: float,
                           entry: float, stop: float,
                           min_units: float = 1.0,
                           max_position_pct: float = 0.20) -> PositionSize:
    """
    Standard fixed fractional position sizing.

    Args:
        account_equity: current account value
        risk_fraction:  fraction of equity to risk per trade (e.g., 0.02)
        entry:          intended entry price
        stop:           stop loss price
        min_units:      minimum order size
        max_position_pct: maximum fraction of equity in any single position
    """
    notes = []
    price_risk = abs(entry - stop)
    if price_risk == 0:
        return PositionSize(0, 0, 0, 0, 0, ["Stop equals entry; cannot size"])

    risk_amount = account_equity * risk_fraction
    raw_units = risk_amount / price_risk
    units = max(min_units, np.floor(raw_units))

    # Cap at max position size
    max_units = (account_equity * max_position_pct) / entry if entry > 0 else units
    if units > max_units:
        units = np.floor(max_units)
        notes.append(f"Position capped at {max_position_pct:.0%} of equity")

    actual_risk = units * price_risk
    position_value = units * entry

    return PositionSize(
        shares=units,
        risk_amount=actual_risk,
        risk_pct=actual_risk / account_equity if account_equity > 0 else 0,
        position_value=position_value,
        position_pct=position_value / account_equity if account_equity > 0 else 0,
        notes=notes
    )


def portfolio_heat_check(open_trades: List[dict], new_risk_pct: float,
                          max_heat: float = 0.06) -> bool:
    """
    Check if adding a new trade would exceed maximum portfolio heat (total open risk).

    Args:
        open_trades: list of dicts with 'risk_pct' key
        new_risk_pct: risk fraction for the new trade
        max_heat: maximum total open risk (default 6%)

    Returns True if trade is safe to take.
    """
    current_heat = sum(t.get('risk_pct', 0) for t in open_trades)
    return (current_heat + new_risk_pct) <= max_heat


# ─── KELLY CRITERION ─────────────────────────────────────────────────────────

@dataclass
class KellyResult:
    full_kelly: float
    half_kelly: float
    quarter_kelly: float
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    b_ratio: float             # win/loss ratio
    expectancy_per_r: float    # expected R per trade


def kelly_criterion(win_rate: float, avg_win_r: float,
                    avg_loss_r: float = 1.0) -> KellyResult:
    """
    Kelly fraction for fixed-size bets.
    f* = (b*p - q) / b
    where b = avg_win_r/avg_loss_r, p = win_rate, q = 1-p

    Args:
        win_rate:   fraction of trades that win (0-1)
        avg_win_r:  average win in R-multiples
        avg_loss_r: average loss in R-multiples (typically 1.0)
    """
    p = np.clip(win_rate, 0.01, 0.99)
    q = 1 - p
    b = avg_win_r / avg_loss_r if avg_loss_r > 0 else avg_win_r

    full_kelly = (b * p - q) / b
    full_kelly = max(0.0, float(full_kelly))

    expectancy = p * avg_win_r - q * avg_loss_r

    return KellyResult(
        full_kelly=full_kelly,
        half_kelly=full_kelly * 0.5,
        quarter_kelly=full_kelly * 0.25,
        win_rate=win_rate,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        b_ratio=b,
        expectancy_per_r=float(expectancy)
    )


# ─── R-MULTIPLE TRACKING ─────────────────────────────────────────────────────

def r_multiple(entry: float, exit_price: float, stop: float,
               direction: str = 'long') -> float:
    """
    Compute R-multiple for a completed trade.
    R = initial_risk = abs(entry - stop)
    R_multiple = profit_or_loss / R
    Positive = win, Negative = loss.
    """
    initial_risk = abs(entry - stop)
    if initial_risk == 0:
        return 0.0
    if direction == 'long':
        pnl = exit_price - entry
    else:
        pnl = entry - exit_price
    return pnl / initial_risk


def trade_statistics(r_multiples: List[float]) -> dict:
    """
    Compute trading statistics from a list of R-multiples.
    """
    if not r_multiples:
        return {}

    arr = np.array(r_multiples)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    win_rate = len(wins) / len(arr) if len(arr) > 0 else 0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0
    expectancy = float(np.mean(arr))
    profit_factor = (float(np.sum(wins)) / abs(float(np.sum(losses)))
                     if len(losses) > 0 and np.sum(losses) != 0 else float('inf'))

    # Sharpe ratio (annualized assuming daily trades)
    if np.std(arr, ddof=1) > 0:
        sharpe = float(expectancy / np.std(arr, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Maximum drawdown in R
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd_r = float(np.max(drawdown))

    return {
        'n_trades': len(arr),
        'win_rate': win_rate,
        'avg_win_r': avg_win,
        'avg_loss_r': avg_loss,
        'expectancy_r': expectancy,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown_r': max_dd_r,
        'std_r': float(np.std(arr, ddof=1)),
        'best_trade_r': float(np.max(arr)),
        'worst_trade_r': float(np.min(arr)),
        'consecutive_losses': _max_consecutive_losses(arr),
        'kelly_fraction': kelly_criterion(win_rate, avg_win, avg_loss).full_kelly
               if avg_loss > 0 else 0.0,
    }


def _max_consecutive_losses(arr: np.ndarray) -> int:
    max_consec = 0
    current = 0
    for r in arr:
        if r < 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    return max_consec


# ─── MONTE CARLO SIMULATION ───────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    risk_fraction: float
    mean_terminal: float
    median_terminal: float
    std_terminal: float
    coeff_variation: float
    mean_max_value: float
    mean_min_value: float
    pct_bankrupt: float
    pct_dd_75: float
    sharpe_ratio: float
    recommended: bool           # True if risk fraction is safe and profitable


def run_monte_carlo(r_multiples: List[float],
                    initial_equity: float = 100_000,
                    risk_fractions: List[float] = None,
                    n_simulations: int = 1000,
                    n_forward_trades: int = 250,
                    bankruptcy_threshold: float = 0.25,
                    seed: int = 42) -> List[MonteCarloResult]:
    """
    Run Monte Carlo simulation for multiple risk fractions.
    Uses bootstrap resampling of historical R-multiples.

    Args:
        r_multiples:    historical trade R-multiples
        initial_equity: starting account balance
        risk_fractions: list of risk fractions to test (default: standard set)
        n_simulations:  number of Monte Carlo runs per risk fraction
        n_forward_trades: trades per simulation run
        bankruptcy_threshold: equity fraction below which we call it "ruined"
    """
    if not r_multiples or len(r_multiples) < 5:
        return []

    if risk_fractions is None:
        risk_fractions = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

    rng = np.random.default_rng(seed)
    r_arr = np.array(r_multiples)
    results = []

    for rf in risk_fractions:
        terminal_values = []
        max_values = []
        min_values = []
        bankrupt_count = 0
        dd_75_count = 0

        for _ in range(n_simulations):
            # Bootstrap sample
            sample = rng.choice(r_arr, size=n_forward_trades, replace=True)
            equity = float(initial_equity)
            peak = equity
            ruined = False
            max_eq = equity
            min_eq = equity

            for r in sample:
                trade_risk = equity * rf
                pnl = trade_risk * r
                equity += pnl
                peak = max(peak, equity)
                max_eq = max(max_eq, equity)
                min_eq = min(min_eq, equity)

                if equity <= 0:
                    equity = 0
                    ruined = True
                    break

            terminal_values.append(equity)
            max_values.append(max_eq)
            min_values.append(min_eq)

            if ruined or equity < initial_equity * bankruptcy_threshold:
                bankrupt_count += 1
            if peak > 0 and (peak - equity) / peak >= 0.75:
                dd_75_count += 1

        term = np.array(terminal_values)
        pct_bankrupt = bankrupt_count / n_simulations
        pct_dd75 = dd_75_count / n_simulations

        # Sharpe of terminal values (simplified)
        if np.std(term) > 0:
            mc_sharpe = float((np.mean(term) - initial_equity) / np.std(term))
        else:
            mc_sharpe = 0.0

        cv = float(np.std(term) / np.mean(term)) if np.mean(term) > 0 else float('inf')
        recommended = (pct_bankrupt < 0.02 and cv < 0.5 and np.mean(term) > initial_equity)

        results.append(MonteCarloResult(
            risk_fraction=rf,
            mean_terminal=float(np.mean(term)),
            median_terminal=float(np.median(term)),
            std_terminal=float(np.std(term)),
            coeff_variation=cv,
            mean_max_value=float(np.mean(max_values)),
            mean_min_value=float(np.mean(min_values)),
            pct_bankrupt=pct_bankrupt,
            pct_dd_75=pct_dd75,
            sharpe_ratio=mc_sharpe,
            recommended=recommended
        ))

    return results


def optimal_risk_fraction(mc_results: List[MonteCarloResult],
                           max_pct_bankrupt: float = 0.02) -> float:
    """Find the optimal risk fraction: highest expected return with bankruptcy < threshold."""
    safe = [r for r in mc_results if r.pct_bankrupt <= max_pct_bankrupt]
    if not safe:
        return 0.01  # fallback to conservative
    return max(safe, key=lambda r: r.mean_terminal).risk_fraction


# ─── DRAWDOWN MANAGEMENT ─────────────────────────────────────────────────────

@dataclass
class DrawdownState:
    current_equity: float
    peak_equity: float
    drawdown_pct: float         # from peak
    drawdown_from_start: float  # from initial equity
    size_multiplier: float      # 0-1, adjust position size
    action: str                 # NORMAL, REDUCE_25, REDUCE_50, REDUCE_75, STOP


def check_drawdown(current_equity: float, peak_equity: float,
                    initial_equity: float) -> DrawdownState:
    """
    Grimes drawdown rules applied as a state machine.

    Drawdown from peak:
    0-5%:   Normal trading (100% size)
    5-10%:  Reduce to 75%
    10-15%: Reduce to 50%
    15-20%: Reduce to 25%
    >20%:   Stop trading; review system
    """
    dd_from_peak = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
    dd_from_start = (initial_equity - current_equity) / initial_equity if initial_equity > 0 else 0

    if dd_from_peak >= 0.20:
        return DrawdownState(current_equity, peak_equity, dd_from_peak,
                             dd_from_start, 0.0, "STOP")
    elif dd_from_peak >= 0.15:
        return DrawdownState(current_equity, peak_equity, dd_from_peak,
                             dd_from_start, 0.25, "REDUCE_75")
    elif dd_from_peak >= 0.10:
        return DrawdownState(current_equity, peak_equity, dd_from_peak,
                             dd_from_start, 0.50, "REDUCE_50")
    elif dd_from_peak >= 0.05:
        return DrawdownState(current_equity, peak_equity, dd_from_peak,
                             dd_from_start, 0.75, "REDUCE_25")
    else:
        return DrawdownState(current_equity, peak_equity, dd_from_peak,
                             dd_from_start, 1.00, "NORMAL")


# ─── EQUITY CURVE ANALYSIS ───────────────────────────────────────────────────

def equity_curve_stats(equity_series: List[float]) -> dict:
    """
    Analyze an equity curve for performance metrics.
    """
    if len(equity_series) < 2:
        return {}

    arr = np.array(equity_series, dtype=float)
    returns = np.diff(arr) / arr[:-1]

    # Max drawdown
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / peak
    max_dd = float(np.max(drawdown))
    max_dd_idx = int(np.argmax(drawdown))

    # CAGR (assuming daily bars, 252 trading days)
    n_days = len(arr)
    if arr[0] > 0:
        cagr = float((arr[-1] / arr[0]) ** (252 / n_days) - 1)
    else:
        cagr = 0.0

    # Sharpe
    if np.std(returns, ddof=1) > 0:
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Calmar ratio
    calmar = cagr / max_dd if max_dd > 0 else float('inf')

    # Recovery factor
    total_profit = arr[-1] - arr[0]
    recovery_factor = total_profit / (max_dd * arr[0]) if max_dd > 0 and arr[0] > 0 else float('inf')

    return {
        'initial_equity': float(arr[0]),
        'final_equity': float(arr[-1]),
        'total_return': float((arr[-1] - arr[0]) / arr[0]),
        'cagr': cagr,
        'max_drawdown': max_dd,
        'max_drawdown_bar': max_dd_idx,
        'sharpe_ratio': sharpe,
        'calmar_ratio': float(calmar),
        'recovery_factor': float(recovery_factor),
        'volatility': float(np.std(returns, ddof=1) * np.sqrt(252)),
    }
