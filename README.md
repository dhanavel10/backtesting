# Realtime S/R Zone Detection System
## For Nifty / BankNifty Intraday Options Buying

---

## Files

| File | Purpose |
|------|---------|
| `tick_processor.py` | WebSocket tick ingester + OHLCV candle aggregator |
| `swing_engine.py` | Causal pivot detection (no lookahead) — Fixed Window + Reversal methods |
| `zone_engine.py` | S/R zone clustering, touch/rejection scoring, zone lifecycle |
| `signal_engine.py` | REJECTION / BREAKOUT / APPROACH signal generator |
| `backtest_validator.py` | Historical replay using **exact same pipeline** — for verification |
| `main.py` | Orchestrator + WebSocket server (port 8766) + HTTP server (port 8080) |
| `dashboard.html` | Live trading dashboard |

---

## Quick Start

### 1. Install dependencies
```bash
pip install yfinance pandas numpy scipy websockets plotly
```

### 2. Run historical backtest (verify accuracy first)
```bash
python backtest_validator.py --ticker ^NSEI --days 30
# BankNifty:
python backtest_validator.py --ticker ^NSEBANK --symbol BANKNIFTY --days 30
```

Backtest flags:
```
--days 30          # lookback period
--rev-pct 0.30     # % reversal to confirm a pivot (0.30 = 0.30%)
                   # On Nifty@24000: 0.30% ≈ 72pts reversal
--min-swing 20     # minimum absolute point swing to count
--half-band 15     # zone = level ± 15pts (30pt total band)
--cluster-tol 15   # pivots within 15pts join same cluster
--min-touches 2    # minimum wick touches to qualify zone
--no-chart         # skip plotly chart
```

### 3. Start live system
```bash
python main.py --symbol NIFTY --tick-uri ws://localhost:8765
```

### 4. Open dashboard
```
http://localhost:8080/dashboard.html
```

---

## Tick Feed Format

Your broker's WebSocket must emit JSON like:
```json
{
  "symbol": "NIFTY",
  "ltp":    24512.50,
  "volume": 123456,
  "ts":     1716000000.123
}
```

Adapt `tick_processor.py → Tick.from_dict()` to match your broker's schema.

---

## Key Design Decisions

### Causal pivot detection (no lookahead)
The **ReversalPivotDetector** confirms a pivot high when price *falls* `rev_pct`%
from the running peak. This is the cleanest causal method:
- No future bars used
- Confirmation lag = time for price to reverse `rev_pct`%
- Directly maps to "level where price reversed significantly"

### Absolute point zones (not percentage)
All zone widths are in **absolute points**, not percentage.
- 15pt band on Nifty@24000 = 0.06% — precision for options entries
- Percentage zones (even 0.1%) = 24pts — too wide

### Zone lifecycle
```
ACTIVE  → price repeatedly respects the zone
BROKEN  → 2 closes beyond the zone edge by breakout_pts
EXPIRED → no touch for max_idle_bars candles
FLIPPED → broken support becomes resistance (auto-detected)
```

### Signal confidence levels
```
HIGH   (4-5 confirmations): wick + close outside + long wick + body + direction
MEDIUM (2-3 confirmations): at least 2 criteria met
LOW    (1 confirmation):    approach / first touch only
```

---

## Tuning Guide

| Scenario | Adjustment |
|----------|-----------|
| Too many zones | Raise `--min-touches`, `--min-sessions` |
| Zones too wide | Lower `--half-band` (try 10) |
| Missing pivots | Lower `--rev-pct` (try 0.20) or `--min-swing` |
| Too many signals | Raise `--min-strength` in signal_engine |
| BankNifty | Raise `--half-band 20`, `--min-swing 30`, `--rev-pct 0.35` |

---

## Architecture

```
localhost WS (tick feed)
        ↓
TickFeedClient.run()          [async]
        ↓  on_tick()
CandleAggregator              [groups ticks into OHLCV bars]
        ↓  on_candle_closed()
ReversalPivotDetector         [causal: emits PivotEvent when price reverses rev_pct%]
        ↓  on_pivot()
ZoneEngine                    [clusters pivots, scores zones, tracks lifecycle]
        ↓  on_candle() + on_pivot()
SignalEngine                  [rejection / breakout / approach signals]
        ↓  on_signal()
WebSocket server (8766)       [broadcast to dashboard]
        ↓
Dashboard (8080)              [realtime zone map + signal log]
```
