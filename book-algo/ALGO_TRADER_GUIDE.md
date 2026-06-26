# Algo Trader's Implementation Guide
## Based on: The Art and Science of Technical Analysis (Adam Grimes)

> **Purpose:** This guide translates every quantifiable concept from Grimes' book into implementable Python trading system components. Use it as the specification document for your strategy engine.

---

## SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                    MARKET DATA FEED                             │
│            (OHLCV bars, multiple timeframes)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  indicators │  │  pivot_    │  │   trend_   │
    │    .py      │  │ detector   │  │  analyzer  │
    │(ATR,EMA,    │  │   .py      │  │    .py     │
    │MACD,Keltner)│  │(HH/HL/LL/ │  │(trend state│
    └────────────┘  │ LH pivots) │  │ Dow Theory)│
           │        └────────────┘  └────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                  ┌─────────────────┐
                  │  market_        │
                  │  structure.py   │
                  │ (regime detect) │
                  └────────┬────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  sr_zones  │  │  channel_  │  │  signal_   │
    │    .py     │  │ detector   │  │  engine    │
    │(S/R clusters│  │    .py     │  │    .py     │
    │ zones)     │  │(channels,  │  │(all 6 trade│
    └────────────┘  │ trendlines)│  │  types)    │
                    └────────────┘  └────────────┘
                                           │
                                    ┌────────────┐
                                    │   risk_    │
                                    │ manager.py │
                                    │(sizing,    │
                                    │ Kelly,     │
                                    │ Monte Carlo│
                                    └────────────┘
                                           │
                                    ┌────────────┐
                                    │  scanner   │
                                    │    .py     │
                                    │(orchestrate│
                                    │ everything)│
                                    └────────────┘
```

---

## MODULE 1 — indicators.py

### Purpose
Foundation layer. All other modules consume these. **Never compute ATR or EMA inline elsewhere.**

### Functions to implement

#### ATR (Average True Range)
```python
# True Range = max(H-L, |H-prev_C|, |L-prev_C|)
# ATR = EMA(TR, period) or Wilder smoothing
# Primary use: Stop placement, overextension detection, position sizing
atr = calculate_atr(high, low, close, period=14)
```

#### EMA (Exponential Moving Average)
```python
# α = 2 / (period + 1)
# EMA[t] = α × price[t] + (1-α) × EMA[t-1]
# Multiple periods needed: 9, 12, 20, 26, 50, 200
ema_20 = calculate_ema(close, 20)
```

#### Keltner Channel
```python
# middle = EMA(20)
# upper = middle + multiplier × ATR(20)
# lower = middle - multiplier × ATR(20)
# multiplier = 2.0 standard; use 1.5 for tighter, 2.5 for wider
kc_upper, kc_mid, kc_lower = keltner_channel(high, low, close, period=20, mult=2.0)
```

#### MACD
```python
# macd_line = EMA(12) - EMA(26)
# signal_line = EMA(9, macd_line)
# histogram = macd_line - signal_line
macd_line, signal, histogram = calculate_macd(close, fast=12, slow=26, signal=9)
```

#### Momentum (Rate of Change)
```python
# ROC = (close[t] - close[t-n]) / close[t-n] × 100
roc = calculate_roc(close, period=14)
```

#### Bar Range Ratio
```python
# Used for climax detection and bar character analysis
# close_position_in_bar = (close - low) / (high - low)
# Used to detect: bullish bar (>0.6), bearish bar (<0.4), indecision (0.4-0.6)
bar_close_position = (close - low) / (high - low)
```

---

## MODULE 2 — pivot_detector.py

### Purpose
Detect all swing highs (pivot highs) and swing lows (pivot lows) with configurable strength. This is the **most critical module** — all subsequent analysis depends on clean pivot detection.

### Algorithm
```python
# Method: Fractal pivot detection
# A pivot high at index i requires:
#   high[i] > high[i-k] for all k in 1..n (left bars)
#   high[i] > high[i+k] for all k in 1..n (right bars)
# Standard: n=2 (tight), n=3 (medium), n=5 (loose/swing)

# Adaptive method: use ATR to filter out pivots that are too close
# Minimum pivot separation: min_separation × ATR

def find_pivot_highs(high, n=3, atr_filter=True):
    # Returns: list of (index, price) for each pivot high

def find_pivot_lows(low, n=3, atr_filter=True):
    # Returns: list of (index, price) for each pivot low
```

### Pivot Strength Scoring
Not all pivots are equal. Score them:
```python
# Pivot score factors:
# 1. How many bars it dominated (look-back window)
# 2. ATR-normalized height: (pivot_price - nearest_neighbor) / ATR
# 3. Volume at pivot if available
# 4. Time since last pivot of similar level (recency)
# 5. Whether it aligned with previous S/R level

pivot_score = (bars_dominated / total_bars) × atr_height_score × recency_factor
```

### Zigzag Reconstruction
Build the market structure skeleton from pivots:
```python
# Alternate between PH and PL to build the zigzag
# Remove lower PH between two PLs (not the highest)
# Remove higher PL between two PHs (not the lowest)
zigzag_points = build_zigzag(pivot_highs, pivot_lows)
```

---

## MODULE 3 — market_structure.py

### Purpose
Determine the current **market regime** based on pivot structure. This module answers: "What phase of the Wyckoff cycle are we in?"

### Market States (Enum)
```python
class MarketState(Enum):
    STRONG_UPTREND    = "strong_uptrend"      # HH + HL confirmed, momentum strong
    WEAK_UPTREND      = "weak_uptrend"         # HH + HL but momentum declining
    ACCUMULATION      = "accumulation"          # Range with springs (Wyckoff)
    RANGE             = "range"                 # Flat range, no clear trend
    DISTRIBUTION      = "distribution"          # Range with upthrusts (Wyckoff)
    WEAK_DOWNTREND    = "weak_downtrend"        # LL + LH but momentum declining
    STRONG_DOWNTREND  = "strong_downtrend"      # LL + LH confirmed, momentum strong
    BREAKOUT_LONG     = "breakout_long"         # Just broke above range resistance
    BREAKOUT_SHORT    = "breakout_short"        # Just broke below range support
    TRANSITION        = "transition"             # Unclear / changing
```

### Regime Detection Algorithm
```python
def detect_market_state(zigzag, prices, atr, keltner):
    """
    1. Check last 4 pivots for HH/HL/LL/LH sequence (Dow Theory)
    2. Measure momentum: impulse leg ATR vs pullback leg ATR
    3. Check for range: price within ±ATR of mean for n bars
    4. Check for spring/upthrust at range extremes
    5. Detect momentum divergences (MACD vs price)
    6. Return MarketState + confidence score (0-1)
    """
```

### Swing Analysis
```python
def analyze_swing_structure(pivots):
    """
    For each pair of consecutive swings, compute:
    - swing_magnitude: abs(high - low) of the swing
    - swing_duration: number of bars
    - swing_velocity: magnitude / duration
    Returns list of swings with metrics for trend health assessment
    """
    
    # Trend health metrics:
    # successive_impulses_shrinking = impulse_n < impulse_n-2 (warning)
    # pullback_depth_increasing = pullback_n > pullback_n-2 (warning)
    # momentum_divergence = price HH but swing velocity decreasing
```

### Pullback Classification
```python
def classify_pullback(pivots, current_index, trend_direction):
    """
    Returns:
    - PullbackType: SIMPLE, COMPLEX, FAILED
    - depth_pct: retracement depth as fraction of prior impulse
    - ab_cd_target: measured move objective for complex pullback
    - is_healthy: bool (depth < 61.8% AND momentum declining in pullback)
    """
```

---

## MODULE 4 — sr_zones.py

### Purpose
Identify and cluster **genuine support and resistance zones** using multiple methods. Critically: filter out random levels as Grimes warned.

### Zone Sources (ranked by significance)
```python
# Priority order (highest to lowest):
# 1. Major swing pivots on HTF (weekly/daily)
# 2. Levels with 3+ touches/tests
# 3. Previous day/week/month OHLC
# 4. Range extremes (highs/lows of trading ranges)
# 5. Broken S/R that flipped (former support now resistance)
# 6. Round numbers (psychological levels)
```

### Zone Clustering Algorithm
```python
def cluster_sr_zones(pivot_prices, atr, merge_distance=0.5):
    """
    1. Collect all pivot high/low prices
    2. Cluster prices within merge_distance × ATR of each other
    3. Each cluster becomes one zone:
       - zone_center = mean of clustered prices
       - zone_top = max of cluster
       - zone_bottom = min of cluster
       - zone_width = zone_top - zone_bottom
       - touch_count = number of pivots in cluster
       - zone_strength = touch_count × recency_weight × atr_normalized_size
    4. Filter: minimum touch_count = 2
    5. Sort by strength descending
    """
```

### Zone Validation (Grimes' Anti-Random Test)
```python
def validate_zone_significance(zone, price_history, atr):
    """
    Grimes warns: random levels look like real S/R.
    Test a zone for statistical significance:
    1. Count clean touches (price came within 0.5×ATR and reversed)
    2. Count bounces with >0.5×ATR reversal from zone
    3. Compute bounce_rate = bounces / total_approaches
    4. Compare to expected bounce_rate for random levels (~50%)
    5. Zone is significant if bounce_rate > 65% with n >= 5 approaches
    """
```

### Spring/Upthrust Detection
```python
def detect_spring(candle, zone, atr):
    """
    Wyckoff Spring (at support zone):
    - low < zone_bottom  (price penetrates support)
    - close > zone_center  (closes back above midpoint of zone)
    - (close - low) / (high - low) > 0.6  (close in upper 40% of bar)
    - duration: this pattern resolves within 1-3 bars
    Returns: spring_detected (bool), spring_strength (0-1)
    """

def detect_upthrust(candle, zone, atr):
    """
    Wyckoff Upthrust (at resistance zone):
    - high > zone_top  (price penetrates resistance)
    - close < zone_center  (closes back below midpoint of zone)
    - (high - close) / (high - low) > 0.6  (close in lower 40% of bar)
    Returns: upthrust_detected (bool), upthrust_strength (0-1)
    """
```

### Zone State Machine
```python
class ZoneState(Enum):
    INTACT    = "intact"        # Not yet tested
    TESTED    = "tested"        # Approached but held
    BROKEN    = "broken"        # Decisively violated
    FLIPPED   = "flipped"       # Broken S became R or broken R became S
    WEAKENED  = "weakened"      # 3+ tests; probability shifting to break
```

---

## MODULE 5 — channel_detector.py

### Purpose
Detect price channels (trend channels), which define the rate of trend and provide overextension reference points.

### Standard Trend Line (Two-Point)
```python
def fit_trend_line(pivots, direction='up', method='least_squares'):
    """
    For uptrend: linear regression through pivot LOWS
    For downtrend: linear regression through pivot HIGHS
    
    Returns:
    - slope: rate of change per bar
    - intercept
    - r_squared: fit quality (0-1)
    - current_value: price at the trend line today
    - distance_from_line: current close minus trend line value
    """
```

### Parallel Channel (Three-Point)
```python
def build_parallel_channel(pivot_highs, pivot_lows, trend_direction):
    """
    1. Draw base trend line through pivot lows (for uptrend)
    2. Draw parallel line through the highest pivot high
    3. Channel top = parallel line
    4. Channel bottom = base trend line
    
    Returns:
    - channel_top: function of bar index
    - channel_bottom: function of bar index  
    - channel_width: in ATR units
    - price_in_channel_pct: where is current price (0=bottom, 1=top)
    """
```

### Keltner Channel Overextension
```python
def check_keltner_position(close, kc_upper, kc_lower, kc_mid):
    """
    Returns:
    - position: normalized 0-1 (0=at lower band, 0.5=at mid, 1=at upper band)
    - is_overextended_up: close > kc_upper
    - is_overextended_down: close < kc_lower
    - bars_outside_channel: consecutive bars outside the channel
    - exhaustion_signal: bars_outside > 2 AND close moves back inside channel
    """
```

### Parabolic Climax Detection
```python
def detect_parabolic_climax(high, low, close, atr, keltner):
    """
    Climax conditions (all should be true for strong signal):
    1. Bar range > 2.0 × ATR(14)  — large range bar
    2. Close > keltner_upper (for bullish) or < keltner_lower (for bearish)
    3. Bar is the largest range bar in the last 20 bars
    4. Close position in bar:
       - Bullish climax: close_position < 0.3 (closed near LOW of large up bar)
       - Bearish climax: close_position > 0.7 (closed near HIGH of large down bar)
    5. Optional: MACD divergence on same bar
    
    Returns: climax_type (NONE, BULLISH_EXHAUSTION, BEARISH_EXHAUSTION), strength (0-1)
    """
```

### Rate of Trend Analysis
```python
def analyze_rate_of_trend(pivots, atr_history):
    """
    1. Fit trend line to each successive pair of pivots
    2. Compute slope for each trend segment
    3. If slopes are accelerating: trend is parabolic (approaching climax)
    4. If slopes are decelerating: trend losing momentum (watch for reversal)
    Returns: slope_series, acceleration, is_parabolic, is_decelerating
    """
```

---

## MODULE 6 — trend_analyzer.py

### Purpose
Comprehensive trend analysis combining all structural elements. This is the "intelligence layer" that synthesizes multiple signals into a coherent trend picture.

### Dow Theory Implementation
```python
def dow_theory_trend(pivots):
    """
    Using the last 4-6 pivots from zigzag:
    
    Uptrend conditions (all must be true):
    - pivot_high[n] > pivot_high[n-2] (Higher High)
    - pivot_low[n] > pivot_low[n-2]   (Higher Low)
    
    Downtrend conditions:
    - pivot_high[n] < pivot_high[n-2] (Lower High)
    - pivot_low[n] < pivot_low[n-2]   (Lower Low)
    
    Trend violation:
    - In uptrend: price closes below most recent Higher Low
    - In downtrend: price closes above most recent Lower High
    
    Returns: TrendDirection (UP, DOWN, FLAT), confidence (0-1)
    """
```

### Momentum Strength Assessment
```python
def assess_trend_strength(impulse_swings, pullback_swings):
    """
    Compare characteristics of impulse vs pullback swings:
    
    Health indicators:
    - impulse_vs_pullback_ratio = avg(impulse) / avg(pullback) 
      → should be > 1.5 for healthy trend
    - velocity_trend = slope of impulse velocities over time
      → positive = strengthening, negative = weakening
    - pullback_depth_trend = slope of pullback retracements over time
      → increasing depth = trend weakening
    
    Returns:
    - strength_score: 0-10
    - is_strengthening: bool
    - is_weakening: bool
    - warning_signs: list of strings
    """
```

### Multi-Timeframe Trend Alignment
```python
def mtf_trend_alignment(daily_state, hourly_state, m15_state):
    """
    Alignment scoring:
    - All three aligned (e.g., all UPTREND): score = 3, enter with full size
    - Two aligned, one neutral: score = 2, enter with 60% size
    - Two aligned, one opposing: score = 1, avoid or skip
    - All misaligned: score = 0, do not trade
    
    Returns: alignment_score, recommended_size_fraction, trade_direction
    """
```

### Trend Change Detection
```python
def detect_trend_change(pivots, current_price, atr, macd_divergence):
    """
    Change of character signals (score each):
    1. First failure to make new high/low (score +2)
    2. Counter-swing larger than previous same-direction swings (score +2)
    3. MACD divergence at new price extreme (score +1)
    4. Parabolic climax detected (score +2)
    5. Time-based: trend lasted > 150% of average trend duration (score +1)
    
    change_score >= 4: high probability of trend change
    change_score >= 2: monitor closely
    Returns: change_score, change_probability, last_safe_entry
    """
```

---

## MODULE 7 — signal_engine.py

### Purpose
Generate trade signals for all six trade types from Grimes' system. Each signal includes entry, stop, targets, and confidence score.

### Signal Data Class
```python
@dataclass
class TradeSignal:
    signal_type: str          # "pullback", "failure_test", "breakout", etc.
    direction: str            # "long" or "short"
    entry_price: float
    stop_price: float
    target_1: float           # First partial profit (1× risk)
    target_2: float           # Second target (pattern-based)
    target_3: float           # Extended target (measured move)
    risk_reward_1: float      # To target_1
    risk_reward_2: float      # To target_2
    confidence: float         # 0-1
    market_state: MarketState
    timeframe: str
    timestamp: datetime
    notes: list[str]
```

### Signal 1: Pullback (Simple and Complex)
```python
def scan_pullback_signal(market_state, pivots, sr_zones, indicators, atr):
    """
    Setup conditions:
    1. market_state in [STRONG_UPTREND, WEAK_UPTREND] (for longs)
    2. Last impulse was at least 1.5 × ATR
    3. Currently in a pullback (counter-trend)
    4. Pullback depth < 61.8% of prior impulse
    5. Pullback momentum declining (ATR of pullback bars < impulse bars)
    
    Entry triggers (choose one):
    A. Price at lower TF support / potential S/R zone
    B. Breakout of resistance within the pullback on LTF
    
    Stop: Below the pullback low minus 0.25 × ATR buffer
    
    Target 1: Prior swing high (for long)
    Target 2: Target 1 + (Target 1 - Entry) × 0.5 (extension)
    Target 3: AB=CD measured move objective (for complex pullback)
    """
```

### Signal 2: Failure Test (Spring/Upthrust)
```python
def scan_failure_test_signal(market_state, sr_zones, candles, atr):
    """
    Spring (long setup):
    1. Price has tested a defined support zone
    2. Spring detected: low < zone_bottom AND close > zone_center
    3. Bar closes in upper 40% of its range
    4. market_state not STRONG_DOWNTREND
    
    Entry: Above the high of the spring bar + 0.1 × ATR
    Stop: Below the spring low
    Target 1: 1× risk above entry
    Target 2: Opposite side of range
    
    Confidence modifiers:
    +0.2 if HTF market_state is ACCUMULATION
    +0.1 if first or second test of zone (not 4th+)
    -0.2 if trend is strongly against the direction
    """
```

### Signal 3: Breakout
```python
def scan_breakout_signal(market_state, sr_zones, pivots, candles, atr):
    """
    Pre-breakout setup conditions:
    1. Trading range with well-defined resistance
    2. Higher lows building into resistance (ascending triangle) OR
    3. Consolidation tight (ATR shrinking: volatility compression)
    
    Entry options:
    A. In the base (anticipatory): buy at support of the base, stop at base low
    B. On breakout bar: buy when high > resistance + 0.1×ATR
    C. On first pullback after breakout
    
    Breakout validation (to avoid fading a real breakout):
    - Breakout bar has range > 1.0 × ATR (momentum)
    - Breakout bar close in top 30% of range
    - First pullback after breakout: depth < 50% of breakout bar
    
    Target: Range height projected from breakout level
    Range_height = resistance - support
    Target = resistance + range_height
    """
```

### Signal 4: The Anti (Trend Termination)
```python
def scan_anti_signal(market_state, trend_change_score, pivots, indicators, atr):
    """
    Setup conditions:
    1. trend_change_score >= 3 (Grimes' change of character)
    2. A sharp counter-trend move has occurred (> 1.5 × avg_impulse)
    3. Counter-trend move creates new momentum extreme (MACD new high/low)
    4. Currently in first pullback of the counter-trend move
    
    Entry: On the pullback after the sharp counter-trend move
    Stop: Beyond the prior trend extreme (strict)
    
    Target 1: MMO of the first counter-trend thrust
    first_thrust = abs(extreme - start_of_thrust)
    Target = entry - first_thrust (for short Anti)
    
    Target 2: Prior major swing level (longer-term target)
    
    Risk management: These trades are countertrend.
    - Size at 50% of normal
    - Take 50% off at Target 1 (no exceptions)
    - Trail stop to breakeven after Target 1
    """
```

### Signal 5: Failed Breakout
```python
def scan_failed_breakout_signal(market_state, sr_zones, candles, atr):
    """
    Setup:
    1. A breakout occurred recently (within last 5 bars)
    2. First pullback after breakout is STRONG (> 75% retracement of breakout thrust)
    3. Price is now violating the breakout level
    
    Entry: When price closes back inside the range (through the former breakout level)
    Stop: Above the extreme of the failed breakout thrust (very strict)
    
    Target 1: Midpoint of the range
    Target 2: Opposite extreme of the range
    
    WARNING: This is the most dangerous trade type.
    - Reduce size to 50%
    - Do not add to position
    - Do not hold through strong counter-thrust
    """
```

### Confluence Scoring
```python
def score_signal_confluence(signal, htf_state, ltf_confirmation, sr_proximity, macd_divergence):
    """
    Each confirming factor adds to confidence:
    +0.15  HTF trend aligned with signal
    +0.10  LTF momentum confirming direction
    +0.10  Entry within 0.5×ATR of significant S/R zone
    +0.10  MACD divergence supporting the setup
    +0.05  Low volatility (ATR contracting before entry)
    +0.05  Multiple TF S/R confluence at the level
    -0.15  HTF trend opposing signal
    -0.10  Taking trade against parabolic trend
    
    Minimum confidence for trade: 0.45
    Ideal confidence: > 0.65
    """
```

---

## MODULE 8 — risk_manager.py

### Purpose
Position sizing, Kelly criterion, Monte Carlo analysis, and portfolio heat management.

### Position Sizing (Fixed Fractional)
```python
def calculate_position_size(account_equity, risk_fraction, entry, stop, tick_size=None):
    """
    risk_amount = account_equity × risk_fraction
    price_risk = abs(entry - stop)
    position_size = risk_amount / price_risk
    
    Constraints:
    - position_size must be whole shares/contracts (floor division)
    - Maximum position value: account_equity × max_position_pct (default 20%)
    - Adjust for minimum tick size if applicable
    
    Returns: position_size, actual_risk_amount, actual_risk_pct
    """
```

### Kelly Criterion Calculator
```python
def calculate_kelly_fraction(win_rate, avg_win_r, avg_loss_r=1.0):
    """
    b = avg_win_r / avg_loss_r  (win-to-loss ratio)
    p = win_rate
    q = 1 - p
    kelly = (b × p - q) / b
    
    Recommendations:
    - Full Kelly: maximum mathematically optimal bet
    - Half Kelly (0.5 × kelly): recommended for real trading
    - Quarter Kelly (0.25 × kelly): conservative; good for uncertain edges
    
    Returns: full_kelly, half_kelly, quarter_kelly
    """
```

### Monte Carlo Position Size Analysis
```python
def run_monte_carlo(trade_history, risk_fractions, n_simulations=1000, n_forward_trades=250):
    """
    For each risk_fraction in risk_fractions:
    1. Bootstrap sample n_forward_trades from trade_history (with replacement)
    2. Simulate account equity curve
    3. Record: terminal_value, max_drawdown, went_bankrupt
    4. Repeat n_simulations times
    
    Returns DataFrame with for each risk_fraction:
    - mean_terminal, median_terminal, std_terminal
    - p_bankruptcy (fraction that went to zero)
    - mean_max_drawdown
    - sharpe_ratio
    - coefficient_of_variation
    """
```

### Drawdown Management
```python
def check_drawdown_rules(current_equity, peak_equity, initial_equity):
    """
    Drawdown-based position sizing adjustment:
    
    drawdown_from_peak = (peak_equity - current_equity) / peak_equity
    
    Rules:
    0-5%:    Normal sizing (100%)
    5-10%:   Reduce to 75% of normal
    10-15%:  Reduce to 50% of normal
    15-20%:  Reduce to 25% of normal
    >20%:    Stop trading, review system
    
    Returns: size_multiplier, action_required (None, REDUCE, STOP)
    """
```

### R-Multiple Tracking
```python
def calculate_r_multiple(entry, exit_price, stop, direction='long'):
    """
    R = initial risk = abs(entry - stop)
    R_multiple = (exit_price - entry) / R  [for long]
    R_multiple = (entry - exit_price) / R  [for short]
    
    Track distribution of R_multiples to:
    - Compute expectancy = mean(R_multiples)
    - Identify system edge
    - Detect if edge is degrading over time
    """
```

---

## MODULE 9 — scanner.py (Main Orchestration)

### Purpose
Orchestrate all modules to scan instruments, detect setups, generate signals, size positions, and produce trade alerts.

### Scanner Flow
```python
def run_scan(instruments, timeframes, account_equity, risk_fraction=0.02):
    """
    For each instrument:
    1. Load OHLCV data for all timeframes
    2. Calculate all indicators (ATR, EMA, MACD, Keltner)
    3. Detect pivots on all timeframes
    4. Build market structure (zigzag, trend state)
    5. Identify S/R zones
    6. Build channel/trend lines
    7. Analyze trend health and change detection
    8. Run signal engine for all 6 trade types
    9. Score signal confluence
    10. Apply risk management rules
    11. Generate alerts for signals with confidence > 0.45
    """
```

### Adaptive Market Regime Logic
```python
def adapt_to_regime(market_state):
    """
    Maps market state to appropriate trade types and parameters:
    
    STRONG_UPTREND:
      - Active: Pullback (simple/complex), Breakout (continuation)
      - Avoid: Short signals, Failure tests at minor lows
      - Size: 100%
    
    WEAK_UPTREND:
      - Active: Complex pullbacks only, Anti setup
      - Caution: No simple pullbacks (too risky)
      - Size: 70%
    
    RANGE / ACCUMULATION:
      - Active: Failure tests (springs at support, upthrusts at resistance)
      - Active: Breakout entry in base (near range bottom)
      - Size: 70% (until breakout direction confirmed)
    
    DISTRIBUTION:
      - Active: Failure tests (upthrusts), Anti signals
      - Avoid: Long breakouts
      - Size: 70%
    
    STRONG_DOWNTREND:
      - Active: Pullback (short), Breakout (continuation down)
      - Size: 100%
    
    TRANSITION:
      - Active: None
      - Action: Wait for regime to clarify
      - Size: 0%
    """
```

---

## COMPLETE QUANTITATIVE RULES REFERENCE

### Pivot Detection Parameters
```
n_left = n_right = 3          # Standard sensitivity
n_left = n_right = 5          # Swing trader (less noise)
min_pivot_distance = 1.0×ATR  # Minimum distance between pivots
```

### Trend Structure Rules
```
HH_HL_confirmation = 2        # Minimum consecutive HH+HL to confirm uptrend
trend_violation = close < most_recent_HL   # For uptrend
trend_confirmed_change = 2 consecutive HL failures
```

### S/R Zone Parameters
```
merge_distance = 0.5×ATR      # Cluster pivots within this distance
min_touch_count = 2           # Minimum touches to form a zone
zone_significance_threshold = 0.65  # Bounce rate to call a zone significant
```

### Entry Rules by Trade Type
```
Pullback entry buffer = 0.1 × ATR above support
Breakout entry = resistance + 0.1 × ATR
Spring entry = high of spring bar + 0.1 × ATR
```

### Stop Placement
```
Default stop buffer = 0.25 × ATR beyond pattern extreme
Minimum stop distance = 0.75 × ATR (avoid stops too close to noise)
Maximum stop distance = 3.0 × ATR (avoid stops too wide; reduce size instead)
```

### Profit Targets
```
Target 1 = entry + 1.0 × initial_risk (take 1/3 to 1/2 here)
Target 2 = prior swing high/low
Target 3 = measured move objective (AB=CD or range projection)
Breakout target = resistance + (resistance - support)
```

### Position Sizing
```
Default risk fraction = 0.02 (2%)
Maximum risk fraction = 0.05 (5%)
Kelly fraction = (b×p - q) / b
Half-Kelly recommended = kelly_fraction × 0.5
Portfolio heat limit = 0.06 (6% total open risk)
```

### Climax Detection
```
climax_range_threshold = 2.5 × ATR(14)
keltner_period = 20
keltner_multiplier = 2.0
climax_close_position_threshold = 0.3 (close in bottom 30% for bullish climax)
```

### MACD Divergence
```
macd_settings = (12, 26, 9)
divergence_lookback = 20 bars  # Look for divergence within last 20 bars
min_price_swing = 0.5 × ATR   # Minimum price swing to check divergence
```

---

## VALIDATION CHECKLIST (Before Any Trade)

```
[ ] 1. HTF trend direction identified
[ ] 2. Market state / Wyckoff phase identified
[ ] 3. Signal type is appropriate for current market state
[ ] 4. Entry has clear invalidation point (stop is defined)
[ ] 5. Minimum 1× reward-to-risk to Target 1
[ ] 6. Signal confidence score > 0.45
[ ] 7. No conflicting HTF signals
[ ] 8. Account drawdown < 15% (normal sizing permitted)
[ ] 9. No parabolic trend (if trend trade) — avoid buying exhaustion
[ ] 10. Trade logged with all parameters before execution
```

---

## PERFORMANCE TRACKING SCHEMA

```python
# Log every trade with these fields:
trade_log = {
    "date": timestamp,
    "instrument": symbol,
    "timeframe": tf,
    "signal_type": type,
    "direction": long/short,
    "entry": price,
    "stop": price,
    "target1": price,
    "target2": price,
    "exit": price,
    "exit_reason": "target1/target2/stop/manual",
    "r_multiple": calculated,
    "market_state": state_at_entry,
    "confidence_score": 0-1,
    "notes": string
}

# Rolling statistics to monitor (recalculate every 20 trades):
metrics = {
    "win_rate": ...,
    "avg_r_multiple": ...,
    "expectancy": ...,  # Should be > 0.2R per trade
    "profit_factor": ...,  # Should be > 1.5
    "max_drawdown": ...,
    "consecutive_losses": ...,
    "sharpe_ratio": ...
}
```
