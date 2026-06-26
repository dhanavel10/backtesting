# The Art and Science of Technical Analysis — Complete Analysis
**Author:** Adam Grimes (2012, Wiley Trading)  
**Scope:** Market Structure, Price Action, Trading Strategies, Risk Management, Trader Psychology

---

## EXECUTIVE SUMMARY

This book presents a rigorous, empirically grounded approach to technical trading. Grimes repeatedly challenges conventional wisdom, arguing that **most published S/R levels are no better than random**, most chart patterns fail, and **a verifiable quantitative edge is non-negotiable**. It is divided into four parts:

1. **Foundation** — probability, the trader's edge, and the Wyckoff market cycle
2. **Market Structure** — trends, trading ranges, and the critical transitions between them
3. **Trading Strategies** — practical templates grounded in market structure
4. **The Individual Trader** — psychology, statistics, performance tracking

---

## PART I — THE FOUNDATION OF TECHNICAL ANALYSIS

### Chapter 1 — The Trader's Edge

#### What Is an Edge?
An **edge** is a systematic bias in the market that, if exploited consistently, produces positive risk-adjusted returns over a large sample size. Without a verifiable edge, consistent profit is mathematically impossible.

**Key insight:** Markets are very close to efficient. Most price movements are random. The burden of proof is on the trader to demonstrate that their method captures something genuinely non-random.

#### Two Forces That Drive All Price Action
Everything in market structure can be reduced to two competing forces:

| Force | Description | Expression |
|-------|-------------|------------|
| **Mean Reversion** | Prices tend to return toward a mean/average | Trading ranges, pullbacks, oscillations |
| **Range Expansion** | Prices can make sustained directional moves | Trends, breakouts, momentum bursts |

These two forces alternate cyclically. Understanding which regime the market is currently in is the foundation of all trade selection.

#### General Chart Reading Principles
- Look at **market structure first**, price action second
- **Market structure** = the pattern of relative highs/lows and the momentum behind moves (static)
- **Price action** = the dynamic process creating structure (must often be inferred)
- Both are **time-frame specific** — a trading range on one TF is a single bar on a higher TF
- Charts are tools, not reality — the bars are a compressed representation of millions of decisions

#### Pivot Points (Definition)
A **pivot high** is a bar whose high is higher than the highs of the n bars on either side.  
A **pivot low** is a bar whose low is lower than the lows of the n bars on either side.  
Pivots define the **skeleton of market structure** — all analysis is built on them.

#### Indicators — The Truth
Grimes is blunt: **indicators are derivatives of price.** They cannot contain more information than is already in the price bars. Their value is in:
- Quantifying momentum (MACD)
- Identifying overextension (Keltner bands)
- Trend direction confirmation (moving averages)

They should confirm, never lead. Never trade indicator signals without price-structure context.

---

### Chapter 2 — The Market Cycle and the Four Trades

#### Wyckoff's Market Cycle (4 Phases)

```
ACCUMULATION → MARKUP → DISTRIBUTION → MARKDOWN
     ↑                                       |
     └───────────────────────────────────────┘
```

| Phase | Description | Who Dominates | Price Action |
|-------|-------------|---------------|--------------|
| **Accumulation** | Smart money quietly buys at low prices | Informed buyers | Low volatility range, springs below support |
| **Markup** | Trend develops as public starts to notice | Trend followers | Impulse legs with pullbacks |
| **Distribution** | Smart money sells into public euphoria | Informed sellers | Range near highs, upthrusts above resistance |
| **Markdown** | Trend down as public finally realizes | Sellers/shorts | Impulse legs down with bounces |

**Wyckoff Springs:** A quick probe below support that immediately reverses — evidence of accumulation. Smart money absorbs the sell orders, trapping bears.

**Wyckoff Upthrusts:** A quick probe above resistance that immediately reverses — evidence of distribution. Smart money sells into the breakout, trapping bulls.

#### The Four Technical Trades
**All technical trades fall into exactly four categories:**

1. **Trend Continuation** — Trading with the trend, entering pullbacks or breakouts to new highs/lows
2. **Trend Termination** — Countertrend trades at exhaustion points; high reward/risk, lower probability
3. **Support/Resistance Holding** — Buying support or selling resistance that is expected to contain price
4. **Support/Resistance Failing** — Breakout or breakdown trades through S/R

**Critical rule:** Each trade type is appropriate for a specific phase of the market cycle. Applying the wrong trade to the wrong phase guarantees losses over time.

---

## PART II — MARKET STRUCTURE

### Chapter 3 — On Trends

#### The Fundamental Trend Pattern
The most important pattern in all of technical analysis:

```
Impulse → Pullback → Impulse → Pullback → Impulse
```

In an **uptrend:** series of higher highs (HH) and higher lows (HL)  
In a **downtrend:** series of lower lows (LL) and lower highs (LH)

Each trend leg (impulse) **fractal**: breaks into a 3-legged structure on the next lower TF.

#### Simple vs. Complex Pullbacks

| Type | Structure | Significance |
|------|-----------|--------------|
| **Simple pullback** | Single counter-trend leg (ABC) | Most common; enter at support of the leg |
| **Complex pullback** | Two counter-trend legs (ABCDE), i.e., a complete trend on the lower TF | Common in mature trends; AB≈CD measured move; wait for the second leg |

**AB = CD Rule (Measured Move Objective — MMO):**  
The second leg of a complex pullback often equals the first leg.  
`Price_target = C + (B - A)`  
This is one of the most reliable quantitative tools in the book.

#### Characteristics of Winning Pullbacks
- **Shallow** — does not retrace more than ~50-61.8% of the prior impulse
- **Low momentum** — counter-trend leg shows declining volume/energy vs impulse
- **No lower TF reversal structure** in the pullback (still looks like a correction, not a new trend)
- **Time compression** — pullback covers less time than the impulse that preceded it
- **No parabolic climax** immediately before the pullback (exhaustion invalidates re-entry)

#### Common Pullback Failure Patterns
1. **Flat pullback** — sideways consolidation after impulse (not counter-trend); suggests trend losing strength
2. **Sharp counter-trend momentum** — the pullback breaks hard against the trend; suggests trend change
3. **Failure near previous swing** — price reaches the prior swing high/low and stalls; take partial profits

#### Trend Analysis — Dow Theory Structure
**Uptrend confirmed:** Higher High AND Higher Low  
**Downtrend confirmed:** Lower Low AND Lower High  
**Trend break warning:** First swing that fails to make a new HH (or HL for uptrend)  
**Trend break confirmed:** Price trades below the most recent HL (uptrend) or above the most recent LH (downtrend)

```python
# Quantitative trend structure rules:
# Uptrend: HH > prev_HH AND HL > prev_HL
# Downtrend: LL < prev_LL AND LH < prev_LH  
# Trend weakening: swing_length[n] < swing_length[n-2] (losing momentum)
# Trend break: price crosses most recent HL (up) or LH (down)
```

#### Trend Lines

| Type | Construction | Significance |
|------|-------------|--------------|
| **Standard trend line** | Connects 2+ pivot lows (uptrend) or highs (downtrend) | Defines rate of trend; break = potential change |
| **Parallel trend line** | Parallel to standard TL, from opposite extremes | Defines channel; price at upper/lower boundary = key area |
| **Micro trend line** | 1-3 bars, very short-term | Used for intraday timing, momentum exhaustion |

**Rate of Trend:** Steeper trend lines break sooner. A series of successively steeper trend lines (each broken and redrawn) indicates **accelerating trend**. Acceleration precedes parabolic blow-off.

#### Character of Trend Legs — Quantifiable Signals

| Signal | What It Means |
|--------|---------------|
| Each successive impulse > prior impulse | Trend strengthening |
| Each successive impulse < prior impulse | Trend weakening — watch for reversal |
| Impulse = 1–2 bar sharp spike | Possible exhaustion/climax |
| 3+ consecutive bars closing on their highs | Counterintuitive **exhaustion** signal (statistically reverting) |
| Pullback retraces > 61.8% of prior impulse | Trend may be failing |
| Lower TF pullback within impulse leg | Normal; shows trend health |
| Lower TF trend within impulse leg | Abnormal; exhaustion possible |

---

### Chapter 4 — On Trading Ranges

#### Support and Resistance — The Critical Reality

> "Always attach the word **potential** to any support or resistance level."

**Key finding:** Almost any random line drawn on a chart will appear to function as support or resistance. The eye is fooled by the presence of lines. This is one of the most important and under-appreciated warnings in all of technical analysis.

**What makes S/R levels more likely to be real:**
- Very obvious on the chart (many participants watching)
- Visible pivots, especially on higher time frames
- Levels tested multiple times (but see caveat below)
- Extremes of large price spikes
- Previous day's high/low
- Round numbers (psychological levels)

#### The Dark Secret of S/R
In a random walk market, buying near support with a 10:1 reward/risk ratio gives you a win rate of ~9%. Expected value = **zero**. You cannot profit from reward/risk ratios alone — you need genuine non-random price action at the level.

#### Broken Support Becomes Resistance (and Vice Versa)
When a resistance level is broken, trapped shorts will buy back (cover) their positions if/when price returns to that level, providing support. This role reversal is real but often overstated.

#### Number of Tests
- **1-3 tests:** Level may still hold
- **3+ tests:** Probability shifts toward breakout — each test weakens the level
- **Grimes explicitly contradicts** the conventional wisdom that "more tests = stronger level"

#### Price Rejection — The Key Pattern
Successful S/R tests show **immediate, sharp price rejection**: price arrives at the level and immediately moves away. It should "not want to be there."  
Slow grinding into a level (price consolidating at S/R) is NOT rejection — it is pressure building toward a break.

#### Wyckoff Springs and Upthrusts (Quantified)
- **Spring:** Price dips below support quickly (1-2 bars) and immediately recovers above  
  `spring = close > support_level AND low < support_level AND (close - low) / (high - low) > 0.6`
- **Upthrust:** Price spikes above resistance briefly then falls back  
  `upthrust = close < resistance_level AND high > resistance_level AND (high - close) / (high - low) > 0.6`

These are the cleanest signal of accumulation/distribution at range extremes.

#### Trading Ranges as Structures
- Ranges are periods of price **consensus** — buyers and sellers agree on a general price range
- Always assume a trading range is a **continuation pattern** unless contradicted by higher TF structure
- Higher TF context provides directional bias for the eventual breakout

**Converging ranges (triangles):** Successively smaller swings zeroing in on a price. Rule: **do not fade the first breakout.**

**Ascending/Descending triangle:** Higher lows into horizontal resistance (ascending) or lower highs into horizontal support (descending) — pressure building against the flat side.

#### Stop Placement at S/R (Three Options)

| Stop Type | Location | Pros | Cons |
|-----------|----------|------|------|
| **A — Inside level** | Within the support zone, before the level | Avoids crowded stop area; no slippage | May be hit while pattern is still valid |
| **B — Just below level** | Tick beyond the most extreme point of S/R | Respects geometry | Visible to everyone; clustered orders |
| **C — Wide stop** | Random distance below | More room | Less precise; poor RR |

**Best practice:** Use Stop A or B; never place stops 2-3 ticks beyond a level (the market trades to the extreme if it goes there at all).

---

### Chapter 5 — Interfaces between Trends and Ranges

#### Breakout (Trading Range → Trend)
A breakout is a price move that breaks definitively out of a trading range, initiating a new trend.

**Characteristics of a real breakout:**
- Strong momentum through the level
- First pullback after breakout is **controlled** (not sharp counter-trend)
- First pullback looks like a pullback in a new trend
- Breakout failures (false breakouts) are **more common** than successful breakouts

**First pullback is critical:**
- Weak first pullback → healthy breakout  
- Strong first pullback counter-trend → probable failure  
- Price violating the breakout level again → probable failure

#### Trend → Trading Range
When an uptrend can no longer make new highs, it enters distribution. The range forms near the highs. Higher TF shows the range as a single consolidation before potential direction change.

#### Parabolic Climax and Exhaustion
A **parabolic move** is when price accelerates beyond the normal rate of trend:
- Successively steeper trend lines, each broken and redrawn
- Price moves far beyond the Keltner channel
- Single large-range bar (climax bar) after an extended run
- "Free bar" — price trades beyond the previous bar's range by a wide margin

**Post-climax behavior:**
1. Sharp counter-trend reaction
2. Consolidation (complex pullback on lower TF)
3. Either: continuation after working off the climax OR new trend begins

**Quantitative climax detection:**
```python
bar_range = high - low
atr = average_true_range(14)
is_climax = bar_range > 2.5 * atr AND position_relative_to_channel > 1.0
```

#### Last Gasp (Wyckoff Upthrust After Extended Consolidation)
After a well-watched topping pattern (head & shoulders, double top), trapped longs and new shorts create explosive conditions:
1. Price probes beyond the previous high (last gasp)
2. Lacks follow-through — quickly reverses
3. Trapped bulls + new shorts = strong downside momentum

This is the highest-probability trend termination pattern in the book.

#### Trend Reversal — Change of Character
Key: **The pattern of the established trend is broken first.**  
Signals of trend change:
- A downswing larger than any previous downswing (in uptrend)
- First HH failure (price fails to make new high in uptrend)
- Momentum divergence — price makes new high but momentum indicator makes lower high
- Structure of pullbacks changes (deeper, longer, more complex)

---

## PART III — TRADING STRATEGIES

### Chapter 6 — Practical Trading Templates

#### 1. Failure Test (Spring/Upthrust)
**Type:** Support/resistance holding  
**Setup:** Price has defined a clear S/R level; now makes a quick probe beyond the level  
**Entry:** When price reverses back through the level  
**Stop:** Beyond the extreme of the probe  
**Best in:** Range extremes, early in accumulation/distribution  

**Quantitative entry rule:**
```
Enter long when:
  - low < support_level
  - close > support_level  
  - (close - low) / bar_range > 0.5
  - This all happens within 1-2 bars
Stop: below the probe low
Target: opposite side of range or 1× initial risk
```

#### 2. Pullback Trade (Trend Continuation)
**Type:** Trend continuation  
**Setup:** Established trend (HH/HL or LL/LH structure), price in pullback  
**Entry (Option A):** At support of the pullback (near the bottom of the retracement)  
**Entry (Option B):** On breakout of lower TF resistance within the pullback  
**Stop:** Below the pullback low (for longs)  
**Target:** Prior swing high (for longs), or 1.5-2× initial risk  

**Quantitative setup conditions:**
```
1. Trend confirmed: n consecutive HH/HL (minimum 2)
2. Pullback shallow: retracement < 61.8% of prior impulse
3. Pullback momentum declining: ATR of pullback bars < ATR of impulse bars
4. Entry timing: pullback near prior support / lower TF breakout
```

#### 3. Complex Pullback Trade
**Extension of pullback trade where:**
- Simple pullback entry failed (you took a small loss)
- Market continues into a second counter-trend leg
- AB = CD measures the approximate termination of the second leg
- Stop is now more precisely defined (below second leg low)
- Entry can be tighter; move out of complex pullback tends to be stronger

#### 4. The Anti (Trend Termination)
**Type:** Trend termination / countertrend  
**Setup:**
1. Evidence of trend exhaustion (loss of momentum, overextension, double top/bottom)
2. A sharp counter-trend move showing "change of character"
**Entry:** First pullback after the sharp counter-trend move  
**Stop:** Beyond the trend extreme (most conservative) or beyond the pullback  
**Target:** Measured move of the first counter-trend thrust OR prior swing  

**MMO for Anti:**
```
counter_thrust = abs(high_A - low_A)
target = entry_price - counter_thrust (for short Anti)
or
target = high_of_consolidation_B  (first target)
```

#### 5. Breakout Trade
**Type:** S/R failing  
**Setup:** Defined trading range; price pressing against one side with higher lows (ascending) into resistance  
**Entry (A):** In the base before the breakout (anticipatory)  
**Entry (B):** On the actual breakout bar  
**Entry (C):** On the first pullback after breakout  
**Stop:** Below the breakout level (for longs)  
**Target:** Width of the prior range projected from the breakout point  

**Range projection:**
```
range_height = resistance - support
target = resistance + range_height  (for upside breakout)
```

#### 6. Failed Breakout Trade
**Type:** S/R holding after apparent failure  
**Setup:**
- Breakout occurs with weak momentum
- First pullback after breakout shows strong counter-trend momentum
- Price violates the breakout level

**Entry:** After the pattern of breakout failure is clear (pullback turns counter-trend)  
**Stop:** Above the extreme of the failed breakout thrust  
**Danger:** Most dangerous trade — extreme volatility and tail risk. Strict discipline required.

---

### Chapter 7 — Tools for Confirmation

#### Moving Averages — The "Still Center"

**No single period is better than another** — all moving averages are roughly equivalent statistically. Their value is structural, not mechanical.

**Valid uses:**
1. **Trend indicator** — slope of MA confirms trend direction; price consistently on one side = trending
2. **Avoid equilibrium** — price chopping around flat MA = no edge; skip
3. **Pullback reference** — MA shows the "average" — don't buy if overextended from it; wait for pullback to MA area

**Invalid uses:**
- Moving average as support/resistance (no statistical basis)
- Crosses as buy/sell signals (no consistent edge found)

#### Channels — Keltner/Price Channels

**Keltner Channel construction:**
```
Middle = EMA(20)
Upper = Middle + multiplier × ATR(20)
Lower = Middle - multiplier × ATR(20)
Standard multiplier = 2.0 (some use 1.5 or 2.5)
```

**Interpretation:**
- Price outside the channel = **overextension / emotional extreme**
- Strong trend: price consistently hugs one band (never reaching the other)
- Climax bar closing beyond the channel = exhaustion signal
- Price between bands in flat MA = range/equilibrium environment

#### MACD — Momentum Divergence Detection

**Standard settings:** 12/26/9 (fast EMA / slow EMA / signal EMA)

**Key use:** Detecting momentum divergences, not crossovers  

**Bearish divergence:** Price makes higher high, MACD makes lower high → trend losing momentum  
**Bullish divergence:** Price makes lower low, MACD makes higher low → selling pressure declining  

**Quantitative divergence rule:**
```python
# Bearish momentum divergence
price_HH = high[n] > high[n-k]  # new price high
macd_HL = macd[n] < macd[n-k]   # MACD fails to confirm
bearish_divergence = price_HH AND macd_HL
```

**Important:** MACD divergences by themselves are not trade signals. They must be placed in the context of price structure (overextension, change of character, etc.).

#### Multiple Time Frame Analysis

**The relationship between time frames is fractal:**
- The pullback on the trading TF = full trend on the lower TF
- The impulse on the trading TF = single bar on the higher TF

**Three TF approach:**
1. **Higher TF (HTF):** Defines the trend bias and key S/R levels
2. **Trading TF (TTF):** Setup and entry conditions identified
3. **Lower TF (LTF):** Entry timing (breakout within pullback, momentum shift)

**Most powerful pattern:** Entry at lower TF momentum exhaustion (selling climax) within a higher TF pullback.

---

### Chapter 8 — Trade Management

#### Initial Stop Placement
1. **Pattern-based:** Stop at the point that proves the trade wrong (geometry of the pattern)
2. **ATR-based:** Stop = Entry ± k × ATR (common: k = 1.5-2.0)
3. **Always define the stop BEFORE entry** — not negotiable

#### Profit Targets

| Target Type | Rule | Best For |
|-------------|------|----------|
| **First target** | 1× initial risk | Most trades; take 33-50% off |
| **Pattern target** | Prior swing high/low | Pullback trades |
| **Measured move** | Range height projected from breakout | Breakout trades |
| **Channel extreme** | Upper/lower Keltner | Trend trades |
| **Fibonacci** | 127.2%, 161.8% extensions | Complex pullbacks |

**Scaling out:** Take 1/3 at 1× risk, hold remainder with trailing stop. Never let a winning trade become a large loser.

#### Active Trade Management (Trailing Stops)

1. **Moving average trailing stop:** Exit when price closes below the 20 EMA (aggressive) or 50 EMA (conservative)
2. **Trend line trailing stop:** Draw successive trend lines; exit on close below current trend line
3. **Parabolic SAR:** Algorithmic trailing stop that accelerates as trend extends
4. **Swing point trailing stop:** Move stop to below each successive higher low (for longs)

#### Position Management Rules
- Take partial profits as the market offers them — this reduces open risk
- If a trade does not act as expected immediately, it is probably wrong — consider reducing
- Catastrophic losses come from adding to losing countertrend positions

---

### Chapter 9 — Risk Management

#### The Mathematics of Risk

**Expected Value (EV):**
```
EV = (Win% × Avg_Win) - (Loss% × Avg_Loss)
   = (p × W) - ((1-p) × L)
```

For a trade to be worth taking: **EV > 0** and **EV / commission_cost > threshold**

**Key insight:** In a random walk, a 10:1 reward/risk gives you a ~9% win rate — EV = 0. You need non-random edge to profit.

#### Fixed Fractional Position Sizing
```
Position_size = (Account_equity × risk_fraction) / (Entry - Stop)
# Example: 2% of $100,000 account
# risk_fraction = 0.02
# Account_equity = $100,000
# Risk per trade = $2,000
# If Stop is $1.50 away, Position_size = $2,000 / $1.50 = 1,333 shares
```

**Results from Monte Carlo analysis (250 trades, 45% win rate, 2:1 R multiple):**
- 2% risk: Mean terminal = $149K, CoV = 0.23, Bankruptcy = 0%
- 5% risk: Mean terminal = $221K, CoV = 0.39, Bankruptcy = 2.2%  
- 10% risk: Mean terminal = $314K, CoV = 0.64, Bankruptcy = 17.6%
- 25% risk: Mean terminal = $463K, CoV = 1.12, Bankruptcy = 47.7%

**Recommendation:** 1-2% fixed fractional for consistency; never exceed 5% per trade.

#### Kelly Criterion
The mathematically optimal fraction to bet:
```
Kelly_fraction = (b × p - q) / b
where:
  b = ratio of win to loss (e.g., 2 for 2:1 R multiple)
  p = win probability
  q = 1 - p (loss probability)

Example: 45% win rate, 2:1 R multiple:
Kelly = (2 × 0.45 - 0.55) / 2 = (0.90 - 0.55) / 2 = 0.175 = 17.5%
```

**Half-Kelly (8.75% in the example above) is commonly recommended** — same asymptotic growth rate with far less variance.

**Warning:** Kelly assumes fixed, known win/loss amounts. Real trading has variable outcomes — Kelly often overestimates the safe fraction. Use fractional Kelly (0.25× to 0.5× Kelly).

#### Monte Carlo Simulation
Run 1,000+ simulations of your trade history with randomized sequence to understand:
- Distribution of terminal wealth
- Probability of drawdown exceeding threshold
- Max adverse excursion
- Whether returns are lognormally distributed (fixed fractional) or normally distributed (fixed dollar)

#### Practical Risk Rules
1. **1-2% risk per trade** from account equity
2. **Maximum 5-6% portfolio heat** at any time (total open risk across all positions)
3. **If down 10%** from equity peak: reduce size by 50%
4. **If down 20%:** Stop trading, review the system
5. **Diversification by non-correlation** — not by holding many correlated trades simultaneously

---

## PART IV — THE INDIVIDUAL TRADER

### Chapter 11 — The Trader's Mind

#### Cognitive Biases (Quantitatively Important)
| Bias | Effect on Trading | Mitigation |
|------|------------------|------------|
| **Confirmation bias** | Seek information confirming existing position | Actively look for counter-evidence |
| **Anchoring** | Over-weight the entry price | Focus on current market structure, not your cost |
| **Loss aversion** | Losses feel ~2× as bad as equal gains | Pre-commit to stops; no exceptions |
| **Recency bias** | Over-weight recent results | Track rolling statistics over 100+ trades |
| **Pattern recognition** | See patterns in random data | Validate every "pattern" statistically |
| **Random reinforcement** | Intermittent reward creates superstitious behavior | Understand expected value; embrace losing trades |

#### Random Reinforcement Problem
Markets randomly reward bad behavior and randomly punish good behavior in the short run. This creates **random reinforcement schedules** that are the most powerful conditioning mechanism known to psychology (slot machines work the same way). The only defense is a large enough sample size and statistical evidence.

#### Flow State
Top traders enter a "flow state" — complete absorption in the task with no self-consciousness. This is enabled by:
- Complete internalization of rules (no conscious calculation needed)
- Total focus on the present moment
- Emotional neutrality toward wins and losses

---

### Chapter 12 — Becoming a Trader

#### Performance Statistics (Must Track)

| Metric | Formula | Target |
|--------|---------|--------|
| **Win rate** | Wins / Total Trades | Context-dependent |
| **Average Win** | Sum of wins / Win count | > 1.5× Average Loss |
| **Average Loss** | Sum of losses / Loss count | Fixed by stop |
| **Expectancy** | (Win% × Avg_Win) - (Loss% × Avg_Loss) | Positive and consistent |
| **Profit Factor** | Gross Profit / Gross Loss | > 1.5 is good |
| **Sharpe Ratio** | (Return - Risk_free) / Std_dev | > 1.0 |
| **Max Drawdown** | Peak to trough equity decline | Monitor; >15% is concerning |
| **R-multiple distribution** | Distribution of (Profit or Loss) / Initial_risk | Should be positively skewed |

#### R-Multiple Analysis
Standardize all trades by their initial risk (R):
```
R_multiple = (Exit_price - Entry_price) / (Entry_price - Stop_price)  [for longs]
```
Plot the distribution of R-multiples. A system with edge will show:
- Mean R > 0
- Positive skew (some big winners, small losers)
- Consistency across market regimes

---

## APPENDIX MATERIAL

### Appendix B — Moving Averages in Depth

**SMA vs EMA:**
- **SMA** — equal weight to all bars in the period; lags more; susceptible to "drop-off effect" (old data leaving the window)
- **EMA** — exponential weighting (more weight on recent data); responds faster; never fully forgets old data

**EMA formula:**
```
EMA[t] = α × Price[t] + (1 - α) × EMA[t-1]
α = 2 / (n + 1)  where n = period
```

**EMA lag approximation:**
```
Lag ≈ (n - 1) / 2  bars
```

**MACD construction:**
```
MACD_line = EMA(12) - EMA(26)
Signal_line = EMA(9) of MACD_line
Histogram = MACD_line - Signal_line
```

---

## QUANTITATIVE SUMMARY TABLE

| Concept | Formula / Rule | Application |
|---------|---------------|-------------|
| AB=CD (MMO) | target = C + (B - A) | Complex pullback target |
| Range projection | target = BO_level + (R - S) | Breakout target |
| Retracement depth | retrace < 61.8% of prior impulse | Pullback validity |
| ATR-based stop | stop = entry ± 1.5 × ATR(14) | Stop placement |
| Keltner bands | EMA(20) ± 2.0 × ATR(20) | Overextension detection |
| Expected value | EV = p×W - (1-p)×L | Trade selection |
| Kelly fraction | (b×p - q) / b | Max position size |
| Fixed fractional | risk = equity × 0.02 | Default sizing |
| Climax bar | range > 2.5 × ATR | Exhaustion signal |
| Divergence | price HH + MACD HL | Momentum failure |
| Spring detection | low < S AND close > S AND close > midpoint | Wyckoff spring |
| Upthrust detection | high > R AND close < R AND close < midpoint | Wyckoff upthrust |

---

## CORE PHILOSOPHICAL PRINCIPLES (GRIMES)

1. **Markets are random most of the time.** Accept this and only trade the non-random moments.
2. **Never trade without a defined edge.** If you cannot explain your edge quantitatively, you don't have one.
3. **Support and resistance are probabilistic, not deterministic.** Always say "potential" S/R.
4. **Failed breakouts are more common than successful ones.** Plan for both.
5. **Simple > Complex.** A few well-understood tools beat many poorly understood ones.
6. **Risk first.** Define your stop before you define your entry.
7. **Track everything.** You cannot manage what you do not measure.
8. **Context is everything.** No pattern works in isolation — it must fit the market structure.
