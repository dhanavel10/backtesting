"""
Nifty 50 - BOS & CHoCH Detector
Timeframe : 5-minute bars, last 60 days
Output    : nifty_bos_choch_report.html
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys

# ─────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────
def fetch_data(ticker="^NSEI", period="60d", interval="5m"):
    print(f"[→] Downloading {ticker} ({interval}, {period}) …")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        print("  ERROR: No data returned. Check your internet connection.")
        sys.exit(1)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    # Convert to IST if UTC
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata")
    df.index = df.index.tz_localize(None)
    print(f"  ✓ {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


# ─────────────────────────────────────────────
# 2. SWING DETECTION  (zigzag pivot)
# ─────────────────────────────────────────────
def detect_swings(df, left=5, right=5):
    """
    Pivot-based swing high / low detection.
    Returns two boolean Series: swing_high, swing_low
    """
    hi = df["High"]
    lo = df["Low"]
    n  = len(df)
    sh = pd.Series(False, index=df.index)
    sl = pd.Series(False, index=df.index)

    for i in range(left, n - right):
        win_hi = hi.iloc[i - left: i + right + 1]
        win_lo = lo.iloc[i - left: i + right + 1]
        if hi.iloc[i] == win_hi.max():
            sh.iloc[i] = True
        if lo.iloc[i] == win_lo.min():
            sl.iloc[i] = True
    return sh, sl


# ─────────────────────────────────────────────
# 3. BOS / CHoCH DETECTION
# ─────────────────────────────────────────────
def detect_bos_choch(df, sh, sl):
    """
    Market Structure Rules
    ─────────────────────
    BOS  (Break of Structure)  – continuation:
      • Bullish BOS : price closes above last confirmed swing HIGH  (uptrend continuation)
      • Bearish BOS : price closes below last confirmed swing LOW   (downtrend continuation)

    CHoCH (Change of Character)  – reversal:
      • Bullish CHoCH : in a downtrend, price closes above last swing HIGH
      • Bearish CHoCH : in an uptrend,  price closes below last swing LOW

    Trend is tracked via the sequence of swing highs/lows:
      HH + HL = uptrend  |  LH + LL = downtrend
    """
    events = []
    close  = df["Close"].values
    dates  = df.index
    sh_idx = np.where(sh.values)[0]
    sl_idx = np.where(sl.values)[0]

    # ── build an ordered list of confirmed pivots
    pivots = []
    for i in sh_idx:
        pivots.append({"idx": i, "type": "SH", "price": df["High"].iloc[i], "date": dates[i]})
    for i in sl_idx:
        pivots.append({"idx": i, "type": "SL", "price": df["Low"].iloc[i],  "date": dates[i]})
    pivots.sort(key=lambda x: x["idx"])

    # ── simple trend tracker
    trend = None          # "up" | "down" | None
    last_sh = None        # last confirmed swing high dict
    last_sl = None        # last confirmed swing low  dict
    prev_sh = None
    prev_sl = None
    broken_levels = set() # track already broken levels to avoid re-firing

    # iterate bar-by-bar
    for bar_i in range(1, len(df)):
        c = close[bar_i]
        dt = dates[bar_i]

        # update confirmed pivots that are to the LEFT of current bar
        for p in pivots:
            if p["idx"] < bar_i:
                if p["type"] == "SH":
                    if last_sh is None or p["idx"] > last_sh["idx"]:
                        prev_sh = last_sh
                        last_sh = p
                elif p["type"] == "SL":
                    if last_sl is None or p["idx"] > last_sl["idx"]:
                        prev_sl = last_sl
                        last_sl = p

        if last_sh is None or last_sl is None:
            continue

        # ── determine current trend from pivot sequence
        def classify_trend():
            nonlocal trend
            if prev_sh and prev_sl:
                if last_sh["price"] > prev_sh["price"] and last_sl["price"] > prev_sl["price"]:
                    trend = "up"
                elif last_sh["price"] < prev_sh["price"] and last_sl["price"] < prev_sl["price"]:
                    trend = "down"
            elif prev_sh and last_sl:
                if last_sl["price"] > prev_sh["price"]:
                    trend = "up"
            elif prev_sl and last_sh:
                if last_sh["price"] < prev_sl["price"]:
                    trend = "down"

        classify_trend()

        # ── check for breaks
        sh_level = last_sh["price"]
        sl_level = last_sl["price"]
        sh_key   = (last_sh["idx"], "SH")
        sl_key   = (last_sl["idx"], "SL")

        # BULLISH break: close above last swing high
        if c > sh_level and sh_key not in broken_levels:
            broken_levels.add(sh_key)
            if trend == "down":
                etype = "CHoCH"
                direction = "Bullish"
                desc = ("Price broke above the last Lower High — "
                        "downtrend structure is disrupted; potential trend reversal upward.")
            else:
                etype = "BOS"
                direction = "Bullish"
                desc = ("Price closed above the previous Higher High — "
                        "uptrend structure confirmed; continuation expected.")
            events.append({
                "date": str(dt)[:19],
                "bar_idx": bar_i,
                "type": etype,
                "direction": direction,
                "level": round(float(sh_level), 2),
                "close": round(float(c), 2),
                "trend_at_break": trend if trend else "undefined",
                "description": desc,
                "swing_date": str(last_sh["date"])[:19],
            })

        # BEARISH break: close below last swing low
        if c < sl_level and sl_key not in broken_levels:
            broken_levels.add(sl_key)
            if trend == "up":
                etype = "CHoCH"
                direction = "Bearish"
                desc = ("Price broke below the last Higher Low — "
                        "uptrend structure is disrupted; potential trend reversal downward.")
            else:
                etype = "BOS"
                direction = "Bearish"
                desc = ("Price closed below the previous Lower Low — "
                        "downtrend structure confirmed; continuation expected.")
            events.append({
                "date": str(dt)[:19],
                "bar_idx": bar_i,
                "type": etype,
                "direction": direction,
                "level": round(float(sl_level), 2),
                "close": round(float(c), 2),
                "trend_at_break": trend if trend else "undefined",
                "description": desc,
                "swing_date": str(last_sl["date"])[:19],
            })

    return events


# ─────────────────────────────────────────────
# 4. COMPUTE SUMMARY STATS
# ─────────────────────────────────────────────
def compute_stats(df, events):
    total_bos   = sum(1 for e in events if e["type"] == "BOS")
    total_choch = sum(1 for e in events if e["type"] == "CHoCH")
    bull_bos    = sum(1 for e in events if e["type"] == "BOS"   and e["direction"] == "Bullish")
    bear_bos    = sum(1 for e in events if e["type"] == "BOS"   and e["direction"] == "Bearish")
    bull_choch  = sum(1 for e in events if e["type"] == "CHoCH" and e["direction"] == "Bullish")
    bear_choch  = sum(1 for e in events if e["type"] == "CHoCH" and e["direction"] == "Bearish")

    price_change   = df["Close"].iloc[-1] - df["Close"].iloc[0]
    pct_change     = (price_change / df["Close"].iloc[0]) * 100
    overall_trend  = "Bullish" if price_change > 0 else "Bearish"

    # daily event counts
    ev_df = pd.DataFrame(events)
    if not ev_df.empty:
        ev_df["date_only"] = pd.to_datetime(ev_df["date"]).dt.date
        daily = ev_df.groupby("date_only").size().reset_index(name="count")
        max_day      = daily.loc[daily["count"].idxmax()]
        most_active  = str(max_day["date_only"])
        most_active_n = int(max_day["count"])
    else:
        most_active   = "N/A"
        most_active_n = 0

    # last 5 events
    last_events = events[-5:] if len(events) >= 5 else events

    return {
        "total_bars"    : len(df),
        "date_from"     : str(df.index[0])[:19],
        "date_to"       : str(df.index[-1])[:19],
        "open_price"    : round(float(df["Open"].iloc[0]), 2),
        "close_price"   : round(float(df["Close"].iloc[-1]), 2),
        "high_price"    : round(float(df["High"].max()), 2),
        "low_price"     : round(float(df["Low"].min()), 2),
        "price_change"  : round(float(price_change), 2),
        "pct_change"    : round(float(pct_change), 2),
        "overall_trend" : overall_trend,
        "total_bos"     : total_bos,
        "total_choch"   : total_choch,
        "bull_bos"      : bull_bos,
        "bear_bos"      : bear_bos,
        "bull_choch"    : bull_choch,
        "bear_choch"    : bear_choch,
        "most_active"   : most_active,
        "most_active_n" : most_active_n,
        "last_events"   : last_events,
    }


# ─────────────────────────────────────────────
# 5. BUILD MINI SPARKLINE  (SVG path)
# ─────────────────────────────────────────────
def build_sparkline(closes, width=300, height=60):
    mn, mx = min(closes), max(closes)
    rng = mx - mn if mx != mn else 1
    pts = []
    for i, v in enumerate(closes):
        x = round(i / (len(closes) - 1) * width, 2)
        y = round(height - ((v - mn) / rng) * height, 2)
        pts.append(f"{x},{y}")
    color = "#22c55e" if closes[-1] >= closes[0] else "#ef4444"
    path  = "M " + " L ".join(pts)
    return f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="sparkline"><path d="{path}" fill="none" stroke="{color}" stroke-width="2"/></svg>'


# ─────────────────────────────────────────────
# 6. BUILD CANDLESTICK CHART  (lightweight SVG)
# ─────────────────────────────────────────────
def build_candle_chart(df, events, max_bars=200):
    """Render last N bars as a compact SVG candlestick chart with BOS/CHoCH markers."""
    sub = df.iloc[-max_bars:].reset_index()
    closes = sub["Close"].tolist()
    opens  = sub["Open"].tolist()
    highs  = sub["High"].tolist()
    lows   = sub["Low"].tolist()

    W, H   = 900, 340
    PAD_L, PAD_R, PAD_T, PAD_B = 60, 20, 20, 40
    chart_w = W - PAD_L - PAD_R
    chart_h = H - PAD_T - PAD_B

    all_hi  = max(highs)
    all_lo  = min(lows)
    rng     = all_hi - all_lo or 1

    def y(price):
        return PAD_T + chart_h - (price - all_lo) / rng * chart_h

    n      = len(sub)
    bar_w  = max(1, chart_w / n)
    cw     = max(1, bar_w * 0.6)

    # ── candles
    candle_svg = []
    for i in range(n):
        x   = PAD_L + i * bar_w + bar_w / 2
        o, c, hi, lo = opens[i], closes[i], highs[i], lows[i]
        col = "#22c55e" if c >= o else "#ef4444"
        y_top  = y(max(o, c))
        y_bot  = y(min(o, c))
        body_h = max(1, y_bot - y_top)
        candle_svg.append(
            f'<line x1="{x:.1f}" y1="{y(hi):.1f}" x2="{x:.1f}" y2="{y(lo):.1f}" stroke="{col}" stroke-width="1"/>'
            f'<rect x="{x - cw/2:.1f}" y="{y_top:.1f}" width="{cw:.1f}" height="{body_h:.1f}" fill="{col}" rx="0.5"/>'
        )

    # ── BOS / CHoCH markers on last max_bars
    start_idx = len(df) - max_bars
    marker_svg = []
    for ev in events:
        bi = ev["bar_idx"] - start_idx
        if 0 <= bi < n:
            x   = PAD_L + bi * bar_w + bar_w / 2
            col = "#22c55e" if ev["direction"] == "Bullish" else "#ef4444"
            lbl = ev["type"]
            dir_arrow = "▲" if ev["direction"] == "Bullish" else "▼"
            yp  = y(ev["close"])
            off = -14 if ev["direction"] == "Bullish" else 14
            marker_svg.append(
                f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{H - PAD_B}" '
                f'stroke="{col}" stroke-width="1" stroke-dasharray="3,3" opacity="0.5"/>'
                f'<text x="{x:.1f}" y="{yp + off:.1f}" text-anchor="middle" '
                f'font-size="8" font-weight="700" fill="{col}" font-family="monospace">'
                f'{lbl} {dir_arrow}</text>'
            )

    # ── price axis (5 ticks)
    axis_svg = []
    for tick in np.linspace(all_lo, all_hi, 5):
        yp = y(tick)
        axis_svg.append(
            f'<line x1="{PAD_L}" y1="{yp:.1f}" x2="{W - PAD_R}" y2="{yp:.1f}" '
            f'stroke="#334155" stroke-width="0.5" stroke-dasharray="2,4"/>'
            f'<text x="{PAD_L - 5}" y="{yp + 3:.1f}" text-anchor="end" '
            f'font-size="9" fill="#94a3b8" font-family="monospace">{tick:,.0f}</text>'
        )

    inner = "\n".join(axis_svg + candle_svg + marker_svg)
    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{W}px;display:block;background:#0f172a;border-radius:8px">'
        f'{inner}</svg>'
    )


# ─────────────────────────────────────────────
# 7. BUILD DAILY BREAKDOWN TABLE DATA
# ─────────────────────────────────────────────
def daily_breakdown(events):
    if not events:
        return []
    ev_df = pd.DataFrame(events)
    ev_df["date_only"] = pd.to_datetime(ev_df["date"]).dt.date
    grp   = ev_df.groupby("date_only")
    rows  = []
    for date, g in grp:
        rows.append({
            "date"       : str(date),
            "total"      : len(g),
            "bos"        : int((g["type"] == "BOS").sum()),
            "choch"      : int((g["type"] == "CHoCH").sum()),
            "bull"       : int((g["direction"] == "Bullish").sum()),
            "bear"       : int((g["direction"] == "Bearish").sum()),
            "first_event": g.iloc[0]["type"] + " " + g.iloc[0]["direction"],
            "last_event" : g.iloc[-1]["type"] + " " + g.iloc[-1]["direction"],
        })
    return rows


# ─────────────────────────────────────────────
# 8. HTML REPORT
# ─────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Nifty 50 | BOS &amp; CHoCH Report</title>
<style>
:root{
  --bg:#060d1a;--surface:#0f172a;--surface2:#1e293b;--surface3:#273447;
  --border:#1e3a5f;--text:#e2e8f0;--muted:#64748b;--accent:#3b82f6;
  --bull:#22c55e;--bear:#ef4444;--warn:#f59e0b;--choch:#a855f7;
  --bos:#3b82f6;
  --font-mono:'JetBrains Mono','Fira Code','Cascadia Code',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;
     font-size:14px;line-height:1.6;min-height:100vh}

/* ── HEADER ── */
.header{background:linear-gradient(135deg,#0a1628 0%,#0f2044 50%,#0a1628 100%);
  border-bottom:1px solid var(--border);padding:28px 32px 24px;position:relative;overflow:hidden}
.header::before{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 60% 80% at 70% 50%,rgba(59,130,246,.08) 0%,transparent 70%)}
.header-inner{max-width:1200px;margin:0 auto;position:relative}
.badge{display:inline-flex;align-items:center;gap:6px;background:rgba(59,130,246,.15);
  border:1px solid rgba(59,130,246,.3);border-radius:20px;padding:4px 12px;
  font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:var(--accent);margin-bottom:12px}
.badge-dot{width:6px;height:6px;background:var(--accent);border-radius:50%;
  animation:pulse 1.8s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
h1{font-size:28px;font-weight:700;letter-spacing:-.02em;line-height:1.2;
  background:linear-gradient(135deg,#e2e8f0 0%,#94a3b8 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
h1 span{background:linear-gradient(135deg,var(--accent),#60a5fa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.subtitle{color:var(--muted);margin-top:6px;font-size:13px}
.meta-row{display:flex;gap:24px;margin-top:16px;flex-wrap:wrap}
.meta-item{display:flex;flex-direction:column;gap:2px}
.meta-label{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.meta-value{font-size:13px;font-weight:600;color:var(--text);font-family:var(--font-mono)}

/* ── LAYOUT ── */
.main{max-width:1200px;margin:0 auto;padding:28px 32px 60px}

/* ── PRICE STRIP ── */
.price-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px;background:var(--border);border:1px solid var(--border);border-radius:10px;
  overflow:hidden;margin-bottom:28px}
.price-cell{background:var(--surface);padding:14px 18px}
.pc-label{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.pc-value{font-size:18px;font-weight:700;font-family:var(--font-mono);margin-top:2px}
.bull{color:var(--bull)}.bear{color:var(--bear)}.neutral{color:var(--text)}

/* ── SUMMARY CARDS ── */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:14px;margin-bottom:28px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:18px 20px;position:relative;overflow:hidden;transition:border-color .2s}
.card:hover{border-color:var(--accent)}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.card.c-bos::before{background:linear-gradient(90deg,var(--bos),#60a5fa)}
.card.c-choch::before{background:linear-gradient(90deg,var(--choch),#c084fc)}
.card.c-bull::before{background:linear-gradient(90deg,var(--bull),#4ade80)}
.card.c-bear::before{background:linear-gradient(90deg,var(--bear),#f87171)}
.card-label{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.card-value{font-size:36px;font-weight:800;font-family:var(--font-mono);line-height:1;
  margin:8px 0 4px}
.card-sub{font-size:12px;color:var(--muted)}
.card-icon{position:absolute;top:16px;right:16px;font-size:28px;opacity:.15}

/* ── CHART SECTION ── */
.section{margin-bottom:32px}
.section-header{display:flex;align-items:center;gap:10px;margin-bottom:16px}
.section-title{font-size:15px;font-weight:700;letter-spacing:-.01em}
.section-badge{font-size:10px;background:var(--surface3);border:1px solid var(--border);
  border-radius:4px;padding:2px 8px;color:var(--muted);font-family:var(--font-mono)}
.chart-wrap{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:16px;overflow:hidden}
.chart-legend{display:flex;gap:16px;margin-top:10px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted)}
.legend-dot{width:10px;height:10px;border-radius:2px}

/* ── EVENTS TABLE ── */
.table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px}
table{width:100%;border-collapse:collapse}
thead{background:var(--surface2)}
th{padding:10px 14px;text-align:left;font-size:10px;text-transform:uppercase;
   letter-spacing:.08em;color:var(--muted);font-weight:600;border-bottom:1px solid var(--border);
   white-space:nowrap}
td{padding:10px 14px;border-bottom:1px solid rgba(30,58,95,.5);vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(59,130,246,.04)}
.tag{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:4px;
  font-size:11px;font-weight:700;letter-spacing:.04em;font-family:var(--font-mono)}
.tag-bos{background:rgba(59,130,246,.15);color:#60a5fa;border:1px solid rgba(59,130,246,.3)}
.tag-choch{background:rgba(168,85,247,.15);color:#c084fc;border:1px solid rgba(168,85,247,.3)}
.tag-bull{background:rgba(34,197,94,.12);color:#4ade80;border:1px solid rgba(34,197,94,.25)}
.tag-bear{background:rgba(239,68,68,.12);color:#f87171;border:1px solid rgba(239,68,68,.25)}
.mono{font-family:var(--font-mono);font-size:12px}
.desc-cell{font-size:12px;color:#94a3b8;max-width:320px}

/* ── DAILY TABLE ── */
.bar-fill{height:6px;background:var(--surface3);border-radius:3px;min-width:40px;margin-top:3px}
.bar-inner{height:100%;border-radius:3px;background:var(--accent)}

/* ── SIGNAL ANALYSIS ── */
.analysis-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:28px}
@media(max-width:640px){.analysis-grid{grid-template-columns:1fr}}
.analysis-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px}
.analysis-title{font-size:12px;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;color:var(--muted);margin-bottom:12px}
.stat-row{display:flex;justify-content:space-between;align-items:center;
  padding:6px 0;border-bottom:1px solid rgba(30,58,95,.4)}
.stat-row:last-child{border-bottom:none}
.stat-key{font-size:12px;color:var(--muted)}
.stat-val{font-size:13px;font-weight:600;font-family:var(--font-mono)}

/* ── FOOTER ── */
.footer{text-align:center;padding:20px;color:var(--muted);font-size:11px;
  border-top:1px solid var(--border)}
.disclaimer{background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2);
  border-radius:8px;padding:12px 16px;margin-bottom:20px;font-size:12px;
  color:#fbbf24;line-height:1.5}

/* ── FILTER TABS ── */
.filter-bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.filter-btn{padding:5px 14px;border-radius:6px;font-size:12px;font-weight:600;
  border:1px solid var(--border);background:var(--surface2);color:var(--muted);
  cursor:pointer;transition:all .15s}
.filter-btn:hover,.filter-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.filter-btn.f-choch.active{background:var(--choch);border-color:var(--choch)}
.filter-btn.f-bull.active{background:var(--bull);border-color:var(--bull)}
.filter-btn.f-bear.active{background:var(--bear);border-color:var(--bear)}

.spark-wrap{display:flex;justify-content:center;margin-top:8px}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-inner">
    <div class="badge"><span class="badge-dot"></span>Live Analysis</div>
    <h1>Nifty 50 &mdash; <span>BOS &amp; CHoCH</span> Report</h1>
    <p class="subtitle">Break of Structure &amp; Change of Character — 5-Minute Bars &bull; Last 60 Trading Days</p>
    <div class="meta-row">
      <div class="meta-item">
        <span class="meta-label">From</span>
        <span class="meta-value">{{DATE_FROM}}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">To</span>
        <span class="meta-value">{{DATE_TO}}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Total Bars</span>
        <span class="meta-value">{{TOTAL_BARS}}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Overall Trend</span>
        <span class="meta-value {{TREND_CLASS}}">{{OVERALL_TREND}}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Generated</span>
        <span class="meta-value">{{GENERATED}}</span>
      </div>
    </div>
  </div>
</div>

<div class="main">

<!-- PRICE STRIP -->
<div class="price-strip">
  <div class="price-cell">
    <div class="pc-label">Open</div>
    <div class="pc-value neutral">{{OPEN}}</div>
  </div>
  <div class="price-cell">
    <div class="pc-label">Close</div>
    <div class="pc-value neutral">{{CLOSE}}</div>
  </div>
  <div class="price-cell">
    <div class="pc-label">High</div>
    <div class="pc-value bull">{{HIGH}}</div>
  </div>
  <div class="price-cell">
    <div class="pc-label">Low</div>
    <div class="pc-value bear">{{LOW}}</div>
  </div>
  <div class="price-cell">
    <div class="pc-label">Net Change</div>
    <div class="pc-value {{CHANGE_CLASS}}">{{CHANGE}} ({{PCT_CHANGE}}%)</div>
  </div>
</div>

<!-- SUMMARY CARDS -->
<div class="cards">
  <div class="card c-bos">
    <div class="card-icon">📊</div>
    <div class="card-label">Total BOS</div>
    <div class="card-value" style="color:var(--bos)">{{TOTAL_BOS}}</div>
    <div class="card-sub">Break of Structure signals</div>
  </div>
  <div class="card c-choch">
    <div class="card-icon">🔄</div>
    <div class="card-label">Total CHoCH</div>
    <div class="card-value" style="color:var(--choch)">{{TOTAL_CHOCH}}</div>
    <div class="card-sub">Change of Character signals</div>
  </div>
  <div class="card c-bull">
    <div class="card-icon">🐂</div>
    <div class="card-label">Bullish Signals</div>
    <div class="card-value bull">{{BULL_TOTAL}}</div>
    <div class="card-sub">{{BULL_BOS}} BOS &bull; {{BULL_CHOCH}} CHoCH</div>
  </div>
  <div class="card c-bear">
    <div class="card-icon">🐻</div>
    <div class="card-label">Bearish Signals</div>
    <div class="card-value bear">{{BEAR_TOTAL}}</div>
    <div class="card-sub">{{BEAR_BOS}} BOS &bull; {{BEAR_CHOCH}} CHoCH</div>
  </div>
</div>

<!-- SPARKLINE -->
<div class="section">
  <div class="section-header">
    <div class="section-title">Price Overview (Close)</div>
    <div class="section-badge">60 days</div>
  </div>
  <div class="chart-wrap">
    <div class="spark-wrap">{{SPARKLINE}}</div>
  </div>
</div>

<!-- CANDLESTICK CHART -->
<div class="section">
  <div class="section-header">
    <div class="section-title">Candlestick Chart with BOS &amp; CHoCH</div>
    <div class="section-badge">Last 200 bars</div>
  </div>
  <div class="chart-wrap">
    {{CANDLE_CHART}}
    <div class="chart-legend">
      <div class="legend-item"><div class="legend-dot" style="background:var(--bull)"></div>Bullish candle</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--bear)"></div>Bearish candle</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--bos)"></div>BOS level</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--choch)"></div>CHoCH level</div>
    </div>
  </div>
</div>

<!-- SIGNAL ANALYSIS -->
<div class="analysis-grid">
  <div class="analysis-card">
    <div class="analysis-title">📈 BOS Breakdown</div>
    <div class="stat-row"><span class="stat-key">Total BOS Signals</span><span class="stat-val">{{TOTAL_BOS}}</span></div>
    <div class="stat-row"><span class="stat-key">Bullish BOS</span><span class="stat-val bull">{{BULL_BOS}}</span></div>
    <div class="stat-row"><span class="stat-key">Bearish BOS</span><span class="stat-val bear">{{BEAR_BOS}}</span></div>
    <div class="stat-row"><span class="stat-key">Bull/Bear Ratio</span><span class="stat-val">{{BOS_RATIO}}</span></div>
  </div>
  <div class="analysis-card">
    <div class="analysis-title">🔄 CHoCH Breakdown</div>
    <div class="stat-row"><span class="stat-key">Total CHoCH Signals</span><span class="stat-val">{{TOTAL_CHOCH}}</span></div>
    <div class="stat-row"><span class="stat-key">Bullish CHoCH</span><span class="stat-val bull">{{BULL_CHOCH}}</span></div>
    <div class="stat-row"><span class="stat-key">Bearish CHoCH</span><span class="stat-val bear">{{BEAR_CHOCH}}</span></div>
    <div class="stat-row"><span class="stat-key">Bull/Bear Ratio</span><span class="stat-val">{{CHOCH_RATIO}}</span></div>
  </div>
  <div class="analysis-card">
    <div class="analysis-title">📅 Activity</div>
    <div class="stat-row"><span class="stat-key">Most Active Day</span><span class="stat-val">{{MOST_ACTIVE}}</span></div>
    <div class="stat-row"><span class="stat-key">Signals That Day</span><span class="stat-val">{{MOST_ACTIVE_N}}</span></div>
    <div class="stat-row"><span class="stat-key">Signal Density</span><span class="stat-val">{{DENSITY}} / 100 bars</span></div>
  </div>
  <div class="analysis-card">
    <div class="analysis-title">🕯️ Latest 5 Signals</div>
    {{LAST_EVENTS_HTML}}
  </div>
</div>

<!-- ALL EVENTS TABLE -->
<div class="section">
  <div class="section-header">
    <div class="section-title">All Detected Signals</div>
    <div class="section-badge" id="evCount">{{TOTAL_EVENTS}} events</div>
  </div>
  <div class="filter-bar">
    <button class="filter-btn active" onclick="filterTable('all')">All</button>
    <button class="filter-btn" onclick="filterTable('BOS')">BOS Only</button>
    <button class="filter-btn f-choch" onclick="filterTable('CHoCH')">CHoCH Only</button>
    <button class="filter-btn f-bull" onclick="filterTable('Bullish')">Bullish</button>
    <button class="filter-btn f-bear" onclick="filterTable('Bearish')">Bearish</button>
  </div>
  <div class="table-wrap">
    <table id="evTable">
      <thead>
        <tr>
          <th>#</th>
          <th>Date &amp; Time</th>
          <th>Type</th>
          <th>Direction</th>
          <th>Broken Level</th>
          <th>Close Price</th>
          <th>Trend at Break</th>
          <th>Swing Origin</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>{{EVENT_ROWS}}</tbody>
    </table>
  </div>
</div>

<!-- DAILY BREAKDOWN TABLE -->
<div class="section">
  <div class="section-header">
    <div class="section-title">Daily Breakdown</div>
    <div class="section-badge">By trading day</div>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Total</th>
          <th>BOS</th>
          <th>CHoCH</th>
          <th>Bullish</th>
          <th>Bearish</th>
          <th>First Signal</th>
          <th>Last Signal</th>
          <th>Distribution</th>
        </tr>
      </thead>
      <tbody>{{DAILY_ROWS}}</tbody>
    </table>
  </div>
</div>

<!-- GLOSSARY -->
<div class="section">
  <div class="section-header"><div class="section-title">📖 Glossary</div></div>
  <div class="analysis-grid">
    <div class="analysis-card">
      <div class="analysis-title">BOS — Break of Structure</div>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7">
        A BOS occurs when price closes beyond a significant swing high or low <em>in the direction of the existing trend</em>.
        It confirms that the current trend is still intact and likely to continue.
        <br><br>
        <strong style="color:var(--bull)">Bullish BOS:</strong> Price closes above a prior swing high in an uptrend — Higher Highs continuation.<br>
        <strong style="color:var(--bear)">Bearish BOS:</strong> Price closes below a prior swing low in a downtrend — Lower Lows continuation.
      </p>
    </div>
    <div class="analysis-card">
      <div class="analysis-title">CHoCH — Change of Character</div>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7">
        A CHoCH occurs when price closes beyond a significant swing high or low <em>against the existing trend</em>.
        It signals a potential reversal and is the first warning that the market's character may be changing.
        <br><br>
        <strong style="color:var(--bull)">Bullish CHoCH:</strong> In a downtrend, price closes above last Lower High — potential bottom forming.<br>
        <strong style="color:var(--bear)">Bearish CHoCH:</strong> In an uptrend, price closes below last Higher Low — potential top forming.
      </p>
    </div>
  </div>
</div>

<!-- DISCLAIMER -->
<div class="disclaimer">
  ⚠️ <strong>Disclaimer:</strong> This report is generated purely for educational and analytical purposes.
  BOS &amp; CHoCH detection is algorithmic and may not reflect all market conditions.
  This is <em>not</em> financial advice. Always conduct your own research before making any trading decisions.
</div>

</div><!-- /main -->

<div class="footer">
  Nifty 50 Market Structure Report &bull; Generated {{GENERATED}} &bull; Data via Yahoo Finance
</div>

<script>
const allEvents = {{EVENTS_JSON}};

function filterTable(f){
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
  const rows=document.querySelectorAll('#evTable tbody tr');
  let vis=0;
  rows.forEach(r=>{
    const type=r.dataset.type||'';
    const dir=r.dataset.dir||'';
    let show=false;
    if(f==='all') show=true;
    else if(f==='BOS'||f==='CHoCH') show=type===f;
    else show=dir===f;
    r.style.display=show?'':'none';
    if(show) vis++;
  });
  document.getElementById('evCount').textContent=vis+' events';
}
</script>
</body>
</html>"""


def build_html(df, events, stats, sh, sl):
    closes_sample = df["Close"].iloc[::max(1, len(df)//300)].tolist()
    sparkline     = build_sparkline(closes_sample, width=800, height=80)
    candle_chart  = build_candle_chart(df, events, max_bars=200)

    # event rows
    rows_html = []
    for i, ev in enumerate(events, 1):
        type_tag   = f'<span class="tag tag-bos">{ev["type"]}</span>' if ev["type"] == "BOS" else f'<span class="tag tag-choch">{ev["type"]}</span>'
        dir_class  = "tag-bull" if ev["direction"] == "Bullish" else "tag-bear"
        dir_arrow  = "▲ " if ev["direction"] == "Bullish" else "▼ "
        dir_tag    = f'<span class="tag {dir_class}">{dir_arrow}{ev["direction"]}</span>'
        trend_cls  = "bull" if ev["trend_at_break"] == "up" else ("bear" if ev["trend_at_break"] == "down" else "")
        rows_html.append(
            f'<tr data-type="{ev["type"]}" data-dir="{ev["direction"]}">'
            f'<td class="mono" style="color:var(--muted)">{i}</td>'
            f'<td class="mono">{ev["date"]}</td>'
            f'<td>{type_tag}</td>'
            f'<td>{dir_tag}</td>'
            f'<td class="mono" style="color:var(--warn)">₹{ev["level"]:,}</td>'
            f'<td class="mono">₹{ev["close"]:,}</td>'
            f'<td class="mono {trend_cls}">{ev["trend_at_break"].capitalize()}</td>'
            f'<td class="mono" style="font-size:11px;color:var(--muted)">{ev["swing_date"]}</td>'
            f'<td class="desc-cell">{ev["description"]}</td>'
            f'</tr>'
        )

    # daily rows
    daily = daily_breakdown(events)
    max_day_total = max((d["total"] for d in daily), default=1)
    daily_rows_html = []
    for d in daily:
        pct = int(d["total"] / max_day_total * 100)
        daily_rows_html.append(
            f'<tr>'
            f'<td class="mono">{d["date"]}</td>'
            f'<td class="mono" style="font-weight:700">{d["total"]}</td>'
            f'<td class="mono" style="color:var(--bos)">{d["bos"]}</td>'
            f'<td class="mono" style="color:var(--choch)">{d["choch"]}</td>'
            f'<td class="mono bull">{d["bull"]}</td>'
            f'<td class="mono bear">{d["bear"]}</td>'
            f'<td class="mono" style="font-size:11px">{d["first_event"]}</td>'
            f'<td class="mono" style="font-size:11px">{d["last_event"]}</td>'
            f'<td><div class="bar-fill"><div class="bar-inner" style="width:{pct}%"></div></div></td>'
            f'</tr>'
        )

    # last 5 events mini
    last_ev_html = []
    for ev in reversed(stats["last_events"]):
        col   = "bull" if ev["direction"] == "Bullish" else "bear"
        arrow = "▲" if ev["direction"] == "Bullish" else "▼"
        last_ev_html.append(
            f'<div class="stat-row">'
            f'<span class="stat-key" style="font-size:11px">{ev["date"][5:16]}</span>'
            f'<span class="stat-val {col}" style="font-size:11px">{arrow} {ev["type"]} ₹{ev["level"]:,}</span>'
            f'</div>'
        )

    # ratios
    bos_ratio   = f'{stats["bull_bos"]}/{stats["bear_bos"]}' if stats["total_bos"]   else "N/A"
    choch_ratio = f'{stats["bull_choch"]}/{stats["bear_choch"]}' if stats["total_choch"] else "N/A"
    density     = round((len(events) / stats["total_bars"]) * 100, 1) if stats["total_bars"] else 0

    trend_class  = "bull" if stats["overall_trend"] == "Bullish" else "bear"
    change_class = "bull" if stats["price_change"] >= 0 else "bear"
    change_str   = f'+{stats["price_change"]:,}' if stats["price_change"] >= 0 else f'{stats["price_change"]:,}'
    pct_str      = f'+{stats["pct_change"]}' if stats["pct_change"] >= 0 else str(stats["pct_change"])

    html = HTML_TEMPLATE
    replacements = {
        "{{DATE_FROM}}"        : stats["date_from"],
        "{{DATE_TO}}"          : stats["date_to"],
        "{{TOTAL_BARS}}"       : f'{stats["total_bars"]:,}',
        "{{OVERALL_TREND}}"    : stats["overall_trend"],
        "{{TREND_CLASS}}"      : trend_class,
        "{{GENERATED}}"        : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "{{OPEN}}"             : f'₹{stats["open_price"]:,}',
        "{{CLOSE}}"            : f'₹{stats["close_price"]:,}',
        "{{HIGH}}"             : f'₹{stats["high_price"]:,}',
        "{{LOW}}"              : f'₹{stats["low_price"]:,}',
        "{{CHANGE}}"           : f'₹{change_str}',
        "{{PCT_CHANGE}}"       : pct_str,
        "{{CHANGE_CLASS}}"     : change_class,
        "{{TOTAL_BOS}}"        : str(stats["total_bos"]),
        "{{TOTAL_CHOCH}}"      : str(stats["total_choch"]),
        "{{BULL_TOTAL}}"       : str(stats["bull_bos"] + stats["bull_choch"]),
        "{{BEAR_TOTAL}}"       : str(stats["bear_bos"] + stats["bear_choch"]),
        "{{BULL_BOS}}"         : str(stats["bull_bos"]),
        "{{BEAR_BOS}}"         : str(stats["bear_bos"]),
        "{{BULL_CHOCH}}"       : str(stats["bull_choch"]),
        "{{BEAR_CHOCH}}"       : str(stats["bear_choch"]),
        "{{BOS_RATIO}}"        : bos_ratio,
        "{{CHOCH_RATIO}}"      : choch_ratio,
        "{{MOST_ACTIVE}}"      : stats["most_active"],
        "{{MOST_ACTIVE_N}}"    : str(stats["most_active_n"]),
        "{{DENSITY}}"          : str(density),
        "{{LAST_EVENTS_HTML}}" : "\n".join(last_ev_html) if last_ev_html else "<p style='color:var(--muted);font-size:12px'>No events detected.</p>",
        "{{SPARKLINE}}"        : sparkline,
        "{{CANDLE_CHART}}"     : candle_chart,
        "{{TOTAL_EVENTS}}"     : str(len(events)),
        "{{EVENT_ROWS}}"       : "\n".join(rows_html) if rows_html else "<tr><td colspan='9' style='text-align:center;color:var(--muted);padding:24px'>No BOS/CHoCH events detected.</td></tr>",
        "{{DAILY_ROWS}}"       : "\n".join(daily_rows_html),
        "{{EVENTS_JSON}}"      : json.dumps(events),
    }
    for k, v in replacements.items():
        html = html.replace(k, v)
    return html


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT = "nifty_bos_choch_report.html"

    df      = fetch_data()
    sh, sl  = detect_swings(df, left=5, right=5)
    events  = detect_bos_choch(df, sh, sl)
    stats   = compute_stats(df, events)
    html    = build_html(df, events, stats, sh, sl)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'─'*50}")
    print(f"  Total bars analysed : {stats['total_bars']:,}")
    print(f"  BOS detected        : {stats['total_bos']}  (↑{stats['bull_bos']} ↓{stats['bear_bos']})")
    print(f"  CHoCH detected      : {stats['total_choch']}  (↑{stats['bull_choch']} ↓{stats['bear_choch']})")
    print(f"  Overall Trend       : {stats['overall_trend']}")
    print(f"  Report saved to     : {OUTPUT}")
    print(f"{'─'*50}\n")