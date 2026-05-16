"""
dashboard.py
============
Plotly-Dash live monitoring dashboard for the breakout strategy.

Run:
    pip install dash dash-core-components dash-html-components
    python dashboard/dashboard.py

Opens at http://localhost:8050

Panels
------
  • Live 5m candlestick chart (last 2 sessions) with zone overlays
  • Zone table — all active zones with strength bars
  • Signal log — last 20 signals / trade outcomes
  • Session P&L ticker
"""

import sys, os

# ── Flat-folder import fix ──────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from dash import Dash, dcc, html, dash_table
    from dash.dependencies import Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from zone_engine  import ZoneEngine, fetch_intraday_chunked
from backtester   import Backtester


# ════════════════════════════════════════════════════════════════
# CHART BUILDER  (standalone — used by both dashboard and CLI)
# ════════════════════════════════════════════════════════════════

def build_chart(df: pd.DataFrame, zones_df: pd.DataFrame,
                signals: list = None, ticker: str = "NIFTY 50",
                sessions: int = 10) -> go.Figure:
    """
    Build a full candlestick + zone overlay chart.
    Can be called standalone (no Dash needed).
    signals: list of Signal objects from BreakoutEngine.signal_log
    """
    last_sessions = sorted(set(df.index.date))[-sessions:]
    df_plot = df[df.index.date >= last_sessions[0]]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )

    # ── Candlestick ──────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"],  high=df_plot["High"],
        low=df_plot["Low"],    close=df_plot["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
        name=ticker,
    ), row=1, col=1)

    # ── Volume ───────────────────────────────────────────────
    vcol = ["#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df_plot["Close"], df_plot["Open"])]
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot["Volume"],
        marker_color=vcol, opacity=0.4, name="Volume",
    ), row=2, col=1)

    # ── Zone bands ───────────────────────────────────────────
    if len(zones_df) > 0:
        for _, z in zones_df.iterrows():
            is_sup = z["type"] == "Support"
            alpha  = 0.06 + (z["strength"] / 100) * 0.18
            la     = 0.45 + (z["strength"] / 100) * 0.55

            fc = f"rgba(38,166,154,{alpha:.2f})" if is_sup else f"rgba(239,83,80,{alpha:.2f})"
            lc = f"rgba(38,166,154,{la:.2f})"   if is_sup else f"rgba(239,83,80,{la:.2f})"

            fig.add_hrect(y0=z["lower"], y1=z["upper"],
                          fillcolor=fc, line_width=0, row=1, col=1)
            fig.add_hline(y=z["price"], line_color=lc,
                          line_width=1.2, row=1, col=1)

            sym   = "S" if is_sup else "R"
            label = (f"{sym} {z['price']:.0f}  "
                     f"[{z['lower']:.0f}–{z['upper']:.0f}]  "
                     f"str={z['strength']:.0f}")
            fig.add_annotation(
                x=df_plot.index[-1], y=z["price"], text=label,
                showarrow=False, xanchor="left", yanchor="middle",
                font=dict(size=8.5, color=lc),
                bgcolor="rgba(0,0,0,0.6)", borderpad=2, row=1, col=1,
            )

    # ── Signal markers ────────────────────────────────────────
    if signals:
        for sig in signals:
            is_long = sig.direction.value == "LONG"
            color   = "#26a69a" if is_long else "#ef5350"
            symbol  = "triangle-up" if is_long else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[sig.timestamp],
                y=[sig.entry_price],
                mode="markers",
                marker=dict(symbol=symbol, size=14, color=color,
                            line=dict(color="white", width=1)),
                name=f"{'L' if is_long else 'S'} {sig.entry_price:.0f}",
                showlegend=False,
            ), row=1, col=1)
            # SL line segment
            if df_plot.index[-1] > sig.timestamp:
                fig.add_shape(
                    type="line",
                    x0=sig.timestamp, x1=df_plot.index[-1],
                    y0=sig.sl, y1=sig.sl,
                    line=dict(color="#ef5350", width=1, dash="dot"),
                    row=1, col=1,
                )
                fig.add_shape(
                    type="line",
                    x0=sig.timestamp, x1=df_plot.index[-1],
                    y0=sig.target, y1=sig.target,
                    line=dict(color="#26a69a", width=1, dash="dot"),
                    row=1, col=1,
                )

    # ── CMP line ─────────────────────────────────────────────
    cp = float(df["Close"].iloc[-1])
    fig.add_hline(y=cp, line_color="rgba(255,235,59,0.8)",
                  line_width=1.5, line_dash="dash", row=1, col=1)
    fig.add_annotation(
        x=df_plot.index[-1], y=cp, text=f"CMP {cp:.2f}",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=10, color="rgba(255,235,59,0.9)"),
        row=1, col=1,
    )

    # ── Session separators ────────────────────────────────────
    for sd in last_sessions[1:]:
        bars = df_plot[df_plot.index.date == sd]
        if len(bars):
            fig.add_vline(x=bars.index[0], line_dash="dot",
                          line_color="rgba(180,180,180,0.15)", row=1, col=1)

    n_zones = len(zones_df) if len(zones_df) > 0 else 0
    fig.update_layout(
        title=f"{ticker} — Precision Zone Breakout  |  "
              f"{n_zones} active zones  |  CMP {cp:.2f}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=700,
        showlegend=False,
        margin=dict(l=60, r=260, t=50, b=40),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(family="monospace"),
    )
    fig.update_yaxes(title_text="Nifty 50", row=1, col=1,
                     gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[15.5, 9.25], pattern="hour"),
    ])
    return fig


# ════════════════════════════════════════════════════════════════
# STATIC CHART MODE  (no Dash required)
# ════════════════════════════════════════════════════════════════

def plot_backtest_chart(ticker="^NSEI", days=60):
    """
    Fetch data, build zones, run a quick backtest, and show the chart.
    No Dash required — opens in browser via plotly.
    """
    print("Fetching data and building zones...")
    df = fetch_intraday_chunked(ticker, days=days)

    ze = ZoneEngine(
        cluster_tolerance=15.0,
        zone_half_band=15.0,
        min_wick_touches=3,
        min_sessions=2,
        min_rejections=1,
    )
    ze.build_from_history(df=df)

    zones = ze.get_active_zones()
    print(f"Active zones: {len(zones)}")

    # Quick backtest for signal markers
    bt = Backtester(
        df,
        warmup_days=20,
        zone_kwargs={"cluster_tolerance": 15.0, "zone_half_band": 15.0,
                     "min_wick_touches": 3, "min_sessions": 2, "min_rejections": 1},
        strat_kwargs={"min_breakout_pts": 20.0, "confirm_candles": 1, "rr_ratio": 2.0},
    )
    bt.run()
    bt.print_summary()

    signals = bt._be.signal_log if bt._be else []
    fig = build_chart(df, zones, signals=signals, ticker=ticker)
    fig.show()
    return fig


# ════════════════════════════════════════════════════════════════
# DASH LIVE DASHBOARD
# ════════════════════════════════════════════════════════════════

def create_dash_app(ze: ZoneEngine, be):
    """
    Create a Dash app for live monitoring.
    ze : running ZoneEngine instance
    be : running BreakoutEngine instance
    """
    if not DASH_AVAILABLE:
        raise ImportError("Install dash: pip install dash")

    app = Dash(__name__, title="Nifty Breakout Monitor")

    DARK_BG   = "#0d1117"
    CARD_BG   = "#161b22"
    TEXT      = "#c9d1d9"
    GREEN     = "#26a69a"
    RED       = "#ef5350"
    YELLOW    = "#ffd54f"

    app.layout = html.Div(style={"background": DARK_BG, "minHeight": "100vh",
                                  "fontFamily": "monospace", "color": TEXT}, children=[
        # Header
        html.Div(style={"padding": "12px 24px", "borderBottom": f"1px solid #30363d",
                         "display": "flex", "justifyContent": "space-between",
                         "alignItems": "center"}, children=[
            html.H2("NIFTY 50 — Precision Zone Breakout",
                    style={"margin": 0, "color": YELLOW, "fontSize": "16px",
                           "letterSpacing": "2px"}),
            html.Div(id="live-clock", style={"color": TEXT, "fontSize": "13px"}),
        ]),

        # Main grid
        html.Div(style={"display": "grid",
                         "gridTemplateColumns": "1fr 340px",
                         "gap": "12px", "padding": "12px"}, children=[

            # Left: chart
            html.Div(style={"background": CARD_BG, "borderRadius": "8px",
                              "padding": "8px"}, children=[
                dcc.Graph(id="main-chart", style={"height": "620px"}),
            ]),

            # Right: zones + signals
            html.Div(style={"display": "flex", "flexDirection": "column",
                              "gap": "12px"}, children=[

                # Zone table
                html.Div(style={"background": CARD_BG, "borderRadius": "8px",
                                 "padding": "12px"}, children=[
                    html.H4("Active Zones", style={"margin": "0 0 8px",
                                                    "color": YELLOW, "fontSize": "12px",
                                                    "letterSpacing": "1px"}),
                    html.Div(id="zone-table"),
                ]),

                # Signal / trade log
                html.Div(style={"background": CARD_BG, "borderRadius": "8px",
                                 "padding": "12px", "flex": "1"}, children=[
                    html.H4("Signal Log", style={"margin": "0 0 8px",
                                                  "color": YELLOW, "fontSize": "12px",
                                                  "letterSpacing": "1px"}),
                    html.Div(id="signal-log"),
                ]),

                # P&L summary
                html.Div(id="pnl-summary",
                         style={"background": CARD_BG, "borderRadius": "8px",
                                 "padding": "12px", "fontSize": "13px"}),
            ]),
        ]),

        # Auto-refresh interval (5 seconds)
        dcc.Interval(id="refresh", interval=5_000, n_intervals=0),
    ])

    # ── Callbacks ────────────────────────────────────────────

    @app.callback(
        [Output("main-chart",  "figure"),
         Output("zone-table",  "children"),
         Output("signal-log",  "children"),
         Output("pnl-summary", "children"),
         Output("live-clock",  "children")],
        Input("refresh", "n_intervals"),
    )
    def refresh(_):
        now    = datetime.now().strftime("%H:%M:%S")
        df     = ze.df
        zones  = ze.get_active_zones()
        sigs   = be.signal_log[-20:] if be else []

        # Chart
        fig = build_chart(df, zones, signals=sigs, sessions=5)

        # Zone table rows
        zone_rows = []
        for _, z in zones.sort_values("dist_pts", key=abs).iterrows():
            is_sup  = z["type"] == "Support"
            color   = GREEN if is_sup else RED
            bar_w   = f"{z['strength']:.0f}%"
            zone_rows.append(
                html.Div(style={"borderBottom": "1px solid #21262d",
                                 "padding": "5px 0", "fontSize": "11px"}, children=[
                    html.Div(style={"display": "flex",
                                     "justifyContent": "space-between"}, children=[
                        html.Span(f"{'S' if is_sup else 'R'}  {z['price']:.0f}",
                                  style={"color": color, "fontWeight": "bold"}),
                        html.Span(f"str={z['strength']:.0f}  dist={z['dist_pts']:+.0f}",
                                  style={"color": "#8b949e"}),
                    ]),
                    html.Div(style={"height": "3px", "background": "#21262d",
                                     "borderRadius": "2px", "marginTop": "3px"}, children=[
                        html.Div(style={"height": "3px", "width": bar_w,
                                         "background": color, "borderRadius": "2px"}),
                    ]),
                ])
            )

        # Signal log rows
        sig_rows = []
        for sig in reversed(sigs):
            is_long = sig.direction.value == "LONG"
            color   = GREEN if is_long else RED
            conf    = "✅" if sig.confirmed else "⏳"
            sig_rows.append(
                html.Div(style={"borderBottom": "1px solid #21262d",
                                 "padding": "4px 0", "fontSize": "11px"}, children=[
                    html.Span(f"{conf} {'▲' if is_long else '▼'} {sig.entry_price:.0f}  "
                              f"SL={sig.sl:.0f}  TGT={sig.target:.0f}  "
                              f"R:R={sig.rr_ratio}",
                              style={"color": color}),
                    html.Br(),
                    html.Span(f"{str(sig.timestamp)[:16]}  zone={sig.zone_price:.0f}  "
                              f"str={sig.zone_strength:.0f}",
                              style={"color": "#8b949e", "fontSize": "10px"}),
                ])
            )

        # P&L
        summary = be.get_trade_summary() if be else {}
        pnl_color = GREEN if summary.get("total_pts", 0) >= 0 else RED
        pnl_div = html.Div([
            html.Div(f"Session P&L: {summary.get('total_pts', 0):+.0f} pts",
                     style={"color": pnl_color, "fontWeight": "bold", "fontSize": "14px"}),
            html.Div(f"Trades: {summary.get('trades', 0)}  "
                     f"Win rate: {summary.get('win_rate', 0)}%",
                     style={"color": TEXT, "fontSize": "11px", "marginTop": "4px"}),
        ])

        return fig, zone_rows, sig_rows, pnl_div, f"Last update: {now}"

    return app


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chart", "dash"], default="chart",
                        help="chart = static plotly | dash = live dashboard")
    parser.add_argument("--ticker", default="^NSEI")
    parser.add_argument("--days",   type=int, default=60)
    args = parser.parse_args()

    if args.mode == "chart":
        plot_backtest_chart(args.ticker, args.days)
    else:
        if not DASH_AVAILABLE:
            print("Install dash: pip install dash")
            sys.exit(1)
        # For live mode, you'd pass your running ze/be instances here
        print("Dash mode requires a running ZoneEngine + BreakoutEngine.")
        print("Import create_dash_app from live_runner.py instead.")
