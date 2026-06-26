"""
market_structure — Modular trading system based on Adam Grimes'
"The Art and Science of Technical Analysis".

Modules:
    indicators      — ATR, EMA, MACD, Keltner Channel, bar character
    pivot_detector  — Swing high/low detection, zigzag construction
    market_structure — Market regime detection (Wyckoff phases, Dow Theory)
    sr_zones        — S/R zone clustering, spring/upthrust detection
    channel_detector — Trend channels, Keltner analysis, climax detection
    trend_analyzer  — Trend health, change detection, MTF alignment
    signal_engine   — All 6 trade type signals with confluence scoring
    risk_manager    — Fixed fractional sizing, Kelly criterion, Monte Carlo
    scanner         — Main orchestration; scan instruments for signals

Quick start (when market_structure/ is on sys.path):
    import sys; sys.path.insert(0, 'path/to/market_structure')
    from scanner import MarketScanner
    scanner = MarketScanner(account_equity=500_000, risk_fraction=0.02)
    result = scanner.analyze(df_ohlcv)
    print(result.summary())
"""
