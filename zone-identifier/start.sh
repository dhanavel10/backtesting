#!/bin/sh
# Ensure index.html exists (handles old images that only have sr_dash.html)
[ ! -f /app/index.html ] && cp /app/sr_dash.html /app/index.html
# Serve index.html on port 2324 — accessible at http://<host>:2324
python -m http.server 2324 --directory /app &
# Nifty tick feed WebSocket on port 8086
uvicorn nifty_websocker:app --host 0.0.0.0 --port 8086 &
# Run the live S/R engine (WebSocket client + dashboard broadcaster)
exec python live_sr.py
