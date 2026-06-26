# nifty_websocker.py
# ─────────────────────────────────────────────────────────────
# Accurate Nifty 50 LTP via yfinance (same library your backtest uses)
# Falls back to NSE unofficial API if yfinance is slow/stale.
#
# Run:
#   pip install fastapi uvicorn yfinance requests beautifulsoup4
#   uvicorn nifty_websocker:app --host 0.0.0.0 --port 8086 --reload
# ─────────────────────────────────────────────────────────────

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio, json, time, requests
from datetime import datetime
import yfinance as yf

app = FastAPI()

# reuse ticker object across calls
_ticker = yf.Ticker("^NSEI")

# NSE session (needs homepage warm-up for cookie)
_nse = requests.Session()
_nse.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
})
_nse_warmed = False


def _yfinance_price() -> float | None:
    try:
        df = _ticker.history(period="1d", interval="1m", auto_adjust=True)
        if df is not None and len(df) > 0:
            return float(df["Close"].iloc[-1])
    except Exception as e:
        print(f"  [yfinance] {e}")
    return None


def _nse_price() -> float | None:
    global _nse_warmed
    try:
        if not _nse_warmed:
            _nse.get("https://www.nseindia.com", timeout=8)
            _nse_warmed = True
        r = _nse.get("https://www.nseindia.com/api/allIndices", timeout=8)
        if r.status_code == 200:
            for item in r.json().get("data", []):
                if item.get("indexSymbol") == "NIFTY 50":
                    return float(item["last"])
    except Exception as e:
        print(f"  [nse] {e}")
        _nse_warmed = False
    return None


def get_live_price() -> dict:
    price = _yfinance_price()
    if price is None:
        print("  yfinance failed, trying NSE...")
        price = _nse_price()
    if price is not None:
        return {"ltp": round(price, 2), "timestamp": int(time.time() * 1000)}
    return {"error": "All sources failed"}


@app.on_event("startup")
async def startup():
    result = get_live_price()
    if "ltp" in result:
        print(f"\n  NIFTY 50 startup price: {result['ltp']}\n")
    else:
        print(f"\n  WARNING: {result['error']}\n")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("  Client connected")

    data = get_live_price()
    print(f"  first tick: {data}")
    await websocket.send_text(json.dumps(data))

    try:
        while True:
            await asyncio.sleep(2)
            data = get_live_price()
            ts   = datetime.now().strftime("%H:%M:%S")
            if "ltp" in data:
                print(f"  {ts}  ltp={data['ltp']}")
            else:
                print(f"  {ts}  {data['error']}")
            await websocket.send_text(json.dumps(data))

    except WebSocketDisconnect:
        print("  Client disconnected")
    except Exception as e:
        print(f"  Error: {e}")


@app.get("/")
def home():
    return {"status": "running", "websocket": "ws://localhost:8086/ws"}


@app.get("/price")
def price_check():
    """Open http://localhost:8086/price in browser to verify."""
    return get_live_price()


if __name__ == "__main__":
    print("Testing...\n")
    p1 = _yfinance_price()
    print(f"yfinance  : {p1 if p1 else 'FAILED'}")
    p2 = _nse_price()
    print(f"NSE       : {p2 if p2 else 'FAILED'}")
    print(f"final     : {get_live_price()}")