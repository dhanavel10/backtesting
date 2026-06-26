#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║         INDIAN MARKET MORNING BRIEF — Daily Sentiment Engine         ║
║                                                                      ║
║  NSE India  →  All Indices, VIX, FII/DII, Pre-open, Gainers,        ║
║               Losers, Most Active, 52W Hi/Lo, Option Chain PCR       ║
║  yfinance   →  US Markets, DXY, Crude, Gold, Global Indices,         ║
║               USD/INR, Bond Yields                                   ║
║  News RSS   →  Google News, ET, LiveMint, Business Standard,         ║
║               Reuters (15 feeds covering all local + global news)    ║
╚══════════════════════════════════════════════════════════════════════╝

INSTALL (one-time):
    pip install requests beautifulsoup4 yfinance schedule pytz colorama tabulate lxml

USAGE:
    python indian_market_morning_brief.py                   # run immediately
    python indian_market_morning_brief.py --schedule        # daily at 08:00 IST
    python indian_market_morning_brief.py --schedule --time 07:30

SSL NOTE:
    If you are behind a corporate proxy or VPN, the script automatically
    applies a TLS 1.2 compatibility adapter so NSE India API calls succeed
    even when SSL inspection is active on your network.
"""

import argparse
import json
import logging
import os
import ssl
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import schedule
import urllib3
import yfinance as yf
from bs4 import BeautifulSoup
from colorama import Back, Fore, Style, init
from tabulate import tabulate
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
init(autoreset=True)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
IST = ZoneInfo("Asia/Kolkata")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("market_brief.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SSL FIX — LegacyTLSAdapter
# ─────────────────────────────────────────────────────────────────────────────
# Many corporate networks / VPNs use SSL inspection proxies that intercept
# HTTPS using TLS 1.2 with older ciphers.  Python 3.10+ defaults to TLS 1.3
# and modern ciphersuites, causing "WRONG_VERSION_NUMBER" SSL errors.
# This adapter forces TLS 1.2 compatibility and disables cert verification
# (safe for read-only market data scraping on a trusted network).
# ─────────────────────────────────────────────────────────────────────────────

_LEGACY_CIPHERS = (
    "ECDH+AESGCM:DH+AESGCM:ECDH+AES256:DH+AES256:"
    "ECDH+AES128:DH+AES:ECDH+HIGH:DH+HIGH:"
    "RSA+AESGCM:RSA+AES:RSA+HIGH:!aNULL:!eNULL:!MD5"
)


class LegacyTLSAdapter(HTTPAdapter):
    """
    Requests adapter that forces TLS 1.2 + legacy ciphers.
    Fixes: SSL WRONG_VERSION_NUMBER errors from corporate proxy SSL inspection.
    """
    def _build_ssl_ctx(self) -> ssl.SSLContext:
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        try:
            ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT   # Python 3.12+
        except AttributeError:
            pass
        try:
            ctx.set_ciphers(_LEGACY_CIPHERS)
        except ssl.SSLError:
            pass
        return ctx

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._build_ssl_ctx()
        super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs["ssl_context"] = self._build_ssl_ctx()
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def _make_session(referer: str = "https://www.nseindia.com/") -> requests.Session:
    """Create a requests Session with full browser headers and SSL fix."""
    s = requests.Session()
    s.mount("https://", LegacyTLSAdapter())
    s.verify = False
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
        "Referer":         referer,
        "Sec-Fetch-Dest":  "empty",
        "Sec-Fetch-Mode":  "cors",
        "Sec-Fetch-Site":  "same-origin",
        "DNT":             "1",
    })
    return s


NSE_SESSION  = _make_session("https://www.nseindia.com/")
NEWS_SESSION = _make_session("https://news.google.com/")

_NSE_WARMED = False


def nse_warm_up():
    """
    NSE India requires a browser session (valid cookies) before allowing API calls.
    We hit the homepage + a real page first to collect those cookies.
    """
    global _NSE_WARMED
    if _NSE_WARMED:
        return
    pages = [
        ("https://www.nseindia.com/",                          "homepage"),
        ("https://www.nseindia.com/market-data/live-equity-market", "market page"),
    ]
    for url, label in pages:
        try:
            log.info(f"NSE warm-up: hitting {label}…")
            NSE_SESSION.get(url, timeout=20)
            time.sleep(2)
        except Exception as e:
            log.warning(f"NSE warm-up ({label}) failed: {e}")
    _NSE_WARMED = True
    log.info("NSE session ready ✅")


def nse_get(url: str, retries: int = 3) -> dict | list | None:
    """GET an NSE API endpoint with cookie warm-up, retry, and back-off."""
    nse_warm_up()
    for attempt in range(1, retries + 1):
        try:
            r = NSE_SESSION.get(url, timeout=20)
            if r.status_code == 401:
                log.warning("NSE 401 — refreshing session cookies…")
                global _NSE_WARMED
                _NSE_WARMED = False
                nse_warm_up()
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 3 * attempt
            log.warning(f"NSE attempt {attempt}/{retries} failed [{url.split('?')[0].split('/')[-1]}]: {e}")
            if attempt < retries:
                log.info(f"  Retrying in {wait}s…")
                time.sleep(wait)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Colour / display helpers
# ─────────────────────────────────────────────────────────────────────────────
def G(t):  return f"{Fore.GREEN}{Style.BRIGHT}{t}{Style.RESET_ALL}"
def R(t):  return f"{Fore.RED}{Style.BRIGHT}{t}{Style.RESET_ALL}"
def Y(t):  return f"{Fore.YELLOW}{Style.BRIGHT}{t}{Style.RESET_ALL}"
def C(t):  return f"{Fore.CYAN}{Style.BRIGHT}{t}{Style.RESET_ALL}"
def B(t):  return f"{Style.BRIGHT}{t}{Style.RESET_ALL}"
def M(t):  return f"{Fore.MAGENTA}{Style.BRIGHT}{t}{Style.RESET_ALL}"

def pct_str(pct) -> str:
    try:
        p = float(str(pct).replace("%","").replace("+","").strip())
    except Exception:
        return Y(str(pct))
    s = f"{p:+.2f}%"
    return G(s) if p > 0 else R(s) if p < 0 else Y(s)

def sentiment(pct) -> str:
    try:
        p = float(str(pct).replace("%","").replace("+","").strip())
    except Exception:
        return Y("N/A")
    if p >= 0.5:  return G("BULLISH 🟢")
    if p <= -0.5: return R("BEARISH 🔴")
    return Y("NEUTRAL 🟡")

def section(title: str):
    w = 76
    print()
    print(C("═" * w))
    print(B(C(f"  {title}")))
    print(C("─" * w))

def print_kv(d: dict):
    for k, v in d.items():
        print(f"  {B(str(k)):<40} {v}")

def print_tbl(rows: list[dict], fmt: str = "simple"):
    if not rows:
        print(Y("  No data available"))
        return
    print(tabulate(rows, headers="keys", tablefmt=fmt))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ══  BLOCK A — NSE INDIA  ════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def nse_market_status() -> dict:
    data = nse_get("https://www.nseindia.com/api/marketStatus")
    if not data:
        return {"Status": R("Unavailable — NSE may be closed or blocked")}
    result = {}
    for mkt in data.get("marketState", []):
        name  = mkt.get("market", "Unknown")
        state = mkt.get("marketStatus", "Unknown")
        trade = mkt.get("tradeDate", "")
        exch  = mkt.get("exchange", "")
        color = G(state) if "open" in state.lower() else R(state)
        result[f"{exch} — {name}"] = f"{color}  [{trade}]"
    return result or {"Status": Y("No data in response")}


def nse_india_vix() -> dict:
    data = nse_get("https://www.nseindia.com/api/equity-stockIndices?index=INDIA%20VIX")
    if not data:
        return {"India VIX": R("Unavailable")}
    meta = data.get("metadata", {})
    vix  = float(meta.get("last", 0) or 0)
    pct  = float(meta.get("percChange", 0) or 0)
    hi   = meta.get("high", "—")
    lo   = meta.get("low", "—")
    prev = meta.get("previousClose", "—")
    if vix == 0:
        return {"India VIX": R("Unavailable or market closed")}
    if vix < 12:   status = G("Calm 😌  Market is confident — low fear")
    elif vix < 16: status = G("Normal 😐  Routine market conditions")
    elif vix < 18: status = Y("Elevated ⚠️  Caution — volatility building")
    elif vix < 22: status = R("Fear 😨  High volatility — reduce risk")
    else:          status = R("PANIC 🚨  Extreme fear — hedge immediately!")
    direction = (
        R("↑ Rising VIX + falling market = STRONG BEARISH signal") if pct > 5 else
        G("↓ Falling VIX + rising market = HEALTHY RALLY signal")  if pct < -5 else
        Y("→ VIX relatively stable")
    )
    return {
        "India VIX":      f"{vix:.2f}",
        "Day High":       str(hi),
        "Day Low":        str(lo),
        "Prev Close":     str(prev),
        "Change%":        pct_str(pct),
        "Status":         status,
        "Signal":         direction,
    }


def nse_all_indices() -> list[dict]:
    data = nse_get("https://www.nseindia.com/api/allIndices")
    if not data:
        return []
    rows = []
    for idx in data.get("data", []):
        name = idx.get("index", "")
        last = idx.get("last", 0)
        chg  = idx.get("variation", 0)
        pct  = idx.get("percentChange", 0)
        adv  = idx.get("advances", "—")
        dec  = idx.get("declines", "—")
        rows.append({
            "Index":    name,
            "Last":     f"{float(last):>12,.2f}" if last else "—",
            "Change":   f"{float(chg):>+10.2f}"  if chg  else "—",
            "Chg%":     pct_str(pct),
            "Adv":      G(str(adv)),
            "Dec":      R(str(dec)),
            "Bias":     sentiment(pct),
        })
    return rows


def nse_index_detail(index_name: str) -> dict:
    enc  = requests.utils.quote(index_name)
    data = nse_get(f"https://www.nseindia.com/api/equity-stockIndices?index={enc}")
    if not data:
        return {index_name: R("Unavailable")}
    meta = data.get("metadata", {})
    adv  = data.get("advance", {})
    pct  = float(meta.get("percChange", 0) or 0)
    return {
        "Index":          meta.get("indexName", index_name),
        "Open":           meta.get("open",          "—"),
        "High":           meta.get("high",          "—"),
        "Low":            meta.get("low",           "—"),
        "Last":           meta.get("last",          "—"),
        "Prev Close":     meta.get("previousClose", "—"),
        "Change":         meta.get("change",        "—"),
        "Chg%":           pct_str(pct),
        "52W High":       meta.get("yearHigh",      "—"),
        "52W Low":        meta.get("yearLow",       "—"),
        "Advances":       G(str(adv.get("advances",  "—"))),
        "Declines":       R(str(adv.get("declines",  "—"))),
        "Unchanged":      Y(str(adv.get("unchanged", "—"))),
        "Sentiment":      sentiment(pct),
    }


def nse_pre_open(symbol: str = "NIFTY") -> list[dict]:
    data = nse_get(f"https://www.nseindia.com/api/market-data-pre-open?key={symbol}")
    if not data:
        return []
    rows = []
    for item in (data.get("data", []) or [])[:20]:
        meta   = item.get("metadata", {})
        detail = item.get("detail", {}).get("preOpenMarket", {})
        sym    = meta.get("symbol", "")
        iep    = detail.get("IEP", 0)
        prev   = float(meta.get("previousClose", 0) or 0)
        try:
            pct = ((float(iep) - prev) / prev) * 100 if prev else 0
        except Exception:
            pct = 0
        rows.append({
            "Symbol":     sym,
            "IEP (₹)":    iep,
            "Prev (₹)":   prev,
            "Est Chg%":   pct_str(pct),
            "Buy Qty":    detail.get("totalBuyQuantity",  "—"),
            "Sell Qty":   detail.get("totalSellQuantity", "—"),
            "Final":      G("✓") if detail.get("finalPrice") else "—",
        })
    return rows


def nse_fii_dii() -> list[dict]:
    data = nse_get("https://www.nseindia.com/api/fiidiiTradeReact")
    if not data or not isinstance(data, list):
        return [{"Note": R("FII/DII unavailable from NSE")}]
    rows = []
    for item in data[:6]:
        date = item.get("date", "")
        for tag, bk, sk in [("FII/FPI", "fiiBuyValue", "fiiSellValue"),
                              ("DII",     "diiBuyValue", "diiSellValue")]:
            try:
                buy  = float(item.get(bk, 0) or 0)
                sell = float(item.get(sk, 0) or 0)
                net  = buy - sell
                rows.append({
                    "Date":       date,
                    "Entity":     B(tag),
                    "Buy (₹Cr)":  f"{buy:>12,.2f}",
                    "Sell (₹Cr)": f"{sell:>12,.2f}",
                    "Net (₹Cr)":  G(f"+{net:>10,.2f}") if net > 0 else R(f"{net:>10,.2f}"),
                    "Signal":     G("NET BUYER 🟢") if net > 0 else R("NET SELLER 🔴"),
                })
            except Exception:
                pass
    return rows


def nse_gainers_losers() -> tuple[list[dict], list[dict]]:
    gainers, losers = [], []
    for kind, store in [("gainers", gainers), ("loosers", losers)]:
        data = nse_get(
            f"https://www.nseindia.com/api/live-analysis-variations?index={kind}&limit=10"
        )
        if not data:
            continue
        items = (data.get("NIFTY", {}).get("data") or
                 data.get("data") or [])
        for item in items[:10]:
            pct = float(item.get("pChange", 0) or 0)
            store.append({
                "Symbol":   item.get("symbol",            "—"),
                "LTP (₹)":  item.get("lastPrice",         "—"),
                "Chg (₹)":  item.get("change",            "—"),
                "Chg%":     pct_str(pct),
                "Volume":   item.get("totalTradedVolume",  "—"),
                "52W H":    item.get("yearHigh",           "—"),
                "52W L":    item.get("yearLow",            "—"),
            })
    return gainers, losers


def nse_most_active() -> list[dict]:
    data = nse_get(
        "https://www.nseindia.com/api/live-analysis-most-active-securities?index=value&limit=10"
    )
    if not data:
        return []
    rows = []
    for item in (data.get("data", []) or [])[:10]:
        pct = float(item.get("pChange", 0) or 0)
        rows.append({
            "Symbol":      item.get("symbol",             "—"),
            "LTP (₹)":     item.get("lastPrice",          "—"),
            "Chg%":        pct_str(pct),
            "Value (₹Cr)": item.get("totalTradedValue",   "—"),
            "Volume":      item.get("totalTradedVolume",   "—"),
        })
    return rows


def nse_52wk() -> tuple[list[dict], list[dict]]:
    highs, lows = [], []
    for kind, store in [("high52", highs), ("low52", lows)]:
        data = nse_get(
            f"https://www.nseindia.com/api/live-analysis-variations?index={kind}&limit=10"
        )
        if not data:
            continue
        for item in (data.get("data", []) or [])[:10]:
            pct = float(item.get("pChange", 0) or 0)
            store.append({
                "Symbol":  item.get("symbol",    "—"),
                "LTP (₹)": item.get("lastPrice", "—"),
                "52W H/L": item.get("yearHigh") or item.get("yearLow", "—"),
                "Chg%":    pct_str(pct),
            })
    return highs, lows


def nse_option_chain_pcr(symbol: str = "NIFTY") -> dict:
    data = nse_get(
        f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    )
    if not data:
        return {"PCR": R("Unavailable (market may be closed)")}
    try:
        records = data.get("records", {})
        expiries = records.get("expiryDates", [])
        nearest  = expiries[0] if expiries else None
        spot     = float(records.get("underlyingValue", 0) or 0)
        total_ce_oi = total_pe_oi = 0
        max_pain_ce = {}
        max_pain_pe = {}
        for row in records.get("data", []):
            if nearest and row.get("expiryDate") != nearest:
                continue
            ce = row.get("CE", {}) or {}
            pe = row.get("PE", {}) or {}
            strike = row.get("strikePrice", 0)
            ce_oi  = ce.get("openInterest", 0) or 0
            pe_oi  = pe.get("openInterest", 0) or 0
            total_ce_oi += ce_oi
            total_pe_oi += pe_oi
            max_pain_ce[strike] = max_pain_ce.get(strike, 0) + ce_oi
            max_pain_pe[strike] = max_pain_pe.get(strike, 0) + pe_oi

        pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi else 0

        # Estimate max pain (strike with maximum combined OI)
        all_strikes = sorted(set(max_pain_ce) | set(max_pain_pe))
        if all_strikes:
            combined = {s: max_pain_ce.get(s, 0) + max_pain_pe.get(s, 0)
                        for s in all_strikes}
            # Max pain is where combined OI is highest (writers' profit point)
            max_pain = max(combined, key=lambda s: combined[s])
        else:
            max_pain = "—"

        if pcr > 1.5:   mood = G("Very Bullish 🐂 (heavy put writing — strong support)")
        elif pcr > 1.2: mood = G("Bullish 🟢 (put writers active)")
        elif pcr > 0.8: mood = Y("Neutral 🟡 (balanced OI)")
        elif pcr > 0.5: mood = R("Bearish 🔴 (call writers dominating)")
        else:           mood = R("Very Bearish 🐻 (heavy call buildup — resistance strong)")

        return {
            "Symbol":         symbol,
            "Nearest Expiry": nearest or "—",
            "Spot Price":     f"{spot:,.2f}",
            "Total Call OI":  f"{total_ce_oi:,}",
            "Total Put OI":   f"{total_pe_oi:,}",
            "PCR":            B(str(pcr)),
            "Max Pain Strike":f"{max_pain:,}" if isinstance(max_pain, (int, float)) else str(max_pain),
            "Market Mood":    mood,
        }
    except Exception as e:
        log.warning(f"Option chain parse error: {e}")
        return {"PCR": Y("Parse error")}


def nse_advance_decline() -> dict:
    data = nse_get(
        "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
    )
    if not data:
        return {"Breadth": R("Unavailable")}
    adv  = data.get("advance", {})
    advances  = int(adv.get("advances",  0) or 0)
    declines  = int(adv.get("declines",  0) or 0)
    unchanged = int(adv.get("unchanged", 0) or 0)
    total = (advances + declines + unchanged) or 1
    ratio = round(advances / declines, 2) if declines else float("inf")
    breadth = (
        G("Strong Breadth 🟢 — broad participation")  if ratio > 1.5 else
        R("Weak Breadth 🔴 — selling widespread")      if ratio < 0.7 else
        Y("Mixed Breadth 🟡 — selective market")
    )
    return {
        "Advances":     G(str(advances)),
        "Declines":     R(str(declines)),
        "Unchanged":    Y(str(unchanged)),
        "Total":        str(total),
        "A/D Ratio":    B(str(ratio)),
        "Advances %":   f"{advances/total*100:.1f}%",
        "Breadth":      breadth,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ══  BLOCK B — GLOBAL DATA (yfinance)  ═══════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def _yf(sym: str, period: str = "5d") -> tuple[float, float, float]:
    """Return (last_close, change, pct_change) for a yfinance ticker."""
    try:
        hist = yf.Ticker(sym).history(period=period)
        if len(hist) >= 2:
            prev = hist["Close"].iloc[-2]
            last = hist["Close"].iloc[-1]
            chg  = last - prev
            pct  = (chg / prev) * 100
            return last, chg, pct
        elif len(hist) == 1:
            return hist["Close"].iloc[-1], 0.0, 0.0
    except Exception as e:
        log.debug(f"yfinance {sym}: {e}")
    return 0.0, 0.0, 0.0


def yf_us_markets() -> list[dict]:
    indices = {
        "S&P 500":       "^GSPC",
        "NASDAQ":        "^IXIC",
        "Dow Jones":     "^DJI",
        "Russell 2000":  "^RUT",
        "US VIX (Fear)": "^VIX",
    }
    rows = []
    for name, sym in indices.items():
        last, chg, pct = _yf(sym)
        if last == 0:
            rows.append({"Index": name, "Last": R("N/A"), "Change": "—", "Chg%": "—", "Sentiment": "—"})
            continue
        # Special handling for VIX
        if "VIX" in name:
            vix_note = (G("Low fear") if last < 15 else
                        Y("Moderate") if last < 20 else
                        R("High fear"))
            rows.append({"Index": name, "Last": f"{last:.2f}", "Change": f"{chg:+.2f}",
                         "Chg%": pct_str(pct), "Sentiment": vix_note})
        else:
            rows.append({"Index": name, "Last": f"{last:,.2f}", "Change": f"{chg:+.2f}",
                         "Chg%": pct_str(pct), "Sentiment": sentiment(pct)})
    return rows


def yf_bond_yields_dxy() -> list[dict]:
    instruments = {
        "US 10Y Treasury Yield": ("^TNX",      "yield"),
        "US 2Y Treasury Yield":  ("^IRX",      "yield"),
        "Dollar Index (DXY)":    ("DX-Y.NYB",  "dxy"),
    }
    rows = []
    for name, (sym, kind) in instruments.items():
        last, chg, pct = _yf(sym)
        if last == 0:
            rows.append({"Instrument": name, "Value": R("N/A"), "Change": "—",
                         "Chg%": "—", "India Impact": "—"})
            continue
        if kind == "dxy":
            impact = (R("Bearish India ⚠️ — strong dollar = FII outflows") if pct > 0.3 else
                      G("Bullish India ✅ — weak dollar = EM inflows")       if pct < -0.3 else
                      Y("Neutral"))
        else:
            impact = (R("FII sell risk ⚠️ — higher US yields pull global money") if pct > 3 else
                      G("EM positive ✅ — falling yields supportive")              if pct < -3 else
                      Y("Watch"))
        rows.append({
            "Instrument":   name,
            "Value":        f"{last:.3f}",
            "Change":       f"{chg:+.4f}",
            "Chg%":         pct_str(pct),
            "India Impact": impact,
        })
    return rows


def yf_commodities() -> list[dict]:
    commodities = {
        "Brent Crude (USD/bbl)": ("BZ=F", "crude"),
        "WTI Crude (USD/bbl)":   ("CL=F", "crude"),
        "Gold (USD/oz)":         ("GC=F", "gold"),
        "Silver (USD/oz)":       ("SI=F", "silver"),
        "Natural Gas":           ("NG=F", "gas"),
    }
    rows = []
    for name, (sym, kind) in commodities.items():
        last, chg, pct = _yf(sym)
        if last == 0:
            rows.append({"Commodity": name, "Price": R("N/A"), "Change": "—",
                         "Chg%": "—", "India Impact": "—"})
            continue
        if kind == "crude":
            impact = (R("Bearish India 🛢️ — high oil hurts CAD + inflation") if pct > 2 else
                      G("Bullish India ✅ — cheaper imports, lower inflation") if pct < -2 else
                      Y("Neutral"))
        elif kind == "gold":
            impact = (R("Risk-off — safe haven demand, fear rising") if pct > 0.8 else
                      G("Risk-on mood 💹") if pct < -0.8 else Y("Neutral"))
        else:
            impact = Y("Monitor")
        rows.append({
            "Commodity":    name,
            "Price":        f"{last:,.2f}",
            "Change":       f"{chg:+.2f}",
            "Chg%":         pct_str(pct),
            "India Impact": impact,
        })
    return rows


def yf_usdinr() -> dict:
    last, chg, pct = _yf("INR=X")
    if last == 0:
        return {"USD/INR": R("N/A — yfinance unavailable")}
    # Higher number = weaker rupee = bearish India
    impact = (R("Rupee weakening 📉 — FII outflows / oil stress / macro fear") if pct > 0.2 else
              G("Rupee strengthening 📈 — FII inflows / BoP positive")          if pct < -0.2 else
              Y("Rupee stable 😐"))
    rbi_note = ("RBI may intervene to stabilise" if abs(pct) > 0.5 else
                "No immediate RBI action expected")
    return {
        "USD/INR Rate":   f"{last:.4f}",
        "Change":         f"{chg:+.4f}",
        "Chg%":           pct_str(pct),
        "Impact":         impact,
        "RBI Watch":      rbi_note,
    }


def yf_global_indices() -> list[dict]:
    indices = {
        "Nikkei 225 (Japan)":    "^N225",
        "Hang Seng (HK)":        "^HSI",
        "Shanghai Comp (China)": "000001.SS",
        "KOSPI (Korea)":         "^KS11",
        "Taiwan TAIEX":          "^TWII",
        "Straits Times (SGX)":   "^STI",
        "DAX (Germany)":         "^GDAXI",
        "FTSE 100 (UK)":         "^FTSE",
        "CAC 40 (France)":       "^FCHI",
        "Euro Stoxx 50":         "^STOXX50E",
    }
    rows = []
    for name, sym in indices.items():
        last, chg, pct = _yf(sym)
        if last == 0:
            rows.append({"Index": name, "Last": R("N/A"), "Change": "—", "Chg%": "—"})
        else:
            rows.append({"Index": name, "Last": f"{last:,.2f}",
                         "Change": f"{chg:+.2f}", "Chg%": pct_str(pct)})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# ══  BLOCK C — NEWS (15 RSS feeds)  ══════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def _rss(url: str, n: int = 5) -> list[dict]:
    """Parse an RSS/Atom feed and return top n headlines."""
    try:
        r = NEWS_SESSION.get(url, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = root.findall(".//item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        out = []
        for item in items[:n]:
            title   = (item.findtext("title") or
                       item.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
            pubdate = (item.findtext("pubDate") or
                       item.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()[:22]
            src_el  = (item.find("{http://purl.org/dc/elements/1.1/}creator") or
                       item.find("source"))
            source  = src_el.text if src_el is not None and src_el.text else ""
            # Clean HTML entities
            for ent, rep in [("&#39;","'"),("&amp;","&"),("&lt;","<"),
                              ("&gt;",">"),("&quot;",'"'),("&#8217;","'")]:
                title = title.replace(ent, rep)
            out.append({
                "Headline": title[:98],
                "Source":   source[:30] if source else "—",
                "Published":pubdate,
            })
        return out
    except Exception as e:
        log.debug(f"RSS [{url[:55]}]: {e}")
        return []


NEWS_FEEDS = {
    # ── Indian Market ──────────────────────────────────────────────────────
    "🇮🇳 NSE / Nifty / Sensex": (
        "https://news.google.com/rss/search"
        "?q=nifty+sensex+NSE+BSE+india+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    "🏦 Indian Banking & RBI": (
        "https://news.google.com/rss/search"
        "?q=RBI+india+bank+HDFC+ICICI+SBI+Kotak+interest+rate&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    "💻 IT & Tech Stocks": (
        "https://news.google.com/rss/search"
        "?q=TCS+Infosys+Wipro+HCL+Tech+Mahindra+IT+sector+india&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    "🏭 India Economy & Macro": (
        "https://news.google.com/rss/search"
        "?q=india+GDP+CPI+inflation+IIP+GST+government+budget+economy&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    "📊 FII DII Institutional Flow": (
        "https://news.google.com/rss/search"
        "?q=FII+DII+foreign+institutional+investor+india+buying+selling&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    # ── Commodities ────────────────────────────────────────────────────────
    "🛢️  Crude Oil & OPEC": (
        "https://news.google.com/rss/search"
        "?q=crude+oil+brent+WTI+OPEC+oil+price+energy&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    "🥇 Gold & Silver": (
        "https://news.google.com/rss/search"
        "?q=gold+price+silver+bullion+XAU+precious+metals&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    # ── Forex ──────────────────────────────────────────────────────────────
    "💵 Rupee & Forex": (
        "https://news.google.com/rss/search"
        "?q=rupee+dollar+USD+INR+forex+RBI+currency+exchange+rate&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    # ── Global ─────────────────────────────────────────────────────────────
    "🇺🇸 US Markets & Fed": (
        "https://news.google.com/rss/search"
        "?q=US+stock+market+Federal+Reserve+Fed+S%26P500+NASDAQ+rate+hike&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    "🌍 Global Economy": (
        "https://news.google.com/rss/search"
        "?q=global+economy+world+markets+trade+war+inflation+recession+China+Europe&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    "🌏 Geopolitics & Risk": (
        "https://news.google.com/rss/search"
        "?q=geopolitics+war+sanctions+Middle+East+Ukraine+China+Taiwan+risk&hl=en-IN&gl=IN&ceid=IN:en"
    ),
    # ── Authoritative Indian Financial News RSS ────────────────────────────
    "📰 Economic Times — Markets": (
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
    ),
    "📰 LiveMint — Markets": (
        "https://www.livemint.com/rss/markets"
    ),
    "📰 Business Standard — Markets": (
        "https://www.business-standard.com/rss/markets-106.rss"
    ),
    "📰 Reuters — Business": (
        "https://feeds.reuters.com/reuters/businessNews"
    ),
}


def fetch_all_news() -> dict[str, list[dict]]:
    results = {}
    for label, url in NEWS_FEEDS.items():
        log.info(f"Fetching: {label}")
        results[label] = _rss(url, n=5)
        time.sleep(0.3)
    return results


def print_news(all_news: dict):
    for label, items in all_news.items():
        print(f"\n  {B(label)}")
        if not items:
            print(f"    {Y('No headlines fetched — check internet / RSS source')}")
            continue
        for i, h in enumerate(items, 1):
            print(f"  {i}. {h['Headline']}")
            print(f"     {C(h['Source'])}  ·  {h['Published']}")


# ─────────────────────────────────────────────────────────────────────────────
# ══  BLOCK D — OVERALL SENTIMENT  ════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def compute_sentiment(
    us_markets:  list[dict],
    commodities: list[dict],
    usdinr:      dict,
    adv_dec:     dict,
    vix:         dict,
    pcr:         dict,
) -> tuple[str, list[str]]:
    score, signals = 0, []

    def _pct(raw) -> float:
        try:
            return float(str(raw).replace("%","").replace("+","").strip())
        except Exception:
            return 0.0

    # US Markets
    for row in us_markets:
        if "VIX" in str(row.get("Index", "")):
            continue
        p = _pct(row.get("Chg%", "0"))
        if p >  1:  score += 2; signals.append(G(f"US {row['Index']} +{p:.1f}% — strong tailwind"))
        elif p > 0: score += 1
        elif p < -1: score -= 2; signals.append(R(f"US {row['Index']} {p:.1f}% — headwind for India"))
        elif p < 0: score -= 1

    # Crude (inverse for India)
    for row in commodities:
        if "Crude" not in str(row.get("Commodity", "")):
            continue
        p = _pct(row.get("Chg%", "0"))
        if p >  3: score -= 2; signals.append(R(f"Crude oil +{p:.1f}% — import cost pressure, CAD worsens"))
        elif p > 1.5: score -= 1
        elif p < -3: score += 2; signals.append(G(f"Crude oil {p:.1f}% — relief for India, lower inflation"))
        elif p < -1.5: score += 1
        break  # Only count Brent once

    # Gold (risk-off indicator)
    for row in commodities:
        if "Gold" not in str(row.get("Commodity", "")):
            continue
        p = _pct(row.get("Chg%", "0"))
        if p > 1: score -= 1; signals.append(Y(f"Gold +{p:.1f}% — safe haven demand, risk-off mood"))
        break

    # Rupee
    p = _pct(usdinr.get("Chg%", "0"))
    if p >  0.3: score -= 2; signals.append(R(f"Rupee weakening {p:+.2f}% — FII outflows expected"))
    elif p < -0.3: score += 1; signals.append(G("Rupee strengthening — FII inflows positive"))

    # India VIX
    try:
        vix_val = float(str(vix.get("India VIX", "0")).replace(",",""))
        vix_chg = _pct(vix.get("Change%", "0"))
        if vix_val > 20:   score -= 2; signals.append(R(f"India VIX {vix_val:.1f} — PANIC zone, extreme caution"))
        elif vix_val > 15: score -= 1; signals.append(Y(f"India VIX {vix_val:.1f} — elevated volatility"))
        elif vix_val < 12: score += 1; signals.append(G(f"India VIX {vix_val:.1f} — calm, confidence high"))
        if vix_chg > 10:   score -= 1; signals.append(R(f"VIX spiking +{vix_chg:.1f}% — fear escalating"))
    except Exception:
        pass

    # Put-Call Ratio
    try:
        pcr_val = float(str(pcr.get("PCR", "1")).replace(",",""))
        if pcr_val > 1.3:  score += 1; signals.append(G(f"PCR {pcr_val} — put writers active, bullish bias"))
        elif pcr_val < 0.7: score -= 1; signals.append(R(f"PCR {pcr_val} — heavy call OI, bearish bias"))
    except Exception:
        pass

    # Advance / Decline
    try:
        adr = float(str(adv_dec.get("A/D Ratio", "1")).replace(",",""))
        if adr > 1.5:   score += 1; signals.append(G(f"A/D ratio {adr:.2f} — broad participation, healthy"))
        elif adr < 0.7: score -= 1; signals.append(R(f"A/D ratio {adr:.2f} — narrow market, weak breadth"))
    except Exception:
        pass

    if score >= 6:   verdict = G("🟢 STRONGLY BULLISH — Gap-up likely, strong broad participation")
    elif score >= 3: verdict = G("🟢 MILDLY BULLISH — Positive bias; confirm in first 15 min")
    elif score >= 1: verdict = G("🟡 SLIGHT BULLISH LEAN — Cautiously positive; stay selective")
    elif score == 0: verdict = Y("🟡 NEUTRAL — Mixed signals; chop or range-bound session likely")
    elif score >= -2:verdict = R("🔴 MILDLY BEARISH — Negative bias; risk-off tone; limit longs")
    elif score >= -4:verdict = R("🔴 BEARISH — Selling pressure expected; manage positions tight")
    else:            verdict = R("🔴 STRONGLY BEARISH — Gap-down likely; high caution; hedge/reduce")

    return verdict, signals


# ─────────────────────────────────────────────────────────────────────────────
# ══  MASTER RUNNER  ══════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def run_morning_brief():
    now = datetime.now(IST)
    report: dict = {"generated_at": now.isoformat()}

    # ── Banner ───────────────────────────────────────────────────────────────
    W = 78
    print()
    print(Back.BLUE + Fore.WHITE + Style.BRIGHT + " " * W)
    print(Back.BLUE + Fore.WHITE + Style.BRIGHT +
          f"  {'INDIAN MARKET MORNING BRIEF':^{W-4}}  ")
    print(Back.BLUE + Fore.WHITE + Style.BRIGHT +
          f"  {now.strftime('%A, %d %B %Y  |  %I:%M %p IST'):^{W-4}}  ")
    print(Back.BLUE + Fore.WHITE + Style.BRIGHT + " " * W)

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK A — NSE INDIA
    # ═════════════════════════════════════════════════════════════════════════
    print(); print(M("  ◀▶  BLOCK A — NSE INDIA LIVE DATA  ◀▶"))

    section("A-1 · NSE MARKET STATUS")
    ms = nse_market_status(); print_kv(ms); report["market_status"] = ms

    section("A-2 · INDIA VIX — FEAR GAUGE")
    vix = nse_india_vix(); print_kv(vix); report["india_vix"] = vix

    section("A-3 · ALL NSE INDICES SNAPSHOT")
    all_idx = nse_all_indices(); print_tbl(all_idx); report["all_indices"] = all_idx

    section("A-4 · NIFTY 50 — DETAILED VIEW")
    n50 = nse_index_detail("NIFTY 50"); print_kv(n50); report["nifty50"] = n50

    section("A-5 · BANK NIFTY — DETAILED VIEW")
    bn = nse_index_detail("NIFTY BANK"); print_kv(bn); report["banknifty"] = bn

    section("A-6 · NIFTY NEXT 50")
    nn50 = nse_index_detail("NIFTY NEXT 50"); print_kv(nn50); report["nifty_next50"] = nn50

    section("A-7 · NIFTY MIDCAP 50")
    mid = nse_index_detail("NIFTY MIDCAP 50"); print_kv(mid); report["midcap50"] = mid

    section("A-8 · PRE-OPEN SESSION — IEP PRICES")
    print(Y("  ⓘ  IEP = Indicative Equilibrium Price — expected opening price"))
    po = nse_pre_open("NIFTY"); print_tbl(po); report["pre_open"] = po

    section("A-9 · NIFTY OPTION CHAIN — PCR & MAX PAIN")
    pcr = nse_option_chain_pcr("NIFTY"); print_kv(pcr); report["option_pcr"] = pcr

    section("A-10 · MARKET BREADTH — ADVANCE / DECLINE")
    adv = nse_advance_decline(); print_kv(adv); report["advance_decline"] = adv

    section("A-11 · FII / DII INSTITUTIONAL FLOWS — LAST 6 DAYS")
    fii = nse_fii_dii(); print_tbl(fii); report["fii_dii"] = fii

    section("A-12 · TOP 10 GAINERS")
    gainers, losers = nse_gainers_losers()
    print_tbl(gainers); report["gainers"] = gainers

    section("A-13 · TOP 10 LOSERS")
    print_tbl(losers); report["losers"] = losers

    section("A-14 · MOST ACTIVE — BY TRADED VALUE")
    ma = nse_most_active(); print_tbl(ma); report["most_active"] = ma

    section("A-15 · 52-WEEK HIGHS TODAY 🚀")
    highs, lows = nse_52wk()
    print_tbl(highs); report["52wk_highs"] = highs

    section("A-16 · 52-WEEK LOWS TODAY 📉")
    print_tbl(lows); report["52wk_lows"] = lows

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK B — GLOBAL DATA (yfinance)
    # ═════════════════════════════════════════════════════════════════════════
    print(); print(M("  ◀▶  BLOCK B — GLOBAL MARKET DATA (yfinance)  ◀▶"))

    section("B-1 · US MARKETS — OVERNIGHT CLOSE")
    print(Y("  ⓘ  Indian markets follow US overnight moves — highest correlation"))
    us = yf_us_markets(); print_tbl(us); report["us_markets"] = us

    section("B-2 · US BOND YIELDS & DOLLAR INDEX (DXY)")
    print(Y("  ⓘ  Rising DXY / yields = FII outflows from India"))
    bd = yf_bond_yields_dxy(); print_tbl(bd); report["yields_dxy"] = bd

    section("B-3 · USD / INR — RUPEE HEALTH")
    rupee = yf_usdinr(); print_kv(rupee); report["usdinr"] = rupee

    section("B-4 · CRUDE OIL, GOLD & KEY COMMODITIES")
    comm = yf_commodities(); print_tbl(comm); report["commodities"] = comm

    section("B-5 · GLOBAL INDICES — ASIA + EUROPE")
    glob = yf_global_indices(); print_tbl(glob); report["global_indices"] = glob

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK C — NEWS (15 feeds)
    # ═════════════════════════════════════════════════════════════════════════
    print(); print(M("  ◀▶  BLOCK C — CURRENT MARKET NEWS (LOCAL + GLOBAL)  ◀▶"))
    section("C · LIVE NEWS HEADLINES — 15 FEEDS")
    all_news = fetch_all_news()
    print_news(all_news)
    report["news"] = {k: v for k, v in all_news.items()}

    # ═════════════════════════════════════════════════════════════════════════
    # BLOCK D — OVERALL SENTIMENT VERDICT
    # ═════════════════════════════════════════════════════════════════════════
    print(); print(M("  ◀▶  BLOCK D — OVERALL SENTIMENT VERDICT  ◀▶"))
    section("★  TODAY'S MARKET SENTIMENT SUMMARY")
    verdict, signals = compute_sentiment(us, comm, rupee, adv, vix, pcr)

    print(f"\n  {B('Overall Verdict:')}")
    print(f"  {verdict}")
    print()
    if signals:
        print(f"  {B('Key Signals Driving Verdict:')}")
        for sig in signals:
            print(f"    → {sig}")

    print()
    print(f"  {B('Trader Checklist:')}")
    checklist = [
        ("📍", "First 15 min",      "often fake move — confirm direction before trading"),
        ("📍", "India VIX > 18",    "reduce position size — volatility kills P&L"),
        ("📍", "FII selling 3+ days","medium-term bearish — wait for reversal sign"),
        ("📍", "PCR extremes",       "< 0.7 or > 1.5 = sentiment extreme = reversal risk"),
        ("📍", "Crude > 90",         "Airline/Shipping/Paints/Chemicals under pressure"),
        ("📍", "Strong DXY",         "FII outflows, Rupee weakness, Nifty headwind"),
        ("📍", "Fed / RBI day",      "VERY dangerous to hold overnight — expect big moves"),
        ("📍", "Earnings season",    "check heavyweights: HDFC/TCS/Reliance guide index"),
        ("📍", "Max Pain",           "Nifty often gravitates toward Max Pain near expiry"),
        ("📍", "A/D ratio < 0.7",    "only frontline stocks rising — weak rally, avoid chasing"),
    ]
    for icon, topic, note in checklist:
        print(f"    {icon}  {Y(B(topic)):<30} {note}")

    report["sentiment_verdict"] = str(verdict)
    report["sentiment_signals"] = [str(s) for s in signals]

    # Footer
    print()
    print(C("═" * 78))
    print(B(C(
        f"  Morning Brief complete · "
        f"NSE India + yfinance + 15 News Feeds · "
        f"{now.strftime('%d %b %Y %H:%M IST')}"
    )))
    print(C("═" * 78))
    print()

    # Save JSON
    fname = f"market_brief_{now.strftime('%Y-%m-%d')}.json"
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        log.info(f"Report saved → {fname}")
    except Exception as e:
        log.warning(f"Could not save JSON: {e}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# ══  SCHEDULER  ══════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduled(time_str: str = "08:00"):
    print(B(C(f"\n  ⏰  Scheduler active — Daily brief at {time_str} IST every day")))
    print(Y(f"  Running once immediately at startup…\n"))
    run_morning_brief()
    schedule.every().day.at(time_str).do(run_morning_brief)
    while True:
        schedule.run_pending()
        nxt = schedule.next_run()
        if nxt:
            secs = (nxt - datetime.now()).total_seconds()
            log.info(f"Next brief in {secs/3600:.1f}h")
        time.sleep(60)


# ─────────────────────────────────────────────────────────────────────────────
# ══  ENTRY POINT  ════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Indian Market Morning Brief — NSE + Global + News",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  Examples:
    python indian_market_morning_brief.py
    python indian_market_morning_brief.py --schedule
    python indian_market_morning_brief.py --schedule --time 07:30

  SSL Note:
    If you get SSL errors (WRONG_VERSION_NUMBER), you are behind a
    corporate proxy.  The script auto-handles this — no config needed.
        """,
    )
    parser.add_argument("--schedule", action="store_true",
                        help="Run daily at specified IST time")
    parser.add_argument("--time", default="08:00",
                        help="Schedule time HH:MM IST (default: 08:00)")
    args = parser.parse_args()
    if args.schedule:
        run_scheduled(args.time)
    else:
        run_morning_brief()


if __name__ == "__main__":
    main()