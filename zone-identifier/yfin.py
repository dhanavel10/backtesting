import yfinance as yf
from datetime import datetime
import pandas as pd

# NIFTY 50 Yahoo Finance symbol
ticker = "^NSEI"

# Download last 60 days
df = yf.download(
    ticker,
    period="3d",
    interval="1m",
    auto_adjust=False
)

# Flatten columns if yfinance returns MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Convert the datetime index to IST (Asia/Kolkata)
if df.index.tz is None:
    # tz-naive -> assume UTC, then convert
    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
else:
    # already tz-aware -> just convert
    df.index = df.index.tz_convert("Asia/Kolkata")

# Save to CSV
filename = "nifty50_1d_1m.csv"
df.to_csv(filename)

print(df.head())
print(f"\nRows downloaded: {len(df)}")
print(f"Saved to: {filename}")