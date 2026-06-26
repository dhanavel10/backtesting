import yfinance as yf

# Download NIFTY 50 data
data = yf.download(
            "^NSEI",
                period="55d",
                        interval="5m"
                        )

# Save as CSV
csv_file = "nifty50.csv"
data.to_csv(csv_file)

print(f"CSV saved successfully: {csv_file}")
