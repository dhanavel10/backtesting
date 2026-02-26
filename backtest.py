# =====================================================
# 9 EMA / 26 EMA Strategy with 0.1% Confirmation
# Indian Market (IST Timezone) + Trade Analysis
# =====================================================

import backtrader as bt
import yfinance as yf
import pandas as pd

# ----------------------------
# 1️⃣ DOWNLOAD DATA
# ----------------------------
stock = "^NSEI"  # Nifty 50 Index
start_date = "2022-01-01"
end_date = "2023-01-01"
interval = "1d"   # Change to "5m" for intraday

data = yf.download(stock, start=start_date, end=end_date, interval=interval)

# Fix MultiIndex columns (new yfinance issue)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Convert timezone safely (important for intraday)
if interval != "1d":
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.index = data.index.tz_convert("Asia/Kolkata")

data.dropna(inplace=True)

# ----------------------------
# 2️⃣ STRATEGY
# ----------------------------

class EMAConfirmationStrategy(bt.Strategy):

    params = dict(confirm_percent=0.001)  # 0.1%

    def __init__(self):
        self.ema9 = bt.ind.EMA(self.data.close, period=9)
        self.ema26 = bt.ind.EMA(self.data.close, period=26)
        self.crossover = bt.ind.CrossOver(self.ema9, self.ema26)

        self.cross_price = None
        self.waiting_long = False
        self.waiting_short = False

        self.highest_price = None
        self.lowest_price = None

        self.trade_count = 0

    def next(self):

        # ============================
        # Detect EMA Crossover
        # ============================
        if not self.position:

            if self.crossover > 0:
                self.cross_price = self.data.close[0]
                self.waiting_long = True
                self.waiting_short = False

            elif self.crossover < 0:
                self.cross_price = self.data.close[0]
                self.waiting_short = True
                self.waiting_long = False

        # ============================
        # Confirm Long Entry
        # ============================
        if self.waiting_long and not self.position:
            if self.data.close[0] >= self.cross_price * (1 + self.p.confirm_percent):
                self.buy()
                self.waiting_long = False
                self.highest_price = self.data.close[0]

        # ============================
        # Confirm Short Entry
        # ============================
        if self.waiting_short and not self.position:
            if self.data.close[0] <= self.cross_price * (1 - self.p.confirm_percent):
                self.sell()
                self.waiting_short = False
                self.lowest_price = self.data.close[0]

        # ============================
        # Long Trailing Exit
        # ============================
        if self.position.size > 0:
            if self.data.close[0] > self.highest_price:
                self.highest_price = self.data.close[0]

            if self.data.close[0] <= self.highest_price * (1 - self.p.confirm_percent):
                self.close()

        # ============================
        # Short Trailing Exit
        # ============================
        if self.position.size < 0:
            if self.data.close[0] < self.lowest_price:
                self.lowest_price = self.data.close[0]

            if self.data.close[0] >= self.lowest_price * (1 + self.p.confirm_percent):
                self.close()

    # ============================
    # TRADE ANALYSIS
    # ============================
    def notify_trade(self, trade):

        if trade.isclosed:

            self.trade_count += 1

            entry_date = bt.num2date(trade.dtopen)
            exit_date = bt.num2date(trade.dtclose)

            pnl = round(trade.pnlcomm, 2)

            trade_type = "LONG" if trade.size > 0 else "SHORT"

            print("\n==============================")
            print(f"Trade #{self.trade_count}")
            print("Type       :", trade_type)
            print("Entry Date :", entry_date)
            print("Exit Date  :", exit_date)
            print("Net PnL    :", pnl)

            if pnl > 0:
                print("Result     : PROFIT ✅")
            else:
                print("Result     : LOSS ❌")

            print("==============================")

# ----------------------------
# 3️⃣ BACKTEST ENGINE
# ----------------------------

cerebro = bt.Cerebro()

data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)

cerebro.addstrategy(EMAConfirmationStrategy)

cerebro.broker.setcash(100000)  # ₹1,00,000
cerebro.broker.setcommission(commission=0.001)

print("Starting Portfolio Value: ₹%.2f" % cerebro.broker.getvalue())

cerebro.run()

print("Final Portfolio Value: ₹%.2f" % cerebro.broker.getvalue())

cerebro.plot()