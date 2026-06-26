import pandas as pd
import numpy as np

df = pd.read_csv('NIFTY 50_5minute_sdv6_trades.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time']  = pd.to_datetime(df['exit_time'])
df['hold_bars']  = ((df['exit_time'] - df['entry_time']).dt.total_seconds() / 300).round(0).astype(int)
df['entry_hour'] = df['entry_time'].dt.hour
df['entry_min']  = df['entry_time'].dt.minute
df['exit_hour']  = df['exit_time'].dt.hour

wins   = df[df['pnl'] > 0]
small  = wins[(wins['pnl'] > 0) & (wins['pnl'] < 50)]
big    = wins[wins['pnl'] >= 50]
losses = df[df['pnl'] <= 0]

print('=== OVERALL ===')
print(f'Total trades        : {len(df)}')
print(f'Winners             : {len(wins)} ({len(wins)/len(df)*100:.1f}%)')
print(f'  Small wins <50 pts: {len(small)} ({len(small)/len(df)*100:.1f}%)')
print(f'  Big   wins >=50   : {len(big)}  ({len(big)/len(df)*100:.1f}%)')
print(f'Losses              : {len(losses)} ({len(losses)/len(df)*100:.1f}%)')
print()

print('=== SMALL WINS (<50 pts) STATS ===')
print(f'Count               : {len(small)}')
print(f'Avg PnL             : {small.pnl.mean():.1f} pts')
print(f'Avg MFE             : {small.mfe_pts.mean():.1f} pts')
print(f'Avg MAE             : {small.mae_pts.mean():.1f} pts')
slip = small.mfe_pts - small.pnl
print(f'Avg pts left on table (MFE-pnl): {slip.mean():.1f} pts')
print(f'Max MFE in small wins: {small.mfe_pts.max():.1f} pts  <-- had this much room')
print(f'Avg hold bars       : {small.hold_bars.mean():.1f} ({small.hold_bars.mean()*5:.0f} mins)')
print()

print('=== BIG WINS (>=50 pts) STATS ===')
print(f'Count               : {len(big)}')
print(f'Avg PnL             : {big.pnl.mean():.1f} pts')
print(f'Avg MFE             : {big.mfe_pts.mean():.1f} pts')
print(f'Avg hold bars       : {big.hold_bars.mean():.1f} ({big.hold_bars.mean()*5:.0f} mins)')
print()

print('=== EXIT REASON: SMALL WINS ===')
print(small['exit_reason'].value_counts().to_string())
print()

print('=== EXIT REASON: BIG WINS ===')
print(big['exit_reason'].value_counts().to_string())
print()

print('=== EXIT REASON: LOSSES ===')
print(losses['exit_reason'].value_counts().to_string())
print()

print('=== DIRECTION BREAKDOWN ===')
print('Small wins by direction:')
print(small['direction'].value_counts().to_string())
print('Big wins by direction:')
print(big['direction'].value_counts().to_string())
print()

print('=== ENTRY PATH BREAKDOWN ===')
print('Small wins by entry path:')
print(small['entry_path'].value_counts().to_string())
print('Big wins by entry path:')
print(big['entry_path'].value_counts().to_string())
print()

print('=== ENTRY SESSION (HOUR) ===')
print('Small wins entry hour:')
print(small['entry_hour'].value_counts().sort_index().to_string())
print('Big wins entry hour:')
print(big['entry_hour'].value_counts().sort_index().to_string())
print()

print('=== HOLD TIME DISTRIBUTION: SMALL WINS ===')
bins = [0, 3, 6, 12, 24, 50, 999]
labels = ['<15m', '15-30m', '30-60m', '1-2h', '2-4h', '>4h']
small_copy = small.copy()
small_copy['hold_bucket'] = pd.cut(small_copy['hold_bars'], bins=bins, labels=labels)
print(small_copy['hold_bucket'].value_counts().sort_index().to_string())
print()

print('=== MFE BUCKETS: SMALL WINS (how far price went before exit) ===')
mfe_bins = [0, 20, 40, 60, 80, 100, 999]
mfe_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '>100']
small_copy['mfe_bucket'] = pd.cut(small_copy['mfe_pts'], bins=mfe_bins, labels=mfe_labels)
print(small_copy['mfe_bucket'].value_counts().sort_index().to_string())
print()

print('=== SMALL WINS WHERE MFE > 60 (big potential, small exit) ===')
missed = small[small['mfe_pts'] > 60]
print(f'Count: {len(missed)} trades had >60pt MFE but exited with <50pt profit')
print(f'Avg PnL  : {missed.pnl.mean():.1f}')
print(f'Avg MFE  : {missed.mfe_pts.mean():.1f}')
print(f'Exit reasons:')
print(missed['exit_reason'].value_counts().to_string())
print()

print('=== SMALL WINS: BE TRIGGERED ===')
print(small['be_triggered'].value_counts().to_string())
print()

print('=== SAMPLE SMALL WIN TRADES (worst slip) ===')
small_sorted = small.copy()
small_sorted['slip'] = small_sorted['mfe_pts'] - small_sorted['pnl']
top_slip = small_sorted.nlargest(10, 'slip')[
    ['entry_time','exit_time','direction','entry_path','entry_price','exit_price',
     'pnl','mfe_pts','mae_pts','slip','exit_reason','be_triggered','hold_bars']
]
print(top_slip.to_string())
