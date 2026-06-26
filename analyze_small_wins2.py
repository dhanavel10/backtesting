import pandas as pd
import numpy as np

df = pd.read_csv('NIFTY 50_5minute_sdv6_trades.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time']  = pd.to_datetime(df['exit_time'])
df['hold_bars']  = ((df['exit_time'] - df['entry_time']).dt.total_seconds() / 300).round(0).astype(int)
df['entry_hour'] = df['entry_time'].dt.hour
df['slip']       = df['mfe_pts'] - df['pnl']

small = df[(df['pnl'] > 0) & (df['pnl'] < 50)]
big   = df[df['pnl'] >= 50]

# ── 1. Giveback lock victims ──────────────────────────────────────────────────
# ARM=45, KEEP=0.5 → lock fires once MFE hits 45, pins SL at entry+22.5
# These are trades that went 45+ pts favorable but closed below 50 pts
print('=== GIVEBACK LOCK VICTIMS (mfe>=45 but pnl<50) ===')
giveback_victims = small[small['mfe_pts'] >= 45]
print(f'Count: {len(giveback_victims)}')
print(f'Avg MFE : {giveback_victims.mfe_pts.mean():.1f} pts  (had this much room)')
print(f'Avg PnL : {giveback_victims.pnl.mean():.1f} pts  (only kept this)')
print(f'Avg slip: {giveback_victims.slip.mean():.1f} pts left on table')
print(f'Exit reasons:')
print(giveback_victims['exit_reason'].value_counts().to_string())
print(f'BE triggered:')
print(giveback_victims['be_triggered'].value_counts().to_string())
print()

# ── 2. Swing trail victims ────────────────────────────────────────────────────
print('=== TRAIL_SL SMALL WINS (trail too tight) ===')
trail_small = small[small['exit_reason'] == 'TRAIL_SL']
print(f'Count: {len(trail_small)}')
print(f'Avg PnL : {trail_small.pnl.mean():.1f} pts')
print(f'Avg MFE : {trail_small.mfe_pts.mean():.1f} pts')
print(f'Avg slip: {trail_small.slip.mean():.1f} pts (exit - mfe gap)')
print(f'Avg hold: {trail_small.hold_bars.mean():.1f} bars ({trail_small.hold_bars.mean()*5:.0f} min)')
print()

# ── 3. EOD forced exits ───────────────────────────────────────────────────────
print('=== EOD_EXIT SMALL WINS (ran out of time) ===')
eod_small = small[small['exit_reason'] == 'EOD_EXIT']
print(f'Count: {len(eod_small)}')
print(f'Avg PnL : {eod_small.pnl.mean():.1f} pts')
print(f'Avg MFE : {eod_small.mfe_pts.mean():.1f} pts')
print(f'Entry hour distribution:')
print(eod_small['entry_hour'].value_counts().sort_index().to_string())
print()

# ── 4. Reversal after big move (SL hit after good MFE) ───────────────────────
print('=== STOP_LOSS SMALL WINS (reversed after good move) ===')
sl_small = small[small['exit_reason'] == 'STOP_LOSS']
print(f'Count: {len(sl_small)}')
print(f'Avg PnL : {sl_small.pnl.mean():.1f} pts')
print(f'Avg MFE : {sl_small.mfe_pts.mean():.1f} pts')
print(f'Avg MAE : {sl_small.mae_pts.mean():.1f} pts')
print(f'BE triggered (has profit stop):')
print(sl_small['be_triggered'].value_counts().to_string())
print()

# ── 5. Dead zone trades (MFE < 45, never armed anything) ─────────────────────
print('=== DEAD ZONE TRADES (mfe<45, never triggered giveback lock) ===')
dead_zone = small[small['mfe_pts'] < 45]
print(f'Count: {len(dead_zone)}')
print(f'Avg PnL : {dead_zone.pnl.mean():.1f} pts')
print(f'Avg MFE : {dead_zone.mfe_pts.mean():.1f} pts  <-- barely moved')
print(f'Exit reasons:')
print(dead_zone['exit_reason'].value_counts().to_string())
print()

# ── 6. Short vs Long deep dive ────────────────────────────────────────────────
print('=== SHORT SMALL WINS DETAIL ===')
short_small = small[small['direction']=='short']
print(f'Count: {len(short_small)}  ({len(short_small)/len(small)*100:.0f}% of small wins)')
print(f'Avg PnL: {short_small.pnl.mean():.1f}  Avg MFE: {short_small.mfe_pts.mean():.1f}')
print(f'Exit reasons:')
print(short_small['exit_reason'].value_counts().to_string())
print()

# ── 7. Compare: what big wins do differently ──────────────────────────────────
print('=== WHAT DIFFERENTIATES BIG WINS ===')
print(f'Big wins avg hold bars : {big.hold_bars.mean():.1f} ({big.hold_bars.mean()*5:.0f} min)')
print(f'Small wins avg hold    : {small.hold_bars.mean():.1f} ({small.hold_bars.mean()*5:.0f} min)')
print(f'Big wins BE rate       : {big.be_triggered.mean()*100:.0f}%')
print(f'Small wins BE rate     : {small.be_triggered.mean()*100:.0f}%')
print(f'Big wins avg MAE       : {big.mae_pts.mean():.1f} pts (how much they went against)')
print(f'Small wins avg MAE     : {small.mae_pts.mean():.1f} pts')
print()

# ── 8. Impact if ARM moved to 60 ──────────────────────────────────────────────
print('=== SIMULATED: if GIVEBACK_ARM raised to 60 (mfe<60 never locked) ===')
would_survive = small[(small['mfe_pts'] >= 45) & (small['mfe_pts'] < 60)]
print(f'Trades that fired at arm=45 but would NOT fire at arm=60: {len(would_survive)}')
print(f'Their avg MFE: {would_survive.mfe_pts.mean():.1f} pts (still had room)')
print(f'Their avg pnl if they ran to 60% of MFE instead: {(would_survive.mfe_pts * 0.6).mean():.1f} pts')
print()

# ── 9. Late-entry EOD problem ────────────────────────────────────────────────
print('=== LATE ENTRIES (after 14:00) ===')
late = df[df['entry_hour'] >= 14]
late_small = late[(late['pnl'] > 0) & (late['pnl'] < 50)]
late_big   = late[late['pnl'] >= 50]
late_loss  = late[late['pnl'] <= 0]
print(f'Total late entries    : {len(late)}')
print(f'Small wins            : {len(late_small)}  avg pnl={late_small.pnl.mean():.1f}' if len(late_small) else 'Small wins: 0')
print(f'Big wins              : {len(late_big)}' if len(late_big) else 'Big wins: 0')
print(f'Losses                : {len(late_loss)}')
