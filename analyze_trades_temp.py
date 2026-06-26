import pandas as pd
import numpy as np

df = pd.read_csv(r'd:\backtest.ai\NIFTY 50_5minute_sdv6_trades.csv')
print("=== COLUMNS ===")
print(df.columns.tolist())
print("\n=== FIRST 3 ROWS ===")
print(df.head(3).to_string())
print("\n=== DTYPES ===")
print(df.dtypes)
print(f"\nTotal rows: {len(df)}")

# Identify PnL column
pnl_col = [c for c in df.columns if 'pnl' in c.lower() or 'profit' in c.lower() or 'pl' in c.lower()]
print("PnL columns:", pnl_col)

# Use first pnl column found
pnl = pnl_col[0] if pnl_col else None
print(f"Using PnL column: {pnl}")

if pnl:
    df[pnl] = pd.to_numeric(df[pnl], errors='coerce')
    total = len(df)

    # Segments
    small_winners = df[(df[pnl] > 0) & (df[pnl] < 50)]
    medium_winners = df[(df[pnl] >= 50) & (df[pnl] < 100)]
    large_winners = df[df[pnl] >= 100]
    all_winners = df[df[pnl] > 0]
    losers = df[df[pnl] <= 0]

    print("\n" + "="*60)
    print("=== Q1: TOTAL TRADES & SMALL WINNERS ===")
    print(f"Total trades: {total}")
    print(f"Small winners (0 < pnl < 50): {len(small_winners)}")

    print("\n" + "="*60)
    print("=== Q2: EXIT_REASON DISTRIBUTION FOR SMALL WINNERS ===")
    if 'exit_reason' in df.columns:
        er = small_winners['exit_reason'].value_counts()
        er_pct = small_winners['exit_reason'].value_counts(normalize=True) * 100
        for val in er.index:
            print(f"  {val}: {er[val]} ({er_pct[val]:.1f}%)")
    else:
        print("  exit_reason column not found")
        exit_cols = [c for c in df.columns if 'exit' in c.lower() or 'reason' in c.lower()]
        print(f"  Similar columns: {exit_cols}")

    print("\n" + "="*60)
    print("=== Q3: ENTRY_PATH DISTRIBUTION FOR SMALL WINNERS ===")
    if 'entry_path' in df.columns:
        ep = small_winners['entry_path'].value_counts()
        ep_pct = small_winners['entry_path'].value_counts(normalize=True) * 100
        for val in ep.index:
            print(f"  {val}: {ep[val]} ({ep_pct[val]:.1f}%)")
    else:
        print("  entry_path column not found")
        path_cols = [c for c in df.columns if 'path' in c.lower() or 'entry' in c.lower()]
        print(f"  Similar columns: {path_cols}")

    print("\n" + "="*60)
    print("=== Q4: DIRECTION DISTRIBUTION FOR SMALL WINNERS ===")
    dir_col = None
    for c in ['direction', 'side', 'type', 'trade_type']:
        if c in df.columns:
            dir_col = c
            break
    if dir_col:
        dc = small_winners[dir_col].value_counts()
        dc_pct = small_winners[dir_col].value_counts(normalize=True) * 100
        for val in dc.index:
            print(f"  {val}: {dc[val]} ({dc_pct[val]:.1f}%)")
    else:
        print("  direction column not found")
        print(f"  All columns: {df.columns.tolist()}")

    print("\n" + "="*60)
    print("=== Q5: AVG MFE_PTS FOR SMALL WINNERS ===")
    mfe_col = [c for c in df.columns if 'mfe' in c.lower()]
    print(f"MFE columns found: {mfe_col}")
    if mfe_col:
        c = mfe_col[0]
        df[c] = pd.to_numeric(df[c], errors='coerce')
        print(f"  Avg {c} for small winners: {small_winners[c].mean():.4f}")
        print(f"  Median {c} for small winners: {small_winners[c].median():.4f}")
    else:
        print("  No MFE column found")

    print("\n" + "="*60)
    print("=== Q6: AVG MAE_PTS FOR SMALL WINNERS ===")
    mae_col = [c for c in df.columns if 'mae' in c.lower()]
    print(f"MAE columns found: {mae_col}")
    if mae_col:
        c = mae_col[0]
        df[c] = pd.to_numeric(df[c], errors='coerce')
        print(f"  Avg {c} for small winners: {small_winners[c].mean():.4f}")
        print(f"  Median {c} for small winners: {small_winners[c].median():.4f}")
    else:
        print("  No MAE column found")

    print("\n" + "="*60)
    print("=== Q7: BE_TRIGGERED DISTRIBUTION FOR SMALL WINNERS ===")
    be_col = [c for c in df.columns if 'be' in c.lower() and 'trigger' in c.lower()]
    if not be_col:
        be_col = [c for c in df.columns if 'be_' in c.lower() or '_be' in c.lower()]
    print(f"BE columns found: {be_col}")
    if be_col:
        c = be_col[0]
        bc = small_winners[c].value_counts()
        bc_pct = small_winners[c].value_counts(normalize=True) * 100
        for val in bc.index:
            print(f"  {val}: {bc[val]} ({bc_pct[val]:.1f}%)")
    else:
        print("  No be_triggered column found")

    print("\n" + "="*60)
    print("=== Q8: LARGE WINNERS (pnl >= 50) STATS ===")
    large = df[df[pnl] >= 50]
    print(f"Large winners count: {len(large)}")
    if mfe_col:
        print(f"  Avg MFE: {large[mfe_col[0]].mean():.4f}")
    if mae_col:
        print(f"  Avg MAE: {large[mae_col[0]].mean():.4f}")
    if 'exit_reason' in df.columns:
        print("  Top 3 exit reasons:")
        top3 = large['exit_reason'].value_counts().head(3)
        top3_pct = large['exit_reason'].value_counts(normalize=True).head(3) * 100
        for val in top3.index:
            print(f"    {val}: {top3[val]} ({top3_pct[val]:.1f}%)")

    print("\n" + "="*60)
    print("=== Q9: TIME OF DAY PATTERNS FOR SMALL WINNERS ===")
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    print(f"Time columns found: {time_cols}")
    for tc in time_cols[:3]:
        print(f"\n  Column: {tc}")
        try:
            dt = pd.to_datetime(df[tc], errors='coerce')
            hours = dt.dt.hour
            sw_hours = hours[small_winners.index]
            hc = sw_hours.value_counts().sort_index()
            for h, cnt in hc.items():
                print(f"    Hour {h:02d}: {cnt}")
        except Exception as e:
            print(f"    Error parsing {tc}: {e}")

    print("\n" + "="*60)
    print("=== Q10: SR_TP DISTRIBUTION FOR SMALL WINNERS ===")
    srtp_col = [c for c in df.columns if 'sr_tp' in c.lower() or ('sr' in c.lower() and 'tp' in c.lower())]
    print(f"SR_TP columns found: {srtp_col}")
    if srtp_col:
        c = srtp_col[0]
        sc = small_winners[c].value_counts()
        sc_pct = small_winners[c].value_counts(normalize=True) * 100
        for val in sc.index:
            print(f"  {val}: {sc[val]} ({sc_pct[val]:.1f}%)")
    else:
        print("  No sr_tp column found")
        # Show all columns for reference
        print(f"  All columns: {df.columns.tolist()}")

    print("\n" + "="*60)
    print("=== LOSING TRADES (pnl <= 0) ===")
    print(f"  Count: {len(losers)}")
    print(f"  Avg PnL: {losers[pnl].mean():.4f}")
    print(f"  Min PnL: {losers[pnl].min():.4f}")

    print("\n" + "="*60)
    print("=== WINNER BREAKDOWN ===")
    print(f"  Small winners (0 < pnl < 50):   {len(small_winners)} ({100*len(small_winners)/total:.1f}%)")
    print(f"  Medium winners (50 <= pnl < 100): {len(medium_winners)} ({100*len(medium_winners)/total:.1f}%)")
    print(f"  Large winners (pnl >= 100):       {len(large_winners)} ({100*len(large_winners)/total:.1f}%)")
    print(f"  All winners:                      {len(all_winners)} ({100*len(all_winners)/total:.1f}%)")
    print(f"  All losers:                       {len(losers)} ({100*len(losers)/total:.1f}%)")

    print("\n" + "="*60)
    print("=== PnL SUMMARY STATS (ALL TRADES) ===")
    print(df[pnl].describe())
