"""
Filter detected patterns to find ones that LOOK like the user's example image:
descending triangle = clearly negative upper slope + nearly-flat lower slope.
Plot the top 6.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from contraction_detector import load_csv, scan_history, detect_breakout, CFG

df = load_csv('NIFTY 50_5minute.csv')
contractions = scan_history(df, CFG)

# score each contraction by "descending-triangle-ness":
#   * upper slope strongly negative
#   * lower slope close to zero (|slope| small)
scored = []
for c in contractions:
    # normalize slopes to price-per-bar / avg_price
    avg_price = (c.upper_intercept + c.upper_slope*c.start_idx) * 0.5  # rough
    if avg_price <= 0:
        continue
    upper_norm = c.upper_slope / avg_price       # negative, want large |this|
    lower_norm = abs(c.lower_slope) / avg_price  # want small
    score = (-upper_norm) - 2.0 * lower_norm     # reward steep down upper, flat lower
    scored.append((score, c))

scored.sort(key=lambda t: t[0], reverse=True)
top = [c for s, c in scored[:6]]

def plot_one(ax, df, c, br, title=""):
    s, e = c.start_idx, c.end_idx
    pad = 15
    a, b = max(0, s - pad), min(len(df), e + pad + 5)
    seg = df.iloc[a:b]
    for gi, row in seg.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.vlines(gi, row["low"], row["high"], color=color, linewidth=0.7, zorder=1)
        ax.add_patch(mpatches.Rectangle(
            (gi - 0.3, min(row["open"], row["close"])),
            0.6, max(abs(row["close"] - row["open"]), 1e-3),
            facecolor=color, edgecolor=color, zorder=2,
        ))
    x_line = np.array([s, e + 3], dtype=float)
    ax.plot(x_line, c.upper_slope * x_line + c.upper_intercept,
            color="#3da5ff", lw=1.8, zorder=3)
    ax.plot(x_line, c.lower_slope * x_line + c.lower_intercept,
            color="#3da5ff", lw=1.8, zorder=3)
    ax.axvspan(s, e, color="#3da5ff", alpha=0.06, zorder=0)
    if br is not None:
        col = "#00e676" if br.breakout_direction == "up" else "#ff5252"
        ax.scatter([br.breakout_idx], [br.breakout_price], color=col, s=80, zorder=5,
                   marker="^" if br.breakout_direction == "up" else "v",
                   edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=9, color="#ddd")
    ax.set_xlim(a - 0.5, b - 0.5)
    ax.tick_params(labelsize=7, colors="#aaa")
    ax.grid(alpha=0.15)
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values():
        sp.set_color("#30363d")

fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor="#0d1117")
for ax, c in zip(axes.flatten(), top):
    br = detect_breakout(df, c, CFG, look_forward=30)
    t1 = df['date'].iloc[c.start_idx].strftime("%Y-%m-%d %H:%M")
    t2 = df['date'].iloc[c.end_idx].strftime("%H:%M")
    ttl = (f"{t1} → {t2}  |  Q={c.quality:.2f}  comp={c.compression:.2f}  "
           f"BO={br.breakout_direction if br else 'none'}")
    plot_one(ax, df, c, br, ttl)
plt.tight_layout()
out = "top_dec.png"
plt.savefig(out, dpi=110, facecolor="#0d1117")
print("saved", out)