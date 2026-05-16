"""
Generate a FULL visual report of every detected contraction.

Outputs:
  - all_contractions.pdf    : one page per contraction, with stats
  - charts/contraction_NNNN.png  : individual PNG per contraction (optional)
  - summary.html            : index page with thumbnails + stats table
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from contraction_detector import load_csv, scan_history, detect_breakout, CFG

CSV_IN = "NIFTY 50_5minute.csv"
OUT_DIR = ""
PNG_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(PNG_DIR, exist_ok=True)

SAVE_INDIVIDUAL_PNGS = True   # set False if you only want the PDF


def plot_contraction(ax, df, c, br, idx_label=""):
    """Plot a single contraction with candles, trendlines, and breakout marker."""
    s, e = c.start_idx, c.end_idx
    pad_left = 20
    pad_right = 20
    a = max(0, s - pad_left)
    b = min(len(df), e + pad_right)
    seg = df.iloc[a:b]

    # candles
    for gi, row in seg.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.vlines(gi, row["low"], row["high"], color=color, linewidth=0.8, zorder=1)
        ax.add_patch(mpatches.Rectangle(
            (gi - 0.35, min(row["open"], row["close"])),
            0.7, max(abs(row["close"] - row["open"]), 1e-3),
            facecolor=color, edgecolor=color, zorder=2,
        ))

    # trendlines (extended slightly past pattern to apex direction)
    x_line = np.array([s, min(e + 5, b - 1)], dtype=float)
    y_up = c.upper_slope * x_line + c.upper_intercept
    y_lo = c.lower_slope * x_line + c.lower_intercept
    ax.plot(x_line, y_up, color="#3da5ff", lw=1.8, zorder=3, label="Upper trendline")
    ax.plot(x_line, y_lo, color="#3da5ff", lw=1.8, zorder=3, label="Lower trendline")

    # shade contraction region
    ax.axvspan(s, e, color="#3da5ff", alpha=0.07, zorder=0)

    # breakout marker
    if br is not None:
        col = "#00e676" if br.breakout_direction == "up" else "#ff5252"
        marker = "^" if br.breakout_direction == "up" else "v"
        ax.scatter([br.breakout_idx], [br.breakout_price], color=col, s=120, zorder=5,
                   marker=marker, edgecolor="white", linewidth=1.2,
                   label=f"Breakout {br.breakout_direction}")

    # title + labels
    t1 = df["date"].iloc[c.start_idx].strftime("%Y-%m-%d %H:%M")
    t2 = df["date"].iloc[c.end_idx].strftime("%Y-%m-%d %H:%M")
    bo_str = f"{br.breakout_direction.upper()} @ {br.breakout_price:.2f} ({br.bars_after_pattern} bars later)" if br else "no breakout"
    title = (f"#{idx_label}   {t1}  →  {t2}   ({c.end_idx - c.start_idx + 1} bars)\n"
             f"Q={c.quality:.2f}  comp={c.compression:.2f}  "
             f"R²(up)={c.upper_r2:.2f}  R²(lo)={c.lower_r2:.2f}  "
             f"contain={c.containment:.0%}  pivots={c.n_pivot_highs}H/{c.n_pivot_lows}L\n"
             f"Breakout: {bo_str}")
    ax.set_title(title, fontsize=10, color="#ddd", loc="left")
    ax.set_xlim(a - 0.5, b - 0.5)
    ax.set_xlabel("Bar index", color="#aaa", fontsize=8)
    ax.set_ylabel("Price", color="#aaa", fontsize=8)
    ax.tick_params(labelsize=8, colors="#aaa")
    ax.grid(alpha=0.15)
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.legend(loc="upper left", fontsize=7, facecolor="#0d1117",
              edgecolor="#30363d", labelcolor="#ccc", framealpha=0.8)


def main():
    print("Loading data...")
    df = load_csv(CSV_IN)
    print(f"  {len(df):,} bars")

    print("Scanning for contractions...")
    contractions = scan_history(df, CFG)
    print(f"  found {len(contractions)} contractions")

    # build metadata table
    rows = []
    for i, c in enumerate(contractions):
        br = detect_breakout(df, c, CFG, look_forward=30)
        rows.append({
            "id": i + 1,
            "start_time": df["date"].iloc[c.start_idx],
            "end_time": df["date"].iloc[c.end_idx],
            "duration_bars": c.end_idx - c.start_idx + 1,
            "quality": round(c.quality, 3),
            "compression": round(c.compression, 3),
            "upper_r2": round(c.upper_r2, 3),
            "lower_r2": round(c.lower_r2, 3),
            "containment": round(c.containment, 3),
            "pivots_high": c.n_pivot_highs,
            "pivots_low": c.n_pivot_lows,
            "breakout_dir": br.breakout_direction if br else "none",
            "breakout_price": round(br.breakout_price, 2) if br else None,
            "bars_to_breakout": br.bars_after_pattern if br else None,
        })
    meta = pd.DataFrame(rows)

    # ---------- PDF: one page per contraction ----------
    pdf_path = os.path.join(OUT_DIR, "all_contractions.pdf")
    print(f"Building PDF: {pdf_path} ...")
    with PdfPages(pdf_path) as pdf:
        # cover page with summary stats
        fig, ax = plt.subplots(figsize=(11, 8.5), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        n_up = (meta["breakout_dir"] == "up").sum()
        n_dn = (meta["breakout_dir"] == "down").sum()
        n_no = (meta["breakout_dir"] == "none").sum()
        cover = (
            f"PRICE CONTRACTION REPORT\n"
            f"{'=' * 50}\n\n"
            f"Instrument:       NIFTY 50 (5-min bars)\n"
            f"Data range:       {df['date'].iloc[0]} → {df['date'].iloc[-1]}\n"
            f"Total bars:       {len(df):,}\n\n"
            f"Detected patterns:    {len(contractions)}\n"
            f"Avg quality score:    {meta['quality'].mean():.3f}\n"
            f"Avg duration:         {meta['duration_bars'].mean():.1f} bars\n"
            f"Avg compression:      {meta['compression'].mean():.3f}\n\n"
            f"Breakouts within 30 bars:\n"
            f"  Upward:   {n_up}  ({n_up/len(meta)*100:.1f}%)\n"
            f"  Downward: {n_dn}  ({n_dn/len(meta)*100:.1f}%)\n"
            f"  None:     {n_no}  ({n_no/len(meta)*100:.1f}%)\n\n"
            f"Detection config:\n"
            f"  Window length:     {CFG.min_len}–{CFG.max_len} bars\n"
            f"  Min R² (lines):    {CFG.min_r2}\n"
            f"  Min containment:   {CFG.min_containment:.0%}\n"
            f"  Compression max:   {CFG.compression_ratio}\n"
            f"  Trend filter:      ≥{CFG.trend_strength}× avg-range over {CFG.trend_lookback} bars\n"
        )
        ax.text(0.05, 0.95, cover, family="monospace", fontsize=11,
                color="#ddd", va="top", ha="left", transform=ax.transAxes)
        pdf.savefig(fig, facecolor="#0d1117")
        plt.close(fig)

        # one page per contraction
        for i, c in enumerate(contractions):
            br = detect_breakout(df, c, CFG, look_forward=30)
            fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0d1117")
            plot_contraction(ax, df, c, br, idx_label=str(i + 1))
            plt.tight_layout()
            pdf.savefig(fig, facecolor="#0d1117")
            plt.close(fig)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(contractions)} pages written")

    print(f"  PDF complete: {pdf_path}")

    # ---------- individual PNGs ----------
    if SAVE_INDIVIDUAL_PNGS:
        print(f"Writing individual PNGs to: {PNG_DIR}/ ...")
        for i, c in enumerate(contractions):
            br = detect_breakout(df, c, CFG, look_forward=30)
            fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0d1117")
            plot_contraction(ax, df, c, br, idx_label=str(i + 1))
            plt.tight_layout()
            png_path = os.path.join(PNG_DIR, f"contraction_{i+1:04d}.png")
            plt.savefig(png_path, dpi=100, facecolor="#0d1117")
            plt.close(fig)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(contractions)} pngs written")

    # ---------- save metadata CSV ----------
    meta_path = os.path.join(OUT_DIR, "contractions_report.csv")
    meta.to_csv(meta_path, index=False)
    print(f"Metadata saved: {meta_path}")

    # ---------- HTML index (browse all charts in browser) ----------
    if SAVE_INDIVIDUAL_PNGS:
        html_path = os.path.join(OUT_DIR, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
            f.write("<title>Contraction Report</title>")
            f.write("<style>")
            f.write("body{background:#0d1117;color:#ddd;font-family:system-ui,sans-serif;margin:20px;}")
            f.write("h1{color:#3da5ff;}")
            f.write(".grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;}")
            f.write(".card{background:#161b22;padding:12px;border:1px solid #30363d;border-radius:6px;}")
            f.write(".card img{width:100%;height:auto;border-radius:4px;}")
            f.write(".meta{font-size:12px;color:#aaa;margin-top:6px;}")
            f.write(".up{color:#00e676;} .down{color:#ff5252;} .none{color:#888;}")
            f.write("</style></head><body>")
            f.write(f"<h1>Contraction Report — {len(contractions)} patterns</h1>")
            f.write(f"<p>Data: {df['date'].iloc[0]} → {df['date'].iloc[-1]}  |  ")
            f.write(f"Up: {n_up} · Down: {n_dn} · None: {n_no}</p>")
            f.write("<div class='grid'>")
            for i, c in enumerate(contractions):
                row = meta.iloc[i]
                bo_class = row["breakout_dir"]
                f.write("<div class='card'>")
                f.write(f"<img src='charts/contraction_{i+1:04d}.png' loading='lazy'>")
                f.write("<div class='meta'>")
                f.write(f"#{row['id']} · {row['start_time']} → {row['end_time']} · ")
                f.write(f"Q={row['quality']} · comp={row['compression']} · ")
                f.write(f"<span class='{bo_class}'>BO: {row['breakout_dir']}</span>")
                f.write("</div></div>")
            f.write("</div></body></html>")
        print(f"HTML index saved: {html_path}")

    print("\nDONE.")


if __name__ == "__main__":
    main()