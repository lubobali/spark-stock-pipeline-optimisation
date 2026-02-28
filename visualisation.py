# Databricks notebook source
# MAGIC %md
# MAGIC # Stock Signal Pipeline — Visualisations
# MAGIC
# MAGIC Charts generated from the pipeline output data. Run `fixed_stock_signal_pipeline.py`
# MAGIC first to populate the output table, then run this notebook to visualise the results.
# MAGIC
# MAGIC **Charts:**
# MAGIC 1. Candlestick OHLCV with Volume
# MAGIC 2. RSI with Overbought/Oversold Signal Bands
# MAGIC 3. Regime Changes Timeline
# MAGIC 4. Cross-Ticker Correlation Heatmap
# MAGIC 5. Performance Before vs After
# MAGIC 6. Shuffle Reduction (Exchange Nodes)

# COMMAND ----------

# MAGIC %pip install mplfinance

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import mplfinance as mpf
import pandas as pd
import numpy as np
from pyspark.sql import functions as F, Window

# ── Configuration ────────────────────────────────────────────────
# Update these to match your catalog/schema
TABLE_NAME = "catalog.schema.stock_bar_minutes"
OUTPUT_TABLE = "catalog.schema.stock_signals_optimised"

# Style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Pipeline Output Data
# MAGIC
# MAGIC Read from the output table created by `fixed_stock_signal_pipeline.py`.
# MAGIC All charts use this same data — no recomputation needed.

# COMMAND ----------

# Read the optimised pipeline output
signal_df = spark.table(OUTPUT_TABLE)

# Read the raw minute bars for candlestick charts
minute_df = spark.table(TABLE_NAME)

# Get list of tickers sorted by total volume
tickers = [
    row["ticker"]
    for row in (
        signal_df
        .groupBy("ticker")
        .agg(F.sum("day_volume").alias("total_vol"))
        .orderBy(F.desc("total_vol"))
        .collect()
    )
]

print(f"Tickers by volume: {tickers}")
print(f"Total signal rows: {signal_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 1 — Candlestick OHLCV with Volume
# MAGIC
# MAGIC Daily candlestick chart for the top 3 tickers by volume. Shows the actual
# MAGIC market data the pipeline processes — open/high/low/close bars with volume
# MAGIC underneath.

# COMMAND ----------

# Top 3 tickers by volume
top_tickers = tickers[:3]

for ticker in top_tickers:
    ticker_pd = (
        signal_df
        .filter(F.col("ticker") == ticker)
        .select("trade_date", "day_open", "day_high", "day_low", "day_close", "day_volume")
        .orderBy("trade_date")
        .toPandas()
    )

    ticker_pd["trade_date"] = pd.to_datetime(ticker_pd["trade_date"])
    ticker_pd = ticker_pd.set_index("trade_date")
    ticker_pd.columns = ["Open", "High", "Low", "Close", "Volume"]

    # mplfinance must manage its own figure to render inline in Databricks
    fig, axlist = mpf.plot(
        ticker_pd,
        type="candle",
        volume=True,
        title=f"\n{ticker} — Daily OHLCV",
        style="charles",
        figsize=(16, 8),
        warn_too_much_data=500,
        returnfig=True,
    )
    display(fig)
    plt.close(fig)

print(f"Candlestick charts rendered for: {top_tickers}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 2 — RSI with Overbought/Oversold Signal Bands
# MAGIC
# MAGIC RSI (Relative Strength Index) plotted per ticker with the 30/70 threshold
# MAGIC bands. Below 30 = oversold (potential BUY). Above 70 = overbought (potential
# MAGIC SELL). The coloured zones show where the pipeline generates BUY/SELL signals.

# COMMAND ----------

# Plot RSI for top 4 tickers
plot_tickers = tickers[:4]

fig, axes = plt.subplots(len(plot_tickers), 1, figsize=(16, 3.5 * len(plot_tickers)), sharex=True)
if len(plot_tickers) == 1:
    axes = [axes]

for ax, ticker in zip(axes, plot_tickers):
    ticker_pd = (
        signal_df
        .filter(F.col("ticker") == ticker)
        .select("trade_date", "rsi")
        .orderBy("trade_date")
        .toPandas()
    )
    ticker_pd["trade_date"] = pd.to_datetime(ticker_pd["trade_date"])

    # RSI line
    ax.plot(ticker_pd["trade_date"], ticker_pd["rsi"], linewidth=1.2, color="#2196F3", label="RSI")

    # Overbought/oversold bands
    ax.axhline(y=70, color="#E53935", linestyle="--", linewidth=0.8, alpha=0.7, label="Overbought (70)")
    ax.axhline(y=30, color="#43A047", linestyle="--", linewidth=0.8, alpha=0.7, label="Oversold (30)")
    ax.axhline(y=50, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)

    # Fill zones
    ax.fill_between(
        ticker_pd["trade_date"], 70, 100,
        alpha=0.08, color="#E53935", label="_nolegend_"
    )
    ax.fill_between(
        ticker_pd["trade_date"], 0, 30,
        alpha=0.08, color="#43A047", label="_nolegend_"
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI")
    ax.set_title(f"{ticker} — RSI with Signal Bands", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=12)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("/tmp/rsi_signals.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 3 — Regime Changes Timeline
# MAGIC
# MAGIC Shows when each ticker's signal regime changed over time. Each dot is a
# MAGIC trading day, coloured by the signal regime. Larger dots mark regime change
# MAGIC points — the transitions the fixed pipeline detects (1,317 total vs 1,049
# MAGIC in the broken version that missed weekend/holiday gaps).

# COMMAND ----------

# Regime colours and ordering (bearish → bullish)
from matplotlib.colors import ListedColormap, BoundaryNorm

regime_order = ["STRONG_BUY", "BUY", "LEAN_BUY", "NEUTRAL", "LEAN_SELL", "SELL", "STRONG_SELL"]
regime_colors_list = ["#1B5E20", "#43A047", "#81C784", "#9E9E9E", "#EF9A9A", "#E53935", "#B71C1C"]
regime_to_num = {r: i for i, r in enumerate(regime_order)}

regime_pd = (
    signal_df
    .select("ticker", "trade_date", "signal_regime", "regime_changed")
    .orderBy("trade_date", "ticker")
    .toPandas()
)
regime_pd["trade_date"] = pd.to_datetime(regime_pd["trade_date"])

# Build a matrix: rows = tickers, columns = dates, values = regime index
sorted_tickers = sorted(regime_pd["ticker"].unique())
sorted_dates = sorted(regime_pd["trade_date"].unique())
date_to_idx = {d: i for i, d in enumerate(sorted_dates)}
ticker_to_idx = {t: i for i, t in enumerate(sorted_tickers)}

matrix = np.full((len(sorted_tickers), len(sorted_dates)), np.nan)
for _, row in regime_pd.iterrows():
    ti = ticker_to_idx[row["ticker"]]
    di = date_to_idx[row["trade_date"]]
    matrix[ti, di] = regime_to_num.get(row["signal_regime"], 3)  # default NEUTRAL

cmap = ListedColormap(regime_colors_list)
bounds = np.arange(-0.5, len(regime_order), 1)
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(18, 5))
ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

# Mark regime changes with black tick marks
changes = regime_pd[regime_pd["regime_changed"] == True]
for _, row in changes.iterrows():
    ti = ticker_to_idx[row["ticker"]]
    di = date_to_idx[row["trade_date"]]
    ax.plot(di, ti, marker="|", color="black", markersize=12, markeredgewidth=1.5)

# Y-axis: ticker labels
ax.set_yticks(range(len(sorted_tickers)))
ax.set_yticklabels(sorted_tickers, fontsize=10)

# X-axis: monthly date labels
n_labels = min(12, len(sorted_dates))
date_indices = np.linspace(0, len(sorted_dates) - 1, n_labels).astype(int)
ax.set_xticks(date_indices)
ax.set_xticklabels(
    [sorted_dates[i].strftime("%b %Y") for i in date_indices],
    fontsize=9, rotation=30, ha="right",
)

ax.set_title(
    f"Signal Regime Timeline — {len(changes)} regime changes detected  ( | = transition )",
    fontsize=14, fontweight="bold",
)

# Legend below chart
patches = [mpatches.Patch(color=regime_colors_list[i], label=regime_order[i]) for i in range(len(regime_order))]
ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=7, fontsize=9, frameon=True)

fig.subplots_adjust(bottom=0.2)
plt.savefig("/tmp/regime_changes.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 4 — Cross-Ticker Correlation Heatmap
# MAGIC
# MAGIC Average intraday price correlation between all ticker pairs. Computed from
# MAGIC the Stage 4 cross-ticker correlation output. Pairs with correlation > 0.8
# MAGIC are considered strongly correlated — they tend to move together intraday.

# COMMAND ----------

# Rebuild correlation from minute data (same logic as Stage 4)
daily_df = (
    minute_df
    .groupBy("ticker", "trade_date")
    .agg(F.avg("close").alias("avg_close"))
)

# Self-join to get all ticker pairs per day
left = daily_df.select(
    F.col("ticker").alias("ticker_a"),
    F.col("trade_date"),
    F.col("avg_close").alias("close_a"),
)
right = daily_df.select(
    F.col("ticker").alias("ticker_b"),
    F.col("trade_date").alias("td_r"),
    F.col("avg_close").alias("close_b"),
)

pairs = left.join(
    F.broadcast(right),
    (F.col("trade_date") == F.col("td_r")) & (F.col("ticker_a") != F.col("ticker_b")),
).drop("td_r")

# Compute average daily close correlation per ticker pair
corr_pd = (
    pairs
    .groupBy("ticker_a", "ticker_b")
    .agg(F.corr("close_a", "close_b").alias("correlation"))
    .toPandas()
)

# Pivot to matrix
corr_matrix = corr_pd.pivot(index="ticker_a", columns="ticker_b", values="correlation")

# Fill diagonal with 1.0
for t in corr_matrix.index:
    if t in corr_matrix.columns:
        corr_matrix.loc[t, t] = 1.0

# Sort both axes
corr_matrix = corr_matrix.sort_index(axis=0).sort_index(axis=1)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix.astype(float),
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Pearson Correlation"},
    ax=ax,
)
ax.set_title("Cross-Ticker Daily Close Correlation", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()
plt.savefig("/tmp/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 5 — Performance: Before vs After
# MAGIC
# MAGIC Side-by-side comparison of wall-clock time per stage. The broken pipeline
# MAGIC took 183 seconds total. After fixing all 8 anti-patterns, the same pipeline
# MAGIC runs in 47 seconds — a **4x overall speedup**.

# COMMAND ----------

# Actual timings from Databricks Serverless runs
stages = ["Stage 1\nLoad &\nValidate", "Stage 2\nDaily\nOHLCV", "Stage 3\nTechnical\nIndicators",
          "Stage 4\nCross-Ticker\nCorrelation", "Stage 5\nSignal\nReport", "Stage 6\nWrite\nDelta", "Stage 7\nSummary\nStats"]
broken_times = [29.49, 4.0, 47.0, 51.11, 19.0, 31.0, 1.0]
fixed_times = [2.0, 1.0, 5.0, 14.69, 14.0, 9.70, 1.0]
speedups = ["15x", "4x", "9x", "3.5x", "1.4x", "3x", "—"]

x = np.arange(len(stages))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 8))

bars_broken = ax.bar(x - width / 2, broken_times, width, label="Broken Pipeline",
                     color="#E53935", alpha=0.85, edgecolor="white")
bars_fixed = ax.bar(x + width / 2, fixed_times, width, label="Fixed Pipeline",
                    color="#43A047", alpha=0.85, edgecolor="white")

# Add speedup labels on top
for i, (b_bar, f_bar, spd) in enumerate(zip(bars_broken, bars_fixed, speedups)):
    if spd != "—":
        ax.annotate(
            spd,
            xy=(x[i], max(b_bar.get_height(), f_bar.get_height()) + 1.5),
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#1565C0",
        )

# Add time labels on bars
for bar in bars_broken:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8, color="#B71C1C")
for bar in bars_fixed:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8, color="#1B5E20")

ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("Pipeline Performance: Broken vs Fixed — 183s → 47s (4x speedup)",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=9)
ax.legend(fontsize=11, loc="upper right")
ax.set_ylim(0, max(broken_times) * 1.25)

# Total bar at the right
ax2_x = len(stages)
ax.bar(ax2_x - width / 2, sum(broken_times), width, color="#E53935", alpha=0.85, edgecolor="white")
ax.bar(ax2_x + width / 2, sum(fixed_times), width, color="#43A047", alpha=0.85, edgecolor="white")
ax.text(ax2_x - width / 2, sum(broken_times) + 1, f"{sum(broken_times):.0f}s", ha="center", fontsize=9, fontweight="bold", color="#B71C1C")
ax.text(ax2_x + width / 2, sum(fixed_times) + 1, f"{sum(fixed_times):.0f}s", ha="center", fontsize=9, fontweight="bold", color="#1B5E20")
ax.annotate("4x", xy=(ax2_x, sum(broken_times) + 5), ha="center", fontsize=13, fontweight="bold", color="#1565C0")

new_labels = stages + ["TOTAL"]
ax.set_xticks(list(x) + [ax2_x])
ax.set_xticklabels(new_labels, fontsize=11)

fig.subplots_adjust(bottom=0.15, top=0.92)
plt.savefig("/tmp/performance_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 6 — Shuffle Reduction: Exchange Nodes
# MAGIC
# MAGIC Each `Exchange` node in the physical plan is a shuffle — data moving across
# MAGIC the network between executors. The broken pipeline has **8+ shuffles**. The
# MAGIC fixed pipeline has **3**. This is where most of the 4x speedup comes from.

# COMMAND ----------

# Shuffle comparison data from physical plan analysis
shuffle_data = {
    "Stage": ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Stage 6"],
    "Broken (Shuffles)": [0, 2, 2, 2, 3, 1],
    "Fixed (Shuffles)": [1, 1, 1, 0, 1, 0],
    "What Changed": [
        "Added groupBy\n(1 efficient shuffle)",
        "Removed repartition\n(was 2 shuffles → 1)",
        "Broadcast join\n(was sort-merge → 0 shuffle)",
        "Broadcast join\n(was sort-merge → 0 shuffle)",
        "Reuse + LAG window\n(was 3 shuffles → 1)",
        "coalesce replaces\nrepartition (0 shuffle)",
    ]
}

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(shuffle_data["Stage"]))
width = 0.3

bars_broken = ax.bar(x - width / 2, shuffle_data["Broken (Shuffles)"], width,
                     label=f"Broken ({sum(shuffle_data['Broken (Shuffles)'])} total shuffles)",
                     color="#E53935", alpha=0.85, edgecolor="white")
bars_fixed = ax.bar(x + width / 2, shuffle_data["Fixed (Shuffles)"], width,
                    label=f"Fixed ({sum(shuffle_data['Fixed (Shuffles)'])} total shuffles)",
                    color="#43A047", alpha=0.85, edgecolor="white")

# Add annotations
for i, txt in enumerate(shuffle_data["What Changed"]):
    ax.annotate(
        txt,
        xy=(x[i], max(shuffle_data["Broken (Shuffles)"][i], shuffle_data["Fixed (Shuffles)"][i]) + 0.15),
        ha="center", va="bottom", fontsize=9.5,
        color="#37474F", style="italic",
    )

ax.set_ylabel("Number of Exchange (Shuffle) Nodes", fontsize=11)
ax.set_title("Shuffle Reduction per Stage — 10 shuffles → 4 (from Physical Plan Analysis)",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(shuffle_data["Stage"], fontsize=10)
ax.legend(fontsize=11, loc="upper right")
ax.set_ylim(0, 5)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("/tmp/shuffle_reduction.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Charts
# MAGIC
# MAGIC All charts have been saved to `/tmp/`. Download them from DBFS or
# MAGIC screenshot the outputs above to add to the `images/` folder in the repo.
# MAGIC
# MAGIC ```
# MAGIC images/
# MAGIC   candlestick_<TICKER>.png  (one per top ticker)
# MAGIC   rsi_signals.png
# MAGIC   regime_changes.png
# MAGIC   correlation_heatmap.png
# MAGIC   performance_comparison.png
# MAGIC   shuffle_reduction.png
# MAGIC ```

# COMMAND ----------

print("All charts generated.")
print("Files saved to /tmp/:")
print("  /tmp/candlestick_<TICKER>.png  (one per top ticker)")
print("  /tmp/rsi_signals.png")
print("  /tmp/regime_changes.png")
print("  /tmp/correlation_heatmap.png")
print("  /tmp/performance_comparison.png")
print("  /tmp/shuffle_reduction.png")
print()
print("Download these or screenshot the cell outputs above,")
print("then add them to the images/ folder in the repo.")
