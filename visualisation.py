# Databricks notebook source
# MAGIC %md
# MAGIC # Stock Signal Pipeline — Interactive Visualisations
# MAGIC
# MAGIC Interactive charts generated from the pipeline output data using **Plotly**.
# MAGIC Run `fixed_stock_signal_pipeline.py` first to populate the output table,
# MAGIC then run this notebook to visualise the results.
# MAGIC
# MAGIC **Charts:**
# MAGIC 1. Candlestick OHLCV with Volume (zoom, pan, hover)
# MAGIC 2. RSI with Overbought/Oversold Signal Bands
# MAGIC 3. Regime Changes Timeline (interactive heatmap)
# MAGIC 4. Cross-Ticker Correlation Heatmap
# MAGIC 5. Performance Before vs After
# MAGIC 6. Shuffle Reduction (Exchange Nodes)

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pyspark.sql import functions as F, Window

# ── Configuration ────────────────────────────────────────────────
# Update these to match your catalog/schema
TABLE_NAME = "catalog.schema.stock_bar_minutes"
OUTPUT_TABLE = "catalog.schema.stock_signals_optimised"

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
# MAGIC Interactive daily candlestick chart for the top 3 tickers by volume.
# MAGIC Drag to zoom into any date range. Hover for exact OHLCV values.
# MAGIC Double-click to reset the view.

# COMMAND ----------

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

    # Volume bar colours: green = close >= open, red = close < open
    vol_colors = [
        "#43A047" if c >= o else "#E53935"
        for c, o in zip(ticker_pd["day_close"], ticker_pd["day_open"])
    ]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Candlestick(
            x=ticker_pd["trade_date"],
            open=ticker_pd["day_open"],
            high=ticker_pd["day_high"],
            low=ticker_pd["day_low"],
            close=ticker_pd["day_close"],
            increasing_line_color="#43A047",
            decreasing_line_color="#E53935",
            name="OHLC",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=ticker_pd["trade_date"],
            y=ticker_pd["day_volume"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(text=f"{ticker} — Daily OHLCV", font=dict(size=16)),
        height=600,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.show()

print(f"Candlestick charts rendered for: {top_tickers}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 2 — RSI with Overbought/Oversold Signal Bands
# MAGIC
# MAGIC RSI (Relative Strength Index) plotted per ticker with 30/70 threshold bands.
# MAGIC Below 30 = oversold (BUY signal). Above 70 = overbought (SELL signal).
# MAGIC Hover for exact RSI values per trading day.

# COMMAND ----------

plot_tickers = tickers[:4]

fig = make_subplots(
    rows=len(plot_tickers), cols=1, shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=[f"{t} — RSI with Signal Bands" for t in plot_tickers],
)

for i, ticker in enumerate(plot_tickers, 1):
    ticker_pd = (
        signal_df
        .filter(F.col("ticker") == ticker)
        .select("trade_date", "rsi")
        .orderBy("trade_date")
        .toPandas()
    )
    ticker_pd["trade_date"] = pd.to_datetime(ticker_pd["trade_date"])

    # RSI line
    fig.add_trace(
        go.Scatter(
            x=ticker_pd["trade_date"],
            y=ticker_pd["rsi"],
            mode="lines",
            name=f"{ticker} RSI",
            line=dict(color="#2196F3", width=1.5),
            showlegend=False,
        ),
        row=i, col=1,
    )

    # Overbought zone (70-100)
    fig.add_hrect(
        y0=70, y1=100, fillcolor="#E53935", opacity=0.08,
        line_width=0, row=i, col=1,
    )
    # Oversold zone (0-30)
    fig.add_hrect(
        y0=0, y1=30, fillcolor="#43A047", opacity=0.08,
        line_width=0, row=i, col=1,
    )

    # Threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color="#E53935", opacity=0.7, row=i, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#43A047", opacity=0.7, row=i, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="grey", opacity=0.4, row=i, col=1)

    fig.update_yaxes(range=[0, 100], title_text="RSI", row=i, col=1)

fig.update_layout(
    height=300 * len(plot_tickers),
    template="plotly_white",
    margin=dict(t=40, b=40),
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 3 — Regime Changes Timeline
# MAGIC
# MAGIC Interactive heatmap showing each ticker's signal regime over time. Each cell
# MAGIC is one trading day — hover for the exact date, ticker, and signal regime.
# MAGIC The fixed pipeline detects **1,317 regime changes** vs 1,049 in the broken
# MAGIC version that missed weekend/holiday gaps.

# COMMAND ----------

regime_order = ["STRONG_BUY", "BUY", "LEAN_BUY", "NEUTRAL", "LEAN_SELL", "SELL", "STRONG_SELL"]
regime_colors = ["#1B5E20", "#43A047", "#81C784", "#9E9E9E", "#EF9A9A", "#E53935", "#B71C1C"]
regime_to_num = {r: i for i, r in enumerate(regime_order)}

regime_pd = (
    signal_df
    .select("ticker", "trade_date", "signal_regime", "regime_changed")
    .orderBy("trade_date", "ticker")
    .toPandas()
)
regime_pd["trade_date"] = pd.to_datetime(regime_pd["trade_date"])

# Pivot to matrices
regime_pd["regime_num"] = regime_pd["signal_regime"].map(regime_to_num).fillna(3).astype(float)
matrix_df = regime_pd.pivot_table(index="ticker", columns="trade_date", values="regime_num", aggfunc="first")
matrix_df = matrix_df.sort_index()

# Regime name matrix for hover
text_df = regime_pd.pivot_table(index="ticker", columns="trade_date", values="signal_regime", aggfunc="first")
text_df = text_df.reindex(index=matrix_df.index, columns=matrix_df.columns).fillna("")

sorted_tickers = list(matrix_df.index)
sorted_dates = [d.strftime("%Y-%m-%d") for d in matrix_df.columns]

# Discrete colorscale: sharp boundaries between regimes
n = len(regime_order)
colorscale = []
for i in range(n):
    colorscale.append([i / n, regime_colors[i]])
    colorscale.append([(i + 1) / n, regime_colors[i]])

n_changes = int((regime_pd["regime_changed"] == True).sum())

fig = go.Figure(data=go.Heatmap(
    z=matrix_df.values,
    x=sorted_dates,
    y=sorted_tickers,
    customdata=text_df.values,
    hovertemplate="<b>%{y}</b><br>Date: %{x}<br>Regime: %{customdata}<extra></extra>",
    colorscale=colorscale,
    zmin=-0.5,
    zmax=n - 0.5,
    showscale=False,
))

# Add legend entries for each regime
for regime, color in zip(regime_order, regime_colors):
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color=color, symbol="square"),
        name=regime, showlegend=True,
    ))

fig.update_layout(
    title=dict(
        text=f"Signal Regime Timeline — {n_changes} regime changes detected",
        font=dict(size=16),
    ),
    height=450,
    template="plotly_white",
    xaxis=dict(title="Date"),
    yaxis=dict(title=""),
    legend=dict(
        orientation="h", yanchor="top", y=-0.15,
        xanchor="center", x=0.5, font=dict(size=11),
    ),
    margin=dict(t=60, b=100),
)
fig.show()

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

corr_pd = (
    pairs
    .groupBy("ticker_a", "ticker_b")
    .agg(F.corr("close_a", "close_b").alias("correlation"))
    .toPandas()
)

# Pivot to matrix
corr_matrix = corr_pd.pivot(index="ticker_a", columns="ticker_b", values="correlation")
for t in corr_matrix.index:
    if t in corr_matrix.columns:
        corr_matrix.loc[t, t] = 1.0
corr_matrix = corr_matrix.sort_index(axis=0).sort_index(axis=1).astype(float)

# Annotation text
z_text = corr_matrix.round(2).values

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.index),
    text=z_text,
    texttemplate="%{text:.2f}",
    textfont=dict(size=13),
    colorscale="RdYlGn",
    zmid=0,
    zmin=-1,
    zmax=1,
    colorbar=dict(title="Pearson<br>Correlation"),
    hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
))

fig.update_layout(
    title=dict(text="Cross-Ticker Daily Close Correlation", font=dict(size=16)),
    height=600,
    width=700,
    template="plotly_white",
    xaxis=dict(title=""),
    yaxis=dict(title="", autorange="reversed"),
    margin=dict(t=60, b=40),
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 5 — Performance: Before vs After
# MAGIC
# MAGIC Side-by-side comparison of wall-clock time per stage. The broken pipeline
# MAGIC took 183 seconds total. After fixing all 8 anti-patterns, the same pipeline
# MAGIC runs in 47 seconds — a **4x overall speedup**.

# COMMAND ----------

stages = [
    "Stage 1<br>Load & Validate", "Stage 2<br>Daily OHLCV", "Stage 3<br>Technical Indicators",
    "Stage 4<br>Cross-Ticker Correlation", "Stage 5<br>Signal Report",
    "Stage 6<br>Write Delta", "Stage 7<br>Summary Stats", "<b>TOTAL</b>",
]
broken_times = [29.49, 4.0, 47.0, 51.11, 19.0, 31.0, 1.0, 182.6]
fixed_times = [2.0, 1.0, 5.0, 14.69, 14.0, 9.70, 1.0, 47.39]
speedups = ["15x", "4x", "9x", "3.5x", "1.4x", "3x", "", "4x"]

fig = go.Figure()

fig.add_trace(go.Bar(
    name="Broken Pipeline",
    x=stages,
    y=broken_times,
    marker_color="#E53935",
    text=[f"{t:.1f}s" for t in broken_times],
    textposition="outside",
    textfont=dict(size=11, color="#B71C1C"),
))

fig.add_trace(go.Bar(
    name="Fixed Pipeline",
    x=stages,
    y=fixed_times,
    marker_color="#43A047",
    text=[f"{t:.1f}s" for t in fixed_times],
    textposition="outside",
    textfont=dict(size=11, color="#1B5E20"),
))

# Speedup annotations above bars
for i, spd in enumerate(speedups):
    if spd:
        fig.add_annotation(
            x=stages[i],
            y=max(broken_times[i], fixed_times[i]) + 6,
            text=f"<b>{spd}</b>",
            showarrow=False,
            font=dict(size=14, color="#1565C0"),
        )

fig.update_layout(
    title=dict(
        text="Pipeline Performance: Broken vs Fixed — 183s → 47s (4x speedup)",
        font=dict(size=16),
    ),
    barmode="group",
    height=550,
    template="plotly_white",
    yaxis=dict(title="Time (seconds)"),
    legend=dict(font=dict(size=13)),
    margin=dict(t=60, b=80),
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart 6 — Shuffle Reduction: Exchange Nodes
# MAGIC
# MAGIC Each `Exchange` node in the physical plan is a shuffle — data moving across
# MAGIC the network between executors. The broken pipeline has **10 shuffles**. The
# MAGIC fixed pipeline has **4**. This is where most of the 4x speedup comes from.

# COMMAND ----------

shuffle_stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Stage 6"]
broken_shuffles = [0, 2, 2, 2, 3, 1]
fixed_shuffles = [1, 1, 1, 0, 1, 0]
what_changed = [
    "Added groupBy<br>(1 efficient shuffle)",
    "Removed repartition<br>(was 2 → 1)",
    "Broadcast join<br>(sort-merge → 0)",
    "Broadcast join<br>(sort-merge → 0)",
    "Reuse + LAG window<br>(was 3 → 1)",
    "coalesce replaces<br>repartition (0 shuffle)",
]

fig = go.Figure()

fig.add_trace(go.Bar(
    name=f"Broken ({sum(broken_shuffles)} total shuffles)",
    x=shuffle_stages,
    y=broken_shuffles,
    marker_color="#E53935",
))

fig.add_trace(go.Bar(
    name=f"Fixed ({sum(fixed_shuffles)} total shuffles)",
    x=shuffle_stages,
    y=fixed_shuffles,
    marker_color="#43A047",
))

# Annotations above bars
for i, txt in enumerate(what_changed):
    fig.add_annotation(
        x=shuffle_stages[i],
        y=max(broken_shuffles[i], fixed_shuffles[i]) + 0.3,
        text=f"<i>{txt}</i>",
        showarrow=False,
        font=dict(size=10, color="#37474F"),
    )

fig.update_layout(
    title=dict(
        text="Shuffle Reduction per Stage — 10 shuffles → 4 (from Physical Plan Analysis)",
        font=dict(size=16),
    ),
    barmode="group",
    height=500,
    template="plotly_white",
    yaxis=dict(title="Number of Exchange (Shuffle) Nodes", dtick=1),
    legend=dict(font=dict(size=13)),
    margin=dict(t=60, b=40),
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC All charts are interactive — hover for values, drag to zoom, double-click to
# MAGIC reset. Screenshot the outputs above to add to the `images/` folder in the repo.

# COMMAND ----------

print("All interactive charts generated.")
print()
print("Screenshot the cell outputs above,")
print("then add them to the images/ folder in the repo.")
