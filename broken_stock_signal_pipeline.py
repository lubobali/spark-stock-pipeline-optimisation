# Databricks notebook source
# MAGIC %md
# MAGIC # Broken Stock Signal Pipeline (Original — Before Optimisation)
# MAGIC
# MAGIC This pipeline reads minute-level stock bar data and builds a daily trading
# MAGIC signal report: technical indicators, cross-ticker correlation pairs, and
# MAGIC buy/sell signals. It produces **correct results** — but the performance is
# MAGIC catastrophic. On a production-sized dataset this would take hours instead
# MAGIC of minutes (or OOM your cluster entirely).
# MAGIC
# MAGIC **8 anti-patterns** are embedded across the 7 stages. See
# MAGIC `fixed_stock_signal_pipeline.py` for the optimised version and
# MAGIC `analysis.md` for the full diagnostic writeup.

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import udf, collect_list, pandas_udf
from pyspark.sql.types import (
    StringType, DoubleType, StructType, StructField,
    ArrayType, IntegerType
)
import time

TABLE_NAME = "catalog.schema.stock_bar_minutes"

# ── Global config ──────────────────────────────────────────────
spark.conf.set("spark.sql.shuffle.partitions", "500")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.adaptive.enabled", "false")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1 — Load & Validate
# MAGIC
# MAGIC Read the stock minute bars and run a quick sanity check.

# COMMAND ----------

start = time.time()

minute_df = spark.table(TABLE_NAME)

total_rows = minute_df.count()
print(f"Loaded {total_rows:,} rows")
minute_df.printSchema()

tickers = [row["ticker"] for row in minute_df.select("ticker").distinct().collect()]
print(f"Tickers ({len(tickers)}): {tickers}")

all_data = minute_df.select("ticker", "trade_date", "close").collect()
print(f"Collected {len(all_data):,} rows to driver for validation")
for ticker in tickers:
    ticker_rows = [r for r in all_data if r["ticker"] == ticker]
    dates = set(r["trade_date"] for r in ticker_rows)
    print(f"  {ticker}: {len(ticker_rows):,} bars across {len(dates)} days")

print(f"⏱  Stage 1: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2 — Daily OHLCV Aggregation
# MAGIC
# MAGIC Roll up minute bars into daily summaries. We need these for the signal
# MAGIC calculations downstream.

# COMMAND ----------

start = time.time()

minute_df_fresh = spark.table(TABLE_NAME)

daily_df = (
    minute_df_fresh
    .repartition(500, "ticker")
    .groupBy("ticker", "trade_date")
    .agg(
        F.first("open").alias("day_open"),
        F.last("close").alias("day_close"),
        F.max("high").alias("day_high"),
        F.min("low").alias("day_low"),
        F.sum("volume").alias("day_volume"),
        F.count("*").alias("bar_count"),
        collect_list("close").alias("all_closes"),
    )
)

daily_count = daily_df.count()
print(f"Daily rows: {daily_count:,}")
daily_df.show(5, truncate=False)

print(f"⏱  Stage 2: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3 — Technical Indicators via UDF
# MAGIC
# MAGIC Calculate RSI (Relative Strength Index) and a custom volatility score
# MAGIC for each ticker-day. RSI is a momentum oscillator that measures the speed
# MAGIC and magnitude of price changes on a 0-100 scale.

# COMMAND ----------

start = time.time()

@udf(returnType=DoubleType())
def compute_rsi(closes_array, period=14):
    """Calculate RSI from an array of intraday close prices."""
    if closes_array is None or len(closes_array) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(closes_array)):
        diff = closes_array[i] - closes_array[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(diff))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


@udf(returnType=StringType())
def classify_signal_regime(rsi_value, volume, avg_volume):
    """Classify the trading regime based on RSI and relative volume."""
    if rsi_value is None or volume is None or avg_volume is None:
        return "UNKNOWN"
    if avg_volume == 0:
        return "UNKNOWN"
    vol_ratio = volume / avg_volume
    if rsi_value > 70 and vol_ratio > 1.5:
        return "STRONG_SELL"
    elif rsi_value > 70:
        return "SELL"
    elif rsi_value < 30 and vol_ratio > 1.5:
        return "STRONG_BUY"
    elif rsi_value < 30:
        return "BUY"
    elif 45 <= rsi_value <= 55:
        return "NEUTRAL"
    elif rsi_value > 55:
        return "LEAN_SELL"
    else:
        return "LEAN_BUY"


avg_volume_df = daily_df.groupBy("ticker").agg(
    F.avg("day_volume").alias("avg_volume")
)

daily_with_avg = daily_df.join(avg_volume_df, on="ticker", how="left")

daily_indicators = (
    daily_with_avg
    .withColumn("rsi", compute_rsi(F.col("all_closes")))
    .withColumn(
        "signal_regime",
        classify_signal_regime(
            F.col("rsi"), F.col("day_volume"), F.col("avg_volume")
        ),
    )
)

daily_indicators.select(
    "ticker", "trade_date", "day_close", "rsi", "signal_regime"
).show(20, truncate=False)

indicator_count = daily_indicators.count()
print(f"Rows with indicators: {indicator_count:,}")

print(f"⏱  Stage 3: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4 — Cross-Ticker Correlation Pairs
# MAGIC
# MAGIC For every pair of tickers on the same trading day, compute the price
# MAGIC return similarity. This helps identify which stocks move together.

# COMMAND ----------

start = time.time()

left = daily_indicators.select(
    F.col("ticker").alias("ticker_a"),
    F.col("trade_date"),
    F.col("day_open").alias("open_a"),
    F.col("day_close").alias("close_a"),
    F.col("day_volume").alias("volume_a"),
    F.col("rsi").alias("rsi_a"),
    F.col("all_closes").alias("closes_a"),
).alias("a")

right = daily_indicators.select(
    F.col("ticker").alias("ticker_b"),
    F.col("trade_date").alias("trade_date_r"),
    F.col("day_open").alias("open_b"),
    F.col("day_close").alias("close_b"),
    F.col("day_volume").alias("volume_b"),
    F.col("rsi").alias("rsi_b"),
    F.col("all_closes").alias("closes_b"),
).alias("b")

ticker_pairs = left.join(
    right,
    (F.col("a.trade_date") == F.col("b.trade_date_r")) &
    (F.col("a.ticker_a") < F.col("b.ticker_b"))
)


@udf(returnType=DoubleType())
def intraday_correlation(closes_a, closes_b):
    """Compute Pearson correlation between two arrays of intraday closes."""
    if closes_a is None or closes_b is None:
        return None
    n = min(len(closes_a), len(closes_b))
    if n < 10:
        return None
    a = closes_a[:n]
    b = closes_b[:n]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
    std_a = (sum((x - mean_a) ** 2 for x in a) / n) ** 0.5
    std_b = (sum((x - mean_b) ** 2 for x in b) / n) ** 0.5
    if std_a == 0 or std_b == 0:
        return None
    return cov / (std_a * std_b)


correlated = ticker_pairs.withColumn(
    "intraday_corr",
    intraday_correlation(F.col("closes_a"), F.col("closes_b")),
).withColumn(
    "return_a", (F.col("close_a") - F.col("open_a")) / F.col("open_a"),
).withColumn(
    "return_b", (F.col("close_b") - F.col("open_b")) / F.col("open_b"),
).withColumn(
    "return_diff", F.abs(F.col("return_a") - F.col("return_b")),
)

strong_pairs = correlated.filter(F.col("intraday_corr") > 0.8)

pair_summary = strong_pairs.groupBy("ticker_a", "ticker_b").agg(
    F.count("*").alias("correlated_days"),
    F.avg("intraday_corr").alias("avg_correlation"),
    F.avg("return_diff").alias("avg_return_diff"),
)

pair_summary.orderBy(F.desc("correlated_days")).show(20, truncate=False)
print(f"Strong correlation pairs: {pair_summary.count():,}")

print(f"⏱  Stage 4: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 5 — Daily Signal Report
# MAGIC
# MAGIC Combine indicators with the previous day's signal (using a lag join)
# MAGIC to detect regime changes, then write the final report.

# COMMAND ----------

start = time.time()

minute_df_again = spark.table(TABLE_NAME)
daily_fresh = (
    minute_df_again
    .groupBy("ticker", "trade_date")
    .agg(
        F.first("open").alias("day_open"),
        F.last("close").alias("day_close"),
        F.max("high").alias("day_high"),
        F.min("low").alias("day_low"),
        F.sum("volume").alias("day_volume"),
        F.count("*").alias("bar_count"),
        collect_list("close").alias("all_closes"),
    )
)

avg_vol_fresh = daily_fresh.groupBy("ticker").agg(
    F.avg("day_volume").alias("avg_volume")
)
daily_ind_fresh = (
    daily_fresh
    .join(avg_vol_fresh, on="ticker", how="left")
    .withColumn("rsi", compute_rsi(F.col("all_closes")))
    .withColumn(
        "signal_regime",
        classify_signal_regime(
            F.col("rsi"), F.col("day_volume"), F.col("avg_volume")
        ),
    )
)

prev_day = daily_ind_fresh.select(
    F.col("ticker"),
    F.date_add(F.col("trade_date"), 1).alias("trade_date"),
    F.col("signal_regime").alias("prev_regime"),
    F.col("rsi").alias("prev_rsi"),
    F.col("day_close").alias("prev_close"),
)

signal_report = daily_ind_fresh.join(
    prev_day,
    on=["ticker", "trade_date"],
    how="left",
)

signal_report = signal_report.withColumn(
    "regime_changed",
    F.when(
        F.col("signal_regime") != F.col("prev_regime"), True
    ).otherwise(False),
).withColumn(
    "day_return_pct",
    F.round(
        (F.col("day_close") - F.col("prev_close")) / F.col("prev_close") * 100,
        2,
    ),
)

regime_changes = signal_report.filter(F.col("regime_changed") == True)
print(f"Regime changes detected: {regime_changes.count():,}")
regime_changes.select(
    "ticker", "trade_date", "prev_regime", "signal_regime",
    "rsi", "day_return_pct",
).orderBy("trade_date", "ticker").show(30, truncate=False)

print(f"⏱  Stage 5: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 6 — Write Final Output

# COMMAND ----------

start = time.time()

OUTPUT_PATH = "/tmp/stock_signals/output"

final = signal_report.drop("all_closes")

(
    final
    .repartition(500)
    .write
    .mode("overwrite")
    .partitionBy("ticker", "signal_regime")
    .format("delta")
    .save(OUTPUT_PATH)
)

written = spark.read.format("delta").load(OUTPUT_PATH)
print(f"Wrote {written.count():,} rows to {OUTPUT_PATH}")
print(f"Partitions on disk: {written.select(F.input_file_name()).distinct().count()} files")

print(f"⏱  Stage 6: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 7 — Summary Stats (Collected to Driver)

# COMMAND ----------

start = time.time()

full_report = spark.read.format("delta").load(OUTPUT_PATH)

summary_rows = (
    full_report
    .groupBy("ticker", "signal_regime")
    .agg(
        F.count("*").alias("days"),
        F.avg("rsi").alias("avg_rsi"),
        F.avg("day_return_pct").alias("avg_return"),
        F.sum("day_volume").alias("total_volume"),
    )
    .collect()
)

print(f"\n{'Ticker':<8} {'Regime':<14} {'Days':>6} {'Avg RSI':>9} {'Avg Ret%':>10} {'Tot Volume':>14}")
print("-" * 65)
for row in sorted(summary_rows, key=lambda r: (r["ticker"], r["signal_regime"])):
    print(
        f"{row['ticker']:<8} {row['signal_regime']:<14} "
        f"{row['days']:>6} {row['avg_rsi']:>9.2f} "
        f"{(row['avg_return'] or 0):>10.2f} {row['total_volume']:>14,.0f}"
    )

print(f"\n⏱  Stage 7: {time.time() - start:.2f}s")

# COMMAND ----------
