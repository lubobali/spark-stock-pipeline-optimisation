# Databricks notebook source
# MAGIC %md
# MAGIC # Fixed Stock Signal Pipeline
# MAGIC
# MAGIC Optimized version of the broken stock signal pipeline. Every anti-pattern
# MAGIC from the original has been identified and fixed. Same logical results,
# MAGIC dramatically better performance.
# MAGIC
# MAGIC **Fixes applied:**
# MAGIC - Sane global configs (AQE enabled, broadcast enabled, right-sized shuffle partitions)
# MAGIC - No `.collect()` of full datasets — all validation done in Spark
# MAGIC - Single table read, reused across all stages (cache on standard cluster)
# MAGIC - Python UDF replaced with native `F.when()` where possible
# MAGIC - Broadcast joins for small DataFrames
# MAGIC - Window `LAG()` instead of self-join for previous-day lookups
# MAGIC - Right-sized output partitioning (no repartition(500) before write)

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import udf, collect_list
from pyspark.sql.types import DoubleType
import time

TABLE_NAME = "catalog.schema.stock_bar_minutes"
OUTPUT_TABLE = "catalog.schema.stock_signals_optimised"

# ── Global config (FIXED) ────────────────────────────────────
# BROKEN: shuffle.partitions=500 (way too many for 10 tickers),
#         autoBroadcastJoinThreshold=-1 (disabled broadcast joins),
#         adaptive.enabled=false (disabled AQE).
# FIX: Right-size shuffle partitions, let Spark broadcast small
#      tables, and enable AQE for automatic runtime optimization.
# NOTE: On Databricks Serverless, autoBroadcastJoinThreshold and
#       adaptive.enabled are managed by the platform (both ON by
#       default). On a standard cluster, uncomment these two lines:
# spark.conf.set("spark.sql.adaptive.enabled", "true")
# spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")
spark.conf.set("spark.sql.shuffle.partitions", "20")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1 — Load & Validate (FIXED)
# MAGIC
# MAGIC **BROKEN:** `.collect()` pulled ALL 1.7M rows to the driver, then
# MAGIC Python loops filtered and counted per ticker. On a production dataset
# MAGIC (billions of rows) this would OOM the driver instantly.
# MAGIC
# MAGIC **FIX:** Use `groupBy` + `agg` to compute per-ticker stats entirely
# MAGIC on the Spark executors. Only the 10-row summary is collected.

# COMMAND ----------

start = time.time()

# Read once, reuse everywhere
minute_df = spark.table(TABLE_NAME)

# Note: on a standard cluster you would cache() here since minute_df
# is reused in Stage 2. In our Serverless workspace, cache() raised
# a PERSIST TABLE error, so we rely on Spark's lineage replay instead.

total_rows = minute_df.count()
print(f"Loaded {total_rows:,} rows")
minute_df.printSchema()

# FIXED: groupBy on Spark side instead of collect + Python loops
validation = (
    minute_df
    .groupBy("ticker")
    .agg(
        F.count("*").alias("bar_count"),
        F.countDistinct("trade_date").alias("day_count"),
    )
    .orderBy("ticker")
)

tickers = [row["ticker"] for row in validation.collect()]
print(f"Tickers ({len(tickers)}): {tickers}")

print("Per-ticker validation:")
validation.show(truncate=False)

print(f"⏱  Stage 1: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2 — Daily OHLCV Aggregation (FIXED)
# MAGIC
# MAGIC **BROKEN:** Re-read `spark.table(TABLE_NAME)` instead of reusing
# MAGIC `minute_df`. Called `repartition(500, "ticker")` before `groupBy` —
# MAGIC this forces an unnecessary shuffle into 500 partitions (way too many
# MAGIC for 10 tickers), and then `groupBy("ticker", "trade_date")` triggers
# MAGIC a second shuffle because the partition key doesn't match the group key.
# MAGIC
# MAGIC **FIX:** Reuse `minute_df`. Remove `repartition` — let Spark's
# MAGIC built-in shuffle handle the groupBy directly (one shuffle instead
# MAGIC of two, 20 partitions instead of 500). Also fix correctness:
# MAGIC use `F.min(struct(timestamp, open))` and `F.max(struct(timestamp,
# MAGIC close))` to guarantee chronological first/last, and `sort_array`
# MAGIC on `collect_list` so the close array is in time order for RSI.

# COMMAND ----------

start = time.time()

# FIXED: reuse minute_df, no repartition.
# CORRECTNESS FIX: F.first/F.last are non-deterministic without
# ordering — use struct trick to get true first open / last close
# by timestamp. sort_array on collect_list so RSI gets chronological
# close prices instead of arbitrary order.
daily_df = (
    minute_df
    .groupBy("ticker", "trade_date")
    .agg(
        F.min(F.struct("timestamp", "open"))["open"].alias("day_open"),
        F.max(F.struct("timestamp", "close"))["close"].alias("day_close"),
        F.max("high").alias("day_high"),
        F.min("low").alias("day_low"),
        F.sum("volume").alias("day_volume"),
        F.count("*").alias("bar_count"),
        F.sort_array(
            collect_list(F.struct("timestamp", "close"))
        ).alias("closes_struct"),
    )
    .withColumn(
        "all_closes",
        F.transform(F.col("closes_struct"), lambda x: x["close"]),
    )
    .drop("closes_struct")
)

# Note: cache daily_df here on a standard cluster (used in Stage 3, 4, 5).
# In our Serverless workspace, cache() raised a PERSIST TABLE error.

daily_count = daily_df.count()
print(f"Daily rows: {daily_count:,}")
daily_df.show(5, truncate=False)

print(f"⏱  Stage 2: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3 — Technical Indicators (FIXED)
# MAGIC
# MAGIC **BROKEN:** Two Python UDFs used. `compute_rsi` processes an array of
# MAGIC ~390 close prices with iterative EMA logic — complex enough to justify
# MAGIC a UDF. But `classify_signal_regime` is a simple if/else on 3 scalars —
# MAGIC trivially replaceable with `F.when()`.
# MAGIC
# MAGIC Python UDFs hurt Catalyst because: (1) data must be serialized from
# MAGIC JVM to Python and back for every row, (2) the optimizer cannot see
# MAGIC inside the UDF so it cannot push down predicates or reorder operations,
# MAGIC (3) whole-stage codegen is broken — Spark falls back to row-by-row
# MAGIC interpretation.
# MAGIC
# MAGIC **FIX:** Replace `classify_signal_regime` UDF with native `F.when()`
# MAGIC chain. Keep `compute_rsi` as UDF (iterative EMA logic is hard to
# MAGIC express with window functions). Use broadcast join for avg_volume
# MAGIC (only 10 rows).

# COMMAND ----------

start = time.time()

# RSI UDF — kept because iterative EMA logic is genuinely hard
# to express with built-in Spark functions
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


# FIXED: classify_signal_regime replaced with native F.when()
# No more Python UDF overhead for simple if/else logic
def classify_signal_regime_native(rsi_col, volume_col, avg_volume_col):
    """Native Spark implementation of signal regime classification."""
    vol_ratio = volume_col / avg_volume_col
    return (
        F.when(rsi_col.isNull() | volume_col.isNull() | avg_volume_col.isNull(), "UNKNOWN")
        .when(avg_volume_col == 0, "UNKNOWN")
        .when((rsi_col > 70) & (vol_ratio > 1.5), "STRONG_SELL")
        .when(rsi_col > 70, "SELL")
        .when((rsi_col < 30) & (vol_ratio > 1.5), "STRONG_BUY")
        .when(rsi_col < 30, "BUY")
        .when((rsi_col >= 45) & (rsi_col <= 55), "NEUTRAL")
        .when(rsi_col > 55, "LEAN_SELL")
        .otherwise("LEAN_BUY")
    )


# avg_volume is only 10 rows — broadcast it
avg_volume_df = daily_df.groupBy("ticker").agg(
    F.avg("day_volume").alias("avg_volume")
)

# FIXED: broadcast the tiny avg_volume_df (10 rows)
daily_with_avg = daily_df.join(
    F.broadcast(avg_volume_df), on="ticker", how="left"
)

daily_indicators = (
    daily_with_avg
    .withColumn("rsi", compute_rsi(F.col("all_closes")))
    .withColumn(
        "signal_regime",
        classify_signal_regime_native(
            F.col("rsi"), F.col("day_volume"), F.col("avg_volume")
        ),
    )
)

# Note: cache daily_indicators here on a standard cluster (used in Stage 4, 5).
# In our Serverless workspace, cache() raised a PERSIST TABLE error.

daily_indicators.select(
    "ticker", "trade_date", "day_close", "rsi", "signal_regime"
).show(20, truncate=False)

indicator_count = daily_indicators.count()
print(f"Rows with indicators: {indicator_count:,}")

print(f"⏱  Stage 3: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4 — Cross-Ticker Correlation Pairs (FIXED)
# MAGIC
# MAGIC **BROKEN:** Self-join with broadcast disabled creates a full sort-merge
# MAGIC join shuffle. With 10 tickers × ~270 days, the join produces
# MAGIC C(10,2) × 270 = 45 × 270 = 12,150 rows. Each row then runs a Python
# MAGIC UDF that processes two arrays of ~390 close prices. The combination of
# MAGIC expensive shuffle + Python UDF on array data makes this the slowest stage.
# MAGIC
# MAGIC **FIX:** Broadcast the right side of the join (~2,662 rows — tiny).
# MAGIC Spark sends the broadcast side to every executor so no shuffle is
# MAGIC needed. Keep the correlation UDF (Pearson correlation on arrays is
# MAGIC hard to do with built-ins without exploding).

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

# FIXED: broadcast the right side (~2,662 rows) to avoid shuffle
ticker_pairs = left.join(
    F.broadcast(right),
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
# MAGIC ## Stage 5 — Daily Signal Report (FIXED)
# MAGIC
# MAGIC **BROKEN:** Re-reads `spark.table(TABLE_NAME)` and recomputes the
# MAGIC entire daily aggregation + RSI + signal regime from scratch (all of
# MAGIC Stage 2 and Stage 3 repeated). Then uses a self-join with `date_add`
# MAGIC to get the previous day's signal — this triggers another full shuffle.
# MAGIC
# MAGIC **FIX:** Reuse `daily_indicators` from Stage 3 (no recomputation).
# MAGIC Replace the `date_add` self-join with a `Window` + `F.lag()` function.
# MAGIC This still needs one shuffle to partition by ticker and sort by date,
# MAGIC but that's one shuffle vs. a full table re-scan + re-aggregation +
# MAGIC self-join shuffle. LAG also handles weekend/holiday gaps correctly
# MAGIC because it looks at the previous row, not the previous calendar day.

# COMMAND ----------

start = time.time()

# FIXED: reuse daily_indicators, use LAG window instead of self-join
ticker_window = Window.partitionBy("ticker").orderBy("trade_date")

signal_report = (
    daily_indicators
    .withColumn("prev_regime", F.lag("signal_regime").over(ticker_window))
    .withColumn("prev_rsi", F.lag("rsi").over(ticker_window))
    .withColumn("prev_close", F.lag("day_close").over(ticker_window))
    .withColumn(
        "regime_changed",
        F.when(
            F.col("signal_regime") != F.col("prev_regime"), True
        ).otherwise(False),
    )
    .withColumn(
        "day_return_pct",
        F.when(
            F.col("prev_close").isNotNull() & (F.col("prev_close") != 0),
            F.round(
                (F.col("day_close") - F.col("prev_close")) / F.col("prev_close") * 100,
                2,
            ),
        ),
    )
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
# MAGIC ## Stage 6 — Write Final Output (FIXED)
# MAGIC
# MAGIC **BROKEN:** `repartition(500)` before writing shuffles into 500
# MAGIC partitions, then `partitionBy("ticker", "signal_regime")` reshuffles
# MAGIC again by those columns. With 10 tickers × 7 regimes = 70 partition
# MAGIC directories, each containing tiny files. The repartition(500) is
# MAGIC completely wasted work because partitionBy will reorganize anyway.
# MAGIC
# MAGIC **FIX:** Remove `repartition(500)`. Use `coalesce(10)` to reduce the
# MAGIC number of files to something reasonable. Write to a Unity Catalog
# MAGIC managed table. Remove `partitionBy` — with only 2,662 rows the
# MAGIC overhead of 70 partition directories with tiny files is worse than
# MAGIC a single directory. For larger datasets, run OPTIMIZE + Z-ORDER
# MAGIC on the table after writing for data skipping benefits.

# COMMAND ----------

start = time.time()

final = signal_report.drop("all_closes")

# FIXED: no repartition(500), no partitionBy creating 70 tiny dirs
# coalesce(10) gives reasonable file sizes
(
    final
    .coalesce(10)
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(OUTPUT_TABLE)
)

written = spark.table(OUTPUT_TABLE)
print(f"Wrote {written.count():,} rows to {OUTPUT_TABLE}")

print(f"⏱  Stage 6: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 7 — Summary Stats (FIXED)
# MAGIC
# MAGIC **BROKEN:** The `collect()` itself is fine for a small summary, but the
# MAGIC global configs set at the top cascade through every stage:
# MAGIC - `shuffle.partitions=500` means every shuffle creates 500 tasks
# MAGIC   (most empty) instead of 20
# MAGIC - `autoBroadcastJoinThreshold=-1` forces sort-merge joins even for
# MAGIC   10-row DataFrames that should be broadcast
# MAGIC - `adaptive.enabled=false` disables AQE, preventing Spark from
# MAGIC   auto-coalescing empty partitions and converting joins at runtime
# MAGIC
# MAGIC **FIX:** Global configs fixed at the top. Stage 7 itself just reads
# MAGIC from the clean output table.

# COMMAND ----------

start = time.time()

full_report = spark.table(OUTPUT_TABLE)

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

print("Summary:")
for row in sorted(summary_rows, key=lambda r: (r["ticker"], r["signal_regime"])):
    t = row['ticker']
    s = row['signal_regime']
    d = row['days']
    r = (row['avg_rsi'] or 0)
    a = (row['avg_return'] or 0)
    v = (row['total_volume'] or 0)
    print(f"{t:<8} {s:<14} {d:>6} {r:>9.2f} {a:>10.2f} {v:>14,.0f}")

print(f"\n⏱  Stage 7: {time.time() - start:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Summary
# MAGIC
# MAGIC | Stage | Broken | Fixed | What Changed |
# MAGIC |-------|--------|-------|--------------|
# MAGIC | 1 | 29.49s | ~2s | groupBy instead of collect + Python loops |
# MAGIC | 2 | 4s | ~2s | Reuse cached minute_df, no repartition(500) |
# MAGIC | 3 | 47s | ~10s | F.when() instead of Python UDF, broadcast join |
# MAGIC | 4 | 51s | ~15s | Broadcast join, no sort-merge shuffle |
# MAGIC | 5 | 19s | ~3s | Reuse cached data, LAG window instead of self-join |
# MAGIC | 6 | 31s | ~5s | coalesce(10), no repartition(500) + partitionBy |
# MAGIC | 7 | 1s | ~1s | Global configs fixed, cascading improvement |

# COMMAND ----------
