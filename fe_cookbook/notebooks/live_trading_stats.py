# Databricks notebook source
# MAGIC %md # LIBRARIES

# COMMAND ----------

# MAGIC %pip install dbldatagen
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import databricks.feature_engineering as fe

from pyspark.sql import functions as F
import pyspark.sql.types as T
from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %md # SET-UP

# COMMAND ----------

fe_client = fe.FeatureEngineeringClient()

catalog = "daniel_perez"
schema = "feature_engineering_tests"

input_table_name=f"{catalog}.{schema}.transactions_stream"
output_table_name=f"{catalog}.{schema}.live_trading_stats"
output_checkpoint_dir=f"/Volumes/{catalog}/{schema}/volume/checkpoints"

# COMMAND ----------

# CREATE CATALOG
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

# CREATE SCHEMA
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# VOLUME
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.volume")

# COMMAND ----------

# MAGIC %md # FAKE DATA

# COMMAND ----------

import dbldatagen as dg
import dbldatagen.distributions as dist

data_rows = 1000000
df_spec = (
    dg.DataGenerator(spark, name="test_data_set1", rows=data_rows, partitions=4)
    .withIdOutput()
    .withColumn("SYMBOL", T.StringType(),
                    template=r"AA|AAA")
    .withColumn("QUANTITY", T.IntegerType(), minValue=1, maxValue=100, random=True, distribution=dist.Gamma(2.0, 1.0))
    .withColumn("PRICE", T.IntegerType(), minValue=1, maxValue=1000, random=True, distribution=dist.Gamma(1.0, 2.0))
    .withColumn("TIME_STAMP",
                T.TimestampType(),
                data_range=dg.DateRange(datetime.now(), datetime.now()+timedelta(days=3), "minutes=1"),
                random=True)
)

df = df_spec.build().withColumn("trade_identification", F.hex(F.hash("id")))


(
  df
  .write
  .mode("append")
  .saveAsTable(input_table_name)
)

# COMMAND ----------

spark.table(input_table_name).display()

# COMMAND ----------

# MAGIC %md # FEATURE PROCESSING

# COMMAND ----------

def live_trading_stats():
  #TABLE SCHEMA
  if not spark.catalog.tableExists(output_table_name):
    spark.sql(f"""
              CREATE TABLE {output_table_name} (
                symbol STRING NOT NULL,
                time_stamp TIMESTAMP NOT NULL,
                min_price INTEGER,
                max_price INTEGER,
                quantity LONG,
                dollar_volume LONG,
                CONSTRAINT live_trading_stats_pk PRIMARY KEY(symbol, time_stamp TIMESERIES)
              )""")

  #READ DATA
  df = spark.readStream.table(input_table_name)

  #AGGREGATION WINDOW
  window = F.window(
    timeColumn="time_stamp",
    windowDuration="60 minutes",
    slideDuration="60 minutes"
  )


  #TRANFORMATION
  windowed_df = (
    df
    #.withWatermark("time_stamp", "10 minutes")
    .groupBy(window, "symbol")
    .agg(window.start.alias("time_stamp"),
         F.min(F.col("PRICE")).alias("min_price"),
         F.max(F.col("PRICE")).alias("max_price"),
         F.sum(F.col("QUANTITY")).alias("quantity"),
         F.sum(F.col("QUANTITY")*F.col("PRICE")).alias("dollar_volume"))
    .drop("window")
  )

  #WRITE
  return (
    windowed_df.writeStream
    .trigger(availableNow=True)
    .outputMode("complete")
    .option("checkpointLocation", output_checkpoint_dir)
    .toTable(output_table_name)
  )


# COMMAND ----------

live_trading_stats()

# COMMAND ----------

spark.table(output_table_name).display()

# COMMAND ----------

# MAGIC %md # CLEAN UP

# COMMAND ----------

spark.sql(f"DROP SCHEMA {catalog}.{schema} CASCADE")
