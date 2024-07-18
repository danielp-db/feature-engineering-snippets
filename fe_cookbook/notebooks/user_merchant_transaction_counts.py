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
output_table_name=f"{catalog}.{schema}.user_merchant_transactions_count"
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
    .withColumn("user_id", T.IntegerType(), minValue=0, maxValue=100, random=True)
    .withColumn("transaction_id", T.IntegerType(), minValue=0, maxValue=10000, random=True)
    .withColumn("category", T.StringType(), values=["a", "b", "c"], random=True)
    .withColumn("amt", T.IntegerType(), minValue=1, maxValue=100, random=True, distribution=dist.Gamma(1.0, 2.0))
    .withColumn("is_fraud", T.StringType(), values=["a", "b", "c"], random=True)
    .withColumn("merchant", T.StringType(), values=["a", "b", "c", "d", "e", "f"], random=True)
    .withColumn("merchant_lat", T.FloatType(), minValue=-90.0, maxValue=90.0, random=True)
    .withColumn("merchant_lon", T.FloatType(), minValue=-180.0, maxValue=180.0, random=True)
    .withColumn(
        "time_stamp",
        T.TimestampType(),
        data_range=dg.DateRange(datetime.now(), datetime.now()+timedelta(days=3), "minutes=1"),
        random=True,
    )
)

df = df_spec.build()

(
  df
  .write
  .mode("append")
  .saveAsTable(input_table_name)
)

# COMMAND ----------

spark.sql(f"SELECT * FROM {input_table_name}").display()

# COMMAND ----------

# MAGIC %md # FEATURE PROCESSING

# COMMAND ----------

def user_merchant_transactions_count():
  #TABLE SCHEMA
  if not spark.catalog.tableExists(output_table_name):
    spark.sql(f"""
              CREATE TABLE {output_table_name} (
                user_id INTEGER NOT NULL,
                merchant STRING NOT NULL,
                transaction_count INTEGER NOT NULL,
                time_stamp TIMESTAMP NOT NULL,
                CONSTRAINT user_merchant_transactions_count_pk PRIMARY KEY(user_id, merchant, time_stamp TIMESERIES)
              )""")

  #READ DATA
  df = spark.readStream.table(input_table_name)

  #AGGREGATION WINDOW
  window = F.window(
    timeColumn="time_stamp",
    windowDuration="30 minutes",
    slideDuration="5 minutes"
  )


  #TRANFORMATION
  windowed_df = (
    df
    .withWatermark("time_stamp", "10 minutes")
    .groupBy(window, "user_id", "merchant")
    .agg(window.start.alias("time_stamp"),
         F.size(F.collect_set("transaction_id")).alias("transaction_count"))
    .drop("window")
  )

  #return windowed_df
  #WRITE
  return (
    windowed_df.writeStream
    .trigger(availableNow=True)
    .outputMode("complete")
    .option("checkpointLocation", output_checkpoint_dir)
    .toTable(output_table_name)
  )


# COMMAND ----------

user_merchant_transactions_count()

# COMMAND ----------

spark.table(output_table_name).display()

# COMMAND ----------

# MAGIC %md # CLEAN UP

# COMMAND ----------

spark.sql(f"DROP SCHEMA {catalog}.{schema} CASCADE")
