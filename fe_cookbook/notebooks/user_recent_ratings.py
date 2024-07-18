# Databricks notebook source
# MAGIC %md # LIBRARIES

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

input_table_name=f"{catalog}.{schema}.user_ratings"
output_table_name=f"{catalog}.{schema}.user_recent_ratings"
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

def generate_random_date():
  import random
  random_seconds = random.randint(0, 1000)
  return datetime.now() - timedelta(seconds=random_seconds)

# COMMAND ----------

#FAKE DATA
data_schema  = T.StructType([
   T.StructField("user_id", T.IntegerType(), False),
   T.StructField("time_stamp", T.TimestampType(), False),
   T.StructField("rating", T.IntegerType(), False),
   T.StructField("book_author", T.StringType(), False),
   T.StructField("category", T.StringType(), False)
])

data = [
  (1, generate_random_date(), 5,"Author A", "Category A"),
  (1, generate_random_date(), 3, "Author B", "Category B"),
  (1, generate_random_date(), 4, "Author A", "Category A"),
  (2, generate_random_date(), 2, "Author C", "Category C"),
  (2, generate_random_date(), 3, "Author A", "Category B"),
  (3, generate_random_date(), 5, "Author B", "Category A"),
]

df = spark.createDataFrame(
  data=data,
  schema=data_schema
)

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

def user_recent_ratings():
  #TABLE SCHEMA
  if not spark.catalog.tableExists(output_table_name):
    spark.sql(f"""
              CREATE TABLE {output_table_name} (
                user_id INTEGER NOT NULL,
                time_stamp TIMESTAMP NOT NULL,
                rating_summary ARRAY<STRING>,
                CONSTRAINT user_recent_ratings_pk PRIMARY KEY(user_id, time_stamp TIMESERIES)
              )""")

  #READ DATA
  df = spark.readStream.table(input_table_name)

  #AGGREGATION WINDOW
  window = F.window(
    timeColumn="time_stamp",
    windowDuration="60 minutes",
    slideDuration="30 minutes"
  )


  #TRANFORMATION
  windowed_df = (
    df
    .withWatermark("time_stamp", "10 minutes")
    .groupBy(window, "user_id")
    .agg(window.start.alias("time_stamp"),
         F.slice(F.collect_set(F.to_json(F.struct("rating","book_author","category"))),start=1,length=2).alias("rating_summary"))
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

user_recent_ratings()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM daniel_perez.feature_engineering_tests.user_recent_ratings

# COMMAND ----------

# MAGIC %md # CLEAN UP

# COMMAND ----------

spark.sql(f"DROP SCHEMA {catalog}.{schema} CASCADE")
