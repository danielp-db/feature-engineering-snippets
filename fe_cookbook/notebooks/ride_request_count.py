# Databricks notebook source
# MAGIC %md # LIBRARIES

# COMMAND ----------

import databricks.feature_engineering as fe

from pyspark.sql import functions as F
import pyspark.sql.types as T
from datetime import datetime, timedelta

import mlflow
from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

# MAGIC %md # SET-UP

# COMMAND ----------

fe_client = fe.FeatureEngineeringClient()

catalog = "daniel_perez"
schema = "feature_engineering_tests"

input_table_name=f"{catalog}.{schema}.stock_transactions_event_source"
output_table_name=f"{catalog}.{schema}.live_trading_stats"
output_checkpoint_dir=f"/Volumes/{catalog}/{schema}/volume/checkpoints"

lookup_key=["request_id"]
timestamp_lookup_key= ["time_stamp"]
lookback_window=timedelta(minutes=60)

mlflow.set_registry_uri("databricks-uc")

label="request_accepted"
model_name = f"{catalog}.{schema}.model_accept_request"

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

#FAKE DATA
data_schema  = T.StructType([
   T.StructField("request_id", T.IntegerType(), False),
   T.StructField("time_stamp", T.TimestampType(), False),
   T.StructField("origin_zipcode", T.IntegerType(), False)
])

data = [
  (1, datetime.strptime("2016-03-11 09:07:00", "%Y-%m-%d %H:%M:%S"), 45389),
  (1, datetime.strptime("2016-03-11 09:08:00", "%Y-%m-%d %H:%M:%S"), 45389),
  (1, datetime.strptime("2016-03-11 10:35:00", "%Y-%m-%d %H:%M:%S"), 45341),

  (2, datetime.strptime("2016-03-11 09:01:00", "%Y-%m-%d %H:%M:%S"), 31311),
  (2, datetime.strptime("2016-03-11 10:50:00", "%Y-%m-%d %H:%M:%S"), 36309),

  (3, datetime.strptime("2016-03-11 10:00:41", "%Y-%m-%d %H:%M:%S"), 12454)
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

# MAGIC %md # FEATURE PROCESSING

# COMMAND ----------

def ride_request_count():
  #TABLE SCHEMA
  if not spark.catalog.tableExists(output_table_name):
    spark.sql(f"""
              CREATE TABLE {output_table_name} (
                request_id INTEGER NOT NULL,
                time_stamp TIMESTAMP NOT NULL,
                last_hour_ride_request_count INTEGER NOT NULL,
                CONSTRAINT hourly_ride_request_pk PRIMARY KEY(request_id, time_stamp TIMESERIES)
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
    .groupBy(window, "request_id")
    .agg(window.start.alias("time_stamp"), F.count("*").alias("last_hour_ride_request_count").astype(T.IntegerType()))
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

ride_request_count()

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE EXTENDED daniel_perez.feature_engineering_tests.hourly_ride_requests

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from daniel_perez.feature_engineering_tests.hourly_ride_requests

# COMMAND ----------

# MAGIC %md # FEATURE CONSUMPTION

# COMMAND ----------

# MAGIC %md ## TRAINING MODEL

# COMMAND ----------

# MAGIC %md ### BUILD TRAINING DATASET

# COMMAND ----------

#FAKE DATA
data_schema  = T.StructType([
   T.StructField("request_id", T.IntegerType(), False),
   T.StructField("time_stamp", T.TimestampType(), False),
   T.StructField("request_accepted", T.BooleanType(), False)
])

data = [
  (1, datetime.strptime("2016-03-11 09:10:00", "%Y-%m-%d %H:%M:%S"), True),
  (2, datetime.strptime("2016-03-11 10:30:00", "%Y-%m-%d %H:%M:%S"), False),
  (3, datetime.strptime("2016-03-11 10:00:00", "%Y-%m-%d %H:%M:%S"), True)
]

df = spark.createDataFrame(
  data=data,
  schema=data_schema
)

# COMMAND ----------

feature_lookups = [
  fe.FeatureLookup(
    table_name=output_table_name,
    lookup_key=lookup_key,
    timestamp_lookup_key=timestamp_lookup_key,
    lookback_window=lookback_window,
    feature_names=["last_hour_ride_request_count"]
  )
]

# COMMAND ----------

training_set = fe_client.create_training_set(
  df=df,
  feature_lookups = feature_lookups,  
  label="request_accepted",
  exclude_columns=lookup_key+timestamp_lookup_key
)

training_df = training_set.load_df()

training_df.display()

# COMMAND ----------

# MAGIC %md ### TRAIN MODEL

# COMMAND ----------

mlflow.autolog()

from sklearn

# COMMAND ----------

pd_df = training_df.toPandas()
X, y = pd_df.drop(label, axis=1), pd_df[label]
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

# COMMAND ----------

clf.predict(X)

# COMMAND ----------

# MAGIC %md ### REGISTER MODEL

# COMMAND ----------

fe_client.log_model(
  model=clf,
  registered_model_name=model_name,
  flavor=mlflow.sklearn,
  artifact_path="model",
  training_set=training_set)

# COMMAND ----------

# MAGIC %md ## BATCH SCORING MODEL

# COMMAND ----------

# MAGIC %md ### BUILD SCORING DATASET

# COMMAND ----------

#FAKE DATA
data_schema  = T.StructType([
   T.StructField("request_id", T.IntegerType(), False),
   T.StructField("time_stamp", T.TimestampType(), False)
])

data = [
  (1, datetime.strptime("2016-03-11 09:10:00", "%Y-%m-%d %H:%M:%S")),
  (2, datetime.strptime("2016-03-11 10:30:00", "%Y-%m-%d %H:%M:%S")),
  (3, datetime.strptime("2016-03-11 10:00:00", "%Y-%m-%d %H:%M:%S"))
]

df = spark.createDataFrame(
  data=data,
  schema=data_schema
)

# COMMAND ----------

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = mlflow.MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

latest_model_version = get_latest_model_version(model_name)


batch_preds = fe_client.score_batch(
  model_uri=f"models:/{model_name}/{latest_model_version}",
  df=df,
  result_type='boolean')

# COMMAND ----------

batch_preds.display()

# COMMAND ----------

# MAGIC %md ## STREAMING SCORING MODEL

# COMMAND ----------

# MAGIC %md #### TODO: 
# MAGIC - HOW IS THIS DONE WITH STREAMING? WITH ONLINE TABLES AND MODELSERVING?
# MAGIC - ADD ONLINE TABLES AND MODEL SERVING

# COMMAND ----------

# MAGIC %md # CLEAN UP

# COMMAND ----------

spark.sql(f"DROP SCHEMA {catalog}.{schema} CASCADE")

# COMMAND ----------

# MAGIC %md #### TODO: 
# MAGIC - DELETE EXPERIMENT
# MAGIC - WILL THIS NOT ORPHAN MODEL/VOLUME DATA?
