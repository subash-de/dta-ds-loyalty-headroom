# Databricks notebook source
# MAGIC %run ../bootstrap

# COMMAND ----------

from ast import literal_eval
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import functions as F

from customer_headroom.etl.build_dataset import TransactionsManagerFixedStretch
from customer_headroom.etl.etl_utils import get_date
from customer_headroom.modelling.predict import PredictorFixedStretch
import customer_headroom.utils.persist_utils as persist_utils

# COMMAND ----------

division = "FD"
category = {"L3_ID": ["MM01"]}
start_date = str(datetime.today().date() - timedelta(days=config.dates.lookback_days))

category_indicator = " OR ".join([f"{key} in ({','.join(map(repr, value))})" for key, value in category.items()])

# COMMAND ----------

config_sim = config["baseline_stretch_simulations"]
config_bd = config["build_dataset"]

# COMMAND ----------

category_trans = spark.sql(
  f"""
  select *
  from {config.factory_tbl_all_transaction_line}
  where 
  trans_line_type = 'S'
  and division_id = '{division}'
  and event_date > '{start_date}'
  and ({category_indicator})
  """)

trx_line_df = persist_utils.read_table(
  table_name = config.factory_tbl_all_transaction_line
)

articles_df = persist_utils.read_table(
  table_name = config.factory_tbl_lu_article
)

# COMMAND ----------

category_trans.select("article_id").distinct().count()

# COMMAND ----------

trx_manager_fixed_stretch = TransactionsManagerFixedStretch(
    baseline_percentiles=config_sim["baseline_percentiles"],
    etl_date=get_date("today"),
    lookback_days=config.dates.lookback_days,
    grouping_columns=[config["build_dataset"]["user_id"], "l3_id"],
    rolling_window=config_sim["rolling_window"],
    rolling_window_col=f"rolling_{config_sim['rolling_window']}_week_sales",
    l1_ids=config_bd["l1_ids"],
    category_level=True,
    l2_ids=config_bd["l2_ids"],
    lx="l3",
    lx_ids={"category": category},
    user_key=config_bd["user_id"],
    exclude_items=literal_eval(config["exclude_items"]),
)

percentile_df = trx_manager_fixed_stretch.get(
  trx_line_df,
  articles_df
).filter(F.col("l3_id") == "category")

# COMMAND ----------

fixed_stretch_predicition_manager = PredictorFixedStretch(
    grouping_columns=[config["build_dataset"]["user_id"], "l3_id"],
    rolling_window_col=f"rolling_{config_sim['rolling_window']}_week_sales",
    baseline_percentiles=config_sim['baseline_percentiles'],
    stretch_amounts=[20, 30],                               
)

stretch_per_customer = fixed_stretch_predicition_manager.get(percentile_df)

# COMMAND ----------

stretch_per_customer.display()

# COMMAND ----------

stretch_per_customer_pandas = stretch_per_customer.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### For category level (especially food), the lowest offer variant is determined by one unit article price.

# COMMAND ----------

category_unit_price = category_trans.groupby("ARTICLE_NUMBER").agg(F.max("UNIT_FULL_PRICE").alias("unit_price")).toPandas()

# COMMAND ----------

plt.hist(category_unit_price['unit_price'], bins=100)
plt.axvline(category_unit_price['unit_price'].astype('float').quantile(0.5), color='orange', linestyle='dashed', label='50th percentile')
plt.axvline(category_unit_price['unit_price'].astype('float').quantile(0.75), linestyle='dashed', label='75th percentile', color='cyan')
plt.axvline(category_unit_price['unit_price'].astype('float').quantile(0.85), color='red', linestyle='dashed', label='85th percentile')
plt.xlim(-1, 25)
current_ticks = plt.xticks()[0][1:]
new_ticks = list(current_ticks) + [
  category_unit_price['unit_price'].astype('float').quantile(0.5),
  category_unit_price['unit_price'].astype('float').quantile(0.75),
  category_unit_price['unit_price'].astype('float').quantile(0.85)
]
plt.xticks(new_ticks, fontsize=8, rotation=30)
plt.xlabel("Product unit price")
plt.ylabel("Frequency")
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC ### The subsequent offer variants are determined through looking at the distribution of the stretched spending. 
# MAGIC ### The highest offer variant is normally determined by 95th - 97th percentile of the stretched spending.

# COMMAND ----------

stretch_per_customer_pandas['85_stretch_30_perc'].quantile([0, 0.25, 0.5, 0.75, 0.95, 0.97, 0.99, 1.0])

# COMMAND ----------

plt.hist(stretch_per_customer_pandas['85_stretch_30_perc'], bins=700, label='85th percentile + 30% stretch')
plt.axvline(stretch_per_customer_pandas['85_stretch_30_perc'].quantile(0.5), color='red', linestyle='dashed', label='50th percentile')
plt.axvline(stretch_per_customer_pandas['85_stretch_30_perc'].quantile(0.85), color='orange', linestyle='dashed', label='85th percentile')
plt.axvline(stretch_per_customer_pandas['85_stretch_30_perc'].quantile(0.95), color='cyan', linestyle='dashed', label='95th percentile')
plt.xlim(-1, 100)
current_ticks = plt.xticks()[0][1:]
new_ticks = list(current_ticks) + [
  stretch_per_customer_pandas['85_stretch_30_perc'].quantile(0.5),
  stretch_per_customer_pandas['85_stretch_30_perc'].quantile(0.85),
  stretch_per_customer_pandas['85_stretch_30_perc'].quantile(0.95)
]
plt.xticks(new_ticks, fontsize=8, rotation=30)
plt.xlabel("Baseline 85th percentile + 30% stretch")
plt.ylabel("Frequency")
plt.legend()

# COMMAND ----------


