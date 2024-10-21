# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basline + stretch simulations

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, weekofyear, to_date, expr, datediff
from pyspark.sql.window import Window
from pyspark.sql.functions import lit
import os
import sys

# COMMAND ----------

import sys
import customer_headroom.utils.persist_utils as persist_utils
# from customer_headroom import config

# COMMAND ----------

from ast import literal_eval
from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from dtaml.utils.table import factory_table

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.etl.build_dataset import TransactionsManager
from customer_headroom.etl.etl_utils import (
    find_all_segments,
    get_campaign,
    get_count,
    get_date,
)

from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T


sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

import customer_headroom.utils.simulation_utils as simulation_utils

# COMMAND ----------

trans_before_june_accu = persist_utils.read_table(
    table_name = config.tables.factory_tbl_all_transaction_line_dev,
    where= "EVENT_DATE >= '2023-06-01'"
           and "EVENT_DATE <= '2024-05-31'"
           and "sparks_reg_date <= '2023-06-01'"
           and "trans_line_type = 'S'"
           and "division_id = 'FD'"
           and "sparks_account_id is not null")

# COMMAND ----------

weekly_sales = simulation_utils.calculate_weekly_rolling_sum(trans_before_june_accu, 4, "EVENT_DATE", "2023-06-01", "sales_amt", "cust_id")

# COMMAND ----------

weekly_sales.display()

# COMMAND ----------

baseline_per_customer = simulation_utils.calculate_baselines_plus_stretch_combs(85, weekly_sales,"rolling_4_week_sales", "cust_id", [0, 10, 30])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculating percentiles of baseline + stretch

# COMMAND ----------

baseline_columns = [col for col in baseline_per_customer.columns if col.startswith("baseline")]
zero_stretch_cols = [col for col in baseline_columns if 'stretch_0_perc' in col]
percentile_columns = list(set(baseline_columns) - set(zero_stretch_cols))

# COMMAND ----------

percentile_df = simulation_utils.calculate_percentile_rank(baseline_per_customer, weekly_sales,"rolling_4_week_sales", percentile_columns,"cust_id")

# COMMAND ----------

# Adjusting percentile values for 0% stretch and number of shops

# Setting 85th baseline + 0% stretch percentile as 85th percentile
percentile_df = percentile_df.withColumn("percentile_baseline_85_stretch_0_perc",lit(0.85))

# Counting number of weekly sales for each customer in the past year
grouped_df = weekly_sales.groupBy("cust_id").agg(
    F.sum(F.when(F.col("sales_amt") > 0, 1).otherwise(0)).alias("non_zero_sales_count")
)
percentile_df = percentile_df.join(grouped_df, on='cust_id', how='left')

# If customer shopped less than two times in the past year, set the percentile to 100%
percentile_df = percentile_df.withColumn("percentile_baseline_85_stretch_10_perc_adjusted",F.when(F.col('non_zero_sales_count') <2, lit(1)).otherwise(F.col("percentile_baseline_85_stretch_10_perc")))
percentile_df = percentile_df.withColumn("percentile_baseline_85_stretch_30_perc_adjusted",F.when(F.col('non_zero_sales_count') <2, lit(1)).otherwise(F.col("percentile_baseline_85_stretch_30_perc")))

# COMMAND ----------

# Finding the average percentile for each baseline + stretch combination
percentile_columns = [col for col in percentile_df.columns if col.startswith("percentile")]
percentile_aggregations = percentile_df.agg(*[F.mean(col).alias(f"{col}_avg") for col in percentile_columns])
percentile_aggregations.display()

# COMMAND ----------

percentile_df_pandas = percentile_df.toPandas()

# COMMAND ----------

# Plotting percentile distributions
simulation_utils.plot_percentiles_distributions(percentile_df_pandas, ['percentile_baseline_85_stretch_0_perc', 'percentile_baseline_85_stretch_10_perc_adjusted','percentile_baseline_85_stretch_30_perc_adjusted'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Deviation from mean & median spending

# COMMAND ----------

# Calculating differences between different baseline + stretches and mean/max/median values
for col in baseline_columns:
  baseline_per_customer = baseline_per_customer.withColumn(col + "_diff_mean", (F.col(col) - F.col("mean_value"))/F.col("mean_value"))
  baseline_per_customer = baseline_per_customer.withColumn(col + "_diff_median", (F.col(col) - F.col("median_value"))/F.col("median_value"))


# COMMAND ----------

difference_columns = [col for col in baseline_per_customer.columns if "_diff_" in col]
baseline_per_customer_agg = baseline_per_customer.agg(*[F.mean(col).alias(f"{col}_avg") for col in difference_columns])
baseline_per_customer_agg.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer segment breakdown

# COMMAND ----------

customer_df = spark.sql("""
                  SELECT *
                  FROM(
                  SELECT 
                      *,
                      ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY yyyymmdd DESC) AS row_num
                  FROM customer_azbase_prod.segtco_history
                  where yyyymmdd <= 20240610)
                  WHERE row_num = 1
                  """)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Percentiles

# COMMAND ----------

# Joining customer breakdown table with percentile rank data
percentile_customer_segment = percentile_df.join(customer_df.select('cust_id', 'cust_band_fd').distinct(), on='cust_id', how='left')

# COMMAND ----------

# Finding the average percentile for each baseline + stretch combination
percentile_aggregations = percentile_customer_segment.groupBy('cust_band_fd').agg(*[F.mean(col).alias(f"{col}_avg") for col in percentile_columns])
percentile_aggregations.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Deviation from mean/median spending

# COMMAND ----------

# Joining customer breakdown table with difference from median/mean
customer_segment_diff = baseline_per_customer.join(customer_df.select('cust_id', 'cust_band_fd').distinct(), on='cust_id', how='left')

# COMMAND ----------

# Finding the average percentile for each baseline + stretch combination
diff_aggregations = customer_segment_diff.groupBy('cust_band_fd').agg(*[F.mean(col).alias(f"{col}_avg") for col in difference_columns])
# diff_aggregations.display()

# COMMAND ----------

percentile_customer_segment_pandas = percentile_customer_segment.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt

# Define the categories you want to plot
categories = percentile_customer_segment_pandas['cust_band_fd'].unique()
categories = [cat for cat in categories if cat is not None]

# Set up the plot
plt.figure(figsize=(10, 6))

# Loop through each category and plot its histogram
for category in categories:
    print("Customer Segment:", category)
    subset = percentile_customer_segment_pandas[percentile_customer_segment_pandas['cust_band_fd'] == category]
    simulation_utils.plot_percentiles_distributions(subset, ['percentile_baseline_85_stretch_0_perc', 'percentile_baseline_85_stretch_10_perc_adjusted', 'percentile_baseline_85_stretch_30_perc_adjusted'])

# COMMAND ----------


