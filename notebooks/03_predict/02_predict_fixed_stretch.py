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

# COMMAND ----------

from ast import literal_eval
from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from dtaml.utils.table import factory_table

import customer_headroom.utils.persist_utils as persist_utils
import customer_headroom.utils.simulation_utils as simulation_utils
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

config_sim = config["baseline_stretch_simulations"]

# COMMAND ----------

def get_campaign(campaign, etl_date):
    if (campaign == "{campaign}") or (campaign == ""):
        campaign = get_date(etl_date)
    return campaign
campaign = get_campaign(config.dates.upcoming_campaign, config.dates.etl_date)

# COMMAND ----------

config_use = config["use_segments"]
segmentations_tbl_name = factory_table(
    table_prefix=config_use.segmentations_tbl.prefix, sensitivity=config.sensitivity
)
segmentations_tbl = persist_utils.read_table(
    table_name=segmentations_tbl_name, where=f"campaign={campaign}"
)
if config_use["all"]:
    logger.info("Use all Segmentations")
    seg_list = find_all_segments(segmentations_tbl, config_use["partitionByList"])
else:
    seg_list = config_use["seg_list"]
logger.info(f"Segmentations: {seg_list}")

# COMMAND ----------

config_bd = config["build_dataset"]
trx_manager = TransactionsManager(
        etl_date=get_date(config.dates.etl_date),
        lookback_days=config.dates.lookback_days,
        l1_ids=config_bd["l1_ids"],
        lx=config_bd["lx"],
        lx_ids=config_bd["lx_ids"],  # getting all the products in this l2 id
        user_key=config_bd["user_id"],
        window_days=config_bd["window_days"],
        time_window_length=config_bd["time_window_days"],
        exclude_items=literal_eval(config["exclude_items"]),
        aggregation_level=config_bd["aggregation_level"],
    )
trx_line = spark.table("analytics_trans_prod.all_transaction_line")
articles_df = spark.table("analytics_trans_prod.lu_article")

# COMMAND ----------

trx_line = (
            trx_manager._add_date(trx_line)
            .filter(F.col("date") <= trx_manager.etl_date)
            .filter(F.col("date") >= trx_manager.lookback_date)
            .filter(F.col("PURCHASE_CHANNEL").isin(trx_manager.channels))
            .filter(F.col("l1_id").isin(list(trx_manager.l1_ids)))
            .filter(trx_manager.get_common_filters())
        )

if trx_manager.christmas_remove_range is not None:
  trx_line = trx_manager.remove_christmas_transactions(
                trx_line, christmas_range=trx_manager.christmas_remove_range
            )

# Remove items from transaction list, e.g. BWS items
trx_line = trx_manager.remove_items(trx_line)

if segmentations_tbl is not None:
    # Only keep customers in segmentations
    trx_line = trx_line.join(
        segmentations_tbl.select(trx_manager.user_key).distinct(), on=trx_manager.user_key
    )

cust_lx_trx = trx_manager.get_customer_transactions(trx_line, articles_df)


# COMMAND ----------

config_sim["rolling_window_col"] = f"rolling_{config_sim['rolling_window']}_week_sales"

# COMMAND ----------

weekly_sales = simulation_utils.calculate_weekly_rolling_sum(cust_lx_trx, 
                                                            config_sim['rolling_window'],
                                                            config_sim['rolling_window_col'],
                                                            config_sim['date_col'], 
                                                            trx_manager.lookback_date, 
                                                            config_sim['col_to_sum'], 
                                                            config_sim['grouping_cols'])

# COMMAND ----------

baseline_per_customer = simulation_utils.calculate_baselines_plus_stretch_combs(config_sim['baseline_percentiles'], 
                                                                                weekly_sales,
                                                                               config_sim['rolling_window_col'], 
                                                                               config_sim['grouping_cols'], 
                                                                               config_sim['stretch_amounts'])
logger.info(f"baseline_per_customer: {baseline_per_customer}")

# COMMAND ----------

baseline_per_customer.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculating percentiles of baseline + stretch

# COMMAND ----------

baseline_columns = [col for col in baseline_per_customer.columns if col.startswith("baseline")]
zero_stretch_cols = [col for col in baseline_columns if 'stretch_0_perc' in col]
percentile_columns = list(set(baseline_columns) - set(zero_stretch_cols))

# COMMAND ----------

percentile_df = simulation_utils.calculate_percentile_rank(baseline_per_customer, 
                                                           weekly_sales,
                                                           config_sim['rolling_window_col'], 
                                                           percentile_columns,
                                                           config_sim['grouping_cols'])

# COMMAND ----------

# Adjusting percentile values for 0% stretch and number of shops

# Setting 85th baseline + 0% stretch percentile as 85th percentile
percentile_df = percentile_df.withColumn("percentile_baseline_85_stretch_0_perc",lit(0.85))

# Counting number of weekly sales for each customer in the past year
grouped_df = weekly_sales.groupBy(config_sim['grouping_cols']).agg(
    F.sum(F.when(F.col(config_sim['col_to_sum']) > 0, 1).otherwise(0)).alias("non_zero_sales_count")
)
percentile_df = percentile_df.join(grouped_df, on=config_sim['grouping_cols'], how='left')

# If customer shopped less than two times in the past year, set the percentile to 100%
percentile_df = percentile_df.withColumn("percentile_baseline_85_stretch_10_perc",F.when(F.col('non_zero_sales_count') <2, lit(1)).otherwise(F.col("percentile_baseline_85_stretch_10_perc")))
percentile_df = percentile_df.withColumn("percentile_baseline_85_stretch_20_perc",F.when(F.col('non_zero_sales_count') <2, lit(1)).otherwise(F.col("percentile_baseline_85_stretch_20_perc")))
percentile_df = percentile_df.withColumn("percentile_baseline_85_stretch_30_perc",F.when(F.col('non_zero_sales_count') <2, lit(1)).otherwise(F.col("percentile_baseline_85_stretch_30_perc")))

# COMMAND ----------

percentile_of_stretches_cols = [col for col in percentile_df.columns if col.startswith("percentile")]
final_df = baseline_per_customer.join(percentile_df.select(['cust_id']+percentile_of_stretches_cols), on=config_sim['grouping_cols'], how="left")
final_df = final_df.drop('mean_value', 'median_value')

# COMMAND ----------

fixed_stretch_tbl_name= persist_utils.create_beam_table(
    table_prefix=config_sim.fixed_stretch_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=final_df,
    partition_by=config_sim.fixed_stretch_tbl.partitionByList,
    overwrite_table=True,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""fixed_stretch_tbl_name: {fixed_stretch_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=fixed_stretch_tbl_name,
    insert_df=final_df,
    insert_append=True,
    add_columns=True,
)

# COMMAND ----------

fixed_stretch_tbl = persist_utils.read_table(
    table_name=fixed_stretch_tbl_name
)
fixed_stretch_tbl.columns
