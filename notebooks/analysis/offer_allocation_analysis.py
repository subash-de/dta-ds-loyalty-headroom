# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.functions import col, weekofyear, to_date, expr, datediff
from pyspark.sql.window import Window

# COMMAND ----------

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import Window as W
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.allocation.allocator import Allocator
from customer_headroom.utils import tmo_utils

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

config_al = config['allocation']

# COMMAND ----------

from ast import literal_eval
from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from dtaml.utils.table import factory_table

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.etl.build_dataset import TransactionsManager, TransactionsManagerFixedStretch
from customer_headroom.etl.etl_utils import (
    find_all_segments,
    get_campaign,
    get_count,
    get_date,
)

from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T

# COMMAND ----------

june_accumulator = spark.sql("""select * from bullseye_azlab_prod.accumulator_240610_export where campaign = 20240610""")

# COMMAND ----------

june_accumulator.display()

# COMMAND ----------

trx_line = spark.table("analytics_trans_prod.all_transaction_line")
articles_df = spark.table("analytics_trans_prod.lu_article")

# COMMAND ----------

def get_date(date):
    if str(date).lower() == "today":
        date = datetime.now().strftime("%Y%m%d")
    return int(date)

# COMMAND ----------

config_sim = config['baseline_stretch_simulations']
config_bd = config['build_dataset']
config_sim["rolling_window_col"] = f"rolling_{config_sim['rolling_window']}_week_sales"
trx_manager_fixed_stretch = TransactionsManagerFixedStretch(
        etl_date=get_date('20240531'),
        lookback_days=config.dates.lookback_days,
        grouping_columns=config_sim['grouping_columns'],
        rolling_window = config_sim['rolling_window'],
        rolling_window_col = config_sim["rolling_window_col"],
        l1_ids=config_bd["l1_ids"],
        lx=config_bd["lx"],
        lx_ids=config_bd["lx_ids"],  # getting all the products in this l2 id
        user_key=config_bd["user_id"],
        exclude_items=literal_eval(config["exclude_items"]),
    )

# COMMAND ----------

fixed_stretch_etl_tbl = trx_manager_fixed_stretch.get(trx_line, articles_df)

# COMMAND ----------

fixed_stretch_etl_tbl.columns

# COMMAND ----------

fixed_stretch_etl_tbl.display()

# COMMAND ----------

#predictions
from customer_headroom.modelling.predict import Predictor,PredictorFixedStretch

# COMMAND ----------

fixed_stretch_predicition_manager = PredictorFixedStretch(
  grouping_columns = config_sim['grouping_columns'],
  rolling_window_col = config_sim["rolling_window_col"],
  baseline_percentiles = config_sim['baseline_percentiles'],
  stretch_amounts = config_sim['stretch_amounts'],                                        
  )

baseline_per_customer = fixed_stretch_predicition_manager.get(fixed_stretch_etl_tbl)

# COMMAND ----------

baseline_per_customer.display()

# COMMAND ----------

#allocation
from customer_headroom.allocation.allocator import Allocator

# COMMAND ----------

fixed_stretch_allocation_manager = Allocator(
        feature_col=config_al["feature_col"],
        offer_limits=config["offer_limits"],
        offer_desc=config["offers_desc"],
        user_key=config_al["user_key"],
        outlier_min=config_al["outlier_min"],
        outlier_max=config_al["outlier_max"],
        max_increase=config_al["max_increase"],
        min_increase=config_al["min_increase"],
        headroom_factor=config_al["headroom_factor"],
        fill_offer=config_al["fill_offer"],
        prev_not_bought_factor=config_al["prev_not_bought_factor"],
    )

# COMMAND ----------

config["offer_limits"]

# COMMAND ----------


def allocate_offers(data_hrm):
  data_hrm = data_hrm.withColumn("rand", F.rand())
  data_hrm_cnt = data_hrm.count()
  data_hrm = data_hrm.withColumn("offer_id", F.lit(None))
  
  for k, v in config["offer_limits"].items():
    offer_id = int(k)
    data_hrm = data_hrm.withColumn(
        "offer_id",
        F.when(
            (F.col("baseline_plus_stretch") >= v[0]) & (F.col("baseline_plus_stretch") < v[1]),
            offer_id,
        ).otherwise(F.col("offer_id")),
    )
    return data_hrm

# COMMAND ----------

def allocate_offers_for_all_baselines(fixed_stretch_tbl):
    # List of baseline columns (automatically detected)
    columns_to_allocate = list(set(fixed_stretch_tbl.columns) - set([col for col in fixed_stretch_tbl.columns if col.endswith("th_percentile")] + ['cust_id'] + ['load_timestamp']))
    
    # Initialize an empty DataFrame to store the combined result
    combined_allocation_df = None
    fixed_stretch_tbl = fixed_stretch_tbl.withColumnRenamed('85th_percentile', 'sum_total_spend')
    
    # Iterate over each baseline column and allocate offers
    for col in columns_to_allocate:
        # Select necessary columns including 'cust_id', '85th_percentile', and the current baseline column
        selected_columns_df = fixed_stretch_tbl.select('cust_id', 'sum_total_spend', col)
        
        # Rename the current baseline column to 'current_baseline' so that allocate_offer can work on it
        renamed_df = selected_columns_df.withColumnRenamed(col, "baseline_plus_stretch")
        
        # Call the allocate_offer function, passing the DataFrame with the renamed column
        allocation_df = allocate_offers(renamed_df)
        
        # Add a new column to indicate the baseline column used for the allocation
        allocation_df = allocation_df.withColumn('test_type', F.lit(col))
                    
        # Combine the result with the previous results
        if combined_allocation_df is None:
            combined_allocation_df = allocation_df
        else:
            combined_allocation_df = combined_allocation_df.unionByName(allocation_df)
    
    return combined_allocation_df

# COMMAND ----------

fixed_stretch_export = allocate_offers_for_all_baselines(baseline_per_customer)

# COMMAND ----------

# def prepare_export(self, data):
#   data_export = (
#       data.withColumn(
#           "offer_id",
#           F.when(F.col("offer_id").isNull(), F.lit(self.fill_offer)).otherwise(
#               F.col("offer_id")
#           ),
#       )
#       .withColumn("spend_plus_stretch", F.round("baseline_plus_stretch", 2))
#       .withColumn("estimated_spend", F.round(F.col("sum_total_spend"), 2))
#       .withColumn(
#           "estimated_stretch",
#           F.round(F.col("spend_plus_stretch") - F.col("sum_total_spend"), 2),
#       )
#       .drop(
#           "sum_total_spend",
#           "baseline_plus_stretch",
#           "large_offers",
#           "small_offers",
#           "rand",
#       )
#   )
  
#   return data_export

# COMMAND ----------

fixed_stretch_export = fixed_stretch_allocation_manager.get(predictions= baseline_per_customer, headroom = False)
# fixed_stretch_export = fixed_stretch_export.withColumn("campaign", F.lit(campaign))

# COMMAND ----------

fixed_stretch_export.columns

# COMMAND ----------

june_accumulator.columns

# COMMAND ----------

june_accumulator = june_accumulator.withColumnRenamed('spend_plus_headroom', 'spend_plus_stretch')

# COMMAND ----------

# merging headroom export and fixed stretch export
exports_merged = june_accumulator.unionByName(fixed_stretch_export.select('cust_id','desc',)).orderBy('cust_id').unionByName(fixed_stretch_export).orderBy('cust_id')

# COMMAND ----------

# Removing customers with no headroom output
headroom_customers = exports_merged.filter(exports_merged["test_type"] == "headroom").select("cust_id").distinct()
all_export = exports_merged.join(headroom_customers, on="cust_id", how="inner")

# COMMAND ----------

all_export.columns

# COMMAND ----------

#check if offer was completed

from pyspark.sql import functions as F

# Assuming your DataFrame is named `df` and the column with descriptions is `offers_desc`
all_export = all_export.withColumn("threshold_for_offer", F.regexp_extract("desc", r"£(\d+)", 1).cast("int"))

# COMMAND ----------

from pyspark.sql import functions as F

# Assuming 'df' is your DataFrame and 'column_name' is the name of the column you want to check
min_value = june_accumulator.select(F.min('load_timestamp')).collect()[0][0]
max_value = june_accumulator.select(F.max('load_timestamp')).collect()[0][0]

print("Minimum value:", min_value)
print("Maximum value:", max_value)


# COMMAND ----------

june_accumulator_transactions = trx_line.filter(F.col("load_timestamp").between(min_value, max_value))

# COMMAND ----------

june_accumulator_spend_per_customer = june_accumulator_transactions.groupBy("cust_id").agg(F.sum("sales_amt").alias("actual_spend"))

# COMMAND ----------

june_export = all_export.join(june_accumulator_spend_per_customer, on="cust_id", how="left")

# COMMAND ----------

june_export.display()

# COMMAND ----------

june_export = june_export.filter(F.col('actual_spend').isNotNull())

# COMMAND ----------

june_export = june_export.withColum('offer_completed', F.when(june_export.threshold_for_offer <= all_export.actual_spend, 1).otherwise(0))

# COMMAND ----------

june_export.groupBy('test_type').agg(F.col('offer_completed').sum()).show()

# COMMAND ----------

from pyspark.sql import functions as F

# Group by `test_type` and calculate the completion rate
result = june_export.groupBy('test_type').agg(
    F.count('*').alias('test_type_count'),  # Count of rows for each `test_type`
    F.sum('offer_completed').alias('offer_completed_sum')  # Sum of completed offers for each `test_type`
).withColumn(
    'completion_rate', F.col('offer_completed_sum') / F.col('test_type_count')
)

result.select('test_type', 'test_type_count', 'offer_completed_sum', 'completion_rate').display()


# COMMAND ----------

# calculating uplift

# 10% stretch
new_completion_customers_10 = june_export.filter((F.col('test_type') == '85_perc_10_stretch') & (F.col('offer_completed') == 1) & (F.col('test_type') == 'headroom') & (F.col('offer_completed') == 0))

new_completion_customers_10 = new_completion_customers_10.withColumn('increase_spending', F.col('threshold_for_offer') - F.col('actual_spend'))
total_increase_spending = new_completion_customers_10.agg(F.sum('increase_spending')).collect()[0][0]
print("financial uplift for 10% stretch customers:",total_increase_spending)

# 20% stretch
new_completion_customers_20 = june_export.filter((F.col('test_type') == '85_perc_20_stretch') & (F.col('offer_completed') == 1) & (F.col('test_type') == 'headroom') & (F.col('offer_completed') == 0))

new_completion_customers_20 = new_completion_customers_20.withColumn('increase_spending', F.col('threshold_for_offer') - F.col('actual_spend'))
total_increase_spending = new_completion_customers_20.agg(F.sum('increase_spending')).collect()[0][0]
print("financial uplift for 20% stretch customers:",total_increase_spending)

# 30% stretch
new_completion_customers_30 = june_export.filter((F.col('test_type') == '85_perc_30_stretch') & (F.col('offer_completed') == 1) & (F.col('test_type') == 'headroom') & (F.col('offer_completed') == 0))

new_completion_customers_30 = new_completion_customers_30.withColumn('increase_spending', F.col('threshold_for_offer') - F.col('actual_spend'))
total_increase_spending = new_completion_customers_30.agg(F.sum('increase_spending')).collect()[0][0]
print("financial uplift for 30% stretch customers:",total_increase_spending)

# COMMAND ----------


