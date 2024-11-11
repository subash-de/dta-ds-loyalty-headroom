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
        baseline_percentiles = config_sim["baseline_percentiles"],
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

config_sim['stretch_amounts']

# COMMAND ----------

baseline_per_customer.columns

# COMMAND ----------

baseline_per_customer.display()

# COMMAND ----------

from pyspark.sql import functions as F

# List of offers and corresponding spend thresholds
offers = [
    '£5 off when you spend £30 on M&S food in store',
    '£5 off when you spend £50 on M&S food in store',
    '£7 off when you spend £70 on M&S food in store',
    '£9 off when you spend £90 on M&S food in store',
    '£11 off when you spend £110 on M&S food in store',
    '£13 off when you spend £130 on M&S food in store',
    '£15 off when you spend £150 on M&S food in store',
    '£17 off when you spend £170 on M&S food in store',
    '£19 off when you spend £190 on M&S food in store',
    '£21 off when you spend £210 on M&S food in store',
    '£23 off when you spend £230 on M&S food in store',
    '£25 off when you spend £250 on M&S food in store',
    '£27 off when you spend £270 on M&S food in store'
]

# Define the spend ranges for each offer
spend_ranges = [
    (0, 21), (21, 50), (50, 70), (70, 90), (90, 110), (110, 130),
    (130, 150), (150, 170), (170, 190), (190, 210), (210, 230),
    (230, 250), (250, 270)
]

# Start the `when` condition chain with the first condition
desc_column = F.when(
    (F.col("85_stretch_0_perc") >= spend_ranges[0][0]) & (F.col("85_stretch_0_perc") < spend_ranges[0][1]),
    offers[0]
)

# Loop through remaining conditions, adding each `when` statement to the column
for (min_val, max_val), offer in zip(spend_ranges[1:], offers[1:]):
    desc_column = desc_column.when(
        (F.col("85_stretch_0_perc") >= min_val) & (F.col("85_stretch_0_perc") < max_val),
        offer
    )

# Add a default value with `otherwise(None)` for rows that don’t match any condition
desc_column = desc_column.otherwise(None)

# Add the new column `desc` to the Spark DataFrame
offers_export = baseline_per_customer.withColumn("desc", desc_column)


# COMMAND ----------

offers_export = offers_export.withColumnRenamed('85_stretch_0_perc', 'spend_plus_stretch')
offers_export = offers_export.withColumn('test_type', F.lit('85_stretch_0_perc'))

# COMMAND ----------

offers_export

# COMMAND ----------

offers_export_0 = offers_export.select('cust_id', 'test_type','spend_plus_stretch', 'desc')

# COMMAND ----------

offers_export.display()

# COMMAND ----------

import re
fixed_stretch_pattern = r"^\d+_stretch_\d+_perc$"
columns_to_allocate = [col for col in baseline_per_customer.columns if re.match(fixed_stretch_pattern, col)]

# COMMAND ----------

columns_to_allocate[1:]

# COMMAND ----------

# List of all columns that should be processed similarly to '85_stretch_0_perc'
stretch_columns = columns_to_allocate[1:]

# Define a function to apply the offer descriptions to a column
def generate_desc_column(column_name):
    # Initialize the `when` condition chain
    desc_col = F.when(
        (F.col(column_name) >= spend_ranges[0][0]) & (F.col(column_name) < spend_ranges[0][1]),
        offers[0]
    )
    # Loop through remaining conditions for the column
    for (min_val, max_val), offer in zip(spend_ranges[1:], offers[1:]):
        desc_col = desc_col.when(
            (F.col(column_name) >= min_val) & (F.col(column_name) < max_val),
            offer
        )
    # Add a default value with `otherwise(None)`
    return desc_col.otherwise(None)

# Create a list to store the individual DataFrames for each column
desc_dfs = []

# Loop through each column, generate the description column, and select the required columns
for col in stretch_columns:
    print("Offer allocation for:", col)
    desc_column = generate_desc_column(col)
    temp_df = baseline_per_customer.withColumn("desc", desc_column) \
                                   .select("cust_id", F.lit(col).alias("test_type"), F.col(col).alias("spend_plus_stretch"), "desc")
    desc_dfs.append(temp_df)

# Union all the DataFrames for each column to create the final long-format DataFrame
final_df = desc_dfs[0]
for df in desc_dfs[1:]:
    final_df = final_df.unionByName(df)

# COMMAND ----------

final_df.columns

# COMMAND ----------

final_df = final_df.unionByName(offers_export_0)

# COMMAND ----------

final_df.display()

# COMMAND ----------

final_df.groupBy('test_type').count().show()

# COMMAND ----------

#allocation
# from customer_headroom.allocation.allocator import Allocator

# COMMAND ----------

# fixed_stretch_allocation_manager = Allocator(
#         feature_col=config_al["feature_col"],
#         offer_limits=config["offer_limits"],
#         offer_desc=config["offers_desc"],
#         user_key=config_al["user_key"],
#         outlier_min=config_al["outlier_min"],
#         outlier_max=config_al["outlier_max"],
#         max_increase=config_al["max_increase"],
#         min_increase=config_al["min_increase"],
#         headroom_factor=config_al["headroom_factor"],
#         fill_offer=config_al["fill_offer"],
#         prev_not_bought_factor=config_al["prev_not_bought_factor"],
#     )

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

# fixed_stretch_export = fixed_stretch_allocation_manager.get(predictions= baseline_per_customer, headroom = False)
# fixed_stretch_export = fixed_stretch_export.withColumn("campaign", F.lit(campaign))

# COMMAND ----------

fixed_stretch_export = final_df

# COMMAND ----------

fixed_stretch_export.columns

# COMMAND ----------

june_accumulator.columns

# COMMAND ----------

june_accumulator = june_accumulator.withColumnRenamed('spend_plus_headroom', 'spend_plus_stretch')
june_accumulator = june_accumulator.withColumn('test_type', F.lit('headroom'))

# COMMAND ----------

# merging headroom export and fixed stretch export
exports_merged = june_accumulator.select('cust_id', 'test_type', 'spend_plus_stretch', 'desc').unionByName(fixed_stretch_export)

# COMMAND ----------

exports_merged.select('test_type').distinct().show()

# COMMAND ----------

exports_merged.columns

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
all_export = all_export.withColumn("reward", F.regexp_extract("desc", r"£(\d+)", 1).cast("int"))
all_export = all_export.withColumn("threshold_for_offer", F.regexp_extract("desc", r"£\d+.*?£(\d+)", 1).cast("int"))

# COMMAND ----------

from pyspark.sql import functions as F

# Assuming 'df' is your DataFrame and 'column_name' is the name of the column you want to check
min_value = june_accumulator_transactions.select(F.min('EVENT_DATE')).collect()[0][0]
max_value = june_accumulator_transactions.select(F.max('EVENT_DATE')).collect()[0][0]

print("Minimum value:", min_value)
print("Maximum value:", max_value)


# COMMAND ----------


june_accumulator_transactions = trx_line.filter(F.col("EVENT_DATE").between(F.lit(get_date('20240610')).cast('timestamp'), F.lit(get_date('20240707')).cast('timestamp')))

# COMMAND ----------

june_accumulator_spend_per_customer = june_accumulator_transactions.groupBy("cust_id").agg(F.sum("sales_amt").alias("actual_spend"))

# COMMAND ----------

june_accumulator_spend_per_customer.columns

# COMMAND ----------

june_export = all_export.join(june_accumulator_spend_per_customer, on="cust_id", how="left")

# COMMAND ----------

june_export.columns

# COMMAND ----------

june_export.display()

# COMMAND ----------

june_export = june_export.filter(F.col('actual_spend').isNotNull())

# COMMAND ----------

june_export = june_export.withColumn('offer_completed', F.when(june_export.threshold_for_offer <= june_export.actual_spend, 1).otherwise(0))

# COMMAND ----------

june_export.columns

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

result.display()

# COMMAND ----------

# calculating uplift

# 10% stretch
new_completion_customers_10 = june_export.filter(
    ((F.col('test_type') == '85_perc_10_stretch') & (F.col('offer_completed') == 1)) |
    ((F.col('test_type') == 'headroom') & (F.col('offer_completed') == 0))
)


new_completion_customers_10 = new_completion_customers_10.withColumn('increase_spending', F.col('threshold_for_offer') - F.col('actual_spend'))
total_increase_spending = new_completion_customers_10.agg(F.sum('increase_spending')).collect()[0][0]
print("financial uplift for 10% stretch customers:",total_increase_spending)

# # 20% stretch
# new_completion_customers_20 = june_export.filter((F.col('test_type') == '85_perc_20_stretch') & (F.col('offer_completed') == 1) & (F.col('test_type') == 'headroom') & (F.col('offer_completed') == 0))

# new_completion_customers_20 = new_completion_customers_20.withColumn('increase_spending', F.col('threshold_for_offer') - F.col('actual_spend'))
# total_increase_spending = new_completion_customers_20.agg(F.sum('increase_spending')).collect()[0][0]
# print("financial uplift for 20% stretch customers:",total_increase_spending)

# # 30% stretch
# new_completion_customers_30 = june_export.filter((F.col('test_type') == '85_perc_30_stretch') & (F.col('offer_completed') == 1) & (F.col('test_type') == 'headroom') & (F.col('offer_completed') == 0))

# new_completion_customers_30 = new_completion_customers_30.withColumn('increase_spending', F.col('threshold_for_offer') - F.col('actual_spend'))
# total_increase_spending = new_completion_customers_30.agg(F.sum('increase_spending')).collect()[0][0]
# print("financial uplift for 30% stretch customers:",total_increase_spending)

# COMMAND ----------

new_completion_customers_10.agg(F.sum('increase_spending')).show()

# COMMAND ----------

# 20% stretch
new_completion_customers_20 = june_export.filter(((F.col('test_type') == '85_perc_20_stretch') & (F.col('offer_completed') == 1)) |
    ((F.col('test_type') == 'headroom') & (F.col('offer_completed') == 0)))

new_completion_customers_20 = new_completion_customers_20.withColumn('increase_spending', F.col('threshold_for_offer') - F.col('actual_spend'))
total_increase_spending = new_completion_customers_20.agg(F.sum('increase_spending')).collect()[0][0]
print("financial uplift for 20% stretch customers:",total_increase_spending)

# 30% stretch
new_completion_customers_30 = june_export.filter(((F.col('test_type') == '85_perc_30_stretch') & (F.col('offer_completed') == 1)) |
    ((F.col('test_type') == 'headroom') & (F.col('offer_completed') == 0)))

new_completion_customers_30 = new_completion_customers_30.withColumn('increase_spending', F.col('threshold_for_offer') - F.col('actual_spend'))
total_increase_spending = new_completion_customers_30.agg(F.sum('increase_spending')).collect()[0][0]
print("financial uplift for 30% stretch customers:",total_increase_spending)

# COMMAND ----------

from pyspark.sql import SparkSession

# Initialize Spark session
# spark = SparkSession.builder.appName("Append DataFrames").getOrCreate()

# Start with empty_df as None
empty_df = None

# Sample DataFrames to append
data1 = [("Alice", 34), ("Bob", 45)]
data2 = [("Charlie", 29), ("David", 50)]

# Creating DataFrames for each data sample
df1 = spark.createDataFrame(data1, ["name", "age"])
df2 = spark.createDataFrame(data2, ["name", "age"])

# Append DataFrames dynamically without defining the schema upfront
for df in [df1, df2]:
    if empty_df is None:
        # Set the first DataFrame as the base DataFrame to define schema
        empty_df = df
    else:
        # Append subsequent DataFrames
        empty_df = empty_df.unionByName(df, allowMissingColumns=True)

# Show the final result
empty_df.show()



# COMMAND ----------


