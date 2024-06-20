# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

# TODO: make segmentation into a job parameter
segmentation = widgets.get("segmentation", "True")

# COMMAND ----------

debug = True

# COMMAND ----------

from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.etl.etl_utils import get_date, get_campaign, write_beam_table
from customer_headroom.etl.segmentation import (
    SegmentationDataManager,
    SegmentationManager,
)



sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------


config_dates = config["dates"]
campaign = get_campaign(config_dates["upcoming_campaign"], config_dates["etl_date"])

date_format = "%Y%m%d"
last_registration_date = int(
    (
        datetime.strptime(str(campaign), date_format)
        - timedelta(days=config_dates["lookback_days_registration"])
    ).strftime(date_format)
)

logger.info(
    f"""
config_dates: {config_dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
)

# COMMAND ----------

# MAGIC %md # Build Segmentations

# COMMAND ----------

# Step 1: Build Segmentation Dataset

logger.info("Begin Building Segmentation Dataset")
config_sg = config["segmentation"]

# load factory tables
# Registered customer information
sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
# All customer attributes (including Experian)
cust_master_df = spark.sql("select * from analytics_trans_prod.customer_master")
# All transactions per customer at line level (with product details)
trx_line_df = spark.table("analytics_trans_prod.all_transaction_line")


# COMMAND ----------


# selecting customer who have registered for a period of time,
#   so that there is some spending data when calculating percentiles.
sparks_account_df = sparks_account_df.filter(
    F.col("registration_date")
    <= datetime.strptime(str(last_registration_date), "%Y%m%d")
)


# COMMAND ----------


# Manager for Segmentation Data
seg_data_manager = SegmentationDataManager(
    etl_date=get_date(config_sg["etl_date"]),
    lookback_days=config_sg["lookback_days"],
    l1_id=config_sg["l1_id"],
    user_id=config_sg["user_id"],
)



# COMMAND ----------

trx_line_df

# COMMAND ----------

sparks_account_df

# COMMAND ----------

cust_master_df

# COMMAND ----------


# Step 1: build the data for segmentation
seg_data = seg_data_manager.get(
    trx_line_df, sparks_account_df, cust_master_df, customer_input=sparks_account_df
)
# adding campaign column
seg_data = seg_data.withColumn("campaign", F.lit(campaign))


# COMMAND ----------

seg_data

# COMMAND ----------

seg_data_table_name = write_beam_table(
  seg_data,
  config,
  "seg_data_tbl",
)

# COMMAND ----------

seg_data_table_name

# COMMAND ----------


logger.info("Begin Segmentation of Dataset")
seg_data_read = persist_utils.read_table(
    table_name=seg_data_table_name, where=f"campaign={campaign}"
)

# COMMAND ----------

# Step 2: Segment the data
seg_manager = SegmentationManager(
    num_cols=config_sg["num_cols"],
    cat_cols=config_sg["cat_cols"],
    frac_lim=config_sg["frac_lim"],
    fail_limit=config_sg["fail_limit"],
    k_search_min=config_sg["k_search_min"],
    k_search_max=config_sg["k_search_max"],
    pca_k=config_sg["pca_k"],
    data_lower_lim=config_sg["data_lower_lim"],
    user_id=config_sg["user_id"],
    verbose=config_sg["verbose"],
)


# COMMAND ----------


segmentations = seg_manager.get(data=seg_data_read).withColumn(
    "campaign", F.lit(campaign)
)

# COMMAND ----------

write_beam_table(
  segmentations,
  config,
  "segmentations_tbl",
)

# COMMAND ----------


