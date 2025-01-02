# Databricks notebook source
# MAGIC %run ../bootstrap

# COMMAND ----------

from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.etl.etl_utils import get_campaign, get_date, write_beam_table
from customer_headroom.etl.segmentation import (
    SegmentationDataManager,
    SegmentationManager,
)

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

debug = False
if debug:
    config.dates.etl_date = config.debug.tables.etl_date
    config.dates.lookback_days = config.debug.tables.lookback_days


campaign = get_campaign(config.dates.upcoming_campaign, config.dates.etl_date)


last_registration_date = int(
    (
        datetime.strptime(str(campaign), config.dates.date_format)
        - timedelta(days=config.dates.lookback_days_registration)
    ).strftime(config.dates.date_format)
)

logger.info(
    f"""
config.dates: {config.dates}
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
sparks_account_df = persist_utils.read_table(
  table_name = config.factory_tbl_sparks_account
)
# All customer attributes (including Experian)
cust_master_df = persist_utils.read_table(
  table_name = config.factory_tbl_customer_master
)

# All transactions per customer at line level (with product details)
trx_line_df = persist_utils.read_table(
  table_name = config.factory_tbl_all_transaction_line
)


# COMMAND ----------


# selecting customer who have registered for a period of time,
#   so that there is some spending data when calculating percentiles.
sparks_account_df = sparks_account_df.filter(
    F.col("registration_date")
    <= datetime.strptime(str(last_registration_date), "%Y%m%d")
)


# COMMAND ----------

# //TODO #23 Can we pass ETL_Date as a parameter of the job itself?
# Manager for Segmentation Data
seg_data_manager = SegmentationDataManager(
    etl_date=get_date(config.dates.etl_date),
    lookback_days=config.dates.lookback_days,
    l1_id=config_sg["l1_id"],
    user_id=config_sg["user_id"],
)


# COMMAND ----------

# Step 1: build the data for segmentation
seg_data = seg_data_manager.get(
    trx_line_df, sparks_account_df, cust_master_df, customer_input=sparks_account_df
)
# adding campaign column
seg_data = seg_data.withColumn("campaign", F.lit(campaign))
seg_data = seg_data.withColumn("l1_id", F.lit(config_sg['l1_id']))
seg_data = seg_data.withColumn("category_level", F.lit(config['category_level']))


# COMMAND ----------

seg_data.display()

# COMMAND ----------

seg_data.count()

# COMMAND ----------

seg_data.select('campaign','l1_id', 'category_level').distinct().display()

# COMMAND ----------

seg_data_table_name = persist_utils.create_beam_table(
        table_prefix=config_sg.seg_data_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_database,
        sensitivity=config.sensitivity,
        schema=seg_data,
        partition_by=config_sg.seg_data_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )

logger.info(f"""seg_data_table_name: {seg_data_table_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=seg_data_table_name,
    insert_df=seg_data,
    add_columns=True,
    insert_append=True,
    delete_where=f"campaign={campaign} and l1_id ='{config_sg['l1_id']}' and category_level={config['category_level']}",
)


# COMMAND ----------

logger.info("Begin Segmentation of Dataset")
seg_data_read = persist_utils.read_table(
    table_name=seg_data_table_name, 
    where=f"campaign={campaign} and l1_id ='{config_sg['l1_id']}' and category_level={config['category_level']}",
)

# COMMAND ----------

seg_data_read.select('campaign', 'l1_id','category_level').distinct().display()

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
segmentations = segmentations.withColumn("l1_id", F.lit(config_sg['l1_id']))
segmentations = segmentations.withColumn("category_level", F.lit(config['category_level']))

# COMMAND ----------

segmentations_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_sg.segmentations_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_database,
        sensitivity=config.sensitivity,
        schema=segmentations,
        partition_by=config_sg.segmentations_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )

logger.info(f"segmentations_tbl_name: {segmentations_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=segmentations_tbl_name,
    insert_df=segmentations,
    add_columns=True,
    insert_append=False,
    delete_where=f"campaign={campaign} and l1_id ='{config_sg['l1_id']}' and category_level={config['category_level']}",
)

# COMMAND ----------


