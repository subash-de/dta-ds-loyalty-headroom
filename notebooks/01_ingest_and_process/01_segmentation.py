# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

segmentation = widgets.get("segmentation", "True")

# COMMAND ----------

segmentation

# COMMAND ----------

from ast import literal_eval
from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.etl.etl_utils import *
from customer_headroom.etl.build_dataset import TransactionsManager
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
# TODO: Must be a Job parameter
if "segmentation" in config.steps:
    logger.info("Begin Building Segmentation Dataset")
    config_sg = config["segmentation"]

    # load factory tables
    # Registered customer information
    sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
    # All customer attributes (including Experian)
    cust_master_df = spark.sql("select * from analytics_trans_prod.customer_master")
    # All transactions per customer at line level (with product details)
    trx_line_df = spark.table("analytics_trans_prod.all_transaction_line")

    # selecting customer who have registered for a period of time,
    #   so that there is some spending data when calculating percentiles.
    sparks_account_df = sparks_account_df.filter(
        F.col("registration_date")
        <= datetime.strptime(str(last_registration_date), "%Y%m%d")
    )

    # Manager for Segmentation Data
    seg_data_manager = SegmentationDataManager(
        etl_date=get_date(config_sg["etl_date"]),
        lookback_days=config_sg["lookback_days"],
        l1_id=config_sg["l1_id"],
        user_id=config_sg["user_id"],
    )

    # Step 1: build the data for segmentation
    # These should be two steps
    seg_data = seg_data_manager.get(
        trx_line_df, sparks_account_df, cust_master_df, customer_input=sparks_account_df
    ).withColumn("campaign", F.lit(campaign))

    # TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
    # Creating the segmented dataset
    seg_data_table_name = persist_utils.create_beam_table(
        table_prefix=config_sg.seg_data_tbl.prefix,
        lab_database=config.dev_database,
        factory_database=config_sg.seg_data_tbl.factory_database,
        sensitivity=config_sg.seg_data_tbl.sensitivity,
        schema=seg_data,
        partition_by=config_sg.seg_data_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )

    logger.info(f"""seg_data_table_name: {seg_data_table_name}""")

    # Persist the segmented data into created dataset
    persist_utils.insert_df_into_table(
        target_tbl_name=seg_data_table_name,
        insert_df=seg_data,
        delete_where=f"campaign={campaign}",
    )

    # Step 2: Segment the data
    logger.info("Begin Segmentation of Dataset")
    seg_data_read = persist_utils.read_table(
        table_name=seg_data_table_name, where=f"campaign={campaign}"
    )
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

    segmentations = seg_manager.get(data=seg_data_read).withColumn(
        "campaign", F.lit(campaign)
    )
    segmentations_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_sg.segmentations_tbl.prefix,
        lab_database=config.dev_database,
        factory_database=config_sg.segmentations_tbl.factory_database,
        sensitivity=config_sg.segmentations_tbl.sensitivity,
        schema=segmentations,
        partition_by=config_sg.segmentations_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )
    logger.info(f"""segmentations_tbl_name: {segmentations_tbl_name}""")

    persist_utils.insert_df_into_table(
        target_tbl_name=segmentations_tbl_name,
        insert_df=segmentations,
        delete_where=f"campaign={campaign}",
    )
