# Databricks notebook source
# MAGIC %run ../setup/bootstrap

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

# Step 3: Get seg_list: List of dictionaries
#   where each dictionary has the following properties:
#       - campaign id
#       - experian_hh_composition
#       - segmentation id (of each experian_hh_composition)

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

# Step 4: Build Dataset

config_use = config["use_segments"]

logger.info("Begin building dataset")
config_bd = config["build_dataset"]

# load factory tables
articles_df = spark.table("analytics_trans_prod.lu_article")
trx_line_df = spark.table("analytics_trans_prod.all_transaction_line")
sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
segtco_history_df = spark.table("customer_azbase_prod.segtco_history")


# COMMAND ----------


# Load Segmentation Dataset
# TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
# TODO: Possibl build data for just 1 segment at a time?
partitionByList = config_use.segmentations_tbl.partitionByList

# build training data
trx_manager = TransactionsManager(
    etl_date=get_date(config_bd["etl_date"]),
    lookback_days=config_bd["lookback_days"],
    l1_ids=config_bd["l1_ids"],
    lx=config_bd["lx"],
    lx_ids=config_bd["lx_ids"],  # getting all the products in this l2 id
    user_key=config_bd["user_id"],
    window_days=config_bd["window_days"],
    time_window_length=config_bd["time_window_days"],
    exclude_items=literal_eval(config["exclude_items"]),
    aggregation_level=config_bd["aggregation_level"],
)

# COMMAND ----------


all_data = trx_manager.get(trx_line_df, articles_df, cust_seg=segmentations_tbl)


# COMMAND ----------


etl_data_tbl_name = persist_utils.create_beam_table(
    table_prefix=config_bd.etl_data_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=all_data,
    partition_by=config_bd.etl_data_tbl.partitionByList,
    overwrite_table=False,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""etl_data_tbl_name: {etl_data_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=etl_data_tbl_name,
    insert_df=all_data,
    add_columns=True,
    insert_append=True,
    delete_where=f"campaign={campaign}",
)

# COMMAND ----------

# Step 4: Build Dataset

config_use = config["use_segments"]

logger.info("Begin building dataset")
config_bd = config["build_dataset"]

# # load factory tables
# articles_df = spark.table("analytics_trans_prod.lu_article")
# trx_line_df = spark.table("analytics_trans_prod.all_transaction_line")
# sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
# segtco_history_df = spark.table("customer_azbase_prod.segtco_history")

# Load Segmentation Dataset
# TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
# TODO: Possibl build data for just 1 segment at a time?
partitionByList = config_use.segmentations_tbl.partitionByList

segmentations_tbl_name = factory_table(
    table_prefix=config_bd.segmentations_tbl.prefix, sensitivity=config.sensitivity
)

logger.info(f"""segmentations_tbl_name: {segmentations_tbl_name}""")

segmentations_tbl = persist_utils.read_table(
    table_name=segmentations_tbl_name, where=f"campaign= {campaign}"
).select([config_bd["user_id"]] + partitionByList)


# COMMAND ----------


# build training data
trx_manager = TransactionsManager(
    etl_date=get_date(config_bd["etl_date"]),
    lookback_days=config_bd["lookback_days"],
    l1_ids=config_bd["l1_ids"],
    lx=config_bd["lx"],
    lx_ids=config_bd["lx_ids"],  # getting all the products in this l2 id
    user_key=config_bd["user_id"],
    window_days=config_bd["window_days"],
    time_window_length=config_bd["time_window_days"],
    exclude_items=literal_eval(config["exclude_items"]),
    aggregation_level=config_bd["aggregation_level"],
)

# COMMAND ----------


all_data = trx_manager.get(trx_line_df, articles_df, cust_seg=segmentations_tbl)


# COMMAND ----------


etl_data_tbl_name = persist_utils.create_beam_table(
    table_prefix=config_bd.etl_data_tbl.prefix,
    lab_database=config.lab_database,
    factory_database=config.factory_database,
    sensitivity=config.sensitivity,
    schema=all_data,
    partition_by=config_bd.etl_data_tbl.partitionByList,
    overwrite_table=False,
    assert_equality=False,
    add_load_timestamp=True,
)
logger.info(f"""etl_data_tbl_name: {etl_data_tbl_name}""")

persist_utils.insert_df_into_table(
    target_tbl_name=etl_data_tbl_name,
    insert_df=all_data,
    add_columns=True,
    insert_append=True,
    delete_where=f"campaign={campaign}",
)

# TODO: Save cust_id to account_id Mapping as done in the customer_purchase work.
# TODO: delete all mentions of validationmanager

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# Step 5: Get ordered segmentation list
seg_cnt = []
for seg in seg_list:
    config_bd = config["build_dataset"]
    seg_cnt.append(
        get_count(
            seg,
            config=config,
        )
    )

seg_cnt.sort(key=lambda i: i[1], reverse=True)

seg_list = [seg[0] for seg in seg_cnt]
logger.info(f"Ordered seg_list: {seg_list}")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="seg_list", value=seg_list)

# COMMAND ----------

# seg_list has
dbutils.notebook.exit(str({"seg_list": seg_list}))

# COMMAND ----------
