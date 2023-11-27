# Databricks notebook source
# MAGIC %run ../notebooks/bootstrap

# COMMAND ----------

import os
from functools import partial
import pandas as pd
from datetime import datetime
from customer_headroom.etl.build_dataset import TransactionsManager
from customer_headroom.etl.segmentation import (
    SegmentationDataManager,
    SegmentationManager,
)
from customer_headroom.modelling.data_process import DataProcessor
from customer_headroom.modelling.fit import build_recommender
from customer_headroom.modelling.predict import Predictor
from customer_headroom.evaluation.model_selection import Evaluator
from customer_headroom.allocation.allocator import Allocator
import offerallocationv2.utils.persist_utils as persist_utils
from dtaml.logging import get_logger
from cdsutils.io_utils import file_exists, save_object, load_object
from multiprocessing.pool import ThreadPool
import seaborn as sns
from datetime import datetime, timedelta
from pyspark.sql import functions as F, DataFrame, Column, Window as W, types as T
from ast import literal_eval

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

def find_all_segments(data, partitionByList):
    segs = (
        data.select(partitionByList)
        .distinct()
        .rdd.map(lambda x: {k: v for (k, v) in zip(partitionByList, x)})
        .collect()
    )
    return segs


def get_date(date):
    if str(date).lower() == "today":
        date = datetime.now().strftime("%Y%m%d")
    return int(date)


def get_campaign(campaign, etl_date):
    if (campaign == "{campaign}") or (campaign == ""):
        campaign = get_date(etl_date)
    return campaign


config_dates = config["dates"]
campaign = get_campaign(config_dates["upcoming_campaign"], config_dates["etl_date"])

date_format = "%Y%m%d"
last_registration_date = int(
    (
        datetime.strptime(str(campaign), date_format)
        - timedelta(days=config_dates["lookback_days_registration"])
    ).strftime(date_format)
)

# campaign = 20230807

logger.info(
    f"""
config_dates: {config_dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
)

# COMMAND ----------

#Get segmentations to use

logger.info("Begin building dataset")
config_bd = config["build_dataset"]
config_use = config["use_segments"]

# load factory tables
articles_df = spark.table("analytics_trans_prod.lu_article")
trx_line_df = spark.table("analytics_trans_prod.all_transaction_line")
sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
segtco_history_df = spark.table("customer_azbase_prod.segtco_history")

# Load Segmentation Dataset
# TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
# TODO: Possibl build data for just 1 segment at a time?
partitionByList = config_use.segmentations_tbl.partitionByList

segmentations_tbl_name = persist_utils.get_table_name(
    table_prefix=config_bd.segmentations_tbl.prefix,
    lab_database=config.dev_database,
    factory_database=config_bd.segmentations_tbl.factory_database,
    sensitivity=config_bd.segmentations_tbl.sensitivity,
)
logger.info(f"""segmentations_tbl_name: {segmentations_tbl_name}""")

segmentations_tbl = persist_utils.read_table(
    table_name=segmentations_tbl_name, where=f"campaign= {campaign}"
).select([config_bd["user_id"]] + partitionByList)

# COMMAND ----------

#Initialize trx_manager class
trx_manager = TransactionsManager(
  etl_date=get_date(config_bd["etl_date"]),
  lookback_days=config_bd["lookback_days"],
  l1_ids=config_bd["l1_ids"],
  lx=config_bd["lx"],
  lx_ids=config_bd["lx_ids"],  # getting all the products in this l2 id
  user_key=config_bd["user_id"],
  window_days=config_bd["window_days"],
  time_window_length=config_bd["time_window_days"],
  exclude_items = literal_eval(config["exclude_items"]), 
)

# COMMAND ----------

all_data = trx_manager.get(trx_line_df, articles_df, cust_seg=segmentations_tbl)

# COMMAND ----------

all_data.write.parquet("/mnt/centralds/offerallocation/headroom/analysis/231121/l3_build_data", mode = "overwrite")

# COMMAND ----------

l3_build_data = spark.read.parquet("/mnt/centralds/offerallocation/headroom/analysis/231121/l3_build_data", mode = "overwrite")

# COMMAND ----------

l3_build_data.display()

# COMMAND ----------

#distinct l3 ids bought in

l3_ids_bought_in = l3_build_data.groupBy('cust_id').agg(F.countDistinct('l3_id').alias('l3s_bought_in')).select('l3s_bought_in').toPandas()

import plotly.express as px
fig = px.box(l3_ids_bought_in, y="l3s_bought_in")
fig.show()

# COMMAND ----------

#l2/l3

articles = spark.read.table('analytics_trans_prod.lu_article')
df = l3_build_data.join(articles, on = 'l3_id', how = 'inner').distinct().select('cust_id','l2_id','l3_id')


# COMMAND ----------

l3_build_data.display()

# COMMAND ----------

a = l3_build_data.filter(F.col('cust_id')==-1000219806380299450)

# COMMAND ----------

a.display()

# COMMAND ----------

a.join(articles, on = 'l3_id', how = 'left').select('cust_id','l2_id','l3_id').distinct().display()

# COMMAND ----------


