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
df = l3_build_data.join(articles, on = 'l3_id', how = 'inner').select('cust_id','l2_id','l3_id').distinct()
df_ = df.groupBy('cust_id','l2_id').agg(F.countDistinct('l3_id').alias('l3s_bought_in_within_l2'))
df_pd = df_.toPandas()

# COMMAND ----------

df_.display()

# COMMAND ----------

#predict breakdown

# COMMAND ----------

#predict run

# COMMAND ----------

if "predict" in config.steps:
    config_pd = config["predict"]
    prediction_tbl_name = persist_utils.get_table_name(
        factory_database=config_pd.prediction_tbl.factory_database,
        lab_database=config.dev_database,
        table_prefix=config_pd.prediction_tbl.prefix,
        sensitivity=config_pd.prediction_tbl.sensitivity,
    )

    logger.info(f"""prediction_tbl_name: {prediction_tbl_name}""")

    prediction_tbl = persist_utils.read_table(
        table_name=prediction_tbl_name, where=f"campaign={campaign}"
    )
    display(prediction_tbl.orderBy(F.rand()))

# COMMAND ----------

logger.info("Begin Allocation")
config_al = config["allocation"]
under_predict_adjustment_factor = config_al["headroom_factor"]

predictions = prediction_tbl

# if its under predicting, then would force the stretch to be 20% 
predictions = (predictions
                .withColumn("prediction_out_orig", F.lit(F.col("prediction_out")))
                .withColumn("prediction_out", F.when(F.col("prediction_out_orig") < F.col(config_al['feature_col']), F.col(config_al['feature_col'])*under_predict_adjustment_factor).otherwise(F.col("prediction_out_orig")))
                )

allocation_manager = Allocator(feature_col=config_al["feature_col"],
                              offer_limits=config["offer_limits"],
                              offer_desc=config["offers_desc"],
                              user_key=config_al["user_key"],
                              lx_key = config_al["lx_key"],
                              outlier_min=config_al["outlier_min"],
                              outlier_max=config_al["outlier_max"],
                              max_increase=config_al["max_increase"],
                              min_increase=config_al["min_increase"],
                              headroom_factor=config_al["headroom_factor"],
                              fill_offer=config_al["fill_offer"],
                              prev_not_bought_factor = config_al["prev_not_bought_factor"],
                              prev_not_bought_factor_l2_id_indpendent = config_al["prev_not_bought_factor_l2_id_indpendent"],
                              aggregate_level = 0,
                              )

headroom_export = (allocation_manager.get(predictions)
                        .withColumn("campaign", F.lit(campaign))
                        ).cache() 


if config_al["aggregate_level"] is None:
      if config['exclude_high_spend'] is not None:
        logger.info(f"Remove customer whos spend_plus_headroom > {config['exclude_high_spend']}")
        headroom_export = (
          headroom_export
          .filter(F.col("spend_plus_headroom") <= config['exclude_high_spend'])
        )

      if config['min_num_basket'] is not None: 
        logger.info(f"Remove customer who have less than {config['min_num_basket']} basket")
        headroom_export = (
          headroom_export
          .join(predictions
                .filter(F.col("count_user_basket") >= config['min_num_basket'] )
                .select("cust_id")
                .distinct(), how = 'inner', on = 'cust_id')
          )

                              

# COMMAND ----------

headroom_export.display()

# COMMAND ----------


