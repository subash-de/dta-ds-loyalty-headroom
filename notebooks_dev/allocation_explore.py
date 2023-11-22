# Databricks notebook source
# MAGIC %run ../notebooks/bootstrap

# COMMAND ----------



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

from offerallocationv2.utils import tmo_utils


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

logger.info("Begin Allocation")
config_al = config["allocation"]
under_predict_adjustment_factor = config_al["headroom_factor"]


prediction_tbl_name = persist_utils.get_table_name(factory_database=config_al.prediction_tbl.factory_database,
                                                    lab_database=config.dev_database,
                                                    table_prefix=config_al.prediction_tbl.prefix,
                                                    sensitivity=config_al.prediction_tbl.sensitivity)
logger.info(f"""prediction_tbl_name: {prediction_tbl_name}""")

predictions = persist_utils.read_table(table_name=prediction_tbl_name, where=f"campaign={campaign}")




# COMMAND ----------

# if its under predicting, then would force the stretch to be 20% 
predictions = (predictions
                .withColumn("prediction_out_orig", F.lit(F.col("prediction_out")))
                .withColumn("prediction_out", F.when(F.col("prediction_out_orig") < F.col(config_al['feature_col']), F.col(config_al['feature_col'])*under_predict_adjustment_factor).otherwise(F.col("prediction_out_orig")))
                )

# COMMAND ----------

logger.info("Allocating all customers")
allocation_manager = Allocator(feature_col=config_al["feature_col"],
                              offer_limits=config["offer_limits"],
                              offer_desc=config["offers_desc"],
                              user_key=config_al["user_key"],
                              outlier_min=config_al["outlier_min"],
                              outlier_max=config_al["outlier_max"],
                              max_increase=config_al["max_increase"],
                              min_increase=config_al["min_increase"],
                              headroom_factor=config_al["headroom_factor"],
                              fill_offer=config_al["fill_offer"],
                              prev_not_bought_factor = config_al["prev_not_bought_factor"],
                              prev_not_bought_factor_l2_id_indpendent = config_al["prev_not_bought_factor_l2_id_indpendent"],
                              aggregate_level = config_al["aggregate_level"],
                              )

headroom_export = (allocation_manager.get(predictions)
                  .withColumn("campaign", F.lit(campaign))
                  ).cache()


# COMMAND ----------

headroom_export.display()

# COMMAND ----------

logger.info("Allocating all customers")
allocation_manager_l2 = Allocator(feature_col=config_al["feature_col"],
                              offer_limits=config["offer_limits"],
                              offer_desc=config["offers_desc"],
                              user_key=config_al["user_key"],
                              outlier_min=config_al["outlier_min"],
                              outlier_max=config_al["outlier_max"],
                              max_increase=config_al["max_increase"],
                              min_increase=config_al["min_increase"],
                              headroom_factor=config_al["headroom_factor"],
                              fill_offer=config_al["fill_offer"],
                              prev_not_bought_factor = config_al["prev_not_bought_factor"],
                              prev_not_bought_factor_l2_id_indpendent = config_al["prev_not_bought_factor_l2_id_indpendent"],
                              aggregate_level = 'l2',
                              )

headroom_export_l2 = (allocation_manager_l2.get(predictions)
                  .withColumn("campaign", F.lit(campaign))
                  ).cache()

# COMMAND ----------

headroom_export_l2.display()

# COMMAND ----------



# COMMAND ----------

df = headroom_export_l2.groupBy('cust_id').agg(F.sum('spend_plus_headroom').alias('sum_spend_plus_headroom'))
join_dfs = headroom_export.join(df, on='cust_id', how='inner').select('cust_id','spend_plus_headroom','sum_spend_plus_headroom')
check = join_dfs.withColumn('diff', F.col('sum_spend_plus_headroom')-F.col('spend_plus_headroom'))

# COMMAND ----------

check.display()

# COMMAND ----------

check.filter(F.col('diff')<-1).display()

# COMMAND ----------


