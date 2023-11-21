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

predictions.display()

# COMMAND ----------



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
                              )
'''
headroom_export = (allocation_manager.get(predictions)
                  .withColumn("campaign", F.lit(campaign))
                  ).cache()
'''

# COMMAND ----------

prediction_scores = allocation_manager.get_prediction_scores(predictions)

# COMMAND ----------

prediction_scores.display()

# COMMAND ----------

prediction_scores_tagged = allocation_manager.tag_outliers(prediction_scores)


# COMMAND ----------

prediction_scores_tagged.display()

# COMMAND ----------

headroom_predictions = allocation_manager.get_headroom(prediction_scores_tagged)


# COMMAND ----------

headroom_predictions.display()

# COMMAND ----------

data_hrm = (prediction_scores_tagged
                    .withColumn("used_headroom_frac",
                                F.when((F.col("pct_error") >= allocation_manager.max_increase) & (F.col("outlier") == 0),
                                       (1. + allocation_manager.max_increase / 100.))
                                .when((F.col("pct_error") <= allocation_manager.min_increase) & (F.col("outlier") == 0),
                                      (1. + allocation_manager.min_increase / 100.))
                                .when((F.col("pct_error") < allocation_manager.max_increase) &
                                      (F.col("pct_error") > allocation_manager.min_increase) & (F.col("outlier") == 0),
                                      1. + F.col("pct_error") / 100.)
                                .otherwise(allocation_manager.headroom_factor)
                                )
                    .withColumn("total_used_headroom_per_id", F.when(F.col(allocation_manager.feature_col) >0, 
                                                                       F.col(allocation_manager.feature_col) * F.col("used_headroom_frac")).otherwise(F.col("prediction_out") * allocation_manager.prev_not_bought_factor)
                                )
                    .groupby(allocation_manager.user_key)
                    .agg(F.sum("total_used_headroom_per_id").alias("total_used_headroom_whole_time_period"),
                    F.sum(allocation_manager.feature_col).alias("sum_total_spend_whole_period"))
                    .withColumn("sum_total_spend", F.col("sum_total_spend_whole_period") )
                    .withColumn("total_used_headroom", F.col("total_used_headroom_whole_time_period") )
                    .withColumn("rand", F.rand())
                    .withColumn("offer_id", F.lit(None))
                    
)                         

# COMMAND ----------

data_hrm.display()

# COMMAND ----------

for k, v in allocation_manager.offer_limits.items():
            offer_id = int(k)
            data_hrm = (data_hrm
                        .withColumn("offer_id", F.when((F.col("total_used_headroom") >= v[0]) &
                                                       (F.col("total_used_headroom") < v[1]), offer_id)
                                    .otherwise(F.col("offer_id"))
                                    )
                        )

# COMMAND ----------

data_hrm.display()

# COMMAND ----------

data_out = (data_hrm
                    # If very large headroom. Probably some outliers. For now random spread these offers over the top offer range.
                    .withColumn("offer_id", F.when((F.col("total_used_headroom") >= allocation_manager.large_lim),
                                                   allocation_manager.get_large_offer(F.col("rand")))
                                .otherwise(F.col("offer_id")))
                    # If offer Id is still null then an outlier. Give a random small offer.
                    .withColumn("offer_id",
                                F.when((F.col("offer_id").isNull()), allocation_manager.get_small_offer(F.col("rand")))
                                .otherwise(F.col("offer_id")))
                    .withColumn("desc", allocation_manager.get_offer_desc_part(F.col("offer_id")))
                    )

# COMMAND ----------

data_out.display()

# COMMAND ----------

headroom_export = allocation_manager.prepare_export(headroom_predictions)

# COMMAND ----------

headroom_export.display()

# COMMAND ----------

headroom_export.groupby('desc').count().display()

# COMMAND ----------


