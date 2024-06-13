# Databricks notebook source
# MAGIC %md # Allocation simulation

# COMMAND ----------

# MAGIC %run ../notebooks/bootstrap

# COMMAND ----------

# MAGIC %md The headroom calculated depends on the customer have bought or not bought into the l2_id in the baseline basket.
# MAGIC - Headroom low - if we only consider customer who have bought into the item
# MAGIC - headroom high - if all the l2 is added up (which include the ones that customer havent bought)
# MAGIC
# MAGIC
# MAGIC Goal of this notebook is to vary the percentage of previousely bought into factor to see what the headroom looks like
# MAGIC - calculate

# COMMAND ----------

import os
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.pool import ThreadPool

import offerallocationv2.utils.persist_utils as persist_utils
import pandas as pd
import seaborn as sns
from cdsutils.io_utils import file_exists, load_object, save_object
from dtaml.logging import get_logger
from pyspark.sql import Column, DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F
from pyspark.sql import types as T

from customer_headroom.allocation.allocator import Allocator
from customer_headroom.etl.build_dataset import TransactionsManager
from customer_headroom.etl.segmentation import (SegmentationDataManager,
                                                SegmentationManager)
from customer_headroom.evaluation.model_selection import Evaluator
from customer_headroom.modelling.data_process import DataProcessor
from customer_headroom.modelling.fit import build_recommender
from customer_headroom.modelling.predict import Predictor

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

campaign = 20230525

logger.info(
    f"""
config_dates: {config_dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
)

# COMMAND ----------

if "allocate" in config.steps:
    logger.info("Begin Allocation")
    config_al = config["allocation"]

    # prediction_tbl_name = persist_utils.get_table_name(factory_database=config_al.prediction_tbl.factory_database,
    #                                                    lab_database=config.dev_database,
    #                                                    table_prefix=config_al.prediction_tbl.prefix,
    #                                                    sensitivity=config_al.prediction_tbl.sensitivity)
    # logger.info(f"""prediction_tbl_name: {prediction_tbl_name}""")
    prediction_tbl_name = "loyalty_azlab_prod.predictions_230522_p_tbl"

    predictions = persist_utils.read_table(
        table_name=prediction_tbl_name, where=f"campaign={campaign}"
    )

    # if its under predicting, then would force the stretch to be 20%
    # replacing the feature col value with pred_out when its 0
    predictions = (
        predictions.withColumn(
            "prediction_out_orig", F.col("prediction_out")
        ).withColumn(
            "prediction_out",
            F.when(
                F.col("prediction_out_orig") < F.col("l2_id_total_spend_basket"),
                F.col("l2_id_total_spend_basket") * 1.2,
            ).otherwise(F.col("prediction_out_orig")),
        )
        # .withColumn("l2_id_total_spend_basket_org", F.col("l2_id_total_spend_basket"))
        # .withColumn("l2_id_total_spend_basket", F.when(F.col("l2_id_total_spend_basket_org")==0, F.col("prediction_out"))
        #             .otherwise(F.col("l2_id_total_spend_basket_org")))
    )

    # allocation_manager = Allocator(feature_col=config_al["feature_col"],
    #                                offer_limits=config_al["offer_limits"],
    #                                offer_desc=config_al["offers_desc"],
    #                                user_key=config_al["user_key"],
    #                                outlier_min=config_al["outlier_min"],
    #                                outlier_max=config_al["outlier_max"],
    #                                max_increase=config_al["max_increase"],
    #                                min_increase=config_al["min_increase"],
    #                                headroom_factor=config_al["headroom_factor"],
    #                                fill_offer=config_al["fill_offer"],
    #                                prev_not_bought_factor = 0, # config_al["prev_not_bought_factor"]
    #                                )

    # headroom_export = (allocation_manager.get(predictions)
    #                    .withColumn("campaign", F.lit(campaign))
    #                    )

    # # headroom_tbl_name = persist_utils.create_beam_table(table_prefix=config_al.headroom_export_tbl.prefix,
    # #                                                     lab_database=config.dev_database,
    # #                                                     factory_database=config_al.headroom_export_tbl.factory_database,
    # #                                                     sensitivity=config_al.headroom_export_tbl.sensitivity,
    # #                                                     schema=headroom_export,
    # #                                                     partition_by=config_al.headroom_export_tbl.partitionByList,
    # #                                                     overwrite_table=True,
    # #                                                     assert_equality=False,
    # #                                                     add_load_timestamp=True
    # #                                                     )
    # # logger.info(f"""headroom_tbl_name: {headroom_tbl_name}""")

    # # persist_utils.insert_df_into_table(target_tbl_name=headroom_tbl_name,
    # #                                    insert_df=headroom_export,
    # #                                    delete_where=f"campaign={campaign}")

# COMMAND ----------


# COMMAND ----------

customer = (predictions.select("cust_id").distinct()).cache()

# COMMAND ----------

customer.count()

# COMMAND ----------

headroom_list = []
for factor in [0, 0.25, 0.5, 0.75, 1]:
    print(f"Allocating with factor : {factor}")
    allocation_manager = Allocator(
        feature_col=config_al["feature_col"],
        offer_limits=config_al["offer_limits"],
        offer_desc=config_al["offers_desc"],
        user_key=config_al["user_key"],
        outlier_min=config_al["outlier_min"],
        outlier_max=config_al["outlier_max"],
        max_increase=config_al["max_increase"],
        min_increase=config_al["min_increase"],
        headroom_factor=config_al["headroom_factor"],
        fill_offer=config_al["fill_offer"],
        prev_not_bought_factor=factor,  # config_al["prev_not_bought_factor"]
    )
    headroom_export = allocation_manager.get(predictions).withColumn(
        "factor", F.lit(f"{factor}")
    )
    headroom_list.append(headroom_export)
    # customer = customer.join(
    #   headroom_export.select(
    #   "cust_id",
    #   F.col("estimated_spend").alias(f"estimated_spend_factor_{factor}"),
    #   F.col("estimated_headroom").alias(f"estimated_headroom_factor_{factor}"),
    #   F.col("spend_plus_headroom").alias(f"spend_plus_headroom_factor_{factor}"),
    #   F.col("desc").alias(f"desc_factor_{factor}"),
    #   ),
    #   how = 'left'
    #   on = 'cust_id'
    # )

# COMMAND ----------

from functools import reduce  # For Python 3.x

from pyspark.sql import DataFrame


def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)


# COMMAND ----------

headroom_compare = unionAll(*headroom_list)

headroom_compare.display()

# COMMAND ----------

headroom_compare.groupBy("desc", "factor").count().withColumn(
    "percentage",
    F.round(F.col("count") / F.sum("count").over(W.partitionBy("factor")), 3),
).display()

# COMMAND ----------

headroom_compare

# COMMAND ----------


# COMMAND ----------


# COMMAND ----------

allocation_manager = Allocator(
    feature_col=config_al["feature_col"],
    offer_limits=config_al["offer_limits"],
    offer_desc=config_al["offers_desc"],
    user_key=config_al["user_key"],
    outlier_min=config_al["outlier_min"],
    outlier_max=config_al["outlier_max"],
    max_increase=config_al["max_increase"],
    min_increase=config_al["min_increase"],
    headroom_factor=config_al["headroom_factor"],
    fill_offer=config_al["fill_offer"],
    prev_not_bought_factor=0,  # config_al["prev_not_bought_factor"]
)

headroom_export = allocation_manager.get(predictions).withColumn(
    "campaign", F.lit(campaign)
)

# COMMAND ----------

headroom_export.display()

# COMMAND ----------

factor = 0
headroom_export.select(
    "cust_id",
    F.col("estimated_spend").alias(f"estimated_spend_factor_{factor}"),
    F.col("estimated_headroom").alias(f"estimated_headroom_factor_{factor}"),
    F.col("spend_plus_headroom").alias(f"spend_plus_headroom_factor_{factor}"),
    F.col("desc").alias(f"desc_factor_{factor}"),
).display()

# COMMAND ----------
