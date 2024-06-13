# Databricks notebook source
# MAGIC %run ./bootstrap $environment=prod

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

campaign = 20230421
date_format = "%Y%m%d"
int((datetime.strptime(str(campaign), date_format) -
                                  timedelta(days=180)).strftime(date_format))

# COMMAND ----------

def days_between(d1, d2, date_format="%Y%m%d"):
    d1 = datetime.strptime(str(d1), date_format)
    d2 = datetime.strptime(str(d2), date_format)
    return abs((d2 - d1).days)

prediction_time_range = [20221023, 20230421]
prediction_time_span = days_between(prediction_time_range[1], prediction_time_range[0]) - 14
prediction_week = prediction_time_span/7.
prediction_month = prediction_time_span/30.

# COMMAND ----------

prediction_period = prediction_week / 2 # want bi-weekly
prediction_period

# COMMAND ----------

class Allocator2(Allocator):

  def get_prediction_scores(self, predictions):
      pred_scores = (predictions
                      .withColumn("pct_error",
                                  100. * (F.col("prediction_out") - F.col(self.feature_col)) / (
                                      F.col(self.feature_col)))
                      # .groupby(self.user_key, "experian_hh_composition", "segmentation")
                      # .agg(F.mean(self.feature_col).alias("mean_input"),
                      #     F.sum(self.feature_col).alias("sum_input"),
                      #     F.sum("prediction_out").alias("sum_prediction"),
                      #     F.mean("pct_error").alias("mean_pct_error"),
                      #     (F.sum(F.col(self.feature_col) * F.col("visits")) / F.sum(
                      #         F.col("visits"))).alias("weightedmean_input"),
                      #     (F.sum(F.col("pct_error") * F.col("visits")) / F.sum(F.col("visits"))).alias(
                      #         "weightedmean_pct_error")
                      #     )
                      .withColumn("offer_id", F.lit(None))
                      )
      return pred_scores

  def tag_outliers(self, data):
      data_tagged = (data
                      .withColumn("outlier", F.when(((F.col("pct_error") >= self.outlier_min) &
                                                    (F.col("pct_error") <= self.outlier_max)
                                                    ), 0).otherwise(1))
                      )
      return data_tagged


  def get_headroom(self, data):

    data_hrm = (data
                .withColumn("used_headroom_frac",
                            F.when((F.col("pct_error") >= self.max_increase) & (F.col("outlier") == 0),
                                    (1. + self.max_increase / 100.))
                            .when((F.col("pct_error") <= self.min_increase) & (F.col("outlier") == 0),
                                  (1. + self.min_increase / 100.))
                            .when((F.col("pct_error") < self.max_increase) &
                                  (F.col("pct_error") > self.min_increase) & (F.col("outlier") == 0),
                                  1. + F.col("pct_error") / 100.)
                            .otherwise(self.headroom_factor)
                            )
                .withColumn("total_used_headroom_per_id",
                            F.col(self.feature_col) * F.col("used_headroom_frac"))
                .groupby(self.user_key, "experian_hh_composition", "segmentation")
                .agg(F.sum("total_used_headroom_per_id").alias("total_used_headroom_whole_time_period"),
                F.sum(self.feature_col).alias("sum_total_spend_whole_period"))
                .withColumn("sum_total_spend", F.col("sum_total_spend_whole_period") / prediction_period)
                .withColumn("total_used_headroom", F.col("total_used_headroom_whole_time_period") / prediction_period)
                .withColumn("rand", F.rand())
                .withColumn("offer_id", F.lit(None))
                )
    for k, v in self.offer_limits.items():
        offer_id = int(k)
        data_hrm = (data_hrm
                    .withColumn("offer_id", F.when((F.col("total_used_headroom") >= v[0]) &
                                                    (F.col("total_used_headroom") < v[1]), offer_id)
                                .otherwise(F.col("offer_id"))
                                )
                    )

    data_out = (data_hrm
                # If very large headroom. Probably some outliers. For now random spread these offers over the top offer range.
                .withColumn("offer_id", F.when((F.col("total_used_headroom") >= self.large_lim),
                                                self.get_large_offer(F.col("rand")))
                            .otherwise(F.col("offer_id")))
                # If offer Id is still null then an outlier. Give a random small offer.
                .withColumn("offer_id",
                            F.when((F.col("offer_id").isNull()), self.get_small_offer(F.col("rand")))
                            .otherwise(F.col("offer_id")))
                .withColumn("desc", self.get_offer_desc_part(F.col("offer_id")))
                )

    return data_out

  def prepare_export(self, data):
      data_export = (data
                      .withColumn("offer_id", F.when(F.col("offer_id").isNull(), F.lit(self.fill_offer))
                                  .otherwise(F.col("offer_id"))
                                  )
                      .withColumn("spend_plus_headroom", F.round("total_used_headroom", 2))
                      .withColumn("estimated_spend", F.round(F.col("sum_total_spend"), 2))
                      .withColumn("estimated_headroom",
                                  F.round(F.col("spend_plus_headroom") - F.col("sum_total_spend"),
                                          2))
                      .select(self.user_key, "offer_id", "estimated_spend", "estimated_headroom",
                              "spend_plus_headroom", "desc")
                      .dropDuplicates(subset=[self.user_key])
                      )
      return data_export

# COMMAND ----------

config_use = config["use_segments"]
persist_utils.get_table_name(
            factory_database=config_use.segmentations_tbl.factory_database,
            lab_database=config.dev_database,
            table_prefix=config_use.segmentations_tbl.prefix,
            sensitivity=config_use.segmentations_tbl.sensitivity)

# COMMAND ----------

def find_all_segments(data, partitionByList):
    segs = data.select(partitionByList).distinct().rdd.map(
        lambda x: {k: v for (k, v) in zip(partitionByList, x)}).collect()
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
print(f"""
config_dates: {config_dates}
campaign: {campaign}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Segmentations

# COMMAND ----------

# %sql select * from analytics_trans_prod.sparks_account

# COMMAND ----------

# campaign = 20230421
# date_format = "%Y%m%d"
# int((datetime.strptime(str(campaign), date_format) -
#                                   timedelta(days=180)).strftime(date_format))

# COMMAND ----------

# datetime.strptime(str(campaign), date_format)

# COMMAND ----------

# sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
# # display(sparks_account_df))
# # sparks_account_df.select(F.min("registration_date")).display()
# sparks_account_df = sparks_account_df.filter(F.col("registration_date") >= datetime.strptime(str("20221023"), date_format))
# display(sparks_account_df)
# # sparks_account_df.select(F.min("registration_date")).display()

# COMMAND ----------

# sparks_account_df.select(F.min("registration_date")).display()

# COMMAND ----------

# sparks_account_df

# COMMAND ----------

if "segmentation" in config.steps:
    logger.info("Begin Building Segmentation Dataset")
    config_sg = config["segmentation"]
    # cust_path = create_path_campaign(config_sg["cust_path"])

    # load factory tables
    sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
    cust_master_df = spark.sql("select * from analytics_trans_prod.customer_master")
    trx_line_df = spark.table("analytics_trans_prod.all_transaction_line")

    # load customer table
    # TODO: Replace with input customer id's if required.
    # if cust_path is None:
    #     custs_etl_data = None
    # else:
    #     custs_etl_data = spark.read.parquet(cust_path)
    sparks_account_df = sparks_account_df.filter(F.col("registration_date") <= datetime.strptime(str("20221023"), "%Y%m%d"))

    custs_etl_data = sparks_account_df

    seg_data_manager = SegmentationDataManager(etl_date=get_date(config_sg["etl_date"]),
                                               lookback_days=config_sg["lookback_days"],
                                               l1_id=config_sg["l1_id"],
                                               user_id=config_sg["user_id"])

    seg_data = (seg_data_manager.get(trx_line_df, sparks_account_df, cust_master_df, customer_input=custs_etl_data)
                .withColumn("campaign", F.lit(campaign))
                )

    # TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
    seg_data_table_name = persist_utils.create_beam_table(table_prefix=config_sg.seg_data_tbl.prefix,
                                                          lab_database=config.dev_database,
                                                          factory_database=config_sg.seg_data_tbl.factory_database,
                                                          sensitivity=config_sg.seg_data_tbl.sensitivity,
                                                          schema=seg_data,
                                                          partition_by=config_sg.seg_data_tbl.partitionByList,
                                                          overwrite_table=False,
                                                          assert_equality=False,
                                                          add_load_timestamp=True
                                                          )

    logger.info(f"""seg_data_table_name: {seg_data_table_name}""")

    persist_utils.insert_df_into_table(target_tbl_name=seg_data_table_name,
                                       insert_df=seg_data,
                                       delete_where=f"campaign={campaign}")

    logger.info("Begin Segmentation of Dataset")
    seg_data_read = persist_utils.read_table(table_name=seg_data_table_name, where=f"campaign={campaign}")
    seg_manager = SegmentationManager(num_cols=config_sg["num_cols"],
                                      cat_cols=config_sg["cat_cols"],
                                      frac_lim=config_sg["frac_lim"],
                                      fail_limit=config_sg["fail_limit"],
                                      k_search_min=config_sg["k_search_min"],
                                      k_search_max=config_sg["k_search_max"],
                                      pca_k=config_sg["pca_k"],
                                      data_lower_lim=config_sg["data_lower_lim"],
                                      user_id=config_sg["user_id"],
                                      verbose=config_sg["verbose"]
                                      )

    segmentations = (seg_manager.get(data=seg_data_read)
                     .withColumn("campaign", F.lit(campaign))
                     )
    segmentations_tbl_name = persist_utils.create_beam_table(table_prefix=config_sg.segmentations_tbl.prefix,
                                                             lab_database=config.dev_database,
                                                             factory_database=config_sg.segmentations_tbl.factory_database,
                                                             sensitivity=config_sg.segmentations_tbl.sensitivity,
                                                             schema=segmentations,
                                                             partition_by=config_sg.segmentations_tbl.partitionByList,
                                                             overwrite_table=False,
                                                             assert_equality=False,
                                                             add_load_timestamp=True
                                                             )
    logger.info(f"""segmentations_tbl_name: {segmentations_tbl_name}""")

    persist_utils.insert_df_into_table(target_tbl_name=segmentations_tbl_name,
                                       insert_df=segmentations,
                                       delete_where=f"campaign={campaign}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Segmentations to use

# COMMAND ----------

if any(step in config.steps for step in ("build_dataset", "fit_rec", "predict")):
    config_use = config["use_segments"]
    if config_use["all"] == True:
        logger.info(f"Use all Segmentations")
        # In
        segmentations_tbl_name = persist_utils.get_table_name(
            factory_database=config_use.segmentations_tbl.factory_database,
            lab_database=config.dev_database,
            table_prefix=config_use.segmentations_tbl.prefix,
            sensitivity=config_use.segmentations_tbl.sensitivity)
        segmentations_tbl = persist_utils.read_table(table_name=segmentations_tbl_name, where=f"campaign={campaign}")
        seg_list = find_all_segments(segmentations_tbl, config_use["partitionByList"])
    else:
        seg_list = config_use["seg_list"]
    logger.info(f"Segmentations: {seg_list}")

# seg_list = [{'experian_hh_composition': 'Cat_U', 'segmentation': 1}, {'experian_hh_composition': 'Cat_00', 'segmentation': 1}, {'experian_hh_composition': 'Cat_00', 'segmentation': 0}, {'experian_hh_composition': 'Cat_05', 'segmentation': 1}, {'experian_hh_composition': 'Cat_01', 'segmentation': 0}, {'experian_hh_composition': 'Cat_05', 'segmentation': 0}, {'experian_hh_composition': 'Cat_03', 'segmentation': 0}, {'experian_hh_composition': 'Cat_01', 'segmentation': 1}, {'experian_hh_composition': 'Cat_U', 'segmentation': 2}, {'experian_hh_composition': 'Cat_U', 'segmentation': 0}, {'experian_hh_composition': 'Cat_02', 'segmentation': 1}, {'experian_hh_composition': 'Cat_02', 'segmentation': 0}, {'experian_hh_composition': 'Cat_03', 'segmentation': 1}, {'experian_hh_composition': 'Cat_08', 'segmentation': 1}, {'experian_hh_composition': 'Cat_04', 'segmentation': 0}, {'experian_hh_composition': 'Cat_08', 'segmentation': 0}, {'experian_hh_composition': 'Cat_04', 'segmentation': 1}, {'experian_hh_composition': 'Cat_07', 'segmentation': 0}, {'experian_hh_composition': 'Cat_10', 'segmentation': 0}, {'experian_hh_composition': 'Cat_07', 'segmentation': 1}, {'experian_hh_composition': 'Cat_10', 'segmentation': 1}, {'experian_hh_composition': 'Cat_06', 'segmentation': 1}, {'experian_hh_composition': 'Cat_06', 'segmentation': 0}, {'experian_hh_composition': 'Cat_09', 'segmentation': 1}, {'experian_hh_composition': 'Cat_11', 'segmentation': 0}, {'experian_hh_composition': 'Cat_09', 'segmentation': 0}, {'experian_hh_composition': 'Cat_11', 'segmentation': 1}]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build dataset

# COMMAND ----------

if "build_dataset" in config.steps:
    logger.info("Begin building dataset")
    config_bd = config["build_dataset"]

    # load factory tables
    articles_df = spark.table("analytics_trans_prod.lu_article")
    trx_line_df = spark.table("analytics_trans_prod.all_transaction_line")
    sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
    segtco_history_df = spark.table("customer_azbase_prod.segtco_history")

    # Load Segmentation Dataset
    # TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
    # TODO: Possibl build data for just 1 segment at a time?
    partitionByList = config_use.segmentations_tbl.partitionByList

    segmentations_tbl_name = persist_utils.create_beam_table(table_prefix=config_bd.segmentations_tbl.prefix,
                                                             lab_database=config.dev_database,
                                                             factory_database=config_bd.segmentations_tbl.factory_database,
                                                             sensitivity=config_bd.segmentations_tbl.sensitivity,
                                                             schema=segmentations,
                                                             partition_by=config_bd.segmentations_tbl.partitionByList,
                                                             overwrite_table=False,
                                                             assert_equality=False,
                                                             add_load_timestamp=True
                                                             )
    segmentations_tbl = (persist_utils.read_table(table_name=segmentations_tbl_name, where=f"campaign={campaign}")
                         .select([config_bd["user_id"]] + partitionByList)
                         )

    # build training data
    trx_manager = TransactionsManager(
        etl_date=get_date(config_bd["etl_date"]),
        lookback_days=config_bd["lookback_days"],
        l1_ids=config_bd["l1_ids"],
        lx=config_bd["lx"],
        lx_ids=config_bd["lx_ids"],
        user_key=config_bd["user_id"],
        window_days=config_bd["window_days"],
    )
    all_data = trx_manager.get(trx_line_df, articles_df, cust_seg=segmentations_tbl)

    etl_data_tbl_name = persist_utils.create_beam_table(table_prefix=config_bd.etl_data_tbl.prefix,
                                                        lab_database=config.dev_database,
                                                        factory_database=config_bd.etl_data_tbl.factory_database,
                                                        sensitivity=config_bd.etl_data_tbl.sensitivity,
                                                        schema=all_data,
                                                        partition_by=config_bd.etl_data_tbl.partitionByList,
                                                        overwrite_table=False,
                                                        assert_equality=False,
                                                        add_load_timestamp=True
                                                        )
    logger.info(f"""etl_data_tbl_name: {etl_data_tbl_name}""")

    persist_utils.insert_df_into_table(target_tbl_name=etl_data_tbl_name,
                                       insert_df=all_data,
                                       delete_where=f"campaign={campaign}")

    # TODO: Save cust_id to account_id Mapping as done in the customer_purchase work.

    # # Check dataset
    # logger.info("Validating training data")
    # all_data = spark.read.parquet(all_data_path)
    # valid_manager = ValidationManager(
    #     start_date=config_bd["start_date"],
    #     end_date=config_bd["end_date"],
    #     date_format=config_bd["date_format"],
    # )
    # is_valid = valid_manager.get(all_data)
    # logger.info(f"Is dataset valid: {is_valid}")
    # logger.info(f"all_data | row count: {all_data.count()}; column count: {len(all_data.columns)}")

# COMMAND ----------

if "build_dataset" in config.steps:
    config_bd = config["build_dataset"]
    etl_data_tbl_name = persist_utils.get_table_name(factory_database=config_bd.etl_data_tbl.factory_database,
                                                     lab_database=config.dev_database,
                                                     table_prefix=config_bd.etl_data_tbl.prefix,
                                                     sensitivity=config_bd.etl_data_tbl.sensitivity)

    etl_data_tbl = persist_utils.read_table(table_name=etl_data_tbl_name, where=f"campaign={campaign}")
    display(etl_data_tbl.orderBy(F.rand()))


# COMMAND ----------

def run_fit_rec(seg, config, database):
    partitionByList = config["partitionByList"]
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
    ext_str = "_".join([str(seg[k]) for k in partitionByList if "campaign"!=k])
    model_tags = {**config.get("model_tags", {}), **{"campaign": campaign}}
    etl_data_tbl_name = persist_utils.get_table_name(factory_database=config.etl_data_tbl.factory_database,
                                                     lab_database=database,
                                                     table_prefix=config.etl_data_tbl.prefix,
                                                     sensitivity=config.etl_data_tbl.sensitivity)

    seg_etl_data_tbl = persist_utils.read_table(table_name=etl_data_tbl_name, where=" and ".join(seg_ext))

    max_size = config["max_train_size"]
    if max_size:
        # Randomly order and limit to max size of training segment
        seg_etl_data_tbl = seg_etl_data_tbl.orderBy(F.rand()).limit(max_size)

    seg_data = seg_etl_data_tbl.toPandas()

    # build surprise preprocessed data
    data_process_manager = DataProcessor(
        feature_col=config["feature_col"],
        item_id=config["item_id"],
        user_id=config["user_id"],
        lognorm=config["lognorm"],
        line_format=config["line_format"],
        min_lim=config["min_lim"],
        max_lim=config["max_lim"]
    )

    seg_data[config["feature_col"]] = seg_data[config["feature_col"]].astype(float)
    rec_data = data_process_manager.get(seg_data)
    logger.info(f"{seg}: Recommender Data Created")

    data_process_manager_name = (config.data_processor_name + "_{ext}").format(campaign=campaign, ext=ext_str)
    logger.info(f"{seg}: Saving Preprocessor obj={data_process_manager}, name={data_process_manager_name}")
    persist_utils.register_model(model_name=data_process_manager_name, model_object=data_process_manager,
                                 tags=model_tags,
                                 description="Headroom: Registered Data Processor Object")

    logger.info(f"{seg}: Build Recommender")

    rec_algo, fit_params = build_recommender(
        X=rec_data,
        method=config["method"],
        params=config["params"],
        param_grid=config["param_grid"]
    )

    rec_name = (config.rec_name + "_{ext}").format(ext=ext_str)
    logger.info(f"{seg}: Saving Recommender obj={rec_algo}, name={rec_name}")
    persist_utils.register_model(model_name=rec_name, model_object=rec_algo,
                                 tags=model_tags,
                                 description="Headroom: Registered Recommender Model")

    param_name = (config.param_name + "_{ext}").format(ext=ext_str)
    logger.info(f"{seg}: Saving Parameters obj={rec_algo}, name={param_name}")
    persist_utils.register_model(model_name=param_name, model_object=fit_params,
                                 tags=model_tags,
                                 description="Headroom: Registered Parameters Object")


if "fit_rec" in config.steps:
    logger.info("Begin Preprocessing dataset")
    config_fr = config["fit_rec"]

    # for seg in seg_list:
    n_threads = int(config_fr["n_threads"])
    pool = ThreadPool(n_threads)
    _pool_res = pool.map(lambda s: run_fit_rec(s, config=config_fr, database=config.dev_database), seg_list)
    pool.close()
    pool.join()

# COMMAND ----------

if "predict" in config.steps:
    logger.info("Begin Predictions")
    config_pd = config["predict"]
    partitionByList = config_pd["partitionByList"]

    etl_data_tbl_name = persist_utils.get_table_name(factory_database=config_pd.etl_data_tbl.factory_database,
                                                     lab_database=config.dev_database,
                                                     table_prefix=config_pd.etl_data_tbl.prefix,
                                                     sensitivity=config_pd.etl_data_tbl.sensitivity)

    for seg in seg_list:
        seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
        ext_str = "_".join([str(seg[k]) for k in partitionByList if k!="campaign"])
        data = persist_utils.read_table(table_name=etl_data_tbl_name, where=" and ".join(seg_ext))

        rec_name = (config_pd.rec_name + "_{ext}").format(ext=ext_str)
        logger.info(f"{seg}: Read Recommender name={rec_name}")
        rec_algo = persist_utils.load_model(model_name=rec_name)

        data_processor_name = (config_pd.data_processor_name + "_{ext}").format(ext=ext_str)
        logger.info(f"{seg}: Read Data Processor name={data_processor_name}")
        data_processor = persist_utils.load_model(model_name=data_processor_name)

        # build surprise preprocessed data
        predictor_manager = Predictor(
            feature_col=config_pd["feature_col"],
            pred_key=f'{config_pd["pred_key"]}_id',
            pred_items=config_pd["pred_items"],
            min_col=data_processor.min_col,
            max_col=data_processor.max_col,
        )

        predictions = predictor_manager.get(data=data, algo=rec_algo)

        prediction_tbl_name = persist_utils.create_beam_table(table_prefix=config_pd.prediction_tbl.prefix,
                                                              lab_database=config.dev_database,
                                                              factory_database=config_pd.prediction_tbl.factory_database,
                                                              sensitivity=config_pd.prediction_tbl.sensitivity,
                                                              schema=predictions,
                                                              partition_by=config_pd.prediction_tbl.partitionByList,
                                                              overwrite_table=False,
                                                              assert_equality=False,
                                                              add_load_timestamp=True
                                                              )
        logger.info(f"""prediction_tbl_name: {prediction_tbl_name}""")

        persist_utils.insert_df_into_table(target_tbl_name=prediction_tbl_name,
                                           insert_df=predictions,
                                           delete_where=" and ".join(seg_ext))

        predictions_read = persist_utils.read_table(table_name=prediction_tbl_name, where=" and ".join(seg_ext))
        logger.info(
            f"predictions {seg} | row count: {predictions_read.count()}; column count: {len(predictions_read.columns)}")

# COMMAND ----------

if "predict" in config.steps:
    config_pd = config["predict"]
    prediction_tbl_name = persist_utils.get_table_name(factory_database=config_pd.prediction_tbl.factory_database,
                                                       lab_database=config.dev_database,
                                                       table_prefix=config_pd.prediction_tbl.prefix,
                                                       sensitivity=config_pd.prediction_tbl.sensitivity)

    prediction_tbl = persist_utils.read_table(table_name=prediction_tbl_name, where=f"campaign={campaign}")
    display(prediction_tbl.orderBy(F.rand()))

# COMMAND ----------

if "allocate" in config.steps:
    logger.info("Begin Allocation")
    config_al = config["allocation"]

    prediction_tbl_name = persist_utils.get_table_name(factory_database=config_al.prediction_tbl.factory_database,
                                                       lab_database=config.dev_database,
                                                       table_prefix=config_al.prediction_tbl.prefix,
                                                       sensitivity=config_al.prediction_tbl.sensitivity)

    predictions = persist_utils.read_table(table_name=prediction_tbl_name, where=f"campaign={campaign}")

    allocation_manager = Allocator2(feature_col=config_al["feature_col"],
                                   offer_limits=config_al["offer_limits"],
                                   offer_desc=config_al["offers_desc"],
                                   user_key=config_al["user_key"],
                                   outlier_min=config_al["outlier_min"],
                                   outlier_max=config_al["outlier_max"],
                                   max_increase=config_al["max_increase"],
                                   min_increase=config_al["min_increase"],
                                   headroom_factor=config_al["headroom_factor"],
                                   fill_offer=config_al["fill_offer"],
                                   )

    headroom_export = (allocation_manager.get(predictions)
                       .withColumn("campaign", F.lit(campaign))
                       )

    headroom_tbl_name = persist_utils.create_beam_table(table_prefix=config_al.headroom_export_tbl.prefix,
                                                        lab_database=config.dev_database,
                                                        factory_database=config_al.headroom_export_tbl.factory_database,
                                                        sensitivity=config_al.headroom_export_tbl.sensitivity,
                                                        schema=headroom_export,
                                                        partition_by=config_al.headroom_export_tbl.partitionByList,
                                                        overwrite_table=True,
                                                        assert_equality=False,
                                                        add_load_timestamp=True
                                                        )
    logger.info(f"""headroom_tbl_name: {prediction_tbl_name}""")

    persist_utils.insert_df_into_table(target_tbl_name=headroom_tbl_name,
                                       insert_df=headroom_export,
                                       delete_where=f"campaign={campaign}")

# COMMAND ----------

# prediction_scores = allocation_manager.get_prediction_scores(predictions)
# display(prediction_scores)

# COMMAND ----------

# prediction_scores_tagged = allocation_manager.tag_outliers(prediction_scores)
# display(prediction_scores)

# COMMAND ----------

# data_hrm = (prediction_scores_tagged
#             .withColumn("used_headroom_frac",
#                         F.when((F.col("pct_error") >= allocation_manager.max_increase) & (F.col("outlier") == 0),
#                                 (1. + allocation_manager.max_increase / 100.))
#                         .when((F.col("pct_error") <= allocation_manager.min_increase) & (F.col("outlier") == 0),
#                               (1. + allocation_manager.min_increase / 100.))
#                         .when((F.col("pct_error") < allocation_manager.max_increase) &
#                               (F.col("pct_error") > allocation_manager.min_increase) & (F.col("outlier") == 0),
#                               1. + F.col("pct_error") / 100.)
#                         .otherwise(allocation_manager.headroom_factor)
#                         )
#             .withColumn("total_used_headroom_per_id",
#                         F.col(allocation_manager.feature_col) * F.col("used_headroom_frac"))
#             .withColumn("rand", F.rand())
#             .groupby(allocation_manager.user_key, "experian_hh_composition", "segmentation")
#             .agg(F.sum("total_used_headroom_per_id").alias("total_used_headroom_whole_time_period"))
#             .withColumn("total_used_headroom", F.col("total_used_headroom_whole_time_period") / F.lit(11.857 ))
#             )
# display(data_hrm)

# COMMAND ----------

# headroom_predictions = allocation_manager.get_headroom(prediction_scores_tagged)
# display(headroom_predictions)

# COMMAND ----------

# headroom_export = allocation_manager.prepare_export(headroom_predictions)

# COMMAND ----------

  # headroom_export = (allocation_manager.get(predictions)
  #                      .withColumn("campaign", F.lit(campaign))
  #                      )

# COMMAND ----------



# COMMAND ----------

if "allocate" in config.steps:
    config_al = config["allocation"]
    headroom_tbl_name = persist_utils.get_table_name(factory_database=config_al.headroom_export_tbl.factory_database,
                                                     lab_database=config.dev_database,
                                                     table_prefix=config_al.headroom_export_tbl.prefix,
                                                     sensitivity=config_al.headroom_export_tbl.sensitivity)

    headroom_tbl = persist_utils.read_table(table_name=headroom_tbl_name, where=f"campaign={campaign}")
    display(headroom_tbl.orderBy(F.rand()))

# COMMAND ----------

config_al["offer_limits"]

# COMMAND ----------

 dict(sorted(config_al["offer_limits"].items(), key=lambda x: max(x[1]),
                                            reverse=True)[:2])

# COMMAND ----------

 dict(sorted(config_al["offer_limits"].items(), key=lambda x: min(x[1]),
                                            reverse=False)[:2])

# COMMAND ----------

headroom_tbl_name

# COMMAND ----------

# headroom_export

headroom_tbl.groupBy('desc').count()\
  .withColumn('percentage', F.round(F.col('count') / F.sum('count')\
  .over(W.partitionBy()),3)).display()

# COMMAND ----------

0.469 + 0.174 + 0.138

# COMMAND ----------

# MAGIC %sql select * from loyalty_azlab_prod.headroom_allocation_p_tbl where campaign = 20230421 and cust_id = '6872732896086312431'

# COMMAND ----------

# MAGIC %sql select * from loyalty_azlab_prod.headroom_allocation_p_tbl where campaign = 20230421 and cust_id = '-7385606211536121860'

# COMMAND ----------

# dbutils.notebook.exit(True)
