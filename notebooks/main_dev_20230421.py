# Databricks notebook source
# MAGIC %run ./bootstrap $environment=prod

# COMMAND ----------

import os
from functools import partial
import pandas as pd
from datetime import datetime
from customer_headroom.etl.build_dataset import TransactionsManager
from customer_headroom.etl.segmentation import SegmentationDataManager, SegmentationManager
from customer_headroom.modelling.data_process import DataProcessor
from customer_headroom.modelling.fit import build_recommender
from customer_headroom.modelling.predict import Predictor
from customer_headroom.evaluation.model_selection import Evaluator
from customer_headroom.allocation.allocator import Allocator
import offerallocationv2.utils.persist_utils as persist_utils
from dtaml.logging import get_logger
from cdsutils.io_utils import file_exists, save_object, load_object
from pyspark.sql import DataFrame, functions as F
from multiprocessing.pool import ThreadPool
import seaborn as sns

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")


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
    custs_etl_data = None

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

    allocation_manager = Allocator(feature_col=config_al["feature_col"],
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

if "allocate" in config.steps:
    config_al = config["allocation"]
    headroom_tbl_name = persist_utils.get_table_name(factory_database=config_al.headroom_export_tbl.factory_database,
                                                     lab_database=config.dev_database,
                                                     table_prefix=config_al.headroom_export_tbl.prefix,
                                                     sensitivity=config_al.headroom_export_tbl.sensitivity)

    headroom_tbl = persist_utils.read_table(table_name=headroom_tbl_name, where=f"campaign={campaign}")
    display(headroom_tbl.orderBy(F.rand()))

# COMMAND ----------

dbutils.notebook.exit(True)
