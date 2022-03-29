# Databricks notebook source
# MAGIC %run ./bootstrap $environment=prod

# COMMAND ----------

devops_token = dbutils.secrets.get("dta-eun-kv-dsc-01", "access-token-devops-artifacts")
pip_url = f"https://{devops_token}@pkgs.dev.azure.com/dta-devops/datascience-platforms/_packaging/dta-ds-libraries/pypi/simple/"  # .format(token=devops_token)
%pip install --extra-index-url "{pip_url}" dtaml cdsutils==0.0.8.2021061002 customer-headroom==0.1.7a57544

# COMMAND ----------

from customer_headroom.config import load_config

config = load_config('dev')
print(f'Config used is: \n{config.dumps()}')

# COMMAND ----------

# segmentation
# config.steps =  ["build_dataset", "fit_rec", "predict", "offline_eval"]

# COMMAND ----------

import os
from functools import partial
import pandas as pd
from customer_headroom.etl.build_dataset import TransactionsManager
from customer_headroom.etl.segmentation import SegmentationDataManager, SegmentationManager
from customer_headroom.modelling.data_process import DataProcessor
from customer_headroom.modelling.fit import build_recommender
from customer_headroom.modelling.predict import Predictor
from customer_headroom.evaluation.model_selection import Evaluator
from dtaml.logging import get_logger
from cdsutils.io_utils import file_exists, save_object, load_object
from pyspark.sql import DataFrame, functions as F
import seaborn as sns

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")


# COMMAND ----------

def write(obj, path, write_mode):
    if write_mode == "errorifexists" and file_exists(path):
        raise IOError(f"{path} exists")
    else:
        if isinstance(obj, DataFrame):
            obj.repartition(1).write.parquet(path, mode=write_mode)
        else:
            save_object(obj, path)


def find_all_segments(data, partitionByList):
    segs = data.select(partitionByList).distinct().rdd.map(
        lambda x: {k: v for (k, v) in zip(partitionByList, x)}).collect()
    return segs


# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Segmentations

# COMMAND ----------

if "segmentation" in config.steps:
    logger.info("Begin Building Segmentation Dataset")
    config_sg = config["segmentation"]
    cust_path = os.path.join(*config_sg["cust_path"])
    seg_data_path = os.path.join(*config_sg["seg_data_path"])
    custs_data_seg_path = os.path.join(*config_sg["custs_data_seg_path"])
    partitionByList = config_sg["partitionByList"]

    # load factory tables
    sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
    cust_master_df = spark.sql("select * from analytics_trans_prod.customer_master")
    trx_line_df = spark.table("analytics_trans_prod.all_transaction_line")

    # load customer table
    # TODO: Replace with input customer id's if required.
    custs_etl_data = spark.read.parquet(cust_path)

    seg_data_manager = SegmentationDataManager(etl_date=config_sg["etl_date"],
                                               lookback_days=config_sg["lookback_days"],
                                               l1_id=config_sg["l1_id"],
                                               user_id=config_sg["user_id"])

    seg_data = seg_data_manager.get(trx_line_df, sparks_account_df, cust_master_df, customer_input=custs_etl_data)
    logger.info(f"Writing seg_data to {seg_data_path}")

    # TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
    seg_data.write.parquet(seg_data_path, mode=config_sg["write_mode"])

    logger.info("Begin Segmentation of Dataset")
    seg_data_read = spark.read.parquet(seg_data_path)
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

    custs_etl_data_seg = seg_manager.get(data=seg_data_read)

    logger.info(f"Writing custs_etl_data_seg to {custs_data_seg_path}")
    custs_etl_data_seg.write.partitionBy(*partitionByList).parquet(custs_data_seg_path, mode=config_sg["write_mode"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Segmentations to use

# COMMAND ----------

if any(step in config.steps for step in ("build_dataset", "fit_rec", "predict")):
    config_use = config["use_segments"]
    if config_use["all"] == True:
        logger.info(f"Use all Segmentations")
        custs_data_seg_path = config_use["custs_data_seg_path"]
        partitionByList = config_use["partitionByList"]
        # In
        custs_data_seg = spark.read.parquet(os.path.join(*custs_data_seg_path))
        seg_list = find_all_segments(custs_data_seg, partitionByList)
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
    cust_path = os.path.join(*config_bd["cust_input_path"])
    partitionByList = config_bd["partitionByList"]

    # load factory tables
    articles_df = spark.table("analytics_trans_prod.lu_article")
    trx_line_df = spark.table("analytics_trans_prod.all_transaction_line")
    sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
    segtco_history_df = spark.table("customer_azbase_prod.segtco_history")

    # Load Segmentation Dataset
    # TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
    # TODO: Possibl build data for just 1 segment at a time?
    custs = spark.read.parquet(cust_path).select([config_bd["user_id"]] + partitionByList)

    # build training data
    trx_manager = TransactionsManager(
        start_date=config_bd["start_date"],
        end_date=config_bd["end_date"],
        l1_ids=config_bd["l1_ids"],
        lx=config_bd["lx"],
        lx_ids=config_bd["lx_ids"],
        user_key=config_bd["user_id"],
        window_days=config_bd["window_days"],
    )
    all_data = trx_manager.get(trx_line_df, articles_df, cust_seg=custs)
    etl_data_path = os.path.join(*config_bd["etl_data_path"])
    logger.info(f"Writing all_data to {etl_data_path}")
    all_data.write.partitionBy(*partitionByList).parquet(etl_data_path, mode=config_bd["write_mode"])
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
    etl_data_path = os.path.join(*config_bd["etl_data_path"])
    all_data_groups = spark.read.parquet(etl_data_path)
    display(all_data_groups.orderBy(F.rand()))

# COMMAND ----------

if "fit_rec" in config.steps:
    logger.info("Begin Preprocessing dataset")
    config_fr = config["fit_rec"]
    etl_data_path = os.path.join(*config_fr["etl_data_path"])
    partitionByList = config_fr["partitionByList"]

    for seg in seg_list:
        seg_ext = [f"{k}={seg[k]}" for k in partitionByList]
        seg_data_path = os.path.join(*([f"{etl_data_path}"] + seg_ext))
        seg_data = spark.read.parquet(seg_data_path).toPandas()

        # build surprise preprocessed data
        data_process_manager = DataProcessor(
            feature_col=config_fr["feature_col"],
            item_id=config_fr["item_id"],
            user_id=config_fr["user_id"],
            lognorm=config_fr["lognorm"],
            line_format=config_fr["line_format"],
            min_lim=config_fr["min_lim"],
            max_lim=config_fr["max_lim"]
        )

        seg_data[config_fr["feature_col"]] = seg_data[config_fr["feature_col"]].astype(float)
        rec_data = data_process_manager.get(seg_data)
        logger.info(f"{seg}: Recommender Data Created")

        data_processor_path = os.path.join(*(config_fr['data_processor_path'] + seg_ext))
        logger.info(f"{seg}: Saving Preprocessor obj={data_process_manager}, path={data_processor_path}")
        write(data_process_manager, os.path.join(data_processor_path), write_mode=config_fr["write_mode"])
        logger.info(f"{seg}: Build Recommender")

        rec_algo, fit_params = build_recommender(
            X=rec_data,
            method=config_fr["method"],
            params=config_fr["params"],
            param_grid=config_fr["param_grid"]
        )

        rec_path = os.path.join(*(config_fr['rec_path'] + seg_ext))
        param_path = os.path.join(*(config_fr['param_path'] + seg_ext))
        logger.info(f"{seg}: Saving Recommender obj={rec_algo}, path={rec_path}")
        write(rec_algo, rec_path, write_mode=config_fr["write_mode"])
        logger.info(f"{seg}: Saving Fit Parameters obj={fit_params}, path={param_path}")
        write(fit_params, param_path, write_mode=config_fr["write_mode"])

# COMMAND ----------

if "predict" in config.steps:
    logger.info("Begin Predictions")
    config_pd = config["predict"]

    pred_data_path = os.path.join(*config_pd["pred_data_path"])
    partitionByList = config_pd["partitionByList"]

    for seg in seg_list:
        seg_ext = [f"{k}={seg[k]}" for k in partitionByList]
        seg_pred_data_path = os.path.join(*([f"{pred_data_path}"] + seg_ext))

        data_processor_path = os.path.join(*(config_pd["data_processor_path"] + seg_ext))
        rec_path = os.path.join(*(config_pd["rec_path"] + seg_ext))
        pred_path = os.path.join(*(config_pd["pred_path"] + seg_ext))

        data = spark.read.parquet(seg_pred_data_path)
        rec_algo = load_object(rec_path)
        data_processor = load_object(data_processor_path)
        # build surprise preprocessed data
        predictor_manager = Predictor(
            feature_col=config_pd["feature_col"],
            pred_key=f'{config_pd["pred_key"]}_id',
            pred_items=config_pd["pred_items"],
            min_col=data_processor.min_col,
            max_col=data_processor.max_col,
        )

        predictions = predictor_manager.get(data=data, algo=rec_algo)

        for obj, path in ((predictions, pred_path),):
            logger.info(f"{seg}: Saving obj={obj}, path={path}")
            write(obj, path, write_mode=config_pd["write_mode"])

        predictions = spark.read.parquet(pred_path)
        logger.info(f"predictions {seg} | row count: {predictions.count()}; column count: {len(predictions.columns)}")

# COMMAND ----------

if "predict" in config.steps:
    config_pd = config["predict"]
    pred_path = os.path.join(*config_pd["pred_path"])
    predictions = spark.read.parquet(pred_path)
    display(predictions.orderBy(F.rand()))

# COMMAND ----------

if "offline_eval" in config.steps:
    logger.info("Begin Offline Evaluation")
    config_ev = config["offline_eval"]

    user_key = config_ev["user_key"]
    pred_key = f'{config_ev["pred_key"]}_id'
    method = config_ev["method"]
    methods = config_ev["methods"]

    for seg in seg_list:
        seg_ext = [f"{k}={seg[k]}" for k in partitionByList]
        seg_pred_data_path = os.path.join(*(config_ev["eval_data_path"] + seg_ext))
        seg_data_processor_path = os.path.join(*(config_ev["data_processor_path"] + seg_ext))

        data_processor = load_object(seg_data_processor_path)
        data = (spark.read.parquet(seg_pred_data_path)
                .select(user_key, pred_key, data_processor.feature_col)
                ).toPandas()

        all_methods = list(set(map(str.lower, methods)).union({str(method).lower()}))
        for m in all_methods:
            logger.info(f"Run Offline Evaluation for Method: {m.upper()}")
            algo_fn = partial(
                build_recommender,
                method=m
            )

            evaluator = Evaluator(
                algorithm=algo_fn,
                data_processor=data_processor,
                user_key=user_key,
                pred_key=pred_key,
                pred_items=config_ev["pred_items"],
                dev_size=config_ev["dev_size"],
                test_size=config_ev["test_size"],
                split_col=config_ev["split_col"],
                sample=config_ev["sample"],
                random_state=config_ev["random_state"]
            )

            if config_ev["kfold"]:
                eval_summary = evaluator.evaluate_kfold(data,
                                                        n_splits=config_ev["n_splits"],
                                                        shuffle=config_ev["shuffle"],
                                                        run_tag=f"{seg}: {m}"
                                                        )
            else:
                eval_summary = evaluator.evaluate(data, run_tag=f"{seg}: {m}")
            eval_summaries[m] = eval_summary
            full_eval_summary = pd.concat(list(eval_summaries.values()))

# COMMAND ----------

if "offline_eval" in config.steps:
    # metrics summary
    display(full_eval_summary)

# COMMAND ----------

dbutils.notebook.exit(True)
