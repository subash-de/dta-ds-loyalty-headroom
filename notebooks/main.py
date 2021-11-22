# Databricks notebook source
# MAGIC %run ./bootstrap $environment=prod

# COMMAND ----------

import os
from customer_headroom.etl.build_dataset import TransactionsManager
from customer_headroom.modelling.data_process import DataProcessor
from customer_headroom.modelling.fit import build_recommender
from customer_headroom.modelling.predict import Predictor
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build dataset

# COMMAND ----------

if "build_dataset" in config.steps:
    logger.info("Begin building dataset")
    config_bd = config["build_dataset"]
    etl_data_path = os.path.join(*config_bd["etl_data_path"])
    cust_path = os.path.join(*config_bd["cust_input_path"])

    # load factory tables
    articles_df = spark.table("analytics_trans_prod.lu_article")
    trx_line_df = spark.table("analytics_trans_prod.all_transaction_line")
    sparks_account_df = spark.table("analytics_trans_prod.sparks_account")
    segtco_history_df = spark.table("customer_azbase_prod.segtco_history")

    # build training data
    trx_manager = TransactionsManager(
        start_date=config_bd["start_date"],
        end_date=config_bd["end_date"],
        l1_ids=config_bd["l1_ids"],
        lx=config_bd["lx"],
        lx_ids=config_bd["lx_ids"],
    )
    all_data = trx_manager.get(trx_line_df, articles_df)
    logger.info(f"Writing all_data to {etl_data_path}")

    # for now use just this top customer sample data.
    # TODO: Replace with customer cluster work to reduce data sizes to appropiate groups.
    custs = spark.read.parquet(cust_path)
    all_data_groups = all_data.join(custs.select("cust_id").distinct(), on="cust_id")
    all_data_groups.write.parquet(etl_data_path, mode=config_bd["write_mode"])

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

    data = spark.read.parquet(etl_data_path).toPandas()

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

    rec_data = data_process_manager.get(data)
    logger.info("Recommender Data Created")

    data_processor_path = os.path.join(*config_fr['data_processor_path'])
    logger.info(f"Saving Preprocessor obj={data_process_manager}, path={data_processor_path}")
    write(data_process_manager, data_processor_path, write_mode=config_fr["write_mode"])
    logger.info("Build Recommender")

    rec_algo = build_recommender(
        X=rec_data,
        method=config_fr["method"]
    )

    rec_path = os.path.join(*config_fr['rec_path'])
    logger.info(f"Saving Recommender obj={rec_algo}, path={rec_path}")
    write(rec_algo, rec_path, write_mode=config_fr["write_mode"])

# COMMAND ----------

if "predict" in config.steps:
    logger.info("Begin Predictions")
    config_pd = config["predict"]
    pred_data_path = os.path.join(*config_pd["pred_data_path"])
    data_processor_path = os.path.join(*config_pd["data_processor_path"])
    rec_path = os.path.join(*config_pd["rec_path"])

    pred_path = os.path.join(*config_pd["pred_path"])

    data = spark.read.parquet(pred_data_path)
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
        logger.info(f"Saving obj={obj}, path={path}")
        write(obj, path, write_mode=config_fr["write_mode"])

    predictions = spark.read.parquet(pred_path)
    logger.info(f"predictions | row count: {predictions.count()}; column count: {len(predictions.columns)}")

# COMMAND ----------

dbutils.notebook.exit(True)
