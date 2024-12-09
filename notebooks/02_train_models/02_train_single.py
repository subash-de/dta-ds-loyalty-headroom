# Databricks notebook source
# MAGIC %run ../bootstrap

# COMMAND ----------

dbutils.widgets.text("seg", "{}", "")

# COMMAND ----------

from datetime import datetime, timedelta
import mlflow
import seaborn as sns
from dtaml.logging import get_logger

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.modelling.data_process import DataProcessor
from customer_headroom.modelling.fit import build_recommender

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

mlflow.set_registry_uri("databricks")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

seg = eval(dbutils.widgets.get("seg"))

if seg == {}:
    dbutils.notebook.exit(True)
else:
    logger.info(f"seg: {seg}")

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


# COMMAND ----------

debug = False
if debug:
    config.dates.etl_date = config.debug.tables.etl_date
    config.dates.lookback_days = config.debug.tables.lookback_days

campaign = get_campaign(config.dates.upcoming_campaign, config.dates.etl_date)


last_registration_date = int(
    (
        datetime.strptime(str(campaign), config.dates.date_format)
        - timedelta(days=config.dates.lookback_days_registration)
    ).strftime(config.dates.date_format)
)

# campaign = 20230522

logger.info(
    f"""
config.dates: {config.dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
)

# COMMAND ----------


def run_fit_rec(seg, config):
    config_fr = config["fit_rec"]
    partitionByList = config_fr["partitionByList"]
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
    ext_str = "_".join([str(seg[k]) for k in partitionByList])

    model_tags = {**config.get("model_tags", {}), **{"campaign": campaign}}
    etl_data_tbl_name = persist_utils.get_table_name(
        factory_database=config.factory_database,
        lab_database=config.lab_database,
        table_prefix=config_fr.etl_data_tbl.prefix,
        sensitivity=config.sensitivity,
    )

    seg_etl_data_tbl = persist_utils.read_table(
        table_name=etl_data_tbl_name, where=" and ".join(seg_ext)
    )

    seg_data = seg_etl_data_tbl.toPandas()

    # build surprise preprocessed data
    data_process_manager = DataProcessor(
        feature_col=config_fr["feature_col"],
        item_id=config_fr["item_id"],
        user_id=config_fr["user_id"],
        lognorm=config_fr["lognorm"],
        line_format=config_fr["line_format"],
        min_lim=config_fr["min_lim"],
        max_lim=config_fr["max_lim"],
    )

    seg_data[config_fr["feature_col"]] = seg_data[config_fr["feature_col"]].astype(
        float
    )
    rec_data = data_process_manager.get(seg_data)
    logger.info(f"{seg}: Recommender Data Created")

    data_process_manager_name = (config_fr.data_processor_name + "_{ext}" + "{model_prefix}").format(
        campaign=campaign, ext=ext_str, model_prefix=config_fr["model_prefix"]
    )
    logger.info(
        f"{seg}: Saving Preprocessor obj={data_process_manager}, name={data_process_manager_name}"
    )
    persist_utils.register_model(
        model_name=data_process_manager_name,
        model_object=data_process_manager,
        tags=model_tags,
        description="Headroom: Registered Data Processor Object",
    )

    logger.info(f"{seg}: Build Recommender")

    rec_algo, fit_params = build_recommender(
        X=rec_data,
        method=config_fr["method"],
        params=config_fr["params"],
        param_grid=config_fr["param_grid"],
    )

    rec_name = (config_fr.rec_name + "_{ext}" + "{model_prefix}").format(ext=ext_str, model_prefix = config_fr["model_prefix"])
    logger.info(f"{seg}: Saving Recommender obj={rec_algo}, name={rec_name}")

    persist_utils.register_model(
        model_name=rec_name,
        model_object=rec_algo,
        tags=model_tags,
        description="Headroom: Registered Recommender Model",
    )

    param_name = (config_fr.param_name + "_{ext}"+ "{model_prefix}").format(ext=ext_str, model_prefix = config_fr['model_prefix'])
    logger.info(f"{seg}: Saving Parameters obj={rec_algo}, name={param_name}")
    persist_utils.register_model(
        model_name=param_name,
        model_object=fit_params,
        tags=model_tags,
        description="Headroom: Registered Parameters Object",
    )


# COMMAND ----------

logger.info("Begin Preprocessing dataset")

run_fit_rec(seg=seg, config=config)

# COMMAND ----------

dbutils.notebook.exit(True)
