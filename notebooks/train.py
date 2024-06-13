# Databricks notebook source
# MAGIC %run ./bootstrap

# COMMAND ----------

dbutils.widgets.text("seg", "{}", "")

# COMMAND ----------

from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.modelling.data_process import DataProcessor
from customer_headroom.modelling.fit import build_recommender

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

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


config_dates = config["dates"]
campaign = get_campaign(config_dates["upcoming_campaign"], config_dates["etl_date"])

date_format = "%Y%m%d"
last_registration_date = int(
    (
        datetime.strptime(str(campaign), date_format)
        - timedelta(days=config_dates["lookback_days_registration"])
    ).strftime(date_format)
)

# campaign = 20230522

logger.info(
    f"""
config_dates: {config_dates}
campaign: {campaign}
last_registration_date: {last_registration_date}
"""
)

# COMMAND ----------


def run_fit_rec(seg, config, database):
    partitionByList = config["partitionByList"]
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
    ext_str = "_".join([str(seg[k]) for k in partitionByList])

    model_tags = {**config.get("model_tags", {}), **{"campaign": campaign}}
    etl_data_tbl_name = persist_utils.get_table_name(
        factory_database=config.etl_data_tbl.factory_database,
        lab_database=database,
        table_prefix=config.etl_data_tbl.prefix,
        sensitivity=config.etl_data_tbl.sensitivity,
    )

    seg_etl_data_tbl = persist_utils.read_table(
        table_name=etl_data_tbl_name, where=" and ".join(seg_ext)
    )

    seg_data = seg_etl_data_tbl.toPandas()

    # build surprise preprocessed data
    data_process_manager = DataProcessor(
        feature_col=config["feature_col"],
        item_id=config["item_id"],
        user_id=config["user_id"],
        lognorm=config["lognorm"],
        line_format=config["line_format"],
        min_lim=config["min_lim"],
        max_lim=config["max_lim"],
    )

    seg_data[config["feature_col"]] = seg_data[config["feature_col"]].astype(float)
    rec_data = data_process_manager.get(seg_data)
    logger.info(f"{seg}: Recommender Data Created")

    data_process_manager_name = (config.data_processor_name + "_{ext}").format(
        campaign=campaign, ext=ext_str
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
        method=config["method"],
        params=config["params"],
        param_grid=config["param_grid"],
    )

    rec_name = (config.rec_name + "_{ext}").format(ext=ext_str)
    logger.info(f"{seg}: Saving Recommender obj={rec_algo}, name={rec_name}")
    persist_utils.register_model(
        model_name=rec_name,
        model_object=rec_algo,
        tags=model_tags,
        description="Headroom: Registered Recommender Model",
    )

    param_name = (config.param_name + "_{ext}").format(ext=ext_str)
    logger.info(f"{seg}: Saving Parameters obj={rec_algo}, name={param_name}")
    persist_utils.register_model(
        model_name=param_name,
        model_object=fit_params,
        tags=model_tags,
        description="Headroom: Registered Parameters Object",
    )


# COMMAND ----------

logger.info("Begin Preprocessing dataset")
config_fr = config["fit_rec"]
run_fit_rec(seg=seg, config=config_fr, database=config.dev_database)

# COMMAND ----------

dbutils.notebook.exit(True)
