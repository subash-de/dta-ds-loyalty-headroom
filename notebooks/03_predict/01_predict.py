# Databricks notebook source
# MAGIC %run ../setup/bootstrap

# COMMAND ----------

dbutils.widgets.text("seg_list", "[]", "")

# COMMAND ----------

from datetime import datetime, timedelta

import seaborn as sns
from dtaml.logging import get_logger
from pyspark.sql import functions as F

import customer_headroom.utils.persist_utils as persist_utils
from customer_headroom.modelling.predict import Predictor

sns.set(style="whitegrid")
logger = get_logger("customer-headroom")

# COMMAND ----------

seg_list = eval(dbutils.widgets.get("seg_list"))
# TODO: remove this, this is just for testing purposes
seg_list = [
    {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 5},
    {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 3},
    {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 6},
    {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 3},
    {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 6},
    {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_00", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 5},
    {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 5},
    {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 6},
    {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 5},
    {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 5},
    {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 3},
    {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 3},
    {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_U", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 3},
    {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 5},
    {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 6},
    {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 6},
    {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_01", "segmentation": 6},
    {"campaign": 20231212, "experian_hh_composition": "Cat_05", "segmentation": 3},
    {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_03", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_02", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 5},
    {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 3},
    {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_06", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 5},
    {"campaign": 20231212, "experian_hh_composition": "Cat_09", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 6},
    {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 5},
    {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 5},
    {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 3},
    {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 3},
    {"campaign": 20231212, "experian_hh_composition": "Cat_11", "segmentation": 0},
    {"campaign": 20231212, "experian_hh_composition": "Cat_07", "segmentation": 6},
    {"campaign": 20231212, "experian_hh_composition": "Cat_08", "segmentation": 6},
    {"campaign": 20231212, "experian_hh_composition": "Cat_04", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 6},
    {"campaign": 20231212, "experian_hh_composition": "Cat_10", "segmentation": 3},
    {"campaign": 20231212, "experian_hh_composition": "Cat_06", "segmentation": 4},
    {"campaign": 20231212, "experian_hh_composition": "Cat_06", "segmentation": 1},
    {"campaign": 20231212, "experian_hh_composition": "Cat_06", "segmentation": 2},
    {"campaign": 20231212, "experian_hh_composition": "Cat_06", "segmentation": 3},
]
if seg_list == []:
    dbutils.notebook.exit(True)
else:
    logger.info(f"seg_list: {seg_list}")

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


logger.info("Begin Predictions")
config_pd = config["predict"]
partitionByList = config_pd["partitionByList"]

# get list of pred items depending on lx id
lu_article = spark.read.table("analytics_trans_prod.lu_article")
pred_items = (
    lu_article.filter(F.col("l2_id").isin(config_pd["pred_items"]))
    .select(f'{config_pd["pred_key"]}_id')
    .distinct()
    .toPandas()
)
list_pred_items = list(pred_items[f'{config_pd["pred_key"]}_id'])

etl_data_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_pd.etl_data_tbl.prefix,
    sensitivity=config.sensitivity,
)


# COMMAND ----------


config_pd = config["predict"]
prediction_tbl_name = persist_utils.get_table_name(
    factory_database=config.factory_database,
    lab_database=config.lab_database,
    table_prefix=config_pd.prediction_tbl.prefix,
    sensitivity=config.sensitivity,
)

logger.info(f"""prediction_tbl_name: {prediction_tbl_name}""")

prediction_tbl = persist_utils.read_table(
    table_name=prediction_tbl_name, where=f"campaign={campaign}"
)
display(prediction_tbl.orderBy(F.rand()))

# COMMAND ----------


for seg in seg_list:
    seg_ext = [f"({k}='{seg[k]}')" for k in partitionByList]
    ext_str = "_".join([str(seg[k]) for k in partitionByList])

    data = persist_utils.read_table(
        table_name=etl_data_tbl_name, where=" and ".join(seg_ext)
    )

    rec_name = (config_pd.rec_name + "_{ext}").format(ext=ext_str)
    logger.info(f"{seg}: Read Recommender name={rec_name}")
    rec_algo = persist_utils.get_latest_version(model_name=rec_name)

    data_processor_name = (config_pd.data_processor_name + "_{ext}").format(ext=ext_str)
    logger.info(f"{seg}: Read Data Processor name={data_processor_name}")
    data_processor = persist_utils.get_latest_version(model_name=data_processor_name)

    # build surprise preprocessed data
    predictor_manager = Predictor(
        feature_col=config_pd["feature_col"],
        pred_key=f'{config_pd["pred_key"]}_id',
        pred_items=list_pred_items,
        min_col=data_processor.min_col,
        max_col=data_processor.max_col,
    )

    predictions = predictor_manager.get(data=data, algo=rec_algo)
    # add the segment here !!! or else the rows are note deleted when inserting
    # new rows are added in the predict step for l2 ids not in etl
    predictions = (
        predictions.withColumn("campaign", F.lit(campaign))
        .drop("load_timestamp")
        .withColumn("experian_hh_composition", F.lit(seg["experian_hh_composition"]))
        .withColumn("segmentation", F.lit(seg["segmentation"]))
    )  # ----------------------------------------------
    prediction_tbl_name = persist_utils.create_beam_table(
        table_prefix=config_pd.prediction_tbl.prefix,
        lab_database=config.lab_database,
        factory_database=config.factory_database,
        sensitivity=config.sensitivity,
        schema=predictions,
        partition_by=config_pd.prediction_tbl.partitionByList,
        overwrite_table=False,
        assert_equality=False,
        add_load_timestamp=True,
    )
    logger.info(f"""prediction_tbl_name: {prediction_tbl_name}""")

    persist_utils.insert_df_into_table(
        target_tbl_name=prediction_tbl_name,
        insert_df=predictions,
        delete_where=" and ".join(seg_ext),
    )

    predictions_read = persist_utils.read_table(
        table_name=prediction_tbl_name, where=" and ".join(seg_ext)
    )
    logger.info(
        f"predictions {seg} | row count: {predictions_read.count()}; column count: {len(predictions_read.columns)}"
    )

# COMMAND ----------

predictions_read = persist_utils.read_table(
    table_name=prediction_tbl_name, where="campaign = 20231212"
)

# COMMAND ----------

display(predictions_read)

# COMMAND ----------

dbutils.notebook.exit(True)

# COMMAND ----------
